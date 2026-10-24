/**
 * Loads the MediaPipe Tasks models from the site's own origin (`assetBase`, filled by scripts/mediapipe-assets.mjs):
 * WASM from `<assetBase>wasm/`, models from `<assetBase>models/`. The packages are imported lazily, so the mock and
 * the console never download them. GPU delegate first, CPU if the GPU path fails to start.
 *
 * Model settings follow valence/perception/video.py and audio.py: GestureRecognizer with up to 8 hands and every
 * canned gesture's score (so Closed_Fist is there even when it is not the top gesture), FaceLandmarker with
 * blendshapes (+ the transformation matrix for the emotion cues) plus a single-face instance for head crops,
 * PoseLandmarker lite standing in for YOLO-pose, and YAMNet restricted to the speech and laughter classes.
 */
import type {
  FaceLandmarker, FaceLandmarkerResult, GestureRecognizer, GestureRecognizerResult, PoseLandmarker,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type { AudioClassifier, AudioClassifierResult } from "@mediapipe/tasks-audio";
import { YAMNET_ALLOWLIST } from "./audioFeatures";

export type Delegate = "GPU" | "CPU";
type Source = TexImageSource;

/** What the video pipeline needs from MediaPipe (an interface, so tests can mock it). */
export interface VisionModels {
  pose(image: Source, tsMs: number): PoseLandmarkerResult;
  faces(image: Source, tsMs: number): FaceLandmarkerResult;
  faceCrop(image: Source): FaceLandmarkerResult;
  gestures(image: Source, tsMs: number): GestureRecognizerResult;
  delegate: Delegate;
  close(): void;
}

export interface AudioModels {
  classify(samples: Float32Array, sampleRate: number): AudioClassifierResult[];
  close(): void;
}

const MODEL = {
  face: "models/face_landmarker.task",
  pose: "models/pose_landmarker_lite.task",
  gesture: "models/gesture_recognizer.task",
  yamnet: "models/yamnet.tflite",
};

const join = (base: string, rel: string) => `${base.endsWith("/") ? base : `${base}/`}${rel}`;

/**
 * "GPU" unless WebGL is software-rendered (SwiftShader, llvmpipe: headless browsers, VMs, GPU blocklists), where
 * MediaPipe's GPU path runs ~10x slower than its CPU (XNNPACK) path: measured in headless Chromium, pose 660 ms vs
 * 68 ms per frame.
 */
export function autoDelegate(): Delegate {
  try {
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : document.createElement("canvas");
    const gl = (canvas.getContext("webgl2") ?? canvas.getContext("webgl")) as WebGLRenderingContext | null;
    if (!gl) return "CPU";
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER) ?? "");
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return /swiftshader|llvmpipe|software|basic render/i.test(renderer) ? "CPU" : "GPU";
  } catch {
    return "GPU";
  }
}

export async function loadVisionModels(
  assetBase: string, opts: { delegate?: Delegate; maxPeople?: number; useModule?: boolean } = {},
): Promise<VisionModels> {
  const mp = await import("@mediapipe/tasks-vision");
  // useModule: the ES-module WASM loader, for module workers (no importScripts there).
  const fileset = await mp.FilesetResolver.forVisionTasks(join(assetBase, "wasm"), opts.useModule ?? false);
  // Every task instantiates its own WASM module through a global ModuleFactory that the loader script defines and
  // the task then clears. importScripts / <script> re-run the loader each time, but an ES-module import runs only
  // once per URL, so in a module worker the second task found no factory: give each task its own loader URL.
  let nTask = 0;
  const filesetFor = () =>
    opts.useModule ? { ...fileset, wasmLoaderPath: `${fileset.wasmLoaderPath}?task=${nTask++}` } : fileset;
  // Fetch each model once (the face model serves two landmarkers) and hand MediaPipe the bytes.
  const [faceBuf, poseBuf, gestureBuf] = await Promise.all(
    [MODEL.face, MODEL.pose, MODEL.gesture].map((file) => fetchModel(join(assetBase, file))),
  );
  const buffers: Record<string, Uint8Array> = { [MODEL.face]: faceBuf!, [MODEL.pose]: poseBuf!, [MODEL.gesture]: gestureBuf! };
  const build = async (delegate: Delegate): Promise<VisionModels> => {
    const base = (file: string) => ({ modelAssetBuffer: buffers[file]!, delegate });
    const created: { close(): void }[] = [];
    try {
      const pose: PoseLandmarker = await mp.PoseLandmarker.createFromOptions(filesetFor(), {
        baseOptions: base(MODEL.pose), runningMode: "VIDEO", numPoses: opts.maxPeople ?? 3,
        // The Python edge's YOLO person threshold (conf=0.4). MediaPipe's default 0.5 found only one of the two
        // people in the e2e two-person scene; 0.4 finds both.
        minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4,
      });
      created.push(pose);
      const faces: FaceLandmarker = await mp.FaceLandmarker.createFromOptions(filesetFor(), {
        baseOptions: base(MODEL.face), runningMode: "VIDEO", numFaces: 3, outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
      created.push(faces);
      const faceCrop: FaceLandmarker = await mp.FaceLandmarker.createFromOptions(filesetFor(), {
        baseOptions: base(MODEL.face), runningMode: "IMAGE", numFaces: 1, outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
      created.push(faceCrop);
      const gestures: GestureRecognizer = await mp.GestureRecognizer.createFromOptions(filesetFor(), {
        baseOptions: base(MODEL.gesture), runningMode: "VIDEO", numHands: 8,
        cannedGesturesClassifierOptions: { maxResults: 8 },
      });
      created.push(gestures);
      return {
        delegate,
        pose: (image, ts) => pose.detectForVideo(image, ts),
        faces: (image, ts) => faces.detectForVideo(image, ts),
        faceCrop: (image) => faceCrop.detect(image),
        gestures: (image, ts) => gestures.recognizeForVideo(image, ts),
        close: () => created.forEach((t) => t.close()),
      };
    } catch (e) {
      created.forEach((t) => t.close());
      throw e;
    }
  };
  if ((opts.delegate ?? autoDelegate()) === "CPU") return build("CPU");
  try {
    return await build("GPU");
  } catch (e) {
    console.warn("[perception] GPU delegate failed, falling back to CPU", e);
    return build("CPU");
  }
}

async function fetchModel(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

export async function loadAudioModels(assetBase: string): Promise<AudioModels> {
  const mp = await import("@mediapipe/tasks-audio");
  const fileset = await mp.FilesetResolver.forAudioTasks(join(assetBase, "wasm"));
  const classifier: AudioClassifier = await mp.AudioClassifier.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: join(assetBase, MODEL.yamnet) },
    categoryAllowlist: [...YAMNET_ALLOWLIST], maxResults: YAMNET_ALLOWLIST.length,
  });
  return {
    classify: (samples, sampleRate) => classifier.classify(samples, sampleRate),
    close: () => classifier.close(),
  };
}

/** Bytes fetched over the network for assets under `assetBase` (0 for HTTP-cache hits). */
export function assetDownloadBytes(assetBase: string): number {
  if (typeof performance === "undefined" || !performance.getEntriesByType) return 0;
  const prefix = new URL(join(assetBase, ""), globalThis.location?.href ?? "http://localhost/").href;
  return (performance.getEntriesByType("resource") as PerformanceResourceTiming[])
    .filter((e) => e.name.startsWith(prefix))
    .reduce((n, e) => n + (e.transferSize || 0), 0);
}
