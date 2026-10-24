/**
 * Web Worker that runs the video pipeline (video.ts) off the page's main thread. The page transfers one ImageBitmap
 * per step; the worker processes it, closes it and replies with the VideoObs (scalars and boxes only). Pixel copies
 * (luminance, head crops) live in the worker's OffscreenCanvases and are wiped after each use, as on the main thread.
 */
import { PixelReader } from "./capture";
import { loadVisionModels, type Delegate } from "./models";
import { VisionPipeline } from "./video";
import type { WorkerReply, WorkerRequest } from "./visionHost";

/** True in a module worker, where MediaPipe must load its ES-module WASM loader instead of importScripts(). */
function isModuleWorker(): boolean {
  try {
    (self as unknown as { importScripts: () => void }).importScripts();
    return false;
  } catch {
    return true;
  }
}

let pipeline: VisionPipeline | null = null;
let closeModels: (() => void) | null = null;
let derived = 0;
const reader = new PixelReader(() => derived++);
const post = (msg: WorkerReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.kind === "init") {
    post({ kind: "started" }); // the page stops timing the start: the models may take a while to download
    try {
      const models = await loadVisionModels(msg.assetBase, {
        ...(msg.delegate ? { delegate: msg.delegate as Delegate } : {}), maxPeople: msg.maxPeople,
        useModule: isModuleWorker(),
      });
      closeModels = () => models.close();
      pipeline = new VisionPipeline(models, reader, {
        maxPeople: msg.maxPeople, minPersonH: msg.minPersonH, faceEveryFrame: msg.faceEveryFrame,
      });
      post({ kind: "ready", delegate: models.delegate });
    } catch (err) {
      post({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  if (msg.kind === "frame") {
    const { bitmap } = msg;
    const t0 = performance.now();
    derived = 0;
    try {
      if (!pipeline) throw new Error("vision worker not initialised");
      const obs = pipeline.process({ image: bitmap, width: bitmap.width, height: bitmap.height }, msg.t, msg.wall, msg.seq,
        msg.decoded);
      bitmap.close();
      post({ kind: "obs", id: msg.id, obs, ms: performance.now() - t0, derived });
    } catch (err) {
      bitmap.close();
      post({ kind: "frame-error", id: msg.id, message: err instanceof Error ? err.message : String(err), derived });
    }
    return;
  }
  if (msg.kind === "close") {
    closeModels?.();
    closeModels = null;
    pipeline = null;
    reader.close();
    self.close();
  }
};
