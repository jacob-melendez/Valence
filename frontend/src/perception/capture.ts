/**
 * Camera and mic capture for the in-browser perception. Frames and audio exist only in memory, for one step:
 * - VideoFrames: one ImageBitmap per processing step, closed right after; pixel reads (luminance for the light
 *   estimate, head crops for the face fallback) go through canvases that are cleared and buffers that are
 *   zero-filled after use. No MediaRecorder, no persisted canvas or image data, nothing sent anywhere.
 * - MicTap: an AudioWorklet copies ~100 ms chunks of the raw mic signal to the page, which zero-fills each chunk
 *   once it has been copied into the current 1 s block.
 */

type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

function makeCanvas(w: number, h: number): { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } {
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d", { willReadFrequently: true, alpha: false });
    if (ctx) return { canvas, ctx };
  }
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true, alpha: false });
  if (!ctx) throw new Error("2D canvas unavailable");
  return { canvas, ctx };
}

/** Called once per derived pixel buffer that has been zero-filled or cleared (governance counter). */
export type PurgeNote = () => void;

/** Reads pixels out of a frame for the light estimate and face crops, wiping every copy after use. */
export class PixelReader {
  private full: { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } | null = null;
  private crop: { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: Canvas2D } | null = null;

  constructor(private readonly onPurge: PurgeNote) {}

  /**
   * Luminance of every `stride`-th pixel of every `stride`-th row (the Python edge's `gray[::stride, ::stride]`), as
   * a ceil(h / stride) x ceil(w / stride) Uint8Array, using OpenCV's BGR2GRAY arithmetic (`toGray`). The full-size
   * RGBA copy is zero-filled and the canvas cleared before returning.
   */
  lumaStrided(
    frame: CanvasImageSource, w: number, h: number, stride: number,
    toGray: (r: number, g: number, b: number) => number,
  ): { gray: Uint8Array; w: number; h: number } {
    if (!this.full || this.full.canvas.width !== w || this.full.canvas.height !== h) this.full = makeCanvas(w, h);
    const { ctx } = this.full;
    ctx.drawImage(frame, 0, 0, w, h);
    const image = ctx.getImageData(0, 0, w, h);
    const px = image.data;
    const ow = Math.ceil(w / stride);
    const oh = Math.ceil(h / stride);
    const gray = new Uint8Array(ow * oh);
    for (let r = 0, i = 0; r < oh; r++) {
      const row = r * stride * w;
      for (let c = 0; c < ow; c++, i++) {
        const p = (row + c * stride) * 4;
        gray[i] = toGray(px[p]!, px[p + 1]!, px[p + 2]!);
      }
    }
    px.fill(0);
    ctx.clearRect(0, 0, w, h);
    this.onPurge();
    return { gray, w: ow, h: oh };
  }

  /** Draw the box (pixels, x2/y2 exclusive) of `frame` resized to size x size; call `wipeCrop` when done. */
  cropTo(frame: CanvasImageSource, box: readonly [number, number, number, number], size: number): OffscreenCanvas | HTMLCanvasElement {
    if (!this.crop || this.crop.canvas.width !== size) this.crop = makeCanvas(size, size);
    const [x1, y1, x2, y2] = box;
    this.crop.ctx.drawImage(frame, x1, y1, x2 - x1, y2 - y1, 0, 0, size, size);
    return this.crop.canvas;
  }

  wipeCrop(): void {
    if (!this.crop) return;
    this.crop.ctx.clearRect(0, 0, this.crop.canvas.width, this.crop.canvas.height);
    this.onPurge();
  }

  close(): void {
    for (const c of [this.full, this.crop]) if (c) c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height);
    if (this.full) this.full.canvas.width = this.full.canvas.height = 0;
    if (this.crop) this.crop.canvas.width = this.crop.canvas.height = 0;
    this.full = this.crop = null;
  }
}

/** Pulls frames from a camera track: one ImageBitmap per call, which the caller must close(). */
export class VideoFrames {
  private video: HTMLVideoElement | null = null;

  constructor(private readonly track: MediaStreamTrack) {}

  async open(timeoutMs = 10000): Promise<void> {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.srcObject = new MediaStream([this.track]);
    this.video = video;
    // Not awaited: play() settles only once a frame has arrived, so a camera that never sends one would hang here
    // instead of reaching the deadline below.
    void video.play().catch(() => undefined);
    const deadline = performance.now() + timeoutMs;
    while (video.readyState < 2 || !video.videoWidth) {
      if (this.video !== video) throw new Error("the camera was closed");
      if (performance.now() > deadline) throw new Error("the camera delivered no frames");
      if (this.track.readyState === "ended") throw new Error("the camera track ended");
      await new Promise((r) => setTimeout(r, 50));
    }
  }

  get live(): boolean {
    return this.track.readyState === "live";
  }

  /** The current frame as an ImageBitmap, or null when none is available yet. */
  async grab(): Promise<ImageBitmap | null> {
    const v = this.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return null;
    return createImageBitmap(v);
  }

  close(): void {
    if (!this.video) return;
    this.video.pause();
    this.video.srcObject = null;
    this.video.removeAttribute("src");
    this.video = null;
  }
}

/** User activations that let a suspended AudioContext start (a touch activates on pointerup, not pointerdown). */
const GESTURES = ["pointerdown", "pointerup", "keydown", "click"] as const;
/** How long open() waits for a suspended context to start by itself. */
const RESUME_WAIT_MS = 1000;

/**
 * Mic tap: ~100 ms raw chunks at the context's native rate, delivered to `onChunk` (which must zero-fill them).
 * `onState` hears the context's state changes after open(): "suspended" (or Safari's "interrupted", e.g. an iOS call
 * or another app taking the audio session) stops the chunks until a gesture or the browser resumes it.
 */
export class MicTap {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private resumeOnGesture: (() => void) | null = null;

  constructor(
    private readonly track: MediaStreamTrack,
    private readonly workletUrl: string,
    private readonly onChunk: (chunk: Float32Array, sampleRate: number) => void,
    private readonly onState?: (state: string) => void,
  ) {}

  /** "running", or "suspended" while the browser waits for a user gesture to allow audio. */
  get state(): AudioContextState | "closed" {
    return this.ctx?.state ?? "closed";
  }

  async open(): Promise<void> {
    const ctx = new AudioContext({ latencyHint: "playback" });
    this.ctx = ctx;
    if (!ctx.audioWorklet) throw new Error("AudioWorklet is not supported in this browser");
    await ctx.audioWorklet.addModule(this.workletUrl);
    const node = new AudioWorkletNode(ctx, "valence-pcm-tap", {
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1, channelCountMode: "explicit",
      processorOptions: { chunkFrames: Math.round(ctx.sampleRate / 10) },
    });
    node.port.onmessage = (e: MessageEvent<Float32Array>) => this.onChunk(e.data, ctx.sampleRate);
    this.node = node;
    this.source = ctx.createMediaStreamSource(new MediaStream([this.track]));
    this.source.connect(node);
    if (ctx.state === "suspended") {
      // Chromium settles resume() only once the context runs, i.e. never without a user activation: wait a little,
      // then retry on every gesture until it runs (one that is not an activation, like a touch's pointerdown, leaves
      // it suspended).
      const wait = new Promise((r) => setTimeout(r, RESUME_WAIT_MS));
      await Promise.race([ctx.resume(), wait]).catch(() => undefined);
      if (this.ctx !== ctx) return; // closed meanwhile (stopped, or the mic ended): arm nothing on a dead context
      if (ctx.state === "suspended") this.resumeOnGestures(ctx);
    }
    ctx.onstatechange = () => {
      if (this.ctx !== ctx) return;
      const state = ctx.state as string;
      if (state === "suspended" || state === "interrupted") {
        // The worklet's partial chunk holds audio from before the gap: wipe it rather than send it with what follows.
        this.node?.port.postMessage("flush");
        this.resumeOnGestures(ctx);
        void ctx.resume().catch(() => undefined);
      } else if (state === "running") {
        this.stopResumeOnGesture();
      }
      this.onState?.(state);
    };
  }

  private resumeOnGestures(ctx: AudioContext): void {
    if (this.resumeOnGesture || typeof document === "undefined") return;
    const resume = () => void ctx.resume().then(() => {
      if (ctx.state === "running") this.stopResumeOnGesture();
    }, () => undefined);
    this.resumeOnGesture = resume;
    for (const ev of GESTURES) document.addEventListener(ev, resume);
  }

  private stopResumeOnGesture(): void {
    if (!this.resumeOnGesture) return;
    for (const ev of GESTURES) document.removeEventListener(ev, this.resumeOnGesture);
    this.resumeOnGesture = null;
  }

  close(): void {
    this.node?.port.postMessage("stop");
    // The port keeps its handler: chunks the worklet posted before "stop" still arrive and onChunk zero-fills them
    // (without a handler they would be dropped with their samples intact).
    this.source?.disconnect();
    this.node?.disconnect();
    this.stopResumeOnGesture();
    if (this.ctx) this.ctx.onstatechange = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = this.node = this.source = null;
  }
}
