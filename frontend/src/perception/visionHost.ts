/**
 * Where the video pipeline runs: in a Web Worker (default when the browser has Worker + OffscreenCanvas, keeping
 * MediaPipe's ~10-100 ms per frame off the console's main thread) or inline on the main thread (fallback, tests).
 * Either way one ImageBitmap goes in per step and is closed after processing; the caller counts it as purged when
 * `process` settles.
 */
import { makeCanvas, PixelReader } from "./capture";
import { loadVisionModels, type Delegate } from "./models";
import type { VideoObs } from "./types";
import { VisionPipeline } from "./video";

export type WorkerRequest =
  | { kind: "init"; assetBase: string; delegate?: Delegate; maxPeople: number; minPersonH: number; faceEveryFrame: boolean }
  | { kind: "frame"; id: number; bitmap: ImageBitmap; t: number; wall: number; seq: number; decoded: number }
  | { kind: "close" };

export type WorkerReply =
  | { kind: "started" }
  | { kind: "ready"; delegate: Delegate }
  | { kind: "error"; message: string }
  | { kind: "obs"; id: number; obs: VideoObs; derived: number }
  | { kind: "frame-error"; id: number; message: string; derived: number };

export interface VisionBackendOptions {
  assetBase: string;
  delegate?: Delegate;
  maxPeople: number;
  minPersonH: number;
  faceEveryFrame?: boolean;
  /** Called with the number of derived pixel buffers wiped while processing. */
  onDerivedPurged: (n: number) => void;
}

export interface VisionBackend {
  readonly where: "worker" | "main";
  readonly delegate: Delegate;
  /** Process and close `bitmap` (ownership passes here). Rejects if the step failed; the bitmap is closed anyway. */
  process(bitmap: ImageBitmap, t: number, wall: number, seq: number, decoded: number): Promise<VideoObs>;
  close(): void;
}

export async function inlineBackend(opts: VisionBackendOptions): Promise<VisionBackend> {
  const models = await loadVisionModels(opts.assetBase, {
    ...(opts.delegate ? { delegate: opts.delegate } : {}), maxPeople: opts.maxPeople,
  });
  const reader = new PixelReader(() => opts.onDerivedPurged(1));
  const pipeline = new VisionPipeline(models, reader, {
    maxPeople: opts.maxPeople, minPersonH: opts.minPersonH, faceEveryFrame: opts.faceEveryFrame ?? false,
  });
  return {
    where: "main",
    delegate: models.delegate,
    async process(bitmap, t, wall, seq, decoded) {
      try {
        return pipeline.process({ image: bitmap, width: bitmap.width, height: bitmap.height }, t, wall, seq, decoded);
      } finally {
        bitmap.close();
      }
    },
    close() {
      // The worker's heap dies with it; this thread's stays, so overwrite MediaPipe's copy of the last frame first.
      const blanks: { width: number; height: number }[] = [];
      pipeline.scrub((w, h) => {
        const { canvas, ctx } = makeCanvas(w, h);
        ctx.fillRect(0, 0, w, h);
        blanks.push(canvas);
        return canvas;
      });
      for (const c of blanks) c.width = c.height = 0;
      models.close();
      reader.close();
    },
  };
}

export async function workerBackend(opts: VisionBackendOptions, timeoutMs = 60000): Promise<VisionBackend> {
  const worker = new Worker(new URL("./vision.worker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, { resolve: (o: VideoObs) => void; reject: (e: Error) => void }>();
  let nextId = 1;
  const delegate = await new Promise<Delegate>((resolve, reject) => {
    // Times the worker's own start only: the model download after "started" can take minutes on a slow network,
    // and giving up then would load everything again on the main thread.
    const timer = setTimeout(() => reject(new Error("vision worker did not start")), timeoutMs);
    worker.onerror = (e) => {
      clearTimeout(timer);
      reject(new Error(`vision worker failed: ${e.message || "script error"}`));
    };
    worker.onmessage = (e: MessageEvent<WorkerReply>) => {
      if (e.data.kind === "started") {
        clearTimeout(timer);
      } else if (e.data.kind === "ready") {
        clearTimeout(timer);
        resolve(e.data.delegate);
      } else if (e.data.kind === "error") {
        clearTimeout(timer);
        reject(new Error(e.data.message));
      }
    };
    const init: WorkerRequest = {
      kind: "init", assetBase: new URL(opts.assetBase, location.href).href, maxPeople: opts.maxPeople,
      minPersonH: opts.minPersonH, faceEveryFrame: opts.faceEveryFrame ?? false,
      ...(opts.delegate ? { delegate: opts.delegate } : {}),
    };
    worker.postMessage(init);
  }).catch((e: unknown) => {
    worker.terminate();
    throw e;
  });
  worker.onmessage = (e: MessageEvent<WorkerReply>) => {
    const msg = e.data;
    if (msg.kind !== "obs" && msg.kind !== "frame-error") return;
    opts.onDerivedPurged(msg.derived);
    const p = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.kind === "obs") p?.resolve(msg.obs);
    else p?.reject(new Error(msg.message));
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(`vision worker crashed: ${e.message}`));
    pending.clear();
  };
  return {
    where: "worker",
    delegate,
    process(bitmap, t, wall, seq, decoded) {
      const id = nextId++;
      return new Promise<VideoObs>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        const req: WorkerRequest = { kind: "frame", id, bitmap, t, wall, seq, decoded };
        worker.postMessage(req, [bitmap]); // transferred: the page no longer holds the frame
      });
    },
    close() {
      for (const p of pending.values()) p.reject(new Error("vision worker closed"));
      pending.clear();
      worker.postMessage({ kind: "close" } satisfies WorkerRequest);
      setTimeout(() => worker.terminate(), 1000);
    },
  };
}

/** Worker when the browser has Worker, OffscreenCanvas and createImageBitmap (and the worker starts), else inline. */
export async function createVisionBackend(opts: VisionBackendOptions & { worker?: boolean }): Promise<VisionBackend> {
  const canWorker = typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined"
    && typeof createImageBitmap !== "undefined";
  if ((opts.worker ?? true) && canWorker) {
    try {
      return await workerBackend(opts);
    } catch (e) {
      console.warn("[perception] vision worker unavailable, running on the main thread", e);
    }
  }
  return inlineBackend(opts);
}
