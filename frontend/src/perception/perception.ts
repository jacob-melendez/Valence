/**
 * createPerception(): MediaPipe Tasks Web perception on a MediaStream, producing the same observations as the
 * Python edge (valence/edge/run.py): one VideoObs per processed frame at `fps` (10 Hz) and one AudioObs per 1 s mic
 * block. Port of the edge runner's loops:
 * - video: a timer at the perception rate takes the current camera frame as an ImageBitmap and hands it to the video
 *   pipeline (video.ts) in a Web Worker (visionHost.ts; main thread where workers can't run it), which closes it;
 *   one frame in flight at most, and when behind it does not accumulate debt (next_t resets to now);
 * - audio: an AudioWorklet delivers ~100 ms raw chunks; 1 s blocks are resampled to 16 kHz and analysed (audio.ts).
 * Every raw buffer is zero-filled or closed right after its step and counted (counters()), and nothing leaves memory.
 *
 * Robustness: no camera or no mic runs the other alone; a tab in the background pauses video (browsers throttle its
 * timers and may stop decoding) while audio continues; a track that ends (permission revoked, device unplugged) stops
 * its loop and is reported in status().degraded. Models run on the CPU delegate by default; a GPU delegate (opt-in)
 * that fails to start falls back to CPU.
 */
import { AudioPipeline } from "./audio";
import { MicTap, VideoFrames } from "./capture";
import { Hub } from "./hub";
import { assetDownloadBytes, loadAudioModels } from "./models";
import { resampleTo16k } from "./resample";
import {
  emptyCounters, idleStatus, type Obs, type Perception, type PerceptionCounters, type PerceptionOptions,
  type PerceptionStatus, type VideoObs,
} from "./types";
import { createVisionBackend, type VisionBackend } from "./visionHost";

const join = (base: string, rel: string) => `${base.endsWith("/") ? base : `${base}/`}${rel}`;

/** The browser-facing pieces, injectable so tests can run the orchestration without a camera or MediaPipe. */
export interface PerceptionDeps {
  visionBackend: typeof createVisionBackend;
  audioModels: typeof loadAudioModels;
  videoFrames: (track: MediaStreamTrack) => Pick<VideoFrames, "open" | "grab" | "close" | "live">;
  micTap: (track: MediaStreamTrack, workletUrl: string, onChunk: (chunk: Float32Array, sampleRate: number) => void) =>
    Pick<MicTap, "open" | "close" | "state">;
  /** document.hidden (the tab is in the background). */
  hidden: () => boolean;
}

const BROWSER: PerceptionDeps = {
  visionBackend: createVisionBackend,
  audioModels: loadAudioModels,
  videoFrames: (track) => new VideoFrames(track),
  micTap: (track, url, onChunk) => new MicTap(track, url, onChunk),
  hidden: () => typeof document !== "undefined" && document.hidden,
};
/**
 * Readable text for a failure. MediaPipe's script loader rejects with the <script> element's error Event when a WASM
 * loader file cannot be loaded (404, offline, CSP), which String() turns into "[object Event]".
 */
export function message(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof Event !== "undefined" && e instanceof Event) {
    if (typeof ErrorEvent !== "undefined" && e instanceof ErrorEvent && e.message) return e.message;
    const src = (e.target as { src?: unknown } | null)?.src;
    return typeof src === "string" && src ? `could not load ${src}` : `${e.type} event`;
  }
  return String(e);
}

class BrowserPerception implements Perception {
  private readonly obsHub = new Hub<Obs>();
  private readonly statusHub = new Hub<PerceptionStatus>();
  private st: PerceptionStatus = idleStatus(false);
  private cnt: PerceptionCounters = emptyCounters();
  private gen = 0;
  private startedAt = 0;
  private lastStatusEmit = 0;
  private cleanups: (() => void)[] = [];
  // video
  private videoSeq = 0;
  private lastVideoT: number | null = null;
  // audio
  private block: Float32Array | null = null;
  /** Latest observations (scalars only), for the DevTools handle. */
  readonly last: { video: Obs | null; audio: Obs | null } = { video: null, audio: null };
  private blockN = 0;
  private audioSeq = 0;

  constructor(
    private readonly opts: PerceptionOptions,
    private readonly deps: PerceptionDeps = BROWSER,
  ) {}

  onObservation = (cb: (obs: Obs) => void) => this.obsHub.on(cb);
  onStatus = (cb: (s: PerceptionStatus) => void) => this.statusHub.on(cb);
  counters = (): PerceptionCounters => ({ ...this.cnt });
  status = (): PerceptionStatus => this.st;

  private setStatus(patch: Partial<PerceptionStatus>, emit = true): void {
    this.st = { ...this.st, ...patch };
    if (emit) {
      this.lastStatusEmit = performance.now();
      this.statusHub.emit(this.st);
    }
  }

  private degrade(reason: string, on: boolean): void {
    const has = this.st.degraded.includes(reason);
    if (on && !has) this.setStatus({ degraded: [...this.st.degraded, reason] });
    if (!on && has) this.setStatus({ degraded: this.st.degraded.filter((d) => d !== reason) });
  }

  private emit(obs: Obs, gen: number): void {
    if (gen !== this.gen) return;
    this.last[obs.type] = obs;
    if (this.st.load_ms === null) {
      this.setStatus({ load_ms: Math.round(performance.now() - this.startedAt), download_bytes: this.downloadBytes() });
    }
    this.obsHub.emit(obs);
    if (performance.now() - this.lastStatusEmit > 1000) this.setStatus({ download_bytes: this.downloadBytes() });
  }

  private downloadBytes(): number {
    return assetDownloadBytes(this.opts.assetBase ?? "/mediapipe/");
  }

  async start(stream?: MediaStream | null): Promise<void> {
    if (this.st.phase === "running" || this.st.phase === "loading") return;
    const live = (tracks: MediaStreamTrack[] | undefined) => tracks?.find((t) => t.readyState === "live");
    const vTrack = this.opts.video === false ? undefined : live(stream?.getVideoTracks());
    const aTrack = this.opts.audio === false ? undefined : live(stream?.getAudioTracks());
    if (!vTrack && !aTrack) throw new Error("perception needs a MediaStream with a live camera or mic track");
    const gen = ++this.gen;
    this.cnt = emptyCounters();
    this.startedAt = performance.now();
    this.videoSeq = this.audioSeq = 0;
    this.lastVideoT = null;
    this.setStatus({
      ...idleStatus(false), phase: "loading",
      degraded: [...(vTrack ? [] : ["no camera"]), ...(aTrack ? [] : ["no microphone"])],
    });
    const [video, audio] = await Promise.allSettled([
      vTrack ? this.startVideo(vTrack, gen) : Promise.resolve(false),
      aTrack ? this.startAudio(aTrack, gen) : Promise.resolve(false),
    ]);
    if (gen !== this.gen) return; // stopped while loading
    const failures: string[] = [];
    if (video.status === "rejected") failures.push(`camera: ${message(video.reason)}`);
    if (audio.status === "rejected") failures.push(`microphone: ${message(audio.reason)}`);
    const camera = video.status === "fulfilled" && video.value;
    const mic = audio.status === "fulfilled" && audio.value;
    if (!camera && !mic) {
      this.teardown();
      const error = failures.join("; ") || "nothing to process";
      this.setStatus({ phase: "error", error, camera: false, mic: false });
      throw new Error(`perception failed to start: ${error}`);
    }
    this.setStatus({
      phase: "running", camera, mic, degraded: [...this.st.degraded, ...failures], download_bytes: this.downloadBytes(),
    });
  }

  stop(): void {
    this.gen++;
    this.teardown();
    // The kill switch leaves no observation behind either (the DevTools handle reads `last`).
    this.last.video = this.last.audio = null;
    if (this.st.phase !== "idle") this.setStatus({ phase: "stopped", camera: false, mic: false, fps: 0 });
  }

  private teardown(): void {
    for (const c of this.cleanups.splice(0).reverse()) {
      try {
        c();
      } catch (e) {
        console.warn("[perception] cleanup failed", e);
      }
    }
    this.dropBlock();
  }

  /** Runs `c` at teardown; the returned function runs it now instead (once). */
  private onTeardown(c: () => void): () => void {
    let done = false;
    const run = () => {
      if (done) return;
      done = true;
      c();
    };
    this.cleanups.push(run);
    return run;
  }

  /** Zero-fills and drops the partial 1 s audio block. */
  private dropBlock(): void {
    if (this.block) {
      this.block.fill(0);
      this.block = null;
      this.blockN = 0;
    }
  }

  // ---------- video ----------

  private async startVideo(track: MediaStreamTrack, gen: number): Promise<boolean> {
    const cnt = this.cnt; // this run's counters: its late reports must not land in a restarted run's
    const fps = this.opts.fps ?? 10;
    const backend = await this.deps.visionBackend({
      assetBase: this.opts.assetBase ?? "/mediapipe/", maxPeople: this.opts.maxPeople ?? 3,
      minPersonH: this.opts.minPersonH ?? 0.25, onDerivedPurged: (n) => (cnt.derived_purged += n),
      faceEveryFrame: this.opts.faceEveryFrame ?? true,
      ...(this.opts.delegate ? { delegate: this.opts.delegate } : {}),
      ...(this.opts.worker !== undefined ? { worker: this.opts.worker } : {}),
    });
    if (gen !== this.gen) {
      backend.close();
      return false;
    }
    const closeBackend = this.onTeardown(() => backend.close());
    const frames = this.deps.videoFrames(track);
    const closeFrames = this.onTeardown(() => frames.close());
    try {
      await frames.open();
    } catch (e) {
      closeFrames(); // no frames: release the camera element and the models now, not at stop()
      closeBackend();
      throw e;
    }
    if (gen !== this.gen) return false; // stopped while the camera opened (teardown released both)
    this.setStatus({ video_backend: `${backend.where}/${backend.delegate}` }, false);
    if (backend.delegate === "CPU" && this.opts.delegate === "GPU") this.degrade("GPU unavailable: CPU delegate", true);

    const onEnded = () => {
      this.degrade("camera stopped", true);
      this.setStatus({ camera: false });
    };
    track.addEventListener("ended", onEnded);
    this.cleanups.push(() => track.removeEventListener("ended", onEnded));

    // Port of EdgeRunner._video_loop: a fixed rate, the newest frame only, no debt when behind.
    const period = 1000 / fps;
    let nextT = performance.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    this.cleanups.push(() => clearTimeout(timer));
    const schedule = (delay: number) => {
      if (gen === this.gen) timer = setTimeout(() => void tick(), delay);
    };
    const tick = async () => {
      if (gen !== this.gen || !frames.live) return;
      const hidden = this.deps.hidden();
      this.degrade("tab hidden: video paused", hidden);
      if (hidden) {
        nextT = performance.now() + period;
        schedule(500);
        return;
      }
      let bitmap: ImageBitmap | null = null;
      try {
        bitmap = await frames.grab();
      } catch (e) {
        console.warn("[perception] frame grab failed", e);
      }
      if (bitmap && gen !== this.gen) bitmap.close();
      else if (bitmap) await this.videoStep(bitmap, backend, gen, cnt);
      nextT += period;
      let delay = nextT - performance.now();
      if (delay < 0) {
        nextT = performance.now(); // behind: don't accumulate debt
        delay = 0;
      }
      schedule(delay);
    };
    schedule(0);
    return true;
  }

  private async videoStep(
    bitmap: ImageBitmap, backend: VisionBackend, gen: number, cnt: PerceptionCounters,
  ): Promise<void> {
    const t = performance.now() / 1000;
    const wall = Date.now() / 1000;
    cnt.frames_decoded += 1;
    cnt.live_raw_buffers += 1;
    const t0 = performance.now();
    let obs: VideoObs | null = null;
    try {
      obs = await backend.process(bitmap, t, wall, this.videoSeq, cnt.frames_decoded);
    } catch (e) {
      if (gen === this.gen) {
        console.error("[perception] video step failed", e);
        this.degrade("video step failing", true);
      }
    } finally {
      // The backend closed the bitmap (in the worker after the transfer, or here on the main thread).
      cnt.frames_purged += 1;
      cnt.live_raw_buffers -= 1;
    }
    if (cnt.live_raw_buffers !== 0) {
      cnt.violations += cnt.live_raw_buffers;
      cnt.live_raw_buffers = 0;
    }
    if (gen !== this.gen) return; // stopped (maybe restarted) meanwhile: the rest belongs to the current run
    if (obs) {
      this.videoSeq += 1;
      this.degrade("video step failing", false);
    }
    const ms = performance.now() - t0;
    const hz = this.lastVideoT !== null && t > this.lastVideoT ? 1 / (t - this.lastVideoT) : 0;
    this.lastVideoT = t;
    const fps = hz ? (this.st.fps ? 0.9 * this.st.fps + 0.1 * hz : hz) : this.st.fps;
    this.setStatus({
      video_ms: this.st.video_ms ? 0.9 * this.st.video_ms + 0.1 * ms : ms,
      fps, face_hz: (this.opts.faceEveryFrame ?? true) ? fps : fps / 2,
    }, false);
    if (obs) this.emit(obs, gen);
  }

  // ---------- audio ----------

  private async startAudio(track: MediaStreamTrack, gen: number): Promise<boolean> {
    const cnt = this.cnt; // this run's counters (see startVideo)
    const base = this.opts.assetBase ?? "/mediapipe/";
    const models = await this.deps.audioModels(base);
    if (gen !== this.gen) {
      models.close();
      return false;
    }
    const closeModels = this.onTeardown(() => models.close());
    const pipeline = new AudioPipeline(models, { splOffsetDb: this.opts.splOffsetDb ?? 94 });
    const tap = this.deps.micTap(track, join(base, "pcm-tap.worklet.js"),
      (chunk, sr) => this.onChunk(chunk, sr, pipeline, gen, cnt, track));
    const closeTap = this.onTeardown(() => tap.close());
    try {
      await tap.open();
    } catch (e) {
      closeTap(); // release the AudioContext and YAMNet now, not at stop()
      closeModels();
      throw e;
    }
    if (gen !== this.gen) return false; // stopped while the tap opened (teardown closed it)
    if (tap.state === "suspended") this.degrade("audio waits for a click on the page", true);
    const onEnded = () => {
      // An ended track still feeds the worklet (silence): close the tap and drop the partial block rather than
      // report a silent room.
      closeTap();
      this.dropBlock();
      this.degrade("microphone stopped", true);
      this.setStatus({ mic: false });
    };
    track.addEventListener("ended", onEnded);
    this.cleanups.push(() => track.removeEventListener("ended", onEnded));
    return true;
  }

  /** One ~100 ms raw chunk: copy into the 1 s block, zero-fill it; a full block is analysed and zero-filled too. */
  private onChunk(
    chunk: Float32Array, sampleRate: number, pipeline: AudioPipeline, gen: number, cnt: PerceptionCounters,
    track: MediaStreamTrack,
  ): void {
    cnt.live_raw_buffers += 1;
    try {
      if (gen !== this.gen || track.readyState !== "live") return; // late chunks (stopped, mic ended): purge only
      this.degrade("audio waits for a click on the page", false);
      if (!this.block || this.block.length !== sampleRate) {
        this.block?.fill(0);
        this.block = new Float32Array(sampleRate);
        this.blockN = 0;
      }
      let i = 0;
      while (i < chunk.length) {
        const take = Math.min(this.block.length - this.blockN, chunk.length - i);
        this.block.set(chunk.subarray(i, i + take), this.blockN);
        this.blockN += take;
        i += take;
        if (this.blockN === this.block.length) this.audioStep(this.block, sampleRate, pipeline, gen, cnt);
      }
    } finally {
      chunk.fill(0);
      cnt.audio_blocks_purged += 1;
      cnt.live_raw_buffers -= 1;
    }
  }

  private audioStep(
    block: Float32Array, sampleRate: number, pipeline: AudioPipeline, gen: number, cnt: PerceptionCounters,
  ): void {
    const t = performance.now() / 1000;
    const wall = Date.now() / 1000;
    const t0 = performance.now();
    const at16k = resampleTo16k(block, sampleRate);
    block.fill(0);
    this.blockN = 0;
    cnt.derived_purged += 1;
    let obs = null;
    try {
      obs = pipeline.process(at16k, t, wall, this.audioSeq);
      this.audioSeq += 1;
      this.degrade("audio step failing", false);
    } catch (e) {
      console.error("[perception] audio step failed", e);
      this.degrade("audio step failing", true);
    } finally {
      at16k.fill(0);
      cnt.derived_purged += 1;
    }
    const ms = performance.now() - t0;
    this.setStatus({ audio_ms: this.st.audio_ms ? 0.9 * this.st.audio_ms + 0.1 * ms : ms }, false);
    if (obs) this.emit(obs, gen);
  }
}

/**
 * `?delegate=CPU` or `?delegate=GPU` on the page URL picks the MediaPipe delegate when the caller does not, so the
 * real-camera check (frontend/e2e/REAL_CAMERA.md) can compare the two on the deployed console without a code change.
 */
function urlDelegate(): PerceptionOptions["delegate"] | undefined {
  try {
    const d = new URLSearchParams(globalThis.location?.search ?? "").get("delegate");
    return d === "CPU" || d === "GPU" ? d : undefined;
  } catch {
    return undefined;
  }
}

export function createPerception(opts: PerceptionOptions = {}, deps?: Partial<PerceptionDeps>): Perception {
  const delegate = opts.delegate ?? urlDelegate() ?? "CPU";
  const p = new BrowserPerception({ ...opts, delegate }, { ...BROWSER, ...deps });
  // DevTools handle for the real-camera check (frontend/e2e/REAL_CAMERA.md): status, counters and the latest
  // observations, all scalars; no frame or sample is reachable from it.
  (globalThis as { __valencePerception?: unknown }).__valencePerception = {
    status: () => p.status(), counters: () => p.counters(), last: () => ({ ...p.last }),
  };
  return p;
}
