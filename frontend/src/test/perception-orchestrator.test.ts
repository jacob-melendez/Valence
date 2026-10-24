import { afterEach, describe, expect, it, vi } from "vitest";
import { MicTap } from "@/perception/capture";
import { createPerception, type PerceptionDeps } from "@/perception/perception";
import type { Obs, PerceptionStatus, VideoObs } from "@/perception/types";
import { workerBackend, type VisionBackend, type WorkerReply, type WorkerRequest } from "@/perception/visionHost";

// The orchestration (loops, purging, counters, robustness) with the browser pieces faked: no camera, no MediaPipe.

function track(kind: "video" | "audio") {
  const listeners = new Map<string, () => void>();
  return {
    kind, readyState: "live" as MediaStreamTrackState, listeners,
    addEventListener: (ev: string, cb: () => void) => listeners.set(ev, cb),
    removeEventListener: (ev: string) => listeners.delete(ev),
    end() {
      this.readyState = "ended";
      listeners.get("ended")?.();
    },
  };
}

function stream(...tracks: ReturnType<typeof track>[]): MediaStream {
  return {
    getVideoTracks: () => tracks.filter((t) => t.kind === "video"),
    getAudioTracks: () => tracks.filter((t) => t.kind === "audio"),
    getTracks: () => tracks,
  } as unknown as MediaStream;
}

function harness(opts: { visionFails?: boolean; audioFails?: boolean } = {}) {
  const state = { hidden: false, bitmapsClosed: 0, bitmapsMade: 0, backendClosed: 0, tapClosed: 0, live: true };
  let onChunk: ((chunk: Float32Array, sr: number) => void) | null = null;
  const backend: VisionBackend = {
    where: "main", delegate: "CPU",
    async process(bitmap, t, wall, seq, decoded) {
      bitmap.close();
      const obs: VideoObs = {
        type: "video", t, wall, frame_seq: seq, frames_decoded: decoded, lux: 300, lux_quality: 0.4, frame_w: 640,
        frame_h: 360, mirrored: false, persons: [],
      };
      return obs;
    },
    close: () => void state.backendClosed++,
  };
  const deps: Partial<PerceptionDeps> = {
    visionBackend: async () => {
      if (opts.visionFails) throw new Error("no WebGL");
      return backend;
    },
    audioModels: async () => {
      if (opts.audioFails) throw new Error("yamnet missing");
      return { classify: () => [], close: () => {} };
    },
    videoFrames: () => ({
      open: async () => {},
      grab: async () => {
        state.bitmapsMade++;
        return { width: 640, height: 360, close: () => void state.bitmapsClosed++ } as unknown as ImageBitmap;
      },
      close: () => {},
      get live() {
        return state.live;
      },
    }),
    micTap: (_t, _url, cb) => {
      onChunk = cb;
      return { open: async () => {}, close: () => void state.tapClosed++, state: "running" as const };
    },
    hidden: () => state.hidden,
  };
  /** Push `n` 100 ms chunks of a 48 kHz sine; returns them so the test can check they were zero-filled. */
  const pushAudio = (n: number) =>
    Array.from({ length: n }, () => {
      const chunk = Float32Array.from({ length: 4800 }, (_, i) => 0.1 * Math.sin(i / 7));
      onChunk!(chunk, 48000);
      return chunk;
    });
  return { deps, state, pushAudio };
}

const collect = (p: ReturnType<typeof createPerception>) => {
  const seen: Obs[] = [];
  p.onObservation((o) => seen.push(o));
  return seen;
};
const videos = (seen: Obs[]) => seen.filter((o) => o.type === "video");

afterEach(() => vi.useRealTimers());

describe("createPerception orchestration", () => {
  it("runs video at the set rate and 1 s audio blocks; every buffer is purged and counted", async () => {
    const { deps, state, pushAudio } = harness();
    const p = createPerception({ fps: 20 }, deps);
    const seen = collect(p);
    await p.start(stream(track("video"), track("audio")));
    expect(p.status()).toMatchObject({ phase: "running", camera: true, mic: true, video_backend: "main/CPU" });
    await vi.waitFor(() => expect(videos(seen).length).toBeGreaterThanOrEqual(5), { timeout: 3000 });
    const chunks = pushAudio(25);
    const audio = seen.filter((o) => o.type === "audio");
    expect(audio).toHaveLength(2); // 25 x 100 ms = two full blocks
    expect(audio[0]).toMatchObject({ block_seq: 0, compliment_hits: 0, laughter_onsets: 0 });
    expect(Number.isFinite((audio[0] as { db_spl: number }).db_spl)).toBe(true);
    expect(chunks.every((c) => c.every((v) => v === 0))).toBe(true);
    p.stop();
    const n = seen.length;
    await new Promise((r) => setTimeout(r, 150)); // an in-flight frame settles; nothing new is emitted
    expect(seen.length).toBe(n);
    const c = p.counters();
    expect(c.frames_decoded).toBe(c.frames_purged);
    expect(c.frames_decoded).toBe(state.bitmapsMade);
    expect(state.bitmapsClosed).toBe(state.bitmapsMade);
    expect(c.audio_blocks_purged).toBe(25);
    expect(c.derived_purged).toBe(4); // per block: the native-rate block and its 16 kHz copy
    expect(c.live_raw_buffers).toBe(0);
    expect(c.violations).toBe(0);
    const v = videos(seen) as VideoObs[];
    expect(v.map((o) => o.frame_seq)).toEqual(v.map((_, i) => i));
    expect(v.at(-1)!.frames_decoded).toBeLessThanOrEqual(c.frames_decoded);
    expect(p.status().phase).toBe("stopped");
    expect(state.backendClosed).toBe(1);
    expect(state.tapClosed).toBe(1);
  });

  it("camera denied: runs on the mic alone and says so", async () => {
    const { deps, pushAudio } = harness();
    const p = createPerception({}, deps);
    const seen = collect(p);
    await p.start(stream(track("audio")));
    expect(p.status()).toMatchObject({ phase: "running", camera: false, mic: true });
    expect(p.status().degraded).toContain("no camera");
    pushAudio(10);
    expect(seen.filter((o) => o.type === "audio")).toHaveLength(1);
    p.stop();
  });

  it("no microphone: video alone, flagged", async () => {
    const { deps } = harness();
    const p = createPerception({ fps: 20 }, deps);
    const seen = collect(p);
    await p.start(stream(track("video")));
    expect(p.status()).toMatchObject({ camera: true, mic: false });
    expect(p.status().degraded).toContain("no microphone");
    await vi.waitFor(() => expect(videos(seen).length).toBeGreaterThan(1));
    p.stop();
  });

  it("refuses a stream without live tracks", async () => {
    const { deps } = harness();
    await expect(createPerception({}, deps).start(stream())).rejects.toThrow(/live camera or mic/);
    const ended = track("video");
    ended.readyState = "ended";
    await expect(createPerception({}, deps).start(stream(ended))).rejects.toThrow(/live camera or mic/);
  });

  it("video models failing leaves audio running; both failing rejects with the reasons", async () => {
    const { deps } = harness({ visionFails: true });
    const p = createPerception({}, deps);
    await p.start(stream(track("video"), track("audio")));
    expect(p.status()).toMatchObject({ phase: "running", camera: false, mic: true });
    expect(p.status().degraded.join(" ")).toMatch(/camera: no WebGL/);
    p.stop();
    const both = createPerception({}, harness({ visionFails: true, audioFails: true }).deps);
    await expect(both.start(stream(track("video"), track("audio")))).rejects.toThrow(/no WebGL.*yamnet missing/);
    expect(both.status().phase).toBe("error");
  });

  it("pauses video while the tab is hidden and resumes after", async () => {
    const { deps, state } = harness();
    const p = createPerception({ fps: 20 }, deps);
    const seen = collect(p);
    await p.start(stream(track("video")));
    await vi.waitFor(() => expect(videos(seen).length).toBeGreaterThan(1));
    state.hidden = true;
    await vi.waitFor(() => expect(p.status().degraded).toContain("tab hidden: video paused"));
    const n = videos(seen).length;
    await new Promise((r) => setTimeout(r, 300));
    expect(videos(seen).length).toBeLessThanOrEqual(n + 1);
    state.hidden = false;
    await vi.waitFor(() => expect(videos(seen).length).toBeGreaterThan(n + 1), { timeout: 3000 });
    expect(p.status().degraded).not.toContain("tab hidden: video paused");
    p.stop();
  });

  it("reports a camera track that ends (permission revoked, unplugged)", async () => {
    const { deps, state } = harness();
    const v = track("video");
    const p = createPerception({ fps: 20 }, deps);
    await p.start(stream(v, track("audio")));
    state.live = false;
    v.end();
    expect(p.status().camera).toBe(false);
    expect(p.status().degraded).toContain("camera stopped");
    p.stop();
  });

  it("exposes a scalar-only DevTools handle", async () => {
    const { deps } = harness();
    const p = createPerception({ fps: 20 }, deps);
    await p.start(stream(track("video")));
    const handle = (globalThis as unknown as { __valencePerception: { status: () => unknown; last: () => { video: unknown } } })
      .__valencePerception;
    await vi.waitFor(() => expect(handle.last().video).not.toBeNull());
    expect(handle.status()).toMatchObject({ phase: "running" });
    p.stop();
    expect(handle.last()).toEqual({ video: null, audio: null }); // the kill switch leaves no observation behind
  });

  it("stop() then start() at once: the old run's in-flight frame and late purge reports stay out of the new counters", async () => {
    const { deps } = harness();
    const purgeReports: ((n: number) => void)[] = [];
    // Like the worker backend: a frame stays in flight until it is answered or the backend closes.
    deps.visionBackend = async (o) => {
      purgeReports.push(o.onDerivedPurged);
      const inFlight = new Set<() => void>();
      return {
        where: "worker", delegate: "GPU",
        process: (bitmap) => new Promise<VideoObs>((_, reject) => {
          inFlight.add(() => {
            bitmap.close();
            reject(new Error("vision worker closed"));
          });
        }),
        close: () => {
          for (const r of inFlight) r();
          inFlight.clear();
        },
      };
    };
    const p = createPerception({ fps: 20 }, deps);
    const s = stream(track("video"));
    await p.start(s);
    await vi.waitFor(() => expect(p.counters().live_raw_buffers).toBe(1));
    p.stop();
    await p.start(s); // e.g. switching cameras
    purgeReports[0]!(2); // the old worker answers the frame it had: two pixel copies wiped
    await vi.waitFor(() => expect(p.counters().frames_decoded).toBe(1)); // the new run's first frame, in flight
    expect(p.counters()).toMatchObject({ frames_purged: 0, live_raw_buffers: 1, derived_purged: 0, violations: 0 });
    p.stop();
  });

  it("stop() while the camera or mic is still opening: no late status, no listener left on the tracks", async () => {
    const { deps } = harness();
    const opening: (() => void)[] = [];
    const frames = deps.videoFrames!, tap = deps.micTap!;
    deps.videoFrames = (t) => Object.assign(frames(t), { open: () => new Promise<void>((r) => opening.push(r)) });
    deps.micTap = (t, u, cb) => Object.assign(tap(t, u, cb), { open: () => new Promise<void>((r) => opening.push(r)) });
    const v = track("video"), a = track("audio");
    const p = createPerception({ fps: 20 }, deps);
    const started = p.start(stream(v, a));
    await vi.waitFor(() => expect(opening).toHaveLength(2));
    p.stop();
    const statuses: PerceptionStatus[] = [];
    p.onStatus((st) => statuses.push(st));
    for (const open of opening) open();
    await started;
    expect(statuses).toEqual([]);
    expect(p.status()).toMatchObject({ phase: "stopped", video_backend: null });
    expect(p.status().degraded).not.toContain("no hardware GPU: CPU delegate");
    expect(v.listeners.size + a.listeners.size).toBe(0);
  });

  it("a mic that ends stops the audio: what the tap still delivers is purged, not reported as a silent room", async () => {
    const { deps, state, pushAudio } = harness();
    const a = track("audio");
    const p = createPerception({}, deps);
    const seen = collect(p);
    await p.start(stream(a));
    const before = pushAudio(5); // half a block
    a.end();
    expect(state.tapClosed).toBe(1);
    const after = pushAudio(15); // Chromium feeds the worklet silence from an ended track until the tap closes
    expect(seen.filter((o) => o.type === "audio")).toEqual([]);
    expect([...before, ...after].every((c) => c.every((x) => x === 0))).toBe(true);
    expect(p.counters()).toMatchObject({ audio_blocks_purged: 20, live_raw_buffers: 0 });
    expect(p.status()).toMatchObject({ mic: false });
    expect(p.status().degraded).toContain("microphone stopped");
    p.stop();
  });

  it("a camera that delivers no frames releases the video models and the camera right away, not at stop()", async () => {
    const { deps, state } = harness();
    let framesClosed = 0;
    const frames = deps.videoFrames!;
    deps.videoFrames = (t) => Object.assign(frames(t), {
      open: async () => {
        throw new Error("the camera delivered no frames");
      },
      close: () => void framesClosed++,
    });
    const p = createPerception({}, deps);
    await p.start(stream(track("video"), track("audio")));
    expect(p.status()).toMatchObject({ phase: "running", camera: false, mic: true });
    expect(p.status().degraded.join(" ")).toMatch(/camera: the camera delivered no frames/);
    expect([state.backendClosed, framesClosed]).toEqual([1, 1]);
    p.stop();
    expect([state.backendClosed, framesClosed]).toEqual([1, 1]); // once
  });
});

describe("MicTap (fake Web Audio)", () => {
  function fakeWebAudio(opts: { allowed: boolean }) {
    type Port = { onmessage: ((e: MessageEvent) => void) | null; postMessage: (m: unknown) => void };
    const audio = { allowed: opts.allowed, resumes: 0, nodes: [] as { port: Port }[] };
    class FakeContext {
      state: AudioContextState = audio.allowed ? "running" : "suspended";
      sampleRate = 48000;
      audioWorklet = { addModule: async () => {} };
      createMediaStreamSource() {
        return { connect() {}, disconnect() {} };
      }
      // As in Chromium: resume() settles once the context runs, so never without a user activation.
      resume() {
        audio.resumes += 1;
        if (!audio.allowed) return new Promise<void>(() => {});
        this.state = "running";
        return Promise.resolve();
      }
      close() {
        this.state = "closed";
        return Promise.resolve();
      }
    }
    class FakeNode {
      port: Port = { onmessage: null, postMessage: () => {} };
      constructor() {
        audio.nodes.push(this);
      }
      disconnect() {}
    }
    vi.stubGlobal("AudioContext", FakeContext);
    vi.stubGlobal("AudioWorkletNode", FakeNode);
    vi.stubGlobal("MediaStream", class {
      constructor(readonly tracks: unknown[]) {}
    });
    return audio;
  }
  const mic = () => track("audio") as unknown as MediaStreamTrack;
  afterEach(() => vi.unstubAllGlobals());

  it("chunks still queued when the tap closes reach onChunk, which zero-fills them, instead of being dropped intact", async () => {
    const audio = fakeWebAudio({ allowed: true });
    const got: Float32Array[] = [];
    const tap = new MicTap(mic(), "/mediapipe/pcm-tap.worklet.js", (c) => {
      got.push(c);
      c.fill(0);
    });
    await tap.open();
    const port = audio.nodes[0]!.port;
    tap.close();
    const late = Float32Array.from([0.3, -0.2, 0.1]); // posted by the worklet before it saw "stop"
    port.onmessage?.({ data: late } as MessageEvent);
    expect(got).toEqual([late]);
    expect([...late]).toEqual([0, 0, 0]);
  });

  it("a suspended context: open() does not wait for a user activation, and every gesture retries until audio runs", async () => {
    vi.useFakeTimers();
    const audio = fakeWebAudio({ allowed: false });
    const tap = new MicTap(mic(), "/mediapipe/pcm-tap.worklet.js", () => {});
    let opened = false;
    void tap.open().then(() => (opened = true));
    await vi.advanceTimersByTimeAsync(2000);
    expect(opened).toBe(true);
    expect(tap.state).toBe("suspended");
    const n = audio.resumes;
    document.dispatchEvent(new Event("pointerdown")); // a touch's pointerdown is not a user activation...
    expect(audio.resumes).toBe(n + 1);
    expect(tap.state).toBe("suspended");
    audio.allowed = true;
    document.dispatchEvent(new Event("pointerup")); // ...its pointerup is
    await vi.advanceTimersByTimeAsync(0);
    expect(tap.state).toBe("running");
    document.dispatchEvent(new Event("keydown"));
    expect(audio.resumes).toBe(n + 2); // running: no longer listening
    tap.close();
  });
});

describe("workerBackend (fake Worker)", () => {
  const opts = { assetBase: "/mediapipe/", maxPeople: 3, minPersonH: 0.25, onDerivedPurged: () => {} };
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.doUnmock("@/perception/models");
    vi.resetModules();
  });

  function fakeWorker(onInit: (reply: (m: WorkerReply) => void) => void) {
    const workers: { terminated: boolean }[] = [];
    vi.stubGlobal("Worker", class {
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: ErrorEvent) => void) | null = null;
      terminated = false;
      constructor() {
        workers.push(this);
      }
      postMessage(msg: WorkerRequest) {
        if (msg.kind === "init") onInit((m) => this.onmessage?.({ data: m } as MessageEvent));
      }
      terminate() {
        this.terminated = true;
      }
    });
    return workers;
  }

  it("a worker that has started is not killed by the start timeout while it downloads the models", async () => {
    vi.useFakeTimers();
    const workers = fakeWorker((reply) => {
      setTimeout(() => reply({ kind: "started" }), 10);
      setTimeout(() => reply({ kind: "ready", delegate: "GPU" }), 90_000); // ~40 MB on a slow venue network
    });
    const backend = workerBackend(opts, 60_000);
    const settled = backend.then((b) => b, (e: unknown) => e);
    await vi.advanceTimersByTimeAsync(90_000);
    expect(await settled).toMatchObject({ where: "worker", delegate: "GPU" });
    expect(workers[0]!.terminated).toBe(false);
  });

  it("a worker that never starts is still given up after the timeout", async () => {
    vi.useFakeTimers();
    const workers = fakeWorker(() => {});
    const failed = expect(workerBackend(opts, 60_000)).rejects.toThrow(/did not start/);
    await vi.advanceTimersByTimeAsync(60_000);
    await failed;
    expect(workers[0]!.terminated).toBe(true);
  });

  it("the worker says it started before it loads the models", async () => {
    vi.resetModules();
    vi.doMock("@/perception/models", () => ({ loadVisionModels: () => new Promise(() => {}) }));
    const posted: unknown[] = [];
    vi.spyOn(self, "postMessage").mockImplementation(((m: unknown) => void posted.push(m)) as typeof self.postMessage);
    await import("@/perception/vision.worker");
    const init: WorkerRequest = { kind: "init", assetBase: "http://localhost/mediapipe/", maxPeople: 3, minPersonH: 0.25 };
    void (self.onmessage as unknown as (e: { data: WorkerRequest }) => Promise<void>)({ data: init });
    await Promise.resolve();
    expect(posted).toEqual([{ kind: "started" }]);
    self.onmessage = null;
  });
});
