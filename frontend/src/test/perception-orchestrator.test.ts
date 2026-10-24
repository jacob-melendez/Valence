import { afterEach, describe, expect, it, vi } from "vitest";
import { createPerception, type PerceptionDeps } from "@/perception/perception";
import type { Obs, VideoObs } from "@/perception/types";
import type { VisionBackend } from "@/perception/visionHost";

// The orchestration (loops, purging, counters, robustness) with the browser pieces faked: no camera, no MediaPipe.

function track(kind: "video" | "audio") {
  const listeners = new Map<string, () => void>();
  return {
    kind, readyState: "live" as MediaStreamTrackState,
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
  });
});
