import { afterEach, describe, expect, it, vi } from "vitest";

import { BrowserSource, type BrowserEngine, type ObsLike, type PerceptionLike } from "@/lib/browser-source";
import type { RoomBaseline, ServerMessage } from "@/lib/valence";

const BASELINE: RoomBaseline = { mode: "auto", complete: false, target_samples: 90, db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null, lux_median: null };

function fakeEngine(log: string[]): BrowserEngine {
  let ticks = 0;
  return {
    states: [{ id: "Focused", display: "Focused", tier: "P0" }],
    featureVersion: "m2_lgbm@test",
    push: (o) => { log.push("persons" in o ? "video" : "audio"); },
    tick: (wall) => {
      ticks += 1;
      return [{ type: "governance", t: wall, tick: ticks, message: "", frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0,
        transcripts_purged: 0, violations: 0, live_raw_buffers: 0, audit_hash: null, retention_s: 0, purge_gap: 0 }];
    },
    setRoomOnly: (v) => { log.push(`room_only ${v}`); },
    setOverlay: (w, e) => { log.push(`overlay ${w} ${e}`); },
    logBreak: () => { log.push("break"); },
    startBaseline: (s, m) => { log.push(`baseline ${s} ${m}`); },
    baseline: () => BASELINE,
    resetPeople: () => { log.push("reset"); },
  };
}

function fakePerception(): PerceptionLike & { emit: (o: ObsLike) => void; stopped: boolean } {
  let cb: ((o: ObsLike) => void) | null = null;
  const p = {
    stopped: false,
    start: () => {},
    stop: () => { p.stopped = true; },
    onObservation: (f: (o: ObsLike) => void) => { cb = f; },
    counters: () => ({ frames_decoded: 3, frames_purged: 3, audio_blocks_purged: 1, fps: 9.6 }),
    emit: (o: ObsLike) => cb?.(o),
  };
  return p;
}

const video = (wall: number, h: number[]): ObsLike => ({ t: wall, wall, frame_w: 1280, frame_h: 720, mirrored: false,
  persons: h.map((height, i) => ({ track_id: i + 1, bbox: [0.1, 0.1, 0.3, 0.1 + height] as const, slump: i === 0 })) });

describe("in-browser source", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("emits hello, tracks for visible people, ticks every 5 s and status every second", async () => {
    vi.useFakeTimers();
    const log: string[] = [], out: ServerMessage[] = [], errors: string[] = [];
    const perception = fakePerception();
    const src = new BrowserSource({ createEngine: async () => fakeEngine(log), createPerception: async () => perception,
      emit: (m) => out.push(m), onBaseline: () => {}, onError: (w) => errors.push(w), now: () => Date.now(), mono: () => Date.now() });
    await src.start();
    expect(out[0]?.type).toBe("hello");
    expect(errors).toContain("camera"); // jsdom has no getUserMedia: reported, perception still starts
    expect(log).toContain("baseline null auto"); // a new session starts a new automatic venue baseline

    perception.emit(video(Date.now() / 1000, [0.6, 0.2])); // the 0.2-tall box is background (< 0.55 x tallest)
    perception.emit({ t: 1, wall: 1, db_spl: 50 });
    const tracks = out.filter((m) => m.type === "tracks");
    expect(tracks).toHaveLength(1);
    expect(tracks[0]?.type === "tracks" && tracks[0].persons.map((p) => [p.id, p.slump])).toEqual([[1, true]]);
    expect(log.filter((l) => l === "video" || l === "audio")).toEqual(["video", "audio"]);

    await vi.advanceTimersByTimeAsync(5000);
    expect(out.filter((m) => m.type === "governance")).toHaveLength(1);
    const status = out.filter((m) => m.type === "status");
    expect(status.length).toBeGreaterThanOrEqual(4);
    expect(status.at(-1)).toMatchObject({ narrator: "template", fps: 9.6, mode: "live" });
    src.stop();
  });

  it("pauses sensing, and the kill switch stops perception and resets people", async () => {
    vi.useFakeTimers();
    const log: string[] = [], out: ServerMessage[] = [];
    const perception = fakePerception();
    const src = new BrowserSource({ createEngine: async () => fakeEngine(log), createPerception: async () => perception,
      emit: (m) => out.push(m), onBaseline: () => {}, onError: () => {} });
    await src.start();
    expect(src.command({ kind: "control", action: "pause" }).ok).toBe(true);
    perception.emit(video(1, [0.5]));
    await vi.advanceTimersByTimeAsync(5000);
    expect(log).not.toContain("video");
    expect(out.some((m) => m.type === "governance")).toBe(false);

    src.command({ kind: "control", action: "stop" });
    expect(perception.stopped).toBe(true);
    expect(log).toContain("reset");
    expect(src.command({ kind: "replay", scenario: "full_demo" }).ok).toBe(false);
    src.command({ kind: "mode", room_only: true });
    src.command({ kind: "sim", which: "low_light", enabled: true });
    src.command({ kind: "baseline", seconds: 90 });
    expect(log).toEqual(expect.arrayContaining(["room_only true", "overlay low_light true", "baseline 90 capturing"]));
    src.stop();
  });
});
