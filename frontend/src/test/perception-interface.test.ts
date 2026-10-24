import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  BLENDSHAPE_NAMES, createMockPerception, createPerception, obsFromDict, obsToDict, openMediaStream,
  type AudioObs, type Obs, type VideoObs,
} from "@/perception";

const REPO = path.resolve(__dirname, "../../..");

function fakeClock() {
  let now = 1000;
  return { clock: { now: () => now, wall: () => 1.7e9 + now }, advance: (s: number) => (now += s) };
}

describe("perception types", () => {
  it("lists the 52 MediaPipe blendshapes once each", () => {
    expect(BLENDSHAPE_NAMES).toHaveLength(52);
    expect(new Set(BLENDSHAPE_NAMES).size).toBe(52);
  });

  it("parses Python JSONL recordings and writes them back unchanged", () => {
    const file = path.join(REPO, "data/real/capture_fists.jsonl");
    const lines = readFileSync(file, "utf8").trim().split("\n").slice(0, 200);
    for (const line of lines) {
      const d = JSON.parse(line) as Record<string, unknown>;
      const obs = obsFromDict(d);
      expect(obs.type).toBe(d["type"]);
      // Older recordings lack the emotion cues; they come back with the Python defaults.
      const cueDefaults = {
        brow_down: null, eye_open: null, mouth_frown: null, smile: null, head_yaw: null, head_pitch: null,
        blink_onsets: 0,
      };
      const expected = d["type"] === "video"
        ? { ...d, persons: (d["persons"] as object[]).map((p) => ({ ...cueDefaults, ...p })) }
        : d;
      expect(obsToDict(obs)).toEqual(expected);
    }
  });

  it("maps NaN to null and back", () => {
    const v: VideoObs = {
      type: "video", t: 1, wall: 2, frame_seq: 3, frames_decoded: 4, lux: NaN, lux_quality: 0, frame_w: 640,
      frame_h: 480, mirrored: false,
      persons: [{
        track_id: 1, bbox: [0, 0, 1, 1], fist_onsets: 0, fist_active: false, slump: false, head_in_hands: false,
        motion: NaN, face_visible: false, face_valence: NaN, brow_down: NaN, eye_open: NaN, mouth_frown: NaN,
        smile: NaN, head_yaw: NaN, head_pitch: NaN, blink_onsets: 0, raw: null,
      }],
    };
    const d = obsToDict(v);
    expect(d["lux"]).toBeNull();
    expect(JSON.parse(JSON.stringify(d))).toEqual(d);
    const back = obsFromDict(d) as VideoObs;
    expect(back.lux).toBeNaN();
    expect(back.persons[0]!.motion).toBeNaN();
    expect(back.persons[0]!.raw).toBeNull();
  });

  it("accepts the older 'kind' key and rejects unknown kinds", () => {
    const a = obsFromDict({ kind: "audio", t: 1, wall: 2, block_seq: 0, db_spl: 50, rms_power: 1e-5 }) as AudioObs;
    expect(a.type).toBe("audio");
    expect(a.speech_prob).toBe(0);
    expect(() => obsFromDict({ type: "nope" })).toThrow(/unknown observation kind/);
  });
});

describe("createMockPerception", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  async function run(seconds: number, opts: Parameters<typeof createMockPerception>[0] = {}) {
    const { clock, advance } = fakeClock();
    const p = createMockPerception({ clock, ...opts });
    const seen: Obs[] = [];
    p.onObservation((o) => seen.push(o));
    await p.start();
    for (let i = 0; i < seconds * 10; i++) {
      advance(0.1);
      vi.advanceTimersByTime(100);
    }
    return { p, seen };
  }

  it("emits video at ~10 Hz and audio once per second, contract-shaped", async () => {
    const { p, seen } = await run(30);
    const video = seen.filter((o): o is VideoObs => o.type === "video");
    const audio = seen.filter((o): o is AudioObs => o.type === "audio");
    expect(video).toHaveLength(300);
    expect(audio).toHaveLength(30);
    expect(video.map((v) => v.frame_seq)).toEqual(video.map((_, i) => i));
    expect(video.at(-1)!.frames_decoded).toBe(300);
    for (const v of video) {
      expect(v.persons).toHaveLength(1);
      for (const person of v.persons) {
        expect(person.bbox.every((x) => x >= 0 && x <= 1)).toBe(true);
        expect(person.bbox[2]).toBeGreaterThan(person.bbox[0]);
        if (person.face_visible) {
          expect(Number.isFinite(person.brow_down)).toBe(true);
          expect(Math.abs(person.head_yaw)).toBeLessThanOrEqual(90);
          expect(person.face_valence).toBeGreaterThanOrEqual(-1);
          expect(person.face_valence).toBeLessThanOrEqual(1);
          expect(Object.keys(person.raw!.blendshapes)).toHaveLength(52);
        } else {
          expect(person.face_valence).toBeNaN();
          expect(person.raw).toBeNull();
        }
      }
    }
    for (const a of audio) {
      expect(a.db_spl).toBeCloseTo(10 * Math.log10(a.rms_power) + 94, 6);
      expect(a.compliment_hits).toBe(0);
    }
    const c = p.counters();
    expect(c.frames_decoded).toBe(c.frames_purged);
    expect(c.audio_blocks_purged).toBe(300);
    expect(c.live_raw_buffers).toBe(0);
    expect(p.status()).toMatchObject({ phase: "running", camera: true, mic: true, mock: true });
    p.stop();
    expect(p.status().phase).toBe("stopped");
  });

  it("walks through the script: fists while frustrated, slump when worn out, laughter when upbeat", async () => {
    const { seen } = await run(180, { people: 2 });
    const video = seen.filter((o): o is VideoObs => o.type === "video");
    const people = video.flatMap((v) => v.persons);
    expect(new Set(people.map((x) => x.track_id))).toEqual(new Set([1, 2]));
    expect(people.reduce((n, x) => n + x.fist_onsets, 0)).toBeGreaterThan(2);
    expect(people.some((x) => x.slump)).toBe(true);
    expect(people.some((x) => x.head_in_hands)).toBe(true);
    const audio = seen.filter((o): o is AudioObs => o.type === "audio");
    expect(audio.reduce((n, a) => n + a.laughter_onsets, 0)).toBeGreaterThan(0);
    expect(people.reduce((n, x) => n + x.blink_onsets, 0)).toBeGreaterThan(20);
    const valences = people.map((x) => x.face_valence).filter(Number.isFinite);
    expect(Math.max(...valences)).toBeGreaterThan(0.5);
    expect(Math.min(...valences)).toBeLessThan(-0.5);
  });

  it("repeats exactly for the same seed and clock", async () => {
    const ra = await run(5, { seed: 3 });
    ra.p.stop();
    const rb = await run(5, { seed: 3 });
    rb.p.stop();
    const [a, b] = [ra.seen, rb.seen];
    expect(a).toHaveLength(55);
    expect(JSON.stringify(a.map(obsToDict))).toBe(JSON.stringify(b.map(obsToDict)));
  });
});

describe("createPerception (interface)", () => {
  it("refuses to start without tracks", async () => {
    await expect(createPerception().start(null)).rejects.toThrow(/MediaStream/);
  });
});

describe("openMediaStream", () => {
  const track = (kind: string) => ({ kind }) as MediaStreamTrack;
  const stream = (tracks: MediaStreamTrack[]) => ({
    getVideoTracks: () => tracks.filter((t) => t.kind === "video"),
    getAudioTracks: () => tracks.filter((t) => t.kind === "audio"),
    getTracks: () => tracks,
  }) as unknown as MediaStream;

  it("asks for both with the browser's speech processing off", async () => {
    const getUserMedia = vi.fn(async (_c: MediaStreamConstraints) => stream([track("video"), track("audio")]));
    const r = await openMediaStream({}, { getUserMedia });
    expect(r).toMatchObject({ camera: true, mic: true, errors: [] });
    const constraints = getUserMedia.mock.calls[0]![0];
    expect(constraints.audio).toMatchObject({ echoCancellation: false, noiseSuppression: false, autoGainControl: false });
  });

  it("keeps the camera when the mic is denied", async () => {
    vi.stubGlobal("MediaStream", class { constructor(readonly tracks: MediaStreamTrack[]) {} getVideoTracks() { return this.tracks; } });
    const getUserMedia = vi.fn(async (c: MediaStreamConstraints) => {
      if (c.audio) throw Object.assign(new Error("Permission denied"), { name: "NotAllowedError" });
      return stream([track("video")]);
    });
    const r = await openMediaStream({}, { getUserMedia });
    expect(r.camera).toBe(true);
    expect(r.mic).toBe(false);
    expect(r.stream).not.toBeNull();
    expect(r.errors.join("\n")).toMatch(/mic: NotAllowedError/);
    vi.unstubAllGlobals();
  });

  it("reports a missing getUserMedia (plain HTTP)", async () => {
    const r = await openMediaStream({}, undefined);
    expect(r.stream).toBeNull();
    expect(r.errors[0]).toMatch(/HTTPS/);
  });
});
