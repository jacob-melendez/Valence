import { act, cleanup, render, screen } from "@testing-library/react";
import { createElement as h } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BrowserSource, type BrowserEngine, type PerceptionLike, type SessionSink } from "@/lib/browser-source";
import { PROTOCOL_V1, protocolMinutes } from "@/lib/protocol";
import { MAX_SESSION_BYTES, SESSION_FORMAT, SessionRecorder, newSessionId } from "@/lib/session-recorder";
import type { Marker, Stamp } from "@/lib/session-recorder";
import type { PredictionMsg, ServerMessage } from "@/lib/valence";

// The session recorder (docs/PROTOCOL.md, valence-session/1), its wiring into the in-browser source, and the
// protocol runner in the store.

type Line = Record<string, unknown> & { type: string };
const parse = (jsonl: string): Line[] => jsonl.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as Line);

const person = (over: Record<string, unknown> = {}) => ({
  track_id: 1, bbox: [0.2, 0.1, 0.6, 0.9], fist_onsets: 0, fist_active: false, slump: false, head_in_hands: false,
  motion: 0.02, face_visible: true, face_valence: 0.1, brow_down: NaN, eye_open: 0.3, mouth_frown: 0, smile: 0.2,
  head_yaw: 1, head_pitch: -2, blink_onsets: 0, ...over,
});
const videoObs = (t: number, persons = [person()]) => ({ type: "video", t, wall: 1_800_000_000 + t, frame_seq: Math.round(t * 10),
  frames_decoded: Math.round(t * 10), lux: 300, lux_quality: 1, frame_w: 1280, frame_h: 720, mirrored: true, persons });
const audioObs = (t: number) => ({ type: "audio", t, wall: 1_800_000_000 + t, block_seq: 1, db_spl: 48, rms_power: 0.001,
  speech_prob: 0.1, laughter_onsets: 0, compliment_hits: 0 });
const prediction = (label = "Focused"): PredictionMsg => ({
  type: "prediction", event_id: "p1-000001", t: 5, person_id: 1, label, label_display: label, confidence: 0.8,
  probabilities: { [label]: 0.8 }, face_only: { label: "Focused", confidence: 0.6 }, mask_override: false, warming_up: true,
  layers: [], data_quality: {} as PredictionMsg["data_quality"], explanation: { text: "", source: "template" }, source: "live",
  simulated_fields: [], affect: { valence: 0.1, arousal: -0.2, label: "Calm", label_display: "Calm", confidence: 0.5, probabilities: {}, model: "emotion-v1@x" },
});
const HEADER = { session_id: "vs-2026-11-08-abcd", started_wall: 1_800_000_000, tz: "America/Los_Angeles",
  feature_version: "m2_lgbm@x", emotion_model: "emotion-v1@x", protocol: PROTOCOL_V1.id };

describe("session recorder (valence-session/1)", () => {
  it("writes the header, observations as obs_to_dict lines without `raw`, ticks, markers and taps", () => {
    const r = new SessionRecorder();
    r.obs(videoObs(0.1)); // not recording yet: dropped
    r.start(HEADER);
    r.obs({ ...videoObs(0.2), persons: [{ ...person(), raw: { blendshapes: { smile: 0.3 }, head_yaw_deg: 1, head_pitch_deg: 2, head_roll_deg: 3 } }] });
    r.obs(audioObs(0.3));
    r.obs({ t: 1 }); // not a perception observation
    r.tick(1_800_000_005, 5, [{ type: "governance" } as unknown as ServerMessage, prediction("Exhausted")]);
    const at: Stamp = { wall: 1_800_000_006, mono: 6 };
    r.marker(at, { kind: "block_start", block_id: "focused", label: "Focused work", scored: true, calm: true, expect: { card: ["Focused"], map: ["Focused", "Calm"] } });
    r.tap(at, { t: 1_800_000_006, person_id: 1, valence: 0.2, arousal: -0.1, feature_version: "emotion-v1", features: { a: 1 } });
    r.stop();
    r.obs(videoObs(9)); // stopped: dropped
    const lines = parse(r.jsonl());
    expect(lines.map((l) => l.type)).toEqual(["session", "video", "audio", "tick", "marker", "self_report"]);
    expect(lines[0]).toEqual({ type: "session", format: SESSION_FORMAT, app: "valence-web", ...HEADER });
    const v = lines[1] as Line & { persons: Record<string, unknown>[] };
    expect(v.persons[0]).not.toHaveProperty("raw");
    expect(v.persons[0]).toMatchObject({ track_id: 1, brow_down: null, smile: 0.2 }); // NaN -> null, as obs_to_dict
    expect(lines[3]).toEqual({ type: "tick", wall: 1_800_000_005, mono: 5, predictions: [{
      person_id: 1, label: "Exhausted", label_display: "Exhausted", confidence: 0.8, warming_up: true, mask_override: false,
      face_only: { label: "Focused", confidence: 0.6 }, affect: { label: "Calm", valence: 0.1, arousal: -0.2 } }] });
    expect(lines[4]).toMatchObject({ type: "marker", wall: 1_800_000_006, mono: 6, kind: "block_start", block_id: "focused", calm: true });
    expect(lines[5]).toMatchObject({ type: "self_report", person_id: 1, valence: 0.2, wall: 1_800_000_006, mono: 6 });
    expect(r.stats()).toMatchObject({ active: false, observations: 2, ticks: 1, markers: 1, taps: 1, session_id: HEADER.session_id });
    expect(r.fileName()).toBe("valence-session-2026-11-08-abcd.jsonl");
    r.clear();
    expect(r.jsonl()).toBe("");
  });

  it("stops itself at the size limit and says so", () => {
    const r = new SessionRecorder();
    r.start(HEADER);
    const big = { ...videoObs(1), persons: Array.from({ length: 2000 }, (_, i) => person({ track_id: i })) };
    let n = 0;
    while (r.active && n < 10_000) { r.obs(big); n += 1; }
    expect(r.active).toBe(false);
    expect(r.stats().full).toBe(true);
    expect(r.stats().bytes).toBeLessThanOrEqual(MAX_SESSION_BYTES);
  });

  it("session ids are dated", () => {
    expect(newSessionId(new Date("2026-11-08T12:00:00Z"), "zz12")).toBe("vs-2026-11-08-zz12");
  });

  it("the protocol is the documented 24 minutes, with a settle block first and expectations on every scored block", () => {
    expect(protocolMinutes(PROTOCOL_V1)).toBe(24);
    expect(PROTOCOL_V1.blocks[0]).toMatchObject({ id: "settle", scored: false });
    for (const b of PROTOCOL_V1.blocks.filter((x) => x.scored)) expect(b.expect.card.length * b.expect.map.length).toBeGreaterThan(0);
  });
});

// ---------- The in-browser source feeds the sink ----------
function sink() {
  const calls: (string | [string, unknown])[] = [];
  const s: SessionSink & { calls: typeof calls } = {
    calls, active: true,
    obs: (o) => { calls.push(["obs", (o as { t: number }).t]); },
    tick: (wall, mono, msgs) => { calls.push(["tick", { wall, mono, n: msgs.length }]); },
    marker: (_at, m: Marker) => { calls.push(["marker", m.action ?? m.kind]); },
  };
  return s;
}
function fakeEngine(): BrowserEngine {
  return {
    states: [], featureVersion: "m2_lgbm@test", emotionModel: "emotion-v1@test", push: () => null, tick: () => [prediction()],
    setRoomOnly() {}, setOverlay() {}, logBreak() {}, startBaseline() {}, resetPeople() {},
    baseline: () => ({ mode: "auto", complete: false, target_samples: 90, db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null, lux_median: null }),
  };
}
function fakePerception() {
  let cb: ((o: unknown) => void) | null = null;
  const p: PerceptionLike & { emit(o: unknown): void } = {
    start() {}, stop() {}, onObservation: (f) => { cb = f as (o: unknown) => void; },
    counters: () => ({ frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0 }), emit: (o) => cb?.(o),
  };
  return p;
}

describe("in-browser source → session sink", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("signals ready before perception starts, records only what the engine took, its ticks and the operator's controls", async () => {
    vi.useFakeTimers();
    const order: string[] = [];
    const s = sink();
    const per = fakePerception();
    let t = 0;
    const src = new BrowserSource({
      createEngine: async () => fakeEngine(), createPerception: async () => { order.push("perception"); return per; },
      emit: () => {}, onBaseline: () => {}, onError: () => {}, needsMedia: false, session: s,
      now: () => 1_800_000_000_000 + t, mono: () => t,
      onReady: (info) => { order.push(`ready ${info.featureVersion} ${info.emotionModel}`); },
    });
    await src.start();
    expect(order).toEqual(["ready m2_lgbm@test emotion-v1@test", "perception"]);
    per.emit(videoObs(0.1));
    per.emit(audioObs(0.2));
    t = 5000;
    await vi.advanceTimersByTimeAsync(5000);
    src.command({ kind: "control", action: "pause" });
    per.emit(videoObs(5.1)); // paused: the engine does not take it, so neither does the recording
    src.command({ kind: "control", action: "start" });
    src.command({ kind: "break" });
    src.command({ kind: "sim", which: "low_light", enabled: true });
    expect(s.calls).toEqual([["obs", 0.1], ["obs", 0.2], ["tick", { wall: 1_800_000_005, mono: 5, n: 1 }],
      ["marker", "pause"], ["marker", "start"], ["marker", "break"], ["marker", "sim_low_light_on"]]);
    expect(src.stamp()).toEqual({ wall: 1_800_000_005, mono: 5 });
    src.stop();
  });

  it("the tour records nothing", async () => {
    vi.useFakeTimers();
    const s = sink();
    const src = new BrowserSource({
      createEngine: async () => fakeEngine(), createPerception: async () => fakePerception(),
      emit: () => {}, onBaseline: () => {}, onError: () => {}, mode: "tour", tourSpeed: 10, session: s,
    });
    await src.start();
    await vi.advanceTimersByTimeAsync(2000);
    src.command({ kind: "break" });
    expect(s.calls).toEqual([]);
    src.stop();
  });
});

// ---------- The store: a recorded protocol run ----------
describe("protocol run in the store", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.doUnmock("@/engine/browser-engine");
    vi.doUnmock("@/perception/index");
    window.history.replaceState(null, "", "/");
  });

  it("restarts the camera session, stamps the blocks on schedule, takes taps, and a Pause aborts the protocol", async () => {
    window.history.replaceState(null, "", "/?source=browser&perception=mock");
    const perceptions: ReturnType<typeof fakePerception>[] = [];
    vi.doMock("@/engine/browser-engine", () => ({ createBrowserEngine: async () => ({
      ...fakeEngine(), latestInputs: () => ({ feature_version: "emotion-v1", features: { a: 1 } }),
      tick: () => [{ ...prediction(), warming_up: false }],
    }) }));
    vi.doMock("@/perception/index", () => ({ createMockPerception: () => { const p = fakePerception(); perceptions.push(p); return p; } }));
    // Timers and the clock are fake; setTimeout stays real so the lazy engine chunk can load.
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "Date"], now: new Date("2026-11-08T18:00:00Z") });
    vi.resetModules();
    const v = await import("@/lib/valence");
    const panels = await import("@/components/console/panels");
    const until = async (ok: () => boolean) => {
      for (let i = 0; i < 200 && !ok(); i++) await act(async () => { await new Promise((r) => setTimeout(r, 5)); });
      expect(ok()).toBe(true);
    };

    v.startBrowserSource("live");
    await until(() => perceptions.length === 1);
    expect(v.startRecordedSession({ protocol: true })).toBe(true);
    // A fresh engine and perception; the recording and the protocol start with it.
    await until(() => perceptions.length === 2 && v.getProtocolRun()?.status === "running");
    expect(v.getSessionStats()).toMatchObject({ active: true, observations: 0 });
    render(h(panels.SessionPanel));
    expect(screen.getByTestId("protocol-block").textContent).toContain("Settle in");
    expect(screen.getByTestId("protocol-countdown").textContent).toBe("2:00");

    perceptions[1]!.emit(videoObs(1));
    await act(async () => { vi.advanceTimersByTime(5000); }); // one engine tick
    expect(v.recordSelfReport(1, 0.4, 0.2)).toBe(true);
    await act(async () => { vi.advanceTimersByTime(115_000); }); // settle ends at 2:00
    expect(v.getProtocolRun()).toMatchObject({ index: 1, status: "running" });
    expect(screen.getByTestId("protocol-block").textContent).toContain("Focused work");

    await act(async () => { await v.sendCommand({ kind: "control", action: "pause" }); });
    expect(v.getProtocolRun()).toMatchObject({ status: "aborted", reason: "operator pause" });
    expect(v.getSessionStats().active).toBe(false);
    expect(screen.getByTestId("recording-done").textContent).toMatch(/^Protocol aborted \(operator pause\)\./);

    const lines = parse(v.sessionJsonl());
    expect(lines[0]).toMatchObject({ type: "session", format: SESSION_FORMAT, protocol: PROTOCOL_V1.id, feature_version: "m2_lgbm@test",
      emotion_model: "emotion-v1@test", tz: Intl.DateTimeFormat().resolvedOptions().timeZone });
    const kinds = lines.map((l) => l.type === "marker" ? `${String(l["kind"])}${l["block_id"] ? `:${String(l["block_id"])}` : ""}${l["action"] ? `:${String(l["action"])}` : ""}` : l.type);
    expect(kinds.filter((k) => k !== "tick")).toEqual(["session", "protocol_start", "block_start:settle", "video", "self_report",
      "block_end:settle", "block_start:focused", "control:pause", "protocol_abort"]);
    const starts = lines.filter((l) => l["kind"] === "block_start");
    expect(Number(starts[1]!["wall"]) - Number(starts[0]!["wall"])).toBe(120); // on schedule, not timer jitter
    expect(starts[1]).toMatchObject({ scored: true, calm: true, expect: { card: ["Focused"], map: ["Focused", "Calm"] } });
    v.setSource("mock");
  });

  it("is camera mode only: no panel and no recording in the tour", async () => {
    window.history.replaceState(null, "", "/?demo=tour");
    vi.resetModules();
    const v = await import("@/lib/valence");
    const panels = await import("@/components/console/panels");
    v.startBrowserSource("tour");
    render(h(panels.SessionPanel));
    expect(screen.queryByTestId("session-panel")).toBeNull();
    expect(v.startRecordedSession({ protocol: true })).toBe(false);
    v.setSource("mock");
  });
});
