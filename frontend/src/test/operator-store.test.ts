import { describe, expect, it } from "vitest";

import {
  CORE_STATES, applyCommandResult, applyHealth, coverBox, applyStateSnapshot, baselineFilling, baselineProgress, baselineWarning,
  describe as describeCommand, getState, handleMessage, parseBaseline, stateName, validStates,
  type HelloMsg, type PredictionMsg, type RoomBaseline, type StateVectorMsg, type StatusMsg,
} from "@/lib/valence";

// Store state is module-wide, so each test uses its own person ids.
const SIX = [
  ...CORE_STATES,
  { id: "Overwhelmed", display: "Overwhelmed / Distressed", tier: "P1" },
  { id: "Tense", display: "Tense / Visible Frustration", tier: "P1" },
];
const hello = (states: unknown, room_only = false) => ({
  type: "hello", contract_version: "1.0.0", mode: "replay", room_only, states, feature_version: "x",
  frame_w: 0, frame_h: 0, mirrored: false,
}) as HelloMsg;
const prediction = (pid: number, n: number, label = "Focused"): PredictionMsg => ({
  type: "prediction", event_id: `p${pid}-${String(n).padStart(6, "0")}`, t: 5000 + n, person_id: pid, label, label_display: label,
  confidence: 0.7, probabilities: { [label]: 0.7 }, face_only: { label: null, confidence: null }, mask_override: false,
  warming_up: false, layers: [], data_quality: { face_visible_frac: 0, lux: null, ood_features: [] },
  explanation: { text: "t", source: "template" }, source: "replay", simulated_fields: [],
});
const vector = (pid: number): StateVectorMsg => ({
  type: "state_vector", t: 5000, person_id: pid, ambient_db: 61, light_lux: 300, hours_since_break: 1, time_of_day: "10:00",
  clenched_fists_5m: 0, compliments_15m: 0, laughter_15m: 0, posture_slump: 0.1, head_in_hands: 0, social_withdrawal: 0,
  face_visible: true, facial_valence: 0.1, window_fill_min: 4, source: "replay", simulated_fields: [],
});
const status = (mode: StatusMsg["mode"]): StatusMsg => ({
  type: "status", mode, camera: true, mic: true, fps: 10, loop_latency_ms: 40, narrator: "ready", offline: true,
  uptime_s: 1, degraded: [], purge_violation: false,
});
const baseline = (patch: Partial<RoomBaseline> = {}): RoomBaseline => ({
  mode: "auto", complete: false, target_samples: 90, db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null,
  lux_median: null, ...patch,
});

describe("state set from hello", () => {
  it("names and orders the P1 states from hello.states", () => {
    expect(stateName("Tense")).toBe("Tense"); // core set: unknown id falls back to the id
    handleMessage(hello(SIX));
    expect(getState().states.map((s) => s.id)).toEqual(SIX.map((s) => s.id));
    expect(stateName("Tense")).toBe("Tense / Visible Frustration");
    expect(stateName("Overwhelmed")).toBe("Overwhelmed / Distressed");
    expect(getState().states.find((s) => s.id === "Tense")?.tier).toBe("P1");
    handleMessage(hello(CORE_STATES));
    expect(getState().states).toHaveLength(4);
  });

  it("keeps the current set on a missing or malformed list and reads room_only", () => {
    handleMessage(hello(SIX));
    handleMessage(hello(undefined, true));
    expect(getState().states).toHaveLength(6);
    expect(getState().roomOnly).toBe(true);
    handleMessage(hello([{ nope: 1 }]));
    expect(getState().states).toHaveLength(6);
    expect(validStates([{ id: "A_B" }])).toEqual([{ id: "A_B", display: "A B", tier: "P0" }]);
    handleMessage(hello(CORE_STATES));
  });
});

describe("room baseline", () => {
  it("parses the baseline object and rejects junk", () => {
    const b = parseBaseline({ ok: true, mode: "captured", complete: true, target_samples: 90, db_samples: 90, lux_samples: 88, db_median: 68.4, db_iqr: 2.1, lux_median: 410 });
    expect(b).toEqual({ mode: "captured", complete: true, target_samples: 90, db_samples: 90, lux_samples: 88, db_median: 68.4, db_iqr: 2.1, lux_median: 410 });
    expect(parseBaseline({ mode: "auto", db_median: null })?.db_median).toBeNull();
    expect(parseBaseline({ mode: "weird" })).toBeNull();
    expect(parseBaseline(null)).toBeNull();
  });

  it("reports capture progress and whether it is still filling", () => {
    expect(baselineProgress(null)).toBe(0);
    expect(baselineProgress(baseline({ mode: "capturing", db_samples: 45 }))).toBe(0.5);
    expect(baselineProgress(baseline({ mode: "capturing", db_samples: 0, lux_samples: 30 }))).toBeCloseTo(1 / 3);
    expect(baselineProgress(baseline({ mode: "captured", complete: true, db_samples: 90 }))).toBe(1);
    // Lux arrives with every video frame (about 10 Hz): progress follows the dB channel the backend completes on.
    expect(baselineProgress(baseline({ mode: "capturing", db_samples: 9, lux_samples: 90 }))).toBeCloseTo(0.1);
    expect(baselineFilling(baseline({ mode: "capturing" }))).toBe(true);
    expect(baselineFilling(baseline({ mode: "auto", complete: true }))).toBe(false);
    expect(baselineFilling(baseline({ mode: "auto" }))).toBe(true);
  });

  it("warns only for live sensors without a captured baseline", () => {
    const live = { source: "live" as const, conn: "open" as const, status: status("live") };
    expect(baselineWarning({ ...live, baseline: baseline() })).toBe(true);
    expect(baselineWarning({ ...live, baseline: baseline({ mode: "capturing" }) })).toBe(true);
    expect(baselineWarning({ ...live, baseline: null })).toBe(true);
    expect(baselineWarning({ ...live, baseline: baseline({ mode: "captured", complete: true }) })).toBe(false);
    expect(baselineWarning({ ...live, status: status("replay"), baseline: baseline() })).toBe(false);
    expect(baselineWarning({ ...live, source: "mock", conn: "mock", baseline: baseline() })).toBe(false);
  });

  it("takes the baseline and edge errors from /api/health", () => {
    applyHealth({ ok: true, edge_error: "camera permission denied", mic_error: null, room_baseline: baseline({ mode: "capturing", db_samples: 12 }) });
    expect(getState().baseline?.db_samples).toBe(12);
    expect(getState().edgeError).toBe("camera permission denied");
    expect(getState().micError).toBeNull();
    applyHealth({ ok: true, edge_error: null, mic_error: null });
    expect(getState().baseline?.db_samples).toBe(12); // no room_baseline: keep the last one
    expect(getState().edgeError).toBeNull();
  });
});

describe("operator commands", () => {
  it("maps each command to its endpoint and body", () => {
    expect(describeCommand({ kind: "baseline", seconds: 90 })).toEqual(["/api/room/baseline", { seconds: 90 }]);
    expect(describeCommand({ kind: "baseline_reset" })).toEqual(["/api/room/baseline/reset", undefined]);
    expect(describeCommand({ kind: "live" })).toEqual(["/api/live", undefined]);
    expect(describeCommand({ kind: "replay", scenario: "full_demo", speed: 20, loop: true })).toEqual(["/api/replay/full_demo", { speed: 20, loop: true }]);
    expect(describeCommand({ kind: "replay", scenario: "focused" })).toEqual(["/api/replay/focused", { speed: 1, loop: false }]);
    expect(describeCommand({ kind: "replay_recording", name: "capture_typing", speed: 10 })).toEqual(["/api/replay_recording/capture_typing", { speed: 10, loop: true }]);
    expect(describeCommand({ kind: "demo_reset" })).toEqual(["/api/demo/reset", undefined]);
    expect(describeCommand({ kind: "mode", room_only: true })).toEqual(["/api/mode", { room_only: true }]);
  });

  it("applies the live response's edge and mic errors and clears the old people", () => {
    handleMessage(prediction(31, 1));
    applyCommandResult({ kind: "live" }, { ok: true, edge_error: null, mic_error: "no audio device" });
    expect(getState().micError).toBe("no audio device");
    expect(getState().edgeError).toBeNull();
    expect(getState().predictions[31]).toBeUndefined();
    expect(getState().history).toEqual([]);
  });

  it("demo reset and room-only mode drop the per-person cards", () => {
    handleMessage(prediction(32, 1));
    handleMessage(vector(32));
    applyCommandResult({ kind: "demo_reset" }, { ok: true });
    expect(getState().predictions[32]).toBeUndefined();
    expect(getState().vectors[32]).toBeUndefined();
    handleMessage(prediction(33, 1));
    applyCommandResult({ kind: "mode", room_only: true }, { ok: true, room_only: true });
    expect(getState().roomOnly).toBe(true);
    expect(getState().predictions[33]).toBeUndefined();
    applyCommandResult({ kind: "mode", room_only: false }, { ok: true, room_only: false });
    expect(getState().roomOnly).toBe(false);
  });

  it("takes the baseline from the capture and reset responses", () => {
    applyCommandResult({ kind: "baseline", seconds: 90 }, { ok: true, ...baseline({ mode: "capturing" }) });
    expect(getState().baseline?.mode).toBe("capturing");
    applyCommandResult({ kind: "baseline_reset" }, { ok: true, ...baseline({ mode: "auto" }) });
    expect(getState().baseline?.mode).toBe("auto");
  });
});

describe("GET /api/state on connect", () => {
  it("fills the cards and state vectors, and pairs later explanations", () => {
    applyStateSnapshot({ people: { "41": { state_vector: vector(41), prediction: prediction(41, 7, "Tense") } }, status: status("replay") });
    expect(getState().predictions[41]?.event_id).toBe("p41-000007");
    expect(getState().vectors[41]?.ambient_db).toBe(61);
    handleMessage({ type: "explanation", event_id: "p41-000007", person_id: 41, text: "llm", source: "llm", latency_ms: 5 });
    expect(getState().llm[41]?.text).toBe("llm");
  });

  it("skips the other mode's people left in the snapshot", () => {
    applyStateSnapshot({ people: { "0": { prediction: prediction(0, 9) }, "43": { prediction: prediction(43, 9) } } });
    expect(getState().predictions[43]?.event_id).toBe("p43-000009");
    expect(getState().predictions[0]).toBeUndefined();
    handleMessage(hello(CORE_STATES, true));
    applyStateSnapshot({ people: { "0": { prediction: prediction(0, 10) }, "44": { prediction: prediction(44, 10) } } });
    expect(getState().predictions[0]?.event_id).toBe("p0-000010");
    expect(getState().predictions[44]).toBeUndefined();
    handleMessage(hello(CORE_STATES, false));
  });

  it("never rolls a card back to an older tick than the WebSocket already delivered", () => {
    const newer = prediction(45, 101, "Focused");
    handleMessage(newer);
    handleMessage({ type: "explanation", event_id: newer.event_id, person_id: 45, text: "llm 101", source: "llm", latency_ms: 5 });
    handleMessage({ ...vector(45), t: newer.t });
    applyStateSnapshot({ people: { "45": { state_vector: { ...vector(45), t: newer.t - 1, ambient_db: 99 }, prediction: prediction(45, 100, "Tense") } } });
    expect(getState().predictions[45]?.event_id).toBe(newer.event_id);
    expect(getState().llm[45]?.text).toBe("llm 101");
    expect(getState().vectors[45]?.ambient_db).toBe(61);
  });

  it("ignores an empty or malformed snapshot", () => {
    expect(() => applyStateSnapshot({ people: {} })).not.toThrow();
    expect(() => applyStateSnapshot({ people: { "42": null } })).not.toThrow();
    expect(() => applyStateSnapshot(null)).not.toThrow();
    expect(getState().predictions[42]).toBeUndefined();
  });
});

describe("box mapping into the 16:9 view", () => {
  it("leaves 16:9 frames and unknown frame sizes unchanged", () => {
    expect(coverBox([0.1, 0.2, 0.4, 0.95], 1280, 720)).toEqual([0.1, 0.2, 0.4, 0.95]);
    expect(coverBox([0.1, 0.2, 0.4, 0.95], 0, 0)).toEqual([0.1, 0.2, 0.4, 0.95]);
  });

  it("crops a 640x480 frame top and bottom like object-cover", () => {
    const [x1, y1, x2, y2] = coverBox([0.1, 0.2, 0.4, 0.95], 640, 480);
    expect([x1, x2]).toEqual([0.1, 0.4]); // full width shown
    expect(y1).toBeCloseTo(0.1); // (0.2 - 0.125) / 0.75
    expect(y2).toBeCloseTo(1.1); // below the visible area, clipped by the view
    const mid = coverBox([0, 0.5, 1, 0.5], 640, 480);
    expect(mid[1]).toBeCloseTo(0.5); // the centre stays put
    expect(coverBox([0, 0.125, 1, 0.875], 640, 480).map((v) => +v.toFixed(6))).toEqual([0, 0, 1, 1]);
  });

  it("crops a wider frame left and right", () => {
    const [x1, , x2] = coverBox([0.2, 0, 0.8, 1], 2560, 1080);
    const vis = (16 / 9) / (2560 / 1080);
    expect(x1).toBeCloseTo((0.2 - (1 - vis) / 2) / vis);
    expect(x2).toBeCloseTo((0.8 - (1 - vis) / 2) / vis);
  });

  it("stores the frame size from tracks", () => {
    handleMessage({ type: "tracks", t: 1, capture_ts_ms: 1, frame_w: 640, frame_h: 480, mirrored: false, persons: [] });
    expect([getState().frameW, getState().frameH]).toEqual([640, 480]);
  });
});
