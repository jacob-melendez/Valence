import { describe, expect, it } from "vitest";

import {
  fmt, fmtSigned, getState, handleMessage, provenanceBadges, shownExplanation,
  type PredictionMsg, type StateVectorMsg, type ValenceState,
} from "@/lib/valence";

// Store state is module-wide, so each test uses its own person id.
let tick = 0;
function prediction(pid: number, label_display: string, label = "Focused"): PredictionMsg {
  tick += 1;
  const event_id = `p${pid}-${String(tick).padStart(6, "0")}`;
  return {
    type: "prediction", event_id, t: 1000 + tick, person_id: pid, label, label_display, confidence: 0.8,
    probabilities: { [label]: 0.8 }, face_only: { label: null, confidence: null }, mask_override: false,
    warming_up: false, layers: [], data_quality: { face_visible_frac: 0, lux: null, ood_features: [] },
    explanation: { text: `template ${event_id}`, source: "template" }, source: "replay", simulated_fields: [],
  };
}
const explain = (p: PredictionMsg, text: string) =>
  handleMessage({ type: "explanation", event_id: p.event_id, person_id: p.person_id, text, source: "llm", latency_ms: 300 });
const shown = (pid: number) => {
  const s = getState();
  const p = s.predictions[pid];
  if (!p) throw new Error(`no prediction for ${pid}`);
  return shownExplanation(p, s.llm[pid]);
};

describe("fmt", () => {
  it("formats numbers and falls back on null, undefined and NaN", () => {
    expect(fmt(72.44)).toBe("72.4");
    expect(fmt(0.123, 2)).toBe("0.12");
    expect(fmt(null)).toBe("—");
    expect(fmt(undefined)).toBe("—");
    expect(fmt(Number.NaN, 1, "n/a")).toBe("n/a");
  });
  it("never prints a negative zero; fmtSigned puts \"+\" on zero and above", () => {
    expect([fmt(-0.004, 2), fmt(-0.04), fmt(-0.4, 0), fmt(-0.006, 2)]).toEqual(["0.00", "0.0", "0", "-0.01"]);
    expect([fmtSigned(-0.004), fmtSigned(0.18), fmtSigned(-0.37), fmtSigned(null)]).toEqual(["+0.00", "+0.18", "-0.37", "—"]);
  });
});

describe("explanation persistence", () => {
  it("keeps the narrator text across newer predictions with the same label_display", () => {
    const e1 = prediction(11, "Focused");
    handleMessage(e1);
    explain(e1, "llm text for e1");
    expect(shown(11)).toMatchObject({ text: "llm text for e1", source: "llm", latency_ms: 300, asOf: null });
    handleMessage(prediction(11, "Focused"));
    handleMessage(prediction(11, "Focused"));
    expect(shown(11)).toMatchObject({ text: "llm text for e1", source: "llm", asOf: e1.event_id });
  });

  it("clears it when label_display changes and shows the new template text", () => {
    const e1 = prediction(12, "Focused");
    handleMessage(e1);
    explain(e1, "llm text");
    const e2 = prediction(12, "Frustrated", "Frustrated_Masked");
    handleMessage(e2);
    expect(shown(12)).toMatchObject({ text: `template ${e2.event_id}`, source: "template" });
    expect(getState().llm[12]).toBeUndefined();
    handleMessage(prediction(12, "Focused")); // back to the old label: the cleared text does not return
    expect(shown(12).source).toBe("template");
  });

  it("accepts a late explanation for an earlier event with the same label", () => {
    const e1 = prediction(13, "Exhausted", "Exhausted");
    handleMessage(e1);
    handleMessage(prediction(13, "Exhausted", "Exhausted"));
    explain(e1, "late llm text");
    expect(shown(13)).toMatchObject({ text: "late llm text", asOf: e1.event_id });
  });

  it("never shows a late explanation for an earlier, different label", () => {
    const e1 = prediction(14, "Focused");
    handleMessage(e1);
    const e2 = prediction(14, "Exhausted", "Exhausted");
    handleMessage(e2);
    explain(e1, "stale llm text");
    expect(shown(14)).toMatchObject({ text: `template ${e2.event_id}`, source: "template" });
  });

  it("drops unknown event ids and never replaces newer text with older text", () => {
    const e1 = prediction(15, "Focused");
    handleMessage(e1);
    const e2 = prediction(15, "Focused");
    handleMessage(e2);
    handleMessage({ type: "explanation", event_id: "p15-999999", person_id: 15, text: "unknown", source: "llm", latency_ms: 1 });
    expect(getState().llm[15]).toBeUndefined();
    explain(e2, "newer");
    explain(e1, "older");
    expect(shown(15).text).toBe("newer");
  });
});

describe("contract nulls and provenance", () => {
  it("accepts a state_vector with null readings", () => {
    const v: StateVectorMsg = {
      type: "state_vector", t: 2000, person_id: 21, ambient_db: null, light_lux: null, hours_since_break: null,
      time_of_day: "15:40", clenched_fists_5m: 0, compliments_15m: 0, laughter_15m: 0, posture_slump: null,
      head_in_hands: null, social_withdrawal: 0, face_visible: false, facial_valence: null, window_fill_min: 3,
      source: "replay", simulated_fields: ["face_occlusion"],
    };
    expect(() => handleMessage(v)).not.toThrow();
    expect(getState().vectors[21]?.ambient_db).toBeNull();
  });

  it("labels mock, replay, simulated and live data", () => {
    const base = getState();
    const live = (patch: Partial<ValenceState>): ValenceState =>
      ({ ...base, source: "live", conn: "open", predictions: {}, vectors: {}, status: null, ...patch });
    const status = (mode: "live" | "replay") => ({
      type: "status" as const, mode, camera: true, mic: true, fps: 10, loop_latency_ms: 40, narrator: "ready",
      offline: true, uptime_s: 1, degraded: [], purge_violation: false,
    });
    expect(provenanceBadges({ ...base, source: "mock", conn: "mock" }).map((b) => b.label)).toEqual(["MOCK"]);
    expect(provenanceBadges(live({ status: status("replay") })).map((b) => b.label)).toEqual(["REPLAY"]);
    expect(provenanceBadges(live({ status: status("live") })).map((b) => b.label)).toEqual(["LIVE"]);
    const sim = { ...prediction(22, "Focused"), source: "live", simulated_fields: ["low_light"] };
    expect(provenanceBadges(live({ status: status("live"), predictions: { 22: sim } })).map((b) => b.label))
      .toEqual(["SIMULATED: low_light"]);
    expect(provenanceBadges(live({ conn: "closed" }))).toEqual([]);
  });
});
