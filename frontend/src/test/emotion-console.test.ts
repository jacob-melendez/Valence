import { afterEach, describe, expect, it } from "vitest";

import {
  clearSelfReports, getSelfReports, getState, handleMessage, recordSelfReport, selfReportsJsonl, setFeatureProvider, setSource,
  type PredictionMsg,
} from "@/lib/valence";

const pred = (pid: number, t: number, affect: NonNullable<PredictionMsg["affect"]> | null): PredictionMsg => ({
  type: "prediction", event_id: `p${pid}-${t}`, t, person_id: pid, label: "Focused", label_display: "Focused", confidence: 0.8,
  probabilities: { Focused: 0.8 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up: false,
  layers: [], data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: "", source: "template" },
  source: "live", simulated_fields: [], affect,
});
const affect = (valence: number, arousal: number) => ({ valence, arousal, label: "Calm", label_display: "Calm", confidence: 0.6,
  probabilities: { Calm: 0.6 }, model: "emotion-v1@test" });

describe("emotion map in the console", () => {
  afterEach(() => { setSource("mock"); setFeatureProvider(null); clearSelfReports(); });

  it("keeps the anchors from hello and a 15-minute valence-arousal trail per person", () => {
    setSource("live", "http://127.0.0.1:1");
    handleMessage({ type: "hello", contract_version: "1.0.0", mode: "live", room_only: false, states: [{ id: "Focused", display: "Focused", tier: "P0" }],
      feature_version: "x", frame_w: 0, frame_h: 0, mirrored: false, affect_states: [{ id: "Calm", display: "Calm", valence: 0.4, arousal: -0.4 }] });
    expect(getState().affectStates.map((a) => a.id)).toEqual(["Calm"]);
    handleMessage(pred(41, 1000, affect(0.1, 0.2)));
    handleMessage(pred(41, 1500, affect(0.3, -0.1)));
    handleMessage(pred(41, 2000, affect(0.4, -0.3))); // the first point is now older than 15 minutes
    handleMessage(pred(42, 2000, null)); // no emotion models: no trail
    expect(getState().affectTrail[41]?.map((p) => p.valence)).toEqual([0.3, 0.4]);
    expect(getState().affectTrail[42]).toBeUndefined();
  });

  it("records self-report taps with the model inputs and writes JSONL", () => {
    expect(recordSelfReport(1, 0.2, 0.3)).toBe(false); // no engine registered
    setFeatureProvider((pid) => (pid === 1 ? { feature_version: "emotion-v1", features: { face_now: 0.1, brow_now: null } } : null));
    expect(recordSelfReport(2, 0, 0)).toBe(false);
    expect(recordSelfReport(1, 1.4, -0.25)).toBe(true);
    const [tap] = getSelfReports();
    expect(tap).toMatchObject({ person_id: 1, valence: 1, arousal: -0.25, feature_version: "emotion-v1", features: { face_now: 0.1, brow_now: null } });
    expect(tap?.session_id).toMatch(/^sr-\d{4}-\d{2}-\d{2}-/);
    const lines = selfReportsJsonl().trim().split("\n");
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0] ?? "{}").features.brow_now).toBeNull();
  });
});
