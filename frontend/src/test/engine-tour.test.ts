import { gunzipSync } from "node:zlib";

import { describe, expect, it } from "vitest";

import { computeEmotion } from "@/engine/emotion-features";
import { emotionInput, loadEmotionModel, predictEmotion } from "@/engine/emotion-model";
import { featuresFromWindow } from "@/engine/features";
import type { FetchLike } from "@/engine/served";
import { parseTour } from "@/engine/tour";
import { RollingWindow } from "@/engine/window";

import { artifactBytes } from "./engine-fixtures";

// End-to-end parity on Cloud B's emotion tour (artifacts/emotion/tour.json.gz): 600 ticks of TickRecords through
// the TypeScript window -> 123 + 31 features -> emotion model, against Python's per-tick outputs.
const tour = parseTour(JSON.parse(gunzipSync(artifactBytes("emotion/tour.json.gz")).toString("utf8")));
const fetchArtifact: FetchLike = async (url) => {
  try {
    const b = artifactBytes(`emotion/${url.split("/").at(-1)}`);
    return { ok: true, status: 200, arrayBuffer: async () => b.slice().buffer as ArrayBuffer };
  } catch {
    return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
  }
};
const maxDiff = (a: ArrayLike<number>, b: readonly (number | null)[]) => {
  let m = 0;
  for (let i = 0; i < b.length; i++) {
    const x = a[i] ?? NaN, y = b[i] ?? NaN;
    if (Number.isNaN(x) !== Number.isNaN(y)) return Infinity;
    if (!Number.isNaN(x)) m = Math.max(m, Math.abs(x - y));
  }
  return m;
};

describe("emotion tour, end to end", () => {
  it("matches Python's features, probabilities, labels and map position on every tick", async () => {
    const model = await loadEmotionModel("/models/emotion/", fetchArtifact);
    expect(model?.modelId).toBe(tour.model_id);
    if (!model) return;
    const win = new RollingWindow(180);
    let worstFeat = 0, worstP = 0, worstVa = 0, labels = 0, onTruth = 0;
    for (const t of tour.ticks) {
      t.records.forEach((r) => win.append(r));
      const person = win.snapshot(1), room = win.room();
      const emo = computeEmotion(person);
      worstFeat = Math.max(worstFeat, maxDiff(emo, t.emotion_features));
      const r = predictEmotion(model, emotionInput(featuresFromWindow(person, room.length ? room : person), emo, model.meta));
      worstP = Math.max(worstP, maxDiff(r.p, t.probabilities));
      worstVa = Math.max(worstVa, Math.abs(r.valence - t.valence), Math.abs(r.arousal - t.arousal));
      if (r.label === t.label) labels += 1;
      if (r.label === t.true_state) onTruth += 1;
    }
    expect(worstFeat).toBeLessThanOrEqual(1e-9);
    expect(worstP).toBeLessThanOrEqual(1e-9);
    expect(worstVa).toBeLessThanOrEqual(1e-9);
    expect(labels).toBe(tour.ticks.length);
    // Not a parity check: how often the model's label is the scripted state (simulated data, window warm-up).
    console.log(`tour: ${tour.ticks.length} ticks, labels identical to Python ${labels}, on the scripted state ${onTruth}`);
  });
});
