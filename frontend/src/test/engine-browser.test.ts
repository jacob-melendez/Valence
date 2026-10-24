import { readFileSync } from "node:fs";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { createBrowserEngine } from "@/engine/browser-engine";
import type { FetchLike } from "@/engine/served";
import { createMockPerception } from "@/perception";
import type { PredictionMsg, ServerMessage } from "@/lib/valence";

// The files scripts/model-files.mjs serves at /models/, read straight from the repository's artifacts/.
const ARTIFACTS = path.resolve(__dirname, "../../../artifacts");
const stubFetch = (served: Record<string, string>): FetchLike => async (url) => {
  const hit = Object.entries(served).find(([prefix]) => url.startsWith(prefix));
  const file = hit ? path.join(hit[1], url.slice(hit[0].length)) : "";
  try {
    const bytes = readFileSync(file);
    return { ok: true, status: 200, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer };
  } catch {
    return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
  }
};
const FILES = { "/models/core/": ARTIFACTS, "/models/emotion/": path.join(ARTIFACTS, "emotion") };

describe("browser engine glue (served predictor + runtime + emotion map)", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("turns mock perception into governance, state_vector and prediction messages with the emotion map", async () => {
    const engine = await createBrowserEngine({ fetchFn: stubFetch(FILES), tz: "UTC" });
    expect(engine.featureVersion).toMatch(/^m2_lgbm@[0-9a-f]{8}$/);
    expect(engine.affectStates?.map((s) => s.id)).toHaveLength(10);
    expect(engine.emotion?.meta.feature_version).toBe("emotion-v1");

    vi.useFakeTimers();
    let mono = 1000, wall = 1792850400; // seconds; advanced with the fake timers
    const clock = { now: () => mono, wall: () => wall };
    const perception = createMockPerception({ people: 2, seed: 3, clock });
    const tracks: ServerMessage[] = [], out: ServerMessage[] = [];
    perception.onObservation((o) => { const t = engine.push(o); if (t) tracks.push(t); });
    await perception.start(null);
    for (let s = 0; s < 24; s++) { // two minutes of 5 s ticks
      for (let k = 0; k < 50; k++) { mono += 0.1; wall += 0.1; await vi.advanceTimersByTimeAsync(100); }
      out.push(...engine.tick(wall, mono, perception.counters()));
    }
    perception.stop();

    expect(tracks.length).toBeGreaterThan(100);
    expect(out.filter((m) => m.type === "governance")).toHaveLength(24);
    const preds = out.filter((m): m is PredictionMsg => m.type === "prediction");
    expect(new Set(preds.map((p) => p.person_id))).toEqual(new Set([1, 2]));
    const last = preds.at(-1);
    expect(last?.layers.length).toBeGreaterThan(0);
    expect(last?.explanation.source).toBe("template");
    expect(Object.keys(last?.affect?.probabilities ?? {})).toHaveLength(10);
    expect(last?.affect?.model).toBe(engine.emotion?.modelId);
    expect(Math.abs(last?.affect?.valence ?? 9)).toBeLessThanOrEqual(1);

    const inputs = engine.latestInputs(1);
    expect(inputs?.feature_version).toBe("emotion-v1");
    expect(Object.keys(inputs?.features ?? {})).toHaveLength(132);
    expect(engine.baseline().target_samples).toBe(90);
    const gov = out.filter((m) => m.type === "governance").at(-1);
    expect(gov?.type === "governance" && gov.frames_purged).toBeGreaterThan(0);
  });

  it("forgets a person's model input once they are no longer predicted (no stale self-report vectors)", async () => {
    const engine = await createBrowserEngine({ fetchFn: stubFetch(FILES), tz: "UTC" });
    vi.useFakeTimers();
    let mono = 1000, wall = 1792850400;
    const perception = createMockPerception({ people: 2, seed: 3, clock: { now: () => mono, wall: () => wall } });
    perception.onObservation((o) => { engine.push(o); });
    await perception.start(null);
    for (let s = 0; s < 12; s++) {
      for (let k = 0; k < 50; k++) { mono += 0.1; wall += 0.1; await vi.advanceTimersByTimeAsync(100); }
      engine.tick(wall, mono, perception.counters());
    }
    expect(engine.latestInputs(1)).not.toBeNull();
    perception.stop(); // everyone leaves the frame
    mono += 5; wall += 5;
    const out = engine.tick(wall, mono, perception.counters());
    expect(out.some((m) => m.type === "prediction")).toBe(false);
    expect(engine.latestInputs(1)).toBeNull();
    expect(engine.latestInputs(2)).toBeNull();
  });

  it("the kill switch's reset drops the room window too, and control actions go into the audit chain", async () => {
    const engine = await createBrowserEngine({ fetchFn: stubFetch(FILES), tz: "UTC" });
    vi.useFakeTimers();
    let mono = 1000, wall = 1792850400;
    const perception = createMockPerception({ people: 1, seed: 5, clock: { now: () => mono, wall: () => wall } });
    perception.onObservation((o) => { engine.push(o); });
    await perception.start(null);
    for (let s = 0; s < 6; s++) {
      for (let k = 0; k < 50; k++) { mono += 0.1; wall += 0.1; await vi.advanceTimersByTimeAsync(100); }
      engine.tick(wall, mono, perception.counters());
    }
    perception.stop();
    const rooms = engine.runtime.window.room().length;
    expect(rooms).toBeGreaterThan(0);
    engine.resetPeople(); // demo reset keeps the room's history
    expect(engine.runtime.window.room().length).toBe(rooms);
    engine.resetPeople(false); // the kill switch (app.py control stop: reset_people(keep_room=False))
    expect(engine.runtime.window.room()).toHaveLength(0);
    engine.audit?.("control_stop");
    const ops = engine.runtime.audit.entries().filter((e) => e.kind === "operator").map((e) => e.payload);
    expect(ops.slice(-2)).toEqual([{ action: "reset", keep_room: false }, { action: "control_stop" }]);
  });

  it("runs without the emotion map when its files are missing", async () => {
    const engine = await createBrowserEngine({ fetchFn: stubFetch({ "/models/core/": ARTIFACTS }), tz: "UTC" });
    expect(engine.emotion).toBeNull();
    expect(engine.affectStates).toEqual([]);
    expect(engine.latestInputs(1)).toBeNull();
  });
});
