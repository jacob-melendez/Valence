/**
 * emotion-features.ts + emotion-model.ts against Cloud B's Python emotion map.
 *
 * Two sources of golden values:
 * - artifacts/emotion/golden.json (scripts/emotion_export.py, Cloud B): tick-channel windows with the 31
 *   compute_emotion features, and 23 model inputs with LightGBM's raw scores, the temperature softmax, the map
 *   position (valence_mix / arousal_mix) and the regressors.
 * - emotion.json.gz (scripts/export_web_golden_emotion.py): simulator sessions of the ten-state set with
 *   compute_emotion at every 4th row of every person, synthetic edge windows (no readable face, partial face,
 *   blink-rate threshold, NaN counters, empty, > 180 ticks), and 261 model inputs (the frozen 123 features +
 *   the 31) with EmotionPredictor's outputs and the Affect message.
 *
 * The models are read from ../artifacts/emotion through loadEmotionModel with a fetch stub, once preferring the
 * LightGBM text models (*.txt.gz) and once forced onto the dump_model() JSON (*.json.gz).
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeTickRecord, type TickRecord } from "@/engine/contracts";
import {
  EMOTION_FEATURE_NAMES,
  EMOTION_FIELDS,
  type EmotionField,
  computeEmotion,
  emotionFromChannels,
} from "@/engine/emotion-features";
import {
  type EmotionMeta,
  type EmotionModel,
  type EmotionResult,
  type FetchLike,
  boosterFromText,
  emotionInput,
  loadEmotionModel,
  makeEmotionModel,
  predictEmotion,
  toAffect,
} from "@/engine/emotion-model";
import { FEATURE_SETS, N_FEATURES } from "@/engine/features";
import { ARTIFACTS_DIR, artifactJson, loadFixture, nn } from "./engine-fixtures";

// ---------- Fixtures ----------

type Num = number | null;
type ChannelsJson = Record<EmotionField, Num[]>;

interface GoldenJson {
  model_id: string;
  input_features: string[];
  states: string[];
  temperature: number;
  tolerance: { features: number; scores: number; pred_contrib: number };
  vectors: {
    note: string;
    x: Num[];
    raw: number[];
    probabilities: number[];
    valence_raw: number;
    arousal_raw: number;
    valence: number;
    arousal: number;
    valence_mix: number;
    arousal_mix: number;
    label: string;
  }[];
  windows: { note: string; person: ChannelsJson; expected: Num[] }[];
}

interface AffectJson {
  valence: number;
  arousal: number;
  label: string;
  label_display: string;
  confidence: number;
  probabilities: Record<string, number>;
  model: string;
}

interface ModelCaseJson {
  name: string;
  state: string;
  window:
    null | { session: number; person_id: number; end: number; hist: number } | { edge: string };
  x123: Num[] | null;
  x: Num[];
  raw: number[];
  p: number[];
  k: number;
  label: string;
  confidence: number;
  valence_mix: number;
  arousal_mix: number;
  valence_raw: number;
  arousal_raw: number;
  valence: number;
  arousal: number;
  affect: AffectJson;
}

interface EmotionFixture {
  version: number;
  feature_version: string;
  model_id: string;
  temperature: number;
  states: string[];
  emotion_feature_names: string[];
  input_features: string[];
  emotion_fields: string[];
  sessions: {
    session_id: string;
    seed: number;
    state_set: string;
    n_ticks: number;
    people: { person_id: number; n_rows: number; channels: ChannelsJson }[];
  }[];
  checkpoints: {
    session: number;
    person_id: number;
    end: number;
    hist: number;
    state: string;
    features: Num[];
  }[];
  edge_windows: { name: string; channels: ChannelsJson; features: Num[] }[];
  model_cases: ModelCaseJson[];
}

const EMO_DIR = path.join(ARTIFACTS_DIR, "emotion");
const golden = artifactJson<GoldenJson>("emotion/golden.json");
const meta = artifactJson<EmotionMeta>("emotion/meta.json");
const fx = loadFixture<EmotionFixture>("emotion");
const TARGET = 1e-9;

// ---------- Helpers ----------

/** Max |a - b| over two arrays with the NaN pattern required identical (a NaN mismatch is Infinity). */
function maxErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    if (Number.isNaN(x) || Number.isNaN(y)) {
      if (Number.isNaN(x) !== Number.isNaN(y)) return Infinity;
      continue;
    }
    if (x === y) continue;
    m = Math.max(m, Math.abs(x - y));
  }
  return m;
}

function bitEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}

/** TickRecords carrying the emotion channels (other fields at their defaults), one per tick. */
function recordsFrom(
  channels: Partial<Record<EmotionField, ArrayLike<number>>>,
  personId = 1,
): TickRecord[] {
  const n = (channels.face_visible ?? channels.brow_down ?? []).length;
  const out: TickRecord[] = [];
  for (let i = 0; i < n; i++) {
    const r = makeTickRecord({
      tick: i,
      ts: i * 5,
      t_mono: i * 5,
      session_id: "golden",
      person_id: personId,
      display_id: `Person #${personId}`,
      n_people: 1,
    });
    for (const f of EMOTION_FIELDS) r[f] = (channels[f] as ArrayLike<number>)[i] as number;
    out.push(r);
  }
  return out;
}

function decodeChannels(ch: ChannelsJson): Record<EmotionField, number[]> {
  const out = {} as Record<EmotionField, number[]>;
  for (const f of EMOTION_FIELDS) out[f] = nn(ch[f]);
  return out;
}

/** A fetch stub over artifacts/emotion: 404 for missing files, plus per-file overrides. */
function stubFetch(
  dir: string,
  opts: {
    missing?: (name: string) => boolean;
    inflate?: (name: string) => boolean;
    log?: string[];
  } = {},
): FetchLike {
  return async (url: string) => {
    opts.log?.push(url);
    const name = url.slice(url.lastIndexOf("/") + 1);
    const file = path.join(dir, name);
    if (opts.missing?.(name) || !existsSync(file)) {
      return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
    }
    let bytes: Uint8Array = readFileSync(file);
    if (opts.inflate?.(name)) bytes = gunzipSync(bytes); // as served with Content-Encoding: gzip
    const copy = bytes.slice();
    return { ok: true, status: 200, arrayBuffer: async () => copy.buffer as ArrayBuffer };
  };
}

const report: Record<string, number | string> = {};
const note = (key: string, v: number | string) => {
  const prev = report[key];
  report[key] = typeof v === "number" && typeof prev === "number" ? Math.max(prev, v) : v;
};
afterAll(() => {
  console.log(`engine-emotion report:\n${JSON.stringify(report, null, 1)}`);
});

// ---------- (a) window features ----------

describe("emotion-v1 window features (emotion-features.ts)", () => {
  it("names match meta.json, golden.json and the fixture", () => {
    expect([...EMOTION_FEATURE_NAMES]).toEqual(meta.emotion_feature_names);
    expect([...EMOTION_FEATURE_NAMES]).toEqual(fx.emotion_feature_names);
    expect([...EMOTION_FIELDS]).toEqual(fx.emotion_fields);
    expect(EMOTION_FEATURE_NAMES).toHaveLength(31);
    expect(meta.input_features).toEqual([...(FEATURE_SETS["m2"] ?? []), ...EMOTION_FEATURE_NAMES]);
  });

  it("golden.json windows (Cloud B)", () => {
    const tol = Math.min(golden.tolerance.features, TARGET);
    expect(golden.windows.length).toBeGreaterThan(0);
    let worst = 0;
    let identical = 0;
    for (const w of golden.windows) {
      const ch = decodeChannels(w.person);
      const want = nn(w.expected);
      const a = emotionFromChannels(ch);
      const b = computeEmotion(recordsFrom(ch));
      const err = Math.max(maxErr(a, want), maxErr(b, want));
      expect(err, w.note).toBeLessThanOrEqual(tol);
      expect(bitEqual(a, b), w.note).toBe(true);
      worst = Math.max(worst, err);
      if (bitEqual(a, want)) identical++;
    }
    note("golden_windows_max_err", worst);
    note("golden_windows_bit_identical", `${identical}/${golden.windows.length}`);
  });

  it("fixture checkpoints: every 4th row of every person, ten states", () => {
    const states = new Set(fx.checkpoints.map((c) => c.state));
    expect(states.size).toBe(10);
    const recs = fx.sessions.map(
      (s) =>
        new Map(
          s.people.map((p) => [p.person_id, recordsFrom(decodeChannels(p.channels), p.person_id)]),
        ),
    );
    let worst = 0;
    let identical = 0;
    let allNaN = 0;
    for (const c of fx.checkpoints) {
      const rows = recs[c.session]?.get(c.person_id) as TickRecord[];
      const want = nn(c.features);
      const got = computeEmotion(rows.slice(c.end - c.hist + 1, c.end + 1));
      const err = maxErr(got, want);
      if (!(err <= TARGET))
        throw new Error(`checkpoint s${c.session} p${c.person_id} row ${c.end}: err ${err}`);
      worst = Math.max(worst, err);
      if (bitEqual(got, want)) identical++;
      if (want.every(Number.isNaN)) allNaN++;
      // the whole history: computeEmotion keeps only the last 180 rows itself
      if (c.end >= 180 && c.end % 40 === 0)
        expect(bitEqual(computeEmotion(rows.slice(0, c.end + 1)), got)).toBe(true);
    }
    expect(allNaN).toBeGreaterThan(0);
    note("fixture_checkpoints", fx.checkpoints.length);
    note("fixture_checkpoints_max_err", worst);
    note("fixture_checkpoints_bit_identical", `${identical}/${fx.checkpoints.length}`);
  });

  it("fixture edge windows (no face, partial face, blink threshold, NaN counters, empty, > 180 ticks)", () => {
    let worst = 0;
    for (const e of fx.edge_windows) {
      const ch = decodeChannels(e.channels);
      const want = nn(e.features);
      const got = emotionFromChannels(ch);
      const err = maxErr(got, want);
      expect(err, e.name).toBeLessThanOrEqual(TARGET);
      expect(bitEqual(computeEmotion(recordsFrom(ch)), got), e.name).toBe(true);
      worst = Math.max(worst, err);
    }
    const byName = new Map(fx.edge_windows.map((e) => [e.name, nn(e.features)]));
    expect(byName.get("no readable face in the window")?.every(Number.isNaN)).toBe(true);
    expect(byName.get("empty window")?.every(Number.isNaN)).toBe(true);
    expect(
      Number.isNaN(byName.get("blink threshold: 2 visible ticks = 10 s")?.[28] as number),
    ).toBe(false);
    expect(Number.isNaN(byName.get("blink threshold: 9.95 s visible")?.[28] as number)).toBe(true);
    expect(computeEmotion([]).every(Number.isNaN)).toBe(true);
    note("fixture_edge_windows_max_err", worst);
  });
});

// ---------- (b, c) models ----------

describe("emotion models (emotion-model.ts)", () => {
  let txt: EmotionModel;
  let json: EmotionModel;
  const urls: string[] = [];
  const jsonUrls: string[] = [];

  beforeAll(async () => {
    let t0 = performance.now();
    const a = await loadEmotionModel("/models/emotion/", stubFetch(EMO_DIR, { log: urls }));
    note("load_txt_ms", Math.round(performance.now() - t0));
    // dump_model() JSON for the classifier and arousal; valence from its text model served already inflated
    t0 = performance.now();
    const b = await loadEmotionModel(
      "/models/emotion",
      stubFetch(EMO_DIR, {
        log: jsonUrls,
        missing: (n) => n === "classifier.txt.gz" || n === "arousal.txt.gz",
        inflate: (n) => n === "valence.txt.gz",
      }),
    );
    note("load_json_ms", Math.round(performance.now() - t0));
    if (!a || !b) throw new Error("emotion model did not load");
    txt = a;
    json = b;
  }, 120_000);

  it("loads meta.json and prefers the text models", () => {
    expect(txt.modelId).toBe(golden.model_id);
    expect(txt.modelId).toBe(fx.model_id);
    expect(txt.meta.temperature).toBe(golden.temperature);
    expect(txt.meta.states.map((s) => s.id)).toEqual(golden.states);
    expect(txt.classifier.numTreePerIteration).toBe(10);
    expect(txt.classifier.numFeatures).toBe(132);
    expect(txt.valence && txt.arousal).toBeTruthy();
    expect(urls).toContain("/models/emotion/classifier.txt.gz");
    expect(urls.some((u) => u.endsWith(".json.gz"))).toBe(false);
    expect(jsonUrls).toContain("/models/emotion/classifier.json.gz");
    expect(jsonUrls).toContain("/models/emotion/arousal.json.gz");
    expect(jsonUrls).not.toContain("/models/emotion/valence.json.gz");
    expect(json.classifier.trees.length).toBe(txt.classifier.trees.length);
  });

  function compareGolden(model: EmotionModel, tag: string) {
    const tol = Math.min(golden.tolerance.scores, TARGET);
    const errs = { raw: 0, p: 0, mix: 0, reg: 0 };
    for (const v of golden.vectors) {
      const r = predictEmotion(model, nn(v.x));
      errs.raw = Math.max(errs.raw, maxErr(r.raw, v.raw));
      errs.p = Math.max(errs.p, maxErr(r.p, v.probabilities));
      errs.mix = Math.max(errs.mix, maxErr([r.valence, r.arousal], [v.valence_mix, v.arousal_mix]));
      errs.reg = Math.max(
        errs.reg,
        maxErr(
          [
            r.valenceReg ?? NaN,
            r.arousalReg ?? NaN,
            r.valenceRegRaw ?? NaN,
            r.arousalRegRaw ?? NaN,
          ],
          [v.valence, v.arousal, v.valence_raw, v.arousal_raw],
        ),
      );
      expect(r.label, v.note).toBe(v.label);
      expect(r.confidence).toBe(r.p[r.k]);
    }
    for (const [k, e] of Object.entries(errs)) {
      expect(e, `${tag} ${k}`).toBeLessThanOrEqual(tol);
      note(`golden_vectors_${tag}_${k}_max_err`, e);
    }
  }

  it("golden.json vectors, classifier from classifier.txt.gz", () => compareGolden(txt, "txt"));
  it("golden.json vectors, classifier from classifier.json.gz", () => compareGolden(json, "json"));

  function compareFixture(model: EmotionModel, tag: string) {
    const errs = { raw: 0, p: 0, mix: 0, reg: 0, confidence: 0 };
    for (const m of fx.model_cases) {
      const r = predictEmotion(model, nn(m.x));
      errs.raw = Math.max(errs.raw, maxErr(r.raw, m.raw));
      errs.p = Math.max(errs.p, maxErr(r.p, m.p));
      errs.confidence = Math.max(errs.confidence, maxErr([r.confidence], [m.confidence]));
      errs.mix = Math.max(errs.mix, maxErr([r.valence, r.arousal], [m.valence_mix, m.arousal_mix]));
      errs.reg = Math.max(
        errs.reg,
        maxErr(
          [
            r.valenceReg ?? NaN,
            r.arousalReg ?? NaN,
            r.valenceRegRaw ?? NaN,
            r.arousalRegRaw ?? NaN,
          ],
          [m.valence, m.arousal, m.valence_raw, m.arousal_raw],
        ),
      );
      expect(r.k, m.name).toBe(m.k);
      expect(r.label, m.name).toBe(m.label);
      expect(r.labelDisplay).toBe(m.affect.label_display);
    }
    for (const [k, e] of Object.entries(errs)) {
      expect(e, `${tag} ${k}`).toBeLessThanOrEqual(TARGET);
      note(`fixture_models_${tag}_${k}_max_err`, e);
    }
  }

  it("fixture model cases (132-value inputs), text models", () => {
    expect(new Set(fx.model_cases.map((m) => m.label)).size).toBe(10);
    compareFixture(txt, "txt");
  });
  it("fixture model cases (132-value inputs), dump models", () => compareFixture(json, "json"));

  it("emotionInput rebuilds the 132 inputs from the 123 features and computeEmotion of the window", () => {
    const recs = fx.sessions.map(
      (s) =>
        new Map(
          s.people.map((p) => [p.person_id, recordsFrom(decodeChannels(p.channels), p.person_id)]),
        ),
    );
    const edges = new Map(fx.edge_windows.map((e) => [e.name, decodeChannels(e.channels)]));
    let n = 0;
    for (const m of fx.model_cases) {
      if (m.window === null || m.x123 === null) continue;
      const w = m.window;
      const emo =
        "edge" in w
          ? emotionFromChannels(edges.get(w.edge) as Record<EmotionField, number[]>)
          : computeEmotion(
              (recs[w.session]?.get(w.person_id) as TickRecord[]).slice(
                w.end - w.hist + 1,
                w.end + 1,
              ),
            );
      const x123 = nn(m.x123);
      expect(x123).toHaveLength(N_FEATURES);
      const x = emotionInput(x123, emo, txt.meta);
      expect(bitEqual(x, nn(m.x)), m.name).toBe(true);
      n++;
    }
    expect(n).toBeGreaterThan(200);
    note("emotion_input_cases_bit_identical", n);
    expect(() => emotionInput(new Float64Array(101), new Float64Array(31), txt.meta)).toThrow();
    expect(() => emotionInput(new Float64Array(123), new Float64Array(30), txt.meta)).toThrow();
  });

  it("toAffect: the Affect message with round(x, 4)", () => {
    let mismatches = 0;
    for (const m of fx.model_cases) {
      const a = toAffect(predictEmotion(txt, nn(m.x)), txt);
      if (JSON.stringify(a) !== JSON.stringify(m.affect)) mismatches++;
      expect(a, m.name).toEqual(m.affect);
    }
    note("affect_mismatches", mismatches);
  });

  it("predictEmotion timing (node)", () => {
    const xs = fx.model_cases.map((m) => Float64Array.from(nn(m.x)));
    const time = (model: EmotionModel, reps: number) => {
      let sink = 0;
      const t0 = performance.now();
      for (let r = 0; r < reps; r++)
        for (const x of xs) sink += (predictEmotion(model, x) as EmotionResult).valence;
      expect(Number.isFinite(sink)).toBe(true);
      return (performance.now() - t0) / (reps * xs.length);
    };
    time(txt, 1); // warm-up
    note("predict_ms_per_vector_with_regressors", Number(time(txt, 5).toFixed(4)));
    const clsOnly = makeEmotionModel(txt.meta, txt.classifier);
    time(clsOnly, 1);
    note("predict_ms_per_vector_classifier_only", Number(time(clsOnly, 5).toFixed(4)));
    note(
      "trees",
      `classifier ${txt.classifier.trees.length}, valence ${txt.valence?.trees.length}, arousal ${txt.arousal?.trees.length}`,
    );
  });
});

// ---------- (d) loader edge cases ----------

describe("loadEmotionModel without the emotion map", () => {
  it("takes the platform fetch as its fetch function (type-level)", () => {
    const f: FetchLike = fetch;
    expect(typeof f).toBe("function");
  });

  it("resolves null when meta.json 404s", async () => {
    const log: string[] = [];
    const got = await loadEmotionModel(
      "/models/emotion/",
      stubFetch(path.join(EMO_DIR, "does-not-exist"), { log }),
    );
    expect(got).toBeNull();
    expect(log).toEqual(["/models/emotion/meta.json"]);
  });

  it("resolves null on a SPA fallback page (index.html with status 200)", async () => {
    const html = new TextEncoder().encode("<!doctype html><html><body>console</body></html>");
    const got = await loadEmotionModel("/m/", async () => ({
      ok: true,
      status: 200,
      arrayBuffer: async () => html.slice().buffer as ArrayBuffer,
    }));
    expect(got).toBeNull();
  });

  it("rejects when meta.json is there but no classifier is", async () => {
    await expect(
      loadEmotionModel(
        "/models/emotion/",
        stubFetch(EMO_DIR, {
          missing: (n) =>
            n.startsWith("classifier") || n.startsWith("valence") || n.startsWith("arousal"),
        }),
      ),
    ).rejects.toThrow(/no classifier/);
  });

  it("rejects on a server error", async () => {
    await expect(
      loadEmotionModel("/models/emotion/", async () => ({
        ok: false,
        status: 500,
        arrayBuffer: async () => new ArrayBuffer(0),
      })),
    ).rejects.toThrow(/500/);
  });

  it("boosterFromText reads both formats of a regressor identically", () => {
    const a = boosterFromText(
      gunzipSync(readFileSync(path.join(EMO_DIR, "valence.txt.gz"))).toString("utf8"),
    );
    const b = boosterFromText(
      gunzipSync(readFileSync(path.join(EMO_DIR, "valence.json.gz"))).toString("utf8"),
    );
    expect(a.trees.length).toBe(b.trees.length);
    const model = makeEmotionModel(
      meta,
      boosterFromText(
        gunzipSync(readFileSync(path.join(EMO_DIR, "classifier.txt.gz"))).toString("utf8"),
      ),
      { valence: b },
    );
    expect(model.arousal).toBeUndefined();
    const r = predictEmotion(model, nn(golden.vectors[0]?.x as Num[]));
    expect(r.arousalReg).toBeNull();
    expect(r.valenceReg).not.toBeNull();
    expect(() => makeEmotionModel(meta, a)).toThrow(/outputs/);
  });
});
