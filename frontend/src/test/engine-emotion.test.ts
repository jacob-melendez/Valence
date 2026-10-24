/**
 * emotion-features.ts + emotion-model.ts against Cloud B's Python emotion map.
 *
 * Two sources of golden values:
 * - artifacts/emotion/golden.json (scripts/emotion_export.py, Cloud B): tick-channel windows with the 31
 *   compute_emotion features, and 23 model inputs with LightGBM's raw scores, the temperature softmax, the map
 *   position (valence_mix / arousal_mix) and the regressors.
 * - emotion.json.gz (scripts/export_web_golden_emotion.py): simulator sessions of the ten-state set with
 *   compute_emotion at every 4th row of every person, synthetic edge windows (no readable face, partial face,
 *   blink-rate threshold, NaN counters, empty, > 180 ticks), 261 model inputs (the frozen 123 features +
 *   the 31) with EmotionPredictor's outputs and the Affect message (as affect_for_window builds it), and whole
 *   TickRecord windows through valence/models/emotion.py affect_for_window.
 * Sections (e) and (f), and the last model tests, pin inputs the fixtures do not cover to literals computed
 * with the .venv: null as missing, the pairwise-summed 10 s blink threshold, old producers without cues,
 * signed yaw, signed zeros, +inf on AvoidInf'd dump thresholds, the row width check, and a toy 3-class dump
 * for class order, temperature, anchors, raw ties and argmax of p.
 *
 * The models are read from ../artifacts/emotion through loadEmotionModel with a fetch stub, once preferring the
 * LightGBM text models (*.txt.gz) and once forced onto the dump_model() JSON (*.json.gz).
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  type Affect,
  makeTickRecord,
  type TickRecord,
  tickRecordFromDict,
} from "@/engine/contracts";
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
  affectForWindow,
  boosterFromText,
  emotionInput,
  loadEmotionModel,
  makeEmotionModel,
  predictEmotion,
  toAffect,
  validateEmotionMeta,
} from "@/engine/emotion-model";
import { FEATURE_SETS, N_FEATURES } from "@/engine/features";
import { loadDumpModel } from "@/engine/lgbm";
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
  /** Whole TickRecord windows through valence/models/emotion.py affect_for_window. */
  affect_windows: {
    name: string;
    person: Record<string, unknown>[];
    room: Record<string, unknown>[];
    affect: AffectJson | null;
  }[];
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

/**
 * Affect vs Python's: label, display and model exact, every number within TARGET (p differs from numpy by a
 * few ulp, see emotion-model.ts); returns the largest numeric error.
 */
function expectAffect(a: Affect, want: AffectJson, what: string): number {
  expect(Object.keys(a), what).toEqual(Object.keys(want));
  expect([a.label, a.label_display, a.model], what).toEqual([
    want.label,
    want.label_display,
    want.model,
  ]);
  expect(Object.keys(a.probabilities), what).toEqual(Object.keys(want.probabilities));
  const ids = Object.keys(want.probabilities);
  const err = maxErr(
    [a.valence, a.arousal, a.confidence, ...ids.map((k) => a.probabilities[k] as number)],
    [
      want.valence,
      want.arousal,
      want.confidence,
      ...ids.map((k) => want.probabilities[k] as number),
    ],
  );
  expect(err, what).toBeLessThanOrEqual(TARGET);
  return err;
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
    const copy = new Uint8Array(bytes); // a real copy: Buffer#slice is a view onto a possibly larger pool
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

  it("toAffect: the Affect message as affect_for_window builds it (unrounded)", () => {
    let identical = 0;
    let worst = 0;
    for (const m of fx.model_cases) {
      const a = toAffect(predictEmotion(txt, nn(m.x)), txt);
      worst = Math.max(worst, expectAffect(a, m.affect, m.name));
      if (JSON.stringify(a) === JSON.stringify(m.affect)) identical++;
    }
    note("affect_max_err", worst);
    note("affect_bit_identical", `${identical}/${fx.model_cases.length}`);
  });

  it("affectForWindow: whole TickRecord windows (affect_for_window, incl. room or person, empty person)", () => {
    expect(fx.affect_windows.length).toBeGreaterThanOrEqual(4);
    for (const w of fx.affect_windows) {
      const got = affectForWindow(
        txt,
        w.person.map(tickRecordFromDict),
        w.room.map(tickRecordFromDict),
      );
      if (w.affect === null) {
        expect(got, w.name).toBeNull();
        continue;
      }
      expect(got, w.name).not.toBeNull();
      note("affect_windows_max_err", expectAffect(got as Affect, w.affect, w.name));
    }
    expect(affectForWindow(txt, [], [])).toBeNull();
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

  // Python literals below: .venv/bin/python -B with valence.models.emotion.EmotionPredictor(artifacts/emotion).

  it("null inputs are missing values, as np.asarray(x, dtype=float) reads None", () => {
    // golden.json vector 0 with every third value null: Python predicts Collaborative_Happy (Calm without
    // the nulls); a null stored as 0 instead of NaN reads as Focused.
    const x = (golden.vectors[0]?.x as Num[]).map((v, j) => (j % 3 === 0 ? null : v));
    for (const model of [txt, json]) {
      const r = predictEmotion(model, x);
      expect(
        bitEqual(
          r.raw,
          [
            -1.8529446928943965, -5.480007054392827, -8.13052744212107, -0.5699124882379649,
            -7.588152795419807, -7.071893911977153, -6.246246243839615, -5.63838010879279,
            -1.1054130901993928, -6.412385136832736,
          ],
        ),
      ).toBe(true);
      expect(r.label).toBe("Collaborative_Happy");
      expect([r.valenceReg, r.arousalReg]).toEqual([0.4711485653986967, 0.34890303780807363]);
      expect(
        maxErr([r.valence, r.arousal], [0.5131636693114864, 0.19293648793990295]),
      ).toBeLessThanOrEqual(1e-15);
    }
    const row = emotionInput(
      new Array<null>(N_FEATURES).fill(null),
      new Array<null>(31).fill(null),
      txt.meta,
    );
    expect(row.every(Number.isNaN)).toBe(true);
  });

  it("a row of the wrong width throws, like LightGBM's shape check", () => {
    // Python: LightGBMError "The number of features in data (131) is not the same as it was in training data (132)"
    expect(() => predictEmotion(txt, new Float64Array(131))).toThrow(/132/);
    expect(() => predictEmotion(txt, new Float64Array(N_FEATURES))).toThrow(/132/);
  });

  it("x = +inf on an AvoidInf'd 1e300 dump threshold decides like LightGBM (both loaders)", () => {
    // blink_rate_1m has dump thresholds printed as 1e300 (Common::AvoidInf); LightGBM sends +inf and 1e301
    // left there. valence/models/emotion_json.py raw_scores reads 1e300 literally and gives -4.240549872195816
    // for class 4 instead; the served Python path (EmotionPredictor, LightGBM) is the reference.
    const want = [
      -4.912850835712999, 1.6677329084260717, -4.898543823628142, -4.528140477128561,
      -4.363780206572808, -5.365849990152937, -4.632673134523343, -4.762989341181959,
      -3.317202148384535, -4.3794788074941975,
    ];
    const f = txt.meta.input_features.indexOf("blink_rate_1m");
    for (const v of [Infinity, 1e301]) {
      const x = nn(golden.vectors[1]?.x as Num[]);
      x[f] = v;
      for (const model of [txt, json]) {
        const r = predictEmotion(model, x);
        expect(bitEqual(r.raw, want), `${v}`).toBe(true);
        expect(r.label).toBe("Frustrated_Masked");
      }
    }
  });

  it("brow_max_1m = -0.0 or +0.0 gives the same prediction (the one non-bit-exact feature)", () => {
    const f = txt.meta.input_features.indexOf("brow_max_1m");
    const a = nn(golden.vectors[0]?.x as Num[]);
    const b = [...a];
    a[f] = 0;
    b[f] = -0;
    const ra = predictEmotion(txt, a);
    const rb = predictEmotion(txt, b);
    expect(bitEqual(ra.raw, rb.raw) && bitEqual(ra.p, rb.p)).toBe(true);
    expect(ra.valenceReg).toBe(rb.valenceReg);
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

  it("a SPA fallback page for classifier.txt.gz falls through to classifier.json.gz", async () => {
    const log: string[] = [];
    const base = stubFetch(EMO_DIR, {
      log,
      missing: (n) => n.startsWith("valence") || n.startsWith("arousal"),
    });
    const html = new TextEncoder().encode("\n  <!doctype html><title>console</title>");
    const m = await loadEmotionModel("/models/emotion/", async (url) =>
      url.endsWith("/classifier.txt.gz")
        ? { ok: true, status: 200, arrayBuffer: async () => html.slice().buffer as ArrayBuffer }
        : base(url),
    );
    expect(m?.classifier.trees.length).toBe(2790);
    expect(m?.valence).toBeUndefined();
    expect(log).toContain("/models/emotion/classifier.json.gz");
  });

  it("a truncated gzip rejects (Python: gzip.decompress raises EOFError)", async () => {
    const base = stubFetch(EMO_DIR);
    const cut = new Uint8Array(
      readFileSync(path.join(EMO_DIR, "classifier.txt.gz")).subarray(0, 50_000),
    );
    await expect(
      loadEmotionModel("/models/emotion/", async (url) =>
        url.endsWith("/classifier.txt.gz")
          ? { ok: true, status: 200, arrayBuffer: async () => cut.slice().buffer as ArrayBuffer }
          : base(url),
      ),
    ).rejects.toThrow();
  });

  it("validateEmotionMeta: own feature names only, states need a display name", () => {
    for (const bad of ["constructor", "__proto__", "toString", "hasOwnProperty"]) {
      expect(
        () => validateEmotionMeta({ ...meta, input_features: [...meta.input_features, bad] }),
        bad,
      ).toThrow(/unknown input feature/);
    }
    const states = meta.states.map((s, i) => (i === 3 ? { ...s, display: undefined } : s));
    expect(() => validateEmotionMeta({ ...meta, states })).toThrow(/display/);
    expect(validateEmotionMeta(meta)).toBe(meta);
  });
});

// ---------- (e) window features on inputs the fixtures do not cover (Python literals) ----------

/** A TickRecord at tick i with every field at its default (cues NaN, blink_count 0, face_visible 0). */
function bareRecord(i: number): TickRecord {
  return makeTickRecord({
    tick: i,
    ts: i * 5,
    t_mono: i * 5,
    session_id: "s",
    person_id: 1,
    display_id: "Person #1",
    n_people: 1,
  });
}

/** Channels for n ticks: every EMOTION_FIELDS channel 0.0, then the overrides. */
function chan(n: number, over: Partial<Record<EmotionField, (number | null)[]>>) {
  const out = {} as Record<EmotionField, (number | null)[]>;
  for (const f of EMOTION_FIELDS) out[f] = over[f] ?? new Array<number>(n).fill(0);
  return out;
}

describe("emotion-v1 window features: adversarial cases (valence/features/emotion.py literals)", () => {
  it("null is NaN in channels and TickRecords, as np.asarray(..., dtype=float) reads None", () => {
    const ch = {} as Record<EmotionField, (number | null)[]>;
    for (const f of EMOTION_FIELDS) ch[f] = [0.2, null, 0.4];
    ch.face_visible = [1.0, null, 1.0];
    ch.blink_count = [1, null, 2];
    // compute_emotion of these channels (a null read as 0 would give a 1m mean of 0.2 and a std of 0.163)
    const cue = [0.4, 0.30000000000000004, 0.30000000000000004, 0.1];
    const want = [
      ...cue,
      ...cue,
      ...cue,
      ...cue,
      ...cue,
      ...cue,
      0.30000000000000004,
      0.4,
      0.30000000000000004,
      0.30000000000000004,
      18.0,
      18.0,
      18.0,
    ];
    expect(bitEqual(emotionFromChannels(ch), want)).toBe(true);
    // rows taken from JSON without tickRecordFromDict carry the nulls themselves
    const recs = [0, 1, 2].map((i) => {
      const r = bareRecord(i) as unknown as Record<string, number | null>;
      for (const f of EMOTION_FIELDS) r[f] = ch[f][i] as number | null;
      return r as unknown as TickRecord;
    });
    expect(bitEqual(computeEmotion(recs), want)).toBe(true);
  });

  it("blink threshold: face_visible 0.2 x 10 sums (numpy pairwise) to exactly 10 s", () => {
    const vis = new Array<number>(10).fill(0.2);
    let naive = 0;
    for (const v of vis) naive += v;
    expect(naive * 5).toBeLessThan(10); // 9.999999999999998: a left-to-right sum would give NaN rates
    const f = emotionFromChannels(
      chan(10, { face_visible: vis, blink_count: [0, 1, 0, 0, 1, 0, 0, 0, 1, 0] }),
    );
    expect([...f.slice(28)]).toEqual([18.0, 18.0, 18.0]); // Python: 60 * 3 / 10.0
  });

  it("an old producer (face visible, no cue fields): 28 NaN cue features, blink rates 0.0", () => {
    // makeTickRecord defaults: the seven cues NaN, blink_count 0 (valence/contracts.py:274-281)
    const recs = Array.from({ length: 15 }, (_, i) => ({ ...bareRecord(i), face_visible: 1 }));
    const f = computeEmotion(recs);
    expect(f.slice(0, 28).every(Number.isNaN)).toBe(true);
    expect([...f.slice(28)]).toEqual([0, 0, 0]);
  });

  it("head_yaw: signed values are averaged as they are (no abs in compute_emotion)", () => {
    const yaw = [-30.0, -10.0, 5.0, -2.5, -30.0, -10.0, 5.0, -2.5, -30.0, -10.0, 5.0, -2.5];
    const f = emotionFromChannels(
      chan(12, { head_yaw: yaw, face_visible: new Array<number>(12).fill(1) }),
    );
    // yaw_now, yaw_mean_1m, yaw_mean_5m, yaw_std_5m
    expect(bitEqual(f.slice(16, 20), [-2.5, -9.375, -9.375, 13.035408509133882])).toBe(true);
  });

  it("brow_max_1m of a 0.0 / -0.0 mix is a zero (numpy may return -0.0: SIMD lane order)", () => {
    const f = emotionFromChannels(chan(2, { brow_down: [-0, 0] }));
    expect(f[25] === 0).toBe(true);
  });
});

// ---------- (f) a toy multiclass dump: class order, temperature, anchors, ties ----------

describe("predictEmotion on a toy 3-class dump (Python: emotion_json.raw_scores + EmotionPredictor.predict)", () => {
  const leaf = (v: number, i: number) => ({ leaf_index: i, leaf_value: v, leaf_count: 10 });
  const single = (idx: number, v: number) => ({
    tree_index: idx,
    num_leaves: 1,
    num_cat: 0,
    shrinkage: 1,
    tree_structure: leaf(v, 0),
  });
  const split = (f: number, thr: number, dl: boolean, mt: string, l: number, r: number) => ({
    tree_index: 0,
    num_leaves: 2,
    num_cat: 0,
    shrinkage: 1,
    tree_structure: {
      split_index: 0,
      split_feature: f,
      split_gain: 1,
      threshold: thr,
      decision_type: "<=",
      default_left: dl,
      missing_type: mt,
      internal_value: 0,
      internal_weight: 0,
      internal_count: 20,
      left_child: leaf(l, 0),
      right_child: leaf(r, 1),
    },
  });
  // round 0: class A splits on brow_now (NaN goes right), B 1.0, C 0.5; round 1: A 0.25, B splits on smile_now
  // (None-type: NaN reads 0), C -0.1. Tree t adds to class t % 3.
  const dump = {
    name: "tree",
    version: "v4",
    num_class: 3,
    num_tree_per_iteration: 3,
    label_index: 0,
    max_feature_idx: 1,
    objective: "multiclass num_class:3",
    average_output: false,
    feature_names: ["brow_now", "smile_now"],
    tree_info: [
      split(0, 0.5, false, "NaN", 1.0, 0.2),
      single(1, 1.0),
      single(2, 0.5),
      single(3, 0.25),
      split(1, 1.0000000180025095e-35, false, "None", 0.25, 0.2500000000000002),
      single(5, -0.1),
    ],
  };
  const toyMeta: EmotionMeta = {
    feature_version: "emotion-v1",
    model_id: "toy@00000000",
    input_features: ["brow_now", "smile_now"],
    emotion_feature_names: [],
    n_features: 2,
    temperature: 1.02,
    states: [
      { id: "A", display: "State A", valence: 0.1, arousal: 0.3 },
      { id: "B", display: "State B", valence: -0.6, arousal: 0.6 },
      { id: "C", display: "State C", valence: 0.7, arousal: -0.4 },
    ],
  };
  const tie = {
    raw: [1.25, 1.25, 0.4],
    p: [0.4107453938418902, 0.4107453938418902, 0.17850921231621972],
    k: 0,
    mix: [-0.08041624829959129, 0.29826716953121324],
  };
  const bWins = {
    raw: [0.45, 1.25, 0.4],
    p: [0.24136718151967113, 0.5288120492425569, 0.229820769237772],
    k: 1,
    mix: [-0.13227597292712662, 0.29776907630632665],
  };
  const cases: { note: string; x: (number | null)[]; want: typeof tie }[] = [
    { note: "exact tie in raw: the first class", x: [0.3, 0], want: tie },
    { note: "class B ahead", x: [0.7, 0], want: bWins },
    { note: "null on a NaN-type split goes the default way (right)", x: [null, 0], want: bWins },
    {
      note: "raw B > A by 1 ulp, equal after / 1.02: argmax of p picks A",
      x: [0.3, 1],
      want: { ...tie, raw: [1.25, 1.2500000000000002, 0.4] },
    },
    { note: "null on a None-type split reads 0", x: [0.3, null], want: tie },
    { note: "5e-36 is in LightGBM's zero band", x: [0, 5e-36], want: tie },
  ];

  it("raw, p, label, map position; no regressors", () => {
    const model = makeEmotionModel(toyMeta, loadDumpModel(dump));
    for (const c of cases) {
      const r = predictEmotion(model, c.x);
      expect(bitEqual(r.raw, c.want.raw), c.note).toBe(true);
      expect(maxErr(r.p, c.want.p), c.note).toBeLessThanOrEqual(1e-15);
      expect([r.k, r.label, r.labelDisplay], c.note).toEqual([
        c.want.k,
        toyMeta.states[c.want.k]?.id,
        toyMeta.states[c.want.k]?.display,
      ]);
      expect(r.confidence).toBe(r.p[r.k]);
      expect(maxErr([r.valence, r.arousal], c.want.mix), c.note).toBeLessThanOrEqual(1e-15);
      expect([r.valenceReg, r.arousalReg, r.valenceRegRaw, r.arousalRegRaw]).toEqual([
        null,
        null,
        null,
        null,
      ]);
      const a = toAffect(r, model);
      expect(Object.keys(a.probabilities)).toEqual(["A", "B", "C"]);
      expect(a.model).toBe("toy@00000000");
    }
  });
});
