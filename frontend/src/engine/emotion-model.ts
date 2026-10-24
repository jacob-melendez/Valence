/**
 * Emotion-map inference for the browser engine: Cloud B's 10-state LightGBM classifier and the valence /
 * arousal regressors (valence/models/emotion.py EmotionPredictor; spec: docs/EMOTION_FEATURES.md section 5).
 *
 * The models are read from /models/emotion/ (frontend/scripts/model-files.mjs serves artifacts/emotion/
 * there): meta.json, then each model as LightGBM text (`*.txt.gz`, what Python loads) or, failing that, as
 * `booster.dump_model()` JSON (`*.json.gz`); lgbm.ts turns both into the same trees. Inputs are the 132
 * values of meta.json input_features: the served M2 columns of the frozen 123 features (features.ts) followed
 * by the 31 emotion-v1 features (emotion-features.ts), NaN = missing (LightGBM's native handling).
 *
 * Inference (meta.json inference):
 *   p = softmax(raw / temperature); the served map position is the probability-weighted anchors
 *   valence = sum_k p_k * states[k].valence (likewise arousal); the regressors, exported for comparison, are
 *   clip(raw, -1, 1).
 *
 * Parity: raw scores and regressors are bit-identical to LightGBM (lgbm.ts), also for inputs exactly on a
 * split threshold, in the +-1e-35 zero band, +-inf and NaN. Probabilities go through Math.exp, and numpy's
 * np.exp (an AVX-512 kernel here) itself differs from libm exp in the last bit for ~4% of inputs: p stays
 * within 3 ulp. Python's `p @ anchors` runs through OpenBLAS and is not even self-consistent (one row vs a
 * batch gives different bits for ~half the values), so the map position (numpy's pairwise order here) has
 * no single bit target; it stays within 4e-16. Tests target <= 1e-9.
 * Golden values: artifacts/emotion/golden.json (Cloud B) and scripts/export_web_golden_emotion.py ->
 * src/test/fixtures/engine/emotion.json.gz.
 */
import type { Affect, TickRecord } from "./contracts";
import { EMOTION_INDEX, N_EMOTION_FEATURES, computeEmotion } from "./emotion-features";
import { FEATURE_INDEX, N_FEATURES, featuresFromWindow } from "./features";
import { type Booster, loadDumpModel, parseLightGBMText, predictRaw } from "./lgbm";
import { clip, isnan, pairwiseSum } from "./num";

// ---------- Types ----------

/** One emotion state with its (assumed) valence-arousal anchor, in class order (meta.json states). */
export interface EmotionStateMeta {
  id: string;
  display: string;
  valence: number;
  arousal: number;
}

/** File names of one model (meta.json files): `python` is the LightGBM text model, `json` the dump. */
export interface EmotionModelFiles {
  json: string;
  python: string;
}

export type EmotionModelName = "classifier" | "valence" | "arousal";

/** artifacts/emotion/meta.json, as written by valence/models/emotion.py:218 export. */
export interface EmotionMeta {
  feature_version: string;
  /** "emotion-v1@<sha8>" over the three dumps (valence/models/emotion.py:140 model_id). */
  model_id: string;
  /** The model input columns, in order (132: FEATURE_SETS.m2 + the 31 emotion features). */
  input_features: string[];
  emotion_feature_names: string[];
  n_features: number;
  states: EmotionStateMeta[];
  anchors_are_assumptions?: boolean;
  /** Softmax temperature fitted on the calibration sessions. */
  temperature: number;
  inference?: Record<string, string>;
  files?: Partial<Record<EmotionModelName, EmotionModelFiles>>;
  /** Feature -> explanation layer. */
  layers?: Record<string, string>;
  training?: Record<string, unknown>;
  summary?: Record<string, unknown>;
}

/** A loaded emotion map. The regressors are optional (the served map position needs only the classifier). */
export interface EmotionModel {
  meta: EmotionMeta;
  modelId: string;
  classifier: Booster;
  valence?: Booster;
  arousal?: Booster;
}

/** One prediction (valence/models/emotion.py:178 EmotionPredictor.predict for a single row). */
export interface EmotionResult {
  /** Classifier raw scores, one per state (class order). */
  raw: Float64Array;
  /** softmax(raw / temperature). */
  p: Float64Array;
  /** argmax p (the first maximum, like np.argmax). */
  k: number;
  label: string;
  labelDisplay: string;
  /** p[k]. */
  confidence: number;
  /** Map position (served): probability-weighted anchors (`valence_mix`). */
  valence: number;
  arousal: number;
  /** Regressors clipped to [-1, 1]; null when that model is not loaded. */
  valenceReg: number | null;
  arousalReg: number | null;
  /** Regressor raw scores (before the clip); null when that model is not loaded. */
  valenceRegRaw: number | null;
  arousalRegRaw: number | null;
}

// ---------- Building a model ----------

function fail(msg: string): never {
  throw new Error(`emotion model: ${msg}`);
}

/**
 * Where an input column comes from: an index into the 123-vector (>= 0), ~index into the 31-vector (< 0), or
 * undefined for an unknown name (Python: KeyError). Own keys only, so "constructor" or "__proto__" in a
 * malformed meta.json cannot resolve to an Object.prototype member.
 */
function inputColumn(name: string): number | undefined {
  if (Object.hasOwn(EMOTION_INDEX, name)) return ~(EMOTION_INDEX[name] as number);
  if (Object.hasOwn(FEATURE_INDEX, name)) return FEATURE_INDEX[name] as number;
  return undefined;
}

/** Shape checks on meta.json; throws on anything the inference below would trip over. */
export function validateEmotionMeta(meta: unknown): EmotionMeta {
  const m = meta as Partial<EmotionMeta> | null;
  if (!m || typeof m !== "object") fail("meta.json is not an object");
  if (!Array.isArray(m.input_features) || !m.input_features.every((n) => typeof n === "string"))
    fail("meta.json input_features missing");
  if (!Array.isArray(m.states) || m.states.length === 0) fail("meta.json states missing");
  for (const s of m.states) {
    if (
      typeof s?.id !== "string" ||
      typeof s.display !== "string" ||
      typeof s.valence !== "number" ||
      typeof s.arousal !== "number"
    )
      fail("meta.json states need id, display, valence and arousal");
  }
  if (typeof m.temperature !== "number" || !(m.temperature > 0))
    fail("meta.json temperature missing");
  if (typeof m.model_id !== "string") fail("meta.json model_id missing");
  for (const n of m.input_features) {
    if (inputColumn(n) === undefined) fail(`unknown input feature ${JSON.stringify(n)}`);
  }
  return m as EmotionMeta;
}

/**
 * Assemble an EmotionModel from loaded boosters, checking that they fit meta.json: the classifier has one
 * tree per state per round and reads meta.input_features; each regressor has one output and the same input.
 */
export function makeEmotionModel(
  meta: EmotionMeta,
  classifier: Booster,
  regressors: { valence?: Booster | null; arousal?: Booster | null } = {},
): EmotionModel {
  const m = validateEmotionMeta(meta);
  const n = m.input_features.length;
  const check = (name: string, b: Booster, outputs: number) => {
    if (b.numFeatures !== n) fail(`${name} reads ${b.numFeatures} features, meta.json lists ${n}`);
    if (b.numTreePerIteration !== outputs)
      fail(`${name} has ${b.numTreePerIteration} outputs, expected ${outputs}`);
  };
  check("classifier", classifier, m.states.length);
  const model: EmotionModel = { meta: m, modelId: m.model_id, classifier };
  if (regressors.valence) {
    check("valence regressor", regressors.valence, 1);
    model.valence = regressors.valence;
  }
  if (regressors.arousal) {
    check("arousal regressor", regressors.arousal, 1);
    model.arousal = regressors.arousal;
  }
  return model;
}

/**
 * A model from its file text: `{...}` is `dump_model()` JSON (loadDumpModel), anything else LightGBM's text
 * format (parseLightGBMText).
 */
export function boosterFromText(text: string): Booster {
  return text.trimStart().startsWith("{")
    ? loadDumpModel(JSON.parse(text))
    : parseLightGBMText(text);
}

// ---------- Loading ----------

/** The part of `fetch`'s Response the loader uses (so tests can pass a stub). */
export interface FetchResponseLike {
  ok: boolean;
  status: number;
  arrayBuffer(): Promise<ArrayBuffer>;
}
export type FetchLike = (url: string) => Promise<FetchResponseLike>;

/**
 * gunzip with the platform's DecompressionStream (browsers and Node >= 18). Like Python's gzip.decompress it
 * reads concatenated members and rejects a truncated stream; unlike it, NUL padding after the last member is
 * rejected (Python strips it). The exporters (valence/models/emotion.py:129 dump_gz, :150 save_text_gz) never
 * pad.
 */
export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const body = new Response(bytes as BodyInit).body;
  if (!body) fail("cannot stream the gzip payload");
  const out = new Response(body.pipeThrough(new DecompressionStream("gzip")));
  return new Uint8Array(await out.arrayBuffer());
}

/**
 * File bytes -> text. gzip payloads (magic 1f 8b) are inflated; anything else is taken as already decoded,
 * which covers hosts that serve `.gz` files with `Content-Encoding: gzip` (the browser inflates those itself).
 */
async function decodeText(bytes: Uint8Array): Promise<string> {
  const raw =
    bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b ? await gunzip(bytes) : bytes;
  return new TextDecoder("utf-8").decode(raw);
}

/** A static host's SPA fallback answers a missing file with index.html (status 200): treat it as missing. */
function looksLikeHtml(text: string): boolean {
  return text.trimStart().startsWith("<");
}

/**
 * Load the emotion map (valence/models/emotion.py:162 EmotionPredictor.__init__, for the browser).
 *
 * Fetches `${baseUrl}meta.json`, then for each model its text model (`classifier.txt.gz`, preferred: smaller
 * and what Python loads) or its dump (`classifier.json.gz`), file names from meta.json `files`. Resolves null
 * when meta.json is missing (404, or a SPA fallback page), so the console runs without the emotion map; a
 * present meta.json without a loadable classifier is an error. The regressors are optional.
 */
export async function loadEmotionModel(
  baseUrl = "/models/emotion/",
  fetchFn: FetchLike = (url) => globalThis.fetch(url),
): Promise<EmotionModel | null> {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  /** Text of a file, or null when it is missing (404 / 410 / SPA fallback). */
  const fetchText = async (name: string): Promise<string | null> => {
    const res = await fetchFn(base + name);
    if (!res.ok) {
      if (res.status === 404 || res.status === 410) return null;
      fail(`GET ${base + name} failed with status ${res.status}`);
    }
    const text = await decodeText(new Uint8Array(await res.arrayBuffer()));
    return looksLikeHtml(text) ? null : text;
  };

  const metaText = await fetchText("meta.json");
  if (metaText === null) return null;
  const meta = validateEmotionMeta(JSON.parse(metaText));

  const loadOne = async (name: EmotionModelName): Promise<Booster | null> => {
    const files = meta.files?.[name];
    for (const file of [files?.python ?? `${name}.txt.gz`, files?.json ?? `${name}.json.gz`]) {
      const text = await fetchText(file);
      if (text !== null) return boosterFromText(text);
    }
    return null;
  };
  const [classifier, valence, arousal] = await Promise.all([
    loadOne("classifier"),
    loadOne("valence"),
    loadOne("arousal"),
  ]);
  if (!classifier) fail(`meta.json found under ${base} but no classifier model next to it`);
  return makeEmotionModel(meta, classifier, { valence, arousal });
}

// ---------- Inputs ----------

/** meta -> per input column: index into the 123-vector (>= 0) or ~index into the 31-vector (< 0). */
const PLANS = new WeakMap<EmotionMeta, Int32Array>();

function inputPlan(meta: EmotionMeta): Int32Array {
  let plan = PLANS.get(meta);
  if (!plan) {
    plan = Int32Array.from(meta.input_features, (name) => {
      const c = inputColumn(name);
      if (c === undefined) fail(`unknown input feature ${JSON.stringify(name)}`);
      return c;
    });
    PLANS.set(meta, plan);
  }
  return plan;
}

/** `np.asarray(x, dtype=float)` for one value: null (JSON's NaN) and undefined are NaN, not 0. */
function asFloat(v: number | null | undefined): number {
  return v === null || v === undefined ? NaN : v;
}

/**
 * The emotion models' input row (valence/features/emotion.py:72 model_input): the frozen 123 features
 * (features.ts compute / featuresFromWindow, unclipped) and the 31 emotion features -> meta.input_features
 * order, each column looked up by name in FEATURE_NAMES or EMOTION_FEATURE_NAMES. null is NaN, as numpy
 * reads None (a plain Float64Array store would make it 0).
 */
export function emotionInput(
  full123: ArrayLike<number | null>,
  emo31: ArrayLike<number | null>,
  meta: EmotionMeta,
): Float64Array {
  if (full123.length !== N_FEATURES)
    fail(`expected the ${N_FEATURES} features, got ${full123.length}`);
  if (emo31.length !== N_EMOTION_FEATURES)
    fail(`expected the ${N_EMOTION_FEATURES} emotion features, got ${emo31.length}`);
  const plan = inputPlan(meta);
  const x = new Float64Array(plan.length);
  for (let i = 0; i < plan.length; i++) {
    const j = plan[i] as number;
    x[i] = asFloat(j >= 0 ? full123[j] : emo31[~j]);
  }
  return x;
}

// ---------- Prediction ----------

/** Weighted anchors `sum_k p_k * a_k` (numpy: p @ anchors); numpy's pairwise order. */
function weighted(
  p: Float64Array,
  states: readonly EmotionStateMeta[],
  key: "valence" | "arousal",
): number {
  const terms = new Float64Array(p.length);
  for (let k = 0; k < p.length; k++)
    terms[k] = (p[k] as number) * (states[k] as EmotionStateMeta)[key];
  return pairwiseSum(terms);
}

/** One regressor's raw score and its clip to [-1, 1]; nulls when the model is absent. */
function regress(b: Booster | undefined, x: ArrayLike<number>): [number | null, number | null] {
  if (!b) return [null, null];
  const raw = predictRaw(b, x)[0] as number;
  return [clip(raw, -1, 1), raw];
}

/**
 * Predict one input row (meta.input_features order, NaN or null = missing), as EmotionPredictor.predict
 * (valence/models/emotion.py:178): `z = raw / T; z -= max(z); p = exp(z) / sum(exp(z))`, the label is the
 * first argmax of p (valence/eval/emotion_eval.py:211; not of raw: two raw scores an ulp apart can divide
 * to the same z), and the map position the probability-weighted anchors. A row of the wrong width throws,
 * as LightGBM's shape check does.
 */
export function predictEmotion(model: EmotionModel, xIn: ArrayLike<number | null>): EmotionResult {
  const { meta } = model;
  const states = meta.states;
  // np.asarray(X, dtype=float): null -> NaN (lgbm.ts would read a null, or a hole, as 0.0)
  const x = new Float64Array(xIn.length);
  for (let i = 0; i < x.length; i++) x[i] = asFloat(xIn[i]);
  const raw = predictRaw(model.classifier, x);
  const K = raw.length;
  const z = new Float64Array(K);
  let zmax = -Infinity; // np.max: NaN propagates
  for (let k = 0; k < K; k++) {
    const v = (raw[k] as number) / meta.temperature;
    z[k] = v;
    if (isnan(v)) zmax = NaN;
    else if (v > zmax) zmax = v;
  }
  const e = new Float64Array(K);
  for (let k = 0; k < K; k++) e[k] = Math.exp((z[k] as number) - zmax);
  const s = pairwiseSum(e);
  const p = new Float64Array(K);
  for (let k = 0; k < K; k++) p[k] = (e[k] as number) / s;

  // np.argmax: the first NaN if any, else the first maximum
  let best = 0;
  for (let k = 0; k < K; k++) {
    const v = p[k] as number;
    if (isnan(v)) {
      best = k;
      break;
    }
    if (v > (p[best] as number)) best = k;
  }
  const st = states[best] as EmotionStateMeta;
  const [valenceReg, valenceRegRaw] = regress(model.valence, x);
  const [arousalReg, arousalRegRaw] = regress(model.arousal, x);
  return {
    raw,
    p,
    k: best,
    label: st.id,
    labelDisplay: st.display,
    confidence: p[best] as number,
    valence: weighted(p, states, "valence"),
    arousal: weighted(p, states, "arousal"),
    valenceReg,
    arousalReg,
    valenceRegRaw,
    arousalRegRaw,
  };
}

// ---------- Affect message ----------

/**
 * The `Prediction.affect` object (valence/contracts.py:373 Affect), as valence/models/emotion.py:217
 * affect_for_window builds it: the served map position (valence_mix / arousal_mix), the label (first argmax of
 * p) with its display name, the confidence p[k], the probabilities keyed by state id in class order, and the
 * model id. Plain floats, no rounding (Python passes `float(...)` straight into Affect).
 */
export function toAffect(result: EmotionResult, model: EmotionModel): Affect {
  const probabilities: Record<string, number> = {};
  model.meta.states.forEach((s, k) => {
    probabilities[s.id] = result.p[k] as number;
  });
  return {
    valence: result.valence,
    arousal: result.arousal,
    label: result.label,
    label_display: result.labelDisplay,
    confidence: result.confidence,
    probabilities,
    model: model.modelId,
  };
}

/**
 * Live path, valence/models/emotion.py:217 affect_for_window: RollingWindow snapshots (TickRecords, oldest
 * first) -> the Affect for the person's latest tick, or null for an empty person window. The input row is
 * `model_input(features_from_window(person, room or person), emotion_from_window(person))`; an empty room
 * window falls back to the person's rows, as in Python.
 */
export function affectForWindow(
  model: EmotionModel,
  person: readonly TickRecord[],
  room: readonly TickRecord[],
): Affect | null {
  if (person.length === 0) return null;
  const x = emotionInput(
    featuresFromWindow(person, room.length ? room : person),
    computeEmotion(person),
    model.meta,
  );
  return toAffect(predictEmotion(model, x), model);
}
