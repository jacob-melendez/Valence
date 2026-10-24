/**
 * The served predictor: M2 (context + face history) with M0 (face snapshot) running in its shadow
 * (valence/models/served.py).
 *
 * Per window: features -> clip to the training range -> M2 calibrated probabilities -> TreeSHAP of the
 * predicted class -> explanation layers. M0 gives the face-only reading; the mask-override condition compares
 * the two (docs/PLAN.md, MaskOverrideDetector). The stateful parts (label hysteresis, override persistence,
 * warm-up) live in smoother.ts.
 *
 * Models come in as already-fetched text (m2.txt, m0.txt) plus the parsed model_meta.json and clip_bounds.json;
 * `loadServed` is the async browser loader for /models/core/ and /models/all/ (frontend/scripts/model-files.mjs).
 *
 * Parity: clipping, the OOD rule, raw scores, SHAP contributions, the face push and the layers are
 * bit-identical to Python (lgbm.ts, treeshap.ts, explainer.ts). Probabilities go through Math.exp, which
 * differs from numpy's SIMD np.exp in the last bit for a few percent of inputs, so p (and the confidence) can
 * be an ulp or two off (162 of 2484 golden values, at most 2 ulp). That can only matter at an exact tie:
 * argmax, the 0.70 / 0.45 override gates and round(p, 4); no golden case or replayed tick hits one.
 * Only the SHAP rows of the predicted class k and the M0 class k0 are computed (Python computes all K; the
 * rows are independent, so the two it reads come out identical).
 * Golden values: scripts/export_web_golden.py -> src/test/fixtures/engine/models.json.gz, plus the tables,
 * synthetic-model and counterfactual cases in src/test/fixtures/engine/smoother.json.gz.
 */
import { layerOf, type Layer, type TickRecord } from "./contracts";
import { buildLayers } from "./explainer";
import { ABS_ROOM, FEATURE_NAMES, N_FEATURES, featureIndex, featuresFromWindow } from "./features";
import { type Booster, parseLightGBMText, predictRaw } from "./lgbm";
import { isnan, pairwiseSum, pyRound } from "./num";
import { sha256Hex, utf8Encode } from "./sha";
import { treeShap } from "./treeshap";

// ---------- Constants ----------

/** Face-only model must clearly read something else (chance is 1/4 or 1/6) (valence/models/served.py:26). */
export const M0_MIN_CONF = 0.45;
/** valence/models/served.py:27 */
export const M2_MIN_CONF = 0.7;
/** A "mask" is a positive-looking face over a non-positive state (valence/models/served.py:28). */
export const POSITIVE: ReadonlySet<string> = new Set(["Collaborative_Happy"]);
/** More out-of-distribution features than this pull probabilities toward uniform (valence/models/served.py:29). */
export const OOD_SHRINK_AFTER = 3;
/** valence/models/served.py:30 */
export const OOD_SHRINK = 0.3;
/** Unbounded features are OOD only this fraction of the training range beyond p1/p99 (valence/models/served.py:31). */
export const OOD_MARGIN = 0.5;

/**
 * Python's `$` (no MULTILINE) also matches just before one trailing "\n"; `\w` is Unicode-aware for str
 * patterns. Feature names are plain ASCII, so this only matters for odd names passed to oodBand directly.
 */
const W = String.raw`[\p{L}\p{N}_]`;
const END = String.raw`(?=\n?$)`;

/**
 * Clipping to the training p1..p99 range is not evidence of a strange input: a face visible on every frame
 * (1.0 vs a training p99 of 0.998) or an 08:00 time of day are legitimate. Features with a physical domain are
 * OOD only outside it; clock/context features never are; the rest only well beyond the training range
 * (valence/models/served.py:36-45). First matching pattern wins.
 */
export const NEVER_OOD: ReadonlySet<string> = new Set([
  "tod_sin",
  "tod_cos",
  "hours_since_break",
  "n_people",
  "fill_ratio",
]);
export const PHYSICAL_DOMAINS: readonly (readonly [RegExp, readonly [number, number]])[] = [
  [new RegExp(`^face_(now|mean_${W}+)${END}`, "u"), [-1.0, 1.0]],
  [new RegExp(`^face_std_5m${END}`, "u"), [0.0, 1.0]],
  [new RegExp(`^(face_visible_${W}+|withdraw_${W}+)${END}`, "u"), [0.0, 1.0]],
  [new RegExp(`^(slump|hih)_(now|mean_${W}+|max_${W}+|ewm_${W}+)${END}`, "u"), [0.0, 1.0]],
  [new RegExp(`^(slump|hih)_std_${W}+${END}`, "u"), [0.0, 0.5]],
  [new RegExp(`^${W}+_cnt_${W}+${END}`, "u"), [0.0, Infinity]], // event counts are legitimately unbounded
  [new RegExp(`^${W}+_since_s${END}`, "u"), [0.0, Infinity]],
];

// ---------- Types ----------

/** The parts of artifacts/model_meta.json the served predictor reads (written by valence/models/train.py). */
export interface ServedMeta {
  states: string[];
  state_set?: string;
  temperatures: { m2: number; m0: number } & Readonly<Record<string, number>>;
  feature_sets: { m2: string[]; m0: string[] } & Readonly<Record<string, string[]>>;
  [key: string]: unknown;
}

/** artifacts/clip_bounds.json: feature -> [p1, p99] of the training data. Missing features are unbounded. */
export type ClipBounds = Readonly<Record<string, readonly [number, number] | readonly number[]>>;

/**
 * One window's prediction (valence/models/predictor.py:23 PredictionOutput). Field names are the Python
 * dataclass's, so engine.py ports line by line. `probabilities` keeps the state order; `values` holds the
 * unclipped M2 inputs (NaN = missing).
 */
export interface PredictionOutput {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  face_only_label: string | null;
  face_only_confidence: number | null;
  layers: Layer[];
  /** This tick's mask-override condition; null when no face this tick. */
  raw_override: boolean | null;
  values: Record<string, number>;
  ood_features: string[];
}

/** What the runtime needs from a predictor (valence/models/predictor.py:35 Predictor protocol). */
export interface Predictor {
  readonly states: readonly string[];
  readonly modelId: string;
  predict(person: readonly TickRecord[], room: readonly TickRecord[]): PredictionOutput | null;
}

/** The M0 shadow reading of one window (only when the face is visible). */
export interface M0Reading {
  raw: Float64Array;
  p: Float64Array;
  k0: number;
  conf: number;
  /** Sum of M2's Facial Surface SHAP values for class k0: how much the face pushed M2 toward M0's reading. */
  facePush: number;
}

/** Every intermediate of predictVector, for tests and the debug console. */
export interface PredictionTrace {
  output: PredictionOutput;
  /** The clipped 123-vector (NaN kept). */
  xc: Float64Array;
  ood: string[];
  /** M2 raw scores, one per state. */
  raw: Float64Array;
  /** M2 SHAP row of the predicted class k: one entry per M2 column, then the bias. */
  contribK: Float64Array;
  /** Probabilities after the temperature softmax and the OOD shrink. */
  p: Float64Array;
  k: number;
  m0: M0Reading | null;
}

/** What `new ServedPredictor` takes: the model files' contents. */
export interface ServedArtifacts {
  /** m2.txt as bytes (hashed for the model id exactly as Python does) or as text (hashed as UTF-8). */
  m2: string | Uint8Array;
  m0: string | Uint8Array;
  meta: ServedMeta;
  clipBounds: ClipBounds;
}

// ---------- Numerics ----------

/** `np.max` over a 1-d array: NaN as soon as any element is NaN; throws on an empty array (numpy: ValueError). */
function npMax(a: ArrayLike<number>): number {
  if (a.length === 0)
    throw new RangeError("zero-size array to reduction operation maximum which has no identity");
  let m = a[0] as number;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (isnan(v)) return NaN;
    if (v > m) m = v;
  }
  return m;
}

/** `int(np.argmax(a))`: the first maximum, or the first NaN if there is one (numpy's rule). */
export function npArgmax(a: ArrayLike<number>): number {
  if (a.length === 0) throw new RangeError("attempt to get argmax of an empty sequence");
  let k = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (isnan(v)) return i;
    if (v > (a[k] as number)) k = i;
  }
  return k;
}

/**
 * valence/models/served.py:61 `_softmax(z, t)`: z / t, exp(z - max), divide by numpy's (pairwise) sum.
 * Math.exp may differ from numpy's np.exp by an ulp.
 */
export function softmax(z: ArrayLike<number>, t: number): Float64Array {
  const zt = Float64Array.from(z, (v) => v / t);
  const m = npMax(zt);
  const e = zt.map((v) => Math.exp(v - m));
  const s = pairwiseSum(e);
  return e.map((v) => v / s);
}

/**
 * Elementwise `np.clip(x, lo, hi)` over arrays, as served.py:94 calls it (numpy clip.cpp: min(max(x, lo), hi)
 * with `a > b ? a : b`, NaN in x propagating). On a signed-zero tie the BOUND wins: clip(-0.0, 0.0, 1.0) is
 * +0.0. This differs from num.ts clip, which follows numpy's scalar path (where the input wins); checked on
 * 3000 random arrays of +-0 / +-1 / +-inf / NaN against numpy 2.5.
 */
function npClipArrayElem(x: number, lo: number, hi: number): number {
  const a = isnan(x) ? x : x > lo ? x : lo;
  return isnan(a) ? a : a < hi ? a : hi;
}

/** `np.asarray(x, dtype=float)` for one 123-vector entry: null / undefined (a hole) is NaN, not 0. */
function asFloat(v: number | null | undefined): number {
  return v === null || v === undefined ? NaN : v;
}

/** Python float() literal syntax (ASCII digits; single underscores between digits). */
const PY_FLOAT_RE =
  /^[+-]?(?:(?:\d(?:_?\d)*)?\.\d(?:_?\d)*|\d(?:_?\d)*\.?)(?:[eE][+-]?\d(?:_?\d)*)?$/;
const PY_FLOAT_SPECIAL_RE = /^[+-]?(?:inf|infinity|nan)$/i;

/**
 * Python `float(v)` for a value read from JSON (valence/models/served.py:75 `float(meta["temperatures"]["m2"])`).
 * Missing (undefined), null, objects and malformed strings raise, as KeyError / TypeError / ValueError do in
 * Python. JS Number() would turn them into NaN or 0 and every prediction would silently go NaN.
 */
function pyFloat(v: unknown, what: string): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") {
    const s = v.trim();
    if (PY_FLOAT_SPECIAL_RE.test(s))
      return /nan/i.test(s) ? NaN : s.startsWith("-") ? -Infinity : Infinity;
    if (PY_FLOAT_RE.test(s)) return Number(s.replaceAll("_", ""));
    throw new Error(`served predictor: ${what}: could not convert string to float: '${v}'`);
  }
  throw new TypeError(
    `served predictor: ${what} must be a real number, not ${v === undefined ? "missing" : typeof v}`,
  );
}

/**
 * One entry of a clip_bounds.json pair as numpy reads it. Python builds `np.array` over all 123 bounds, so a
 * null turns the array into dtype=object and ood_band (math.isfinite) or the first np.clip raises a
 * TypeError; a bool is 1.0 / 0.0. Only numbers and bools are accepted here.
 */
function clipBound(v: unknown, what: string): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  throw new TypeError(
    `served predictor: ${what} must be a real number, not ${v === null ? "null" : typeof v}`,
  );
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// ---------- OOD bands ----------

/** Values outside the returned band count as out of distribution (valence/models/served.py:48 ood_band). */
export function oodBand(name: string, lo: number, hi: number): [number, number] {
  if (NEVER_OOD.has(name)) return [-Infinity, Infinity];
  for (const [pattern, domain] of PHYSICAL_DOMAINS) {
    if (pattern.test(name)) return [domain[0], domain[1]];
  }
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) return [-Infinity, Infinity];
  const margin = OOD_MARGIN * (hi - lo);
  return [lo - margin, hi + margin];
}

// ---------- ServedPredictor ----------

function asText(v: string | Uint8Array): string {
  return typeof v === "string" ? v : new TextDecoder("utf-8").decode(v);
}

function fail(msg: string): never {
  throw new Error(`served predictor: ${msg}`);
}

/** valence/models/served.py:67 ServedPredictor. */
export class ServedPredictor implements Predictor {
  readonly meta: ServedMeta;
  readonly states: readonly string[];
  /** "m2_lgbm@" + the first 8 hex digits of sha256(m2.txt bytes) (valence/models/served.py:72). */
  readonly modelId: string;
  readonly m2: Booster;
  readonly m0: Booster;
  readonly t2: number;
  readonly t0: number;
  /** Feature sets come from the trained artifact, so code changes cannot silently misalign a model. */
  readonly cols2: readonly string[];
  readonly cols0: readonly string[];
  readonly idx2: Int32Array;
  readonly idx0: Int32Array;
  /** Clip bounds and OOD bands per feature, in FEATURE_NAMES order. */
  readonly lo: Float64Array;
  readonly hi: Float64Array;
  readonly oodLo: Float64Array;
  readonly oodHi: Float64Array;
  /** A feature neither model reads cannot make an input unusual. */
  readonly oodUsed: Uint8Array;
  /** Per M2 column: is it a Facial Surface feature (summed into the face push). */
  readonly faceCols: Uint8Array;
  private readonly faceNowIdx: number;

  constructor(a: ServedArtifacts) {
    const meta = a.meta;
    this.meta = meta;
    this.states = Object.freeze([...meta.states]);
    const m2Bytes = typeof a.m2 === "string" ? utf8Encode(a.m2) : a.m2;
    this.modelId = "m2_lgbm@" + sha256Hex(m2Bytes).slice(0, 8);
    this.m2 = parseLightGBMText(asText(a.m2));
    this.m0 = parseLightGBMText(asText(a.m0));
    // float(meta["temperatures"]["m2"]): a missing or null temperature raises instead of giving NaN / 0
    const temps: unknown = (meta as { temperatures?: unknown }).temperatures;
    if (!isPlainObject(temps)) fail("model_meta.json has no temperatures object");
    this.t2 = pyFloat(temps["m2"], "temperatures.m2");
    this.t0 = pyFloat(temps["m0"], "temperatures.m0");
    this.cols2 = Object.freeze([...meta.feature_sets.m2]);
    this.cols0 = Object.freeze([...meta.feature_sets.m0]);
    this.idx2 = Int32Array.from(this.cols2, featureIndex); // throws on an unknown name, like the KeyError
    this.idx0 = Int32Array.from(this.cols0, featureIndex);
    // Python fails later (contrib.reshape(len(states), -1), Booster.predict's width check); fail at load
    const K = this.states.length;
    if (this.m2.numTreePerIteration !== K || this.m0.numTreePerIteration !== K)
      fail(
        `models have ${this.m2.numTreePerIteration} / ${this.m0.numTreePerIteration} classes, meta has ${K} states`,
      );
    if (this.m2.numFeatures !== this.cols2.length || this.m0.numFeatures !== this.cols0.length)
      fail("model widths do not match meta feature_sets");

    const clip: unknown = a.clipBounds;
    // clip.get(...) needs a dict (a JSON list or null raises AttributeError in Python)
    if (!isPlainObject(clip)) fail("clip_bounds.json must be an object of feature -> [p1, p99]");
    this.lo = new Float64Array(N_FEATURES);
    this.hi = new Float64Array(N_FEATURES);
    this.oodLo = new Float64Array(N_FEATURES);
    this.oodHi = new Float64Array(N_FEATURES);
    this.oodUsed = new Uint8Array(N_FEATURES);
    const used = new Set([...this.cols2, ...this.cols0]);
    FEATURE_NAMES.forEach((n, i) => {
      // clip.get(n, [-inf, inf])[0], [1]. Python raises on anything but a list of >= 2 real numbers: a short
      // list (IndexError), a scalar (not subscriptable), a null entry (object array: TypeError at load or first
      // predict). Number(null) / undefined would instead clip silently to NaN.
      const b: unknown = Object.hasOwn(clip, n) ? clip[n] : undefined;
      if (b !== undefined && !(Array.isArray(b) && b.length >= 2))
        fail(`clip bounds of ${n} must be a list of two numbers`);
      const lo = b === undefined ? -Infinity : clipBound(b[0], `clip bounds of ${n}`);
      const hi = b === undefined ? Infinity : clipBound(b[1], `clip bounds of ${n}`);
      this.lo[i] = lo;
      this.hi[i] = hi;
      const [bl, bh] = oodBand(n, lo, hi);
      this.oodLo[i] = bl;
      this.oodHi[i] = bh;
      this.oodUsed[i] = used.has(n) ? 1 : 0;
    });
    this.faceCols = Uint8Array.from(this.cols2, (c) => (layerOf(c) === "Facial Surface" ? 1 : 0));
    this.faceNowIdx = featureIndex("face_now");
  }

  // ---------- core ----------

  /** The 123-vector as doubles; throws on a wrong length (numpy's broadcast error). */
  private vector(x: ArrayLike<number | null>): Float64Array {
    if (x.length !== N_FEATURES)
      throw new RangeError(`expected ${N_FEATURES} features, got ${x.length}`);
    return Float64Array.from({ length: N_FEATURES }, (_, i) => asFloat(x[i]));
  }

  /**
   * valence/models/served.py:92 `_clip`: clip to the training range (NaN stays NaN) and list, in feature
   * order, the features either model reads whose UNCLIPPED value is outside its OOD band.
   */
  clip(xIn: ArrayLike<number | null>): { xc: Float64Array; ood: string[] } {
    const x = this.vector(xIn);
    const xc = new Float64Array(N_FEATURES);
    const ood: string[] = [];
    for (let i = 0; i < N_FEATURES; i++) {
      const v = x[i] as number;
      const ok = !isnan(v);
      xc[i] = ok ? npClipArrayElem(v, this.lo[i] as number, this.hi[i] as number) : v;
      if (ok && this.oodUsed[i] && (v < (this.oodLo[i] as number) || v > (this.oodHi[i] as number)))
        ood.push(FEATURE_NAMES[i] as string);
    }
    return { xc, ood };
  }

  /** M2's input row: `xc[idx2]`. */
  m2Row(xc: ArrayLike<number>): Float64Array {
    return Float64Array.from(this.idx2, (i) => xc[i] as number);
  }

  /** M0's input row: `xc[idx0]`. */
  m0Row(xc: ArrayLike<number>): Float64Array {
    return Float64Array.from(this.idx0, (i) => xc[i] as number);
  }

  /** M2 raw scores (valence/models/served.py:100), from the clipped 123-vector. */
  m2Raw(xc: ArrayLike<number>): Float64Array {
    return predictRaw(this.m2, this.m2Row(xc));
  }

  /**
   * M2 SHAP rows (valence/models/served.py:101 pred_contrib), one per requested class (default: all), each
   * one entry per M2 column then the bias.
   */
  m2Contrib(xc: ArrayLike<number>, classes?: readonly number[]): Float64Array[] {
    return treeShap(this.m2, this.m2Row(xc), classes);
  }

  /** M0 raw scores (valence/models/served.py:114), from the clipped 123-vector. */
  m0Raw(xc: ArrayLike<number>): Float64Array {
    return predictRaw(this.m0, this.m0Row(xc));
  }

  /** predictVector with every intermediate (valence/models/served.py:104). */
  trace(xIn: ArrayLike<number | null>, todH: number | null = null): PredictionTrace {
    const x = this.vector(xIn);
    const { xc, ood } = this.clip(x);
    const row2 = this.m2Row(xc);
    const raw = predictRaw(this.m2, row2);
    let p = softmax(raw, this.t2);
    if (ood.length > OOD_SHRINK_AFTER) {
      const K = p.length;
      p = p.map((v) => (1 - OOD_SHRINK) * v + OOD_SHRINK / K);
    }
    const k = npArgmax(p);
    const label = this.states[k] as string;

    let m0: M0Reading | null = null;
    let contribK: Float64Array;
    if (!isnan(x[this.faceNowIdx] as number)) {
      const raw0 = this.m0Raw(xc);
      const p0 = softmax(raw0, this.t0);
      const k0 = npArgmax(p0);
      const [ck, c0] = treeShap(this.m2, row2, [k, k0]) as [Float64Array, Float64Array];
      contribK = ck;
      // contrib[k0, :-1][face_cols].sum(): numpy's pairwise sum over the selected Facial Surface columns
      const face: number[] = [];
      for (let j = 0; j < this.faceCols.length; j++)
        if (this.faceCols[j]) face.push(c0[j] as number);
      m0 = { raw: raw0, p: p0, k0, conf: p0[k0] as number, facePush: pairwiseSum(face) };
    } else {
      contribK = (treeShap(this.m2, row2, [k]) as [Float64Array])[0];
    }
    const faceLabel = m0 === null ? null : (this.states[m0.k0] as string);
    const faceConf = m0 === null ? null : m0.conf;
    const facePush = m0 === null ? 0.0 : m0.facePush;
    const rawOverride =
      faceLabel === null
        ? null
        : POSITIVE.has(faceLabel) &&
          !POSITIVE.has(label) &&
          (faceConf as number) >= M0_MIN_CONF &&
          (p[k] as number) >= M2_MIN_CONF &&
          facePush > 0;

    const names = this.cols2;
    const values: Record<string, number> = {};
    const contribs = new Map<string, number>();
    names.forEach((n, j) => {
      values[n] = x[this.idx2[j] as number] as number;
      const c = contribK[j] as number;
      if (c !== 0.0) contribs.set(n, c); // `if c != 0.0`: drops -0.0 too, keeps NaN
    });
    const absolute = new Map<string, number>(
      ABS_ROOM.map((n) => [n, x[featureIndex(n)] as number]),
    );
    const layers = buildLayers(contribs, values, { todH, absolute });
    const output: PredictionOutput = {
      label,
      confidence: p[k] as number,
      probabilities: Object.fromEntries(this.states.map((s, i) => [s, p[i] as number])),
      face_only_label: faceLabel,
      face_only_confidence: faceConf,
      layers,
      raw_override: rawOverride,
      values,
      ood_features: ood,
    };
    return { output, xc, ood, raw, contribK, p, k, m0 };
  }

  /** valence/models/served.py:104 predict_vector: one unclipped 123-vector -> PredictionOutput. */
  predictVector(x: ArrayLike<number | null>, todH: number | null = null): PredictionOutput {
    return this.trace(x, todH).output;
  }

  /** valence/models/served.py:134 predict: null for an empty person window; the room falls back to the person. */
  predict(person: readonly TickRecord[], room: readonly TickRecord[]): PredictionOutput | null {
    const last = person[person.length - 1];
    if (last === undefined) return null;
    const x = featuresFromWindow(person, room.length ? room : person);
    return this.predictVector(x, last.tod_h);
  }

  // ---------- counterfactual ("what if this one feature were different?") ----------

  /** valence/models/served.py:141 counterfactual (the dict the API returns; confidences rounded to 4 places). */
  counterfactual(
    person: readonly TickRecord[],
    room: readonly TickRecord[],
    feature: string,
    value: number,
  ): Counterfactual {
    const fi = featureIndex(feature); // KeyError: unknown feature
    const x = featuresFromWindow(person, room.length ? room : person);
    const todH = (person[person.length - 1] as TickRecord).tod_h;
    const before = this.predictVector(x, todH);
    const x2 = Float64Array.from(x);
    x2[fi] = value;
    const after = this.predictVector(x2, todH);
    const orig = x[fi] as number;
    return {
      feature,
      original_value: isnan(orig) ? null : orig,
      counterfactual_value: value,
      original: { label: before.label, confidence: pyRound(before.confidence, 4) },
      counterfactual: {
        label: after.label,
        confidence: pyRound(after.confidence, 4),
        probabilities: Object.fromEntries(
          Object.entries(after.probabilities).map(([s, v]) => [s, pyRound(v, 4)]),
        ),
      },
      label_changed: before.label !== after.label,
      simulated_fields: [feature],
    };
  }
}

/** The counterfactual API payload (valence/models/served.py:149). */
export interface Counterfactual {
  feature: string;
  original_value: number | null;
  counterfactual_value: number;
  original: { label: string; confidence: number };
  counterfactual: { label: string; confidence: number; probabilities: Record<string, number> };
  label_changed: boolean;
  simulated_fields: string[];
}

// ---------- Loading ----------

/** The part of `fetch`'s Response the loader uses (so tests can pass a stub). */
export interface FetchResponseLike {
  ok: boolean;
  status: number;
  arrayBuffer(): Promise<ArrayBuffer>;
}
export type FetchLike = (url: string) => Promise<FetchResponseLike>;

const PY_JSON_NAN = "\u0000\u0001py-json:NaN";
const PY_JSON_INF = "\u0000\u0001py-json:Infinity";
const PY_JSON_NINF = "\u0000\u0001py-json:-Infinity";
const PY_JSON_TOKENS: readonly (readonly [string, string])[] = [
  ["-Infinity", PY_JSON_NINF],
  ["Infinity", PY_JSON_INF],
  ["NaN", PY_JSON_NAN],
];

/**
 * Python's `json.loads` (served.py:82, :164) for the model JSON files. Two differences from JSON.parse:
 * - it accepts the bare tokens `NaN`, `Infinity` and `-Infinity` (what json.dumps writes for non-finite
 *   floats, e.g. a clip bound of a feature with infinite values): JSON.parse would reject the whole file;
 * - it rejects a leading BOM ("Unexpected UTF-8 BOM"), which the caller keeps in the text for that reason.
 * The tokens are swapped for sentinel strings outside string literals, then mapped back by the reviver; a
 * malformed neighbour ("NaN1", "-NaN", "nan") still fails, as in Python, and so does a token used as a key.
 */
export function parsePythonJson(text: string): unknown {
  let out = "";
  let i = 0;
  let inString = false;
  while (i < text.length) {
    const ch = text[i] as string;
    if (inString) {
      out += ch;
      if (ch === "\\") {
        out += text[i + 1] ?? "";
        i += 2;
        continue;
      }
      if (ch === '"') inString = false;
      i++;
      continue;
    }
    if (ch === '"') {
      inString = true;
      out += ch;
      i++;
      continue;
    }
    const tok = PY_JSON_TOKENS.find(([t]) => text.startsWith(t, i));
    if (tok !== undefined) {
      out += JSON.stringify(tok[1]);
      i += tok[0].length;
      continue;
    }
    out += ch;
    i++;
  }
  return JSON.parse(out, (key, v: unknown) => {
    if (key === PY_JSON_NAN || key === PY_JSON_INF || key === PY_JSON_NINF)
      throw new SyntaxError("Expecting property name enclosed in double quotes");
    if (v === PY_JSON_NAN) return NaN;
    if (v === PY_JSON_INF) return Infinity;
    if (v === PY_JSON_NINF) return -Infinity;
    return v;
  });
}

/** Path.read_text() for a JSON file: strict UTF-8 (UnicodeDecodeError on bad bytes), a BOM kept as text. */
function readJsonText(bytes: Uint8Array): string {
  return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
}

/** A static host's SPA fallback answers a missing file with index.html (status 200): treat it as missing. */
function looksLikeHtml(bytes: Uint8Array): boolean {
  for (const b of bytes) {
    if (
      b === 0x20 ||
      b === 0x09 ||
      b === 0x0a ||
      b === 0x0d ||
      b === 0xef ||
      b === 0xbb ||
      b === 0xbf
    )
      continue;
    return b === 0x3c; // "<"
  }
  return false;
}

/**
 * Load the served predictor for the browser (valence/models/served.py:160 load_served).
 *
 * Fetches `${baseUrl}{m2.txt, m0.txt, model_meta.json, clip_bounds.json}` (served from artifacts/ under
 * /models/core/ and /models/all/ by frontend/scripts/model-files.mjs). Resolves null, like load_served, when
 * model_meta.json, m2.txt or m0.txt is missing (404 / 410 / SPA fallback page) or, when `stateSet` is given,
 * the meta's state_set differs from it (Python always passes Settings.state_set; null skips the check). A
 * missing clip_bounds.json next to present models is an error, as in Python. Other HTTP errors throw.
 * The two JSON files are read like Path.read_text + json.loads (strict UTF-8, NaN / Infinity tokens accepted,
 * a BOM rejected: parsePythonJson).
 */
export async function loadServed(
  baseUrl = "/models/core/",
  fetchFn: FetchLike = (url) => globalThis.fetch(url),
  stateSet: string | null = null,
): Promise<ServedPredictor | null> {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  /** Bytes of a file, or null when it is missing. */
  const fetchBytes = async (name: string): Promise<Uint8Array | null> => {
    const res = await fetchFn(base + name);
    if (!res.ok) {
      if (res.status === 404 || res.status === 410) return null;
      fail(`GET ${base + name} failed with status ${res.status}`);
    }
    const bytes = new Uint8Array(await res.arrayBuffer());
    return looksLikeHtml(bytes) ? null : bytes;
  };

  const [metaBytes, m2, m0] = await Promise.all([
    fetchBytes("model_meta.json"),
    fetchBytes("m2.txt"),
    fetchBytes("m0.txt"),
  ]);
  if (metaBytes === null || m2 === null || m0 === null) return null;
  const meta = parsePythonJson(readJsonText(metaBytes)) as ServedMeta;
  // meta.get("state_set"): a JSON list or null raises AttributeError in Python
  if (!isPlainObject(meta)) fail("model_meta.json must be an object");
  if (stateSet !== null && meta.state_set !== stateSet) return null;
  const clipBytes = await fetchBytes("clip_bounds.json");
  if (clipBytes === null) fail(`models found under ${base} but no clip_bounds.json next to them`);
  const clipBounds = parsePythonJson(readJsonText(clipBytes)) as ClipBounds;
  return new ServedPredictor({ m2, m0, meta, clipBounds });
}
