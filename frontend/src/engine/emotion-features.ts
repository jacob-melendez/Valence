/**
 * Emotion-map window features (feature version `emotion-v1`): a person's rolling window -> the 31 face-cue
 * features the emotion models read next to the frozen 123 (valence/features/emotion.py; spec:
 * docs/EMOTION_FEATURES.md section 3).
 *
 * Inputs are the person's own tick rows, oldest first (the same window features.ts sees); only the eight
 * emotion-v1 cue fields of TickRecord and face_visible are read. A tick without a readable face carries NaN
 * cues and 0 visible seconds; window statistics skip NaN, and a window with no readable face at all gives
 * NaN for every feature. Nothing is imputed or clipped.
 *
 * Parity: the reductions are num.ts's numpy-faithful nanmean / nanstd (two-pass, population) / nanmax and
 * pairwise sums, so every feature is bit-identical to Python (no transcendental functions are involved).
 * One exception, by design (num.ts): when the last 12 brow_down values mix 0.0 and -0.0 and nothing larger,
 * brow_max_1m is +0.0 here while numpy's SIMD fmax may return -0.0. The models cannot tell the two apart
 * (lgbm.ts predictorRow reads both as 0.0).
 * Golden values: artifacts/emotion/golden.json -> windows (Cloud B) and
 * scripts/export_web_golden_emotion.py -> src/test/fixtures/engine/emotion.json.gz.
 */
import type { TickRecord } from "./contracts";
import { MAX_TICKS, TICK_S, WIN } from "./features";
import { allNaN, nanToNum, nanmax, nanmean, nanstd, pairwiseSum, tail } from "./num";

// ---------- Channels and names ----------

/** Feature version of this module (valence/features/emotion.py:17). */
export const EMOTION_FEATURE_VERSION = "emotion-v1";

/** Cue prefix -> TickRecord field, in feature order (valence/features/emotion.py:20 CUE_CHANNELS). */
export const CUE_CHANNELS = {
  brow: "brow_down",
  eye: "eye_open",
  frown: "mouth_frown",
  smile: "smile",
  yaw: "head_yaw",
  pitch: "head_pitch",
} as const satisfies Record<string, keyof TickRecord>;

/** Every TickRecord field the emotion features read (valence/features/emotion.py:28 EMOTION_FIELDS). */
export const EMOTION_FIELDS = [
  ...Object.values(CUE_CHANNELS),
  "head_away",
  "blink_count",
  "face_visible",
] as const satisfies readonly (keyof TickRecord)[];
export type EmotionField = (typeof EMOTION_FIELDS)[number];

/** Blink rates need at least this much visible face in the window (valence/features/emotion.py:29). */
export const BLINK_MIN_VISIBLE_S = 10.0;
/** valence/features/emotion.py BLINK_EPS_S: the 10 s test must not depend on summation order. */
export const BLINK_EPS_S = 1e-9;

/**
 * The 31 feature names in vector order (valence/features/emotion.py:31-33; also artifacts/emotion/meta.json
 * -> emotion_feature_names).
 */
export const EMOTION_FEATURE_NAMES: readonly string[] = [
  ...Object.keys(CUE_CHANNELS).flatMap((c) => [
    `${c}_now`,
    `${c}_mean_1m`,
    `${c}_mean_5m`,
    `${c}_std_5m`,
  ]),
  "eye_mean_15m",
  "brow_max_1m",
  "away_mean_1m",
  "away_mean_5m",
  "blink_rate_1m",
  "blink_rate_5m",
  "blink_rate_15m",
];
/** Number of emotion features (31). */
export const N_EMOTION_FEATURES = EMOTION_FEATURE_NAMES.length;
/** Feature name -> index in the 31-vector (valence/features/emotion.py:34 EMOTION_INDEX). */
export const EMOTION_INDEX: Readonly<Record<string, number>> = Object.fromEntries(
  EMOTION_FEATURE_NAMES.map((n, i) => [n, i]),
);

// ---------- Reductions ----------

/** `_nanstat(fn, x)` (valence/features/compute.py:84): NaN for an empty or all-NaN window, else fn(x). */
function nanstat(fn: (a: Float64Array) => number, x: Float64Array): number {
  if (x.length === 0 || allNaN(x)) return NaN;
  return fn(x);
}

/**
 * `_blink_rate(blinks, visible, eye, n)` (valence/features/emotion.py:43): blinks per minute of visible face over
 * the last n ticks whose cues were produced (`eye_open` finite; a visible face from an older producer is missing,
 * not "no blinks"): `60 * sum(blinks) / (5 * sum(visible))` over those ticks, NaN counted as 0 in both sums
 * (np.where + np.nan_to_num, then numpy's pairwise sum); NaN when that holds less than 10 s (minus 1e-9, so the
 * test does not depend on summation order) of visible face.
 */
function blinkRate(blinks: Float64Array, visible: Float64Array, eye: Float64Array, n: number): number {
  const cued = tail(eye, n).map((v) => (Number.isNaN(v) ? 0 : 1));
  const vis = nanToNum(tail(visible, n), { nan: 0.0 }).map((v, i) => (cued[i] ? v : 0.0));
  const seconds = pairwiseSum(vis) * TICK_S;
  if (seconds < BLINK_MIN_VISIBLE_S - BLINK_EPS_S) return NaN;
  const bl = nanToNum(tail(blinks, n), { nan: 0.0 }).map((v, i) => (cued[i] ? v : 0.0));
  return (60.0 * pairwiseSum(bl)) / seconds;
}

/**
 * One value as `np.asarray([v], dtype=float)` reads it: null (NaN as JSON writes it, `TickRecord.to_dict()`)
 * is NaN, as Python's None is; a Float64Array store alone would turn null into 0 and undefined into NaN.
 */
function asFloat(v: number | null | undefined): number {
  return v === null || v === undefined ? NaN : v;
}

/**
 * `np.asarray(d[key], dtype=float)[-MAX_TICKS:]` (valence/features/emotion.py:51): a float64 copy of the last
 * MAX_TICKS values, null read as NaN; throws like Python's KeyError when the channel is missing.
 */
function channel(person: EmotionChannels, key: EmotionField): Float64Array {
  const v = person[key];
  if (v === undefined) throw new Error(`missing channel ${JSON.stringify(key)}`);
  const start = Math.max(0, v.length - MAX_TICKS);
  const out = new Float64Array(v.length - start);
  for (let i = start; i < v.length; i++) out[i - start] = asFloat(v[i]);
  return out;
}

// ---------- Features ----------

/**
 * Per-tick channel arrays by TickRecord field, oldest first (Python: dict[str, np.ndarray]); null = NaN, so
 * channels decoded from JSON can be passed as they are.
 */
export type EmotionChannels = Readonly<Partial<Record<EmotionField, ArrayLike<number | null>>>>;

/**
 * `compute_emotion(person)` (valence/features/emotion.py:49): the person's channel arrays (oldest first; every
 * EMOTION_FIELDS key required, longer inputs cut to their last MAX_TICKS values) -> the 31 features in
 * EMOTION_FEATURE_NAMES order. An empty window gives 31 NaN, as in Python.
 */
export function emotionFromChannels(person: EmotionChannels): Float64Array {
  const p = {} as Record<EmotionField, Float64Array>;
  for (const k of EMOTION_FIELDS) p[k] = channel(person, k);
  const out = new Float64Array(N_EMOTION_FEATURES);
  let i = 0;
  for (const field of Object.values(CUE_CHANNELS)) {
    const x = p[field];
    out[i++] = x.length ? (x[x.length - 1] as number) : NaN;
    out[i++] = nanstat(nanmean, tail(x, WIN["1m"]));
    out[i++] = nanstat(nanmean, tail(x, WIN["5m"]));
    out[i++] = nanstat(nanstd, tail(x, WIN["5m"]));
  }
  out[i++] = nanstat(nanmean, tail(p.eye_open, WIN["15m"]));
  out[i++] = nanstat(nanmax, tail(p.brow_down, WIN["1m"]));
  out[i++] = nanstat(nanmean, tail(p.head_away, WIN["1m"]));
  out[i++] = nanstat(nanmean, tail(p.head_away, WIN["5m"]));
  for (const w of ["1m", "5m", "15m"] as const)
    out[i++] = blinkRate(p.blink_count, p.face_visible, p.eye_open, WIN[w]);
  return out;
}

/**
 * Live path: a RollingWindow snapshot of one person (TickRecords, oldest first) -> the 31 emotion features
 * (valence/features/emotion.py:67 emotion_from_window). Only the last MAX_TICKS rows are used. A field that
 * is null (a row taken from JSON without tickRecordFromDict) is NaN, as numpy reads None.
 */
export function computeEmotion(personRecords: readonly TickRecord[]): Float64Array {
  const n = Math.min(personRecords.length, MAX_TICKS);
  const start = personRecords.length - n;
  const ch = {} as Record<EmotionField, Float64Array>;
  for (const f of EMOTION_FIELDS) {
    const a = new Float64Array(n);
    for (let j = 0; j < n; j++)
      a[j] = asFloat((personRecords[start + j] as TickRecord)[f] as number | null);
    ch[f] = a;
  }
  return emotionFromChannels(ch);
}

/** Alias named after the Python function (valence/features/emotion.py:67). */
export const emotionFromWindow = computeEmotion;
