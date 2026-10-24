/**
 * Per-person posture signals from COCO-17 pose keypoints. Port of valence/perception/posture.py.
 *
 * Turns one person's keypoint stream into three somatic signals: a slump (head sinking toward the shoulders relative
 * to that person's own calibrated baseline), head in hands, and torso/arm motion. Everything is normalized by
 * shoulder width so it is independent of distance to the camera. The tracker keeps only a handful of scalars plus its
 * own copy of the previous sample's 17x2 keypoint coordinates; it never sees pixels.
 *
 * Keypoints are passed flat (the Python (17, 2) array, row-major): `kptsXy` = [x0, y0, x1, y1, ..., x16, y16] in
 * pixels, COCO-17 order, and `kptsConf` = the 17 confidences. Any ArrayLike works (number[], a model's
 * Float32Array, ...); both are copied into float64 and never retained.
 */
import { TIME_EPS } from "./debounce";

export const NOSE = 0;
export const L_EYE = 1;
export const R_EYE = 2;
export const L_EAR = 3;
export const R_EAR = 4;
export const L_SHOULDER = 5;
export const R_SHOULDER = 6;
export const L_WRIST = 9;
export const R_WRIST = 10;
export const HEAD_POINTS: readonly number[] = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR];
/** Shoulders, elbows, wrists, hips (COCO 5-12): motion ignores the head. */
export const BODY_POINTS: readonly number[] = [5, 6, 7, 8, 9, 10, 11, 12];
/** Left/right twins, which pose models swap. */
export const BODY_PAIRS: ReadonlyArray<readonly [number, number]> = [[5, 6], [7, 8], [9, 10], [11, 12]];
export const N_KPTS = 17;
export const MIN_BASELINE_SAMPLES = 10;

/** One sample's posture signals; `headDrop` is NaN until a nose and shoulders are seen. */
export interface PostureState {
  readonly slump: boolean;
  readonly headInHands: boolean;
  readonly motion: number;
  readonly headDrop: number;
  readonly calibrated: boolean;
}

/** Python's keyword arguments of PostureTracker, camelCased; each defaults as in Python. */
export interface PostureTrackerOptions {
  baselineS?: number;
  slumpRatio?: number;
  slumpHoldS?: number;
  hihHoldS?: number;
  hihRadius?: number;
  emaAlpha?: number;
  minConf?: number;
  earRatio?: number;
  noseOnlyRatio?: number;
  motionMinConf?: number;
  motionJump?: number;
  motionMinPoints?: number;
}

const x = (xy: Float64Array, k: number): number => xy[2 * k]!;
const y = (xy: Float64Array, k: number): number => xy[2 * k + 1]!;

/**
 * Euclidean length of (dx, dy) as np.linalg.norm(..., axis=1) computes it: sqrt(dx * dx + dy * dy), so NaN and inf
 * propagate like numpy (Math.hypot returns Infinity for (Infinity, NaN) and rounds differently).
 */
function norm2(dx: number, dy: number): number {
  return Math.sqrt(dx * dx + dy * dy);
}

/** numpy's pairwise summation of a float64 vector (8 accumulators, blocks of 128), as np.add.reduce runs it. */
function pairwiseSum(a: ArrayLike<number>, lo: number, n: number): number {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += a[lo + i]!;
    return res;
  }
  if (n <= 128) {
    const r = new Float64Array(8);
    for (let j = 0; j < 8; j++) r[j] = a[lo + j]!;
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = r[j]! + a[lo + i + j]!;
    let res = ((r[0]! + r[1]!) + (r[2]! + r[3]!)) + ((r[4]! + r[5]!) + (r[6]! + r[7]!));
    for (; i < n; i++) res += a[lo + i]!;
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
}

/** np.mean of a float64 vector: (0.0 + pairwise sum) / n; NaN when empty. */
function npMean(a: ArrayLike<number>): number {
  return (0.0 + pairwiseSum(a, 0, a.length)) / a.length;
}

/** np.median of finite samples: the mean of the middle one or two of the sorted values. */
function npMedian(values: readonly number[]): number {
  const s = Float64Array.from(values).sort();
  const mid = s.length >> 1;
  return npMean(s.length % 2 ? s.subarray(mid, mid + 1) : s.subarray(mid - 1, mid + 1));
}

/** NaN-skipping exponential moving average. */
function ema(prev: number, value: number, alpha: number): number {
  if (Number.isNaN(value)) return prev;
  if (Number.isNaN(prev)) return value;
  return alpha * value + (1.0 - alpha) * prev;
}

/** Mean y of the visible shoulders (one is enough for an above/below test). */
function shoulderLineY(xy: Float64Array): number {
  const ys = [y(xy, L_SHOULDER), y(xy, R_SHOULDER)].filter((v) => Number.isFinite(v));
  return ys.length ? npMean(ys) : NaN;
}

/** Track how long `condition` has been continuously true; true once held `holdS`. */
function held(since: number | null, condition: boolean, t: number, holdS: number): [number | null, boolean] {
  if (!condition) return [null, false];
  const start = since === null ? t : since;
  return [start, t - start >= holdS - TIME_EPS];
}

/**
 * Slump / head-in-hands / motion for one display id.
 *
 * `headDrop` is the nose height above the shoulder midpoint in shoulder widths (EMA); `earDrop` is the same for the
 * mean of the visible ears. Baselines are the medians of the raw valid samples seen during the first `baselineS`
 * after the first update (at least 10 samples; calibration continues past the window until the nose has them, and
 * restarts if its median is not positive).
 *
 * A slump needs the whole head to sink, not just tip forward: looking down at a keyboard pitches the head, which
 * swings the nose down but leaves the ears (near the pitch axis) in place. So slump = nose < `slumpRatio` x baseline
 * AND ears < `earRatio` x baseline, held `slumpHoldS`; when the ears were never calibrated or aren't visible, the
 * nose alone must fall below `noseOnlyRatio` x baseline.
 *
 * `motion` is the mean displacement of the torso and arm keypoints (COCO 5-12) in shoulder widths per second (EMA).
 * A point counts only if it has conf >= `motionMinConf` in both samples and moved at most `motionJump` shoulder
 * widths since the last one (a bigger per-sample jump is a keypoint switching identity). A left/right twin pair seen
 * in both samples is matched whichever way is shorter, so a one-sample L/R label swap is not motion. With fewer than
 * `motionMinPoints` survivors, unless both shoulders survive (a measurable torso), the last value is held; motion is
 * NaN only until the first measurable pair.
 */
export class PostureTracker {
  readonly baselineS: number;
  readonly slumpRatio: number;
  readonly earRatio: number;
  readonly noseOnlyRatio: number;
  readonly slumpHoldS: number;
  readonly hihHoldS: number;
  readonly hihRadius: number;
  readonly emaAlpha: number;
  readonly minConf: number;
  /** Never looser than `minConf`. */
  readonly motionMinConf: number;
  readonly motionJump: number;
  readonly motionMinPoints: number;
  private t0: number | null = null;
  private samples: number[] = [];
  private earSamples: number[] = [];
  private _baseline: number | null = null;
  private _earBaseline: number | null = null;
  private shoulderW = NaN;
  private headDrop = NaN;
  private earDrop = NaN;
  private motion = NaN;
  private slumpSince: number | null = null;
  private hihSince: number | null = null;
  private prevXy: Float64Array | null = null;
  private prevT: number | null = null;

  constructor({
    baselineS = 20.0,
    slumpRatio = 0.7,
    slumpHoldS = 3.0,
    hihHoldS = 2.0,
    hihRadius = 0.5,
    emaAlpha = 0.3,
    minConf = 0.5,
    earRatio = 0.93,
    noseOnlyRatio = 0.6,
    motionMinConf = 0.6,
    motionJump = 1.0,
    motionMinPoints = 3,
  }: PostureTrackerOptions = {}) {
    if (!(0.0 < emaAlpha && emaAlpha <= 1.0)) throw new Error("emaAlpha must be in (0, 1]");
    this.baselineS = baselineS;
    this.slumpRatio = slumpRatio;
    this.earRatio = earRatio;
    this.noseOnlyRatio = noseOnlyRatio;
    this.slumpHoldS = slumpHoldS;
    this.hihHoldS = hihHoldS;
    this.hihRadius = hihRadius;
    this.emaAlpha = emaAlpha;
    this.minConf = minConf;
    this.motionMinConf = motionMinConf > minConf ? motionMinConf : minConf; // Python max(minConf, motionMinConf)
    this.motionJump = motionJump;
    this.motionMinPoints = motionMinPoints;
  }

  get baseline(): number | null {
    return this._baseline;
  }

  get earBaseline(): number | null {
    return this._earBaseline;
  }

  get shoulderWidth(): number {
    return this.shoulderW;
  }

  /**
   * Feed one pose sample at `t`: `kptsXy` = 34 pixel coordinates, row-major (x0, y0, ..., x16, y16); `kptsConf` = 17
   * per-keypoint confidences. Throws if the lengths are not 34 and 17.
   */
  update(kptsXy: ArrayLike<number>, kptsConf: ArrayLike<number>, t: number): PostureState {
    const [xy, body] = this.mask(kptsXy, kptsConf);
    const t0 = (this.t0 = this.t0 === null ? t : this.t0);

    this.updateShoulderWidth(xy);
    const rawDrop = this.rawHeadDrop(xy);
    const rawEar = this.rawEarDrop(xy);
    this.headDrop = ema(this.headDrop, rawDrop, this.emaAlpha);
    this.earDrop = Number.isNaN(rawEar) ? rawEar : ema(this.earDrop, rawEar, this.emaAlpha);
    this.calibrate(rawDrop, rawEar, t, t0);

    const [slumpSince, slump] = held(this.slumpSince, this.belowBaseline(), t, this.slumpHoldS);
    this.slumpSince = slumpSince;
    const [hihSince, hih] = held(this.hihSince, this.handsOnHead(xy), t, this.hihHoldS);
    this.hihSince = hihSince;
    this.motion = ema(this.motion, this.rawMotion(body, t), this.emaAlpha);
    this.prevXy = body;
    this.prevT = t;

    return {
      slump,
      headInHands: hih,
      motion: this.motion,
      headDrop: this.headDrop,
      calibrated: this._baseline !== null,
    };
  }

  /** (all keypoints at `minConf`, torso/arm keypoints at `motionMinConf`), both 34 flat; NaN = not valid. */
  private mask(kptsXy: ArrayLike<number>, kptsConf: ArrayLike<number>): [Float64Array, Float64Array] {
    if (kptsXy.length !== 2 * N_KPTS || kptsConf.length !== N_KPTS) {
      throw new Error(`expected 34 (17x2) keypoint coordinates and 17 conf, got ${kptsXy.length}, ${kptsConf.length}`);
    }
    const xy = new Float64Array(2 * N_KPTS);
    const body = new Float64Array(2 * N_KPTS).fill(NaN);
    for (let k = 0; k < N_KPTS; k++) {
      const conf = Number(kptsConf[k]);
      const valid = conf >= this.minConf;
      xy[2 * k] = valid ? Number(kptsXy[2 * k]) : NaN;
      xy[2 * k + 1] = valid ? Number(kptsXy[2 * k + 1]) : NaN;
      if (BODY_POINTS.includes(k) && conf >= this.motionMinConf) {
        body[2 * k] = xy[2 * k]!;
        body[2 * k + 1] = xy[2 * k + 1]!;
      }
    }
    return [xy, body];
  }

  private updateShoulderWidth(xy: Float64Array): void {
    // np.linalg.norm of a single vector goes through BLAS ddot, which may fuse the multiply-add (1 ulp apart).
    const width = norm2(x(xy, L_SHOULDER) - x(xy, R_SHOULDER), y(xy, L_SHOULDER) - y(xy, R_SHOULDER));
    if (Number.isFinite(width) && width > 0.0) this.shoulderW = ema(this.shoulderW, width, this.emaAlpha);
  }

  private rawHeadDrop(xy: Float64Array): number {
    const midY = (y(xy, L_SHOULDER) + y(xy, R_SHOULDER)) / 2.0;
    if (Number.isNaN(this.shoulderW)) return NaN;
    return (midY - y(xy, NOSE)) / this.shoulderW;
  }

  private rawEarDrop(xy: Float64Array): number {
    const ears = [L_EAR, R_EAR].filter((k) => Number.isFinite(x(xy, k)) && Number.isFinite(y(xy, k)));
    const earYs = ears.map((k) => y(xy, k));
    const midY = (y(xy, L_SHOULDER) + y(xy, R_SHOULDER)) / 2.0;
    if (earYs.length === 0 || Number.isNaN(this.shoulderW) || Number.isNaN(midY)) return NaN;
    return (midY - npMean(earYs)) / this.shoulderW;
  }

  private calibrate(rawDrop: number, rawEar: number, t: number, t0: number): void {
    if (this._baseline !== null) return;
    const inWindow = t - t0 < this.baselineS - TIME_EPS;
    if (inWindow || this.samples.length < MIN_BASELINE_SAMPLES) {
      if (Number.isFinite(rawDrop)) this.samples.push(rawDrop);
      if (Number.isFinite(rawEar)) this.earSamples.push(rawEar);
    }
    if (inWindow || this.samples.length < MIN_BASELINE_SAMPLES) return;
    const baseline = npMedian(this.samples);
    const ear = this.earSamples.length >= MIN_BASELINE_SAMPLES ? npMedian(this.earSamples) : null;
    this.samples = [];
    this.earSamples = [];
    if (baseline > 0.0) {
      this._baseline = baseline;
      this._earBaseline = ear !== null && ear > 0.0 ? ear : null;
    } else {
      this.t0 = t;
    }
  }

  private belowBaseline(): boolean {
    if (this._baseline === null || Number.isNaN(this.headDrop)) return false;
    if (this._earBaseline !== null && !Number.isNaN(this.earDrop)) {
      return this.headDrop < this.slumpRatio * this._baseline && this.earDrop < this.earRatio * this._earBaseline;
    }
    return this.headDrop < this.noseOnlyRatio * this._baseline;
  }

  private handsOnHead(xy: Float64Array): boolean {
    if (Number.isNaN(this.shoulderW)) return false;
    const finite = (k: number): boolean => Number.isFinite(x(xy, k)) && Number.isFinite(y(xy, k));
    const wrists = [L_WRIST, R_WRIST];
    const head = HEAD_POINTS.filter(finite);
    const shoulderY = shoulderLineY(xy);
    if (head.length === 0 || !wrists.every(finite) || Number.isNaN(shoulderY)) return false;
    const radius = this.hihRadius * this.shoulderW;
    const dist = (w: number): number => Math.min(...head.map((h) => norm2(x(xy, w) - x(xy, h), y(xy, w) - y(xy, h))));
    const near = wrists.every((w) => dist(w) <= radius);
    const raised = wrists.every((w) => y(xy, w) < shoulderY);
    return near && raised;
  }

  /** Torso/arm speed in shoulder widths/s for this pair; NaN (so the EMA holds) when unmeasurable. */
  private rawMotion(body: Float64Array, t: number): number {
    if (this.prevXy === null || this.prevT === null || Number.isNaN(this.shoulderW)) return NaN;
    const dt = t - this.prevT;
    if (dt <= 0.0) return NaN;
    const prev = this.prevXy;
    const sw = this.shoulderW;
    const disp = new Float64Array(N_KPTS); // NaN unless valid in both
    for (let k = 0; k < N_KPTS; k++) disp[k] = norm2(x(body, k) - x(prev, k), y(body, k) - y(prev, k)) / sw;
    for (const [left, right] of BODY_PAIRS) {
      const swapLeft = norm2(x(body, left) - x(prev, right), y(body, left) - y(prev, right)) / sw;
      const swapRight = norm2(x(body, right) - x(prev, left), y(body, right) - y(prev, left)) / sw;
      if (swapLeft + swapRight < disp[left]! + disp[right]!) { // false when any is NaN: keep the labels
        disp[left] = swapLeft;
        disp[right] = swapRight;
      }
    }
    const ok = Array.from(disp, (d) => Number.isFinite(d) && d <= this.motionJump);
    const torso = ok[L_SHOULDER]! && ok[R_SHOULDER]!;
    const kept = disp.filter((_, k) => ok[k]!);
    if (kept.length < this.motionMinPoints && !torso) return NaN;
    return npMean(kept) / dt;
  }
}
