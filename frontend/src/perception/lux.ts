/**
 * Ambient Physics layer: room light (lux) from the webcam, no light sensor needed. Port of valence/perception/lux.py.
 *
 * Auto-exposure normalises brightness, so background luminance alone is a weak lux proxy; in a dim room the camera
 * raises gain and sensor noise rises with it. Both are measured on the background only (person boxes masked) from a
 * single frame and mapped to lux with a log-linear model. Only scalars leave this module: the decimated luminance
 * copy is zero-filled before backgroundStats returns, and no frame, thumbnail or frame difference is ever kept.
 *
 * Not ported: `LuxModel.fit` (the least-squares calibration fit runs offline in scripts/calibrate_lux.py;
 * `LuxModel.fromDict` can load its result, though the browser runs on the uncalibrated defaults).
 */

export const CLIP_LO = 5;
export const CLIP_HI = 250;
export const MAX_WIDTH = 320;
export const MAX_LUX = 200_000.0;
export const DEFAULT_A = 3.59;
export const DEFAULT_B = 0.01463;
/** Immerkaer (1996) noise kernel, row-major 3 x 3 (symmetric, so correlation == convolution). */
const IMMERKAER = [1, -2, 1, -2, 4, -2, 1, -2, 1] as const;
const SIGMA_SCALE = Math.sqrt(Math.PI / 2.0) / 6.0;

/** math.floor / math.ceil: like Python, a non-finite value cannot become an integer. */
function pyFloor(x: number): number {
  if (!Number.isFinite(x)) throw new RangeError(`cannot convert float ${x} to integer`);
  return Math.floor(x) + 0;
}

function pyCeil(x: number): number {
  if (!Number.isFinite(x)) throw new RangeError(`cannot convert float ${x} to integer`);
  return Math.ceil(x) + 0;
}

/** np.clip on a scalar: NaN stays NaN, signed zeros pass through. */
function npClip(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Single-frame background statistics: mean luminance, noise sigma, coverage, clipping. */
export class BgStats {
  constructor(
    readonly yBg: number,
    readonly sigma: number,
    readonly bgFrac: number,
    readonly clippedFrac: number,
  ) {}

  get valid(): boolean {
    return this.bgFrac > 0.0 && Number.isFinite(this.yBg);
  }
}

/** 1 inside any finite normalized (x1, y1, x2, y2) box (corners in either order), row-major h x w. */
function personMask(h: number, w: number, boxes: Iterable<ArrayLike<number>>): Uint8Array {
  const mask = new Uint8Array(h * w);
  for (const box of boxes) {
    if (box.length !== 4) throw new TypeError(`a box needs 4 values (x1, y1, x2, y2), got ${box.length}`);
    const x1 = box[0]!, y1 = box[1]!, x2 = box[2]!, y2 = box[3]!;
    if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
    const c0 = npClip(pyFloor((x2 < x1 ? x2 : x1) * w), 0, w), c1 = npClip(pyCeil((x2 > x1 ? x2 : x1) * w), 0, w);
    const r0 = npClip(pyFloor((y2 < y1 ? y2 : y1) * h), 0, h), r1 = npClip(pyCeil((y2 > y1 ? y2 : y1) * h), 0, h);
    for (let r = r0; r < r1; r++) mask.fill(1, r * w + c0, r * w + Math.max(c0, c1));
  }
  return mask;
}

/**
 * Pixels whose whole 3x3 neighbourhood is outside `excluded`, minus the outer ring: cv2.dilate with a 3x3 ones kernel
 * (its default border never contributes), then `== 0`, then the ring zeroed. Every pixel left has its 3x3 support
 * inside the image.
 */
function interiorKeep(excluded: Uint8Array, h: number, w: number): Uint8Array {
  const keep = new Uint8Array(h * w);
  for (let r = 1; r < h - 1; r++) {
    for (let c = 1; c < w - 1; c++) {
      let hit = 0;
      for (let dr = -1; dr <= 1; dr++) {
        const row = (r + dr) * w + c;
        hit |= excluded[row - 1]! | excluded[row]! | excluded[row + 1]!;
      }
      keep[r * w + c] = hit ? 0 : 1;
    }
  }
  return keep;
}

/**
 * sigma = sqrt(pi/2) / 6 * mean|I * N| over background pixels whose 3x3 support is entirely background (Immerkaer
 * 1996), so person/background edges do not count as noise. Hard-clipped pixels (0 or 255) carry no noise and would
 * dilute the estimate, so their 3x3 neighbourhoods are skipped too, unless that leaves nothing to measure.
 *
 * cv2.filter2D(img, CV_32F, N, BORDER_REPLICATE) is evaluated only where it is summed: kept pixels are interior, so
 * the border mode never reaches them, and each response is an integer of magnitude <= 4080, exact in float32 (the
 * Math.fround is the identity, kept to mark the float32 output). The abs-sum is float64, as in numpy.
 */
function immerkaerSigma(img: Uint8Array, h: number, w: number, person: Uint8Array): number {
  if (Math.min(h, w) < 3) return NaN;
  const excluded = new Uint8Array(h * w);
  for (let i = 0; i < excluded.length; i++) excluded[i] = person[i]! | (img[i] === 0 || img[i] === 255 ? 1 : 0);
  let keep = interiorKeep(excluded, h, w);
  if (!keep.includes(1)) keep = interiorKeep(person, h, w);
  const offsets = [-w - 1, -w, -w + 1, -1, 0, 1, w - 1, w, w + 1];
  let n = 0;
  let total = 0;
  for (let i = 0; i < keep.length; i++) {
    if (!keep[i]) continue;
    let response = 0;
    for (let k = 0; k < 9; k++) response += IMMERKAER[k]! * img[i + offsets[k]!]!;
    total += Math.abs(Math.fround(response));
    n += 1;
  }
  if (n === 0) return NaN;
  return (SIGMA_SCALE * total) / n;
}

/**
 * Background luminance and sensor-noise sigma from ONE uint8 luminance frame (`gray`, row-major width x height).
 *
 * The frame is decimated (strided, not averaged: `gray[::stride, ::stride]`) to <= `maxWidth` columns for speed:
 * decimation keeps per-pixel noise statistics intact, whereas area averaging would shrink sigma by the stride.
 * `personBoxesNorm` are (x1, y1, x2, y2) in [0, 1].
 */
export function backgroundStats(
  gray: Uint8Array,
  width: number,
  height: number,
  personBoxesNorm: Iterable<ArrayLike<number>> = [],
  maxWidth = MAX_WIDTH,
): BgStats {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 0 || height < 0
    || gray.length !== width * height) {
    throw new RangeError("backgroundStats expects a uint8 width x height luminance image");
  }
  if (maxWidth === 0) throw new RangeError("division by zero");
  const stride = Math.max(1, pyCeil(width / maxWidth));
  const h = height > 0 ? Math.floor((height - 1) / stride) + 1 : 0;
  const w = width > 0 ? Math.floor((width - 1) / stride) + 1 : 0;
  let img = gray;
  if (stride > 1) {
    img = new Uint8Array(h * w);
    for (let r = 0, i = 0; r < h; r++) for (let c = 0; c < w; c++, i++) img[i] = gray[r * stride * width + c * stride]!;
  }
  try {
    const person = personMask(h, w, personBoxesNorm);
    let nBg = 0;
    let sum = 0;
    let clipped = 0;
    for (let i = 0; i < person.length; i++) {
      if (person[i]) continue;
      const v = img[i]!;
      nBg += 1;
      sum += v;
      if (v <= CLIP_LO || v >= CLIP_HI) clipped += 1;
    }
    if (nBg === 0) return new BgStats(NaN, NaN, 0.0, 0.0);
    return new BgStats(sum / nBg, immerkaerSigma(img, h, w, person), nBg / person.length, clipped / nBg);
  } finally {
    if (img !== gray) img.fill(0);
  }
}

/**
 * lux = exp(a + b * yBg + c * sigma). The default (uncalibrated) model maps yBg 30 -> ~55, 128 -> ~240, 220 -> ~900
 * lux, ignoring noise.
 */
export class LuxModel {
  constructor(
    readonly a = DEFAULT_A,
    readonly b = DEFAULT_B,
    readonly c = 0.0,
    readonly calibrated = false,
  ) {}

  predict(yBg: number, sigma: number): number {
    // Python's `c * sigma if c else 0.0`: NaN is truthy there, unlike in JS.
    const z = this.a + this.b * yBg + (this.c !== 0 ? this.c * sigma : 0.0);
    if (!Number.isFinite(z)) return NaN;
    return Math.min(Math.exp(Math.min(z, 50.0)), MAX_LUX);
  }

  toDict(): { a: number; b: number; c: number; calibrated: boolean } {
    return { a: this.a, b: this.b, c: this.c, calibrated: this.calibrated };
  }

  /**
   * From a calibration mapping (calibration.json "lux"): missing keys take the defaults, values go through Python's
   * float() / bool() (numeric strings such as " 1e-2 " or "nan" parse, null or a list throws). An `undefined` value
   * counts as missing.
   */
  static fromDict(d: Readonly<Record<string, unknown>> | ReadonlyMap<string, unknown>): LuxModel {
    const get = (k: string, dflt: unknown): unknown => {
      const v = d instanceof Map ? d.get(k) : (d as Readonly<Record<string, unknown>>)[k];
      return v === undefined ? dflt : v;
    };
    return new LuxModel(
      pyFloat(get("a", DEFAULT_A)),
      pyFloat(get("b", DEFAULT_B)),
      pyFloat(get("c", 0.0)),
      pyBool(get("calibrated", false)),
    );
  }
}

const PY_SPACE = "[ \\t\\n\\v\\f\\r\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]*";
const PY_DIGITS = "\\d(?:_?\\d)*";
const PY_FLOAT = new RegExp(
  `^${PY_SPACE}([+-]?(?:(?:${PY_DIGITS})?\\.${PY_DIGITS}|${PY_DIGITS}\\.?)(?:[eE][+-]?${PY_DIGITS})?)${PY_SPACE}$`,
);
const PY_SPECIAL = new RegExp(`^${PY_SPACE}([+-]?)(inf|infinity|nan)${PY_SPACE}$`, "i");

const ND = /\p{Nd}/u;

/** Python float() first maps every Unicode decimal digit to ASCII; Nd digits come in contiguous blocks of 0-9 runs. */
function asciiDigits(s: string): string {
  return s.replace(/\p{Nd}/gu, (d) => {
    const cp = d.codePointAt(0)!;
    let start = cp;
    while (ND.test(String.fromCodePoint(start - 1))) start -= 1;
    return String((cp - start) % 10);
  });
}

/** Python float() of a JSON-ish value. */
function pyFloat(v: unknown): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1.0 : 0.0;
  if (typeof v === "bigint") {
    const f = Number(v);
    if (!Number.isFinite(f)) throw new RangeError("int too large to convert to float");
    return f;
  }
  if (typeof v === "string") {
    const text = asciiDigits(v);
    const m = PY_FLOAT.exec(text);
    if (m) return Number(m[1]!.replaceAll("_", ""));
    const s = PY_SPECIAL.exec(text);
    if (s) return s[2]!.toLowerCase() === "nan" ? NaN : s[1] === "-" ? -Infinity : Infinity;
    throw new TypeError(`could not convert string to float: '${v}'`);
  }
  throw new TypeError(`float() argument must be a string or a real number, not '${v === null ? "None" : typeof v}'`);
}

/** Python bool() of a JSON-ish value. */
function pyBool(v: unknown): boolean {
  if (v === null || v === undefined) return false;
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "bigint") return v !== 0n;
  if (typeof v === "string" || Array.isArray(v)) return v.length > 0;
  if (v instanceof Map || v instanceof Set) return v.size > 0;
  if (typeof v === "object") return Object.keys(v).length > 0;
  return true;
}

/** np.percentile(sorted, 100 q), default 'linear' method: numpy's virtual index, neighbours and _lerp. */
function percentileSorted(sorted: Float64Array, q: number): number {
  const n = sorted.length;
  const vi = (n - 1) * q;
  let lo = Math.floor(vi);
  let hi = lo + 1;
  if (vi >= n - 1) lo = hi = -1; // past the last index: numpy takes index -1 (and gamma = vi + 1)
  const gamma = vi - lo;
  const a = sorted.at(lo)!, b = sorted.at(hi)!;
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

/**
 * Rolling `windowS` median of per-frame lux, plus a [0, 1] quality score: bgFrac * (1 - clippedFrac) * stability,
 * stability = 1 / (1 + IQR / median), halved while the model is uncalibrated.
 */
export class LuxEstimator {
  /** (t, lux), oldest first. */
  private readonly history: [number, number][] = [];

  constructor(
    public model: LuxModel = new LuxModel(),
    public windowS = 10.0,
  ) {}

  reset(): void {
    this.history.length = 0;
  }

  /**
   * Returns (lux median over the window, lux quality); (NaN, 0.0) before any valid frame. A non-finite `t` is ignored
   * (the current window is reported unchanged), so it can never wedge the eviction of old readings.
   */
  update(stats: BgStats, t: number): [number, number] {
    if (!Number.isFinite(t)) return this.summary(stats);
    if (stats.valid) {
      const lux = this.model.predict(stats.yBg, stats.sigma);
      if (Number.isFinite(lux)) this.history.push([t, lux]);
    }
    while (this.history.length && t - this.history[0]![0] > this.windowS) this.history.shift();
    return this.summary(stats);
  }

  private summary(stats: BgStats): [number, number] {
    if (!this.history.length) return [NaN, 0.0];
    const values = Float64Array.from(this.history, ([, v]) => v).sort();
    const q25 = percentileSorted(values, 25.0 / 100);
    const median = percentileSorted(values, 50.0 / 100);
    const q75 = percentileSorted(values, 75.0 / 100);
    const stability = median > 0 ? 1.0 / (1.0 + (q75 - q25) / median) : 0.0;
    let quality = stats.bgFrac * (1.0 - stats.clippedFrac) * stability;
    if (!this.model.calibrated) quality *= 0.5;
    return [median, npClip(quality, 0.0, 1.0)];
  }
}
