/**
 * numpy- and CPython-faithful numerics for the browser engine.
 *
 * The Python engine is the reference: every feature, probability, explanation string and audit hash
 * the TypeScript port produces must be bit-identical to what valence/ produces for the same inputs.
 * Plain JS arithmetic is not enough for that, because numpy sums float64 arrays pairwise (not left to
 * right), the nan* functions sum zero-filled arrays, and CPython rounds and prints floats from their
 * exact binary value with ties to even (JS toFixed/toPrecision round ties away from zero).
 *
 * Everything here is pure and synchronous; inputs are 1-D float64 arrays (any ArrayLike<number>).
 * Golden values: scripts/export_web_golden_num.py -> src/test/fixtures/engine/num.json.gz.
 *
 * Not bit-exact by design: the sign of a zero returned by nanmax/nanmin/median/percentileLinear when the
 * input mixes 0.0 and -0.0 (numpy's answer there depends on its SIMD lanes and partition order).
 * Out of scope: exp/log/pow/sin/cos. V8's Math functions, glibc (CPython `math`) and numpy's AVX-512
 * builds (np.exp, np.power, np.log10) disagree in the last bit for a few percent of inputs, so values that
 * pass through them need a ulp tolerance (ulpDistance) rather than bit equality.
 */

// ---------- NaN helpers ----------

/** `math.isnan(x)` / `np.isnan(x)` for one value. */
export function isnan(x: number): boolean {
  return x !== x;
}

/** `np.isnan(a).any()`. */
export function anyNaN(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i++) if (isnan(a[i] as number)) return true;
  return false;
}

/** `np.all(np.isnan(a))`; true for an empty array, as in numpy. */
export function allNaN(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i++) if (!isnan(a[i] as number)) return false;
  return true;
}

/** `int((~np.isnan(a)).sum())`. */
export function countNotNaN(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (!isnan(a[i] as number)) n++;
  return n;
}

/**
 * `np.nan_to_num(a, nan=nan)`: NaN -> `nan`, +inf -> the largest finite double, -inf -> its negative
 * (numpy's defaults for posinf/neginf, used by valence/features/compute.py:144 and :163).
 */
export function nanToNum(
  a: ArrayLike<number>,
  opts: { nan?: number; posinf?: number; neginf?: number } = {},
): Float64Array {
  const nan = opts.nan ?? 0.0;
  const posinf = opts.posinf ?? Number.MAX_VALUE;
  const neginf = opts.neginf ?? -Number.MAX_VALUE;
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    out[i] = isnan(v) ? nan : v === Infinity ? posinf : v === -Infinity ? neginf : v;
  }
  return out;
}

/**
 * Python slice `a[-n:]` as a Float64Array copy. Note Python's quirk: `a[-0:]` is the whole array.
 * Used by valence/features/compute.py for the 1m/5m/15m windows.
 */
export function tail(a: ArrayLike<number>, n: number): Float64Array {
  const len = a.length;
  const start = n <= 0 ? 0 : Math.max(0, len - n);
  const out = new Float64Array(len - start);
  for (let i = start; i < len; i++) out[i - start] = a[i] as number;
  return out;
}

// ---------- Summation ----------

/** numpy's PW_BLOCKSIZE (numpy/_core/src/umath/loops_utils.h.src). */
const PW_BLOCKSIZE = 128;

/**
 * numpy's `pairwise_sum` kernel over `a[lo, lo + n)`: plain loop below 8 elements, 8 running partial
 * sums up to 128, recursive halving (at a multiple of 8) above.
 */
function pairwiseKernel(a: ArrayLike<number>, lo: number, n: number): number {
  if (n < 8) {
    let res = 0.0; // starts at +0.0, so a lone -0.0 sums to +0.0 exactly like numpy
    for (let i = 0; i < n; i++) res += a[lo + i] as number;
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    let r0 = a[lo] as number, r1 = a[lo + 1] as number, r2 = a[lo + 2] as number, r3 = a[lo + 3] as number;
    let r4 = a[lo + 4] as number, r5 = a[lo + 5] as number, r6 = a[lo + 6] as number, r7 = a[lo + 7] as number;
    const stop = n - (n % 8);
    let i = 8;
    for (; i < stop; i += 8) {
      const b = lo + i;
      r0 += a[b] as number; r1 += a[b + 1] as number; r2 += a[b + 2] as number; r3 += a[b + 3] as number;
      r4 += a[b + 4] as number; r5 += a[b + 5] as number; r6 += a[b + 6] as number; r7 += a[b + 7] as number;
    }
    let res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++) res += a[lo + i] as number;
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseKernel(a, lo, n2) + pairwiseKernel(a, lo + n2, n - n2);
}

/**
 * `np.add.reduce(a)` / `np.sum(a)` for a contiguous 1-D float64 array, bit-exact.
 *
 * numpy (2.5, verified empirically by the exporter) seeds the output with the identity +0.0 and adds the
 * pairwise sum of ALL elements to it; the first element does not seed the accumulator. Hence the result
 * is never -0.0, and an empty array sums to 0.0.
 */
export function pairwiseSum(a: ArrayLike<number>): number {
  return 0.0 + pairwiseKernel(a, 0, a.length);
}

/** `float(np.sum(a))`; alias of pairwiseSum. */
export const npSum = pairwiseSum;

/** `float(np.mean(a))`: pairwise sum / n; NaN for an empty array (numpy warns and returns nan). */
export function npMean(a: ArrayLike<number>): number {
  return pairwiseSum(a) / a.length;
}

/**
 * CPython 3.12 builtin `sum()` over floats: Neumaier-compensated, left to right, starting from int 0
 * (so the first float is added to 0 exactly, turning -0.0 into 0.0). Python/bltinmodule.c builtin_sum_impl.
 * Counts (ints) also sum exactly with this, as long as they stay below 2**53.
 */
export function pySum(a: ArrayLike<number>): number {
  if (a.length === 0) return 0;
  let f = 0 + (a[0] as number);
  let c = 0.0;
  for (let i = 1; i < a.length; i++) {
    const x = a[i] as number;
    const t = f + x;
    if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x;
    else c += (x - t) + f;
    f = t;
  }
  // "Avoid losing the sign on a negative result, and don't let adding the compensation convert an
  // infinite or overflowed sum to a NaN."
  if (c !== 0 && Number.isFinite(c)) f += c;
  return f;
}

// ---------- nan* reductions ----------

/** numpy `_replace_nan(a, 0)`: a copy with NaN -> +0.0, plus the NaN count left out. */
function zeroFilled(a: ArrayLike<number>): { arr: Float64Array; mask: Uint8Array; cnt: number } {
  const n = a.length;
  const arr = new Float64Array(n);
  const mask = new Uint8Array(n);
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    const v = a[i] as number;
    if (isnan(v)) mask[i] = 1;
    else { arr[i] = v; cnt++; }
  }
  return { arr, mask, cnt };
}

/**
 * `np.nanmean(a)` (numpy/lib/_nanfunctions_impl.py nanmean): NaNs become 0.0 IN PLACE and the whole
 * array is summed pairwise, so the NaN positions shape the summation tree. NaN when nothing is finite.
 */
export function nanmean(a: ArrayLike<number>): number {
  const { arr, cnt } = zeroFilled(a);
  return pairwiseSum(arr) / cnt;
}

/**
 * `np.nanvar(a, ddof=ddof)` (two-pass): mean of the zero-filled array, deviations with the NaN slots
 * reset to 0, pairwise sum of squares / (count - ddof). NaN when count - ddof <= 0.
 */
export function nanvar(a: ArrayLike<number>, ddof = 0): number {
  const { arr, mask, cnt } = zeroFilled(a);
  const avg = pairwiseSum(arr) / cnt;
  for (let i = 0; i < arr.length; i++) {
    const d = mask[i] ? 0.0 : (arr[i] as number) - avg;
    arr[i] = d * d;
  }
  const dof = cnt - ddof;
  if (dof <= 0) return NaN;
  return pairwiseSum(arr) / dof;
}

/** `np.nanstd(a, ddof=ddof)` = sqrt(nanvar). Math.sqrt is correctly rounded, like C sqrt. */
export function nanstd(a: ArrayLike<number>, ddof = 0): number {
  return Math.sqrt(nanvar(a, ddof));
}

/**
 * `np.nanmax(a)` (np.fmax.reduce): NaN only when every element is NaN; NaN for an empty array (numpy
 * raises there; callers in valence/ guard it). See the module note about signed zeros.
 */
export function nanmax(a: ArrayLike<number>): number {
  let best = NaN;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (isnan(v)) continue;
    if (isnan(best) || v > best || (v === 0 && best === 0 && !Object.is(v, -0))) best = v;
  }
  return best;
}

/** `np.nanmin(a)` (np.fmin.reduce); see nanmax. */
export function nanmin(a: ArrayLike<number>): number {
  let best = NaN;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (isnan(v)) continue;
    if (isnan(best) || v < best || (v === 0 && best === 0 && Object.is(v, -0))) best = v;
  }
  return best;
}

// ---------- Order statistics ----------

/** Ascending copy with NaNs last: the values numpy's partition leaves at every index we read. */
function sortedCopy(a: ArrayLike<number>): Float64Array {
  return Float64Array.from(a).sort();
}

/**
 * `np.median(a)` (numpy/lib/_function_base_impl.py _median): mean of the middle one or two values via
 * np.mean (so a lone -0.0 comes back +0.0); NaN when any element is NaN or the array is empty.
 */
export function median(a: ArrayLike<number>): number {
  const n = a.length;
  if (n === 0) return NaN;
  const s = sortedCopy(a);
  if (isnan(s[n - 1] as number)) return NaN;
  const h = Math.floor(n / 2);
  return n % 2 === 1 ? npMean([s[h] as number]) : npMean([s[h - 1] as number, s[h] as number]);
}

/**
 * `np.nanmedian(a)` for 1-D input: median of the non-NaN values; NaN when there are none (numpy
 * returns `arr1d[-1]`, which is then NaN) or the array is empty (numpy returns nanmean(empty)).
 */
export function nanmedian(a: ArrayLike<number>): number {
  const kept: number[] = [];
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (!isnan(v)) kept.push(v);
  }
  return kept.length ? median(kept) : NaN;
}

/**
 * `np.percentile(a, q)` with the default 'linear' method, for a scalar q in [0, 100]
 * (numpy/lib/_function_base_impl.py _quantile, _get_indexes, _get_gamma, _lerp).
 *
 * The interpolation is numpy's `_lerp`: `b - (b - a) * (1 - t)` when t >= 0.5, else `a + (b - a) * t`,
 * with t measured from the possibly clamped lower index (which is -1, i.e. the last element, when the
 * virtual index reaches the end). NaN when any element is NaN; NaN for an empty array (numpy raises).
 */
export function percentileLinear(a: ArrayLike<number>, q: number): number {
  const n = a.length;
  if (n === 0) return NaN;
  const s = sortedCopy(a);
  if (isnan(s[n - 1] as number)) return NaN;
  const quantile = q / 100;
  const virtual = (n - 1) * quantile;
  let prev = Math.floor(virtual);
  let next = prev + 1;
  if (virtual >= n - 1) { prev = -1; next = -1; }
  if (virtual < 0) { prev = 0; next = 0; }
  if (isnan(virtual)) { prev = -1; next = -1; }
  const gamma = virtual - prev; // numpy subtracts the clamped intp index, -1 included
  const lo = s[prev < 0 ? n + prev : prev] as number;
  const hi = s[next < 0 ? n + next : next] as number;
  const diff = hi - lo;
  return gamma >= 0.5 ? hi - diff * (1 - gamma) : lo + diff * gamma;
}

/**
 * Python `statistics.median(data)` (valence/runtime/aggregator.py:215, :237): sorts, then the middle
 * value or `(lo + hi) / 2`. NaN for empty data (Python raises StatisticsError).
 */
export function statsMedian(a: ArrayLike<number>): number {
  const n = a.length;
  if (n === 0) return NaN;
  // Python's sorted() is stable and only uses `<`; a numeric comparator with no ties broken matches it.
  const s = Array.from(a).sort((x, y) => (x < y ? -1 : y < x ? 1 : 0));
  const h = Math.floor(n / 2);
  return n % 2 === 1 ? (s[h] as number) : ((s[h - 1] as number) + (s[h] as number)) / 2;
}

/**
 * Scalar `np.clip(x, lo, hi)` (valence/models/served.py:94): NaN if any of the three is NaN; otherwise x is
 * kept unless it is strictly outside [lo, hi], so on a signed-zero tie the input's zero wins
 * (`clip(-0.0, 0.0, 1.0) == -0.0`). Rule determined empirically against numpy 2.5 by the exporter.
 */
export function clip(x: number, lo: number, hi: number): number {
  if (isnan(x) || isnan(lo) || isnan(hi)) return NaN;
  const v = x < lo ? lo : x;
  return v > hi ? hi : v;
}

// ---------- Exact decimal arithmetic on doubles ----------

const F64 = new Float64Array(1);
const U32 = new Uint32Array(F64.buffer);
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const POW10: bigint[] = [1n];

function pow10(k: number): bigint {
  for (let i = POW10.length; i <= k; i++) POW10.push((POW10[i - 1] as bigint) * 10n);
  return POW10[k] as bigint;
}

/** True when the sign bit is set (catches -0.0, which `x < 0` misses). */
function signBit(x: number): boolean {
  return x < 0 || Object.is(x, -0);
}

/** |x| = mant * 2**exp exactly, for finite x. */
function decompose(x: number): { mant: bigint; exp: number } {
  F64[0] = x;
  const hi = (LITTLE_ENDIAN ? U32[1] : U32[0]) as number;
  const lo = (LITTLE_ENDIAN ? U32[0] : U32[1]) as number;
  const e = (hi >>> 20) & 0x7ff;
  const frac = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  return e === 0 ? { mant: frac, exp: -1074 } : { mant: frac | (1n << 52n), exp: e - 1075 };
}

/** Position of a double on the ordered number line; -0.0 and +0.0 both map to 0. */
function orderedBits(x: number): bigint {
  F64[0] = x;
  const hi = (LITTLE_ENDIAN ? U32[1] : U32[0]) as number;
  const lo = (LITTLE_ENDIAN ? U32[0] : U32[1]) as number;
  const mag = (BigInt(hi & 0x7fffffff) << 32n) | BigInt(lo);
  return hi >>> 31 ? -mag : mag;
}

/**
 * How many representable doubles apart a and b are: 0 when equal (either zero sign), 0 for two NaNs,
 * Infinity when only one is NaN. For tolerance checks on values that went through exp/log/pow/sin/cos:
 * V8's Math functions, glibc's (CPython `math`) and numpy's AVX-512 SIMD builds (np.exp, np.power,
 * np.log10) each round the last bit differently, so those values cannot be bit-exact across them.
 */
export function ulpDistance(a: number, b: number): number {
  if (isnan(a) || isnan(b)) return isnan(a) && isnan(b) ? 0 : Infinity;
  const d = orderedBits(a) - orderedBits(b);
  return Number(d < 0n ? -d : d);
}

/** round-half-even(num / den) for non-negative bigints. */
function divRoundHalfEven(num: bigint, den: bigint): bigint {
  const q = num / den;
  const r2 = (num - q * den) * 2n;
  if (r2 > den || (r2 === den && (q & 1n) === 1n)) return q + 1n;
  return q;
}

/** round-half-even(|x| * 10**k) computed from the exact binary value of x (finite x, any integer k). */
function scaledRound(mant: bigint, exp: number, k: number): bigint {
  let num = mant;
  let den = 1n;
  if (exp >= 0) num <<= BigInt(exp);
  else den <<= BigInt(-exp);
  if (k >= 0) num *= pow10(k);
  else den *= pow10(-k);
  return divRoundHalfEven(num, den);
}

/**
 * |x| rounded half-even to `precision` significant digits: the digit string (exactly `precision` digits)
 * and the decimal point position `decpt` such that |x| ~= 0.d1d2... * 10**decpt (dtoa's convention).
 */
function roundSignificant(x: number, precision: number): { digits: string; decpt: number } {
  const { mant, exp } = decompose(x);
  const lower = pow10(precision - 1);
  const upper = pow10(precision);
  let e10 = Math.floor(Math.log10(Math.abs(x)));
  for (let guard = 0; guard < 4; guard++) {
    const n = scaledRound(mant, exp, precision - 1 - e10);
    if (n >= upper) { e10++; continue; }
    if (n < lower) { e10--; continue; }
    return { digits: n.toString(), decpt: e10 + 1 };
  }
  throw new Error(`roundSignificant did not converge for ${x}`);
}

/**
 * Shortest round-trip digits of |x| (finite, non-zero) and dtoa's `decpt`. ECMAScript's Number::toString
 * picks the shortest digit string that reads back as x and, among those, the closest to x, which is what
 * CPython's repr (dtoa mode 0) prints.
 */
function shortestDigits(x: number): { digits: string; decpt: number } {
  const s = String(Math.abs(x));
  const ePos = s.indexOf("e");
  const mantissa = ePos < 0 ? s : s.slice(0, ePos);
  const exp10 = ePos < 0 ? 0 : Number(s.slice(ePos + 1));
  const dot = mantissa.indexOf(".");
  const intPart = dot < 0 ? mantissa : mantissa.slice(0, dot);
  const fracPart = dot < 0 ? "" : mantissa.slice(dot + 1);
  let digits = intPart + fracPart;
  let decpt = intPart.length + exp10;
  let lead = 0;
  while (lead < digits.length - 1 && digits[lead] === "0") lead++;
  digits = digits.slice(lead);
  decpt -= lead;
  return { digits: digits.replace(/0+$/, "") || "0", decpt };
}

/**
 * CPython's format_float_short layout (Python/pystrtod.c) for the 'g' and 'r' codes: places the point,
 * pads zeros, switches to an exponent and adds ".0" to integers in repr mode.
 */
function layoutShort(
  neg: boolean,
  digits: string,
  decptIn: number,
  code: "g" | "r",
  precision: number,
  alwaysSign: boolean,
): string {
  let decpt = decptIn;
  let useExp = false;
  let exp = 0;
  if (code === "g") useExp = decpt <= -4 || decpt > precision;
  else useExp = decpt <= -4 || decpt > 16;
  if (useExp) { exp = decpt - 1; decpt = 1; }
  const addDot0 = code === "r";
  const len = digits.length;
  const vEnd = !useExp && addDot0 ? Math.max(len, decpt + 1) : Math.max(len, decpt);
  let body: string;
  if (decpt <= 0) body = "0." + "0".repeat(-decpt) + digits + "0".repeat(vEnd - len);
  else if (decpt <= len) body = digits.slice(0, decpt) + "." + digits.slice(decpt) + "0".repeat(vEnd - len);
  else body = digits + "0".repeat(decpt - len) + "." + "0".repeat(vEnd - decpt);
  if (body.endsWith(".")) body = body.slice(0, -1);
  if (useExp) {
    const a = Math.abs(exp);
    body += "e" + (exp < 0 ? "-" : "+") + (a < 10 ? "0" : "") + String(a);
  }
  return (neg ? "-" : alwaysSign ? "+" : "") + body;
}

/** CPython's spelling of nan/inf in every float format: the sign of a NaN is ignored. */
function nonFinite(x: number, alwaysSign: boolean): string {
  if (isnan(x)) return alwaysSign ? "+nan" : "nan";
  return x < 0 ? "-inf" : alwaysSign ? "+inf" : "inf";
}

// ---------- Python float formatting and rounding ----------

/**
 * Python `round(x, ndigits)` for a float (Objects/floatobject.c float___round___impl / double_round):
 * the exact binary value rounded half-even to `ndigits` decimals (negative allowed), read back correctly
 * rounded. So `round(2.675, 2) == 2.67`, `round(0.125, 2) == 0.12`, `round(-0.001, 2) == -0.0`.
 * NaN and +-inf come back unchanged; a result too large for a double throws (Python: OverflowError).
 */
export function pyRound(x: number, ndigits = 0): number {
  if (!Number.isFinite(x)) return x;
  if (!Number.isInteger(ndigits)) throw new TypeError(`ndigits must be an integer, got ${ndigits}`);
  if (ndigits > 323) return x; // NDIGITS_MAX: x always rounds to itself
  if (ndigits < -308) return 0.0 * x; // NDIGITS_MIN: +-0.0 with the sign of x
  const neg = signBit(x);
  const { mant, exp } = decompose(x);
  if (ndigits >= 0 && exp >= -ndigits) return x; // x * 10**ndigits is already an integer
  const n = scaledRound(mant, exp, ndigits);
  if (n === 0n) return neg ? -0 : 0;
  const out = Number(`${neg ? "-" : ""}${n.toString()}e${-ndigits}`);
  // e.g. round(1.7976931348623157e308, -308): Python raises OverflowError instead of returning inf
  if (!Number.isFinite(out)) throw new RangeError("rounded value too large to represent");
  return out;
}

/**
 * Python `round(x)` with no ndigits: the nearest integer, ties to even, as a (JS) number. Python raises
 * on NaN (ValueError) and on +-inf (OverflowError); so does this (RangeError).
 */
export function pyIntRound(x: number): number {
  if (!Number.isFinite(x)) throw new RangeError(`cannot round ${x} to an integer`);
  const f = Math.floor(x);
  const d = x - f; // exact: |x| < 2**52 here, or x is already an integer and d == 0
  let r: number;
  if (d < 0.5) r = f;
  else if (d > 0.5) r = f + 1;
  else r = f % 2 === 0 ? f : f + 1;
  return r + 0; // Python ints have no -0
}

/** `np.rint(x)`: round half to even, keeping the sign of zero (`rint(-0.4) == -0.0`). */
export function rint(x: number): number {
  const r = Math.round(x); // half-up; |x| < 2**52 makes r - x exact, beyond that x is integral
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** numpy's power_of_ten (numpy/_core/src/multiarray/calculation.c): a table to 1e8, then repeated *10. */
function npPowerOfTen(n: number): number {
  if (n < 9) return [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8][n] as number;
  let r = 1e9;
  for (let i = n; i > 9; i--) r *= 10;
  return r;
}

/**
 * `np.round(x, decimals)`, which is also what `round(np.float64(x), decimals)` does: the fast, inexact
 * `rint(x * 10**d) / 10**d` (or `rint(x / 10**-d) * 10**-d`), NOT CPython's exact rounding. Use it only
 * where the Python value is a numpy scalar; Python floats go through pyRound.
 */
export function npRound(x: number, decimals = 0): number {
  if (decimals >= 0) {
    const f = npPowerOfTen(decimals);
    return rint(x * f) / f;
  }
  const f = npPowerOfTen(-decimals);
  return rint(x / f) * f;
}

/**
 * Python `f"{x:.{digits}f}"`, or `f"{x:+.{digits}f}"` with `sign: true`: fixed point from the exact
 * binary value, ties to even (`f"{0.125:.2f}" == "0.12"`), full digits for huge values, and a "-" on
 * anything negative that rounds to zero (`f"{-0.001:.2f}" == "-0.00"`, `f"{-0.0:+.2f}" == "-0.00"`).
 */
export function pyFixed(x: number, digits: number, opts: { sign?: boolean } = {}): string {
  const always = opts.sign ?? false;
  if (!Number.isFinite(x)) return nonFinite(x, always);
  if (!Number.isInteger(digits) || digits < 0) throw new RangeError(`bad precision ${digits}`);
  const neg = signBit(x);
  let s = "0";
  if (x !== 0) {
    const { mant, exp } = decompose(x);
    s = scaledRound(mant, exp, digits).toString();
  }
  if (digits > 0) {
    if (s.length <= digits) s = "0".repeat(digits + 1 - s.length) + s;
    s = s.slice(0, s.length - digits) + "." + s.slice(s.length - digits);
  }
  return (neg ? "-" : always ? "+" : "") + s;
}

/**
 * Python `f"{x:g}"` (precision 6) or `f"{x:.{precision}g}"`: round half-even to `precision` significant
 * digits, drop trailing zeros, exponent form when the exponent is < -4 or >= precision
 * (`f"{1e16:g}" == "1e+16"`, `f"{100000.0:g}" == "100000"`, `f"{-0.0:g}" == "-0"`).
 */
export function pyG(x: number, precision = 6, opts: { sign?: boolean } = {}): string {
  const always = opts.sign ?? false;
  if (!Number.isFinite(x)) return nonFinite(x, always);
  const p = precision === 0 ? 1 : precision;
  const neg = signBit(x);
  if (x === 0) return layoutShort(neg, "0", 1, "g", p, always);
  const { digits, decpt } = roundSignificant(x, p);
  return layoutShort(neg, digits.replace(/0+$/, "") || "0", decpt, "g", p, always);
}

/**
 * Python `repr(x)` / `str(x)` for a float: shortest round-trip digits, exponent form below 1e-4 or from
 * 1e16 up, ".0" on integral values: '1e-05', '1e+16', '1000000000000000.0', '2.0', '-0.0', 'nan', 'inf'.
 */
export function pyRepr(x: number): string {
  if (!Number.isFinite(x)) return nonFinite(x, false);
  const neg = signBit(x);
  if (x === 0) return neg ? "-0.0" : "0.0";
  const { digits, decpt } = shortestDigits(x);
  return layoutShort(neg, digits, decpt, "r", 0, false);
}
