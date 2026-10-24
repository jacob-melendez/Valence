/**
 * Polyphase resampling to 16 kHz. Port of scipy.signal.resample_poly(x, up, down) with its defaults
 * (window=('kaiser', 5.0), padtype='constant', 1-D input) as valence/perception/audio_features.py's
 * resample_48k_to_16k calls it: the input is cast to float32 first, and scipy then filters a float32 signal in
 * float32 (the float64 firwin taps are cast to the input dtype, `h *= up` and every product and sum of upfirdn are
 * float32). This port does the same, rounding each operation with Math.fround (exact float32 arithmetic: double
 * rounding is innocuous for + and *), so its output is bit-identical to scipy's. The taps (firwin, kaiser, i0) are
 * float64, as in scipy.
 */
import { pairwiseSum } from "./audioFeatures";

/** np.finfo(np.float64).eps: np.sinc's stand-in for an argument of 0. */
const EPS = 2.220446049250313e-16;

/** Error-free a * b = p + e (Dekker; JS has no fma). */
function twoProd(a: number, b: number): [number, number] {
  const p = a * b;
  const ca = 134217729 * a;
  const ah = ca - (ca - a);
  const al = a - ah;
  const cb = 134217729 * b;
  const bh = cb - (cb - b);
  const bl = b - bh;
  return [p, ((ah * bh - p) + ah * bl + al * bh) + al * bl];
}

/** Error-free a + b = s + e. */
function twoSum(a: number, b: number): [number, number] {
  const s = a + b;
  const bb = s - a;
  return [s, (a - (s - bb)) + (b - bb)];
}

/**
 * Modified Bessel function of order 0 (scipy.special.i0) by its power series sum_k ((x/2)^k / k!)^2, summed in
 * double-double arithmetic: relative error ~1e-16 (the plain float64 series drifts past 1e-15 for |x| > 10). Like
 * scipy's Cephes code, +inf once exp(|x|) overflows (|x| > 709.78) and NaN for +-inf. Terms are carried times
 * 2^-600 for |x| > 600 so the Dekker splits cannot overflow.
 */
export function i0(x: number): number {
  const ax = Math.abs(x);
  if (Number.isNaN(x) || ax === Infinity) return NaN; // Cephes: exp(x) * chbevl(..) / sqrt(x) = inf / inf
  if (ax > 709.782712893384) return Infinity; // Cephes: exp(x) overflows
  const half = 0.5 * ax;
  const [qh, ql] = twoProd(half, half); // q = (x / 2)^2
  const scale = half > 300 ? 2 ** -600 : 1;
  let th = scale;
  let tl = 0;
  let sh = scale;
  let sl = 0;
  for (let k = 1; th > sh * 1e-34; k++) {
    // term *= q
    const [p, pe] = twoProd(th, qh);
    const e = pe + (th * ql + tl * qh);
    th = p + e;
    tl = e - (th - p);
    // term /= k^2 (exact divisor)
    const d = k * k;
    const q1 = th / d;
    const [p1, p2] = twoProd(q1, d);
    let [s, r] = twoSum(th, -p1);
    r = r + tl - p2;
    const q2 = (s + r) / d;
    th = q1 + q2;
    tl = q2 - (th - q1);
    // sum += term
    [s, r] = twoSum(sh, th);
    r += sl + tl;
    sh = s + r;
    sl = r - (sh - s);
  }
  return (sh + sl) / scale;
}

/** scipy.signal.windows.kaiser(M, beta, sym=True): i0(beta sqrt(1 - ((n - alpha) / alpha)^2)) / i0(beta). */
export function kaiser(M: number, beta: number): Float64Array {
  if (!Number.isInteger(M) || M < 0) throw new Error("Window length M must be a non-negative integer");
  if (M <= 1) return new Float64Array(M).fill(1.0);
  const w = new Float64Array(M);
  const alpha = (M - 1) / 2.0;
  const denom = i0(beta);
  for (let n = 0; n < M; n++) {
    const r = (n - alpha) / alpha;
    w[n] = i0(beta * Math.sqrt(1 - r * r)) / denom;
  }
  return w;
}

/**
 * scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)) for one low-pass cutoff (relative to Nyquist,
 * fs=2, pass_zero=True, scale=True): windowed sinc normalised to unit gain at DC.
 */
export function firwin(numtaps: number, cutoff: number, beta = 5.0): Float64Array {
  if (cutoff <= 0 || cutoff >= 1) {
    throw new Error("Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.");
  }
  const alpha = 0.5 * (numtaps - 1);
  const win = kaiser(numtaps, beta);
  const h = new Float64Array(numtaps);
  for (let i = 0; i < numtaps; i++) {
    const m = i - alpha;
    const x = Math.PI * (cutoff * m);
    const y = x ? x : EPS; // np.sinc
    h[i] = cutoff * (Math.sin(y) / y) * win[i]!;
  }
  // scale_frequency 0: c = cos(pi m 0) = 1, so s = sum(h * c) = sum(h) (np.sum: 0.0 + pairwise sum).
  const s = 0.0 + pairwiseSum(h);
  for (let i = 0; i < numtaps; i++) h[i]! /= s;
  return h;
}

/** math.gcd for non-negative integers. */
function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

/** Cython _output_len: ceil(((in_len - 1) * up + len_h) / down) in floored integer arithmetic. */
function outputLen(lenH: number, inLen: number, up: number, down: number): number {
  return Math.floor(((inLen - 1) * up + lenH - 1) / down) + 1;
}

/** firwin taps for reduced (up, down), cast to float32 and multiplied by up in float32; built once per ratio. */
const tapsCache = new Map<string, Float32Array>();

function scaledTaps(up: number, down: number): Float32Array {
  const key = `${up}/${down}`;
  let h = tapsCache.get(key);
  if (h === undefined) {
    const maxRate = Math.max(up, down);
    const h64 = firwin(2 * (10 * maxRate) + 1, 1 / maxRate, 5.0);
    h = new Float32Array(h64.length);
    for (let i = 0; i < h64.length; i++) h[i] = Math.fround(Math.fround(h64[i]!) * up);
    tapsCache.set(key, h);
  }
  return h;
}

/**
 * Resample `x` by up / down with scipy.signal.resample_poly's defaults: zero-phase Kaiser(5.0) low-pass of
 * 2 * 10 * max(up, down) + 1 taps after gcd reduction, zeros beyond both ends, ceil(len(x) * up / down) samples
 * whose first sample lines up with x[0]. The result is `resample_poly(np.asarray(x, np.float32), up, down)`;
 * up == down returns a float32 copy. Temporary copies are zero-filled; nothing is retained.
 */
export function resamplePoly(x: ArrayLike<number>, up: number, down: number): Float32Array {
  if (!Number.isInteger(up)) throw new Error("up must be an integer");
  if (!Number.isInteger(down)) throw new Error("down must be an integer");
  if (up < 1 || down < 1) throw new Error("up and down must be >= 1");
  const owned = !(x instanceof Float32Array);
  const xf = x instanceof Float32Array ? x : Float32Array.from(x);
  try {
    const g = gcd(up, down);
    up /= g;
    down /= g;
    if (up === 1 && down === 1) return xf.slice();
    const nIn = xf.length;
    const nOut = Math.floor((nIn * up) / down) + (nIn * up % down ? 1 : 0);
    const out = new Float32Array(nOut);
    if (nOut === 0) return out;

    const h = scaledTaps(up, down);
    const halfLen = 10 * Math.max(up, down);
    // Zero-pad the filter to put the output samples at the centre.
    const nPrePad = down - (halfLen % down);
    let nPostPad = 0;
    const nPreRemove = Math.floor((halfLen + nPrePad) / down);
    while (outputLen(h.length + nPrePad + nPostPad, nIn, up, down) < nOut + nPreRemove) nPostPad += 1;
    const lenH = h.length + nPrePad + nPostPad;
    const lenOut = outputLen(lenH, nIn, up, down);

    // _pad_h: zero-pad to a multiple of up, then store each phase's coefficients flipped.
    const hPerPhase = (lenH + (((-lenH % up) + up) % up)) / up;
    const hTransFlip = new Float32Array(hPerPhase * up);
    for (let c = 0; c < up; c++) {
      for (let j = 0; j < hPerPhase; j++) {
        const k = (hPerPhase - 1 - j) * up + c - nPrePad; // index into the unpadded taps
        hTransFlip[c * hPerPhase + j] = k >= 0 && k < h.length ? h[k]! : 0;
      }
    }

    // _apply_impl (mode 'constant', cval 0), keeping only outputs nPreRemove .. nPreRemove + nOut - 1.
    const end = Math.min(lenOut, nPreRemove + nOut);
    const paddedLen = nIn + hPerPhase - 1;
    let xIdx = 0;
    let yIdx = 0;
    let t = 0;
    while (xIdx < nIn) {
      if (yIdx >= nPreRemove) {
        let hIdx = t * hPerPhase;
        let xConvIdx = xIdx - hPerPhase + 1;
        if (xConvIdx < 0) {
          hIdx -= xConvIdx;
          xConvIdx = 0;
        }
        let acc = 0;
        for (; xConvIdx <= xIdx; xConvIdx++) {
          acc = Math.fround(acc + Math.fround(xf[xConvIdx]! * hTransFlip[hIdx++]!));
        }
        out[yIdx - nPreRemove] = acc;
      }
      yIdx += 1;
      if (yIdx >= end) return out;
      t += down;
      xIdx += Math.floor(t / up);
      t = t % up;
    }
    // Flush the last outputs: samples beyond the end are cval = 0 (their products are still added, as in scipy).
    while (xIdx < paddedLen) {
      if (yIdx >= nPreRemove) {
        let hIdx = t * hPerPhase;
        let acc = 0;
        for (let xConvIdx = xIdx - hPerPhase + 1; xConvIdx <= xIdx; xConvIdx++) {
          const xval = xConvIdx >= nIn || xConvIdx < 0 ? 0 : xf[xConvIdx]!;
          acc = Math.fround(acc + Math.fround(xval * hTransFlip[hIdx++]!));
        }
        out[yIdx - nPreRemove] = acc;
      }
      yIdx += 1;
      if (yIdx >= end) return out;
      t += down;
      xIdx += Math.floor(t / up);
      t = t % up;
    }
    return out;
  } finally {
    if (owned) xf.fill(0);
  }
}

/** Polyphase 48 kHz -> 16 kHz (anti-aliased), float32 (resample_48k_to_16k). */
export function resample48kTo16k(x: ArrayLike<number>): Float32Array {
  return resamplePoly(x, 1, 3);
}

/**
 * Any integer sample rate -> 16 kHz: up / down = 16000 / sampleRate reduced by their gcd (48000 -> 1/3,
 * 44100 -> 160/441, 32000 -> 1/2, 16000 -> a float32 copy).
 */
export function resampleTo16k(x: ArrayLike<number>, sampleRate: number): Float32Array {
  if (!Number.isInteger(sampleRate) || sampleRate < 1) throw new Error("sampleRate must be a positive integer");
  const g = gcd(16000, sampleRate);
  return resamplePoly(x, 16000 / g, sampleRate / g);
}
