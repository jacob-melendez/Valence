/**
 * Image-region helpers: head boxes and box normalization. Port of valence/perception/regions.py.
 *
 * Not ported: `black_out` (the edge's --obscure-face switch, zeroing head regions of the live frame in place); the
 * browser never writes frame pixels. Nothing here copies, keeps or returns pixel data.
 */

export const HEAD_POINTS = [0, 1, 2, 3, 4] as const;
export const L_EYE = 1;
export const R_EYE = 2;
export const L_SHOULDER = 5;
export const R_SHOULDER = 6;
export const SHOULDER_TO_HEAD = 0.6;
/** Ear-to-ear span ~ 2x eye distance; keeps the box face-sized when ears are hidden. */
export const EYES_TO_EARS = 2.0;
export const FALLBACK_HEAD_FRAC = 0.15;

/** math.floor / math.ceil: like Python, a non-finite value cannot become an integer. */
function pyFloor(x: number): number {
  if (!Number.isFinite(x)) throw new RangeError(`cannot convert float ${x} to integer`);
  return Math.floor(x) + 0;
}

function pyCeil(x: number): number {
  if (!Number.isFinite(x)) throw new RangeError(`cannot convert float ${x} to integer`);
  return Math.ceil(x) + 0;
}

const F64 = new DataView(new ArrayBuffer(8));

/** Finite nonzero x as m * 2^e, m an integer (|m| < 2^53). */
function decompose(x: number): [bigint, number] {
  F64.setFloat64(0, x);
  const hi = F64.getUint32(0), lo = F64.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  if (biased) m |= 1n << 52n;
  return [x < 0 ? -m : m, biased ? biased - 1075 : -1074];
}

/** n * 2^e rounded to the nearest double, ties to even (n != 0). */
function roundToDouble(n: bigint, e: number): number {
  let m = n < 0n ? -n : n;
  const q = Math.max(m.toString(2).length + e - 53, -1074); // exponent of the result's last mantissa bit
  if (q > e) {
    const shift = BigInt(q - e);
    const rem = m & ((1n << shift) - 1n);
    const half = 1n << (shift - 1n);
    m >>= shift;
    if (rem > half || (rem === half && (m & 1n) === 1n)) m += 1n;
    e = q;
  }
  let x = Number(m); // <= 2^53: exact; the scaling below is exact unless it overflows
  for (; e > 1023; e -= 1023) x *= 2 ** 1023;
  for (; e < -1022; e += 1022) x *= 2 ** -1022;
  x *= 2 ** e;
  return n < 0n ? -x : x;
}

/** Correctly rounded a * b + c with a single rounding (C fma, Python math.fma). */
export function fma(a: number, b: number, c: number): number {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a * b + c;
  if (!Number.isFinite(c)) return c;
  if (a === 0 || b === 0) return a * b + c;
  if (c === 0) return a * b;
  const [ma, ea] = decompose(a), [mb, eb] = decompose(b), [mc, ec] = decompose(c);
  const e = Math.min(ea + eb, ec);
  const n = ((ma * mb) << BigInt(ea + eb - e)) + (mc << BigInt(ec - e));
  return n === 0n ? 0 : roundToDouble(n, e);
}

/**
 * np.linalg.norm of the 2-vector (dx, dy): sqrt(x.dot(x)), and the dot is BLAS ddot, which on the reference build
 * (OpenBLAS, x86-64) evaluates as fma(dy, dy, dx * dx). Neither the naive sum nor Math.hypot matches it bit for bit.
 */
function norm2(dx: number, dy: number): number {
  return Math.sqrt(fma(dy, dy, dx * dx));
}

/**
 * Square box (x1, y1, x2, y2) (x2/y2 exclusive) around a person's head; `kptsXy` holds the 17 COCO keypoints as
 * [x0, y0, x1, y1, ...] in pixels, `kptsConf` their confidences.
 *
 * Centred on the mean of the confident nose/eye/ear keypoints. Its side is `scale` times the largest of the head
 * points' spread, 2 x eye distance (both eyes confident; covers hidden ears) and 0.6 x shoulder width (both
 * shoulders confident); with a single head point and no shoulders it falls back to 0.15 x frameW. Clipped to the
 * frame; null if nothing is visible. Like Python, throws when the box edge is not finite (NaN or infinite scale).
 */
export function headBoxPx(
  kptsXy: ArrayLike<number>,
  kptsConf: ArrayLike<number>,
  frameW: number,
  frameH: number,
  scale = 1.8,
  minConf = 0.3,
): [number, number, number, number] | null {
  const n = kptsConf.length;
  if (n < 7 || kptsXy.length !== 2 * n) {
    throw new RangeError("headBoxPx expects n >= 7 keypoints: 2n coordinates and n confidences");
  }
  const x = (k: number) => kptsXy[2 * k]!;
  const y = (k: number) => kptsXy[2 * k + 1]!;
  const ok = (k: number) => kptsConf[k]! >= minConf && Number.isFinite(x(k)) && Number.isFinite(y(k));

  const head = HEAD_POINTS.filter(ok);
  if (!head.length) return null;
  const xs = head.map(x), ys = head.map(y);
  const dxSpan = Math.max(...xs) - Math.min(...xs), dySpan = Math.max(...ys) - Math.min(...ys);
  let spread = dySpan > dxSpan ? dySpan : dxSpan;
  if (ok(L_EYE) && ok(R_EYE)) {
    const v = EYES_TO_EARS * norm2(x(L_EYE) - x(R_EYE), y(L_EYE) - y(R_EYE));
    if (v > spread) spread = v;
  }
  if (ok(L_SHOULDER) && ok(R_SHOULDER)) {
    const v = SHOULDER_TO_HEAD * norm2(x(L_SHOULDER) - x(R_SHOULDER), y(L_SHOULDER) - y(R_SHOULDER));
    if (v > spread) spread = v;
  }
  if (spread <= 0.0) spread = FALLBACK_HEAD_FRAC * frameW;

  // head.mean(axis=0): a left-to-right sum, then / count.
  let sx = 0, sy = 0;
  for (let i = 0; i < head.length; i++) {
    sx += xs[i]!;
    sy += ys[i]!;
  }
  const cx = sx / head.length, cy = sy / head.length;
  const half = (scale * spread) / 2.0;
  // Python max(0, v) / min(frame, v): the second argument wins only if strictly larger / smaller.
  const x1 = pyFloor(cx - half), y1 = pyFloor(cy - half), x2 = pyCeil(cx + half), y2 = pyCeil(cy + half);
  const box: [number, number, number, number] = [
    x1 > 0 ? x1 : 0, y1 > 0 ? y1 : 0, x2 < frameW ? x2 : frameW, y2 < frameH ? y2 : frameH,
  ];
  return box[2] > box[0] && box[3] > box[1] ? box : null;
}

/** Python min(1.0, max(0.0, v)): NaN maps to 0.0. */
function clip01(v: number): number {
  const lo = v > 0.0 ? v : 0.0;
  return lo < 1.0 ? lo : 1.0;
}

/** Pixel (x1, y1, x2, y2) -> fractions of the frame, each clipped to [0, 1]. */
export function normBbox(xyxyPx: ArrayLike<number>, frameW: number, frameH: number): [number, number, number, number] {
  if (frameW <= 0 || frameH <= 0) throw new RangeError("frame size must be positive");
  if (xyxyPx.length !== 4) throw new TypeError(`normBbox expects (x1, y1, x2, y2), got ${xyxyPx.length} values`);
  return [
    clip01(xyxyPx[0]! / frameW), clip01(xyxyPx[1]! / frameH), clip01(xyxyPx[2]! / frameW), clip01(xyxyPx[3]! / frameH),
  ];
}
