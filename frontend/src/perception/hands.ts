/**
 * Hands -> people, fist heuristics and per-person fist onsets. Port of valence/perception/hands.py.
 *
 * MediaPipe's GestureRecognizer sees hands but not people; the pose model sees people and their wrists. `assignHands`
 * joins the two by wrist position. `isCurlFist` is a geometric fallback for when the gesture head is unsure.
 * `FistTracker` turns noisy per-hand Closed_Fist scores into debounced onsets per display id. Everything here is
 * coordinates and scalars; no image data is touched.
 */
import { fistDebouncer, TIME_EPS, type HysteresisDebouncer } from "./debounce";

export type Side = "L" | "R";
/** (display id, side) */
export type HandKey = [number, Side];

export const SIDES: readonly Side[] = ["L", "R"];
export const WRIST = 0;
export const FINGER_TIP_PIP: ReadonlyArray<readonly [number, number]> = [[8, 6], [12, 10], [16, 14], [20, 18]];

const VIEW = new DataView(new ArrayBuffer(8));
const TWO_64 = 18446744073709551616; // 2 ** 64
const DBL_MIN = 2.2250738585072014e-308; // 2 ** -1022, the smallest normal double
const VELTKAMP = 134217729; // 2 ** 27 + 1

/** frexp(x)[1] for a finite x > 0: x = m * 2 ** e with 0.5 <= m < 1. */
function frexpExp(x: number): number {
  VIEW.setFloat64(0, x);
  const biased = (VIEW.getUint32(0) >>> 20) & 0x7ff;
  return biased === 0 ? frexpExp(x * TWO_64) - 64 : biased - 1022;
}

/** ldexp(1.0, k), exactly, for -1074 <= k <= 1023. */
function pow2(k: number): number {
  if (k < -1022) return pow2(k + 64) / TWO_64;
  VIEW.setUint32(0, (k + 1023) << 20);
  VIEW.setUint32(4, 0);
  return VIEW.getFloat64(0);
}

/** a * b - p exactly, for p = fl(a * b) (Dekker's product with Veltkamp splitting; what CPython gets from fma). */
function productError(a: number, b: number, p: number): number {
  let c = VELTKAMP * a;
  const ah = c - (c - a);
  const al = a - ah;
  c = VELTKAMP * b;
  const bh = c - (c - b);
  const bl = b - bh;
  return ah * bh - p + ah * bl + al * bh + al * bl;
}

/** CPython's vector_norm (Modules/mathmodule.c) for two non-negative coordinates. */
function vectorNorm(a: number, b: number, max: number, foundNan: boolean): number {
  if (max === Infinity) return max;
  if (foundNan) return NaN;
  if (max === 0.0) return max;
  const maxE = frexpExp(max);
  if (maxE < -1023) return DBL_MIN * vectorNorm(a / DBL_MIN, b / DBL_MIN, max / DBL_MIN, foundNan);
  const scale = pow2(-maxE);
  let csum = 1.0;
  let frac1 = 0.0;
  let frac2 = 0.0;
  for (const v of [a, b]) {
    const x = v * scale; // lossless scaling
    const sq = x * x;
    const sum = csum + sq; // dl_fast_sum(csum, sq)
    frac1 += productError(x, x, sq);
    frac2 += sq - (sum - csum);
    csum = sum;
  }
  let h = Math.sqrt(csum - 1.0 + (frac1 + frac2));
  const sq = -h * h;
  const sum = csum + sq;
  frac1 += productError(-h, h, sq);
  frac2 += sq - (sum - csum);
  csum = sum;
  h += (csum - 1.0 + (frac1 + frac2)) / (2.0 * h); // differential correction
  return h / scale;
}

/**
 * math.hypot(x, y) bit for bit, as CPython computes it (scaled, compensated, with a differential correction; checked
 * against 3.13). Math.hypot rounds differently in the last place, which would move distance ties and gate boundaries.
 * Infinity wins over NaN.
 */
export function pyHypot(x: number, y: number): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  let max = 0.0;
  if (ax > max) max = ax;
  if (ay > max) max = ay;
  return vectorNorm(ax, ay, max, Number.isNaN(ax) || Number.isNaN(ay));
}

/** Python's ordering of two tuple elements (no NaN here: gated distances are never NaN; sides are "L" < "R"). */
const cmp = <T extends number | string>(a: T, b: T): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * Match each MediaPipe hand (wrist landmark, in pixels) to a person's wrist.
 *
 * `personWrists` maps display id -> `{ L: [x, y, conf], R: [x, y, conf] }` from COCO keypoints 9/10. Matching is
 * globally greedy: every hand/wrist pair within `maxDistFrac * frameW` is sorted by distance (ties by hand index, then
 * display id, then side) and accepted nearest first, so each hand and each wrist is used at most once. Wrists below
 * `minConf` or with a non-finite position are ignored. Returns `[displayId, side]` or null per hand.
 */
export function assignHands(
  handWristsPx: ReadonlyArray<readonly [number, number]>,
  personWrists: ReadonlyMap<number, Readonly<Record<Side, readonly [number, number, number]>>>,
  frameW: number,
  maxDistFrac = 0.12,
  minConf = 0.5,
): ([number, Side] | null)[] {
  const maxDist = maxDistFrac * frameW;
  const candidates: [number, Side, number, number][] = [];
  for (const [pid, sides] of personWrists) {
    for (const [side, [wx, wy, conf]] of Object.entries(sides) as [Side, readonly [number, number, number]][]) {
      if (conf >= minConf && Number.isFinite(wx) && Number.isFinite(wy)) candidates.push([pid, side, wx, wy]);
    }
  }
  const pairs: [number, number, number, Side][] = [];
  handWristsPx.forEach(([hx, hy], h) => {
    for (const [pid, side, wx, wy] of candidates) {
      const d = pyHypot(hx - wx, hy - wy);
      if (d <= maxDist) pairs.push([d, h, pid, side]);
    }
  });
  pairs.sort((a, b) => cmp(a[0], b[0]) || cmp(a[1], b[1]) || cmp(a[2], b[2]) || cmp(a[3], b[3]));
  const result: ([number, Side] | null)[] = handWristsPx.map(() => null);
  const used = new Set<string>();
  for (const [, h, pid, side] of pairs) {
    const key = handKey(pid, side);
    if (result[h] === null && !used.has(key)) {
      result[h] = [pid, side];
      used.add(key);
    }
  }
  return result;
}

/**
 * Count curled fingers (index..pinky) from 21 normalized hand landmarks, each `[x, y]` or `[x, y, z]`.
 *
 * A finger is curled when its tip is closer to the wrist than its PIP joint. Distances use x as-is and y multiplied by
 * `aspect` (frame h / w), i.e. frame widths, so they are isotropic; z is ignored. Non-finite landmarks never count.
 * Throws on anything that is not a (21, >= 2) array, like the Python ValueError.
 */
export function curledFingers(landmarks: ReadonlyArray<ArrayLike<number>>, aspect = 9 / 16): number {
  const cols = landmarks[0]?.length ?? 0;
  if (landmarks.some((row) => row.length !== cols)) throw new Error("expected (21, 2|3) landmarks, got a ragged array");
  if (landmarks.length !== 21 || cols < 2) {
    throw new Error(`expected (21, 2|3) landmarks, got (${landmarks.length}, ${cols})`);
  }
  // xy = pts[:, :2] * [1, aspect]; dist = np.linalg.norm(xy - xy[WRIST], axis=1) = sqrt(dx * dx + dy * dy)
  const wrist = landmarks[WRIST]!;
  const wx = wrist[0]!;
  const wy = wrist[1]! * aspect;
  const dist = (k: number): number => {
    const p = landmarks[k]!;
    const dx = p[0]! - wx;
    const dy = p[1]! * aspect - wy;
    return Math.sqrt(dx * dx + dy * dy);
  };
  let curled = 0;
  for (const [tip, pip] of FINGER_TIP_PIP) if (dist(tip) < dist(pip)) curled += 1;
  return curled;
}

/** Fallback fist test: at least `minCurled` of the four fingers curled. */
export function isCurlFist(landmarks: ReadonlyArray<ArrayLike<number>>, aspect = 9 / 16, minCurled = 3): boolean {
  return curledFingers(landmarks, aspect) >= minCurled;
}

/** Map key of a (display id, side) hand: -0 and 0 coincide, as they do in a Python dict. */
function handKey(displayId: number, side: string): string {
  return `${displayId}|${side}`;
}

/**
 * One hysteresis debouncer per (display id, side), reported per person.
 *
 * A person's fist onset is a rising edge of "any hand clenched": clenching both fists together is one event, and a
 * second hand closing while the first is still clenched, or within the refractory, adds nothing. `personState`
 * returns the onsets since the previous call for that person (a delta, so callers can sum it into a rolling window)
 * and whether any of their hands is clenched.
 */
export class FistTracker {
  private readonly debouncers = new Map<string, { key: HandKey; deb: HysteresisDebouncer }>();
  private readonly pending = new Map<number, number>();
  private readonly lastOnset = new Map<number, number>();

  constructor(private readonly factory: () => HysteresisDebouncer = fistDebouncer) {}

  /** Feed one Closed_Fist score for a hand; true iff a person-level onset was counted. */
  update(displayId: number, side: Side, score: number, t: number): boolean {
    if (!SIDES.includes(side)) throw new Error(`side must be one of L, R, got ${JSON.stringify(side)}`);
    const key = handKey(displayId, side);
    let entry = this.debouncers.get(key);
    if (entry === undefined) {
      entry = { key: [displayId, side], deb: this.factory() };
      this.debouncers.set(key, entry);
    }
    const deb = entry.deb;
    const other = this.debouncers.get(handKey(displayId, SIDES[1 - SIDES.indexOf(side)]!));
    const otherActive = other !== undefined && other.deb.active;
    if (!deb.update(score, t) || otherActive) return false;
    if (t - (this.lastOnset.get(displayId) ?? -Infinity) < deb.refractoryS - TIME_EPS) return false;
    this.lastOnset.set(displayId, t);
    this.pending.set(displayId, (this.pending.get(displayId) ?? 0) + 1);
    return true;
  }

  /** Feed NaN ("not observed") to every known hand not in `seen` this tick. */
  markUnobserved(t: number, seen: Iterable<readonly [number, Side]>): void {
    const seenKeys = new Set<string>();
    for (const [displayId, side] of seen) seenKeys.add(handKey(displayId, side));
    for (const [key, { deb }] of this.debouncers) if (!seenKeys.has(key)) deb.update(NaN, t);
  }

  /** `[onsets since the last call for this person, any hand active]`. */
  personState(displayId: number): [number, boolean] {
    const onsets = this.pending.get(displayId) ?? 0;
    this.pending.delete(displayId);
    const active = SIDES.some((side) => this.debouncers.get(handKey(displayId, side))?.deb.active === true);
    return [onsets, active];
  }

  /** Forget a person entirely (their display id was released). */
  drop(displayId: number): void {
    for (const side of SIDES) this.debouncers.delete(handKey(displayId, side));
    this.pending.delete(displayId);
    this.lastOnset.delete(displayId);
  }

  /** Hands currently being tracked (in first-seen order). */
  known(): [number, Side][] {
    return [...this.debouncers.values()].map(({ key }) => [key[0], key[1]]);
  }
}
