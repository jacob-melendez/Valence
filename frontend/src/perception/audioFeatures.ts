/**
 * Audio scalars for the Ambient Physics and Somatic & Social layers. Port of valence/perception/audio_features.py.
 *
 * Callers reduce each raw block to a mean power and YAMNet allowlist scores, then purge it. Loudness is smoothed in
 * the power domain and converted to dB last: averaging dB values would understate bursts of sound.
 *
 * Not ported: the compliment counter (COMPLIMENT_PHRASES, vosk_grammar, count_compliments) and its near-field check
 * (PhraseLoudness). The browser has no keyword recognizer, so compliment counts stay 0. resample_48k_to_16k lives
 * in ./resample.
 */
import { HysteresisDebouncer, laughterDebouncer } from "./debounce";

export const POWER_FLOOR = 1e-12;
export const YAMNET_ALLOWLIST: readonly string[] = [
  "Speech", "Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
];
export const LAUGH_FAMILY: readonly string[] = YAMNET_ALLOWLIST.slice(1);

/**
 * numpy's pairwise summation (np.add.reduce over a contiguous float64 array) of a[lo .. lo + n - 1], or of their
 * squares when `square` is set (np.sum(np.square(a, dtype=np.float64)) without the temporary array).
 */
export function pairwiseSum(a: ArrayLike<number>, lo = 0, n = a.length, square = false): number {
  if (n < 8) {
    let res = -0.0;
    for (let i = lo; i < lo + n; i++) res += square ? a[i]! * a[i]! : a[i]!;
    return res;
  }
  if (n <= 128) {
    const r = new Float64Array(8);
    for (let j = 0; j < 8; j++) r[j] = square ? a[lo + j]! * a[lo + j]! : a[lo + j]!;
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let j = 0; j < 8; j++) r[j]! += square ? a[lo + i + j]! * a[lo + i + j]! : a[lo + i + j]!;
    }
    let res = ((r[0]! + r[1]!) + (r[2]! + r[3]!)) + ((r[4]! + r[5]!) + (r[6]! + r[7]!));
    for (; i < n; i++) res += square ? a[lo + i]! * a[lo + i]! : a[lo + i]!;
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2, square) + pairwiseSum(a, lo + n2, n - n2, square);
}

/** np.iinfo(dtype).max + 1 for integer typed arrays (integer PCM full scale); null for float data. */
function integerFullScale(x: ArrayLike<number>): number | null {
  if (x instanceof Int16Array) return 32768;
  if (x instanceof Int8Array) return 128;
  if (x instanceof Uint8Array || x instanceof Uint8ClampedArray) return 256;
  if (x instanceof Uint16Array) return 65536;
  if (x instanceof Int32Array) return 2147483648;
  if (x instanceof Uint32Array) return 4294967296;
  return null;
}

/**
 * float64 mean of squares; integer typed arrays (PCM) are scaled to [-1, 1). NaN for an empty block. Plain arrays
 * and Float32Array / Float64Array are float data.
 */
export function meanPower(x: ArrayLike<number>): number {
  const n = x.length;
  if (n === 0) return NaN;
  let p = (0.0 + pairwiseSum(x, 0, n, true)) / n;
  const full = integerFullScale(x);
  if (full !== null) p /= full * full;
  return p;
}

/** 10 log10(max(p, 1e-12)) + offset (offset = spl_offset_db gives dB SPL). */
export function powerToDb(p: number, offset = 0.0): number {
  if (Number.isNaN(p)) return NaN;
  return 10.0 * Math.log10(Math.max(p, POWER_FLOOR)) + offset;
}

/** Exponential moving average of mean power (never of dB). Non-finite or negative input is skipped. */
export class PowerSmoother {
  constructor(
    public alpha = 0.3,
    public value = NaN,
  ) {
    if (!(0.0 < alpha && alpha <= 1.0)) throw new Error("alpha must be in (0, 1]");
  }

  update(p: number): number {
    if (Number.isFinite(p) && p >= 0.0) {
      this.value = Number.isNaN(this.value) ? p : this.alpha * p + (1.0 - this.alpha) * this.value;
    }
    return this.value;
  }

  db(offset = 0.0): number {
    return powerToDb(this.value, offset);
  }

  reset(): void {
    this.value = NaN;
  }
}

/** YAMNet class name -> score; a missing class counts as 0. */
export type YamnetScores = Readonly<Record<string, number>>;

/** Max score over LAUGH_FAMILY (missing = 0, NaN ignored; first maximum kept, like Python's max); NaN if all NaN. */
export function familyMax(scores: YamnetScores): number {
  let best = NaN;
  for (const k of LAUGH_FAMILY) {
    const s = scores[k];
    const v = s === undefined ? 0.0 : Number(s);
    if (Number.isNaN(v)) continue;
    if (Number.isNaN(best) || v > best) best = v;
  }
  return best;
}

/**
 * YAMNet allowlist scores -> [laugh, speech, onset] with debounced laughter onsets.
 *
 * laugh = max over LAUGH_FAMILY; a missing class counts as 0. Onsets accumulate until `takeOnsets` hands out the
 * delta since its previous call.
 */
export class LaughterTracker {
  private readonly debouncer: HysteresisDebouncer;
  private pending = 0;

  constructor(debouncer: HysteresisDebouncer | null = null) {
    this.debouncer = debouncer ?? laughterDebouncer();
  }

  get active(): boolean {
    return this.debouncer.active;
  }

  update(scores: YamnetScores, t: number): [number, number, boolean] {
    const laugh = familyMax(scores);
    const s = scores["Speech"];
    const speech = s === undefined ? 0.0 : Number(s);
    const onset = this.debouncer.update(laugh, t);
    if (onset) this.pending += 1;
    return [laugh, speech, onset];
  }

  takeOnsets(): number {
    const n = this.pending;
    this.pending = 0;
    return n;
  }
}
