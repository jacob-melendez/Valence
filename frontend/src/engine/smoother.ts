/**
 * Per-person smoothing of the served predictions (valence/models/predictor.py:49 PersonSmoother).
 *
 * The predictor is stateless (one window in, one calibrated distribution out); the live display needs three
 * stateful rules on top: EMA + 2-tick hysteresis on the shown *label* (the shown confidence stays the
 * calibrated, unsmoothed probability), the 3-of-6 persistence rule for the mask override, and the warm-up
 * blocks (no label claims before the window holds warmup_s, no override before override_block_s).
 *
 * Parity: the EMA is plain elementwise double arithmetic, as numpy does it, so it is bit-identical; argmax
 * follows numpy (first maximum, a NaN wins). Golden values: src/test/fixtures/engine/smoother.json.gz.
 */
import type { Settings } from "./contracts";
import type { PredictionOutput } from "./served";
import { isnan } from "./num";

// ---------- Types ----------

/** valence/models/predictor.py:43 SmoothedPrediction. */
export interface SmoothedPrediction {
  label: string;
  mask_override: boolean;
  warming_up: boolean;
}

/**
 * PersonSmoother's keyword arguments, named and defaulted as in Python (valence/models/predictor.py:50), so
 * engine.py's `PersonSmoother(states, s.tick_s, s.warmup_s, s.override_block_s)` ports directly.
 */
export interface PersonSmootherOptions {
  tick_s?: number;
  warmup_s?: number;
  override_block_s?: number;
  /** EMA weight of the newest distribution. */
  alpha?: number;
  /** Ticks a new top state must persist before the label switches. */
  hold?: number;
  /** Override switches on at >= this many hits in the window ... */
  override_on?: number;
  /** ... and off at <= this many. */
  override_off?: number;
  /** Face-visible ticks the override rule looks back over (deque maxlen). */
  override_window?: number;
}

/** The parts of a PredictionOutput the smoother reads. */
export type SmootherInput = Pick<PredictionOutput, "label" | "raw_override"> & {
  probabilities: Readonly<Record<string, number>> | ReadonlyMap<string, number>;
};

// ---------- Helpers ----------

/** Python `int(x)` for a float: truncation; NaN and +-inf raise. */
function pyInt(x: number): number {
  if (isnan(x)) throw new RangeError("cannot convert float NaN to integer");
  if (!Number.isFinite(x)) throw new RangeError("cannot convert float infinity to integer");
  return Math.trunc(x) + 0; // + 0 turns a -0 into 0 (Python ints have no sign of zero)
}

/** Python `a / b` on floats: division by zero raises. */
function pyDiv(a: number, b: number): number {
  if (b === 0) throw new RangeError("float division by zero");
  return a / b;
}

/** `probabilities.get(state, 0.0)`: own keys only, so "constructor" and friends are absent. */
function probOf(probs: SmootherInput["probabilities"], state: string): number {
  if (probs instanceof Map) return probs.has(state) ? (probs.get(state) as number) : 0.0;
  const rec = probs as Readonly<Record<string, number>>;
  return Object.hasOwn(rec, state) ? (rec[state] as number) : 0.0;
}

/** `int(np.argmax(a))`: the first maximum, or the first NaN if there is one. */
function argmax(a: Float64Array): number {
  if (a.length === 0) throw new RangeError("attempt to get argmax of an empty sequence");
  let k = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (isnan(v)) return i;
    if (v > (a[k] as number)) k = i;
  }
  return k;
}

// ---------- PersonSmoother ----------

/** valence/models/predictor.py:49 PersonSmoother. Public fields mirror the Python attributes. */
export class PersonSmoother {
  readonly states: readonly string[];
  readonly tickS: number;
  readonly warmupTicks: number;
  readonly overrideBlockTicks: number;
  readonly alpha: number;
  readonly hold: number;
  readonly overrideOnN: number;
  readonly overrideOffN: number;
  /** deque maxlen of overrideHist. */
  readonly overrideWindow: number;
  ema: Float64Array | null = null;
  label: string | null = null;
  candidate: string | null = null;
  candidateN = 0;
  /** Last `overrideWindow` face-visible ticks: did the override condition hold for the shown label? */
  overrideHist: boolean[] = [];
  overrideOn = false;

  constructor(states: readonly string[], opts: PersonSmootherOptions = {}) {
    const tickS = opts.tick_s ?? 5.0;
    this.states = states;
    this.tickS = tickS;
    this.warmupTicks = pyInt(pyDiv(opts.warmup_s ?? 60.0, tickS));
    this.overrideBlockTicks = pyInt(pyDiv(opts.override_block_s ?? 120.0, tickS));
    this.alpha = opts.alpha ?? 0.5;
    this.hold = opts.hold ?? 2;
    this.overrideOnN = opts.override_on ?? 3;
    this.overrideOffN = opts.override_off ?? 1;
    const win = opts.override_window ?? 6;
    // deque(maxlen=...) rejects these at construction
    if (!Number.isInteger(win)) throw new TypeError("an integer is required");
    if (win < 0) throw new RangeError("maxlen must be non-negative");
    this.overrideWindow = win;
  }

  /**
   * The smoother valence/runtime/engine.py:377 builds per person:
   * `PersonSmoother(states, s.tick_s, s.warmup_s, s.override_block_s)`.
   */
  static fromSettings(
    states: readonly string[],
    s: Pick<Settings, "tick_s" | "warmup_s" | "override_block_s">,
  ): PersonSmoother {
    return new PersonSmoother(states, {
      tick_s: s.tick_s,
      warmup_s: s.warmup_s,
      override_block_s: s.override_block_s,
    });
  }

  /**
   * One tick (valence/models/predictor.py:68 update). `windowTicks` is the person's window length in ticks
   * (len(person) in engine.py), which gates the warm-up and the override block.
   */
  update(out: SmootherInput, windowTicks: number): SmoothedPrediction {
    const p = Float64Array.from(this.states, (s) => probOf(out.probabilities, s));
    const prev = this.ema;
    const a = this.alpha;
    // alpha * p + (1 - alpha) * ema, elementwise (numpy: two multiplies, one add, no fused multiply-add)
    this.ema = prev === null ? p : p.map((v, i) => a * v + (1 - a) * (prev[i] as number));
    const top = this.states[argmax(this.ema)] as string;
    if (this.label === null) {
      this.label = top;
    } else if (top !== this.label) {
      this.candidateN = top === this.candidate ? this.candidateN + 1 : 1;
      this.candidate = top;
      if (this.candidateN >= this.hold) {
        this.label = top;
        this.candidate = null;
        this.candidateN = 0;
        this.overrideHist = []; // an override belongs to the label it was raised for
        this.overrideOn = false;
      }
    } else {
      this.candidate = null;
      this.candidateN = 0;
    }
    // Override hysteresis: on when the condition held on >= 3 of the last 6 face-visible ticks, off at <= 1.
    // A tick without a readable face holds the current state instead of resetting it.
    if (out.raw_override !== null) {
      this.overrideHist.push(Boolean(out.raw_override && out.label === this.label));
      while (this.overrideHist.length > this.overrideWindow) this.overrideHist.shift();
      const hits = this.overrideHist.reduce((n, h) => n + (h ? 1 : 0), 0);
      if (hits >= this.overrideOnN) this.overrideOn = true;
      else if (hits <= this.overrideOffN) this.overrideOn = false;
    }
    const override = this.overrideOn && windowTicks >= this.overrideBlockTicks;
    return {
      label: this.label,
      mask_override: override,
      warming_up: windowTicks < this.warmupTicks,
    };
  }
}
