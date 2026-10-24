/**
 * Hysteresis debouncer: noisy per-sample scores -> stable state + counted onsets. Port of
 * valence/perception/debounce.py, shared by fist detection (per person x hand) and laughter detection.
 */

/** Absorbs float error in timestamp differences (2.3 - 1.3 < 1.0). */
export const TIME_EPS = 1e-9;

/**
 * Enter when `score >= on` for `onCount` consecutive samples spanning at least `onHoldS` seconds (first to last high
 * sample); exit once `score < off` has held for `offHoldS` seconds (0 exits on the first low sample). A rising edge
 * counts as an onset only if `refractoryS` has passed since the previous counted onset. Scores between `off` and
 * `on` keep the current state (and break an entry streak). NaN means "not observed": it breaks an entry streak and
 * counts as low for exit.
 */
export class HysteresisDebouncer {
  active = false;
  /** Cumulative counted rising edges. */
  onsets = 0;
  private streak = 0;
  private highSince: number | null = null;
  private lowSince: number | null = null;
  private lastOnsetT = -Infinity;

  constructor(
    readonly on: number,
    readonly off: number,
    readonly onCount = 1,
    readonly offHoldS = 0.0,
    readonly refractoryS = 0.0,
    readonly onHoldS = 0.0,
  ) {
    if (!(off <= on)) throw new Error("off threshold must be <= on threshold");
  }

  reset(): void {
    this.active = false;
    this.onsets = 0;
    this.streak = 0;
    this.highSince = null;
    this.lowSince = null;
    this.lastOnsetT = -Infinity;
  }

  /** Feed one sample; returns true iff an onset was counted at this sample. */
  update(score: number, t: number): boolean {
    const observed = !Number.isNaN(score);
    if (!this.active) {
      if (observed && score >= this.on) {
        this.streak += 1;
        if (this.highSince === null) this.highSince = t;
      } else {
        this.streak = 0;
        this.highSince = null;
      }
      if (this.streak < this.onCount || t - this.highSince! < this.onHoldS - TIME_EPS) return false;
      this.active = true;
      this.streak = 0;
      this.highSince = null;
      this.lowSince = null;
      if (t - this.lastOnsetT >= this.refractoryS - TIME_EPS) {
        this.lastOnsetT = t;
        this.onsets += 1;
        return true;
      }
      return false;
    }
    if (observed && score >= this.off) {
      this.lowSince = null;
      return false;
    }
    if (this.lowSince === null) this.lowSince = t;
    if (t - this.lowSince >= this.offHoldS - TIME_EPS) {
      this.active = false;
      this.lowSince = null;
    }
    return false;
  }
}

/** Closed_Fist: enter at >= 0.7 held >= 0.5 s, exit < 0.4 for 1 s, 3 s refractory. */
export const fistDebouncer = () => new HysteresisDebouncer(0.7, 0.4, 1, 1.0, 3.0, 0.5);

/** YAMNet laughter family: on >= 0.05 for 1 result, off below 0.02 held 1.5 s, 5.5 s refractory. */
export const laughterDebouncer = () => new HysteresisDebouncer(0.05, 0.02, 1, 1.5, 5.5);
