/**
 * Keypoint smoothing for the posture signals (browser only; the Python edge has no counterpart because YOLO-pose
 * does not need it).
 *
 * BlazePose keypoints carry frame-to-frame white noise that YOLO's do not: on identical frames (a still portrait with
 * sensor noise) the head and shoulders jitter at 0.11-0.22 shoulder widths/s against YOLO's 0.004-0.02, and
 * PostureTracker's motion reads 0.46 sw/s against 0.026. A One Euro filter per keypoint (min cutoff 0.2 Hz, beta 8
 * per shoulder width/s, derivative cutoff 0.5 Hz), with motion on the shoulders only, brought motion to 1.3x YOLO on
 * still and drifting scenes while keeping 0.93x of real 0.1-0.3 sw/s motion, and left slump onsets within 0.1 s
 * (side-by-side run of both models through the real Python PostureTracker; docs/hackohio/status-C.md).
 *
 * The cutoff rises with the keypoint's smoothed speed in shoulder widths per second (the raw shoulder distance,
 * EMA 0.3), so beta does not depend on the distance to the camera. The first sample passes through; a keypoint that
 * is not finite holds its state (its output is NaN); one filter per tracked person, reset with the display id.
 * Only coordinates are filtered; confidences pass unchanged. Only scalars are kept.
 */

const N = 17;
const L_SHOULDER = 5;
const R_SHOULDER = 6;

/** Exponential smoothing factor for a first-order low-pass at cutoff `fc` Hz over `dt` seconds. */
export function oneEuroAlpha(dt: number, fc: number): number {
  const tau = 1.0 / (2 * Math.PI * fc);
  return 1.0 / (1.0 + tau / dt);
}

export class KeypointSmoother {
  private xPrev = new Float64Array(2 * N).fill(NaN);
  private dxPrev = new Float64Array(2 * N);
  private tPrev: number | null = null;
  private sw = NaN;

  constructor(
    readonly minCutoff = 0.2,
    readonly beta = 8.0,
    readonly dCutoff = 0.5,
  ) {}

  /** Filter one frame of 17 keypoints (34 numbers, x/y interleaved, pixels) taken at `t` seconds. */
  update(xy: ArrayLike<number>, t: number): Float64Array {
    const out = new Float64Array(2 * N);
    const w = Math.hypot(xy[2 * L_SHOULDER]! - xy[2 * R_SHOULDER]!, xy[2 * L_SHOULDER + 1]! - xy[2 * R_SHOULDER + 1]!);
    if (Number.isFinite(w) && w > 0) this.sw = Number.isNaN(this.sw) ? w : 0.3 * w + 0.7 * this.sw;
    const finite = (k: number) => Number.isFinite(xy[2 * k]!) && Number.isFinite(xy[2 * k + 1]!);

    if (this.tPrev === null || !this.xPrev.every(Number.isFinite)) {
      for (let k = 0; k < N; k++) {
        out[2 * k] = xy[2 * k]!;
        out[2 * k + 1] = xy[2 * k + 1]!;
        if (finite(k)) {
          this.xPrev[2 * k] = xy[2 * k]!;
          this.xPrev[2 * k + 1] = xy[2 * k + 1]!;
        }
      }
      this.tPrev = t;
      return out;
    }
    const dt = t - this.tPrev;
    if (!(dt > 0)) {
      // A repeated or backwards timestamp: report the current estimate, change nothing.
      for (let k = 0; k < N; k++) {
        out[2 * k] = finite(k) ? this.xPrev[2 * k]! : NaN;
        out[2 * k + 1] = finite(k) ? this.xPrev[2 * k + 1]! : NaN;
      }
      return out;
    }
    const aD = oneEuroAlpha(dt, this.dCutoff);
    const scale = Number.isFinite(this.sw) && this.sw > 0 ? this.sw : 1.0;
    for (let k = 0; k < N; k++) {
      if (!finite(k)) {
        out[2 * k] = out[2 * k + 1] = NaN; // state held
        continue;
      }
      const dx = aD * ((xy[2 * k]! - this.xPrev[2 * k]!) / dt) + (1 - aD) * this.dxPrev[2 * k]!;
      const dy = aD * ((xy[2 * k + 1]! - this.xPrev[2 * k + 1]!) / dt) + (1 - aD) * this.dxPrev[2 * k + 1]!;
      const a = oneEuroAlpha(dt, this.minCutoff + this.beta * (Math.hypot(dx, dy) / scale));
      const x = a * xy[2 * k]! + (1 - a) * this.xPrev[2 * k]!;
      const y = a * xy[2 * k + 1]! + (1 - a) * this.xPrev[2 * k + 1]!;
      out[2 * k] = this.xPrev[2 * k] = x;
      out[2 * k + 1] = this.xPrev[2 * k + 1] = y;
      this.dxPrev[2 * k] = dx;
      this.dxPrev[2 * k + 1] = dy;
    }
    this.tPrev = t;
    return out;
  }
}
