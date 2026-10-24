/**
 * Facial Surface layer: valence from FaceLandmarker blendshapes. Port of valence/perception/face_valence.py.
 *
 * Deliberately naive, as in Python: a smile reads as positive whether or not it is felt, and cheekSquint (the
 * Duchenne cue) is NOT used, so the face-only baseline cannot see through a masked smile.
 */
import { linearSumAssignment } from "./assign";

export const VALENCE_WEIGHTS: ReadonlyArray<readonly [string, number]> = [
  ["mouthSmile", 1.0],
  ["mouthFrown", -0.7],
  ["browDown", -0.5],
  ["noseSneer", -0.3],
  ["mouthPress", -0.2],
];
export const VALENCE_BLENDSHAPES: readonly string[] = VALENCE_WEIGHTS.flatMap(([stem]) => [
  `${stem}Left`, `${stem}Right`,
]);
export const NOSE_TIP_INDEX = 1;
export const CALIBRATION_S = 20.0;
export const GAIN = 2.5;
export const MAX_CALIBRATION_SAMPLES = 4096;

export interface Category {
  categoryName: string;
  score: number;
}

/** Keep only the scores valenceRaw needs. */
export function blendshapeScores(categories: Iterable<Category>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of categories) if (VALENCE_BLENDSHAPES.includes(c.categoryName)) out[c.categoryName] = Number(c.score);
  return out;
}

/** Pixel position of the nose-tip landmark. */
export function noseTipPx(landmarks: ArrayLike<{ x: number; y: number }>, frameW: number, frameH: number): [number, number] {
  const tip = landmarks[NOSE_TIP_INDEX]!;
  return [tip.x * frameW, tip.y * frameH];
}

function bilateral(bs: Readonly<Record<string, number>>, stem: string): number | null {
  const values: number[] = [];
  for (const k of [`${stem}Left`, `${stem}Right`]) {
    const v = bs[k];
    if (v !== undefined) values.push(Number(v));
  }
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

/**
 * smile - 0.7 frown - 0.5 browDown - 0.3 noseSneer - 0.2 mouthPress (each L/R-averaged). A missing side falls back
 * to the other, a missing category counts as 0, and a mapping with none of them (no face) yields NaN.
 */
export function valenceRaw(bs: Readonly<Record<string, number>>): number {
  let total = 0;
  let seen = false;
  for (const [stem, weight] of VALENCE_WEIGHTS) {
    const v = bilateral(bs, stem);
    if (v !== null) {
      total += weight * v;
      seen = true;
    }
  }
  return seen ? total : NaN;
}

/** statistics.median: the middle value, or the mean of the two middle values. */
export function median(values: readonly number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  if (!n) throw new Error("median of an empty list");
  const mid = n >> 1;
  return n % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/**
 * Per-person neutral baseline: valence relative to the median raw valence over that person's first
 * `calibrationS` seconds of visible face (running median while calibrating, frozen afterwards). Visible time accrues
 * only between consecutive visible samples, each gap capped at `maxGapS`; a non-finite `t` accrues nothing.
 * Non-finite input means "face not visible": NaN out, never enters the baseline.
 */
export class FaceValenceCalibrator {
  private samples: number[] = [];
  private visibleS = 0;
  private prevT: number | null = null;
  private prevVisible = false;
  private frozen: number | null = null;

  constructor(
    readonly calibrationS = CALIBRATION_S,
    readonly gain = GAIN,
    readonly maxGapS = 1.0,
  ) {}

  get calibrated(): boolean {
    return this.frozen !== null;
  }

  get baseline(): number {
    if (this.frozen !== null) return this.frozen;
    return this.samples.length ? median(this.samples) : 0.0;
  }

  get visibleSeconds(): number {
    return this.visibleS;
  }

  /** Feed one raw valence sample; returns clip(tanh(gain * (vRaw - baseline)), -1, 1). */
  update(vRaw: number, t: number): number {
    const visible = Number.isFinite(vRaw);
    if (!Number.isFinite(t)) {
      this.prevT = null;
    } else {
      if (visible && this.prevVisible && this.prevT !== null) {
        this.visibleS += Math.min(Math.max(t - this.prevT, 0.0), this.maxGapS);
      }
      this.prevT = t;
    }
    this.prevVisible = visible;
    if (!visible) return NaN;
    if (this.frozen === null) {
      if (this.visibleS < this.calibrationS && this.samples.length < MAX_CALIBRATION_SAMPLES) {
        this.samples.push(vRaw);
      } else {
        this.frozen = this.baseline;
        this.samples = [];
      }
    }
    return Math.max(-1.0, Math.min(1.0, Math.tanh(this.gain * (vRaw - this.baseline))));
  }
}

/**
 * Match each face (nose-tip landmark, px) to a tracked person (pose nose, px): optimal one-to-one assignment on
 * Euclidean distance, gated at `maxDistFrac * frameW`. Gated pairs cost more than any full set of valid pairs, so
 * the solver first maximises the number of valid matches, then minimises distance. Person noses below `minConf`
 * are ignored. Returns a person id or null per face.
 */
export function assignFaces(
  facePointsPx: ReadonlyArray<readonly [number, number]>,
  personNosesPx: ReadonlyMap<number, readonly [number, number, number]>,
  frameW: number,
  maxDistFrac = 0.15,
  minConf = 0.3,
): (number | null)[] {
  const out: (number | null)[] = facePointsPx.map(() => null);
  const pids = [...personNosesPx.entries()]
    .filter(([, [x, y, conf]]) => conf >= minConf && Number.isFinite(x) && Number.isFinite(y))
    .map(([pid]) => pid);
  if (!out.length || !pids.length) return out;
  const gate = maxDistFrac * frameW;
  const dist = facePointsPx.map(([fx, fy]) => pids.map((pid) => {
    const [nx, ny] = personNosesPx.get(pid)!;
    return Math.hypot(fx - nx, fy - ny);
  }));
  const valid = dist.map((row) => row.map((d) => Number.isFinite(d) && d <= gate));
  if (!valid.some((row) => row.some(Boolean))) return out;
  const blocked = gate * (Math.min(dist.length, pids.length) + 1) + 1.0;
  const cost = dist.map((row, r) => row.map((d, c) => (valid[r]![c] ? d : blocked)));
  const [rows, cols] = linearSumAssignment(cost);
  rows.forEach((r, i) => {
    const c = cols[i]!;
    if (valid[r]![c]) out[r] = pids[c]!;
  });
  return out;
}
