/**
 * Emotion-map face cues per frame (docs/EMOTION_FEATURES.md section 1). Port of valence/features/face_cues.py, the
 * reference for parity: raw channels from the blendshapes and the facial transformation matrix, a per-person
 * neutral baseline (first 20 s of visible face) and a blink detector. Only scalars; landmarks are never touched.
 */
import { pyMax, pyMin, SortedSamples } from "./faceValence";

export const CUE_CALIBRATION_S = 20.0;
export const MAX_GAP_S = 1.0;
export const BLINK_CLOSE = 0.3;
export const BLINK_OPEN = 0.15;
export const BLINK_MAX_MS = 600.0;
export const RAW_CHANNELS = ["brow", "closure", "frown", "smile", "yaw", "pitch"] as const;
export type RawChannel = (typeof RAW_CHANNELS)[number];
export type RawCues = Record<RawChannel, number>;

export interface FaceCues {
  brow_down: number;
  eye_open: number;
  mouth_frown: number;
  smile: number;
  head_yaw: number;
  head_pitch: number;
  blink_onsets: number;
}

export const NO_CUES: Readonly<FaceCues> = Object.freeze({
  brow_down: NaN, eye_open: NaN, mouth_frown: NaN, smile: NaN, head_yaw: NaN, head_pitch: NaN, blink_onsets: 0,
});

type Blendshapes = Readonly<Record<string, number | null | undefined>>;

function avg(b: Blendshapes, stem: string): number {
  const vals: number[] = [];
  for (const k of [`${stem}Left`, `${stem}Right`]) {
    const v = b[k];
    if (v !== undefined && v !== null && !Number.isNaN(Number(v))) vals.push(Number(v)); // a NaN side is missing
  }
  return vals.length ? vals.reduce((a, c) => a + c, 0) / vals.length : NaN;
}

/** math.degrees: one multiplication by 180/pi (CPython's radToDeg), not (rad * 180) / pi, which rounds differently. */
const deg = (rad: number) => rad * (180.0 / Math.PI);

/** Column-major (MediaPipe's packing) unless the translation sits in the last row. */
const columnMajor = (d: ArrayLike<number>) => Math.abs(d[14]!) >= Math.abs(d[11]!);

/**
 * (yaw, pitch) in degrees from the 16 numbers of the facial transformation matrix. yaw > 0: nose toward the image's
 * right; pitch > 0: face tilted up.
 */
export function headPoseDeg(data: ArrayLike<number> | null | undefined): [number, number] {
  if (!data || data.length !== 16) return [NaN, NaN];
  const [fx, fy, fz] = columnMajor(data) ? [data[8]!, data[9]!, data[10]!] : [data[2]!, data[6]!, data[10]!];
  return [deg(Math.atan2(fx, fz)), deg(Math.atan2(fy, Math.hypot(fx, fz)))];
}

/**
 * Roll in degrees (browser extra, not in the Python reference): the angle of the face's up axis (the matrix's second
 * column) from the image's vertical; > 0 when the head's top tilts toward the image's right (clockwise).
 */
export function headRollDeg(data: ArrayLike<number> | null | undefined): number {
  if (!data || data.length !== 16) return NaN;
  const [ux, uy] = columnMajor(data) ? [data[4]!, data[5]!] : [data[1]!, data[5]!];
  return deg(Math.atan2(ux, uy));
}

/** Raw (not yet neutral-relative) channels for one face. */
export function rawCues(blendshapes: Blendshapes, matrix: ArrayLike<number> | null | undefined): RawCues {
  const [yaw, pitch] = matrix ? headPoseDeg(matrix) : [NaN, NaN];
  return {
    brow: avg(blendshapes, "browDown"), closure: avg(blendshapes, "eyeBlink"), frown: avg(blendshapes, "mouthFrown"),
    smile: avg(blendshapes, "mouthSmile"), yaw, pitch,
  };
}

const clip = (v: number, lo: number, hi: number) => (Number.isNaN(v) ? v : pyMax(lo, pyMin(hi, v)));

/**
 * Python's float `a % b` as CPython computes it: fmod, then add `b` once if the signs differ (one rounding, so a tiny
 * negative remainder becomes `b` itself, e.g. -2.8e-14 % 360 == 360.0), and a zero remainder takes the sign of `b`.
 */
function pyMod(a: number, b: number): number {
  const mod = a % b;
  if (mod !== 0) return b < 0 !== mod < 0 ? mod + b : mod; // signs differ: add b once
  return b < 0 ? -0 : 0;
}

/** Python's (v + 180) % 360 - 180. */
const wrapDeg = (v: number) => (Number.isNaN(v) ? v : pyMod(v + 180.0, 360.0) - 180.0);

const emptySamples = (): Record<RawChannel, SortedSamples> => ({
  brow: new SortedSamples(), closure: new SortedSamples(), frown: new SortedSamples(), smile: new SortedSamples(),
  yaw: new SortedSamples(), pitch: new SortedSamples(),
});

/** Neutral baseline (first 20 s of visible face, running median, then frozen) and the blink detector. */
export class FaceCueTracker {
  private samples = emptySamples();
  private frozen: RawCues | null = null;
  private visibleS = 0;
  private prevT: number | null = null;
  private prevVisible = false;
  private closedAt: number | null = null;

  constructor(readonly calibrationS = CUE_CALIBRATION_S) {}

  baseline(): RawCues {
    if (this.frozen) return this.frozen;
    const out = {} as RawCues;
    for (const k of RAW_CHANNELS) out[k] = this.samples[k].length ? this.samples[k].median() : 0.0;
    return out;
  }

  /** One frame. `raw` is null (or all NaN) when no face is visible. Returns the PersonObs cues. */
  update(raw: RawCues | null, tMs: number): FaceCues {
    const visible = raw !== null && !RAW_CHANNELS.every((k) => Number.isNaN(raw[k]));
    const t = tMs / 1000.0;
    if (visible && this.prevVisible && this.prevT !== null) {
      this.visibleS += pyMin(pyMax(t - this.prevT, 0.0), MAX_GAP_S);
    }
    this.prevT = t;
    this.prevVisible = visible;
    if (!visible || raw === null) {
      this.closedAt = null;
      return { ...NO_CUES };
    }
    if (this.frozen === null) {
      if (this.visibleS < this.calibrationS) {
        for (const k of RAW_CHANNELS) if (!Number.isNaN(raw[k])) this.samples[k].push(raw[k]);
      } else {
        this.frozen = this.baseline();
        this.samples = emptySamples();
      }
    }
    const b = this.baseline();
    let blink = 0;
    const c = raw.closure;
    if (!Number.isNaN(c)) {
      if (this.closedAt === null && c >= b.closure + BLINK_CLOSE) {
        this.closedAt = tMs;
      } else if (this.closedAt !== null && c <= b.closure + BLINK_OPEN) {
        blink = tMs - this.closedAt <= BLINK_MAX_MS ? 1 : 0;
        this.closedAt = null;
      }
    }
    return {
      brow_down: clip(raw.brow - b.brow, -1.0, 1.0),
      eye_open: clip(b.closure - c, -1.0, 1.0),
      mouth_frown: clip(raw.frown - b.frown, -1.0, 1.0),
      smile: clip(raw.smile - b.smile, -1.0, 1.0),
      head_yaw: clip(wrapDeg(raw.yaw - b.yaw), -90.0, 90.0),
      head_pitch: clip(raw.pitch - b.pitch, -90.0, 90.0),
      blink_onsets: blink,
    };
  }
}
