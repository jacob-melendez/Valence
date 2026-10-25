/**
 * Video perception: one camera frame in, one VideoObs out. Port of valence/perception/video.py `VideoPerceiver`.
 *
 * Per call (10 Hz): pose every time (PoseLandmarker lite + a simple tracker, standing in for YOLO-pose + ByteTrack);
 * gestures on even iterations (5 Hz); face landmarks on odd ones (5 Hz, the Python cadence) or, with `faceEveryFrame`
 * (createPerception's default), on every one; background lux at 1 Hz. Faces: one full-frame
 * FaceLandmarker pass, matched to people by the pose nose; anyone left unmatched gets a second try on a head crop.
 * Output is scalars, booleans and normalized boxes only; the frame belongs to the caller (closed right after), the
 * luminance and crop copies are wiped by the PixelReader, and nothing image-shaped is kept here. Display ids are
 * ephemeral; per-person baselines live in memory only and die with the track.
 *
 * Posture runs on One Euro-smoothed keypoints with motion on the shoulders only (smoothing.ts): BlazePose jitters
 * where YOLO does not. Hands, faces and head crops use the raw keypoints, as in Python.
 *
 * Browser additions: the emotion-v1 face cues (faceCues.ts) and the raw extras (52 blendshapes, head pose) per
 * person, held like face_valence for FACE_HOLD_S after the face pass that produced them.
 */
import { blazeposeToCoco, poseBoxPx } from "./coco";
import { CUE_CALIBRATION_S, FaceCueTracker, headPoseDeg, headRollDeg, NO_CUES, rawCues, type FaceCues } from "./faceCues";
import { assignFaces, blendshapeScores, FaceValenceCalibrator, noseTipPx, valenceRaw } from "./faceValence";
import { assignHands, FistTracker, isCurlFist, type Side } from "./hands";
import { DisplayIdMapper } from "./ids";
import { grayPixel } from "./luma";
import { backgroundStats, LuxEstimator, MAX_WIDTH } from "./lux";
import type { VisionModels } from "./models";
import { PostureTracker, type PostureState } from "./posture";
import { headBoxPx, normBbox } from "./regions";
import { KeypointSmoother } from "./smoothing";
import { IouTracker, type BoxPx } from "./tracker";
import type { BBox, FaceRaw, PersonObs, VideoObs } from "./types";

const NOSE = 0, L_EYE = 1, R_EYE = 2, L_WRIST = 9, R_WRIST = 10;
export const LUX_PERIOD_S = 1.0;
export const FACE_CROP_SCALE = 2.2;
export const FACE_CROP_PX = 256;
/** A processing pause this long restarts per-person baselines: the engine's break_absent_min (10 min). */
export const BREAK_GAP_S = 10 * 60;
/**
 * A face seen within the last FACE_HOLD_S still counts as visible. At the edge's cadence (faces every other frame at
 * 10 Hz) that is 2.25 face-pass intervals; when passes come slower than every 0.2 s the hold stretches to 2.25
 * measured intervals (`faceHoldS`), otherwise a face would drop out between passes (below ~4.5 fps at the Python
 * cadence). Identical to Python at 10 Hz and above.
 */
export const FACE_HOLD_S = 0.45;
export const FACE_HOLD_PASSES = 2.25;
/**
 * A longer gap between face passes is a pause (tab hidden, a stall), not the device's pace: it is not measured,
 * or a minute in a hidden tab would hold a face that has left for ~20 s.
 */
export const FACE_PASS_MAX_GAP_S = 2.0;
/** Hands whose wrist is in the bottom 20% of the frame are typing, not fists. */
export const KEYBOARD_ZONE = 0.8;

export type FrameImage = TexImageSource & CanvasImageSource;

export interface FrameInput {
  image: FrameImage;
  width: number;
  height: number;
}

/** Pixel access (capture.ts PixelReader): every copy it makes is wiped before it returns or on wipeCrop(). */
export interface PixelSource {
  lumaStrided(frame: FrameImage, w: number, h: number, stride: number,
    toGray: (r: number, g: number, b: number) => number): { gray: Uint8Array; w: number; h: number };
  cropTo(frame: FrameImage, box: readonly [number, number, number, number], size: number): FrameImage;
  wipeCrop(): void;
}

interface Det {
  trackId: number;
  boxPx: BoxPx;
  kxy: Float64Array; // 17 x 2 pixels, row-major
  kconf: Float64Array; // 17
}

interface FaceHit {
  valence: number;
  blendshapes: Record<string, number>;
  matrix: ArrayLike<number> | null;
  /**
   * Found on the head-crop retry. Its matrix is in the crop's own virtual camera, so yaw and pitch are off by the
   * crop's viewing angle (a median 2-3 degrees, p95 6, against the full frame on the same face): they are reported
   * as missing. Roll (in-plane) is unaffected and kept.
   */
  fromCrop: boolean;
}

/** Ephemeral per-display-id state: scalars and keypoint history only. */
class Person {
  /** Posture on One Euro-smoothed keypoints, motion on the shoulders only (smoothing.ts: BlazePose jitter). */
  posture = new PostureTracker({ motionPoints: [5, 6] });
  smoother = new KeypointSmoother();
  face = new FaceValenceCalibrator();
  cues = new FaceCueTracker(CUE_CALIBRATION_S, true); // crop-only calibration leaves yaw/pitch missing, not 0
  faceValence = NaN;
  faceSeenT = -Infinity;
  lastCues: FaceCues = { ...NO_CUES };
  blinkPending = 0;
  raw: FaceRaw | null = null;
}

export interface VisionPipelineOptions {
  /** Keep only the largest this many people (settings.max_tracks = 3). */
  maxPeople?: number;
  /** Ignore people shorter than this share of the frame height (settings.min_box_height = 0.25). */
  minPersonH?: number;
  /**
   * Run the face pass on every frame instead of every other one (the Python edge's cadence, this class's default;
   * createPerception turns it on). Doubles the face rate for the emotion cues' blink detector
   * (docs/EMOTION_FEATURES.md 1.3 wants >= 15 face frames/s), at the cost of one more model per even frame.
   */
  faceEveryFrame?: boolean;
}

const area = (b: BoxPx) => (b[2] - b[0]) * (b[3] - b[1]);

export class VisionPipeline {
  private readonly ids = new DisplayIdMapper();
  private readonly fists = new FistTracker();
  private readonly lux = new LuxEstimator();
  private readonly tracker = new IouTracker();
  private readonly people = new Map<number, Person>();
  private readonly maxPeople: number;
  private readonly minPersonH: number;
  private readonly faceEveryFrame: boolean;
  private iter = 0;
  private lastProcessT: number | null = null;
  private lastProcessWall = 0;
  private readonly lastTs = { pose: -1, faces: -1, gestures: -1 };
  private lastLuxT = -Infinity;
  private lastFacePassT: number | null = null;
  private facePassInterval = 0;
  private luxValue = NaN;
  private luxQ = 0.0;

  constructor(
    private readonly models: Pick<VisionModels, "pose" | "faces" | "faceCrop" | "gestures">,
    private readonly pixels: PixelSource,
    opts: VisionPipelineOptions = {},
  ) {
    this.maxPeople = opts.maxPeople ?? 3;
    this.minPersonH = opts.minPersonH ?? 0.25;
    this.faceEveryFrame = opts.faceEveryFrame ?? false;
  }

  process(frame: FrameInput, t: number, wall: number, seq: number, framesDecoded = 0): VideoObs {
    const { width: w, height: h } = frame;
    const it = this.iter++;
    // A pause in processing (tab hidden, camera muted) is missing data, not an absence: tracks reattach to their
    // seats by tracker id or by position (ids.ts) and keep their baselines. Only a pause as long as the engine's break
    // (break_absent_min) restarts the per-person baselines, since whoever sits there now may be someone else; the
    // display ids stay where they are. Wall time counts too: the engine's break clock runs on it, and the monotonic clock
    // stops while the computer sleeps.
    if (this.lastProcessT !== null
      && Math.max(t - this.lastProcessT, wall - this.lastProcessWall) >= BREAK_GAP_S) {
      for (const did of this.people.keys()) this.fists.drop(did);
      this.people.clear();
    }
    this.lastProcessT = t;
    this.lastProcessWall = wall;
    if (w > 0 && h > 0) this.ids.aspect = h / w;

    const allDets = this.track(frame, w, h, t);
    const dets = allDets
      .filter((d) => d.boxPx[3] - d.boxPx[1] >= this.minPersonH * h)
      .sort((a, b) => area(b.boxPx) - area(a.boxPx))
      .slice(0, this.maxPeople);
    const idMap = this.ids.update(dets.map((d) => [d.trackId, normBbox(d.boxPx, w, h)] as const), t);
    for (const gone of this.ids.popReleased()) {
      this.people.delete(gone);
      this.fists.drop(gone);
    }
    const byId = new Map<number, Det>();
    for (const d of dets) {
      const did = idMap.get(d.trackId);
      if (did !== undefined) byId.set(did, d);
    }
    const postures = new Map<number, PostureState>();
    for (const [did, d] of byId) {
      let p = this.people.get(did);
      if (!p) this.people.set(did, (p = new Person()));
      postures.set(did, p.posture.update(p.smoother.update(d.kxy, t), d.kconf, t));
    }

    if (it % 2 === 0) this.updateFists(frame, byId, w, h, t);
    if (it % 2 === 1 || this.faceEveryFrame) this.updateFaces(frame, byId, w, h, t);

    if (t - this.lastLuxT >= LUX_PERIOD_S) {
      this.lastLuxT = t;
      const boxesNorm = allDets.map((d) => normBbox(d.boxPx, w, h));
      const stride = Math.max(1, Math.ceil(w / MAX_WIDTH));
      const g = this.pixels.lumaStrided(frame.image, w, h, stride, grayPixel);
      [this.luxValue, this.luxQ] = this.lux.update(backgroundStats(g.gray, g.w, g.h, boxesNorm), t);
      g.gray.fill(0);
    }

    const persons: PersonObs[] = [];
    for (const [did, d] of [...byId].sort((a, b) => a[0] - b[0])) {
      const p = this.people.get(did)!;
      const ps = postures.get(did)!;
      const [onsets, active] = this.fists.personState(did);
      const visible = t - p.faceSeenT <= this.faceHoldS;
      if (!visible) p.cues.update(null, t * 1000); // a frame without a visible face resets the blink detector
      const cues = visible ? { ...p.lastCues, blink_onsets: p.blinkPending } : NO_CUES;
      p.blinkPending = 0;
      persons.push({
        track_id: did, bbox: normBbox(d.boxPx, w, h) as BBox, fist_onsets: onsets, fist_active: active,
        slump: ps.slump, head_in_hands: ps.headInHands, motion: ps.motion, face_visible: visible,
        face_valence: visible ? p.faceValence : NaN, ...cues, raw: visible ? p.raw : null,
      });
    }
    return {
      type: "video", t, wall, frame_seq: seq, frames_decoded: framesDecoded, lux: this.luxValue,
      lux_quality: this.luxQ, frame_w: w, frame_h: h, mirrored: false, persons,
    };
  }

  /** FACE_HOLD_S, stretched to FACE_HOLD_PASSES face-pass intervals when passes come slower than every 0.2 s. */
  get faceHoldS(): number {
    return Math.max(FACE_HOLD_S, FACE_HOLD_PASSES * this.facePassInterval);
  }

  /** VIDEO mode (pose, gestures) needs strictly increasing ms timestamps per task; IMAGE-mode faces ignore it. */
  private ts(task: keyof VisionPipeline["lastTs"], t: number): number {
    this.lastTs[task] = Math.max(Math.floor(t * 1000), this.lastTs[task] + 1);
    return this.lastTs[task];
  }

  private track(frame: FrameInput, w: number, h: number, t: number): Det[] {
    const res = this.models.pose(frame.image, this.ts("pose", t));
    const boxes: BoxPx[] = [];
    const poses: { kxy: Float64Array; kconf: Float64Array }[] = [];
    for (const lms of res.landmarks) {
      const box = poseBoxPx(lms, w, h);
      if (!box) continue;
      const coco = blazeposeToCoco(lms, w, h);
      boxes.push(box);
      poses.push({ kxy: coco.xy, kconf: coco.conf });
    }
    const trackIds = this.tracker.update(boxes, t);
    return boxes.map((boxPx, i) => ({ trackId: trackIds[i]!, boxPx, kxy: poses[i]!.kxy, kconf: poses[i]!.kconf }));
  }

  private updateFists(frame: FrameInput, byId: Map<number, Det>, w: number, h: number, t: number): void {
    const res = this.models.gestures(frame.image, this.ts("gestures", t));
    const wristsPx: [number, number][] = [];
    const scores: number[] = [];
    res.landmarks.forEach((lms, n) => {
      const wrist = lms[0];
      if (!wrist || wrist.y >= KEYBOARD_ZONE) return; // keyboard zone: never scored, so the hand counts as unobserved
      wristsPx.push([wrist.x * w, wrist.y * h]);
      const cats = res.gestures[n] ?? [];
      let fist = cats.find((c) => c.categoryName === "Closed_Fist")?.score;
      if (fist === undefined) fist = isCurlFist(lms.map((lm) => [lm.x, lm.y]), h / w) ? 1.0 : 0.0;
      scores.push(Number(fist));
    });
    const personWrists = new Map<number, Record<Side, [number, number, number]>>();
    for (const [did, d] of byId) {
      personWrists.set(did, {
        L: [d.kxy[2 * L_WRIST]!, d.kxy[2 * L_WRIST + 1]!, d.kconf[L_WRIST]!],
        R: [d.kxy[2 * R_WRIST]!, d.kxy[2 * R_WRIST + 1]!, d.kconf[R_WRIST]!],
      });
    }
    const seen: [number, Side][] = [];
    assignHands(wristsPx, personWrists, w).forEach((key, i) => {
      if (key === null) return;
      this.fists.update(key[0], key[1], scores[i]!, t);
      seen.push([key[0], key[1]]);
    });
    this.fists.markUnobserved(t, seen);
  }

  private updateFaces(frame: FrameInput, byId: Map<number, Det>, w: number, h: number, t: number): void {
    if (this.lastFacePassT !== null && t > this.lastFacePassT && t - this.lastFacePassT <= FACE_PASS_MAX_GAP_S) {
      const dt = t - this.lastFacePassT;
      this.facePassInterval = this.facePassInterval ? 0.8 * this.facePassInterval + 0.2 * dt : dt;
    }
    this.lastFacePassT = t;
    const res = this.models.faces(frame.image, this.ts("faces", t));
    const points = res.faceLandmarks.map((lms) => noseTipPx(lms, w, h));
    const hits: FaceHit[] = res.faceBlendshapes.map((bs, i) => ({
      valence: valenceRaw(blendshapeScores(bs.categories)),
      blendshapes: Object.fromEntries(bs.categories.map((c) => [c.categoryName, c.score])),
      matrix: res.facialTransformationMatrixes?.[i]?.data ?? null,
      fromCrop: false,
    }));
    const noses = new Map<number, [number, number, number]>();
    for (const [did, d] of byId) noses.set(did, VisionPipeline.nose(d));
    const matched = assignFaces(points, noses, w);
    const found = new Map<number, FaceHit>();
    matched.forEach((did, i) => {
      if (did !== null && i < hits.length) found.set(did, hits[i]!);
    });
    for (const [did, d] of byId) {
      if (found.has(did)) continue;
      const hit = this.faceFromCrop(frame, d, w, h);
      if (hit) found.set(did, hit);
    }
    for (const did of byId.keys()) {
      const p = this.people.get(did)!;
      const hit = found.get(did);
      if (hit) {
        p.faceValence = p.face.update(hit.valence, t);
        p.faceSeenT = t;
        const [yaw, pitch] = hit.fromCrop ? [NaN, NaN] : headPoseDeg(hit.matrix);
        p.raw = { blendshapes: hit.blendshapes, head_yaw_deg: yaw, head_pitch_deg: pitch, head_roll_deg: headRollDeg(hit.matrix) };
      }
      // Emotion cues (docs/EMOTION_FEATURES.md 1.2-1.3): a face without a valence reading counts as not visible. A miss
      // while the face is still held (face_visible stays true) is a held frame, like the fist frames: no update, so it
      // neither resets the blink detector nor breaks the visible-time clock. Frames past the hold reset in process().
      if (hit === undefined) continue;
      const readable = Number.isFinite(p.faceValence);
      const cues = p.cues.update(readable ? rawCues(hit.blendshapes, hit.fromCrop ? null : hit.matrix) : null, t * 1000);
      if (readable) {
        p.lastCues = cues;
        p.blinkPending = cues.blink_onsets;
      }
    }
  }

  private faceFromCrop(frame: FrameInput, d: Det, w: number, h: number): FaceHit | null {
    const box = headBoxPx(d.kxy, d.kconf, w, h, FACE_CROP_SCALE);
    if (box === null || box[2] - box[0] < 16 || box[3] - box[1] < 16) return null;
    const crop = this.pixels.cropTo(frame.image, box, FACE_CROP_PX);
    let res;
    try {
      res = this.models.faceCrop(crop);
    } finally {
      this.pixels.wipeCrop();
    }
    const bs = res.faceBlendshapes[0];
    if (!bs) return null;
    return {
      valence: valenceRaw(blendshapeScores(bs.categories)),
      blendshapes: Object.fromEntries(bs.categories.map((c) => [c.categoryName, c.score])),
      matrix: res.facialTransformationMatrixes?.[0]?.data ?? null,
      fromCrop: true,
    };
  }

  /** The pose nose, or the eyes' midpoint when the nose is unsure. */
  private static nose(d: Det): [number, number, number] {
    if (d.kconf[NOSE]! >= 0.3) return [d.kxy[0]!, d.kxy[1]!, d.kconf[NOSE]!];
    return [
      (d.kxy[2 * L_EYE]! + d.kxy[2 * R_EYE]!) / 2, (d.kxy[2 * L_EYE + 1]! + d.kxy[2 * R_EYE + 1]!) / 2,
      Math.min(d.kconf[L_EYE]!, d.kconf[R_EYE]!),
    ];
  }
}
