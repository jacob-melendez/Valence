/**
 * BlazePose (MediaPipe PoseLandmarker, 33 landmarks) -> the COCO-17 keypoints the Python posture, hands, face
 * assignment and head-box code expects from YOLO-pose, in pixels, with a confidence per keypoint.
 *
 * Confidence is BlazePose's `visibility` (the likelihood that the landmark is visible in the image), which plays the
 * role of YOLO's keypoint confidence for the 0.3 / 0.5 / 0.6 thresholds. A landmark outside the frame gets
 * confidence 0: YOLO never places a confident keypoint outside the image, while BlazePose extrapolates the whole
 * body (hips and knees below a desk-height frame).
 */

/** COCO index i <- BlazePose landmark BLAZEPOSE_TO_COCO[i]. */
export const BLAZEPOSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] as const;
export const N_KPTS = 17;

export interface Landmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export interface CocoPose {
  /** 17 x 2 pixel coordinates, row-major: [x0, y0, x1, y1, ...]. */
  xy: Float64Array;
  /** 17 confidences in [0, 1]. */
  conf: Float64Array;
}

const inFrame = (v: number) => Number.isFinite(v) && v >= 0 && v <= 1;

export function blazeposeToCoco(landmarks: ArrayLike<Landmark>, frameW: number, frameH: number): CocoPose {
  const xy = new Float64Array(N_KPTS * 2);
  const conf = new Float64Array(N_KPTS);
  BLAZEPOSE_TO_COCO.forEach((src, i) => {
    const lm = landmarks[src];
    if (!lm) {
      xy[2 * i] = xy[2 * i + 1] = NaN;
      return;
    }
    xy[2 * i] = lm.x * frameW;
    xy[2 * i + 1] = lm.y * frameH;
    const vis = Number.isFinite(lm.visibility) ? lm.visibility! : 0;
    conf[i] = inFrame(lm.x) && inFrame(lm.y) ? Math.min(1, Math.max(0, vis)) : 0;
  });
  return { xy, conf };
}

/**
 * Person box in pixels (x1, y1, x2, y2) from all 33 landmarks, standing in for YOLO's person box. BlazePose
 * extrapolates hidden and off-frame parts, so the extent reaches the frame edge for a person cut by it (as YOLO's
 * box does); the top is raised by half the eyes-to-shoulders height because the highest landmark (an eye or ear)
 * sits well below the top of the head. Clipped to the frame; null without finite landmarks.
 */
export function poseBoxPx(landmarks: ArrayLike<Landmark>, frameW: number, frameH: number): [number, number, number, number] | null {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i]!;
    if (!Number.isFinite(lm.x) || !Number.isFinite(lm.y)) continue;
    x1 = Math.min(x1, lm.x);
    x2 = Math.max(x2, lm.x);
    y1 = Math.min(y1, lm.y);
    y2 = Math.max(y2, lm.y);
  }
  if (!Number.isFinite(x1)) return null;
  const eyeY = Math.min(landmarks[2]?.y ?? NaN, landmarks[5]?.y ?? NaN);
  const shoulderY = ((landmarks[11]?.y ?? NaN) + (landmarks[12]?.y ?? NaN)) / 2;
  const headroom = Number.isFinite(eyeY) && Number.isFinite(shoulderY) ? Math.max(0, shoulderY - eyeY) * 0.5 : 0;
  y1 -= headroom;
  const clip = (v: number, hi: number) => Math.min(hi, Math.max(0, v));
  const box: [number, number, number, number] = [
    clip(x1 * frameW, frameW), clip(y1 * frameH, frameH), clip(x2 * frameW, frameW), clip(y2 * frameH, frameH),
  ];
  return box[2] > box[0] && box[3] > box[1] ? box : null;
}
