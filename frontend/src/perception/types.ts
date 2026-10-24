/**
 * Browser perception contracts.
 *
 * PersonObs, VideoObs and AudioObs mirror valence/contracts.py field for field: same snake_case names, units and
 * meaning, so the in-browser engine consumes exactly what the Python edge emits. Two deliberate differences:
 * - a `type` tag ("video" | "audio"), the same key `obs_to_dict` writes to JSONL, so recordings parse as-is
 *   (`obsFromDict`);
 * - browser-only extras on each person (`raw`: all 52 blendshapes and head yaw/pitch/roll), which the Python
 *   edge does not have.
 *
 * Numbers that have no reading are NaN, as in Python (`obsToDict` writes them as null, like `obs_to_dict`).
 * Times are seconds: `t` = performance.now() / 1000 (monotonic, like time.monotonic()), `wall` = Date.now() / 1000.
 * Only scalars, booleans and normalized boxes: never pixels, audio samples or landmarks.
 */

/** Normalized x1, y1, x2, y2, each in [0, 1]. */
export type BBox = [number, number, number, number];

/** Face extras the browser gets for free from FaceLandmarker (held like face_valence: null when not visible). */
export interface FaceRaw {
  /** All 52 blendshape scores (0..1) by MediaPipe category name, `_neutral` included. */
  blendshapes: Record<string, number>;
  /**
   * Absolute head pose from the facial transformation matrix, degrees, signed as in docs/EMOTION_FEATURES.md 1.1:
   * yaw > 0 the nose points toward the image's right; pitch > 0 the face tilts up; roll > 0 the head tilts clockwise
   * in the image (its top toward the image's right). All about 0 when facing the lens.
   */
  head_yaw_deg: number;
  head_pitch_deg: number;
  head_roll_deg: number;
}

/** One tracked person in one processed video frame (~10 Hz). contracts.PersonObs. */
export interface PersonObs {
  /** Ephemeral display id 1..N ("Person #N"); never linked to identity. */
  track_id: number;
  bbox: BBox;
  /** Debounced fist rising edges in this frame. */
  fist_onsets: number;
  fist_active: boolean;
  /** Debounced posture slump. */
  slump: boolean;
  head_in_hands: boolean;
  /** EMA of mean torso/arm keypoint displacement, shoulder widths per second; NaN when undefined. */
  motion: number;
  face_visible: boolean;
  /** [-1, 1] relative to the person's neutral baseline; NaN when the face is not visible. */
  face_valence: number;
  // Emotion-map face cues (docs/EMOTION_FEATURES.md section 1), relative to the person's neutral (first 20 s of
  // visible face); NaN (blink_onsets 0) when the face is not visible. Python defaults them to NaN / 0.
  /** browDown L/R mean minus neutral, -1..1. */
  brow_down: number;
  /** Neutral eyeBlink L/R mean minus this frame's, -1..1 (> 0 eyes wider than neutral). */
  eye_open: number;
  /** mouthFrown L/R mean minus neutral, -1..1. */
  mouth_frown: number;
  /** mouthSmile L/R mean minus neutral, -1..1. */
  smile: number;
  /** Degrees from the neutral pose, -90..90; > 0 nose toward the image's right. */
  head_yaw: number;
  /** Degrees from the neutral pose, -90..90; > 0 face tilted up. */
  head_pitch: number;
  /** Blinks completed on this frame (0 or 1). */
  blink_onsets: number;
  /**
   * Browser-only extras. The browser source always sets it (null when the face is not visible); it is absent on
   * observations replayed from Python recordings.
   */
  raw?: FaceRaw | null;
}

/** Result of processing one video frame. contracts.VideoObs. */
export interface VideoObs {
  type: "video";
  t: number;
  wall: number;
  frame_seq: number;
  /** Cumulative frames taken from the camera since start(); must equal frames purged. */
  frames_decoded: number;
  /** Estimated lux (10 s median, updated at 1 Hz, repeated on every frame); NaN before the first reading. */
  lux: number;
  /** 0..1. */
  lux_quality: number;
  frame_w: number;
  frame_h: number;
  mirrored: boolean;
  persons: PersonObs[];
}

/** Result of one ~1 s audio step. contracts.AudioObs. */
export interface AudioObs {
  type: "audio";
  t: number;
  wall: number;
  block_seq: number;
  /** Approximate dB SPL (Leq over the step): 10 log10(rms_power) + SPL offset (94 dB, uncalibrated). */
  db_spl: number;
  /** Linear mean(x^2) over the step at 16 kHz, for power-domain averaging. */
  rms_power: number;
  speech_prob: number;
  /** Debounced laughter rising edges. */
  laughter_onsets: number;
  /** Always 0 in the browser: there is no keyword recognizer. */
  compliment_hits: number;
}

export type Obs = VideoObs | AudioObs;

/** Governance counters (what the console's governance panel shows). */
export interface PerceptionCounters {
  /** Camera frames taken for processing since start() (each becomes one ImageBitmap). */
  frames_decoded: number;
  /** Of those, frames released: bitmap closed and every pixel copy zero-filled. */
  frames_purged: number;
  /** Raw mic buffers (~100 ms each, like the Python edge's capture parts) zero-filled after use. */
  audio_blocks_purged: number;
  /** Derived copies zero-filled: luminance and crop buffers, 1 s audio blocks, resampled audio. */
  derived_purged: number;
  /** Always 0: there is no speech recognizer in the browser. */
  transcripts_purged: number;
  /** Raw buffers alive right now (taken but not yet purged); 0 between steps. */
  live_raw_buffers: number;
  /** Buffers found alive when a step ended (should stay 0). */
  violations: number;
}

export type PerceptionPhase = "idle" | "loading" | "running" | "stopped" | "error";

export interface PerceptionStatus {
  phase: PerceptionPhase;
  /** A live video track is being processed. */
  camera: boolean;
  /** A live audio track is being processed. */
  mic: boolean;
  /** Video observations per second (EMA). */
  fps: number;
  /** Face passes per second (= fps by default, fps / 2 at the Python cadence): report it next to blink counts. */
  face_hz: number;
  /** Time to process one video frame, ms (EMA). */
  video_ms: number;
  /** Time to process one audio block, ms (EMA). */
  audio_ms: number;
  /** From start() to the first observation, ms; null until then. */
  load_ms: number | null;
  /**
   * Bytes of models and WASM the page fetched itself (0 when served from the HTTP cache or a mock). With the vision
   * worker (the default) that is the audio side only: the worker's downloads are not in the page's resource timeline.
   */
  download_bytes: number;
  /** Human-readable reasons the output is partial, e.g. "no microphone", "tab hidden: video paused". */
  degraded: string[];
  error: string | null;
  /** True for createMockPerception(). */
  mock: boolean;
  /** Where the video models run and on what ("worker/GPU", "main/CPU"); null until loaded or without a camera. */
  video_backend: string | null;
}

export interface PerceptionOptions {
  /** URL prefix of the self-hosted MediaPipe files (WASM under `wasm/`, models under `models/`). */
  assetBase?: string;
  /** Video processing rate, Hz (Python edge: settings.perception_hz = 10). */
  fps?: number;
  /** Process the stream's video track (default true when it has one). */
  video?: boolean;
  /** Process the stream's audio track (default true when it has one). */
  audio?: boolean;
  /**
   * MediaPipe delegate. Default "CPU": on identical frames the CPU path matches the Python edge to ~1e-4
   * (blendshapes, valence, head pose), while the GPU path is unmeasured on real hardware and drifted 100-300x more on
   * software WebGL. "GPU" (also `?delegate=GPU` on the page URL) falls back to "CPU" if it fails to start; on
   * software-rendered WebGL it runs, but far slower than CPU (headless Chromium: under 1 fps).
   */
  delegate?: "GPU" | "CPU";
  /** dB SPL = dBFS + this (Python default spl_offset_db = 94, uncalibrated). */
  splOffsetDb?: number;
  /** At most this many people (Python settings.max_tracks = 3). */
  maxPeople?: number;
  /** Ignore people shorter than this fraction of the frame height (Python settings.min_box_height = 0.25). */
  minPersonH?: number;
  /** Run the video models in a Web Worker (default true where supported; falls back to the main thread). */
  worker?: boolean;
  /**
   * Face pass on every video frame (default true) instead of every other one (false: the Python edge's cadence).
   * The emotion cues' blink detector needs the face rate (docs/EMOTION_FEATURES.md 1.3); measured on a CPU-only
   * machine: 7.2 fps with 7.2 face passes/s, against 8.4 fps with 4.2 at the Python cadence.
   */
  faceEveryFrame?: boolean;
}

export type ObservationListener = (obs: Obs) => void;
export type StatusListener = (status: PerceptionStatus) => void;

export interface Perception {
  /**
   * Start processing `stream` (camera and/or mic). Resolves once the models are loaded and the loops run; rejects
   * if nothing can be processed. The mock ignores the stream.
   */
  start(stream?: MediaStream | null): Promise<void>;
  /** Stop all processing and release every buffer. The caller still owns the stream and its tracks. */
  stop(): void;
  /** Subscribe to observations; returns an unsubscribe function. */
  onObservation(cb: ObservationListener): () => void;
  counters(): PerceptionCounters;
  status(): PerceptionStatus;
  /** Subscribe to status changes; returns an unsubscribe function. */
  onStatus(cb: StatusListener): () => void;
}

/** MediaPipe's 52 blendshape categories, in model order. */
export const BLENDSHAPE_NAMES = [
  "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
  "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
  "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
  "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
  "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
  "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
  "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
  "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
  "noseSneerLeft", "noseSneerRight",
] as const;

export const emptyCounters = (): PerceptionCounters => ({
  frames_decoded: 0,
  frames_purged: 0,
  audio_blocks_purged: 0,
  derived_purged: 0,
  transcripts_purged: 0,
  live_raw_buffers: 0,
  violations: 0,
});

export const idleStatus = (mock = false): PerceptionStatus => ({
  phase: "idle",
  camera: false,
  mic: false,
  fps: 0,
  face_hz: 0,
  video_ms: 0,
  audio_ms: 0,
  load_ms: null,
  download_bytes: 0,
  degraded: [],
  error: null,
  mock,
  video_backend: null,
});

// ---------- JSON form (contracts.obs_to_dict / obs_from_dict) ----------

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

const nanToNull = (v: number): number | null => (Number.isNaN(v) ? null : v);
const nullToNan = (v: unknown): number => (v === null || v === undefined ? NaN : Number(v));

/** JSON-safe dict, NaN -> null, like contracts.obs_to_dict (every PersonObs field, plus `raw` when present). */
export function obsToDict(obs: Obs): { [k: string]: Json } {
  if (obs.type === "audio") {
    return {
      type: "audio", t: obs.t, wall: obs.wall, block_seq: obs.block_seq, db_spl: nanToNull(obs.db_spl),
      rms_power: nanToNull(obs.rms_power), speech_prob: nanToNull(obs.speech_prob),
      laughter_onsets: obs.laughter_onsets, compliment_hits: obs.compliment_hits,
    };
  }
  return {
    type: "video", t: obs.t, wall: obs.wall, frame_seq: obs.frame_seq, frames_decoded: obs.frames_decoded,
    lux: nanToNull(obs.lux), lux_quality: obs.lux_quality, frame_w: obs.frame_w, frame_h: obs.frame_h,
    mirrored: obs.mirrored,
    persons: obs.persons.map((p) => {
      const d: { [k: string]: Json } = {
        track_id: p.track_id, bbox: [...p.bbox], fist_onsets: p.fist_onsets, fist_active: p.fist_active,
        slump: p.slump, head_in_hands: p.head_in_hands, motion: nanToNull(p.motion), face_visible: p.face_visible,
        face_valence: nanToNull(p.face_valence), brow_down: nanToNull(p.brow_down), eye_open: nanToNull(p.eye_open),
        mouth_frown: nanToNull(p.mouth_frown), smile: nanToNull(p.smile), head_yaw: nanToNull(p.head_yaw),
        head_pitch: nanToNull(p.head_pitch), blink_onsets: p.blink_onsets,
      };
      if (p.raw !== undefined) {
        d["raw"] = p.raw === null ? null : {
          blendshapes: { ...p.raw.blendshapes }, head_yaw_deg: nanToNull(p.raw.head_yaw_deg),
          head_pitch_deg: nanToNull(p.raw.head_pitch_deg), head_roll_deg: nanToNull(p.raw.head_roll_deg),
        };
      }
      return d;
    }),
  };
}

/**
 * Inverse of obsToDict; also reads Python JSONL lines (`obs_to_dict`, or the older "kind" key). Fields an older
 * recording lacks take the Python defaults (NaN, 0).
 */
export function obsFromDict(d: Record<string, unknown>): Obs {
  const kind = d["type"] ?? d["kind"];
  const num = (k: string) => nullToNan(d[k]);
  if (kind === "audio") {
    return {
      type: "audio", t: num("t"), wall: num("wall"), block_seq: num("block_seq"), db_spl: num("db_spl"),
      rms_power: num("rms_power"), speech_prob: d["speech_prob"] == null ? 0 : num("speech_prob"),
      laughter_onsets: Number(d["laughter_onsets"] ?? 0), compliment_hits: Number(d["compliment_hits"] ?? 0),
    };
  }
  if (kind !== "video") throw new Error(`unknown observation kind ${JSON.stringify(kind)}`);
  const persons = ((d["persons"] as Record<string, unknown>[] | undefined) ?? []).map((p): PersonObs => {
    const bbox = p["bbox"] as number[];
    if (!Array.isArray(bbox) || bbox.length !== 4) throw new Error("bbox must be (x1, y1, x2, y2)");
    const person: PersonObs = {
      track_id: Number(p["track_id"]), bbox: [bbox[0]!, bbox[1]!, bbox[2]!, bbox[3]!],
      fist_onsets: Number(p["fist_onsets"] ?? 0), fist_active: Boolean(p["fist_active"]), slump: Boolean(p["slump"]),
      head_in_hands: Boolean(p["head_in_hands"]), motion: nullToNan(p["motion"]),
      face_visible: Boolean(p["face_visible"]), face_valence: nullToNan(p["face_valence"]),
      brow_down: nullToNan(p["brow_down"]), eye_open: nullToNan(p["eye_open"]), mouth_frown: nullToNan(p["mouth_frown"]),
      smile: nullToNan(p["smile"]), head_yaw: nullToNan(p["head_yaw"]), head_pitch: nullToNan(p["head_pitch"]),
      blink_onsets: Number(p["blink_onsets"] ?? 0),
    };
    if ("raw" in p) {
      const r = p["raw"] as Record<string, unknown> | null;
      person.raw = r === null ? null : {
        blendshapes: { ...(r["blendshapes"] as Record<string, number>) }, head_yaw_deg: nullToNan(r["head_yaw_deg"]),
        head_pitch_deg: nullToNan(r["head_pitch_deg"]), head_roll_deg: nullToNan(r["head_roll_deg"]),
      };
    }
    return person;
  });
  return {
    type: "video", t: num("t"), wall: num("wall"), frame_seq: num("frame_seq"),
    frames_decoded: Number(d["frames_decoded"] ?? 0), lux: num("lux"), lux_quality: Number(d["lux_quality"] ?? 0),
    frame_w: Number(d["frame_w"] ?? 0), frame_h: Number(d["frame_h"] ?? 0), mirrored: Boolean(d["mirrored"]),
    persons,
  };
}
