/**
 * TypeScript mirrors of the Valence runtime contracts (valence/contracts.py), the runtime settings
 * (valence/config.py) and the shared tick-channel rules (valence/features/spec.py).
 *
 * Field names stay snake_case so records and observations round-trip through the same JSON the Python
 * side writes (recordings, golden fixtures, WebSocket messages). Missing numbers are NaN in memory and
 * null in JSON, exactly like `_nan_to_none` / `_none_to_nan`.
 */
import type { StateInfo } from "@/lib/valence";

export type {
  ExplanationMsg,
  GovernanceMsg,
  HeartbeatMsg,
  HelloMsg,
  PredictionMsg,
  ServerMessage,
  StateInfo,
  StateVectorMsg,
  StatusMsg,
  TrackPerson,
  TracksMsg,
} from "@/lib/valence";

/** valence/contracts.py:23 */
export const CONTRACT_VERSION = "1.0.0";

// ---------- Emotional states ----------

/** valence/contracts.py:29 */
export const CORE_STATES = ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"] as const;
/** valence/contracts.py:30 */
export const P1_STATES = ["Overwhelmed", "Tense"] as const;
/** valence/contracts.py:31 */
export const ALL_STATES = [...CORE_STATES, ...P1_STATES] as const;
export type StateId = (typeof ALL_STATES)[number];
/** valence/contracts.py:32, keyed by Settings.state_set. */
export const STATE_SETS: Readonly<Record<"core" | "all", readonly string[]>> = { core: CORE_STATES, all: ALL_STATES };

/** valence/contracts.py:34 */
export const STATE_TIER: Readonly<Record<string, "P0" | "P1">> = {
  Focused: "P0",
  Frustrated_Masked: "P0",
  Exhausted: "P0",
  Collaborative_Happy: "P0",
  Overwhelmed: "P1",
  Tense: "P1",
};

/** valence/contracts.py:36 */
export const STATE_DISPLAY: Readonly<Record<string, string>> = {
  Focused: "Focused",
  Frustrated_Masked: "Frustrated / Masked Stress",
  Exhausted: "Exhausted",
  Collaborative_Happy: "Collaborative / Happy",
  Overwhelmed: "Overwhelmed / Distressed",
  Tense: "Tense / Visible Frustration",
};

/**
 * Word stems the narrator validator uses (valence/contracts.py:48). Case-insensitive substring matching;
 * test Tense's stems before Frustrated_Masked's ("visibly frustrat" contains "frustrat").
 */
export const STATE_LEXICON: Readonly<Record<string, readonly string[]>> = {
  Focused: ["focus", "concentrat"],
  Frustrated_Masked: ["frustrat", "masked stress"],
  Exhausted: ["exhaust", "fatigu"],
  Collaborative_Happy: ["collaborat", "upbeat", "happy"],
  Overwhelmed: ["overwhelm", "distress"],
  Tense: ["tense", "visibly frustrat"],
};

/** The hello / GET /api/states entry list for a state set (valence/api/app.py:136). */
export function stateInfos(states: readonly string[]): StateInfo[] {
  return states.map((id) => ({ id, display: displayName(id), tier: STATE_TIER[id] ?? "P0" }));
}

function displayName(label: string): string {
  const name = STATE_DISPLAY[label];
  if (name === undefined) throw new Error(`unknown state ${JSON.stringify(label)}`); // Python: KeyError
  return name;
}

// ---------- Emotion map (emotion-v1, docs/EMOTION_FEATURES.md) ----------
// Ten states on the valence-arousal plane, kept apart from STATE_SETS: the served 4- and 6-state models,
// the narrator and the lexicon do not use them.

/** valence/contracts.py:60 */
export const EXTRA_STATES = ["Bored", "Confused", "Calm", "Anxious"] as const;
/** valence/contracts.py:61 */
export const EMOTION_STATES = [...ALL_STATES, ...EXTRA_STATES] as const;
/** valence/contracts.py:62 */
export const EMOTION_DISPLAY: Readonly<Record<string, string>> = {
  ...STATE_DISPLAY, Bored: "Bored", Confused: "Confused", Calm: "Calm", Anxious: "Anxious",
};
/** (valence, arousal) anchors, each in [-1, 1]; assumptions, not measurements (valence/contracts.py:66). */
export const AFFECT_ANCHORS: Readonly<Record<string, readonly [number, number]>> = {
  Focused: [0.1, 0.3],
  Frustrated_Masked: [-0.6, 0.6],
  Exhausted: [-0.4, -0.7],
  Collaborative_Happy: [0.7, 0.5],
  Overwhelmed: [-0.6, 0.8],
  Tense: [-0.4, 0.5],
  Bored: [-0.3, -0.5],
  Confused: [-0.2, 0.3],
  Calm: [0.4, -0.4],
  Anxious: [-0.5, 0.6],
};

/** One emotion-map state and its anchor, as listed in hello.affect_states (valence/contracts.py:364). */
export interface AffectStateInfo {
  id: string;
  display: string;
  valence: number;
  arousal: number;
}

/** Emotion-map reading for one person, prediction.affect (valence/contracts.py:373 Affect). */
export interface Affect {
  valence: number;
  arousal: number;
  label: string;
  label_display: string;
  confidence: number;
  probabilities: Record<string, number>;
  model: string;
}

/**
 * Human-readable label for a prediction (valence/contracts.py:80 display_label). With only the core states,
 * Frustrated_Masked reads "Frustrated" unless the mask override fired; once Tense is in the set it always
 * shows its full name. A lone person is never "Collaborative".
 */
export function displayLabel(
  label: string,
  opts: { maskOverride?: boolean; nPeople?: number; states?: readonly string[] } = {},
): string {
  const maskOverride = opts.maskOverride ?? false;
  const nPeople = opts.nPeople ?? 2;
  const states = opts.states ?? CORE_STATES;
  if (label === "Frustrated_Masked" && !states.includes("Tense") && !maskOverride) return "Frustrated";
  if (label === "Collaborative_Happy" && nPeople < 2) return "Upbeat";
  return displayName(label);
}

// ---------- Explanation layers and narrator contract ----------

/** valence/contracts.py:105 */
export const LAYERS = ["Ambient Physics", "Time Dynamics", "Somatic & Social", "Facial Surface"] as const;
export type LayerName = (typeof LAYERS)[number];

/** valence/contracts.py Driver (pydantic, every field required). */
export interface Driver {
  feature: string;
  display: string;
  value: number | null;
  unit: string;
  shap: number;
  level: "high" | "elevated" | "normal" | "low";
  effect: "supports" | "opposes";
  scope: "room" | "person";
}

/** valence/contracts.py Layer */
export interface Layer {
  name: LayerName;
  contribution: number;
  drivers: Driver[];
}

/** valence/contracts.py FaceOnly: label/confidence are null when no face was visible. */
export interface FaceOnly {
  label: string | null;
  confidence: number | null;
  face_valence: number | null;
}

/** valence/contracts.py DataQuality */
export interface DataQuality {
  face_visible_frac: number;
  lux: number | null;
  ood_features: string[];
}

/** valence/contracts.py FactSheet: what the narrator (and the template fallback) reads. */
export interface FactSheet {
  event_id: string;
  person: string;
  label: string;
  label_display: string;
  confidence: number;
  confidence_display: string;
  probabilities: Record<string, number>;
  window_fill_min: number;
  n_people: number;
  face_only: FaceOnly;
  face_visible: boolean;
  mask_override: boolean;
  layers: Layer[];
  data_quality: DataQuality;
}

export type NarrationSource = "llm" | "llm_retry" | "template";

/** valence/contracts.py NarrationResult */
export interface NarrationResult {
  text: string;
  source: NarrationSource;
  latency_ms: number;
  attempts: number;
}

/** valence/contracts.py:416 */
export const GOVERNANCE_LINE =
  "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.";

// ---------- Perception observations ----------

/** One tracked person in one processed video frame (valence/contracts.py:122 PersonObs). */
export interface PersonObs {
  track_id: number;
  bbox: [number, number, number, number];
  fist_onsets: number;
  fist_active: boolean;
  slump: boolean;
  head_in_hands: boolean;
  motion: number;
  face_visible: boolean;
  face_valence: number;
  // Emotion-map face cues (docs/EMOTION_FEATURES.md section 1), relative to the person's neutral; NaN
  // (blink_onsets 0) when the face is not visible or the producer does not compute them.
  brow_down: number;
  eye_open: number;
  mouth_frown: number;
  smile: number;
  head_yaw: number;
  head_pitch: number;
  blink_onsets: number;
}

/** One processed video frame (valence/contracts.py:154 VideoObs); `type` is the obs_to_dict tag. */
export interface VideoObs {
  type: "video";
  t: number;
  wall: number;
  frame_seq: number;
  frames_decoded: number;
  lux: number;
  lux_quality: number;
  frame_w: number;
  frame_h: number;
  mirrored: boolean;
  persons: PersonObs[];
}

/** One ~1 s audio analysis step (valence/contracts.py:170 AudioObs). */
export interface AudioObs {
  type: "audio";
  t: number;
  wall: number;
  block_seq: number;
  db_spl: number;
  rms_power: number;
  speech_prob: number;
  laughter_onsets: number;
  compliment_hits: number;
}

export type Obs = VideoObs | AudioObs;

type FieldKind = "num" | "bool" | "bbox" | "persons";
interface FieldSpec { name: string; kind: FieldKind; required: boolean; dflt?: number | boolean }

// Dataclass field order matters: obs_to_dict emits keys in this order.
const PERSON_FIELDS: readonly FieldSpec[] = [
  { name: "track_id", kind: "num", required: true },
  { name: "bbox", kind: "bbox", required: true },
  { name: "fist_onsets", kind: "num", required: false, dflt: 0 },
  { name: "fist_active", kind: "bool", required: false, dflt: false },
  { name: "slump", kind: "bool", required: false, dflt: false },
  { name: "head_in_hands", kind: "bool", required: false, dflt: false },
  { name: "motion", kind: "num", required: false, dflt: NaN },
  { name: "face_visible", kind: "bool", required: false, dflt: false },
  { name: "face_valence", kind: "num", required: false, dflt: NaN },
  { name: "brow_down", kind: "num", required: false, dflt: NaN },
  { name: "eye_open", kind: "num", required: false, dflt: NaN },
  { name: "mouth_frown", kind: "num", required: false, dflt: NaN },
  { name: "smile", kind: "num", required: false, dflt: NaN },
  { name: "head_yaw", kind: "num", required: false, dflt: NaN },
  { name: "head_pitch", kind: "num", required: false, dflt: NaN },
  { name: "blink_onsets", kind: "num", required: false, dflt: 0 },
];
const VIDEO_FIELDS: readonly FieldSpec[] = [
  { name: "t", kind: "num", required: true },
  { name: "wall", kind: "num", required: true },
  { name: "frame_seq", kind: "num", required: true },
  { name: "frames_decoded", kind: "num", required: false, dflt: 0 },
  { name: "lux", kind: "num", required: false, dflt: NaN },
  { name: "lux_quality", kind: "num", required: false, dflt: 0.0 },
  { name: "frame_w", kind: "num", required: false, dflt: 0 },
  { name: "frame_h", kind: "num", required: false, dflt: 0 },
  { name: "mirrored", kind: "bool", required: false, dflt: false },
  { name: "persons", kind: "persons", required: false },
];
const AUDIO_FIELDS: readonly FieldSpec[] = [
  { name: "t", kind: "num", required: true },
  { name: "wall", kind: "num", required: true },
  { name: "block_seq", kind: "num", required: true },
  { name: "db_spl", kind: "num", required: true },
  { name: "rms_power", kind: "num", required: true },
  { name: "speech_prob", kind: "num", required: false, dflt: 0.0 },
  { name: "laughter_onsets", kind: "num", required: false, dflt: 0 },
  { name: "compliment_hits", kind: "num", required: false, dflt: 0 },
];

/** valence/contracts.py:116 _check_unit_interval */
function checkUnitInterval(name: string, v: number): void {
  if (!(v >= -1e-6 && v <= 1 + 1e-6)) throw new RangeError(`${name}=${v} is outside [0, 1]`);
}

/** PersonObs.__post_init__ (valence/contracts.py:144): bbox of 4 values in [0, 1], face_valence in [-1, 1]. */
function validatePerson(p: PersonObs): PersonObs {
  if (p.bbox.length !== 4) throw new RangeError("bbox must be (x1, y1, x2, y2)");
  for (const v of p.bbox) checkUnitInterval("bbox", v);
  if (!Number.isNaN(p.face_valence) && !(p.face_valence >= -1 - 1e-6 && p.face_valence <= 1 + 1e-6)) {
    throw new RangeError(`face_valence=${p.face_valence} is outside [-1, 1]`);
  }
  return p;
}

/** Field values from a dict with `_none_to_nan` applied; unknown or missing required keys throw like Python's TypeError. */
function readFields(d: Record<string, unknown>, specs: readonly FieldSpec[], what: string): Record<string, unknown> {
  const known = new Set(specs.map((s) => s.name));
  for (const k of Object.keys(d)) if (!known.has(k)) throw new TypeError(`${what}: unexpected field ${JSON.stringify(k)}`);
  const out: Record<string, unknown> = {};
  for (const s of specs) {
    const has = Object.prototype.hasOwnProperty.call(d, s.name);
    if (!has && s.required) throw new TypeError(`${what}: missing field ${JSON.stringify(s.name)}`);
    const v = has ? d[s.name] : s.dflt;
    if (s.kind === "num") out[s.name] = v === null || v === undefined ? NaN : Number(v);
    else if (s.kind === "bool") {
      if (v === null) throw new TypeError(`${what}: ${s.name} is null`);
      out[s.name] = Boolean(v);
    } else out[s.name] = v;
  }
  return out;
}

function personFromDict(p: Record<string, unknown>): PersonObs {
  const f = readFields(p, PERSON_FIELDS, "PersonObs");
  const bbox = f["bbox"];
  if (!Array.isArray(bbox)) throw new TypeError("PersonObs: bbox must be a list");
  return validatePerson({ ...(f as unknown as PersonObs), bbox: bbox.map(Number) as PersonObs["bbox"] });
}

/** A PersonObs with the dataclass defaults filled in (validated like __post_init__). */
export function makePersonObs(init: Partial<PersonObs> & Pick<PersonObs, "track_id" | "bbox">): PersonObs {
  return validatePerson({
    fist_onsets: 0, fist_active: false, slump: false, head_in_hands: false, motion: NaN,
    face_visible: false, face_valence: NaN, brow_down: NaN, eye_open: NaN, mouth_frown: NaN, smile: NaN,
    head_yaw: NaN, head_pitch: NaN, blink_onsets: 0, ...init,
  });
}

/** A VideoObs with the dataclass defaults filled in. */
export function makeVideoObs(init: Partial<Omit<VideoObs, "type">> & Pick<VideoObs, "t" | "wall" | "frame_seq">): VideoObs {
  return {
    type: "video", frames_decoded: 0, lux: NaN, lux_quality: 0.0, frame_w: 0, frame_h: 0, mirrored: false,
    persons: [], ...init,
  };
}

/** An AudioObs with the dataclass defaults filled in. */
export function makeAudioObs(
  init: Partial<Omit<AudioObs, "type">> & Pick<AudioObs, "t" | "wall" | "block_seq" | "db_spl" | "rms_power">,
): AudioObs {
  return { type: "audio", speech_prob: 0.0, laughter_onsets: 0, compliment_hits: 0, ...init };
}

/**
 * Inverse of obsToDict (valence/contracts.py:212 obs_from_dict): `type` (or the older `kind`) picks the
 * class, null -> NaN, missing optional fields take the dataclass defaults.
 */
export function obsFromDict(dIn: Record<string, unknown>): Obs {
  const d = { ...dIn };
  const tag = d["type"] ?? d["kind"];
  delete d["type"];
  delete d["kind"];
  if (tag === "video") {
    const persons = d["persons"] ?? [];
    delete d["persons"];
    if (!Array.isArray(persons)) throw new TypeError("VideoObs: persons must be a list");
    const f = readFields(d, VIDEO_FIELDS.filter((s) => s.kind !== "persons"), "VideoObs");
    return { type: "video", ...(f as Omit<VideoObs, "type" | "persons">), persons: persons.map((p) => personFromDict(p as Record<string, unknown>)) };
  }
  if (tag === "audio") {
    const f = readFields(d, AUDIO_FIELDS, "AudioObs");
    return { type: "audio", ...(f as Omit<AudioObs, "type">) };
  }
  throw new Error(`unknown observation kind ${JSON.stringify(tag ?? null)}`);
}

const nanToNull = (v: unknown): unknown => (typeof v === "number" && Number.isNaN(v) ? null : v);

/**
 * JSON-safe dict for features-only recordings (valence/contracts.py:203 obs_to_dict):
 * `{"type": "video"|"audio", ...fields in dataclass order}` with NaN -> null (bbox values untouched).
 */
export function obsToDict(obs: Obs): Record<string, unknown> {
  const out: Record<string, unknown> = { type: obs.type };
  const src = obs as unknown as Record<string, unknown>;
  if (obs.type === "video") {
    for (const s of VIDEO_FIELDS) {
      if (s.kind === "persons") continue;
      out[s.name] = nanToNull(src[s.name]);
    }
    out["persons"] = obs.persons.map((p) => {
      const ps = p as unknown as Record<string, unknown>;
      const row: Record<string, unknown> = {};
      for (const s of PERSON_FIELDS) row[s.name] = s.kind === "bbox" ? [...p.bbox] : nanToNull(ps[s.name]);
      return row;
    });
    return out;
  }
  for (const s of AUDIO_FIELDS) out[s.name] = nanToNull(src[s.name]);
  return out;
}

// ---------- Per-tick record ----------

/** Room slot id in the rolling window and in scenario data (valence/features/window.py:15). */
export const ROOM = 0;

export type Source = "live" | "replay" | "sim" | "sim_seed";

/**
 * One 5 s row per person (valence/contracts.py:241 TickRecord). Room-level fields are copied into each
 * person's row. NaN marks a missing reading.
 */
export interface TickRecord {
  tick: number;
  ts: number;
  t_mono: number;
  session_id: string;
  person_id: number;
  display_id: string;
  n_people: number;
  source: string;
  simulated_fields: string[];
  // Ambient Physics (room)
  ambient_db: number;
  db_p90: number;
  light_lux: number;
  lux_quality: number;
  room_db_base: number;
  room_lux_base: number;
  // Somatic & Social
  laughter_count: number;
  compliments_count: number;
  clenched_fists_count: number;
  posture_slump: number;
  head_in_hands_frac: number;
  social_withdrawal_flag: number;
  motion_energy: number;
  // Facial Surface
  face_valence: number;
  face_visible: number;
  // Time Dynamics
  hours_since_break: number;
  tod_h: number;
  // Emotion-map face cues per tick (docs/EMOTION_FEATURES.md section 2): medians over the tick's face
  // frames, NaN when the face was never read; the room row keeps the defaults.
  brow_down: number;
  eye_open: number;
  mouth_frown: number;
  smile: number;
  head_yaw: number;
  head_pitch: number;
  head_away: number;
  blink_count: number;
}

/** TickRecord as JSON (`TickRecord.to_dict()`): NaN numbers are null. */
export type TickRecordDict = { [K in keyof TickRecord]: TickRecord[K] extends number ? number | null : TickRecord[K] };

/** Field names in dataclass order (valence/contracts.py:287 TICK_FIELDS). */
export const TICK_FIELDS = [
  "tick", "ts", "t_mono", "session_id", "person_id", "display_id", "n_people", "source", "simulated_fields",
  "ambient_db", "db_p90", "light_lux", "lux_quality", "room_db_base", "room_lux_base",
  "laughter_count", "compliments_count", "clenched_fists_count", "posture_slump", "head_in_hands_frac",
  "social_withdrawal_flag", "motion_energy", "face_valence", "face_visible", "hours_since_break", "tod_h",
  "brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away", "blink_count",
] as const satisfies readonly (keyof TickRecord)[];

/** Fields typed `int` in the dataclass (they serialise without ".0" in Python JSON). */
export const TICK_INT_FIELDS: readonly (keyof TickRecord)[] = [
  "tick", "person_id", "n_people", "laughter_count", "compliments_count", "clenched_fists_count",
  "social_withdrawal_flag", "blink_count",
];

const TICK_STRING_FIELDS = new Set<string>(["session_id", "display_id", "source"]);
const TICK_REQUIRED = ["tick", "ts", "t_mono", "session_id", "person_id", "display_id", "n_people"] as const;

/** Dataclass defaults for the optional TickRecord fields. */
const TICK_DEFAULTS = {
  source: "live", ambient_db: NaN, db_p90: NaN, light_lux: NaN, lux_quality: 0.0, room_db_base: NaN,
  room_lux_base: NaN, laughter_count: 0, compliments_count: 0, clenched_fists_count: 0, posture_slump: NaN,
  head_in_hands_frac: NaN, social_withdrawal_flag: 0, motion_energy: NaN, face_valence: NaN, face_visible: 0.0,
  hours_since_break: NaN, tod_h: NaN, brow_down: NaN, eye_open: NaN, mouth_frown: NaN, smile: NaN, head_yaw: NaN,
  head_pitch: NaN, head_away: NaN, blink_count: 0,
} as const;

export type TickRecordInit = Partial<TickRecord> & Pick<TickRecord, (typeof TICK_REQUIRED)[number]>;

/** A TickRecord with the dataclass defaults filled in (a fresh simulated_fields list each time). */
export function makeTickRecord(init: TickRecordInit): TickRecord {
  return { ...TICK_DEFAULTS, ...init, simulated_fields: [...(init.simulated_fields ?? [])] };
}

/**
 * TickRecord from its JSON dict (`TickRecord(**{k: row[k] for k in TICK_FIELDS if k in row})`, as
 * valence/runtime/engine.py:98 builds them) with null -> NaN. Keys outside TICK_FIELDS are ignored.
 */
export function tickRecordFromDict(d: Record<string, unknown>): TickRecord {
  for (const k of TICK_REQUIRED) {
    if (!Object.prototype.hasOwnProperty.call(d, k)) throw new TypeError(`TickRecord: missing field ${JSON.stringify(k)}`);
  }
  const out: Record<string, unknown> = { ...TICK_DEFAULTS, simulated_fields: [] };
  for (const k of TICK_FIELDS) {
    if (!Object.prototype.hasOwnProperty.call(d, k)) continue;
    const v = d[k];
    if (k === "simulated_fields") out[k] = Array.isArray(v) ? v.map(String) : [];
    else if (TICK_STRING_FIELDS.has(k)) out[k] = String(v);
    else out[k] = v === null || v === undefined ? NaN : Number(v);
  }
  return out as unknown as TickRecord;
}

/** `TickRecord.to_dict()` (valence/contracts.py:283): every field in order, NaN -> null. */
export function tickRecordToDict(r: TickRecord): TickRecordDict {
  const src = r as unknown as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const k of TICK_FIELDS) out[k] = k === "simulated_fields" ? [...r.simulated_fields] : nanToNull(src[k]);
  return out as TickRecordDict;
}

// ---------- Tick channels and shared rules (valence/features/spec.py) ----------

/** valence/features/spec.py:30 */
export const ROOM_CHANNELS = ["ambient_db", "light_lux", "laughter_count", "compliments_count"] as const;
/** valence/features/spec.py:31 */
export const PERSON_CHANNELS = [
  "motion_energy", "posture_slump", "head_in_hands_frac", "clenched_fists_count", "social_withdrawal_flag", "n_people",
] as const;
/** valence/features/spec.py:39 */
export const TIME_CHANNELS = ["hours_since_break", "tod_h"] as const;
/** valence/features/spec.py:40 */
export const FACE_CHANNELS = ["face_valence", "face_visible"] as const;

/** valence/features/spec.py:43 */
export const WITHDRAW_SLUMP_MIN = 0.6;
/** valence/features/spec.py:44; shoulder-widths/s (live capture: still ~0.014, typing ~0.03). */
export const WITHDRAW_MOTION_MAX = 0.02;

/** Mostly slumped and nearly still for the tick; same rule in sim and live (valence/features/spec.py:47). */
export function withdrawalFlag(postureSlump: number, motionEnergy: number): number {
  if (Number.isNaN(postureSlump) || Number.isNaN(motionEnergy)) return 0;
  return postureSlump >= WITHDRAW_SLUMP_MIN && motionEnergy <= WITHDRAW_MOTION_MAX ? 1 : 0;
}

/** valence/features/spec.py:55 */
export const ROOM_PREFIXES = ["db_", "lux_", "laugh_", "compl_"] as const;
/** valence/features/spec.py:56 */
export const TIME_FEATURES = ["hours_since_break", "tod_sin", "tod_cos"] as const;
/** valence/features/spec.py:57 */
export const NEVER_NARRATED = ["fill_ratio"] as const;

/** Explanation layer of a feature name (valence/features/spec.py:60 layer_of). */
export function layerOf(feature: string): LayerName {
  if (feature.startsWith("db_") || feature.startsWith("lux_")) return "Ambient Physics";
  if ((TIME_FEATURES as readonly string[]).includes(feature)) return "Time Dynamics";
  if (feature.startsWith("face_")) return "Facial Surface";
  return "Somatic & Social";
}

/** "room" or "person" scope of a feature name (valence/features/spec.py:70 scope_of). */
export function scopeOf(feature: string): "room" | "person" {
  return ROOM_PREFIXES.some((p) => feature.startsWith(p)) || feature === "n_people" ? "room" : "person";
}

// ---------- Settings ----------

/** Runtime configuration (valence/config.py Settings); snake_case like the Python dataclass and fixtures. */
export interface Settings {
  tick_s: number;
  window_ticks: number;
  perception_hz: number;
  warmup_s: number;
  override_block_s: number;
  max_tracks: number;
  min_box_height: number;
  participant_rel_height: number;
  participant_min_presence: number;
  break_absent_min: number;
  person_drop_min: number;
  db_cal_offset: number;
  near_field_db: number;
  room_baseline_s: number;
  state_set: "core" | "all";
  artifacts_dir: string;
  api_host: string;
  api_port: number;
  narrator_url: string;
  narrator_deadline_s: number;
  mode: "live" | "replay" | "hybrid";
  room_only: boolean;
  audit_dir: string;
  audit_ttl_s: number;
}

/** valence/config.py:10 defaults (no VALENCE_* environment overrides in the browser). */
export const DEFAULT_SETTINGS: Readonly<Settings> = Object.freeze({
  tick_s: 5.0,
  window_ticks: 180,
  perception_hz: 10.0,
  warmup_s: 60.0,
  override_block_s: 120.0,
  max_tracks: 3,
  min_box_height: 0.25,
  participant_rel_height: 0.55,
  participant_min_presence: 0.3,
  break_absent_min: 10.0,
  person_drop_min: 15.0,
  db_cal_offset: 94.0,
  near_field_db: 3.0,
  room_baseline_s: 90.0,
  state_set: "core",
  artifacts_dir: "artifacts",
  api_host: "127.0.0.1",
  api_port: 8000,
  narrator_url: "http://127.0.0.1:8081",
  narrator_deadline_s: 2.5,
  mode: "replay",
  room_only: false,
  audit_dir: "var/audit",
  audit_ttl_s: 86_400,
});

/** DEFAULT_SETTINGS with overrides (e.g. the `settings` object of a golden fixture). */
export function makeSettings(overrides: Partial<Settings> = {}): Settings {
  return { ...DEFAULT_SETTINGS, ...overrides };
}
