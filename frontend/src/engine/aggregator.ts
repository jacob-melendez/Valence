/**
 * Aggregator: perception observations (~10 Hz video, ~1 Hz audio) -> one TickRecord per person per tick
 * (valence/runtime/aggregator.py).
 *
 * Also owns the presence-based break clock, the venue room baseline, and near-field gating of room audio
 * events: laughter and compliments only count when their audio block is clearly louder than the room's
 * baseline (or, before one exists, its recent median), so other tables in a noisy hall do not register as
 * social events at this desk.
 *
 * Synchronous and single-threaded: the Python lock has no counterpart. Arithmetic follows the Python
 * expressions term by term (numpy reductions through num.ts), so records match the reference bit for bit
 * except where a value passes through pow/log10 (the tick Leq and recent_level), which can differ from
 * numpy in the last bit.
 */
import {
  makeTickRecord,
  ROOM,
  withdrawalFlag,
  type AudioObs,
  type Obs,
  type PersonObs,
  type Settings,
  type TickRecord,
  type VideoObs,
} from "./contracts";
import { isnan, median, npMean, percentileLinear, pyRound, statsMedian } from "./num";

export { ROOM };

// ---------- Constants ----------

/** valence/runtime/aggregator.py:25 */
export const NEAR_FIELD_MIN_BLOCKS = 10;
/** Speech recognizers report a phrase 0-2 blocks after it ends: gate on the loudest recent block (aggregator.py:26). */
export const NEAR_FIELD_BLOCKS = 3;
/** valence/runtime/aggregator.py:27 */
export const DRIFT_WINDOW_S = 300.0;
/** Room 5-min median this far above/below the baseline -> "re-capture the baseline" (aggregator.py:28). */
export const DRIFT_DB = 12.0;
/** Light halved or doubled (aggregator.py:29). */
export const DRIFT_LUX_LOG2 = 1.0;
/** Emotion-map face cues: per-tick medians over the face frames (aggregator.py:31). */
export const CUE_FIELDS = [
  "brow_down",
  "eye_open",
  "mouth_frown",
  "smile",
  "head_yaw",
  "head_pitch",
] as const;
/** valence/runtime/aggregator.py:32 */
export const HEAD_AWAY_DEG = 20.0;
/** Block levels kept for the near-field running median (aggregator.py:232, `obs.t - 900`). */
const BLOCK_DB_WINDOW_S = 900;

type CueField = (typeof CUE_FIELDS)[number];
type BBox = PersonObs["bbox"];

/** `time.monotonic()` for the last_video_t / last_audio_t status fields (never part of a record). */
function monotonicNow(): number {
  const perf = (globalThis as { performance?: { now(): number } }).performance;
  return perf ? perf.now() / 1000 : Date.now() / 1000;
}

/** `time.time()` in seconds. */
function wallNow(): number {
  return Date.now() / 1000;
}

/** Python `max(items, default=d)` over floats: the first maximum wins, later items replace it only if `>`. */
function pyMax(items: readonly number[], dflt: number): number {
  if (items.length === 0) return dflt;
  let best = items[0] as number;
  for (let i = 1; i < items.length; i++) if ((items[i] as number) > best) best = items[i] as number;
  return best;
}

/** Python `sorted()` of strings: code point order (JS's default sort compares UTF-16 code units). */
function sortedStrings(items: Iterable<string>): string[] {
  return Array.from(items).sort((a, b) => {
    const ca = Array.from(a, (ch) => ch.codePointAt(0) as number);
    const cb = Array.from(b, (ch) => ch.codePointAt(0) as number);
    for (let i = 0; i < Math.min(ca.length, cb.length); i++) {
      if (ca[i] !== cb[i]) return (ca[i] as number) - (cb[i] as number);
    }
    return ca.length - cb.length;
  });
}

/** `10 * log10(mean(10 ** (db / 10)))`: energy-average level of dB values (aggregator.py:357, :311). */
function leqOf(db: readonly number[]): number {
  const power = db.map((v) => Math.pow(10.0, v / 10.0));
  return 10.0 * Math.log10(npMean(power));
}

// ---------- Per-tick accumulators ----------

/** One person's tick so far (aggregator.py:36 _PersonAcc). */
interface PersonAcc {
  frames: number;
  /** Sum of box heights (normalized), for the participant filter. */
  height: number;
  fists: number;
  slump: number;
  hih: number;
  face_frames: number;
  motion: number[];
  face: number[];
  cues: Record<CueField, number[]>;
  pose_frames: number;
  away_frames: number;
  blinks: number;
}

function newAcc(): PersonAcc {
  return {
    frames: 0,
    height: 0.0,
    fists: 0,
    slump: 0,
    hih: 0,
    face_frames: 0,
    motion: [],
    face: [],
    cues: { brow_down: [], eye_open: [], mouth_frown: [], smile: [], head_yaw: [], head_pitch: [] },
    pose_frames: 0,
    away_frames: 0,
    blinks: 0,
  };
}

/** The tick's emotion-map face cues for one person, NaN when the face was never read (aggregator.py:51). */
function cueValues(acc: PersonAcc): Pick<TickRecord, CueField | "head_away" | "blink_count"> {
  const med = (vals: readonly number[]): number => (vals.length ? median(vals) : NaN);
  return {
    brow_down: med(acc.cues.brow_down),
    eye_open: med(acc.cues.eye_open),
    mouth_frown: med(acc.cues.mouth_frown),
    smile: med(acc.cues.smile),
    head_yaw: med(acc.cues.head_yaw.map((v) => Math.abs(v))),
    head_pitch: med(acc.cues.head_pitch),
    head_away: acc.pose_frames ? acc.away_frames / acc.pose_frames : NaN,
    blink_count: acc.blinks,
  };
}

// ---------- Room baseline ----------

export type BaselineMode = "auto" | "capturing" | "captured";

/** RoomBaseline.snapshot() (aggregator.py:103); rounded medians, null when missing. */
export interface BaselineSnapshot {
  mode: BaselineMode;
  complete: boolean;
  target_samples: number;
  db_samples: number;
  lux_samples: number;
  db_median: number | null;
  db_iqr: number | null;
  lux_median: number | null;
}

/** Aggregator.baseline_status(): the snapshot plus how far the room drifted (aggregator.py:252). */
export interface BaselineStatus extends BaselineSnapshot {
  drift_db: number | null;
  drift_lux_log2: number | null;
  drifted: boolean;
}

/**
 * The venue's room level with the person sitting neutral: dB and lux medians, RAM only
 * (valence/runtime/aggregator.py:64 RoomBaseline).
 *
 * `auto` collects from the start of a live session or recording; an operator capture restarts collection
 * (`capturing` -> `captured`). Until `target` samples (~1 per second per channel) exist, the running median
 * is used, so room-relative features exist from the first tick.
 */
export class RoomBaseline {
  db: number[] = [];
  lux: number[] = [];

  constructor(
    public target: number,
    public mode: BaselineMode = "auto",
  ) {}

  addDb(v: number): void {
    if (this.db.length < this.target) {
      this.db.push(v);
      this.settle();
    }
  }

  addLux(v: number): void {
    if (this.lux.length < this.target) {
      this.lux.push(v);
      this.settle();
    }
  }

  private settle(): void {
    // A capture with no microphone completes on light alone.
    if (
      this.mode === "capturing" &&
      (this.db.length >= this.target || (this.db.length === 0 && this.lux.length >= this.target))
    ) {
      this.mode = "captured";
    }
  }

  get complete(): boolean {
    return this.db.length >= this.target;
  }

  /** `float(np.median(db))`, NaN before the first sample. */
  get dbBase(): number {
    return this.db.length ? median(this.db) : NaN;
  }

  /** `float(np.median(lux))`, NaN before the first sample. */
  get luxBase(): number {
    return this.lux.length ? median(this.lux) : NaN;
  }

  /** valence/runtime/aggregator.py:103 snapshot */
  snapshot(): BaselineSnapshot {
    const r = (v: number, nd = 1): number | null => (isnan(v) ? null : pyRound(v, nd));
    // np.subtract(*np.percentile(db, [75, 25])): each quantile is computed like the scalar call.
    const iqr =
      this.db.length >= 4 ? percentileLinear(this.db, 75) - percentileLinear(this.db, 25) : NaN;
    return {
      mode: this.mode,
      complete: this.complete,
      target_samples: this.target,
      db_samples: this.db.length,
      lux_samples: this.lux.length,
      db_median: r(this.dbBase),
      db_iqr: r(iqr),
      lux_median: r(this.luxBase, 0),
    };
  }
}

// ---------- Break clock ----------

/** valence/runtime/aggregator.py:114 _PersonClock */
interface PersonClock {
  first_seen: number;
  last_seen: number;
  break_start: number;
  seeded: boolean;
}

// ---------- Aggregator ----------

/** The room channels every record of a tick shares (the `room` dict of Aggregator.flush, aggregator.py:366). */
export type RoomFields = Pick<
  TickRecord,
  | "tick"
  | "ts"
  | "t_mono"
  | "session_id"
  | "n_people"
  | "source"
  | "ambient_db"
  | "db_p90"
  | "light_lux"
  | "lux_quality"
  | "room_db_base"
  | "room_lux_base"
  | "laughter_count"
  | "compliments_count"
  | "tod_h"
>;

/**
 * Observations in, one TickRecord per participant per tick out (valence/runtime/aggregator.py:121).
 * Python's private methods are private here (camelCase); public state keeps the Python meaning.
 */
export class Aggregator {
  readonly s: Settings;
  sessionId: string;
  source: string;
  /** Frames decoded since the current source began (across capture restarts). */
  framesDecoded = 0;
  frameW = 0;
  frameH = 0;
  mirrored = false;
  /** Laughter onsets and compliment hits dropped by the near-field gate. */
  gatedOut = 0;
  latestBoxes = new Map<number, BBox>();
  videoFramesTotal = 0;
  audioBlocksTotal = 0;
  /** Venue recalibration (POST /api/calibrate/room). */
  dbCorrection = 0.0;
  luxScale = 1.0;
  /** POST /api/sim/lux (SIMULATED). */
  luxOverride: number | null = null;
  lastVideoT = 0.0;
  lastAudioT = 0.0;
  baseline: RoomBaseline;

  private tickNo = 0;
  /** (t, db) for the 15-min room median. */
  private blockDb: [number, number][] = [];
  private lux = NaN;
  private luxQuality = 0.0;
  private decodedOffset = 0;
  private decodedLast = 0;
  private clocks = new Map<number, PersonClock>();
  private luxSampleT = -Infinity;
  /** (t, lux) at ~1 Hz for the drift check. */
  private luxHist: [number, number][] = [];
  /** The last NEAR_FIELD_BLOCKS block levels (a deque with maxlen in Python). */
  private recentDb: number[] = [];

  // Per-tick state (_reset_tick)
  private acc = new Map<number, PersonAcc>();
  private db: number[] = [];
  private laugh = 0;
  private compl = 0;
  private maxPeople = 0;
  private videoFrames = 0;
  private audioBlocks = 0;
  private simFields = new Set<string>();

  constructor(settings: Settings, sessionId: string, source = "live") {
    this.s = settings;
    this.sessionId = sessionId;
    this.source = source;
    this.baseline = new RoomBaseline(Math.trunc(settings.room_baseline_s));
  }

  private resetTick(): void {
    this.acc = new Map();
    this.db = [];
    this.laugh = 0;
    this.compl = 0;
    this.maxPeople = 0;
    this.videoFrames = 0;
    this.audioBlocks = 0;
    this.simFields = new Set();
  }

  // ---------- Ingestion ----------

  /** valence/runtime/aggregator.py:165 push */
  push(obs: Obs): void {
    if (obs.type === "video") this.pushVideo(obs);
    else if (obs.type === "audio") this.pushAudio(obs);
  }

  /** valence/runtime/aggregator.py:172 _push_video */
  private pushVideo(obs: VideoObs): void {
    this.videoFrames += 1;
    this.videoFramesTotal += 1;
    if (obs.frames_decoded < this.decodedLast) this.decodedOffset += this.decodedLast; // a new capture: its counter restarts at 0
    this.decodedLast = obs.frames_decoded;
    this.framesDecoded = this.decodedOffset + obs.frames_decoded;
    this.frameW = obs.frame_w;
    this.frameH = obs.frame_h;
    this.mirrored = obs.mirrored;
    this.lastVideoT = monotonicNow();
    if (!isnan(obs.lux)) {
      this.lux = obs.lux * this.luxScale;
      this.luxQuality = obs.lux_quality;
      if (obs.t - this.luxSampleT >= 1.0) {
        // lux arrives per video frame; the baseline wants ~1 sample/s
        this.luxSampleT = obs.t;
        this.baseline.addLux(this.lux);
        this.luxHist.push([obs.t, this.lux]);
        while (
          this.luxHist.length &&
          (this.luxHist[0] as [number, number])[0] < obs.t - DRIFT_WINDOW_S
        )
          this.luxHist.shift();
      }
    }
    this.maxPeople = Math.max(this.maxPeople, obs.persons.length);
    const boxes = new Map<number, BBox>();
    for (const p of obs.persons) {
      let acc = this.acc.get(p.track_id);
      if (acc === undefined) this.acc.set(p.track_id, (acc = newAcc()));
      acc.frames += 1;
      acc.height += p.bbox[3] - p.bbox[1];
      acc.fists += p.fist_onsets;
      acc.slump += p.slump ? 1 : 0;
      acc.hih += p.head_in_hands ? 1 : 0;
      if (!isnan(p.motion)) acc.motion.push(p.motion);
      acc.blinks += p.blink_onsets;
      if (p.face_visible && !isnan(p.face_valence)) {
        acc.face_frames += 1;
        acc.face.push(p.face_valence);
        for (const k of CUE_FIELDS) {
          const v = p[k];
          if (!isnan(v)) acc.cues[k].push(v);
        }
        if (!(isnan(p.head_yaw) || isnan(p.head_pitch))) {
          acc.pose_frames += 1;
          acc.away_frames +=
            Math.abs(p.head_yaw) > HEAD_AWAY_DEG || Math.abs(p.head_pitch) > HEAD_AWAY_DEG ? 1 : 0;
        }
      }
      boxes.set(p.track_id, p.bbox);
      this.seen(p.track_id, obs.wall);
    }
    this.latestBoxes = boxes;
  }

  /** valence/runtime/aggregator.py:215 _push_audio */
  private pushAudio(obs: AudioObs): void {
    this.audioBlocks += 1;
    this.audioBlocksTotal += 1;
    this.lastAudioT = monotonicNow();
    let db = obs.db_spl;
    if (!isnan(db)) {
      // `if self.db_correction`: Python's truthiness, where NaN counts as set (JS would treat it as unset).
      if (this.dbCorrection !== 0) db = db + this.dbCorrection;
      this.db.push(db);
      this.baseline.addDb(db);
      this.recentDb.push(db);
      if (this.recentDb.length > NEAR_FIELD_BLOCKS) this.recentDb.shift();
    }
    if (obs.laughter_onsets !== 0 || obs.compliment_hits !== 0) {
      if (this.nearField(pyMax(this.recentDb, db))) {
        this.laugh += obs.laughter_onsets;
        this.compl += obs.compliment_hits;
      } else {
        this.gatedOut += obs.laughter_onsets + obs.compliment_hits;
      }
    }
    if (!isnan(db)) {
      this.blockDb.push([obs.t, db]);
      while (
        this.blockDb.length &&
        (this.blockDb[0] as [number, number])[0] < obs.t - BLOCK_DB_WINDOW_S
      )
        this.blockDb.shift();
    }
  }

  /** Is this block level clearly louder than the room? (aggregator.py:235 _near_field) */
  private nearField(db: number): boolean {
    if (isnan(db)) return true;
    if (this.baseline.complete) return db >= this.baseline.dbBase + this.s.near_field_db;
    if (this.blockDb.length < NEAR_FIELD_MIN_BLOCKS) return true;
    const med = statsMedian(this.blockDb.map(([, v]) => v));
    return db >= med + this.s.near_field_db;
  }

  // ---------- Venue room baseline ----------

  /** Start a fresh baseline: `capturing` from the operator, `auto` on a new source (aggregator.py:246). */
  startBaseline(seconds: number | null = null, mode: BaselineMode = "capturing"): BaselineSnapshot {
    // `int(seconds or self.s.room_baseline_s)`: 0 and None both mean the configured length.
    this.baseline = new RoomBaseline(Math.trunc(seconds || this.s.room_baseline_s), mode);
    return this.baseline.snapshot();
  }

  /** valence/runtime/aggregator.py:252 baseline_status */
  baselineStatus(): BaselineStatus {
    return { ...this.baseline.snapshot(), ...this.drift() };
  }

  /** How far the room's last 5 minutes sit from the baseline, null until both exist (aggregator.py:256). */
  private drift(): Pick<BaselineStatus, "drift_db" | "drift_lux_log2" | "drifted"> {
    const out: Pick<BaselineStatus, "drift_db" | "drift_lux_log2" | "drifted"> = {
      drift_db: null,
      drift_lux_log2: null,
      drifted: false,
    };
    if (!this.baseline.complete) return out;
    const last = this.blockDb[this.blockDb.length - 1];
    if (last !== undefined) {
      const tEnd = last[0];
      const recent = this.blockDb.filter(([t]) => t >= tEnd - DRIFT_WINDOW_S).map(([, v]) => v);
      out.drift_db = pyRound(statsMedian(recent) - this.baseline.dbBase, 1);
    }
    const baseLux = this.baseline.luxBase;
    if (this.luxHist.length && baseLux > 0) {
      const med = statsMedian(this.luxHist.map(([, v]) => v));
      out.drift_lux_log2 = med > 0 ? pyRound(Math.log2(med / baseLux), 2) : null;
    }
    out.drifted =
      (out.drift_db !== null && Math.abs(out.drift_db) > DRIFT_DB) ||
      (out.drift_lux_log2 !== null && Math.abs(out.drift_lux_log2) > DRIFT_LUX_LOG2);
    return out;
  }

  // ---------- Break clock ----------

  /** valence/runtime/aggregator.py:274 _seen */
  private seen(pid: number, wall: number): void {
    const clock = this.clocks.get(pid);
    if (clock === undefined) {
      this.clocks.set(pid, { first_seen: wall, last_seen: wall, break_start: wall, seeded: false });
      return;
    }
    if (wall - clock.last_seen >= this.s.break_absent_min * 60) {
      clock.break_start = wall; // came back from a break
      clock.seeded = false;
    }
    clock.last_seen = wall;
  }

  /** Hours since the person's last break, NaN for an unknown person (aggregator.py:284 hours_since_break). */
  hoursSinceBreak(pid: number, wall: number): number {
    const clock = this.clocks.get(pid);
    if (clock === undefined) return NaN;
    const h = (wall - clock.break_start) / 3600.0;
    return h > 0.0 ? h : 0.0; // Python max(0.0, h): 0.0 unless h is greater (so -0.0 and NaN give 0.0)
  }

  /** Restart the break clock of one person, or of everyone (aggregator.py:288 log_break). */
  logBreak(pid: number | null = null, wall: number | null = null): void {
    const w = wall ?? wallNow();
    for (const [p, clock] of this.clocks) {
      if (pid === null || p === pid) {
        clock.break_start = w;
        clock.seeded = false;
      }
    }
  }

  /** Operator-set clock for every known person, logged SIMULATED (aggregator.py:296 seed_hours). */
  seedHours(hours: number, wall: number | null = null): void {
    const w = wall ?? wallNow();
    for (const clock of this.clocks.values()) {
      clock.break_start = w - hours * 3600.0;
      clock.seeded = true;
    }
  }

  // ---------- Calibration and operator actions ----------

  /** Leq of the last `seconds` of audio blocks, for venue recalibration (aggregator.py:304 recent_level). */
  recentLevel(seconds = 10.0): number {
    const last = this.blockDb[this.blockDb.length - 1];
    if (last === undefined) return NaN;
    const tEnd = last[0];
    return leqOf(this.blockDb.filter(([t]) => t >= tEnd - seconds).map(([, v]) => v));
  }

  /** valence/runtime/aggregator.py:313 calibrate */
  calibrate(
    splReference: number | null = null,
    luxReference: number | null = null,
  ): { db_correction?: number; lux_scale?: number } {
    const out: { db_correction?: number; lux_scale?: number } = {};
    if (splReference !== null) {
      const level = this.recentLevel();
      if (!isnan(level)) {
        this.dbCorrection += splReference - level;
        this.baseline.db = this.baseline.db.map((v) => v + splReference - level); // same units as live
      }
      out.db_correction = this.dbCorrection;
    }
    if (luxReference !== null && !isnan(this.lux) && this.lux > 0) {
      const factor = luxReference / this.lux;
      this.luxScale *= factor;
      this.baseline.lux = this.baseline.lux.map((v) => v * factor);
      out.lux_scale = this.luxScale;
    }
    return out;
  }

  /** SIMULATED events (fallback when a detector fails on stage), tagged in simulated_fields (aggregator.py:328). */
  inject(
    opts: {
      laughter?: number;
      compliments?: number;
      fists?: Readonly<Record<number, number>>;
    } = {},
  ): void {
    const laughter = opts.laughter ?? 0;
    const compliments = opts.compliments ?? 0;
    if (laughter) {
      this.laugh += laughter;
      this.simFields.add("laughter_count");
    }
    if (compliments) {
      this.compl += compliments;
      this.simFields.add("compliments_count");
    }
    for (const [key, n] of Object.entries(opts.fists ?? {})) {
      const pid = Number(key);
      let acc = this.acc.get(pid);
      if (acc === undefined) this.acc.set(pid, (acc = newAcc()));
      acc.fists += n;
      this.simFields.add("clenched_fists_count");
    }
  }

  /** valence/runtime/aggregator.py:341 reset_frame_counter */
  resetFrameCounter(): void {
    this.framesDecoded = this.decodedOffset = this.decodedLast = 0;
  }

  /** valence/runtime/aggregator.py:345 reset_people */
  resetPeople(): void {
    this.clocks.clear();
    this.acc.clear();
    this.latestBoxes = new Map();
  }

  // ---------- Flush (once per tick) ----------

  /**
   * Close the tick: the room record first, then one record per participant in person id order
   * (valence/runtime/aggregator.py:352 flush).
   */
  flush(wall: number, tMono: number, todH: number): TickRecord[] {
    const tick = this.tickNo;
    this.tickNo += 1;
    let leq: number;
    let p90: number;
    if (this.db.length) {
      leq = leqOf(this.db);
      p90 = percentileLinear(this.db, 90);
    } else {
      leq = p90 = NaN;
    }
    let lux = this.lux;
    if (this.luxOverride !== null) {
      lux = this.luxOverride;
      this.simFields.add("light_lux");
    }
    const simTick = sortedStrings(this.simFields);
    const room: RoomFields = {
      tick,
      ts: wall,
      t_mono: tMono,
      session_id: this.sessionId,
      n_people: this.maxPeople,
      source: this.source,
      ambient_db: leq,
      db_p90: p90,
      light_lux: lux,
      lux_quality: this.luxQuality,
      room_db_base: this.baseline.dbBase,
      room_lux_base: this.baseline.luxBase,
      laughter_count: this.laugh,
      compliments_count: this.compl,
      tod_h: todH,
    };
    const people: TickRecord[] = [];
    const participants = this.participants();
    room.n_people = participants.size;
    const pids = [...this.acc.keys()].sort((a, b) => a - b);
    for (const pid of pids) {
      if (!participants.has(pid)) continue;
      const acc = this.acc.get(pid) as PersonAcc;
      const slump = acc.slump / acc.frames;
      const motion = acc.motion.length ? npMean(acc.motion) : NaN;
      const clock = this.clocks.get(pid);
      people.push(
        makeTickRecord({
          ...room,
          person_id: pid,
          display_id: `Person #${pid}`,
          simulated_fields: clock?.seeded ? [...simTick, "hours_since_break"] : [...simTick],
          clenched_fists_count: acc.fists,
          posture_slump: slump,
          head_in_hands_frac: acc.hih / acc.frames,
          social_withdrawal_flag: withdrawalFlag(slump, motion),
          motion_energy: motion,
          face_valence: acc.face.length ? median(acc.face) : NaN,
          face_visible: acc.face_frames / acc.frames,
          hours_since_break: this.hoursSinceBreak(pid, wall),
          ...cueValues(acc),
        }),
      );
    }
    this.dropAbsent(wall);
    this.resetTick();
    return [roomRecord(room, people, simTick), ...people];
  }

  /**
   * People at the desk: present for a good share of the tick and close to the camera
   * (valence/runtime/aggregator.py:393 _participants).
   *
   * Background figures (people walking by, posters, a TV) are tracked but excluded: on the demo Mac they
   * were present in 80-95% of frames at a third of the main person's box height.
   */
  private participants(): Set<number> {
    if (this.acc.size === 0 || !this.videoFrames) return new Set();
    const meanH = new Map<number, number>();
    for (const [pid, acc] of this.acc) if (acc.frames) meanH.set(pid, acc.height / acc.frames);
    // Python max() over an empty sequence raises ValueError (only reachable when inject() created every
    // accumulator and no frame had a person); mirror it rather than invent a tick.
    if (meanH.size === 0) throw new Error("max() arg is an empty sequence");
    const tallest = pyMax([...meanH.values()], NaN);
    const out = new Set<number>();
    for (const [pid, acc] of this.acc) {
      if (
        acc.frames >= this.s.participant_min_presence * this.videoFrames &&
        (meanH.get(pid) as number) >= this.s.participant_rel_height * tallest
      )
        out.add(pid);
    }
    return out;
  }

  /** Forget the break clocks of people absent longer than person_drop_min (aggregator.py:407 _drop_absent). */
  private dropAbsent(wall: number): void {
    for (const [pid, c] of [...this.clocks]) {
      if (wall - c.last_seen > this.s.person_drop_min * 60) this.clocks.delete(pid);
    }
  }
}

// ---------- Room record ----------

/**
 * Room slot: room channels plus aggregates of whoever is present; also used by room-only mode
 * (valence/runtime/aggregator.py:412 room_record).
 */
export function roomRecord(
  room: RoomFields,
  people: readonly TickRecord[],
  simTick: readonly string[] = [],
): TickRecord {
  const mean = (
    attr:
      | "posture_slump"
      | "head_in_hands_frac"
      | "motion_energy"
      | "face_visible"
      | "hours_since_break",
  ): number => {
    const vals = people.map((p) => p[attr]).filter((v) => !isnan(v));
    return vals.length ? npMean(vals) : NaN;
  };
  const faces = people.map((p) => p.face_valence).filter((v) => !isnan(v));
  const sim = new Set<string>(simTick);
  for (const p of people) for (const f of p.simulated_fields) sim.add(f);
  let fists = 0;
  for (const p of people) fists += p.clenched_fists_count;
  return makeTickRecord({
    ...room,
    person_id: ROOM,
    display_id: "Room",
    simulated_fields: sortedStrings(sim),
    clenched_fists_count: fists,
    posture_slump: mean("posture_slump"),
    head_in_hands_frac: mean("head_in_hands_frac"),
    social_withdrawal_flag: people.some((p) => p.social_withdrawal_flag) ? 1 : 0,
    motion_energy: mean("motion_energy"),
    face_valence: faces.length ? median(faces) : NaN,
    face_visible: people.length ? mean("face_visible") : 0.0,
    hours_since_break: mean("hours_since_break"),
  });
}

// ---------- Local time of day ----------

/** C `round()`: half away from zero (exact: the fractional part of a double is representable). */
function cRound(x: number): number {
  const r = Math.trunc(x);
  return Math.abs(x - r) >= 0.5 ? r + Math.sign(x) : r;
}

/**
 * Whole seconds of a timestamp as `datetime.fromtimestamp` sees them: CPython first rounds to whole
 * microseconds, half to even (_PyTime_ObjectToTimeval with _PyTime_ROUND_HALF_EVEN), carrying into the
 * seconds, so a wall of ...04.9999996 reads as second 05.
 */
function fromtimestampSeconds(wall: number): number {
  let intpart = Math.trunc(wall);
  const floatpart = (wall - intpart) * 1e6; // modf, then scale to the denominator
  let us = cRound(floatpart);
  if (Math.abs(floatpart - us) === 0.5) us = 2.0 * cRound(floatpart / 2.0);
  if (us >= 1e6) intpart += 1;
  else if (us < 0) intpart -= 1;
  return intpart;
}

const tzFormatters = new Map<string, Intl.DateTimeFormat>();

function hmsIn(seconds: number, tz: string): [number, number, number] {
  let fmt = tzFormatters.get(tz);
  if (fmt === undefined) {
    fmt = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      hourCycle: "h23",
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
    });
    tzFormatters.set(tz, fmt);
  }
  const parts = fmt.formatToParts(new Date(seconds * 1000));
  const get = (type: string): number => Number(parts.find((p) => p.type === type)?.value ?? NaN);
  return [get("hour") % 24, get("minute"), get("second")];
}

/**
 * Local time of day in hours from integer seconds (valence/runtime/engine.py:52 local_tod):
 * `hour + minute / 60 + second / 3600`. `tz` is "UTC", an IANA zone name, or undefined for the
 * runtime's local zone (what Python's datetime.fromtimestamp uses).
 */
export function localTod(wall: number, tz?: string): number {
  const secs = fromtimestampSeconds(wall);
  let h: number;
  let m: number;
  let s: number;
  if (tz === "UTC" || tz === "Etc/UTC") {
    const day = ((secs % 86400) + 86400) % 86400;
    h = Math.floor(day / 3600);
    m = Math.floor((day % 3600) / 60);
    s = day % 60;
  } else if (tz === undefined) {
    const d = new Date(secs * 1000);
    h = d.getHours();
    m = d.getMinutes();
    s = d.getSeconds();
  } else {
    [h, m, s] = hmsIn(secs, tz);
  }
  return h + m / 60.0 + s / 3600.0;
}
