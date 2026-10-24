/**
 * The runtime engine: one call per 5 s tick turns the tick's TickRecords into the WebSocket messages
 * (valence/runtime/engine.py Runtime, plus the hello builder and the operator endpoints of valence/api/app.py
 * that only touch the runtime).
 *
 * Each step: apply the SIMULATED overlays, append the records to the rolling window, close the purge-gate tick
 * (governance message), forget people absent too long, then predict per person (or for the room in room-only
 * mode): served predictor -> PersonSmoother -> fact sheet -> template text -> state_vector + prediction, and a
 * throttled 'prediction' entry in the hash-chained audit log.
 *
 * Python's Runtime pulls records from a TickSource inside step(); here the caller hands them in
 * (`step(records, wall, mono)`, e.g. a scenario replay) or lets the runtime flush its own Aggregator
 * (`stepLive`, the LiveSource). The ticker thread, the event bus and the narrator HTTP client have no
 * counterpart: step() returns the messages it published, in emission order (governance, then per target
 * state_vector and prediction), and `onMessage` sees each one as it is published.
 *
 * The purge gate lives in perception (frontend/src/perception), so its cumulative counters come in through
 * setPurgeCounters / setFramesDecoded; left alone they stay 0, which is what a scenario replay reports.
 *
 * Parity: every message field, key order included, equals Python's model_dump() and the audit chain hashes
 * match byte for byte (golden values: scripts/export_web_golden.py -> src/test/fixtures/engine/runtime.json.gz).
 * Every number in a message is either a tick clock or a value Python rounds (3 or 4 decimals), so the last-bit
 * differences upstream (Math.exp in the softmax, Math.sin/cos/log2 in a few features) only show at an exact
 * rounding tie; no golden tick hits one.
 */
import { Aggregator, localTod, type BaselineSnapshot } from "./aggregator";
import { AuditChain, comparePyStr, type AuditPayload, type PyJson } from "./audit";
import {
  CONTRACT_VERSION,
  displayLabel,
  GOVERNANCE_LINE,
  makeTickRecord,
  ROOM,
  stateInfos,
  type FactSheet,
  type GovernanceMsg,
  type HelloMsg,
  type Layer,
  type Obs,
  type PredictionMsg,
  type Settings,
  type StateVectorMsg,
  type TickRecord,
  type TracksMsg,
} from "./contracts";
import { band, buildFactSheet } from "./explainer";
import { clip, isnan, npMean, pyIntRound, pyRound, pySum, tail } from "./num";
import type { Predictor } from "./served";
import { PersonSmoother } from "./smoother";
import { templateText } from "./template";
import { RollingWindow } from "./window";

export { ROOM };

// ---------- Types ----------

/** What one step publishes, in order: governance, then per target state_vector and prediction. */
export type TickMessage = GovernanceMsg | StateVectorMsg | PredictionMsg;
/** Everything the runtime publishes (Python's EventBus): tick messages and push_obs's tracks. */
export type RuntimeMessage = TickMessage | TracksMsg;

/** The SIMULATED overlays an operator can switch on (valence/runtime/engine.py:124). */
export type OverlayName = "face_occlusion" | "low_light";

/** Source label (Python: `self.source.label`): "live" counts decoded frames, "replay" reports its purges. */
export type SourceLabel = "live" | "replay";

/**
 * The purge gate's cumulative counters at the end of a tick (valence/governance/purge.py:41 PurgeStats).
 * Perception owns the gate in the browser and reports these; all default to 0 (a scenario replay purges
 * nothing).
 */
export interface PurgeCounters {
  /** `purged["video"]`: decoded camera frames purged since the gate started. */
  video_purged: number;
  /** `purged["audio"]`: captured audio blocks purged. */
  audio_purged: number;
  transcripts_purged: number;
  violations: number;
  live_raw_buffers: number;
}

/** The narrator client the runtime calls (valence/explain/narrator_client.py NarratorClient, async there). */
export interface NarratorLike {
  request(sheet: FactSheet, personId: number): void;
  clearCache?(): void;
}

/** What `latest` keeps per person: the last state_vector and prediction (GET /api/state). */
export interface LatestPerson {
  state_vector: StateVectorMsg;
  prediction: PredictionMsg;
}

export interface RuntimeOptions {
  /** Python: f"session-{datetime.now():%Y%m%d-%H%M%S}" (local time of `clock`). */
  sessionId?: string;
  /** Default: a fresh Aggregator(settings, sessionId), as Python's Runtime builds one. */
  aggregator?: Aggregator;
  /** Default: a fresh in-memory AuditChain(sessionId). */
  audit?: AuditChain;
  narrator?: NarratorLike | null;
  /** Wall clock in seconds (Python `time.time()`): step's default wall, audit timestamps, seed, history. */
  clock?: () => number;
  /** Monotonic clock in seconds (Python `time.monotonic()`): step's default mono, fps, tracks throttle. */
  monoClock?: () => number;
  /** Milliseconds for loop_latency_ms (Python `time.perf_counter()`). */
  perfClock?: () => number;
  /** Standard normal draws for the low_light overlay (Python `np.random.normal(0, 0.35)` = 0.35 * draw). */
  gauss?: () => number;
  /** Zone for the time of day of live ticks (local_tod): "UTC", an IANA name, or undefined for local time. */
  tz?: string;
  /** Called with every message as it is published (the EventBus). */
  onMessage?: (msg: RuntimeMessage) => void;
}

// ---------- Python helpers ----------

/** Python `int(x)` for a float: truncation toward zero; NaN and +-inf raise (ValueError / OverflowError). */
function pyInt(x: number): number {
  if (isnan(x)) throw new RangeError("cannot convert float NaN to integer");
  if (!Number.isFinite(x)) throw new RangeError("cannot convert float infinity to integer");
  return Math.trunc(x) + 0; // Python ints have no -0
}

/** Python `a % b` for numbers: the result takes the divisor's sign (floor modulo). */
function pyMod(a: number, b: number): number {
  const r = a % b;
  return r !== 0 && r < 0 !== b < 0 ? r + b : r + 0;
}

/** Python `f"{n:0{width}d}"` for an int: zero padding after the sign. */
function zeroPad(n: number, width: number): string {
  const digits = BigInt(n) < 0n ? (-BigInt(n)).toString() : BigInt(n).toString();
  const sign = n < 0 ? "-" : "";
  return sign + digits.padStart(width - sign.length, "0");
}

/** Python `max(1, n)`: n only when it is greater (a NaN loses, unlike Math.max). */
function atLeastOne(n: number): number {
  return n > 1 ? n : 1;
}

/**
 * An audit payload number that came from JSON: integral -> Python int (bigint), else a Python float (NaN and
 * +-inf stay numbers, so the audit check refuses them as Python's does). BigInt() alone would throw on 1.5.
 */
function pyIntOrFloat(n: number | null): bigint | number | null {
  return n !== null && Number.isInteger(n) ? BigInt(n) : n;
}

/** `str(s)[:n]`: the first n code points (a UTF-16 slice could split a surrogate pair). */
function pySlice(s: string, n: number): string {
  return Array.from(s).slice(0, n).join("");
}

/** Standard normal by the polar method (any generator will do: the overlay is noise, not a reference). */
function polarGauss(): number {
  for (;;) {
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const s = u * u + v * v;
    if (s > 0 && s < 1) return u * Math.sqrt((-2 * Math.log(s)) / s);
  }
}

/** f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}" for a wall clock in seconds (local time). */
function sessionIdAt(wall: number): string {
  const d = new Date(wall * 1000);
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return (
    `session-${p(d.getFullYear(), 4)}${p(d.getMonth() + 1)}${p(d.getDate())}-` +
    `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
}

/** A baseline snapshot as an audit payload: its counts are Python ints (bigint), its medians floats. */
function snapshotPayload(s: BaselineSnapshot): Record<string, PyJson> {
  return {
    mode: s.mode,
    complete: s.complete,
    target_samples: BigInt(s.target_samples),
    db_samples: BigInt(s.db_samples),
    lux_samples: BigInt(s.lux_samples),
    db_median: s.db_median,
    db_iqr: s.db_iqr,
    lux_median: s.lux_median,
  };
}

/** `time.perf_counter()` / `time.monotonic()` in ms: performance.now() where it exists (browser, worker, Node). */
function perfNow(): number {
  const perf = (globalThis as { performance?: { now(): number } }).performance;
  return perf !== undefined ? perf.now() : Date.now();
}

/**
 * A deep copy of a message (plain objects, arrays and scalars only): Python keeps `model_dump()` dicts in
 * history / latest, separate from the model it published, so a listener that edits a message it received
 * (sorting its layers, say) must not reach the runtime's read models.
 */
function plainCopy<T>(v: T): T {
  if (Array.isArray(v)) return v.map(plainCopy) as T;
  if (v === null || typeof v !== "object") return v;
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v as Record<string, unknown>)) out[k] = plainCopy(x);
  return out as T;
}

/** Layer.model_dump(): fresh objects in the pydantic field order (the message must not share the output's). */
function dumpLayers(layers: readonly Layer[]): Layer[] {
  return layers.map((l) => ({
    name: l.name,
    contribution: l.contribution,
    drivers: l.drivers.map((d) => ({
      feature: d.feature,
      display: d.display,
      value: d.value,
      unit: d.unit,
      shap: d.shap,
      level: d.level,
      effect: d.effect,
      scope: d.scope,
    })),
  }));
}

// ---------- Runtime ----------

const ZERO_COUNTERS: Readonly<PurgeCounters> = Object.freeze({
  video_purged: 0,
  audio_purged: 0,
  transcripts_purged: 0,
  violations: 0,
  live_raw_buffers: 0,
});

/**
 * valence/runtime/engine.py:108 Runtime. Public fields keep the Python attribute meaning (camelCase); the
 * maps and the window are live state, read them but change them only through the methods.
 */
export class Runtime {
  /** Python `self.s`. */
  readonly settings: Settings;
  predictor: Predictor;
  narrator: NarratorLike | null;
  sessionId: string;
  readonly window: RollingWindow;
  readonly aggregator: Aggregator;
  readonly audit: AuditChain;
  /** Python `self.source.label`; useLive/useScenario/useRecording set it. */
  sourceLabel: SourceLabel = "live";
  mode: Settings["mode"];
  roomOnly: boolean;
  readonly overlays: Record<OverlayName, boolean> = { face_occlusion: false, low_light: false };
  speed = 1.0;
  /** Set when a step gets no records (a finished scenario); the caller's ticker honours it. */
  paused = false;
  readonly smoothers = new Map<number, PersonSmoother>();
  readonly latest = new Map<number, LatestPerson>();
  /** deque(maxlen=window_ticks * 8) of state_vector and prediction messages, oldest first. */
  history: (StateVectorMsg | PredictionMsg)[] = [];
  loopLatencyMs = 0.0;
  readonly started: number;
  camera = false;
  mic = false;
  fps = 0.0;
  /** Frames decoded but not (yet) purged at the last tick; > 2 for 3 ticks is status "purge_gap". */
  purgeGap = 0;
  /** Consecutive ticks with purgeGap > 2. */
  gapTicks = 0;
  framesPurged = 0;
  framesDecoded = 0;

  private readonly clock: () => number;
  private readonly monoClock: () => number;
  private readonly perfClock: () => number;
  private readonly gauss: () => number;
  private readonly tz: string | undefined;
  private readonly onMessage: ((msg: RuntimeMessage) => void) | undefined;
  private lastNarrated = new Map<number, [string, string, number]>();
  private lastAudited = new Map<number, [string, number]>();
  private baselineLogged: object | null = null;
  private tickI = 0;
  private lastTracks = 0.0;
  private fpsFrames: [number, number];
  private purgedBase = 0; // purged video frames before the current source began
  private purge: PurgeCounters = { ...ZERO_COUNTERS };
  private decodedOverride: number | null = null;
  /** Failed audit appends (Python logs them and carries on). */
  auditFailures = 0;

  constructor(settings: Settings, predictor: Predictor, opts: RuntimeOptions = {}) {
    this.settings = settings;
    this.predictor = predictor;
    this.narrator = opts.narrator ?? null;
    this.clock = opts.clock ?? (() => Date.now() / 1000);
    this.monoClock = opts.monoClock ?? (() => perfNow() / 1000);
    this.perfClock = opts.perfClock ?? perfNow;
    this.gauss = opts.gauss ?? polarGauss;
    this.tz = opts.tz;
    this.onMessage = opts.onMessage;
    this.sessionId = opts.sessionId ?? sessionIdAt(this.clock());
    this.window = new RollingWindow(settings.window_ticks);
    this.aggregator = opts.aggregator ?? new Aggregator(settings, this.sessionId);
    this.mode = settings.mode;
    this.roomOnly = settings.room_only;
    this.started = this.clock();
    this.audit = opts.audit ?? new AuditChain(this.sessionId);
    this.fpsFrames = [this.monoClock(), 0];
  }

  /** Ticks stepped so far (Python `_tick_i`): the next governance tick and event-id suffix. */
  get tickIndex(): number {
    return this.tickI;
  }

  /** valence/runtime/engine.py:466 states */
  states(): readonly string[] {
    return this.predictor.states;
  }

  // ---------- Purge-gate counters ----------

  /** The perception purge gate's cumulative counters (merged into the current ones). */
  setPurgeCounters(c: Partial<PurgeCounters>): void {
    this.purge = { ...this.purge, ...c };
  }

  /**
   * Frames decoded since the current source began, for the "live" label. null (the default) reads the
   * aggregator's count from the VideoObs it was fed, as Python does.
   */
  setFramesDecoded(n: number | null): void {
    this.decodedOverride = n;
  }

  /** valence/runtime/engine.py:219 _new_source_counters: align decoded/purged to the source starting now. */
  newSourceCounters(): void {
    this.aggregator.resetFrameCounter();
    this.purgedBase = this.purge.video_purged;
    this.gapTicks = 0;
  }

  // ---------- Sources and operator actions ----------

  /**
   * valence/runtime/engine.py:183 use_scenario, minus the source: the caller then steps the scenario's records.
   * Starts from an empty window and audits the switch.
   */
  useScenario(name: string, speed = 1.0): void {
    this.newSourceCounters();
    this.sourceLabel = "replay";
    this.mode = "replay";
    this.speed = speed;
    this.resetPeople(false);
    this.recordAudit("operator", { action: "replay", scenario: name, speed });
  }

  /** valence/runtime/engine.py:192 use_live: ticks come from the aggregator (stepLive). */
  useLive(): void {
    this.newSourceCounters();
    this.sourceLabel = "live";
    this.aggregator.source = "live";
    // an operator capture at the venue survives
    if (this.aggregator.baseline.mode !== "captured") this.aggregator.startBaseline(null, "auto");
    this.mode = "live";
    this.speed = 1.0;
    this.recordAudit("operator", { action: "live" });
  }

  /**
   * valence/runtime/engine.py:203 use_recording, minus the feeder: the caller pushes the recording's
   * observations (pushObs) and steps with stepLive, e.g. through replay.ts replayInto. `name` is what Python
   * audits, `path.name`: the recording's file name with its extension (POST /api/replay_recording/capture_fists
   * audits "capture_fists.jsonl").
   */
  useRecording(name: string, speed = 1.0): void {
    this.newSourceCounters();
    this.resetPeople(false);
    this.sourceLabel = "replay";
    this.aggregator.source = "replay";
    this.aggregator.startBaseline(null, "auto"); // the recording's own room
    this.mode = "replay";
    this.speed = speed; // ticks follow the playback speed, so each tick still holds 5 s of the recording
    this.recordAudit("operator", { action: "replay_recording", recording: pySlice(name, 80) });
  }

  /** valence/runtime/engine.py:231 reset_people: forget every person (keepRoom: the room history stays). */
  resetPeople(keepRoom = true): void {
    if (keepRoom) this.window.resetPeople();
    else this.window.clear();
    this.aggregator.resetPeople();
    this.smoothers.clear();
    this.latest.clear();
    this.history = [];
    this.lastNarrated.clear(); // person ids restart, so their throttles must too
    this.lastAudited.clear();
    this.narrator?.clearCache?.();
    this.recordAudit("operator", { action: "reset", keep_room: keepRoom });
  }

  /** valence/runtime/engine.py:246 set_overlay */
  setOverlay(name: OverlayName, enabled: boolean): void {
    this.overlays[name] = enabled;
    this.recordAudit("operator", { action: "sim", overlay: name, enabled });
  }

  /**
   * valence/runtime/engine.py:250 seed: pre-fill room history from a scenario (context only, tagged sim_seed).
   * `rows` are the scenario's rows (any order of person ids, ticks ascending); the room rows of the first
   * `minutes` are re-stamped to end now. Returns how many were added.
   */
  seed(
    rows: readonly TickRecord[],
    scenario: string,
    minutes: number,
    hoursSinceBreak: number | null,
  ): number {
    const room = rows.filter(
      (r) => r.person_id === ROOM && r.tick < (minutes * 60) / this.settings.tick_s,
    );
    const now = this.clock();
    const mono = this.monoClock();
    const n = room.length;
    room.forEach((row, i) => {
      const offset = (n - i) * this.settings.tick_s;
      this.window.append(
        makeTickRecord({
          ...row,
          ts: now - offset,
          t_mono: mono - offset,
          source: "sim_seed",
          simulated_fields: ["room_history"],
          session_id: this.sessionId,
        }),
      );
    });
    if (hoursSinceBreak !== null) this.aggregator.seedHours(hoursSinceBreak, this.clock());
    this.recordAudit("operator", {
      action: "demo_seed",
      scenario,
      minutes,
      hours_since_break: hoursSinceBreak,
      simulated_fields: ["room_history"],
    });
    return n;
  }

  /**
   * valence/runtime/engine.py:270 _audit: append to the audit chain at the wall clock; a refused payload is
   * counted, not thrown (Python logs it). Python ints in a payload must be bigint (audit.ts header).
   */
  recordAudit(kind: string, payload: AuditPayload): void {
    try {
      this.audit.append(kind, payload, this.clock());
    } catch {
      this.auditFailures += 1;
    }
  }

  // ---------- Operator endpoints (valence/api/app.py) ----------

  /** POST /api/mode (valence/api/app.py:542 set_mode): switch room-only mode. */
  setRoomOnly(roomOnly: boolean): void {
    if (roomOnly !== this.roomOnly) this.latest.clear(); // the other mode's cards would otherwise linger
    this.roomOnly = roomOnly;
    this.recordAudit("operator", { action: "mode", room_only: this.roomOnly });
  }

  /**
   * POST /api/events/break (valence/api/app.py:392 log_break): restart one person's break clock, or everyone's.
   * The body's person_id goes into the audit payload as JSON gave it: an integral id is a Python int (bigint),
   * anything else a float (a JSON `1.0` is the one case JS cannot tell apart; it is audited as the int 1).
   */
  logBreak(personId: number | null = null): void {
    this.aggregator.logBreak(personId, this.clock());
    this.recordAudit("operator", { action: "break", person_id: pyIntOrFloat(personId) });
  }

  /**
   * POST /api/room/baseline (valence/api/app.py:510 capture_room_baseline): capture ~seconds of the venue's
   * noise and light. Seconds outside [10, 600] (or NaN) throw, where the endpoint answers 422.
   */
  captureRoomBaseline(seconds: number = this.settings.room_baseline_s): BaselineSnapshot {
    if (!(seconds >= 10 && seconds <= 600)) throw new RangeError("seconds must be in [10, 600]");
    const snap = this.aggregator.startBaseline(seconds);
    this.recordAudit("operator", { action: "room_baseline", seconds });
    return snap;
  }

  /** POST /api/room/baseline/reset (valence/api/app.py:534 reset_room_baseline). */
  resetRoomBaseline(): BaselineSnapshot {
    const snap = this.aggregator.startBaseline(null, "auto");
    this.recordAudit("operator", { action: "room_baseline_reset" });
    return snap;
  }

  // ---------- Perception sink ----------

  /**
   * valence/runtime/engine.py:277 push_obs: feed the aggregator; a video frame also updates fps and, at most
   * every 0.1 s, publishes the participants' boxes (returned too; null otherwise).
   */
  pushObs(obs: Obs): TracksMsg | null {
    this.aggregator.push(obs);
    if (obs.type !== "video") {
      this.mic = true;
      return null;
    }
    this.camera = true;
    let [t0, n] = this.fpsFrames;
    n += 1;
    const now = this.monoClock();
    if (now - t0 >= 2.0) {
      this.fps = n / (now - t0);
      [t0, n] = [now, 0];
    }
    this.fpsFrames = [t0, n];
    if (!(now - this.lastTracks >= 0.1)) return null;
    this.lastTracks = now;
    let tallest = 0.0; // max(..., default=0.0): later heights replace it only when greater
    obs.persons.forEach((p, i) => {
      const h = p.bbox[3] - p.bbox[1];
      if (i === 0 || h > tallest) tallest = h;
    });
    const shown = obs.persons.filter(
      (p) => p.bbox[3] - p.bbox[1] >= this.settings.participant_rel_height * tallest,
    );
    const msg: TracksMsg = {
      type: "tracks",
      t: obs.wall,
      capture_ts_ms: pyInt(obs.wall * 1000),
      frame_w: obs.frame_w,
      frame_h: obs.frame_h,
      mirrored: obs.mirrored,
      persons: shown.map((p) => ({
        id: p.track_id,
        label: `Person #${p.track_id}`,
        bbox: [p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]],
        fist_active: p.fist_active,
        slump: p.slump,
      })),
    };
    this.onMessage?.(msg);
    return msg;
  }

  // ---------- The tick ----------

  /**
   * valence/runtime/engine.py:303 step, with the tick's records handed in (room row first, as a TickSource
   * returns them; null = the scenario finished, which pauses). `wall`/`mono` default to the clocks. Returns
   * every message published, in order.
   */
  step(
    records: readonly TickRecord[] | null,
    wall: number = this.clock(),
    mono: number = this.monoClock(),
  ): TickMessage[] {
    return this.tick(() => records, wall, mono);
  }

  /** step() on the LiveSource (valence/runtime/engine.py:70): flush the aggregator for this tick. */
  stepLive(wall: number = this.clock(), mono: number = this.monoClock()): TickMessage[] {
    return this.tick(() => this.aggregator.flush(wall, mono, localTod(wall, this.tz)), wall, mono);
  }

  private tick(
    next: () => readonly TickRecord[] | null,
    wall: number,
    mono: number,
  ): TickMessage[] {
    const t0 = this.perfClock();
    const got = next();
    if (got === null) {
      this.paused = true; // scenario finished
      return [];
    }
    const records = got.map((r) => this.overlay(r));
    const baseline = this.aggregator.baseline;
    if (baseline.mode === "captured" && this.baselineLogged !== baseline) {
      this.baselineLogged = baseline;
      this.recordAudit("operator", {
        action: "room_baseline_captured",
        ...snapshotPayload(baseline.snapshot()),
      });
    }
    for (const r of records) this.window.append(r);
    const purge = this.purge; // gate.end_tick()
    // Both counters cover the current source only. Replayed recordings carry the original capture's counter,
    // and this page decodes no frames for them, so replay reports its own purged count.
    this.framesPurged = purge.video_purged - this.purgedBase;
    const decoded =
      this.sourceLabel === "live"
        ? (this.decodedOverride ?? this.aggregator.framesDecoded)
        : this.framesPurged;
    this.framesDecoded = decoded;
    // Every decoded frame must be purged; allow the capture's in-flight slot (<= 2 frames).
    const gap = decoded - this.framesPurged;
    this.purgeGap = gap > 0 ? gap : 0;
    this.gapTicks = this.purgeGap > 2 ? this.gapTicks + 1 : 0;
    const out: TickMessage[] = [];
    const publish = (m: TickMessage): void => {
      out.push(m);
      this.onMessage?.(m);
    };
    publish({
      type: "governance",
      t: wall,
      tick: this.tickI,
      message: GOVERNANCE_LINE,
      frames_decoded: decoded,
      frames_purged: this.framesPurged,
      audio_blocks_purged: purge.audio_purged,
      transcripts_purged: purge.transcripts_purged,
      violations: purge.violations,
      live_raw_buffers: purge.live_raw_buffers,
      audit_hash: this.audit.lastHash.slice(0, 16),
      retention_s: this.settings.audit_ttl_s,
      purge_gap: this.purgeGap,
    });
    this.evict(wall);
    const targets = this.roomOnly
      ? [ROOM]
      : [...new Set(records.map((r) => r.person_id).filter((p) => p !== ROOM))].sort(
          (a, b) => a - b,
        );
    for (const pid of targets) this.predict(pid, wall, publish);
    this.tickI += 1;
    this.loopLatencyMs = this.perfClock() - t0;
    return out;
  }

  /** valence/runtime/engine.py:346 _evict: forget people absent longer than person_drop_min. */
  private evict(wall: number): void {
    for (const pid of this.window.people()) {
      const rows = this.window.snapshot(pid);
      const last = rows[rows.length - 1];
      if (last !== undefined && wall - last.ts > this.settings.person_drop_min * 60) {
        this.window.drop(pid);
        this.smoothers.delete(pid);
        this.latest.delete(pid);
      }
    }
  }

  /**
   * valence/runtime/engine.py:355 _overlay: the SIMULATED low-light / face-occlusion overlays. The window
   * keeps its own copy of every record, as Python's source hands over fresh ones.
   */
  private overlay(rIn: TickRecord): TickRecord {
    let r: TickRecord = { ...rIn, simulated_fields: [...rIn.simulated_fields] };
    if (!this.overlays.low_light && !this.overlays.face_occlusion) return r;
    const sim = [...r.simulated_fields];
    if (this.overlays.low_light) {
      const noisy = isnan(r.face_valence)
        ? r.face_valence
        : clip(r.face_valence + (0 + 0.35 * this.gauss()), -1, 1);
      r = {
        ...r,
        light_lux: !isnan(r.light_lux) ? r.light_lux * 0.15 : 30.0,
        face_valence: noisy,
      };
      sim.push("low_light");
    }
    if (this.overlays.face_occlusion) {
      r = { ...r, face_valence: NaN, face_visible: 0.0 };
      sim.push("face_occlusion");
    }
    return { ...r, simulated_fields: sim };
  }

  /** valence/runtime/engine.py:369 _predict: one target's state_vector + prediction (nothing for an empty window). */
  private predict(pid: number, wall: number, publish: (m: TickMessage) => void): void {
    const person = this.window.snapshot(pid);
    const last = person[person.length - 1];
    if (last === undefined) return;
    const room = this.window.room();
    const out = this.predictor.predict(person, room);
    if (out === null) return;
    let sm = this.smoothers.get(pid);
    if (sm === undefined) {
      sm = PersonSmoother.fromSettings(this.predictor.states, this.settings);
      this.smoothers.set(pid, sm);
    }
    const smooth = sm.update(out, person.length);
    const states = this.predictor.states;
    const label = smooth.label;
    // out.probabilities.get(label, out.confidence): unrounded (the fact sheet rounds it once)
    const conf = Object.hasOwn(out.probabilities, label)
      ? (out.probabilities[label] as number)
      : out.confidence;
    const nPeople = atLeastOne(last.n_people);
    const shown = displayLabel(label, { maskOverride: smooth.mask_override, nPeople, states });
    const name = pid === ROOM ? "Room" : `Person #${pid}`;
    const eventId = `p${pid}-${zeroPad(this.tickI, 6)}`;
    const faceVisible = last.face_visible > 0 && !isnan(last.face_valence);
    const sheet = buildFactSheet({
      event_id: eventId,
      person: name,
      label,
      label_display: shown,
      confidence: conf,
      probabilities: out.probabilities,
      window_fill_min: (person.length * this.settings.tick_s) / 60,
      n_people: nPeople,
      face_only_label: out.face_only_label,
      face_only_confidence: out.face_only_confidence,
      face_valence: isnan(last.face_valence) ? null : last.face_valence,
      face_visible: faceVisible,
      mask_override: smooth.mask_override,
      layers: out.layers,
      face_visible_frac: npMean(
        tail(
          Float64Array.from(person, (r) => r.face_visible),
          60,
        ),
      ),
      lux: isnan(last.light_lux) ? null : last.light_lux,
      ood_features: out.ood_features,
    });
    const text = templateText(sheet);
    const simSet = new Set(last.simulated_fields);
    for (const r of room.slice(-180)) for (const f of r.simulated_fields) simSet.add(f);
    const simFields = [...simSet].sort(comparePyStr);
    const pred: PredictionMsg = {
      type: "prediction",
      event_id: eventId,
      t: wall,
      person_id: pid,
      label,
      label_display: shown,
      confidence: pyRound(conf, 4),
      probabilities: { ...sheet.probabilities },
      face_only: { label: sheet.face_only.label, confidence: sheet.face_only.confidence },
      mask_override: smooth.mask_override,
      warming_up: smooth.warming_up,
      layers: dumpLayers(out.layers),
      data_quality: {
        face_visible_frac: sheet.data_quality.face_visible_frac,
        lux: sheet.data_quality.lux,
        ood_features: [...sheet.data_quality.ood_features],
      },
      explanation: { text, source: "template" },
      source: last.source,
      simulated_fields: simFields,
      affect: null,
    };
    const sv = this.stateVector(pid, person, room, wall, simFields);
    publish(sv);
    publish(pred);
    // sv_d, pred_d = sv.model_dump(), pred.model_dump(): history and latest share one copy, the published
    // messages stay the listeners' own
    const svD = plainCopy(sv);
    const predD = plainCopy(pred);
    this.history.push(svD, predD);
    const maxHistory = this.settings.window_ticks * 8;
    if (this.history.length > maxHistory)
      this.history.splice(0, this.history.length - Math.max(0, maxHistory));
    this.latest.set(pid, { state_vector: svD, prediction: predD });
    this.maybeNarrate(pid, sheet, wall);
    this.maybeAudit(pid, pred, wall);
  }

  /** valence/runtime/engine.py:419 _state_vector */
  private stateVector(
    pid: number,
    person: readonly TickRecord[],
    room: readonly TickRecord[],
    wall: number,
    simFields: string[],
  ): StateVectorMsg {
    const last = person[person.length - 1] as TickRecord;
    const room15 = room.length ? room.slice(-180) : person.slice(-180); // room[-180:] or person[-180:]
    const nz = (v: number): number | null => (isnan(v) ? null : pyRound(v, 3));
    const sumInt = (rows: readonly TickRecord[], f: (r: TickRecord) => number): number =>
      pyInt(pySum(rows.map(f))); // int(sum(...)): builtin sum, then truncation
    const tod = last.tod_h;
    const h = isnan(tod) ? 0 : pyInt(tod);
    const m = isnan(tod) ? 0 : pyMod(pyIntRound((tod - h) * 60), 60);
    return {
      type: "state_vector",
      t: wall,
      person_id: pid,
      ambient_db: nz(last.ambient_db),
      light_lux: nz(last.light_lux),
      hours_since_break: nz(last.hours_since_break),
      time_of_day: `${zeroPad(pyMod(h, 24), 2)}:${zeroPad(m, 2)}`,
      clenched_fists_5m: sumInt(person.slice(-60), (r) => r.clenched_fists_count),
      compliments_15m: sumInt(room15, (r) => r.compliments_count),
      laughter_15m: sumInt(room15, (r) => r.laughter_count),
      posture_slump: nz(last.posture_slump),
      head_in_hands: nz(last.head_in_hands_frac),
      social_withdrawal: pyInt(last.social_withdrawal_flag),
      face_visible: last.face_visible > 0,
      facial_valence: nz(last.face_valence),
      window_fill_min: pyRound((person.length * this.settings.tick_s) / 60, 2),
      source: last.source,
      simulated_fields: [...simFields],
    };
  }

  /** valence/runtime/engine.py:441 _maybe_narrate: on a new label/band, else at most every 30 s. */
  private maybeNarrate(pid: number, sheet: FactSheet, wall: number): void {
    if (this.narrator === null) return;
    const prev = this.lastNarrated.get(pid);
    const key: [string, string] = [sheet.label_display, band(sheet.confidence)];
    if (prev === undefined || prev[0] !== key[0] || prev[1] !== key[1] || wall - prev[2] >= 30) {
      this.lastNarrated.set(pid, [key[0], key[1], wall]);
      this.narrator.request(sheet, pid);
    }
  }

  /** valence/runtime/engine.py:450 _maybe_audit: a 'prediction' entry on a new shown label, else every 30 s. */
  private maybeAudit(pid: number, pred: PredictionMsg, wall: number): void {
    const prev = this.lastAudited.get(pid);
    if (prev === undefined || prev[0] !== pred.label_display || wall - prev[1] >= 30) {
      this.lastAudited.set(pid, [pred.label_display, wall]);
      const layers: Record<string, PyJson> = {};
      for (const layer of pred.layers) layers[layer.name] = layer.contribution;
      this.recordAudit("prediction", {
        person: pid === ROOM ? "Room" : `Person #${pid}`,
        label: pred.label,
        confidence: pred.confidence,
        mask_override: pred.mask_override,
        source: pred.source,
        layers,
        simulated_fields: pred.simulated_fields.slice(0, 8),
      });
    }
  }

  // ---------- Read models for the API ----------

  /** valence/runtime/engine.py:462 history_since: the messages of the last `minutes`. */
  historySince(minutes: number): (StateVectorMsg | PredictionMsg)[] {
    const cutoff = this.clock() - minutes * 60;
    return this.history.filter((m) => m.t >= cutoff);
  }

  /**
   * The hello message for this runtime (valence/api/app.py:131 hello): the state list, the model id as
   * feature_version ("m2_lgbm@" + sha256(m2.txt)[:8] for the served predictor) and the camera geometry.
   */
  hello(): HelloMsg {
    const agg = this.aggregator;
    return {
      type: "hello",
      contract_version: CONTRACT_VERSION,
      // a mode outside the Literal (Settings.mode is a plain str in Python) is announced as "replay"
      mode: this.mode === "live" || this.mode === "hybrid" ? this.mode : "replay",
      room_only: this.roomOnly,
      states: stateInfos(this.predictor.states),
      feature_version: this.predictor.modelId,
      frame_w: agg.frameW,
      frame_h: agg.frameH,
      mirrored: agg.mirrored,
      affect_states: null,
    };
  }

  /** valence/runtime/engine.py:162 close: drop the audit entries (Python deletes the session's audit file). */
  close(): void {
    this.audit.close(true);
  }
}
