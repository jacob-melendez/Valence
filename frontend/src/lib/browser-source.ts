/**
 * In-browser source: the whole Valence pipeline runs in this page (Valence Web).
 * Camera and microphone are opened here, turned into observations by the perception module
 * (frontend/src/perception), aggregated and scored by the TypeScript engine (frontend/src/engine), and
 * dropped. Nothing is recorded or uploaded (the deployed CSP has connect-src 'self'; status().sealed says whether
 * this host actually serves it).
 * It emits the same messages the backend WebSocket does (docs/API.md section 3), so every panel works unchanged.
 */
import type { AffectStateInfo, Command, EvalSummary, HelloMsg, RoomBaseline, ServerMessage, StatusMsg } from "./valence";

// ---------- Seams (implemented by frontend/src/engine and frontend/src/perception) ----------
/** A person in a video observation (valence/contracts.py PersonObs; only the fields this module reads). */
export interface ObsPerson {
  track_id: number; bbox: readonly [number, number, number, number]; fist_active?: boolean; slump?: boolean;
}
export interface VideoObsLike { t: number; wall: number; frame_w: number; frame_h: number; mirrored: boolean; persons: readonly ObsPerson[] }
export interface AudioObsLike { t: number; wall: number; db_spl: number }
export type ObsLike = VideoObsLike | AudioObsLike;
export const isVideo = (o: ObsLike): o is VideoObsLike => "persons" in o;

export interface AuditInfo { entries: number; head: string; verified: boolean }
export interface PerceptionCounters {
  frames_decoded: number; frames_purged: number; audio_blocks_purged: number;
  transcripts_purged?: number; violations?: number; live_raw_buffers?: number; fps?: number;
}
/** The parts of perception's status() this module shows (frontend/src/perception/types.ts PerceptionStatus). */
export interface PerceptionStatusLike {
  fps?: number; face_hz?: number; degraded?: string[]; video_backend?: string | null; download_bytes?: number;
}
/** frontend/src/perception's createPerception() / createMockPerception() result. */
export interface PerceptionLike {
  start(stream: MediaStream | null): void | Promise<void>;
  stop(): void;
  onObservation(cb: (o: ObsLike) => void): void | (() => void);
  counters(): PerceptionCounters;
  status?(): PerceptionStatusLike;
}

/** The engine side: the aggregator + runtime of valence/runtime, ported (frontend/src/engine). */
export interface BrowserEngine {
  states: { id: string; display: string; tier: string }[];
  featureVersion: string;
  /** Emotion-map states with their anchors (empty without the emotion models). */
  affectStates?: AffectStateInfo[];
  /** Feed one observation; a video frame may return the participants' `tracks` message (<= 10 Hz). */
  push(obs: ObsLike): ServerMessage | null | void;
  /** One 5 s tick: flush the aggregator and run the runtime step (governance, state_vector, prediction). */
  tick(wall: number, mono: number, counters: PerceptionCounters): ServerMessage[];
  setRoomOnly(roomOnly: boolean): void;
  setOverlay(which: "face_occlusion" | "low_light", enabled: boolean): void;
  logBreak(wall: number): void;
  startBaseline(seconds: number | null, mode: "capturing" | "auto"): void;
  baseline(): RoomBaseline;
  /** Forget every person; keepRoom false also drops the room's history (the kill switch, as app.py control stop). */
  resetPeople(keepRoom?: boolean): void;
  /** An operator entry in the audit chain (valence/api/app.py control: `control_{action}`). */
  audit?(action: string): void;
  /** The emotion model's current input for a person (self-report capture), when the emotion map is loaded. */
  latestInputs?(personId: number): { feature_version: string; features: Record<string, number | null> } | null;
  /** The in-memory audit chain: entries, head hash and whether every hash recomputes. */
  auditInfo?(): AuditInfo;
  /** The audit log as the desktop backend's JSONL file (verifiable with scripts/audit_verify.py). */
  auditJsonl?(): string;
  /** The Evaluation panel's numbers from the loaded models' metadata (held-out scores on simulated sessions). */
  evalSummary?(): EvalSummary;
}

export interface BrowserSourceDeps {
  createEngine: () => Promise<BrowserEngine>;
  createPerception: () => Promise<PerceptionLike>;
  emit: (m: ServerMessage) => void;
  onBaseline: (b: RoomBaseline) => void;
  /** The loaded models' evaluation summary, once the engine is up. */
  onEval?: (e: EvalSummary) => void;
  onError: (what: "camera" | "mic" | "engine", detail: string) => void;
  now?: () => number; // epoch ms
  mono?: () => number; // ms
  /** "tour": replay the engine's scripted emotion tour (no camera, no mic), `tourSpeed` times faster than real time. */
  mode?: "live" | "tour";
  tourSpeed?: number;
  /** false: open no camera or mic and start perception with null (?perception=mock). Default true. */
  needsMedia?: boolean;
  /** Whether this host serves a CSP that limits connect-src to 'self' (checked once at start; cspSealed()). */
  cspSealed?: () => Promise<boolean>;
  /** A camera/mic open begins (first start, or Start after pause, stop or a refused prompt): the last attempt's
   *  camera and mic errors no longer apply, and this attempt reports its own. */
  onOpenSensors?: () => void;
}

/** status() of this source: the backend's StatusMsg plus the browser-only fields the console shows. */
export type BrowserStatus = StatusMsg & { face_hz?: number; video_backend?: string | null; sealed?: boolean; loading?: string; audit?: AuditInfo };

// ---------- Shared camera stream (the preview reuses it instead of opening the camera twice) ----------
let stream: MediaStream | null = null;
const streamListeners = new Set<(s: MediaStream | null) => void>();
const setStream = (s: MediaStream | null) => { stream = s; streamListeners.forEach((l) => l(s)); };
export const getBrowserStream = () => stream;
export function onBrowserStream(l: (s: MediaStream | null) => void): () => void {
  streamListeners.add(l);
  return () => { streamListeners.delete(l); };
}
const stopTracks = (s: MediaStream | null | undefined) => s?.getTracks().forEach((t) => t.stop());

// ---------- Is the page sealed? ----------
/** True when a CSP limits connect-src to 'self' alone (only then may the console say the page is sealed). */
export const sealsConnect = (csp: string | null | undefined) => !!csp && /(?:^|;)\s*connect-src\s+'self'\s*(?:;|$)/i.test(csp);
/** Read this page's own CSP header (a same-origin HEAD, which connect-src 'self' allows); false when there is none. */
export async function cspSealed(fetchFn: typeof fetch = (...a) => fetch(...a)): Promise<boolean> {
  try {
    if (typeof location === "undefined") return false;
    const r = await fetchFn(location.pathname, { method: "HEAD", cache: "no-store" });
    return sealsConnect(r.headers.get("content-security-policy"));
  } catch { return false; }
}

// ---------- Camera and microphone ----------
const errName = (e: unknown) => (e && typeof e === "object" && "name" in e ? String((e as { name: unknown }).name) : "");
export const NO_MEDIA_API = "Camera and microphone need a secure page (HTTPS or localhost): this browser offers no getUserMedia here.";
/** Plain, actionable text for a getUserMedia failure (by DOMException name). */
export function mediaErrorText(e: unknown, what: "camera" | "mic" | "both"): string {
  const dev = what === "mic" ? "Microphone" : what === "camera" ? "Camera" : "Camera and microphone";
  switch (errName(e)) {
    case "NotAllowedError": return `${dev} blocked: allow ${what === "both" ? "them" : "it"} for this site (the camera icon in the address bar), then press Start.`;
    case "SecurityError": return `${dev} blocked by the browser's security policy (the page needs HTTPS and a policy that allows the camera).`;
    case "NotFoundError": return what === "mic" ? "No microphone found." : what === "camera" ? "No camera found." : "No camera or microphone found.";
    case "NotReadableError": return `${dev} could not start: another app may be using ${what === "both" ? "them" : "it"}, or the system blocked access.`;
    case "OverconstrainedError": return `${dev} cannot deliver the requested format.`;
    case "AbortError": return `${dev} start was interrupted: press Start to try again.`;
    default: return `${dev} failed: ${String(e)}`;
  }
}
// Worth asking again device by device: one device is missing, busy or picky. A refusal (NotAllowed, Security) covers
// both, and asking again only repeats the prompt (and counts toward the browser's dismissal embargo).
const RETRY_EACH = new Set(["NotFoundError", "NotReadableError", "OverconstrainedError"]);

/** Open camera (16:9, as the backend edge asks) and microphone (raw: no echo cancellation, noise suppression or AGC,
 *  so levels mean what the Python edge's levels mean). Either may fail on its own; `alive` false stops further prompts. */
async function openMedia(onError: BrowserSourceDeps["onError"], alive: () => boolean): Promise<MediaStream | null> {
  const md = typeof navigator !== "undefined" ? navigator.mediaDevices : undefined;
  if (!md?.getUserMedia) { onError("camera", NO_MEDIA_API); return null; }
  const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 };
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  try { return await md.getUserMedia({ video, audio }); } catch (e) {
    if (!RETRY_EACH.has(errName(e))) { onError("camera", mediaErrorText(e, "both")); return null; }
  }
  const out = new MediaStream();
  const each: ["camera" | "mic", MediaStreamConstraints][] = [["camera", { video }], ["mic", { audio }]];
  for (const [what, c] of each) {
    if (!alive()) break;
    try { (await md.getUserMedia(c)).getTracks().forEach((t) => out.addTrack(t)); } catch (e) { onError(what, mediaErrorText(e, what)); }
  }
  return out.getTracks().length ? out : null;
}

/** Bytes fetched so far under a prefix since the page loaded (resource timing, finished files only). */
function fetchedBytes(prefix: string): number {
  try {
    const base = new URL(prefix, location.href).href;
    return (performance.getEntriesByType("resource") as PerformanceResourceTiming[])
      .filter((e) => e.name.startsWith(base)).reduce((n, e) => n + (e.transferSize || 0), 0);
  } catch { return 0; }
}

// ---------- Explanation wording (describe.ts stays identical to Python; only the shown text changes) ----------
// The engine's break clock starts when a person is first seen (aggregator.ts seen), so live, before "Log a break",
// its "hours since the last break" is time on camera.
const SINCE_BREAK = / hours since the last break/g;
const SINCE_ARRIVAL = " hours since you appeared on camera (no break logged)";
// Signed values are formatted like Python's "+.2f", which prints a value just under zero as "-0.00" (a neutral face
// in the tour, for one): shown as "+0.00".
const NEG_ZERO = /(^|[^\d.])-(0\.0+)(?!\d)/g;
function relabelText(m: ServerMessage, re: (s: string) => string): ServerMessage {
  if (m.type !== "prediction") return m;
  return { ...m, explanation: { ...m.explanation, text: re(m.explanation.text) },
    layers: m.layers.map((l) => ({ ...l, drivers: l.drivers.map((d) => ({ ...d, display: re(d.display) })) })) };
}

// ---------- Evaluation rows in plain words ----------
// evalSummaryOf (engine/browser-engine.ts) names its rows by model id; the panel shows what each model is, with the
// id as the row's tooltip.
const EVAL_ROWS: Record<string, readonly [id: string, label: string]> = {
  "m2 · served, 4 states (context + face)": ["m2", "Context model · context + face, 4 states"],
  "m0 · face only, 4 states": ["m0", "Face-only baseline · face snapshot (2 features), 4 states"],
  "emotion-v1 · 10 states (context + face)": ["emotion-v1 full", "Emotion map · context + face, 10 states"],
  "emotion-v1 · face only": ["emotion-v1 face_only", "Emotion map · face only (38 face features)"],
  "emotion-v1 · context only": ["emotion-v1 context_only", "Emotion map · context only"],
};
/** The engine's evaluation summary with plain row names, the 4-state headline, and which measurement those rows are. */
export function plainEval(e: EvalSummary): EvalSummary {
  if (!Array.isArray(e.metrics)) return e;
  const metrics = e.metrics.map((r) => {
    const known = EVAL_ROWS[r.model];
    return known ? { ...r, id: known[0], model: known[1] } : r;
  });
  const f1 = (id: string) => metrics.find((r) => r.id === id)?.macro_f1;
  const m2 = f1("m2"), m0 = f1("m0");
  // model_meta.json's summary is the trainer's own test split (mixed history lengths); MODEL_CARD section 5's ladder
  // table measures windows with up to 15 min of history, so its figures differ slightly.
  const note = typeof m2 === "number" && typeof m0 === "number" && Number.isFinite(m2) && Number.isFinite(m0)
    ? ` On the 4 states, context + face scores ${m2.toFixed(2)} macro-F1 against ${m0.toFixed(2)} for the face snapshot alone.`
      + " Those two rows are the trainer's test score over mixed history lengths (model_meta.json); MODEL_CARD section 5's"
      + " ladder table measures windows with up to 15 min of history and reads 0.797 and 0.422."
    : "";
  return { ...e, metrics, ...(note ? { disclosure: `${e.disclosure ?? ""}${note}`.trim() } : {}) };
}

// ---------- The source ----------
const TICK_MS = 5000;
const MODELS = "/mediapipe/";

export class BrowserSource {
  private engine: BrowserEngine | null = null;
  private perception: PerceptionLike | null = null;
  /** The stream this source opened (published to the preview through setStream). */
  private media: MediaStream | null = null;
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private statusTimer: ReturnType<typeof setInterval> | null = null;
  private running: "running" | "paused" | "stopped" = "running";
  private startedAt = 0;
  private lastVideoAt = 0;
  private lastAudioAt = 0;
  private loopLatency = 0;
  private tourTicks = 0;
  /** "Log a break" pressed since the people were last reset: the break clock then means what it says. */
  private breakLogged = false;
  // Operator toggles, kept so a toggle pressed while the engine loads still reaches it.
  private roomOnly = false;
  private readonly overlays = { face_occlusion: false, low_light: false };
  private sealed: boolean | null = null;
  private engineLoading = false;
  private sensorLoading: "media" | "perception" | null = null;
  /** Model bytes already fetched when this perception load began (an earlier session's, or none). */
  private bytesBefore = 0;
  /** Bumped by start() / stop(): anything an older session still had in flight is dropped. */
  private generation = 0;
  /** Bumped by openSensors() / closeSensors() (pause, stop): an open still in flight releases what it got. */
  private sensorGen = 0;
  private readonly now: () => number;
  private readonly mono: () => number;

  private get tour() { return this.deps.mode === "tour"; }

  constructor(private readonly deps: BrowserSourceDeps) {
    this.now = deps.now ?? (() => Date.now());
    this.mono = deps.mono ?? (() => performance.now());
  }

  /** Load the engine, open camera + mic, start perception and the tick timer (status from the first second on). */
  async start(): Promise<void> {
    const gen = ++this.generation;
    this.clearTimers();
    this.startedAt = this.now();
    this.engineLoading = true;
    // Status from the start, so the console says what it waits for (engine, permission, ~40 MB of models).
    const everySecond = () => { if (gen === this.generation) this.deps.emit(this.status()); };
    this.statusTimer = setInterval(everySecond, 1000);
    everySecond();
    this.deps.cspSealed?.().then((s) => { if (gen === this.generation) this.sealed = s; }, () => {});
    let engine: BrowserEngine;
    try { engine = await this.deps.createEngine(); } catch (e) {
      if (gen === this.generation) { this.engineLoading = false; this.deps.onError("engine", String(e)); }
      return;
    }
    if (gen !== this.generation) return;
    this.engine = engine;
    this.engineLoading = false;
    if (engine.evalSummary) this.deps.onEval?.(plainEval(engine.evalSummary()));
    // Toggles pressed while it loaded (the console already shows them as set).
    if (this.roomOnly) engine.setRoomOnly(true);
    for (const w of ["face_occlusion", "low_light"] as const) if (this.overlays[w]) engine.setOverlay(w, true);
    this.deps.emit(this.hello());
    // Paused or stopped while the engine loaded: Start opens the sensors later.
    if (!this.tour && this.running === "running") await this.openSensors(gen);
    if (gen !== this.generation) return;
    this.tickTimer = setInterval(() => this.tick(), this.tour ? TICK_MS / (this.deps.tourSpeed ?? 10) : TICK_MS);
    this.deps.onBaseline(engine.baseline());
  }

  private async openSensors(gen: number) {
    const g = ++this.sensorGen;
    const alive = () => gen === this.generation && g === this.sensorGen && this.running === "running";
    const onError: BrowserSourceDeps["onError"] = (what, detail) => { if (alive()) this.deps.onError(what, detail); };
    if (!alive()) return;
    this.deps.onOpenSensors?.();
    let media: MediaStream | null = null, p: PerceptionLike | null = null;
    try {
      if (this.deps.needsMedia !== false) {
        this.sensorLoading = "media";
        media = await openMedia(onError, alive);
        if (!alive()) return this.release(media, null);
        // Neither camera nor mic: the cause is reported, and perception would have nothing to run on.
        if (!media) { this.sensorLoading = null; this.emitStatus(); return; }
        if (stream && stream !== media) stopTracks(stream); // never leave an earlier stream running unseen
        this.media = media;
        setStream(media);
      }
      this.sensorLoading = "perception";
      this.bytesBefore = fetchedBytes(MODELS);
      p = await this.deps.createPerception();
      if (!alive()) return this.release(media, p);
      const mine = p;
      this.perception = p;
      p.onObservation((o) => { if (this.perception === mine) this.onObs(o); });
      await p.start(media);
      if (!alive()) return this.release(media, p);
      this.sensorLoading = null;
      this.emitStatus();
    } catch (e) {
      const live = alive();
      this.release(media, p); // no perception: keep neither camera nor mic open for nothing
      if (live) { this.sensorLoading = null; onError("camera", `perception failed: ${String(e)}`); this.emitStatus(); }
    }
  }

  /** Status now rather than at the next second (the Welcome line follows a Stop or a finished open at once). */
  private emitStatus() { if (this.statusTimer) this.deps.emit(this.status()); }

  /** Stop a perception and close a stream this source opened, leaving whatever a newer open owns. */
  private release(media: MediaStream | null, p: PerceptionLike | null) {
    p?.stop();
    stopTracks(media);
    if (p && this.perception === p) this.perception = null;
    if (media && this.media === media) this.media = null;
    if (media && stream === media) setStream(null);
  }

  private closeSensors() {
    this.sensorGen += 1; // an open in flight sees this and releases what it got
    this.sensorLoading = null;
    const p = this.perception, media = this.media;
    this.perception = this.media = null;
    p?.stop();
    stopTracks(media);
    stopTracks(stream);
    setStream(null);
  }

  private clearTimers() {
    if (this.tickTimer) clearInterval(this.tickTimer);
    if (this.statusTimer) clearInterval(this.statusTimer);
    this.tickTimer = this.statusTimer = null;
  }

  /** Stop everything: timers, perception, camera and mic (source changes). */
  stop() {
    this.generation += 1;
    this.clearTimers();
    this.closeSensors();
    this.engine = null;
    this.engineLoading = false;
  }

  private onObs(o: ObsLike) {
    if (this.running !== "running" || !this.engine) return;
    // The engine's push_obs publishes the participants' boxes (valence/runtime/engine.py: >= 0.55x the tallest,
    // at most every 0.1 s).
    const tracks = this.engine.push(o);
    if (tracks) this.deps.emit(tracks);
    if (isVideo(o)) this.lastVideoAt = this.now(); else this.lastAudioAt = this.now();
  }

  /** The emotion model's current input for a person (for the self-report capture); none while paused or stopped. */
  /** The audit chain's state (checked at most every 5 s: verify recomputes every hash). */
  auditInfo(): AuditInfo | null {
    const e = this.engine;
    if (!e?.auditInfo) return null;
    const t = this.now();
    if (!this.auditCache || t - this.auditCache.at >= 5000) this.auditCache = { at: t, info: e.auditInfo() };
    return this.auditCache.info;
  }
  auditJsonl(): string | null { return this.engine?.auditJsonl?.() ?? null; }
  private auditCache: { at: number; info: AuditInfo } | null = null;

  latestInputs(personId: number) { return this.running === "running" ? this.engine?.latestInputs?.(personId) ?? null : null; }

  private tick() {
    if (this.running !== "running" || !this.engine) return;
    const t0 = this.mono();
    const counters = this.perception?.counters() ?? { frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0 };
    let msgs: ServerMessage[] = [];
    // The tour runs on a virtual clock (5 s per tick) so the 15-minute window and timeline read like real time.
    const [wall, mono] = this.tour
      ? [this.startedAt / 1000 + this.tourTicks * 5, this.tourTicks * 5]
      : [this.now() / 1000, this.mono() / 1000];
    this.tourTicks += 1;
    try { msgs = this.engine.tick(wall, mono, counters); } catch (e) { this.deps.onError("engine", String(e)); }
    this.loopLatency = this.mono() - t0;
    const sinceArrival = !this.tour && !this.breakLogged;
    msgs = msgs.map((m) => relabelText(m, (t) => {
      const u = t.replace(NEG_ZERO, "$1+$2");
      return sinceArrival ? u.replace(SINCE_BREAK, SINCE_ARRIVAL) : u;
    }));
    msgs.forEach((m) => this.deps.emit(m));
    this.deps.onBaseline(this.engine.baseline());
  }

  hello(): HelloMsg {
    const affect = this.engine?.affectStates ?? [];
    return { type: "hello", contract_version: "1.0.0", mode: this.tour ? "replay" : "live", room_only: this.roomOnly,
      states: this.engine?.states ?? [], feature_version: this.engine?.featureVersion ?? "", frame_w: 0, frame_h: 0, mirrored: false,
      ...(affect.length ? { affect_states: affect } : {}) };
  }

  /** What the page is still waiting for, or null once everything runs. */
  private loadingText(ps: PerceptionStatusLike | undefined): string | null {
    if (this.engineLoading) return this.tour ? "loading the engine and the tour" : "loading the engine and models";
    if (this.sensorLoading === "media") return "waiting for camera and microphone";
    if (this.sensorLoading !== "perception") return null;
    // Both are page-wide sums (perception's download_bytes reads the same resource timing): count this load only, so a
    // Start after Pause (models from the HTTP cache) does not show the first session's MB again.
    const mb = Math.max(0, Math.max(ps?.download_bytes ?? 0, fetchedBytes(MODELS)) - this.bytesBefore) / 1e6;
    return `loading perception models${mb >= 0.1 ? ` · ${mb.toFixed(1)} MB` : ""}`;
  }

  status(): BrowserStatus {
    const t = this.now(), c = this.perception?.counters(), ps = this.perception?.status?.();
    // Camera and mic as the browser has them open (its camera light), not as last seen: an open stream that nothing
    // reads must still show ON.
    const live = (kind: string) => !!this.media?.getTracks().some((tr) => tr.kind === kind && tr.readyState === "live");
    const camera = live("video"), mic = live("audio");
    const loading = this.loadingText(ps);
    const flowing = (on: boolean, at: number) => on && t - at < 3000;
    const degraded = this.tour || this.running !== "running" || loading ? []
      : this.deps.needsMedia === false ? ["mock perception (no camera or mic)", ...(ps?.degraded ?? [])]
      : [...(flowing(camera, this.lastVideoAt) ? [] : ["camera"]), ...(flowing(mic, this.lastAudioAt) ? [] : ["mic"]), ...(ps?.degraded ?? [])];
    const r1 = (v: number | undefined) => Math.round((v ?? 0) * 10) / 10;
    return { type: "status", mode: this.tour ? "replay" : "live", camera, mic, fps: r1(ps?.fps ?? c?.fps),
      // Face passes per second: blink numbers are only comparable at the same face rate (docs/EMOTION_FEATURES.md 1.3).
      ...(this.tour ? {} : { face_hz: r1(ps?.face_hz ?? ps?.fps ?? c?.fps), video_backend: ps?.video_backend ?? null }),
      ...(this.sealed === null ? {} : { sealed: this.sealed }),
      ...(loading ? { loading } : {}),
      ...(() => { const a = this.auditInfo(); return a ? { audit: a } : {}; })(),
      loop_latency_ms: Math.round(this.loopLatency * 10) / 10,
      // The LLM narrator is desktop-only; the browser always uses the template explanation.
      narrator: "template", offline: false, uptime_s: Math.round((t - this.startedAt) / 1000),
      degraded, purge_violation: false };
  }

  /** Operator commands, handled locally (no backend). Returns a short result line for the console. */
  command(c: Command): { ok: boolean; text: string } {
    const e = this.engine;
    const loading = { ok: false, text: "in-browser engine still loading" };
    switch (c.kind) {
      case "control": {
        const was = this.running;
        if (c.action === "start") {
          this.running = "running";
          // From pause or stop (or after a refused prompt) reopen camera and mic; before the engine exists start() does.
          if (!this.tour && e && (was !== "running" || (!this.perception && !this.sensorLoading))) void this.openSensors(this.generation);
        } else {
          // Pause releases camera and mic too (app.py control: stop_edge) but keeps the window and people; the kill
          // switch also forgets everyone, room history included (reset_people(keep_room=False)).
          this.running = c.action === "pause" ? "paused" : "stopped";
          this.closeSensors();
          if (c.action === "stop") { e?.resetPeople(false); this.breakLogged = false; }
        }
        e?.audit?.(`control_${c.action}`);
        this.emitStatus();
        return { ok: true, text: `control ${c.action} (in-browser)` };
      }
      case "break": if (!e) return loading; e.logBreak(this.now() / 1000); this.breakLogged = true; return { ok: true, text: "break logged (in-browser)" };
      // Only the cards go (runtime.setRoomOnly clears `latest`); windows and break clocks stay, as in app.py set_mode.
      case "mode": this.roomOnly = c.room_only; e?.setRoomOnly(c.room_only); return { ok: true, text: `room-only ${c.room_only}` };
      case "sim": this.overlays[c.which] = c.enabled; e?.setOverlay(c.which, c.enabled); return { ok: true, text: `sim ${c.which} ${c.enabled}` };
      case "baseline":
      case "baseline_reset":
        // The tour's records never pass the aggregator, so a capture would never fill.
        if (this.tour) return { ok: false, text: "the tour carries its own room baseline: capture needs the live camera and mic" };
        if (!e) return loading;
        if (c.kind === "baseline") e.startBaseline(c.seconds, "capturing"); else e.startBaseline(null, "auto");
        this.deps.onBaseline(e.baseline());
        return { ok: true, text: c.kind === "baseline" ? `baseline ${c.seconds} s (in-browser)` : "baseline reset (in-browser)" };
      // Forgotten people get fresh break clocks (aggregator resetPeople): time on camera again.
      case "demo_reset": if (!e) return loading; e.resetPeople(); this.breakLogged = false; return { ok: true, text: "people reset (in-browser)" };
      case "live": return { ok: true, text: "already live (in-browser)" };
      case "replay":
      case "replay_recording":
        return { ok: false, text: "replays need the desktop backend (LIVE source)" };
    }
  }
}
