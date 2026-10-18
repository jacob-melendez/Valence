/**
 * Valence Console data layer: message types, store, mock engine, live transport, commands.
 * Every component reads through useValence() and writes through sendCommand(); switching
 * mock <-> live changes nothing outside this module.
 * Camera frames never pass through here: the webcam is preview-only in the UI.
 * Message shapes follow contract 1.0 (valence/contracts.py, docs/API.md).
 */
import { useSyncExternalStore } from "react";

// ---------- Types ----------
export type StateKey = "Focused" | "Frustrated_Masked" | "Exhausted" | "Collaborative_Happy";
export const STATES: StateKey[] = ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"];
export const STATE_DISPLAY: Record<StateKey, string> = {
  Focused: "Focused",
  Frustrated_Masked: "Frustrated / Masked Stress",
  Exhausted: "Exhausted",
  Collaborative_Happy: "Collaborative / Happy",
};
export const LAYERS = ["Ambient Physics", "Time Dynamics", "Somatic & Social", "Facial Surface"] as const;
/** The core state set, used until the backend's hello / GET /api/states arrives. */
export const CORE_STATES: StateInfo[] = STATES.map((id) => ({ id, display: STATE_DISPLAY[id], tier: "P0" }));

/** Full name of a state id from the active state set (hello.states), then the core names, then the id itself. */
export const stateName = (id: string | null | undefined): string =>
  id ? (state.states.find((x) => x.id === id)?.display
    ?? (STATE_DISPLAY as Record<string, string | undefined>)[id] ?? id.replace(/_/g, " ")) : "—";

/** Null-safe number formatting: the backend sends `null` when a sensor had no reading. */
export function fmt(v: number | null | undefined, digits = 1, fallback = "—"): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : fallback;
}

export interface TrackPerson { id: number; label: string; bbox: [number, number, number, number]; fist_active: boolean; slump: boolean }
export interface TracksMsg {
  type: "tracks"; t: number; capture_ts_ms: number; frame_w: number; frame_h: number; mirrored: boolean;
  persons: TrackPerson[];
}
export interface GovernanceMsg {
  type: "governance"; t: number; tick: number; message: string;
  frames_decoded: number; frames_purged: number; audio_blocks_purged: number; transcripts_purged: number;
  violations: number; live_raw_buffers: number; audit_hash: string | null; retention_s: number; purge_gap: number;
}
export interface StateVectorMsg {
  type: "state_vector"; t: number; person_id: number; ambient_db: number | null; light_lux: number | null;
  hours_since_break: number | null; time_of_day: string; clenched_fists_5m: number; compliments_15m: number;
  laughter_15m: number; posture_slump: number | null; head_in_hands: number | null; social_withdrawal: number;
  face_visible: boolean; facial_valence: number | null; window_fill_min: number;
  source: string; simulated_fields: string[];
}
export interface Driver {
  feature: string; display: string; value: number | null; unit: string; shap: number;
  level?: "high" | "elevated" | "normal" | "low"; effect?: "supports" | "opposes"; scope?: "room" | "person";
}
export interface Layer { name: string; contribution: number; drivers: Driver[] }
export type ExplanationSource = "llm" | "llm_retry" | "template";
export interface DataQuality { face_visible_frac: number; lux: number | null; ood_features: string[] }
export interface PredictionMsg {
  type: "prediction"; event_id: string; t: number; person_id: number; label: string;
  label_display: string; confidence: number; probabilities: Record<string, number>;
  /** Face-only baseline; both fields are null when no face was visible. */
  face_only: { label: string | null; confidence: number | null }; mask_override: boolean; warming_up: boolean;
  layers: Layer[]; data_quality: DataQuality;
  explanation: { text: string; source: ExplanationSource; latency_ms?: number };
  source: string; simulated_fields: string[];
}
export interface ExplanationMsg {
  type: "explanation"; event_id: string; person_id: number; text: string;
  source: ExplanationSource; latency_ms: number;
}
export interface StatusMsg {
  type: "status"; mode: "live" | "replay" | "hybrid"; camera: boolean; mic: boolean; fps: number;
  loop_latency_ms: number; narrator: string; offline: boolean; uptime_s: number;
  degraded: string[]; purge_violation: boolean;
}
export interface StateInfo { id: string; display: string; tier: string }
export interface HelloMsg {
  type: "hello"; contract_version: string; mode: "live" | "replay" | "hybrid"; room_only: boolean;
  states: StateInfo[]; feature_version: string; frame_w: number; frame_h: number; mirrored: boolean;
}
export interface HeartbeatMsg { type: "heartbeat"; t: number }
export type ServerMessage =
  | HelloMsg | TracksMsg | GovernanceMsg | StateVectorMsg | PredictionMsg | ExplanationMsg | StatusMsg | HeartbeatMsg;

export interface HistoryPoint {
  t: number; person_id: number; label?: string; confidence?: number;
  ambient_db?: number | null; light_lux?: number | null; hours_since_break?: number | null;
  posture_slump?: number | null; facial_valence?: number | null; window_fill_min?: number;
}
/** Mock-only placeholder row. */
export interface EvalMetricRow { model: string; accuracy: number; macro_f1: number; ece: number }
/** reports/metrics.json as served in GET /api/eval/summary `metrics` (keys defined by valence/eval/evaluate.py). */
export interface EvalReport {
  states?: string[];
  models?: Record<string, { macro_f1?: number; balanced_accuracy?: number; [k: string]: unknown }>;
  calibration_m2?: { ece_before?: number; ece_after?: number; [k: string]: unknown };
  generated_at?: string;
  disclosure?: string;
  [k: string]: unknown;
}
export interface EvalSummary {
  /** false until reports/metrics.json exists. */
  available?: boolean;
  accuracy_vs_history?: { minutes: number; context: number; face_only: number }[];
  robustness?: { condition: string; context: number; face_only: number }[];
  confusion?: { name: string; labels: string[]; matrix: number[][] }[];
  calibration?: { predicted: number; observed: number }[];
  metrics?: EvalMetricRow[] | EvalReport;
  figures?: string[];
  [k: string]: unknown;
}
export type Scenario = string;
/** GET /api/room/baseline, and `room_baseline` in GET /api/health (docs/API.md 7.4.12). */
export interface RoomBaseline {
  mode: "auto" | "capturing" | "captured"; complete: boolean; target_samples: number;
  db_samples: number; lux_samples: number; db_median: number | null; db_iqr: number | null; lux_median: number | null;
}
export interface ScenarioInfo { name: string; description?: string; minutes?: number }
export interface RecordingInfo { name: string; bytes?: number }
export interface ReplayOpts { speed: number; loop: boolean }
export const REPLAY_SPEEDS = [1, 4, 8, 20] as const;
export type Command =
  | { kind: "control"; action: "start" | "stop" | "pause" }
  | { kind: "break" }
  | ({ kind: "replay"; scenario: Scenario } & Partial<ReplayOpts>)
  | ({ kind: "replay_recording"; name: string } & Partial<ReplayOpts>)
  | { kind: "live" }
  | { kind: "demo_reset" }
  | { kind: "mode"; room_only: boolean }
  | { kind: "baseline"; seconds: number }
  | { kind: "baseline_reset" }
  | { kind: "sim"; which: "face_occlusion" | "low_light"; enabled: boolean };

/** Latest narrator text for a person, kept while its label_display is still the current one. */
export interface LlmExplanation {
  event_id: string; label_display: string; text: string; source: ExplanationSource; latency_ms: number;
}

export type ConnState = "mock" | "connecting" | "open" | "closed";
export interface ValenceState {
  source: "mock" | "live";
  conn: ConnState;
  baseUrl: string;
  persons: TrackPerson[];
  /** Box coordinates are in mirrored image space (tracks.mirrored). */
  mirrored: boolean;
  govLog: GovernanceMsg[];
  gov: GovernanceMsg | null;
  vectors: Record<number, StateVectorMsg>;
  predictions: Record<number, PredictionMsg>;
  llm: Record<number, LlmExplanation>;
  history: HistoryPoint[];
  status: StatusMsg | null;
  retention: Record<string, unknown> | null;
  evalSummary: EvalSummary | null;
  /** Active state set from hello.states or GET /api/states (4 core, or 6 with the P1 states). */
  states: StateInfo[];
  roomOnly: boolean;
  baseline: RoomBaseline | null;
  /** Why the edge runner (camera) or the mic failed to start, from POST /api/live or GET /api/health. */
  edgeError: string | null;
  micError: string | null;
  scenarios: ScenarioInfo[];
  recordings: RecordingInfo[];
  sim: { face_occlusion: boolean; low_light: boolean };
  running: "running" | "paused" | "stopped";
  lastCommand: { text: string; ok: boolean } | null;
  lastMessageAt: number;
}

// ---------- Backend URL ----------
// 127.0.0.1, not localhost: macOS resolves localhost to ::1 first and the API binds IPv4 loopback.
export const DEFAULT_API = "http://127.0.0.1:8000";
const API_STORAGE_KEY = "valence.api";
const cleanBase = (u: string) => u.trim().replace(/\/+$/, "");
const validBase = (u: string | null | undefined) => (u && /^https?:\/\/[^\s/]+/i.test(cleanBase(u)) ? cleanBase(u) : null);
function saveBase(u: string) {
  try { window.localStorage.setItem(API_STORAGE_KEY, u); } catch { /* storage blocked */ }
}
/** `?api=http://host:port` wins, then the last URL saved in localStorage, then DEFAULT_API. Client-only. */
export function resolveBaseUrl(): string {
  if (typeof window === "undefined") return DEFAULT_API;
  let fromQuery: string | null = null;
  try { fromQuery = validBase(new URLSearchParams(window.location.search).get("api")); } catch { /* ignore */ }
  if (fromQuery) { saveBase(fromQuery); return fromQuery; }
  try { return validBase(window.localStorage.getItem(API_STORAGE_KEY)) ?? DEFAULT_API; } catch { return DEFAULT_API; }
}

// ---------- Store ----------
let state: ValenceState = {
  source: "mock", conn: "mock", baseUrl: DEFAULT_API,
  persons: [], mirrored: false, govLog: [], gov: null, vectors: {}, predictions: {}, llm: {}, history: [],
  status: null, retention: null, evalSummary: null,
  states: CORE_STATES, roomOnly: false, baseline: null, edgeError: null, micError: null, scenarios: [], recordings: [],
  sim: { face_occlusion: false, low_light: false }, running: "running",
  lastCommand: null, lastMessageAt: 0,
};
const listeners = new Set<() => void>();
function set(patch: Partial<ValenceState>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}
export const getState = (): ValenceState => state;
export function useValence(): ValenceState {
  return useSyncExternalStore(
    (l) => { listeners.add(l); return () => listeners.delete(l); },
    () => state,
    () => state,
  );
}

const HISTORY_WINDOW_S = 15 * 60;
function pushHistory(p: HistoryPoint) {
  const h = state.history.filter((x) => x.t >= p.t - HISTORY_WINDOW_S);
  const last = [...h].reverse().find((x) => x.person_id === p.person_id && Math.abs(x.t - p.t) < 0.01);
  if (last) Object.assign(last, p); else h.push(p);
  return h;
}

// event_id -> label_display of each person's recent predictions, to pair late `explanation` messages.
const RECENT_EVENTS = 20;
let recentEvents: Record<number, { event_id: string; label_display: string }[]> = {};
let lastTracksAt = 0;
const labelKey = (p: { label_display?: string; label: string }) => p.label_display || p.label;

/**
 * Explanation to show for a prediction (docs/API.md section 5): the narrator text while the person's
 * label_display is unchanged since the narrated event, otherwise the prediction's own template text.
 */
export function shownExplanation(p: PredictionMsg, llm: LlmExplanation | undefined) {
  if (llm && llm.label_display === labelKey(p))
    return { text: llm.text, source: llm.source, latency_ms: llm.latency_ms as number | null, asOf: llm.event_id === p.event_id ? null : llm.event_id };
  return { text: p.explanation?.text ?? "", source: p.explanation?.source ?? "template", latency_ms: p.explanation?.latency_ms ?? null, asOf: null as string | null };
}

export function handleMessage(msg: ServerMessage) {
  const now = Date.now();
  switch (msg.type) {
    case "hello": {
      const states = validStates(msg.states);
      return set({ ...(states ? { states } : {}), roomOnly: !!msg.room_only, lastMessageAt: now });
    }
    case "heartbeat":
      return set({ lastMessageAt: now });
    case "tracks":
      lastTracksAt = now;
      return set({ persons: Array.isArray(msg.persons) ? msg.persons : [], mirrored: !!msg.mirrored, lastMessageAt: now });
    case "governance":
      return set({ gov: msg, govLog: [...state.govLog, msg].slice(-200), lastMessageAt: now });
    case "state_vector":
      return set({
        vectors: { ...state.vectors, [msg.person_id]: msg },
        history: pushHistory({
          t: msg.t, person_id: msg.person_id, ambient_db: msg.ambient_db, light_lux: msg.light_lux,
          hours_since_break: msg.hours_since_break, posture_slump: msg.posture_slump,
          facial_valence: msg.facial_valence, window_fill_min: msg.window_fill_min,
        }),
        lastMessageAt: now,
      });
    case "prediction": {
      const pid = msg.person_id, label = labelKey(msg);
      recentEvents[pid] = [...(recentEvents[pid] ?? []), { event_id: msg.event_id, label_display: label }].slice(-RECENT_EVENTS);
      const llm = state.llm[pid];
      let nextLlm = state.llm;
      if (llm && llm.label_display !== label) { nextLlm = { ...state.llm }; delete nextLlm[pid]; } // label changed
      return set({
        predictions: { ...state.predictions, [pid]: msg },
        llm: nextLlm,
        history: pushHistory({ t: msg.t, person_id: pid, label: msg.label, confidence: msg.confidence }),
        lastMessageAt: now,
      });
    }
    case "explanation": {
      const recent = recentEvents[msg.person_id] ?? [];
      const idx = recent.findIndex((e) => e.event_id === msg.event_id);
      const hit = recent[idx];
      if (!hit) return; // unknown event id (e.g. from before a reset or reconnect): drop
      const cur = state.llm[msg.person_id];
      if (cur && recent.findIndex((e) => e.event_id === cur.event_id) > idx) return; // never replace newer text
      return set({
        llm: {
          ...state.llm,
          [msg.person_id]: { event_id: msg.event_id, label_display: hit.label_display, text: msg.text, source: msg.source, latency_ms: msg.latency_ms },
        },
        lastMessageAt: now,
      });
    }
    case "status": {
      const stale = state.persons.length > 0 && now - lastTracksAt > 1500; // clear boxes once tracks stop
      return set({ status: msg, lastMessageAt: now, ...(stale ? { persons: [] } : {}) });
    }
  }
}

/** A usable state list from hello.states / GET /api/states, or null to keep the current one. */
export function validStates(v: unknown): StateInfo[] | null {
  if (!Array.isArray(v)) return null;
  const out = v.filter((x): x is StateInfo => !!x && typeof x === "object" && typeof (x as StateInfo).id === "string")
    .map((x) => ({ id: x.id, display: typeof x.display === "string" && x.display ? x.display : x.id.replace(/_/g, " "), tier: String(x.tier ?? "P0") }));
  return out.length ? out : null;
}

/** A room baseline object from GET /api/room/baseline, a baseline POST response or health.room_baseline. */
export function parseBaseline(v: unknown): RoomBaseline | null {
  if (!v || typeof v !== "object") return null;
  const b = v as { [K in keyof RoomBaseline]?: unknown };
  const mode = b.mode === "capturing" || b.mode === "captured" || b.mode === "auto" ? b.mode : null;
  if (!mode) return null;
  const n = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : null);
  return {
    mode, complete: !!b.complete, target_samples: n(b.target_samples) ?? 0,
    db_samples: n(b.db_samples) ?? 0, lux_samples: n(b.lux_samples) ?? 0,
    db_median: n(b.db_median), db_iqr: n(b.db_iqr), lux_median: n(b.lux_median),
  };
}

/** Capture progress 0-1: the dB channel's samples of the target (lux when no audio has arrived). */
export function baselineProgress(b: RoomBaseline | null): number {
  if (!b) return 0;
  if (b.mode === "captured" || (b.complete && b.mode !== "capturing")) return 1;
  if (!b.target_samples) return 0;
  return Math.max(0, Math.min(1, Math.max(b.db_samples, b.lux_samples) / b.target_samples));
}

/** Live sensors without a captured venue baseline: the operator should capture one before judging. */
export function baselineWarning(s: Pick<ValenceState, "source" | "conn" | "status" | "baseline">): boolean {
  return s.source === "live" && s.conn === "open" && s.status?.mode === "live" && s.baseline?.mode !== "captured";
}

/** Provenance badges for the header: MOCK, or (when connected) REPLAY / HYBRID / SIMULATED, else LIVE. */
export function provenanceBadges(s: ValenceState): { label: string; tone: "mock" | "live" | "warn"; title: string }[] {
  if (s.source === "mock") return [{ label: "MOCK", tone: "mock", title: "Generated in the browser, no backend" }];
  if (s.conn !== "open") return [];
  const latest = [...Object.values(s.predictions), ...Object.values(s.vectors)];
  const sim = [...new Set(latest.flatMap((m) => (Array.isArray(m.simulated_fields) ? m.simulated_fields : [])))];
  const out: { label: string; tone: "mock" | "live" | "warn"; title: string }[] = [];
  if (s.status?.mode === "replay" || latest.some((m) => m.source === "replay"))
    out.push({ label: "REPLAY", tone: "warn", title: "Replayed scenario or recording, not live sensors" });
  else if (s.status?.mode === "hybrid")
    out.push({ label: "HYBRID", tone: "warn", title: "One modality live, the other replayed or simulated" });
  if (sim.length) out.push({ label: `SIMULATED: ${sim.join(", ")}`, tone: "warn", title: "Simulated or operator-set data in the 15-minute window" });
  if (!out.length) out.push({ label: "LIVE", tone: "live", title: "Live camera and microphone" });
  return out;
}

function resetData() {
  recentEvents = {};
  set({
    persons: [], mirrored: false, govLog: [], gov: null, vectors: {}, predictions: {}, llm: {}, history: [], status: null, retention: null, evalSummary: null,
    states: CORE_STATES, roomOnly: false, baseline: null, edgeError: null, micError: null, scenarios: [], recordings: [],
  });
}

/** The backend forgot its people (demo reset, mode or source change): drop their cards, boxes and narrator text. */
function clearPeople(history = false) {
  recentEvents = {};
  set({ persons: [], vectors: {}, predictions: {}, llm: {}, ...(history ? { history: [] } : {}) });
}

// ---------- Mock engine ----------
const CYCLE_S = 120;
const PHASE: StateKey[] = ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"];
const GOV_LINES = [
  "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.",
  "[GOVERNANCE] Audio block reduced to dB level and purged. 0 raw samples retained.",
  "[GOVERNANCE] Transcript tokenized for compliment/laughter counts and purged.",
] as const;
const rnd = (a: number, b: number) => a + Math.random() * (b - a);
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

const mock = {
  timer: null as ReturnType<typeof setInterval> | null,
  t: 1000,
  tick: 0,
  counters: { frames: 17000, audio: 3400, transcripts: 200 },
  scenario: "full_demo" as Scenario,
  lastBreakT: 1000 - 3.6 * 3600,
  /** Mock baseline capture: one dB and one lux sample per mock second. */
  stepBaseline() {
    const b = state.baseline;
    if (!b || !baselineFilling(b)) return;
    const n = Math.min(b.target_samples, b.db_samples + 1);
    const done = n >= b.target_samples;
    set({ baseline: { ...b, db_samples: n, lux_samples: n, complete: done, mode: done && b.mode === "capturing" ? "captured" : b.mode,
      db_median: +(57 + rnd(-0.3, 0.3)).toFixed(1), db_iqr: 2.3, lux_median: Math.round(410 + rnd(-5, 5)) } });
  },
  start() {
    if (this.timer) return;
    this.timer = setInterval(() => this.step(), 1000);
  },
  stop() { if (this.timer) clearInterval(this.timer); this.timer = null; },
  stateAt(t: number, pid: number): StateKey {
    const sc = this.scenario;
    if (sc !== "full_demo") return PHASE[["focused", "frustrated_masked", "exhausted", "collaborative_happy"].indexOf(sc)] ?? "Focused";
    const offset = pid === 2 ? 3 : 0; // person 2 lags slightly
    const idx = Math.floor((((t - offset) % CYCLE_S) + CYCLE_S) % CYCLE_S / (CYCLE_S / 4));
    const s = PHASE[idx] ?? "Focused";
    if (pid === 2 && s === "Frustrated_Masked") return "Focused";
    return s;
  },
  vector(t: number, pid: number, s: StateKey, fill: number): StateVectorMsg {
    const base = {
      Focused: { db: 48, lux: 480, slump: 0.12, fists: 0, comp: 0, laugh: 0, wd: 0, val: 0.1 },
      Frustrated_Masked: { db: 72, lux: 220, slump: 0.35, fists: 3, comp: 0, laugh: 0, wd: 0.2, val: 0.62 },
      Exhausted: { db: 55, lux: 140, slump: 0.78, fists: 0, comp: 0, laugh: 0, wd: 0.6, val: -0.2 },
      Collaborative_Happy: { db: 63, lux: 450, slump: 0.1, fists: 0, comp: 3, laugh: 4, wd: 0, val: 0.7 },
    }[s];
    const low = state.sim.low_light, hidden = state.sim.face_occlusion;
    const hrs = Math.max(0, (t - this.lastBreakT) / 3600) + (s === "Exhausted" ? 2 : 0);
    const d = new Date(Date.now() - (this.t - t) * 1000);
    return {
      type: "state_vector", t, person_id: pid,
      ambient_db: +(base.db + rnd(-2, 2)).toFixed(1),
      light_lux: Math.round(low ? rnd(8, 20) : base.lux + rnd(-20, 20)),
      hours_since_break: +hrs.toFixed(1),
      time_of_day: `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`,
      clenched_fists_5m: base.fists + (base.fists ? Math.round(rnd(-1, 1)) : 0),
      compliments_15m: base.comp, laughter_15m: base.laugh,
      posture_slump: +clamp(base.slump + rnd(-0.05, 0.05), 0, 1).toFixed(2),
      head_in_hands: s === "Exhausted" ? +rnd(0.05, 0.2).toFixed(2) : 0,
      social_withdrawal: base.wd > 0.5 ? 1 : 0,
      face_visible: !hidden,
      facial_valence: hidden ? null : +(base.val + rnd(-0.05, 0.05)).toFixed(2),
      window_fill_min: +fill.toFixed(1),
      source: this.scenario === "full_demo" ? "live" : "replay",
      simulated_fields: [...(hidden ? ["face_occlusion"] : []), ...(low ? ["low_light"] : [])],
    };
  },
  prediction(v: StateVectorMsg, s: StateKey): PredictionMsg {
    const conf = rnd(0.78, 0.92);
    const probs = {} as Record<StateKey, number>;
    const others = STATES.filter((x) => x !== s);
    const w = others.map(() => rnd(0.2, 1));
    const ws = w.reduce((a, b) => a + b, 0);
    probs[s] = conf;
    others.forEach((o, i) => (probs[o] = ((1 - conf) * (w[i] ?? 0)) / ws));
    const val = v.facial_valence;
    const db = v.ambient_db ?? 0, lux = v.light_lux ?? 0, hrs = v.hours_since_break ?? 0, slump = v.posture_slump ?? 0;
    const faceLabel: StateKey | null = !v.face_visible || val === null ? null
      : val > 0.4 ? "Collaborative_Happy" : val < -0.1 ? "Exhausted" : "Focused";
    const faceConf = faceLabel ? +rnd(0.6, 0.78).toFixed(2) : null;
    const mask = faceLabel !== null && faceLabel !== s && s === "Frustrated_Masked";
    const L = (name: string, drivers: Driver[]): Layer => ({ name, contribution: +drivers.reduce((a, d) => a + d.shap, 0).toFixed(2), drivers });
    const sign = (good: boolean, m: number) => +((good ? 1 : -1) * m * rnd(0.85, 1.15)).toFixed(2);
    const layers: Layer[] = [
      L("Somatic & Social", [
        { feature: "clenched_fists_count_5m", display: `${v.clenched_fists_5m} clenched-fist events in the last 5 minutes`, value: v.clenched_fists_5m, unit: "events", shap: sign(s === "Frustrated_Masked" ? true : v.clenched_fists_5m === 0, s === "Frustrated_Masked" ? 0.38 : 0.06) },
        { feature: "posture_slump", display: `posture slump ${Math.round(slump * 100)}%`, value: v.posture_slump, unit: "", shap: sign(s === "Exhausted", s === "Exhausted" ? 0.34 : 0.05) },
        { feature: "laughter_15m", display: `${v.laughter_15m} laughs and ${v.compliments_15m} compliments in 15 min`, value: v.laughter_15m, unit: "events", shap: sign(s === "Collaborative_Happy", s === "Collaborative_Happy" ? 0.36 : 0.04) },
      ]),
      L("Ambient Physics", [
        { feature: "ambient_db_mean_15m", display: `average noise of ${Math.round(db)} dB over 15 minutes`, value: v.ambient_db, unit: "dB", shap: sign(s !== "Focused" || db < 55, s === "Frustrated_Masked" ? 0.22 : 0.1) },
        { feature: "light_lux_mean_15m", display: `room light ${lux} lux`, value: v.light_lux, unit: "lux", shap: sign(s === "Exhausted" ? lux < 200 : lux > 300, 0.07) },
      ]),
      L("Time Dynamics", [
        { feature: "hours_since_break", display: `${hrs} hours since the last break`, value: v.hours_since_break, unit: "h", shap: sign(s === "Exhausted" || s === "Frustrated_Masked", s === "Exhausted" ? 0.28 : 0.15) },
      ]),
      L("Facial Surface", [
        faceLabel !== null && val !== null
          ? { feature: "face_last", display: `${val > 0.3 ? "smiling" : val < -0.1 ? "drooping" : "neutral"} expression (valence ${val >= 0 ? "+" : ""}${fmt(val, 2)})`, value: val, unit: "", shap: sign(!mask, mask ? 0.18 : 0.08) }
          : { feature: "face_visible", display: "face not visible — facial layer ignored", value: 0, unit: "", shap: 0 },
      ]),
    ];
    // Same rule as the backend's display_label(): with the core states, Frustrated_Masked reads "Frustrated" unless the mask override fired.
    const disp = s === "Frustrated_Masked" && !mask ? "Frustrated" : STATE_DISPLAY[s];
    const pid = v.person_id;
    const top = layers.slice().sort((a, b) => b.contribution - a.contribution)[0]?.name.toLowerCase() ?? "context";
    const tpl = mask
      ? `Model predicted ${disp} (${Math.round(conf * 100)}%). Person #${pid} appears to smile, but ${v.clenched_fists_5m} clenched-fist events, ${Math.round(db)} dB of sustained noise and ${hrs} hours without a break point to masked stress.`
      : `Model predicted ${disp} (${Math.round(conf * 100)}%) for Person #${pid} based on ${top} signals.`;
    return {
      type: "prediction", event_id: `p${pid}-${String(this.tick).padStart(6, "0")}`, t: v.t, person_id: pid,
      label: s, label_display: disp, confidence: +conf.toFixed(2), probabilities: probs,
      face_only: { label: faceLabel, confidence: faceConf }, mask_override: mask,
      warming_up: v.window_fill_min < 5, layers,
      data_quality: { face_visible_frac: v.face_visible ? 1 : 0, lux: v.light_lux, ood_features: [] },
      explanation: { text: tpl, source: "template" },
      source: v.source, simulated_fields: v.simulated_fields,
    };
  },
  llmText(p: PredictionMsg): string {
    const top = p.layers.flatMap((l) => l.drivers).sort((a, b) => b.shap - a.shap).slice(0, 2).map((d) => d.display);
    if (p.mask_override)
      return `Although Person #${p.person_id}'s face reads as ${stateName(p.face_only.label).toLowerCase()}, the surrounding context tells a different story: ${top.join(" and ")}. Taken together, Valence judges the smile to be a mask over stress and suggests checking in or offering a short break.`;
    return `Person #${p.person_id} most likely feels ${p.label_display.toLowerCase()} (${Math.round(p.confidence * 100)}%). The strongest evidence is ${top.join(", plus ")}. No facial imagery was retained to reach this conclusion.`;
  },
  step() {
    this.t += 1; this.tick += 1;
    this.stepBaseline();
    const t = this.t;
    const ids = state.roomOnly ? [0] : [1, 2];
    const persons: TrackPerson[] = ids.map((id) => {
      const cx = (id === 1 ? 0.3 : 0.7) + Math.sin(t / 6 + id) * 0.02;
      return { id, label: id === 0 ? "Room" : `Person #${id}`, bbox: [+(cx - 0.14).toFixed(3), 0.2, +(cx + 0.14).toFixed(3), 0.95], fist_active: false, slump: false };
    });
    handleMessage({ type: "tracks", t, capture_ts_ms: Date.now(), frame_w: 1280, frame_h: 720, mirrored: true, persons });
    const fill = Math.min(15, 5 + (t - 1000) / 60);
    for (const id of ids) {
      const s = this.stateAt(t, id);
      const v = this.vector(t, id, s, fill);
      handleMessage(v);
      if (this.tick % 2 === 0) {
        const p = this.prediction(v, s);
        handleMessage(p);
        const lat = Math.round(rnd(900, 1900));
        setTimeout(() => handleMessage({ type: "explanation", event_id: p.event_id, person_id: id, text: this.llmText(p), source: Math.random() < 0.1 ? "llm_retry" : "llm", latency_ms: lat }), lat);
      }
    }
    this.counters.frames += 10; this.counters.audio += 2;
    if (this.tick % 7 === 0) this.counters.transcripts += 1;
    if (this.tick % 5 === 0) {
      handleMessage({
        type: "governance", t, tick: this.tick,
        message: GOV_LINES[Math.floor(this.tick / 5) % 3] ?? GOV_LINES[0],
        frames_decoded: this.counters.frames, frames_purged: this.counters.frames, audio_blocks_purged: this.counters.audio,
        transcripts_purged: this.counters.transcripts, violations: 0, live_raw_buffers: 0,
        audit_hash: null, retention_s: 86400, purge_gap: 0,
      });
    }
    handleMessage({
      type: "status", mode: this.scenario === "full_demo" ? "live" : "replay", camera: true, mic: true,
      fps: +rnd(9.2, 10).toFixed(1), loop_latency_ms: Math.round(rnd(70, 95)), narrator: "ready",
      offline: true, uptime_s: t - 1000 + 1820, degraded: [], purge_violation: false,
    });
  },
  backfill() {
    const start = this.t - HISTORY_WINDOW_S;
    const h: HistoryPoint[] = [];
    for (let t = start; t <= this.t; t += 10) {
      for (const id of [1, 2]) {
        const s = this.stateAt(t, id);
        const v = this.vector(t, id, s, Math.min(15, (t - start) / 60));
        h.push({ t, person_id: id, label: s, confidence: rnd(0.75, 0.9), ambient_db: v.ambient_db, light_lux: v.light_lux, hours_since_break: v.hours_since_break, posture_slump: v.posture_slump, facial_valence: v.facial_valence, window_fill_min: v.window_fill_min });
      }
    }
    set({ history: h });
  },
  command(c: Command) {
    if (c.kind === "control") {
      if (c.action === "start") { this.start(); set({ running: "running" }); }
      else { this.stop(); set({ running: c.action === "pause" ? "paused" : "stopped" }); }
    } else if (c.kind === "break") this.lastBreakT = this.t;
    else if (c.kind === "replay" || c.kind === "replay_recording" || c.kind === "live") {
      this.scenario = c.kind === "replay" ? c.scenario : "full_demo";
      clearPeople(); this.start(); set({ running: "running" });
    } else if (c.kind === "demo_reset") clearPeople();
    else if (c.kind === "mode") { clearPeople(); set({ roomOnly: c.room_only }); }
    else if (c.kind === "baseline")
      set({ baseline: { mode: "capturing", complete: false, target_samples: Math.round(c.seconds), db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null, lux_median: null } });
    else if (c.kind === "baseline_reset") set({ baseline: { ...MOCK_BASELINE, mode: "auto", complete: false, db_samples: 0, lux_samples: 0 } });
    else set({ sim: { ...state.sim, [c.which]: c.enabled } });
  },
};

export const MOCK_EVAL: EvalSummary = {
  accuracy_vs_history: [0, 1, 3, 5, 10, 15].map((m) => ({ minutes: m, context: +(0.52 + 0.36 * (1 - Math.exp(-m / 4))).toFixed(2), face_only: 0.49 })),
  robustness: [
    { condition: "Normal", context: 0.87, face_only: 0.61 },
    { condition: "Face hidden", context: 0.81, face_only: 0.25 },
    { condition: "Low light", context: 0.83, face_only: 0.34 },
  ],
  confusion: [
    { name: "Context model", labels: STATES, matrix: [[46, 2, 1, 1], [3, 41, 4, 2], [1, 3, 44, 2], [1, 2, 1, 46]] },
    { name: "Face-only baseline", labels: STATES, matrix: [[30, 5, 9, 6], [4, 12, 6, 28], [8, 4, 31, 7], [4, 3, 2, 41]] },
  ],
  calibration: [0.1, 0.3, 0.5, 0.7, 0.9].map((p) => ({ predicted: p, observed: +clamp(p + rnd(-0.05, 0.05), 0, 1).toFixed(2) })),
  metrics: [
    { model: "Valence context", accuracy: 0.87, macro_f1: 0.86, ece: 0.04 },
    { model: "Face-only baseline", accuracy: 0.57, macro_f1: 0.51, ece: 0.14 },
  ],
};
// Same shape as GET /api/retention.
const MOCK_RETENTION = {
  raw_media_stored: 0,
  held: [
    { what: "rolling state vector (numeric TickRecords)", where: "RAM", how_long: "15 minutes" },
    { what: "recent messages for /api/history", where: "RAM", how_long: "about 20 minutes" },
    { what: "audit log: labels, confidences, SHAP layer sums, operator actions", where: "var/audit (mock)", how_long: "deleted at session end; TTL 86400 s after a crash" },
  ],
  never_kept: ["video frames (zeroed and released each frame)", "audio blocks (zeroed each block)", "transcripts (Vosk grammar: only lexicon hits exist)", "pose keypoints (stay inside perception)", "face images or embeddings"],
};
const MOCK_BASELINE: RoomBaseline = { mode: "captured", complete: true, target_samples: 90, db_samples: 90, lux_samples: 90, db_median: 57.2, db_iqr: 2.3, lux_median: 410 };
const MOCK_SCENARIOS: ScenarioInfo[] = ["full_demo", "focused", "frustrated_masked", "exhausted", "collaborative_happy"].map((name) => ({ name }));
const MOCK_RECORDINGS: RecordingInfo[] = [{ name: "golden_demo" }, { name: "capture_typing" }];

// ---------- Live transport ----------
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let backoff = 1000;
let liveActive = false;

const wsUrl = (base: string) => base.replace(/^http/, "ws").replace(/\/$/, "") + "/ws/events";
const api = (path: string) => state.baseUrl.replace(/\/$/, "") + path;
export const figureUrl = (name: string) => api(`/api/eval/figures/${name}.png`);

async function getJson(path: string): Promise<unknown> {
  const r = await fetch(api(path));
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

/** Latest card per person from GET /api/state, so the console is filled before the next tick arrives. */
export function applyStateSnapshot(j: unknown) {
  if (!j || typeof j !== "object") return;
  const { people, status } = j as { people?: Record<string, { state_vector?: StateVectorMsg; prediction?: PredictionMsg }>; status?: StatusMsg };
  for (const [key, entry] of Object.entries(people && typeof people === "object" ? people : {})) {
    // /api/state can keep the other mode's people after a room-only toggle: take person 0 only in room-only mode.
    if ((Number(key) === 0) !== state.roomOnly) continue;
    if (entry?.state_vector && typeof entry.state_vector === "object") handleMessage({ ...entry.state_vector, type: "state_vector" });
    if (entry?.prediction && typeof entry.prediction === "object") handleMessage({ ...entry.prediction, type: "prediction" });
  }
  if (status && typeof status === "object" && !state.status) handleMessage({ ...status, type: "status" });
}

/** /api/health: room baseline plus the edge / mic start errors. */
export function applyHealth(j: unknown) {
  if (!j || typeof j !== "object") return;
  const h = j as { room_baseline?: unknown; edge_error?: unknown; mic_error?: unknown };
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  set({ baseline: parseBaseline(h.room_baseline) ?? state.baseline, edgeError: str(h.edge_error), micError: str(h.mic_error) });
}

// Room-baseline poll: every 2 s while a capture (or the automatic baseline) is still filling, else health every 10 s.
const BASELINE_POLL_MS = 2000;
const HEALTH_EVERY = 5;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let pollCount = 0;
export const baselineFilling = (b: RoomBaseline | null) => !!b && (b.mode === "capturing" || !b.complete);
async function pollOnce() {
  pollCount += 1;
  try {
    if (baselineFilling(state.baseline) && pollCount % HEALTH_EVERY !== 0) {
      const b = parseBaseline(await getJson("/api/room/baseline"));
      if (b) set({ baseline: b });
    } else if (pollCount % HEALTH_EVERY === 0) applyHealth(await getJson("/api/health"));
  } catch { /* backend restarting: the WebSocket shows the connection state */ }
}
function startPoll() {
  stopPoll();
  pollCount = 0;
  pollTimer = setInterval(() => { if (liveActive && state.conn === "open") void pollOnce(); }, BASELINE_POLL_MS);
}
function stopPoll() { if (pollTimer) clearInterval(pollTimer); pollTimer = null; }

async function loadRest() {
  try { set({ states: validStates(await getJson("/api/states")) ?? state.states }); } catch { /* hello carries them too */ }
  try { applyStateSnapshot(await getJson("/api/state")); } catch { /* ignore */ }
  try { applyHealth(await getJson("/api/health")); } catch { /* ignore */ }
  try { const j = await getJson("/api/scenarios"); set({ scenarios: Array.isArray(j) ? (j as ScenarioInfo[]).filter((x) => typeof x?.name === "string") : [] }); } catch { /* ignore */ }
  try { const j = await getJson("/api/recordings"); set({ recordings: Array.isArray(j) ? (j as RecordingInfo[]).filter((x) => typeof x?.name === "string") : [] }); } catch { /* ignore */ }
  try {
    const j = await getJson("/api/history?minutes=15");
    const pts = (Array.isArray(j) ? j : []) as HistoryPoint[];
    set({ history: [...pts, ...state.history].sort((a, b) => a.t - b.t) });
  } catch { /* backend may not be up yet */ }
  try { set({ retention: (await getJson("/api/retention")) as Record<string, unknown> }); } catch { /* ignore */ }
  try { set({ evalSummary: (await getJson("/api/eval/summary")) as EvalSummary }); } catch { /* ignore */ }
}

function connectWs() {
  if (!liveActive) return;
  set({ conn: "connecting" });
  try { ws = new WebSocket(wsUrl(state.baseUrl)); } catch { return scheduleReconnect(); }
  ws.onopen = () => { backoff = 1000; set({ conn: "open" }); loadRest(); };
  ws.onmessage = (e) => { try { handleMessage(JSON.parse(e.data) as ServerMessage); } catch { /* bad frame */ } };
  ws.onclose = () => { set({ conn: "closed" }); scheduleReconnect(); };
  ws.onerror = () => ws?.close();
}
function scheduleReconnect() {
  if (!liveActive) return;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connectWs, backoff);
  backoff = Math.min(backoff * 2, 10000);
}
function stopLive() {
  liveActive = false;
  stopPoll();
  if (reconnectTimer) clearTimeout(reconnectTimer);
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
}

// ---------- Public API ----------
export function setSource(source: "mock" | "live", baseUrl = state.baseUrl) {
  const base = cleanBase(baseUrl) || state.baseUrl;
  stopLive(); mock.stop(); resetData();
  set({ source, baseUrl: base, running: "running" });
  if (source === "mock") {
    set({ conn: "mock", retention: MOCK_RETENTION, evalSummary: MOCK_EVAL, baseline: { ...MOCK_BASELINE }, scenarios: MOCK_SCENARIOS, recordings: MOCK_RECORDINGS });
    mock.backfill(); mock.start();
  } else {
    if (validBase(base)) saveBase(base);
    liveActive = true; backoff = 1000; connectWs(); startPoll();
  }
}

let booted = false;
export function boot() { if (!booted) { booted = true; setSource("mock", resolveBaseUrl()); } }

const replayBody = (c: Partial<ReplayOpts>, loopDefault: boolean) => ({ speed: c.speed ?? 1, loop: c.loop ?? loopDefault });

/** Path and JSON body of a command (docs/API.md section 7.4). */
export function describe(c: Command): readonly [string, Record<string, unknown> | undefined] {
  switch (c.kind) {
    case "control": return [`/api/control/${c.action}`, undefined];
    case "break": return ["/api/events/break", undefined];
    case "replay": return [`/api/replay/${encodeURIComponent(c.scenario)}`, replayBody(c, false)];
    case "replay_recording": return [`/api/replay_recording/${encodeURIComponent(c.name)}`, replayBody(c, true)];
    case "live": return ["/api/live", undefined];
    case "demo_reset": return ["/api/demo/reset", undefined];
    case "mode": return ["/api/mode", { room_only: c.room_only }];
    case "baseline": return ["/api/room/baseline", { seconds: c.seconds }];
    case "baseline_reset": return ["/api/room/baseline/reset", undefined];
    case "sim": return [`/api/sim/${c.which}`, { enabled: c.enabled }];
  }
}

/** The fields of a command response the console reads. */
export interface CommandReply { ok?: unknown; detail?: unknown; edge_error?: unknown; mic_error?: unknown; room_only?: unknown }

/** Store update for a successful command response (`j` is the parsed JSON body, if any). */
export function applyCommandResult(c: Command, j: CommandReply) {
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  switch (c.kind) {
    case "sim": return set({ sim: { ...state.sim, [c.which]: c.enabled } });
    case "control":
      if (c.action === "stop") clearPeople();
      return set({ running: c.action === "start" ? "running" : c.action === "pause" ? "paused" : "stopped" });
    case "live":
      clearPeople(true);
      return set({ running: "running", edgeError: str(j.edge_error), micError: str(j.mic_error) });
    case "replay":
    case "replay_recording":
      clearPeople(true);
      return set({ running: "running" });
    case "demo_reset": return clearPeople();
    case "mode":
      clearPeople();
      return set({ roomOnly: typeof j.room_only === "boolean" ? j.room_only : c.room_only });
    case "baseline":
    case "baseline_reset": {
      const b = parseBaseline(j);
      return b ? set({ baseline: b }) : undefined;
    }
  }
}

export async function sendCommand(c: Command) {
  const [path, body] = describe(c);
  if (state.source === "mock") {
    mock.command(c);
    return set({ lastCommand: { text: `POST ${path} (mock)`, ok: true } });
  }
  try {
    const init: RequestInit = { method: "POST" };
    if (body) { init.headers = { "Content-Type": "application/json" }; init.body = JSON.stringify(body); }
    const r = await fetch(api(path), init);
    let j: CommandReply = {};
    try { const parsed: unknown = await r.json(); if (parsed && typeof parsed === "object") j = parsed as CommandReply; } catch { /* no JSON body */ }
    if (r.ok) applyCommandResult(c, j);
    const ok = r.ok && j.ok !== false;
    const detail = typeof j.detail === "string" ? ` · ${j.detail}` : !ok && typeof j.edge_error === "string" ? ` · ${j.edge_error}` : "";
    set({ lastCommand: { text: `POST ${path} → ${r.status}${detail}`, ok } });
  } catch {
    set({ lastCommand: { text: `POST ${path} failed (backend unreachable)`, ok: false } });
  }
}
