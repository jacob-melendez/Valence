/**
 * In-browser source: the whole Valence pipeline runs in this page (Valence Web).
 * Camera and microphone are opened here, turned into observations by the perception module
 * (frontend/src/perception), aggregated and scored by the TypeScript engine (frontend/src/engine), and
 * dropped. Nothing is recorded or sent anywhere (the deployed CSP has connect-src 'self').
 * It emits the same messages the backend WebSocket does (docs/API.md section 3), so every panel works unchanged.
 */
import type { Command, HelloMsg, RoomBaseline, ServerMessage, StatusMsg, TrackPerson } from "./valence";

// ---------- Seams (implemented by frontend/src/engine and frontend/src/perception) ----------
/** A person in a video observation (valence/contracts.py PersonObs; only the fields this module reads). */
export interface ObsPerson {
  track_id: number; bbox: readonly [number, number, number, number]; fist_active?: boolean; slump?: boolean;
}
export interface VideoObsLike { t: number; wall: number; frame_w: number; frame_h: number; mirrored: boolean; persons: readonly ObsPerson[] }
export interface AudioObsLike { t: number; wall: number; db_spl: number }
export type ObsLike = VideoObsLike | AudioObsLike;
export const isVideo = (o: ObsLike): o is VideoObsLike => "persons" in o;

export interface PerceptionCounters { frames_decoded: number; frames_purged: number; audio_blocks_purged: number; fps?: number }
/** frontend/src/perception's createPerception() / createMockPerception() result. */
export interface PerceptionLike {
  start(stream: MediaStream | null): void | Promise<void>;
  stop(): void;
  onObservation(cb: (o: ObsLike) => void): void | (() => void);
  counters(): PerceptionCounters;
}

/** The engine side: the aggregator + runtime of valence/runtime, ported (frontend/src/engine). */
export interface BrowserEngine {
  states: { id: string; display: string; tier: string }[];
  featureVersion: string;
  push(obs: ObsLike): void;
  /** One 5 s tick: flush the aggregator and run the runtime step (governance, state_vector, prediction). */
  tick(wall: number, mono: number, counters: PerceptionCounters): ServerMessage[];
  setRoomOnly(roomOnly: boolean): void;
  setOverlay(which: "face_occlusion" | "low_light", enabled: boolean): void;
  logBreak(wall: number): void;
  startBaseline(seconds: number | null, mode: "capturing" | "auto"): void;
  baseline(): RoomBaseline;
  resetPeople(): void;
}

export interface BrowserSourceDeps {
  createEngine: () => Promise<BrowserEngine>;
  createPerception: () => Promise<PerceptionLike>;
  emit: (m: ServerMessage) => void;
  onBaseline: (b: RoomBaseline) => void;
  onError: (what: "camera" | "mic" | "engine", detail: string) => void;
  now?: () => number; // epoch ms
  mono?: () => number; // ms
}

// ---------- Shared camera stream (the preview reuses it instead of opening the camera twice) ----------
let stream: MediaStream | null = null;
const streamListeners = new Set<(s: MediaStream | null) => void>();
const setStream = (s: MediaStream | null) => { stream = s; streamListeners.forEach((l) => l(s)); };
export const getBrowserStream = () => stream;
export function onBrowserStream(l: (s: MediaStream | null) => void): () => void {
  streamListeners.add(l);
  return () => { streamListeners.delete(l); };
}

/** Open camera (16:9, as the backend edge asks) and microphone (raw: no echo cancellation, noise suppression or AGC,
 *  so levels mean what the Python edge's levels mean). Either may fail independently. */
async function openMedia(onError: BrowserSourceDeps["onError"]): Promise<MediaStream | null> {
  const md = typeof navigator !== "undefined" ? navigator.mediaDevices : undefined;
  if (!md?.getUserMedia) { onError("camera", "getUserMedia is not available (needs HTTPS or localhost)"); return null; }
  const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 };
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  try { return await md.getUserMedia({ video, audio }); } catch { /* try each on its own */ }
  const out = new MediaStream();
  try { (await md.getUserMedia({ video })).getTracks().forEach((t) => out.addTrack(t)); } catch (e) { onError("camera", String(e)); }
  try { (await md.getUserMedia({ audio })).getTracks().forEach((t) => out.addTrack(t)); } catch (e) { onError("mic", String(e)); }
  return out.getTracks().length ? out : null;
}

// ---------- The source ----------
const TICK_MS = 5000;
const TRACKS_MIN_MS = 100; // tracks at most 10 Hz, like the backend
const PARTICIPANT_REL_HEIGHT = 0.55; // valence/config.py participant_rel_height: boxes shown in `tracks`

export class BrowserSource {
  private engine: BrowserEngine | null = null;
  private perception: PerceptionLike | null = null;
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private statusTimer: ReturnType<typeof setInterval> | null = null;
  private running: "running" | "paused" | "stopped" = "running";
  private startedAt = 0;
  private lastVideoAt = 0;
  private lastAudioAt = 0;
  private lastTracksAt = 0;
  private loopLatency = 0;
  private roomOnly = false;
  private generation = 0;
  private readonly now: () => number;
  private readonly mono: () => number;

  constructor(private readonly deps: BrowserSourceDeps) {
    this.now = deps.now ?? (() => Date.now());
    this.mono = deps.mono ?? (() => performance.now());
  }

  /** Load the engine, open camera + mic, start perception and the tick/status timers. */
  async start(): Promise<void> {
    const gen = ++this.generation;
    this.startedAt = this.now();
    try { this.engine = await this.deps.createEngine(); } catch (e) { this.deps.onError("engine", String(e)); return; }
    if (gen !== this.generation) return;
    this.deps.emit(this.hello());
    await this.openSensors(gen);
    if (gen !== this.generation) return;
    this.tickTimer = setInterval(() => this.tick(), TICK_MS);
    this.statusTimer = setInterval(() => this.deps.emit(this.status()), 1000);
    this.deps.onBaseline(this.engine.baseline());
  }

  private async openSensors(gen: number) {
    const media = await openMedia(this.deps.onError);
    if (gen !== this.generation) { media?.getTracks().forEach((t) => t.stop()); return; }
    setStream(media);
    try {
      this.perception = await this.deps.createPerception();
      this.perception.onObservation((o) => this.onObs(o));
      await this.perception.start(media);
    } catch (e) { this.deps.onError("camera", `perception failed: ${String(e)}`); }
    // A new live session starts a new automatic venue baseline (valence/runtime/engine.py go_live).
    this.engine?.startBaseline(null, "auto");
  }

  private closeSensors() {
    this.perception?.stop();
    this.perception = null;
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
  }

  /** Stop everything: timers, perception, camera and mic (the kill switch and source changes). */
  stop() {
    this.generation += 1;
    if (this.tickTimer) clearInterval(this.tickTimer);
    if (this.statusTimer) clearInterval(this.statusTimer);
    this.tickTimer = this.statusTimer = null;
    this.closeSensors();
    this.engine = null;
  }

  private onObs(o: ObsLike) {
    if (this.running !== "running" || !this.engine) return;
    this.engine.push(o);
    const t = this.now();
    if (!isVideo(o)) { this.lastAudioAt = t; return; }
    this.lastVideoAt = t;
    if (t - this.lastTracksAt < TRACKS_MIN_MS) return;
    this.lastTracksAt = t;
    // Same rule as valence/runtime/engine.py push_obs: show boxes at least 0.55x the tallest one.
    const tallest = Math.max(0, ...o.persons.map((p) => p.bbox[3] - p.bbox[1]));
    const persons: TrackPerson[] = o.persons
      .filter((p) => p.bbox[3] - p.bbox[1] >= PARTICIPANT_REL_HEIGHT * tallest)
      .map((p) => ({ id: p.track_id, label: `Person #${p.track_id}`, bbox: [...p.bbox] as TrackPerson["bbox"],
        fist_active: !!p.fist_active, slump: !!p.slump }));
    this.deps.emit({ type: "tracks", t: o.wall, capture_ts_ms: Math.round(o.wall * 1000), frame_w: o.frame_w,
      frame_h: o.frame_h, mirrored: o.mirrored, persons });
  }

  private tick() {
    if (this.running !== "running" || !this.engine) return;
    const t0 = this.mono();
    const counters = this.perception?.counters() ?? { frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0 };
    let msgs: ServerMessage[] = [];
    try { msgs = this.engine.tick(this.now() / 1000, this.mono() / 1000, counters); } catch (e) { this.deps.onError("engine", String(e)); }
    this.loopLatency = this.mono() - t0;
    msgs.forEach((m) => this.deps.emit(m));
    this.deps.onBaseline(this.engine.baseline());
  }

  hello(): HelloMsg {
    return { type: "hello", contract_version: "1.0.0", mode: "live", room_only: this.roomOnly,
      states: this.engine?.states ?? [], feature_version: this.engine?.featureVersion ?? "", frame_w: 0, frame_h: 0, mirrored: false };
  }

  status(): StatusMsg {
    const t = this.now(), c = this.perception?.counters();
    const camera = this.running === "running" && t - this.lastVideoAt < 3000;
    const mic = this.running === "running" && t - this.lastAudioAt < 3000;
    const degraded = [...(camera ? [] : ["camera"]), ...(mic ? [] : ["mic"])];
    return { type: "status", mode: "live", camera, mic, fps: Math.round((c?.fps ?? 0) * 10) / 10,
      loop_latency_ms: Math.round(this.loopLatency * 10) / 10,
      // The LLM narrator is desktop-only; the browser always uses the template explanation.
      narrator: "template", offline: false, uptime_s: Math.round((t - this.startedAt) / 1000),
      degraded: this.running === "running" ? degraded : [], purge_violation: false };
  }

  /** Operator commands, handled locally (no backend). Returns a short result line for the console. */
  command(c: Command): { ok: boolean; text: string } {
    const e = this.engine;
    switch (c.kind) {
      case "control":
        if (c.action === "start" && this.running !== "running") {
          if (this.running === "stopped") void this.openSensors(this.generation);
          this.running = "running";
        } else if (c.action === "pause") this.running = "paused";
        else if (c.action === "stop") { this.running = "stopped"; this.closeSensors(); e?.resetPeople(); }
        return { ok: true, text: `control ${c.action} (in-browser)` };
      case "break": e?.logBreak(this.now() / 1000); return { ok: true, text: "break logged (in-browser)" };
      case "mode": this.roomOnly = c.room_only; e?.setRoomOnly(c.room_only); e?.resetPeople(); return { ok: true, text: `room-only ${c.room_only}` };
      case "sim": e?.setOverlay(c.which, c.enabled); return { ok: true, text: `sim ${c.which} ${c.enabled}` };
      case "baseline": e?.startBaseline(c.seconds, "capturing"); if (e) this.deps.onBaseline(e.baseline()); return { ok: true, text: `baseline ${c.seconds} s (in-browser)` };
      case "baseline_reset": e?.startBaseline(null, "auto"); if (e) this.deps.onBaseline(e.baseline()); return { ok: true, text: "baseline reset (in-browser)" };
      case "demo_reset": e?.resetPeople(); return { ok: true, text: "people reset (in-browser)" };
      case "live": return { ok: true, text: "already live (in-browser)" };
      case "replay":
      case "replay_recording":
        return { ok: false, text: "replays need the desktop backend (LIVE source)" };
    }
  }
}
