/**
 * Session recorder for the real-data benchmark (docs/PROTOCOL.md, format `valence-session/1`).
 *
 * One JSON line per perception observation (exactly `obsToDict`, the Python edge's `obs_to_dict`, without the `raw`
 * debug block), per engine tick (the wall/mono it stepped with and what each prediction showed), per marker (protocol
 * blocks, operator controls) and per self-report tap. Features only: counts, levels, flags, normalized boxes, face-cue
 * scalars and model outputs; never frames, audio, landmarks or transcripts. Held in this tab's memory until Clear or
 * the tab closes (Download keeps it); nothing is uploaded. `valence/eval/benchmark.py` replays and scores the file.
 */
import { obsToDict, type Obs } from "@/perception/types";
import type { ServerMessage } from "./valence";

export const SESSION_FORMAT = "valence-session/1";
/** A recording stops itself past this size (about 40 minutes of one person at 7 fps is ~25 MB). */
export const MAX_SESSION_BYTES = 64 * 1024 * 1024;

export interface SessionHeader {
  session_id: string; started_wall: number; started_mono: number; tz: string | null; feature_version: string;
  emotion_model: string | null; protocol: string | null;
}
export interface Stamp { wall: number; mono: number }
export type MarkerKind = "block_start" | "block_end" | "protocol_start" | "protocol_end" | "protocol_abort" | "control";
export interface Marker {
  kind: MarkerKind; block_id?: string; label?: string; scored?: boolean; calm?: boolean;
  expect?: { card: readonly string[]; map: readonly string[] }; action?: string; reason?: string;
}
export interface SessionStats {
  active: boolean; full: boolean; lines: number; bytes: number; observations: number; ticks: number;
  markers: number; taps: number; started_wall: number | null; session_id: string | null;
}

export function newSessionId(now: Date, rnd = Math.random().toString(36).slice(2, 6)): string {
  return `vs-${now.toISOString().slice(0, 10)}-${rnd}`;
}

export class SessionRecorder {
  private lines: string[] = [];
  private bytes = 0;
  private counts = { observations: 0, ticks: 0, markers: 0, taps: 0 };
  private header: SessionHeader | null = null;
  private recording = false;
  private full = false;

  /** Begin a new recording (drops any previous one) with its header line. */
  start(h: SessionHeader): void {
    this.clear();
    this.header = h;
    this.recording = true;
    this.push({ type: "session", format: SESSION_FORMAT, app: "valence-web", ...h });
  }
  /** Stop adding lines; what was recorded stays until clear(). */
  stop(): void { this.recording = false; }
  clear(): void {
    this.lines = []; this.bytes = 0; this.header = null; this.recording = false; this.full = false;
    this.counts = { observations: 0, ticks: 0, markers: 0, taps: 0 };
  }
  get active(): boolean { return this.recording; }

  /** A perception observation (the perception module's Obs; other shapes are ignored). */
  obs(o: unknown): void {
    if (!this.recording) return;
    const kind = (o as { type?: unknown } | null)?.type;
    if (kind !== "video" && kind !== "audio") return;
    const d = obsToDict(o as Obs);
    if (Array.isArray(d["persons"])) for (const p of d["persons"]) if (p && typeof p === "object" && !Array.isArray(p)) delete p["raw"];
    if (this.push(d)) this.counts.observations += 1;
  }

  /** One engine tick: the wall/mono it stepped with and what each prediction showed. */
  tick(wall: number, mono: number, msgs: readonly ServerMessage[]): void {
    if (!this.recording) return;
    const predictions = msgs.flatMap((m) => m.type !== "prediction" ? [] : [{
      person_id: m.person_id, label: m.label, label_display: m.label_display, confidence: m.confidence,
      warming_up: m.warming_up, mask_override: m.mask_override,
      face_only: { label: m.face_only?.label ?? null, confidence: m.face_only?.confidence ?? null },
      affect: m.affect ? { label: m.affect.label, valence: m.affect.valence, arousal: m.affect.arousal } : null,
    }]);
    if (this.push({ type: "tick", wall, mono, predictions })) this.counts.ticks += 1;
  }

  marker(at: Stamp, m: Marker): void {
    if (!this.recording) return;
    if (this.push({ type: "marker", wall: at.wall, mono: at.mono, ...m })) this.counts.markers += 1;
  }

  /** A self-report tap (as selfReportsJsonl writes it) with the source's clocks. */
  tap(at: Stamp, tap: object): void {
    if (!this.recording) return;
    if (this.push({ type: "self_report", ...tap, wall: at.wall, mono: at.mono })) this.counts.taps += 1;
  }

  jsonl(): string { return this.lines.length ? this.lines.join("\n") + "\n" : ""; }
  fileName(): string { return `valence-session-${this.header?.session_id.replace(/^vs-/, "") ?? "empty"}.jsonl`; }
  stats(): SessionStats {
    return { active: this.recording, full: this.full, lines: this.lines.length, bytes: this.bytes, ...this.counts,
      started_wall: this.header?.started_wall ?? null, session_id: this.header?.session_id ?? null };
  }

  private push(line: object): boolean {
    const s = JSON.stringify(line);
    if (this.bytes + s.length + 1 > MAX_SESSION_BYTES) { this.full = true; this.recording = false; return false; }
    this.lines.push(s);
    this.bytes += s.length + 1;
    return true;
  }
}
