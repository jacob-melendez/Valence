import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { evalSummaryOf } from "@/engine/browser-engine";
import type { EmotionModel } from "@/engine/emotion-model";
import {
  BrowserSource, NO_MEDIA_API, cspSealed, getBrowserStream, mediaErrorText, plainEval, sealsConnect,
  type BrowserEngine, type BrowserSourceDeps, type BrowserStatus, type ObsLike, type PerceptionLike,
} from "@/lib/browser-source";
import type { EvalMetricRow, RoomBaseline, ServerMessage } from "@/lib/valence";

import metrics from "../../../reports/metrics.json";

const BASELINE: RoomBaseline = { mode: "auto", complete: false, target_samples: 90, db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null, lux_median: null };

function fakeEngine(log: string[]): BrowserEngine {
  let ticks = 0;
  return {
    states: [{ id: "Focused", display: "Focused", tier: "P0" }],
    featureVersion: "m2_lgbm@test",
    push: (o) => {
      log.push("persons" in o ? "video" : "audio");
      if (!("persons" in o)) return null;
      // the engine's push_obs: participants' boxes (>= 0.55x the tallest)
      const tallest = Math.max(...o.persons.map((p) => p.bbox[3] - p.bbox[1]));
      return { type: "tracks", t: o.wall, capture_ts_ms: Math.round(o.wall * 1000), frame_w: o.frame_w, frame_h: o.frame_h,
        mirrored: o.mirrored, persons: o.persons.filter((p) => p.bbox[3] - p.bbox[1] >= 0.55 * tallest)
          .map((p) => ({ id: p.track_id, label: `Person #${p.track_id}`, bbox: [...p.bbox] as [number, number, number, number], fist_active: false, slump: !!p.slump })) };
    },
    tick: (wall) => {
      ticks += 1;
      return [{ type: "governance", t: wall, tick: ticks, message: "", frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0,
        transcripts_purged: 0, violations: 0, live_raw_buffers: 0, audit_hash: null, retention_s: 0, purge_gap: 0 }];
    },
    setRoomOnly: (v) => { log.push(`room_only ${v}`); },
    setOverlay: (w, e) => { log.push(`overlay ${w} ${e}`); },
    logBreak: () => { log.push("break"); },
    startBaseline: (s, m) => { log.push(`baseline ${s} ${m}`); },
    baseline: () => BASELINE,
    resetPeople: (keepRoom = true) => { log.push(`reset keep_room=${keepRoom}`); },
    audit: (action) => { log.push(`audit ${action}`); },
  };
}

type FakePerception = PerceptionLike & { emit: (o: ObsLike) => void; started: boolean; stopped: boolean; startedWith: unknown };
function fakePerception(opts: { start?: () => void | Promise<void>; status?: () => object } = {}): FakePerception {
  let cb: ((o: ObsLike) => void) | null = null;
  const p: FakePerception = {
    started: false, stopped: false, startedWith: undefined,
    start: (s) => { p.started = true; p.startedWith = s; return opts.start?.(); },
    stop: () => { p.stopped = true; },
    onObservation: (f: (o: ObsLike) => void) => { cb = f; },
    counters: () => ({ frames_decoded: 3, frames_purged: 3, audio_blocks_purged: 1, fps: 9.6 }),
    ...(opts.status ? { status: opts.status } : {}),
    emit: (o: ObsLike) => cb?.(o),
  };
  return p;
}

const video = (wall: number, h: number[]): ObsLike => ({ t: wall, wall, frame_w: 1280, frame_h: 720, mirrored: false,
  persons: h.map((height, i) => ({ track_id: i + 1, bbox: [0.1, 0.1, 0.3, 0.1 + height] as const, slump: i === 0 })) });

// ---------- Fake camera and microphone (MediaStream tracks with a readyState, a getUserMedia the test answers) ----------
class FakeTrack {
  readyState: "live" | "ended" = "live";
  constructor(readonly kind: "video" | "audio") {}
  stop() { this.readyState = "ended"; }
}
class FakeStream {
  constructor(readonly tracks: FakeTrack[] = []) {}
  getTracks() { return this.tracks; }
  getVideoTracks() { return this.tracks.filter((t) => t.kind === "video"); }
  getAudioTracks() { return this.tracks.filter((t) => t.kind === "audio"); }
  addTrack(t: FakeTrack) { this.tracks.push(t); }
}
const domErr = (name: string) => Object.assign(new Error(`${name}: denied`), { name });
interface Ask { c: MediaStreamConstraints; grant: () => void; deny: (name: string) => void }
/** getUserMedia that waits for the test (`hold`), or answers at once with both devices (default). */
function installMedia(answer: "grant" | "hold" | ((c: MediaStreamConstraints) => FakeTrack[] | Error) = "grant") {
  const calls: MediaStreamConstraints[] = [], asks: Ask[] = [], tracks: FakeTrack[] = [];
  const make = (c: MediaStreamConstraints) => {
    const t = [...(c.video ? [new FakeTrack("video")] : []), ...(c.audio ? [new FakeTrack("audio")] : [])];
    tracks.push(...t);
    return new FakeStream(t);
  };
  const getUserMedia = (c: MediaStreamConstraints) => {
    calls.push(c);
    if (answer === "grant") return Promise.resolve(make(c));
    if (typeof answer === "function") {
      const r = answer(c);
      if (r instanceof Error) return Promise.reject(r);
      tracks.push(...r);
      return Promise.resolve(new FakeStream(r));
    }
    return new Promise((res, rej) => { asks.push({ c, grant: () => res(make(c)), deny: (n) => rej(domErr(n)) }); });
  };
  vi.stubGlobal("MediaStream", FakeStream);
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
  return { calls, asks, tracks, live: () => tracks.filter((t) => t.readyState === "live").map((t) => t.kind) };
}
const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
function deferred<T>() {
  let resolve!: (v: T) => void, reject!: (e: unknown) => void;
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}

function source(over: Partial<BrowserSourceDeps> & { perceptions?: FakePerception[] } = {}) {
  const log: string[] = [], out: ServerMessage[] = [], errors: string[] = [];
  const perceptions = over.perceptions ?? [];
  const src = new BrowserSource({
    createEngine: async () => fakeEngine(log),
    createPerception: async () => { const p = fakePerception(); perceptions.push(p); return p; },
    emit: (m) => out.push(m), onBaseline: () => {}, onError: (w, d) => errors.push(`${w}: ${d}`),
    ...over,
  });
  const lastStatus = () => out.filter((m): m is BrowserStatus => m.type === "status").at(-1);
  return { src, log, out, errors, perceptions, lastStatus };
}

describe("in-browser source", () => {
  beforeEach(() => { vi.useRealTimers(); });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    delete (navigator as { mediaDevices?: unknown }).mediaDevices;
  });

  it("emits status from the start, then hello, tracks for visible people, ticks every 5 s and status every second", async () => {
    vi.useFakeTimers();
    const media = installMedia();
    const perception = fakePerception();
    const { src, out, log, errors } = source({ createPerception: async () => perception, now: () => Date.now(), mono: () => Date.now() });
    await src.start();
    expect(out[0]).toMatchObject({ type: "status", loading: "loading the engine and models" }); // before the engine
    expect(out[1]?.type).toBe("hello");
    expect(errors).toEqual([]);
    expect(perception.startedWith).toBeInstanceOf(FakeStream);
    expect(media.live()).toEqual(["video", "audio"]);

    perception.emit(video(Date.now() / 1000, [0.6, 0.2])); // the 0.2-tall box is background (< 0.55 x tallest)
    perception.emit({ t: 1, wall: 1, db_spl: 50 });
    const tracks = out.filter((m) => m.type === "tracks");
    expect(tracks).toHaveLength(1);
    expect(tracks[0]?.type === "tracks" && tracks[0].persons.map((p) => [p.id, p.slump])).toEqual([[1, true]]);
    expect(log.filter((l) => l === "video" || l === "audio")).toEqual(["video", "audio"]);

    await vi.advanceTimersByTimeAsync(5000);
    expect(out.filter((m) => m.type === "governance")).toHaveLength(1);
    const status = out.filter((m) => m.type === "status");
    expect(status.length).toBeGreaterThanOrEqual(5);
    expect(status.at(-1)).toMatchObject({ narrator: "template", fps: 9.6, mode: "live", camera: true, mic: true });
    expect((status.at(-1) as BrowserStatus).loading).toBeUndefined();
    src.stop();
    expect(media.live()).toEqual([]);
  });

  it("pause releases camera, mic and perception but keeps people; Start reopens them; the kill switch also forgets everyone", async () => {
    const media = installMedia();
    const { src, log, out, perceptions, lastStatus } = source();
    await src.start();
    expect(src.status()).toMatchObject({ camera: true, mic: true });

    expect(src.command({ kind: "control", action: "pause" }).ok).toBe(true);
    expect(perceptions[0]?.stopped).toBe(true);
    expect(media.live()).toEqual([]);
    expect(getBrowserStream()).toBeNull();
    expect(src.status()).toMatchObject({ camera: false, mic: false });
    perceptions[0]?.emit(video(1, [0.5]));
    expect(log).not.toContain("video");
    expect(log.some((l) => l.startsWith("reset"))).toBe(false); // the window and people stay

    src.command({ kind: "control", action: "start" });
    await flush();
    expect(media.calls).toHaveLength(2);
    expect(perceptions[1]?.started).toBe(true);
    expect(media.live()).toEqual(["video", "audio"]);

    src.command({ kind: "control", action: "stop" });
    expect(perceptions[1]?.stopped).toBe(true);
    expect(media.live()).toEqual([]);
    expect(log).toContain("reset keep_room=false"); // room history too, as app.py control stop
    expect(log.filter((l) => l.startsWith("audit"))).toEqual(["audit control_pause", "audit control_start", "audit control_stop"]);
    expect(lastStatus()).toBeDefined();
    expect(out.some((m) => m.type === "governance")).toBe(false);
    src.stop();
  });

  it("the kill switch during the permission prompt releases the camera once the prompt resolves", async () => {
    const media = installMedia("hold");
    const { src, perceptions } = source();
    const started = src.start();
    await flush();
    expect(media.asks).toHaveLength(1);
    src.command({ kind: "control", action: "stop" });
    media.asks[0]?.grant();
    await started; await flush();
    expect(media.live()).toEqual([]);
    expect(getBrowserStream()).toBeNull();
    expect(perceptions).toHaveLength(0);
    expect(src.status()).toMatchObject({ camera: false, mic: false });
    src.stop();
  });

  it("Stop, Start, Stop, Start during the prompt leaves exactly one stream and one perception, and stop() ends both", async () => {
    const media = installMedia("hold");
    const { src, perceptions } = source();
    const started = src.start();
    await flush();
    src.command({ kind: "control", action: "stop" });
    src.command({ kind: "control", action: "start" }); await flush();
    src.command({ kind: "control", action: "stop" });
    src.command({ kind: "control", action: "start" }); await flush();
    expect(media.asks).toHaveLength(3);
    for (const a of [...media.asks]) { a.grant(); await flush(); }
    await started; await flush();
    expect(media.live()).toEqual(["video", "audio"]); // only the last open survives
    expect(perceptions.filter((p) => p.started && !p.stopped)).toHaveLength(1);
    expect(getBrowserStream()?.getTracks().every((t) => t.readyState === "live")).toBe(true);
    src.stop(); // a source switch
    expect(media.live()).toEqual([]);
    expect(perceptions.every((p) => !p.started || p.stopped)).toBe(true);
  });

  it("Stop while the engine loads: getUserMedia is never called; Start opens the sensors afterwards", async () => {
    const media = installMedia();
    const engine = deferred<BrowserEngine>();
    const log: string[] = [];
    const { src, perceptions } = source({ createEngine: () => engine.promise });
    const started = src.start();
    src.command({ kind: "control", action: "stop" });
    engine.resolve(fakeEngine(log));
    await started; await flush();
    expect(media.calls).toHaveLength(0);
    expect(perceptions).toHaveLength(0);
    src.command({ kind: "control", action: "start" });
    await flush();
    expect(media.calls).toHaveLength(1);
    expect(perceptions[0]?.started).toBe(true);
    src.stop();
  });

  it("a stopped source asks no further prompt and reports nothing when its pending prompt is denied", async () => {
    const media = installMedia("hold");
    const { src, errors } = source();
    const started = src.start();
    await flush();
    src.stop(); // e.g. the visitor clicked "Watch the 10-state tour"
    media.asks[0]?.deny("NotAllowedError");
    await started; await flush();
    expect(media.calls).toHaveLength(1);
    expect(errors).toEqual([]);
  });

  it("stop() while the perception chunk loads: that perception is stopped and its failure is not reported", async () => {
    const media = installMedia();
    const chunk = deferred<void>();
    const p = fakePerception({ start: () => { throw new Error("perception needs a MediaStream with a live camera or mic track"); } });
    const { src, errors } = source({ createPerception: async () => { await chunk.promise; return p; } });
    const started = src.start();
    await flush();
    src.stop();
    expect(media.live()).toEqual([]);
    chunk.resolve();
    await started; await flush();
    expect(p.started).toBe(false);
    expect(errors).toEqual([]);
  });

  it("perception failing to start releases camera and mic and reports why", async () => {
    const media = installMedia();
    const p = fakePerception({ start: () => Promise.reject(new Error("perception failed to start: camera: HTTP 404")) });
    const { src, errors } = source({ createPerception: async () => p });
    await src.start();
    expect(media.live()).toEqual([]);
    expect(getBrowserStream()).toBeNull();
    expect(p.stopped).toBe(true);
    expect(errors).toEqual(["camera: perception failed: Error: perception failed to start: camera: HTTP 404"]);
    expect(src.status()).toMatchObject({ camera: false, mic: false });
    src.stop();
  });

  it("without getUserMedia: reports the real cause and starts no perception", async () => {
    const { src, errors, perceptions } = source(); // jsdom has no navigator.mediaDevices
    await src.start();
    expect(errors).toEqual([`camera: ${NO_MEDIA_API}`]);
    expect(perceptions).toHaveLength(0);
    src.stop();
  });

  it("a refused prompt is asked once (no per-device retry) and explained in plain words", async () => {
    const media = installMedia(() => domErr("NotAllowedError"));
    const { src, errors, perceptions } = source();
    await src.start();
    expect(media.calls).toHaveLength(1);
    expect(perceptions).toHaveLength(0);
    expect(errors).toEqual([`camera: ${mediaErrorText(domErr("NotAllowedError"), "both")}`]);
    expect(errors[0]).toMatch(/Camera and microphone blocked: allow them .* then press Start/);
    // Start after allowing it in the browser asks again.
    src.command({ kind: "control", action: "start" });
    await flush();
    expect(media.calls).toHaveLength(2);
    src.stop();
  });

  it("each open first clears the last attempt's errors (onOpenSensors); a granted retry runs and says so at once", async () => {
    let asked = 0;
    const media = installMedia(() => (asked++ === 0 ? domErr("NotAllowedError") : [new FakeTrack("video"), new FakeTrack("audio")]));
    const events: string[] = [];
    const { src, perceptions, lastStatus } = source({ onOpenSensors: () => events.push("open"), onError: (w, d) => events.push(`${w}: ${d}`) });
    await src.start();
    expect(events).toHaveLength(2);
    expect(events.join(" | ")).toMatch(/^open \| camera: Camera and microphone blocked/);
    expect(lastStatus()?.loading).toBeUndefined(); // the refusal shows now, not at the next second
    src.command({ kind: "control", action: "start" }); // the visitor allowed it in the browser
    await flush();
    expect(events.slice(2)).toEqual(["open"]); // cleared before asking; nothing new to report
    expect(perceptions[0]?.started).toBe(true);
    expect(lastStatus()).toMatchObject({ camera: true, mic: true }); // emitted when the open finished
    expect(lastStatus()?.loading).toBeUndefined();
    src.command({ kind: "control", action: "start" }); // already running: no new prompt, nothing cleared
    await flush();
    expect([media.calls.length, events.filter((e) => e === "open").length]).toEqual([2, 2]);
    src.stop();
  });

  it("Stop during the prompt shows at once: status goes out with the command, not at the next second", async () => {
    const media = installMedia("hold");
    const { src, lastStatus } = source();
    const started = src.start();
    await flush();
    expect(src.status().loading).toBe("waiting for camera and microphone");
    expect(lastStatus()?.loading).toBe("loading the engine and models"); // the 1 s timer has not fired yet
    src.command({ kind: "control", action: "stop" });
    expect(lastStatus()).toMatchObject({ camera: false, mic: false, degraded: [] });
    expect(lastStatus()?.loading).toBeUndefined();
    media.asks[0]?.deny("NotAllowedError");
    await started;
    src.stop();
  });

  it("the loading MB counts this load only: Start after Pause (models from the HTTP cache) does not repeat the first session's", async () => {
    installMedia();
    const entries: { name: string; transferSize: number }[] = [];
    const spy = vi.spyOn(performance, "getEntriesByType").mockImplementation(() => entries as unknown as PerformanceEntryList);
    const model = (mb: number) => ({ name: new URL(`/mediapipe/file${entries.length}.task`, location.href).href, transferSize: mb * 1e6 });
    const loads = [deferred<void>(), deferred<void>()];
    const ps: FakePerception[] = [];
    const { src } = source({ createPerception: async () => { const i = ps.length, p = fakePerception({ start: () => loads[i]?.promise }); ps.push(p); return p; } });
    try {
      const started = src.start();
      await flush();
      entries.push(model(10.7), { name: new URL("/assets/perception-x.js", location.href).href, transferSize: 5e6 });
      expect(src.status().loading).toBe("loading perception models · 10.7 MB");
      loads[0]?.resolve();
      await started;
      src.command({ kind: "control", action: "pause" });
      src.command({ kind: "control", action: "start" });
      await flush();
      expect(src.status().loading).toBe("loading perception models"); // nothing fetched yet for this load
      entries.push(model(2));
      expect(src.status().loading).toBe("loading perception models · 2.0 MB");
      loads[1]?.resolve();
      await flush();
      expect(src.status().loading).toBeUndefined();
    } finally {
      src.stop();
      spy.mockRestore();
    }
  });

  it("a missing camera is retried device by device and runs on the mic alone", async () => {
    const media = installMedia((c) => (c.video ? domErr("NotFoundError") : [new FakeTrack("audio")]));
    const { src, errors, perceptions } = source();
    await src.start();
    expect(media.calls.map((c) => Object.keys(c).sort().join("+"))).toEqual(["audio+video", "video", "audio"]);
    expect(errors).toEqual(["camera: No camera found."]);
    expect(perceptions[0]?.started).toBe(true);
    expect(src.status()).toMatchObject({ camera: false, mic: true });
    src.stop();
  });

  it("status reports the tracks the browser has open, and the loading phase with the MB downloaded", async () => {
    installMedia();
    const loaded = deferred<void>();
    const p = fakePerception({ start: () => loaded.promise, status: () => ({ phase: "loading", download_bytes: 12_345_678 }) });
    const { src } = source({ createPerception: async () => p });
    const started = src.start();
    await flush();
    expect(src.status()).toMatchObject({ camera: true, mic: true, loading: "loading perception models · 12.3 MB", degraded: [] });
    loaded.resolve();
    await started;
    expect(src.status().loading).toBeUndefined();
    const tracks = getBrowserStream()?.getTracks() ?? [];
    tracks.forEach((t) => t.stop()); // the OS took the camera away
    expect(src.status()).toMatchObject({ camera: false, mic: false });
    src.stop();
  });

  it("?perception=mock (needsMedia false) opens no camera and starts perception with null", async () => {
    const media = installMedia();
    const { src, perceptions } = source({ needsMedia: false });
    await src.start();
    expect(media.calls).toHaveLength(0);
    expect(perceptions[0]?.started).toBe(true);
    expect(perceptions[0]?.startedWith).toBeNull();
    expect(getBrowserStream()).toBeNull();
    src.stop();
  });

  it("room-only toggles keep everyone's window; toggles pressed while the engine loads reach it", async () => {
    installMedia();
    const engine = deferred<BrowserEngine>();
    const log: string[] = [];
    const { src, out } = source({ createEngine: () => engine.promise });
    const started = src.start();
    expect(src.command({ kind: "mode", room_only: true }).ok).toBe(true);
    expect(src.command({ kind: "sim", which: "low_light", enabled: true }).ok).toBe(true);
    expect(src.command({ kind: "break" })).toEqual({ ok: false, text: "in-browser engine still loading" });
    expect(src.command({ kind: "baseline", seconds: 90 }).ok).toBe(false);
    engine.resolve(fakeEngine(log));
    await started;
    expect(log).toEqual(expect.arrayContaining(["room_only true", "overlay low_light true"]));
    expect(out.find((m) => m.type === "hello")).toMatchObject({ room_only: true });
    src.command({ kind: "mode", room_only: false });
    expect(log).toContain("room_only false");
    expect(log.some((l) => l.startsWith("reset"))).toBe(false); // app.py set_mode only clears the latest cards
    src.command({ kind: "baseline", seconds: 90 });
    src.command({ kind: "demo_reset" });
    expect(log).toEqual(expect.arrayContaining(["baseline 90 capturing", "reset keep_room=true"]));
    expect(src.command({ kind: "replay", scenario: "full_demo" }).ok).toBe(false);
    src.stop();
  });

  it("the tour refuses a baseline capture (its records never reach the aggregator) and never opens a camera", async () => {
    const media = installMedia();
    const { src, log } = source({ mode: "tour" });
    await src.start();
    const r = src.command({ kind: "baseline", seconds: 90 });
    expect(r.ok).toBe(false);
    expect(r.text).toMatch(/tour carries its own room baseline/);
    expect(src.command({ kind: "baseline_reset" }).ok).toBe(false);
    expect(log.some((l) => l.startsWith("baseline"))).toBe(false);
    src.command({ kind: "control", action: "stop" });
    src.command({ kind: "control", action: "start" });
    await flush();
    expect(media.calls).toHaveLength(0);
    src.stop();
  });

  it("status says sealed only when this host's CSP limits connect-src to 'self'", async () => {
    const sealed = source({ needsMedia: false, cspSealed: async () => true });
    await sealed.src.start();
    expect(sealed.src.status().sealed).toBe(true);
    sealed.src.stop();
    const open = source({ needsMedia: false, cspSealed: async () => false });
    await open.src.start();
    expect(open.src.status().sealed).toBe(false);
    open.src.stop();
    expect(source().src.status().sealed).toBeUndefined(); // not checked yet

    expect(sealsConnect("default-src 'self'; connect-src 'self'; img-src 'self' data:")).toBe(true);
    expect(sealsConnect("default-src 'self'; connect-src 'self' https://x.example")).toBe(false);
    expect(sealsConnect("default-src 'self'")).toBe(false);
    expect(sealsConnect(null)).toBe(false);
    const head = (csp: string | null) => (async (url: RequestInfo | URL, init?: RequestInit) => {
      expect([String(url), init?.method, init?.cache]).toEqual([location.pathname, "HEAD", "no-store"]);
      return new Response(null, { headers: csp ? { "content-security-policy": csp } : {} });
    }) as typeof fetch;
    expect(await cspSealed(head("default-src 'self'; connect-src 'self'"))).toBe(true);
    expect(await cspSealed(head(null))).toBe(false);
    expect(await cspSealed((async () => { throw new TypeError("offline"); }) as typeof fetch)).toBe(false);
  });

  it("live, the break clock reads as time on camera until a break restarts that person's clock (the engine's text stays Python's); the tour is untouched", async () => {
    vi.useFakeTimers();
    const LINE = "Person #1 reads as Focused. Main evidence: 0.0 hours since the last break and room light 336 lux.";
    const pred = { type: "prediction", event_id: "p1-1", t: 1, person_id: 1, label: "Focused", label_display: "Focused", confidence: 0.9,
      probabilities: { Focused: 0.9 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up: false,
      layers: [{ name: "Time Dynamics", contribution: 1, drivers: [{ feature: "hours_since_break", display: "0.0 hours since the last break", value: 0, unit: "h", shap: 1 }] }],
      data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: LINE, source: "template" },
      source: "live", simulated_fields: [] } satisfies ServerMessage;
    const engine = () => ({ ...fakeEngine([]), tick: (): ServerMessage[] => [structuredClone(pred)] });
    const said = (out: ServerMessage[]) => out.filter((m) => m.type === "prediction").map((m) => [m.person_id, m.explanation.text, m.layers[0]?.drivers[0]?.display]);
    // Live, the engine predicts everyone it has seen on camera (each one gets a break clock when first seen).
    const seen = new Set<number>();
    const liveEngine = () => ({ ...fakeEngine([]),
      push: (o: ObsLike) => { if ("persons" in o) o.persons.forEach((p) => seen.add(p.track_id)); return null; },
      tick: (): ServerMessage[] => [...seen].sort((a, b) => a - b).map((id) => ({ ...structuredClone(pred), person_id: id, event_id: `p${id}-1` })) });
    const perceptions: FakePerception[] = [];
    const live = source({ needsMedia: false, perceptions, createEngine: async () => liveEngine() });
    await live.src.start();
    live.src.command({ kind: "break" }); // before anyone is on camera: restarts no one's clock
    perceptions[0]?.emit(video(1, [0.5]));
    await vi.advanceTimersByTimeAsync(5000);
    live.src.command({ kind: "break" }); // restarts person 1's clock
    perceptions[0]?.emit(video(2, [0.5, 0.5])); // person 2 first appears after the break: a clock from arrival
    await vi.advanceTimersByTimeAsync(5000);
    live.src.command({ kind: "demo_reset" }); // fresh clocks: time on camera again
    await vi.advanceTimersByTimeAsync(5000);
    const arrival = "0.0 hours since you appeared on camera (no break logged)";
    const onCamera = [LINE.replace("0.0 hours since the last break", arrival), arrival];
    expect(said(live.out)).toEqual([
      [1, ...onCamera],
      [1, LINE, "0.0 hours since the last break"], [2, ...onCamera],
      [1, ...onCamera], [2, ...onCamera],
    ]);
    live.src.stop();
    const tour = source({ mode: "tour", tourSpeed: 1, createEngine: async () => engine() });
    await tour.src.start();
    await vi.advanceTimersByTimeAsync(5000);
    expect(said(tour.out)).toEqual([[1, LINE, "0.0 hours since the last break"]]);
    tour.src.stop();
  });

  it("live, the compliment count is marked as not measured (no speech recognition in the browser); the tour's scripted ones stay", async () => {
    vi.useFakeTimers();
    // Every compliment phrasing describe.ts has (compl_cnt_*, compl_since_s, the unnarrated compl_cnt_now).
    const shown = ["no compliments in the room in the last 15 minutes", "1 compliment in the room in the last minute",
      "3 compliments in the room in the last 5 minutes", "no compliment in the last 15 minutes", "last compliment 40 s ago",
      "compl_cnt_now = 0", "no laughter events in the room in the last 5 minutes"];
    const LINE = `Person #1 reads as Focused. Main evidence: ${shown[0]}, ${shown[6]} and ${shown[3]}.`;
    const pred = { type: "prediction", event_id: "p1-1", t: 1, person_id: 1, label: "Focused", label_display: "Focused", confidence: 0.9,
      probabilities: { Focused: 0.9 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up: false,
      layers: [{ name: "Ambient Physics", contribution: 1, drivers: shown.map((display, i) => ({ feature: `f${i}`, display, value: 0, unit: "", shap: 0.1 })) }],
      data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: LINE, source: "template" },
      source: "live", simulated_fields: [] } satisfies ServerMessage;
    const engine = async () => ({ ...fakeEngine([]), tick: (): ServerMessage[] => [structuredClone(pred)] });
    const said = (out: ServerMessage[]) => out.filter((m) => m.type === "prediction").map((m) => [m.explanation.text, m.layers[0]?.drivers.map((d) => d.display)]);
    const NOT = " (not measured: no speech recognition)";
    const live = source({ needsMedia: false, createEngine: engine });
    await live.src.start();
    await vi.advanceTimersByTimeAsync(5000);
    expect(said(live.out)).toEqual([[
      `Person #1 reads as Focused. Main evidence: ${shown[0]}${NOT}, ${shown[6]} and ${shown[3]}${NOT}.`,
      [...shown.slice(0, 6).map((s) => `${s}${NOT}`), shown[6]],
    ]]);
    live.src.stop();
    const tour = source({ mode: "tour", tourSpeed: 1, createEngine: engine });
    await tour.src.start();
    await vi.advanceTimersByTimeAsync(5000);
    expect(said(tour.out)).toEqual([[LINE, shown]]);
    tour.src.stop();
  });

  it("a value just under zero reads +0.00 in the explanation and drivers, not Python's -0.00; real negatives stay", async () => {
    vi.useFakeTimers();
    const LINE = "Person #1 reads as Focused. Main evidence: neutral expression (valence -0.00), motion changing (-0.000) and light changing (-0.4 lux/min).";
    const pred = { type: "prediction", event_id: "p1-000079", t: 1, person_id: 1, label: "Focused", label_display: "Focused", confidence: 0.9,
      probabilities: { Focused: 0.9 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up: false,
      layers: [{ name: "Facial Surface", contribution: 1, drivers: [{ feature: "face_now", display: "neutral expression (valence -0.00)", value: -0.001, unit: "", shap: 1 }] }],
      data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: LINE, source: "template" },
      source: "replay", simulated_fields: [] } satisfies ServerMessage;
    const { src, out } = source({ mode: "tour", tourSpeed: 1, createEngine: async () => ({ ...fakeEngine([]), tick: (): ServerMessage[] => [structuredClone(pred)] }) });
    await src.start();
    await vi.advanceTimersByTimeAsync(5000);
    const p = out.find((m) => m.type === "prediction");
    expect(p?.type === "prediction" && [p.explanation.text, p.layers[0]?.drivers[0]?.display, p.event_id]).toEqual([
      "Person #1 reads as Focused. Main evidence: neutral expression (valence +0.00), motion changing (+0.000) and light changing (-0.4 lux/min).",
      "neutral expression (valence +0.00)", "p1-000079"]);
    src.stop();
  });

  it("names the Evaluation rows in plain words (ids kept for the tooltip) and says which measurement the 4-state rows are", () => {
    const served = { summary: { m2: { test_macro_f1: 0.7911 }, m0: { test_macro_f1: 0.4153 } } };
    const emotion = { meta: { summary: { full: { macro_f1: 0.6815, accuracy: 0.6964, ece: 0.0074 },
      face_only: { macro_f1: 0.6155, accuracy: 0.6264, ece: 0.0089 }, context_only: { macro_f1: 0.4248, accuracy: 0.447, ece: 0.0047 } } } } as unknown as EmotionModel;
    const e = plainEval(evalSummaryOf(served, emotion));
    const rows = e.metrics as EvalMetricRow[];
    expect(rows.map((r) => [r.id, r.model])).toEqual([
      ["m2", "Context model · context + face, 4 states"],
      ["m0", "Face-only baseline · face snapshot (2 features), 4 states"],
      ["emotion-v1 full", "Emotion map · context + face, 10 states"],
      ["emotion-v1 face_only", "Emotion map · face only (38 face features)"],
      ["emotion-v1 context_only", "Emotion map · context only"],
    ]);
    expect(rows.map((r) => r.macro_f1)).toEqual([0.7911, 0.4153, 0.6815, 0.6155, 0.4248]);
    expect(e.disclosure).toMatch(/^Held-out test scores on simulated sessions/);
    expect(e.disclosure).toMatch(/context \+ face scores 0\.79 macro-F1 against 0\.42 for the face snapshot alone\. Those two rows are the trainer's test score over mixed history lengths/);
    // The ladder figures it quotes are MODEL_CARD section 5's, from the tracked reports/metrics.json: a re-run that
    // changes them fails here until the disclosure is updated.
    expect(e.disclosure).toContain(`reads ${metrics.models.m2.macro_f1.toFixed(3)} and ${metrics.models.m0.macro_f1.toFixed(3)}.`);
  });

  it("latestInputs gives nothing while paused or stopped", async () => {
    const engine = { ...fakeEngine([]), latestInputs: () => ({ feature_version: "emotion-v1", features: { a: 1 } }) };
    const { src } = source({ needsMedia: false, createEngine: async () => engine });
    await src.start();
    expect(src.latestInputs(1)).not.toBeNull();
    src.command({ kind: "control", action: "pause" });
    expect(src.latestInputs(1)).toBeNull();
    src.stop();
  });

  it("hands the loaded engine's evaluation summary to the console once the engine is up", async () => {
    const summary = { available: true, metrics: [{ model: "m2", accuracy: NaN, macro_f1: 0.79, ece: NaN }], disclosure: "simulated" };
    const gate = deferred<BrowserEngine>();
    const got: unknown[] = [];
    const { src } = source({ needsMedia: false, createEngine: () => gate.promise, onEval: (e) => got.push(e) });
    const started = src.start();
    await flush();
    expect(got).toEqual([]); // nothing while the models load
    gate.resolve({ ...fakeEngine([]), evalSummary: () => summary });
    await started;
    expect(got).toEqual([summary]);
    src.stop();
  });
});
