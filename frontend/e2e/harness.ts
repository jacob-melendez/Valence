/**
 * Perception end-to-end harness page: opens the (fake) camera and mic like the console does, runs the real
 * createPerception(), and exposes compact summaries on window.__e2e for frontend/e2e/run.mjs. Query parameters:
 * delegate=GPU|CPU, faceEveryFrame=0 (the Python edge's face cadence), record=1 (keep every observation as an
 * obs_to_dict line), mock=1 (createMockPerception instead).
 */
import {
  createMockPerception, createPerception, obsToDict, openMediaStream, type Obs, type PerceptionOptions,
} from "@/perception";

interface E2E {
  ready: boolean;
  error: string | null;
  media: { camera: boolean; mic: boolean; errors: string[] } | null;
  video: unknown[];
  audio: unknown[];
  status: unknown;
  counters: unknown;
  violations: string[];
  t0: number;
  /** record=1: every observation in contracts.obs_to_dict form (PersonObs fields only, no `raw`). */
  records: unknown[];
  /** Main-thread tasks over 50 ms after start(): count and total ms (what the page's UI would feel). */
  longTasks: { n: number; ms: number; max: number };
}

const e2e: E2E = {
  ready: false, error: null, media: null, video: [], audio: [], status: null, counters: null, violations: [],
  t0: performance.now(), longTasks: { n: 0, ms: 0, max: 0 }, records: [],
};
(window as unknown as { __e2e: E2E }).__e2e = e2e;
document.addEventListener("securitypolicyviolation", (e) => e2e.violations.push(`${e.violatedDirective} ${e.blockedURI}`));

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById("status")!;
const latestEl = document.getElementById("latest")!;
const r3 = (v: number) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);

const RECORD = params.get("record") === "1";

function summarize(o: Obs): void {
  if (RECORD && e2e.records.length < 20000) {
    const d = obsToDict(o);
    if (Array.isArray(d["persons"])) for (const p of d["persons"] as Record<string, unknown>[]) delete p["raw"];
    e2e.records.push(d);
  }
  if (o.type === "audio") {
    e2e.audio.push({ t: r3(o.t), seq: o.block_seq, db: r3(o.db_spl), speech: r3(o.speech_prob), laugh: o.laughter_onsets });
    return;
  }
  e2e.video.push({
    t: r3(o.t), seq: o.frame_seq, w: o.frame_w, h: o.frame_h, decoded: o.frames_decoded, lux: r3(o.lux),
    luxq: r3(o.lux_quality),
    persons: o.persons.map((p) => ({
      id: p.track_id, bbox: p.bbox.map(r3), face: p.face_visible, valence: r3(p.face_valence), motion: r3(p.motion),
      slump: p.slump, fist: p.fist_active, hih: p.head_in_hands, brow: r3(p.brow_down), eye: r3(p.eye_open),
      smile: r3(p.smile), yaw: r3(p.head_yaw), pitch: r3(p.head_pitch), blink: p.blink_onsets,
      nbs: p.raw ? Object.keys(p.raw.blendshapes).length : 0,
      yawAbs: p.raw ? r3(p.raw.head_yaw_deg) : null, pitchAbs: p.raw ? r3(p.raw.head_pitch_deg) : null,
      rollAbs: p.raw ? r3(p.raw.head_roll_deg) : null, smileRaw: p.raw ? r3(p.raw.blendshapes["mouthSmileLeft"] ?? NaN) : null,
    })),
  });
  if (e2e.video.length > 3000) e2e.video.splice(0, 1000);
  latestEl.textContent = JSON.stringify(e2e.video.at(-1), null, 1);
}

async function main(): Promise<void> {
  const opened = await openMediaStream();
  e2e.media = { camera: opened.camera, mic: opened.mic, errors: opened.errors };
  const opts: PerceptionOptions = {};
  const delegate = params.get("delegate");
  if (delegate === "GPU" || delegate === "CPU") opts.delegate = delegate;
  if (params.get("worker") === "0") opts.worker = false;
  const every = params.get("faceEveryFrame");
  if (every !== null) opts.faceEveryFrame = every !== "0";
  const perception = params.get("mock") ? createMockPerception() : createPerception(opts);
  perception.onObservation(summarize);
  perception.onStatus((s) => {
    e2e.status = s;
    statusEl.textContent = JSON.stringify(s, null, 1);
  });
  setInterval(() => {
    e2e.counters = perception.counters();
    e2e.status = perception.status();
  }, 250);
  await perception.start(opened.stream);
  e2e.ready = true;
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        e2e.longTasks.n += 1;
        e2e.longTasks.ms += e.duration;
        e2e.longTasks.max = Math.max(e2e.longTasks.max, e.duration);
      }
    }).observe({ type: "longtask" });
  } catch {
    /* longtask timing unsupported */
  }
}

main().catch((e: unknown) => {
  e2e.error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  statusEl.textContent = `error: ${e2e.error}`;
});
