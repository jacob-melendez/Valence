// End-to-end test of the in-browser perception in headless Chromium with a fake camera and microphone (files made by
// make_media.py from freely licensed media; see README.md). Builds the standalone harness (vite.config.ts), serves it
// with a strict CSP, and fails unless faces are detected, observations flow, every request stays on the origin,
// nothing violates the CSP and no page error occurs. Writes .out/report-<scenario>.json with fps, model load time and
// download size.
//
//   node e2e/run.mjs [--scenario one|two|roll_cw|roll_ccw|portrait] [--size WxH] [--seconds 25] [--delegate GPU|CPU] [--csp task|deploy]
//                    [--audio file.wav] [--video file.y4m] [--record out.jsonl [--record-seconds N]] [--python-cadence]
//                    [--main-thread] [--deny camera|mic] [--cpu-throttle N] [--profile out.cpuprofile] [--no-build]
//                    [--headed]
//
// Run from frontend/. Needs `npm install`, the repo's Python venv (for make_media.py) and Playwright's Chromium
// (NODE_PATH or /opt/node-tools/node_modules provides `playwright`).

import { spawnSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const E2E = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(E2E, "..");
const REPO = path.resolve(FRONTEND, "..");
const require = createRequire(import.meta.url);
const { chromium } = (() => {
  for (const id of ["playwright", "/opt/node-tools/node_modules/playwright"]) {
    try {
      return require(id);
    } catch {
      /* try the next one */
    }
  }
  throw new Error("playwright not found (npm i -D playwright, or set NODE_PATH)");
})();

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const SCENARIO = opt("--scenario", "one");
// --deny camera|mic: the user refuses that device: getUserMedia rejects any request for it with NotAllowedError, as a
// browser does after "Block" (headless Chromium has no permission prompt to click).
const DENY = opt("--deny", "");
const TAG = DENY ? `${SCENARIO}-no-${DENY}` : SCENARIO;
const SECONDS = Number(opt("--seconds", "25"));
const DELEGATE = opt("--delegate", "");
const CSP_KIND = opt("--csp", "task");
const MEDIA = path.join(E2E, ".media");
const AUDIO = path.resolve(opt("--audio", path.join(MEDIA, "speech.wav")));
// --video file.y4m: any clip instead of a scenario scene (the report is tagged with --scenario).
const VIDEO = path.resolve(opt("--video", path.join(MEDIA, `${SCENARIO}.y4m`)));
// --record out.jsonl [--record-seconds N]: write the observations (obs_to_dict lines, PersonObs fields only) from the
// first video observation for N seconds (default: the whole run), e.g. one pass of a clip.
const RECORD = opt("--record", null);
const RECORD_S = Number(opt("--record-seconds", "Infinity"));
const DIST = path.join(E2E, "dist");
const OUT = path.join(E2E, ".out");

// The CSP from the task brief, or the one the deploy writes (frontend/scripts/security-headers.mjs, no inline scripts
// in this harness, so no hashes).
const TASK_CSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:";
async function cspHeader() {
  if (CSP_KIND === "task") return TASK_CSP;
  const { contentSecurityPolicy } = await import(path.join(FRONTEND, "scripts", "security-headers.mjs"));
  return contentSecurityPolicy([]);
}

function run(cmd, argv, cwd) {
  const r = spawnSync(cmd, argv, { cwd, stdio: "inherit" });
  if (r.status !== 0) throw new Error(`${cmd} ${argv.join(" ")} failed (${r.status})`);
}

// 1. Assets, media, build.
const { prepare } = await import(path.join(FRONTEND, "scripts", "mediapipe-assets.mjs"));
const assets = await prepare({ check: true, log: () => {} });
if (assets.problems.length) run(process.execPath, [path.join(FRONTEND, "scripts", "mediapipe-assets.mjs")], FRONTEND);
if (!existsSync(VIDEO) || !existsSync(path.join(MEDIA, "speech.wav")))
  run(path.join(REPO, ".venv", "bin", "python"), [path.join(E2E, "make_media.py"), MEDIA], REPO);
if (!args.includes("--no-build") || !existsSync(path.join(DIST, "index.html")))
  run("npx", ["vite", "build", "--config", path.join(E2E, "vite.config.ts")], FRONTEND);

// 2. Static server with the CSP on every response; counts the bytes it serves.
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".wasm": "application/wasm", ".task": "application/octet-stream",
  ".tflite": "application/octet-stream", ".png": "image/png", ".jpg": "image/jpeg", ".woff2": "font/woff2",
};
const CSP = await cspHeader();
const served = [];
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://x").pathname;
  let file = path.join(DIST, decodeURIComponent(url));
  if (!file.startsWith(DIST) || !existsSync(file) || statSync(file).isDirectory()) file = path.join(DIST, "index.html");
  res.setHeader("Content-Security-Policy", CSP);
  res.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
  res.setHeader("Cache-Control", "no-store");
  served.push({ url, bytes: statSync(file).size });
  createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${server.address().port}`;

// 3. Chromium with a fake camera and mic fed from the files.
const browser = await chromium.launch({
  headless: !args.includes("--headed"),
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${VIDEO}`,
    `--use-file-for-fake-audio-capture=${AUDIO}`,
    "--autoplay-policy=no-user-gesture-required",
    "--enable-unsafe-swiftshader",
  ],
});
const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
await context.grantPermissions(["camera", "microphone"], { origin });
if (DENY) {
  await context.addInitScript((deny) => {
    const md = navigator.mediaDevices;
    const real = md.getUserMedia.bind(md);
    md.getUserMedia = async (c) => {
      if ((deny === "camera" && c?.video) || (deny === "mic" && c?.audio)) {
        throw new DOMException("Permission denied", "NotAllowedError");
      }
      return real(c);
    };
  }, DENY);
}
const page = await context.newPage();
// --cpu-throttle N: slow the page and its workers N-fold (DevTools CPU throttling), roughly a phone.
const THROTTLE = Number(opt("--cpu-throttle", "1"));
if (THROTTLE > 1) {
  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: THROTTLE });
}
const offOrigin = [];
const consoleCsp = [];
const pageErrors = [];
const consoleLog = [];
context.on("request", (r) => {
  const u = r.url();
  if (!u.startsWith(origin) && !u.startsWith("data:") && !u.startsWith("blob:")) offOrigin.push(u);
});
page.on("console", (m) => {
  const text = m.text();
  if (/Content Security Policy|Refused to/i.test(text)) consoleCsp.push(text);
  if (m.type() === "error" || m.type() === "warning") consoleLog.push(`${m.type()}: ${text.slice(0, 300)}`);
});
page.on("pageerror", (e) => pageErrors.push(String(e)));
page.on("worker", (w) => w.on("console", (m) => /Content Security Policy|Refused to/i.test(m.text()) && consoleCsp.push(m.text())));

const params = new URLSearchParams();
if (DELEGATE) params.set("delegate", DELEGATE);
if (args.includes("--python-cadence")) params.set("faceEveryFrame", "0");
if (args.includes("--main-thread")) params.set("worker", "0");
if (RECORD) params.set("record", "1");
// --size WxH: the camera track's resolution (the harness asks for 1280x720, which crops a portrait clip to a square).
const SIZE = opt("--size", "");
if (SIZE) params.set("size", SIZE);
const query = params.size ? `?${params}` : "";
const resp = await page.goto(`${origin}/${query}`, { waitUntil: "load" });
const startedAt = Date.now();
// --profile out.cpuprofile: a CPU profile of the page's main thread over the run (after models load), for DevTools.
const PROFILE = opt("--profile", "");
let profiler = null;
let state;
for (;;) {
  await page.waitForTimeout(500);
  state = await page.evaluate(() => {
    const e = window.__e2e;
    return { ready: e.ready, error: e.error, n: e.video.length, a: e.audio.length };
  });
  if (PROFILE && !profiler && state.n > 0) {
    profiler = await context.newCDPSession(page);
    await profiler.send("Profiler.enable");
    await profiler.send("Profiler.start");
  }
  if (state.error || Date.now() - startedAt > SECONDS * 1000) break;
}
if (profiler) writeFileSync(path.resolve(PROFILE), JSON.stringify((await profiler.send("Profiler.stop")).profile));
const data = await page.evaluate(() => window.__e2e);
mkdirSync(OUT, { recursive: true });
await page.screenshot({ path: path.join(OUT, `screenshot-${TAG}.png`) });
await browser.close();
server.close();

// 4. Checks and report.
const video = data.video;
const audio = data.audio;
const people = video.flatMap((v) => v.persons);
const faces = people.filter((p) => p.face && p.valence !== null);
const maxPeople = Math.max(0, ...video.map((v) => v.persons.length));
const median = (xs) => {
  const s = xs.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
};
const mediapipeBytes = served.filter((s) => s.url.startsWith("/mediapipe/")).reduce((n, s) => n + s.bytes, 0);
const status = data.status ?? {};
const counters = data.counters ?? {};
const expectPeople = SCENARIO === "two" ? 2 : 1;
const videoChecks = {
  video_observations: video.length >= 20,
  faces_detected: faces.length >= 5,
  people_tracked: maxPeople >= expectPeople,
  blendshapes_52: faces.some((p) => p.nbs === 52),
  frames_purged: counters.frames_decoded > 0 && counters.frames_decoded - counters.frames_purged <= 1,
  ...(SIZE ? { frame_size: video.length > 0 && video.every((v) => `${v.w}x${v.h}` === SIZE) } : {}),
};
const audioChecks = {
  audio_observations: audio.length >= 3,
  speech_heard: audio.some((a) => a.speech !== null && a.speech > 0.5),
};
const checks = {
  csp_header_applied: (resp?.headers()["content-security-policy"] ?? "") === CSP,
  perception_running: status.phase === "running" && !data.error,
  ...(DENY === "camera"
    ? { camera_off_reported: status.camera === false && (status.degraded ?? []).includes("no camera"), no_video: video.length === 0 }
    : videoChecks),
  ...(DENY === "mic"
    ? { mic_off_reported: status.mic === false && (status.degraded ?? []).includes("no microphone"), no_audio: audio.length === 0 }
    : audioChecks),
  no_live_buffers_or_violations: counters.violations === 0,
  no_off_origin_requests: offOrigin.length === 0,
  no_csp_violations: consoleCsp.length === 0 && data.violations.length === 0,
  no_page_errors: pageErrors.length === 0,
};
const report = {
  scenario: TAG,
  csp: CSP,
  delegate_requested: DELEGATE || "default (CPU)",
  seconds: SECONDS,
  checks,
  ok: Object.values(checks).every(Boolean),
  error: data.error,
  media: data.media,
  status,
  counters,
  metrics: {
    load_ms: status.load_ms ?? null,
    fps: status.fps ?? null,
    video_ms: status.video_ms ?? null,
    audio_ms: status.audio_ms ?? null,
    video_observations: video.length,
    audio_observations: audio.length,
    face_frames: faces.length,
    max_people: maxPeople,
    mediapipe_bytes_served: mediapipeBytes,
    mediapipe_files_served: served.filter((s) => s.url.startsWith("/mediapipe/")).map((s) => `${s.url} ${s.bytes}`),
    total_bytes_served: served.reduce((n, s) => n + s.bytes, 0),
    main_thread_long_tasks: data.longTasks,
  },
  medians: {
    face_valence: median(faces.map((p) => p.valence)),
    head_yaw_abs: median(faces.map((p) => p.yawAbs)),
    head_pitch_abs: median(faces.map((p) => p.pitchAbs)),
    head_roll_abs: median(faces.map((p) => p.rollAbs)),
    smile_raw: median(faces.map((p) => p.smileRaw)),
    motion: median(people.map((p) => p.motion)),
    lux: median(video.map((v) => v.lux)),
    lux_quality: median(video.map((v) => v.luxq)),
    db_spl: median(audio.map((a) => a.db)),
    speech: median(audio.map((a) => a.speech)),
  },
  laughter_onsets: audio.reduce((n, a) => n + a.laugh, 0),
  blink_onsets: people.reduce((n, p) => n + p.blink, 0),
  off_origin_requests: offOrigin,
  csp_violations: [...consoleCsp, ...data.violations],
  page_errors: pageErrors,
  console: consoleLog.slice(0, 40),
  sample_video: video.slice(-3),
  sample_audio: audio.slice(-3),
};
writeFileSync(path.join(OUT, `report-${TAG}.json`), `${JSON.stringify(report, null, 2)}\n`);
if (RECORD) {
  const t0 = data.records.find((r) => r.type === "video")?.t ?? 0;
  const kept = data.records.filter((r) => r.t >= t0 && r.t < t0 + RECORD_S);
  writeFileSync(path.resolve(RECORD), kept.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(`recorded ${kept.length} observations (${kept.filter((r) => r.type === "video").length} video) -> ${RECORD}`);
}
console.log(JSON.stringify({ scenario: TAG, ok: report.ok, checks, metrics: report.metrics, medians: report.medians,
  error: data.error, errors: pageErrors.slice(0, 5), csp: report.csp_violations.slice(0, 5), off: offOrigin.slice(0, 5),
  console: consoleLog.slice(0, 10) }, null, 2));
console.log(report.ok ? `e2e ${TAG}: OK` : `e2e ${TAG}: FAILED`);
process.exit(report.ok ? 0 : 1);
