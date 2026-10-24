// Self-hosted MediaPipe assets for the in-browser perception (frontend/src/perception), so the page never loads
// anything from another origin:
//   - WASM runtimes copied from node_modules/@mediapipe/tasks-{vision,audio}/wasm (the pinned package versions);
//   - model files downloaded once from Google's model bucket at the URLs below, checked against pinned SHA-256s
//     (the same files the Python edge uses, scripts/download_models.sh).
// Everything lands in public/mediapipe/{wasm,models}/ (gitignored) plus a manifest.json with sizes and hashes.
// Runs before `npm run build` and `npm run dev` (prebuild/predev); idempotent: a file whose hash matches is kept.
//
//   node scripts/mediapipe-assets.mjs [--check]     (--check: verify only, never download; exit 1 if anything is off)

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(FRONTEND, "public", "mediapipe");
const MP = "https://storage.googleapis.com/mediapipe-models";

export const MODELS = [
  {
    file: "face_landmarker.task",
    url: `${MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
    sha256: "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    bytes: 3758596,
  },
  {
    file: "pose_landmarker_lite.task",
    url: `${MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
    sha256: "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    bytes: 5777746,
  },
  {
    file: "gesture_recognizer.task",
    url: `${MP}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`,
    sha256: "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    bytes: 8373440,
  },
  {
    file: "yamnet.tflite",
    url: `${MP}/audio_classifier/yamnet/float32/1/yamnet.tflite`,
    sha256: "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf",
    bytes: 4126810,
  },
];

// FilesetResolver picks `<task>_wasm[_module][_nosimd]_internal.{js,wasm}` at runtime; ship every variant it can ask for.
const WASM = [
  ["tasks-vision", ["vision_wasm_internal", "vision_wasm_nosimd_internal", "vision_wasm_module_internal"]],
  ["tasks-audio", ["audio_wasm_internal", "audio_wasm_nosimd_internal", "audio_wasm_module_internal"]],
];

const sha256 = (file) => createHash("sha256").update(readFileSync(file)).digest("hex");
const sameBytes = (a, b) => existsSync(b) && statSync(a).size === statSync(b).size && sha256(a) === sha256(b);

async function download(url, dest, expected) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const got = createHash("sha256").update(buf).digest("hex");
  if (got !== expected) throw new Error(`${url}: sha256 ${got} does not match the pinned ${expected}`);
  writeFileSync(`${dest}.part`, buf);
  renameSync(`${dest}.part`, dest);
}

export async function prepare({ check = false, out = OUT, log = console.log } = {}) {
  const problems = [];
  const manifest = { generated_by: "frontend/scripts/mediapipe-assets.mjs", packages: {}, wasm: [], models: [] };
  for (const dir of ["wasm", "models"]) mkdirSync(path.join(out, dir), { recursive: true });

  for (const [pkg, stems] of WASM) {
    const root = path.join(FRONTEND, "node_modules", "@mediapipe", pkg);
    if (!existsSync(root)) {
      problems.push(`@mediapipe/${pkg} is not installed (run npm install)`);
      continue;
    }
    manifest.packages[`@mediapipe/${pkg}`] = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8")).version;
    for (const stem of stems) {
      for (const ext of [".js", ".wasm"]) {
        const src = path.join(root, "wasm", stem + ext);
        const dest = path.join(out, "wasm", stem + ext);
        if (!sameBytes(src, dest)) {
          if (check) problems.push(`wasm/${stem + ext} is missing or stale`);
          else copyFileSync(src, dest);
        }
        if (existsSync(dest)) manifest.wasm.push({ file: `wasm/${stem + ext}`, bytes: statSync(dest).size });
      }
    }
  }

  for (const m of MODELS) {
    const dest = path.join(out, "models", m.file);
    const ok = existsSync(dest) && statSync(dest).size === m.bytes && sha256(dest) === m.sha256;
    if (!ok) {
      if (check) problems.push(`models/${m.file} is missing or does not match its pinned sha256`);
      else {
        log(`[mediapipe-assets] downloading ${m.file} (${(m.bytes / 1048576).toFixed(1)} MiB)`);
        await download(m.url, dest, m.sha256);
      }
    }
    manifest.models.push({ file: `models/${m.file}`, bytes: m.bytes, sha256: m.sha256, source: m.url });
  }

  if (!check) writeFileSync(path.join(out, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  const total = [...manifest.wasm, ...manifest.models].reduce((n, f) => n + f.bytes, 0);
  log(`[mediapipe-assets] ${problems.length ? "PROBLEMS" : "ok"}: ${manifest.wasm.length} wasm files, `
    + `${manifest.models.length} models, ${(total / 1048576).toFixed(1)} MiB in ${path.relative(FRONTEND, out)}`);
  return { problems, manifest };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
const proxied = process.env.HTTPS_PROXY || process.env.https_proxy;
if (isMain && proxied && !process.env.NODE_USE_ENV_PROXY) {
  // Node's fetch ignores HTTPS_PROXY unless NODE_USE_ENV_PROXY=1 is set at startup (behind a corporate or CI proxy).
  const r = spawnSync(process.execPath, process.argv.slice(1), {
    stdio: "inherit",
    env: { ...process.env, NODE_USE_ENV_PROXY: "1" },
  });
  process.exit(r.status ?? 1);
} else if (isMain) {
  const check = process.argv.includes("--check");
  prepare({ check })
    .then(({ problems }) => {
      for (const p of problems) console.error(`[mediapipe-assets] ${p}`);
      process.exit(problems.length ? 1 : 0);
    })
    .catch((e) => {
      console.error(`[mediapipe-assets] ${e.message}`);
      process.exit(1);
    });
}
