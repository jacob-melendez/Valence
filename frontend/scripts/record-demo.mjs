// Records a short demo video of the camera-free tour on the static build (headless Chromium, no camera, no real
// people): the welcome banner, a person card filling in, the emotion map's trail and the governance panel.
//
//   npm run build && node scripts/record-demo.mjs [--out ../docs/media/valence-web-tour.webm] [--speed 30]
//
// The tour is a scripted, simulated session (the page's badge says so); say that wherever the video is shown.

import { createReadStream, existsSync, mkdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const ROOT = path.resolve("dist/client");
const OUT = path.resolve(opt("--out", "../docs/media/valence-web-tour.webm")); // webm (Playwright); see docs/media/README.md for the mp4
const SPEED = opt("--speed", "30");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".woff2": "font/woff2", ".txt": "text/plain", ".gz": "application/gzip", ".wasm": "application/wasm", ".ico": "image/x-icon" };

// Serve dist/client with its _headers (same rules check-static applies).
const rules = [];
for (const line of existsSync(path.join(ROOT, "_headers")) ? readFileSync(path.join(ROOT, "_headers"), "utf8").split("\n") : []) {
  if (!line.trim()) continue;
  if (!/^\s/.test(line)) rules.push({ pattern: line.trim(), headers: {} });
  else { const i = line.indexOf(":"); rules.at(-1).headers[line.slice(0, i).trim()] = line.slice(i + 1).trim(); }
}
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://x").pathname;
  let file = path.join(ROOT, decodeURIComponent(url));
  if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory()) file = path.join(ROOT, "index.html");
  for (const r of rules) {
    const hit = r.pattern.endsWith("*") ? url.startsWith(r.pattern.slice(0, -1)) : url === r.pattern;
    if (hit) for (const [k, v] of Object.entries(r.headers)) res.setHeader(k, v);
  }
  res.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
  createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const HOST = "valence-web.example";
const origin = `http://${HOST}:${server.address().port}`;

const tmp = path.join(path.dirname(OUT), ".record-tmp");
mkdirSync(tmp, { recursive: true });
const browser = await chromium.launch({ args: [`--host-resolver-rules=MAP ${HOST} 127.0.0.1`] });
const size = { width: 1280, height: 760 };
const context = await browser.newContext({ viewport: size, recordVideo: { dir: tmp, size } });
const page = await context.newPage();
const scrollTo = async (sel, block = "start") => {
  await page.evaluate(([s, b]) => document.querySelector(s)?.scrollIntoView({ behavior: "smooth", block: b }), [sel, block]);
  await page.waitForTimeout(1200);
};

await page.goto(`${origin}/?demo=tour&speed=${SPEED}`, { waitUntil: "load" });
await page.waitForTimeout(7000); // the welcome banner, the tour starting
await scrollTo("main");
await page.waitForTimeout(16000); // the person card filling in: probabilities, layers, template text
await scrollTo("[data-testid=affect-map]", "center");
await page.waitForTimeout(30000); // the emotion map's trail moving through the states, the scripted state as a ring
await scrollTo("[data-testid=audit-chain]", "center");
await page.waitForTimeout(8000); // governance: the audit chain, verified
await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
await page.waitForTimeout(4000);

const video = page.video();
await context.close();
await browser.close();
server.close();
const src = await video.path();
renameSync(src, OUT);
console.log(`[record-demo] ${OUT} (${(statSync(OUT).size / 1e6).toFixed(1)} MB)`);
