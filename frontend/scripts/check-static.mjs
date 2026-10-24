// End-to-end check of the static build as Cloudflare will serve it: dist/client behind a tiny server that applies
// dist/client/_headers, loaded in headless Chromium. Fails if any request leaves the origin, if anything violates
// the CSP (in the page, a Worker or a worklet: the served CSP gets a report-uri), if the page throws, or if the
// console shows its red error box (--allow-errors to accept it).
//
//   npm run build && node scripts/check-static.mjs [--path /?source=browser] [--fake-media] [--screenshot out.png]
//   node scripts/check-static.mjs --path "/?source=browser&perception=mock" --fake-media --wait 25000 --expect "Top drivers by layer"
//   node scripts/check-static.mjs --path "/?demo=tour" --wait 15000 --expect "Top drivers by layer" --forbid-media
//
// --fake-media starts Chromium with a fake camera and microphone (and auto-accepts the permission prompt), so the
// in-browser source can be exercised without hardware. --forbid-media fails if the page calls getUserMedia at all.

import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
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
const ROOT = path.resolve(opt("--dir", "dist/client"));
const PAGE = opt("--path", "/");
const WAIT_MS = Number(opt("--wait", "6000"));
const SHOT = opt("--screenshot", null);
// --fake-video file.y4m / --fake-audio file.wav: feed the fake camera / mic from files (e.g. frontend/e2e/.media from
// e2e/make_media.py, a real face and speech) instead of Chromium's generated test pattern and beep.
const FAKE_VIDEO = opt("--fake-video", null);
const FAKE_AUDIO = opt("--fake-audio", null);
const FAKE = args.includes("--fake-media") || Boolean(FAKE_VIDEO || FAKE_AUDIO);
// --expect "a|b": each text must be on the page at the end. For predictions use PersonCard-only text such as
// "Top drivers by layer": "Person #1" is also the label of the tracking box, which appears before any prediction.
const EXPECT = (opt("--expect", "") ?? "").split("|").filter(Boolean);
const ALLOW_ERRORS = args.includes("--allow-errors");
const FORBID_MEDIA = args.includes("--forbid-media");
const REPORT_PATH = "/__csp-report";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".woff2": "font/woff2",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".task": "application/octet-stream",
  ".tflite": "application/octet-stream",
  ".gz": "application/gzip",
};

/** Parse a Cloudflare _headers file into [{pattern, headers}] (only the `/*` suffix wildcard is used here). */
function parseHeaders(file) {
  if (!existsSync(file)) return [];
  const rules = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    if (!line.trim() || line.trim().startsWith("#")) continue;
    if (!/^\s/.test(line)) rules.push({ pattern: line.trim(), headers: {} });
    else {
      const i = line.indexOf(":");
      rules.at(-1).headers[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
  }
  return rules;
}
const RULES = parseHeaders(path.join(ROOT, "_headers"));
const matches = (pattern, url) =>
  pattern.endsWith("*") ? url.startsWith(pattern.slice(0, -1)) : url === pattern;

// CSP reports: the browser sends them for violations in Workers and worklets too, which raise no document event,
// no console message and (blocked before the network) no request event.
const reports = [];
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://x").pathname;
  if (url === REPORT_PATH && req.method === "POST") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => { reports.push(body); res.statusCode = 204; res.end(); });
    return;
  }
  let file = path.join(ROOT, decodeURIComponent(url));
  if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory())
    file = path.join(ROOT, "index.html");
  for (const r of RULES)
    if (matches(r.pattern, url))
      for (const [k, v] of Object.entries(r.headers))
        res.setHeader(k, k.toLowerCase() === "content-security-policy" ? `${v}; report-uri ${REPORT_PATH}` : v);
  res.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
  createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
// --hostname name: load the page as http://name:port (mapped to this server), e.g. to see the hosted-site defaults.
const HOSTNAME = opt("--hostname", null);
const origin = `http://${HOSTNAME ?? "127.0.0.1"}:${server.address().port}`;

const browser = await chromium.launch({
  args: FAKE
    ? [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        ...(FAKE_VIDEO ? [`--use-file-for-fake-video-capture=${path.resolve(FAKE_VIDEO)}`] : []),
        ...(FAKE_AUDIO ? [`--use-file-for-fake-audio-capture=${path.resolve(FAKE_AUDIO)}`] : []),
        "--autoplay-policy=no-user-gesture-required",
        ...(HOSTNAME ? [`--host-resolver-rules=MAP ${HOSTNAME} 127.0.0.1`] : []),
      ]
    : HOSTNAME ? [`--host-resolver-rules=MAP ${HOSTNAME} 127.0.0.1`] : [],
});
// --mobile: an iPhone-sized viewport (390x844, touch, DPR 3) to check the layout judges see on a phone.
const MOBILE = args.includes("--mobile");
const context = await browser.newContext(
  MOBILE ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true } : { viewport: { width: 1440, height: 1000 } },
);
if (FAKE) await context.grantPermissions(["camera", "microphone"], { origin });
const page = await context.newPage();
const offOrigin = [],
  violations = [],
  errors = [];
await page.addInitScript(() => {
  window.__csp = [];
  document.addEventListener("securitypolicyviolation", (e) =>
    window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`),
  );
  // Every getUserMedia call, before the page's own code runs (the tour and ?perception=mock must make none).
  window.__gum = [];
  const md = navigator.mediaDevices;
  if (md?.getUserMedia) {
    const orig = md.getUserMedia.bind(md);
    md.getUserMedia = (c) => { window.__gum.push(JSON.stringify(c)); return orig(c); };
  }
});
page.on("request", (r) => {
  const u = r.url();
  if (!u.startsWith(origin) && !u.startsWith("data:") && !u.startsWith("blob:")) offOrigin.push(u);
});
page.on("console", (m) => {
  if (/Content Security Policy/i.test(m.text())) violations.push(m.text());
});
page.on("pageerror", (e) => errors.push(String(e)));

const resp = await page.goto(origin + PAGE, { waitUntil: "load" });
await page.waitForTimeout(WAIT_MS);
violations.push(...(await page.evaluate(() => window.__csp)));
const gum = await page.evaluate(() => window.__gum);
const edgeErrors = await page.evaluate(() => {
  const el = document.querySelector("[data-testid=edge-errors]");
  return el && el.getClientRects().length ? el.innerText : null;
});
await page.waitForTimeout(1000); // CSP reports are sent asynchronously
const csp = resp?.headers()["content-security-policy"] ?? "";
const title = await page.title();
const bodyText = await page.evaluate(() => document.body.innerText);
// innerText applies CSS text-transform (panel titles are uppercase): compare case-insensitively.
const missing = EXPECT.filter((t) => !bodyText.toLowerCase().includes(t.toLowerCase()));
if (SHOT) await page.screenshot({ path: SHOT, fullPage: args.includes("--full") });
await browser.close();
server.close();

const report = {
  page: PAGE,
  title,
  csp_applied: csp.startsWith("default-src 'self'"),
  off_origin_requests: offOrigin,
  csp_violations: violations,
  csp_reports: reports,
  page_errors: errors,
  edge_errors: edgeErrors,
  getusermedia_calls: gum,
  missing_text: missing,
};
console.log(JSON.stringify(report, null, 2));
const ok =
  report.csp_applied && !offOrigin.length && !violations.length && !reports.length && !errors.length &&
  !missing.length && (ALLOW_ERRORS || !edgeErrors) && !(FORBID_MEDIA && gum.length);
console.log(ok ? "check-static: OK" : "check-static: FAILED");
process.exit(ok ? 0 : 1);
