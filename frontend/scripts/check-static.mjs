// End-to-end check of the static build as Cloudflare will serve it: dist/client behind a tiny server that applies
// dist/client/_headers, loaded in headless Chromium. Fails if any request leaves the origin, if anything violates
// the CSP, or if the page throws.
//
//   npm run build && node scripts/check-static.mjs [--path /?source=browser] [--fake-media] [--screenshot out.png]
//
// --fake-media starts Chromium with a fake camera and microphone (and auto-accepts the permission prompt), so the
// in-browser source can be exercised without hardware.

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
const FAKE = args.includes("--fake-media");

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

const server = createServer((req, res) => {
  const url = new URL(req.url, "http://x").pathname;
  let file = path.join(ROOT, decodeURIComponent(url));
  if (!file.startsWith(ROOT) || !existsSync(file) || statSync(file).isDirectory())
    file = path.join(ROOT, "index.html");
  for (const r of RULES)
    if (matches(r.pattern, url))
      for (const [k, v] of Object.entries(r.headers)) res.setHeader(k, v);
  res.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
  createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  args: FAKE ? ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] : [],
});
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
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
const csp = resp?.headers()["content-security-policy"] ?? "";
const title = await page.title();
if (SHOT) await page.screenshot({ path: SHOT, fullPage: false });
await browser.close();
server.close();

const report = {
  page: PAGE,
  title,
  csp_applied: csp.startsWith("default-src 'self'"),
  off_origin_requests: offOrigin,
  csp_violations: violations,
  page_errors: errors,
};
console.log(JSON.stringify(report, null, 2));
const ok = report.csp_applied && !offOrigin.length && !violations.length && !errors.length;
console.log(ok ? "check-static: OK" : "check-static: FAILED");
process.exit(ok ? 0 : 1);
