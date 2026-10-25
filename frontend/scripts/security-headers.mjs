// Writes dist/client/_headers for the static Cloudflare deploy (Workers static assets / Pages read this file).
//
// The console runs the whole Valence engine in the page: camera and mic are processed in memory and nothing is
// sent anywhere. The CSP makes that checkable: connect-src 'self' means the page cannot send data to any other
// origin, and every script must come from the site itself. TanStack's prerendered shell carries a few inline
// scripts (hydration, scroll restoration) whose content changes per build, so their SHA-256 hashes are computed
// here, after the shell exists, instead of allowing 'unsafe-inline'.
//
//   node scripts/security-headers.mjs [dist/client]     (also runs automatically at the end of `vite build`)

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Inline <script> bodies (no src attribute) in an HTML document, in document order, as the browser parses them.
 * TanStack's dehydrated router state contains a NUL character; the HTML parser turns U+0000 in script data into
 * U+FFFD, and the CSP hash is computed over the parsed text, so hash the replaced text.
 */
export function inlineScripts(html) {
  return [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)].map((m) =>
    m[1].replaceAll("\u0000", "�"),
  );
}

/** Absolute http(s) URLs in src/href attributes: the shell must not load anything from another origin. */
export function externalRefs(html) {
  return [...html.matchAll(/\b(?:src|href)="(https?:\/\/[^"]+)"/gi)].map((m) => m[1]);
}

export function contentSecurityPolicy(scriptHashes) {
  return [
    "default-src 'self'",
    // 'wasm-unsafe-eval' lets MediaPipe compile its WebAssembly; it does not allow JS eval.
    `script-src 'self' 'wasm-unsafe-eval' ${scriptHashes.join(" ")}`.trim(),
    // Inline styles cannot send data (fetches from CSS are bound by img-src/font-src 'self').
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    "media-src 'self' blob:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

export function headersFile(html) {
  const hashes = inlineScripts(html).map(
    (body) => `'sha256-${createHash("sha256").update(body, "utf8").digest("base64")}'`,
  );
  return [
    "/*",
    `  Content-Security-Policy: ${contentSecurityPolicy(hashes)}`,
    "  Permissions-Policy: camera=(self), microphone=(self), geolocation=(), payment=(), usb=()",
    "  Referrer-Policy: no-referrer",
    "  X-Content-Type-Options: nosniff",
    "  Cross-Origin-Opener-Policy: same-origin",
    "",
    "/assets/*",
    "  Cache-Control: public, max-age=31536000, immutable",
    "",
    // MediaPipe's WASM, glue and worklet and the engine's models have unversioned URLs, while the hashed bundles that
    // load them change on every deploy: revalidate on each load (Cloudflare answers 304 from the ETag, so the ~11 MB
    // WASM downloads again only when it changed). A max-age would pair new code with cached old files after a deploy.
    "/mediapipe/*",
    "  Cache-Control: no-cache",
    "",
    "/models/*",
    "  Cache-Control: no-cache",
    "",
  ].join("\n");
}

export function writeSecurityHeaders(dir = "dist/client") {
  const index = path.join(dir, "index.html");
  if (!existsSync(index)) throw new Error(`${index} not found: run the build first`);
  const html = readFileSync(index, "utf8");
  const external = externalRefs(html);
  if (external.length)
    throw new Error(`index.html loads from other origins: ${external.join(", ")}`);
  writeFileSync(path.join(dir, "_headers"), headersFile(html));
  console.log(
    `[security-headers] ${path.join(dir, "_headers")}: ${inlineScripts(html).length} inline scripts hashed`,
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  writeSecurityHeaders(process.argv[2]);
}
