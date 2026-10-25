# Deploying Valence Web (Cloudflare)

Valence Web is a static site. The console and the in-browser engine run entirely in the page, so camera and
microphone data never leave the device. There is no server code. The site is served by Cloudflare as a
static-assets-only Worker.

## One-time setup (about 5 minutes, from the Cloudflare dashboard)

1. Go to **Workers & Pages → Create → Import a repository** and pick `jacob-melendez/Valence`.
2. Configure the build:

   | Setting | Value |
   |---|---|
   | Project name | `valence-web` (must match `name` in `frontend/wrangler.jsonc`) |
   | Production branch | `claude/happy-ride-7w3ng3` |
   | Root directory | `frontend` |
   | Build command | `npm install && npm run build` |
   | Deploy command | `npx wrangler deploy` |

3. Save. Every push to the branch then rebuilds and redeploys. The site appears at
   `https://valence-web.<your-subdomain>.workers.dev`.

From a laptop instead: `cd frontend && npm install && npm run build && npx wrangler deploy` (log in once with
`npx wrangler login`).

## What the build produces

`npm run build` writes `frontend/dist/client/`:

- **`index.html`:** the prerendered SPA shell. `vite.config.ts` uses SPA mode with `nitro: false`.
- **`assets/`:** the hashed JS and CSS.
- **`fonts/`:** self-hosted font files (see `public/fonts/README.md`).
- **`models/`:** the engine's model files, copied from `artifacts/` by `scripts/model-files.mjs` at the end of the
  build: the served M2/M0 models for 4 states (`core/`) and 6 states (`all/`), and the emotion-v1 models and the tour
  (`emotion/`).
- **`mediapipe/`:** MediaPipe's WASM and models, self-hosted. The `prebuild` step (`scripts/mediapipe-assets.mjs`)
  writes them to `public/mediapipe/`, which is gitignored.
- **`_headers`:** written by `scripts/security-headers.mjs` at the end of the build, and applied by Cloudflare to
  every response:

  | Header | Value |
  |---|---|
  | `Content-Security-Policy` | `default-src 'self'`; `connect-src 'self'` (the page cannot send data to any other origin); `script-src 'self' 'wasm-unsafe-eval'` plus SHA-256 hashes of the shell's inline scripts (no `'unsafe-inline'`) |
  | `Permissions-Policy` | camera and microphone for this origin only |
  | `Referrer-Policy` | `no-referrer` |
  | `X-Content-Type-Options` | `nosniff` |
  | `Cross-Origin-Opener-Policy` | `same-origin` |
  | `Cache-Control` | `/assets/*`: `public, max-age=31536000, immutable`; `/mediapipe/*` and `/models/*`: `no-cache` (revalidated on each load) |

## Checking a build the way Cloudflare serves it

```
cd frontend && npm run build && node scripts/check-static.mjs
node scripts/check-static.mjs --path "/?demo=tour" --wait 15000 --expect "Top drivers by layer" --forbid-media
node scripts/check-static.mjs --path "/?source=browser&perception=mock" --fake-media --wait 25000 --expect "Top drivers by layer" --forbid-media
node scripts/check-static.mjs --hostname valence-web.example --path "/?utm_source=x" --wait 12000 --expect "Watch the 10-state tour"
```

The check serves `dist/client` with the `_headers` rules applied and opens the page in headless Chromium. It fails on:

- any request to another origin;
- any CSP violation (in the page, a Worker or a worklet), or a CSP that was not applied;
- any page error, or the console's red error box (`--allow-errors` accepts it);
- an `--expect` text missing from the page;
- with `--forbid-media`, any camera or microphone request (a `getUserMedia` call).

To exercise the in-browser source with a fake camera and microphone:
`node scripts/check-static.mjs --fake-media --path "/?source=browser"`.

## Browser support

The tour needs a browser with `DecompressionStream` to unpack the gzipped models and the tour script: Chrome/Edge 80+,
Firefox 113+, Safari 16.4+ (iOS 16.4, March 2023). Camera mode runs MediaPipe in a Web Worker where the browser has module
workers and WebGL on an OffscreenCanvas (Chrome/Edge, Firefox 114+, Safari 17+). Elsewhere it runs on the page's main
thread (Safari 16.4–16.x, Firefox < 114: `video_backend` reads `main/CPU`, and the UI is slower). Details are in
`docs/hackohio/status-C.md`. The cloud tests run Chromium only, so check an
iPhone by hand after deploying (`docs/HACKOHIO_FINISH.md` step 2).

## Verifying the privacy claim on the live site

In the browser, open DevTools → Network and use the console for a few minutes. Every request goes to the site's own
origin, and only for static files. The response headers of `/` show the CSP above.
