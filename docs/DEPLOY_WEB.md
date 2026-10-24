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
- **`_headers`:** written by `scripts/security-headers.mjs` at the end of the build, and applied by Cloudflare to
  every response:

  | Header | Value |
  |---|---|
  | `Content-Security-Policy` | `default-src 'self'`; `connect-src 'self'` (the page cannot send data to any other origin); `script-src 'self' 'wasm-unsafe-eval'` plus SHA-256 hashes of the shell's inline scripts (no `'unsafe-inline'`) |
  | `Permissions-Policy` | camera and microphone for this origin only |
  | `Referrer-Policy` | `no-referrer` |
  | `X-Content-Type-Options` | `nosniff` |
  | `Cross-Origin-Opener-Policy` | `same-origin` |

## Checking a build the way Cloudflare serves it

```
cd frontend && npm run build && node scripts/check-static.mjs
node scripts/check-static.mjs --path "/?demo=tour" --wait 15000 --expect "Top drivers by layer" --forbid-media
node scripts/check-static.mjs --path "/?source=browser&perception=mock" --fake-media --wait 25000 --expect "Top drivers by layer" --forbid-media
node scripts/check-static.mjs --hostname valence-web.example --path "/?utm_source=x" --wait 12000 --expect "Watch the 10-state tour"
```

The check serves `dist/client` with the `_headers` rules applied and opens the page in headless Chromium. It fails on:

- any request to another origin;
- any CSP violation;
- any page error.

To exercise the in-browser source with a fake camera and microphone:
`node scripts/check-static.mjs --fake-media --path "/?source=browser"`.

## Verifying the privacy claim on the live site

In the browser, open DevTools → Network and use the console for a few minutes. Every request goes to the site's own
origin, and only for static files. The response headers of `/` show the CSP above.
