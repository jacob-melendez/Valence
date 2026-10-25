# Hack OHI/O: the finish line (what only you can do)

The three cloud sessions can't do these steps: they need a Cloudflare account, a real camera, a person, Devpost, or a
tag push. Do them in this order, and stop wherever time runs out: each step stands on its own. The event ends at
**10:00 ET Oct 25**, and the feature freeze for the cloud sessions is at 08:00 ET.

| # | Step | Time | Why |
|---|---|---|---|
| 1 | Deploy to Cloudflare | 5 min | Judges get a link that runs everything in their own tab |
| 2 | Check the live site, real camera included | 10 min | The cloud tests used fake cameras |
| 3 | Record the live-camera demo | 10 min | The video judges watch |
| 4 | Devpost submission | 15 min | Required |
| 5 | Push the final tag | 1 min | Closes the work log at a fixed commit |
| 6 | Optional: a self-report session | 10 min | The first real-data check of the emotion map |

## 1. Deploy (5 min)

Follow [`docs/DEPLOY_WEB.md`](DEPLOY_WEB.md):

1. Cloudflare dashboard → Workers & Pages → Import a repository → `jacob-melendez/Valence`.
2. Use these settings:

   | Setting | Value |
   |---|---|
   | Production branch | `claude/happy-ride-7w3ng3` |
   | Root directory | `frontend` |
   | Build | `npm install && npm run build` |
   | Deploy | `npx wrangler deploy` |

The site appears at `https://valence-web.<your-subdomain>.workers.dev`. Paste that URL to any cloud session (or to
Cloud A) and it goes into the README, the pitch and the Devpost draft.

## 2. Check the live site (10 min)

1. **Phone first** (what a judge sees). The link opens on the tour, with no permission prompt and one orange
   "IN-BROWSER · TOUR (SIMULATED)" badge, and the person card fills in within about 20 s.
2. **Laptop, Chrome.** DevTools → Network: every request goes to the site's own origin. The response headers of `/`
   include `content-security-policy: … connect-src 'self'`.
3. **Real camera.** Press **Use my camera (on-device)**, allow camera and mic, and sit neutral for 20 s. Then follow
   [`frontend/e2e/REAL_CAMERA.md`](../frontend/e2e/REAL_CAMERA.md) sections 1–2: load time, fps, smile/frown, nod.
   Paste the numbers in bold into `docs/hackohio/status-C.md` or send them to a session.

If anything fails: the tour still demonstrates everything except your own face, and
[`docs/media/valence-web-tour.mp4`](media/valence-web-tour.mp4) is a 75 s backup clip.

## 3. Record the demo (10 min)

Follow the table in [`docs/HACKOHIO_PITCH.md`](HACKOHIO_PITCH.md) ("Demo script", about 3 minutes; QuickTime screen
recording works). The privacy beat at 1:30 is the one to land: Network tab, the CSP header, the governance counters.
Say "signals" and "simulated data", never "detects emotions" (see "Do not say" in the pitch).

## 4. Devpost (15 min)

Copy from [`docs/HACKOHIO_PITCH.md`](HACKOHIO_PITCH.md) → "Devpost draft". Put the **continuation disclosure first**
(baseline tag `hackohio-2026-baseline`, commit `cc28be9`, and the log `docs/HACKOHIO_LOG.md`). Add:

- the Cloudflare URL;
- the demo video;
- the repo link.

The pitch's numbers are re-checked against the final log at 08:00 ET.

## 5. Push the final tag (1 min, from the Mac, before 10:00 ET)

```
git fetch origin claude/happy-ride-7w3ng3
git tag hackohio-2026-final origin/claude/happy-ride-7w3ng3
git push origin hackohio-2026-final
```

`git diff hackohio-2026-baseline..hackohio-2026-final` then shows exactly the event's work.

## 6. Optional: a self-report session (10 min)

This is the first real labeled data for the emotion map; until now every model number is from simulated data.

1. On the live site, press **Use my camera (on-device)** and work normally for 10 minutes.
2. When the **Self-report** panel asks (every 2 minutes, after the first minute), tap how you feel on the grid.
3. Press **Download JSONL**. The file has only feature numbers and your taps: no images, no audio.
4. Commit the file under `data/self_report/`, or hand it to Cloud B. Cloud B runs
   `python -m valence.eval.self_report <file>.jsonl`, and the model card's real-data table fills in.

Five taps is a smoke test, not an evaluation; the model card will say so.
