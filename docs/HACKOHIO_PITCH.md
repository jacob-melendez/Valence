# Hack OHI/O 2026: pitch, demo script and Devpost draft (Valence Web)

Draft written during the event by Cloud A. Numbers come from `docs/HACKOHIO_LOG.md`, `docs/WEB_ENGINE.md`,
`docs/hackohio/status-{B,C}.md` and `docs/MODEL_CARD.md`. Re-check them against the final log before submitting.
Every model number is **on simulated data**, so say so whenever one is shown.

## One line

**Valence Web:** context-aware emotion signals, computed entirely in your browser. It reads the room, the body and the
face over 15 minutes and explains every call. It uploads nothing, and its CSP forbids it from connecting anywhere but its
own origin.

## Continuation disclosure (put this first in the submission)

Valence is a continuation of our TrojanHacks @ USC 2026 project (AI Data Governance track winner). Hack OHI/O allows
continued projects with a precise log of the work done at the event:

- **Baseline:** commit `cc28be9`, tagged `hackohio-2026-baseline`, made before the event. Everything after it is
  event work. `git diff cc28be9..hackohio-2026-final` shows all of it.
- **Log:** `docs/HACKOHIO_LOG.md` has a narrative entry every ~2 h, plus a table generated from git (every commit,
  time, track, files) that flags anything outside 10:00 ET Oct 24 → 10:00 ET Oct 25.
- **Before the event, Valence was a desktop Python engine:**
  - laptop camera and mic;
  - 4 or 6 states;
  - a FastAPI backend and a React console;
  - governance: purge gate, audit log, no-network sandbox.
- **Built during the event:**
  1. **Valence Web.** The whole engine, ported to TypeScript and running in the page:
     - perception with MediaPipe in a Web Worker;
     - the aggregator and the 123 features;
     - LightGBM with TreeSHAP;
     - the smoother and the mask override;
     - explanations and the audit chain.
     It is checked against the Python engine on golden data and deployed as a static site with a strict CSP.
  2. **The emotion map.** A valence–arousal core with 10 states (4 new ones: Bored, Confused, Calm, Anxious) and 31
     new face-cue features, plus a self-report capture for real-data evaluation.

## Demo script (about 3 minutes; a recorded video or live)

| Time | Show | Say |
|---|---|---|
| 0:00 | The hosted link opens on the **tour** (no permission prompt). The welcome banner is visible; the badge reads IN-BROWSER · TOUR (SIMULATED). | "This is Valence Web. Everything you're about to see runs inside this tab: the perception, the model and the explanations." |
| 0:15 | The tour plays at 10×. Scroll to the emotion map below the person card: its trail moves through the states, with a dashed ring at the scripted state. | "This is a scripted, simulated session walking through ten states. The dot is the model; the dashed ring is what the script says. The probabilities place the point on the plane." |
| 0:40 | Back to the person card, which names its 4-state context model: probabilities, the face-only baseline next to it, the template explanation, the four layers with their drivers. | "Every call comes with its evidence: room noise and light, posture and motion, time since a break, and the face, each with its SHAP contribution. The card is a 4-state model; the map is a separate 10-state one, so they can differ." |
| 1:00 | Click **Use my camera (on-device)**. Allow the camera and mic, and keep a neutral face. Your box appears; the card shows a **warming up** badge for the first minute. | "Now it's my camera. The first 20 seconds learn my neutral face, and labels are provisional for the first minute. Frames are processed in memory and dropped." |
| 1:30 | DevTools → Network: only the site's own files. Response headers: `Content-Security-Policy: … connect-src 'self'`. The governance panel: frames purged keeps up with frames decoded (at most the one frame in flight behind), Violations 0. | "Here is the privacy claim you can check yourself. The page is not allowed to connect anywhere else. Each frame is closed in the same step that reads it, so purged keeps up with decoded by construction; the counters show every step finished, and Violations would count a raw buffer left open." |
| 2:05 | Smile, then frown and lower your brows; look down at the keyboard. The emotion map and the cues move. | "Face cues like brow lowering, blinks and head pose feed the emotion map, together with the room and the body." |
| 2:30 | The **Self-report** panel (taps open after the warm-up): tap how you feel, then Download JSONL. | "For real-data evaluation we ask people, not a dataset of actors: every two minutes you tap how you feel, and only the numbers are kept." |
| 2:45 | `docs/WEB_ENGINE.md`: the parity table. | "The browser engine reproduces the Python engine: identical model scores, explanations and audit hashes on golden data; a 3-person tick takes 26 ms." |

Backup: if the camera misbehaves, stay on the tour; it shows everything except your own face. A 75 s recording of
the tour is in `docs/media/valence-web-tour.mp4` (simulated, and captioned so).

Optional 20 s for a governance-minded judge: in the governance panel press **Download audit log**, then run
`python scripts/audit_verify.py <file> --head <head>` from the repo. The desktop's verifier accepts the browser's
chain ("chain intact"), and editing any line makes it fail at that line.

## Devpost draft

**Inspiration.** Emotion AI usually reads one face in one frame, and faces mask. A stressed person can smile. Our
first version (TrojanHacks, AI Data Governance winner) showed that context over time beats the face snapshot on one
laptop, with governance built in. The obvious next question was whether anyone could try it without installing
anything, and without trusting us with their camera.

**What it does.**
- Valence Web runs the whole pipeline in a browser tab:
  1. camera and microphone;
  2. MediaPipe face, pose and hand models in a Web Worker, and its YAMNet sound model on the page's thread;
  3. a 15-minute rolling window of room, body and face features;
  4. LightGBM models with TreeSHAP explanations;
  5. a face-only shadow model whose call context can overrule (the "mask override");
  6. template narration;
  7. a hash-chained audit log.
- The emotion map places each person on a valence–arousal plane with 10 named states, from new face cues (brow
  lowering, eye openness, blinks, head pose, frown, smile) plus context.
- The page's Content-Security-Policy forbids connecting anywhere except its own origin, so the privacy claim is
  something you can check, not just trust.

**How we built it.** Three parallel coding sessions, each with its own files:
- **The engine:** a TypeScript port of the Python engine.
- **The emotion map:** simulator, features and models, in Python.
- **Browser perception:** MediaPipe Tasks Web.

Python stayed the reference. An exporter wrote golden fixtures from the Python engine, and every TypeScript module
had to match them. That meant reproducing numpy's pairwise summation, CPython's rounding and float formatting, and
LightGBM's TreeSHAP exactly. The site is static on Cloudflare, with every asset (fonts, MediaPipe WASM, models)
self-hosted.

**Challenges.**
- **Bit-level parity.** numpy's sums are pairwise, Python rounds ties to even, and LightGBM's TreeSHAP weights paths
  by row counts, not hessians.
- **A strict CSP with a framework.** TanStack's prerendered shell has a NUL byte in an inline script. Browsers parse
  it as U+FFFD, so the CSP hash had to be computed on the replaced text.
- **A telemetry-free MediaPipe.** Every MediaPipe Tasks Web 1.x build posts usage logs to Google, so we pinned
  0.10.35, which has no external URLs.
- **BlazePose jitter.** Compared side by side with the desktop's YOLO pose on identical frames, BlazePose read about
  18 times more motion on a still scene (0.46 against 0.026 shoulder-widths/s). We added a One Euro filter and compute
  motion from the shoulders only.

**Accomplishments.**
- **Parity with Python:**
  - identical LightGBM scores, TreeSHAP contributions, explanation strings (12,644), audit hashes, and all 2,964
    runtime messages over 996 golden ticks;
  - features and probabilities within 1–2 ulp;
  - on the emotion tour, 600/600 labels identical to Python.
- **A strict CSP that holds in practice:** 0 off-origin requests and 0 CSP violations in the end-to-end tests with
  fake media.
- **The 10-state emotion map (simulated data, 90 held-out sessions):**

  | Measure | Result |
  |---|---|
  | Macro-F1 | 0.681 (95% CI 0.667–0.695) |
  | Face only | 0.616 |
  | Calibration error (ECE) | 0.007 |

**What we learned.** The governance claim gets much stronger when it moves from "we don't store frames" to "the
browser won't let this page send them".

**What's next.**
- Real-data evaluation from consented self-report sessions.
- Measuring real face-cue spreads against the simulator. This is the main risk: doubling every cue spread drops
  macro-F1 to 0.554 in simulation.
- Real-camera tuning of BlazePose slump detection.

**Built with.** TypeScript, React, TanStack Start, Vite, MediaPipe Tasks Web, LightGBM (training in Python; inference
ported), Python, numpy, Cloudflare Workers static assets, Playwright, vitest.

## Do not say

- Never "detects emotions" without "signals" and "on simulated data".
- No clinical or diagnostic claims, and nothing about lecture or classroom monitoring.
- Not "offline". The page is online but sealed by its CSP.
- Never present the tour as real data: it is a scripted simulated session, and the badge says so.
