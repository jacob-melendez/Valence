# Hack OHI/O 2026: work log (Valence, continued project)

Hack OHI/O (The Ohio State University), October 24–25, 2026, virtual, 24 hours. Valence is a continuation of the
project that won the AI Data Governance track at TrojanHacks @ USC (October 2026). Hack OHI/O allows continued
projects with a precise log of the work done during the event; this file is that log.

## Baseline: what existed before the event

- **Baseline commit `cc28be9`** (Oct 23, before the event; tag `hackohio-2026-baseline`) marks the repository as it
  stood before the event: <https://github.com/jacob-melendez/Valence/commit/cc28be9fe5315791e8dcc715c382ae94869f61d0>.
  Everything after it is event work, and `git diff cc28be9..HEAD` shows all of it. A commit hash cannot be moved
  or edited, so the baseline is fixed even without the tag.
- At the baseline, Valence is an on-device, privacy-first context-aware emotion recognition engine (Python):
  - laptop perception (pose, face blendshapes, gestures, laughter, compliment keywords), with raw media zeroed after
    feature extraction;
  - a 15-minute rolling window of room-relative features feeding a LightGBM model: 4 core states plus 2 extra
    (Focused, Frustrated / Masked, Exhausted, Collaborative / Happy; Overwhelmed, Tense);
  - a face-only baseline model and the mask override;
  - TreeSHAP explanations and a local LLM narrator;
  - a FastAPI + WebSocket backend and a React console;
  - governance: purge gate, hash-chained audit log, no-network sandbox.

  Evaluation at the baseline is in `docs/RESULTS.md` and `docs/MODEL_CARD.md`, on simulated data plus small
  real captures from one person.
- The only commit made before the event for Hack OHI/O is the logging tooling itself: this file,
  `scripts/hack_log.py`, its test and `make hacklog`. Planning notes in `docs/HACKOHIO_PLAN.md` describe intent,
  not code.

## Event goals (tracks)

| Tag | Track | Goal |
|---|---|---|
| `[web]` | **Valence Web** | Run the pipeline entirely in the browser, as a hosted link anyone can try live. Camera and mic data never leave the device. |
| `[emotion]` | **Emotion map** | A valence–arousal core with about 10 named states mapped onto it, and a first evaluation on real labeled data. |
| `[eval]` | Evaluation | Tests, parity checks and measurements supporting both tracks. |
| `[infra]`, `[docs]`, `[pitch]` | Supporting work | Tooling, documentation and the presentation. |

## How this log is kept

1. **Every commit subject starts with its track tag**, e.g. `[web] Port the feature function to TypeScript`. The
   commit time is the record of when the work happened.
2. **A manual entry at least every 2 hours** in "Narrative" below: time, what was done, why, and the result
   (numbers, screenshots, links).
3. **The generated section** is rebuilt from git with
   `make hacklog START=<ISO start> END=<ISO end>`. It lists every commit since the baseline, by track, with time,
   lines changed and files, and flags anything outside the event window.

## Narrative

_Times in America/New_York. Add entries as `### HH:MM — title`, newest last._

### Before the event (Oct 23)

Logging tooling only: this file, `scripts/hack_log.py` (+ test), `make hacklog`, and the baseline (`cc28be9`). The plan
is in `docs/HACKOHIO_PLAN.md`. One more pre-event commit pins the baseline commit in the script and this file; the
generated log lists it, marked as before the window.

### 10:00 — Event start

The event window is 10:00 ET Oct 24 → 10:00 ET Oct 25 (7:00 → 7:00 PT). The baseline tag `hackohio-2026-baseline`
is pushed and points at `cc28be9`.

### 12:20 — Three cloud sessions, ownership fixed

The work runs as three cloud sessions on `claude/happy-ride-7w3ng3`, so progress can be followed from a phone:

| Session | Track | Owns |
|---|---|---|
| **Cloud A** (this log's keeper) | `[web]` engine | `frontend/src/engine/**`, the console source switch and panels, golden exporter, deploy config, this log |
| **Cloud B** | `[emotion]` emotion map | `valence/{sim,features,models,eval}/**`, `artifacts/emotion/**`, `docs/EMOTION_FEATURES.md` |
| **Cloud C** | `[web]` browser perception | `frontend/src/perception/**`, `frontend/public/mediapipe/**`, `frontend/e2e/**`, MediaPipe dependencies |

Decisions:
- **Perception moves from the Mac to the cloud.** It is tested in headless Chromium with a fake camera; real-camera
  checks happen after deploy.
- **Hosting is Cloudflare,** the console build's existing target.
- **The existing 123 features, TickRecord fields and served models are frozen** while Cloud A ports them to
  TypeScript, so parity can't drift. Cloud B adds its new features in a separate module.
- **Status files.** B and C write `docs/hackohio/status-{B,C}.md` every ~2 h; this log summarizes them.

Cloud A's code survey, read-only, found:
- **Port all 123 features**, including the 22 the served model does not use: they feed the room-relative features
  and the explanations.
- **The served model splits in three ways.** With missing type None, a NaN input becomes 0.0. TreeSHAP weights paths
  by row counts.
- **The display strings use Python's rounding,** which rounds ties to even.

### 12:45 — First pushes from all three sessions

**Cloud A (web):**
- **Static Cloudflare deploy** (`f6b8879`). The console builds as a static site with a prerendered shell. The
  headers file applies a strict CSP: `connect-src 'self'`, and scripts only from the site plus SHA-256 hashes of the
  shell's inline scripts.
  - One bug found on the way: TanStack's hydration script contains a NUL byte. Browsers parse it as U+FFFD before
    hashing, so the hash has to be computed on the replaced text.
  - The fonts are now self-hosted.
  - `scripts/check-static.mjs` serves the build with the real headers in headless Chromium. Result: 0 off-origin
    requests, 0 CSP violations, 0 page errors.
  - One-time Cloudflare setup: `docs/DEPLOY_WEB.md`.
- **In-browser source** (`9249fd1`), a third source next to MOCK and LIVE:
  - one camera and mic stream, shared with the preview;
  - messages in the backend's shapes and cadence;
  - operator commands handled locally.
  - Tested with a fake camera at `/?source=browser`.
- **Emotion map panel and self-report capture** (`b163c46`), built against Cloud B's contract:
  - a valence–arousal plane with the anchors and a trail per person;
  - a tap grid every 2 minutes that exports the taps as JSONL.
- **Model files** (`e356acf`) are served at `/models/` from `artifacts/`, with no copies in git.
- **The engine port** runs as a multi-agent workflow in the background: numeric helpers, golden exporter,
  modules, served predictor, runtime and review.

**Cloud B (emotion):** published the `emotion-v1` spec (`d012452`):
- per-frame face cues from MediaPipe, a per-person neutral baseline and a blink rule;
- 8 per-tick fields and 31 window features;
- 10 states with valence–arousal anchors.

The changes are additive. Existing corpora are hash-identical to the pre-event simulator, and the full suite passes
(687 tests).

**Cloud C (perception):**
- The interface (`22edbbc`): observation types mirroring `contracts.py`, a deterministic mock perception and a
  media helper with AEC/NS/AGC off.
- B's face cues in `PersonObs` (`edee3b6`) and the debouncer port (`c834be6`).
- MediaPipe Tasks Web is pinned to 0.10.35. Every 1.x release bundles a usage logger that posts to Google.

## Generated from git

<!-- hacklog:start -->
_Run `make hacklog START=... END=...` to fill this section._
<!-- hacklog:end -->
