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

### 14:15 — Engine modules at parity, emotion model trained and ported, real perception running

**Cloud A (web):** the engine port (golden exporter → modules → served predictor → runtime) is past its module and
review stages. **326 engine tests pass** against fixtures exported from the Python engine. Highlights:
- **Numerics:** numpy's pairwise sum and nan statistics, plus CPython's `round`, float formatting and `repr`. Both
  are bit-exact on more than 2000 reference cases.
- **Features:** the 123 features are bit-identical to Python at the scenario checkpoints.
- **LightGBM:** raw scores are bit-identical. TreeSHAP weights paths by row counts, as LightGBM does.
- **Audit:** the audit chain reproduces Python's `json.dumps` and SHA-256 hashes.
- **The served predictor and smoother** are ported. The runtime stage is running.
- **The emotion-v1 port** is a second workflow (`8e62e64`, `a045957`):
  - the 31 window features are bit-identical (1454 checkpoints, plus a fuzz of 2196 windows);
  - the 10-state model's raw scores are bit-identical to LightGBM through both of Cloud B's model formats;
  - about 0.6 ms per prediction;
  - an adversarial review found and fixed 5 divergences, chiefly missing values read as 0 and the Affect
    rounding.
- **Deploy:** cache headers for MediaPipe and the models. The full `npm run build` with Cloud C's assets (72 MiB)
  passes the static check.

**Cloud B (emotion):** the 10-state classifier and the valence/arousal regressors are trained and exported
(`80ac70e`, `emotion-v1@378fa16c`). Results on simulated data, 90 held-out sessions:

| Model | Macro-F1 | ECE | Valence MAE / CCC | Arousal MAE / CCC |
|---|---|---|---|---|
| Full (served) | 0.688 | 0.007 | 0.176 / 0.78 | 0.238 / 0.72 |
| Face only | 0.621 | — | — | — |
| Context only | 0.425 | — | — | — |

- The weakest state is Confused (F1 0.48).
- The map position is the probability-weighted anchors, which beat the regressors on both axes.
- A spec review (`140d7e5`) changed the blink-rate rule: cue-less ticks are now missing. The rule is ported here.

**Cloud C (perception):** the real MediaPipe pipeline runs end to end in headless Chromium, in a Web Worker
(`51ab99b`).
- Faces carry 52 blendshapes, and the observations follow the Python edge's semantics.
- In every scenario: 0 off-origin requests and 0 CSP violations. Throughput is about 8.5 fps on the CPU.
- Pose motion was found to differ from YOLO: BlazePose jitter reads about 0.46 sw/s on a still scene. After a
  side-by-side study, a One Euro filter with shoulder-only motion brings this to about 0.13
  (`8cc9899`/`9837d71`).
- Slump detection with BlazePose stays an open item for the real-camera check.

### 15:50 — Valence Web runs end to end; engine port gate passed

**Cloud A (web):**
- **Engine glue** (`01ca763`). `/?source=browser` now runs, all in the page:
  1. perception;
  2. the aggregator;
  3. the served M2 + M0 models with TreeSHAP layers;
  4. the template explanation;
  5. the audit chain;
  6. the emotion map.

  On the production build in headless Chromium, with a fake camera and the strict CSP, every panel fills in. There
  are 0 off-origin requests, 0 CSP violations and 0 page errors. Self-report taps now record the emotion model's
  132 inputs.
- **Camera-free demo** (`424729c`). `/?demo=tour` replays Cloud B's scripted 50-minute walk through the 10 states.
  It runs through the same engine at 10× speed on a virtual clock, and the emotion map draws the scripted state as
  a dashed ring.
  - End-to-end parity on the tour: all 600 ticks are within 1e-9 of Python, and the labels are identical on
    600/600.
  - The label matches the scripted state on 449/600 ticks.
- **Engine-port integration gate passed.** The workflow ran 17 agents.
  - Frontend: 860 tests in 26 files, `tsc` clean. Python: 41 tests.
  - Exact matches: LightGBM scores, TreeSHAP, explanations (12,644 strings), audit hashes, and all 2,964 runtime
    messages over 996 ticks.
  - Features and probabilities are within 1–2 ulp: V8 vs glibc math.
  - Speed: a 3-person tick takes 26 ms (p95 43 ms) against a 5 s budget.
  - Write-up: `docs/WEB_ENGINE.md`.
- **Status panel** (`7b6674d`). In browser mode it shows the face rate (blink numbers depend on it) and the video
  backend. The network line reads "sealed (`connect-src 'self'`)" instead of a red "online".

**Cloud B (emotion):**
- **Retrained** after two reviews (`emotion-v1@006a8d7d`): the simulator's head pose is now computed per frame.
- **Results on simulated data:** macro-F1 0.681 (CI 0.667–0.695) against 0.616 for face only, a context gain of
  +0.066. ECE 0.007.
- **Robustness:** doubling every face-cue spread drops macro-F1 to 0.554. This is the main risk until real cue
  spreads are measured.

**Cloud C (perception):**
- **Audio:** browser YAMNet is bit-identical to the edge on the same windows. Laughter onsets fall in the same
  blocks.
- **Faces:** browser and edge agree to p95 1.3e-4 on blendshapes and 0.004° on head pose, on identical frames.
  Faces now run in IMAGE mode, as on the edge.
- **Head-crop fix:** a crop's pose was in the crop's own camera; crop hits now report yaw and pitch as missing.
- **Web Worker:** moving MediaPipe to a worker cut main-thread long tasks from 158 to 1.
- **Next:** the whole console running on real perception with fake media files.

### 18:00 — Glue reviewed and hardened; hosted landing; real browser cues

**Cloud A (web):**
- **Hosted landing** (`fa02cf9`). A bare visit to the hosted site opens on the camera-free tour, with a welcome
  banner that offers "Use my camera (on-device)". Localhost keeps the operator defaults.
- **Docs:** the pitch draft with the continuation disclosure, a 3-minute demo script and the Devpost text
  (`docs/HACKOHIO_PITCH.md`), and `docs/WEB_ENGINE.md`.
- **Reproducibility** (`c8b9f69`). Every golden expectation in the engine tests now comes from a generator in the
  repo (`scripts/golden_runtime/`; `check.py` compares 32/32).
- **Glue review** (39 agents, three lenses: lifecycle, privacy, robustness). Every finding was adversarially
  verified: 31 confirmed, 5 rejected. All 31 are fixed and were re-verified independently, plus 3 small
  regressions caught in a second round (`e0526ba`). The ones that mattered:
  - the kill switch failed if pressed while the camera permission prompt was open, and quick Stop/Start sequences
    could leave a camera stream live for the life of the tab;
  - Pause kept the camera and mic open while the UI said "sensing suspended";
  - the "camera-free" tour and `?perception=mock` still asked for camera permission on load;
  - tracking parameters such as `?utm_source=` sent hosted visitors to the operator's mock;
  - "SEALED" was hard-coded. It now shows only when the served CSP really has `connect-src 'self'` (a HEAD check).
  - `check-static` now also catches CSP violations from workers and worklets, a visible error box, and any camera
    call (`--forbid-media`).

  894 frontend tests pass (33 new). The tour, mock-perception and hosted `?utm_source` production checks are all
  clean.

**Cloud B (emotion):**
- A second verification pass on the spec and the model card.
- **First real browser cues:** Cloud C's two CC BY clips, run through the real in-browser perception. Per-tick cues
  on neutral faces are much steadier than in the simulator, so the simulated within-state noise is pessimistic.
  Between-state differences remain unmeasured. The map reads Focused or Calm on them, never a negative state.

**Cloud C (perception):**
- **The whole console on real perception** with fake media files: every panel fills in, 0 off-origin requests,
  0 CSP violations.
- **15-minute soak:** 6.7 fps sustained, 5,646 frames decoded against 5,645 purged, 0 violations, no memory-leak
  trend.
- **CPU delegate by default.** It matches the native edge to about 1e-4 on identical frames; `?delegate=GPU` is
  there to try.
- **Features-only captures** of both clips for Cloud B (`data/browser/`).

## Generated from git

<!-- hacklog:start -->
_Run `make hacklog START=... END=...` to fill this section._
<!-- hacklog:end -->
