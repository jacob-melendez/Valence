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

### 20:15 — A judge's-eye pass on the hosted page

**Cloud A (web):**
- **Evaluation panel in the browser** (`a889852`). It used to say "Evaluation summary unavailable". It now shows
  the loaded models' held-out scores, labelled as simulated data: m2/m0 macro-F1 and emotion-v1 accuracy, F1 and ECE.
- **README** (`89ce0a2`): a Valence Web section at the top, covering what moved into the browser, how to try it and
  where the docs are.
- **Judge-view review** (42 agents, four lenses: phone, laptop, live camera, claims). Each finding was
  adversarially verified: 35 confirmed, 3 rejected. They were fixed in six batches (`a075b9c`). Fresh reviewers then
  listed 19 follow-ups, all fixed and re-reviewed (`4f808e1`). The ones that mattered:
  - **Dead controls on the hosted site.** It showed the MOCK/LIVE switch and a `127.0.0.1:8000` Connect form.
    LIVE led to a dead backend that the CSP blocks. The hosted site now shows one badge and no operator chrome;
    localhost keeps every control.
  - **Readings buried.** On a phone the first reading sat more than 2 screens down, behind an empty Room view and
    the operator panels. On a laptop it was below the fold. The person card is now on the first laptop screen, and
    phones see the readings before the operator panels.
  - **Purge log contradicting its counters.** The tour's governance log said "raw biometric frame purged" while
    every counter read 0, and a mic-only live run did the same. The engine's line (kept byte-identical for parity)
    now shows only when a frame really was purged.
  - **Two unlabelled confidences** for the same person: the 4-state card and the 10-state map. Each now names its
    model, and a warm-up badge covers the first minute and the 20 s face baseline.
  - **The page's own clock as evidence.** The live explanation's top evidence was "0.0 hours since the last break",
    which is just time since the page opened. Until a break is logged it now reads as time on camera, per person.
    Compliments are marked unmeasured, because there is no speech recognition in the browser.
  - **Page metadata** still carried template strings and the forbidden phrasing "not faces". It now says signals,
    in-browser and simulated, and a test guards it.

  913 frontend tests pass. tsc is clean. The build and the tour, hosted, mock-perception and localhost checks pass,
  and phones at 390 px no longer scroll sideways.

**Cloud B (emotion):**
- The model card is ready to report real self-report results as soon as taps exist.
- **Public dataset blocked again.** RAVDESS is unreachable: the network policy denies zenodo.org and
  huggingface.co. Recorded in model card §9.6.
- Cross-checked the browser Evaluation panel against `meta.json`.
- Fixed the consent wording in the spec: taps stay in the tab until Clear or close (`b0c10c5`).

**Cloud C (perception):**
- **19:45 check-in, healthy.** All six end-to-end scenarios run at 4.5–7.5 fps with 0 off-origin requests and 0 CSP
  violations, and there is no fixture drift.
- **Bug hunt in progress** across six areas: lifecycle, privacy and purge counts, non-48 kHz audio, video parity,
  Safari/Firefox, and the console contract.
- `REAL_CAMERA.md` now starts the camera from the welcome card (`acdd410`).

### 22:15 — Bug hunt on perception; counters per source; the finish line written down

**Cloud A (web):**
- **Governance counters per source** (`e403179`). Cloud C's bug hunt found that after Pause → Start the governance
  counters fell back to 0, including any recorded violation, because every Start opens a new perception that counts
  from 0. The desktop keeps them per source. The browser source now does too; the new test fails on the old code.
- **One reason per stalled device** (`5f2a8fd`). When perception names the cause ("camera muted: video paused",
  "microphone muted"), the status no longer also lists the bare device name.
- **Tour clip re-recorded** on the current page (`4ab5ea6`).
- **Finish-line checklist** for the user (`378ee86`, `64c0c00`). `docs/HACKOHIO_FINISH.md` lists the six steps only
  a person can do: deploy, live check, demo video, Devpost, final tag, and an optional self-report session.
  `docs/DEPLOY_WEB.md` now has a browser-support note.
- **Integration gate:**
  - Python: 725 passed, 22 skipped.
  - Frontend: 928 tests pass; the build passes; the tour, hosted and mock-perception checks are clean.

**Cloud B (emotion):**
- **One command from taps to model card** (`474628f`): `python -m valence.eval.self_report <taps.jsonl>` writes the
  report, the figure and model card §9.6. It labels fewer than 20 taps as a smoke test.
- No new data and no number changes (`emotion-v1@006a8d7d`).

**Cloud C (perception):**
- **Bug hunt** (six finders, each finding adversarially verified): 14 of 16 findings held, which come to 11
  issues, fixed in `ad7082e`. Among them:
  - a camera that never sends a frame hung the start;
  - Safari muting the camera for another tab gave frozen frames;
  - ended tracks left the status live;
  - the worker died on iOS Chrome/Firefox;
  - the pose cap let small bystanders hide the participant.
- **Review of those fixes:** 4 more issues, fixed in `c34942f`.
- **Person box** (`48694ef`): measured against the desktop's YOLO, it stays full-extent.
- **Browser support for camera mode, from code and published support:**
  - Chrome/Edge (tested), Safari 17+ and iOS browsers: in a worker.
  - Safari 16.4–16.x and Firefox < 114: on the page's thread, which works but is slower.
  - A real Safari or phone still has to be checked by hand after the deploy.

### 00:15 — Perception bug hunt round 2; reviews catch regressions; caching fixed

**Cloud A (web):**
- **Caching** (`ac1a24e`, found in Cloud C's round 2). `/mediapipe/*` and `/models/*` had unversioned URLs with a
  1-day / 1-hour max-age, so after a redeploy a returning visitor could pair new code with old WASM or models. They
  are now revalidated on every load (304 from the ETag), so the 11 MB WASM downloads again only when it changes.
- **Gate on the current head:** 937 frontend tests pass; the build passes; the tour, hosted and mock-perception
  checks are clean. Python is unchanged since the 22:15 run (725 passed).

**Cloud B (emotion):**
- Quiet, with no self-report taps yet.
- The spec records the browser's late calibration of a face-cue channel that has no samples at the 20 s freeze (an
  edge case; parity is unaffected otherwise) (`6741b3d`).
- Numbers unchanged.

**Cloud C (perception):**
- **Round 2** (worker protocol, phone cameras, long sessions, asset build): 8 findings held, fixed in `212d79f`.
  - **Safari 16.4–16.x:** the worker started without WebGL and then failed every frame. A WebGL probe now sends video
    to the page's thread. This corrects the 21:45 browser-support note.
  - **WebGL context loss:** MediaPipe silently repeated a frozen face. Now it shows as "video step failing".
  - **The last frame** stayed in MediaPipe's WASM heap after stop. The heap is now zero-filled when the models close
    (verified on real MediaPipe).
- **Reviews of those fixes caught two of C's own regressions**, corrected in `f1de56c` and `69d00c8`:
  - a 3 s pause reset that could swap seated people's ids is reverted;
  - a blank-frame scrub that could not reach the face crop was replaced by the heap wipe.
  - Test gaps closed.

### 02:15 — Round 3 fixes; soak holds; quiet otherwise

**Cloud A (web)**, from Cloud C's third bug-hunt round (`a99dbbc`, `37faa3d`):
- **Pause → Start now always starts the people fresh** (the room's history stays). A new perception numbers people
  from 1 in detection order, so two people could swap 15-minute histories. Even a presenter alone could inherit
  someone else's history, by coming back as #1 after holding #2. The paused line says "Press Start to begin again
  (people start fresh)".
- **"no microphone"** is shown once.
- **Governance note and pitch:** purged keeps up with decoded by construction, with at most the one frame in flight
  between them.
- **`DEPLOY_WEB.md`:** camera mode runs in a worker on Chrome/Edge, Firefox 114+ and Safari 17+, and on the page's
  main thread elsewhere.
- **Kept on purpose:** a perception pause of 10+ min reads as a break and then an absence, as on the desktop.
- **Gate:** 940 frontend tests pass; the build and the checks are clean. App code is unchanged since; Python is
  unchanged since 22:15.

**Cloud B (emotion):** quiet. No taps yet, and no contract changes (`emotion-v1@006a8d7d`).

**Cloud C (perception):**
- **Round 3** (`d659c41`): 21 findings held, mostly wording. `REAL_CAMERA.md` was rewritten to match the deployed
  console.
- **Portrait phone frames** (720×1280) are now a seventh end-to-end scenario (`644ec53`).
- **40-minute two-person soak:** GC sawtooth with no rising peaks; 11,018 frames decoded against 11,017 purged;
  0 violations; 0 page errors.
- **Browser support correction:** iOS Chrome and Firefox behave like Safari on the same iOS (WebKit).

### 04:15 — Final reviews before the freeze; quiet otherwise

**Cloud A (web):**
- **Gate on the current head:** 944 frontend tests pass; the build passes; the tour, hosted and mock-perception
  checks are clean.
- **No new code.** Cloud C's `d87fbd6` fixed, on its side, a purge gap that grew by one frame on every Pause. The
  browser source's per-source counters (`e403179`) needed no change.

**Cloud B (emotion):**
- Its full pytest on `e276b25`: 706 passed, 41 skipped. The skips are Mac-only dependencies; this container ran
  725/22 at 22:15.
- Still no self-report taps, and the model is unchanged.

**Cloud C (perception):**
- **Final pre-freeze review.** It found the in-flight frame miscounted at Pause (`d87fbd6`). A lost WebGL context now
  stops video with its own reason, and a race found in review is fixed (`52fc8bb`).
- **Slow devices** (4× CPU throttle): the worker path holds 5.9 fps. The main-thread fallback (Safari before 17) now
  rests between steps so the page stays usable (`e276b25`).
- **A fresh clone** (`git clone`, `npm ci`, `npm run build`) builds and passes the production check, with real
  perception and a fake camera.
- `status-C.md` now opens with a "Read first" summary: what works, the gates, what needs a real camera, and known
  limits.

### 06:15 — Quiet before the freeze

- **No code changes since 04:15.** The 04:15 gate still holds: 944 frontend tests and the build pass.
- **Cloud B:** `status-B.md` now opens with a "Read first" summary. It covers what exists, the numbers (simulated
  data only), the real data so far, the main risks, and the one-command self-report path.
- **Cloud C:** its 05:45 check-in is healthy. All seven end-to-end scenarios pass with 0 off-origin requests and
  0 CSP violations. fps came in a little lower (3.7–5.8), which C attributes to load on the shared container; no
  perception code changed.
- Freeze at 08:00 ET.

### 08:00 — Feature freeze

**State at the freeze.** Valence Web runs the whole Valence pipeline in a browser tab:
- MediaPipe perception (video models in a Web Worker, YAMNet on the page's thread);
- the 15-minute window and the 123 features;
- LightGBM with TreeSHAP;
- the smoother and the mask override;
- template explanations;
- a hash-chained audit log that the desktop's Python verifier accepts;
- the 10-state emotion map.

It is a static site with a strict CSP (`connect-src 'self'`). The hosted landing opens on a camera-free, scripted,
simulated tour, and one button switches to the visitor's own camera.

**Gates on the head:**
- **Frontend:** 944 tests in 29 files pass; tsc is clean; the build passes.
- **check-static** (production headers, headless Chromium): the tour, the hosted landing and in-browser perception
  with a fake camera are all clean, with 0 off-origin requests, 0 CSP violations, 0 page errors and no camera
  request in the tour.
- **Python:** 725 passed, 22 skipped (the skips are Mac-only dependencies).
- **Cloud C's end-to-end suite:** seven scenarios with real MediaPipe, fake cameras and 0 violations, plus a
  40-minute soak with no leak.

**Overnight:**
- **Cloud A:**
  - A judge's-eye review of the hosted page: 35 findings, plus 19 follow-ups.
  - Fixes for Cloud C's bug-hunt findings in Cloud A's files: counters per source; Pause → Start starts people fresh;
    revalidated caching; deduplicated status reasons.
  - A finish-line checklist for the user.
  - An audit of every number and claim in the judge-facing docs (14 corrections, `d2256e9`).
- **Cloud B:**
  - The one-command self-report evaluation into the model card.
  - Status only otherwise. No real taps arrived, so every model number is still on simulated data.
- **Cloud C:** three bug-hunt rounds on the perception module, each followed by a review of its own fixes. Two of
  those reviews caught regressions, which were reverted.

**Left for a person** (`docs/HACKOHIO_FINISH.md`):
1. Deploy to Cloudflare.
2. Check the live site on a phone, plus a real-camera check.
3. Record the demo video.
4. Submit on Devpost.
5. Push the `hackohio-2026-final` tag from the Mac before 10:00 ET.
6. Optionally, run a self-report session.

The generated table below runs to the freeze; Cloud A refreshes it at 09:40 ET if anything lands after.

## Generated from git

<!-- hacklog:start -->
_Generated by `scripts/hack_log.py` from git: 148 commits since `hackohio-2026-baseline` (`cc28be9`), times in America/New_York._

- Window: Sat Oct 24 10:00 → Sun Oct 25 10:00 — **1 commit(s) outside the window** (marked ⚠)
- Total change since the baseline: 222 files changed, 47463 insertions(+), 167 deletions(-)

| Track | Commits | Lines + / − |
|---|---|---|
| Valence Web | 89 | +39601 / −1597 |
| Emotion map | 26 | +4915 / −652 |
| Docs | 16 | +626 / −15 |
| Evaluation | 12 | +4588 / −404 |
| Pitch | 4 | +229 / −5 |
| Infrastructure | 1 | +17 / −7 |

### Timeline

| Time | Track | Commit | Change | Files |
|---|---|---|---|---|
| Fri Oct 23 19:57 ⚠ | Infrastructure | `aff901d` Pin the Hack OHI/O baseline to commit cc28be9 (before the event) | +17 / −7 | `docs/HACKOHIO_LOG.md`, `scripts/hack_log.py` |
| Sat Oct 24 12:18 | Docs | `09ea054` Log: event start, three cloud sessions and file ownership | +30 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 12:29 | Valence Web | `f6b8879` Static Cloudflare deploy with a strict CSP and self-hosted fonts | +335 / −5 | `frontend/scripts/check-static.mjs`, `frontend/scripts/security-headers.mjs`, `docs/DEPLOY_WEB.md` +7 more |
| Sat Oct 24 12:32 | Emotion map | `d012452` Publish the emotion-v1 spec: face cues, tick fields, window features, 10 states | +890 / −6 | `docs/EMOTION_FEATURES.md`, `tests/test_emotion_features.py`, `valence/features/face_cues.py` +6 more |
| Sat Oct 24 12:32 | Valence Web | `22edbbc` Browser perception interface: contract types, mock perception, stub, media helper | +1237 / −16 | `frontend/src/perception/types.ts`, `frontend/src/perception/mock.ts`, `frontend/src/test/perception-interface.test.ts` +10 more |
| Sat Oct 24 12:34 | Valence Web | `edee3b6` Perception: emotion-v1 face cues on PersonObs (port of face_cues.py), mock fills them | +205 / −11 | `frontend/src/perception/faceCues.ts`, `frontend/src/perception/types.ts`, `frontend/src/test/perception-interface.test.ts` +2 more |
| Sat Oct 24 12:34 | Valence Web | `9249fd1` In-browser source in the console (engine and perception plug in lazily) | +439 / −23 | `frontend/src/lib/browser-source.ts`, `frontend/src/test/browser-source.test.ts`, `frontend/src/lib/valence.ts` +1 more |
| Sat Oct 24 12:36 | Valence Web | `c834be6` Perception: port of the hysteresis debouncer; parity fixture loader | +113 / −0 | `frontend/src/perception/debounce.ts`, `frontend/src/test/perception-fixture-util.ts` |
| Sat Oct 24 12:38 | Valence Web | `b163c46` Emotion map panel and self-report capture in the console | +248 / −6 | `frontend/src/components/console/panels.tsx`, `frontend/src/lib/valence.ts`, `frontend/src/test/emotion-console.test.ts` +1 more |
| Sat Oct 24 12:40 | Valence Web | `e356acf` Serve the engine's model files at /models/ from artifacts/ (dev middleware, copied at build) | +66 / −1 | `frontend/scripts/model-files.mjs`, `frontend/vite.config.ts` |
| Sat Oct 24 12:40 | Docs | `9a989f0` Log: 12:45 entry (first pushes from all three sessions) | +38 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 12:41 | Valence Web | `ab89dfc` Engine foundation checkpoint: Python-faithful numerics, contracts, SHA; golden fixtures | +3695 / −0 | `scripts/export_web_golden.py`, `frontend/src/engine/contracts.ts`, `frontend/src/engine/num.ts` +13 more |
| Sat Oct 24 12:49 | Emotion map | `cd8f56a` 10-state classifier and valence/arousal regressors: train, evaluate, export | +1070 / −8 | `valence/models/emotion.py`, `valence/eval/emotion_eval.py`, `valence/eval/self_report.py` +6 more |
| Sat Oct 24 12:55 | Valence Web | `af8f48b` Self-hosted MediaPipe assets (prebuild), mic worklet, e2e media with licenses | +313 / −0 | `frontend/scripts/mediapipe-assets.mjs`, `frontend/e2e/make_media.py`, `frontend/e2e/README.md` +6 more |
| Sat Oct 24 12:57 | Valence Web | `299a162` Status C: MediaPipe under the strict CSP, load and per-frame timings, decisions | +22 / −2 | `docs/hackohio/status-C.md` |
| Sat Oct 24 13:02 | Valence Web | `e8b552a` Perception ports: audio features + scipy resample_poly, posture; capture, model loading, pose glue | +3678 / −4 | `frontend/src/test/perception-posture.test.ts`, `frontend/src/test/perception-fixtures/gen_posture.py`, `frontend/src/test/perception-audio.test.ts` +28 more |
| Sat Oct 24 13:04 | Emotion map | `80ac70e` Export the 10-state emotion map: dumps, meta, golden vectors and a 10-state tour | +1534 / −13 | `artifacts/emotion/meta.json`, `artifacts/emotion/metrics.json`, `scripts/emotion_tour.py` +13 more |
| Sat Oct 24 13:04 | Emotion map | `21e1285` Status B: model trained and exported | +36 / −0 | `docs/hackohio/status-B.md` |
| Sat Oct 24 13:04 | Emotion map | `d2dfcce` Status B: fix entry times (ET) | +2 / −2 | `docs/hackohio/status-B.md` |
| Sat Oct 24 13:04 | Emotion map | `6bfb729` Status B: spec entry time is the push time (12:32 ET) | +1 / −1 | `docs/hackohio/status-B.md` |
| Sat Oct 24 13:06 | Valence Web | `b14d3a3` Status C: BlazePose thresholds and lower-body decision from the test scenes | +15 / −1 | `docs/hackohio/status-C.md` |
| Sat Oct 24 13:06 | Valence Web | `fb49a8f` Serve Cloud B's emotion text models (txt.gz) next to the dumps | +6 / −2 | `frontend/scripts/model-files.mjs` |
| Sat Oct 24 13:07 | Valence Web | `4092c09` Engine modules checkpoint: features, aggregator, LightGBM + TreeSHAP, explanations | +4616 / −0 | `frontend/src/engine/aggregator.ts`, `frontend/src/test/engine-aggregator.test.ts`, `frontend/src/engine/features.ts` +13 more |
| Sat Oct 24 13:08 | Valence Web | `8fdb7aa` Perception models: pose thresholds 0.4 (both people found), lower body never confident, CPU on software WebGL, model bytes fetched once | +14 / −4 | `frontend/src/perception/coco.ts`, `frontend/src/perception/models.ts` |
| Sat Oct 24 13:10 | Emotion map | `c59f8c8` Model card: Emotion map section (simulated results, controls, robustness, limits) | +261 / −0 | `docs/MODEL_CARD.md`, `valence/eval/emotion_eval.py`, `artifacts/emotion/robustness.json` |
| Sat Oct 24 13:13 | Evaluation | `8aaf6cd` Emotion map on real captures: cue replay tool, live Affect helper, pre-event captures | +1064 / −1 | `artifacts/emotion/real_captures.json`, `valence/eval/emotion_capture.py`, `tests/test_emotion_models.py` +2 more |
| Sat Oct 24 13:14 | Emotion map | `3e4c390` Spec: self-report session protocol | +12 / −0 | `docs/EMOTION_FEATURES.md` |
| Sat Oct 24 13:15 | Emotion map | `51afdf8` Affect-map figure for the model card (anchors vs mean served position) | +90 / −0 | `scripts/emotion_figures.py`, `docs/MODEL_CARD.md`, `artifacts/emotion/figs/affect_map.png` |
| Sat Oct 24 13:22 | Valence Web | `2141e7b` Face ports proven against Python: scipy's LSA solver ported, numpy-exact distances and float modulo | +1590 / −55 | `frontend/src/test/perception-fixtures/gen_face.py`, `frontend/src/test/perception-face.test.ts`, `frontend/src/perception/assign.ts` +8 more |
| Sat Oct 24 13:24 | Valence Web | `a1ca8cb` Perception ports: hands.py (assign_hands, curled fingers, FistTracker) and ids.py (DisplayIdMapper) | +2083 / −0 | `frontend/src/test/perception-fixtures/gen_hands.py`, `frontend/src/test/perception-hands.test.ts`, `frontend/src/perception/hands.ts` +7 more |
| Sat Oct 24 13:26 | Evaluation | `ce72d07` Emotion map: domain-randomization experiment (noise-scale mix), not adopted | +21 / −2 | `valence/models/emotion.py`, `docs/MODEL_CARD.md` |
| Sat Oct 24 13:26 | Valence Web | `fe6d507` Perception review fixes: numpy's fused norm in posture, JSON null as NaN, debouncer on_count=0 raises like Python | +284 / −43 | `frontend/src/test/perception-fixtures/gen_posture.py`, `frontend/src/perception/posture.ts`, `frontend/src/test/perception-posture.test.ts` +11 more |
| Sat Oct 24 13:28 | Evaluation | `1d74cc9` Emotion map: session-bootstrap intervals (context adds +0.067 macro-F1, CI excludes 0) | +146 / −7 | `valence/eval/emotion_eval.py`, `artifacts/emotion/ci.json`, `docs/MODEL_CARD.md` |
| Sat Oct 24 13:30 | Valence Web | `51ab99b` Real in-browser perception: MediaPipe video pipeline in a Web Worker, audio pipeline, lux/regions/BGR2GRAY ports | +3094 / −32 | `frontend/src/test/perception-fixtures/gen_lux.py`, `frontend/src/test/perception-lux.test.ts`, `frontend/src/perception/perception.ts` +16 more |
| Sat Oct 24 13:32 | Evaluation | `7624700` Self-report eval: warm-window subset and a taps-vs-prediction figure | +39 / −0 | `valence/eval/self_report.py` |
| Sat Oct 24 13:32 | Valence Web | `44dd07b` Ignore plain-JSON perception fixtures (only the gzipped exports are committed) | +2 / −0 | `frontend/src/test/perception-fixtures/.gitignore` |
| Sat Oct 24 13:33 | Valence Web | `f942ab5` Cache headers for /mediapipe/* (1 day) and /models/* (1 hour) | +7 / −0 | `frontend/scripts/security-headers.mjs` |
| Sat Oct 24 13:33 | Valence Web | `6cc805e` Engine checkpoint: reviewed features/aggregator, audit chain, emotion port started | +3536 / −64 | `frontend/src/test/engine-aggregator.test.ts`, `frontend/src/engine/audit.ts`, `frontend/src/test/engine-emotion.test.ts` +10 more |
| Sat Oct 24 13:33 | Valence Web | `c367b0b` Perception README: pipeline, file map with Python references, privacy, deliberate differences, tests | +87 / −0 | `frontend/src/perception/README.md` |
| Sat Oct 24 13:34 | Emotion map | `806ff5f` Model card: what the 10-state classifier leans on (TreeSHAP shares) | +6 / −0 | `docs/MODEL_CARD.md` |
| Sat Oct 24 13:42 | Valence Web | `8e62e64` Emotion-v1 in the TypeScript engine: 31 window features and the 10-state model | +485 / −48 | `frontend/src/test/engine-emotion.test.ts`, `scripts/export_web_golden_emotion.py`, `frontend/src/engine/emotion-model.ts` +1 more |
| Sat Oct 24 13:42 | Valence Web | `082ba13` Engine checkpoint: explanation and feature review fixes in progress | +297 / −29 | `frontend/src/test/engine-explain.test.ts`, `frontend/src/engine/explainer.ts`, `frontend/src/test/engine-features.test.ts` +3 more |
| Sat Oct 24 13:45 | Emotion map | `140d7e5` Spec review fixes: cue-less blink rate, LightGBM tiny-input zeroing, exact edge rules | +214 / −61 | `scripts/emotion_export.py`, `docs/EMOTION_FEATURES.md`, `valence/eval/self_report.py` +9 more |
| Sat Oct 24 13:47 | Valence Web | `a045957` Emotion features: Cloud B's cue-less blink-rate rule (140d7e5) | +18 / −10 | `frontend/src/engine/emotion-features.ts`, `frontend/src/test/engine-emotion.test.ts` |
| Sat Oct 24 13:48 | Valence Web | `4a3fa4f` Engine checkpoint: LightGBM review fix in progress | +38 / −11 | `frontend/src/engine/lgbm.ts` |
| Sat Oct 24 13:48 | Valence Web | `3d4b0ea` Engine checkpoint: TreeSHAP review fix in progress | +7 / −14 | `frontend/src/engine/treeshap.ts` |
| Sat Oct 24 13:49 | Emotion map | `1418119` emotion_json: dump thresholds of +-1e300 are infinite (LightGBM AvoidInf) | +23 / −5 | `tests/test_emotion_models.py`, `valence/models/emotion_json.py`, `docs/EMOTION_FEATURES.md` |
| Sat Oct 24 14:02 | Valence Web | `6c9e3f9` Perception glue review: 10 fixes (cue tracker on held faces, worker start timeout, suspended audio, ended mic, stale runs) | +460 / −59 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/perception.ts`, `frontend/src/test/perception-pipeline.test.ts` +6 more |
| Sat Oct 24 14:07 | Valence Web | `9837d71` Posture on One Euro-smoothed keypoints, motion on the shoulders (BlazePose jitter vs YOLO, measured side by side) | +162 / −3 | `frontend/src/perception/smoothing.ts`, `frontend/src/test/perception-pipeline.test.ts`, `frontend/src/perception/posture.ts` +1 more |
| Sat Oct 24 14:08 | Valence Web | `514c035` Status C 14:00: YOLO vs BlazePose side by side, motion fix, glue review; checklist adds keyboard gaze and stillness | +52 / −3 | `docs/hackohio/status-C.md`, `frontend/e2e/REAL_CAMERA.md` |
| Sat Oct 24 14:09 | Valence Web | `95d0f14` Face cues follow the updated reference: a NaN blendshape side counts as missing (face_cues._avg, 140d7e5) | +1 / −1 | `frontend/src/perception/faceCues.ts`, `frontend/src/test/perception-fixtures/face-pose.json.gz` |
| Sat Oct 24 14:14 | Valence Web | `8cc9899` Face pass on every frame by default (blink detector), face_hz in status; Python cadence one option away | +57 / −12 | `frontend/src/test/perception-pipeline.test.ts`, `frontend/src/perception/video.ts`, `frontend/src/perception/visionHost.ts` +8 more |
| Sat Oct 24 14:15 | Valence Web | `417bcce` Engine checkpoint: served predictor + smoother, audit/LightGBM/TreeSHAP review fixes | +2085 / −130 | `frontend/src/test/engine-served.test.ts`, `frontend/src/engine/served.ts`, `frontend/src/engine/audit.ts` +6 more |
| Sat Oct 24 14:16 | Docs | `df30142` Log: 14:15 entry (engine at parity, emotion model trained and ported, real perception) | +41 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 14:18 | Valence Web | `6e9d11f` e2e: refused camera and refused mic scenarios (NotAllowedError), Python-cadence switch | +55 / −17 | `frontend/e2e/run.mjs`, `frontend/e2e/run-all.mjs`, `frontend/e2e/README.md` |
| Sat Oct 24 14:20 | Valence Web | `1f3f486` e2e measures main-thread long tasks; --main-thread switch (worker: 1 long task in 25 s, main: 158 totalling 22.8 s) | +18 / −2 | `frontend/e2e/harness.ts`, `frontend/e2e/run.mjs` |
| Sat Oct 24 14:47 | Valence Web | `739eadf` Status C 14:50: audio identical to the edge on the same sound (YAMNet bit-identical, onsets in the same blocks) | +37 / −0 | `docs/hackohio/status-C.md` |
| Sat Oct 24 15:04 | Emotion map | `cb7c2bc` Simulate head pose per frame so head_away agrees with yaw/pitch; retrain (emotion-v1@006a8d7d) | +544 / −528 | `artifacts/emotion/metrics.json`, `artifacts/emotion/meta.json`, `valence/sim/generator.py` +13 more |
| Sat Oct 24 15:04 | Evaluation | `8eb7c8c` Emotion-track review fixes; model card section 9 generated from the artifacts | +976 / −362 | `artifacts/emotion/real_captures.json`, `scripts/emotion_card.py`, `docs/MODEL_CARD.md` +8 more |
| Sat Oct 24 15:04 | Emotion map | `0631fc5` Status B 15:25: reviews, fixes, retrain | +50 / −0 | `docs/hackohio/status-B.md` |
| Sat Oct 24 15:04 | Emotion map | `311b772` Status B: entry time 15:04 ET | +1 / −1 | `docs/hackohio/status-B.md` |
| Sat Oct 24 15:06 | Valence Web | `f5b79c0` Emotion parity against Cloud B's retrained model (emotion-v1@006a8d7d) | +47 / −27 | `frontend/src/test/engine-emotion.test.ts`, `scripts/export_web_golden_emotion.py`, `frontend/src/test/fixtures/engine/emotion.json.gz` |
| Sat Oct 24 15:07 | Valence Web | `124228e` Engine checkpoint: runtime port (messages in the backend's shapes) and served review | +2266 / −10 | `frontend/src/engine/runtime.ts`, `frontend/src/test/engine-runtime.test.ts`, `frontend/src/engine/index.ts` +2 more |
| Sat Oct 24 15:13 | Valence Web | `01ca763` Valence Web runs end to end: engine glue for the in-browser source | +235 / −32 | `frontend/src/engine/browser-engine.ts`, `frontend/src/test/engine-browser.test.ts`, `frontend/src/lib/browser-source.ts` +3 more |
| Sat Oct 24 15:13 | Valence Web | `7b6674d` System status in browser mode: face rate for blink numbers, video backend, sealed network | +10 / −4 | `frontend/src/components/console/panels.tsx` |
| Sat Oct 24 15:14 | Valence Web | `6982910` check-static: --fake-video / --fake-audio feed the console from files; the whole console runs on real perception | +27 / −2 | `frontend/scripts/check-static.mjs`, `frontend/e2e/README.md`, `frontend/e2e/console-in-browser.jpg` |
| Sat Oct 24 15:15 | Valence Web | `f95a7e9` e2e README: screenshot time | +1 / −1 | `frontend/e2e/README.md` |
| Sat Oct 24 15:19 | Valence Web | `424729c` Camera-free tour demo (?demo=tour) and end-to-end parity on Cloud B's emotion tour | +195 / −20 | `frontend/src/test/engine-tour.test.ts`, `frontend/src/engine/tour.ts`, `frontend/src/engine/browser-engine.ts` +4 more |
| Sat Oct 24 15:20 | Valence Web | `7090cea` Engine checkpoint: runtime review fixes in progress | +972 / −105 | `frontend/src/test/engine-runtime.test.ts`, `frontend/src/engine/runtime.ts`, `frontend/src/engine/index.ts` |
| Sat Oct 24 15:25 | Emotion map | `4da1b42` Docs re-check fixes: inf-threshold splits, state cues, away semantics, parity and region wording | +31 / −19 | `scripts/emotion_card.py`, `docs/MODEL_CARD.md`, `docs/EMOTION_FEATURES.md` |
| Sat Oct 24 15:26 | Emotion map | `b2f138a` Model card: name the states next to the region F1s | +3 / −3 | `scripts/emotion_card.py`, `docs/MODEL_CARD.md` |
| Sat Oct 24 15:26 | Valence Web | `114e92e` e2e: --video any clip, --record observations as obs_to_dict JSONL (PersonObs fields only) | +30 / −5 | `frontend/e2e/harness.ts`, `frontend/e2e/run.mjs` |
| Sat Oct 24 15:27 | Valence Web | `ef2806d` data/browser: features-only observation streams from the real browser perception on two CC BY 4.0 clips | +2112 / −0 | `data/browser/female.jsonl`, `data/browser/male.jsonl`, `data/browser/README.md` |
| Sat Oct 24 15:31 | Evaluation | `ab05fa5` Emotion map on real browser cues (Cloud C's two CC BY 4.0 clips) | +418 / −1 | `artifacts/emotion/browser_captures.json`, `scripts/emotion_card.py`, `docs/MODEL_CARD.md` +1 more |
| Sat Oct 24 15:34 | Valence Web | `aec533d` Faces in IMAGE mode as on the edge; no head yaw/pitch from the head-crop retry; ?delegate= for the GPU check | +48 / −8 | `frontend/src/perception/perception.ts`, `frontend/src/perception/video.ts`, `frontend/e2e/REAL_CAMERA.md` +3 more |
| Sat Oct 24 15:35 | Valence Web | `29151a4` Status C 15:40: faces identical to the edge on the same frames (CPU), IMAGE mode, crop pose fix, console end to end | +33 / −0 | `docs/hackohio/status-C.md` |
| Sat Oct 24 15:36 | Valence Web | `c60fcc9` Engine: drop a reviewer's scratch test that read a session-local directory | +204 / −40 | `frontend/src/test/engine-runtime.test.ts`, `frontend/src/test/engine-aggregator.test.ts`, `frontend/src/test/fixtures/engine/features.json.gz` +1 more |
| Sat Oct 24 15:38 | Valence Web | `7c00652` e2e/soak.mjs: long run of the real perception with Chromium memory sampled every 30 s | +52 / −0 | `frontend/e2e/soak.mjs` |
| Sat Oct 24 15:41 | Evaluation | `eed6e31` Commit the smoother and m0-dump fixture generators | +447 / −0 | `scripts/export_web_golden_smoother.py`, `scripts/export_web_golden_lgbm.py` |
| Sat Oct 24 15:42 | Docs | `7cfc8a4` WEB_ENGINE.md: how Valence Web is put together, parity table, speed, reproducing fixtures | +97 / −0 | `docs/WEB_ENGINE.md` |
| Sat Oct 24 15:43 | Docs | `0433189` Log: 15:50 entry (Valence Web end to end, engine gate passed, tour demo) | +46 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 15:46 | Valence Web | `fa02cf9` Hosted site opens on the camera-free tour with a welcome banner | +58 / −6 | `frontend/src/components/console/panels.tsx`, `frontend/src/lib/valence.ts`, `frontend/scripts/check-static.mjs` +1 more |
| Sat Oct 24 15:47 | Valence Web | `8cf5186` Emotion map: anchor labels no longer overlap (side by valence, nudges in the crowded corner, tooltips) | +16 / −6 | `frontend/src/components/console/panels.tsx` |
| Sat Oct 24 15:48 | Pitch | `7c46f16` Hack OHI/O pitch draft: continuation disclosure, 3-minute demo script, Devpost text | +126 / −0 | `docs/HACKOHIO_PITCH.md` |
| Sat Oct 24 15:49 | Evaluation | `5194b3a` Runtime-test generators moved into scripts/golden_runtime/ (in progress) | +973 / −0 | `scripts/golden_runtime/extra_runs.py`, `scripts/golden_runtime/edge_cases.py`, `scripts/golden_runtime/specs.py` +3 more |
| Sat Oct 24 15:51 | Valence Web | `810d3c4` Soak: 15 min of real perception, no leak trend, decoded = purged, 0 violations | +21 / −0 | `docs/hackohio/status-C.md`, `frontend/e2e/README.md` |
| Sat Oct 24 15:54 | Valence Web | `c539e9c` CPU delegate by default (matches the native edge to ~1e-4 on identical frames); ?delegate=GPU to try the GPU | +40 / −13 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/e2e/REAL_CAMERA.md`, `frontend/src/perception/types.ts` +3 more |
| Sat Oct 24 16:10 | Evaluation | `c8b9f69` Runtime-test generators reproducible from the repo (scripts/golden_runtime/, check.py) | +352 / −17 | `scripts/golden_runtime/check.py`, `scripts/golden_runtime/README.md`, `frontend/src/test/engine-runtime.test.ts` +3 more |
| Sat Oct 24 16:36 | Emotion map | `7babd7e` Status B 16:38: docs re-verified, first real browser cues | +28 / −0 | `docs/hackohio/status-B.md` |
| Sat Oct 24 16:39 | Evaluation | `aee859e` Model card renders real self-report results once artifacts/emotion/self_report.json exists | +34 / −2 | `scripts/emotion_card.py` |
| Sat Oct 24 17:59 | Valence Web | `e0526ba` Fix 31 review findings in the Valence Web glue (kill switch, pause, camera prompts, honest status) | +1073 / −125 | `frontend/src/test/browser-source.test.ts`, `frontend/src/lib/browser-source.ts`, `frontend/src/test/browser-console.test.ts` +9 more |
| Sat Oct 24 18:00 | Docs | `81c1188` Log: 18:00 entry (glue review and fixes, hosted landing, real browser cues) | +39 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 18:03 | Valence Web | `91c3f63` Audit chain in the governance panel, downloadable and verifiable by the desktop's Python verifier | +98 / −2 | `frontend/src/test/engine-browser.test.ts`, `frontend/src/components/console/panels.tsx`, `frontend/src/lib/browser-source.ts` +3 more |
| Sat Oct 24 18:04 | Valence Web | `28902cd` Review pass: docs match the code, load failures read "could not load <url>" | +116 / −73 | `docs/hackohio/status-C.md`, `frontend/src/perception/README.md`, `frontend/src/perception/perception.ts` +18 more |
| Sat Oct 24 18:06 | Valence Web | `40352fe` Phones see the readings first: operator panels move below them on small screens | +9 / −3 | `frontend/scripts/check-static.mjs`, `frontend/src/routes/index.tsx` |
| Sat Oct 24 18:08 | Evaluation | `ff485b0` Model card: RAVDESS considered for the public-dataset stretch, hosts denied by the network policy | +10 / −4 | `docs/MODEL_CARD.md`, `scripts/emotion_card.py` |
| Sat Oct 24 18:09 | Pitch | `ba199f7` 75 s demo recording of the camera-free tour (docs/media) and a recorder script | +97 / −4 | `frontend/scripts/record-demo.mjs`, `frontend/src/lib/valence.ts`, `docs/media/README.md` +2 more |
| Sat Oct 24 18:09 | Pitch | `28302ce` Demo script: backup clip and the cross-language audit verification step | +6 / −1 | `docs/HACKOHIO_PITCH.md` |
| Sat Oct 24 18:17 | Valence Web | `a889852` In-browser Evaluation panel: held-out scores from the loaded models' metadata, labelled simulated | +79 / −7 | `frontend/src/engine/browser-engine.ts`, `frontend/src/test/engine-browser.test.ts`, `frontend/src/test/browser-source.test.ts` +3 more |
| Sat Oct 24 18:18 | Docs | `89ce0a2` README: Valence Web section up top (what moved into the browser, how to try it, where the docs are) | +32 / −0 | `README.md` |
| Sat Oct 24 18:51 | Emotion map | `b12eb08` Status B 18:53: waiting on taps, public data blocked by network policy | +24 / −0 | `docs/hackohio/status-B.md` |
| Sat Oct 24 19:50 | Valence Web | `56895ce` status-C: 19:45 check-in | +10 / −0 | `docs/hackohio/status-C.md` |
| Sat Oct 24 19:59 | Valence Web | `a075b9c` Judge-view pass: hosted page without operator chrome, readings first, honest tour and live labels | +491 / −127 | `frontend/src/components/console/panels.tsx`, `frontend/src/test/browser-console.test.ts`, `frontend/src/test/browser-source.test.ts` +7 more |
| Sat Oct 24 20:01 | Valence Web | `acdd410` REAL_CAMERA: start the camera from "Use my camera (on-device)" (header switch is gone) | +2 / −1 | `frontend/e2e/REAL_CAMERA.md` |
| Sat Oct 24 20:01 | Emotion map | `b0c10c5` Spec: self-report taps stay in the tab until Clear or close (Download keeps them); first prompt after warm-up | +6 / −4 | `docs/EMOTION_FEATURES.md` |
| Sat Oct 24 20:16 | Valence Web | `4f808e1` Judge-view follow-ups: one product name, mic-only purge log, per-person break wording, guarded metadata | +263 / −63 | `frontend/src/test/browser-console.test.ts`, `frontend/src/components/console/panels.tsx`, `frontend/src/test/browser-source.test.ts` +6 more |
| Sat Oct 24 20:16 | Docs | `7677957` Log: 20:15 entry (judge's-eye pass on the hosted page, B and C status) | +44 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 20:32 | Pitch | `4ab5ea6` Re-record the 75 s tour clip on the current page (Valence Web header, readings first, honest tour labels) | +0 / −0 | `docs/media/valence-web-tour.mp4` |
| Sat Oct 24 20:34 | Docs | `378ee86` Finish-line checklist for the user (deploy, live check, demo, Devpost, final tag, self-report) and browser support | +90 / −0 | `docs/HACKOHIO_FINISH.md`, `docs/DEPLOY_WEB.md` |
| Sat Oct 24 20:46 | Valence Web | `ad7082e` Perception robustness from the bug hunt: muted/ended tracks, no-frame camera, iOS audio interruptions | +377 / −54 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/perception.ts`, `frontend/src/perception/capture.ts` +4 more |
| Sat Oct 24 20:50 | Valence Web | `e403179` Governance counters stay per source across Pause → Start (a past violation never resets) | +51 / −3 | `frontend/src/lib/browser-source.ts`, `frontend/src/test/browser-source.test.ts` |
| Sat Oct 24 20:52 | Valence Web | `5f2a8fd` Status names a muted camera or mic once: perception's reason replaces the bare device name | +19 / −1 | `frontend/src/test/browser-source.test.ts`, `frontend/src/lib/browser-source.ts` |
| Sat Oct 24 20:56 | Valence Web | `48694ef` Person box stays full-extent: measured against YOLO, a visibility gate shrinks the participant | +7 / −0 | `frontend/src/perception/README.md`, `docs/hackohio/status-C.md` |
| Sat Oct 24 21:09 | Evaluation | `474628f` One command from taps to model card: self_report writes the artifact, figure and card section | +108 / −8 | `artifacts/emotion/card_stats.json`, `valence/eval/self_report.py`, `scripts/emotion_card.py` |
| Sat Oct 24 21:10 | Emotion map | `a93029a` Status B 21:12: self-report path ready end to end | +18 / −0 | `docs/hackohio/status-B.md` |
| Sat Oct 24 21:10 | Docs | `64c0c00` Finish checklist: run the self-report evaluation yourself if no cloud session is up (outputs to commit) | +7 / −1 | `docs/HACKOHIO_FINISH.md` |
| Sat Oct 24 21:15 | Valence Web | `c34942f` Review of the robustness fixes: hidden time, stale audio chunk, guarded releases | +128 / −27 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/perception.ts`, `frontend/src/perception/capture.ts` |
| Sat Oct 24 21:50 | Valence Web | `2f9a586` status-C: 21:45 check-in (bug hunt fixed, browser support) | +36 / −0 | `docs/hackohio/status-C.md` |
| Sat Oct 24 22:18 | Docs | `57cea76` Log: 22:15 entry (perception bug hunt, counters per source, finish-line checklist, gates green) | +36 / −0 | `docs/HACKOHIO_LOG.md` |
| Sat Oct 24 22:31 | Valence Web | `212d79f` Bug hunt round 2: Safari 16 worker fallback, WebGL context loss, pauses, portrait frames | +257 / −18 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/models.ts`, `frontend/src/test/perception-pipeline.test.ts` +8 more |
| Sat Oct 24 22:33 | Valence Web | `ac1a24e` Revalidate /mediapipe/* and /models/* on each load (no-cache) instead of a 1-day / 1-hour max-age | +5 / −3 | `frontend/scripts/security-headers.mjs` |
| Sat Oct 24 23:04 | Valence Web | `f1de56c` Review of round 2: pauses keep seats and baselines, heaps wiped, late yaw/pitch calibration | +169 / −91 | `frontend/src/test/perception-pipeline.test.ts`, `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/models.ts` +6 more |
| Sat Oct 24 23:16 | Emotion map | `6741b3d` Spec: record the browser's late calibration of a cue channel with no samples at the freeze (f1de56c) | +4 / −0 | `docs/EMOTION_FEATURES.md` |
| Sat Oct 24 23:16 | Emotion map | `0b70c7c` Status B 23:18 | +13 / −0 | `docs/hackohio/status-B.md` |
| Sat Oct 24 23:18 | Valence Web | `69d00c8` Break-length pause measured on wall time too; tests pin the YAMNet heap wipe, close failures, late calibration | +57 / −15 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/test/perception-face.test.ts`, `frontend/src/perception/video.ts` +2 more |
| Sat Oct 24 23:50 | Valence Web | `4ec8845` status-C: 23:45 check-in (bug hunt round 2, reviews of the fixes) | +28 / −0 | `docs/hackohio/status-C.md` |
| Sun Oct 25 00:18 | Docs | `cada54d` Log: 00:15 entry (perception round 2, reviews catch regressions, caching fixed, gates green) | +27 / −0 | `docs/HACKOHIO_LOG.md` |
| Sun Oct 25 00:23 | Valence Web | `d659c41` Round 3: real-camera checklist matches the deployed console; mock follows the status contract | +120 / −40 | `frontend/e2e/REAL_CAMERA.md`, `frontend/src/perception/mock.ts`, `frontend/src/test/perception-interface.test.ts` +5 more |
| Sun Oct 25 00:27 | Valence Web | `a99dbbc` Cloud C round 3: Pause → Start with several people seated starts them fresh; "no microphone" once; honest purge note | +37 / −9 | `frontend/src/test/browser-source.test.ts`, `frontend/src/lib/browser-source.ts`, `docs/DEPLOY_WEB.md` +3 more |
| Sun Oct 25 00:31 | Valence Web | `644ec53` e2e: portrait phone frames (720x1280) as a seventh scenario, with a frame-size check | +37 / −15 | `frontend/e2e/make_media.py`, `frontend/e2e/harness.ts`, `frontend/e2e/run.mjs` +2 more |
| Sun Oct 25 00:33 | Valence Web | `37faa3d` Pause → Start always starts the people fresh (a lone presenter can come back under another id too) | +13 / −14 | `frontend/src/lib/browser-source.ts`, `frontend/src/test/browser-source.test.ts`, `frontend/src/components/console/panels.tsx` +1 more |
| Sun Oct 25 01:21 | Emotion map | `4928c23` Status B 01:21 | +12 / −0 | `docs/hackohio/status-B.md` |
| Sun Oct 25 01:50 | Valence Web | `7d189ed` status-C: 01:45 check-in (round 3, portrait e2e, 40-min soak) | +38 / −0 | `docs/hackohio/status-C.md` |
| Sun Oct 25 01:58 | Valence Web | `2af2fa0` e2e: --cpu-throttle and --profile; document the main thread's frame-grab cost | +23 / −1 | `frontend/e2e/run.mjs`, `frontend/src/perception/README.md`, `frontend/e2e/README.md` |
| Sun Oct 25 02:17 | Docs | `aa53b46` Log: 02:15 entry (round 3 fixes, Pause → Start starts people fresh, 40-min soak) | +26 / −0 | `docs/HACKOHIO_LOG.md` |
| Sun Oct 25 02:33 | Valence Web | `d87fbd6` stop() settles the frame in flight; a lost WebGL context stops video with its own reason | +88 / −9 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/perception.ts`, `frontend/src/perception/README.md` |
| Sun Oct 25 02:41 | Valence Web | `52fc8bb` start() does not report the camera live when video already stopped on a lost WebGL context | +29 / −2 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/perception.ts` |
| Sun Oct 25 02:51 | Valence Web | `e276b25` Main-thread fallback rests half a step after each step so a slow page stays usable | +38 / −1 | `frontend/src/test/perception-orchestrator.test.ts`, `frontend/src/perception/perception.ts`, `frontend/src/perception/README.md` |
| Sun Oct 25 03:29 | Emotion map | `b993e37` Status B 03:29: full suite green, no taps | +9 / −0 | `docs/hackohio/status-B.md` |
| Sun Oct 25 03:50 | Valence Web | `430f8c9` status-C: 03:45 check-in (final review, slow devices, fresh clone) | +25 / −0 | `docs/hackohio/status-C.md` |
| Sun Oct 25 03:51 | Valence Web | `0a59ac3` status-C: a 'Read first' summary at the top | +23 / −0 | `docs/hackohio/status-C.md` |
| Sun Oct 25 04:18 | Docs | `eed9f0b` Log: 04:15 entry (final reviews before the freeze, gates green) | +23 / −0 | `docs/HACKOHIO_LOG.md` |
| Sun Oct 25 04:57 | Emotion map | `caf3595` Status B: a 'Read first' summary at the top | +33 / −1 | `docs/hackohio/status-B.md` |
| Sun Oct 25 05:51 | Valence Web | `62f86cd` status-C: 05:45 check-in | +10 / −0 | `docs/hackohio/status-C.md` |
| Sun Oct 25 06:17 | Docs | `a9a27ae` Log: 06:15 entry (quiet before the freeze) | +10 / −0 | `docs/HACKOHIO_LOG.md` |
| Sun Oct 25 06:33 | Docs | `d2256e9` Number and claim audit of the judge-facing docs: 14 corrections | +40 / −14 | `docs/HACKOHIO_FINISH.md`, `docs/DEPLOY_WEB.md`, `docs/WEB_ENGINE.md` +2 more |
| Sun Oct 25 07:47 | Valence Web | `fa08e51` status-C: 'Read first' summary at the freeze state | +10 / −6 | `docs/hackohio/status-C.md` |
| Sun Oct 25 07:50 | Valence Web | `a16f09f` status-C: 07:45 final check-in before the freeze | +11 / −0 | `docs/hackohio/status-C.md` |
<!-- hacklog:end -->
