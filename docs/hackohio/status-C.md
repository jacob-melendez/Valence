# Cloud C status: browser perception (`[web]`)

Owner: Cloud C. Scope: `frontend/src/perception/**`, `frontend/src/test/perception*`, `frontend/public/mediapipe/**`,
`frontend/e2e/**`, `scripts/export_perception_fixtures.py`, the MediaPipe dependencies. Times are America/New_York.
Newest entry last.

## Read first (state at the 08:00 freeze, Oct 25)

- **What works.** The browser perception (MediaPipe Tasks Web 0.10.35, self-hosted) produces the Python edge's
  VideoObs and AudioObs with parity-tested ports. The video models run in a Web Worker on the CPU delegate; YAMNet
  runs on the page's thread. It feeds the console end to end. Nothing leaves the origin, and it runs under the strict
  CSP. Frames and audio are purged and counted.
- **Gates at the freeze.**
  - 944 frontend tests pass.
  - `e2e/run-all.mjs` passes seven scenarios (one, two, roll ±15°, portrait 720×1280, camera denied, mic denied).
  - A 40-minute soak showed no leak.
  - A fresh clone builds and passes the deploy check.
- **Robustness covered.** Missing, denied, muted or ended camera or mic; a hidden tab; iOS audio interruptions; a
  lost WebGL context; Safari before 17 (the page's thread, with a rest between steps); slow devices. Every change
  tonight had its own adversarial review, and the regressions they found are fixed (log 21:45–03:45).
- **Needs a real camera** (`frontend/e2e/REAL_CAMERA.md`, ~15 min after deploy):
  - fps on a real Mac and phone;
  - slump and keyboard-gaze with BlazePose;
  - GPU against CPU;
  - Safari's camera hand-off between tabs.
- **Known limits.**
  - The first 20 s on camera are the face and posture calibration: sit neutral.
  - Blink counts undercount below ~15 Hz.
  - A camera that stalls without being muted repeats its last frame until the browser mutes it.
  - On the page's thread (Safari before 17), video is slower.
  - Each frame grab costs ~5.5 ms on the page's thread (headless); `VideoFrame` would cut it to ~0.1 ms but is
    untested on Safari and Firefox.

## Decisions

| Decision | Why |
|---|---|
| **MediaPipe Tasks Web 0.10.35** (`@mediapipe/tasks-vision`, `@mediapipe/tasks-audio`), pinned exactly | Published 2026-04-27, well past the 24 h guard. Every 1.x release (1.0.0 to 1.1.1) bundles a usage logger that POSTs to `odml.pa.googleapis.com/v1/log` every 60 s, created unconditionally in `createFromOptions`. Under our CSP it would be blocked, but it would still be a third-party attempt and a CSP violation. 0.10.35 has no such code: its bundle contains no external URL. The Python edge runs mediapipe 1.1.1 with the same `.task` models. |
| Observations keep NaN for "no reading" | Same as `contracts.py`; `obsToDict` writes null like `obs_to_dict`, and `obsFromDict` reads the Python JSONL recordings unchanged (tested on `data/real/capture_fists.jsonl`). |
| A `type` tag on each observation | The same key `obs_to_dict` writes, so TypeScript gets a discriminated union and recordings parse as-is. |
| `audio_blocks_purged` counts ~100 ms raw mic buffers | Same meaning as the Python edge, where the gate counts each capture part. 1 s blocks and resampled copies count under `derived_purged`. |
| Model and WASM files self-hosted, never committed | `frontend/scripts/mediapipe-assets.mjs` (prebuild and predev) copies the WASM from `node_modules` and downloads the four models from Google's model bucket with pinned SHA-256s (the files `scripts/download_models.sh` gives the Python edge). The Cloudflare build (`npm install && npm run build`) has internet access; the browser never contacts Google. |
| CPU delegate by default (15:55) | On identical frames the CPU path matches the native edge to ~1e-4. The GPU path is unmeasured on real hardware and drifted 100–300x more on software WebGL. `?delegate=GPU` tries it, falling back to CPU only if it fails to start; on software WebGL it runs, ~10x slower (pose 660 vs 68 ms per frame). |
| Faces on every frame by default (14:50) | The emotion cues' blink detector needs the face rate: 7.2 fps with 7.2 face passes/s, against 8.4 fps with 4.2 at the Python cadence (`faceEveryFrame: false`); 22 and 24 blinks instead of 13 and 15 on the two real-person clips (`data/browser`). `status().face_hz` reports the rate. |
| Faces in IMAGE mode, as on the edge (15:40) | VIDEO mode was bit-identical wherever the detector fired and only added a stale ghost face about once per 2000 frames. The head-crop retry gives no yaw/pitch (its matrix is in the crop's own camera), only roll. |
| Video models in a Web Worker | MediaPipe takes 10-100 ms per frame; in a worker the console's main thread stays free. Main-thread fallback where workers can't run it. |
| Pose: PoseLandmarker lite + a simple IoU tracker | Stands in for YOLO26n-pose + ByteTrack (track buffer 6 s, as `bytetrack_valence.yaml`). BlazePose landmarks 0,2,5,7,8,11-16,23-28 map to COCO 0-16; confidence = BlazePose visibility, 0 outside the frame. The person box is the landmarks' extent with the top raised above the eyes. |
| Up to 6 poses tracked, then the 25% filter and largest 3 (21:00) | Capped at 3, MediaPipe's VIDEO mode stops detecting once 3 poses are tracked, so small bystanders seen first could hide a participant who sits down close. YOLO hands the filter every detection. |
| Person box keeps BlazePose's predicted legs (21:00) | Measured against YOLO on 486 synthetic frames (desk-occluded bystanders, seated participants): bystanders' boxes run 0.06 taller, but BlazePose finds only a third of them and none next to a close participant. Gating out landmarks with visibility < 0.5 shrank the main participant's box at 720 px from 0.85 to 0.57 of the frame (YOLO 0.97). |
| Muted tracks pause, ended tracks release (21:00) | A muted camera (Safari giving it to another tab, Chromium after ~25 missed frames) would repeat its last frame or black, and a muted mic renders silence: video pauses and audio is purged, both reported in `degraded`. A track that ends, with or without its event, releases its models at once. |
| Pose detection and presence thresholds 0.4 | The Python edge's YOLO person threshold. MediaPipe's default 0.5 found one of the two people in the two-person scene (15/15 frames); 0.4, 0.3 and 0.2 all find both. |
| Hips, knees, ankles (COCO 11-16) never confident | BlazePose always predicts a whole body: on the waist-up portrait it put the hips at 72% of the frame height with visibility 1.00 (knees 0.5-0.6, ankles 0.3). YOLO at a desk gives hidden hips low confidence, and of the Python rules only motion reads them (COCO 5-12), which leaves shoulders, elbows and wrists. |
| Posture on One Euro-smoothed keypoints, motion on the shoulders only (14:00) | BlazePose keypoints jitter (white noise) where YOLO's do not: raw, motion read 0.46 sw/s on a still scene against YOLO's 0.026. Smoothed shoulders read 1.3x YOLO on still and drifting scenes and keep 0.93x of real motion (`smoothing.ts`). |
| Confidence thresholds 0.3 / 0.5 / 0.6 kept | BlazePose visibility on the e2e scenes: nose, eyes, ears, shoulders 1.00; elbows 0.6-0.8; partly hidden wrists 0.45-0.8 (YOLO usually reports 0.8-0.95 for visible wrists, so a wrist near 0.5 may miss the hand assignment: on the real-camera checklist). |
| `frontend/package-lock.json` re-synced | `npm ci` failed on the committed lock (ajv 6/8 entries missing). `npm install` fixed it while adding MediaPipe; `bun.lock` got the two MediaPipe entries by hand (bun can't reach the private registry its lock names for `@lovable.dev/vite-tanstack-config`). |

## Log

### 12:33 — Interface pushed

- `frontend/src/perception/`: `types.ts` (PersonObs / VideoObs / AudioObs mirrors, raw extras: 52 blendshapes and
  head yaw/pitch/roll in degrees, counters, status), `createPerception()` (stub: reports a "stub" degradation and
  emits nothing), `createMockPerception()` (10 Hz video, 1 s audio, deterministic per seed, a 180 s script per
  person: focused, upbeat, frustrated, worn out), `openMediaStream()` (echo cancellation, noise suppression and AGC
  off; falls back to camera-only or mic-only).
- `faceValence.ts`: port of `face_valence.py` (formula, calibrator, Hungarian face assignment), used by the mock.
- Tests: `src/test/perception-interface.test.ts` (types, JSONL round trip, mock cadence and content, media fallbacks).
- Next: the pure ports with parity fixtures, then the MediaPipe pipeline and the headless-Chromium end-to-end test.

### 12:35 — Emotion cues in the interface

Cloud B's `docs/EMOTION_FEATURES.md` (emotion-v1) landed: `PersonObs` gains `brow_down`, `eye_open`, `mouth_frown`,
`smile`, `head_yaw`, `head_pitch` and `blink_onsets`. They are in `types.ts` (required; `obsFromDict` gives older
recordings the Python defaults), `faceCues.ts` ports `valence/features/face_cues.py` (baseline, blink detector, head
pose from the matrix), and the mock fills them. The raw extras follow B's sign convention (pitch > 0 = face up).

### 12:57 — MediaPipe runs under the strict CSP

- A throwaway probe loaded all four models from `/mediapipe/` under `default-src 'self'; script-src 'self'
  'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:` in headless Chromium with a fake camera playing
  the test portrait: 0 off-origin requests, no CSP errors, 1 face with 52 blendshapes (smile 0.90), 1 pose, matrix
  column-major with the translation at (-4, 23, -70) cm as `docs/EMOTION_FEATURES.md` assumes.
- Load: vision models 0.93-1.0 s, YAMNet 0.2 s (local server, warm disk). Per frame on this 4-core container
  (CPU delegate, other jobs running): pose 68 ms, face 33 ms, gestures 58 ms, `createImageBitmap` 6 ms; YAMNet 76 ms
  per 0.975 s clip.
- Pushed: the asset script and prebuild hook, the mic worklet, the e2e media (public-domain NASA portrait from
  scikit-image, CC BY 4.0 LibriSpeech clip) and their licenses.
- In progress: the ports of `audio_features.py` + scipy `resample_poly` (done: bit-identical to scipy), `posture.py`,
  `hands.py` + `ids.py`, `lux.py` + `regions.py` + OpenCV's BGR2GRAY, and the face parity checks run as parallel
  agents, each followed by an adversarial review. The video pipeline (`video.ts`), the audio pipeline, the capture
  loop and the worker are written and wait on them.

### 13:10 — Ports landing, pose checked on the test scenes

- Pushed `e8b552a`: `audio_features.py` + scipy `resample_poly` (bit-identical to scipy in float32, also for
  44.1 kHz) and `posture.py`, with 124 parity and unit tests on 2.4 MB of gzipped fixtures written by
  `scripts/export_perception_fixtures.py`; plus the pose glue, capture, model loading, the audio pipeline and the
  e2e runner. A clean worktree of that commit: 212 frontend tests pass, `tsc` clean, `npm run build` OK.
- BlazePose on the e2e scenes (CPU delegate): 70-100 ms per frame on this container; the threshold and lower-body
  decisions above.
- Running: the `hands.py` + `ids.py`, `lux.py` + `regions.py` + BGR2GRAY ports and the face parity agent; the
  audio and posture adversarial reviews.

### 13:30 — Real perception runs end to end in headless Chromium

`node frontend/e2e/run-all.mjs`: the real `createPerception()` on Chromium's fake camera and microphone (the scenes
from `make_media.py`), served with the brief's CSP. Every check passes in every scenario: faces with 52 blendshapes,
observations flowing, speech heard, frames decoded = purged, **0 off-origin requests, 0 CSP violations, 0 page
errors**.

| Scenario | Load | fps | ms/frame | Backend | People | Notes |
|---|---|---|---|---|---|---|
| one (portrait, drifting) | 1.5 s | 8.5 | 110 | worker/CPU | 1 | speech 0.99, lux 239 (q 0.35), face valence 0 (its own baseline) |
| two (side by side) | 1.7 s | 5.8 | 153 | worker/CPU | 2 | each face assigned to its own person |
| roll_cw / roll_ccw (±15°) | 1.5 / 1.6 s | 8.0 / 8.6 | 117 / 112 | worker/CPU | 1 | roll +17.9 / −10.7 vs +3.0 upright: **sign confirmed** (clockwise > 0) |

Audio: 22-35 ms per 1 s block (48 kHz -> 16 kHz resample + YAMNet). Load = `start()` to the first observation, local
server, 4-core container without a GPU (CPU delegate, chosen automatically on SwiftShader).

- Download per first visit (simd variants, one copy each): vision WASM 11.2 MB + loader 0.3 MB, audio WASM 6.3 MB,
  models 22.0 MB (face 3.8, pose 5.8, gestures 8.4, YAMNet 4.1); the e2e server sends `no-store`, so it re-serves
  the vision WASM per task (4x).
- Fixed on the way: in a module worker MediaPipe's loader is an ES module, imported once per URL, so the second
  task found no `ModuleFactory` and the worker fell back to the main thread; each task now gets its own loader URL.
- **Motion differs from YOLO**, as the plan feared: on the near-still portrait (true drift ~0.06 sw/s) BlazePose
  head/shoulder points jitter 0.10-0.21 sw/s and elbows/wrists 0.3-0.8, so PostureTracker's motion reads ~0.45 sw/s
  (the Python edge read 0.033 upright, live). An agent is running YOLO26n-pose and BlazePose side by side on
  identical synthetic frames (still, drift, slump, known-speed motion) to pick a fix (smoothing or keypoint gating).
- Reviews done: audio no differences; posture two fixed (numpy's fused norm, null as NaN); face five fixed (scipy's
  LSA solver ported, numpy-exact distances, CPython float modulo, Python min/max with NaN, `math.degrees`); hands/ids
  exact (CPython `math.hypot` ported).
- Face hold scales with the device: 0.45 s = 2.25 face passes at 10 Hz; slower devices hold 2.25 measured passes.

### 14:00 — YOLO vs BlazePose side by side; motion fixed; glue reviewed

**Side by side on identical frames.** YOLO26n-pose and BlazePose lite were each run through the real Python
`PostureTracker`, YOLO with ultralytics 8.4.171 as `video.py` calls it, BlazePose through MediaPipe 0.10.35 in
Python with the `coco.ts` mapping. The sequences:
- synthetic 1280x720, 10 fps, sensor noise σ 3: still, drift, slump 35% and 50%, known-speed motion, fidget;
- two CC BY 4.0 clips of real people framed head and shoulders (intel-iot-devkit sample videos).

Python reproduces the browser's numbers, so the gap is the model, not the browser runtime.

| motion p50 (sw/s) | still | drift | 0.3 sw/s | 0.1 sw/s | real F | real M |
|---|---|---|---|---|---|---|
| YOLO | 0.026 | 0.072 | 0.366 | 0.173 | 0.036 | 0.022 |
| BlazePose, raw | 0.464 | 0.613 | 1.090 | 0.872 | 0.120 | 0.059 |
| BlazePose + One Euro + shoulders only (**shipped**) | 0.054 | 0.148 | 0.557 | 0.264 | 0.033 | 0.013 |

- **Cause.** BlazePose jitter is white noise: lag-1 autocorrelation ≈ 0, and VIDEO mode returns the same
  landmarks as IMAGE mode. Head and shoulders move 0.11–0.22 sw/s on a still frame, elbows and wrists 0.5–1.0;
  YOLO's points move 0.004–0.02.
- **Fix.** Of ~1000 variants, the best was a per-keypoint One Euro filter (0.2 Hz, β 8 per sw/s, d 0.5 Hz) with
  motion on the shoulders only. Still and drift read 1.3× YOLO and real motion 0.93×. The TS filter matches the
  evaluated Python one within 1e-6 on 2346 recorded frames. In the e2e test, motion on the drifting portrait went
  from 0.46 to 0.13.
- **Slump, open.** On the spacesuit portrait BlazePose places the shoulders on the neck ring, and they slide down
  with a pasted-lower head, so both synthetic slumps were missed (YOLO caught them).
- **Keyboard gaze, open.** On real head pitch-down, BlazePose's ears drop to 0.74× baseline against YOLO's 0.92×,
  so the 0.93 ear rule that keeps keyboard gaze from counting as a slump protects less: one 3.8 s false slump in
  one clip. Both are on the real-camera checklist; the thresholds stay Python's until a real check says otherwise.

**Glue review** (two independent reviewers, every finding proven by a failing test first). One fidelity fix: a
missed face pass inside the hold no longer resets the blink detector or stalls the cue baseline. Nine
privacy/robustness fixes:
- the worker's start timeout no longer kills a slow first download;
- a suspended AudioContext no longer hangs `start()`, and every gesture retries it, touch included;
- an ended mic stops producing AudioObs from silence;
- stop/start races are closed: per-run counters and generation checks after each open;
- failed opens release their models at once;
- worklet chunks still queued on close are zero-filled;
- a hidden tab is not mistaken for a slow device;
- `stop()` clears the DevTools handle.

**Tests now.** 720 frontend tests pass, 300+ of them perception, with parity fixtures for every ported module.
`e2e/run-all.mjs` passes in all four scenarios at 5.8–8.4 fps on this CPU-only container.

### 14:50 — Audio checked against the edge on identical sound; faces every frame; more e2e

**Audio side by side.** The test file was 90 s: room noise, LibriSpeech speech (CC BY 4.0) and eight ESC-50
laughing clips (CC BY-NC, kept out of the repo). It played through Chromium's fake microphone into the real
`createPerception()`, and through the edge's real `AudioPerceiver` (Python MediaPipe 0.10.35, compliments off).
- **The model is identical.** Browser (WASM) and Python YAMNet scored the same 15600-sample windows bit for bit:
  199/199 windows, all 6 classes, max |diff| 0.0.
- **End to end, given the exact same 1 s span:** speech r 0.999, laughter-family r 0.998, 100% on the same side of
  the 0.5 / 0.05 thresholds, db_spl +0.001 dB on average (0.000 on clean blocks), and **laughter onsets in exactly
  the same blocks** in all three runs (7 of 8 laughs; the 8th falls in the 5.5 s refractory on both sides).
- **What differs is only where the blocks fall.** The browser's 1 s grid starts at an arbitrary point in the sound,
  and the edge's own scores move as much when its grid shifts (speech r 0.98, laughter r 0.94 at 100 ms). The
  onset count stays the same.
- **Chromium findings, not bugs in the page:**
  - The headless AudioContext runs at 44.1 kHz; the 160/441 resample branch is exercised and correct.
  - When capture hand-off runs dry under CPU load, Chromium inserts 10–40 ms buffers of zeros (71–252 ms per 200 s).
    They shift the grid and make the affected blocks read up to 0.2 dB low.

**Faces on every frame (default).** For the emotion cues' blink detector. On this CPU-only container: 7.2 fps with
7.2 face passes/s, against 8.4 fps with 4.2 at the Python cadence (`faceEveryFrame: false`).
`status().face_hz` reports the rate, and Cloud A has the field.

**e2e.** `run-all.mjs` passes six scenarios, including a refused camera (audio alone, "no camera") and a refused
microphone (video alone, "no microphone"), simulated as `getUserMedia` NotAllowedError. Main-thread long tasks
during a 25 s run: **1 (171 ms) with the worker, 158 totalling 22.8 s without it**, both at 7.6 fps. That is the
case for the worker.

**Reference drift check.** `scripts/export_perception_fixtures.py` now runs at every status. It caught Cloud B's
`_avg` change (a NaN blendshape side counts as missing); the port follows it (`95d0f14`). At 14:47 all fixtures
reproduce byte for byte.

**Interpretation for Cloud B (emotion spec section 1).** With faces on alternating frames (`faceEveryFrame: false`),
a frame between face passes, or a face pass that misses while the face is still held (`face_visible` true within
the 0.45 s hold), counts as visible. It neither enters the baseline nor resets the blink detector. Only frames whose
hold has expired count as "not visible" (`update(None)`). Otherwise a single missed pass would erase a blink in
progress.

### 15:40 — Faces checked against the edge on identical frames; whole console on real perception

**Faces, side by side.** The browser (WASM) and the edge (native MediaPipe 0.10.35) ran on identical frames: the e2e
scenes and two CC BY 4.0 clips of real people (intel-iot-devkit). Each frame carried an index marker, and colour was
converted with libyuv's I420ToARGB, which matches Chromium bit for bit (verified on full-frame dumps).
- **Implementation parity (CPU delegate).** Blendshapes agree to p95 1.3e-4, valence_raw to 1.1e-4, head yaw and
  pitch to 0.004°, landmarks to 0.003 px, and the face count matches on every frame.
- **PersonObs against an IMAGE-mode edge replica on the browser's timestamps.** face_valence and all cues agree at
  r ≥ 0.987 (mostly ≥ 0.998), with median |d| ~1e-5. Blinks: 22 vs 22 on female, 24 vs 23 on male.
- **VIDEO vs IMAGE mode.** The output was bit-identical on all 2297 frames where the detector fired. The detector
  runs every frame either way, and VIDEO only added a stale ghost face about once per 2000 frames. **Faces now run in
  IMAGE mode, as on the edge.**
- **Defect fixed.** The head-crop retry's matrix is in the crop's own camera: yaw and pitch were off by a median
  2–3° (p95 6°), and the head_pitch cue stepped ~8.5° at each full-frame/crop switch. Crop hits now report yaw and
  pitch as missing and keep roll.
- **Open.** On software WebGL the GPU delegate drifts 100–300× more than CPU (face_valence p95 0.06, landmarks
  ~1 px). Real GPUs are unmeasured, so `?delegate=CPU|GPU` on the console URL and REAL_CAMERA.md step 6 compare them.
- **Cadence.** Faces on every frame, against the edge's odd frames, give 22 and 24 blinks instead of 13 and 15 on
  the two clips: the reason for the default.

**The whole console on real perception.** Production build, `/?source=browser`, fake camera and mic fed with the
portrait and speech (new `--fake-video` / `--fake-audio` flags in `scripts/check-static.mjs`).
- **Panels:** Person #1 tracked, a prediction with all four layers and template text, the emotion map, and the
  governance panel: 236 frames decoded, 235 purged, 428 audio blocks purged, 0 violations.
- **Network and errors:** 0 off-origin requests, 0 CSP violations, 0 page errors. Screenshot in
  `frontend/e2e/console-in-browser.jpg`.

**For Cloud B.** `data/browser/{female,male}.jsonl`: one 132 s pass of each CC BY clip as features-only
`obs_to_dict` lines with the emotion-v1 cues, for its real-vs-simulated spread check. Source, license and stats are
in `data/browser/README.md`.

**Fixtures.** At 15:38 every one reproduces byte for byte.

### 15:55 — 15-minute soak

`frontend/e2e/soak.mjs`: the real perception in headless Chromium for 15 min (worker, CPU delegate), with every
Chromium process's RSS sampled every 30 s.
- **Throughput:** 6.7 fps sustained.
- **Counters:** 5,646 frames decoded and 5,645 purged (one in flight), 8,981 audio chunks purged, 0 violations,
  0 page errors.
- **Main thread:** 3 long tasks, 234 ms in total.
- **Memory:** total Chromium RSS moved in a sawtooth between 1.60 and 1.65 GB and fell back after each collection
  (at 4.5 and 12 min). No leak trend.

### 17:45 — check-in: review pass, clearer load errors

- **Health.** No fixture drift. The frontend suite passes (862 tests) and the build passes. `e2e/run-all.mjs`: all
  six scenarios pass at 4.5–6.8 fps (worker/CPU), 0 off-origin requests, 0 CSP violations; the roll-sign and
  two-people cross checks hold.
- **Fresh-eyes review** of docs against code (a separate agent):
  - Fixed several docs that still said GPU is the default or that GPU falls back to CPU on software WebGL. An explicit
    GPU request runs on GPU (headless: 0.5 fps), so the default stays CPU.
  - Also fixed docs that described faces on odd frames only and motion on elbows and wrists.
  - `download_bytes` counts only the page's own fetches: 10.7 MB with the worker, against ~40 MB served.
  - Models are 22.0 MB. Added "from a fresh clone" setup steps to the e2e README.
  - Removed six unused members (tsc clean).
- **Load errors (Cloud A's report).** A WASM loader that fails to load used to surface as `[object Event]`. It now
  reads `could not load <url>`, with a test.
- **Restart (Cloud A's question).** `stop()` then `start(freshStream)` is safe: every per-run object (frame source,
  mic tap, pipelines, trackers, calibrators) and the counters start fresh, and a late report from the old run cannot
  land in the new one.
- **Docs.** `e2e/README.md`'s console check now expects "Top drivers by layer" (Cloud A's request).

### 19:45 — check-in

- **Health.** No fixture drift. The frontend suite passes (898 tests, including the other teams' new ones) and tsc is
  clean. `e2e/run-all.mjs`: all six scenarios pass at 4.5–7.5 fps (worker/CPU), with 0 off-origin requests and 0 CSP
  violations. The roll-sign and two-people cross checks hold.
- **Since 17:45.** Nobody else touched the perception module or the Python it ports.
- **Now.** A bug hunt over the module, with one adversarial verifier per finding, across six areas: lifecycle,
  privacy and purge counts, the audio path at non-48 kHz rates, video parity, Safari/Firefox compatibility, and the
  contract and its use in the console. Fixes for confirmed findings land as separate commits.

### 21:45 — bug hunt fixed, browser support

- **Health.** No fixture drift. 928 frontend tests pass and the build passes. `e2e/run-all.mjs`: all six scenarios
  pass at 4.3–7.6 fps (worker/CPU), with 0 off-origin requests and 0 CSP violations; both cross checks hold.
- **Bug hunt.** Six finders covered the module, and each finding then had a verifier try to refute it. 14 of 16
  findings held, which overlap to 11 issues:
  - **Fixed in ad7082e:**
    - a camera that never sends a frame hung `start()`, because `play()` was awaited;
    - muted tracks: Safari's other-tab capture and Chromium's stall mute gave frozen frames or a "silent room";
    - tracks ending without a seen event, which left status live and models loaded;
    - stale video status after the camera ended;
    - YAMNet's last window lingering in MediaPipe's WASM heap;
    - iOS audio interruptions never noticed;
    - the pose cap applied before the height filter, so 3 small bystanders could hide a participant;
    - the worker dying on iOS Chrome/Firefox (MediaPipe's user-agent check calls `document` in the worker);
    - fps counting failed steps.
  - **Measured, kept as is:** the person box reaching BlazePose's predicted legs (Decisions table).
  - **Cloud A's code:** governance counters went back to 0 after Pause → Start. Cloud A fixed it in e403179.
- **Review of those fixes** (three lenses, plus verifiers): 5 findings held, which come to 4 issues, all fixed in
  c34942f:
  - the 10 s no-frame deadline counted time in the background, so a phone switching apps while the models load lost
    the camera;
  - a chunk posted just before an audio suspension cleared the warning;
  - releases that throw skipped the mic-ended status;
  - "audio step failing" outlived the mic.
- **Browser support for camera mode.** Measured on headless Chromium only; the rest comes from the code and the
  browsers' published support:
  - **Chrome/Edge desktop and Android:** worker, CPU delegate (tested).
  - **Safari 17+ on macOS and iOS:** worker. MediaPipe needs WebGL in an OffscreenCanvas there, which Safari has had
    since 17. AudioWorklet and `wasm-unsafe-eval` are supported.
  - **iOS Chrome/Firefox (WebKit):** worker since ad7082e; before that they fell back to the main thread.
  - **Safari 16.4–16.x and Firefox < 114** (no WebGL in a worker, or no module workers): the worker fails and video
    runs on the page's thread (`video_backend` reads `main/CPU`). It works, but the UI is slower.
  - **Not testable here:** a real Safari, Firefox or phone. `frontend/e2e/REAL_CAMERA.md` steps 1 and 5.5 cover it
    after deploy.

### 23:45 — bug hunt round 2, and reviews of my own fixes

- **Health.** No fixture drift. 937 frontend tests pass and the build passes. `e2e/run-all.mjs`: all six scenarios
  pass at 4.5–7.0 fps (worker/CPU), with 0 off-origin requests and 0 CSP violations; both cross checks hold. A
  `--main-thread` run passes too.
- **Round 2** looked at the worker protocol, phone cameras, long sessions and the asset build. 8 findings held. I fixed
  7 in 212d79f; the eighth was in Cloud A's file:
  - **Safari 16.4–16.x:** my ad7082e canvas fix let the worker start without WebGL and then fail every frame. A WebGL
    probe now sends video back to the page's thread. This corrects the 21:45 browser-support claim, which was wrong
    between ad7082e and 212d79f.
  - **WebGL context loss:** MediaPipe silently repeated a frozen face or a phantom person; now every call fails, shown
    as "video step failing".
  - **Main-thread heap:** after stop, MediaPipe's WASM heap held the last frame. It is now zero-filled when the models
    close, which covers the face crops and the YAMNet window too; verified on real MediaPipe in Chromium.
  - **Portrait frames:** display-id distances now use the real frame aspect.
  - **Crop-only calibration:** a face-cue channel empty at the freeze stays missing until it has 20 s of its own.
  - **fps while paused:** it reads 0 while video is paused.
  - **Caching (Cloud A's file):** `/mediapipe/*` was cached for 24 h with no version in the URL. Cloud A made it
    `no-cache` in ac1a24e.
- **Reviews of my fixes caught mistakes, which f1de56c and 69d00c8 correct:**
  - A 3 s pause reset I added in 212d79f renumbered seated people in tracker-id order and wiped the baselines of people
    who stayed. It is reverted. A pause keeps seats and baselines again; only a 10 min pause, measured on wall time like
    the engine's break clock, restarts the baselines.
  - My blank-frame scrub could not reach the face crop. The heap wipe above replaced it.
  - Test gaps closed: the YAMNet heap wipe, a task that fails to close, and late calibration.
- **Lesson.** Each fix round got its own review, and two of the three reviews found a regression of mine. I'll keep
  doing that until the freeze.

### 01:45 — round 3 (docs, engine, mock), portrait frames, 40-minute soak

- **Health.** No fixture drift. 940 frontend tests pass and the build passes. `e2e/run-all.mjs` now has seven
  scenarios, all passing at 4.3–7.3 fps (worker/CPU), with 0 off-origin requests and 0 CSP violations; both cross
  checks hold.
- **Round 3** (docs against code, how the engine uses my observations, mock against real). 21 findings held, mostly
  wording:
  - `REAL_CAMERA.md` was rewritten to match the deployed console (d659c41):
    - the 20 s face and posture calibration now comes right after allowing the camera; the baselines freeze in the
      first 20 s on camera, not in sections 2–3;
    - denying the mic blocks both devices on the console, which does not retry each device after a refusal;
    - reload `/?source=browser`; the bare URL lands on the tour;
    - the purge gap reads 0 or 1, with one frame in flight;
    - `load_ms` is the time to the first observation;
    - onset counts are read from the State vector;
    - iPhone DevTools come through the Mac's Web Inspector.
  - The mock follows the status contract: no `load_ms` or rates before the first frame, both rates 0 after `stop()`.
    Tired eyes stay under the blink reopen level, so blinks count. People 2 and 3 calibrate on the neutral face.
  - **Cloud A fixed their items:**
    - "mic, no microphone" is shown once;
    - Pause → Start always starts people fresh: a new perception numbers people from 1 in detection order, so even
      a lone presenter can come back under another id;
    - the governance note and the DEPLOY_WEB worker wording are corrected.

    They kept 10+ min pauses as breaks on purpose, as on the desktop; my 10 min baseline restart stays in step.
- **Correction to 21:45 browser support.** iOS Chrome and Firefox use the system WebKit, so they behave like Safari on
  the same iOS: the worker on iOS 17+, and the page's thread on iOS 16.4–16.x (212d79f probe). Android Chrome:
  the worker is expected but untested; only headless desktop Chromium was measured.
- **Portrait frames end to end** (644ec53). New `portrait` scenario: 720×1280 with real MediaPipe, faces on every
  frame, one person, 6.6–6.9 fps. The harness's 1280×720 request cropped a portrait clip to a square, so
  `run.mjs --size WxH` sets the track size and checks every frame.
- **Soak, two-person scene, after tonight's changes:**
  - **15 min:** RSS stayed on plateaus of ~1.62–1.64 GB, with no trend.
  - **40 min:** GC sawtooth with peaks of 1,661, 1,669 and 1,667 MiB, not rising, and troughs 1,618 → 1,643.
    The harness keeps every audio observation (2,389 by the end), which may explain the troughs' rise.
  - **Counters:** 11,018 frames decoded, 11,017 purged (one in flight); 0 violations; 0 page errors; 8 main-thread
    long tasks, the longest 109 ms.

### 03:45 — final review, slow devices, fresh clone

- **Health.** No fixture drift. 944 frontend tests pass and the build passes. `e2e/run-all.mjs` passes all seven
  scenarios at 4.4–6.9 fps (worker/CPU), with 0 off-origin requests and 0 CSP violations; both cross checks hold.
- **Final pre-freeze review**, on how tonight's changes interact:
  - **Purge gap after Pause (d87fbd6).** With the worker, a frame is almost always in flight. `stop()` counted it
    only on the step's later rejection, so the console's per-source counters, read right after `stop()`, came out
    one frame short on every Pause, and the governance purge gap crept up for good. `stop()` now settles the frame
    in flight; nothing changes on Cloud A's side.
  - **Lost WebGL context.** Instead of failing every frame under "video step failing", video now stops, releases
    the models and reports "video stopped: WebGL context lost (restart the camera)".
  - **Its review found one race (52fc8bb).** If the context is lost while the mic is still loading, `start()` no
    longer reports the camera as live.
- **Slow devices, measured with DevTools CPU throttling at 4×** (it slows the page's thread, not workers):
  - **Worker path:** video holds 5.9 fps. Most of the page's work is `createImageBitmap(video)`: about 5.5 ms per
    1280×720 frame headless, against about 0.1 ms for `new VideoFrame(video)`, which MediaPipe also accepts. Not
    switched before the freeze: handing a VideoFrame to WebGL in a worker is untested on Safari and Firefox.
  - **Main-thread fallback (Safari before 17):** the page was busy 52 of 60 s at 1.6 fps. The loop now rests at least
    half a step after each one (e276b25): 37.5 of 60 s at 1.1 fps; unthrottled, 6.0 → 4.9 fps.
  - **New tooling:** `run.mjs --cpu-throttle N` and `--profile out.cpuprofile`.
- **Fresh clone.** `git clone`, `npm ci`, `npm run build`: the strict prebuild downloads the four models with
  checked hashes (72 MB in `dist/client/mediapipe`, largest file 11 MB). `scripts/check-static.mjs` with the
  production CSP, real perception and the fake camera then passes: "Top drivers by layer" appears, with 0 off-origin
  requests, 0 CSP violations and 0 page errors.

### 05:45 — check-in

- **Health.** No fixture drift. 944 frontend tests pass. `e2e/run-all.mjs` passes all seven scenarios with 0
  off-origin requests and 0 CSP violations; both cross checks hold.
- **fps.** 3.7–5.8 (worker/CPU), a little under the earlier runs (4.3–7.7). No perception code changed since 03:45
  (only other teams' docs), so this is load on the shared container, not a regression.
- **Review of e276b25** (main-thread rest): nothing confirmed. The one claim, that the face-hold stretch is lost on
  very slow devices, only applies to the Python face cadence, which nothing ships.
- **Next.** Last check-in at 07:45; freeze at 08:00. After that, only fixes, docs and status.
