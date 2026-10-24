# Cloud C status: browser perception (`[web]`)

Owner: Cloud C. Scope: `frontend/src/perception/**`, `frontend/src/test/perception*`, `frontend/public/mediapipe/**`,
`frontend/e2e/**`, `scripts/export_perception_fixtures.py`, the MediaPipe dependencies. Times are America/New_York.
Newest entry last.

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
