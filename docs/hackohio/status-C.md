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
| CPU delegate when WebGL is software-rendered | In headless Chromium (SwiftShader) MediaPipe's GPU path is ~10x slower: pose 660 ms vs 68 ms per frame, face 180 vs 33 ms, gestures 306 vs 58 ms. `autoDelegate()` checks the WebGL renderer string. |
| Video models in a Web Worker | MediaPipe takes 10-100 ms per frame; in a worker the console's main thread stays free. Main-thread fallback where workers can't run it. |
| Pose: PoseLandmarker lite + a simple IoU tracker | Stands in for YOLO26n-pose + ByteTrack (track buffer 6 s, as `bytetrack_valence.yaml`). BlazePose landmarks 0,2,5,7,8,11-16,23-28 map to COCO 0-16; confidence = BlazePose visibility, 0 outside the frame. The person box is the landmarks' extent with the top raised above the eyes. |
| Pose detection and presence thresholds 0.4 | The Python edge's YOLO person threshold. MediaPipe's default 0.5 found one of the two people in the two-person scene (15/15 frames); 0.4, 0.3 and 0.2 all find both. |
| Hips, knees, ankles (COCO 11-16) never confident | BlazePose always predicts a whole body: on the waist-up portrait it put the hips at 72% of the frame height with visibility 1.00 (knees 0.5-0.6, ankles 0.3). YOLO at a desk gives hidden hips low confidence, and of the Python rules only motion reads them (COCO 5-12), so motion is measured on shoulders, elbows and wrists. |
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
  models 21.3 MB (face 3.8, pose 5.8, gestures 8.4, YAMNet 4.1); the e2e server sends `no-store`, so it re-serves
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
