# Browser perception (Valence Web)

Camera and microphone in, the Python edge's observations out, entirely inside the page. This module is the browser
counterpart of `valence/edge` + `valence/perception`: it emits `VideoObs` (~10 Hz) and `AudioObs` (1 s) with the
same fields, units and meaning as `valence/contracts.py`, so the in-browser engine (`src/engine`) consumes exactly
what the desktop engine consumes.

```ts
import { createPerception, openMediaStream } from "@/perception";
const { stream } = await openMediaStream();            // AEC / noise suppression / AGC off, camera-only or mic-only fallbacks
const perception = createPerception();                 // createMockPerception() works without a camera
perception.onObservation((obs) => engine.push(obs));
await perception.start(stream);
perception.counters();                                 // governance: frames decoded / purged, audio blocks purged
perception.status();                                   // fps, time to first observation, where the models run, degradations
perception.stop();
```

## Pipeline

```
getUserMedia ─┬─ video track ─ <video> ─ ImageBitmap per step (10 Hz, newest frame, no debt) ──► Web Worker
              │                                                                                     │
              │      video.ts (port of VideoPerceiver.process):                                     │
              │        PoseLandmarker lite ─► COCO-17 (coco.ts) ─► IoU tracker ─► DisplayIdMapper (ids.ts)
              │        ─► One Euro (smoothing.ts) ─► PostureTracker (posture.ts): slump, head in hands, motion
              │        even frames: GestureRecognizer ─► Closed_Fist / curl fallback ─► assignHands ─► FistTracker
              │        every frame: FaceLandmarker ─► assignFaces ─► FaceValenceCalibrator + FaceCueTracker
              │                     (head crop retry for unmatched people; odd frames only with faceEveryFrame: false)
              │        1 Hz: strided luminance (luma.ts, OpenCV BGR2GRAY) ─► backgroundStats ─► LuxEstimator
              │      bitmap closed, pixel copies wiped ─► VideoObs ─────────────────────────────────┘
              │
              └─ audio track ─ AudioWorklet (public/mediapipe/pcm-tap.worklet.js): ~100 ms raw chunks
                   ─► 1 s block (zero-filled chunks) ─► resample_poly to 16 kHz (resample.ts)
                   ─► audio.ts: Leq dB SPL, YAMNet speech/laughter on the last 0.975 s, laughter debouncer
                   ─► block and copy zero-filled ─► AudioObs
```

| File | Role | Python reference |
|---|---|---|
| `types.ts` | `PersonObs`, `VideoObs`, `AudioObs`, counters, status; `obsToDict` / `obsFromDict` (JSONL form) | `valence/contracts.py` |
| `perception.ts` | the loops, buffers, counters and robustness | `valence/edge/run.py`, `capture.py` |
| `video.ts` | one frame -> one `VideoObs` | `valence/perception/video.py` |
| `audio.ts` | one 1 s block -> one `AudioObs` (no compliment recognizer: `compliment_hits` stays 0) | `valence/perception/audio.py` |
| `posture.ts`, `hands.ts`, `ids.ts`, `debounce.ts`, `faceValence.ts`, `lux.ts`, `regions.ts`, `audioFeatures.ts` | exact ports | same names under `valence/perception/` |
| `faceCues.ts` | emotion-v1 face cues (brow, eye openness, frown, smile, head yaw/pitch, blinks) | `valence/features/face_cues.py`, `docs/EMOTION_FEATURES.md` |
| `resample.ts`, `assign.ts`, `luma.ts` | scipy `resample_poly` and `linear_sum_assignment`, OpenCV `BGR2GRAY` | scipy, OpenCV |
| `coco.ts`, `tracker.ts` | BlazePose -> COCO-17 and a simple tracker, standing in for YOLO-pose + ByteTrack | — |
| `smoothing.ts` | One Euro keypoint filter for posture (BlazePose jitters where YOLO does not) | — |
| `models.ts`, `visionHost.ts`, `vision.worker.ts`, `capture.ts`, `media.ts` | MediaPipe loading, the worker, frame and mic capture, getUserMedia | — |
| `mock.ts` | deterministic observations without a camera | — |

## Privacy

- Each frame exists as one `ImageBitmap` for one step and is closed in the worker right after. The luminance read
  (1 Hz) and head crops go through canvases that are cleared, and their buffers are zero-filled. MediaPipe copies each
  frame and its face crops into its WASM heap; the worker's heap dies with the worker, and on the page's thread (the
  fallback) each task's heap is zero-filled when it closes. There is no
  MediaRecorder, and no canvas or image data is kept or persisted.
- Mic chunks are zero-filled once copied into the 1 s block. The block and its 16 kHz copy are zero-filled after
  analysis. MediaPipe keeps its own copy of the last YAMNet window in its WASM heap; closing the classifier (stop(),
  or a mic that ends) overwrites it with a silent window and then zero-fills the heap.
- `counters()` reports frames decoded and purged, raw audio buffers purged, derived buffers wiped, and buffers alive
  at the end of a step (`violations`). This mirrors the Python purge gate's numbers.
- No request leaves the origin. WASM and models come from `/mediapipe/` (`scripts/mediapipe-assets.mjs`: WASM
  copied from `node_modules`, models downloaded at build time with pinned SHA-256s). MediaPipe is pinned to 0.10.35
  because every 1.x release bundles a usage logger that posts to Google. Everything works under
  `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self' blob:`.
- Observations hold scalars, booleans and normalized boxes, plus per person the 52 blendshape scores and head pose
  (`raw`). They never hold landmarks or pixels.

## Where it differs from the Python edge (on purpose)

- **Pose.** PoseLandmarker lite + an IoU tracker replace YOLO26n-pose + ByteTrack. BlazePose `visibility` is used as
  keypoint confidence; it is 0 outside the frame and for hips and legs, which BlazePose extrapolates with visibility
  ~1. Detection and presence thresholds are 0.4, YOLO's `conf`. It tracks up to 6 poses (YOLO: every detection), then
  the same 25% height filter and largest 3 apply.
- **Person box.** The extent of all 33 landmarks, so it reaches legs BlazePose predicts behind a desk. Measured
  against YOLO on 486 synthetic frames: the main participant's box is 0.05 shorter (mostly at the top), and desk-occluded
  bystanders are 0.06 taller, though BlazePose finds only a third of them and none next to a close participant.
  Gating hidden landmarks out (visibility < 0.5) shrank the main participant's box by up to 0.4, so the box stays.
- **Posture.** BlazePose keypoints jitter where YOLO's do not, so posture runs on One Euro-smoothed keypoints and
  motion is measured on the shoulders only (`smoothing.ts`; Python: raw keypoints, COCO 5-12).
- **Faces.** FaceLandmarker in IMAGE mode as on the edge, up to 3 faces, plus the transformation matrix for head
  pose; on every frame by default (Python: every other frame, still available as `faceEveryFrame: false`), because
  the emotion cues' blink detector needs the face rate (`status().face_hz` reports it). Head yaw and pitch are not
  taken from the head-crop retry (its matrix is in the crop's own camera); roll is.
- **Face hold.** 0.45 s as on the edge (2.25 face passes at its every-other-frame cadence at 10 Hz); when face passes
  come slower than every 0.2 s (a slow device), it stretches to 2.25 measured pass intervals.
- **Workers and delegates.** The video models run in a Web Worker, on MediaPipe's CPU delegate by default. On
  identical frames that matches the native edge to ~1e-4. `?delegate=GPU` on the page URL tries the GPU instead.
  MediaPipe sends images through WebGL on either delegate: a worker without WebGL on OffscreenCanvas (Safari before 17)
  hands video to the page's thread, and a lost WebGL context stops video and releases the models ("video stopped: WebGL
  context lost (restart the camera)") instead of repeating MediaPipe's last results.
- **Pauses.** While nothing is processed (tab hidden, camera muted) nobody is seen leaving, so people reattach to their
  seats afterwards with their ids and baselines. A pause as long as the engine's break (`break_absent_min`, 10 min,
  on wall time like the engine's clock, so a computer's sleep counts) restarts the per-person baselines, since someone
  else may be sitting there; the display ids stay.
- **Frame shape.** Display-id reuse distances use the real frame aspect (Python: 16:9, the edge's capture size), so
  portrait phone frames measure vertical distance correctly. A face-cue channel with no samples at the 20 s freeze
  stays missing (Python: 0) until it has 20 s of its own, then freezes: only the browser's head-crop retry, which gives
  no yaw or pitch, can leave one empty.
- **Muted and stalled cameras.** A muted track (Safari giving the camera to another tab, an OS privacy toggle,
  Chromium after ~25 missed frames) pauses video like a hidden tab, and a muted mic is purged instead of analysed,
  both reported in `status().degraded`. A camera that stalls without being muted repeats its last frame until the
  browser mutes it (Python: `take()` never returns a frame twice).
- **Main-thread cost.** The page's thread only grabs frames, hands them to the worker and runs YAMNet once a second.
  The biggest part is `createImageBitmap(video)`: ~5.5 ms per 1280x720 frame in headless Chromium (software GPU),
  against ~0.1 ms for `new VideoFrame(video)`, which MediaPipe 0.10.35 also accepts. Untested on Safari and Firefox,
  so not switched yet (`e2e/run.mjs --cpu-throttle 4 --profile` shows it). Without a worker (Safari before 17), each
  step blocks the page for its whole duration, so the loop rests at least half a step after each one: at 4x CPU
  throttling the page was busy 52 of 60 s at 1.6 fps, and 37.5 of 60 s at 1.1 fps with the rest.
- **Not included.** No compliment keywords, no lux or SPL calibration (the uncalibrated defaults, as on a fresh
  desktop install).

## Decisions

| Decision | Why |
|---|---|
| **MediaPipe Tasks Web 0.10.35** (`@mediapipe/tasks-vision`, `@mediapipe/tasks-audio`), pinned exactly | Published 2026-04-27, well past the 24 h guard. Every 1.x release (1.0.0 to 1.1.1) bundles a usage logger that POSTs to `odml.pa.googleapis.com/v1/log` every 60 s, created unconditionally in `createFromOptions`. Under our CSP it would be blocked, but it would still be a third-party attempt and a CSP violation. 0.10.35 has no such code: its bundle contains no external URL. The Python edge runs mediapipe 1.1.1 with the same `.task` models. |
| Observations keep NaN for "no reading" | Same as `contracts.py`; `obsToDict` writes null like `obs_to_dict`, and `obsFromDict` reads the Python JSONL recordings unchanged (tested on `data/real/capture_fists.jsonl`). |
| A `type` tag on each observation | The same key `obs_to_dict` writes, so TypeScript gets a discriminated union and recordings parse as-is. |
| `audio_blocks_purged` counts ~100 ms raw mic buffers | Same meaning as the Python edge, where the gate counts each capture part. 1 s blocks and resampled copies count under `derived_purged`. |
| Model and WASM files self-hosted, never committed | `frontend/scripts/mediapipe-assets.mjs` (prebuild and predev) copies the WASM from `node_modules` and downloads the four models from Google's model bucket with pinned SHA-256s (the files `scripts/download_models.sh` gives the Python edge). The Cloudflare build (`npm install && npm run build`) has internet access; the browser never contacts Google. |
| CPU delegate by default | On identical frames the CPU path matches the native edge to ~1e-4. The GPU path is unmeasured on real hardware and drifted 100–300x more on software WebGL. `?delegate=GPU` tries it, falling back to CPU only if it fails to start; on software WebGL it runs, ~10x slower (pose 660 vs 68 ms per frame). |
| Faces on every frame by default | The emotion cues' blink detector needs the face rate: 7.2 fps with 7.2 face passes/s, against 8.4 fps with 4.2 at the Python cadence (`faceEveryFrame: false`); 22 and 24 blinks instead of 13 and 15 on the two real-person clips (`data/browser`). `status().face_hz` reports the rate. |
| Faces in IMAGE mode, as on the edge | VIDEO mode was bit-identical wherever the detector fired and only added a stale ghost face about once per 2000 frames. The head-crop retry gives no yaw/pitch (its matrix is in the crop's own camera), only roll. |
| Video models in a Web Worker | MediaPipe takes 10-100 ms per frame; in a worker the console's main thread stays free. Main-thread fallback where workers can't run it. |
| Pose: PoseLandmarker lite + a simple IoU tracker | Stands in for YOLO26n-pose + ByteTrack (track buffer 6 s, as `bytetrack_valence.yaml`). BlazePose landmarks 0,2,5,7,8,11-16,23-28 map to COCO 0-16; confidence = BlazePose visibility, 0 outside the frame. The person box is the landmarks' extent with the top raised above the eyes. |
| Up to 6 poses tracked, then the 25% filter and largest 3 | Capped at 3, MediaPipe's VIDEO mode stops detecting once 3 poses are tracked, so small bystanders seen first could hide a participant who sits down close. YOLO hands the filter every detection. |
| Person box keeps BlazePose's predicted legs | Measured against YOLO on 486 synthetic frames (desk-occluded bystanders, seated participants): bystanders' boxes run 0.06 taller, but BlazePose finds only a third of them and none next to a close participant. Gating out landmarks with visibility < 0.5 shrank the main participant's box at 720 px from 0.85 to 0.57 of the frame (YOLO 0.97). |
| Muted tracks pause, ended tracks release | A muted camera (Safari giving it to another tab, Chromium after ~25 missed frames) would repeat its last frame or black, and a muted mic renders silence: video pauses and audio is purged, both reported in `degraded`. A track that ends, with or without its event, releases its models at once. |
| Pose detection and presence thresholds 0.4 | The Python edge's YOLO person threshold. MediaPipe's default 0.5 found one of the two people in the two-person scene (15/15 frames); 0.4, 0.3 and 0.2 all find both. |
| Hips, knees, ankles (COCO 11-16) never confident | BlazePose always predicts a whole body: on the waist-up portrait it put the hips at 72% of the frame height with visibility 1.00 (knees 0.5-0.6, ankles 0.3). YOLO at a desk gives hidden hips low confidence, and of the Python rules only motion reads them (COCO 5-12), which leaves shoulders, elbows and wrists. |
| Posture on One Euro-smoothed keypoints, motion on the shoulders only | BlazePose keypoints jitter (white noise) where YOLO's do not: raw, motion read 0.46 sw/s on a still scene against YOLO's 0.026. Smoothed shoulders read 1.3x YOLO on still and drifting scenes and keep 0.93x of real motion (`smoothing.ts`). |
| Confidence thresholds 0.3 / 0.5 / 0.6 kept | BlazePose visibility on the e2e scenes: nose, eyes, ears, shoulders 1.00; elbows 0.6-0.8; partly hidden wrists 0.45-0.8 (YOLO usually reports 0.8-0.95 for visible wrists, so a wrist near 0.5 may miss the hand assignment: on the real-camera checklist). |


## Measured against the Python edge

**Pose: YOLO vs BlazePose on identical frames.** YOLO26n-pose and BlazePose lite each ran through the real Python
`PostureTracker` (YOLO with ultralytics 8.4.171 as `video.py` calls it, BlazePose through MediaPipe 0.10.35 in Python
with the `coco.ts` mapping) on synthetic 1280x720 10 fps sequences (still, drift, slump 35% and 50%, known-speed
motion, fidget; sensor noise σ 3) and two CC BY 4.0 clips of real people framed head and shoulders (intel-iot-devkit
samples). Python reproduces the browser's numbers, so the gap is the model, not the browser runtime.

| motion p50 (sw/s) | still | drift | 0.3 sw/s | 0.1 sw/s | real F | real M |
|---|---|---|---|---|---|---|
| YOLO | 0.026 | 0.072 | 0.366 | 0.173 | 0.036 | 0.022 |
| BlazePose, raw | 0.464 | 0.613 | 1.090 | 0.872 | 0.120 | 0.059 |
| BlazePose + One Euro + shoulders only (**shipped**) | 0.054 | 0.148 | 0.557 | 0.264 | 0.033 | 0.013 |

- **Cause.** BlazePose jitter is white noise (lag-1 autocorrelation ≈ 0; VIDEO mode returns the same landmarks as
  IMAGE mode). Head and shoulders move 0.11–0.22 sw/s on a still frame, elbows and wrists 0.5–1.0; YOLO's 0.004–0.02.
- **Fix.** Of ~1000 variants, the best was a per-keypoint One Euro filter (0.2 Hz, β 8 per sw/s, d 0.5 Hz) with
  motion on the shoulders only: still and drift read 1.3× YOLO and real motion 0.93×. The TS filter matches the
  evaluated Python one within 1e-6 on 2346 recorded frames.
- **Open: slump.** On a spacesuit portrait BlazePose places the shoulders on the neck ring, and they slide down with a
  lowered head, so both synthetic slumps were missed (YOLO caught them).
- **Open: keyboard gaze.** On a real head pitch-down, BlazePose's ears drop to 0.74× baseline against YOLO's 0.92×,
  so the 0.93 ear rule that keeps keyboard gaze from counting as a slump protects less (one 3.8 s false slump in one
  clip). Thresholds stay Python's until a real-camera check says otherwise (`frontend/e2e/REAL_CAMERA.md`).

**Audio on identical sound.** A 90 s file (room noise, LibriSpeech speech, eight ESC-50 laughs) played through
Chromium's fake microphone into `createPerception()` and through the edge's `AudioPerceiver` (Python MediaPipe
0.10.35, compliments off). Browser and Python YAMNet scored the same 15600-sample windows bit for bit (199/199
windows, max |diff| 0.0). Given the same 1 s span: speech r 0.999, laughter-family r 0.998, db_spl +0.001 dB on
average, and laughter onsets in exactly the same blocks. What differs is only where the 1 s blocks fall.

**Faces on identical frames.** Browser (WASM) and edge (native MediaPipe 0.10.35) on the e2e scenes and the two CC
BY clips, colour converted bit-identically to Chromium: blendshapes agree to p95 1.3e-4, valence_raw to 1.1e-4, head
yaw and pitch to 0.004°, and the face count matches on every frame. face_valence and all cues agree at r ≥ 0.987
(mostly ≥ 0.998); blinks 22 vs 22 and 24 vs 23. On software WebGL the GPU delegate drifts 100–300× more than CPU,
which is why CPU is the default.

**Soak.** 15 min (one person): 6.7 fps sustained, 5,646 frames decoded and 5,645 purged (one in flight),
0 violations, Chromium RSS in a flat 1.60–1.65 GB sawtooth. 40 min (two people): peaks 1,661 / 1,669 / 1,667 MiB, not
rising; 11,018 frames decoded and 11,017 purged; 0 violations; 0 page errors.

**Slow devices** (DevTools CPU throttling 4×, which slows the page's thread, not workers). Worker path: video holds
5.9 fps; most of the page's work is `createImageBitmap(video)` (~5.5 ms per 1280x720 frame headless, against ~0.1 ms
for `new VideoFrame(video)`, untested on Safari and Firefox). Main-thread fallback: the loop rests at least half a
step after each one so the page stays usable (busy 37.5 of 60 s at 1.1 fps, from 52 of 60 s).

**Browser support for camera mode** (measured on headless Chromium; the rest from the code and published support):
- Chrome/Edge desktop and Android: worker, CPU delegate (Chromium tested; Android expected, untested).
- Safari 17+ on macOS and iOS, and iOS Chrome/Firefox (system WebKit): worker (WebGL in an OffscreenCanvas).
- Safari 16.4–16.x and Firefox < 114: a WebGL probe sends video to the page's thread (`video_backend` reads
  `main/CPU`); it works, but the UI is slower.

## Tests

- `src/test/perception-*.test.ts` (vitest; `npx vitest run src/test/perception-*.test.ts` from `frontend/` after
  `npm install`): unit tests plus **parity** against the Python functions on shared inputs. The fixtures are
  committed; `scripts/export_perception_fixtures.py` rewrites them (repo venv with numpy, scipy and OpenCV) from the
  generators in `src/test/perception-fixtures/gen_*.py`.
- `e2e/run-all.mjs`: headless Chromium with a fake camera and mic, behind the strict CSP. It checks that faces are
  detected, observations flow, nothing leaves the origin and nothing violates the CSP. See `e2e/README.md` (it needs
  Playwright, which is not a frontend dependency).
- `e2e/REAL_CAMERA.md`: the checklist for a real camera after deploy.
