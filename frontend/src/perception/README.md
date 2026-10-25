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
perception.status();                                   // fps, load time, where the models run, degradations
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
  hands video to the page's thread, and a lost WebGL context fails every later step ("video step failing") instead of
  repeating MediaPipe's last results.
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
- **Not included.** No compliment keywords, no lux or SPL calibration (the uncalibrated defaults, as on a fresh
  desktop install).

## Tests

- `src/test/perception-*.test.ts` (vitest; `npx vitest run src/test/perception-*.test.ts` from `frontend/` after
  `npm install`): unit tests plus **parity** against the Python functions on shared inputs. The fixtures are
  committed; `scripts/export_perception_fixtures.py` rewrites them (repo venv with numpy, scipy and OpenCV) from the
  generators in `src/test/perception-fixtures/gen_*.py`.
- `e2e/run-all.mjs`: headless Chromium with a fake camera and mic, behind the strict CSP. It checks that faces are
  detected, observations flow, nothing leaves the origin and nothing violates the CSP. See `e2e/README.md` (it needs
  Playwright, which is not a frontend dependency).
- `e2e/REAL_CAMERA.md`: the checklist for a real camera after deploy.
