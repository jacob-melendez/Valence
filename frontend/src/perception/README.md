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
              │        ─► PostureTracker (posture.ts): slump, head in hands, motion
              │        even frames: GestureRecognizer ─► Closed_Fist / curl fallback ─► assignHands ─► FistTracker
              │        odd frames:  FaceLandmarker ─► assignFaces ─► FaceValenceCalibrator + FaceCueTracker
              │                     (head crop retry for unmatched people)
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
| `models.ts`, `visionHost.ts`, `vision.worker.ts`, `capture.ts`, `media.ts` | MediaPipe loading, the worker, frame and mic capture, getUserMedia | — |
| `mock.ts` | deterministic observations without a camera | — |

## Privacy

- Each frame exists as one `ImageBitmap` for one step and is closed in the worker right after. The luminance read
  (1 Hz) and head crops go through canvases that are cleared, and their buffers are zero-filled. There is no
  MediaRecorder, and no canvas or image data is kept or persisted.
- Mic chunks are zero-filled once copied into the 1 s block. The block and its 16 kHz copy are zero-filled after
  analysis.
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
  ~1. Detection and presence thresholds are 0.4, YOLO's `conf`.
- **Faces.** FaceLandmarker in IMAGE mode as on the edge, up to 3 faces, plus the transformation matrix for head
  pose; on every frame by default (Python: every other frame, still available as `faceEveryFrame: false`), because
  the emotion cues' blink detector needs the face rate (`status().face_hz` reports it). Head yaw and pitch are not
  taken from the head-crop retry (its matrix is in the crop's own camera); roll is.
- **Face hold.** 0.45 s, i.e. 2.25 face passes at 10 Hz, stretched to 2.25 measured passes on slower devices.
- **Workers and delegates.** The video models run in a Web Worker. The GPU delegate is used unless WebGL is
  software-rendered.
- **Not included.** No compliment keywords, no lux or SPL calibration (the uncalibrated defaults, as on a fresh
  desktop install).

## Tests

- `src/test/perception-*.test.ts` (vitest): unit tests plus **parity** against the Python functions on shared inputs.
  The fixtures are written by `scripts/export_perception_fixtures.py`; the generators are in
  `src/test/perception-fixtures/gen_*.py`.
- `e2e/run-all.mjs`: headless Chromium with a fake camera and mic, behind the strict CSP. It checks that faces are
  detected, observations flow, nothing leaves the origin and nothing violates the CSP. See `e2e/README.md`.
- `e2e/REAL_CAMERA.md`: the checklist for a real camera after deploy.
