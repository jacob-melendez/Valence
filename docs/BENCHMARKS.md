# Edge benchmarks (Apple Silicon)

Machine: MacBook Pro Mac15,11, **Apple M3 Max** (14 cores: 10P + 4E), 36 GB, macOS 14.6, plugged in.
Python 3.12.14 arm64 (uv-managed), torch 2.14.1 (MPS), ultralytics 8.4.171, mediapipe 1.1.1,
opencv-contrib 4.14.0.94, mlx 0.32.4 / mlx-lm 0.31.3, vosk 0.3.44.

Reproduce: `python scripts/spikes/bench_video.py [--camera N]`, `python scripts/spikes/bench_audio.py --say`,
`python scripts/spikes/bench_llm.py`. Every frame/audio block in the spikes goes through the purge gate; only
timings and counts are printed.

## Video (spikes 1-3)

1280x720 input (ultralytics sample images: 2-3 people), 60 timed iterations after warm-up.

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| YOLO26n-pose + ByteTrack, MPS, imgsz 640 | 18.4 ms | 21.0 ms | `classes=[0], conf=0.4`, track_buffer 60 |
| YOLO26n-pose + ByteTrack, MPS, imgsz 480 | 17.8 ms | 20.2 ms | barely faster: MPS is launch-bound, not FLOP-bound |
| BGR->RGB + `mp.Image` | 0.6 ms | 0.9 ms | |
| GestureRecognizer (VIDEO, 8 hands) | 16.4 ms | 16.8 ms | full frame |
| FaceLandmarker (VIDEO, 4 faces, blendshapes) | 5.5 ms | 9.3 ms | full frame: **found 0 faces** in a 2-person shot |
| FaceLandmarker (IMAGE) on a 256 px head crop | ~11 ms / person | | found the face the full frame missed |
| PoseLandmarker lite (4 poses), fallback | 23.4 ms | 24.1 ms | slower than YOLO on MPS: fallback not needed |
| **`VideoPerceiver.process` end to end** (2 people) | **45 ms** | ~51 ms | YOLO + alternating gesture/face crops + lux |

**Live webcam** (built-in FaceTime HD, index 0, 1280x720 requested; one person at the desk plus background people):

| Stage | p50 | p95 |
|---|---|---|
| YOLO26n-pose + ByteTrack, MPS, imgsz 640 | 20.7 ms | 23.1 ms |
| YOLO26n-pose + ByteTrack, MPS, imgsz 480 | 17.4 ms | 20.7 ms |
| GestureRecognizer (VIDEO, 8 hands), full frame | 30.5 ms | 43.5 ms |
| FaceLandmarker (4 faces, blendshapes), full frame | 10.4 ms | 11.4 ms |
| **Edge runner, video + audio together** (`python -m valence.edge.run`) | **45-57 ms/frame, 10.0 Hz** | 0 sink errors |

**Decisions**
- Loop budget is 100 ms at 10 Hz; measured ~45 ms, so **keep imgsz 640** and YOLO (no PoseLandmarker fallback).
- Faces: one **full-frame** FaceLandmarker pass (IMAGE mode, 4 faces) matched to people by the YOLO nose keypoint,
  then a **head-crop** retry (box from YOLO nose/eye/ear keypoints, scale 2.2, 256 px) for anyone left unmatched.
  The short-range face detector misses faces that fill little of a 720p frame (0 faces in the 2-person sample);
  crop-only gave the desk user a visible face only 67% of frames live, hence the hybrid.
- People: only boxes >= 25% of the frame height, at most the 3 largest (`settings.min_box_height`, `max_tracks`):
  a hackathon room puts 2-3 extra people at the frame edges (box height 0.3-0.57).
- Capture: `grab()` every camera frame, `retrieve()` (decode) only at 10 Hz; ~2/3 of the 30 fps stream is never decoded.
- Ultralytics keeps the last input image on its predictor (`batch`, `results`, `dataset.im0`); the perceiver clears
  them after every call. A weakref check confirms no frame survives `process()`.

**Camera permission gotcha.** Claude Code's shells run under the helper bundle `com.anthropic.claude-code`, whose
Info.plist has a microphone usage string but no camera usage string, so macOS can never grant it the camera. Run every
camera step from Terminal.app (or iTerm / VS Code), which gets the normal permission prompt.

## Audio (spikes 4-5)

Live "MacBook Pro Microphone", 16 kHz mono float32, blocksize 1600 (16 kHz accepted natively; no resampling needed).

| Stage | Cost |
|---|---|
| YAMNet AudioClassifier, AUDIO_STREAM, 6-class allowlist, 1 s chunks (spike) | callback ~6 ms after submit, 1 result/s |
| YAMNet AUDIO_CLIPS, synchronous on the block's last 0.975 s (shipped) | one result per block, attributed exactly |
| Vosk small-en-us 0.15, restricted grammar (8 phrases + `[unk]`), per 100 ms block | p50 0.84 ms, max 29 ms |
| dBFS from mean power | negligible |

Notes: the default input device on this Mac is a Bluetooth headset, so capture is pinned by name to
"MacBook Pro Microphone". `say` through the speakers is only ~3 dB above the room floor at the mic (-58 vs -61 dBFS),
yet Vosk still caught "amazing"; real-voice validation happens in the Step 3 exit test. AudioClassifier streaming works,
so the ai-edge-litert fallback is not needed; Vosk works, so the whisper fallback is not needed.

## Narrator LLM (spike 6)

`apply_chat_template(..., enable_thinking=False)` (template ends in an empty `<think>\n\n</think>` block), sampling
temp 0.7 / top_p 0.8 / top_k 20 / repetition penalty 1.05, max 110 tokens.

| Model (MLX 4-bit) | Load | Warm-up | Prompt | Generation | Wall per answer | Peak mem |
|---|---|---|---|---|---|---|
| **Qwen3-1.7B** | 1.2 s | 4.0 s | 1752 tok/s | **187 tok/s** | 536 ms (55 tok) | 1.27 GB |
| Llama-3.2-3B-Instruct (fallback) | | 3.2 s | 1099 tok/s | 107 tok/s | 1108 ms (89 tok) | 2.25 GB |

Qwen3-1.7B fits the 2.5 s narrator deadline with room for two retries even with a ~1k-token few-shot prompt
(~0.6 s prefill). Warm-up at service start hides the 4 s Metal kernel compile.

## torch + LightGBM in one process (libomp)

Live mode runs YOLO (torch) and the LightGBM predictor in the API process. LightGBM 4.7.0 needs Homebrew `libomp`
(`brew install libomp`; it won't import otherwise) and torch 2.14.1 bundles its own, so two OpenMP runtimes meet.
Test: YOLO26n-pose load + 3 MPS inferences + LightGBM `predict(pred_contrib=True)` in one process, 5 runs each.

| setup | result |
|---|---|
| `import torch` first, LightGBM `num_threads=1` or 0 | **OK** 5/5 |
| torch first + `KMP_DUPLICATE_LIB_OK=TRUE` only | OK |
| torch first + `OMP_NUM_THREADS=2` | segfault |
| torch first, LightGBM default threads (None / -1 / 4) | segfault |
| `import lightgbm` before torch (any of the above) | segfault, in torch's conv/bn fusion during YOLO load |

`valence.edge.run` imports torch at module load and warns if lightgbm got there first; the API must import it before
lightgbm and predict single-threaded (requested in `docs/CONTRACT_REQUESTS.md`).

## Live feature capture (Step 3.5, `scripts/spikes/live_capture.py`)

One person at the desk, built-in camera, features only (`data/real/capture_*.jsonl`, summary in
`data/real/capture_summary_run1.json`). One edge runner spans the session, so baselines come from the first 25 s.
10.0 Hz video throughout; every segment 180 s unless noted.

| Segment | Result |
|---|---|
| typing (quiet room) | **fist onsets 0.0/min**, slump 0%, head-in-hands 0%, face visible 100%, motion p10/p50/p90 0.014/0.027/0.048 sw/s, 52 dB median |
| fists (15 cued two-handed clenches, every 12 s) | **recall 15/15, 0 false onsets**; 28 per-hand onsets before onsets became person-level (now 1 per clench) |
| keyboard gaze (typing while looking down) | old nose-only slump rule: **slumped 93% of the time** (false). Fixed: slump now needs nose < 0.70x **and** ears < 0.93x baseline |
| covered camera (15 s) | no person tracked, so no face/fist/slump output |
| slump, audio | re-recorded in run 2 below (run 1 had no upright phases; audio exposed the Vosk zeroing bug, fixed in e38f955) |

**Posture probe** (`scripts/spikes/posture_probe.py`, cued upright / keyboard gaze / slump x 3 cycles, ratios of
head-point height above the shoulder line to the upright baseline, medians):

| | upright | gaze | slump |
|---|---|---|---|
| nose | 1.00 | 0.84 | 0.42 |
| eyes | 1.00 | 0.89 | 0.58 |
| ears | 1.00 | 0.96 | 0.86 |
| shoulder line drop (frame heights) | 0.00 | +0.01 | +0.12 |

Head pitch (looking at the keys) swings the nose but barely moves the ears; a slump lowers the whole head. Replaying the
probe through `PostureTracker` (3 s hold): new rule slump 0.84 / gaze 0.00 / upright 0.00; nose-only 0.95 / 0.00 / 0.00
on this probe, but nose-only failed on the deeper real typing gaze above. Shoulder drop alone is baseline-sensitive
(fails when the first upright phase is taken as baseline), so it is not used.

## Offline check

`valence.edge.run` sets `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` on import.
With those, the full perception stack under load (`scripts/spikes/gpu_load.py`: YOLO + MediaPipe + lux at 10 Hz)
held **no inet sockets** (`lsof -i -p <pid>` empty after 15 s). Wi-Fi-off demo rehearsal still to do.

### Run 2 (2026-10-18, after the slump, fist and Vosk fixes; `data/real/capture_summary.json`)

Participants only (the recorder drops bystanders). 10.0 Hz throughout.

| Segment | Result |
|---|---|
| keyboard gaze (typing while looking down, 3 min) | **slump 0%** (run 1, nose-only rule: 93%), fist onsets 0, face visible 100%, motion p50 0.021 |
| slump (6 cued 20 s slumps, 10 s upright gaps) | **6/6 detected**, 4.3-11.3 s after the cue (3 s hold + EMA), slump in 71% of cued frames, **2.7% while upright** |
| audio ("great job" x3, then laughing) | **compliment hits 4** (>= 3 expected; phrases are not logged by design), laughter onsets 0 under the old rule (see probe) |

**Laugh probe** (`scripts/spikes/laugh_probe.py`, cued quiet / laugh / talk, YAMNet AUDIO_CLIPS per second):
quiet and talking: laughter family **0.00** every second (Speech up to 0.99); real laughs: Speech 0.1-0.98 with the
laughter family at 0.04-0.08, and one second at Laughter 0.50 / Snicker 0.26 / Chuckle 0.26. The planned rule
(>= 0.30 for 2 results) can never fire on this mic; the shipped rule is >= 0.05 for one result, off < 0.02, 4 s
refractory, which counts 1 onset per real laugh phase and 0 in quiet/talk on the probe.

**Run 3 (audio only, new laughter rule):** 3x "great job" -> **3 compliment hits**; one ~5 s laugh -> **2 laughter
onsets** (4 s apart: a long laugh can count twice under the 4 s refractory); 0 events during the quiet baseline.
