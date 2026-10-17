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

**Decisions**
- Loop budget is 100 ms at 10 Hz; measured ~45 ms, so **keep imgsz 640** and YOLO (no PoseLandmarker fallback).
- Faces are read from **per-person head crops** (box from YOLO nose/eye/ear keypoints, scale 2.2, resized to 256 px)
  with FaceLandmarker in IMAGE mode. The short-range face detector misses faces that fill little of a 720p frame;
  a crop also belongs to exactly one person, so face-to-person matching is exact.
- Ultralytics keeps the last input image on its predictor (`batch`, `results`, `dataset.im0`); the perceiver clears
  them after every call. A weakref check confirms no frame survives `process()`.

Live-camera numbers: _pending camera permission_ (rerun `bench_video.py --camera N` and the 10-min edge run).

## Audio (spikes 4-5)

Live "MacBook Pro Microphone", 16 kHz mono float32, blocksize 1600 (16 kHz accepted natively; no resampling needed).

| Stage | Cost |
|---|---|
| YAMNet AudioClassifier, AUDIO_STREAM, 6-class allowlist, 1 s chunks | callback ~6 ms after submit, 1 result/s |
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
