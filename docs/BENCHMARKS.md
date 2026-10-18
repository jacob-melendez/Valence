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

**Laughter double count (P1-4), offline replay.** The probe's per-second laughter-family max went through
`HysteresisDebouncer(on=0.05, off=0.02)` with different exit holds and refractories, plus synthetic 1 s sequences.
"hold" = how long the score must stay below 0.02 before the state exits. Onsets per segment at an exact 1.000 s cadence:

| rule | quiet 0-5 | laugh 6-11 | talk 12-19 | laugh 20-25 | quiet 26-31 | laugh 32-37 | quiet 38-41 | 5 s laugh, 2 low s | 5 s laugh, 3 low s | 2 laughs 10 s apart | 2 laughs 6 s apart |
|---|---|---|---|---|---|---|---|---|---|---|---|
| old: hold 0, refr 4 s | 0 | 0 | 0 | 1 | 0 | 1 | 0 | **2** | **2** | **4** | 2 |
| A: hold 2 s, refr 4 s | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | **2** | 2 | 2 |
| B: hold 2 s, refr 6 s | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 2 | 2 |
| C: hold 3 s, refr 6 s | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 2 | **1** |
| **B': hold 1.5 s, refr 5.5 s (shipped)** | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 2 | 2 |

Synthetic laughs: `0.08 0 0 0.04 0.08` (2 low s) and `0.08 0 0 0 0.08` (3 low s), shaped like the run-3 laugh with onsets
4 s apart; the 10 s case is `0.08 0.04 0.08 0 0.08` twice, onsets 10 s apart; the 6 s case is `0.08 0 0.08` twice,
onsets 6 s apart. Every rule scores the same on the probe: the in-laugh dips there are a single 0.00 second or are
followed by 0.04, which holds the state. So the choice comes from the synthetic cases. The old rule counts each
dipping 5 s laugh twice. A fixes only the 2-low-second dip. C merges laughs 6 s apart.

**Timestamp jitter.** Live `t` is `time.monotonic()` when a block finishes, so the cadence is 1 s plus or minus a few ms.
B's 2.0 s hold and 6.0 s refractory sit exactly on the sample grid. At a 0.999 s or 0.98 s cadence, B exits on the 4th
low result instead of the 3rd and merges the two laughs 6 s apart (1 onset); A does the same. At a 1.02 s cadence B
matches its 1.000 s row. B' gives the 1.000 s row above at 0.98, 0.999, 1.000 and 1.02 s cadences, so it ships. B' keeps
`active` for about 2 s after a laugh ends, and onset latency does not change. (The old rule's double count is also
cadence-dependent: its re-onset 4 blocks later only cleared the 4 s refractory when the blocks were at least 1.000 s apart.)
The first probe laugh phase (6-11) scored 0.00 every second, so no debouncer can count it. That miss is in the detector.

## Live demo smoke (P0-1, 2026-10-18, `FRONTEND=1 /bin/bash scripts/spikes/run_demo.sh`)

| | unsandboxed | sandboxed (`valence_offline.sb`, default now) |
|---|---|---|
| edge / mic | up / up | up / up |
| video FPS | 10.0 | 10.2 |
| predictions / explanations in 90 s | 18 / 5 (llm 3, retry 1, template 1) | 34 / 10 (llm 9, retry 1) |
| frames_decoded = frames_purged, purge_gap, violations | 1950 = 1950, 0, 0 | 1099 = 1099, 0, 0 |
| venue baseline step | captured 42.1 dB / 309 lux | skipped (BASELINE=0) |
| **external network from the API process** | **4 established HTTPS connections to Google** (MediaPipe Clearcut) | **none established**; a one-time startup burst of 64 unconnected sockets (refused with EPERM) was gone 2 min later; fds flat at 379 |

MediaPipe 1.1.1 (also 1.0.0 and 0.10.35) embeds `ClearcutLoggingClient` posting usage logs to
`https://play.googleapis.com/log`, with no opt-out. `run_demo.sh` therefore runs the narrator and the API under
`sandbox-exec -f scripts/spikes/valence_offline.sb` (deny network-outbound except loopback). Camera, mic, MPS, MLX and
the localhost API/narrator/WebSocket traffic all work inside it. The frontend dev server is not sandboxed (it never
sees camera or mic data).

## Venue robustness (offline replay)

**Question:** does the served context model (`m2_lgbm@8a78efc7`) still read the captures correctly in a loud, bright
hall? Offline, CPU only. The captures `data/real/capture_*.jsonl` were shifted to hall conditions (features only; no
media was read or written), then replayed exactly as `valence/eval/negative_control.py` does: `Settings`,
`EventBus`, `load_served`, `Runtime(..., gate=PurgeGate())`, `use_live()`, `aggregator.source = "replay"`,
`replay_offline`. The baseline is the replay's automatic one. The script is in the session scratchpad, not in the repo, and the shifted copies were deleted afterwards. The math
is given in full below so the run can be reproduced. (The original `venue/venue_replay.py` was removed with the
scratchpad's `venue/` folder; the adversarial review's `venue/review/verify.py` re-implements the transform
independently and is the reproducible version. See "Adversarial review" at the end of this section for corrections.)

**Transformations** (per observation, in file order; offset = 94.0 dB, the `spl_offset_db` default; the captures
were uncalibrated, and `db_spl = 10·log10(rms_power) + 94` holds exactly on the unsmoothed blocks):
- **Hall noise H**:
  - `P_H = 10^((H − 94)/10)`, and `rms_power' = rms_power + P_H`.
  - `db_spl' = 10·log10(10^((db_spl − 94)/10) + A(k)) + 94`. **Correction (review):** only `capture_typing` and
    `capture_fists` were recorded with the older EMA meter (α 0.3); for them `A(k) = P_s(k) = 0.3·P_H(k) + 0.7·P_s(k−1)`
    is exact because the EMA is linear in power. `keyboard_gaze`, `slump`, `audio` and `baseline` store the block Leq
    (`db_spl == 10·log10(rms_power) + 94` on every block, as the current `AudioPerceiver` emits), so `A(k) = P_H(k)`.
    The original run used the EMA for all six. For a constant H the two are identical after block 1, so the hall,
    late and burst cells are unaffected beyond a ≤ 3-block ramp at the step; the babble cells on the four Leq captures
    had their ±2.5 dB jitter smoothed to about ±1.05 dB. Babble rows below are the corrected re-run (marked †).
  - `laughter_onsets`, `compliment_hits` and `speech_prob` are unchanged. The core's near-field gate (baseline
    + 6 dB) then decides on `db_spl'`. The *ungated* variant forces `Aggregator._near_field` to true.
  - H ∈ {none, 62, 70, 76}.
- **Bright hall L**: `lux' = lux · L`, with `lux_quality` unchanged. L ∈ {1, 2, 4}.
- **Baseline conditions**:
  - **hall** (i): the shift is present from the first observation, so the baseline is captured in the hall (the
    realistic demo case).
  - **late** (ii): the baseline is captured quiet and dim, and H and L start at t ≥ 120 s.
  - **burst** (iii): H + 6 dB on the first 90 audio blocks (the baseline window), H after that, and L from the
    start.
  - **babble** (extra): H = 70 plus iid N(0, 2.5) dB per block, plus a judge talking nearby (+8 dB for 15 s every
    60 s from t = 100 s), with seed 20261018.
- **Recordings**:
  - `capture_audio` is 45 s, shorter than the 60 s warm-up. It was stitched after the same-day `keyboard_gaze` (180 s)
    and `baseline` (25 s) captures, played twice, and only ticks inside the audio part were scored (18 ticks).
  - `capture_baseline` (25 s) was looped 8 times as a neutral control.
  - The slump (26) and audio (18) tick counts include the replay's post-loop flush tick, which holds only 0.03 s / 0.21 s
    of data. Dropping it changes no share (review).
- **Scope**: all cells, with no subsampling (238 replays, 13 s on 6 CPU workers).

**Cells:** `X s · c` means the smoothed label X (F Focused, C Collaborative/Upbeat, M Frustrated/Masked, E Exhausted)
held share s of the non-warming-up ticks, and c is the mean confidence. **`mask_override` was 0 in every cell**, and
also 0 unshifted. Expected share: Focused for typing and gaze, Frustrated/Tense for fists, Exhausted for slump,
Collaborative for audio, and "not stress" for baseline.

| recording | baseline cond. | H (dB) | L = 1 | L = 2 | L = 4 | expected share (L=1/2/4) | ticks |
|---|---|---|---|---|---|---|---|
| typing | hall | none | F 1.00 · 0.930 | F 1.00 · 0.914 | F 1.00 · 0.710 | 1.00/1.00/1.00 | 25 |
| typing | hall | 62 | F 1.00 · 0.935 | F 1.00 · 0.919 | F 1.00 · 0.715 | 1.00/1.00/1.00 | 25 |
| typing | hall | 70 | F 1.00 · 0.935 | F 1.00 · 0.919 | F 1.00 · 0.715 | 1.00/1.00/1.00 | 25 |
| typing | hall | 76 | F 1.00 · 0.935 | F 1.00 · 0.919 | F 1.00 · 0.715 | 1.00/1.00/1.00 | 25 |
| typing | late | none | (= unshifted) | F 1.00 · 0.828 | F 1.00 · 0.809 | 1.00/1.00/1.00 | 25 |
| typing | late | 62 | F 1.00 · 0.923 | F 1.00 · 0.827 | F 1.00 · 0.808 | 1.00/1.00/1.00 | 25 |
| typing | late | 70 | F 1.00 · 0.924 | F 1.00 · 0.832 | F 1.00 · 0.814 | 1.00/1.00/1.00 | 25 |
| typing | late | 76 | F 1.00 · 0.836 | F 1.00 · 0.824 | F 1.00 · 0.815 | 1.00/1.00/1.00 | 25 |
| typing | burst | 62 | F 1.00 · 0.934 | F 1.00 · 0.918 | F 1.00 · 0.713 | 1.00/1.00/1.00 | 25 |
| typing | burst | 70 | F 1.00 · 0.934 | F 1.00 · 0.917 | F 1.00 · 0.712 | 1.00/1.00/1.00 | 25 |
| typing | burst | 76 | F 1.00 · 0.934 | F 1.00 · 0.917 | F 1.00 · 0.712 | 1.00/1.00/1.00 | 25 |
| typing | babble† | 70 | F 1.00 · 0.933 | F 1.00 · 0.918 | — | 1.00/1.00/— | 25 |
| keyboard gaze | hall | none | F 1.00 · 0.996 | F 1.00 · 0.996 | F 1.00 · 0.772 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | hall | 62 | F 1.00 · 0.996 | F 1.00 · 0.995 | F 1.00 · 0.772 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | hall | 70 | F 1.00 · 0.996 | F 1.00 · 0.995 | F 1.00 · 0.772 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | hall | 76 | F 1.00 · 0.996 | F 1.00 · 0.995 | F 1.00 · 0.772 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | late | none | (= unshifted) | F 1.00 · 0.897 | F 1.00 · 0.888 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | late | 62 | F 1.00 · 0.991 | F 1.00 · 0.895 | F 1.00 · 0.886 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | late | 70 | F 1.00 · 0.893 | F 1.00 · 0.895 | F 1.00 · 0.886 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | late | 76 | F 1.00 · 0.893 | F 1.00 · 0.886 | F 1.00 · 0.886 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | burst | 62 | F 1.00 · 0.996 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | burst | 70 | F 1.00 · 0.996 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | burst | 76 | F 1.00 · 0.996 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 25 |
| keyboard gaze | babble† | 70 | F 1.00 · 0.996 | F 1.00 · 0.995 | — | 1.00/1.00/— | 25 |
| fists | hall | none | F 0.96 C 0.04 · 0.838 | F 0.96 C 0.04 · 0.821 | F 0.92 C 0.08 · 0.681 | 0.00/0.00/0.00 | 25 |
| fists | hall | 62 | F 0.96 C 0.04 · 0.838 | F 0.96 C 0.04 · 0.819 | F 0.92 C 0.08 · 0.681 | 0.00/0.00/0.00 | 25 |
| fists | hall | 70 | F 0.96 C 0.04 · 0.838 | F 0.96 C 0.04 · 0.819 | F 0.92 C 0.08 · 0.682 | 0.00/0.00/0.00 | 25 |
| fists | hall | 76 | F 0.96 C 0.04 · 0.838 | F 0.96 C 0.04 · 0.819 | F 0.92 C 0.08 · 0.682 | 0.00/0.00/0.00 | 25 |
| fists | late | none | (= unshifted) | F 0.92 C 0.08 · 0.843 | F 0.92 C 0.08 · 0.783 | 0.00/0.00/0.00 | 25 |
| fists | late | 62 | F 0.92 C 0.08 · 0.852 | F 0.92 C 0.08 · 0.829 | F 0.92 C 0.08 · 0.772 | 0.00/0.00/0.00 | 25 |
| fists | late | 70 | F 0.92 C 0.08 · 0.813 | F 0.92 C 0.08 · 0.803 | F 0.92 C 0.08 · 0.771 | 0.00/0.00/0.00 | 25 |
| fists | late | 76 | F 0.92 C 0.08 · 0.786 | F 0.92 C 0.08 · 0.785 | F 0.92 C 0.08 · 0.768 | 0.00/0.00/0.00 | 25 |
| fists | burst | 62 | F 0.96 C 0.04 · 0.851 | F 0.96 C 0.04 · 0.827 | F 0.96 C 0.04 · 0.660 | 0.00/0.00/0.00 | 25 |
| fists | burst | 70 | F 0.96 C 0.04 · 0.850 | F 0.96 C 0.04 · 0.827 | F 0.96 C 0.04 · 0.658 | 0.00/0.00/0.00 | 25 |
| fists | burst | 76 | F 0.96 C 0.04 · 0.850 | F 0.96 C 0.04 · 0.826 | F 0.96 C 0.04 · 0.658 | 0.00/0.00/0.00 | 25 |
| fists | babble† | 70 | F 0.96 C 0.04 · 0.840 | F 0.92 C 0.08 · 0.858 | — | 0.00/0.00/— | 25 |
| slump | hall | none | F 1.00 · 0.695 | F 1.00 · 0.696 | F 1.00 · 0.686 | 0.00/0.00/0.00 | 26 |
| slump | hall | 62 | F 1.00 · 0.683 | F 1.00 · 0.685 | F 1.00 · 0.669 | 0.00/0.00/0.00 | 26 |
| slump | hall | 70 | F 1.00 · 0.682 | F 1.00 · 0.684 | F 1.00 · 0.667 | 0.00/0.00/0.00 | 26 |
| slump | hall | 76 | F 1.00 · 0.682 | F 1.00 · 0.684 | F 1.00 · 0.667 | 0.00/0.00/0.00 | 26 |
| slump | late | none | (= unshifted) | F 1.00 · 0.735 | F 1.00 · 0.734 | 0.00/0.00/0.00 | 26 |
| slump | late | 62 | F 1.00 · 0.633 | F 1.00 · 0.693 | F 1.00 · 0.702 | 0.00/0.00/0.00 | 26 |
| slump | late | 70 | F 1.00 · 0.630 | F 1.00 · 0.693 | F 1.00 · 0.701 | 0.00/0.00/0.00 | 26 |
| slump | late | 76 | F 1.00 · 0.631 | F 1.00 · 0.693 | F 1.00 · 0.700 | 0.00/0.00/0.00 | 26 |
| slump | burst | 62 | F 1.00 · 0.698 | F 1.00 · 0.703 | F 1.00 · 0.698 | 0.00/0.00/0.00 | 26 |
| slump | burst | 70 | F 1.00 · 0.698 | F 1.00 · 0.702 | F 1.00 · 0.698 | 0.00/0.00/0.00 | 26 |
| slump | burst | 76 | F 1.00 · 0.698 | F 1.00 · 0.702 | F 1.00 · 0.698 | 0.00/0.00/0.00 | 26 |
| slump | babble† | 70 | F 1.00 · 0.675 | F 1.00 · 0.671 | — | 0.00/0.00/— | 26 |
| audio | hall | none | F 1.00 · 0.698 | F 1.00 · 0.667 | F 1.00 · 0.542 | 0.00/0.00/0.00 | 18 |
| audio | hall | 62 | F 1.00 · 0.756 | F 1.00 · 0.673 | F 1.00 · 0.546 | 0.00/0.00/0.00 | 18 |
| audio | hall | 70 | F 1.00 · 0.757 | F 1.00 · 0.674 | F 1.00 · 0.547 | 0.00/0.00/0.00 | 18 |
| audio | hall | 76 | F 1.00 · 0.756 | F 1.00 · 0.676 | F 1.00 · 0.548 | 0.00/0.00/0.00 | 18 |
| audio | late | none | (= unshifted) | F 1.00 · 0.751 | F 1.00 · 0.646 | 0.00/0.00/0.00 | 18 |
| audio | late | 62 | F 1.00 · 0.597 | F 1.00 · 0.630 | F 1.00 · 0.635 | 0.00/0.00/0.00 | 18 |
| audio | late | 70 | F 1.00 · 0.493 | F 1.00 · 0.633 | F 1.00 · 0.638 | 0.00/0.00/0.00 | 18 |
| audio | late | 76 | F 1.00 · 0.487 | F 1.00 · 0.634 | F 1.00 · 0.638 | 0.00/0.00/0.00 | 18 |
| audio | burst | 62 | F 1.00 · 0.739 | F 1.00 · 0.670 | F 1.00 · 0.544 | 0.00/0.00/0.00 | 18 |
| audio | burst | 70 | F 1.00 · 0.741 | F 1.00 · 0.665 | F 1.00 · 0.540 | 0.00/0.00/0.00 | 18 |
| audio | burst | 76 | F 1.00 · 0.741 | F 1.00 · 0.665 | F 1.00 · 0.540 | 0.00/0.00/0.00 | 18 |
| audio | babble† | 70 | F 1.00 · 0.738 | F 1.00 · 0.700 | — | 0.00/0.00/— | 18 |
| audio (ungated) | hall | none | F 1.00 · 0.712 | F 1.00 · 0.686 | F 1.00 · 0.555 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | hall | 62 | F 1.00 · 0.725 | F 1.00 · 0.648 | F 1.00 · 0.528 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | hall | 70 | F 1.00 · 0.724 | F 1.00 · 0.646 | F 1.00 · 0.527 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | hall | 76 | F 1.00 · 0.723 | F 1.00 · 0.648 | F 1.00 · 0.528 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | late | none | (= unshifted) | F 1.00 · 0.757 | F 1.00 · 0.649 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | late | 62 | F 1.00 · 0.597 | F 1.00 · 0.630 | F 1.00 · 0.635 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | late | 70 | F 1.00 · 0.493 | F 1.00 · 0.633 | F 1.00 · 0.638 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | late | 76 | F 1.00 · 0.487 | F 1.00 · 0.634 | F 1.00 · 0.638 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | burst | 62 | F 1.00 · 0.727 | F 1.00 · 0.665 | F 1.00 · 0.540 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | burst | 70 | F 1.00 · 0.727 | F 1.00 · 0.656 | F 1.00 · 0.534 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | burst | 76 | F 1.00 · 0.727 | F 1.00 · 0.657 | F 1.00 · 0.534 | 0.00/0.00/0.00 | 18 |
| audio (ungated) | babble† | 70 | F 1.00 · 0.736 | F 1.00 · 0.698 | — | 0.00/0.00/— | 18 |
| baseline | hall | none | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | hall | 62 | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | hall | 70 | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | hall | 76 | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | late | none | (= unshifted) | F 1.00 · 0.914 | F 1.00 · 0.868 | 1.00/1.00/1.00 | 29 |
| baseline | late | 62 | F 1.00 · 0.910 | F 1.00 · 0.880 | F 1.00 · 0.865 | 1.00/1.00/1.00 | 29 |
| baseline | late | 70 | F 1.00 · 0.878 | F 1.00 · 0.873 | F 1.00 · 0.865 | 1.00/1.00/1.00 | 29 |
| baseline | late | 76 | F 1.00 · 0.870 | F 1.00 · 0.865 | F 1.00 · 0.865 | 1.00/1.00/1.00 | 29 |
| baseline | burst | 62 | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | burst | 70 | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | burst | 76 | F 1.00 · 0.995 | F 1.00 · 0.995 | F 1.00 · 0.771 | 1.00/1.00/1.00 | 29 |
| baseline | babble† | 70 | F 1.00 · 0.994 | F 1.00 · 0.994 | — | 1.00/1.00/— | 29 |

**Flips relative to the unshifted replay:**
- **No cell changes its majority label.**
- Per-tick disagreements occur only in **fists**, and only on **1 tick** (tick 34, the second-to-last; tick 35, the
  last, reads Collaborative in every replay including unshifted, which is the 0.04 C share). They appear
  at L = 4 under hall (every H), in every late cell and in babble at L = 2, but never under burst.
  - The model's raw reading on that tick is Collaborative in every replay, unshifted included: raw P(C) is 0.79–0.90,
    with `face_now` 0.72 (a smile). Unshifted, hysteresis still holds Focused on tick 34, shown at 0.14
    confidence. Shifted, the EMA crosses one tick earlier.
  - **Correction (review): the decisive tick is 33, not 34.** The smoother (EMA α 0.5, hold 2) switches on the second
    consecutive tick whose EMA argmax differs. Unshifted, the EMA argmax first turns Collaborative at tick 34 (shown
    at 35); shifted, at tick 33 (raw P(Focused) 0.25 → 0.13 at L = 4, → 0.08 late H 62), so it shows at 34. The tick-33
    drivers are the same features as the tick-34 ones below: L = 4 hall ΔFocused −0.66 (`lux_std_5m` 19 → 77, −0.12;
    `lux_slope_5m` −11.6 → −46.4, −0.11; `lux_std_15m` −0.08; `lux_rel_15m` −7.7 → −30.9, −0.07); late H 62 ΔFocused
    −0.83 (`db_room_mean_1m` 0.2 → 12.6, −0.25; `db_slope_5m` −0.1 → 6.5, −0.22; `db_std_5m` 1.2 → 6.7, −0.21;
    `db_room_now` −0.6 → 15.2, −0.12). Tick-34 SHAP (as originally reported):
  - **L = 4 hall:** the change in SHAP against unshifted for Focused is −0.54 in total. The top terms are
    `lux_slope_5m` −9.4 → −37.7 lux/min (−0.12), `lux_std_5m` 19 → 78 lux (−0.12) and `lux_std_15m` (−0.08).
  - **H 62, late:** the change for Focused is −0.89. The top terms are `db_room_mean_1m` 0.1 → 13.8 dB (−0.26),
    `db_slope_5m` −0.2 → 6.6 dB/min (−0.25), `db_std_5m` 1.2 → 6.8 (−0.20) and `db_room_now` −0.8 → 15.2 (−0.11).
  - The face layer dominates in every case: `face_now` is about ±2.0 SHAP. The shift only changes when the
    hysteresis lets go.

**Confidence (no label change):**
- **L = 4 (about 1,100–1,600 lux on the edge's scale)**, under every baseline condition:
  - The cost is a flat −0.22 to −0.23: gaze 0.996 → 0.772, typing 0.93 → 0.71, audio 0.70 → 0.54.
  - The SHAP change is about 0. The cause is the **OOD shrink** (`p·0.7 + 0.075`): 9–12 features per tick are out of
    range.
  - They are the absolute `lux_now / lux_mean_* / lux_ewm_* / lux_max_*`, **none of which is an input of M2 or M0**.
  - Their OOD bands end at 1,066–1,115 lux, so from about **1,090 lux steady** four of them are out of range and
    every tick shrinks.
  - L = 2 (600–830 lux) costs ≤ 0.02 on typing, gaze, fists, slump and baseline. **Correction (review):** the stitched
    audio replay loses 0.08 at L = 2 (0.757 → 0.675 at H 70) with ≤ 2 OOD features, so no shrink: the raw-lux spread
    inputs (ask 2) scale the artificial lux step at the stitch joins (397 → 275 → 288 lux) by L.
  - **Reachability (review):** the edge's lux is uncalibrated (`data/real/calibration.json` absent, default
    `lux = exp(3.59 + 0.01463·y_bg)`), so 1,090 lux needs a background mean luma ≥ 233/255 and the model cannot exceed
    1,511 lux (y_bg = 255). L = 4 on gaze, slump and audio (1,590–1,620 lux) is beyond what the edge can report, and
    webcam auto-exposure pulls a brighter room back towards mid-grey. Treat L = 4 as a stress bound; L = 2 (y_bg
    about 190–210, a bright or backlit background) is the realistic bright-hall case.
- **late** (a +20–36 dB step on a quiet baseline):
  - Confidence falls: gaze 0.996 → 0.89, audio 0.70 → 0.49, slump 0.70 → 0.63.
  - Up to 8 features are OOD: `db_room_now`, `db_rel_15m`, `db_std_*`, `db_slope_*`.
  - A late lux step of ×2 or ×4 on a dim baseline costs gaze 0.996 → 0.89 and typing 0.93 → 0.81.
- **hall / burst / babble at L ≤ 2:** confidence stays within ±0.03 of unshifted at every H, including 76 dB, except
  on audio (review): +0.06 at L = 1 (0.70 → 0.76) because the gate drops the two laughs once H ≥ 62 (ungated: +0.01),
  and −0.08 at L = 2 (above). A burst-inflated baseline only moves `db_room_*` to about −6 dB, which the model ignores.
- **Replay clock (review):** `replay_offline` maps the recording onto the current wall clock, and `tod_sin`/`tod_cos`
  are M2 inputs, so confidences depend on when the replay runs. Pinned to 11:00, 13:00, 15:00 and 17:00 local, every
  majority label is unchanged; confidences move by up to 0.03 (fists 0.838/0.811/0.822/0.819, audio
  0.694/0.676/0.663/0.673 unshifted). Differences under about 0.03 in the table are within that noise.

**Expected readings:**
- **Typing, gaze, baseline:** Focused / not stress = 1.00 in every cell.
- **Fists (stress 0.00), slump (Exhausted 0.00), audio (Collaborative 0.00):** these fail in **every cell including
  unshifted**. The venue shift neither causes nor fixes them.
  - Fists read Focused, as in the core's round-7 result.
  - On slump, all 26 ticks are OOD on `motion_*` (the P1-1 held-slump artifact), so confidence is about 0.69
    everywhere.
  - Audio is Collaborative on 11% of raw-argmax ticks (the laugh-face ticks) in every condition, and never after
    smoothing.

**Near-field gate (laughter and compliments):**
- Unshifted, **6 of 10 events are gated**: the 3 compliment hits, played twice. They arrive on blocks at
  39.3–39.8 dB, the room floor, while the two laughs arrive at 61.8 and 63.9 dB. The Vosk hit lands on the block after
  the phrase, which is quiet again.
- In every hall and burst cell, **10 of 10 are gated**. At-desk laughter at about 62–64 dB power-sums to about H + 1 dB
  under a 62–76 dB hall, below baseline + 6.
- In babble, 2 events passed (3 in the corrected re-run), and only because a judge burst raised their blocks.
- **Realism (review):** the transform adds hall power to the recorded quiet-room laughs but cannot model the Lombard
  effect (people laugh and talk louder in noise, typically several dB, up to about 0.5 dB per dB of noise above
  ~50 dB). A block passes when the user's own level is ≥ H + 4.7 dB. The recorded 63.9 dB laugh needs +2.8 dB at
  H 62 (likely to pass in reality), +10.8 dB at H 70 (borderline) and +16.8 dB at H 76 (blocked). "10 of 10 gated"
  is the no-Lombard worst case at 62 dB and close to reality at 70–76 dB.
- Ungated replays change mean P(Collaborative) by +0.03 to +0.04 and no labels, so with these captures the gate has
  no effect on the label.

**Asks for the core side** (corrected by the adversarial review where marked):
1. **The OOD shrink should count only model inputs.**
   - `ServedPredictor._clip` reports OOD over all 123 `FEATURE_NAMES`, and 22 of them feed neither M2 nor M0. Among
     those 22 are the absolute dB/lux levels that round 7 removed from the model for venue shift.
   - Keep clipping every feature, but build `ood` from the M2 and M0 columns only.
   - Effect: in a hall above about 1,090 lux (edge scale), every prediction loses about 0.22 confidence and
     `data_quality.ood_features` reports "unusual input", although the model's own inputs are fine.
   - Measured what-if (review, OOD band masked to M2 ∪ M0 columns): L = 4 hall H 70 recovers typing 0.714 → 0.913, gaze
     0.772 → 0.995, fists 0.684 → 0.837, baseline 0.771 → 0.995; audio (6 model inputs OOD, the stitch artifact) and
     slump (P1-1) are unchanged; unshifted home replays are identical on all six captures, so home behaviour is kept.
2. **Linear-lux inputs leak absolute light.**
   - M2 still uses `lux_std_5m / lux_slope_5m / lux_std_15m / lux_slope_15m / lux_d1m15m / lux_rel_15m` in lux units.
     They scale ×L with the hall's brightness and moved the fists tick above.
   - Use log2-lux versions, or divide by `room_lux_base`, as `lux_room_*` already does.
   - The `db_*` spread and slope inputs are dB-domain and invariant to an additive shift. Under a loud hall they
     collapse (typing `db_std_5m` 0.12 at H 70 against a training p1 of 0.91), yet Focused held, so this is lower
     priority.
3. **The lux baseline fills in about 9 s, not 90 s.**
   - `RoomBaseline.add_lux` is called on every `VideoObs` (10 Hz), and the target is 90 samples. After a
     `POST /api/room/baseline`, `lux_samples` is 90 after about 9 s, and those 9 s fix `lux_base` for the session.
   - Sample lux at 1 Hz, for example only when `obs.t` crosses a new second.
   - Measured what-if (review, every 10th frame): slump's `lux_base` 398 → 321 lux and confidence 0.689 → 0.725 at
     home (its lux drifts 276–400 during the first 90 s); other captures move ≤ 0.002; no label changes.
4. **The near-field gate on compliments uses the wrong block.**
   - Gate on the max `db_spl` of the last 2–3 audio blocks (the phrase's own blocks), not the block on which the Vosk
     hit arrives.
   - In the hall, a +6 dB rule over a 62–76 dB median gates all at-desk laughter, and judge bursts pass it instead.
     Consider a margin from the baseline's IQR or p90, or a speech_prob-weighted test.
   - The core decides whether social events should count in a 70 dB hall at all. If not, the SIMULATED inject is the
     stage fallback.
5. **Late arrival:**
   - A large step against an old baseline (condition ii) is the only case that reaches 0.49 confidence.
   - Suggestion: warn in `/api/health` or the operator panel when `db_room_mean_5m` > +12 dB or `|lux_room_mean_5m|`
     > 1 for 5 min, to prompt a re-capture.
   - A burst-inflated baseline (condition iii) did not hurt labels. **Correction (review):** `db_iqr` would not catch
     one: a burst that covers the whole window gives `db_iqr` 0.0 (all burst cells). What does show it is a negative
     `db_room_mean_5m` (about −6 dB) after capture, so make the warning two-sided: re-capture prompt also when
     `db_room_mean_5m` < −4 dB.
   - Timing (review): over the 180 s captures `db_room_mean_5m` reached only +8 to +18 dB after a +20–36 dB step
     (the window still holds the quiet minutes), against ≤ +3.5 dB in every home replay. Use `db_room_mean_1m` > +12 dB
     held for 2 min to warn sooner; home `|lux_room_mean_5m|` peaked at 0.36, so the lux threshold of 1 is safe.
6. **Replay reproducibility (review):** let `replay_offline` take a fixed start wall time (or tod) so benchmark
   confidences do not depend on the hour the replay runs.

**Adversarial review (re-run independently, CPU, same served model):**
- **Confirmed:** the power-sum math and offset (94 dB, captures uncalibrated); each baseline condition is applied
  (replay baseline `db_median` 70.0 in hall, 40.7/48.7 quiet in late, H + 6 in burst); warm-up ticks are excluded
  (`warming_up`); one prediction per tick and person (predictor calls = messages), so the SHAP vectors line up; no
  majority label flips; `mask_override` 0 everywhere; L = 4 shrink caused by OOD on non-input absolute lux features;
  compliments gated because the Vosk hit lands on the block after the phrase (phrase blocks 56.9–58.6 dB, hit blocks
  39.3–39.8 dB); ungated changes P(Collaborative) by 0.01–0.03 and no labels; 22 of 123 features feed neither model.
- **Corrected above:** EMA applied to Leq captures (babble re-run); flip tick is decided at tick 33; "L = 2 ≤ 0.02" and
  "±0.03" do not hold for the stitched audio replay; `db_iqr` cannot reveal a full-window burst; replay confidences
  carry ±0.03 time-of-day noise; L = 4 exceeds what the uncalibrated edge can report on three captures; the gate result
  at 62 dB is a no-Lombard worst case.
- **Demo implication (unchanged):** capture the baseline after arriving, seated and neutral, outside an announcement
  (condition i). Expect the same labels as at home; expect lower confidence only if `lux_median` prints above about
  1,000 (until ask 1 lands) or if the room gets much louder after the capture (re-capture). Do not rely on laughter or
  compliments registering at 70 dB; the SIMULATED inject is the fallback.

## Audio detectors in crowd noise (synthetic)

Question: in a 65-75 dB hall, do the edge audio detectors invent compliments or laughs from crowd babble, and does
the core's near-field gate (event block >= room baseline + 6 dB) keep the user's own events? Synthetic test, no mic:
macOS `say` TTS plus generated pink noise, mixed in memory and fed to `AudioPerceiver.process` as 16 kHz float32
1 s blocks (YAMNet AUDIO_CLIPS + Vosk restricted grammar, the shipped path). Peak < 1 throughout. The WAVs lived only
in the session scratchpad and were deleted afterwards.

- **Compliments:** 5 phrases ("great job", "nice work", "thank you", "well done", "that is amazing") x 4 voices
  (Samantha, Daniel, Karen, Fred) = 20 utterances. Each SNR condition is one continuous stream with an utterance every
  15 s, so the recognizer runs the way it does live.
- **Babble:** 6-10 overlapping talkers (14 other voices), each at a random -10..0 dB distance gain, reading 37
  hackathon chatter sentences, plus pink noise 6 dB below the babble. The sentences include 5 with a grammar phrase
  verbatim ("thank you for coming", "awesome demo", "I love it here", ...) and 7 with grammar words split or out of
  order ("the job is great", "nice, it works", "well, done soon", ...).
- **SNR** = the utterance's active-speech power over the hall's power. If the hall is 70 dB, +10 / +5 / 0 / -5 dB
  means the user's speech reaches the mic at about 80 / 75 / 70 / 65 dB.

### Babble only: false compliments and laughs (5 min each; pink 3 min, near-silence 1 min)

Three runs per row: the original run and an adversarial re-run with mixing seeds 7 and 11 (TTS regenerated, same
script). Cells give the range over the runs. The seed-7 re-run did not reproduce the original bit for bit (6-voice
row 4.4 against 6.2/min), so a single run carries about +-2/min of noise. The reverberant row was added by the review: each of 24 talkers is convolved with
an exponential-decay room response (RT60 0.8 s, direct-to-reverberant -6 dB), which is closer to a crowd heard from a
laptop in a hall than the dry 6-10 talker mix.

| condition | false compliments/min, shipped rule -> with the phrase loudness check | verbatim-phrase sentences/min in the mix | laughter onsets/min | laughter family max | `speech_prob` p10 / p50 / p90 |
|---|---|---|---|---|---|
| dry babble, 6 voices + pink | **4.4-6.2 -> 0.2-1.0** | 15.6-16.2 | 0 | <= 0.012 | 0.20-0.33 / 0.59-0.67 / 0.85-0.89 |
| dry babble, 10 voices + pink | 0.4-1.0 -> 0.0 | 26.4-30.6 | 0 | <= 0.016 | 0.41 / 0.74 / 0.89 |
| dry babble, 8 voices + pink, no verbatim-phrase sentences | 2.8-4.4 -> 0.0-0.2 | 0 | 0 | <= 0.016 | 0.26-0.41 / 0.74 / 0.89 |
| dry babble 8 + pink + one non-stop talker at the next seat, +6 dB over the crowd | 4.2-5.0 -> 4.2-5.0 | 24.0-28.2 | 0 | <= 0.016 | 0.33-0.41 / 0.80 / 0.92-0.94 |
| **reverberant crowd, 24 voices + pink** (review), with and without verbatim-phrase sentences | **0.0 -> 0.0** | 72.8 / 0 | 0 | 0.004 | 0.20 / 0.59 / 0.80-0.85 |
| pink noise only | 0 -> 0 | 0 | 0 | 0 | 0.01 / 0.01 / 0.02 |
| near-silence (-90 dBFS) | 0 -> 0 | 0 | 0 | 0 | 0.01 / 0.01 / 0.02 |

- **Laughter:** 0 false onsets in every condition and run. The family never exceeded 0.016, against the 0.05 on
  threshold. Laughter recall in noise was not measured: TTS cannot laugh.
- **Compliments:**
  - Shipped rule (consecutive words, conf >= 0.7): with a *dry* mix of 6-10 talkers, Vosk's 8-phrase grammar forces
    babble into phrases: up to 4.4-6.2 false compliments per minute, even with no grammar phrase in the mix
    (2.8-4.4/min). Confidence does not separate them: false hits read conf p50 0.70-0.96, p90 1.00 (TTS true hits
    read 1.00; real voices will read lower). Part of the 6-voice rate is genuine: bystanders saying "thank you" or
    "awesome demo" verbatim.
  - **The dry mix overstates the crowd risk.** Each dry talker is as intelligible as a person 1-3 m away with no
    room reverberation. In the reverberant 24-talker crowd, Vosk produced **no** phrase at all, before or after the
    check, even with 73 verbatim-phrase sentences per minute in the mix.
  - **The realistic false-compliment source is one intelligible talker close to the laptop** (the next-seat row:
    4.2-5.0/min). It is louder than the crowd, so the loudness check keeps all of it, and only the core's gate can
    drop it (see the gate section: it does so only if that talker was already present during the baseline).
  - Against dry mid-distance talkers, the word-aligned loudness test separates them; confidence and word span do not.
- **`speech_prob`:** babble reads as Speech (p50 0.59-0.80), so it says nothing about who is talking.

### Compliment hit rate in a 70 dB hall (20 utterances per row)

Ranges over the original run and the two re-runs (dry 8-voice hall); the last column is the review's reverberant
24-voice hall. The re-runs score a hit as right only if its Vosk word times overlap the utterance (+-0.3 s).

| SNR | user's speech at the mic | right phrase, shipped rule (+ false hits in the stream) | right phrase, with the loudness check (+ false hits) | latency after the phrase ends, p50 / max | 1 s speech-block rise over the hall, p10 / p50 / p90 | reverberant hall: shipped -> check |
|---|---|---|---|---|---|---|
| clean (quiet) | | 20/20 (+0) | **20/20** (+0) | 1.1-1.3 / 1.8 s | (quiet room) | 20/20 -> 20/20 |
| +10 dB | ~80 dB | 19-20/20 (+0-1) | **19-20/20** (+0) | 1.0-1.1 / 1.6 s | 7.5-7.7 / 8.4-8.5 / 9.3-9.4 dB | 20/20 -> 20/20 |
| +5 dB | ~75 dB | 16-18/20 (+3-9) | 16-17/20 (+0) | 1.1-1.3 / 2.2 s | 3.7-3.9 / 4.6-4.8 / 5.3-5.5 dB | 17/20 -> 17/20 |
| 0 dB | ~70 dB | 10-13/20 (+9-12) | 8-13/20 (+0-1) | 1.0 / 2.1 s | 1.1-1.6 / 2.0-2.2 / 2.8-3.1 dB | 12/20 -> 10/20 |
| -5 dB | ~65 dB | 2-5/20 (+6-11) | 0/20 (+0-2) | | -0.1 to 0.3 / 0.7-1.2 / 1.5-1.8 dB | 0/20 -> 0/20 |
| user talks non-stop, compliments inside the talk, quiet room | | 20/20 | 20/20, own-talk false 0.8-1.8/min | 1.3-1.4 / 2.6 s | | 20/20, 1.6/min |
| same, hall, +5 dB | | 16-19/20 | 15-19/20, own-talk false 0.4-1.0/min | 1.3 / 1.9 s | | 17/20, 1.2/min |
| same, hall, +10 dB | | 20/20 | 20/20, own-talk false 0-1.0/min | 1.2 / 2.6 s | | 20/20, 0.2/min |

- **The check costs nothing at clean, +10 and +5 dB in any run** (same hits before and after); at 0 dB it drops 0-2
  of 10-13 true hits.
- **Detection collapses around 0 dB SNR,** before any gating. For a compliment to register, the user must be louder
  than the hall at the laptop mic: lean in and say it at a raised voice.
- **Own-talk false compliments:** the person's own ordinary talk at the desk gives about 1-2 false compliments per
  minute. It is near-field, so no level test can remove it. A non-stop talker at the next seat looks the same.
- **Latency** (time-aligned): the final arrives 0-2 blocks after the block holding the phrase's end (0 blocks: 20-55%,
  1 block: 45-80%, 2 blocks: <= 15%), at most 2.6 s after the phrase ends. Hits seen 7-12 blocks later in the original window
  scoring were later false hits of the same phrase, not slow finals.

### Edge mitigation (shipped): the phrase loudness check

`PhraseLoudness` (`valence/perception/audio_features.py`) keeps the 100 ms mean powers of the audio given to Vosk on
the recognizer's clock: scalars only, 90 s. A phrase counts only if:

- the mean power over its own words (Vosk `start` of the first word to `end` of the last) is **>= 4 dB** above
- the **20th percentile** of the 100 ms powers in the **30 s before it**.

The low percentile keeps the floor at the room's level while the person talks most of the time (the gaps between
words are the room). With less than 1 s of history the check passes; a phrase without word times never counts.
`AudioPerceiver` passes `near=self.loudness.is_near` to `count_compliments`.

Choosing the values (offline grid on the stored results: windows 10/30/60 s, percentiles 10/20/50, mean or peak
power, 4-7 dB):
- p50 floors or a 6-7 dB threshold also remove the 0 dB hits;
- p10 floors let half the babble through;
- 30 s / p20 / 4 dB keeps clean and +10 at 20/20 and +5 at 17/20, and gives the babble numbers above.
- Review: the values were tuned on the seed-7 run only. On the fresh seed-11 run and the reverberant hall, the check
  still keeps every clean, +10 and +5 hit; at 0 dB it costs 1-2 hits. Re-building the check offline from the stored
  100 ms powers gives exactly the live perceiver's counts in every stream, so the recognizer clock and the loudness
  clock agree.
- Home captures cannot be replayed through it (features only), but at home the floor is the quiet room (~40 dB) and
  the phrase blocks were +14-19 dB over it, far above 4 dB.

Tests: `tests/perception/test_audio_features.py`, the `test_near_check_*`, `test_loudness_*` and
`test_perceiver_applies_the_loudness_check_on_the_recognizer_clock` tests (9 functions). They cover: the span passed,
rejection, missing times, +10 dB over the floor, a crowd-level phrase rejected, the person's own talk not raising the
floor, block size, edges, scalars only, and (review, needs the YAMNet model) the `AudioPerceiver` wiring with a fake
recognizer: a phrase at the room's level gives 0 hits, one 12 dB above it gives 1.

### Core near-field gate: would it keep the user's own events?

**At home, it already drops compliments.** Vosk finalizes a phrase at the endpoint, about 1 s after the phrase ends.
So `compliment_hits` arrives on the *next* block, which is quiet. The gate compares *that* block's dB with the baseline.

| home capture | quiet median | compliment speech block | block carrying the hit | laughter onset block | real `Aggregator` replay |
|---|---|---|---|---|---|
| run 3 (`capture_audio.jsonl`) | 39.7 dB | +17.2 / +18.4 / +18.9 dB | **+0.0 / -0.1 / -0.4 dB** | +24.2 / +22.1 dB | compliments 2 of 3 gated out, laughs 2/2 kept |
| run 2 (`791601a`) | 41.2 dB | +14.5 / +17.9 / +17.0 / +16.2 dB | **-1.4 / -0.9 / -0.8 / +0.2 dB** | (none) | compliments 3 of 4 gated out |

The one kept compliment in each run fell inside the first 10 blocks, before the gate arms (`NEAR_FIELD_MIN_BLOCKS`).
Laughter is fine at home: YAMNet scores the block the laugh is in.

Review: confirmed by re-running the replay (`Aggregator`, `Settings()` defaults). These captures are 45 audio blocks,
shorter than the 90 s baseline, so the replay gated against the running median (the pre-baseline fallback), not
`baseline + 6`. Live, after 90 s, the lag block is compared with the baseline instead and reads the same: the quiet
room. In run 2, two of the speech blocks are 2 blocks before the hit, not 1. `db_spl` here is the raw block Leq
(`10·log10(rms_power) + 94` holds exactly on every block), not a smoothed value, so nothing carries the speech level
into the lag block.

**In a 70 dB hall, the +6 dB margin rejects normal speech even without the lag.** The power sum of the hall (70 dB)
and the user's block Leq gives these rises above the baseline:

| user's block Leq at the mic | 60 | 64 | 65 | 66 | 68 | 70 | 72 | 75 | 80 dB |
|---|---|---|---|---|---|---|---|---|---|
| block rise above a 70 dB baseline | +0.4 | +1.0 | +1.2 | +1.5 | +2.1 | +3.0 | +4.1 | **+6.2** | +10.4 dB |

- **At home:** speech blocks were 56-59 dB and laughs 62-64 dB (default 94 dB offset, uncalibrated; the quiet room
  read 40-41 dB, so the offset is plausible but not verified).
- **In the hall:** even with a Lombard rise of 5-8 dB, speech reaches about 61-67 dB and rises 0.5-1.8 dB over the
  baseline. Laughs at the home level rise 0.6-1.0 dB, or 1.8-4.1 dB if they also rise 5-8 dB. All are gated out at
  +6. (Review: corrected from "64-67 dB, 1.0-1.8 dB" and "laughs 1-3 dB"; the power-sum table above was right.)
- **What passing takes:** the 1 s block's Leq must reach about 75 dB, which is about 77 dB of active speech. A block
  includes pauses: the +5 dB SNR rows above rose only 4.8 dB.

Gate variants on the synthetic hall streams (after the edge check; base = the hall's median):

| SNR | current: hit block >= base + 6 | max(hit block, 2 blocks before) >= base + 6 | max(hit block, 2 blocks before) >= base + 3 |
|---|---|---|---|
| clean | 4/20 (re-runs 3/20; reverberant 6/20) | 20/20 | 20/20 |
| +10 dB | 0/20 (re-runs 0/19; reverberant 2/20) | 19-20 of 19-20 | all |
| +5 dB | 0/17 (re-runs 0) | 0-2 of 16-17 | 15-17 of 16-17 |
| 0 dB | 0/13 (re-runs 0) | 0 | 1-2 of 9-13 |
| babble-only false hits after the check (mostly the next-seat talker's; base includes that talker) | 0/27 (re-runs 0/17, 0/26) | 0 | 0 |
| **review:** next-seat talker's false hits, base = crowd only (the talker sits down after the baseline) | 5/16, 10/20 | 16/16, 18/20 | 16/16, 20/20 |
| **review:** user talks non-stop in the hall, base = hall without the user (neutral baseline), +5 / +10 dB | 3-5 of 21-22 / 11-14 of 20-24 | 9-12 of 21-22 / 20-23 of 20-24 | all / all |

Review notes on this table:
- The zero false hits in the original "babble-only" row hold only because the base was the stream's own median, which
  already contained the next-seat talker. If that talker sits down after the baseline capture, its false compliments
  are 6-7 dB above the baseline and pass every variant, the current gate included.
- With the hit-block lag fixed (max of 3 blocks), the margin trades the user's +5 dB compliments against a nearby
  talker: no level gate separates the two.

Asks for the core side:
- Gate `compliment_hits` on the max of the hit block and the 2 blocks before it. Observed true-hit latency is 0-2
  blocks (at most 2.6 s after the phrase ends) in every run. This is needed at home too: today 2 of 3 and 3 of 4 home
  compliments are dropped. Alternatively skip the block gate for compliments, but the edge check is relative to the
  last 30 s of the room, not to the session baseline, so it does not stop a nearby talker either.
- Scale the margin down in loud rooms, for example +3 dB when the baseline is >= 65 dB. In a 70 dB hall, at-desk
  speech and laughs rise only 0.5-4 dB, so +6 drops them all. Accept that a chatty neighbour then counts too.
- Optionally, a contract field from the edge for the event's own rise (phrase words over the room floor), so the core
  does not infer it from block levels.
- Demo: capture the 90 s baseline while the hall is at its normal level; the `POST /api/sim/event` inject is the
  backup and skips the gate, but is marked SIMULATED and audited.
