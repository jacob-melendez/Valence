# Contract requests / assumptions from the Mac (edge) session

The edge session builds against the shared contract as written. These are the assumptions it makes where the
contract is silent, plus small asks for files the core session owns. Please confirm or correct.

## Semantics the edge emits (confirm)

1. **Counts are per-observation deltas, not cumulative.** `PersonObs.fist_onsets` = debounced fist rising edges
   counted *in this frame* (almost always 0, occasionally 1); `AudioObs.laughter_onsets` and
   `AudioObs.compliment_hits` = counts within this ~1 s block. Aggregate by summing over the window.
2. **`PersonObs.motion`** = EMA (alpha 0.3) of mean keypoint displacement in **shoulder-widths per second**
   (rate-independent; ~0.0-0.1 still, 0.3+ fidgeting). 0.0 when undefined.
3. **`VideoObs.lux`** is a 10 s median, refreshed at 1 Hz; **NaN until the first estimate**. `lux_quality` in
   [0, 1] (background share x unclipped share x stability, halved while uncalibrated).
4. **`face_valence`** is NaN whenever `face_visible` is False. It is relative to the person's own neutral baseline
   (median of their first 20 s of visible face), so ~0 means "their neutral".
5. **Display ids** (`track_id`) are 1..N by first appearance and reused if a person reappears within 3 s and
   0.25 frame-widths of where they were lost; after that the id is released and the person's baselines are
   forgotten (nothing about them is kept).
6. **Features-only JSONL** (`data/real/*.jsonl`, written by `python -m valence.edge.run --jsonl ...`): one object
   per line, `{"type": "video" | "audio", ...dataclass fields...}`, tuples as lists, NaN/inf as `null`.
   Replay mode can read these directly.

## Purge gate API (confirm)

`valence/perception/gate.py` is the edge's single import point:
`from valence.governance.purge import gate; gate.purge(buf, kind)` with `kind in {"video", "audio"}`.
If `purge.py` exposes a different name (e.g. a module-level `purge()`), tell me and I'll change one line.

## Small asks for core-owned files

- **`.gitignore`**: please add `models/`, `adapters/`, `*.pt`, `*.task`, `*.tflite`, `*.safetensors`
  (the Mac session uses `.git/info/exclude` locally until then).
- **`pyproject.toml`**: make `valence` importable from the repo root for scripts and tests
  (e.g. `[tool.pytest.ini_options] pythonpath = ["."]` and an installable package), since `scripts/*.py`
  import `valence.*`. Locally I use a `.pth` file in the venv.
- **AST privacy test**: `scripts/spikes/bench_video.py` uses `cv2.imread` (read-only) on the bundled ultralytics
  sample images; reads are not writes, please don't flag them.
- **FactSheet with a hidden face**: the narrator accepts `face_only.label: null` and `face_only.face_valence: null`
  when `face_visible` is false. If the core uses a different marker (e.g. `"Unknown"`), tell me.

## Round 2 (after contract 1.0.0)

Everything in CONTRACT_RESPONSES.md is adopted on the edge side (frames_decoded via grab()/retrieve(),
frame_w/frame_h/mirrored=False, <=3 largest tracks >=25% height from `settings`, contract `obs_to_dict` in the
recorder, YAMNet AUDIO_CLIPS synchronous on the block's last 0.975 s, `note_transcript_purged()` per Vosk result).

### 1. libomp: torch + LightGBM in one process (live mode) — please act on this

Measured on the M3 Max (torch 2.14.1 bundles its own libomp; LightGBM 4.7.0 needs Homebrew `libomp`):

| setup (one process: YOLO on MPS + LightGBM predict/pred_contrib) | result |
|---|---|
| `import torch` first, LightGBM `num_threads=1` (or 0) | OK (5/5) |
| torch first + `KMP_DUPLICATE_LIB_OK=TRUE` only | OK |
| torch first + `OMP_NUM_THREADS=2` (the `.env.example` suggestion) | **segfault** |
| torch first, LightGBM default threads (None / -1 / 4) | **segfault** |
| **`import lightgbm` before torch** (any setting) | **segfault** (torch conv/bn fusion in YOLO load) |

Asks: (a) in live mode import `valence.edge.run` (it imports torch at module load) **before** anything imports
lightgbm, e.g. at the top of `valence/api/app.py` when `settings.mode == "live"`; (b) predict with
`num_threads=1` (pred_contrib on one row is <5 ms anyway); (c) drop `OMP_NUM_THREADS=2` from `.env.example`.
`valence.edge.run` logs a warning if it sees lightgbm already imported without torch.
Mac setup needs `brew install libomp` (LightGBM won't load without it); added to the Mac install notes.

### 2. Derived-buffer purge kinds

`frames_decoded == purged["video"]` only holds if just the decoded camera frames are purged as `"video"`.
Perception also purges copies it derives from a frame (RGB conversion, head crops, grayscale for lux) and from
audio (1 s concatenation, int16 PCM for Vosk, YAMNet clip). Please add `"video_derived"` and `"audio_derived"`
to `KINDS` (counted separately, same zero + weakref tracking). `valence/perception/gate.py:purge_derived`
already uses them when present and falls back to `"video"`/`"audio"` until then (so until then
purged["video"] > frames_decoded).

Also: `VideoObs.frames_decoded` is the capture's count when that frame was handed to perception; the capture
thread may have decoded (and purged as displaced) 1-2 more by the time you compare, so compare with a
tolerance of the in-flight slot (<= 2) or compare at stop.

### 3. `motion` NaN

Will emit NaN when undefined (after the posture module review lands).

### 4. URGENT: `.gitignore` `models/` hides `valence/models/`

The pattern `models/` matches every directory named `models`, so `valence/models/` (predictor) was never
committed: on a fresh checkout `tests/test_runtime.py`, `tests/test_api.py` and `valence/api/app.py` fail with
`ModuleNotFoundError: valence.models`. Please change it to `/models/` (and `/adapters/`), then `git add
valence/models` and push. (Same fix applied to the Mac's local exclude.)

## Round 3 (edge -> core): real-capture measurements for the simulator — negative control fails

`python -m valence.eval.negative_control data/real/capture_typing.jsonl` (served m2_lgbm@f69de179):
**Frustrated_Masked 49%, Exhausted 35%, Collaborative 16%, Focused 0%** -> gate fails (stress 0.49 > 0.20).
The detectors are *not* the cause: during 3 min of quiet typing there were 0 fist onsets, 0% slump, 0% head-in-hands.
The cause is sim-vs-real **scale**. Top supporting drivers over the replay (count of ticks):
`motion_mean_1m` (29 Frustrated + 23 Exhausted), `face_now` / `face_mean_5m` (25 + 20), `tod_*`, `lux_slope_1m`,
`db_slope_*`. A mid-replay tick: "nearly still (motion 0.04)" -> Exhausted with SHAP **+1.67** (largest by far).

Measured on the demo Mac (one person at a desk, `data/real/capture_*.jsonl`, `docs/BENCHMARKS.md`):

| quantity | typing | fists (cued) | baseline (still, upright) | sim params today |
|---|---|---|---|---|
| motion sw/s p10 / p50 / p90 | 0.014 / **0.027** / 0.048 | 0.014 / 0.031 / 0.091 | 0.011 / 0.014 / 0.031 | Focused 0.08, Exhausted 0.04, Frustrated 0.20, Happy 0.18 |
| face_valence p10 / p50 / p90 | 0.001 / **0.005** / 0.021 | -0.012 / 0.003 / 0.856 | -0.005 / 0.001 / 0.004 | Focused face_mu -0.15, Exhausted -0.10 |
| lux (uncalibrated) p50, quality | 316, 0.12 | 268, 0.20 | 233, 0.15 | lux_range 120-600 |
| dB SPL (offset +94, uncalibrated) p50 | 52 | 47 | 52 | Focused db_target 48 |
| fist onsets / min | **0.0** | 15/15 cued, 0 false | 0 | — |
| slump duty | **0%** | 0% | 0% | — |

Asks:
1. **Motion scale**: real motion is ~3x below the sim's per-state means. Either scale the sim (Focused ~0.03,
   Exhausted ~0.01-0.02, agitated states ~0.08-0.15; keyboard gaze measured 0.07, deliberate slumping 0.08-0.4)
   or drive the model from the per-session-relative feature (`motion_rel_sess`) rather than absolute motion.
2. **Face valence is baseline-relative** (contract item 4): a neutral person reads ~0.00, not -0.15. Center the sim's
   Focused/Exhausted face_mu near 0 (Focused "resting focus face" was measured at +0.005) and keep the forced-smile
   +0.45 relative to neutral.
3. **Lux/dB are uncalibrated until Step 7** (the user needs a phone lux/SPL app): lux noise from webcam auto-exposure
   and the person moving makes `lux_slope_1m` swing 2-80 lux/min with no real change; dB offset is the +94 default.
   Consider down-weighting `*_slope_1m`/`lux_std_1m` (or widening their "elevated" bands) and treating `lux_quality`
   < 0.3 as low-confidence light.
4. Re-run `make negative` after the retrain; the keyboard-gaze, slump and audio captures will follow (being re-recorded
   with the fixed slump rule and the Vosk fix).

### Round 3b: live API segfault fixed on the edge side (FYI, no action needed)

`VALENCE_MODE=live uvicorn valence.api.app:app` segfaulted at startup even with torch imported first: LightGBM
**parsing the model file** (`lgb.Booster(model_file=...)`, `valence/models/served.py:45`) runs a multi-threaded
OpenMP region, and `params={"num_threads": 1}` does not apply during parsing. Measured (3/3 each): plain -> segfault;
`params={"num_threads":1}` -> segfault; `OMP_NUM_THREADS=1` -> OK; `DYLD_LIBRARY_PATH=<site-packages>/torch/lib` -> OK
with a single libomp image in the process (LightGBM then shares torch's runtime). `valence.edge.run` (which the API
imports first in live mode) now does `os.environ.setdefault("OMP_NUM_THREADS", "1")` before importing torch, so the
API needs no change; the full app + served model + YOLO-in-a-thread path passes. If you ever need multi-threaded
LightGBM in the live process, launch with `DYLD_LIBRARY_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')")`.

Replay-mode note: `/api/replay_recording` shows the recording's `frames_decoded` against `frames_purged = 0`, which
reads as a governance mismatch on screen; consider hiding the comparison (or labelling it "recorded") in replay.
Venue note: background people >= 25% of frame height still get predictions; `VALENCE_MIN_BOX_HEIGHT=0.35` keeps only
people at the desk.
