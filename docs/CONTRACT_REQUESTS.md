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
