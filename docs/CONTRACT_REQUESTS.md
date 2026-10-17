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
