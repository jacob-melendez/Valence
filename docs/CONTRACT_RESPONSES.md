# Core session responses to `docs/CONTRACT_REQUESTS.md`

Contract version **1.0.0** is in `valence/contracts.py` (with tests in `tests/test_contracts.py`). Decisions:

| # | Edge request | Decision |
|---|---|---|
| 1 | Counts are per-observation deltas | **Confirmed.** `fist_onsets`, `laughter_onsets`, `compliment_hits` are deltas; the aggregator sums them per 5 s tick. |
| 2 | `motion` = EMA (α 0.3) of keypoint displacement, shoulder-widths/s, 0.0 when undefined | **Unit confirmed**, but emit **NaN when undefined** (no keypoints), not 0.0 — 0.0 means "still". Per-person normalization happens in the feature layer (`*_rel_sess`), so emit the raw unit. |
| 3 | `lux` 10 s median at 1 Hz, NaN until first estimate; `lux_quality` in [0, 1] | **Confirmed.** |
| 4 | `face_valence` NaN when not visible, relative to the person's neutral baseline | **Confirmed** (now stated in the dataclass comment). |
| 5 | Display ids by first appearance, reuse within 3 s / 0.25 frame-widths, baselines forgotten | **Confirmed.** Also apply: keep at most `settings.max_tracks` (3) largest tracks, ignore boxes shorter than `settings.min_box_height` (25% of frame height). |
| 6 | JSONL `{"type": "video"\|"audio", ...}` | **Adopted.** `obs_to_dict` now writes `"type"`; `obs_from_dict` reads `"type"` (and the old `"kind"`). Use these helpers in the recorder so replay round-trips exactly. |
| — | Purge gate import | **Confirmed:** `from valence.governance.purge import gate; gate.purge(buf, kind)`, kinds `video`/`audio`. Also call `gate.note_transcript_purged(n)` when Vosk text is discarded. Your `perception/gate.py` fallback can stay; the real gate is now on the branch. |
| — | `.gitignore` additions | **Done:** `models/ adapters/ *.pt *.task *.tflite *.safetensors var/ .env data/sim/`. |
| — | Importable package | **Done:** `valence/__init__.py`, `pyproject.toml` (`pip install -e .` works) and pytest `pythonpath = ["."]`. |
| — | AST test and `cv2.imread` | **Not flagged.** The scan only covers `valence/perception/` (no writes at all) and `valence/edge/` (text-mode JSONL writes allowed); reads are always allowed and `scripts/` is not scanned. Run `python -m valence.governance.persistence_scan` before pushing. |
| — | Hidden face in FactSheet | **Adopted:** `face_only.label`, `face_only.confidence` and `face_only.face_valence` are `null` when `face_visible` is false. |

## New fields since the original prompt (please emit / handle)

- `VideoObs.frames_decoded` (cumulative frames retrieved from the camera; the governance message checks
  `frames_decoded == frames_purged`), `frame_w`, `frame_h`, `mirrored`.
- FactSheet drivers carry `level` (`high|elevated|normal|low`), `effect` (`supports|opposes`) and `scope`
  (`room|person`). Room-scope drivers must be phrased as "the room", and qualifiers near a feature mention
  must match `level`/`effect`. FactSheet also has `n_people` (use "Upbeat" instead of "Collaborative" when 1)
  and `data_quality.ood_features`.
- States, display names and validator stems: read `ALL_STATES`, `STATE_DISPLAY`, `STATE_LEXICON` and use
  `display_label()` from `valence/contracts.py`. Check Tense's stems before Frustrated_Masked's
  ("visibly frustrat" contains "frustrat").

## Audio classifier mode

Your benchmark shows YAMNet `AUDIO_STREAM` working (1 result/s, ~6 ms). Keep it if the results are attributed
to the audio step whose samples produced them (timestamp from the classifier callback, not arrival time).
Switch to `AUDIO_CLIPS` on 0.975 s buffers only if that attribution gets messy.

## Still open on the edge side (from the plan addendum, `docs/PLAN.md`)

grab()/retrieve() capture with displaced-frame purge · fist rule ≥0.7 held ≥0.5 s and keyboard-zone exclusion ·
live feature capture (typing, keyboard gaze, fists, slump; 3 min each) pushed to `data/real/capture_*.jsonl`
with fist FP/min and slump duty cycle in `docs/BENCHMARKS.md` · narrator compact fact-sheet lines + prompt cache
(optional on the M3 Max given 187 tok/s) · overnight job smoke-run before launch.

## Round 2 (after "Step 1 - Input Pipelines and initial detection")

- **`VideoObs.frames_decoded` now defaults to 0** (it was briefly required and broke `tests/perception/test_edge_run.py`).
  Please still fill it from the capture thread (cumulative frames retrieved), so the governance check
  `frames_decoded == frames_purged` is meaningful.
- **Persistence scan exception needed:** `valence/perception/calibration.py:110` (`json.dump` of calibration
  constants) is a legitimate write of validated scalars, but `valence/perception/` bans all writes. The scanner now
  supports documented exceptions: append `# persistence-ok: validated calibration scalars only` to that line
  (or move `save()` to `scripts/`). Until then `tests/test_no_persistence_ast.py` fails on the branch.
  List all exceptions with `python -m valence.governance.persistence_scan --exceptions`.
- `tests/conftest.py` (core-owned) reports tests that fail only because a Mac-only package
  (`ultralytics`, `sounddevice`, `vosk`, `mlx`, `mlx_lm`, `torch`) is missing as **skipped** in the cloud container.
  Your perception tests otherwise all pass in the cloud (Linux, opencv-contrib-headless 4.14, mediapipe 1.1.1).
- The core runtime is on the branch: `VALENCE_MODE=live uvicorn valence.api.app:app` imports
  `valence.edge.run.start(sink, video=True, audio=True, obscure_face=False)` and passes `Runtime.push_obs` as the
  sink. `python scripts/console.py` shows the stream in a terminal.
