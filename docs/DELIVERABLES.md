# Valence: deliverables checklist (live)

This file tracks every hackathon deliverable for Valence (TrojanHacks @ USC 2026, AI Data Governance track).
Keep it current through the build. Scope, schedule and priorities come from `docs/PLAN.md`. Field names come
from `valence/contracts.py`.

**Snapshot:** 2026-10-17 23:00 UTC (≈ 4:00 PM PT Sat). Branch `claude/happy-ride-7w3ng3`, HEAD `9f31355`
(pushed), tag `slice-v0` on `0987043`. Update this file at every sync point and before each push.

## How to read this file

- `[x]`: the artifact exists in the working tree at the snapshot. It does **not** mean that tests pass or
  that results are good.
- `[ ]`: not done yet. The path shown is where the artifact will live. Paths come from `docs/PLAN.md`,
  except where marked *proposed* (the plan names no path).
- Line format: **Item**: Priority · Owner · When · `artifact path` · note.
- Tags: **uncommitted** = the file exists but git does not track it yet. **local only** = gitignored on
  purpose and rebuilt by a command. **this batch** = a doc written in the current docs batch.
- Priority = the tier in PLAN.md "Priority tiers & drop order". An item the plan does not tier gets the tier
  of the feature it supports.
- Owner: **CLOUD** = Linux session (ML core, runtime, API, most docs). **LOCAL** = Mac session (perception,
  edge, narrator). **YOU** = the builder (acting, calibration, Lovable frontend, rehearsals, submission).
- This file contains no result numbers. Hardware measurements live only in `docs/BENCHMARKS.md`. Evaluation
  numbers go to `reports/metrics.json` and then to `docs/RESULTS.md` once they are produced.

## Sync points and hours

H0 = plan approval ≈ 2:30 PM PT Saturday. H22 = submission ≈ 12:30 PM PT Sunday.

| Mark | Plan hour | ≈ PT | What lands |
|---|---|---|---|
| S1 | H0:45 | 3:15 PM Sat | Contracts, governance core + tests, docs batch, LOCAL/FRONTEND prompts |
| S2 | H2:30 | 5:00 PM | Vertical slice, tag `slice-v0` |
| | H2:30–4:00 | | Simulator v2 (untuned), scenarios, legacy generator |
| | H4:00–5:15 | | Features + parity / no-lookahead / window tests |
| | H5:15–6:00 | | Tune `noise_scale` on one criterion, freeze it, generate 300 sessions |
| S3 | H7:00 | 9:30 PM | Full ladder, calibration, clip bounds, override validated, fact sheets exported |
| | H7:00–8:00 | | Evaluate v1, proof 2/3 charts; live feature capture (LOCAL + YOU) |
| S4 | H9:15 | 11:45 PM | Explainer, template, narrator client, counterfactual; M2 served, tag `model-v1` |
| | H9:15–10:30 | | Capture-based sim update, negative control, live wiring, first live E2E |
| Gate | H10:30 | 1:00 AM Sun | Core-4 live E2E and negative control pass → P1 states; otherwise fix live and skip P1 |
| S5 | H12:00 | 2:30 AM | P1 states retrained, sheets re-exported; acted set |
| | H12:00–12:30 | | Acted-set scoring; LOCAL smoke-runs every overnight job, then launches them |
| | H12:30–15:30 | | M3, CIs, sweeps, chaos/e2e; GOVERNANCE, MODEL_CARD, RESULTS, DEMO; Devpost draft |
| | H15:30–17:00 | | Regenerate figures, finalize docs, 60-min soak, PERCEPTION/NARRATOR docs |
| Freeze | H17:00 | 7:30 AM | Feature freeze, tag `demo-v1` |
| | H17–20 | | Rehearsal fixes only; 2 rehearsals with Wi-Fi off (live + chaos) |
| | H20–21 | | Backup screen recording, 5 slides |
| Submit | H21–22 | 11:30 AM–12:30 PM | README check, Devpost, 2-min recording, submit |
| Venue | Sunday, before judging | | Recalibrate, start the pipeline ≥ 20 min early, reset between judge groups |

## Progress at the snapshot

| Section | Done | Open |
|---|---|---|
| Blockers | 4 | 1 |
| A. Code | 51 | 16 |
| B. Model & data artifacts | 4 | 13 |
| C. Judging proofs | 16 | 24 |
| D. Tests | 14 | 10 |
| E. Documentation | 11 | 7 |
| F. Demo & submission | 2 | 19 |
| G. Frontend outputs (contract) | 20 | 0 |

## Blockers

Resolved blockers stay listed so other sessions can see what changed. Open ones must be fixed before the
next push.

- [x] **`.gitignore` hid `valence/models/`**: P0 · CLOUD · fixed in `9f31355` · `.gitignore` · The pattern
  `models/` matched every directory named `models`, so `predictor.py`, `served.py` and `train_ladder.py` were
  missing from every commit before `9f31355` (LOCAL flagged it in CONTRACT_REQUESTS.md round 2 §4). The
  patterns are now anchored (`/models/`, `/adapters/`, `/var/`, `/data/sim/`) and `valence/models/` is
  committed.
- [x] **libomp load order in live mode**: P0 · CLOUD · fixed in `9f31355` · `valence/api/app.py`,
  `valence/models/served.py`, `.env.example` · On the demo Mac, importing LightGBM before torch, or setting
  `OMP_NUM_THREADS=2`, segfaults (measured by LOCAL, CONTRACT_REQUESTS.md round 2 §1). Live mode now imports
  `valence.edge.run` first, the served model predicts with `num_threads=1`, and `.env.example` no longer
  suggests `OMP_NUM_THREADS`.
- [x] **Derived-buffer purge kinds**: P0 · CLOUD · fixed in `9f31355` · `valence/governance/purge.py`
  (`KINDS` now includes `video_derived`, `audio_derived`) · Copies made inside perception are counted
  separately, so `frames_decoded == purged["video"]` can hold (compare with a tolerance of ≤ 2 in-flight
  frames, or at stop; CONTRACT_REQUESTS.md round 2 §2).
- [x] **Persistence scan clean**: P0 · LOCAL · fixed in `c009f50` · the calibration writer moved to
  `valence/edge/calibration_store.py`. `python -m valence.governance.persistence_scan` reported 0 violations
  at the snapshot.
- [ ] **Commit this docs batch and the figures**: P0 · CLOUD · next push · `docs/PRD.md`,
  `docs/ARCHITECTURE.md`, `docs/DATASET.md`, `docs/API.md`, `docs/DELIVERABLES.md`, `README.md`,
  `reports/figs/`

## A. Code

### A1. Contract, configuration, packaging

- [x] **Shared contract v1.0.0** (states, `display_label()`, `PersonObs`/`VideoObs`/`AudioObs`,
  `TickRecord`, `FactSheet`, WS messages, `json_schemas()`): P0 · CLOUD · S1 · `valence/contracts.py`
- [x] **Runtime settings** (`VALENCE_*` overrides) and the offline env template: P0 · CLOUD · S1 ·
  `valence/config.py`, `.env.example`
- [x] **Packaging and pins**: P0 · CLOUD · S1 · `pyproject.toml`, `requirements.txt`, `requirements.lock`
- [x] **Makefile** (`test`, `scan`, `replay`, `live`, `console`, `train`, `eval`, `facts`, `negative`,
  `smoke`): P0 · CLOUD · S2 → S3 · `Makefile`
- [x] **Mac dependency set** (needs `brew install libomp` for LightGBM): P0 · LOCAL · S1 (Step 0) ·
  `requirements-mac.txt`
- [x] **Model downloads**: P0 · LOCAL · S1 (Step 0) · `scripts/download_models.sh`

### A2. Governance

- [x] **PurgeGate**: P0 · CLOUD · S1 · `valence/governance/purge.py` · id→`weakref.ref` callback dict,
  in-place `fill(0)` of the owning buffer, `end_tick()` counts survivors as PURGE_VIOLATION, read-only
  buffers counted as unzeroable, `note_transcript_purged()`, separate `video_derived` / `audio_derived`
  kinds for copies made inside perception.
- [x] **Perception gate import point**: `purge()` for raw capture buffers, `purge_derived()` for copies made
  inside perception: P0 · LOCAL · S1 → Step 2 · `valence/perception/gate.py`
- [x] **Raw-media log filter**: P0 · CLOUD · S1 · `valence/governance/logfilter.py`
- [x] **Persistence AST scan** with documented `# persistence-ok:` exceptions: P0 · CLOUD · S1 ·
  `valence/governance/persistence_scan.py`
- [x] **Calibration writer outside perception** (perception never writes): P1 · LOCAL · Step 2 ·
  `valence/edge/calibration_store.py`
- [x] **Audit hash chain + session TTL**: P1 · CLOUD · S1 · `valence/governance/audit.py` · stores labels,
  SHAP summaries and operator actions only. Refuses strings over 200 characters. Deletes the session file on
  close and removes files older than `audit_ttl_s` at start.
- [x] **Audit wiring** for operator actions (replay, live, reset, sim overlays, seed, calibrate, mode,
  self-report): P1 · CLOUD · S2 · `valence/runtime/engine.py`, `valence/api/app.py`
- [x] **Capture**: `grab()` on every frame, `retrieve()` at the perception rate, `frames_decoded` counted,
  displaced slot frames purged; `VideoObs` carries `frames_decoded`, `frame_w`, `frame_h`, `mirrored`: P0 ·
  LOCAL · Step 2 · `valence/edge/capture.py`, `valence/edge/run.py`
- [ ] **`audit_verify.py` CLI**: P1 · CLOUD · H12:30–15:30 · `scripts/audit_verify.py` · the API equivalent
  `GET /api/governance/verify` exists.

### A3. Runtime and API

- [x] **Event bus** (bounded per-client queues, drop-oldest): P0 · CLOUD · S2 · `valence/runtime/bus.py`
- [x] **Aggregator** (room + person slots, near-field gate at room median + 6 dB, break clock, calibration,
  simulated injections): P0 · CLOUD · S2 · `valence/runtime/aggregator.py`
- [x] **Rolling window** (180 ticks, room slot, a new person inherits room history): P0 · CLOUD · S2 ·
  `valence/features/window.py`
- [x] **Ticker, scenario replay and live sources, overlays, seeding, room-only**: P0 · CLOUD · S2 ·
  `valence/runtime/engine.py` · The plan named `ticker.py` and `replay.py`; both are in `engine.py`.
- [x] **Features-only JSONL replay** (live feeder for the camera/mic fallback, deterministic offline replay
  for the negative control), `GET /api/recordings`, `POST /api/replay_recording/{name}`: P0 · CLOUD ·
  H9:15–10:30 · `valence/runtime/recording.py`, `valence/api/app.py`
- [x] **Predictor interface**: EMA + hysteresis(2), warm-up, override hysteresis and the 2-min block, rule
  stub: P0 · CLOUD · S2 · `valence/models/predictor.py`
- [x] **FastAPI REST + `/ws/events`**: every route listed in PLAN.md, including `demo/seed`, `demo/reset`,
  `calibrate/room`, `mode`, `label`, `sim/event`, `sim/lux` and the counterfactual, plus `POST /api/live`
  and the recording routes: P0 · CLOUD · S2 → H9:15–10:30 · `valence/api/app.py`
- [x] **Terminal console**: P0 · CLOUD · S2 · `scripts/console.py`
- [x] **Fact-sheet export for the narrator** (`make facts`): P1 · CLOUD · S3 · `scripts/export_fact_sheets.py`
  · output listed in B.
- [ ] **Hybrid mode** (live sensors where available, the rest simulated and flagged). The contract allows
  `mode: "hybrid"`, but the engine has no hybrid source yet: P0 · CLOUD · H9:15–10:30 ·
  `valence/runtime/engine.py`
- [ ] **First live end-to-end run on the Mac** (edge → runtime → WS → Lovable): P0 · LOCAL + YOU ·
  H9:15–10:30 · `make live`
- [ ] **`e2e_check.py`** (pre-demo replay check, `--speed 20`): P0 · CLOUD · H12:30–15:30 ·
  `scripts/e2e_check.py`
- [ ] **`mem_soak.py`** (memory + governance counters over 60 min): P1 · CLOUD · H12:30–15:30 ·
  `scripts/mem_soak.py`

### A4. Simulator and scenarios

- [x] **Simulator v2** (semi-Markov hidden state, lagged room context, per-session mic offset, nuisance
  terms, `noise_scale`, P1 state parameters): P0 · CLOUD · H2:30–4:00 · `valence/sim/generator.py`,
  `valence/sim/params.py`
- [x] **Legacy generator** + `to_legacy_csv()`: P0 · CLOUD · H2:30–4:00 · `valence/sim/legacy.py`,
  `valence/sim/generator.py`
- [x] **Scenario loader + 8 scenarios** (focused, frustrated_masked, exhausted, collaborative_happy,
  overwhelmed, tense, context_blindness, full_demo): P0 · CLOUD · H2:30–4:00 · `valence/sim/scenarios.py`,
  `scenarios/*.yaml`
- [ ] **Difficulty tuning on one criterion** (M2 macro-F1 ≤ .92 plus a live-vs-sim feature variance match),
  with the frozen value recorded: P0 · CLOUD · H5:15–6:00 · `valence/sim/tune_difficulty.py` · The current
  artifacts were trained at `noise_scale` 1.0 (`artifacts/model_meta.json`). No tuning script exists yet.
- [ ] **Capture-informed false-positive rates and motion/slump scales**: P0 · CLOUD · H9:15–10:30 ·
  `valence/sim/params.py` · needs the live feature capture (B).

### A5. Features

- [x] **Sensor spec, tick channels, layer and scope rules**: P0 · CLOUD · H4:00–5:15 ·
  `valence/features/spec.py`
- [x] **One `compute` for offline and live** (NaN-native): P0 · CLOUD · H4:00–5:15 ·
  `valence/features/compute.py`
- [x] **Offline dataset builder** (random-length windows): P0 · CLOUD · H4:00–5:15 ·
  `valence/features/build_dataset.py`

### A6. Models, explanation, evaluation

- [x] **Ladder trainer** (m0, m0t, m1, m2_noface, m2_noabsdb, m2_notime, m2; 70/15/15 session split;
  early-stop and calibration halves; temperature scaling; clip bounds): P0 · CLOUD · S3 ·
  `valence/models/train_ladder.py`
- [x] **Served predictor** (M2 with M0 in shadow, clipping + `ood_features`, `pred_contrib` layers, raw
  mask-override condition): P0 · CLOUD · S4 · `valence/models/served.py`
- [x] **MaskOverrideDetector**: P0 · CLOUD · S3 · `valence/models/served.py` (condition) +
  `valence/models/predictor.py` (hysteresis) · The implemented rule differs from PLAN.md: M0 ≥ 0.45 (plan
  0.55), M0 must read `Collaborative_Happy` while M2 does not, and "on at ≥ 3 of the last 6 face-visible
  ticks, off at ≤ 1" (plan: 3 consecutive). PRD §2.8, ARCHITECTURE §9.4 and API §3.7 now state both rules;
  choosing the final one is open.
- [x] **Counterfactual** (re-run M2 with one feature changed) and
  `GET /api/explain/{event_id}/counterfactual`: P1 · CLOUD · S4 · `valence/models/served.py` (the plan named
  `counterfactual.py`), `valence/api/app.py`
- [ ] **M3 GRU(64) + attention benchmark**: P1 · CLOUD · H12:30–15:30 · `valence/models/temporal.py` · the
  file appeared (uncommitted) after the snapshot; no results yet.
- [x] **Feature descriptions** (display, unit, `level`, `scope`), layer builder, FactSheet builder: P0 ·
  CLOUD · S4 · `valence/explain/describe.py`, `valence/explain/explainer.py`
- [x] **Instant template text**: P0 · CLOUD · S4 · `valence/explain/template.py`
- [x] **Narrator client** (latest-only per person, triggers, cache, never blocks the tick): P1 · CLOUD · S4
  · `valence/explain/narrator_client.py`
- [x] **Evaluation for proofs 2 and 3** (`make eval`): P0 · CLOUD · H7:00–8:00 · `valence/eval/evaluate.py`,
  `valence/eval/style.py` · outputs are listed in C2/C3.
- [x] **Negative control and acted-set scoring** (`make negative`; `--expect` labels): P0 (negative control)
  / P1 (acted set) · CLOUD · H9:15–10:30 and H12:00–12:30 · `valence/eval/negative_control.py` · The plan's
  separate `score_acted.py` is folded into this script.
- [ ] **P1 states** (Frustrated_Masked becomes mask-only, Tense takes unmasked frustration), retrained
  with `make train STATES=all`: P1 · CLOUD · S5 · `artifacts/` · simulator parameters for P1 exist; no
  models yet.

### A7. Perception and edge (LOCAL)

- [x] **Hardware spikes and self-test**: P0 · LOCAL · H0:45–2:30 (Step 1) · `scripts/spikes/bench_video.py`,
  `bench_audio.py`, `bench_llm.py`, `selftest.py`
- [x] **Video perceiver** (YOLO26n-pose + ByteTrack, display ids, full-frame faces with head-crop fallback,
  blendshape face valence, posture with `motion` NaN when undefined, lux): P0 (boxes/IDs, face valence,
  fists) / P1 (slump, head-in-hands, lux) · LOCAL · H2:30–6:00 (Step 2) · `valence/perception/video.py`,
  `ids.py`, `face_valence.py`, `hands.py`, `posture.py`, `lux.py`, `regions.py`, `bytetrack_valence.yaml`
- [x] **Fist rule** (score ≥ 0.7 held ≥ 0.5 s, hands in the bottom 20% keyboard zone excluded): P0 · LOCAL ·
  Step 2 · `valence/perception/debounce.py`, `valence/perception/hands.py`, `valence/perception/video.py`
- [x] **Person filter** (≤ 3 largest tracks with box height ≥ 25%, from `settings.max_tracks` and
  `settings.min_box_height`): P0 · LOCAL · Step 2 · `valence/perception/video.py`
- [x] **Obscure-face mode** (MediaPipe never sees a face): P1 · LOCAL · Step 2 ·
  `valence/perception/video.py` (`obscure_face`)
- [x] **Audio perceiver** (calibrated Leq dB; YAMNet in AUDIO_CLIPS mode on the block's last 0.975 s; Vosk
  restricted-grammar compliments; transcript purges counted): P0 (dB) / P1 (YAMNet, Vosk) · LOCAL ·
  H5:15–7:00 (Step 3) · `valence/perception/audio.py`, `valence/perception/audio_features.py`
- [x] **Calibration constants + interactive SPL/lux scripts**: P1 · LOCAL · H8:00–9:15 ·
  `valence/perception/calibration.py`, `scripts/calibrate_spl.py`, `scripts/calibrate_lux.py`
- [x] **Capture threads** (camera 1-slot handoff, mic queue, permission-failure symptoms): P0 · LOCAL ·
  Step 4 · `valence/edge/capture.py`
- [x] **Edge runner** `start(sink, *, video, audio, obscure_face) -> stop` with `--print` / `--jsonl`
  (contract `obs_to_dict`, offline env defaults, imports torch before LightGBM): P0 · LOCAL · H9:15–10:30
  (Step 4) · `valence/edge/run.py`

### A8. Narrator (LOCAL)

- [x] **Fact-sheet schema, lexicon, validator** (numbers ⊆ sheet, state named, top drivers, mask language,
  banned terms, length): P0 · LOCAL · S4 · `valence/narrator/schema.py`, `lexicon.py`, `validator.py`
- [x] **Template fallback + synthetic fact sheets**: P0 · LOCAL · S4 · `valence/narrator/templates.py`,
  `fake_fact_sheets.py`
- [x] **Narrator service `:8081`, Stage A** (`POST /narrate`, `GET /health`, `--template-only`; up to 3
  validated attempts within the 2.5 s deadline, then the template; prompt cache; one MLX worker thread;
  compact `key: value` sheet lines + few-shots; default model Qwen3-1.7B-4bit): P1 · LOCAL · H8:00–9:15
  (Step 5) · `valence/narrator/service.py`, `engine.py`, `prompt.py` · committed in `c009f50` as "drafts,
  under review". The plan named `prompts.py`.
- [x] **Offline narrator evaluation** (pass rate, hallucinated numbers, false mask language, banned terms,
  latency; LLM or template mode; optional adapter): P1 · LOCAL · H12:30–17:00 · `valence/narrator/evaluate.py`
  · produces the C4 table.
- [ ] **Validator reads the v1.0.0 sheet fields** (addendum 8): qualifiers within 6 tokens must match
  `level`/`effect`; `scope: room` drivers phrased as "the room"; "Upbeat" when `n_people` == 1;
  `data_quality.ood_features`; cached text re-validated against the current sheet: P0 · LOCAL · S4 ·
  `valence/narrator/validator.py` · `prompt.py` renders `level`/`effect`/`scope`, but the validator has no
  such checks yet.
- [ ] **Grammar-generated targets + LoRA config** (no self-distillation): P1 · LOCAL · S5 ·
  `valence/narrator/grammar.py`, `valence/narrator/build_data.py`, `valence/narrator/lora.yaml`
- [ ] **Overnight runner** (`set -uo pipefail`, one status file per job, smoke-run first,
  `caffeinate -dims`): P1 · LOCAL · H12:00–12:30 · `scripts/overnight.sh`
- [ ] **Narrator ship flag**: ship the LoRA only if its pass rate ≥ Stage A and p95 ≤ 2.5 s with YOLO
  running: P1 · LOCAL · H15:30–17:00 · narrator config (path TBD)

### A9. P2 (only if time)

- [ ] **Perception subprocess** (`spawn`, emits feature dicts only), only if perception runs below 8 Hz:
  P2 · LOCAL · as needed · path TBD
- [ ] **EmotiEffLib face model**: P2 · LOCAL · as needed · path TBD
- [ ] **Attention slide** (M3 attention weights): P2 · CLOUD · H12:30–15:30 · `reports/figs/` (*proposed*)
- Self-distillation for narrator data: cut by the plan (grammar-only targets). Not tracked.

## B. Model & data artifacts

- [x] **Served models + metadata** (core 4 states; 300 sessions, seed 7, `noise_scale` 1.0 per
  `model_meta.json`): P0 · CLOUD · S3 · `artifacts/m0.txt`, `artifacts/m2.txt`, `artifacts/model_meta.json`
  (temperatures, feature sets, states, split seed) · committed in `1c2f393`. Retrain after tuning (A4) and
  the capture update.
- [x] **Clip bounds** (train p1–p99): P0 · CLOUD · S3 · `artifacts/clip_bounds.json` · committed.
- [x] **Rest of the ladder**: P0 (M1, M2-noface) / P1 (M0t, M2-noabsdB, M2-notime) · CLOUD · S3 ·
  `artifacts/m0t.txt`, `m1.txt`, `m2_noface.txt`, `m2_noabsdb.txt`, `m2_notime.txt` · **local only**;
  `make train` rebuilds them.
- [x] **Evaluation figures, v1**: P0 · CLOUD · H7:00–8:00 · `reports/figs/` · **uncommitted** · details in
  C2/C3.
- [ ] **Frozen feature order as its own file**: P0 · CLOUD · S3 · `artifacts/feature_spec.json` · The order
  is currently stored as `feature_names` in `model_meta.json` (114 features). PRD, ARCHITECTURE and API now
  point to `model_meta.json` and note the gap; write the file or update PLAN.md.
- [ ] **Simulator datasets, split by session**: P0 · CLOUD · H5:15–6:00 · `data/sim/` (gitignored by design)
  · Today the trainer regenerates sessions in memory from `seed` + `sessions`, and `evaluate.py` rebuilds
  the held-out split the same way. No dataset or split manifest is on disk (*proposed*:
  `artifacts/splits.json`).
- [ ] **Evaluation metrics**: P0 · CLOUD · H7:00–8:00 · `reports/metrics.json` · TBD, produced by
  `valence/eval/evaluate.py`
- [ ] **Golden features-only recording** for the replay fallback: P0 · LOCAL · H9:15–10:30 ·
  `data/real/golden_*.jsonl` (*proposed*) · the replay code exists (`valence/runtime/recording.py`).
- [ ] **Live feature capture**, 3 min each of quiet typing, keyboard gaze, fist clenching and slumping:
  P0 · LOCAL + YOU · H7:00–8:00 · `data/real/capture_*.jsonl`, plus fist FP/min while typing, slump duty
  cycle and motion distribution in `docs/BENCHMARKS.md`
- [ ] **Sensor calibration constants** (SPL offset, lux model): P1 · LOCAL + YOU · H8:00–9:15 ·
  `data/real/calibration.json`
- [ ] **Self-report labels** every 10 min: P1 · YOU · from H9:15 · `POST /api/label` →
  `var/labels/<session_id>.jsonl` (gitignored; stores the label plus the latest numeric tick only)
- [ ] **P1 six-state models**: P1 · CLOUD · S5 · `artifacts/` (`state_set` = `all`)
- [ ] **M3 benchmark results**: P1 · CLOUD · H12:30–15:30 · `reports/metrics.json`
- [ ] **Narrator fact sheets from M2 SHAP** (`make facts`): P1 · CLOUD · S3 ·
  `data/narrator/fact_sheets_sim.jsonl`
- [ ] **Narrator train/valid/test** (1500/150/300, balanced states × {face agrees, disagrees, hidden}):
  P1 · LOCAL · S5 · `data/narrator/{train,valid,test}.jsonl`
- [ ] **LoRA adapter** (rank 8, scale 20, 16 layers, 600 iters): P1 · LOCAL · H12:30–15:30 · `adapters/`
  (gitignored)
- [ ] **Acted set**, 6 scenarios × 3 min, features only: P1 · LOCAL + YOU · S5 · `data/real/` (file names
  TBD)

## C. Judging proofs

### C1. Data minimization

- [x] **PurgeGate code for the slide** (`purge()`, `_on_free`, `end_tick()`): P0 · CLOUD · S1 ·
  `valence/governance/purge.py`
- [x] **Purge tests**: zeroing + finalizer, leaked reference → violation, a view zeroes its owner, read-only
  counted, cycles collected before a violation, held detector results leak the frame (the fake-Results
  test), transcript counter: P0 · CLOUD · S1 · `tests/test_purge.py`
- [x] **AST no-persistence test and log-filter test**: P0 · CLOUD · S1 · `tests/test_no_persistence_ast.py`,
  `tests/test_logfilter.py`
- [x] **Perceiver privacy probes**: no frame retained after `process()`, audio block purged, obscure-face:
  P0 · LOCAL · Step 2 · `tests/perception/test_perceiver_privacy.py` · `docs/BENCHMARKS.md` notes that the
  perceiver clears Ultralytics' retained input references after each call.
- [ ] **`pytest -v` output** of test_purge, test_no_persistence_ast and test_logfilter, for the slide:
  P0 · YOU · H20–21 · slides
- [ ] **`frames_decoded == frames_purged` shown live** in `governance` (tolerance ≤ 2 in-flight frames):
  P0 · LOCAL + YOU · H9:15–10:30 · `valence/edge/capture.py`, `valence/governance/purge.py` · both sides are
  implemented (capture counts decoded frames; derived copies use their own kinds). Show it on the Mac.
- [ ] **`/api/governance/stats` after a 60-min soak** (0 violations, 0 `live_raw_buffers`, `retention_s`):
  P1 · LOCAL · H15:30–17:00 · `reports/soak_governance.json` (*proposed*)
- [ ] **`memory_flat.png`**: P1 · CLOUD (script) + LOCAL (run) · H15:30–17:00 ·
  `reports/figs/memory_flat.png` via `scripts/mem_soak.py`
- [ ] **`audit_verify.py` prints OK**: P1 · CLOUD · H12:30–15:30 · `scripts/audit_verify.py`
- [ ] **`lsof` during a live run shows only the audit log open for writing**: P1 · YOU · H17–20 ·
  recorded in `docs/DEMO.md`
- [ ] **Demo with Wi-Fi off** (`status.offline` is true): P0 · YOU · H17–20 · rehearsal notes
- [ ] **Honest limits**: `del` is not erasure; GPU allocator caches, AVFoundation buffers and `mp.Image`
  copies cannot be zeroed; the Vosk grammar means no free transcript; keypoints never leave perception; the
  audit log stores labels + SHAP summaries only and is deleted at session end: P1 · CLOUD · H12:30–15:30 ·
  `docs/GOVERNANCE.md`

### C2. Context vs face

Every figure here is v1: core 4 states, `noise_scale` 1.0, no M3. Regenerate them at H15:30–17:00.

- [x] **`accuracy_vs_history.png`** (macro-F1 at 0/1/2/5/10/15 min, session-bootstrap 95% CIs): P0 · CLOUD
  · H7:00–8:00 · `reports/figs/accuracy_vs_history.png` · M3 is added when it exists.
- [x] **Confusion M0 vs M2**: P0 · CLOUD · H7:00–8:00 · `reports/figs/confusion_m0_vs_m2.png`
- [x] **`acc_vs_episode_age.png`**: P0 · CLOUD · H7:00–8:00 · `reports/figs/acc_vs_episode_age.png`
- [x] **`reliability.png`**: P0 · CLOUD · H7:00–8:00 · `reports/figs/reliability.png`
- [ ] **Headline numbers**: Frustrated_Masked recall, and the "Focused paradox" (M0 Focused→Frustrated/Tense
  confusion vs M2): P0 · CLOUD · H7:00–8:00 · `reports/metrics.json` → `docs/RESULTS.md` · TBD, produced
  by `valence/eval/evaluate.py`
- [ ] **ECE before/after temperature scaling**: P0 · CLOUD · H7:00–8:00 · `reports/metrics.json` · TBD,
  produced by `valence/eval/evaluate.py`
- [ ] **Sensitivity table** (`noise_scale` ×0.5 / ×1 / ×1.5): P0 · CLOUD · H12:30–15:30 · `docs/RESULTS.md`
  · TBD, produced by `valence/sim/tune_difficulty.py`
- [ ] **Legacy vs v2 slide** (the original mock is separable from context alone): P0 · CLOUD + YOU ·
  H7:00–8:00, slide at H20–21 · `reports/metrics.json` (legacy result from `evaluate.py`) → slide
- [x] **`context_blindness` scenario** (same head-in-hands pose in two contexts, no clock reset): P0 · CLOUD
  · H2:30–4:00 · `scenarios/context_blindness.yaml`
- [ ] **`context_blindness` replay shows two labels** with M2 served: P0 · CLOUD · S4 ·
  `POST /api/replay/context_blindness` · `tests/test_served.py` checks other scenarios, not this one.
- [ ] **Live mask flip** with `mask_override`: P0 · YOU + LOCAL · H17–20, then the venue · live demo
- [ ] **Counterfactual demo** ("still Frustrated at 0.5 h because of fists + noise"): P1 · CLOUD · S4 ·
  `GET /api/explain/{event_id}/counterfactual?feature=hours_since_break&value=0.5`
- [ ] **M2-notime and M2-noabsdB ablations reported**: P1 · CLOUD · H12:30–15:30 · `reports/metrics.json` ·
  the models exist locally; the numbers are TBD.
- [ ] **Claim wording used everywhere**: "Under explicit, published behavioral assumptions, a temporal
  context model beats a face snapshot; in-the-wild validation is future work.": P0 · CLOUD · H12:30–15:30 ·
  `docs/RESULTS.md`, `docs/MODEL_CARD.md`, slides

### C3. Anti-bias, narrowed to "decision stability when face evidence is missing or degraded"

- [x] **Robustness curves that degrade only the face channel** (lux feature held fixed) for M0, M0t, M2 and
  M2-noface: P0 · CLOUD · H7:00–8:00 · `reports/figs/robustness_face_dropout.png` · v1 (core 4,
  `noise_scale` 1.0), **uncommitted**
- [x] **Lux shift reported separately**: P0 · CLOUD · H7:00–8:00 · `reports/figs/robustness_lux.png` · v1,
  **uncommitted**
- [ ] **M2 vs M2-noface gap table**: P0 · CLOUD · H7:00–8:00 · `reports/metrics.json` → `docs/RESULTS.md` ·
  the target from PLAN.md verification is M2-noface ≥ M2 − .05 (core 4).
- [ ] **Live obscure-face**: the label holds and the narration switches to the occlusion clause: P1 ·
  LOCAL + YOU · H17–20 · `obscure_face=True` (code in A7)
- [ ] **Live face-detection rate at 3 light levels**: P1 · LOCAL + YOU · S5 · `docs/BENCHMARKS.md` (or
  `docs/PERCEPTION.md`)
- [ ] **MediaPipe model-card fairness evaluations cited**: P1 · CLOUD · H12:30–15:30 · `docs/MODEL_CARD.md`
- [x] **Demographic validation listed as an explicit gap**: P0 · CLOUD · S1 · `docs/PRD.md` §1.3 (this
  batch) · repeat it in `docs/MODEL_CARD.md`.

### C4. Narrator table

- [ ] **Stage A vs LoRA vs template**: first-try pass rate, invented numbers, wrong qualifiers, false mask
  claims, demographic mentions (target 0), p50/p95 latency with YOLO running, distinct-2 / self-BLEU: P1 ·
  LOCAL · H12:30–17:00 · `docs/NARRATOR.md` · TBD, produced by `valence/narrator/evaluate.py` (A/B job in
  `scripts/overnight.sh`)
- [x] **Generation-speed spike without YOLO load** (context for the table, not the table itself): P1 · LOCAL
  · Step 1 · `docs/BENCHMARKS.md` (Narrator LLM section)

### C5. Governance framing

- [x] **EU AI Act Art. 5(1)(f) framing**: emotion recognition in workplaces and education is prohibited
  regardless of consent (medical/safety carve-outs only). Valence is a research prototype for consented
  self-monitoring outside those settings. Pseudonymous behavioral data is still personal data, hence TTL +
  session deletion: P0 · CLOUD · S1 · `docs/PRD.md` §1.4 (this batch)
- [x] **Room-only compliance configuration** (one aggregate room label, no per-person inference): P1 ·
  CLOUD · H9:15–10:30 · `POST /api/mode` with `{"room_only": true}`, `valence/runtime/engine.py`,
  `tests/test_runtime.py::test_overlays_and_room_only`
- [x] **TTL + session deletion implemented**: P1 · CLOUD · S1 · `valence/governance/audit.py`
- [ ] **Same framing in the model card**: P1 · CLOUD · H12:30–15:30 · `docs/MODEL_CARD.md`
- [ ] **Pitch slide**: P0 · YOU · H20–21 · slides (order in `docs/PRD.md` §4.5)

## D. Tests

### D1. Present (a file exists; pass/fail is whatever `make test` reports)

- [x] `tests/test_purge.py`: P0 · CLOUD · S1
- [x] `tests/test_no_persistence_ast.py`: P0 · CLOUD · S1
- [x] `tests/test_logfilter.py`: P0 · CLOUD · S1
- [x] `tests/test_audit.py` (chain verifies and detects tampering, payload refusal, delete on close + TTL):
  P1 · CLOUD · S1
- [x] `tests/test_contracts.py` (state consistency, display rules, NaN round-trip, `PersonObs` validation,
  schema export, frontend prediction example): P0 · CLOUD · S1
- [x] `tests/test_api.py` (REST endpoints, WS stream from `hello` to prediction/governance/state_vector,
  history): P0 · CLOUD · S2
- [x] `tests/test_runtime.py` (replay scenario end to end, overlays + room-only, seeding, room slot survives
  a person reset, near-field gate, break clock, offline recording replay): P0 · CLOUD · S2
- [x] `tests/test_feature_parity.py` (feature names/sets, live window == offline builder, no-lookahead,
  short/missing windows): P0 · CLOUD · H4:00–5:15
- [x] `tests/test_served.py` (SHAP additivity, override fires on frustrated_masked and not elsewhere,
  scripted scenarios track their script, counterfactual changes only the named feature): P0 · CLOUD · S3–S4
- [x] `tests/conftest.py` (tests that fail only because a Mac-only package is missing are reported as
  skipped in the cloud): P0 · CLOUD · S2
- [x] `tests/perception/`, 11 files (audio_features, calibration, debounce, edge_run, face_valence, hands,
  ids, lux, perceiver_privacy, posture, regions): P0 · LOCAL · Steps 2–4
- [x] `tests/narrator/`, 6 files (validator, templates, prompt, engine, service, fake_fact_sheets): P0 ·
  LOCAL · S4
- [x] **torch + LightGBM in one process**, measured matrix of import orders and thread settings: P0 · LOCAL
  · S1 (Step 0) · `docs/CONTRACT_REQUESTS.md` round 2 §1 · not yet copied into `docs/BENCHMARKS.md`.
- [x] **Persistence scan run**: P0 · CLOUD + LOCAL · every push · `python -m valence.governance.persistence_scan`
  (0 violations at the snapshot)

### D2. To build

- [ ] **test_window** (900 s eviction, 4 writer threads, room-slot inheritance): P0 · CLOUD · H4:00–5:15 ·
  `tests/test_window.py` · the room-slot case is partly covered in `test_runtime.py`.
- [ ] **Model gates (targets)**: no session in two splits; M2 macro-F1 ≤ .92 and ≥ .85; M2-noface ≥ M2 − .05
  (core 4); ECE drops; at 15 min M2 > M1 > M0 with non-overlapping CIs: P0 · CLOUD · S3 ·
  `tests/test_models.py` (*proposed*), reading `reports/metrics.json`
- [ ] **Simulator checks**: dwell/transition stats match the config; same seed → same hash: P0 · CLOUD ·
  H2:30–4:00 · `tests/test_sim.py` (*proposed*)
- [ ] **test_e2e_replay with M2**: correct label on ≥ 80% of ticks, override fires, 0 violations, tick
  < 200 ms; chaos → `status` event + fallback while ticks continue: P0 · CLOUD · H12:30–15:30 ·
  `tests/test_e2e_replay.py`
- [ ] **API contract hardening**: every WS message validated against its JSON Schema; `explanation` carries
  its prediction's `event_id`; per-client queues stay bounded with no client connected: P0 · CLOUD · S2 ·
  extend `tests/test_contracts.py`, `tests/test_api.py`
- [ ] **Negative control gate run**: the typing-capture replay predicts Frustrated/Tense on ≤ 20% of ticks;
  ≥ 90% of live features fall inside the clip bounds: P0 · CLOUD · H10:30 gate · `make negative` · needs
  the live capture (B).
- [ ] **Narrator adversarial set**: "91%", "she", "depressed", "high noise (45 dB)" (qualifier vs `level`),
  mask language when the face agrees, stale cached numbers: P0 · LOCAL · S4 · `tests/narrator/` ·
  unknown numbers, banned terms and mask terms are covered in `test_validator.py`; the qualifier and
  stale-cache cases are not.
- [ ] **live_probe**: 5 fists → 5 ± 1; 3 min of typing → fist FP rate logged; a 10 s slump sets the flag;
  covered camera → `face_visible` 0; "great job" ×3 → 3; dB within ±3 of a phone meter: P0 · LOCAL ·
  H9:15–10:30 · `scripts/live_probe.py` (*proposed*)
- [ ] **Wi-Fi-off smoke** at H1 and H14: P0 · LOCAL · H1, H14 · record in `docs/BENCHMARKS.md`
- [ ] **Live 60-min run**: ≥ 8 Hz, 0 violations, flat memory, narrator fallback rate logged: P1 · LOCAL ·
  H15:30–17:00 · `docs/PERCEPTION.md`

## E. Documentation

- [x] `docs/PLAN.md` (approved execution plan): P0 · CLOUD · H0
- [x] `docs/PRD.md` (v2): P0 · CLOUD · S1 · this batch
- [x] `docs/ARCHITECTURE.md` (replaces the Code Architecture note): P0 · CLOUD · S1 · this batch
- [x] `docs/DATASET.md` (replaces the mock dataset script; generator v2 + parameter table, legacy
  compatibility): P0 · CLOUD · S1 · this batch
- [x] `docs/API.md` (REST, WebSocket, narrator contract): P0 · CLOUD · S1 · this batch
- [x] `docs/DELIVERABLES.md` (this file; update at every sync point): P0 · CLOUD · S1 · this batch
- [x] `README.md`: P0 · CLOUD · S1, final check by YOU at H21–22 · this batch
- [x] `docs/LOCAL_SESSION_PROMPT.md` (v1 prompt + ⚑ addendum): P0 · CLOUD · S1
- [x] `docs/FRONTEND_PROMPT.md` (v2): P0 · CLOUD · S1
- [x] `docs/CONTRACT_REQUESTS.md` (LOCAL) and `docs/CONTRACT_RESPONSES.md` (CLOUD): P0 · LOCAL + CLOUD ·
  S1–S2 · The edge's round-2 asks (libomp, derived purge kinds, `.gitignore`) were applied in `9f31355`;
  CONTRACT_RESPONSES.md does not record that yet.
- [x] `docs/BENCHMARKS.md` (Apple M3 Max, 36 GB): P0 · LOCAL · Steps 0–1 · Still open: live-camera numbers,
  live feature capture stats, the face-detection rate at 3 light levels, and a copy of the libomp matrix.
- [ ] `docs/RESULTS.md`: P0 · CLOUD · H12:30–15:30
- [ ] `docs/DEMO.md` (venue recalibration, judge-group reset, fallbacks): P0 · CLOUD · H12:30–15:30
- [ ] `docs/GOVERNANCE.md` (honest limits; `purge.py` already points to it): P1 · CLOUD · H12:30–15:30
- [ ] `docs/MODEL_CARD.md` (full simulator parameter table, claim wording, fairness citations,
  Art. 5(1)(f)): P1 · CLOUD · H12:30–15:30
- [ ] `docs/PERCEPTION.md`: P1 · LOCAL · H15:30–17:00
- [ ] `docs/NARRATOR.md`: P1 · LOCAL · H15:30–17:00
- [ ] PDF exports of the updated docs: P2 · CLOUD · on request only

## F. Demo & submission

- [x] **Tag `slice-v0`**: P0 · CLOUD · S2 · git tag on `0987043` · this commit lacks `valence/models/`
  (fixed in `9f31355`); a fresh checkout of the tag does not import.
- [ ] **Tag `model-v1`** (M2 served): P0 · CLOUD · S4 · git tag
- [ ] **Feature freeze + tag `demo-v1`**: P0 · CLOUD · H17:00 · git tag
- [x] **Lovable app in the repo**: P0 · YOU · started at H0 · `frontend/` (subtree merged in `8f2e105`) ·
  built from the v1 chat prompt.
- [ ] **Lovable app updated to the v2 prompt**: states from `GET /api/states` (4 or 6, no hard-coded list),
  `frame_w`/`frame_h`/`mirrored`, `source`/`simulated_fields` provenance, reset and room-only controls, the
  full governance counters: P0 · YOU · before H9:15–10:30 · `frontend/` (prompt in
  `docs/FRONTEND_PROMPT.md`) · none of these appear in `frontend/src` at the snapshot.
- [ ] **Lovable connected to the live backend**: P0 · YOU · H9:15–10:30 · runs locally next to the backend
- [ ] **Rehearsal 1**: live, Wi-Fi off: P0 · YOU · H17–20
- [ ] **Rehearsal 2**: chaos (kill the narrator, bad camera, muted mic) → `status.degraded` + fallbacks while
  ticks continue: P0 · YOU · H17–20
- [ ] **Backup screen recording** of a full replay run: P0 · LOCAL · H20–21 · keep it outside git (media
  file)
- [ ] **5 slides** from `reports/figs/`: P0 · YOU · H20–21 · order in `docs/PRD.md` §4.5
- [ ] **Devpost text**: P0 · CLOUD drafts (H12:30–15:30), YOU finalizes (H21–22) · `docs/DEVPOST.md`
  (*proposed*)
- [ ] **2-minute video**: P0 · YOU · H21–22
- [ ] **README final check**: P0 · YOU · H21–22 · `README.md`
- [ ] **Submit on Devpost**: P0 · YOU · by H22 (≈ 12:30 PM Sun)

### Venue checklist (Sunday, before judging)

- [ ] **Power and offline**: plugged in, `caffeinate -dims`, Wi-Fi off: P0 · YOU · venue
- [ ] **Devices**: camera/mic permission granted to the app that launches Python; mic pinned by name
  ("MacBook Pro Microphone", per `docs/BENCHMARKS.md`): P0 · YOU · venue
- [ ] **Pre-demo checks**: `make smoke && pytest -q && python scripts/e2e_check.py --speed 20`, narrator
  warm-up: P0 · YOU · venue
- [ ] **10-min recalibration**: `POST /api/calibrate/room` (SPL offset, 2-point lux, noise floor) with phone
  meters: P0 · YOU · venue
- [ ] **Start the live pipeline at the table ≥ 20 min before judging**, so the window is real and not seeded
  (`make live`; never `--reload` with live sensors): P0 · YOU · venue
- [ ] **Reset between judge groups**: `POST /api/demo/reset` (clears people, keeps room history): P0 · YOU ·
  venue
- [ ] **Demo split**: masked flip and obscure-face live; other states by replay with `source=replay` visible.
  Fallbacks: `POST /api/demo/seed` (context only, tagged `sim_seed`) if the window is empty;
  `POST /api/replay_recording/{name}` or scenario replay as the main demo if live fails: P0 · YOU · venue

## G. Frontend outputs (contract only)

This section covers the backend side only; the Lovable app itself is tracked in F. `[x]` means the message
or endpoint is defined in `valence/contracts.py` and emitted or served by `valence/runtime/engine.py`,
`valence/explain/narrator_client.py` or `valence/api/app.py`. The prompt is `docs/FRONTEND_PROMPT.md`; the
schemas are in `docs/API.md`. Owner CLOUD, synced at S2 unless noted.

### Outputs

- [x] **Webcam view + boxes**: P0 · `tracks` (`persons[].id`, `label`, `bbox`, `fist_active`, `slump`;
  `frame_w`, `frame_h`, `mirrored`, `capture_ts_ms`). The browser camera is a local preview only.
- [x] **Governance purge log + counters**: P0 · `governance` (`message`, `frames_decoded`, `frames_purged`,
  `audio_blocks_purged`, `transcripts_purged`, `violations`, `live_raw_buffers`, `audit_hash`,
  `retention_s`); `GET /api/governance/stats` adds the derived-buffer counts.
- [x] **Per-person state + confidence**: P0 · `prediction.label_display`, `prediction.confidence`
  (sim-calibrated, unsmoothed)
- [x] **All state probabilities**: P0 · `prediction.probabilities`; the state list comes from
  `GET /api/states` and `hello.states`
- [x] **Face-only vs context + override flag**: P0 · `prediction.face_only` (`label`, `confidence`; null =
  no face visible), `prediction.mask_override`
- [x] **Narrator text + source**: P0 (template) / P1 (LLM) · `prediction.explanation` (`text`, `source`),
  replaced by `explanation` (`event_id`, `text`, `source`, `latency_ms`)
- [x] **Layer-grouped drivers**: P0 · `prediction.layers[]` (`name`, `contribution`, `drivers[]`: `feature`,
  `display`, `value`, `unit`, `shap`, `level`, `effect`, `scope`)
- [x] **Current state-vector values**: P0 · `state_vector` (`ambient_db`, `light_lux`, `hours_since_break`,
  `time_of_day`, `clenched_fists_5m`, `compliments_15m`, `laughter_15m`, `posture_slump`, `head_in_hands`,
  `social_withdrawal`, `face_visible`, `facial_valence`, `window_fill_min`)
- [x] **15-min timeline + warm-up**: P0 · `GET /api/history?minutes=15`, `prediction.warming_up`,
  `state_vector.window_fill_min`
- [x] **System status**: P0 · `status` (`mode`, `camera`, `mic`, `fps`, `loop_latency_ms`, `narrator`,
  `offline`, `uptime_s`, `degraded`, `purge_violation`)
- [x] **Retention summary**: P0 · `GET /api/retention`
- [x] **Evaluation figures**: P0 · `GET /api/eval/summary`, `GET /api/eval/figures/{name}.png` (serves
  `reports/figs/`; contents as in C2/C3)
- [x] **`simulated_fields` provenance**: P0 · `source` and `simulated_fields` on `state_vector` and
  `prediction`; `data_quality` (`face_visible_frac`, `lux`, `ood_features`) on `prediction`
- [x] **Session header**: P0 · `hello` (`contract_version`, `mode`, `room_only`, `states`,
  `feature_version`, `frame_w`, `frame_h`, `mirrored`)

### Inputs

- [x] **Start / stop / pause** (privacy kill switch): P0 · `POST /api/control/{start|stop|pause}`
- [x] **Log a break**: P0 · `POST /api/events/break`
- [x] **Replay picker**: P0 · `GET /api/scenarios`, `POST /api/replay/{scenario}`
- [x] **Face-hidden / low-light toggles**: P0 · `POST /api/sim/face_occlusion`, `POST /api/sim/low_light`
  with `{"enabled": true|false}`
- [x] **Reset between visitors**: P0 · `POST /api/demo/reset`
- [x] **Room-only mode**: P1 · `POST /api/mode` with `{"room_only": true|false}`

## Drop order (from PLAN.md)

If any block runs more than 30 min over, take its fallback. Cut features in this order:

1. P2 (attention slide, EmotiEffLib, perception subprocess, self-distillation)
2. P1 states (Overwhelmed, Tense): they land only at the H10:30 gate and are the first real cut
3. Social withdrawal
4. Lux fit → raw brightness + `POST /api/sim/lux`
5. Acted set → 2 scenarios
6. M3
7. Vosk → `POST /api/sim/event`
8. YAMNet
9. LoRA → ship Stage A
10. GestureRecognizer → curl heuristic
11. LLM narrator → template only

**Never drop P0 governance or evaluation.** If S4 slips past H10:30, replay becomes the main demo and live
runs in hybrid mode with SIMULATED flags. No new code after sleep: triage only.
