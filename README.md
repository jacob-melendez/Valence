# Valence

> *The environment carries the signal, not the face.*

Valence is a hybrid individual-and-environment system that processes live webcam and microphone input on
the edge. It detects environmental stimuli (noise, light, time since a break), body mechanics (clenched
fists, slumping, head in hands) and social triggers (laughter, compliments). Raw frames and audio blocks are
zero-filled in RAM right after feature extraction, and only counts, levels, flags and normalized boxes move
on. Those values form a 15-minute **Rolling State Vector**. A classifier labels each person's emotional
state from it, and every prediction comes with a plain-English explanation of the context behind it.

Built for **TrojanHacks @ USC 2026, AI Data Governance track**. This is a **build in progress**; see
[Status](#status).

## Valence Web (Hack OHI/O 2026)

Valence is a continuation project at Hack OHI/O 2026. The event baseline is the tag `hackohio-2026-baseline`
(commit `cc28be9`), and [`docs/HACKOHIO_LOG.md`](docs/HACKOHIO_LOG.md) logs every change since.

At the event, the whole engine moved into the browser.

- **Everything runs in the page:**
  - perception (MediaPipe in a Web Worker);
  - the 15-minute window and the 123 features;
  - LightGBM with TreeSHAP;
  - the smoother and the mask override;
  - explanations and the hash-chained audit log.
- **Checked against the Python engine on golden data:** identical model scores, explanations and audit hashes;
  features and probabilities within 1–2 ulp.
- **A new 10-state emotion map** on a valence–arousal plane, with a self-report capture for real-data evaluation.
- **A sealed static site.** The Content-Security-Policy allows `connect-src 'self'` only, so the page cannot send
  camera or microphone data to any other origin.

All model scores come from simulated data.

To try it: `cd frontend && npm install && npm run dev`, then open `/?demo=tour`. This plays a scripted, simulated
session and needs no camera. Press **Use my camera (on-device)** for your own.

| Doc | What it covers |
|---|---|
| [`docs/WEB_ENGINE.md`](docs/WEB_ENGINE.md) | Architecture and the parity table |
| [`docs/DEPLOY_WEB.md`](docs/DEPLOY_WEB.md) | Cloudflare deploy and the security headers |
| [`docs/HACKOHIO_PITCH.md`](docs/HACKOHIO_PITCH.md) | Demo script |
| [`docs/media/valence-web-tour.mp4`](docs/media/valence-web-tour.mp4) | 75 s recording of the tour |

## Contents

- [Valence Web (Hack OHI/O 2026)](#valence-web-hack-ohio-2026)
- [What Valence is](#what-valence-is)
- [Status](#status)
- [Quickstart](#quickstart)
- [Make targets](#make-targets)
- [Repository layout and ownership](#repository-layout-and-ownership)
- [Documentation](#documentation)
- [How the two build sessions coordinate](#how-the-two-build-sessions-coordinate)
- [License](#license)

---

## What Valence is

### Thesis

Emotion recognition that reads only a face fails in three ways: a forced smile reads as happiness (the
masking dilemma); head in hands or a furrowed "focus face" is read without context (context blindness); and
pipelines store biometric data and give no reason for their labels (the black box). Valence reads the room
and the behavior over time as well as the face, keeps none of the raw media, and explains each prediction
from the model's own drivers.

What we claim, and only this: **under explicit, published behavioral assumptions, a temporal context model
beats a face snapshot; in-the-wild validation is future work.** The models are trained and evaluated on a
simulator whose parameters are published in [`docs/DATASET.md`](docs/DATASET.md).

```
webcam + mic -> edge perception -> PURGE GATE -> aggregator -> 15-min rolling window (5 s ticks)
               (counts, levels,   (zero-fill     (1 room slot     -> 123 features
                flags, boxes)      raw buffers)   + venue baseline
                                                  + <=3 person slots)
  -> M2 LightGBM (101 features: context + face history), M0 face-only model as a shadow
  -> TreeSHAP drivers grouped into layers -> MaskOverrideDetector -> FactSheet
  -> template explanation now, local LLM rewrite + validator later
  -> FastAPI REST + /ws/events on 127.0.0.1:8000 -> Lovable "Valence Audit Console", scripts/console.py
```

Explanation layers: Ambient Physics (dB, lux), Time Dynamics (hours since break, time of day), Somatic &
Social (fists, slump, head in hands, motion, withdrawal, laughter, compliments, people present) and Facial
Surface (face valence). The model reads noise and light only relative to the venue's room baseline, captured on
arrival with the person sitting neutral (`POST /api/room/baseline`), so a loud hall is not read as a loud,
frustrated room; explanations still quote the absolute level ("average noise of 72 dB over the last 5 minutes, 6 dB above
this room's baseline").

### States

Single source of truth: `valence/contracts.py` (`CORE_STATES`, `P1_STATES`, `STATE_DISPLAY`, `STATE_TIER`).
The active set is `VALENCE_STATE_SET=core` (4 states) or `all` (6 states).

| `label` | `STATE_DISPLAY` | Tier |
|---|---|---|
| `Focused` | Focused | P0 |
| `Frustrated_Masked` | Frustrated / Masked Stress | P0 |
| `Exhausted` | Exhausted | P0 |
| `Collaborative_Happy` | Collaborative / Happy | P0 |
| `Overwhelmed` | Overwhelmed / Distressed | P1 |
| `Tense` | Tense / Visible Frustration | P1 |

`display_label()` sets the shown name (`label_display`). With only the core states, `Frustrated_Masked`
also covers unmasked frustration and reads "Frustrated" unless `mask_override` fired. Once `Tense` is in the
set, `Frustrated_Masked` is mask-only and always shows its full name. `Collaborative_Happy` reads "Upbeat"
when `n_people` < 2. People are only ever "Person #N".

### The three proofs

Refined from the original PRD's three proofs ([`docs/PRD.md`](docs/PRD.md) section 4.2). No evaluation
result is reported here; everything below is the evidence the build is designed to produce.

| # | Proof | Evidence | Status |
|---|---|---|---|
| 1 | **Data minimization** (the code) | `PurgeGate` zero-fills each raw buffer and tracks it with a weak reference; a buffer still alive at tick end is a `PURGE_VIOLATION`. Capture uses `grab()`/`retrieve()` so `frames_decoded == frames_purged`. `pytest -v tests/test_purge.py tests/test_no_persistence_ast.py tests/test_logfilter.py`; `/api/governance/stats` after a 60-min soak (target: 0 `violations`, 0 `live_raw_buffers`); `scripts/audit_verify.py`; demo with Wi-Fi off. | Gate, tests and stats endpoint exist. Soak, `memory_flat.png` and `audit_verify.py`: TBD |
| 2 | **Context vs face** (the evaluation, on simulated data) | `accuracy_vs_history.png`: macro-F1 at 0/1/2/5/10/15 min of history for M0, M0t, M1, M2-noface, M2 (and M3 if time allows) with session-bootstrap 95% CIs. Headline numbers: `Frustrated_Masked` recall and the "Focused paradox" (true Focused read as Frustrated/Tense), M0 vs M2. `context_blindness` replay: same head-in-hands pose, two contexts, two labels. Live mask flip with `mask_override`; counterfactual ("still Frustrated at 0.5 h because of fists and noise"). | Produced by `valence/eval/evaluate.py`: `reports/metrics.json` and `reports/figs/` (core 4 states, untuned `noise_scale` 1.0, simulated), summarized in [`docs/RESULTS.md`](docs/RESULTS.md), plus a venue-shift evaluation (loud hall) |
| 3 | **Decision stability when face evidence is missing or degraded** (narrowed from v1's "anti-bias") | Robustness curves that degrade only the face channel for M0, M0t, M2, M2-noface; M2 vs M2-noface gap table; live obscure-face (`--obscure-face`): the label holds and the narration switches to its occlusion clause; face-detection rate at 3 light levels. **Demographic validation is an explicit gap, not a claim.** | Face-dropout and low-light robustness figures and tables in `reports/` and [`docs/RESULTS.md`](docs/RESULTS.md) (simulated); live parts TBD (acted set, rehearsal) |

Two supporting deliverables: a narrator faithfulness table (template vs prompt-only LLM vs LoRA: pass rate,
invented numbers, wrong qualifiers, false mask claims, demographic mentions, latency with YOLO running), and
the governance framing below.

### Offline and privacy guarantees

- **Offline at runtime.** Internet is used only for one-time model downloads. `.env.example` sets
  `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. The API binds to
  `127.0.0.1:8000` and the narrator service to `127.0.0.1:8081`, loopback only. The demo plan is
  to rehearse and present with Wi-Fi off.
- **Raw media is zeroed within one processing step.** Every frame and audio block goes through
  `gate.purge(buf, kind)` (in-place `fill(0)`, weak-reference tracking) right after feature extraction;
  purged buffers still referenced at the end of a tick are counted as `violations` and reported in every
  `governance` message.
- **Only scalars, flags and normalized boxes leave perception.** No pixels, audio, transcripts, body
  keypoints or identity embeddings cross into the runtime or onto the WebSocket. IDs are ephemeral display
  IDs ("Person #N"). Compliments come from a restricted Vosk grammar, so no free transcript ever exists.
- **Nothing raw reaches disk or logs.** An AST scan (`python -m valence.governance.persistence_scan`) bans
  file writes in `valence/perception/` and binary/media writes in `valence/edge/`; `RawMediaFilter` drops
  any log record that carries an array or bytes; the rolling window lives only in memory (at most 15 min).
- **The audit log records decisions, not data.** Labels, confidences, SHAP layer summaries and operator
  actions go into a hash-chained JSONL file (`var/audit/<session_id>.jsonl`). It is deleted when
  the session closes, and files older than `audit_ttl_s` (86 400 s) are removed at startup.

Honest limits: `del` removes a name, not memory, and copies inside native libraries (GPU allocator caches,
AVFoundation pixel buffers, MediaPipe's internal `mp.Image`) cannot be zeroed from Python. See the
`valence/governance/purge.py` docstring and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) section 11.3.

### Intended use

EU AI Act Article 5(1)(f) prohibits emotion recognition in workplaces and education institutions regardless
of consent, with only medical and safety exceptions. Valence is a **research prototype for consented
self-monitoring and personal wellbeing outside those settings**. It is not a tool for employers, schools,
proctoring or productivity scoring. `room_only` mode (one aggregate room label, no per-person inference) is
the compliance-oriented configuration; we do not claim it makes workplace use lawful. Ephemeral IDs make the
timeline pseudonymous, not anonymous, which is why the audit log has a TTL and is deleted at session end.
Details: [`docs/PRD.md`](docs/PRD.md) section 1.4.

---

## Status

**Build in progress, 2026-10-17 (Saturday evening; judging Sunday morning).** The replay path is
implemented end to end (simulator scenarios → aggregator → features → served M2 v1 → explanations → REST and
`/ws/events`). Live perception and the narrator exist as code on the branch but have not had their first live
end-to-end run. A file existing does not mean its tests pass. The live checklist is
[`docs/DELIVERABLES.md`](docs/DELIVERABLES.md). Rows below were last updated on 2026-10-18 (venue room baseline,
capture runs 2–3).

| Area | State |
|---|---|
| Shared contract `valence/contracts.py` (`CONTRACT_VERSION` 1.0.0) | On the branch; `tests/test_contracts.py` |
| Governance core: purge gate (incl. derived-buffer kinds), audit chain, raw-media log filter, persistence scan | On the branch; `tests/test_purge.py`, `test_audit.py`, `test_logfilter.py`, `test_no_persistence_ast.py`. The persistence scan currently reports 0 violations. |
| Runtime and API: aggregator (with the venue room baseline, `/api/room/baseline`), event bus, engine, features-only recording replay, REST + `/ws/events`, `scripts/console.py` | On the branch; `tests/test_runtime.py`, `tests/test_api.py`. Hybrid mode not implemented yet |
| Simulator v2 (with a random venue hall level per session), legacy generator, 9 scenarios (including `loud_venue`) | On the branch; `tests/test_scenarios.py`. Difficulty tuning (`tune_difficulty.py`) not written yet |
| Features (123 computed, 101 used by M2; room level relative to the venue baseline; same code offline and live), dataset builder | On the branch; `tests/test_feature_parity.py` |
| Model ladder (M0 to M2 variants), served M2 with M0 shadow, counterfactual | Trained (core 4 states, 300 sessions, seed 7, untuned `noise_scale` 1.0; `artifacts/model_meta.json`): served `m2_lgbm@8a78efc7`, the previous `85874e97` kept in `artifacts/legacy/` as a rollback; `tests/test_served.py`. Six-state models in `artifacts/all/`; M3 benchmark in `reports/m3_metrics.json` |
| Explainer, template text, narrator client | On the branch |
| Evaluation (`valence/eval/evaluate.py`), negative control | On the branch. `reports/metrics.json` and `reports/figs/` committed; the negative control passes on the typing capture (in-sample; `docs/MODEL_CARD.md` §5) |
| Mac perception + edge runner (`valence/perception/`, `valence/edge/`) | On the branch (video, audio, face valence, fists, posture, lux, IDs, capture threads). Live feature captures, runs 1–3, recorded (`data/real/`, `docs/BENCHMARKS.md`) |
| Narrator (`valence/narrator/`) | Stage A service (:8081), validator and templates on the branch as drafts; LoRA planned |
| Docs `GOVERNANCE.md`, `MODEL_CARD.md`, `RESULTS.md`, `DEMO.md`, `PERCEPTION.md`, `NARRATOR.md` | Written |

**Measured so far** (Apple M3 Max, 36 GB; the only source of measured numbers is
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)):

| Stage | Measured |
|---|---|
| `VideoPerceiver.process` end to end, 2 people, sample images | 45 ms p50 (loop budget 100 ms at 10 Hz) |
| YOLO26n-pose + ByteTrack, MPS, imgsz 640 | 18.4 ms p50 / 21.0 ms p95 |
| Narrator LLM Qwen3-1.7B 4-bit (MLX) | 187 tok/s generation; 536 ms wall for a 55-token answer; 1.27 GB peak |
| Edge runner live, video + audio (built-in webcam) | 45–57 ms/frame at 10.0 Hz |

---

## Quickstart

### Core (any OS, Python 3.12)

The core (contracts, governance, runtime, simulator, models, API) needs no camera or GPU.

```bash
git clone https://github.com/jacob-melendez/Valence.git
cd Valence
git checkout claude/happy-ride-7w3ng3

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'     # a uv venv has no pip; with `uv venv --seed`, `pip install -e '.[dev]'` works too

pytest -q
python -m valence.governance.persistence_scan             # exit code 1 on any violation
python -m valence.governance.persistence_scan --exceptions  # also list documented exceptions
```

- Tests that fail only because a Mac-only package (`ultralytics`, `sounddevice`, `vosk`, `mlx`, `mlx_lm`,
  `torch`) is missing are reported as skipped (`tests/conftest.py`).
- Exact versions of the core environment are frozen in `requirements.lock`.
- Optional extras in `pyproject.toml`: `bench` (torch, for the M3 GRU benchmark) and `slides` (shap, for
  beeswarm figures only; serving uses LightGBM `pred_contrib`).

Replay demo (serves the trained M2 when `artifacts/model_meta.json` matches `VALENCE_STATE_SET`, otherwise
the rule stub, flagged `predictor:stub` in `status.degraded`):

```bash
make replay                     # SCENARIO=full_demo SPEED=1 by default; API on 127.0.0.1:8000
python scripts/console.py       # second terminal: terminal view of /ws/events
```

### Mac edge (Apple Silicon, arm64 Python 3.12)

Live perception needs an arm64 Python: MediaPipe ships no x86_64 macOS wheel, so a Rosetta Python fails.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
python -c "import platform; assert platform.machine() == 'arm64'"

uv pip install -r requirements-mac.txt
uv pip install -e '.[dev]'
# ultralytics pulls opencv-python, which shadows the contrib build; swap it back (do this last):
uv pip uninstall opencv-python opencv-python-headless
uv pip install --no-deps --reinstall opencv-contrib-python==4.14.0.94

bash scripts/download_models.sh            # one-time downloads into models/ (gitignored)
# SKIP_LLM=1 bash scripts/download_models.sh   # perception and audio models only

cp .env.example .env                       # offline switches + VALENCE_* runtime settings
```

Nothing reads `.env` automatically yet. Export it before starting (`set -a; source .env; set +a`), or pass
`--env-file .env` to uvicorn. Every `VALENCE_<NAME>` variable overrides the matching field in
`valence/config.py`. LightGBM needs `brew install libomp`. torch and LightGBM share one process in live mode:
`valence/api/app.py` imports the edge (torch) before LightGBM and the served model predicts with
`num_threads=1`. Do **not** set `OMP_NUM_THREADS` (measured on the demo Mac: segfault; see
`docs/CONTRACT_REQUESTS.md` round 2). Uncomment `KMP_DUPLICATE_LIB_OK=TRUE` in `.env` only if a libomp clash
still appears.

Then:

```bash
python scripts/spikes/selftest.py               # camera and mic deliver real signal, not silent zeros
python -m valence.edge.run --video --audio --print   # features-only readout; never media
make live                                       # API + live edge on 127.0.0.1:8000
```

- macOS grants camera and microphone access to the app that launched Python (Terminal, iTerm, VS Code), not
  to Python. A denied camera gives black frames and a denied mic gives exact zeros; the self-test fails
  loudly on both.
- Never run uvicorn with `--reload` while live sensors are on (it opens the camera twice).
- Room calibration: `scripts/calibrate_spl.py` (mic dBFS to dB SPL against a phone meter) and
  `scripts/calibrate_lux.py` (camera to lux against a phone lux meter).
- Venue room baseline: on arrival, sitting neutral, `curl -s -X POST 127.0.0.1:8000/api/room/baseline -H
  'content-type: application/json' -d '{"seconds": 90}'`; without it the first 90 s of the live session are used
  (`docs/DEMO.md` §2, `docs/API.md` §7.4.12).
- The narrator service (`python -m valence.narrator.service`, Stage A draft) runs in its own venv,
  `valence/narrator/.venv`. Without it, the template explanation in each `prediction` stands.

---

## Make targets

`Makefile` uses `.venv/bin/python` (override with `PY=...`).

| Target | What it does | Status |
|---|---|---|
| `make test` | `pytest -q` | In `Makefile` |
| `make scan` | Persistence scan | In `Makefile` |
| `make replay` | API + scripted replay (`SCENARIO=full_demo SPEED=1`) | In `Makefile`; serves M2 when the artifacts match the state set |
| `make live` | API + live edge (Mac only) | In `Makefile`; depends on the Mac edge setup above |
| `make console` | Terminal view of the event stream | In `Makefile` |
| `make train` | Simulate the corpus in memory and train the model ladder into `artifacts/` (`python -m valence.models.train_ladder --sessions 300 --seed 7 --noise-scale 1.0 --state-set core`; `STATES=all` for 6 states) | In `Makefile`. `noise_scale` 1.0 is untuned until `tune_difficulty.py` exists |
| `make eval` | `python -m valence.eval.evaluate --bootstrap 200` into `reports/metrics.json` and `reports/figs/*.png` | In `Makefile` |
| `make facts` | Balanced narrator fact sheets into `data/narrator/fact_sheets_sim.jsonl` | In `Makefile` |
| `make negative` | Negative control on the live typing capture (`data/real/capture_typing*.jsonl`, `--expect Focused`) | In `Makefile`; passes on the committed capture (in-sample, `docs/MODEL_CARD.md` §5) |
| `make smoke` | Pre-demo check | In `Makefile` as `test` + `scan`. **Planned** pre-demo sequence: `make smoke && pytest -q && python scripts/e2e_check.py --speed 20` (`e2e_check.py` not written yet) |

There is no separate data target: `make train` regenerates the simulator sessions from the seed. Writing the
corpus to `data/sim/` with a manifest is planned (`docs/DATASET.md` section 15).

---

## Repository layout and ownership

**C** = CLOUD session (Linux container, no camera or GPU). **L** = LOCAL session (Claude Desktop on the demo
Mac). Ownership is disjoint; full module map in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) section 4.

| Path | Owner | Contents | State |
|---|---|---|---|
| `valence/contracts.py` | C | States, display rules, `PersonObs` / `VideoObs` / `AudioObs`, `TickRecord`, `FactSheet`, `NarrationResult`, WebSocket messages, `json_schemas()` | Exists |
| `valence/config.py` | C | `Settings`; every field overridable by `VALENCE_<NAME>` | Exists |
| `valence/governance/` | C | `purge.py` (`PurgeGate`, module-level `gate`), `audit.py` (`AuditLog`), `logfilter.py` (`RawMediaFilter`), `persistence_scan.py` | Exists; the retention summary is served by `valence/api/app.py` |
| `valence/runtime/` | C | `bus.py` (bounded per-client queues, drop-oldest), `aggregator.py` (room/person slots, venue room baseline, near-field gate, per-person break clocks, calibration), `engine.py` (ticker, scenario replay and live sources), `recording.py` (features-only JSONL replay) | Exists |
| `valence/sim/` | C | `generator.py` (simulator v2), `params.py`, `scenarios.py`, `legacy.py` (original mock, kept for comparison) | Exists; `tune_difficulty.py` planned |
| `valence/features/` | C | `spec.py`, `window.py`, `compute.py` (123 features, 101 used by M2, same code offline and live), `build_dataset.py` | Exists |
| `valence/models/` | C | `predictor.py` (interface, smoothing, stub), `served.py` (M2 + M0 shadow, counterfactual), `train_ladder.py`, `temporal.py` (M3) | Exists |
| `valence/explain/` | C | `describe.py`, `explainer.py` (layers, fact sheet), `template.py`, `narrator_client.py` | Exists |
| `valence/eval/` | C | `evaluate.py` into `reports/`, `negative_control.py` (also scores acted recordings with `--expect`), `style.py` | Exists |
| `valence/api/app.py` | C | Lifespan, REST under `/api`, `/ws/events` | Exists |
| `valence/perception/` | L | Video (YOLO26n-pose + ByteTrack, hands, posture, face valence from blendshapes, lux), audio (dB, YAMNet laughter, Vosk grammar), debouncing, IDs, calibration constants, `gate.py` (single import point for the purge gate) | Exists |
| `valence/edge/` | L | `run.py`: `start(sink, *, video, audio, obscure_face) -> stop`; CLI `--print` / `--jsonl`; `capture.py`; `calibration_store.py` | Exists |
| `valence/narrator/` | L | `service.py` (:8081), `engine.py`, `prompt.py`, `validator.py`, `lexicon.py`, `schema.py`, `templates.py`, `fake_fact_sheets.py`, `evaluate.py`; runs in its own venv `valence/narrator/.venv` (gitignored) | Stage A drafts exist; LoRA planned |
| `scenarios/*.yaml` | C | `focused`, `frustrated_masked`, `exhausted`, `collaborative_happy`, `overwhelmed`, `tense`, `context_blindness`, `full_demo`, `loud_venue` | Exists |
| `scripts/` | C / L | C: `console.py`, `export_fact_sheets.py` (exist); `e2e_check.py`, `mem_soak.py`, `audit_verify.py` (planned). L: `spikes/`, `download_models.sh`, `calibrate_spl.py`, `calibrate_lux.py` (exist); `overnight.sh` (planned) | Mixed |
| `tests/` | C; L for `tests/perception/`, `tests/narrator/` | Unit and integration tests | Exists, growing |
| `docs/` | C; L for `BENCHMARKS.md`, `PERCEPTION.md`, `NARRATOR.md`, `CONTRACT_REQUESTS.md` | See [Documentation](#documentation) | Mixed |
| `data/` | C: `sim/` (gitignored), `narrator/fact_sheets_sim.jsonl`. L: `real/` (features only), `narrator/{train,valid,test}.jsonl` | Features-only datasets | Planned |
| `artifacts/`, `reports/` | C | `artifacts/`: served models (`m0.txt`, `m2.txt` committed; the rest of the ladder local only), `model_meta.json` (temperatures, feature names, states), `clip_bounds.json`; `artifacts/all/` (6 states); `artifacts/legacy/` (the previous served M2, rollback with `VALENCE_ARTIFACTS_DIR=artifacts/legacy`). `reports/`: figures, `metrics.json`, `m3_metrics.json`, `capture_vs_sim.json` | Exist |
| `models/`, `adapters/`, `var/` | — | Downloaded weights, LoRA adapters, runtime files such as `var/audit/` | Gitignored, never committed |
| `pyproject.toml`, `requirements.txt`, `requirements.lock`, `Makefile`, `.gitignore`, `.env.example` | C | Packaging, core dependencies, commands, offline settings | Exists |
| `requirements-mac.txt` | L | Mac-only perception, audio and narrator dependencies | Exists |
| `Valence-PRD.pdf`, `Valence-CodeArch.pdf`, `Valence-MockDataset.pdf` | — | Original documents, superseded by `docs/PRD.md`, `docs/ARCHITECTURE.md`, `docs/DATASET.md` | Kept for reference |

---

## Documentation

| Document | What it covers | Owner |
|---|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Approved execution plan: architecture, states, data contracts, model ladder, schedule, proofs, risks. Source of truth for design decisions | C |
| [`docs/DELIVERABLES.md`](docs/DELIVERABLES.md) | Live checklist of code, artifacts, proofs, tests and docs | C |
| [`docs/PRD.md`](docs/PRD.md) | Product requirements v2 (replaces the v1 PDF): thesis, claims and non-claims, intended use, data lifecycle, proofs, success metrics (targets) | C |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Data flow, threads, module map, aggregator, features, model ladder, inference path, governance controls, fallbacks | C |
| [`docs/DATASET.md`](docs/DATASET.md) | Simulator v2 specification and full parameter table, legacy compatibility, difficulty tuning, scenarios, splits | C |
| [`docs/API.md`](docs/API.md) | REST endpoints, WebSocket messages, provenance (`source`, `simulated_fields`), audit log entries | C |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Measured edge performance on the demo Mac (the only source of measured numbers) | L |
| [`docs/CONTRACT_REQUESTS.md`](docs/CONTRACT_REQUESTS.md) | Edge session's assumptions and asks against the shared contract | L |
| [`docs/CONTRACT_RESPONSES.md`](docs/CONTRACT_RESPONSES.md) | Core session's decisions on those requests, by round | C |
| [`docs/FRONTEND_PROMPT.md`](docs/FRONTEND_PROMPT.md) | Prompt for the Lovable "Valence Audit Console" frontend (built outside this repo) | C |
| [`docs/LOCAL_SESSION_PROMPT.md`](docs/LOCAL_SESSION_PROMPT.md) | Prompt that started the Mac session, plus the v2 addendum | C |

Also: [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md),
[`docs/RESULTS.md`](docs/RESULTS.md), [`docs/DEMO.md`](docs/DEMO.md) (C); [`docs/PERCEPTION.md`](docs/PERCEPTION.md),
[`docs/NARRATOR.md`](docs/NARRATOR.md) (L).

When documents disagree: `valence/contracts.py` wins on field names, states and messages; `docs/PLAN.md` wins
on design; `docs/BENCHMARKS.md` is the only source of measured numbers.

---

## How the two build sessions coordinate

Valence is built by one person running two Claude sessions in parallel:

- **CLOUD** (Linux container, no camera or GPU) builds the hardware-independent core: contracts, governance,
  runtime, simulator, features, models, explainer, evaluation, API, most tests and docs.
- **LOCAL** (Claude Desktop on the demo Mac) builds everything that needs the camera, microphone or Apple
  Silicon: perception, the edge runner, the narrator, spikes, calibration and benchmarks.

They coordinate through files on one branch:

1. **One contract file.** `valence/contracts.py` defines every type and field that crosses the boundary
   (`VideoObs` / `AudioObs` from the edge, `FactSheet` to the narrator, `NarrationResult` back, all WebSocket
   messages). It changes only through the request process below. Additive changes (new optional fields with
   defaults) are safe; renames and removals are not. Example: `VideoObs.frames_decoded` defaults to 0 so
   existing constructors keep working.
2. **Requests and responses.** LOCAL writes assumptions and asks to
   [`docs/CONTRACT_REQUESTS.md`](docs/CONTRACT_REQUESTS.md) and the builder relays them. CLOUD answers in
   [`docs/CONTRACT_RESPONSES.md`](docs/CONTRACT_RESPONSES.md) with a decision per item (confirmed, adopted,
   or corrected), in numbered rounds.
3. **Disjoint ownership.** Each path has one owner ([layout table](#repository-layout-and-ownership)). A
   session never edits the other's files; it asks through the contract documents instead.
4. **Git.** Both sessions push to `claude/happy-ride-7w3ng3` and run `git pull --rebase` before every push.
   Small commits. Never commit weights, `models/`, `adapters/`, `.venv/` or any raw media.
5. **Pre-push checks.** `pytest -q` and `python -m valence.governance.persistence_scan`. Mac-only
   dependencies stay in `requirements-mac.txt`; core code stays Python 3.12 compatible so it runs in the
   cloud container and on the Mac.

---

## License

MIT. Copyright (c) 2026 Jacob Melendez. See [`LICENSE`](LICENSE).
