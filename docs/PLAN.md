# Valence — 22-Hour Backend/ML Execution Plan

## Context
TrojanHacks @ USC 2026, AI Data Governance track; judging Sunday morning. Solo builder, ~22 h left, Apple
Silicon Mac demo machine. Repo `jacob-melendez/Valence` is empty (README, LICENSE, Python .gitignore);
branch `claude/happy-ride-7w3ng3` (not yet on GitHub — first push creates it from `main`).
Scope = backend/ML only, "full model architecture": live edge perception (webcam + mic) → biometric purge
gate → 15-min rolling state vector → trained temporal classifier vs face-only baseline → SHAP explanations →
**local small LLM** narrator (specialized to Valence's states; varied but fact-grounded) → local
FastAPI/WebSocket interface consumed by a Lovable frontend (frontend itself out of scope). Fully on-device;
internet only for one-time downloads.

Problems in the provided materials this plan fixes: mock rows are i.i.d. (the window has nothing to exploit);
classes trivially separable on context (~100% — judges see through it); no laughter, noise, missing
modalities or multi-person; no link between synthetic scales and live sensors; laughter treated as a
"keyword"; `del frame` alone is not erasure.

Method: 4 research briefs (versions verified 2026-10-17) → 3 competing plans (risk-first 41/50,
judging-first 40, ML-depth 39) → merged on the risk-first backbone → adversarial critique (3 blockers,
~20 majors, all folded in below and marked ⚑).

## Locked decisions
- Two parallel Claude sessions: **CLOUD** (this one; Linux, no camera/GPU) builds the hardware-independent ML
  core; **LOCAL** (Claude Desktop on the Mac) builds perception + narrator. Local prompt already given in
  chat; the ⚑ addendum at the end of this plan must be sent to it.
- Stack: Python 3.12 arm64 (uv) · MediaPipe Tasks (GestureRecognizer, FaceLandmarker, YAMNet AudioClassifier in
  AUDIO_CLIPS mode ⚑) · Ultralytics YOLO26n-pose + ByteTrack (MPS) · sounddevice · Vosk 0.3.44 restricted
  grammar · LightGBM (served) · PyTorch GRU (benchmark only) · LightGBM `pred_contrib` TreeSHAP · mlx-lm 0.31.3
  + Qwen3-1.7B-4bit in its own venv ⚑ (fallback Llama-3.2-3B-Instruct-4bit, downloaded only if Qwen fails) ·
  FastAPI + uvicorn.
- Narrator: local LLM; prompt-only Stage A, then QLoRA Stage B; faithfulness validator; template fallback.
- States: 4 core (P0) + Overwhelmed and Tense (P1, user-approved), cap 6, config-driven.
- Frontend: Lovable (GitHub sync → run locally next to the backend); reads states from `GET /api/states`.

## Emotional states (single source of truth: `valence/contracts.py`)
`STATES`, `STATE_DISPLAY`, `STATE_LEXICON` (validator synonyms), `STATE_TIER`.
| State | Display | Face | Context signature | Tier |
|---|---|---|---|---|
| Focused | Focused | −0.15 (looks angry) | quiet 40–55 dB, normal light, 1–3 h since break, head-in-hands .2 ("thinking") | P0 |
| Frustrated_Masked | Frustrated / Masked Stress | +0.45 forced smile | loud, dim, 3.5–6 h, fists ~0.8/min | P0 |
| Exhausted | Exhausted | −0.1 flat | silent, very dim, 5–8 h, slump/withdrawal, head-in-hands .4 | P0 |
| Collaborative_Happy | Collaborative / Happy ("Upbeat" when 1 person ⚑) | +0.55 | lively 60–75 dB, bright, fresh break, laughter/compliments, ≥2 people | P0 |
| Overwhelmed | Overwhelmed / Distressed | hidden 40–70% (hands), −0.3 | head-in-hands frequent, slump, 3–7 h, 60–75 dB | P1 |
| Tense | Tense / Visible Frustration | −0.45 (agrees) | fists, 65–80 dB, often ≥2 people, high motion | P1 |
⚑ Core-4 phase: Frustrated_Masked covers unmasked frustration too (30%), so its display is "Frustrated" and
becomes "Frustrated / Masked Stress" only when `mask_override` fires. When P1 lands, Frustrated_Masked becomes
mask-only (forced smile ≥90%) and Tense takes unmasked frustration; displays become fixed. Tense vs
Frustrated_Masked is separable only via the face → M2-noface confuses that pair; report honestly ("face is
informative but not sufficient"). P1 lands only at the H10:30 gate, and is the first thing cut.

## Architecture
```
camera → capture thread: grab() every frame, retrieve() only at 10 Hz ⚑ → perception thread       PURGE GATE
  YOLO26n-pose+ByteTrack (desk ROI, box h ≥25%, ≤3 tracks ⚑) | FaceLandmarker 5 Hz | Gesture 5 Hz | lux 1 Hz ─┐ weakref
mic 16 kHz/100 ms → audio worker: Leq/p90 | YAMNet AUDIO_CLIPS 0.975 s | Vosk grammar ────────────────────┤ → fill(0)
ReplaySource: scenario YAML | features JSONL | /api/demo/seed ──────────────────────────┐                  └ → del
   (only VideoObs/AudioObs scalars, boxes and flags cross this line — keypoints stay in perception ⚑)
Aggregator → ROOM slot (dB, lux, laughter/compliments gated near-field ≥ room median+6 dB ⚑, time, break clock)
          + PERSON slots (fists, slump, head-in-hands, motion, withdrawal, face) → 5 s TickRecord
 → RollingWindow (180 ticks; new person inherits room history ⚑) → compute_features (~110) → clip (+ood list)
 → M2 LightGBM + M0 face-only shadow → softmax(z/T) [displayed confidence, unsmoothed ⚑] → EMA → hysteresis(2)
 → pred_contrib SHAP → layers → MaskOverrideDetector → FactSheet → template text (instant `prediction`)
 → EventBus (bounded, per-client drop-oldest ⚑) → FastAPI :8000 (/api + /ws/events) + scripts/console.py ⚑
 → AuditLog (hash-chained: labels + SHAP summaries only, session TTL ⚑) → Narrator :8081 → Validator → `explanation`
```
One main process (uvicorn main thread; daemon capture/perception/audio/ticker threads → `call_soon_threadsafe`;
camera/mic released in lifespan shutdown; never `--reload` with live sensors ⚑); narrator is a separate process
in its own venv. If perception < 8 Hz, move it to a `spawn` subprocess emitting only feature dicts. Sources
per modality `--video live|replay --audio live|replay`; every record carries `source` and `simulated_fields`.
`--room-only` mode emits one aggregate room label, no per-person inference ⚑ (compliance configuration).

## Repo layout & ownership (C = cloud, L = local)
```
valence/contracts.py      C  states, PersonObs/VideoObs/AudioObs, TickRecord, Prediction, FactSheet, WS msgs (pydantic → JSON Schema)
valence/config.py         C  TICK_S=5, WINDOW=180, thresholds, calibration offsets; .env.example with offline vars ⚑
valence/governance/       C  purge.py (PurgeGate), logfilter.py, audit.py (hash chain + TTL), retention
valence/runtime/          C  bus.py, aggregator.py (room/person slots, break clock, near-field gate), ticker.py, replay.py
valence/sim/              C  generator.py (v2), scenarios.py, legacy.py, tune_difficulty.py
valence/features/         C  spec.py (SENSOR_SPEC w/ units ⚑, CHANNELS ⚑, FEATURE_NAMES, FACE_FEATURES, LAYER_MAP), window.py, build_dataset.py
valence/models/           C  train_ladder.py, temporal.py (M3), predictor.py (clip, T, EMA, hysteresis), counterfactual.py ⚑
valence/explain/          C  explainer.py (layers, MaskOverrideDetector, build_fact_sheet w/ level/effect/scope ⚑), template.py, narrator_client.py
valence/eval/             C  evaluate.py, score_acted.py, negative_control.py ⚑ → reports/figs/*.png, reports/metrics.json
valence/api/app.py        C  lifespan starts threads; REST + /ws/events
scripts/                  C: console.py ⚑ e2e_check.py mem_soak.py audit_verify.py export_fact_sheets.py · L: spikes/ download_models.sh calibrate_*.py overnight.sh
valence/perception/       L  video.py tracking.py face.py fist.py posture.py lux.py audio.py
valence/edge/run.py       L  start(sink, *, video, audio, obscure_face) -> stop ; CLI --print/--jsonl
valence/narrator/         L  service.py (:8081), prompts.py, validator.py, grammar.py, build_data.py, lora.yaml (own .venv)
scenarios/*.yaml          C  focused, frustrated_masked, exhausted, collaborative_happy, overwhelmed, tense, context_blindness ⚑, full_demo
tests/                    C except tests/perception, tests/narrator (L)
docs/                     C except BENCHMARKS.md PERCEPTION.md NARRATOR.md (L); CONTRACT_REQUESTS.md (L writes, user relays)
data/                     sim/ + narrator/fact_sheets_sim.jsonl C · real/ + narrator/{train,valid,test}.jsonl L
artifacts/ reports/ requirements.txt requirements.lock ⚑ pyproject.toml Makefile .gitignore  C ; requirements-mac.txt L ; models/ adapters/ gitignored
```
Git: both sessions push to `claude/happy-ride-7w3ng3`, `git pull --rebase` before every push, disjoint ownership.

## Data contracts (`docs/API.md`; additive to the Lovable + local prompts already given)
- **Perception → runtime** (local `sink(obs)`): `PersonObs{track_id, bbox(x1,y1,x2,y2 norm), fist_onsets,
  fist_active, slump, head_in_hands, motion (per-person normalized ⚑), face_visible, face_valence|NaN}`;
  `VideoObs{t, wall, frame_seq, frames_decoded ⚑, lux, lux_quality, frame_w, frame_h, mirrored ⚑, persons}`;
  `AudioObs{t, wall, block_seq, db_spl, rms_power, speech_prob, laughter_onsets, compliment_hits}`.
- **TickRecord** (5 s, per person; identical for sim, replay, live; every live field is simulated ⚑):
  `tick, ts, t_mono, session_id, person_id, display_id, n_people, source, simulated_fields, ambient_db, db_p90,
  light_lux, lux_quality, laughter_count, compliments_count, clenched_fists_count, posture_slump,
  head_in_hands_frac, social_withdrawal_flag, motion_energy, face_valence|null, face_visible,
  hours_since_break, tod_h`. Per-minute rates → per-tick Poisson counts.
- **CHANNELS** (frozen ⚑): db, lux, motion, slump, hih, fist, laugh, compl, withdraw, n_people,
  hours_since_break, tod_sin, tod_cos; face separate.
- **Rolling State Vector (~110 features, order frozen in `artifacts/feature_spec.json`)**: S ∈ {db, lux, motion,
  slump, hih}: `S_now, S_{mean,std,slope,max}_{1m,5m,15m}, S_ewm_{1m,5m}, S_d1m15m, S_rel_sess` (85) ·
  E ∈ {fist, laugh, compl}: `E_cnt_{1m,5m,15m}, E_since_s` (12) · `withdraw_now, withdraw_frac_5m, n_people` (3) ·
  `hours_since_break, tod_sin, tod_cos, fill_ratio` (4) · FACE: `face_now, face_mean_{1m,5m,15m}, face_std_5m,
  face_visible_frac_5m` (6). NaN native; clipped to train p1–p99 with `data_quality.ood_features` reported ⚑
  (>3 clipped → lower confidence band, narrated). LAYER_MAP: db/lux → Ambient Physics; hours_since_break/tod →
  Time Dynamics; fist/laugh/compl/slump/hih/motion/withdraw/n_people → Somatic & Social; face → Facial Surface.
- **WebSocket `/ws/events`**: `hello` (mode, states, feature version, frame_w/h, mirrored) · `tracks` ≤10 Hz
  (ids, labels, normalized boxes, fist/slump flags, capture_ts_ms; never pixels or keypoints ⚑) · `governance`
  per tick (PRD line, frames_decoded, frames_purged, audio/transcripts purged, violations, live_raw_buffers,
  audit_hash, retention_s ⚑) · `state_vector` 5 s/person · `prediction` 1 Hz/person (label, label_display,
  confidence "sim-calibrated" ⚑, probabilities, face_only, mask_override, warming_up, layers, data_quality,
  explanation{text, source:"template"}) · `explanation` · `status` · `heartbeat`.
- **REST `/api`**: GET `states, state, history?minutes=15, governance/stats, governance/audit?since=,
  governance/verify, retention, eval/summary, eval/figures/{name}.png, scenarios, models, health,
  narrator/health, explain/{event_id}/counterfactual?feature=value ⚑` · POST `control/{start|stop|pause},
  events/break, replay/{scenario}, sim/{face_occlusion|low_light}, demo/seed, demo/reset ⚑, calibrate/room ⚑,
  mode {room_only} ⚑, label {state} ⚑ (self-report), sim/event, sim/lux`. sim/*, seed and operator-set fields are
  logged SIMULATED and shown in `simulated_fields`. CORS: localhost + Lovable origin.
- **FactSheet** (cloud → `POST :8081/narrate`): `event_id, person, label, label_display, confidence,
  confidence_display, probabilities, window_fill_min, n_people, face_only{label, confidence, face_valence},
  face_visible, mask_override, layers[{name, contribution, drivers[{feature, display, value, unit, shap,
  level: high|elevated|normal|low, effect: supports|opposes, scope: room|person}]}] ⚑,
  data_quality{face_visible_frac, lux, ood_features}`.

## Model architecture
- **Simulator v2**: ~300 sessions × 1–3 people × 60–120 min at 5 s (~500k person-ticks), seeds logged. Per-person
  semi-Markov hidden state (gamma dwell, mean 4–10 min, min 2 min); logistic tilt toward Frustrated/Exhausted/
  Overwhelmed after ~3 h; Poisson breaks (~1/90 min) reset the clock; Happy likelier with ≥2 people. Room dB
  drifts toward occupants' state targets (2–4 min lag, σ 4 dB, **±12–15 dB per-session mic offset ⚑**); lux
  per-session baseline 120–600 + time-of-day drift. Event rates/min: fists Frustrated/Tense 0.8 bursty; laughter
  Happy 0.5; compliments Happy 0.2. **False-positive rates and motion/slump scales are set from the live feature
  capture (typing, keyboard gaze) ⚑**, not guessed. head_in_hands per state (Exhausted .4, Frustrated .25,
  Focused .2, Happy .1 during laughter, Overwhelmed high) ⚑. Face per state table; noise grows below 200 lux;
  face NaN on 5–40% of ticks. 2% per-modality dropout. `legacy.py` reproduces the original ~100% for the "why
  we rebuilt the data" slide; `to_legacy_csv()` keeps the original 9 columns.
- **Difficulty ⚑ (anti-circularity)**: tune one `noise_scale` against ONE criterion that is not the result —
  M2 macro-F1 ≤ .92 (no ceiling) and live-vs-sim feature variance match. Report M0/M1 wherever they land; show
  ×0.5/×1/×1.5 sensitivity and the legacy slide as evidence the ordering is structural. Full simulator
  parameter table in MODEL_CARD.md. Claim wording: "Under explicit, published behavioral assumptions, a temporal
  context model beats a face snapshot; in-the-wild validation is future work."
- **Features**: one plain-numpy `compute_features(window)` offline and live; training windows every 6 ticks,
  random length 1–180 ticks.
- **Model ladder**: M0 face snapshot · M0t face history · M1 context snapshot (CHANNELS) · M2-noface ·
  M2-noabsdB ⚑ · M2-notime ⚑ · **M2 = context + face history, served live** · M3 GRU(64) + attention (benchmark,
  cloud CPU, P1). LightGBM: 31 leaves, lr .05, ≤800 rounds, early stop 50, min_data_in_leaf 100, fractions .8,
  balanced weights. Split by session 70/15/15; validation split into early-stop half and calibration half ⚑;
  temperature scaling.
- **Explainability**: `pred_contrib` (<5 ms) → predicted-class slice → layer sums → top-3 drivers with display
  strings, `level`, `effect`, `scope`. **MaskOverrideDetector ⚑**: M0 argmax ≠ M2 label, M0 conf ≥ .55, M2 conf
  ≥ .70, face-layer SHAP for M0's class inside M2 positive, 3 consecutive ticks, blocked first 2 min; validated
  on scenarios as soon as M2 exists. **Counterfactual ⚑**: re-run M2 with one feature changed (e.g.
  hours_since_break=0.5) to defuse "operator-typed driver" attacks.
- **Narrator** (local): FactSheet → compact `key: display` lines (~120 tokens) ⚑ → Qwen3-1.7B-4bit
  (`enable_thinking=False`) with mlx_lm prompt cache for the fixed system prompt + few-shots ⚑, max_tokens 90.
  Stage B QLoRA (rank 8, scale 20, all attn+MLP projections, 16 layers, lr 1e-4, batch 4, 600 iters,
  `--mask-prompt --grad-checkpoint`; 8 GB Mac: batch 1–2, 8 layers ⚑) on 1500/150/300 grammar-generated
  targets from M2 SHAP sheets, balanced states × {face agrees, disagrees, hidden}. Self-distillation cut ⚑.
  Validator: numbers ⊆ sheet; names predicted state; ≥2 of top-3 drivers, nothing outside top-5; qualifier
  within 6 tokens of a feature must match its `level`/`effect` ⚑; room-scope drivers phrased as "the room" ⚑;
  mask language only with mask_override; no pronouns/demographics/diagnoses; 20–90 words. Cached text is
  re-validated against the current sheet ⚑. Ship LoRA only if pass rate ≥ Stage A and p95 ≤ 2.5 s **measured
  with YOLO running** ⚑.
- **Per-tick inference**: features → clip → M2 + M0 → temperature (displayed confidence) → EMA/hysteresis
  (label only) → SHAP → override → FactSheet → `prediction` with template text → narration (latest-only per
  person; triggers: label change, band change, every 30 s) → `explanation`. `warming_up` first 60 s.

## Schedule (H0 = approval ≈ 2:30 PM PT Sat; H22 = submission ≈ 12:30 PM Sun; ★ critical path; S = sync point)
| Time | CLOUD (this session) | LOCAL (Mac session) | YOU |
|---|---|---|---|
| H0–0:45 ★ | First push: skeleton, requirements, .env.example, contracts, PurgeGate (id→weakref dict ⚑) + audit chain + logfilter + tests, docs (PRD v2, ARCHITECTURE, DATASET, API, DELIVERABLES, LOCAL/FRONTEND prompts v2) → **S1** | Step 0 env + hardware check (chip/RAM) + torch+LightGBM same-process test ⚑ + Wi-Fi-off smoke ⚑ | Start local session, paste prompt + ⚑ addendum; grant camera/mic; start Lovable |
| H0:45–2:30 ★ | Slice: bus, aggregator (room/person slots), window, ticker, replay, stub predictor, FastAPI, console.py, e2e replay test → `slice-v0` **S2** | Step 1 spikes + benchmarks | |
| H2:30–4:00 ★ | Simulator v2 untuned (noise_scale=1), scenarios incl. context_blindness, legacy ⚑ reorder | Step 2 video (3.5 h ⚑) | |
| H4:00–5:15 ★ | Features + parity / no-lookahead / window tests | Step 2 | |
| H5:15–6:00 ★ | Quick M0/M2 trainer → tune noise_scale (one criterion) → freeze → generate 300 sessions | Step 2 → Step 3 audio (1.5 h ⚑) | Meal |
| H6:00–7:00 ★ | Full ladder + calibration + clip bounds; override detector validated; export fact sheets → **S3** | Step 3 | |
| H7:00–8:00 ★ | evaluate v1 → proof 2/3 charts (checkpoint B) | **Live feature capture ⚑** (typing, keyboard gaze, fists, slump × 3 min, features only) → push | Act the capture |
| H8:00–9:15 ★ | Explainer, template text, narrator client, counterfactual; stub → M2 → `model-v1` **S4** | Step 5 narrator Stage A (prompt cache, compact sheet) | SPL/lux calibration with phone apps |
| H9:15–10:30 ★ | Apply capture measurements → sim FP rates/scales → regenerate + retrain (minutes); negative-control gate (typing replay ≤20% Frustrated ⚑); live-mode wiring, hybrid, seed/reset/calibrate/room-only/label endpoints | Step 4 edge runner → first live end-to-end; start self-report labels every 10 min ⚑ | Verify live; connect Lovable live |
| **H10:30 gate** | Core 4 live E2E + negative control pass? yes → P1 states; no → fix live, skip P1 | | Decide |
| H10:30–12:00 | P1 states (Frustrated_Masked → mask-only, Tense = unmasked), retrain, re-eval, re-export sheets → **S5** | Acted set: 6 scenarios × 3 min + face detection rate at 3 light levels ⚑; narrator grammar + LoRA data from S5 | Act scenarios |
| H12:00–12:30 | score_acted, negative control on acted typing | **Dry-run every overnight job in smoke mode ⚑** (LoRA --iters 10, A/B on 5 sheets), then launch `overnight.sh` (`set -uo pipefail`, status file per job) | Confirm smoke outputs; sleep |
| H12:30–15:30 | M3 GRU, bootstrap CIs, sensitivity sweeps, chaos/e2e tests; GOVERNANCE.md, MODEL_CARD.md, RESULTS.md, DEMO.md (venue recalibration + judge-group reset), Devpost draft, README | LoRA training → A/B with YOLO replay running | **Sleep** |
| H15:30–17:00 | Regenerate figures, finalize docs, fix triage issues | Triage status files, flip narrator ship flag, 60-min soak in background ⚑, PERCEPTION.md, NARRATOR.md | Triage (no new code after sleep ⚑) |
| **H17:00** | **FEATURE FREEZE** → tag `demo-v1` | | |
| H17–20 ★ | Rehearsal bug fixes only | Rehearsal bug fixes only | 2 rehearsals with Wi-Fi off: live + chaos (kill narrator, bad camera, muted mic) |
| H20–21 | Buffer | Backup screen recording of full replay run | 5 slides from reports/figs |
| H21–22 ★ | Submission support | | README check, Devpost, 2-min recording, submit ⚑ |
At the venue Sunday ⚑: 10-min recalibration (`POST /api/calibrate/room`: SPL offset, 2-point lux, noise
floor); start the live pipeline at the table ≥20 min before judging (real window, not seeded); `POST
/api/demo/reset` between judge groups; demo the masked flip + obscure-face live, other states via replay with
`source=replay` visible.
Gates: any block >30 min over → take its fallback. S4 past H10:30 → replay is the main demo, live runs hybrid
with SIMULATED flags. Commit at every checkpoint.

## Priority tiers & drop order
- **P0**: replay/scenarios/seed/reset; simulator v2 + tuning; features + parity; M0/M1/M2/M2-noface +
  calibration; proof 2/3 charts; SHAP layers + override; template text + validator; PurgeGate + tests + stats;
  FastAPI/WS + console.py; live YOLO boxes/IDs, face valence, fists, dB; live capture + negative control; docs
  PRD v2, ARCHITECTURE, DATASET, API, RESULTS, DEMO; submission block.
- **P1**: narrator Stage A → LoRA; M0t; M2-noabsdB/notime; counterfactual; M3; YAMNet; Vosk;
  slump/head-in-hands/withdrawal; lux estimate; obscure-face; audit chain; acted set; 60-min soak; self-report
  labels; Overwhelmed + Tense; room-only mode; GOVERNANCE, MODEL_CARD.
- **P2**: attention slide, EmotiEffLib, perception subprocess, self-distillation.
- **Drop order**: P2 → P1 states → withdrawal → lux fit (raw brightness + `/sim/lux`) → acted set to 2 scenarios
  → M3 → Vosk (`/sim/event`) → YAMNet → LoRA (ship Stage A) → GestureRecognizer (curl heuristic) → LLM
  (template only). Never drop P0 governance/evaluation.
- **Fallbacks**: YOLO MPS → CPU imgsz 480 → PoseLandmarker(4)+IoU · face → null (NaN-native) · YAMNet →
  ai-edge-litert → `/sim/event` · Vosk → pywhispercpp tiny.en + hallucination filter → `/sim/event` · LightGBM
  libomp clash → `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=2`, `num_threads=1` → sklearn HistGradientBoosting ·
  camera/mic → hybrid/replay from golden JSONL · narrator → Llama-3.2-3B → template · empty window →
  `/api/demo/seed` (context only, tagged sim_seed).

## Judging-proof deliverables
1. **Data minimization**: PurgeGate snippet (id→`weakref.ref` callback dict, `fill(0)`, live = purged −
   finalized, PURGE_VIOLATION at tick end) ⚑; capture uses grab()/retrieve() so `frames_decoded ==
   frames_purged` and displaced slot frames are purged ⚑; YOLO outputs copied and `del results` before purge,
   with a fake-Results test ⚑; `pytest -v` for test_purge, test_no_persistence_ast, test_logfilter;
   `/governance/stats` after 60-min soak (0 violations, 0 live buffers, `retention_s`); `memory_flat.png`;
   `audit_verify.py` OK; `lsof` shows only the audit log written; demo with Wi-Fi off; GOVERNANCE.md honest
   limits (`del` is not erasure; GPU allocator caches, AVFoundation buffers, mp.Image copies can't be zeroed;
   Vosk grammar means no free transcript; keypoints never leave perception; audit log stores labels + SHAP
   summaries only, deleted at session end).
2. **Context vs face**: `accuracy_vs_history.png` (macro-F1 at 0/1/2/5/10/15 min for M0, M0t, M1, M2-noface,
   M2, M3 with session-bootstrap 95% CIs); confusion M0 vs M2; headline numbers = Frustrated_Masked recall AND
   M0 Focused→Frustrated/Tense confusion vs M2 ("Focused paradox") ⚑; `acc_vs_episode_age.png`;
   `reliability.png` + ECE; sensitivity table; legacy-vs-v2 slide; `context_blindness` replay (same
   head-in-hands pose, two contexts, two labels) ⚑; live mask flip with mask_override; counterfactual demo
   ("still Frustrated at 0.5 h because of fists + noise") ⚑; M2-notime ablation.
3. **Anti-bias, narrowed ⚑ to "decision stability when face evidence is missing or degraded"**: robustness
   curves degrading only the face channel (lux feature held fixed; lux shift reported separately) ⚑ for M0,
   M0t, M2, M2-noface; M2 vs M2-noface gap table; live obscure-face (label holds, narration switches to the
   occlusion clause); live face-detection rate at 3 light levels; cite MediaPipe model-card fairness evals;
   demographic validation listed as an explicit gap.
4. **Narrator table**: Stage A vs LoRA vs template — first-try pass rate, invented numbers, wrong qualifiers,
   false mask claims, demographic mentions (target 0), p50/p95 latency with YOLO running, distinct-2/self-BLEU.
5. **Governance framing ⚑** (PRD v2, MODEL_CARD, pitch): EU AI Act Art. 5(1)(f) prohibits emotion recognition
   in workplaces and education regardless of consent (only medical/safety carve-outs); Valence is a research
   prototype for consented self-monitoring/personal wellbeing outside those settings; `--room-only` is the
   compliance-oriented configuration; pseudonymous behavioral data is still personal data → TTL + session
   deletion.

## Deliverables (also written to `docs/DELIVERABLES.md` as a live checklist)
- Code: perception (video, audio), purge gate, aggregator/window, feature spec, simulator v2, model ladder,
  explainer + override + counterfactual, narrator (service, validator, LoRA), API, replay, console.py,
  Makefile.
- Artifacts: sim datasets (session-split), models + calibrators + feature spec + clip bounds, narrator JSONL +
  adapter, live capture + acted features-only sets, self-report labels.
- Proofs 1–5 above. Tests: purge, AST, logfilter, parity, no-lookahead, window, SHAP additivity, contracts,
  e2e, negative control, validator, perception probes.
- Docs: PRD v2, ARCHITECTURE, DATASET, API, GOVERNANCE, MODEL_CARD, RESULTS, DEMO, DELIVERABLES,
  LOCAL_SESSION_PROMPT, FRONTEND_PROMPT, README.
- Demo/submission: backup recording, Devpost text, 5 slides, Lovable frontend (outside backend scope).
- Frontend outputs (contract only): webcam view + boxes; governance purge log + counters; per-person state +
  confidence; all state probabilities; face-only vs context + override flag; narrator text + source;
  layer-grouped drivers; current state-vector values; 15-min timeline + warm-up; system status; retention
  summary; evaluation figures; `simulated_fields` provenance. Inputs: start/stop/pause, log break, replay picker,
  face/low-light toggles, reset, room-only.

## Document updates (first cloud push)
Originals are uploaded PDFs; updated versions go to `docs/*.md` (PDF export on request).
- `docs/PRD.md` (v2): same thesis/track; fully offline edge; room + person slots; laughter as an audio event;
  6 states with tiers; data-lifecycle table (captured / derived / retained / purged / TTL); face valence from
  blendshapes (no identity embedding, ephemeral IDs, keypoints stay in perception); local LLM narrator +
  validator; refined proofs (narrowed anti-bias claim); honest simulator framing; corrected EU AI Act framing +
  room-only mode; non-goals; success metrics.
- `docs/ARCHITECTURE.md` replaces the Code Architecture note; `docs/DATASET.md` replaces the mock dataset doc
  (generator v2 + parameter table, legacy compatibility); `docs/API.md`; `docs/DELIVERABLES.md`; README.
- `docs/LOCAL_SESSION_PROMPT.md` (chat prompt + ⚑ addendum merged) and `docs/FRONTEND_PROMPT.md` (chat prompt
  + `GET /api/states`, frame_w/h/mirrored, simulated_fields, reset/room-only controls).

## Apple Silicon / packaging risks
Rosetta Python (assert arm64) · OpenCV conflict (opencv-contrib-python 4.14.0.94 `--no-deps`) · pre-install
`lap`; `.env`: `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`; Wi-Fi-off smoke at H1 and H14 ·
`uv pip freeze > requirements.lock`, no upgrades after H1; pin the oldest mediapipe 1.x that passes smoke ·
torch + LightGBM libomp clash → same-process smoke test at H0 · narrator in its own venv · camera/mic
permission belongs to the launching app and fails silently (self-tests; pin devices by name) · MediaPipe Tasks
only, VIDEO mode, increasing ms timestamps · YOLO vs MLX GPU contention (narrate on triggers; YOLO → CPU 480 if
FPS drops >30%) · RAM: never train while live runs; 8 GB fallback config · thermals: 10 Hz cap, plugged in ·
cloud container is Python 3.13 x86: write 3.12-compatible code; Mac-only deps in `requirements-mac.txt`.

## Verification
| Stage | Check |
|---|---|
| Simulator | dwell/transition stats match config; same seed → same hash; sensitivity sweep printed |
| Features | `test_feature_parity` (NaN-aware, atol 1e-9), `test_no_lookahead`, `test_window` (900 s eviction, 4 writer threads, room slot inheritance) |
| Models | no session in two splits; M2 ≤ .92 and ≥ .85; M2-noface ≥ M2 − .05 (core 4); ECE drops; at 15 min M2 > M1 > M0 with non-overlapping CIs |
| Explainer | `test_shap_additivity` (1e-6); override fires within 6 ticks on frustrated_masked, never on happy/focused; counterfactual changes only the named feature |
| Governance | test_purge (id→weakref path, zeroed buffer, leaked ref → violation), fake-Results test, AST ban, logfilter, audit_verify, frames_decoded == frames_purged, 60-min soak flat memory |
| API | `test_contracts` validates every WS message against JSON Schema; explanation carries its prediction's event_id; bounded per-client queues under no-client load |
| End to end (cloud) | `test_e2e_replay`: correct label ≥80% of ticks, override fires, 0 violations, tick < 200 ms; chaos → `status` event + fallback while ticks continue |
| Sim-to-real | negative control: typing capture replay predicts Frustrated/Tense on ≤20% of ticks; ≥90% of live features inside clip bounds; acted-set majority-label agreement table |
| Perception (local) | `live_probe`: 5 fists → 5±1, typing 3 min → fist FP rate logged; 10 s slump sets flag; covered camera → face_visible 0; "great job" ×3 → 3; dB within ±3 of phone meter |
| Narrator (local) | adversarial strings ("91%", "she", "depressed", "high noise (45 dB)", mask language when face agrees) rejected; stale cached numbers rejected; A/B on 300 sheets with YOLO running |
| Live (Mac) | 60 min: ≥8 Hz, 0 violations, flat memory, narrator fallback rate logged |
| Pre-demo | `make smoke && pytest -q && python scripts/e2e_check.py --speed 20`, narrator warm-up, venue recalibration |

## ⚑ Addendum to send to the LOCAL session (also merged into `docs/LOCAL_SESSION_PROMPT.md`)
1. Step 0: record chip/RAM/Air-vs-Pro in docs/BENCHMARKS.md (8 GB → LoRA batch 1–2, 8 layers, 5 Hz loop);
   import ultralytics + lightgbm in one process and run a predict (libomp clash → `KMP_DUPLICATE_LIB_OK=TRUE`,
   `OMP_NUM_THREADS=2`); `.env` with `YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false HF_HUB_OFFLINE=1`; smoke test with
   Wi-Fi off; narrator gets its own venv `valence/narrator/.venv`; download the Llama fallback only if Qwen fails;
   pin the oldest mediapipe 1.x that passes.
2. Capture: `cap.grab()` every frame, `cap.retrieve()` only at 10 Hz; purge any frame displaced from the slot;
   emit `frames_decoded`; add `frame_w, frame_h, mirrored` to VideoObs.
3. YOLO: copy boxes/ids/keypoints with `.cpu().numpy().copy()`, `del results` before `gate.purge(frame)`;
   desk ROI + min box height 25% of frame + keep ≤3 largest tracks. Keypoints never leave perception.
4. Fists: score ≥0.7 held ≥0.5 s, ignore hands whose wrist is in the bottom 20% of the frame (keyboard zone).
   motion and slump are per-person normalized (z vs the person's own baseline).
5. YAMNet in AUDIO_CLIPS mode, called synchronously on 0.975 s rolling buffers (not AUDIO_STREAM).
   (Near-field gating of laughter/compliments is done by the cloud aggregator — just emit raw counts.)
6. New Step 3.5 right after audio — live feature capture (needs me): 3 min each of quiet typing, looking at the
   keyboard, clenching fists, slumping → `data/real/capture_*.jsonl` (features only); report fist FP/min while
   typing, slump duty cycle while typing, motion distribution in docs/BENCHMARKS.md; push and tell me.
7. Narrator: send the fact sheet as compact `key: display` lines (~120 tokens); build an mlx_lm prompt cache for
   the fixed system prompt + few-shots at startup and reuse a copy per request; max_tokens 90; measure latency
   with YOLO running on a looped replay.
8. Validator: each driver now carries `level` (high|elevated|normal|low), `effect` (supports|opposes) and
   `scope` (room|person) — check adjectives within 6 tokens of a feature mention against `level`/`effect`;
   phrase room-scope drivers as "the room"; use "Upbeat" instead of "Collaborative" when n_people == 1;
   re-validate any cached text against the current sheet. Read STATES / STATE_DISPLAY / STATE_LEXICON from
   `valence/contracts.py`; P1 adds Overwhelmed ("overwhelm", "distress") and Tense ("tense", "visibly frustrat").
9. LoRA: grammar-only targets (no self-distillation). Before sleeping, run every overnight job in smoke mode
   (LoRA `--iters 10`, A/B on 5 sheets), then launch `scripts/overnight.sh` with `set -uo pipefail`,
   independent jobs and one status file per job, under `caffeinate -dims`, plugged in.
10. After sleep: no new code — triage, flip the narrator ship flag, run a 60-min soak, rehearse.


## Build-time decisions (recorded after approval)

Changes made while building, with the reason. The rest of this plan is unchanged.

| Area | Plan said | Built / decided | Why |
|---|---|---|---|
| Mask override | M0 conf >= .55, any M0/M2 disagreement, 3 consecutive ticks | M0 reads Collaborative_Happy (positive face) while M2 reads a non-positive state; M0 conf >= .45; M2 conf >= .70; face SHAP for M0's class positive; on at >= 3 of the last 6 face-visible ticks, off at <= 1; a tick without a face holds; reset when the shown label changes | "Mask" means a positive-looking face over a non-positive state; a flat face read as Focused under Exhausted context is a face-only error, not a mask. Strict 3-in-a-row flickered on every face dropout. |
| Features | ~110, `S_rel_sess` | 114 (adds `E_cnt_now`, `face_visible_now`), `S_rel_15m` = now minus 15-min median | Keeps the feature function pure (window-only) so train/live parity is exact. |
| Feature spec file | `artifacts/feature_spec.json` | order, sets and temperatures in `artifacts/model_meta.json` | One metadata file per trained ladder. |
| Break clock | room-level | per person, presence-based (absence >= `break_absent_min`) | Matches the edge's per-person tracking; a judge who sits down starts at 0 h. |
| Withdrawal | slump + low motion + turned away | slump >= 60% of the tick and motion <= 0.06 sw/s (`valence/features/spec.py`), same rule in sim and live | Turned-away needs keypoints, which never leave perception. |
| YAMNet | AUDIO_CLIPS | AUDIO_CLIPS (edge implemented after first benchmarking AUDIO_STREAM) | Deterministic attribution of laughter to its audio step. |
| libomp | `OMP_NUM_THREADS=2` fallback | torch imported before LightGBM in live mode; LightGBM `num_threads=1`; never set `OMP_NUM_THREADS` | Measured segfaults on the demo Mac (CONTRACT_REQUESTS round 2). |
| M3 | cloud CPU, P1 | cloud CPU (torch from PyPI; the PyTorch index is blocked by the network policy) | Benchmark only; M2 stays served. |
| Simulator breaks | 1/90 min, 5-15 min | 1/120 min, 10-20 min | Long stretches without a break must occur for the fatigue logistic to matter; breaks shorter than the live 10-min rule would reset the clock in sim but not live. |
| Difficulty | tune `noise_scale` to M2 <= .92 | `noise_scale` = 1.0, untuned (M2 macro-F1 0.82 is already under the ceiling) | Anti-circularity: the headline numbers were not tuned toward. |
| OOD reporting | clipped features = `ood_features` | clip input to p1–p99, but count a feature as OOD only outside its physical domain (bounded features), never for clock features, and 50% of the training range beyond p1/p99 for db/lux/motion | Real captures had > 3 clipped features on every tick (face visible on every frame, morning time of day), so every live prediction was shrunk and the override needed a raw M2 confidence of ~0.89. |
| Replay clock | — | recordings re-stamped to the current clock (feeder) or one synthetic clock (offline); `Runtime.step(wall, mono)` | Break clocks started at the capture time, inflating `hours_since_break` to ~8.5 h; the earlier "fists -> Frustrated 60%" came from this. Corrected: Focused 23/25. |
| `full_demo` | seed 106, 3.4 h | seed 100, 2.0 h (and `overwhelmed` 4.5 -> 2.5 h) | Seed 106's Focused opening read Frustrated on 60/60 ticks; chosen by a seed x clock sweep, guarded by `tests/test_scenarios.py` (every scenario, >= 80% per segment). |
| Mic failure | — | the server retries the edge video-only and reports `mic_error` | A wrong `VALENCE_MIC` took the camera down with it. |
