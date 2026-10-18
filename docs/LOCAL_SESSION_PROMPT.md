# Local (Mac) session prompt

This is the prompt that started the LOCAL Claude session on the Mac (v1, sent in chat), followed by the v2
addendum produced by the plan's adversarial review. Contract decisions since then are in
`docs/CONTRACT_RESPONSES.md`; the shared contract is `valence/contracts.py`.

## v1 prompt

You are the LOCAL (Mac) build session for Valence, a TrojanHacks @ USC 2026 project (AI Data Governance track). Demo is Sunday morning; ~21 hours remain. I'm a solo builder running two Claude sessions: you, on my Apple Silicon MacBook, and a CLOUD session building the hardware-independent ML core (data generator, features, LightGBM models, evaluation, SHAP, purge gate, rolling window, FastAPI, replay mode, docs). You build everything that needs the camera, mic, or Apple Silicon. No frontend work.

### What Valence is
Privacy-preserving emotion inference: "the environment carries the signal, not the face." Webcam + mic + light -> local edge inference (counts behaviors, measures dB/lux) -> biometric purge gate (raw frames/audio zeroed and deleted immediately) -> 15-minute rolling state vector (5 s ticks) -> LightGBM classifier vs a face-only baseline -> SHAP drivers -> a LOCAL small LLM writes the audit explanation. Fully offline after one-time downloads. States: Focused, Frustrated_Masked, Exhausted, Collaborative_Happy (+ Overwhelmed, Tense as P1).

### File ownership
YOU own: valence/perception/**, valence/edge/**, valence/narrator/**, scripts/spikes/**, scripts/download_models.sh, scripts/calibrate_*.py, scripts/overnight.sh, requirements-mac.txt, docs/BENCHMARKS.md, docs/PERCEPTION.md, docs/NARRATOR.md, docs/CONTRACT_REQUESTS.md, tests/perception/**, tests/narrator/**, data/real/**, data/narrator/{train,valid,test}.jsonl, adapters/ (gitignored).
CLOUD owns everything else (contracts, governance, runtime, sim, features, models, explain, eval, api, requirements.txt, pyproject.toml, .gitignore, README, other docs/tests).

### Git
Branch claude/happy-ride-7w3ng3; `git pull --rebase` before every push; small commits; never commit weights, models/, adapters/, .venv or raw media.

### Privacy hard rules
No raw frames/audio/transcripts on disk, in logs or stdout (enforced by `python -m valence.governance.persistence_scan` and `tests/test_no_persistence_ast.py`). Every raw buffer: `gate.purge(buf, kind)` then `del`. Perception returns only scalars, booleans, normalized boxes; no embeddings, no re-identification; keypoints never leave perception. Compliments via Vosk restricted grammar. Narrator never mentions gender, age, race, appearance or diagnoses.

### Work order
0. Environment (Python 3.12 arm64 via uv, requirements-mac.txt, OpenCV contrib swap, model downloads, permission self-tests).
1. Spikes + benchmarks (YOLO26n-pose+ByteTrack, GestureRecognizer, FaceLandmarker, mic dB + YAMNet, Vosk, mlx_lm) -> docs/BENCHMARKS.md.
2. Video perception -> VideoObs (IDs, debounced fists, slump, head-in-hands, motion, blendshape face valence without cheekSquint, `--obscure-face`, lux regression).
3. Audio perception -> AudioObs (calibrated Leq dB, YAMNet laughter onsets, Vosk compliment hits).
4. Edge runner `valence.edge.run.start(sink, *, video, audio, obscure_face) -> stop` + features-only JSONL recorder.
5. Narrator Stage A (prompt-only service on :8081 + validator + template fallback).
6. Narrator Stage B (grammar-generated targets from cloud fact sheets, QLoRA, A/B vs Stage A vs template).
7. Calibration + acted validation set (features only).
8. Live integration + hardening (15-60 min soak, PERCEPTION.md).

Cut order if behind: LoRA -> head-in-hands/withdrawal -> Vosk -> lux regression -> YOLO (MediaPipe pose fallback). Never cut: purge gate, face valence, fist events, dB, features-only JSONL.

## v2 addendum (from the plan's adversarial review)

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
