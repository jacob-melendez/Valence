# Valence — Product Requirements Document (v2)

| | |
|---|---|
| Project | Valence |
| Event / track | TrojanHacks @ USC 2026 — AI Data Governance |
| Version | v2, 2026-10-17 (replaces the v1 PRD PDF) |
| Audience | Judges and teammates |
| Source of truth | Architecture and schedule: `docs/PLAN.md`. Field names, states and messages: `valence/contracts.py` (contract 1.0.0). Measured hardware numbers: `docs/BENCHMARKS.md`. |

**Core thesis (unchanged from v1):** *The environment carries the signal, not the face.* Emotion recognition
that reads only a face is context-blind and depends on harvesting biometric data. Valence reads the room and
the behavior over time (noise, light, time since a break, clenched fists, slumping, laughter) as well as the
face, keeps none of the raw media, and explains every prediction.

**How to read this document.** Valence is being built during the hackathon. At this update (Saturday
evening, 2026-10-17; partly updated on 2026-10-18 for the venue room baseline) the repo contains the shared
contract, configuration and governance modules with their tests; the runtime, REST/WebSocket API and console;
simulator v2 and the 9 replay scenarios; the feature code; a trained model ladder (core 4 states, untuned
`noise_scale` 1.0) with M2 served; the explainer,
template text and counterfactual; the evaluation and negative-control scripts; the Mac perception and edge
runner; and a draft narrator service. Per-item status is in `docs/DELIVERABLES.md`. A file existing does not
mean its tests pass. Where the implementation already differs from `docs/PLAN.md`, the difference is stated in
place. **No accuracy result is reported here:** every metric below is a **target**; the simulated results are
in `docs/RESULTS.md` and `docs/MODEL_CARD.md`, and items not yet produced say "TBD — produced by `<script>`". The
only measured numbers are hardware timings from `docs/BENCHMARKS.md` (Apple M3 Max, 36 GB).

---

## 1. Executive summary and problem statement

### 1.1 The problem

| Problem | What goes wrong |
|---|---|
| **The masking dilemma** | Face models map surface expressions. A forced, polite smile reads as happiness while frustration underneath is missed. |
| **Context blindness** | Head in hands can mean weeping, laughing or deep focus. A furrowed "resting focus face" reads as anger. Without environmental, temporal and behavioral context a model defaults to the wrong answer. |
| **The black box and the privacy problem** | Typical pipelines store frames and identity data indefinitely and output labels with no account of why. |

### 1.2 The solution

Valence is a hybrid individual-and-environment system that runs entirely on one laptop.

1. **Edge perception.** Webcam and microphone are processed locally. Perception turns each frame or audio
   step into scalars, flags and normalized boxes: room noise (dB), light (lux), per-person clenched-fist
   events, posture slump, head in hands, motion, a face valence score, laughter events and compliment
   phrases.
2. **Biometric purge gate.** Every raw frame and audio block is zero-filled in place right after feature
   extraction, and a weak-reference check at the end of every 5 s tick counts any buffer something still
   references as a violation.
3. **Rolling state vector.** A 15-minute window of 5 s ticks, with one **room slot** (shared signals) and up
   to three **person slots** (body signals), grouped into the three context layers plus a Facial Surface.
4. **Context model vs face model.** A LightGBM classifier on the full window (M2) is served live next to a
   face-only baseline (M0) running as a shadow, so every prediction shows what the face alone would have
   said.
5. **Explanation.** TreeSHAP drivers are grouped by layer, a template sentence is shown immediately, and a
   **local small LLM** rewrites it in plain English. A validator rejects any text that is not grounded in the
   drivers; on rejection or timeout the template text stays.
6. **Governance by construction.** Fully offline after one-time downloads; by default the only file the live
   pipeline writes is a hash-chained audit log of decisions (labels and SHAP summaries, never features or media),
   deleted when the session ends and expired by TTL.

### 1.3 What we claim and what we do not

| We claim (subject to the proofs in §4) | We do not claim |
|---|---|
| Under explicit, published behavioral assumptions, a temporal context model beats a face snapshot. | In-the-wild accuracy. The model is trained and evaluated on a simulator; in-the-wild validation is future work. |
| Raw frames and audio are zeroed and released within one processing step, and the code can show it. | That `del` or zero-filling erases every copy (native library buffers are out of reach; see §2.4). |
| Decisions stay stable when face evidence is missing or degraded. | That Valence prevents demographic bias. No demographic validation has been done; that gap is listed explicitly. |
| Every explanation is checked against the model's own drivers. | Clinical validity. Valence diagnoses nothing. |

The simulator's parameters are published (`docs/DATASET.md`, `docs/MODEL_CARD.md`). Two small sanity sets
recorded by the builder (an acted set and periodic self-reports, features only) check direction, not
accuracy.

### 1.4 Intended use and regulatory position

- **EU AI Act, Article 5(1)(f)** prohibits placing on the market, putting into service or using AI systems
  that infer the emotions of a natural person **in the workplace or in education institutions**, except for
  medical or safety reasons. The prohibition applies regardless of consent. Valence uses facial blendshapes
  (biometric data), so it is an emotion recognition system in the Act's sense.
- Outside those settings, emotion recognition is still listed as high-risk (Annex III, point 1(c)) and
  deployers must inform the people exposed to it (Article 50(3)).
- **Valence's position:** a research prototype for **consented self-monitoring and personal wellbeing outside
  workplace and education settings**. It is not a tool for employers, schools, proctoring or productivity
  scoring (see non-goals). The hackathon demo analyzes only the builder and volunteers who agree at the table.
- **`--room-only` mode** is the compliance-oriented configuration: one aggregate room label, no per-person
  inference. We do not claim it makes workplace or classroom use lawful; that would need legal review.
- **Data protection:** ephemeral IDs make the behavioral timeline pseudonymous, not anonymous. Pseudonymous
  behavioral data is still personal data, so the audit log is deleted at session end and expired by TTL, and
  the rolling window lives only in memory.

We use the EU AI Act as the design bar because it is the strictest current reference, even though the demo
takes place in the US.

### 1.5 Non-goals

- Identity recognition, face recognition, re-identification, or any identity embedding.
- Storing, recording or transmitting video, audio, transcripts or keypoints.
- Cloud inference or any network dependency at runtime.
- Workplace or classroom monitoring, proctoring, hiring, productivity or performance scoring.
- Medical or psychological diagnosis; claims of clinical validity.
- Inferring demographics or appearance (the narrator is forbidden to mention them).
- Attributing speech or laughter to a specific person (audio is a room-scope signal).
- More than 3 tracked people, multiple cameras, or tracking people across sessions.
- Building the frontend in this repo (it is built separately in Lovable; §3).

---

## 2. System architecture and data lifecycle

### 2.1 Pipeline

```
camera ─► capture: grab() every frame, retrieve() (decode) at 10 Hz ─► video perception ──┐
mic ────► audio worker: 16 kHz mono, 100 ms blocks, ~1 s analysis steps ──────────────────┤─► PURGE GATE
                                                                                           │   zero-fill in place,
         only scalars, flags and normalized boxes cross this line;                         │   weakref check per tick
         keypoints, crops, blendshapes and recognizer text stay inside perception          ▼
replay: scenario YAML | features-only JSONL | /api/demo/seed ──► aggregator: ROOM slot + PERSON slots
  ─► 5 s TickRecord per person ─► rolling window (180 ticks = 15 min) ─► 123 features ─► clip (+ OOD list)
  ─► M2 LightGBM (served, 101 features) + M0 face-only shadow ─► calibrated probabilities ─► smoothed label
  ─► TreeSHAP ─► 4 layers ─► mask-override check ─► FactSheet ─► template text (instant)
  ─► WebSocket + REST on 127.0.0.1:8000 ─► Lovable console (frontend)
  ─► audit log (labels + SHAP summaries)      ─► narrator on 127.0.0.1:8081 ─► validator ─► LLM text or template
```

One main process (FastAPI/uvicorn plus capture, perception, audio and ticker threads) and one narrator
process in its own virtual environment. Each modality can be switched independently between `live` and
`replay`; the running mode is `live`, `replay` or `hybrid`. Every record carries `source`
(`live | replay | sim | sim_seed`) and `simulated_fields`, so any simulated or operator-set value is visible.

### 2.2 Fully offline edge

| Item | Detail |
|---|---|
| One-time downloads | `scripts/download_models.sh`: YOLO26n-pose weights; MediaPipe GestureRecognizer, FaceLandmarker, PoseLandmarker lite (fallback); YAMNet audio classifier; Vosk small-en-us 0.15; Qwen3-1.7B-4bit (MLX) with Llama-3.2-3B-Instruct-4bit as fallback. Plus pinned Python packages. |
| Offline switches | `.env`: `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. |
| Network surface | API on `127.0.0.1:8000`, narrator on `127.0.0.1:8081` (loopback only). CORS: localhost and the Lovable origin. The frontend runs locally next to the backend for the demo. |
| Proof | Rehearsals and the demo run with Wi-Fi off; the `status` message carries an `offline` flag. |

### 2.3 Perception: what each sensor produces

| Sensor | Model / method | Emitted (contract) |
|---|---|---|
| Camera, people | YOLO26n-pose + ByteTrack (desk region, box height ≥ 25% of frame, ≤ 3 largest tracks) | `PersonObs.track_id`, normalized `bbox` |
| Camera, hands | MediaPipe GestureRecognizer; fist score ≥ 0.7 held ≥ 0.5 s; hands with the wrist in the bottom 20% of the frame (keyboard zone) ignored | `fist_onsets` (debounced rising edges), `fist_active` |
| Camera, posture | Pose keypoints, per-person normalized | `slump`, `head_in_hands`, `motion` (shoulder-widths/s; NaN when undefined) |
| Camera, face | FaceLandmarker blendshapes: one full-frame pass matched to people by the YOLO nose keypoint, then a per-person head-crop retry for anyone unmatched | `face_visible`, `face_valence` in [−1, 1] or NaN (§2.6) |
| Camera, light | Brightness-based lux estimate, 10 s median at 1 Hz | `VideoObs.lux`, `lux_quality` |
| Mic, level | Calibrated Leq | `AudioObs.db_spl`, `rms_power` |
| Mic, laughter | YAMNet audio classifier (AUDIO_CLIPS mode on each step's last 0.975 s): laughter is an **audio event**, debounced into onsets | `laughter_onsets` |
| Mic, compliments | Vosk with a **restricted grammar** (a fixed phrase list plus `[unk]`) | `compliment_hits` |
| Mic, speech activity | Speech probability per step | `speech_prob` |
| Capture bookkeeping | | `frame_seq`, `frames_decoded`, `frame_w`, `frame_h`, `mirrored`; audio `block_seq` |

Laughter and compliments count only when the audio step is at least 6 dB above the venue room baseline
(`near_field_db`; before the baseline is complete, 6 dB above the room's 15-minute median), so the desk area is
measured rather than the whole venue. Both are room-scope signals.

Measured on the demo Mac (`docs/BENCHMARKS.md`, 1280×720 sample images, not yet the live camera): video
perception end to end ~45 ms p50 for 2 people against a 100 ms budget at 10 Hz; Vosk restricted grammar p50
0.84 ms per 100 ms block; YAMNet ~6 ms per 1 s result (measured in AUDIO_STREAM mode).

### 2.4 Biometric purge gate

`valence/governance/purge.py` (exists, with `tests/test_purge.py`).

| Step | Mechanism |
|---|---|
| Extract | Perception computes scalars from the buffer. Detector outputs are copied (`.cpu().numpy().copy()`) and the detector's result objects deleted first, because they hold references to the input frame. |
| Purge | `gate.purge(buf, kind)` zero-fills the buffer in place (or the array that owns it, if it is a view), then registers a `weakref` callback. The caller then `del`s its own name. `kind` is `"video"` / `"audio"` for raw capture buffers and `"video_derived"` / `"audio_derived"` for copies made inside perception (RGB conversion, head crops, PCM for Vosk, the YAMNet clip). |
| Verify | At the end of every tick, `end_tick()` runs one garbage collection if needed; any purged buffer still alive is a `PURGE_VIOLATION`. |
| Count | `frames_decoded` (from capture) must equal `frames_purged` (the `"video"` count), compared with a tolerance of ≤ 2 in-flight frames or at stop. Capture decodes only at 10 Hz (`grab()` every frame, `retrieve()` at 10 Hz) and purges any frame displaced from the hand-off slot. Recognizer text cannot be zeroed (Python strings are immutable), so it is counted with `note_transcript_purged()`. |
| Enforce | `valence/governance/persistence_scan.py` (AST scan; `tests/test_no_persistence_ast.py`) bans file writes in `valence/perception/` and binary writes in `valence/edge/`. `valence/governance/logfilter.py` drops any log record that carries an array or bytes. |

**Honest limits** (also in `docs/GOVERNANCE.md`). `del` removes a name, not memory. Zero-filling first means the bytes are gone even if
the pages stay in the allocator. Copies made inside native libraries — GPU allocator caches, AVFoundation
pixel buffers, MediaPipe's internal `mp.Image` — cannot be reached from Python and are not zeroed.

### 2.5 Rolling state vector: room slot and person slots

Every 5 s the aggregator writes one `TickRecord` per person. Room fields are computed once and copied into
each person's row. A person who arrives mid-session inherits the room's history, so their window is not empty
for room signals.

| Slot | `TickRecord` fields | Layer |
|---|---|---|
| Room | `ambient_db`, `db_p90`, `light_lux`, `lux_quality`, `room_db_base`, `room_lux_base` (venue room baseline) | Ambient Physics |
| Room | `laughter_count`, `compliments_count` (near-field gated) | Somatic & Social |
| Room | `tod_h` | Time Dynamics |
| Person | `hours_since_break` (per-person, presence-based break clock) | Time Dynamics |
| Person | `clenched_fists_count`, `posture_slump`, `head_in_hands_frac`, `social_withdrawal_flag`, `motion_energy` | Somatic & Social |
| Person | `face_valence` (median over the tick, NaN if never visible), `face_visible` (fraction of the tick) | Facial Surface |
| Both | `tick`, `ts`, `t_mono`, `session_id`, `person_id`, `display_id`, `n_people`, `source`, `simulated_fields` | — |

Each person's break clock resets on `POST /api/events/break` or when that person is absent for at least
`break_absent_min` (default 10 min; shortened for demos). A logged break is a real event entered by the
operator: it is written to the audit log but not tagged in `simulated_fields`. An operator-set clock value
(`POST /api/demo/seed` with `hours_since_break`) is tagged. (`docs/PLAN.md` lists the break clock under the
room slot; the implementation and `docs/DATASET.md` keep one clock per person.)

**The three context layers plus the Facial Surface** (`contracts.LAYERS`):

| Layer | Signals | Purpose |
|---|---|---|
| Ambient Physics | noise (dB), light (lux) | Physical stressors in the room |
| Time Dynamics | hours since break, time of day | Fatigue over time |
| Somatic & Social | fists, laughter, compliments, slump, head in hands, motion, withdrawal, number of people | Behaviors that reveal inner state or social interaction |
| Facial Surface | face valence history and visibility | The face, as one input among several, so its weight is visible in every explanation |

**Window features** (~110 in the plan; 123 as implemented in `valence/features/compute.py`; order frozen as
`feature_names` in `artifacts/model_meta.json`): for noise, light, motion, slump and head in hands — current
value, mean/std/slope/max over 1, 5 and 15 min, two EWMs, a 1-vs-15-minute difference and the current value
relative to its 15-minute median; noise and light relative to the venue room baseline (dB above it, log2 of the
light ratio: now, means and a max/EWM); for fists, laughter and compliments — counts in the current tick and over
1, 5 and 15 min, and time since the last event; withdrawal now and over 5 min; number of people; hours since
break; time of day (sine/cosine); window fill ratio; face now, face visible now, face mean over 1/5/15 min,
face std over 5 min and face-visible fraction over 5 min. The served M2 reads 101 of them: **no absolute dB or lux
level**, so a loud hall is this room's normal rather than a loud, frustrated room; the absolute levels are still
computed and quoted in explanations ("average noise of 72 dB over the last 5 minutes, 6 dB above this room's
baseline"). The baseline is captured on arrival with the person sitting neutral (`POST /api/room/baseline`), or
taken automatically from the first 90 s of a session. NaN is native (no imputation). Features are clipped to the
training 1st–99th percentiles; features outside their physical domain (or, for dB/lux/motion, far outside the
training range) are reported in `data_quality.ood_features`, and more than three of them lowers the confidence
(probabilities are pulled toward uniform) and is narrated.

### 2.6 Facial Surface: face valence without identity

| Rule | Design |
|---|---|
| Input | FaceLandmarker blendshape scores: one full-frame pass, each face matched to a person by the YOLO nose keypoint; anyone left unmatched gets a second try on a per-person head crop (benchmarks showed the full-frame detector misses small faces; a crop belongs to exactly one person). |
| Output | One scalar `face_valence` in [−1, 1], **relative to that person's neutral baseline** (median of their first 20 s of visible face), so ~0 means "their neutral". NaN whenever `face_visible` is false. |
| No identity | No face embeddings, no recognition, no matching against anything stored. Landmarks and blendshapes never leave perception. |
| Ephemeral IDs | `track_id` 1..N by first appearance, shown as "Person #N". Reused only if the person reappears within 3 s and 0.25 frame-widths; otherwise the ID is released and the person's baselines are forgotten. |
| Missing face | NaN flows through the features; M0 reports `face_only.label = null` and the narrator uses its occlusion clause. `--obscure-face` (live) and `POST /api/sim/face_occlusion` exercise this path in the demo. |

### 2.7 Emotional states

Single source of truth: `CORE_STATES`, `P1_STATES`, `STATE_DISPLAY`, `STATE_TIER`, `STATE_LEXICON` in
`valence/contracts.py`. The active set is chosen by `VALENCE_STATE_SET=core|all` (cap: 6). The frontend reads
it from `GET /api/states` and the `hello` message.

| State id | Display (`STATE_DISPLAY`) | Tier | Face (simulator) | Context signature (simulator) |
|---|---|---|---|---|
| `Focused` | Focused | P0 | −0.15 (looks angry) | quiet 40–55 dB, normal light, 1–3 h since break, some head in hands ("thinking") |
| `Frustrated_Masked` | Frustrated / Masked Stress | P0 | +0.45 (forced smile) | loud, dim, 3.5–6 h since break, fists ~0.8/min |
| `Exhausted` | Exhausted | P0 | −0.1 (flat) | silent, very dim, 5–8 h, slump and withdrawal, frequent head in hands |
| `Collaborative_Happy` | Collaborative / Happy | P0 | +0.55 | lively 60–75 dB, bright, fresh break, laughter and compliments, usually ≥ 2 people |
| `Overwhelmed` | Overwhelmed / Distressed | P1 | hidden 40–70% of the time (hands), −0.3 | frequent head in hands, slump, 3–7 h, 60–75 dB |
| `Tense` | Tense / Visible Frustration | P1 | −0.45 (face agrees) | fists, 65–80 dB, often ≥ 2 people, high motion |

P1 states land only if the core four pass live end-to-end and the negative control (schedule gate H10:30);
they are the first thing cut.

**Display-label rules** (`contracts.display_label`; the prediction carries the result as `label_display`):

| Condition | `label_display` |
|---|---|
| `Frustrated_Masked`, core set, `mask_override` false | "Frustrated" |
| `Frustrated_Masked`, core set, `mask_override` true | "Frustrated / Masked Stress" |
| `Frustrated_Masked`, `Tense` in the active set | "Frustrated / Masked Stress" (always) |
| `Collaborative_Happy`, `n_people` < 2 | "Upbeat" (a lone person is never "Collaborative") |
| Any other case | `STATE_DISPLAY[label]` |

Why: with only the core four, `Frustrated_Masked` also covers unmasked frustration (30% of its simulated
episodes), so calling it "masked" is only justified when the override detector fires. With P1, it becomes
mask-only and `Tense` takes unmasked frustration. `Tense` and `Frustrated_Masked` differ mainly in the face,
so the no-face model will confuse that pair; we report it as "the face is informative but not sufficient".

### 2.8 Prediction engine

| Model | Inputs | Role |
|---|---|---|
| M0 | face snapshot | Baseline; runs live as a shadow (`face_only`) |
| M0t | face history | Baseline (P1) |
| M1 | context snapshot (current values of the context channels; no face, no history) | Baseline |
| M2-noface | full window without face | Ablation: how much the face adds |
| M2-abs | full window with absolute dB/lux levels instead of room-relative ones | Ablation: what absolute levels cost in a loud hall (P1) |
| M2-notime | full window without time features | Ablation: is "hours since break" doing all the work? (P1) |
| **M2** | **full window: context + face history** | **Served live** |
| M3 | GRU(64) + attention over the window | Benchmark only (P1) |

- LightGBM, session-level 70/15/15 split; the validation split is halved into an early-stopping half and a
  calibration half.
- Displayed `confidence` is a temperature-scaled softmax, **unsmoothed**, labeled "sim-calibrated" because
  the calibration data is simulated. The displayed label is smoothed (EMA plus a 2-tick hysteresis).
  `warming_up` is true for the first 60 s.
- **Mask override (plan rule).** `mask_override` is true only when: M0's label differs from M2's; M0
  confidence ≥ 0.55; M2 confidence ≥ 0.70; the Facial Surface SHAP for M0's class inside M2 is positive (M2
  sees the smile too); all of this for 3 consecutive ticks; never in the first 2 minutes.
- **Mask override (as implemented, v1).** `valence/models/served.py` and `valence/models/predictor.py`
  differ from the plan rule: M0 confidence ≥ 0.45; M0's label must be `Collaborative_Happy` (a positive face)
  while M2's is not; M2 confidence ≥ 0.70 and the face-SHAP check as above; the flag turns on when the
  condition held on ≥ 3 of the last 6 face-visible ticks and off at ≤ 1; never in the first 2 minutes. The
  positive-face requirement keeps a furrowed Focused face from counting as a "mask". Which rule is final is
  an open item (§4.7).
- **Counterfactual.** `GET /api/explain/{event_id}/counterfactual?feature=<name>&value=<number>` re-runs M2
  with exactly one feature changed (for example `feature=hours_since_break&value=0.5`). It answers "is this
  label just the number the operator typed?" by showing whether the label holds without that driver.
  Implemented in `valence/models/served.py`; it currently re-runs on the person's current window, not on the
  stored feature vector of that `event_id`.

### 2.9 Explanation and narrator

1. **SHAP.** LightGBM `pred_contrib` (TreeSHAP), predicted-class slice, summed per layer; top drivers carry
   `feature`, `display`, `value`, `unit`, `shap`, `level` (`high|elevated|normal|low`), `effect`
   (`supports|opposes`) and `scope` (`room|person`).
2. **FactSheet** (`contracts.FactSheet`): `event_id`, `person`, `label`, `label_display`, `confidence`,
   `confidence_display`, `probabilities`, `window_fill_min`, `n_people`, `face_only{label, confidence,
   face_valence}`, `face_visible`, `mask_override`, `layers`, `data_quality{face_visible_frac, lux,
   ood_features}`.
3. **Template text** is produced from the FactSheet immediately and sent inside the `prediction` message
   (`explanation.source = "template"`).
4. **Local narrator** (`POST 127.0.0.1:8081/narrate`): Qwen3-1.7B-4bit via mlx-lm, thinking disabled, a
   compact ~120-token rendering of the FactSheet, a cached system prompt with few-shot examples, at most 90
   output tokens. Fallback model: Llama-3.2-3B-Instruct-4bit. Deadline 2.5 s (`narrator_deadline_s`).
   Triggers: label change, confidence-band change, or every 30 s; only the latest request per person is kept.
   Stage A is prompt-only; Stage B (QLoRA on grammar-generated targets) ships only if its validator pass rate
   is at least Stage A's and its p95 latency is ≤ 2.5 s **measured with YOLO running**.
5. **Faithfulness validator** (rejects the text, then one retry, then template):

| Check | Rule |
|---|---|
| Numbers | Every number in the text appears in the FactSheet |
| State | Names the predicted state (stems from `STATE_LEXICON`; `Tense` stems tested before `Frustrated_Masked`'s) |
| Drivers | Mentions at least 2 of the top 3 drivers and nothing outside the top 5 |
| Qualifiers | An adjective within 6 tokens of a feature must match that driver's `level` / `effect` |
| Scope | Room-scope drivers are phrased as "the room" |
| Mask | Mask language only when `mask_override` is true |
| People | "Person #N" only; no pronouns; never mentions demographic attributes, appearance or diagnoses |
| Length | 20–90 words |
| Cache | Cached text is re-validated against the current FactSheet |

The draft narrator (`valence/narrator/`, Stage A) implements the number, state, driver, mask, banned-term
and length checks; the qualifier, room-scope and cached-text checks are still open (`docs/DELIVERABLES.md`
A8). It also differs in two parameters: up to 3 attempts (temperature 0.7, 0.4, then greedy) instead of one
retry, and `max_tokens` 110 instead of 90.

The result arrives as an `explanation` message with the same `event_id` and `source` = `llm`,
`llm_retry` or `template`.

Measured (`docs/BENCHMARKS.md`, without YOLO running): Qwen3-1.7B-4bit generates 187 tok/s, 536 ms wall time
for a 55-token answer, 1.27 GB peak memory; Llama-3.2-3B fallback 1108 ms for 89 tokens. Latency with YOLO
running: TBD — produced by the narrator A/B run (local session, `docs/NARRATOR.md`).

### 2.10 Data lifecycle

| Data | Captured / derived from | What is kept or passed on | Purged how | Where it lives | How long |
|---|---|---|---|---|---|
| Raw video frame | Camera; decoded at 10 Hz | Only scalars, flags, normalized boxes (`VideoObs`) | `gate.purge(frame, "video")` zero-fill + `del`; detector results deleted first; displaced frames purged; `frames_decoded == frames_purged` | RAM, perception thread | One processing step (~45 ms p50 measured for 2 people on sample images) |
| Head crop | Cut from the frame per person | Blendshape scores (inside perception only) | Purged as raw video | RAM, perception | Same step |
| Body keypoints | YOLO26n-pose | Slump, head in hands, motion, head-crop box | Dropped after the frame; **never leave perception** (not in `VideoObs`, never on the WebSocket) | RAM, perception | One step |
| Face landmarks and blendshapes | FaceLandmarker | One `face_valence` scalar | Discarded after the scalar is computed; no embeddings | RAM, perception | One step |
| Neutral face baseline | Median of a person's first 20 s of visible face | Reference for `face_valence` | Forgotten when the ephemeral ID is released | RAM, perception | While the track lives (+3 s reuse window) |
| Raw audio block | Mic, 16 kHz mono, 100 ms blocks | `db_spl`, `rms_power`, `speech_prob`, `laughter_onsets`, `compliment_hits` (`AudioObs`) | `gate.purge(block, "audio")` | RAM, audio worker | ~1 s rolling analysis buffer |
| Recognizer text | Vosk restricted grammar | A count of matched phrases | Discarded inside perception, counted as `transcripts_purged`. **No free transcript ever exists**: the grammar can only output its fixed phrases or `[unk]` | RAM, perception | One audio step |
| Observations | Perception output | Aggregated into ticks | Released after aggregation | RAM, main process | Until the next 5 s tick |
| Venue room baseline | Room dB and lux samples, ~90 s on arrival (or the first 90 s of a session) | Median dB, dB IQR, median lux: room constants, no media and nothing per person | Replaced by the next capture or reset; audited as `room_baseline`, `room_baseline_captured`, `room_baseline_reset` | RAM, main process | Until reset, a recording replay or process exit |
| `TickRecord` / rolling window | Aggregator | 123 features per tick (101 read by M2); `state_vector` messages | Evicted after 15 min; a person's rows dropped after `person_drop_min` (15 min) absence; cleared by `POST /api/demo/reset` | RAM, main process (not written to disk) | ≤ 15 min |
| Predictions, SHAP, FactSheets, explanation text | Models, explainer, narrator | `prediction`, `explanation` messages on loopback | Not persisted (except the audit summary below) | RAM; loopback HTTP / WebSocket | Latest per person |
| **Audit log** | Decisions | Labels, confidences, SHAP layer summaries, operator actions (seed, simulated inputs, calibration); periodic purge-counter entries are planned. Strings > 200 chars, NaN and deep nesting are refused, so free text cannot be logged | Hash-chained (`GET /api/governance/verify`; offline `scripts/audit_verify.py` planned); file **deleted at session close**; files older than `audit_ttl_s` (86 400 s) deleted at startup | `var/audit/<session_id>.jsonl` — the only file the live pipeline writes by default | Session; at most 24 h after a crash |
| Application logs | Code | Text only | `RawMediaFilter` drops records carrying arrays or bytes | stderr | Process lifetime |
| Features-only recordings (opt-in) | `valence.edge.run --jsonl` | `VideoObs` / `AudioObs` as JSONL, no media | — | `data/real/` (builder's calibration capture and acted set) | Kept as evaluation data |
| Self-report labels (P1) | `POST /api/label` | State label + time + that person's latest numeric `TickRecord` (no media); audited as `self_report` | — (not deleted at session close; not yet listed by `GET /api/retention`) | `var/labels/<session_id>.jsonl` (gitignored; builder only) | Kept as evaluation data |
| Frontend camera preview | Browser `getUserMedia` | Nothing | Never recorded, stored or uploaded | Browser | Display only |

`GET /api/retention` returns this policy at runtime; the `governance` message carries `retention_s`.

### 2.11 Room-only mode

`--room-only` (config `room_only`, `POST /api/mode {"room_only": true}`, reported in `hello.room_only`)
produces one aggregate label for the room and no per-person inference. It is the configuration to start from in any compliance review (§1.4).

---

## 3. Product features and UI scope (what the backend exposes)

The judge-facing "Valence Audit Console" is built separately in Lovable (`docs/FRONTEND_PROMPT.md`) and runs
locally next to the backend. This repo provides only the contract below; full schemas are in `docs/API.md`
and `contracts.json_schemas()`.

### 3.1 Outputs

| UI element | Backend source | Key fields |
|---|---|---|
| Live view with boxes ("Person #1") | `tracks` (≤ 10 Hz) — boxes only; the browser shows its own camera preview | `persons[{id, label, bbox, fist_active, slump}]`, `frame_w`, `frame_h`, `mirrored`, `capture_ts_ms` |
| Governance purge log and counters | `governance` (every tick), `GET /api/governance/stats`, `/audit?since=`, `/verify` | `message`, `frames_decoded`, `frames_purged`, `audio_blocks_purged`, `transcripts_purged`, `violations`, `live_raw_buffers`, `audit_hash`, `retention_s` |
| Detected state and confidence | `prediction` (1 Hz per person) | `label`, `label_display`, `confidence` |
| All state probabilities | `prediction` | `probabilities` (4 or 6 states) |
| Face-only vs context, override flag | `prediction` | `face_only{label, confidence}` (null when no face), `mask_override` |
| Audit explanation and its source | `prediction.explanation`, then `explanation` with the same `event_id` | `text`, `source` (`template`, `llm`, `llm_retry`), `latency_ms` |
| Drivers grouped by layer | `prediction` | `layers[{name, contribution, drivers[...]}]` |
| Current state-vector values | `state_vector` (5 s per person) | `ambient_db`, `light_lux`, `hours_since_break`, `time_of_day`, `clenched_fists_5m`, `compliments_15m`, `laughter_15m`, `posture_slump`, `head_in_hands`, `social_withdrawal`, `face_visible`, `facial_valence`, `window_fill_min` |
| 15-minute timeline and warm-up | `GET /api/history?minutes=15`, then live messages | `window_fill_min`, `warming_up` |
| Data provenance | every `state_vector` and `prediction` | `source`, `simulated_fields` |
| Data quality | `prediction` | `data_quality{face_visible_frac, lux, ood_features}` |
| System status | `status` | `mode`, `camera`, `mic`, `fps`, `loop_latency_ms`, `narrator`, `offline`, `uptime_s`, `degraded`, `purge_violation` |
| State list | `GET /api/states`, `hello.states` | `{id, display, tier}` |
| Retention summary | `GET /api/retention` | |
| Venue room baseline | `GET /api/room/baseline`, `/api/health` → `room_baseline` | `mode` (auto / capturing / captured), `db_median`, `db_iqr`, `lux_median` |
| Evaluation results | `GET /api/eval/summary`, `GET /api/eval/figures/{name}.png` | `reports/metrics.json`, `reports/figs/` (produced by `valence/eval/evaluate.py`) |
| Counterfactual | `GET /api/explain/{event_id}/counterfactual?feature=<name>&value=<number>` | original and counterfactual label and confidence, `label_changed` |

### 3.2 Inputs

| Control | Endpoint |
|---|---|
| Start / stop / pause (privacy kill switch) | `POST /api/control/{start\|stop\|pause}` |
| Log a break | `POST /api/events/break` |
| Reset people between visitors | `POST /api/demo/reset` |
| Replay a scenario | `GET /api/scenarios`, `POST /api/replay/{scenario}` |
| Replay a features-only recording (camera/mic fallback) | `GET /api/recordings`, `POST /api/replay_recording/{name}` |
| Return to live sensors | `POST /api/live` |
| Simulate face hidden / low light | `POST /api/sim/face_occlusion`, `POST /api/sim/low_light` `{"enabled": bool}` |
| Room-only mode | `POST /api/mode {"room_only": bool}` |
| Venue calibration | `POST /api/calibrate/room` (SPL offset, 2-point lux, noise floor) |
| Capture the venue room baseline (on arrival, sitting neutral) | `POST /api/room/baseline {"seconds": 90}`; `POST /api/room/baseline/reset` returns to automatic |
| Seed an empty window | `POST /api/demo/seed` (context only, tagged `sim_seed`) |
| Self-report label (P1) | `POST /api/label {state}` |

Every control call is written to the audit log. Inputs that inject data that was not sensed (`sim/*`,
`demo/seed`, including a seeded `hours_since_break`) are also tagged in `source` / `simulated_fields`. A
logged break, venue calibration and the room baseline are real events or measurements, so they are audited but not tagged
(`docs/API.md` §6 and §8).

### 3.3 Example audit explanation (illustrative format, not a result)

The v1 example is kept, in v2 form. The numbers are placeholders (the same ones as the frontend mock data and
the `prediction` example in `docs/API.md` §3.7); the sentence shape is the one `valence/explain/template.py`
produces.

> **Person #1: Frustrated / Masked Stress — 88%** (face-only: Collaborative / Happy; context corrected the
> facial mask)
>
> *Template:* "Model predicted Frustrated / Masked Stress (88%) for Person #1. The face alone suggests
> Collaborative / Happy, but 3 clenched-fist events in the last 5 minutes, average noise of 72 dB over 15
> minutes and 4.2 hours since the last break point the other way. Context corrected the facial mask."

The label reads "Frustrated / Masked Stress" here only because `mask_override` is true (§2.7). The noise
driver has `scope: room`, so an LLM narration must phrase it as "the room" (see the `explanation` example in
`docs/API.md` §3.8).

---

## 4. Hackathon test plan and pitch strategy

### 4.1 Data used for training and evaluation

| Set | What it is | Used for | Status |
|---|---|---|---|
| Simulator v2 | ~300 sessions × 1–3 people × 60–120 min at 5 s (~500k person-ticks), seeded. Per-person semi-Markov hidden state (dwell 4–10 min, minimum 2 min), fatigue tilt after ~3 h, Poisson breaks, room dB that lags occupants' states, ±12 dB per-session mic offset (plan range 12–15 dB), a random venue hall level per session (40–75 dB, power-summed, with the arrival baseline an operator would capture), face missing on 5–40% of ticks, 2% per-modality dropout | Training, calibration, main evaluation | Generator in the repo (`valence/sim/generator.py`, `params.py`). The served models were trained on 300 sessions at the untuned `noise_scale` 1.0 (`artifacts/model_meta.json`), on the simulator refit to capture run 1; corpus statistics TBD |
| Legacy | The original mock generator: one random state per 5 s row (nothing for a window to learn) and context ranges that barely overlap | "Why we rebuilt the data" slide | Generator in the repo (`valence/sim/legacy.py`); slide result in `reports/metrics.json` (`legacy_mock`, from `valence/eval/evaluate.py`) |
| Live feature capture | Builder, 3 min each: quiet typing, looking at the keyboard, clenching fists, slumping (features only); plus audio | Sets the simulator's false-positive rates and motion/slump scales from measurements instead of guesses; negative control | Done: run 1 (typing, fists, baseline, covered camera) set the simulator's scales; runs 2–3 (keyboard gaze, slump, audio) checked them without a refit (`data/real/capture_*.jsonl`, `docs/BENCHMARKS.md`, `docs/MODEL_CARD.md` §5) |
| Acted set | 6 scenarios × 3 min, plus face-detection rate at 3 light levels (features only) | Sanity check (majority-label agreement) | TBD — scored by `valence/eval/negative_control.py --expect <state>` (the plan's `score_acted.py` is folded into it) |
| Self-report | Builder's own label every 10 min during live runs | Sanity check | P1; endpoint exists (`POST /api/label`); labels TBD |
| Scenarios | Replay YAML: one per state, `context_blindness`, `full_demo`, `loud_venue` (a 72 dB hall) | Demo, end-to-end tests, override validation | 9 files in `scenarios/`; `tests/test_scenarios.py` |

**Anti-circularity.** Simulator difficulty is one `noise_scale`, tuned against a single criterion that is not
the headline result: M2 macro-F1 ≤ 0.92 (no ceiling effect) and a match between live and simulated feature
variance. It is then frozen. M0 and M1 are reported wherever they land, with a ×0.5 / ×1 / ×1.5 sensitivity
table and the legacy slide as evidence that the ordering is structural. The full parameter table is
published in `docs/DATASET.md` and `docs/MODEL_CARD.md`. Not done yet: the tuning script
(`valence/sim/tune_difficulty.py`) does not exist, so the v1 models use the untuned `noise_scale` 1.0.

### 4.2 The proofs

**Proof 1 — Data minimization (the code).**
PurgeGate snippet (zero-fill, weakref callback dict, violation at tick end) and capture with
`grab()`/`retrieve()`; `pytest -v tests/test_purge.py tests/test_no_persistence_ast.py tests/test_logfilter.py`
(including a test that holding a detector's results leaks the frame and is caught); `/api/governance/stats`
after a 60-minute soak (target: 0 violations, 0 live raw buffers); `memory_flat.png`; `audit_verify.py` OK;
`lsof` showing only the audit log being written; demo with Wi-Fi off; the honest limits from §2.4.

**Proof 2 — Context vs face (the evaluation), on simulated data.**

| Evidence | Detail |
|---|---|
| Headline metric 1 | **`Frustrated_Masked` recall**, M0 vs M2 (the masking dilemma) |
| Headline metric 2 | **The "Focused paradox"**: share of true-`Focused` ticks that M0 labels `Frustrated_Masked` (or `Tense`), vs M2 (context blindness) |
| `accuracy_vs_history.png` | Macro-F1 at 0, 1, 2, 5, 10, 15 min of history for M0, M0t, M1, M2-noface, M2, M3, with session-bootstrap 95% CIs |
| Other figures | Confusion matrices M0 vs M2; `acc_vs_episode_age.png`; `reliability.png` + ECE; sensitivity table; M2-notime ablation; legacy vs v2 slide |
| `context_blindness` replay | The same head-in-hands posture in two different contexts produces two different labels |
| Live mask flip | `mask_override` fires and the explanation says why |
| Counterfactual | E.g. "still Frustrated at 0.5 h since a break, because of fists and noise" |

**Proof 3 — Decision stability when face evidence is missing or degraded** (narrowed from v1's "anti-bias").
Robustness curves that degrade only the face channel (missing rate and noise), with the lux feature held
fixed, for M0, M0t, M2 and M2-noface; lux shifts reported separately; an M2 vs M2-noface gap table; live
obscure-face (the label holds and the narration switches to its occlusion clause); live face-detection rate
at 3 light levels; MediaPipe model-card fairness evaluations cited. **Demographic validation is an explicit
gap**, not a claim.

**Proof 4 — Narrator faithfulness.** Template vs Stage A vs Stage B: first-try validator pass rate, invented
numbers, wrong qualifiers, false mask claims, demographic mentions (target 0), p50/p95 latency with YOLO
running, distinct-2 and self-BLEU for variety.

**Proof 5 — Governance framing.** §1.4: Article 5(1)(f) applies regardless of consent; Valence is a research
prototype for consented self-monitoring outside workplace and education; `--room-only` is the
compliance-oriented configuration; TTL and session deletion because pseudonymous data is personal data.

### 4.3 Success metrics (targets; results, where produced, are in `docs/RESULTS.md` and `docs/MODEL_CARD.md`)

| Area | Metric | Target | Produced by |
|---|---|---|---|
| Model | M2 macro-F1, held-out simulated sessions | 0.85–0.92 | `valence/eval/evaluate.py` → `reports/metrics.json` (produced) |
| Model | Ordering at 15 min of history | M2 > M1 > M0, non-overlapping 95% CIs | `evaluate.py` (produced) |
| Model | Face dependence (core 4) | M2-noface ≥ M2 − 0.05 | `evaluate.py` (produced) |
| Model | Headline: `Frustrated_Masked` recall, Focused-paradox confusion | M2 better than M0 on both; values reported as measured | `evaluate.py` (produced) |
| Model | Calibration | ECE lower after temperature scaling | `evaluate.py` (produced) |
| Explainer | SHAP additivity | Within 1e-6 | `tests/test_served.py::test_shap_additivity` (test exists) |
| Explainer | Mask override | Fires within 6 ticks on `frustrated_masked`; never on `collaborative_happy` or `focused` | `tests/test_served.py::test_mask_override_fires_on_masked_and_not_elsewhere` (test exists) |
| Explainer | Counterfactual | Changes only the named feature | `tests/test_served.py::test_counterfactual_changes_only_the_named_feature` (test exists) |
| Sim-to-real | Negative control: replay of typing capture | ≤ 20% of ticks predicted `Frustrated_Masked`/`Tense` | `valence/eval/negative_control.py` (passes, in-sample; `docs/MODEL_CARD.md` §5) |
| Sim-to-real | Venue shift: Focused misread as negative from the session's own room to a loud hall | Reported as measured (simulated) | `evaluate.py` `venue_shift` (produced; `docs/MODEL_CARD.md` §5) |
| Sim-to-real | Live features inside clip bounds | ≥ 90% | `data_quality.ood_features` logs (TBD) |
| Sim-to-real | Acted set | Majority-label agreement table (reported, no threshold) | `negative_control.py --expect` (TBD) |
| Governance | Purge | 0 violations, 0 live raw buffers, `frames_decoded == frames_purged` over 60 min | `scripts/mem_soak.py`, `/api/governance/stats` (TBD) |
| Governance | Audit chain | Verifies; tampering detected | `tests/test_audit.py` (exists), `GET /api/governance/verify`, `scripts/audit_verify.py` (TBD) |
| Governance | Persistence | Only the audit log is written; works with Wi-Fi off | `lsof`, rehearsal (TBD) |
| Runtime | End-to-end replay | Correct label on ≥ 80% of ticks; override fires; tick < 200 ms | `tests/test_scenarios.py` (label ≥ 80% per segment after a 2-min grace, override on masked segments only: passes on all 9 scenarios); `tests/test_e2e_replay` and `scripts/e2e_check.py` (TBD) |
| Runtime | Live loop | ≥ 8 Hz for 60 min, flat memory | 60-min soak (TBD) |
| Runtime | Chaos | Narrator killed / camera lost / mic muted → `status` event and fallback; ticks continue | chaos tests and rehearsal (TBD) |
| Perception | Probes | 5 fists → 5 ± 1; 10 s slump sets the flag; covered camera → `face_visible` 0; dB within ±3 of a phone meter | `live_probe` (local, TBD) |
| Narrator | Faithfulness | 0 demographic mentions, 0 invented numbers after validation | narrator A/B (TBD) |
| Narrator | Latency | p95 ≤ 2.5 s with YOLO running | narrator A/B (TBD) |

### 4.4 Demo plan (Sunday)

1. **Venue setup (~10 min):** `POST /api/calibrate/room` (SPL offset, 2-point lux, noise floor), then capture the
   venue room baseline sitting neutral (`POST /api/room/baseline {"seconds": 90}`, check `GET /api/room/baseline`
   reads `captured`). Start the live pipeline at the table at least 20 minutes before judging so the 15-minute
   window is real, not seeded.
2. **Governance first:** counters with `frames_decoded == frames_purged`, 0 violations, 0 live raw buffers,
   `offline: true`; the purge gate code and its tests.
3. **Live:** the masked flip (face-only says happy, context and `mask_override` say otherwise) and
   obscure-face (label holds, narration switches). Any simulated input, such as a sim overlay or a seeded
   break clock, shows in `simulated_fields`; every operator action, including a logged break, is in the audit
   log.
4. **Replay** for the other states and `context_blindness`, with `source=replay` visible.
5. **Counterfactual** on the current event.
6. **Evidence:** figures from `reports/figs/` (§4.2).
7. `POST /api/demo/reset` between judge groups.

Two rehearsals with Wi-Fi off are planned, including chaos (kill the narrator, bad camera, muted mic). A backup
screen recording of a full replay run is made before judging, for the case where live capture fails.

### 4.5 Pitch structure (5 slides, figures from `reports/figs/`)

1. Problem and thesis: the masking dilemma and context blindness, with the v1 mock data's flaw (legacy slide).
2. Architecture and data lifecycle: purge gate, what is kept, what is never kept.
3. Context vs face: `accuracy_vs_history.png` and the two headline metrics.
4. Missing or degraded face: robustness curves and the M2 vs M2-noface gap.
5. Governance and limits: Article 5(1)(f), room-only mode, simulator-trained, not clinically validated,
   demographic validation and in-the-wild validation as future work.

### 4.6 Risks and fallbacks

| Risk | Fallback |
|---|---|
| Live sensors fail or late integration | Hybrid or replay from golden features-only JSONL, flagged `source=replay` / SIMULATED |
| Face not detected | NaN-native; M0 reports null; M2 continues on context |
| YOLO too slow | CPU at imgsz 480 → PoseLandmarker + IoU tracking |
| YAMNet / Vosk fail | Alternative runtime → `POST /api/sim/event` (flagged simulated) |
| Narrator slow or down | Llama-3.2-3B → template text (always available) |
| Venue noise far from training | Venue room baseline captured on arrival; room-relative features (no absolute dB/lux in the model); venue-randomized training; M2-abs ablation and venue-shift evaluation; rollback to the previous model (`VALENCE_ARTIFACTS_DIR=artifacts/legacy`) |
| Empty window at demo start | Start early; `/api/demo/seed` (context only, tagged `sim_seed`) as last resort |
| Behind schedule | Drop order: P2 → P1 states → withdrawal → lux fit → acted set to 2 scenarios → M3 → Vosk → YAMNet → LoRA → GestureRecognizer → LLM (template only). Governance and evaluation (P0) are never dropped. |

### 4.7 Known gaps and open items

- Model results (simulated) are in `reports/metrics.json` and `docs/RESULTS.md`, not in this document; the
  narrator A/B is TBD.
- Difficulty tuning (`valence/sim/tune_difficulty.py`) is not done. The capture-based simulator update is done
  (refit to capture run 1; runs 2–3 checked it, `docs/MODEL_CARD.md` §5).
- The venue room baseline assumes the arrival capture is neutral; a sustained forced-smile capture and a fix for
  the held-slump motion artifact (perception) are still open (`docs/MODEL_CARD.md` §7).
- Mask-override rule: the implementation (§2.8, "as implemented") differs from the plan rule. Pick one and
  align `docs/PLAN.md`, this document and the code.
- Live-camera perception timings are measured (edge runner 45–57 ms/frame at 10.0 Hz, `docs/BENCHMARKS.md`); the
  60-min soak is not.
- YAMNet mode (resolved): the edge runs AUDIO_CLIPS on each step's last 0.975 s, as planned; the only YAMNet
  timing in `docs/BENCHMARKS.md` was measured in AUDIO_STREAM mode.
- Room-only mode switches off per-person labels, but perception still runs per person (boxes, face valence)
  and the room record uses the median face. Switching off per-person face analysis in room-only mode, and
  which model should serve the room label, are open (`docs/API.md` §4.4, `docs/ARCHITECTURE.md` §16).
- No demographic validation and no in-the-wild data; both are future work.

---

## 5. What changed from v1 and why

| Topic | v1 | v2 | Why |
|---|---|---|---|
| Thesis, track, structure | Context over face; AI Data Governance; 4 sections | Unchanged | Still the core idea |
| Deployment | "Local edge inference" | Fully offline after one-time downloads; loopback-only services; Wi-Fi-off demo | A governance claim needs no network path at all |
| Purge | `del frame` and "instantly purge" | Zero-fill in place + weakref check + violation count + `frames_decoded == frames_purged`; honest limits stated | `del` removes a name, not the bytes |
| State vector | One vector | Room slot + up to 3 person slots; room history inherited by newcomers | Rooms have several people; noise and light are shared, bodies are not |
| Laughter | A "keyword" | An audio event (YAMNet), near-field gated; compliments via a restricted grammar | Laughter is not a word; a restricted grammar means no free transcript exists |
| Room noise and light | Absolute dB and lux | Relative to the venue room baseline captured on arrival; absolute levels only quoted in explanations | A loud demo hall must not read as a loud, frustrated room |
| States | 4 | 4 core (P0) + `Overwhelmed`, `Tense` (P1), cap 6, config-driven | Unmasked frustration and distress needed their own classes |
| Display labels | "Frustrated" in the text, "Frustrated / Masked Stress" in the label | `display_label` rules: "Frustrated" unless the override fires (core set); "Upbeat" for one person | The label must not claim masking without evidence, or collaboration with one person |
| Layers | 3 context layers | 3 context layers + Facial Surface | The face is an M2 input; its weight must be visible to explain an override |
| Face | "Facial valence snapshot" | Blendshape valence relative to a personal neutral baseline; full-frame pass with a per-person head-crop retry; no identity embeddings; ephemeral IDs; keypoints stay in perception | Resting faces differ between people; minimize identity data |
| Explanation | Template pseudocode over raw rows | TreeSHAP drivers by layer → template → local LLM → faithfulness validator → template fallback | An explanation must reflect the model's actual reasons, and an LLM must not invent any |
| Confidence | "88% confidence" | Temperature-scaled, unsmoothed, labeled "sim-calibrated" | Calibration was fit on simulated data |
| Audit log | "Natural language audit log" | Hash-chained decision log: labels + SHAP summaries only; deleted at session end; TTL | Pseudonymous behavioral data is personal data; tamper evidence |
| Training data | i.i.d. mock CSV, one random state per row, near-separable context | Simulator v2 with temporal dynamics, noise, missing modalities, multi-person, published parameters; legacy kept for comparison | A 15-minute window cannot help on i.i.d. rows, and ~100% separable classes prove nothing |
| Proof 1 | Show `del frame` | Purge gate, tests, soak, audit verification, `lsof`, Wi-Fi off | Evidence instead of a snippet |
| Proof 2 | "Higher accuracy over time in-the-wild" | On published simulator assumptions; headline metrics `Frustrated_Masked` recall and the Focused paradox; `context_blindness` replay; counterfactual | There is no in-the-wild data; the claim must match the evidence |
| Proof 3 | "Prevents demographic or lighting bias" | "Decision stability when face evidence is missing or degraded"; demographic validation listed as a gap | Bias prevention was not tested and cannot be claimed |
| Proofs 4–5 | — | Narrator faithfulness table; governance framing | The LLM and the legal position are both judgeable claims |
| Regulation | Implied that privacy and consent were enough | EU AI Act Art. 5(1)(f) prohibits workplace/education emotion recognition regardless of consent; research prototype for consented self-monitoring outside those settings; `--room-only` mode | Consent does not lift the prohibition; v1's framing implied that it did |
| Validation framing | — | Simulator-trained; acted and self-report sanity sets; not clinically validated; in-the-wild validation is future work | Honesty about what the evidence supports |
| UI scope | UI described as part of the product | Backend exposes a versioned contract; the frontend is built separately in Lovable | Separate build; one contract keeps both sides consistent |
| Scope control | — | Non-goals, success metrics as targets, priority tiers and drop order | Prevents scope creep and unsupported claims |
