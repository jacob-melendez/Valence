# Valence architecture

This document replaces the original "Code Architecture Breakdown" note. It describes how Valence turns a webcam
and a microphone into a per-person emotional-state label with an explanation, and where each data-governance
control sits.

Sources of truth:

- `docs/PLAN.md`: the approved execution plan (architecture, states, model ladder, proofs).
- `valence/contracts.py` (`CONTRACT_VERSION = "1.0.0"`): every field name used below. If this document and
  the contract disagree, the contract wins.
- `docs/CONTRACT_RESPONSES.md`: confirmed perception-to-runtime semantics.
- `docs/BENCHMARKS.md`: the only source of measured numbers here. Machine: MacBook Pro, Apple M3 Max, 36 GB.

Status legend used in tables: **exists** = the file is on the branch now; **planned** = this is the design
the code will implement. No evaluation results exist yet. Anything not measured is written as a target or as
"TBD — produced by `<script>`".

---

## 1. Overview

Valence runs entirely on one laptop. Edge perception reduces each video frame and audio block to a few numbers
(counts, flags, levels, normalized boxes). The raw buffer is then zeroed by the purge gate. The runtime turns
those numbers into one 5-second record per person and keeps the last 15 minutes in a rolling window. A
LightGBM model reads about 110 features from that window. A face-only model runs next to it as a baseline.
SHAP contributions are grouped into the PRD's layers, a rule-based detector flags "context corrected the
face", and the result is streamed to a local web UI. A small local LLM in a separate process rewrites the
explanation, and a validator checks it against the facts. The internet is used only for one-time model
downloads.

### Key runtime parameters (`valence/config.py`, exists)

Every value can be overridden with a `VALENCE_<NAME>` environment variable.

| Setting | Default | Meaning |
|---|---|---|
| `tick_s` | 5.0 | Tick length in seconds |
| `window_ticks` | 180 | Rolling window length (15 min) |
| `perception_hz` | 10.0 | Frames retrieved for perception per second |
| `warmup_s` | 60 | `warming_up` is true for this long |
| `override_block_s` | 120 | MaskOverrideDetector cannot fire during the first 2 min |
| `max_tracks` | 3 | Keep at most the 3 largest person tracks |
| `min_box_height` | 0.25 | Ignore person boxes shorter than 25% of frame height |
| `break_absent_min` | 10.0 | Absence that counts as a break (about 0.5 for demos) |
| `person_drop_min` | 15.0 | Unobserved person slot is dropped after this long |
| `db_cal_offset` | 94.0 | dBFS to approximate dB SPL until room calibration replaces it |
| `near_field_db` | 6.0 | Laughter and compliments count only at room median + 6 dB or louder |
| `state_set` | `core` | `core` (4 states) or `all` (6 states) |
| `api_host`, `api_port` | 127.0.0.1, 8000 | FastAPI bind (loopback only) |
| `narrator_url` | http://127.0.0.1:8081 | Narrator service |
| `narrator_deadline_s` | 2.5 | After this, the template text stands |
| `mode` | `replay` | `live`, `replay` or `hybrid` |
| `room_only` | False | Compliance configuration: one aggregate room label |
| `audit_dir`, `audit_ttl_s` | `var/audit`, 86 400 | Audit log location and time to live |

---

## 2. Data flow

`*` marks data that never crosses the purge gate.

```
 MAC SENSORS        PERCEPTION  (LOCAL: valence/perception, valence/edge)    raw media exists only in this band
 ───────────        ─────────────────────────────────────────────────────
 webcam ─► capture thread ─► 1-slot buffer ─► perception thread (VideoPerceiver, 10 Hz)
           grab() every frame  displaced       YOLO26n-pose + ByteTrack (MPS, 640) ─► boxes, ids, keypoints*
           retrieve() @10 Hz   frame → purge   GestureRecognizer ┐ alternate frames ─► fist scores
                                               FaceLandmarker    ┘ head crops       ─► blendshapes*
                                               lux (10 s median, 1 Hz) · posture · motion · debouncers
 mic ────► sounddevice callback ─► audio worker (16 kHz mono, 100 ms blocks → ~1 s analysis step)
           (pinned by name)        Leq + rms_power · YAMNet laughter · Vosk restricted grammar ─► text*
                                       │ frame / block / text: features are extracted first, then
                                       ▼
 ╔═══════════════════════ PURGE GATE   valence/governance/purge.py ═════════════════════════════════╗
 ║ gate.purge(buf, kind): zero the owning buffer in place, track it by weakref, caller `del`s refs  ║
 ║ gate.note_transcript_purged(n) · ticker calls gate.end_tick(): any survivor = PURGE_VIOLATION    ║
 ║ Only VideoObs / AudioObs cross: scalars, flags, normalized boxes. Nothing marked * crosses.      ║
 ╚══════════════════════════════════════╤═══════════════════════════════════════════════════════════╝
                                        │ sink(obs)        valence.edge.run.start(sink, ...)
 ReplaySource: features JSONL ─────────►│                  (same sink; source = "replay")
                                        ▼
 RUNTIME (CLOUD: valence/runtime)
 Aggregator ─ ROOM slot:    Leq dB, p90, lux, laughter / compliments (near-field gated), n_people,
            │               time of day, break clock
            └ PERSON slots: fists, slump, head-in-hands, motion, withdrawal, face valence / visibility
                    │ every 5 s (ticker thread)
                    ▼
 TickRecord per person (source, simulated_fields) ◄── scenario YAML / POST /api/demo/seed (valence/sim)
                    ▼
 RollingWindow: 180 ticks = 15 min per person; a new person inherits room history
                    ▼
 compute_features (~110, order frozen in artifacts/feature_spec.json) ─► clip to train p1–p99 (+ ood_features)
                    ▼
 M2 LightGBM (served)  +  M0 face-only shadow
                    ▼
 softmax(z/T) ─► confidence (calibrated, unsmoothed) ─► EMA ─► hysteresis(2) ─► label   (smoothing: label only)
                    ▼
 pred_contrib TreeSHAP ─► layer sums ─► top drivers (level / effect / scope)
                    ▼
 MaskOverrideDetector ─► display_label() ─► FactSheet ─► template text ─► `prediction`
                    ▼
 EventBus (bounded, per-client drop-oldest) ─┬─► FastAPI :8000  /api + /ws/events ─► Lovable UI
                    │                        └─► scripts/console.py
                    ├─► AuditLog: hash chain of labels, SHAP layer sums, purge counters (per session, TTL)
                    └─► narrator_client (latest-only per person) ── POST /narrate ──► Narrator :8081
                                                                     (separate process, own venv)
                        `explanation` (same event_id) ◄── NarrationResult ◄── Qwen3-1.7B-4bit ─► validator
                                                                                (template on failure)
```

The gate line is a hard boundary. Above it, code may hold pixels, audio samples, keypoints, blendshapes and
recognizer text for the duration of one processing call. Below it, the system holds only the dataclasses in
`contracts.py`, and those carry no media.

---

## 3. Process and thread model

| Unit | Kind | Owner | Does | Hands off via |
|---|---|---|---|---|
| uvicorn main thread (asyncio loop) | main process | CLOUD | FastAPI routes, `/ws/events` fan-out, narrator client tasks, lifespan start/stop | EventBus |
| Capture thread | daemon thread | LOCAL | `cap.grab()` on every frame; `cap.retrieve()` only at 10 Hz into a 1-slot buffer; a frame displaced from the slot is purged | slot |
| Perception thread | daemon thread | LOCAL | `VideoPerceiver.process(frame)` → `VideoObs`; `gate.purge(frame, "video")`; `del frame` | `sink(obs)` |
| Audio callback + worker | sounddevice callback + daemon thread | LOCAL | 100 ms blocks → ~1 s `AudioObs`; each block purged once consumed | `sink(obs)` |
| Replay thread | daemon thread | CLOUD | Paces recorded observations or scenario ticks by timestamp (speed factor) | `sink(obs)` or window |
| Ticker thread | daemon thread | CLOUD | Every `tick_s`: close the tick, build TickRecords, window, features, inference, explanation, audit append, `gate.end_tick()` | `loop.call_soon_threadsafe` → EventBus |
| Narrator | separate process, own venv `valence/narrator/.venv`, port 8081 | LOCAL | MLX inference + validator + template fallback | HTTP |

Rules:

- The `sink` is called from the perception and audio threads at once, so the aggregator is lock-protected.
  `test_window` (planned) covers 4 concurrent writer threads.
- Worker threads never touch asyncio objects directly. They schedule onto the loop with
  `call_soon_threadsafe`.
- Lifespan startup (planned `valence/api/app.py`) installs the log filter, opens the session `AuditLog`
  (which deletes expired files), loads the model artifacts, and starts sources per mode plus the ticker.
  Shutdown calls the edge `stop()` (releases camera and mic), stops the ticker, and closes the audit log with
  `delete=True`.
- Never run uvicorn with `--reload` while live sensors are on. The reloader starts a second process, and
  camera/mic permission belongs to the launching app and fails silently.
- YOLO (torch, MPS) and LightGBM share the main process. The libomp clash is checked by a same-process smoke
  test. The fallback is in section 13.
- The narrator runs in its own process and venv (mlx-lm 0.31.3). Its dependencies cannot conflict with
  perception, and its GPU work is separate from YOLO's. Contention is managed by narrating only on triggers
  (section 9.6). If YOLO frame rate drops by more than 30%, YOLO moves to CPU at imgsz 480.
- Escalation (P2): if perception falls below 8 Hz, it moves to a `spawn` subprocess that emits only
  feature dicts. Raw frames would then never enter the API process at all.

---

## 4. Module map and ownership

C = CLOUD session (Linux, no camera). L = LOCAL session (the Mac). Ownership is disjoint. Contract changes go
through `docs/CONTRACT_REQUESTS.md`.

| Path | Owner | Responsibility | Status |
|---|---|---|---|
| `valence/contracts.py` | C | States, display rules, `PersonObs`/`VideoObs`/`AudioObs`, `TickRecord`, `FactSheet`, WS messages, JSON Schema export | exists |
| `valence/config.py` | C | `Settings` (section 1); `.env.example` with offline vars | exists (`.env.example` planned) |
| `valence/governance/purge.py` | C | `PurgeGate`, module-level `gate` | exists |
| `valence/governance/audit.py` | C | `AuditLog`: hash chain, payload checks, TTL, session deletion | exists |
| `valence/governance/logfilter.py` | C | `RawMediaFilter`, `install()` | exists |
| `valence/governance/persistence_scan.py` | C | AST scan for disk writes in raw-media paths | exists |
| `valence/governance/` retention | C | Retention summary for `GET /api/retention` and `retention_s` | planned |
| `valence/runtime/bus.py` | C | EventBus with bounded per-client queues (drop-oldest) | planned |
| `valence/runtime/aggregator.py` | C | Room and person slots, near-field gate, break clock, TickRecord assembly | planned |
| `valence/runtime/ticker.py` | C | 5 s loop: window, features, predictor, explainer, audit, `end_tick()` | planned |
| `valence/runtime/replay.py` | C | ReplaySource: scenario YAML, features JSONL, seed | planned |
| `valence/sim/` | C | `generator.py` (v2), `scenarios.py`, `legacy.py`, `tune_difficulty.py` | planned |
| `valence/features/spec.py` | C | `SENSOR_SPEC` (with units), `CHANNELS`, `FEATURE_NAMES`, `FACE_FEATURES`, `LAYER_MAP` | planned |
| `valence/features/window.py` | C | `RollingWindow`, `compute_features(window)` | planned |
| `valence/features/build_dataset.py` | C | Training windows from simulated sessions | planned |
| `valence/models/train_ladder.py` | C | M0 to M2 variants, calibration, clip bounds | planned |
| `valence/models/temporal.py` | C | M3 GRU benchmark | planned |
| `valence/models/predictor.py` | C | Clip, temperature, EMA, hysteresis | planned |
| `valence/models/counterfactual.py` | C | One-feature what-if re-runs of M2 | planned |
| `valence/explain/explainer.py` | C | Layers, MaskOverrideDetector, `build_fact_sheet` | planned |
| `valence/explain/template.py` | C | Instant template explanation | planned |
| `valence/explain/narrator_client.py` | C | Async latest-only narration client | planned |
| `valence/eval/` | C | `evaluate.py`, `score_acted.py`, `negative_control.py` → `reports/` | planned |
| `valence/api/app.py` | C | Lifespan, REST, `/ws/events` | planned |
| `valence/perception/gate.py` | L | Single import point for the purge gate (zero-fill fallback if governance is missing) | exists |
| `valence/perception/debounce.py` | L | `HysteresisDebouncer`, `fist_debouncer()`, `laughter_debouncer()` | exists |
| `valence/perception/` `video.py tracking.py face.py fist.py posture.py lux.py audio.py` | L | `VideoPerceiver` and audio perception | planned on the branch (benchmarked locally) |
| `valence/edge/run.py` | L | `start(sink, *, video, audio, obscure_face) -> stop`; CLI `--print` / `--jsonl` | planned (`edge/__init__.py` exists) |
| `valence/narrator/` | L | `service.py` (:8081), `prompts.py`, `validator.py`, `grammar.py`, `build_data.py`, `lora.yaml`; own `.venv` | planned (`__init__.py` exists) |
| `scripts/` | C | `console.py`, `e2e_check.py`, `mem_soak.py`, `audit_verify.py`, `export_fact_sheets.py` | planned |
| `scripts/` | L | `spikes/` and `download_models.sh` (exist); `calibrate_*.py`, `overnight.sh` (planned) | mixed |
| `scenarios/*.yaml` | C | focused, frustrated_masked, exhausted, collaborative_happy, overwhelmed, tense, context_blindness, full_demo | planned |
| `tests/` | C (L: `tests/perception`, `tests/narrator`) | Existing: `test_purge.py`, `test_audit.py`, `test_logfilter.py`, `test_no_persistence_ast.py`, `test_contracts.py` | partly exists |
| `data/` | C: `sim/`, `narrator/fact_sheets_sim.jsonl` · L: `real/`, `narrator/{train,valid,test}.jsonl` | Features-only data | planned |
| `artifacts/`, `reports/` | C | Models, calibrators, `feature_spec.json`, clip bounds; metrics and figures | planned |

---

## 5. Perception → runtime contract

The LOCAL side implements `EdgeStart`: `valence.edge.run.start(sink, *, video=True, audio=True,
obscure_face=False)` returns a `stop()` callable. It calls `sink(obs)` with `Obs = VideoObs | AudioObs`. The
dataclasses carry only scalars, booleans and normalized boxes. They never carry pixels, audio, keypoints,
blendshapes or text.

### 5.1 `PersonObs` (one tracked person in one processed frame, about 10 Hz)

| Field | Type | Semantics (confirmed in `CONTRACT_RESPONSES.md`) |
|---|---|---|
| `track_id` | int | Ephemeral display id 1..N, assigned by first appearance ("Person #N"). Reused if the person reappears within 3 s and 0.25 frame-widths of where the track was lost. After that the id is released and the person's baselines are forgotten. Never linked to identity. |
| `bbox` | (x1, y1, x2, y2) | Normalized to [0, 1]; validated in `__post_init__`. |
| `fist_onsets` | int | **Delta**: debounced fist rising edges in this frame (usually 0, sometimes 1). The aggregator sums them per tick. |
| `fist_active` | bool | Debounced fist currently held. |
| `slump` | bool | Debounced posture slump, judged against the person's own baseline posture. |
| `head_in_hands` | bool | Hands at the head, from pose keypoints. |
| `motion` | float | EMA (α 0.3) of mean keypoint displacement in shoulder-widths per second, in raw units. **NaN when undefined** (no keypoints). 0.0 means "still". Per-person normalization happens in the feature layer (`*_rel_sess`), not here. |
| `face_visible` | bool | A face was read for this person in this frame. |
| `face_valence` | float | [-1, 1], relative to the person's neutral baseline (median of their first 20 s of visible face). NaN whenever `face_visible` is false. |

### 5.2 `VideoObs` (one processed frame)

| Field | Semantics |
|---|---|
| `t`, `wall` | `time.monotonic()` and `time.time()` |
| `frame_seq` | Sequence number of the processed frame |
| `frames_decoded` | Cumulative frames retrieved from the camera. Must equal the gate's video purge count, and the `governance` message checks this. |
| `lux`, `lux_quality` | Estimated lux: a 10 s median refreshed at 1 Hz. NaN until the first estimate and on frames where it was not refreshed. `lux_quality` in [0, 1] = background share × unclipped share × stability, halved while uncalibrated. |
| `frame_w`, `frame_h`, `mirrored` | Lets the UI overlay normalized boxes correctly |
| `persons` | At most `max_tracks` (3) largest tracks, boxes at least `min_box_height` (25%) of frame height, inside the desk ROI |

### 5.3 `AudioObs` (one ~1 s analysis step)

| Field | Semantics |
|---|---|
| `t`, `wall`, `block_seq` | Timestamps and sequence |
| `db_spl` | Calibrated approximate dB SPL (Leq over the step) |
| `rms_power` | Linear `mean(x**2)` over the step, so the aggregator can average in the power domain |
| `speech_prob` | From the audio classifier |
| `laughter_onsets` | **Delta**: debounced laughter rising edges in this step. Raw count; near-field gating is done by the aggregator. |
| `compliment_hits` | **Delta**: lexicon phrase matches in this step from the Vosk restricted grammar. The text itself is discarded inside perception and counted with `gate.note_transcript_purged(n)`. |

### 5.4 Recording format

Features-only JSONL (`python -m valence.edge.run --jsonl ...`, written to `data/real/*.jsonl`) has one object
per line: `{"type": "video" | "audio", ...dataclass fields...}`. Tuples are written as lists and NaN as `null`.
The recorder uses `obs_to_dict` and replay uses `obs_from_dict` (which also accepts the older `"kind"` key),
so recordings round-trip exactly.

### 5.5 Inside perception (LOCAL design, measured in `BENCHMARKS.md`)

| Stage | Design | Measured p50 / p95 |
|---|---|---|
| Capture | `grab()` every frame, `retrieve()` at 10 Hz; a displaced slot frame is purged; `frames_decoded` counts retrievals | — |
| Person tracking | YOLO26n-pose + ByteTrack on MPS, imgsz 640, `classes=[0]`, `conf=0.4`, `track_buffer` 60. Boxes, ids and keypoints are copied with `.cpu().numpy().copy()`, then `del results` before the purge. The predictor's cached input image (`batch`, `results`, `dataset.im0`) is cleared after each call, and a weakref check confirms no frame survives `process()`. | 18.4 / 21.0 ms |
| Fists | GestureRecognizer (VIDEO mode). Hands whose wrist is in the bottom 20% of the frame (keyboard zone) are ignored. | 16.4 / 16.8 ms (full frame) |
| Face | FaceLandmarker in IMAGE mode on a **per-person head crop**: box from YOLO nose/eye/ear keypoints, scale 2.2, resized to 256 px. Blendshapes → valence. On the full 720p frame the detector found 0 faces in a 2-person shot; the crop found them. A crop belongs to exactly one person, so face-to-person matching is exact. `obscure_face=True` disables face reading for the live occlusion demo. | ~11 ms per person |
| Gesture/face schedule | GestureRecognizer and FaceLandmarker alternate frames (about 5 Hz each) | — |
| Posture, motion | Slump and head-in-hands from keypoints; motion EMA as in 5.1 | — |
| Lux | Brightness regression → 10 s median at 1 Hz | — |
| **`VideoPerceiver.process` end to end** | YOLO + alternating gesture/face crops + lux, 2 people, 1280×720 sample images | **45 ms / ~51 ms** |
| Audio input | sounddevice, 16 kHz mono float32, blocksize 1600 (100 ms), device pinned by name ("MacBook Pro Microphone") | — |
| Laughter | YAMNet AudioClassifier, 6-class allowlist, `laughter_debouncer()` (on at ≥ 0.30 for 2 results, off below 0.15, 3 s refractory) | callback ~6 ms, 1 result/s |
| Compliments | Vosk small-en-us 0.15, restricted grammar (8 phrases + `[unk]`) | p50 0.84 ms, max 29 ms per 100 ms block |

The loop budget is 100 ms at 10 Hz, so the measured ~45 ms leaves headroom. Live-camera numbers are still
pending camera permission (see `BENCHMARKS.md`).

Decisions that refine the plan:

- **YAMNet mode.** The plan specified `AUDIO_CLIPS` on 0.975 s buffers. `AUDIO_STREAM` was measured to work
  (1 result/s). It is kept, provided each result is attributed to the audio step whose samples produced it
  (classifier-callback timestamp, not arrival time). `AUDIO_CLIPS` is the fallback if that attribution gets
  messy.
- **Motion units.** The plan described `motion` as per-person normalized. The confirmed contract emits the raw
  unit (NaN when undefined) and normalizes in features (`motion_rel_sess`).
- **Fist rule.** The plan's rule is score ≥ 0.7 held ≥ 0.5 s plus the keyboard-zone exclusion. The current
  `fist_debouncer()` in `debounce.py` still uses the earlier values (on at ≥ 0.6 for 3 samples, off below 0.4
  for 1 s, 3 s refractory). `CONTRACT_RESPONSES.md` lists this as open on the edge side.

---

## 6. Aggregator (`valence/runtime/aggregator.py`, planned)

The aggregator keeps one **room slot** and up to `max_tracks` **person slots**. It closes a tick every
`tick_s` and emits one `TickRecord` per person. Room fields are copied into every person's row.

### 6.1 Room slot

| TickRecord field | Built from |
|---|---|
| `ambient_db` | Power-domain mean of the tick's `AudioObs.rms_power`, converted to dB SPL with the calibration offset (Leq) |
| `db_p90` | 90th percentile of the tick's step levels. Carried for display and QA; not in the frozen `CHANNELS`. |
| `light_lux`, `lux_quality` | Latest non-NaN `VideoObs.lux` / `lux_quality` in the tick |
| `laughter_count`, `compliments_count` | Sum of `laughter_onsets` / `compliment_hits` from steps that pass the **near-field gate** |
| `n_people` | Number of active person slots |
| `tod_h` | Local time of day in hours [0, 24) |
| `hours_since_break` | Room break clock (below) |

**Near-field gate.** An audio step's laughter and compliment counts are kept only if that step's `db_spl` is
at least the room's running median plus `near_field_db` (6 dB). This counts laughter and praise at the desk,
not from the next table in a loud hall. Perception emits raw counts; gating happens only here.

**Break clock.** One communal clock, as in the original PRD's "last communal room break". It starts at
session start and resets on `POST /api/events/break`, or when no person has been tracked for
`break_absent_min` (10 min by default; about 0.5 min for demos).

### 6.2 Person slots

Keyed by `track_id`.

| TickRecord field | Built from |
|---|---|
| `clenched_fists_count` | Sum of `fist_onsets` over the tick |
| `posture_slump` | Fraction of the tick's frames with `slump` true |
| `head_in_hands_frac` | Fraction of the tick's frames with `head_in_hands` true |
| `motion_energy` | NaN-aware mean of `motion` over the tick |
| `face_valence` | Median over the tick of visible-face frames; NaN when never visible |
| `face_visible` | Fraction of the tick with a visible face |
| `social_withdrawal_flag` | Derived from person-slot signals. The original dataset defined it as "slumped over / isolated". Exact rule TBD in `aggregator.py`; P1 and early in the drop order. |
| `person_id`, `display_id` | `track_id` and "Person #N" |
| `tick`, `ts`, `t_mono`, `session_id` | Tick bookkeeping |
| `source`, `simulated_fields` | Provenance (section 12) |

A slot not observed for `person_drop_min` (15 min) is dropped. `POST /api/demo/reset` clears all person slots
and windows between judge groups.

### 6.3 New people inherit room history

When a new person appears mid-session, their window is pre-filled with the room's history for the room-level
channels (dB, lux, laughter, compliments, `n_people`, time). Their person channels start empty (NaN). A late
arrival in a room that has been loud for 12 minutes is therefore judged in that context, but nothing about any
other person is copied. `fill_ratio` and `window_fill_min` still reflect how much of the window is filled.
`test_window` (planned) covers this inheritance along with 900 s eviction.

### 6.4 Room-only mode

With `--room-only` or `POST /api/mode {"room_only": true}`, the aggregator emits one aggregate room record. The
API publishes one room label: no per-person predictions, per-person drivers or face results. This is the
compliance-oriented configuration (section 11.3).

---

## 7. Rolling window and features (`valence/features/`, planned)

### 7.1 RollingWindow

There is one thread-safe deque of `TickRecord`s per person, capped at `window_ticks` = 180 (15 min). Records
older than 900 s are evicted. The window lives in RAM only. One plain-numpy function,
`compute_features(window)`, is used both offline (training) and live (serving). `test_feature_parity`
(NaN-aware, atol 1e-9) and `test_no_lookahead` (both planned) guard this. Training windows are cut every 6
ticks with random length 1–180 ticks, so the model also learns to handle partly filled windows.

### 7.2 CHANNELS (frozen)

| Channel | From TickRecord | Slot | Layer |
|---|---|---|---|
| `db` | `ambient_db` | room | Ambient Physics |
| `lux` | `light_lux` | room | Ambient Physics |
| `motion` | `motion_energy` | person | Somatic & Social |
| `slump` | `posture_slump` | person | Somatic & Social |
| `hih` | `head_in_hands_frac` | person | Somatic & Social |
| `fist` | `clenched_fists_count` | person | Somatic & Social |
| `laugh` | `laughter_count` | room | Somatic & Social |
| `compl` | `compliments_count` | room | Somatic & Social |
| `withdraw` | `social_withdrawal_flag` | person | Somatic & Social |
| `n_people` | `n_people` | room | Somatic & Social |
| `hours_since_break` | `hours_since_break` | room | Time Dynamics |
| `tod_sin`, `tod_cos` | `tod_h` | room | Time Dynamics |
| face (separate) | `face_valence`, `face_visible` | person | Facial Surface |

`SENSOR_SPEC` in `spec.py` records each channel's unit. A driver's `scope` (`room` or `person`) follows its
channel's slot. That is what lets the narrator say "the room was loud" rather than attributing room noise to
a person.

### 7.3 Feature families (~110; order frozen in `artifacts/feature_spec.json`)

| Family | Features | Count |
|---|---|---|
| Continuous S ∈ {db, lux, motion, slump, hih} | `S_now`; `S_{mean,std,slope,max}_{1m,5m,15m}`; `S_ewm_{1m,5m}`; `S_d1m15m` (short vs long trend); `S_rel_sess` (relative to the person's own session baseline; removes per-session mic offset and per-person motion scale) | 17 × 5 = 85 |
| Events E ∈ {fist, laugh, compl} | `E_cnt_{1m,5m,15m}`, `E_since_s` | 4 × 3 = 12 |
| Social | `withdraw_now`, `withdraw_frac_5m`, `n_people` | 3 |
| Time and window | `hours_since_break`, `tod_sin`, `tod_cos`, `fill_ratio` | 4 |
| Face (`FACE_FEATURES`) | `face_now`, `face_mean_{1m,5m,15m}`, `face_std_5m`, `face_visible_frac_5m` | 6 |
| **Total** | | **110** |

Missing values stay NaN; LightGBM handles NaN natively. Face features are NaN when no face was visible.

### 7.4 LAYER_MAP

| Layer (`LAYERS`) | Feature prefixes |
|---|---|
| Ambient Physics | `db_*`, `lux_*` |
| Time Dynamics | `hours_since_break`, `tod_*` |
| Somatic & Social | `fist_*`, `laugh_*`, `compl_*`, `slump_*`, `hih_*`, `motion_*`, `withdraw_*`, `n_people` |
| Facial Surface | `face_*` |

### 7.5 Clipping and out-of-distribution reporting

Before inference, every feature is clipped to the training p1–p99 range stored with the model artifacts. The
names of clipped features go to `data_quality.ood_features`. If more than 3 features are clipped, the
confidence is shown in a lower band and the narration says so. The sim-to-real target is that at least 90% of
live features fall inside the clip bounds (TBD — measured on live captures).

---

## 8. Model ladder (`valence/models/`, planned)

All models are trained on simulator v2 data (see `docs/DATASET.md`), split by session 70/15/15. No session
appears in two splits. The validation split is halved: one half for early stopping, one half for calibration
(temperature scaling).

LightGBM settings: 31 leaves, learning rate 0.05, up to 800 rounds, early stop 50, `min_data_in_leaf` 100,
feature and bagging fractions 0.8, balanced class weights.

| Model | Inputs | Algorithm | Purpose | Served? |
|---|---|---|---|---|
| M0 | Face snapshot (`face_now`) | LightGBM | Context-blind face baseline | Yes, as the live shadow behind `face_only`; null when no face is visible |
| M0t | Face history (the 6 `FACE_FEATURES`) | LightGBM | Does face history alone close the gap? | No (eval) |
| M1 | Context snapshot: current value of each `CHANNELS` entry, no history | LightGBM | Context without time | No (eval) |
| M2-noface | All non-face features | LightGBM | "Face is informative but not sufficient"; stability when the face is missing | No (eval) |
| M2-noabsdB | M2 without absolute dB levels (relative and trend dB kept) | LightGBM | Robustness to mic calibration (the simulator applies a ±12–15 dB per-session mic offset) | No (eval, P1) |
| M2-notime | M2 without `hours_since_break`, `tod_sin`, `tod_cos` | LightGBM | Is time a shortcut? | No (eval, P1) |
| **M2** | All ~110 features (context + face history) | LightGBM | The served model | **Yes** |
| M3 | Tick sequence | PyTorch GRU(64) + attention | Temporal benchmark (cloud CPU) | No (P1) |

Acceptance targets from the plan (not results): M2 macro-F1 between 0.85 and 0.92 (the 0.92 ceiling is the
anti-circularity criterion used to tune `noise_scale`); M2-noface ≥ M2 − 0.05 on the core 4 states; ECE drops
after calibration; at 15 min of history, M2 > M1 > M0 with non-overlapping session-bootstrap 95% CIs.
Results: TBD — produced by `valence/eval/evaluate.py` → `reports/metrics.json`, `reports/figs/*.png`.

Artifacts per model in `artifacts/`: booster, temperature `T`, clip bounds, `feature_spec.json`.

---

## 9. Per-tick inference path

The ticker runs this for each person on every completed tick. Target: features to `prediction` in under
200 ms (`test_e2e_replay`). The API streams `prediction` at most once per second per person.

### 9.1 Features, clip, models

1. Append the new `TickRecord` to the person's window, then `compute_features(window)` → about 110 floats.
2. Clip to p1–p99 and collect `ood_features`.
3. Run M2 and M0 (M0 on the face snapshot; skipped, giving a null `face_only`, when no face was visible this
   tick).

### 9.2 Confidence and label

4. `softmax(z / T)` on M2 margins gives the calibrated `probabilities` and `confidence`. The displayed
   confidence is **unsmoothed** and labeled "sim-calibrated" in the UI, because calibration was fit on
   simulated data.
5. The **label only** is smoothed: an EMA over the probability vector, then hysteresis(2), so the label
   changes only after the new class wins 2 consecutive ticks. Confidence numbers are never smoothed, so the
   narrator never quotes a number the model did not produce.
6. `warming_up` is true for the first `warmup_s` (60 s).

### 9.3 SHAP layers and drivers

7. LightGBM `pred_contrib` (TreeSHAP; target < 5 ms) on M2. Take the predicted class's slice, sum
   contributions by `LAYER_MAP` into `Layer.contribution`, and pick the top drivers. Each `Driver` carries
   `feature`, `display` (pre-formatted, e.g. "average noise of 72 dB over 15 minutes"), `value`, `unit`,
   `shap`, `level` (`high|elevated|normal|low`), `effect` (`supports` if `shap` > 0, else `opposes`) and
   `scope` (`room|person`). `test_shap_additivity` (planned) checks contributions sum to the margin within
   1e-6.

### 9.4 MaskOverrideDetector

8. `mask_override` turns true only when **all** of these hold:

| Condition | Threshold |
|---|---|
| M0's argmax differs from M2's label | — |
| M0 confidence | ≥ 0.55 |
| M2 confidence | ≥ 0.70 |
| Inside M2, the Facial Surface layer's SHAP toward M0's class | > 0 (the face really does look like M0's class; context outweighs it) |
| Consecutive ticks meeting the above | 3 |
| Time since the person's window started | ≥ `override_block_s` (120 s) |

Verification target: fires within 6 ticks on the `frustrated_masked` scenario and never on
`collaborative_happy` or `focused`. It is validated on scenarios as soon as M2 exists.

### 9.5 Display label, FactSheet, template text

9. `label_display = display_label(label, mask_override=..., n_people=..., states=...)`:
   - In the core-4 phase, `Frustrated_Masked` also covers unmasked frustration and reads "Frustrated". It reads
     "Frustrated / Masked Stress" only when `mask_override` fired. Once `Tense` is in the state set,
     `Frustrated_Masked` is mask-only and always shows its full name.
   - `Collaborative_Happy` reads "Upbeat" when `n_people` < 2.
10. Build the `FactSheet`: `event_id`, `person`, `label`, `label_display`, `confidence`,
    `confidence_display`, `probabilities`, `window_fill_min`, `n_people`,
    `face_only{label, confidence, face_valence}` (all null when `face_visible` is false), `face_visible`,
    `mask_override`, `layers[{name, contribution, drivers[...]}]`,
    `data_quality{face_visible_frac, lux, ood_features}`.
11. `template.py` renders instant text from the FactSheet alone: the person, `label_display` and
    `confidence_display`, the top drivers' `display` strings (room-scope drivers phrased as "the room"), a face
    clause (agrees / disagrees / "no face visible"), and mask language only when `mask_override` is true. The
    `prediction` message goes out with `explanation{text, source: "template"}`.
12. The audit log receives a features-free entry: label, confidence, layer sums, `mask_override`, purge
    counters.

### 9.6 Async narration

13. `narrator_client.py` keeps a **latest-only queue per person**. At most one request is in flight per
    person, and a newer FactSheet replaces any pending one. Triggers: label change, confidence-band change,
    and every 30 s.
14. `POST http://127.0.0.1:8081/narrate` with the FactSheet. The narrator (LOCAL) turns it into compact
    `key: display` lines (~120 tokens), runs Qwen3-1.7B-4bit with `enable_thinking=False`, an mlx_lm prompt
    cache for the fixed system prompt and few-shots, and `max_tokens` 90. It returns
    `NarrationResult{text, source: llm|llm_retry|template, latency_ms, attempts}`.
15. The narrator's validator rejects any text with numbers not on the sheet, a wrong state name, fewer than 2
    of the top-3 drivers or anything outside the top 5, a qualifier within 6 tokens of a feature that
    contradicts its `level`/`effect`, room-scope drivers not phrased as "the room", mask language without
    `mask_override`, pronouns, demographics or diagnoses, or a length outside 20–90 words. Cached text is
    re-validated against the current sheet. On failure it retries, then falls back to the template.
16. The runtime publishes an `explanation` message with the **same `event_id`** as its `prediction`. The UI
    replaces the template text and shows `source`. If `narrator_deadline_s` (2.5 s) passes, the template text
    stays and `status.narrator` reports `fallback` or `down`.

Measured (spike, no YOLO load): Qwen3-1.7B 4-bit generates at 187 tok/s, 536 ms wall for a 55-token answer,
1.27 GB peak memory. The ship gate is p95 ≤ 2.5 s **measured with YOLO running** (TBD — produced by the
LOCAL narrator A/B run).

### 9.7 Counterfactual

`GET /api/explain/{event_id}/counterfactual?feature=value` re-runs M2 on that event's feature vector with one
feature changed (for example `hours_since_break=0.5`) and returns the new probabilities. Recent feature vectors
are held in RAM with the same 15-minute lifetime as the window. Verification target: only the named feature
changes.

---

## 10. Output interfaces (summary; full schemas in `docs/API.md`)

| WS message (`/ws/events`) | Rate | Carries |
|---|---|---|
| `hello` | on connect | `mode`, `room_only`, `states`, `feature_version`, `frame_w`, `frame_h`, `mirrored` |
| `tracks` | ≤ 10 Hz | Ids, labels, normalized boxes, `fist_active`, `slump`, `capture_ts_ms`. Never pixels or keypoints. |
| `governance` | per tick | `message` (PRD line), `frames_decoded`, `frames_purged`, `audio_blocks_purged`, `transcripts_purged`, `violations`, `live_raw_buffers`, `audit_hash`, `retention_s` |
| `state_vector` | 5 s per person | Current values plus `window_fill_min`, `source`, `simulated_fields` |
| `prediction` | ≤ 1 Hz per person | Label, `label_display`, `confidence`, `probabilities`, `face_only`, `mask_override`, `warming_up`, `layers`, `data_quality`, template `explanation`, provenance |
| `explanation` | async | `event_id`, `person_id`, `text`, `source`, `latency_ms` |
| `status` | periodic | Camera, mic, fps, loop latency, narrator state, offline, `degraded`, `purge_violation` |
| `heartbeat` | periodic | `t` |

Every message is validated against the JSON Schema from `json_schemas()` in `test_contracts` (the existing
`tests/test_contracts.py` covers the contract module; message-level API tests are planned). The EventBus gives
each client a bounded queue that drops the oldest message, so a slow or absent client cannot grow memory.
`scripts/console.py` is a terminal client of the same bus, so the demo does not depend on the web UI.

REST (`/api`): `states`, `state`, `history?minutes=15`, `governance/stats`, `governance/audit?since=`,
`governance/verify`, `retention`, `eval/summary`, `eval/figures/{name}.png`, `scenarios`, `models`, `health`,
`narrator/health`, `explain/{event_id}/counterfactual`; POST `control/{start|stop|pause}`, `events/break`,
`replay/{scenario}`, `sim/{face_occlusion|low_light}`, `sim/event`, `sim/lux`, `demo/seed`, `demo/reset`,
`calibrate/room`, `mode`, `label`. CORS allows localhost and the Lovable origin.

---

## 11. Governance controls

### 11.1 Where each control lives

| Control | Location | Mechanism | Evidence |
|---|---|---|---|
| **PurgeGate** | `valence/governance/purge.py` (exists) | `gate.purge(buf, kind)` with `kind` in `video`/`audio`. If `buf` is a view, it walks to the owning array; zero-fills it in place (read-only buffers are counted in `unzeroable`); registers `weakref.ref(owner, callback)` in an id→weakref dict. The callback removes the entry when the array is actually freed. `end_tick()` runs one `gc.collect()` if anything is still alive, then counts each survivor as a violation, logs `PURGE_VIOLATION` and stops tracking it. `live_raw_buffers` = purged buffers not yet freed. | `tests/test_purge.py` (zeroing, finalizer, leaked reference, views, read-only, cycles, held detector results); `/api/governance/stats` after a 60-min soak (target: 0 violations, 0 live buffers) |
| Purge call sites | perception (LOCAL) | Purge right after feature extraction, then `del`. Displaced capture-slot frames are purged too, so `frames_decoded == frames_purged`. YOLO outputs are copied and `del results` happens before the purge; predictor caches are cleared. | `governance` message per tick; fake-Results test (`test_holding_detector_results_leaks_the_frame` exists) |
| Transcript discard | perception audio (LOCAL) | Vosk restricted grammar: only 8 phrases + `[unk]` can be recognized, so there is no free transcript. Matched text is dropped in perception and counted via `note_transcript_purged`. Python strings cannot be zeroed; this limit is documented. | `transcripts_purged` counter |
| Boundary contract | `valence/contracts.py` (exists) | `PersonObs`/`VideoObs`/`AudioObs` have no field that can carry media, keypoints or text | `tests/test_contracts.py` |
| **Raw-media log filter** | `valence/governance/logfilter.py` (exists) | `RawMediaFilter` drops any log record whose message or arguments contain `np.ndarray`, `bytes`, `bytearray` or `memoryview`, including inside dicts, lists, tuples and sets. `install()` attaches it to the root logger and all its handlers and counts drops. | `tests/test_logfilter.py` |
| **Persistence scan** | `valence/governance/persistence_scan.py` (exists) | AST scan. `valence/perception/`: no file writes at all. `valence/edge/`: text-mode writes only (features JSONL). Both ban `imwrite`, `VideoWriter`, `save`, `savez`, `savetxt`, `tofile`, `dump`, `to_pickle`, `wave.open`, `soundfile.write`, `wavfile.write`, `write_bytes`. Reads are allowed. `python -m valence.governance.persistence_scan` exits 1 on any violation. | `tests/test_no_persistence_ast.py` |
| **Audit chain** | `valence/governance/audit.py` (exists) | One JSONL file per session, `var/audit/<session_id>.jsonl`. Each entry has `seq`, `ts`, `kind`, `payload`, `prev`, `hash` (SHA-256 over prev + canonical body). The payload checker refuses NaN/inf, strings over 200 chars (no transcript can be logged), lists over 64 items, nesting deeper than 4, and non-JSON types. Contents: labels, confidences, SHAP layer summaries, purge counters, operator actions (seed, simulated inputs, calibration). | `tests/test_audit.py`; `GET /api/governance/verify`; `scripts/audit_verify.py` (planned); `audit_hash` in each `governance` message |
| **Retention** | `audit.py` + retention module (planned) | Audit file deleted on session close (`close(delete=True)`); files older than `audit_ttl_s` (86 400 s) deleted at startup, so a crash cannot leave timelines on disk. The window is RAM-only, 15 min. | `GET /api/retention`, `retention_s` |
| Offline operation | `.env`, config | `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`; API and narrator bind to 127.0.0.1 | Wi-Fi-off rehearsal; `status.offline` |
| No pixels to the UI | API | `tracks` carries normalized boxes only; the UI uses the browser's own camera for preview | `test_contracts` schemas |
| Room-only mode | aggregator + API | One aggregate label, no per-person inference | `hello.room_only` |

### 11.2 Data lifetime

| Data | Held in | Lifetime |
|---|---|---|
| Video frame | capture slot → perception call | One `process()` call (measured ~45 ms p50), then zeroed |
| Audio block (100 ms) | audio worker | Until its analysis step consumes it, then zeroed |
| Keypoints, blendshapes, recognizer text | perception locals | Within one processing call; never cross the gate |
| Per-person baselines (face neutral, posture) | perception | Forgotten when the display id is released (3 s after loss) |
| `VideoObs` / `AudioObs` | aggregator current-tick buffers | Until the tick closes (5 s) |
| `TickRecord`s, features | RollingWindow (RAM) | 15 min; cleared by `demo/reset` or person drop |
| Audit entries (no features) | `var/audit/<session_id>.jsonl` | Deleted at session close; TTL 86 400 s as a backstop |
| Features-only recordings | `data/real/*.jsonl` | Operator-initiated captures only (live feature capture, acted set); features, never media |

### 11.3 Honest limits and framing

- `del` removes a name, not memory. Zeroing first means the bytes are gone even if the allocator keeps the
  pages. Copies inside native libraries (GPU allocator caches, AVFoundation pixel buffers, MediaPipe's
  `mp.Image`) cannot be reached from Python and are not zeroed. These limits are stated in `purge.py` and
  will be in `docs/GOVERNANCE.md`.
- Pseudonymous behavioral data is still personal data, which is why the audit log has a TTL and session
  deletion.
- EU AI Act Art. 5(1)(f) prohibits emotion recognition in workplaces and education regardless of consent
  (medical and safety carve-outs only). Valence is a research prototype for consented self-monitoring outside
  those settings. `--room-only` is the compliance-oriented configuration.

---

## 12. Modes and provenance

### 12.1 Modes

| Mode (`hello.mode`, `status.mode`) | Video | Audio | Use |
|---|---|---|---|
| `live` | camera | mic | The real pipeline |
| `replay` | recorded / scenario | recorded / scenario | Default (`settings.mode`); scripted states for the demo and tests |
| `hybrid` | live or replay | live or replay | One modality live and the other replayed, or live with simulated inputs |

Sources are chosen per modality with `--video live|replay --audio live|replay`.

ReplaySource inputs (`valence/runtime/replay.py`, planned):

- **Features JSONL** (`data/real/*.jsonl` in the format of 5.4) re-enters through the same `sink(obs)` the edge
  uses, so it exercises the aggregator exactly as live data does.
- **Scenario YAML** (`scenarios/*.yaml`) is rendered by the simulator into `TickRecord`s. These have the same
  shape as live ticks; the simulator produces every field the live pipeline produces.
- **`POST /api/demo/seed`** fills an empty window with context history only (no face), tagged
  `source = "sim_seed"`. It is a fallback for an empty window. At the venue, the plan is to start live at
  least 20 minutes early so the window is real.
- Replay is paced by timestamps with a speed factor (`scripts/e2e_check.py --speed 20`).

### 12.2 Provenance fields

Every `TickRecord`, `state_vector` and `prediction` carries:

- `source`: one of `live`, `replay`, `sim`, `sim_seed` (`Source` in the contract).
- `simulated_fields`: names of fields whose values were set by an operator or simulator rather than sensed:
  any `POST /api/sim/*` input (`face_occlusion`, `low_light`, `event`, `lux`), seeded history, or an
  operator-set value such as `hours_since_break`.

The UI shows a "REPLAY" or "SIMULATED: <field>" badge whenever `source` ≠ `live` or `simulated_fields` is
non-empty. Every operator action is also written to the audit chain. `POST /api/label {state}` records a
self-report label for sim-to-real comparison; it never changes a prediction.

### 12.3 Calibration

`POST /api/calibrate/room` (planned) sets the SPL offset (replacing `db_cal_offset`), a 2-point lux fit and the
noise floor. It is run for 10 minutes at the venue before judging.

---

## 13. Fallbacks

| Failure | Fallback chain | Notes |
|---|---|---|
| YOLO on MPS | CPU at imgsz 480 → PoseLandmarker (4 poses) + IoU tracking | MPS measured 18.4 ms p50; PoseLandmarker 23.4 ms, so not needed on the M3 Max |
| YOLO and MLX GPU contention | Narrate only on triggers; YOLO → CPU 480 if frame rate drops > 30% | |
| Perception < 8 Hz | Move perception to a `spawn` subprocess emitting feature dicts | P2 |
| 8 GB Mac | 5 Hz loop, LoRA batch 1–2, 8 layers | Demo Mac has 36 GB |
| Face not found or hidden | `face_valence` NaN, `face_only` null; M2 continues on context (NaN-native features) | Narration switches to the occlusion clause |
| GestureRecognizer | Finger-curl heuristic from hand landmarks | In the drop order |
| YAMNet | `AUDIO_CLIPS` mode → ai-edge-litert → `POST /api/sim/event` | `AUDIO_STREAM` measured working |
| Vosk | pywhispercpp tiny.en + hallucination filter → `POST /api/sim/event` | Vosk measured working |
| Lux regression | Raw brightness + `POST /api/sim/lux` | |
| LightGBM / torch libomp clash | `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=2`, `num_threads=1` → sklearn `HistGradientBoostingClassifier` | |
| Camera or mic unavailable | Hybrid or replay from golden features JSONL | `status.camera` / `status.mic` false, `degraded` lists it |
| Narrator LLM | Llama-3.2-3B-Instruct-4bit (downloaded only if Qwen fails) → template | Llama measured 1108 ms per 89-token answer |
| Narrator slow or down | Template text stands; `status.narrator` = `fallback` / `down`; ticks continue | Chaos test target: `status` event + fallback while ticks continue |
| LoRA fails the ship gate | Ship Stage A (prompt-only) | |
| Empty window at demo time | `POST /api/demo/seed` (context only, `sim_seed`) | |
| Live end-to-end late (S4 past H10:30) | Replay is the main demo; live runs hybrid with SIMULATED flags | |

Drop order under time pressure (from the plan): P2 → P1 states → withdrawal → lux fit → acted set to 2
scenarios → M3 → Vosk → YAMNet → LoRA → GestureRecognizer → LLM (template only). P0 governance and evaluation
are never dropped.

---

## 14. Mapping from the original Code Architecture note

The original note relied on i.i.d. mock rows that were close to 100% separable on context alone. It had no
temporal structure for a window to exploit, no laughter, noise or missing modalities, one person, and laughter
treated as a keyword. The two "paradoxes" it described are kept as the project's headline experiments, but each
is now measured on temporally structured data and surfaced through explicit mechanisms.

### A. "Frustrated_Masked paradox" (beating the face model)

| Original | New design |
|---|---|
| Mock rows: `facial_valence_snapshot` 0.4–0.8, `ambient_db` up to 85, `clenched_fists_count` > 0, `hours_since_break` > 3.5, drawn independently per row | Simulator v2: per-person semi-Markov states with dwell times, fatigue tilt after ~3 h, Poisson breaks, room dB that drifts toward occupants' states with a per-session mic offset, bursty fist events, face noise, 5–40% missing face. Frustrated_Masked face mean +0.45 (forced smile). |
| "Train a Decision Tree on the face column vs the full vector" | M0 (face snapshot) vs M2 (context + face history) LightGBM, with M0t, M1 and M2-noface as controls, compared at 0–15 min of history with bootstrap CIs. Headline: Frustrated_Masked recall, M0 vs M2 (TBD — produced by `valence/eval/evaluate.py`). |
| `if predicted_state == "Frustrated_Masked" and face_prediction == "Happy"` | `MaskOverrideDetector` (9.4): confidence thresholds on both models, face-layer SHAP check, 3 consecutive ticks, blocked for the first 2 min. Sets `mask_override`; only then may the display read "Frustrated / Masked Stress" (core-4 phase) and the narrator use mask language. |
| Explanation string interpolating raw row values (`ambient_db`, `hours_since_break`) | `FactSheet` built from SHAP drivers with `level`/`effect`/`scope` → instant template text → LLM narration that the validator checks against the sheet. "Context corrected the facial mask" appears only with `mask_override`. |
| Shown on mock data only | Also shown live: the masked flip with `mask_override`, plus the counterfactual ("still Frustrated at 0.5 h because of fists and noise"). |

### B. "Focused paradox" (curing context blindness)

| Original | New design |
|---|---|
| Focused rows: face −0.4 to 0.0 ("looks angry"), 40–55 dB, bright, stable breaks | Simulator v2 Focused: face mean −0.15, quiet 40–55 dB, normal light, 1–3 h since break, head-in-hands 0.2 ("thinking"). The negative face overlaps with frustration (and with Tense when P1 lands). |
| "Context clears up resting-hacker-face misclassifications" | Headline metric: M0's Focused → Frustrated/Tense confusion vs M2's (TBD — produced by `valence/eval/evaluate.py`). |
| (no scenario) | `scenarios/context_blindness.yaml`: the same head-in-hands pose in two contexts gives two labels. |
| — | Here the correction is visible as `face_only.label` ≠ `label` on the `prediction`, with drivers explaining it. It is not a mask, so `mask_override` must stay false on the `focused` scenario (section 9.4 target). |

### C. XAI audit trail

The original pseudocode built one string from raw row values. The new trail has three parts:

1. A per-prediction `FactSheet` with the PRD's three context layers plus the Facial Surface layer (`LAYERS`).
2. Grounded text, from the template instantly and from the local LLM when validated.
3. A tamper-evident `AuditLog` of decisions (never features), verifiable via `/api/governance/verify`.

---

## 15. Performance budgets

| Stage | Budget / target | Measured (`BENCHMARKS.md`, M3 Max 36 GB) |
|---|---|---|
| Perception loop | 100 ms per frame at 10 Hz; live ≥ 8 Hz over 60 min | `VideoPerceiver.process` 45 ms p50 / ~51 ms p95 (2 people, sample images); live camera pending |
| YOLO26n-pose + ByteTrack (MPS, 640) | — | 18.4 / 21.0 ms |
| FaceLandmarker on head crop | — | ~11 ms per person |
| GestureRecognizer (full frame) | — | 16.4 / 16.8 ms |
| YAMNet (stream) | 1 result/s | ~6 ms callback |
| Vosk per 100 ms block | — | p50 0.84 ms, max 29 ms |
| SHAP `pred_contrib` | < 5 ms | TBD — produced by the planned predictor benchmark |
| Tick: features → `prediction` | < 200 ms | TBD — produced by `test_e2e_replay` |
| Narrator | p95 ≤ 2.5 s with YOLO running | Qwen3-1.7B: 536 ms for 55 tokens without YOLO load; with YOLO TBD (LOCAL A/B) |
| Memory | Flat over a 60-min soak | TBD — produced by `scripts/mem_soak.py` (`memory_flat.png`) |

---

## 16. Open items

1. **Fist rule.** Align `fist_debouncer()` (`debounce.py`, currently ≥ 0.6 for 3 samples) with the plan's
   rule (≥ 0.7 held ≥ 0.5 s, keyboard-zone exclusion). This is an edge-side task.
2. **Mask override on Focused.** Under the generic rule in 9.4, a Focused person whose face M0 reads as
   Frustrated could trigger `mask_override`. The target says it must never fire on `focused`. If scenario
   validation shows it firing, restrict the detector, for example to M2 labels where a mask is meaningful.
   Decide once M2 exists.
3. **Display-id release vs `person_drop_min`.** The edge releases an id 3 s after a track is lost and may
   reuse it for someone new; the aggregator keeps a slot for 15 min. The aggregator should treat an id that
   reappears after a gap longer than the release window as a new person: clear the slot and inherit room
   history. To be confirmed in `aggregator.py` and `test_window`.
4. **Driver scope for `hours_since_break`.** The break clock is room-level (6.1), but the API example in
   `FRONTEND_PROMPT.md` tags it `scope: "person"`. Fix the scope of the time channels, and the `LAYER_MAP`
   entry for `fill_ratio`, in `spec.py`.
5. **Room-only model.** Which model serves the single room label in `--room-only` (for example M2-noface on
   room-aggregated channels) is not yet specified.
6. **`social_withdrawal_flag` rule.** Not yet defined (P1).
7. **`end_tick()` timing.** The ticker thread calls `end_tick()` while perception threads are purging. A
   buffer purged microseconds earlier whose caller has not yet run `del` would count as a violation. Purge must
   be the last use of a buffer. If the 60-min soak shows spurious violations, check only buffers purged
   before the previous `end_tick()`.
