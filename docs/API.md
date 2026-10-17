# Valence API

Contract version **1.0.0** (`CONTRACT_VERSION` in `valence/contracts.py`).

This is the interface between the Valence backend and any client: the Lovable "Valence Audit Console"
(`docs/FRONTEND_PROMPT.md`), `scripts/console.py`, tests, and the local narrator service. Field names in this
document are copied from `valence/contracts.py`. If the two ever disagree, `contracts.py` wins. Report the
mismatch through `docs/CONTRACT_REQUESTS.md`.

**All numbers in the examples are illustrative placeholders, not measured results.** Measured performance
is in `docs/BENCHMARKS.md`. Evaluation results are TBD and will come from `valence/eval/evaluate.py`.

## Contents

1. [Implementation status](#1-implementation-status)
2. [Base URL, CORS and conventions](#2-base-url-cors-and-conventions)
3. [WebSocket `/ws/events`](#3-websocket-wsevents)
4. [Drawing boxes over a browser camera preview](#4-drawing-boxes-over-a-browser-camera-preview)
5. [Pairing `prediction` and `explanation` by `event_id`](#5-pairing-prediction-and-explanation-by-event_id)
6. [Provenance: `source` and `simulated_fields`](#6-provenance-source-and-simulated_fields)
7. [REST endpoints](#7-rest-endpoints)
8. [Audit log: operator and SIMULATED actions](#8-audit-log-operator-and-simulated-actions)
9. [Internal narrator API (`127.0.0.1:8081`)](#9-internal-narrator-api-1270018081)
10. [Versioning and JSON Schema export](#10-versioning-and-json-schema-export)
11. [Known gaps and open items](#11-known-gaps-and-open-items)

---

## 1. Implementation status

Status as of Saturday evening, 2026-10-17. A committed file is not a claim that its tests pass.

| Piece | File | Status |
|---|---|---|
| Message models, `FactSheet`, `NarrationResult`, `json_schemas()` | `valence/contracts.py` | Committed. Tests in `tests/test_contracts.py`. |
| REST and WebSocket server | `valence/api/app.py` | Committed. Tests in `tests/test_api.py`. This document describes the behavior it implements; planned behavior is marked. |
| Runtime that produces the messages | `valence/runtime/engine.py`, `bus.py`, `aggregator.py`, `recording.py` | Committed. Tests in `tests/test_runtime.py`. |
| Served model | `valence/models/served.py` (M2 + M0 shadow), `valence/models/predictor.py` (smoothing, rule stub) | The trained v1 M2 (`model_id` `"m2_lgbm@<8 hex>"`, core 4 states) is served when `artifacts/model_meta.json` matches `settings.state_set`; otherwise the rule stub (`"stub-rules"`) is served. |
| Counterfactual | `valence/models/served.py` (`ServedPredictor.counterfactual`) | Implemented for the served M2; the stub returns `501`. |
| Evaluation reports | `valence/eval/evaluate.py` → `reports/metrics.json`, `reports/figs/*.png` | Script committed; v1 figures exist (uncommitted); `reports/metrics.json` does not exist yet, so `GET /api/eval/summary` returns `{"available": false}`. |
| Narrator service | `valence/narrator/service.py` (LOCAL session) | Stage A draft committed ("under review"). Without a running narrator, the template text in each `prediction` is the explanation. |

---

## 2. Base URL, CORS and conventions

### 2.1 Endpoints and ports

| Service | Address | Notes |
|---|---|---|
| REST | `http://127.0.0.1:8000/api/...` | The frontend's default `http://localhost:8000` points to the same server. If `localhost` resolves to IPv6 `::1` and fails, use `127.0.0.1`. |
| WebSocket | `ws://127.0.0.1:8000/ws/events` | Not under `/api`. |
| OpenAPI (FastAPI default) | `http://127.0.0.1:8000/docs`, `/openapi.json` | Request bodies are untyped there (see 7.1). For message schemas, use `json_schemas()` (section 10). |
| Narrator (internal) | `http://127.0.0.1:8081` | Called only by the backend. Clients never call it (section 9). |

Both services bind to loopback only (`settings.api_host` = `127.0.0.1`, `settings.api_port` = `8000`,
`settings.narrator_url` = `http://127.0.0.1:8081`). There is no authentication, no database and no user
account, because nothing is reachable from another machine.

Start the server:

```bash
VALENCE_MODE=replay uvicorn valence.api.app:app   # replay demo (default mode)
VALENCE_MODE=live   uvicorn valence.api.app:app   # live camera + mic (Mac). Never use --reload with live sensors.
```

In replay mode the server starts by looping the scenario `VALENCE_SCENARIO` (default `full_demo`) at speed
`VALENCE_SPEED` (default `1`).

### 2.2 CORS

The REST API allows these browser origins (regex in `valence/api/app.py`):

```
^(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://[a-z0-9-]+\.(lovable\.app|lovableproject\.com))$
```

That covers any port on `localhost` or `127.0.0.1` (the exported Lovable app run locally) and the hosted
Lovable preview origins. All methods and headers are allowed, and credentials (cookies) are not used. For the
demo, run the frontend locally. A hosted `https://` page that calls `http://localhost` can trigger Chrome's
local-network permission prompt, and Safari may block it.

CORS does not apply to WebSockets. See section 11 for the planned `Origin` check on `/ws/events`.

### 2.3 Conventions

| Topic | Rule |
|---|---|
| Encoding | JSON, UTF-8, `snake_case` keys. WebSocket messages are JSON text frames, one message per frame, discriminated by `"type"`. |
| Missing values | `null`. NaN and infinity are never sent. Fields typed `float \| null` are `null` when a sensor had no reading (for example `facial_valence` when no face is visible). |
| Times | `t` is wall-clock Unix time in seconds (`time.time()`) for `governance`, `state_vector`, `prediction` and `heartbeat`. **Exception:** `tracks.t` is the backend's monotonic perception clock. Use `tracks.capture_ts_ms` (Unix ms) for wall time. |
| People | `person_id` is an ephemeral integer display id: 1, 2, 3 in order of first appearance. It is reused if the person reappears within 3 s near the same spot, and released after that. It is never linked to identity. Display it as `"Person #N"`. `person_id` `0` is the room (room-only mode, section 7.4.10). At most `settings.max_tracks` (3) people are tracked. |
| States | Use the `id` (for example `Frustrated_Masked`) as a key and `label_display` for text. Get the list from `GET /api/states` or `hello.states` (4 core states, or 6 with the P1 states). Never hardcode it. |
| Defaults | The server always sends every field, including fields that have defaults. In the JSON Schemas, `required` lists only fields without defaults (section 10). |
| Errors | See 7.1. |

---

## 3. WebSocket `/ws/events`

### 3.1 Connection lifecycle

1. Connect to `ws://127.0.0.1:8000/ws/events`. The server accepts and immediately sends **`hello`**.
2. The server then forwards every message on the event bus. The server never reads from the socket, so
   anything the client sends is ignored. All control goes through REST.
3. Nothing is replayed after `hello`. To fill the screen right away, call `GET /api/state` (latest
   `state_vector` and `prediction` per person) and `GET /api/history?minutes=15` (timeline) after connecting.
4. Each client gets a **bounded queue of 200 messages**. If the client falls behind, the **oldest** messages
   are dropped. A slow or absent client never blocks or slows the pipeline.
5. `status` arrives about every 1 s and `heartbeat` about every 5 s. If nothing arrives for 10 s, reconnect
   with backoff (for example 0.5 s doubling to 5 s) and repeat step 3.

### 3.2 Message summary

| `type` | When | Rate | Scope |
|---|---|---|---|
| `hello` | First message on every connection | Once per connection | Session |
| `tracks` | A video observation arrives (live camera, or a features-only JSONL replay) | ≤ 10 Hz (throttled to one per 100 ms). Not sent during scenario replay, which has no boxes. | All visible people |
| `governance` | End of every tick, after the purge-gate check | One per tick: every 5 s at speed 1 | Session |
| `state_vector` | Every tick, just before that person's `prediction` | One per person per tick | Person (or room) |
| `prediction` | Every tick, once the person has a window | One per person per tick. The plan caps it at 1 Hz per person; sped-up replay shortens the tick, so rely on `t`, not arrival rate. | Person (or room) |
| `explanation` | Asynchronously, when the narrator returns text for a narrated `prediction` | Only on narration triggers: label change, confidence-band change, or 30 s since that person's last narration | Person (or room) |
| `status` | Periodic system status | About 1 Hz | Session |
| `heartbeat` | Keep-alive | About every 5 s | Session |

A tick is `settings.tick_s` = 5 s of wall time. During replay at speed *k*, a tick takes 5 s / *k* of wall time.

### 3.3 `hello`

Sent once, first, on every connection. It is not re-sent when settings change, so after a control call that
changes mode or room-only, re-read the REST response or `GET /api/state`.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"hello"` | Discriminator |
| `contract_version` | string | `CONTRACT_VERSION`, currently `"1.0.0"`. Check the major version (section 10). |
| `mode` | `"live" \| "replay" \| "hybrid"` | Current input mode. `hybrid` means one modality live and the other replayed, or live with simulated inputs. |
| `room_only` | bool | `true`: one aggregate room label and no per-person inference (section 7.4.10) |
| `states` | `StateInfo[]` | Active state set, in the order used by `probabilities`. Each item is `{id, display, tier}`: `id` is the state key, `display` its full name from `STATE_DISPLAY`, and `tier` is `"P0"` or `"P1"`. |
| `feature_version` | string | Opaque version of the model and feature spec. Currently the served model id, for example `"m2_lgbm@1a2b3c4d"` (or `"stub-rules"` when no trained model matches the state set). Show it, but do not parse it. |
| `frame_w`, `frame_h` | int | Size in pixels of the backend's processed video frame. `0` before the first frame and in scenario replay. |
| `mirrored` | bool | `true` if the backend frame, and so the box coordinates, are horizontally mirrored (section 4) |

```json
{"type": "hello", "contract_version": "1.0.0", "mode": "live", "room_only": false, "states": [{"id": "Focused", "display": "Focused", "tier": "P0"}, {"id": "Frustrated_Masked", "display": "Frustrated / Masked Stress", "tier": "P0"}, {"id": "Exhausted", "display": "Exhausted", "tier": "P0"}, {"id": "Collaborative_Happy", "display": "Collaborative / Happy", "tier": "P0"}], "feature_version": "m2_lgbm@1a2b3c4d", "frame_w": 1280, "frame_h": 720, "mirrored": false}
```

### 3.4 `tracks`

Normalized boxes for the people currently tracked. Sent at most every 100 ms while video observations flow.
It contains **no pixels, no keypoints and no face data**, only boxes, ephemeral ids and two booleans
(section 4.4). Tracks are not stored: they are not in `/api/history` and not in the audit log.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"tracks"` | Discriminator |
| `t` | float | Backend monotonic clock in seconds (`VideoObs.t`). Use it only to order messages and measure intervals. |
| `capture_ts_ms` | int | Wall-clock Unix time in ms of the processed frame. Use it for staleness checks (section 4.3). |
| `frame_w`, `frame_h` | int | Size of the frame the boxes refer to. Takes precedence over `hello`. |
| `mirrored` | bool | Box coordinates are in mirrored image space (section 4.2) |
| `persons` | `TrackPerson[]` | One entry per tracked person (at most 3). An empty list means nobody is tracked. |
| `persons[].id` | int | Ephemeral display id (same as `person_id` elsewhere) |
| `persons[].label` | string | `"Person #N"` |
| `persons[].bbox` | `[x1, y1, x2, y2]` floats | Normalized to [0, 1] relative to `frame_w` × `frame_h`. Origin top-left, x to the right, y down, with `x1 < x2` and `y1 < y2`. |
| `persons[].fist_active` | bool | A debounced clenched fist is held in this frame |
| `persons[].slump` | bool | Debounced posture slump in this frame |

```json
{"type": "tracks", "t": 48211.37, "capture_ts_ms": 1792276799871, "frame_w": 1280, "frame_h": 720, "mirrored": false, "persons": [{"id": 1, "label": "Person #1", "bbox": [0.12, 0.2, 0.41, 0.95], "fist_active": true, "slump": false}, {"id": 2, "label": "Person #2", "bbox": [0.55, 0.18, 0.86, 0.97], "fist_active": false, "slump": false}]}
```

### 3.5 `governance`

One per tick. It is the data-minimization evidence shown in the console's purge log. All counters except
`live_raw_buffers` are cumulative for the server process.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"governance"` | Discriminator |
| `t` | float | Wall time of the tick (s) |
| `tick` | int | Tick counter of this server process |
| `message` | string | The PRD log line `GOVERNANCE_LINE`: `"[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline."`. Display it, but never parse it; use the counters. |
| `frames_decoded` | int | Frames retrieved from the camera, as reported by the edge (`VideoObs.frames_decoded`) |
| `frames_purged` | int | Video buffers zero-filled by the `PurgeGate`. This includes frames displaced from the capture slot before processing. **Expected: `frames_decoded == frames_purged`.** Both are 0 in scenario replay. |
| `audio_blocks_purged` | int | Audio buffers zero-filled by the `PurgeGate` |
| `transcripts_purged` | int | Recognizer outputs discarded inside perception. The restricted grammar can only emit fixed phrases or `[unk]`, so no free transcript exists. |
| `violations` | int | Purged buffers still referenced at the end of a tick (`PURGE_VIOLATION`). Target: 0. |
| `live_raw_buffers` | int | Purged buffers still alive when this tick closed. Survivors are counted as violations and then no longer tracked. Target: 0. |
| `audit_hash` | string \| null | First 16 hex characters of the latest audit-entry SHA-256. Sixteen zeros (genesis) before the first entry. The full hash is in `GET /api/governance/stats`. |
| `retention_s` | int | Longest time any derived record can remain on disk: the audit-file TTL (`settings.audit_ttl_s` = 86 400 s), which applies only after a crash. Normally the file is deleted at session close. |

```json
{"type": "governance", "t": 1792276800.0, "tick": 348, "message": "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.", "frames_decoded": 17402, "frames_purged": 17402, "audio_blocks_purged": 3480, "transcripts_purged": 212, "violations": 0, "live_raw_buffers": 0, "audit_hash": "9f2c4e1a7b3d5c60", "retention_s": 86400}
```

### 3.6 `state_vector`

The current values of one person's rolling state vector, for the "current values" panel and the timeline.
It is sent every tick per person, just before the `prediction` for the same tick.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"state_vector"` | Discriminator |
| `t` | float | Wall time of the tick (s) |
| `person_id` | int | Ephemeral id (0 = room) |
| `ambient_db` | float \| null | Room noise this tick: Leq, approximate dB SPL (calibrated mic dBFS + offset). Room scope. |
| `light_lux` | float \| null | Estimated light level in lux (10 s median from the webcam, or simulated). Room scope. |
| `hours_since_break` | float \| null | Hours since the last break: presence-based clock, reset by `POST /api/events/break` or by an absence of `settings.break_absent_min` |
| `time_of_day` | string | Local time `"HH:MM"` |
| `clenched_fists_5m` | int | Debounced fist onsets for this person in the last 5 min |
| `compliments_15m` | int | Near-field compliment phrase hits in the room in the last 15 min |
| `laughter_15m` | int | Near-field laughter onsets in the room in the last 15 min |
| `posture_slump` | float \| null | Fraction [0, 1] of this tick's frames with a debounced slump |
| `head_in_hands` | float \| null | Fraction [0, 1] of this tick's frames with head in hands |
| `social_withdrawal` | int | 1 if this tick was mostly slumped and nearly still, else 0 |
| `face_visible` | bool | A readable face was seen during this tick |
| `facial_valence` | float \| null | Median face valence this tick, in [-1, 1], relative to the person's own neutral baseline (0 is their neutral). `null` when no face was visible. |
| `window_fill_min` | float | Minutes of this person's 15-minute window that are filled (0–15) |
| `source` | string | Provenance of this person's latest tick (section 6) |
| `simulated_fields` | string[] | Simulated or operator-set data inside this window (section 6) |

"Near-field" means a laughter or compliment event counts only if its audio block is at least
`settings.near_field_db` (6 dB) above the room's running median, so the next table in a loud hall does not
count.

```json
{"type": "state_vector", "t": 1792276800.0, "person_id": 1, "ambient_db": 72.4, "light_lux": 210.0, "hours_since_break": 4.2, "time_of_day": "15:40", "clenched_fists_5m": 3, "compliments_15m": 0, "laughter_15m": 0, "posture_slump": 0.35, "head_in_hands": 0.0, "social_withdrawal": 0, "face_visible": true, "facial_valence": 0.62, "window_fill_min": 12.5, "source": "live", "simulated_fields": []}
```

### 3.7 `prediction`

One per person per tick. It carries the label, all probabilities, the face-only baseline, explanation layers
and an instant template explanation.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"prediction"` | Discriminator |
| `event_id` | string | Unique within the server process. Used to pair with `explanation` (section 5) and for the counterfactual endpoint. It is currently formatted `p<person_id>-<tick, 6 digits>` (for example `p1-000348`), but clients must treat it as opaque. |
| `t` | float | Wall time of the tick (s) |
| `person_id` | int | Ephemeral id (0 = room) |
| `label` | string | State id. It is **smoothed**: an EMA over the probabilities plus hysteresis, so it changes only after a new state wins 2 consecutive ticks. |
| `label_display` | string | Text to show, from `display_label()`. Examples: `"Frustrated"` (core-4 phase, no mask override), `"Frustrated / Masked Stress"` (mask override, or once Tense is in the state set), `"Upbeat"` (Collaborative_Happy with one person). |
| `confidence` | float | `probabilities[label]`: calibrated (temperature-scaled softmax) and **unsmoothed**. Calibration was fit on simulated data, so label it **"sim-calibrated"** in the UI. While the label is changing, `label` can briefly differ from this tick's highest probability. |
| `probabilities` | object `{state_id: float}` | Calibrated probability of every active state, summing to about 1. Show all of them. |
| `face_only` | `{label, confidence}` | Face-only baseline (M0) for this tick. **Both `null` when no face was visible.** `label` is a state id. |
| `mask_override` | bool | The face-only baseline and the context model disagree with enough confidence for several ticks, the face evidence inside M2 points to M0's state, and the person's window is at least 2 min old. As implemented: M0 reads `Collaborative_Happy` with confidence ≥ 0.45 while M2 (confidence ≥ 0.70) does not; on after ≥ 3 of the last 6 face-visible ticks, off at ≤ 1. The plan's rule is M0 ≥ 0.55 for 3 consecutive ticks (`docs/ARCHITECTURE.md` 9.4). Show "Context corrected the facial mask". |
| `warming_up` | bool | `true` during the person's first `settings.warmup_s` (60 s) |
| `layers` | `Layer[]` | Explanation layers, strongest support first (see below) |
| `data_quality` | object | `face_visible_frac`: fraction of the last 5 min with a visible face. `lux`: latest lux or `null`. `ood_features`: names of features clipped to the training range; when more than 3 are clipped, confidence is lowered and the narration says so. |
| `explanation` | `{text, source}` | Instant explanation built from the FactSheet. `source` is always `"template"` here. A later `explanation` message may replace it (section 5). |
| `source`, `simulated_fields` | string, string[] | Provenance (section 6) |

`Layer` and `Driver`, shared with the FactSheet (section 9):

| Field | Type | Meaning |
|---|---|---|
| `layers[].name` | `"Ambient Physics" \| "Time Dynamics" \| "Somatic & Social" \| "Facial Surface"` | The PRD's three context layers plus the face |
| `layers[].contribution` | float | Sum of the attributions of all narratable features in the layer toward the predicted label (`fill_ratio` is never narrated and is left out): TreeSHAP in model-margin units for M2, linear terms for the stub. Positive supports the label, negative opposes it. The Facial Surface layer is kept even when it opposes, so a corrected mask stays visible. |
| `layers[].drivers` | `Driver[]` | Up to 3 strongest drivers in the layer by absolute `shap`. They need not add up to `contribution`. |
| `drivers[].feature` | string | Feature name, `<channel>_<stat>_<window>` (for example `db_mean_15m`, `fist_cnt_5m`, `face_now`, `hours_since_break`) |
| `drivers[].display` | string | Pre-formatted text, for example `"average noise of 72 dB over 15 minutes"`. The only place numbers are formatted. |
| `drivers[].value` | float \| null | Raw feature value |
| `drivers[].unit` | string | Unit, for example `dB`, `lux`, `h`, `events`, `%`, `sw/s` (shoulder-widths/s), or `""` |
| `drivers[].shap` | float | This feature's attribution |
| `drivers[].level` | `"high" \| "elevated" \| "normal" \| "low"` | Where the value sits against fixed thresholds (for example noise: high ≥ 65 dB, elevated ≥ 58, low < 45) |
| `drivers[].effect` | `"supports" \| "opposes"` | `supports` if `shap` ≥ 0 |
| `drivers[].scope` | `"room" \| "person"` | `room` for noise, light, laughter, compliments and `n_people`; `person` otherwise. Text should phrase room drivers as "the room". |

```json
{"type": "prediction", "event_id": "p1-000348", "t": 1792276800.0, "person_id": 1, "label": "Frustrated_Masked", "label_display": "Frustrated / Masked Stress", "confidence": 0.88, "probabilities": {"Focused": 0.04, "Frustrated_Masked": 0.88, "Exhausted": 0.05, "Collaborative_Happy": 0.03}, "face_only": {"label": "Collaborative_Happy", "confidence": 0.71}, "mask_override": true, "warming_up": false, "layers": [{"name": "Somatic & Social", "contribution": 0.41, "drivers": [{"feature": "fist_cnt_5m", "display": "3 clenched-fist events in the last 5 minutes", "value": 3.0, "unit": "events", "shap": 0.38, "level": "high", "effect": "supports", "scope": "person"}]}, {"name": "Ambient Physics", "contribution": 0.22, "drivers": [{"feature": "db_mean_15m", "display": "average noise of 72 dB over 15 minutes", "value": 72.4, "unit": "dB", "shap": 0.22, "level": "high", "effect": "supports", "scope": "room"}]}, {"name": "Time Dynamics", "contribution": 0.15, "drivers": [{"feature": "hours_since_break", "display": "4.2 hours since the last break", "value": 4.2, "unit": "h", "shap": 0.15, "level": "high", "effect": "supports", "scope": "person"}]}, {"name": "Facial Surface", "contribution": -0.18, "drivers": [{"feature": "face_now", "display": "smiling expression (valence +0.62)", "value": 0.62, "unit": "", "shap": -0.18, "level": "high", "effect": "opposes", "scope": "person"}]}], "data_quality": {"face_visible_frac": 0.9, "lux": 210.0, "ood_features": []}, "explanation": {"text": "Model predicted Frustrated / Masked Stress (88%) for Person #1. The face alone suggests Collaborative / Happy, but 3 clenched-fist events in the last 5 minutes, average noise of 72 dB over 15 minutes and 4.2 hours since the last break point the other way. Context corrected the facial mask.", "source": "template"}, "source": "live", "simulated_fields": []}
```

When no face is visible, the same message has `"face_only": {"label": null, "confidence": null}`, the
Facial Surface layer reduces to face-visibility drivers or disappears, and the template switches to its
occlusion clause ("The face is not readable right now, so the call rests on context: ...").

### 3.8 `explanation`

The narrator's text for one earlier `prediction`, matched by `event_id` (section 5).

| Field | Type | Meaning |
|---|---|---|
| `type` | `"explanation"` | Discriminator |
| `event_id` | string | The `event_id` of the `prediction` this text explains |
| `person_id` | int | Same as that prediction |
| `text` | string | Validated narration: 20–90 words, numbers only from the fact sheet, "Person #N" only, no pronouns, demographics or diagnoses |
| `source` | `"llm" \| "llm_retry" \| "template"` | `llm`: the first generation passed the validator. `llm_retry`: a retry passed. `template`: every attempt failed validation and the narrator returned its template fallback. |
| `latency_ms` | int | Narrator round trip measured by the backend. On a backend cache hit, it is the latency reported for the original narration. |

```json
{"type": "explanation", "event_id": "p1-000348", "person_id": 1, "text": "Person #1 reads as Frustrated / Masked Stress (88%). The face alone suggests Collaborative / Happy, but 3 clenched-fist events in the last 5 minutes, the room averaging 72 dB over 15 minutes and 4.2 hours since the last break outweigh the smile. Context corrected the facial mask.", "source": "llm", "latency_ms": 540}
```

If the narrator is down, slow or returns an invalid response, **no `explanation` is sent**. The template
text already in the `prediction` stays, and `status.narrator` reports the problem.

### 3.9 `status`

Periodic system status, about 1 Hz.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"status"` | Discriminator |
| `mode` | `"live" \| "replay" \| "hybrid"` | Current input mode |
| `camera` | bool | Live mode and a video observation arrived in the last 3 s |
| `mic` | bool | Live mode and an audio observation arrived in the last 3 s |
| `fps` | float | Processed video observations per second (perception rate; target 10, 0 in scenario replay) |
| `loop_latency_ms` | float | Time the runtime took to process the last tick: windows, features, models, SHAP, messages |
| `narrator` | `"ready" \| "fallback" \| "down"` | `ready`: the narrator answered its last health check or request. `fallback`: it did not, so template text is in use. `down` is reserved for a narrator that is known to be unreachable; the current runtime reports only `ready` or `fallback`. |
| `offline` | bool | The offline settings (`HF_HUB_OFFLINE`, `YOLO_OFFLINE`) are set, so no library will try to download anything. It is a configuration check, not a network probe. Valence needs no network after the one-time model downloads. |
| `uptime_s` | float | Seconds since the runtime started |
| `degraded` | string[] | Components not working normally. Current values: `camera`, `mic` (live mode with no recent observation), `narrator`, `predictor:stub` (trained model not loaded), `edge` (the perception runner failed to start). |
| `purge_violation` | bool | `true` once any purge violation has occurred in this process (it stays set). Show it as an alarm. |

```json
{"type": "status", "mode": "live", "camera": true, "mic": true, "fps": 9.6, "loop_latency_ms": 45.0, "narrator": "ready", "offline": true, "uptime_s": 1820.0, "degraded": [], "purge_violation": false}
```

### 3.10 `heartbeat`

Keep-alive, about every 5 s.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"heartbeat"` | Discriminator |
| `t` | float | Wall time (s) |

```json
{"type": "heartbeat", "t": 1792276805.0}
```

---

## 4. Drawing boxes over a browser camera preview

The console shows a camera preview from the **browser's own** `getUserMedia` stream. The backend opens the
same physical camera separately for perception. Frames never travel between the two: the backend sends only
normalized boxes, and the browser preview never leaves the browser.

### 4.1 Recommended setup

1. After `hello`, request the same resolution as the backend so both views share an aspect ratio:
   `navigator.mediaDevices.getUserMedia({video: {width: {ideal: frame_w}, height: {ideal: frame_h}}})`.
   Pick the same device the backend uses (`enumerateDevices()`, match by label). A different camera, for
   example a Continuity Camera phone, will not line up.
2. Put the `<video>` and an overlay `<canvas>` in one container with CSS `aspect-ratio: frame_w / frame_h`.
   Give the video `object-fit: cover`, and position the canvas absolutely over the full container.
3. Then a normalized coordinate maps directly to pixels: `px = x * W`, `py = y * H`, where `W` and `H` are the
   container's CSS size. If the browser stream's aspect ratio still differs, `object-fit: cover`
   center-crops it, and alignment is approximate.

### 4.2 Mirroring

`mirrored: true` means the backend flipped its frame horizontally before detection, so box x coordinates are
in mirrored space. A plain `<video>` element shows the camera unmirrored. Therefore:

- If `mirrored` is true, flip the boxes: `x1' = 1 − x2`, `x2' = 1 − x1` (y is unchanged).
- To show a selfie-style mirrored preview, mirror the **whole container** (video and canvas together) with
  CSS `transform: scaleX(-1)`. The boxes then follow automatically, and the rule above still applies. If you
  mirror the container, draw the label text un-mirrored.

```ts
function drawTracks(msg: Tracks, canvas: HTMLCanvasElement): void {
  const W = canvas.clientWidth, H = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (Date.now() - msg.capture_ts_ms > 1000) return;          // stale: draw nothing
  for (const p of msg.persons) {
    let [x1, y1, x2, y2] = p.bbox;
    if (msg.mirrored) [x1, x2] = [1 - x2, 1 - x1];
    ctx.strokeRect(x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
    ctx.fillText(p.label, x1 * W + 4, y1 * H + 14);            // "Person #1"
  }
}
```

### 4.3 Timing

- Boxes arrive at up to 10 Hz, about one perception step behind the backend's camera. The measured
  perception step is about 45 ms p50 for 2 people on sample images (`docs/BENCHMARKS.md`). Live-camera
  numbers are still pending.
- `capture_ts_ms` and `Date.now()` come from the same machine's clock. Clear the boxes when the newest
  `tracks` message is more than about 1 s old, or when no `tracks` message has arrived for 1 s.
- **Never delay or buffer the browser video to line it up with the boxes.** That would mean storing frames.
  Draw the latest boxes on the live preview and accept a small lag.

### 4.4 Modes without a camera

| Situation | What arrives | What to draw |
|---|---|---|
| Live | `tracks` at ≤ 10 Hz | Boxes over the browser preview |
| Replay of a features-only JSONL recording (`POST /api/replay_recording/{name}`) | `tracks` from the recorded boxes | Boxes on a blank canvas sized `frame_w` × `frame_h` |
| Scenario replay (`POST /api/replay/{scenario}`) | No `tracks` (simulated scenarios have no boxes) | A blank canvas. List people from `prediction.person_id`. |
| Room-only mode | `tracks` unchanged: detection still counts people and feeds aggregate body signals | Boxes, but no per-person state |

### 4.5 Why pixels and keypoints are never sent

- **Data minimization is the project's thesis.** Each raw frame goes through the `PurgeGate`
  (`valence/governance/purge.py`) right after feature extraction: it is zero-filled in place, tracked by a
  weak reference until freed, and any survivor at tick end counts as a `PURGE_VIOLATION`. Sending pixels
  would create copies outside the gate and break `frames_decoded == frames_purged`.
- **Keypoints are body biometrics.** Pose keypoints stay inside the perception thread and are dropped after
  each frame. They are not in `VideoObs`, not in `TickRecord`, and not on the WebSocket. Only the derived
  flags (`fist_active`, `slump`) and a normalized box leave perception.
- **No face data leaves perception** except one scalar per person (`face_valence`), which is relative to the
  person's own neutral baseline. No landmarks, embeddings or face crops leave.
- **The browser preview is not backend data.** The API has no endpoint that accepts images or audio, and the
  only images it serves are offline evaluation charts (`/api/eval/figures/{name}.png`).
- Boxes are coarse, live-only and short-lived. They are not written to history, the audit log or disk.

---

## 5. Pairing `prediction` and `explanation` by `event_id`

1. Every `prediction` carries an `event_id` and an instant template explanation
   (`explanation.source` = `"template"`).
2. The backend asks the narrator for text only on **triggers**: a change of `label_display`, a change of
   confidence band (high ≥ 0.85, medium ≥ 0.60, low below that), or 30 s since that person's last narration.
   At most one request per person is in flight, and a newer fact sheet replaces any queued one.
3. When the narrator answers, an `explanation` with the **same `event_id`** follows, usually within the
   narrator deadline (`settings.narrator_deadline_s` = 2.5 s). The backend's HTTP timeout is 3.0 s.

Client rules:

| Case | Rule |
|---|---|
| `explanation.event_id` equals the person's latest `prediction.event_id` | Replace the template text with `explanation.text` and show `explanation.source`. |
| A newer `prediction` arrives with the same `label_display` | Keep showing the latest `explanation` text, marked "as of" its prediction's time. Its numbers describe that earlier event. |
| A newer `prediction` arrives with a different `label_display` | Show the new prediction's template text until its own `explanation` arrives. |
| `explanation` for an `event_id` older than what is shown, or unknown | Never let it replace text from a newer event. Drop explanations for unknown event ids (for example from before a reset or reconnect). |
| No `explanation` ever arrives | Normal when the narrator is down or between triggers. The template text is a complete, validated-format explanation. |

`GET /api/history` returns `prediction` messages with their template text only. `explanation` messages are
live-only.

---

## 6. Provenance: `source` and `simulated_fields`

Every `TickRecord`, `state_vector` and `prediction` carries both fields.

### 6.1 `source`

| Value | Meaning |
|---|---|
| `live` | From the camera and mic through the live aggregator |
| `replay` | From a scenario rendered by the simulator (`POST /api/replay/{scenario}`), or from a recorded features-only JSONL file (`POST /api/replay_recording/{name}`) |
| `sim` | Simulator training data. Normally never on the live stream. |
| `sim_seed` | Room history pre-filled by `POST /api/demo/seed` (context only, no face) |

On `state_vector` and `prediction`, `source` is the source of **that person's latest tick**. A live person
in a seeded room shows `source: "live"` plus `"room_history"` in `simulated_fields`.

### 6.2 `simulated_fields`

The names of values that were set by an operator or simulator instead of sensed. On `state_vector` and
`prediction`, it is the union of the tags on the person's latest tick and on **every room tick still in the
15-minute window**. A tag therefore stays visible until the simulated data has left the window, up to 15 min
after the input was switched off.

| Tag | Set by | Meaning |
|---|---|---|
| `room_history` | `POST /api/demo/seed` | Room slot pre-filled from a scenario. Those rows have `source = "sim_seed"`. |
| `hours_since_break` | `POST /api/demo/seed` with `hours_since_break` | Operator-set break clock for the people present at seed time |
| `face_occlusion` | `POST /api/sim/face_occlusion` | Face fields blanked (`face_valence` NaN, `face_visible` 0) |
| `low_light` | `POST /api/sim/low_light` | Lux scaled down and face valence made noisier. Current implementation: lux × 0.15 (30 lux if unknown), Gaussian noise σ 0.35 on valence. |
| `laughter_count`, `compliments_count`, `clenched_fists_count` | `POST /api/sim/event` | Injected event counts (`aggregator.inject`) |
| `light_lux` | `POST /api/sim/lux` | Overridden lux (`aggregator.lux_override`) |

A logged break (`POST /api/events/break`) and venue calibration (`POST /api/calibrate/room`) are real events or
measurements entered by the operator. They are written to the audit log but not tagged here.

**UI rule** (`docs/FRONTEND_PROMPT.md`, item 10): show a badge when `source` ≠ `"live"` (for example "REPLAY")
or when `simulated_fields` is non-empty (for example "SIMULATED: hours_since_break").

---

## 7. REST endpoints

### 7.1 Common behavior

- Prefix `/api`. JSON in, JSON out, except the PNG figures.
- Request bodies are free-form JSON objects. They are **not schema-validated yet**: unknown keys are
  ignored, and a wrong type can produce a `500`. Send exactly the documented fields and types. A missing
  optional body is treated as `{}`.
- Every POST returns `{"ok": true, ...}` with the new relevant state, so the UI does not have to wait for the
  WebSocket.

| Status | Body | When |
|---|---|---|
| `200` | Endpoint-specific | Success. Some endpoints report a soft failure inside the body (`"ok": false`, `"available": false`, `"healthy": false`). |
| `404` | `{"detail": "<message>"}` | Unknown scenario, overlay, control action, figure, or a person no longer in the window |
| `422` | `{"detail": [...]}` (FastAPI validation) or `{"detail": "<message>"}` | Query parameter of the wrong type, missing required body, invalid state name |
| `501` | `{"detail": "<message>"}` | Feature needs the trained model (counterfactual) |
| `503` | `{"detail": "runtime not started"}` | Request before startup finished or after shutdown began |
| `500` | Plain text | Unhandled error, for example a malformed body value |

### 7.2 Endpoint index

"Audit" names the `operator` action written to the hash-chained audit log (section 8). "SIMULATED" means the
call injects data that is not sensed, so it is tagged in `simulated_fields` or `source`.

| Method | Path | Purpose | Audit action | SIMULATED |
|---|---|---|---|---|
| GET | `/api/states` | Active state list | — | — |
| GET | `/api/state` | Latest per-person snapshot and status | — | — |
| GET | `/api/history?minutes=15` | Timeline of recent `state_vector` and `prediction` messages | — | — |
| GET | `/api/governance/stats` | Purge counters | — | — |
| GET | `/api/governance/audit?since=` | Audit entries | — | — |
| GET | `/api/governance/verify` | Recompute the hash chain | — | — |
| GET | `/api/retention` | Data-retention summary | — | — |
| GET | `/api/eval/summary` | Evaluation metrics (TBD) | — | — |
| GET | `/api/eval/figures/{name}.png` | Evaluation chart (TBD) | — | — |
| GET | `/api/scenarios` | Replay scenarios | — | — |
| GET | `/api/recordings` | Features-only recordings in `data/real/` | — | — |
| GET | `/api/models` | Served model | — | — |
| GET | `/api/health` | Backend health | — | — |
| GET | `/api/narrator/health` | Narrator health | — | — |
| GET | `/api/explain/{event_id}/counterfactual?feature=&value=` | One-feature what-if | — | Result is hypothetical; nothing is changed |
| POST | `/api/control/{start\|stop\|pause}` | Start, privacy pause, stop and clear | `control_start` / `control_pause` / `control_stop` (+ `reset` on stop) | no |
| POST | `/api/live` | Switch to live sensors | `live`, `reset` | no |
| POST | `/api/events/break` | Log a break | `break` | no (a real event entered by the operator) |
| POST | `/api/replay/{scenario}` | Play a scenario | `replay`, `reset` | `source = "replay"` |
| POST | `/api/replay_recording/{name}` | Replay a features-only recording through the live pipeline | `replay_recording`, `reset` | `source = "replay"` |
| POST | `/api/sim/face_occlusion` | Simulate a hidden face | `sim` | **yes**: `face_occlusion` |
| POST | `/api/sim/low_light` | Simulate low light | `sim` | **yes**: `low_light` |
| POST | `/api/sim/event` | Inject laughter, compliment or fist events | `sim_event` | **yes**: event-count tags |
| POST | `/api/sim/lux` | Override lux | `sim_lux` | **yes**: `light_lux` |
| POST | `/api/demo/seed` | Pre-fill room history | `demo_seed` | **yes**: `room_history`, `sim_seed`, optionally `hours_since_break` |
| POST | `/api/demo/reset` | Forget all people, keep the room | `reset` | no |
| POST | `/api/calibrate/room` | Venue calibration | `calibrate_room` | no (changes calibration, not data) |
| POST | `/api/mode` | Room-only on or off | `mode` | no |
| POST | `/api/label` | Self-report label (P1) | `self_report` | no (never enters the model) |

`POST /api/live`, `GET /api/recordings` and `POST /api/replay_recording/{name}` are not in the plan's endpoint
list. `live` is how the server returns from replay to live sensors; the recording routes are the camera/mic
fallback (replay a golden features-only recording made on the demo Mac).

### 7.3 Read endpoints

#### 7.3.1 `GET /api/states`

The active state set as a bare array (4 core states, or 6 with `settings.state_set = "all"`). `display` is
the full `STATE_DISPLAY` name; the per-prediction text (`"Frustrated"`, `"Upbeat"`) is in
`prediction.label_display`.

```json
[{"id": "Focused", "display": "Focused", "tier": "P0"}, {"id": "Frustrated_Masked", "display": "Frustrated / Masked Stress", "tier": "P0"}, {"id": "Exhausted", "display": "Exhausted", "tier": "P0"}, {"id": "Collaborative_Happy", "display": "Collaborative / Happy", "tier": "P0"}]
```

With 6 states, two more entries follow:
`{"id": "Overwhelmed", "display": "Overwhelmed / Distressed", "tier": "P1"}` and
`{"id": "Tense", "display": "Tense / Visible Frustration", "tier": "P1"}`. Errors: `503` only.

#### 7.3.2 `GET /api/state`

The latest `state_vector` and `prediction` per person, plus a `status` object. `people` is an object keyed
by `person_id` as a string. It is empty before the first prediction and after a reset.

```jsonc
{
  "people": {
    "1": {
      "state_vector": { "type": "state_vector", "person_id": 1, "...": "same shape as 3.6" },
      "prediction":   { "type": "prediction", "event_id": "p1-000348", "...": "same shape as 3.7 (template explanation)" }
    }
  },
  "status": { "type": "status", "mode": "live", "...": "same shape as 3.9" }
}
```

Errors: `503` only.

#### 7.3.3 `GET /api/history?minutes=15`

Recent `state_vector` and `prediction` messages, oldest first, as a bare array of the WebSocket message
objects. Use it to draw the 15-minute timeline, then extend the timeline from the live stream.

| Query | Type | Default | Notes |
|---|---|---|---|
| `minutes` | float | `15` | Messages with `t` ≥ now − `minutes`. Values above 60 are clamped to 60. A non-number gives `422`. |

The buffer holds 1,440 messages (two per person per tick): about 60 min for one person or 20 min for three.
It lives in RAM only and is never written to disk.

```jsonc
[
  { "type": "state_vector", "t": 1792275900.0, "person_id": 1, "...": "..." },
  { "type": "prediction",   "t": 1792275900.0, "person_id": 1, "event_id": "p1-000168", "...": "..." }
  // ... up to now
]
```

#### 7.3.4 `GET /api/governance/stats`

Process-wide purge-gate counters: the numbers to show after a soak test.

```json
{"frames_decoded": 17402, "frames_purged": 17402, "audio_blocks_purged": 3480, "derived_buffers_purged": {"video_derived": 26103, "audio_derived": 6960}, "freed": {"video": 17402, "audio": 3480, "video_derived": 26103, "audio_derived": 6960}, "unzeroable": {}, "transcripts_purged": 212, "violations": 0, "live_raw_buffers": 0, "retention_s": 86400, "near_field_gated_out": 7, "audit_hash": "9f2c4e1a7b3d5c60a1e8d4b27f0c3e9a5b6d1f8e2c7a4b9d0e3f6a1c8b5d2e7f"}
```

| Field | Meaning |
|---|---|
| `frames_decoded`, `frames_purged`, `audio_blocks_purged`, `transcripts_purged`, `violations`, `live_raw_buffers`, `retention_s` | As in the `governance` message (3.5) |
| `derived_buffers_purged` | Copies made inside perception (RGB conversion, head crops, PCM for Vosk, the YAMNet clip) zero-filled by the `PurgeGate`, by kind (`video_derived`, `audio_derived`). Counted separately so that `frames_decoded == frames_purged` can hold. |
| `freed` | Purged buffers whose memory was actually released, by kind (`video`, `audio`, `video_derived`, `audio_derived`). Kinds with a zero count are omitted. |
| `unzeroable` | Purged buffers that were read-only and could not be zero-filled, by kind. Target: empty. |
| `near_field_gated_out` | Laughter and compliment events discarded by the near-field gate |
| `audit_hash` | Full SHA-256 of the latest audit entry (64 hex characters) |

Errors: `503` only.

#### 7.3.5 `GET /api/governance/audit?since=0`

Entries of this session's audit log with `seq` ≥ `since`, as a bare array. `since` is an integer and
defaults to `0`. A non-integer gives `422`. To poll, pass the last `seq` you received + 1. Entry format and
action names are in section 8.

```json
[{"seq": 0, "ts": 1792275000.123, "kind": "operator", "payload": {"action": "live"}, "prev": "0000000000000000000000000000000000000000000000000000000000000000", "hash": "3b1d0c9e5f7a2846b1c3d5e7f9a0b2c4d6e8f0a1b3c5d7e9f1a2b4c6d8e0f2a4"}, {"seq": 1, "ts": 1792275005.456, "kind": "prediction", "payload": {"person": "Person #1", "label": "Focused", "confidence": 0.74, "mask_override": false, "source": "live", "layers": {"Ambient Physics": 0.31, "Time Dynamics": 0.12, "Somatic & Social": 0.08, "Facial Surface": -0.05}, "simulated_fields": []}, "prev": "3b1d0c9e5f7a2846b1c3d5e7f9a0b2c4d6e8f0a1b3c5d7e9f1a2b4c6d8e0f2a4", "hash": "c07e2a9d4b6f8103e5a7c9d1f3b5a7e9c1d3f5a7b9e1c3d5f7a9b1c3e5d7f9a1"}]
```

#### 7.3.6 `GET /api/governance/verify`

Recomputes the hash chain over the whole session file.

```json
{"ok": true, "entries": 412}
```

`ok: false` means an entry was edited, removed or reordered. In that case `entries` is the 1-based position
of the first entry that fails, so the bad `seq` is `entries − 1`. The same check runs offline in
`scripts/audit_verify.py` (planned). Errors: `503` only.

#### 7.3.7 `GET /api/retention`

A plain-language summary of what is held, where, and for how long. The full data-lifecycle table is in
`docs/PRD.md`.

```json
{"raw_media_stored": 0, "held": [{"what": "rolling state vector (numeric TickRecords)", "where": "RAM", "how_long": "15 minutes"}, {"what": "recent messages for /api/history", "where": "RAM", "how_long": "about 20 minutes"}, {"what": "audit log: labels, confidences, SHAP layer sums, operator actions", "where": "var/audit/session-20261017-151500.jsonl", "how_long": "deleted at session end; TTL 86400 s after a crash"}], "never_kept": ["video frames (zeroed and released each frame)", "audio blocks (zeroed each block)", "transcripts (Vosk grammar: only lexicon hits exist)", "pose keypoints (stay inside perception)", "face images or embeddings"]}
```

Self-report labels (`POST /api/label`) are also written to `var/labels/<session_id>.jsonl`. They are not yet
listed here (section 11).

#### 7.3.8 `GET /api/eval/summary`

Evaluation metrics from `reports/metrics.json` and the names of the available figures in `reports/figs/`.
**TBD: produced by `valence/eval/evaluate.py`.** Until `reports/metrics.json` exists (it does not yet), the
response is:

```json
{"available": false}
```

Once it has run, the response is `{"available": true, "metrics": {...}, "figures": ["accuracy_vs_history", ...]}`.
`metrics` is the contents of `reports/metrics.json`, whose keys are defined by `evaluate.py`. The plan's
headline items are macro-F1 vs history length with session-bootstrap 95% CIs for M0, M0t, M1, M2-noface, M2
and M3; `Frustrated_Masked` recall; the Focused-paradox confusion; ECE; robustness when the face is
degraded; and the negative control. In mock mode, the frontend shows placeholders.

#### 7.3.9 `GET /api/eval/figures/{name}.png`

Returns `image/png` from `reports/figs/{name}.png`. Names come from `figures` in `/api/eval/summary`, and
any file present is served even while the summary reports `available: false`. v1 files present (uncommitted;
core 4 states, untuned `noise_scale` 1.0): `accuracy_vs_history`, `acc_vs_episode_age`, `reliability`,
`confusion_m0_vs_m2`, `robustness_face_dropout`, `robustness_lux`. Planned: `memory_flat` (60-min soak). Paths are confined to `reports/figs/`. An unknown name or a
traversal attempt gives `404 {"detail": "figure not found"}`. These are offline charts, never camera images.

#### 7.3.10 `GET /api/scenarios`

Replay scenarios from `scenarios/*.yaml`, as a bare array. `minutes` is the sum of the segment lengths.

```json
[{"name": "collaborative_happy", "description": "Two people fresh off a break, laughing and complimenting each other in a bright, lively room.", "minutes": 12.0}, {"name": "frustrated_masked", "description": "Focused work turns into masked frustration - a forced smile over clenched fists, rising noise and a long stretch without a break.", "minutes": 14.0}, {"name": "full_demo", "description": "The pitch storyline - focused work, then masked frustration (the face says happy, context says otherwise), a break, and a collaborative recovery.", "minutes": 26.0}]
```

The full list is `collaborative_happy`, `context_blindness`, `exhausted`, `focused`, `frustrated_masked`,
`full_demo`, `overwhelmed` and `tense`. `overwhelmed` and `tense` script the P1 states and are meaningful
only with the 6-state set.

#### 7.3.11 `GET /api/models`

```jsonc
{"served": "m2_lgbm@1a2b3c4d", "states": ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"], "meta": {"state_set": "core", "sessions": 300, "seed": 7, "noise_scale": 1.0, "feature_names": ["db_now", "..."], "...": "..."}}
```

`served` is the served model's id: `"m2_lgbm@"` plus the first 8 hex characters of the SHA-256 of
`artifacts/m2.txt`, or `"stub-rules"` (transparent linear rules standing in for M2) when no trained model
matches `settings.state_set`. `meta` is the contents of `artifacts/model_meta.json` for the served M2
(states, training settings, temperatures, feature names and sets) and `{}` for the stub. Its keys are defined
by `valence/models/train_ladder.py`; do not rely on them in the UI.

#### 7.3.12 `GET /api/health`

```json
{"ok": true, "contract_version": "1.0.0", "mode": "replay", "model": "m2_lgbm@1a2b3c4d", "edge_error": null}
```

`edge_error` is the reason the perception runner failed to start (missing module, camera or mic permission),
or `null`. `ok` is `true` whenever the server can answer. Use `status` and `degraded` for component health.

#### 7.3.13 `GET /api/narrator/health`

The backend calls the narrator's `GET /health` (1 s timeout) and reports the result. It always returns
`200`, even when the narrator is down.

```json
{"healthy": true, "url": "http://127.0.0.1:8081", "failures": 0}
```

`failures` counts failed `/narrate` calls since startup.

#### 7.3.14 `GET /api/explain/{event_id}/counterfactual?feature=<name>&value=<number>`

A what-if: re-run the served model with exactly **one** feature changed and report the new probabilities.
It answers the question "is this driver really what drove the label?", for example: is Person #1 still
Frustrated with `hours_since_break=0.5` because of fists and noise? Nothing is changed and nothing is
published.

| Param | Type | Notes |
|---|---|---|
| `event_id` (path) | string | A `prediction.event_id`. The current implementation uses it only to select the person (`p<person_id>-...`) and re-runs on that person's **current** window. |
| `feature` (query) | string | A feature name from the served model's feature spec (for example `hours_since_break`, `db_mean_15m`, `fist_cnt_5m`) |
| `value` (query) | float | New value for that feature |

Response (`ServedPredictor.counterfactual` in `valence/models/served.py`). Labels are the unsmoothed
argmax of each re-run; `original_value` is `null` when the feature was NaN. The numbers below are
placeholders:

```json
{"feature": "hours_since_break", "original_value": 4.2, "counterfactual_value": 0.5, "original": {"label": "Frustrated_Masked", "confidence": 0.88}, "counterfactual": {"label": "Frustrated_Masked", "confidence": 0.81, "probabilities": {"Focused": 0.08, "Frustrated_Masked": 0.81, "Exhausted": 0.08, "Collaborative_Happy": 0.03}}, "label_changed": false, "simulated_fields": ["hours_since_break"]}
```

`simulated_fields` names the feature that was set by hand, so a UI can mark the result as hypothetical. The
design also called for echoing `event_id` and a `changed_features` check; the check lives in
`tests/test_served.py::test_counterfactual_changes_only_the_named_feature` instead.

| Error | When |
|---|---|
| `501` | The served model has no counterfactual support (the rule stub) |
| `404` | The person is no longer in the window (absent more than 15 min, reset, or replay restarted) |
| `422` | `feature` or `value` missing, or `value` not a number |
| `500` | `event_id` not in the `p<id>-<tick>` form, or `feature` not in the model's feature list (not yet mapped to `422`) |

### 7.4 Control and operator endpoints

All of these write an `operator` entry to the audit log (section 8).

#### 7.4.1 `POST /api/control/{start|stop|pause}`

No body.

| Action | Effect |
|---|---|
| `start` | Unpause. In live mode, (re)start the edge runner (camera and mic). Start the ticker if it is not running. |
| `pause` | **Privacy pause.** Stop ticking and release the camera and mic in live mode. The windows are kept in RAM. |
| `stop` | Pause, release the sensors, then **clear every window** (people and room). Resuming with `start` begins from an empty window. |

Response:

```json
{"ok": true, "paused": true}
```

Errors: an unknown action gives `404 {"detail": "action must be start, pause or stop"}`. Audit:
`control_start`, `control_pause` or `control_stop`. `stop` also writes `reset` with `keep_room: false`.

#### 7.4.2 `POST /api/live`

Switch the input to live sensors: clear all windows, start the edge runner, unpause, and restart the ticker.
No body.

```json
{"ok": true, "edge_error": null}
```

If the camera or mic cannot start, the call still returns `200` with `"ok": false` and the reason in
`edge_error`. `status.degraded` then contains `edge`. Audit: `live`, then `reset`.

#### 7.4.3 `POST /api/events/break`

Reset the break clock: `hours_since_break` restarts at 0. This affects the live aggregator only; scenario
replay uses the scenario's own clock.

| Body field | Type | Default | Notes |
|---|---|---|---|
| `person_id` | int \| null | `null` | `null` resets every tracked person. An unknown id does nothing. |

```json
{"person_id": 1}
```

```json
{"ok": true}
```

Audit: `break` with `person_id`. The break is a real event entered by the operator, so it is not added to
`simulated_fields`.

#### 7.4.4 `POST /api/replay/{scenario}`

Play a scenario from `GET /api/scenarios`. This releases live sensors, clears all windows, renders the
scenario with the simulator, and restarts the ticker. All messages then carry `source: "replay"`.

| Body field | Type | Default | Notes |
|---|---|---|---|
| `speed` | float | `1.0` | Ticks run `speed` times faster than real time. Must be > 0 (not validated yet). `20` is used by `scripts/e2e_check.py`. |
| `loop` | bool | `false` | Restart at the end. Without looping, the runtime pauses when the scenario ends; use another replay or `POST /api/live` to continue. |

```json
{"speed": 1.0, "loop": false}
```

```json
{"ok": true, "scenario": "frustrated_masked", "speed": 1.0}
```

Errors: `404 {"detail": "unknown scenario <name>"}`. Audit: `replay` with `scenario` and `speed`, and `reset`.

#### 7.4.5 `POST /api/sim/face_occlusion` and `POST /api/sim/low_light`

Turn a simulation overlay on or off. It applies to every subsequent tick in any mode.

| Body field | Type | Default |
|---|---|---|
| `enabled` | bool | `true` (send it explicitly) |

```json
{"enabled": true}
```

```json
{"ok": true, "overlays": {"face_occlusion": true, "low_light": false}}
```

Effect and tags: see 6.2. With `face_occlusion`, `face_only` becomes null, the label is expected to hold, and
the template uses its occlusion clause. Errors: an unknown overlay gives `404`. Audit: `sim` with `overlay`
and `enabled`. **SIMULATED.**

#### 7.4.6 `POST /api/sim/event`

Inject audio or gesture events into the next live tick (`aggregator.inject`). This is the fallback when
YAMNet or Vosk is unavailable. Injected counts bypass the near-field gate.

| Body field | Type | Default | Notes |
|---|---|---|---|
| `laughter` | int | `0` | Room laughter onsets to add |
| `compliments` | int | `0` | Room compliment hits to add |
| `fists` | object `{"<person_id>": int}` | `{}` | Fist onsets to add per person |

```json
{"laughter": 2, "compliments": 1, "fists": {"1": 0}}
```

```json
{"ok": true}
```

This affects the live aggregator only; scenario replay is not changed. Audit: `sim_event` with `laughter`
and `compliments`. **SIMULATED** (tags in 6.2).

#### 7.4.7 `POST /api/sim/lux`

Override the light level for subsequent live ticks. This is the fallback if the webcam lux estimate is
dropped.

| Body field | Type | Notes |
|---|---|---|
| `lux` | float \| null | New lux value. `null` clears the override and returns to sensed lux. |

```json
{"lux": 80.0}
```

```json
{"ok": true, "lux_override": 80.0}
```

Audit: `sim_lux` with `lux`. **SIMULATED** (tag `light_lux`).

#### 7.4.8 `POST /api/demo/seed`

Fill the room slot with history from a scenario. It adds context only (noise, light, laughter, compliments,
time), with no face and no person channels. Use it only as a last resort when the window is empty at demo
time; the venue plan is to start live at least 20 minutes early. Call `POST /api/control/stop` or
`POST /api/live` first so the window is empty, because seeding does not check for existing rows.

| Body field | Type | Default | Notes |
|---|---|---|---|
| `scenario` | string | `"full_demo"` | Scenario to take room history from |
| `minutes` | float | `12` | Minutes of history to add, ending now. At most 15 are kept. |
| `hours_since_break` | float \| null | `null` | Operator-set break clock for the people tracked right now |

```json
{"scenario": "frustrated_masked", "minutes": 12, "hours_since_break": 4.2}
```

```json
{"ok": true, "room_ticks_seeded": 144, "simulated_fields": ["room_history", "hours_since_break"]}
```

Seeded rows have `source = "sim_seed"` and the tag `room_history`. Errors: an unknown scenario currently
gives `500`. Audit: `demo_seed` with `scenario`, `minutes`, `hours_since_break` and `simulated_fields`.
**SIMULATED.**

#### 7.4.9 `POST /api/demo/reset`

Forget every person (windows, break clocks, smoothing state, latest predictions) and **keep the room
history**. Use it between judge groups. No body. Overlays and mode are unchanged.

```json
{"ok": true}
```

Audit: `reset` with `keep_room: true`.

#### 7.4.10 `POST /api/mode`

Turn room-only mode on or off. Room-only mode is the compliance-oriented configuration: one aggregate room
label, no per-person inference. Each tick then produces one `state_vector` and one `prediction` with
`person_id: 0`. The fact sheet uses `person: "Room"`, and `event_id`s look like `p0-000349`. `tracks`
continue (4.4). `hello` is not re-sent, so use this response.

| Body field | Type | Default |
|---|---|---|
| `room_only` | bool | `false` |

```json
{"room_only": true}
```

```json
{"ok": true, "room_only": true}
```

Audit: `mode` with `room_only`.

#### 7.4.11 `POST /api/calibrate/room`

Venue calibration (about 10 minutes at the venue before judging). The operator reads a phone SPL meter and
lux meter at the desk and posts the readings. The backend pairs them with its current measurements
(`aggregator.calibrate`).

| Body field | Type | Notes |
|---|---|---|
| `spl_reference` | float \| null | Phone SPL reading in dB taken now. Adds a dB correction (reference minus the Leq of the last 10 s of audio) on top of the levels the edge reports (the edge's own offset defaults to `settings.db_cal_offset` = 94). Ignored for the correction if no audio has arrived. |
| `lux_reference` | float \| null | Phone lux reading taken now. Multiplies the lux scale by reference ÷ current lux estimate (a one-point scale per call; the plan's 2-point fit is not implemented). Ignored if there is no lux estimate. |

```json
{"spl_reference": 52.0, "lux_reference": 320.0}
```

Response: `{"ok": true, "db_correction": <float>, "lux_scale": <float>}`. `db_correction` is returned whenever
`spl_reference` is sent (unchanged if no audio has arrived yet); `lux_scale` only when the lux reference was
applied. Numeric values are rounded to 3 decimals in the audit entry. Noise-floor measurement,
listed in the plan, is not implemented.

Calibration applies to new observations only. The window then mixes old and new calibration for up to
15 min, so calibrate first and call `POST /api/control/stop` then `start` before the real run. Audit:
`calibrate_room` with the applied values. Measured values, not simulated.

#### 7.4.12 `POST /api/label`

Self-report label (P1), used to compare live predictions against what the person reports. **It never
changes a prediction or any model input.**

| Body field | Type | Default | Notes |
|---|---|---|---|
| `state` | string | required | One of the six state ids in `STATE_DISPLAY` |
| `person_id` | int | `1` | |

```json
{"state": "Focused", "person_id": 1}
```

```json
{"ok": true}
```

The label, the time and that person's latest numeric `TickRecord` (no media) are appended to
`var/labels/<session_id>.jsonl` as evaluation data. Errors: an invalid `state` gives
`422 {"detail": "state must be one of [...]"}`, and a missing body gives `422`. Audit: `self_report` with
`label` and `person_id`.

#### 7.4.13 `GET /api/recordings` and `POST /api/replay_recording/{name}`

The camera/mic fallback: replay a features-only JSONL recording (`data/real/<name>.jsonl`, written by
`python -m valence.edge.run --jsonl`) through the live aggregator, with its original timing scaled by `speed`.
It exercises the same path as live data, so `tracks` are sent from the recorded boxes.

`GET /api/recordings` returns a bare array, for example `[{"name": "golden_demo", "bytes": 482113}]`.

`POST /api/replay_recording/{name}` releases live sensors, clears all windows, starts the feeder and restarts
the ticker. All messages then carry `source: "replay"`.

| Body field | Type | Default | Notes |
|---|---|---|---|
| `speed` | float | `1.0` | Playback speed |
| `loop` | bool | `true` | Restart at the end of the file |

```json
{"ok": true, "recording": "golden_demo"}
```

Errors: `404 {"detail": "unknown recording <name>"}` (paths are confined to `data/real/`). Audit: `reset`, then
`replay_recording` with the file name.

---

## 8. Audit log: operator and SIMULATED actions

The audit log (`valence/governance/audit.py`) records **decisions, never data**. It is one JSONL file per
session at `var/audit/<session_id>.jsonl`, deleted when the session closes, and any file older than
`settings.audit_ttl_s` (86 400 s) is removed at startup. Each entry is hash-chained:

| Field | Meaning |
|---|---|
| `seq` | 0-based sequence number |
| `ts` | Unix time, 3 decimals |
| `kind` | `prediction` or `operator` (purge-counter entries are planned) |
| `payload` | JSON object. Strings over 200 characters, lists over 64 items, nesting deeper than 4, NaN and infinity are refused, so free text such as a transcript can never be logged. |
| `prev` | `hash` of the previous entry (64 zeros for the first) |
| `hash` | SHA-256 of `prev` + the canonical JSON of `{seq, ts, kind, payload}` |

`prediction` entries are written per person when `label_display` changes, and otherwise at most every 30 s.
The payload holds `person`, `label`, `confidence`, `mask_override`, `source`, `layers` (layer name →
contribution) and `simulated_fields` (up to 8 entries).

`operator` entries have `payload.action`:

| `action` | Written by | Injects SIMULATED data |
|---|---|---|
| `control_start`, `control_pause`, `control_stop` | `POST /api/control/*` | no |
| `live` | `POST /api/live` | no |
| `replay` | `POST /api/replay/{scenario}`, and the startup scenario in replay mode | scenario data (`source = "replay"`) |
| `replay_recording` | `POST /api/replay_recording/{name}` | recorded data (`source = "replay"`) |
| `reset` | `demo/reset`, `replay`, `replay_recording`, `live`, `control/stop` | no |
| `break` | `POST /api/events/break` | no |
| `sim` | `POST /api/sim/face_occlusion`, `POST /api/sim/low_light` | **yes** |
| `sim_event` | `POST /api/sim/event` | **yes** |
| `sim_lux` | `POST /api/sim/lux` | **yes** |
| `demo_seed` | `POST /api/demo/seed` | **yes** |
| `calibrate_room` | `POST /api/calibrate/room` | no |
| `mode` | `POST /api/mode` | no |
| `self_report` | `POST /api/label` | no |

The SIMULATED actions (`sim`, `sim_event`, `sim_lux`, `demo_seed`) are the inputs the plan requires to be
"logged SIMULATED". The data they inject is also tagged in `simulated_fields` (section 6), so the console
shows it live and the audit chain proves it afterwards.

---

## 9. Internal narrator API (`127.0.0.1:8081`)

The narrator is a separate process in its own venv (`valence/narrator/.venv`), owned by the LOCAL session
(`valence/narrator/service.py`, a Stage A draft committed "under review"; run with
`python -m valence.narrator.service [--port 8081] [--template-only]`). Only the backend's `NarratorClient`
calls it. Browsers and other clients never do. This section is the contract; where the draft service differs,
the row says so.

### 9.1 `POST /narrate`

Request: a `FactSheet` as JSON (`Content-Type: application/json`). Response: `200` with a
`NarrationResult`.

What the narrator does with it: compact the sheet to `key: display` lines (about 120 tokens), generate with
Qwen3-1.7B-4bit (MLX, `enable_thinking=False`, prompt cache, `max_tokens` 90; the draft uses 110), validate, retry, and fall back
to its template. The fallback chain is Qwen3-1.7B, then Llama-3.2-3B-Instruct-4bit (if Qwen fails), then the
template.

| Behavior | Rule |
|---|---|
| Deadline | Answer within `settings.narrator_deadline_s` (2.5 s) including retries. The backend's HTTP timeout is 3.0 s. |
| Validation failure | Return `200` with the template text and `source: "template"`. Never return unvalidated text. |
| Invalid FactSheet | `422`. `contracts.FactSheet` uses `extra="forbid"`; the draft service validates with its own permissive schema (`valence/narrator/schema.py`, `extra="allow"`), so unknown fields pass there (section 11). |
| Model not loaded or warming up | Contract: `503`. The draft service instead answers with template text (`source: "template"`) if the model failed to load. Measured Qwen3-1.7B load is 1.2 s plus a 4.0 s warm-up at service start (`docs/BENCHMARKS.md`). |
| Any non-200, timeout or unparseable body | The backend counts a failure, sends **no** `explanation`, keeps the template text, and reports `status.narrator = "fallback"`. |
| Caching (backend side) | LLM results (not template results) are cached, keyed on person, label, `label_display`, confidence band, `confidence_display`, `mask_override`, `face_visible` and every driver `display` string. A cached text is reused only when every number it could mention is identical. The narrator also re-validates any text it caches itself against the current sheet. |

Validator rules the narrator must enforce (summary; details in `docs/NARRATOR.md`, LOCAL). The draft
`valence/narrator/validator.py` checks numbers, the state name, drivers, mask language, banned terms and length;
the qualifier, room-scope, "Upbeat" and cached-text checks are not implemented yet (`docs/DELIVERABLES.md` A8):

- Numbers in the text must be a subset of the numbers on the sheet.
- The text names the predicted state, using `STATE_LEXICON` stems and checking Tense's stems before
  Frustrated_Masked's.
- It mentions at least 2 of the top-3 drivers and nothing outside the top 5.
- A qualifier within 6 tokens of a feature mention must match that driver's `level` and `effect`.
- Room-scope drivers are phrased as "the room".
- Mask language appears only when `mask_override` is true. "Upbeat", not "Collaborative", when `n_people` is 1.
- No pronouns, demographics or diagnoses. Refer to people as "Person #N".
- 20–90 words.

Measured on the demo Mac (Apple M3 Max, 36 GB) without YOLO running: Qwen3-1.7B 4-bit generates at 187
tok/s, 536 ms wall for a 55-token answer, 1.27 GB peak memory. The ship gate, p95 ≤ 2.5 s **with YOLO
running**, is TBD and will come from the LOCAL narrator A/B run.

#### `FactSheet` fields

| Field | Type | Meaning |
|---|---|---|
| `event_id` | string | Same as the `prediction`, echoed into the `explanation` |
| `person` | string | `"Person #N"`, or `"Room"` in room-only mode |
| `label` | string | Smoothed state id |
| `label_display` | string | Text from `display_label()` (`"Frustrated"`, `"Upbeat"`, ...) |
| `confidence` | float | Calibrated, unsmoothed `probabilities[label]` |
| `confidence_display` | string | Rounded percent, for example `"88%"`. The only form the text may quote. |
| `probabilities` | object | Every active state |
| `window_fill_min` | float | Minutes of window filled (1 decimal) |
| `n_people` | int | People present (at least 1) |
| `face_only` | `{label, confidence, face_valence}` | Face-only baseline. **All three are `null` when `face_visible` is false.** `face_valence` is rounded to 2 decimals. |
| `face_visible` | bool | A face was readable on the latest tick |
| `mask_override` | bool | Mask language allowed only when true |
| `layers` | `Layer[]` | As in 3.7, with `level`, `effect` and `scope` on every driver |
| `data_quality` | `{face_visible_frac, lux, ood_features}` | As in 3.7. `lux` may be null. |

The WebSocket `prediction.face_only` (`FaceOnlyBrief`) has only `label` and `confidence`. The FactSheet's
`FaceOnly` adds `face_valence`.

#### Example: hidden face, one person, core states

Note the null `face_only`, a room-scope driver that `opposes` at level `elevated`, and person-scope drivers
at levels `high` and `low`.

```json
{"event_id": "p1-000512", "person": "Person #1", "label": "Exhausted", "label_display": "Exhausted", "confidence": 0.81, "confidence_display": "81%", "probabilities": {"Focused": 0.09, "Frustrated_Masked": 0.07, "Exhausted": 0.81, "Collaborative_Happy": 0.03}, "window_fill_min": 15.0, "n_people": 1, "face_only": {"label": null, "confidence": null, "face_valence": null}, "face_visible": false, "mask_override": false, "layers": [{"name": "Time Dynamics", "contribution": 0.52, "drivers": [{"feature": "hours_since_break", "display": "5.6 hours since the last break", "value": 5.6, "unit": "h", "shap": 0.47, "level": "high", "effect": "supports", "scope": "person"}]}, {"name": "Somatic & Social", "contribution": 0.44, "drivers": [{"feature": "slump_mean_5m", "display": "slumped 71% of the last 5 minutes", "value": 0.71, "unit": "%", "shap": 0.29, "level": "high", "effect": "supports", "scope": "person"}, {"feature": "motion_mean_5m", "display": "nearly still (motion 0.03)", "value": 0.03, "unit": "sw/s", "shap": 0.18, "level": "low", "effect": "supports", "scope": "person"}]}, {"name": "Ambient Physics", "contribution": 0.12, "drivers": [{"feature": "db_mean_15m", "display": "average noise of 41 dB over 15 minutes", "value": 41.3, "unit": "dB", "shap": 0.21, "level": "low", "effect": "supports", "scope": "room"}, {"feature": "lux_mean_5m", "display": "normal light (380 lux)", "value": 380.0, "unit": "lux", "shap": -0.09, "level": "elevated", "effect": "opposes", "scope": "room"}]}, {"name": "Facial Surface", "contribution": 0.01, "drivers": [{"feature": "face_visible_frac_5m", "display": "face visible 10% of the last 5 minutes", "value": 0.1, "unit": "%", "shap": 0.01, "level": "low", "effect": "supports", "scope": "person"}]}], "data_quality": {"face_visible_frac": 0.1, "lux": 380.0, "ood_features": []}}
```

When the face is visible, `face_only` is filled, for example
`{"label": "Collaborative_Happy", "confidence": 0.71, "face_valence": 0.62}` (see the masked case in 3.7).

#### `NarrationResult`

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Validated narration (or the template fallback) |
| `source` | `"llm" \| "llm_retry" \| "template"` | As in 3.8 |
| `latency_ms` | int | Narrator-side time to produce the answer |
| `attempts` | int | Generations tried, default 1 |

```json
{"text": "Person #1 reads as Exhausted with 81% confidence. The face is not readable right now, so the call rests on context: 5.6 hours since the last break, slumped 71% of the last 5 minutes, and the room quiet at an average of 41 dB over 15 minutes.", "source": "llm", "latency_ms": 610, "attempts": 1}
```

### 9.2 `GET /health`

`200` when the model is loaded and warmed up, so `/narrate` can be called. Any other status, or no answer
within 1 s, means not ready. The backend checks only the status code (on startup, about every 10 s, and on
`GET /api/narrator/health`). The body is informational and defined by the LOCAL session; a suggested shape is
`{"ok": true, "model": "Qwen3-1.7B-4bit", "adapter": null}`. The draft service returns `200` with the model,
warm-up state and per-source counts in the body, including while it is in template-only fallback, so a
status-only check cannot yet tell the two apart (section 11).

---

## 10. Versioning and JSON Schema export

### 10.1 Rules

- `CONTRACT_VERSION` in `valence/contracts.py` (currently `"1.0.0"`) is sent in `hello.contract_version`,
  `GET /api/health` and the OpenAPI `version`.
- **Additive changes only:** new optional fields with defaults, new message types, new endpoints. They bump
  the minor version. Renaming, removing or retyping a field, or changing a field's meaning, is breaking and
  would bump the major version. No breaking change is planned before judging.
- Contract changes go through `docs/CONTRACT_REQUESTS.md` (requests) and `docs/CONTRACT_RESPONSES.md`
  (decisions). The CLOUD session owns `contracts.py`.
- **Clients** must ignore unknown fields and unknown `type` values, and should warn when the major version
  differs from the one they were built for.
- **Strict validators:** the pydantic models use `extra="forbid"`, and the exported schemas say
  `"additionalProperties": false`. A consumer that validates strictly against an older schema will reject
  messages carrying a newly added field. The backend and narrator share this repo and must run the same
  `CONTRACT_VERSION`. The frontend should regenerate its types from the schemas or not validate strictly.

### 10.2 Export the JSON Schemas

`json_schemas()` returns a JSON Schema for every WebSocket message (`Hello`, `Tracks`, `Governance`,
`StateVector`, `Prediction`, `Explanation`, `Status`, `Heartbeat`) and for the narrator contract
(`FactSheet`, `NarrationResult`). Run from the repo root:

```bash
python -c "from valence.contracts import json_schemas; import json; print(json.dumps(json_schemas(), indent=2))" > valence_schemas.json
```

The schemas can be fed to a JSON-Schema-to-TypeScript generator for the frontend. Two schema details matter
for generated types:

- `bbox` is a 4-tuple (`prefixItems`, `minItems` and `maxItems` 4).
- `required` lists only fields without defaults, but the server always sends every field (2.3). Treat
  defaulted fields such as `type`, `source`, `simulated_fields` and `room_only` as always present.

`tests/test_contracts.py` covers the contract module, including that the schemas export and that the
frontend's `prediction` example validates. `tests/test_api.py` exercises the REST endpoints and the WebSocket
stream. Strict validation of every live message against these schemas is planned (`test_contracts`,
`test_e2e_replay`).

---

## 11. Known gaps and open items

| Item | Current state | Intended |
|---|---|---|
| WebSocket `Origin` check | CORS does not cover WebSockets, and `/ws/events` accepts any origin, so any page open in the same browser could read the stream from loopback. | Reject handshakes whose `Origin` header is present and does not match the CORS regex (2.2). Non-browser clients send no `Origin` and stay allowed. Recommended hardening; not in the plan. |
| `status.narrator = "down"` | Never emitted; an unreachable narrator shows `fallback`. | Report `down` after repeated failed health checks. |
| Counterfactual | Implemented for the served M2 (`501` with the stub). Uses the person's current window, not the stored feature vector of `event_id`. An unknown feature gives `500`. | Re-run on the event's stored vector (held in RAM up to 15 min, per `docs/ARCHITECTURE.md` 9.7); `422` for an unknown feature. |
| Request validation | Bodies are untyped dicts. Bad values can give `500`, an unknown scenario in `demo/seed` gives `500`, and `replay` does not check `speed`. | Typed request models: `422` for bad input and `404` for an unknown scenario. |
| `demo/seed` on a non-empty window | Not checked. Seeded rows are appended after existing rows. | Refuse with `409` unless the room window is empty. |
| `calibrate/room` | dB correction and a one-point lux scale per call (7.4.11). Noise floor is not accepted. | SPL offset, 2-point lux, noise floor (plan). |
| `sim_event` audit payload | Fist counts are not in the `sim_event` audit payload (laughter and compliments are). | Include fists in the audit payload. |
| Self-report labels file | `var/labels/<session_id>.jsonl` is written but not listed by `GET /api/retention`, and not deleted at session close (it is evaluation data). | List it in `/api/retention`. |
| `hello` on settings change | Sent only on connect. | Optionally re-send it to all clients after `mode`, `live`, `replay` or `control` calls (additive). |
| `feature_version` | Carries the served model id. The feature order is stored as `feature_names` in `artifacts/model_meta.json`; the plan's `artifacts/feature_spec.json` does not exist. | Carry a feature-spec version once the spec file exists. |
| `mask_override` rule | The implemented rule (3.7) differs from the plan's (M0 ≥ 0.55, 3 consecutive ticks). | Pick one rule and align the plan, `docs/ARCHITECTURE.md` 9.4, `docs/PRD.md` 2.8 and the code. |
| Narrator draft vs this contract | `valence/narrator/schema.py` keeps its own copy of the state list (core 4 only) and display names instead of reading `valence/contracts.py`, accepts extra fields, never returns `503`, and `/health` is `200` even in template-only fallback. It uses `max_tokens` 110 (plan: 90) and up to 3 attempts. | Read states from `contracts.py` (6 states with P1), return `503` or a distinguishable health status when the model is not ready. LOCAL session. |
| Audit purge-counter entries | Not written yet. | Periodic `purge` entries with the counters (per `audit.py` docstring). |
