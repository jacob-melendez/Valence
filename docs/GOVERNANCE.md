# Valence governance and data lifecycle

| | |
|---|---|
| Scope | What Valence captures, derives, keeps and destroys, read from the code on branch `claude/happy-ride-7w3ng3` on 2026-10-18, after the narrator update (`c61bed2`), the governance fixes in §8.1, the modality-dropout retrain and the venue room baseline (`bbd46a4`; served `m2_lgbm@8a78efc7`, the previous `85874e97` kept in `artifacts/legacy/`) |
| Audience | TrojanHacks @ USC 2026 judges (AI Data Governance track) and the builder running the demo |
| Status words | **built** = in the code now · **not built** = planned in `docs/PLAN.md`, absent from the code · **not run** = no result exists yet |
| Results | This document reports no model result. Model results in `reports/metrics.json` are on **simulated data** generated under the assumptions in `docs/DATASET.md` ("they test the architecture, not real-world accuracy"). The only measured numbers below are hardware timings from `docs/BENCHMARKS.md` (Apple M3 Max) and test/scan output. |

**In one paragraph.** Camera frames and audio blocks exist only in RAM, inside the capture and perception
threads: a frame for one ~45 ms perception call, an audio block until its ~1 s analysis step. Each is
zero-filled in place right after feature extraction and tracked until Python frees it.
Only scalars, flags and normalized boxes leave perception. The rolling window is RAM-only (15 min). By
default the live pipeline writes one file, a hash-chained audit log of decisions, and deletes it at shutdown.
The limits of all this (native-library copies, immutable strings, no soak run yet) are in §3.5 and §8.

---

## 1. Principles

| Principle | What it means in Valence | Built mechanism |
|---|---|---|
| Data minimization (incl. storage limitation) | Perception reduces each frame or audio step to scalars, flags and normalized boxes. Raw buffers are zero-filled after use. Compliments come from a restricted grammar (8 phrases + `[unk]`), so no free transcript exists. No face embeddings, no identity. Window RAM-only for 15 min; audit file deleted at shutdown, 24 h TTL backstop. | `valence/contracts.py` (`PersonObs`/`VideoObs`/`AudioObs` have no media, keypoint or text field), `valence/governance/purge.py`, `COMPLIMENT_PHRASES` in `valence/perception/audio_features.py`, `valence/features/window.py`, `valence/governance/audit.py` |
| Purpose limitation | One purpose: consented self-monitoring in a research demo. Display ids are ephemeral; no re-identification or cross-session tracking. Self-report labels never enter the model. | `valence/perception/ids.py`, `POST /api/label` (writes a file, changes no prediction), non-goals in `docs/PRD.md` §1.5 |
| Transparency | Every prediction carries SHAP layer sums, the face-only shadow (`face_only`), `source` and `simulated_fields`. Purge counters every tick. Retention policy and the audit chain are readable at runtime. | `valence/runtime/engine.py` (`Governance`, `Prediction` messages), `GET /api/governance/*`, `GET /api/retention` |
| Human control | Pause releases camera and mic; stop also clears all windows; reset forgets every person; room-only mode; logged breaks. Every operator action is written to the audit chain. | `POST /api/control/{start,pause,stop}`, `/api/demo/reset`, `/api/mode`, `/api/events/break` in `valence/api/app.py` |

## 2. Data lifecycle

**Who can read, by default.** The API has no authentication. It binds to 127.0.0.1 because the Makefile runs
uvicorn without `--host` (uvicorn's default; `settings.api_host` exists but nothing reads it), so any process on
the machine can call it. Browsers are limited by CORS on REST (localhost, `*.lovable.app`,
`*.lovableproject.com`), but `/ws/events` does not check `Origin` (§8). Files get the process's default
permissions. "Perception thread only" below means no reference leaves that thread's code.

| # | Data item | Where it lives | How long | Who can read | How it is destroyed |
|---|---|---|---|---|---|
| 1 | Video frame (BGR). Capture `grab()`s every frame and decodes only at 10 Hz; ~2/3 of the 30 fps stream is never decoded (`docs/BENCHMARKS.md`) | 1-slot handoff (`valence/edge/capture.py`) → `VideoPerceiver.process` (`valence/perception/video.py`) | One `process()` call: 45 ms p50 (2 people, sample images); 45-57 ms/frame live (`docs/BENCHMARKS.md`) | Perception thread only | `purge(frame, "video")` zero-fill + weakref, then `del`. Frames displaced from the slot, the warm-up frame and the slot frame at `close()` are purged by capture |
| 2 | Derived frame copies: RGB conversion, grayscale (lux, 1 Hz), 256 px head crops, `mp.Image` | Locals in `process()` | Same call | Perception thread only | RGB, gray, crop: `purge_derived(..., "video")` (kind `video_derived`). `mp.Image` holds MediaPipe's own copy: `del` only, not zeroed |
| 3 | Audio block, 100 ms, 16 kHz mono float32 | PortAudio callback → copy into `MicCapture._q` (max 100 blocks = 10 s) | Until concatenated into a ~1 s step | Audio thread only | Each copy `purge(part, "audio")` after concatenation; queue drained and purged at `close()`. PortAudio's own callback buffer is not reachable |
| 4 | Derived audio: 1 s block, 48 kHz native block (only if 16 kHz is rejected), int16 PCM for Vosk; YAMNet gets a view of the block's last 0.975 s | Locals in `MicCapture.read` and `AudioPerceiver.process` (`valence/perception/audio.py`) | One ~1 s step | Audio thread only | `purge_derived(..., "audio")` (kind `audio_derived`); the block is purged once, after Vosk has read it. MediaPipe's `AudioData` copy of the YAMNet clip is not zeroed |
| 5 | Recognizer text (Vosk `Result()` JSON) | Returned by the recognizer in `AudioPerceiver.process`, parsed by `count_compliments` (`audio_features.py`) | One audio step | Audio thread only; never logged, returned or put in an exception | A Python `str` cannot be zeroed: dropped and counted with `note_transcript_purged()`. Only phrase counts leave |
| 6 | Pose keypoints (COCO-17, YOLO26n-pose) | Copied into `_Det` per call; `PostureTracker` keeps the previous sample's 17×2 coordinates and baseline samples per display id | `_Det`: one call. Tracker: while the display id lives (released 3 s after the track is lost, `docs/CONTRACT_REQUESTS.md`) | Perception thread only; not in `VideoObs`, `TickRecord`, WebSocket or disk | Garbage-collected when the id is released (`ids.pop_released()` → `people.pop`); not zeroed |
| 7 | Face landmarks, blendshapes → `face_valence` | Landmarks and scores in `_update_faces`; `FaceValenceCalibrator` keeps raw valence samples from the first 20 s of visible face (≤ 4096) for the person's neutral baseline (`face_valence.py`) | Landmarks: one call (`del res`). Baseline: while the display id lives | Perception thread; only one `face_valence` scalar per person crosses the boundary | `del`; calibrator dropped with the display id; not zeroed |
| 8 | Bounding boxes (normalized x1, y1, x2, y2) | `PersonObs` → `tracks` WS message, ≤ 10 Hz, only boxes ≥ 0.55 × the tallest in the frame (`settings.participant_rel_height`, `engine.py`); latest frame's boxes also in `Aggregator.latest_boxes` | Latest frame | Any WebSocket client | Overwritten each frame. Not in `TickRecord`, history or audit. Exception: features-only recordings (row 15) contain them |
| 9 | `TickRecord` (5 s, per participant + room slot): numeric fields, `session_id`, display id, `source`, `simulated_fields` | Built by `Aggregator.flush` from per-tick accumulators, which are reset every tick. Participants: present in ≥ 30% of the tick's frames with mean box height ≥ 0.55 × the tallest (`_participants`); other tracks get no record | Moves into the window (row 10) | Main process | Accumulators cleared by `_reset_tick()` |
| 10 | Rolling window | `RollingWindow`: 180 ticks per slot (15 min), RAM only | ≤ 15 min while a slot keeps receiving ticks; a slot with no tick for `person_drop_min` (15 min) is evicted with its smoother and latest state (`Runtime._evict`) | Main process; values exposed in `state_vector` messages and `/api/state` | `deque(maxlen=180)` eviction; `_evict`; `clear()` on stop, `POST /api/live`, replay; `reset_people()` on `/api/demo/reset`; process exit |
| 11 | History buffer (`state_vector` + `prediction` messages) | `Runtime.history`, `deque(maxlen=1440)` | ~60 min for 1 person, ~20 min for 3 (`docs/ARCHITECTURE.md` §11.2); `/api/history` serves ≤ 60 min | Any local client of `/api/history` | Eviction; cleared by reset, stop, `POST /api/live` and replay start (`reset_people`); process exit. Pause keeps it |
| 12 | Predictions, FactSheets, explanation text | `Runtime.latest`; narrator client: latest-only slot per person + a 256-entry cache of LLM texts, oldest evicted first (`valence/explain/narrator_client.py`); narrator process on 127.0.0.1:8081 with its own ≤ 256-entry text cache, reused only after re-validation against the current sheet (`valence/narrator/engine.py`) | Latest per person; caches until evicted | Local clients (`/api/state`, WebSocket) | Overwritten; `latest`, the client cache and pending requests cleared on reset and stop (`narrator_client.clear_cache`). The narrator process's cache lives until that process exits (§8 gap 6) |
| 13 | Audit log | `var/audit/<session_id>.jsonl` (gitignored) | Until shutdown; other files older than 86 400 s deleted at the next startup | Local user with file access; any client of `/api/governance/audit` | `Path.unlink` on shutdown (`Runtime.close` → `AuditLog.close(delete=True)`). Unlink is not a secure erase |
| 14 | Self-report labels | `var/labels/<session_id>.jsonl` (gitignored), only on `POST /api/label`: label, time, person id, that person's latest numeric `TickRecord` | Until a startup finds the file older than `audit_ttl_s` (86 400 s) | Local user with file access | `expire_labels` at API startup (`valence/api/app.py`), else manual deletion. Not deleted at session close; not listed by `/api/retention` (§8 gap 7) |
| 15 | Features-only JSONL recordings | `data/real/*.jsonl`, only from `python -m valence.edge.run --jsonl` or `scripts/spikes/live_capture.py`: every `VideoObs`/`AudioObs` (boxes, face valence, fists, slump, motion, dB, wall-clock time) | Indefinitely (builder's capture and acted set) | Anyone with the repo: `data/real/` is **not** gitignored, and capture run 1 (`capture_{baseline,covered,fists,typing}.jsonl`) is committed (`74ff5f1`), as are runs 2–3 (`capture_{keyboard_gaze,slump,audio}.jsonl`, recorded with the recorder dropping bystanders, `docs/BENCHMARKS.md`). Since the bystander scrub the files keep only person observations whose box is ≥ 0.55 × the tallest in that frame; tracks 2-4 still appear where they pass that rule (no face valence), and earlier commits contain all their rows (see gap 8) | Manual deletion (and history rewrite once pushed) |
| 16 | Calibration constants | `data/real/calibration.json` (SPL offset, lux model; written by `scripts/calibrate_*.py` through `valence/edge/calibration_store.py`, scalars only, atomic write). Venue calibration (`db_correction`, `lux_scale`) in aggregator RAM, values audited | File: indefinitely. Runtime: process lifetime | Local user; loaded by perception | Manual deletion. Room constants, not data about a person |
| 17 | Model artifacts | `artifacts/m2.txt`, `m0.txt`, `model_meta.json`, `clip_bounds.json` and the same four in `artifacts/all/` (6 states) and `artifacts/legacy/` (the previous served model, rollback) committed; other ladder models local only (`.gitignore`) | Indefinitely | Public with the repo | Trained only on simulator output generated in memory (`make train`: 300 sessions, seed 7), plus face-hidden or degraded copies of those simulated windows for the M2 family (modality dropout, `train_ladder.py`). The simulator's scales were fit to summary statistics of the builder's capture run 1 (`reports/capture_vs_sim.json`; runs 2–3 were compared against it, not used to refit); no recorded rows enter training |
| 18 | Application logs | stderr | Process lifetime | Terminal | Not written to a file by default. `RawMediaFilter` drops records carrying raw media on every logger and handler present at startup, on `logging.lastResort` and on a dedicated `valence` handler (§4) |
| 19 | Venue room baseline | `Aggregator.baseline` (`RoomBaseline`, `valence/runtime/aggregator.py`), RAM only: up to `room_baseline_s` (90 by default; ≤ 600 via `POST /api/room/baseline`) room dB levels and lux estimates, about one per second, reduced to the median dB, dB IQR and median lux. Room constants: no media, no per-person value. Copied into every `TickRecord` as `room_db_base`, `room_lux_base` | Until the next capture, `POST /api/room/baseline/reset`, a recording replay (which starts an automatic one) or process exit. Survives `control/stop`/`start` and `/api/demo/reset`; a captured one also survives `POST /api/live` | Main process; `GET /api/room/baseline`, `/api/health` (`room_baseline`) | Dropped with the object; never written to disk except as the audit entries `room_baseline` (seconds), `room_baseline_captured` (the medians, the IQR and the sample counts) and `room_baseline_reset` |

The frontend's camera preview uses the browser's own `getUserMedia` stream. It never reaches the backend
and the backend has no endpoint that accepts images or audio, but that stream is outside the purge gate.

## 3. The purge gate (`valence/governance/purge.py`)

### 3.1 `gate.purge(buf, kind)`

1. Checks `kind` is in `KINDS` = `video`, `audio`, `video_derived`, `audio_derived` (`ValueError` otherwise) and
   that `buf` is a numpy array (`TypeError` otherwise).
2. Walks `.base` to the ndarray that owns the memory. Purging a view (e.g. `frame[:, :, ::-1]`) zeroes the
   whole underlying buffer.
3. `owner.fill(0)` if writeable. A read-only buffer is not zeroed and is counted in `unzeroable[kind]`.
4. Registers `weakref.ref(owner, callback)` in a dict keyed by `id(owner)` and increments `purged[kind]`. The
   callback fires when the array is actually freed: it removes the entry (only if the entry still holds this
   ref, because ids are reused) and increments `freed[kind]`. The lock is an `RLock` because a callback can fire
   inside `purge()`.
5. The caller then `del`s its own reference. `purge` must be the buffer's last use.

`note_transcript_purged(n)` counts recognizer strings that were dropped (strings cannot be zeroed).

### 3.2 `gate.end_tick()`, called by the ticker thread once per 5 s tick (`engine.py`, `Runtime.step`)

- If any purged buffer is still alive, one `gc.collect()` runs (reference cycles get one chance).
- Each survivor counts as a violation: a `PURGE_VIOLATION` warning with counts by kind, then tracking is
  cleared so each leak is reported once.
- The tick's `governance` message carries `frames_decoded`, `frames_purged`, `audio_blocks_purged`,
  `transcripts_purged`, `violations`, `live_raw_buffers` (survivors this tick), `audit_hash` (16 hex),
  `retention_s` and `purge_gap` (§3.4). `status.purge_violation` stays true once any violation has occurred.

### 3.3 Call sites

| Buffer | Kind | Where |
|---|---|---|
| Decoded camera frame | `video` | End of `VideoPerceiver.process`; `CameraCapture` for displaced, warm-up and final slot frames |
| RGB copy, grayscale, head crop | `video_derived` | `valence/perception/video.py` |
| 100 ms mic block copy | `audio` | `MicCapture.read` / `close` |
| 1 s block, 48 kHz native block, int16 PCM | `audio_derived` | `valence/edge/capture.py`, `valence/perception/audio.py` |
| Calibration script frames and blocks | `video` / `audio` | `scripts/calibrate_lux.py`, `scripts/calibrate_spl.py` (`tests/perception/test_calibration.py`) |

Detector outputs: YOLO boxes and keypoints are copied (`.cpu().numpy()` then `.copy()`), the `Results` object is
deleted, and Ultralytics' predictor caches (`batch`, `results`, `dataset`, `plotted_img`) are set to `None`
before the frame is purged (`_track`, `_scrub_yolo`). `test_holding_detector_results_leaks_the_frame` shows a
held results object is caught as a violation.

### 3.4 `frames_decoded == frames_purged`

- Every decoded frame increments the capture's `frames_decoded` and is purged as `video` by exactly one owner
  (perception, or capture if displaced). Derived copies use `*_derived` kinds so they do not inflate
  `frames_purged`.
- `VideoObs.frames_decoded` carries the capture's count at the moment that frame was decoded; capture may
  decode and purge 1-2 displaced frames before the next observation, so `frames_purged` can lead by the
  in-flight slot. Documented tolerance: ≤ 2 (`docs/CONTRACT_REQUESTS.md` round 2).
- **Checked every tick.** `Runtime.step` computes `purge_gap = max(0, frames_decoded − frames_purged)` (decoded
  frames not yet purged) and publishes it in the `governance` message, `/api/governance/stats` and the contract
  (`Governance.purge_gap`, `valence/contracts.py`). A gap above 2 for 3 consecutive ticks adds `purge_gap` to
  `status.degraded` (`valence/api/app.py`). It is an alarm, not a stop: capture keeps running.
- **Replay.** When the tick source is not live (scenario or recording replay), the `governance` message and
  `/api/governance/stats` both report the current source's own `frames_purged` as `frames_decoded` (`engine.py`,
  `valence/api/app.py`), so the pair is equal by construction, `purge_gap` is 0 and proves nothing. The counts
  stamped in a recording are never shown. `/api/governance/stats` also reports `frames_purged_process`, the
  video frames purged by the whole API process. Read the pair only in live mode.
- **Capture restarts and source changes.** Both counters cover the current source only. Each edge start
  (pause/stop → start) opens a new capture whose counter restarts at 0; the aggregator sees the count drop and
  adds the previous capture's last count as an offset (`Aggregator._push_video`), so `frames_decoded` keeps
  counting across the restart, as `frames_purged` does. Every source change (`POST /api/live`, scenario or
  recording replay) re-aligns the pair (`Runtime._new_source_counters`): the aggregator's decoded count and
  offset go back to 0, `frames_purged` counts from the gate's purged-video total at that moment, and the run of
  ticks above the threshold restarts. `purge_gap` therefore stays within the in-flight tolerance across a
  capture restart or source change unless frames really go unpurged: a restart does not blind the check, and a
  recording replay leaves no false alarm behind (§8 gap 3,
  `tests/test_runtime.py::test_frame_counter_survives_capture_restart_and_source_change`). Frames the old capture
  decoded after its last observation are purged but not in the offset, so each restart can add up to the
  in-flight slot to `frames_purged`'s lead.

### 3.5 Honest limits

- `del` removes a name, not memory. Zeroing first means the purged ndarray's bytes are gone even though
  Python's and numpy's allocators may keep the freed pages and reuse them.
- Copies outside numpy's reach are not zeroed: GPU allocator caches (YOLO's input tensor on MPS),
  AVFoundation/OpenCV capture pixel buffers, MediaPipe's internal `mp.Image` and `AudioData` copies and model
  input tensors, PortAudio's callback buffer, Vosk/Kaldi and YAMNet (TFLite) internal buffers.
- Purging a view zeroes its owner, so a call site may purge only after every reader is done. An earlier
  `audio.py` purged the YAMNet clip (a view) and so zeroed the block before Vosk read it; fixed on 2026-10-17
  (block purged once, at the end), covered by `test_vosk_reads_the_real_block_before_it_is_purged`.
- Temporaries are freed without zeroing: `np.clip(block, -1, 1) * 32767.0` before `astype(int16)`
  (`audio.py`), `np.square` in `mean_power`, `resample_poly` internals. On mic-queue overflow the copied block
  is dropped unpurged (`capture.py`, `overflows += 1`).
- Immutable `str`/`bytes` cannot be zeroed: Vosk result JSON and token strings are dropped and counted, not erased.
- The gate only knows buffers handed to it. A copy that never reaches `purge()` is invisible to every counter.
- The OS can page a buffer to swap before it is zeroed.
- `end_tick()` runs on the ticker thread while perception threads purge; a buffer purged an instant before its
  caller's `del` would count as a spurious violation (`docs/ARCHITECTURE.md` §16 item 7). Not measured: no soak.

## 4. Enforcement

| Control | What the code does | Limits |
|---|---|---|
| Boundary contract | `PersonObs`, `VideoObs`, `AudioObs` hold ints, floats, bools and one normalized 4-tuple box (validated in [0, 1]) (`tests/test_contracts.py`) | Recordings (row 15) still persist these fields |
| Persistence AST scan (`valence/governance/persistence_scan.py`, `tests/test_no_persistence_ast.py`, `make scan`) | `valence/perception/`: no file writes (also no `write_text`). `valence/edge/`: text-mode writes only. Banned: `imwrite`, `VideoWriter(_fourcc)`, `save`, `savez(_compressed)`, `savetxt`, `tofile`, `dump`, `to_pickle`, `wave.open`, `soundfile`/`sf.write`, `wavfile.write`, `write_bytes`, and `open`/`.open` with `w`, `a`, `x`, `+` or a non-literal mode. Banned attribute names are caught on any module alias (`import cv2 as c; c.imwrite`). Also caught: the banned names above and `wave.open` imported from a media module under any alias (`from cv2 import imwrite as w`, `from numpy import save`), `shutil.copy/copy2/copyfile/copyfileobj/move` in both directories, and `os.write/os.writev/os.fdopen/os.open` in `perception/`. Exception pragma `# persistence-ok: <reason>` on the call's line; a pragma without a reason does not count; `--exceptions` lists them. Exit 1 on any violation. Current output: `0 violation(s), 0 documented exception(s)` | Scans only those two directories (not `api/`, `runtime/`, `scripts/`). Still misses (probed): `import shutil as sh` and `from shutil import copy`, `from os import write` and `import os as o`, `os.write`/`os.fdopen` in `edge/`, `from soundfile import write`, `from scipy.io.wavfile import write` and `write` on an aliased `soundfile`/`wavfile` import, and `Path.open("wb")` with the mode as the only positional argument (§8 gap 12). A lint against accidents, not a proof |
| Raw-media log filter (`valence/governance/logfilter.py`, `tests/test_logfilter.py`) | Drops records whose message or args contain an ndarray, bytes, bytearray or memoryview, recursing into dict/list/tuple/set. `install()` at API startup attaches it to every existing logger and handler (root, uvicorn, ...) and to `logging.lastResort`, and gives the `valence` logger its own filtered stderr handler with propagation off, so `valence.*` records are filtered even when no root handler exists | Handlers added after startup are not filtered |
| Audit chain (`valence/governance/audit.py`, `tests/test_audit.py`) | One file per session. Entry `{seq, ts, kind, payload, prev, hash}`; `hash` = SHA-256 of `prev` + canonical JSON of `{seq, ts, kind, payload}`; genesis = 64 zeros. Payload refused if a string > 200 chars, a list > 64 items, depth > 4, NaN/inf, non-string keys or non-JSON types, so no transcript or array can be logged. `prediction` entries per person on display-label change, else ≤ every 30 s (person, label, confidence, mask_override, source, layer sums, ≤ 8 simulated_fields); `operator` entries per §5. Deleted at shutdown; startup deletes other `*.jsonl` older than `audit_ttl_s` (86 400 s). `GET /api/governance/verify` recomputes the chain: `{ok, entries}`, where on failure `entries` is the 1-based position of the first bad entry | Tamper-evident, not tamper-proof: someone with file access can rewrite a consistent chain or cut the tail; only the live `audit_hash` anchors it. A refused payload is logged and skipped (`Runtime._audit`). TTL runs only at startup. Purge-counter entries and `scripts/audit_verify.py`: not built |
| Bounded queues | Camera: 1-slot handoff, displaced frames purged. Mic: 100-block queue, overflow counted. EventBus: 200 messages per WebSocket client, drop-oldest, nothing queued with no clients (`valence/runtime/bus.py`). Window 180 ticks per slot; history 1 440 messages; narrator latest-only per person + 256-entry cache; room dB history 15 min; venue baseline `room_baseline_s` samples per channel | Bounds are on count, not on what is kept (rows 10-12) |
| No outbound network | `.env.example` sets `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; `valence/edge/run.py` sets the same with `setdefault` at import (live mode). API on 127.0.0.1:8000, narrator on 127.0.0.1:8081 (`narrator/service.py` `HOST`). `status.offline` is true when `HF_HUB_OFFLINE` and `YOLO_OFFLINE` are set. Measured: the perception stack under load held no inet sockets (`lsof -i`, `docs/BENCHMARKS.md` "Offline check") | `status.offline` is a configuration check, not a network probe; the API process itself has not been `lsof`-checked. Nothing loads `.env` automatically: export it for replay mode. CORS allows `*.lovable.app` and `*.lovableproject.com`; for a Wi-Fi-off demo the frontend must run locally |
| Pause / kill switch | `POST /api/control/pause`: ticker skips ticks and the edge `stop()` runs (threads joined, `cap.release()`, slot frame purged, mic stream closed and queue purged). Windows stay in RAM. `stop` does the same and also clears every window, history, `latest` and the backend's narration cache (`reset_people`). `start` resumes and reopens sensors in live mode. Audited as `control_*` | Pause keeps history, `latest` and narration cache. The narrator process's text cache survives both. In recording replay, pause does not stop the JSONL feeder |

## 5. Provenance and operator actions

Every `TickRecord`, `state_vector` and `prediction` carries `source` (`live`, `replay`, `sim`, `sim_seed`) and
`simulated_fields`. On a prediction, `simulated_fields` is the union of the person's latest tick and every room
tick in the window, so a tag stays visible until the simulated data has left the 15-min window (`docs/API.md` §6).

| Endpoint | Audit action (`operator`) | SIMULATED marking |
|---|---|---|
| `POST /api/sim/face_occlusion` | `sim` | `face_occlusion` (valence NaN, visible 0) |
| `POST /api/sim/low_light` | `sim` | `low_light` (lux × 0.15, or 30 if unknown; valence noise σ 0.35; `engine.py` `_overlay`) |
| `POST /api/sim/event` | `sim_event` (fist counts not in payload) | `laughter_count`, `compliments_count`, `clenched_fists_count` |
| `POST /api/sim/lux` | `sim_lux` | `light_lux` |
| `POST /api/demo/seed` | `demo_seed` | `room_history`, `source = sim_seed`; `hours_since_break` if given |
| `POST /api/replay/{scenario}`, `POST /api/replay_recording/{name}` | `reset`, then `replay` / `replay_recording` | `source = replay` |
| `control/{start,pause,stop}`, `live`, `demo/reset`, `events/break`, `calibrate/room`, `room/baseline`, `room/baseline/reset`, `mode`, `label` | `control_*`, `live`, `reset`, `break`, `calibrate_room`, `room_baseline` (then `room_baseline_captured` when the capture completes), `room_baseline_reset`, `mode`, `self_report` | None: real events, measurements or settings; audited only |

The four "logged SIMULATED" actions the plan requires are `sim`, `sim_event`, `sim_lux` and `demo_seed`
(`docs/API.md` §8). Startup in replay mode also writes `reset` and `replay`; `POST /api/live` writes `live`
and `reset`; `control/stop` writes `reset` before `control_stop`.

## 6. How to verify each claim yourself

Start the API first (`make live` on the Mac, or `make replay`); it listens on 127.0.0.1:8000.

| Claim | Command or endpoint | Expected |
|---|---|---|
| Gate zeroes, tracks views, catches leaks and held detector results; scan, filter and chain work | `.venv/bin/python -m pytest -v tests/test_purge.py tests/test_no_persistence_ast.py tests/test_logfilter.py tests/test_audit.py` | 22 passed (cloud container, 2026-10-18) |
| Real perceivers zero every buffer, keep no reference, and `--obscure-face` hides every face (Mac, models downloaded) | `.venv/bin/python -m pytest -v tests/perception/test_perceiver_privacy.py` | All 5 pass. Not run here: in the cloud all 5 skip (no models) |
| No persistence in raw-media code | `.venv/bin/python -m valence.governance.persistence_scan --exceptions` | `0 violation(s), 0 documented exception(s)`, exit 0 |
| Counters | `curl -s 127.0.0.1:8000/api/governance/stats` | `violations` 0; `live_raw_buffers` 0 (a momentary in-flight buffer is possible); `unzeroable` `{}`; live, for the current source (§3.4; each capture restart can add up to 2 to the lead): `frames_purged − frames_decoded` in 0..2 and `purge_gap` ≤ 2, no `purge_gap` in `status.degraded`; `make replay`: both 0 |
| Chain integrity | `curl -s 127.0.0.1:8000/api/governance/verify`; then change one digit of a payload value in `var/audit/<session>.jsonl` (keep the line valid JSON: a broken line makes the endpoint fail with HTTP 500) and call it again | `{"ok": true, ...}`, then `"ok": false` with the bad position |
| Retention policy | `curl -s 127.0.0.1:8000/api/retention` | `raw_media_stored: 0`, held items, never-kept list |
| Audit deleted at shutdown | Ctrl-C the server, then `ls var/audit/` | That session's file is gone |
| Kill switch stops capture | `curl -s -X POST 127.0.0.1:8000/api/control/pause` | `{"ok": true, "paused": true}`; macOS camera indicator off; `frames_decoded` stops growing; `status.camera` false after 3 s |
| Only expected files written | `touch /tmp/valence.mark` before starting; after shutdown: `find . -newer /tmp/valence.mark -type f -not -path './.venv/*' -not -path '*/__pycache__/*'` | Nothing, or `var/labels/*` if `/api/label` was used (audit file already deleted) |
| No media files or outside sockets | `lsof -nP -p $(pgrep -f 'app:app --port 8000')` | No media files; network entries only loopback: `127.0.0.1:8000 (LISTEN)`, its local WebSocket/REST clients, and connections to `:8081`. The audit file is opened per append, so it rarely appears |
| Works with no network | Turn Wi-Fi off, start the narrator and `make live`, watch the console | Predictions keep flowing; `status.offline` true. **Not run yet** (`docs/DELIVERABLES.md`) |
| 60-min soak: 0 violations, flat memory | `scripts/mem_soak.py` | **Not built, not run** |

## 7. Regulatory framing

- **EU AI Act Art. 5(1)(f)** prohibits placing on the market, putting into service for this purpose, or using AI
  systems that infer the emotions of a natural person in the workplace or in education institutions, except for
  medical or safety reasons. There is no consent exception. Valence infers emotions from facial blendshapes and
  body keypoints (biometric data), so it is an emotion recognition system in the Act's sense (Art. 3(39)).
  Recital 18 excludes physical states such as fatigue, but frustration and happiness are emotions, so the
  system as a whole is in scope.
- Outside those settings, emotion recognition is high-risk (Annex III point 1(c)) and deployers must inform the
  people exposed (Art. 50(3)).
- **Position:** a research prototype for consented self-monitoring and personal wellbeing outside workplace and
  education settings. Not for employers, schools, proctoring or productivity scoring. The Act's territorial
  scope (Art. 2) does not reach a demo at a US event; it is used as the design bar (`docs/PRD.md` §1.4). A
  hackathon on a university campus is close to the education setting, so the demo analyzes only the builder and
  volunteers who opt in at the table. That opt-in is a data-protection courtesy, not an Art. 5(1)(f) exemption:
  if the prohibition applied, consent would not lift it.
- **Room-only configuration:** `VALENCE_ROOM_ONLY=1` or `POST /api/mode {"room_only": true}` (a literal
  `--room-only` CLI flag is not built). Each tick yields one aggregate prediction (`person_id` 0, "Room") and no
  per-person predictions (`tests/test_runtime.py::test_overlays_and_room_only`). Perception still runs per
  person (face valence, keypoints) and `tracks` still carries per-person boxes, so it still processes biometric
  data. It reduces exposure; it does not make workplace or classroom use lawful. That would need legal review.
- **GDPR:** ephemeral ids make the behavioral timeline pseudonymous, not anonymous, and pseudonymous data is
  still personal data (Recital 26). Hence the RAM-only window, session deletion and TTL. Inferred states such as
  exhaustion or distress may count as health data (Art. 9); Valence treats them as sensitive. Valence does no
  biometric identification (no embeddings, no matching). Section 1 maps to Art. 5(1)(b) purpose limitation,
  (c) data minimisation and (e) storage limitation.
- This is design framing for a hackathon prototype, not legal advice.

## 8. Known gaps

Numbers are kept stable so other documents can cite them; fixed items moved to §8.1.

| # | Gap | Evidence / where | Status |
|---|---|---|---|
| 2 | No soak, memory-flat chart, API-process `lsof` check or Wi-Fi-off rehearsal yet (the perception stack alone under load held no inet sockets, `docs/BENCHMARKS.md` "Offline check"); `scripts/mem_soak.py` and `scripts/audit_verify.py` do not exist | `docs/DELIVERABLES.md`, `docs/BENCHMARKS.md` | Not built / not run |
| 3 | (Fixed after the review) The decoded/purged pair now covers the current source only: capture restarts add an offset (`Aggregator._push_video`), every source change re-aligns both counters (`Runtime._new_source_counters`), and replay reports this process's own count in both the `governance` message and `/api/governance/stats` (which also shows `frames_purged_process`) | `valence/runtime/aggregator.py`, `valence/runtime/engine.py`, `valence/api/app.py` | Fixed; `tests/test_runtime.py::test_frame_counter_survives_capture_restart_and_source_change` |
| 4 | `end_tick()` race can report spurious violations | `docs/ARCHITECTURE.md` §16 item 7 | Open |
| 5 | Live perception privacy tests (`tests/perception/test_perceiver_privacy.py`) need the downloaded models; in the cloud all 5 skip | Mac only | Run on the Mac before judging |
| 6 | The narrator process's own text cache (≤ 256 texts, RAM) survives reset and stop; it is reused only after re-validation against the current sheet and dies with that process (the backend side is fixed, §8.1) | `valence/narrator/engine.py` `_texts` | Open (minor) |
| 7 | Self-report labels are not deleted at session close and not listed by `/api/retention` (the TTL is fixed, §8.1) | `valence/api/app.py` `retention` | Open |
| 8 | Capture run 1 (`74ff5f1`) contained features-only rows (boxes, face valence) of background tracks 2-4 | `data/real/capture_*.jsonl` | **Mitigated in current files:** captures are solo, so each file now keeps only its dominant track (the builder); no other track remains. The earlier commits still contain the original rows; rewriting the shared branch's history was not done. The edge recorder should drop non-participants before writing (requested in CONTRACT_RESPONSES round 5) |
| 9 | Audit TTL enforced only at startup; tail truncation undetectable without the live `audit_hash`; `unlink` is not a secure erase; no purge-counter entries; `sim_event` payload omits fists | `valence/governance/audit.py`, `docs/API.md` §11 | Open |
| 10 | No authentication on any endpoint; `/ws/events` accepts any `Origin` | `docs/API.md` §11 | Open |
| 11 | Room-only mode still runs per-person perception and `tracks`; whether a dedicated room model serves the label is undecided | `docs/PRD.md` §4.7, `docs/ARCHITECTURE.md` §16 item 3 | Open |
| 12 | Remaining persistence-scan blind spots: directories other than `valence/perception/` and `valence/edge/` are not scanned; low-level `os` writes are allowed in `edge/` (text JSONL recorder); calls through dynamic attributes (`getattr`) or `exec` are invisible to static analysis. Module aliases (`import shutil as sh`, `import os as o`), `from` imports of `shutil`/`os`/`soundfile`/`scipy.io.wavfile` writers and `Path.open("wb")` with a positional mode are now caught | §4, `valence/governance/persistence_scan.py` | Open (by design for static analysis); covered cases tested in `tests/test_no_persistence_ast.py::test_scanner_resolves_module_aliases_and_path_open_modes` |
| 13 | Display-id release (3 s) vs `person_drop_min` (15 min): a reused id within 15 min can inherit an earlier person's slot and break clock | `docs/ARCHITECTURE.md` §16 item 2 | Open |
| 14 | Hybrid mode and per-modality `--video`/`--audio` sources | `docs/ARCHITECTURE.md` §12.1 | Not built |
| 15 | No demographic validation and no labeled in-the-wild data; models are trained and evaluated on simulated data only (the live captures, runs 1–3, are unlabeled and from one person) | `reports/metrics.json` `disclosure`, `docs/PRD.md` §1.3 | Future work |

### 8.1 Fixed

| # | Was | Fixed in the code | Check |
|---|---|---|---|
| 1 | Log filter missed `valence.*` records under uvicorn (no root handler at `install()`, so the last-resort handler printed them unfiltered) | `logfilter.install()` covers every existing logger and handler, `logging.lastResort` and a filtered `valence` handler with propagation off (§4) | `tests/test_logfilter.py::test_install_filters_valence_child_loggers_without_root_handlers` |
| 3 | `frames_decoded == frames_purged` was displayed, not checked | `purge_gap` in the `governance` message, `/api/governance/stats` and `Governance.purge_gap`; `purge_gap` in `status.degraded` after 3 ticks above 2 (`valence/runtime/engine.py`, `valence/api/app.py`, `valence/contracts.py`). In replay both report the current source's own count (§3.4). Holds across capture restarts and source changes (gap 3, fixed) | §6 "Counters" |
| 6 | Absent people's window slots were never evicted; history and the narration cache survived stop and reset | `Runtime._evict` drops the slot, smoother and latest state after `person_drop_min`; `reset_people` (reset, stop, `POST /api/live`, replay) clears history and calls `narrator_client.clear_cache` (`engine.py`) | `tests/test_runtime.py::test_absent_people_are_evicted_and_reset_clears_history` |
| 7 | Self-report labels had no TTL | `expire_labels` deletes label files older than `audit_ttl_s` at API startup (`valence/api/app.py`) | Startup only, like the audit TTL |
| 12 | Scan missed aliased media imports, `os.write`, `os.fdopen`, `shutil` copies | `persistence_scan.py`: banned names imported from a media module under any alias, `os.write/writev/fdopen/open` in `perception/`, `shutil.copy*`/`move` (§4); the rest stays in gap 12 | `tests/test_no_persistence_ast.py::test_scanner_catches_aliases_low_level_writes_and_copies` |
