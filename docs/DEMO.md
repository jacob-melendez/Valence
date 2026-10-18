# Valence demo runbook

For the solo builder, Sunday judging (TrojanHacks @ USC 2026, AI Data Governance track). Every model number
below is from `reports/metrics.json` and is measured on **simulated data** (45 held-out simulator sessions,
`noise_scale` 1.0, untuned), so it tests the architecture and says nothing about real-world accuracy. Say
"on simulated data" every time you quote one.

> `metrics.json` (generated 2026-10-18 00:04:46) evaluates the served build `m2_lgbm@e696c749`, trained on the
> simulator refit to live capture run 1 (`docs/MODEL_CARD.md`). If you retrain, re-run `make eval`, update every
> number below, and pitch only charts from the run that wrote `metrics.json`.

Ports: live API `127.0.0.1:8000`, narrator `127.0.0.1:8081`, and an optional second, replay-only API on
`127.0.0.1:8001`. Use the second API for replays: `POST /api/replay/{scenario}` on `:8000` releases the camera
and **clears every window**, and the live window then needs 15 minutes to refill.

```bash
A=http://127.0.0.1:8000; R=http://127.0.0.1:8001; J='content-type: application/json'   # used below
```

---

## 1. Night before (Mac, Wi-Fi on)

| # | Step | Command / check |
|---|---|---|
| 1 | arm64 venv + deps | `uv venv --python 3.12 .venv && source .venv/bin/activate`; `python -c "import platform; assert platform.machine() == 'arm64'"`; `uv pip install -r requirements-mac.txt && uv pip install -e '.[dev]'`; then `uv pip uninstall opencv-python opencv-python-headless && uv pip install --no-deps --reinstall opencv-contrib-python==4.14.0.94` |
| 2 | OpenMP + jq | `brew install libomp jq`. LightGBM does not import without libomp (`docs/BENCHMARKS.md`). You only need jq for the snippets in this runbook. |
| 3 | Models (one-time) | `bash scripts/download_models.sh`. Downloads YOLO26n-pose, the MediaPipe tasks, YAMNet, Vosk and Qwen3-1.7B-4bit into `models/`. If only the mirror downloaded, set `VALENCE_NARRATOR_MODEL=models/llm/Qwen3-1.7B-MLX-4bit` (the default path is `models/llm/Qwen3-1.7B-4bit`, `valence/narrator/engine.py`). |
| 4 | `.env` | `cp .env.example .env`. Keep `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `VALENCE_STATE_SET=core`. `core` must match `state_set` in `artifacts/model_meta.json`; otherwise the rule stub is served. Add `VALENCE_MIC="MacBook Pro Microphone"` **with the quotes** (the default input on the demo Mac is a Bluetooth headset; without the variable the edge already prefers a device named "MacBook … Microphone", `valence/edge/capture.py`, so this only makes it explicit). Do not set `OMP_NUM_THREADS`: `valence.edge.run` sets it to 1 before importing torch, and `OMP_NUM_THREADS=2` segfaulted (`docs/BENCHMARKS.md`). Nothing loads `.env` automatically, so run `set -a; source .env; set +a` in every terminal. |
| 5 | Camera/mic permission | macOS grants it to the app that launches Python. Use Terminal.app, iTerm or VS Code. Claude Code's shell can never get the camera (`docs/BENCHMARKS.md`). Then `python scripts/spikes/selftest.py` (fails loudly on black frames or silent zeros) and `python -m valence.edge.run --video --audio --print --duration 30`. |
| 6 | Tests | `make test` must report 0 failed. In the cloud container at `7722719` (2026-10-18) it reported 487 passed, 7 skipped; the skips need Mac-only packages or the downloaded models and should run on the Mac. `make scan` should print `0 violation(s), 0 documented exception(s)`. Also run `.venv/bin/python -m pytest -v tests/perception/test_perceiver_privacy.py`: all 5 must pass (they skip in the cloud). |
| 7 | Edge calibration (optional) | `python scripts/calibrate_spl.py` and `python scripts/calibrate_lux.py` write constants only to `data/real/calibration.json`. The venue step in §2 corrects on top of them. |
| 8 | Golden recording (camera fallback) | Play noise, smile and clench your fists for 10 min: `python -m valence.edge.run --jsonl data/real/golden_demo.jsonl --duration 600`. The file holds features only (boxes, scalars, wall-clock times) of everyone the tracker sees, not only you. `data/real/` is **not** gitignored (capture run 1 is already committed), so do not commit it by accident. |
| 9 | Frontend | `cd frontend && npm install` needs the network, so do it tonight. The app loads Google Fonts, which fall back to system fonts offline. The snapshot was built from the v1 prompt (`docs/DELIVERABLES.md`), so `scripts/console.py` is the reliable view. |
| 10 | Rehearse | Run §2–§4 once with Wi-Fi off and make a backup screen recording of a replay run (keep it outside git). |

## 2. At the venue, T-30 min

| When | Action |
|---|---|
| T-30 | Plug in. Frame the camera on your seat only. Volunteers opt in at the table (`docs/PRD.md` §1.4). Optional: `export VALENCE_MIN_BOX_HEIGHT=0.35` tracks only people near the camera (default 0.25, `valence/config.py`; the edge reads it). |
| T-29 | **T1 narrator:** `set -a; source .env; set +a; .venv/bin/python -m valence.narrator.service` (from `valence/narrator/.venv` if you built that venv; `.venv` also has mlx-lm via `requirements-mac.txt`). Wait for the `narrator model ready` log line (measured load 1.2 s + warm-up 4.0 s). |
| T-28 | **T2 live API:** `set -a; source .env; set +a; caffeinate -dims make live`. Check `curl -s $A/api/health`: `"mode": "live"`, `"model": "m2_lgbm@..."`, `"edge_error": null`. |
| T-27 | **T3 console:** `make console`. Governance lines should scroll every 5 s. |
| T-26 | **Venue recalibration** (table below), then `curl -s -X POST $A/api/control/stop` and `curl -s -X POST $A/api/control/start` so the window does not mix old and new calibration. |
| T-25 | Sit down with a **neutral face, upright, for 20 s**. Face valence and slump are measured against your own first 20 s (`docs/PERCEPTION.md`). Do not leave the frame for more than 3 s from now on: perception releases your display id and recomputes both baselines from your next 20 s on return. The window slot and break clock are keyed by display id, so they continue if you come back as the same Person #N (`docs/GOVERNANCE.md` §8 gap 13). |
| T-24 | **T4 replay API:** `VALENCE_MODE=replay VALENCE_SCENARIO=context_blindness VALENCE_SPEED=20 .venv/bin/python -m uvicorn valence.api.app:app --port 8001`. **T5:** `.venv/bin/python scripts/console.py --url ws://127.0.0.1:8001/ws/events`. |
| T-23 | **T6 frontend:** `cd frontend && npm run dev`. Set its backend URL to `http://127.0.0.1:8000`; `localhost` can resolve to `::1` and fail. |
| T-22 | **Wi-Fi off.** Check `curl -s $A/api/state \| jq .status.offline` returns `true`. That field only checks the env vars, so also check the radio icon and `lsof -nP -p $(pgrep -f 'app:app --port 8000')`: network entries should be loopback only (`127.0.0.1:8000`, its local clients, connections to `:8081`). |
| T-20…T-3 | The window fills (180 ticks × 5 s = 15 min). Run the §7 checklist. Pre-open the 4 pitch charts from `reports/figs/` in Preview, in pitch order: `accuracy_vs_history`, `confusion_m0_vs_m2`, `robustness_face_dropout`, `robustness_lux`. |
| T-3 | **Start the masked act:** noise from a phone or speaker at about 65–70 dB at the desk, a smile, and fist clenches (details in §3). |

**Venue recalibration: `POST /api/calibrate/room`.** The dB step uses the Leq of the last 10 s of audio
blocks, so wait at least 10 s after audio starts; with no audio yet `db_correction` stays unchanged. The lux
step needs a lux estimate, which the edge reports from its first valid background reading (a rolling 10 s
median, `valence/perception/lux.py`).

```bash
curl -s -X POST $A/api/calibrate/room -H "$J" -d '{"spl_reference": 62.0, "lux_reference": 320}'
# -> {"ok": true, "db_correction": <float>, "lux_scale": <float>}
```

| Field | What you read | What the backend does (`valence/runtime/aggregator.py`) | Not built |
|---|---|---|---|
| `spl_reference` | Phone SPL app held next to the laptop mic, averaged over ~10 s of steady broadband room noise (not a pure tone; the phone reads dBA, the mic measures unweighted power) | `db_correction += reference − Leq of the last 10 s`; added to every later audio level, on top of the edge offset (`spl_offset_db` in `data/real/calibration.json`, default 94) | Noise-floor measurement |
| `lux_reference` | Phone lux meter where you sit | `lux_scale *= reference ÷ current lux estimate` (one point per call); `lux_scale` is returned only if an estimate exists | 2-point lux fit |

- The calibration is held in RAM and survives `control/stop` and `control/start`. **A server restart loses it:
  recalibrate.**
- It is audited as `calibrate_room` and is a measurement, not SIMULATED.
- Training range is roughly 34–73 dB for `db_mean_5m` and 89–575 lux for `lux_mean_5m`
  (`artifacts/clip_bounds.json`). Values outside are clipped and listed in `data_quality.ood_features`.

## 3. The 4-minute pitch

**Masked act, started at T-3 and kept up until about 2:00.**

- **Noise:** about 65–70 dB at the desk. The simulator's Frustrated_Masked room target is 66 dB
  (`valence/sim/params.py`).
- **Smile:** this makes the face-only model read Collaborative_Happy.
- **Fists:** clench both fists for about 1 s, wrists above the bottom 20% of the frame (that strip is ignored
  as the keyboard zone), then relax. A fist counts at a score of 0.7 or more held for at least 0.5 s, with a
  3 s refractory (`docs/PERCEPTION.md`).
- **Pace:** one clench every 45–60 s. The trained clip range is 0–2 fists per minute and 0–7 per 5 minutes
  (`artifacts/clip_bounds.json`); faster clenching is clipped and listed as out of range.
- **Why start early:** context lags behavior. On simulated data M2 is right 37.6% of the time in the first
  30 s of an episode, 68.5% at 1–2 min and 86.3% at 2–5 min (`accuracy_by_episode_age`). The override is also
  blocked for the first 120 s of a person's window (`override_block_s`).

| Clock | Proof | Exact action | Point at / say |
|---|---|---|---|
| 0:00–0:20 | Thesis | none | "A face snapshot misses masked stress and misreads resting focus. 15 minutes of context fixes most of that on simulated data, and no frame or audio is kept." |
| 0:20–1:05 | **1 Data minimization** | Show the governance log in the console or frontend. Run `curl -s $A/api/governance/stats \| jq '{frames_decoded, frames_purged, violations, live_raw_buffers, unzeroable, retention_s}'`, then `curl -s $A/api/retention \| jq .raw_media_stored`, then `curl -s $A/api/governance/verify`. | `frames_purged − frames_decoded` is 0–2 (in-flight tolerance); `violations` 0; `live_raw_buffers` 0; `unzeroable` `{}`; `raw_media_stored` 0; chain `ok: true`; Wi-Fi off, `offline: true`. Frames are zero-filled, not just `del`'d. |
| 1:05–1:45 | **2 Context vs face (live)** | Console line for Person #1. | `Frustrated / Masked Stress … (face-only: Collaborative_Happy) MASK OVERRIDE`, with the template line "The face alone suggests Collaborative / Happy, but … point the other way. Context corrected the facial mask." |
| 1:45–2:20 | 2 | Charts `accuracy_vs_history.png`, then `confusion_m0_vs_m2.png` | On simulated data: macro-F1 at 15 min is M2 0.791 [0.776, 0.810] vs M0 0.425. Frustrated_Masked recall is 69.4% vs 11.7%. Focused misread as negative is 18.3% vs 55.9%. |
| 2:20–3:00 | **3 Missing/degraded face** | `curl -s -X POST $A/api/sim/face_occlusion -H "$J" -d '{"enabled": true}'`; show the console; then the same call with `false`. Charts `robustness_face_dropout.png`, `robustness_lux.png`. | The label should hold. The text switches to "The face is not readable right now, so the call rests on context: …" and the console shows `SIMULATED: face_occlusion`. On simulated data, with the face hidden on every tick M2 scores 0.662 vs M0 0.121; at 10% light, 0.615 vs 0.254. Say that both are below the no-face model (0.693): degraded face evidence costs M2 more than no face (MODEL_CARD §5). |
| 3:00–3:20 | 2 (counterfactual) | `EID=$(curl -s $A/api/state \| jq -r '.people["1"].prediction.event_id')`, then `curl -s "$A/api/explain/$EID/counterfactual?feature=hours_since_break&value=0.5" \| jq '{original, cf: .counterfactual.label, label_changed}'` | "Same window, break clock set to 30 min: does the label survive?" Read `label_changed` as it comes. Only that one feature changes; nothing is published. |
| 3:20–3:50 | 2 (context blindness) | `curl -s -X POST $R/api/replay/context_blindness -H "$J" -d '{"speed": 20}'` and watch T5 | The same head-in-hands pose in both halves. Laughter and compliments read as Collaborative / Happy. Silence, dim light and a long stretch without a break (4.4 h at the scenario's start, about 4.5 h by the second half) read as Exhausted. That segment starts at minute 9 of 18 (`scenarios/context_blindness.yaml`), about 27 s in at speed 20. The messages say `source: replay`. A cloud replay with `e696c749` at `7722719` read the first half as Collaborative / Happy throughout and the second as Exhausted, with 5 transition ticks (Upbeat, Frustrated) between. |
| 3:50–4:00 | Close | none | Simulated data only. EU AI Act Art. 5(1)(f), so not for workplaces or schools. Room-only mode exists. Demographic and in-the-wild validation are explicit gaps. |

**Live label caveat.** The served model passes `make negative` (quiet typing reads Focused), but only on the
capture the simulator was refit to (MODEL_CARD §5). A live MASK OVERRIDE shows the pipeline working on one person;
it is not evidence for proof 2. Say so, and let the `frustrated_masked` replay on `$R` (scripted, `source: replay`)
carry the evidence.

If the live flip has not appeared by 1:05, do not wait. Run
`curl -s -X POST $R/api/replay/frustrated_masked -H "$J" -d '{"speed": 20}'`. The override fires within 30 ticks
of the masked segment's start at minute 4 (`tests/test_served.py`), so by about 20 s after the replay starts at
speed 20. Run the counterfactual against `$R` afterwards.

## 4. Between judge groups

```bash
curl -s -X POST $A/api/sim/face_occlusion -H "$J" -d '{"enabled": false}'   # overlays survive a reset
curl -s -X POST $A/api/demo/reset                                           # forget all people, keep room history
```

- Reset **as soon as a group leaves**. It also restarts *your* person window and break clock:
  - `warming_up` lasts 60 s and the override is blocked for 120 s (`valence/config.py`).
  - In the simulator a person returns from a break as Focused or (with 2+ people) Collaborative_Happy
    (`valence/sim/generator.py`), so a fresh break clock pulls predictions away from Frustrated.
  - A counterfactual probe on the end of the `frustrated_masked` replay (simulated, `e696c749` at `7722719`)
    kept Frustrated_Masked at `hours_since_break` 0.1–1.0 and turned Collaborative_Happy at 0.0–0.05.
  - So expect the live flip only after about 6+ minutes back on the clock. Re-check after any retrain with the
    counterfactual endpoint on `$R`.
- Do not use `/api/demo/seed` to restore the clock: it appends room rows without checking for existing ones.
- A `SIMULATED: face_occlusion` tag stays on predictions until those room ticks leave the 15-min window. This is
  intended provenance, and you can explain it.
- The reset does not clear `/api/history` (1 440 messages) or the narration cache (`docs/GOVERNANCE.md` §8
  gap 6).

## 5. Fallback ladder

| Symptom (where you see it) | Action | Provenance judges will see |
|---|---|---|
| Camera fails: `degraded` has `camera`/`edge`, `edge_error` in `/api/health` | Was it launched from Terminal/iTerm? Run `python scripts/spikes/selftest.py`. Otherwise `curl -s $A/api/recordings`, then `curl -s -X POST $A/api/replay_recording/golden_demo -H "$J" -d '{"loop": true}'`. Last resort: `curl -s -X POST $A/api/replay/full_demo -H "$J" -d '{"loop": true}'`. Both clear the window. During any replay the frame counters prove nothing: `/api/governance/stats` shows the recording's stamped `frames_decoded`, and the `governance` message reports this process's own count (`docs/GOVERNANCE.md` §3.4). Skip proof 1's counter check then | `source: replay` |
| Mic fails: `degraded` has `mic` | Check `VALENCE_MIC`. Inject events with `curl -s -X POST $A/api/sim/event -H "$J" -d '{"laughter": 2, "compliments": 1}'` (or `"fists": {"1": 1}`). dB stays missing; M2 is NaN-native. | SIMULATED: `laughter_count`, `compliments_count`, `clenched_fists_count`; audit `sim_event` |
| Narrator down: `status.narrator = "fallback"` | Nothing to do: each `prediction` already carries template text and ticks continue. Restart T1, or use `--template-only`. | `explanation.source: template` |
| Model artifacts missing: `/api/models` says `stub-rules`, `degraded` has `predictor:stub`, counterfactual returns 501 | `git checkout artifacts/` (`m2.txt`, `m0.txt`, `model_meta.json` and `clip_bounds.json` are committed), check `VALENCE_STATE_SET=core`, restart T2, recalibrate. The stub keeps the demo running meanwhile. | `feature_version: stub-rules` |
| libomp segfault at start | Start only with `make live`: `VALENCE_MODE=live` makes `app.py` import `valence.edge.run`, which sets `OMP_NUM_THREADS=1` and imports torch before LightGBM; LightGBM predicts with `num_threads=1`. Do not set `OMP_NUM_THREADS` to anything else; `brew install libomp`. Last resort: `KMP_DUPLICATE_LIB_OK=TRUE` in `.env` (OK in `docs/BENCHMARKS.md`). | none |
| Window empty at demo time (late restart) | `curl -s -X POST $A/api/control/stop`, then `curl -s -X POST $A/api/control/start`, then `curl -s -X POST $A/api/demo/seed -H "$J" -d '{"scenario": "frustrated_masked", "minutes": 12}'`. Adds context only, no face. | `source: sim_seed`, SIMULATED `room_history`; audit `demo_seed` |
| Live flip does not come | Replay `frustrated_masked` on `$R` (§3) | `source: replay` |
| Frontend will not connect | Use `127.0.0.1`, not `localhost`; present from the console (T3/T5). | none |
| `purge_violation: true` | Say it plainly: `end_tick()` can race a perception thread and count a spurious violation (`docs/GOVERNANCE.md` §8 gap 4). Show `live_raw_buffers`. | `violations` > 0 |
| Port 8000 busy after a crash | `lsof -nP -iTCP:8000 -sTCP:LISTEN`, kill it, restart, recalibrate | none |

## 6. Q&A prep

1. **"It's all synthetic."** Yes. Every accuracy metric is simulated: 300 simulated sessions (seed 7), 45 of them
   held out for test. The behavioral assumptions are published in `docs/DATASET.md` and `docs/MODEL_CARD.md`. The
   claim is only "under published assumptions, temporal context beats a face snapshot". In-the-wild validation is
   future work. The live demo shows the pipeline running, not its accuracy. The one real-data check, 3 min of the
   builder's quiet typing, read mostly Frustrated before the simulator was refit to live capture and reads Focused
   on every tick now. The refit used that same capture, so it is a consistency check, not validation (MODEL_CARD §5).
2. **"You wrote the simulator, so context wins by construction."**
   - `noise_scale` was left at 1.0, untuned, so the headline was never tuned toward (`docs/PLAN.md` build-time
     decisions).
   - Distributions overlap: the room targets are 66 dB for Frustrated_Masked and 63 dB for Collaborative_Happy,
     and only 70% of Frustrated_Masked episodes mask (`valence/sim/params.py`).
   - M2 is *worse* than the face in the first 30 s of an episode (37.6% vs 44.0%).
   - The original mock was rejected because context alone scored 1.0 on it.
   - What remains, that states persist for minutes with context signatures, is an assumption only real data
     can test.
3. **"`del frame` isn't erasure."** Agreed. The PurgeGate zero-fills each buffer, tracks it by weakref and
   counts survivors as violations at tick end. Capture decodes only 10 of 30 frames/s. Not zeroed: GPU
   allocator caches, AVFoundation buffers, MediaPipe `mp.Image`/`AudioData` copies, PortAudio buffers and
   swap (`docs/GOVERNANCE.md` §3.5).
4. **"EU AI Act?"** Art. 5(1)(f) bans emotion recognition in workplaces and education institutions, except for
   medical or safety reasons; consent does not lift it. Valence is a research prototype for consented
   self-monitoring outside those settings, and the Act's territorial scope does not reach a US event (it is our
   design bar). Room-only mode (`POST /api/mode`) reduces exposure but does not make workplace use lawful, and
   per-person perception still runs in it. A campus hackathon is close to the education setting, so only the
   builder and opt-in volunteers are analyzed; the opt-in is a courtesy, not an Art. 5 exemption.
5. **"Demographic bias?"** Not validated, and we list that as an explicit gap. Proof 3 is narrowed to decision
   stability when the face is missing or degraded. Face valence is relative to each person's own neutral. The
   narrator may not mention demographics. Upstream fairness evaluations are in the MediaPipe model cards.
6. **"What is SIMULATED here?"**
   - Every metric.
   - Replays (`source: replay`) and seeded rows (`sim_seed`).
   - Anything from `/api/sim/*` or `/api/demo/seed`, tagged in `simulated_fields`, shown in magenta in the
     console and audited as `sim`, `sim_event`, `sim_lux` or `demo_seed` (`curl -s "$A/api/governance/audit?since=0"`).
   - Counterfactual outputs, tagged with the changed feature.
7. **"Why not just a better face model?"**
   - A forced smile *is* a smile. Our M0 is deliberately naive (`cheekSquint` excluded, `docs/PERCEPTION.md`)
     so the correction is measurable. A Duchenne-aware model would narrow the masking gap, and we say so.
   - Context carries most of the signal: M2-noface scores 0.694, M0 0.426, M2 0.805.
   - It keeps working without a face: 0.662 with the face hidden on every tick, though that is below the no-face
     model's 0.693; a fallback to it is not built (all simulated).
   - A better face model also means more face data, which is exactly what we minimize.
8. **"Latency?"**
   - Edge: 45–57 ms/frame at 10.0 Hz live; YOLO 20.7 ms p50 live (`docs/BENCHMARKS.md`).
   - Labels update per 5 s tick with EMA and 2-tick hysteresis, so they lag behavior by design.
     `status.loop_latency_ms` shows the tick cost.
   - Narrator: Qwen3-1.7B took 536 ms for 55 tokens without YOLO running. The p95 with YOLO is not measured,
     and the 2.5 s deadline falls back to the template.
9. **"What leaves the device?"**
   - Nothing. Both services bind to loopback, the offline vars are set, and the demo runs with Wi-Fi off.
   - No endpoint accepts images or audio. The WebSocket carries normalized boxes, labels and numbers, never
     pixels or keypoints.
   - Caveats: no authentication, `/ws/events` does not check `Origin`, and the frontend's camera preview is the
     browser's own stream, outside the gate.
10. **"What if it crashes?"**
    - The window and history are RAM-only and die with the process. The audit log is deleted on clean shutdown.
    - After a crash, `var/audit/<session>.jsonl` stays until a later startup finds it older than 86 400 s
      (TTL runs only at startup).
    - `var/labels/` (only if `/api/label` was used) and `data/real/` persist.
    - Recovery: restart, recalibrate, and the window restarts empty (seed fallback in §5).

## 7. Pre-demo checklist (T-20 to T-5)

- [ ] `make smoke` (= `make test` + `make scan`) passes. Run it in a spare terminal; it does not touch `:8000`.
- [ ] e2e: `.venv/bin/python -m pytest -q tests/test_runtime.py::test_replay_scenario_end_to_end tests/test_served.py`
      (`scripts/e2e_check.py` is **not built**). Then a dry run on `$R`: replay `frustrated_masked` at speed 20 and
      see `MASK OVERRIDE` in T5 within about 20 s.
- [ ] Narrator warm-up:
  - `curl -s 127.0.0.1:8081/health | jq '{mode, warm, error}'` → `"llm"`, `true`, `null`. `/health` returns
    200 even in template fallback, so check `mode`.
  - `.venv/bin/python -m valence.narrator.fake_fact_sheets --n 1 | curl -s -X POST 127.0.0.1:8081/narrate -H "$J" -d @- | jq '{source, latency_ms}'`
  - `curl -s $A/api/narrator/health` → `"healthy": true`
- [ ] `curl -s $A/api/state | jq .status`: `mode` live, `camera` and `mic` true, `fps` about 10, `narrator`
      `ready`, `offline` true, `degraded` `[]`.
- [ ] `curl -s $A/api/governance/verify` → `ok: true`; `curl -s $A/api/eval/summary | jq .available` → `true`.
- [ ] Person #1 is present and `warming_up` is false. `curl -s $A/api/models | jq .served` starts with `m2_lgbm@`.
- [ ] The `face_occlusion` overlay is off, Wi-Fi is off, the laptop is plugged in, the charts are open, and the
      noise source is charged.

## 8. Known gaps that affect the demo

| Gap | Effect |
|---|---|
| `--obscure-face` is a CLI flag of `python -m valence.edge.run` only. The API starts the edge with `obscure_face=False` and has no endpoint or env var for it. | Live proof 3 uses `POST /api/sim/face_occlusion` (SIMULATED) or a sheet of paper held in front of the face (keep your hands away from your head, or head-in-hands fires). Running the CLI alongside `make live` opens the camera twice. |
| `scripts/e2e_check.py`, `scripts/mem_soak.py`, `scripts/audit_verify.py`, hybrid mode | Not built. No 60-min soak or memory chart exists. |
| 2-point lux fit and noise floor in `/api/calibrate/room` | Not built (one-point lux, additive dB only). |
| Narrator p95 with YOLO running, LoRA | Not measured / not shipped. The prompt-only Stage A or the template is what runs. |
| Negative control (`make negative`) | Passes on the served `e696c749` (Focused on every tick of `data/real/capture_typing.jsonl`), but in-sample: the simulator was refit to that capture run. Re-run after any retrain and quote only the latest result. Keyboard-gaze, slump and audio captures are still wanted (`docs/CONTRACT_RESPONSES.md` round 4). |
| `reports/figs/` vs `metrics.json` | All six figures and `metrics.json` were written by one run (2026-10-18 00:03–00:04, `e696c749`). A partial `make eval` run can overwrite only some figures: check timestamps and pitch only figures from the run that wrote `metrics.json`. |
| `VALENCE_DB_CAL_OFFSET` in `.env.example` | No code reads `settings.db_cal_offset`. The edge's offset is `spl_offset_db` in `data/real/calibration.json`. |
