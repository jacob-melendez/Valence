# LOCAL (Mac) session: next assignment (2026-10-18, after capture runs 2–3)

Start with `git pull --rebase origin claude/happy-ride-7w3ng3`. File ownership, the privacy rules and the git
rules are unchanged from [LOCAL_SESSION_PROMPT.md](LOCAL_SESSION_PROMPT.md). Never write frames, audio or
transcripts to disk. Push small commits. Put questions and measurements in `docs/CONTRACT_REQUESTS.md` (round 5
onward); answers go in `docs/CONTRACT_RESPONSES.md`.

## What changed on the core side since your last pull (details: CONTRACT_RESPONSES round 7)

**Venue room baseline (your round 4).**
- `POST /api/room/baseline {"seconds": 90}` captures the venue's noise and light, with the person sitting neutral.
- The served models now read the room only relative to that baseline: `db_room_*` (dB above it) and `lux_room_*`
  (log2 of the light ratio). Absolute dB and lux are quoted in the text only.
- The models were retrained on a simulator that draws a random venue (40–75 dB) per session. Served core model:
  `m2_lgbm@8a78efc7`. Rollback: `VALENCE_ARTIFACTS_DIR=artifacts/legacy`.

**Other fixes since your last pull:**
- **Replay clock:** replays used the recording's own time, so break clocks read about 8.5 h. Fixed. Your old
  "fists → Frustrated 60%" result came from this bug; corrected, fists read Focused 24/25.
- **OOD rule:** the out-of-distribution check uses physical domains now, so a face visible 100% of the time and
  morning hours no longer shrink confidence.
- **Mic failure:** if the mic fails, the edge restarts video-only and reports `mic_error`.
- **`full_demo` scenario:** re-seeded (100, 2.0 h). `tests/test_scenarios.py` checks every scenario.
- **Frontend:** the P0 crashes are fixed (contract types, `label_display`, explanation persistence, provenance
  badge, error boundaries, default `http://127.0.0.1:8000`).

**Your captures, scale check** (`reports/capture_vs_sim.json`):
- **Keyboard gaze:** slump 0%, motion matches the simulator, and the model reads Focused 25/25.
- **Slump:** detection works, but *held-slump motion* is a problem (P1-1 below).
- **Audio:** laughter and compliments both register. One laugh tick read face valence 0.98.

## P0: demo-critical, in this order

**P0-1. Smoke-test the new backend on the Mac.**
- Pull, then run `.venv/bin/python -m pytest -q`. The Mac-only tests run there; the cloud skips them.
- Run `bash scripts/spikes/run_demo.sh` live.
- Check `/api/health`: the edge is up, `mic_error` is null, and `room_baseline` is present.
- Check that predictions stream and the narrator says ready.
- Done when the suite is green on the Mac and a live run streams `prediction` + `explanation`. Report anything
  red in CONTRACT_REQUESTS.

**P0-2. `scripts/spikes/run_demo.sh` fixes.**
- **(a) bash 3.2:** line 62 expands an empty array under `set -u`: `"${NARGS[@]}"`. Stock macOS bash 3.2 aborts on
  this with "unbound variable" in the default LLM mode. Use `${NARGS[@]+"${NARGS[@]}"}`, and do the same for any
  other array that can be empty.
- **(b) Template mode:** skip the `models/llm/Qwen3-1.7B-4bit` preflight when `NARRATOR=template`.
- **(c) Venue baseline:** in live mode, after the API is up:
  - print "sit neutral at the desk for 90 s";
  - `POST /api/room/baseline {"seconds": 90}`;
  - poll `GET /api/room/baseline` until `mode` is `captured`;
  - print `db_median` and `lux_median`;
  - `BASELINE=0` skips this step.
- **(d) Optional frontend:** `FRONTEND=1` also starts `npm run dev -- --host 127.0.0.1` in `frontend/` and prints the
  URL with `?api=http://127.0.0.1:8000`.
- Done when it runs under `/bin/bash` 3.2 in both narrator modes and prints the captured baseline.

**P0-3. Golden recording `data/real/golden_demo.jsonl`.**
- Record features only, with `--jsonl`. The recorder already keeps participants only.
- Cued segments:
  1. 90 s neutral (the baseline);
  2. 4 min focused typing;
  3. about 8 min of the masked act;
  4. about 2 min away from the desk (a break);
  5. 3–4 min laughing and chatting, with a second person if you can.
- The masked act:
  - room noise at about 65–70 dB (for example, hall noise playing from a phone at the desk);
  - a broad smile held the whole time;
  - a two-handed fist clench every 45–60 s;
  - occasional sighs.
- Push the recording and a summary JSON. In the summary, report face valence p10/p50/p90 during the held smile:
  the face-only model reads Collaborative at ≥ 0.5 only from a valence of about 0.4 up, and no capture has measured
  a held forced smile yet.
- Replay it with `curl -X POST 127.0.0.1:8000/api/replay_recording/golden_demo -H 'content-type: application/json' -d '{"speed": 1, "loop": false}'`.
- Watch for "Frustrated / Masked Stress" with `mask_override: true`. The override needs at least 3 of the last 6
  face-visible ticks.
- Done when the file and summary are pushed and BENCHMARKS.md records whether the override fired and when. If it
  doesn't fire, push anyway: the core side will diagnose with it.

**P0-4. Offline proof.**
- Do a Wi-Fi-off run of `run_demo.sh` in live mode for about 10 min: `status.offline` is true and `lsof -i -p <api pid>`
  is empty.
- Note the FPS and the narrator fallback rate.
- Do the 2-point lux re-check (`scripts/calibrate_lux.py --two-point`) and the SPL check with a phone meter (the venue
  routine, rehearsed at home).
- Record everything in BENCHMARKS.md.

**P0-5. Frontend on the Mac with the real camera.**
- Run `cd frontend && npm install && npm run dev -- --host 127.0.0.1`, then open the printed URL with
  `?api=http://127.0.0.1:8000`.
- Check that:
  - the person's box lines up with the camera view (mirroring could not be checked without a camera);
  - labels show `label_display`;
  - narrator text persists between same-label predictions;
  - the governance counters move;
  - the evaluation panel renders;
  - no "Panel error" appears.
- Report issues in CONTRACT_REQUESTS. `frontend/` is edited on the core side unless the user says otherwise.

## P1: should do, after the P0s

**P1-1. Held-slump motion artifact (perception).**
- The problem, on `capture_slump`:
  - motion p50 is 0.101 shoulder widths/s during a held slump, 0.033 upright and 0.575 in transitions;
  - the simulator's Exhausted is about 0.012;
  - face visibility drops to 0.33 while slumped.
- It looks like keypoint jitter when the head drops. A slumped, tired person then reads as agitated.
- Suggested fixes:
  - compute motion from torso and arm keypoints only (shoulders, elbows, wrists, hips) with a confidence floor;
  - or hold the last value while those keypoints are low-confidence;
  - keep the EMA.
- Re-run the posture probe or a slump capture with 30 s *holds*. Done when held-slump motion is at or below upright
  motion; report the numbers so the core side can refit.

**P1-2. Narrator with the room-relative wording.**
- New driver texts look like "average noise of 72 dB over the last 5 minutes, 6 dB above this room's baseline" or
  "180 lux, dimmer than this room's usual light". `channel_of` maps them to `db` and `lux`.
- The core side checked your grammar: 354 new-style sheets, 0 template fallbacks.
- Please run your LLM/LoRA A/B on a fresh export (`python scripts/export_fact_sheets.py --out /tmp/fs_new.jsonl`) and
  compare the validator pass rate and latency with the shipped result.
- `valence/narrator/fake_fact_sheets.py` got a minimal guard. Its `FEATURE_NAMES` excludes `ROOM_REL`, so its seeded
  output is byte-identical; its test was adjusted to match. Add room-relative drivers there and in the LoRA data only
  if the pass rate drops.

**P1-3. 60-minute live soak.**
- Target: ≥ 8 Hz, 0 purge violations, flat RSS (sample every minute), and the narrator fallback rate.
- Also check `lsof`: only the audit log may be open for writing.
- Record in BENCHMARKS.md. The core side can add `scripts/mem_soak.py` for the chart if you want it; ask.

**P1-4. Social capture of at least 3 minutes.**
- Two people if possible: talking, laughing, 2–3 compliments. `capture_audio` was 45 s, shorter than the 60 s
  warm-up, so it was never scored.
- Also look at the laughter double count: one 5 s laugh gives 2 onsets with the 4 s refractory. Consider holding off
  while the laughter score stays above the off threshold.

**P1-5. Live face-detection rate at 3 light levels** (normal, dim, very dim), for BENCHMARKS.md.

**P1-6. Backup screen recording** of a full `full_demo` replay with the frontend. Keep it outside git: it is media.

## P2 (only if everything above is done)
- EmotiEffLib face model.
- Acted set: 6 scenarios × 3 min, features only.
- Tick off the stale DELIVERABLES items you finished (LoRA adapter, grammar targets, PERCEPTION.md, NARRATOR.md).

## Core side (for context, not yours)
- Docs synced to the venue baseline.
- `e2e_check.py`, `audit_verify.py` and `mem_soak.py`.
- The label/confidence mismatch: a label held by hysteresis can show 2–7% confidence.
- `/api/live` passing `obscure_face` through.
- The Devpost draft.
- Refitting the simulator from P1-1 and P0-3.
