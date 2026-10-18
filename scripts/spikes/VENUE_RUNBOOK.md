# Venue runbook (Mac, demo day)

Everything below runs from **Terminal.app** (Claude Code's shell can't get camera permission). Repo root:
`cd ~/Desktop/Projects/Valence`. Total setup time at the table: about 10 minutes.

## Before leaving home

- [ ] `git pull --rebase=merges origin claude/happy-ride-7w3ng3` and `.venv/bin/python -m pytest -q` (all green).
- [ ] `ls adapters/narrator/adapters.safetensors` (the fine-tuned narrator is local only, not in git).
- [ ] `ls data/real/golden_demo.jsonl` (the replay fallback; record it first if missing).
- [ ] Charger in the bag. Battery throttling turned every explanation into the template fallback in testing.

## At the table, in this order

1. **Plug in the charger.** Lid open; built-in camera at the top, face lit from the front, not from behind.
2. **Free port 8000** if `startup-tracker` came back after a reboot:
   `launchctl bootout gui/$(id -u)/local.startup-tracker.web` (restore later with
   `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.startup-tracker.web.plist`).
3. **Wi-Fi off** (menu bar). The servers already run in a no-outbound-network sandbox, so this is belt and braces,
   and it is the visible proof for judges.
4. **Light check, 2 min** (before the demo starts, nobody in view):
   `.venv/bin/python scripts/calibrate_lux.py --two-point` and type in a phone lux-meter reading for a dim setup
   (shade the table / lower screen) and the normal hall light. It saves `data/real/calibration.json`.
5. **Start the demo:** `FRONTEND=1 bash scripts/spikes/run_demo.sh`.
   - It prints `sandbox: narrator + API run with outbound network denied`, then `narrator: LLM ready`.
   - **Venue baseline, 90 s:** when it prints ">>> Venue baseline: sit NEUTRAL ...", sit still with a neutral face and
     don't talk. Hall noise and judges talking nearby are fine: that *is* the baseline. It ends with
     `room baseline captured: NN dB, NNN lux`.
6. **Sound check, 30 s:** phone SPL app next to the laptop; compare with the dB shown in the console
   (`.venv/bin/python scripts/console.py` in a second Terminal tab). Within ~3 dB is fine; the models only use dB
   relative to the venue baseline, so an offset error doesn't change predictions, only the quoted numbers.
7. **Open the frontend** at the printed URL (`http://127.0.0.1:8080/?api=http://127.0.0.1:8000`), press **Connect**
   until the badge reads **LIVE · CONNECTED**, and check your box follows you.
8. **First minute is warm-up** (the window fills); predictions say `warming_up` until then.

## If something goes wrong

| symptom | fix |
|---|---|
| `PREFLIGHT FAIL: port 8000 is taken` | step 2, or `VALENCE_API_PORT=8010` and open the frontend with `?api=http://127.0.0.1:8010` |
| `launched from Claude's shell` | run it from Terminal.app |
| edge failed / camera black | lid closed or another app holds the camera; `VALENCE_CAMERA=0`; or replay (below) |
| `mic_error` | runs video-only automatically; check System Settings > Privacy > Microphone > Terminal |
| narrator `fallback` | the explanation text comes from templates (still faithful); check the charger, then restart the demo |
| box offset vertically in the browser | known frontend 4:3-vs-16:9 crop (reported to the core session); predictions are unaffected |
| anything live is flaky | **replay the golden take:** `curl -X POST 127.0.0.1:8000/api/replay_recording/golden_demo -H 'content-type: application/json' -d '{"speed": 1, "loop": false}'` |

Stop everything with Ctrl-C in the demo window. To go back to live after a replay: `curl -X POST 127.0.0.1:8000/api/live`.

## What to show judges (edge side)

- **Purge gate:** governance counters in the UI: frames decoded == frames purged, 0 violations, 0 live raw buffers.
- **No network:** `lsof -nP -i -p $(lsof -t -iTCP:8000 -sTCP:LISTEN)` shows only loopback; MediaPipe's built-in
  usage logger (play.googleapis.com) is blocked by the sandbox profile `scripts/spikes/valence_offline.sb`.
- **No transcript:** Vosk only knows 8 compliment phrases + `[unk]`; recognized words never leave one function.
- **Room-relative context:** the baseline you just captured; a loud hall isn't read as a stressed person.
