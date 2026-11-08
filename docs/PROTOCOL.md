# Real-data protocol and session format (benchmark, phase 0)

Every model number so far comes from simulated data. This protocol produces **labeled real sessions**: a person works
through timed behaviour blocks in the web console while it records the session's features (never frames or audio).
`python -m valence.eval.benchmark` replays those sessions through the Python runtime and scores each model per block.

## 1. The protocol (`valence-protocol/1`, about 24 minutes)

Run it from the console: **Session** panel → **Start protocol**. The panel shows the current block, what to do and a
countdown, and it records the block boundaries. Keep the camera on and stay in frame; the self-report panel still asks
how you feel every 2 minutes (tap honestly; those taps are the second label source).

| # | Block (`id`) | Minutes | What to do | Card (4-state) should read | Emotion map should read | Scored |
|---|---|---|---|---|---|---|
| 1 | Settle in (`settle`) | 2 | Sit as you normally work, relaxed; a neutral face for the first 20 s | — | — | no: face and room baselines |
| 2 | Focused work (`focused`) | 6 | Read or type something engaging; stay quiet and still | Focused | Focused, Calm | yes (calm block) |
| 3 | Restless (`restless`) | 3 | Fidget, shift in your seat, tap, clench your fists now and then, sigh | Frustrated_Masked | Frustrated_Masked, Tense, Anxious, Overwhelmed | yes |
| 4 | Frustrating task (`frustrated`) | 4 | Mental arithmetic against the clock (count down from 1000 by 7s out loud); let it show | Frustrated_Masked | Frustrated_Masked, Tense, Confused, Overwhelmed, Anxious | yes |
| 5 | Tired (`tired`) | 3 | Slump, rest your head in your hands, look down, slow down | Exhausted | Exhausted, Bored | yes |
| 6 | Upbeat (`upbeat`) | 3 | Watch or listen to something funny, or chat with someone; smile and laugh | Collaborative_Happy | Collaborative_Happy | yes |
| 7 | Calm (`calm`) | 3 | Sit back, breathe slowly, relaxed face | Focused | Calm, Focused | yes (calm block) |

Notes:
- **The order is fixed** so sessions are comparable, and the 8 minutes before the first non-calm block give the
  context model its history (it needs about 10 minutes to read well; that is one of the things this measures).
- **Do not press Pause or Stop** during the protocol: either one ends the protocol and the recording (the file keeps
  what was recorded, with the reason), and Start afterwards forgets the people. To stop early, use **Abort protocol**.
- **Alone, the card shows Collaborative_Happy as "Upbeat".** The expectation is the state id, not the display text.
- **Consent.** Record yourself, or people who agreed to it knowing what is kept (section 2). If the recordings become
  research data about other people, check with your institution's review board first.
- **Several sessions beat one long one.** Different days, rooms, lighting and times of day are the point.

## 2. What a session file holds (privacy)

A session file (`valence-session-<date>-<id>.jsonl`) holds only what the perception layer already reduces frames and
audio to: counts, levels, flags, normalized person boxes, face-cue scalars (brow lowering, eye openness, smile, head
pose…), the model outputs, block boundaries and self-report taps. **No frames, no audio, no landmarks, no
transcripts.** It exists in the tab's memory until you download or clear it, and is never uploaded: the page cannot
connect anywhere but its own origin. Recording is off unless you turn it on (or start the protocol).

## 3. File format (`valence-session/1`)

JSON Lines, in the order the browser engine took them (oldest first). Every line has a `type`. Times: `wall` is Unix
seconds (UTC), `mono` is the page's monotonic clock in seconds, the clock observation `t` values use. The benchmark
skips line types it does not know; `valence/runtime/recording.py iter_obs` (observation replay) skips the four meta
types below (`session`, `tick`, `marker`, `self_report`) and still rejects anything else.

| `type` | When | Fields |
|---|---|---|
| `session` | First line, once | `format` (`"valence-session/1"`), `session_id`, `started_wall` and `started_mono` (the recording's start on both clocks), `tz` (IANA zone of the browser, for the time of day; null if the browser gave none, and the benchmark then fixes the time of day at 12:00 unless `--tz` is passed), `feature_version` (served model id), `emotion_model` (id or null), `protocol` (id or null), `app` (`"valence-web"`) |
| `video`, `audio` | Each perception observation | Exactly `obs_to_dict` / `obsToDict` (`valence/contracts.py`, `frontend/src/perception/types.ts`) without the optional `raw` debug block. `valence/runtime/recording.py iter_obs` reads these lines and skips the others. |
| `tick` | Each engine tick (5 s) | `wall`, `mono` (the values the engine stepped with) and `predictions`: one entry per prediction message, `{person_id, label, label_display, confidence, warming_up, mask_override, face_only: {label, confidence}, affect: {label, valence, arousal} or null}` |
| `marker` | Block boundaries and operator actions | `wall`, `mono`, `kind` (`block_start`, `block_end`, `protocol_start`, `protocol_end`, `protocol_abort`, `control`). Blocks add `block_id`, `label`, `scored` (bool), `calm` (bool), `expect: {card: [...], map: [...]}`; `protocol_start`/`protocol_end` add `label` (the protocol id) and `protocol_abort` adds `reason`; `control` adds `action`: `start`, `pause`, `stop`, `break`, `demo_reset`, `room_only_on`/`room_only_off`, `sim_<overlay>_on`/`_off`, `baseline_<seconds>` or `baseline_default`, `baseline_reset` |
| `self_report` | Each self-report tap | The tap as `selfReportsJsonl` writes it (`t`, `person_id`, `valence`, `arousal`, `feature_version`, `features`, optional `label`) plus `wall` and `mono` |

A legacy recording (`data/real/capture_*.jsonl`: only `video`/`audio` lines) is scored with a manifest that gives the
whole file one block (`data/real/manifest.json`, format `valence-legacy-manifest/1`, a note on each entry); its ticks are
synthesized every 5 s from the first observation.

## 4. The benchmark

```
python -m valence.eval.benchmark sessions/*.jsonl --out reports/benchmark
python -m valence.eval.benchmark --legacy data/real/manifest.json --out reports/benchmark
```

For each session it replays the observations through the Python runtime (the served model, smoother and mask
override, exactly what the card shows) in file order, stepping at the session's recorded tick times (`--order mono`
interleaves by the page clock instead, which moves a frame that reached the engine just after a tick into that tick).
The emotion map is computed beside the runtime from its window, as the browser does. Per block, for the card, the raw
M2 argmax, the face-only M0 and the emotion map:

- **hit rate**: over the block's ticks after a grace period (default 60 s, `--grace`), the share whose label is in the
  block's `expect` list. A tick where the person was predicted but that model had no reading (M0 without a face)
  counts as a miss; ticks where the person was not predicted at all are reported as absent and left out;
- **time to first hit**: seconds from the block's start to its first expected label, over all of the block's ticks
  (grace included, so a label carried over from the previous block counts);
- **label distribution** and majority label (after the grace period);
- **calm false-alarm rate**: in scored calm blocks, the share of ticks with a reading whose card label is not Focused
  (map: not Calm or Focused);
- **self-report agreement**: map valence/arousal MAE against each tap, matched to the latest tick at or before it
  (the lag is reported), next to a constant (0, 0) guess;
- **parity**: the share of ticks where the replayed label equals what the browser showed (tick lines), for the card,
  M0 and the map, with the largest confidence and valence/arousal differences. In file order it should be 100% with
  differences of 0, because the browser engine is a parity port (verified on a real headless-Chromium recording).
