# Benchmark: legacy-keyboard_gaze

- File: `data/real/capture_keyboard_gaze.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 2, 3 min of typing while looking down at the keyboard (slump 0% under the fixed rule, face visible 100%). Looking down at the keyboard must not read as slumped or Exhausted, so this is a calm block expected Focused; any Exhausted tick is a false alarm. Not in the simulator's refit data.
- Started: 2026-10-18 14:58:59 UTC; time of day: fixed at 12:00
- Replay: 1981 observations, 35 ticks (synthesized every 5 s from the first observation), 175 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Typing, looking down at the keyboard (`keyboard_gaze`, calm) | Focused | 35 | 24 | 100% | 5 s | Focused (100%) |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Typing, looking down at the keyboard (`keyboard_gaze`, calm) | Focused | 35 | 24 | 100% | 5 s | Focused (100%) |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Typing, looking down at the keyboard (`keyboard_gaze`, calm) | Focused | 35 | 24 | 100% | 5 s | Focused (100%) |

## Emotion map (emotion-v1)

Not scored: no emotion-v1 face cues in this recording: the map was not read.

## Session

| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |
|---|---|---|
| Card | — (0 blocks) | 0% (0/24 ticks) |
| Raw M2 | — (0 blocks) | 0% (0/24 ticks) |
| M0 | — (0 blocks) | 0% (0/24 ticks) |
| Map | — (0 blocks) | — (0/0 ticks) |

Self-report: no taps.
Parity: no recorded ticks (nothing to compare).
