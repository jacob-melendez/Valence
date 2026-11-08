# Benchmark: legacy-typing

- File: `data/real/capture_typing.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 1, 3 min of quiet typing (0 fist onsets, 0% slump, face visible 100%): the negative-control capture (valence.eval.negative_control). A calm block, expected Focused. In-sample: the simulator was refit to run 1.
- Started: 2026-10-17 22:59:10 UTC; time of day: fixed at 12:00
- Replay: 1981 observations, 35 ticks (synthesized every 5 s from the first observation), 175 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Quiet typing (`typing`, calm) | Focused | 35 | 24 | 100% | 5 s | Focused (100%) |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Quiet typing (`typing`, calm) | Focused | 35 | 24 | 92% | 5 s | Focused (92%) |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Quiet typing (`typing`, calm) | Focused | 35 | 24 | 92% | 5 s | Focused (92%) |

## Emotion map (emotion-v1)

Not scored: no emotion-v1 face cues in this recording: the map was not read.

## Session

| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |
|---|---|---|
| Card | — (0 blocks) | 0% (0/24 ticks) |
| Raw M2 | — (0 blocks) | 8% (2/24 ticks) |
| M0 | — (0 blocks) | 8% (2/24 ticks) |
| Map | — (0 blocks) | — (0/0 ticks) |

Self-report: no taps.
Parity: no recorded ticks (nothing to compare).
