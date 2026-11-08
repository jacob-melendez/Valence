# Benchmark: legacy-baseline

- File: `data/real/capture_baseline.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 3, 25 s sitting neutral in a quiet room (face visible 100%, no fists or slump). A calm block: the card should read Focused. It is shorter than the 60 s grace, so it has no scored ticks at the default grace.
- Started: 2026-10-18 15:21:43 UTC; time of day: fixed at 12:00
- Replay: 275 observations, 4 ticks (synthesized every 5 s from the first observation), 20 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Baseline (sitting neutral) (`baseline`, calm) | Focused | 4 | 0 | — | 5 s | — |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Baseline (sitting neutral) (`baseline`, calm) | Focused | 4 | 0 | — | 5 s | — |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Baseline (sitting neutral) (`baseline`, calm) | Focused | 4 | 0 | — | 5 s | — |

## Emotion map (emotion-v1)

Not scored: no emotion-v1 face cues in this recording: the map was not read.

## Session

| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |
|---|---|---|
| Card | — (0 blocks) | — (0/0 ticks) |
| Raw M2 | — (0 blocks) | — (0/0 ticks) |
| M0 | — (0 blocks) | — (0/0 ticks) |
| Map | — (0 blocks) | — (0/0 ticks) |

Self-report: no taps.
Parity: no recorded ticks (nothing to compare).
