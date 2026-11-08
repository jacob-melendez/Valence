# Benchmark: legacy-covered

- File: `data/real/capture_covered.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 1, 15 s with the camera covered (lux about 37, no person tracked in any frame). There is no right label to hit: with nobody in frame the runtime makes no person prediction, which the report shows as absent ticks. Unscored; kept as the check that nothing is predicted for a hidden face.
- Started: 2026-10-17 23:17:14 UTC; time of day: fixed at 12:00
- Replay: 166 observations, 3 ticks (synthesized every 5 s from the first observation), 15 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Camera covered (`covered`, unscored) | — | 3 (3 absent) | 0 | — | — | — |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Camera covered (`covered`, unscored) | — | 3 (3 absent) | 0 | — | — | — |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Camera covered (`covered`, unscored) | — | 3 (3 absent) | 0 | — | — | — |

## Emotion map (emotion-v1)

Not scored: nobody was predicted, so the map was not read.

## Session

| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |
|---|---|---|
| Card | — (0 blocks) | — (0/0 ticks) |
| Raw M2 | — (0 blocks) | — (0/0 ticks) |
| M0 | — (0 blocks) | — (0/0 ticks) |
| Map | — (0 blocks) | — (0/0 ticks) |

Self-report: no taps.
Parity: no recorded ticks (nothing to compare).
