# Benchmark: legacy-slump

- File: `data/real/capture_slump.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 2, 3 min of cued 20 s slumps with upright gaps (slump duty 45%, face visible 55%): the protocol's tired block, expected Exhausted. Intermittent cued slumping in a fresh 3-minute window is a weak version of exhaustion (no break-clock pressure), and a held slump reads restless live (MODEL_CARD section 7), so expect a low hit rate.
- Started: 2026-10-18 15:02:07 UTC; time of day: fixed at 12:00
- Replay: 1981 observations, 36 ticks (synthesized every 5 s from the first observation), 180 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Cued slumps (`slump`) | Exhausted | 36 | 25 | 0% | never | Focused (100%) |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Cued slumps (`slump`) | Exhausted | 36 | 25 | 0% | never | Focused (100%) |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Cued slumps (`slump`) | Exhausted | 36 | 25 | 44% | 10 s | Focused (56%) |

## Emotion map (emotion-v1)

Not scored: no emotion-v1 face cues in this recording: the map was not read.

## Session

| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |
|---|---|---|
| Card | 0% (1 blocks) | — (0/0 ticks) |
| Raw M2 | 0% (1 blocks) | — (0/0 ticks) |
| M0 | 44% (1 blocks) | — (0/0 ticks) |
| Map | — (0 blocks) | — (0/0 ticks) |

Self-report: no taps.
Parity: no recorded ticks (nothing to compare).
