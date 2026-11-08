# Benchmark: legacy-fists

- File: `data/real/capture_fists.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 1, 3 min of cued fist clenches (28 onsets, 9.3/min, recall 1.0) in a quiet room with a fresh break clock, otherwise still with a neutral face: the protocol's restless block, expected Frustrated_Masked. MODEL_CARD section 5 already reports Focused on 24/25 ticks: fists alone are not enough for M2, so expect a low hit rate.
- Started: 2026-10-17 23:08:12 UTC; time of day: fixed at 12:00
- Replay: 1981 observations, 36 ticks (synthesized every 5 s from the first observation), 180 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Cued fist clenches (restless) (`fists`) | Frustrated_Masked | 36 | 25 | 0% | never | Focused (96%) |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Cued fist clenches (restless) (`fists`) | Frustrated_Masked | 36 | 25 | 4% | 160 s | Focused (84%) |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Cued fist clenches (restless) (`fists`) | Frustrated_Masked | 36 | 25 | 0% | never | Focused (84%) |

## Emotion map (emotion-v1)

Not scored: no emotion-v1 face cues in this recording: the map was not read.

## Session

| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |
|---|---|---|
| Card | 0% (1 blocks) | — (0/0 ticks) |
| Raw M2 | 4% (1 blocks) | — (0/0 ticks) |
| M0 | 0% (1 blocks) | — (0/0 ticks) |
| Map | — (0 blocks) | — (0/0 ticks) |

Self-report: no taps.
Parity: no recorded ticks (nothing to compare).
