# Benchmark: legacy-audio

- File: `data/real/capture_audio.jsonl` (legacy observation recording, no tick lines)
- Manifest note: Run 3, 45 s of talking, laughing and complimenting at the desk (2 laughter onsets, 3 compliments, one laugh tick with face valence 0.98): the protocol's upbeat block, expected Collaborative_Happy (shown as Upbeat for one person). Shorter than the 60 s grace, so no scored ticks at the default grace.
- Started: 2026-10-18 15:24:21 UTC; time of day: fixed at 12:00
- Replay: 496 observations, 9 ticks (synthesized every 5 s from the first observation), 45 s, order `mono`; served `m2_lgbm@8a78efc7`, emotion map `emotion-v1@006a8d7d`
- Scored: person 1, the first 60 s of each block skipped; hit rate = scored ticks whose label is expected (no reading counts as a miss); first hit = seconds from the block start, grace included

## Card (served M2: smoothed label with mask override)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Laughter and compliments (`audio`) | Collaborative_Happy | 9 | 0 | — | never | — |

## Raw M2 argmax (before smoothing)

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Laughter and compliments (`audio`) | Collaborative_Happy | 9 | 0 | — | 35 s | — |

## Face-only M0

| Block | Expect | In block | Scored | Hit rate | First hit | Majority |
|---|---|---|---|---|---|---|
| Laughter and compliments (`audio`) | Collaborative_Happy | 9 | 0 | — | 35 s | — |

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
