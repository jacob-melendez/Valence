# Benchmark summary

7 session(s); the first 60 s of each block are not scored. Hit rate = share of scored ticks whose label is in the block's expect list (`python -m valence.eval.benchmark`; definitions in the module docstring and docs/PROTOCOL.md section 4).

## Sessions

Mean hit rate over the scored non-calm blocks, calm false-alarm rate over the scored calm blocks.

| Session | Ticks | Card | Raw M2 | M0 | Map | Card calm FA | Map calm FA | Parity (card) |
|---|---|---|---|---|---|---|---|---|
| legacy-baseline | 4 | — | — | — | — | — | — | — |
| legacy-typing | 35 | — | — | — | — | 0% | — | — |
| legacy-keyboard_gaze | 35 | — | — | — | — | 0% | — | — |
| legacy-fists | 36 | 0% | 4% | 0% | — | — | — | — |
| legacy-slump | 36 | 0% | 0% | 44% | — | — | — | — |
| legacy-covered | 3 | — | — | — | — | — | — | — |
| legacy-audio | 9 | — | — | — | — | — | — | — |

## Blocks (pooled over sessions)

Hits / ticks over every session's scored ticks of the block (— : no reading, or no scored tick).

| Block | Expect (card) | Sessions | Ticks | Card | Raw M2 | M0 | Map |
|---|---|---|---|---|---|---|---|
| Baseline (sitting neutral) (`baseline`, calm) | Focused | 1 | 0 | — | — | — | — |
| Quiet typing (`typing`, calm) | Focused | 1 | 24 | 100% | 92% | 92% | — |
| Typing, looking down at the keyboard (`keyboard_gaze`, calm) | Focused | 1 | 24 | 100% | 100% | 100% | — |
| Cued fist clenches (restless) (`fists`) | Frustrated_Masked | 1 | 25 | 0% | 4% | 0% | — |
| Cued slumps (`slump`) | Exhausted | 1 | 25 | 0% | 0% | 44% | — |
| Laughter and compliments (`audio`) | Collaborative_Happy | 1 | 0 | — | — | — | — |

Map: no emotion-v1 face cues in this recording: the map was not read (6 session(s)); nobody was predicted, so the map was not read (1 session(s)).
