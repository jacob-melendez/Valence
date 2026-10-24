# Cloud B status: emotion map

_Times in America/New_York. Newest last._

### 12:32 ET — Spec published (emotion-v1)

Done:
- `docs/EMOTION_FEATURES.md`: per-frame face cues from MediaPipe (brow lowering, eye openness, blinks, head
  yaw/pitch, frown, smile; exact blendshape names, matrix layout and sign convention), the per-person neutral
  baseline, the blink rule, per-tick aggregation, 31 new window features (`emotion-v1`), 10 states with the plan's
  valence–arousal anchors (marked as assumptions), the artifact layout and the self-report JSONL format.
- `valence/contracts.py` (additive only): `PersonObs` and `TickRecord` cue fields (NaN / 0 defaults),
  `EMOTION_STATES`, `EMOTION_DISPLAY`, `AFFECT_ANCHORS`, `Affect`, optional `Prediction.affect`, `Hello.affect_states`.
- Reference code: `valence/features/face_cues.py` (per frame), `valence/runtime/aggregator.py` (per tick, additive),
  `valence/features/emotion.py` (31 window features), simulator cues for all states + Bored, Confused, Calm, Anxious
  (state set `emotion`; separate random stream).

Numbers: full suite 687 passed, 41 skipped. Existing scenarios and core/all corpora hash-identical to the pre-event
simulator on the original columns (`tests/test_emotion_features.py`).

Next: 10-state classifier + valence/arousal regressors, evaluation, JSON export to `artifacts/emotion/`.

Blockers: none.

### 13:05 ET — 10-state model trained and exported (emotion-v1@378fa16c)

Done:
- `artifacts/emotion/`: `classifier.json.gz` (2.78 MB), `valence.json.gz` (0.45 MB), `arousal.json.gz` (0.71 MB)
  as `dump_model()` JSON; `meta.json` (input order, states + anchors, temperature 0.948, layers, summary);
  `golden.json` (23 vectors with raw/probabilities/V-A/`pred_contrib`, 8 feature windows, 370 face-cue frames);
  `tour.json.gz` (a 50-min scripted walk through all 10 states: tick records + Python's per-tick outputs, for a
  camera-free affect-map demo and end-to-end parity). `metrics.json` has everything below.
- `valence/models/emotion_json.py` evaluates the JSON dumps without LightGBM: matches LightGBM exactly on every
  golden vector (reference for the TS runtime).
- `valence/eval/self_report.py` runs on a synthetic fixture until real taps arrive.

Numbers (simulated, 600 sessions, 90 held-out test sessions, 24 548 windows):

| Model | Macro-F1 | ECE | Valence MAE / CCC | Arousal MAE / CCC |
|---|---|---|---|---|
| **Full (served)**, map position = probability-weighted anchors | **0.688** | 0.007 | **0.176 / 0.78** | **0.238 / 0.72** |
| Full, regressors | — | — | 0.199 / 0.75 | 0.267 / 0.70 |
| Face only (38 features) | 0.621 | 0.007 | 0.205 / 0.71 | 0.275 / 0.65 |
| Context only (94) | 0.425 | 0.005 | 0.274 / 0.48 | 0.329 / 0.51 |
| Full, face hidden at test time | 0.414 | 0.014 | 0.275 / 0.48 | 0.329 / 0.51 |
| Shuffled context, test time | 0.429 | 0.163 | | |
| Shuffled context, trained | 0.618 (≈ face only: no leak) | 0.007 | | |

Per-state F1: Collaborative 0.84, Tense 0.79, Frustrated 0.76, Exhausted 0.75, Overwhelmed 0.68, Focused 0.68,
Calm 0.67, Anxious 0.62, Bored 0.61, **Confused 0.48** (weakest; confused with Focused 13%, Anxious 10%).
Frustrated/Tense/Anxious/Overwhelmed: 83% of their windows land inside that region, 87% of those on the right state.

Decision: the regressors lose to the probability-weighted anchors on both axes (the simulator's within-state spread
has little observable signal), so the served map position is `Σ p_k · anchor_k`; the regressors stay exported and
reported. Written in `docs/EMOTION_FEATURES.md` §5.

Next: model-card section; self-report evaluation when real taps exist; error analysis on Confused.

Blockers: none.

### 15:04 ET — Two reviews, fixes, retrain (emotion-v1@006a8d7d)

Done:
- **Spec-vs-reference review** (independent agent, 10 findings, all fixed in `140d7e5`, `1418119`):
  - blink rate is missing, not 0, when a visible face has no cues (older producers);
  - a 1e-9 tolerance on the 10 s blink test;
  - the JSON tree reader zeroes |x| ≤ 1e-35f and reads ±1e300 thresholds as ±inf, so it now matches LightGBM on
    4 600 probes with inf, huge, tiny and NaN inputs;
  - golden data adds 17 blink edge-case frames and 3 missing-data windows;
  - the spec now states the exact edge rules. Cloud A ported these with 0 mismatches.
- **Four-lens review workflow** (38 agents, 17 findings, 14 confirmed by adversarial verification), all addressed:
  - The simulator's head pose is now per frame, so `head_away` agrees with yaw and pitch as it does live. Retrained:
    `emotion-v1@006a8d7d`.
  - The trained shuffled-context control is no longer cited as a leak test.
  - Robustness conditions now stress every face-cue spread.
  - Self-report: the evaluator skips duplicate and non-object lines and has a fair constant baseline.
  - Capture replay handles time of day and short recordings.
  - Model card §9 is now generated from the artifacts (`scripts/emotion_card.py`).

Numbers (simulated; 90 test sessions):

| Metric | Result |
|---|---|
| Macro-F1 | 0.681 (95% CI 0.667–0.695) |
| Face only | 0.616 |
| Context gain | +0.066 (CI +0.059 to +0.074) |
| ECE | 0.007 |
| Map-position MAE | valence 0.174, arousal 0.236 |
| Weakest state | Confused, F1 0.46 |

Robustness:

| Shift | Macro-F1 |
|---|---|
| None (fresh sessions) | 0.691 |
| Every face-cue spread × 2 | 0.554 |
| Between-person cue spread × 1.5 | 0.649 |
| Pitch sign flipped | 0.604 |
| Blinks halved | 0.681 |

Pre-event real captures (no cues):

- Quiet typing reads Focused 78%, Calm 22%, and no stress state.
- Cued fists read Focused 83% and Frustrated 12%.

Next: real self-report taps once Cloud A's browser-engine glue registers the feature provider. Then re-run
`valence.eval.emotion_capture` on browser recordings to measure real cue spreads against the simulator.

Blockers: none (self-report capture depends on Cloud A's glue, in progress).

### 16:38 ET — Docs re-verified; first real browser cues

Done:
- **Second verification pass** (24 agents) of `EMOTION_FEATURES.md` against the code and model card §9 against
  the artifacts: 6 confirmed and 3 split findings, all wording or number fixes (`4da1b42`, `b2f138a`). The card now
  states:
  - the web port's parity exactly: bit-identical raw scores and regressors, p within 3 ulp, the map position
    within 4e-16, and the 600-tick tour within 1e-9 with identical labels;
  - that the shared negative region is told apart mostly by face cues: face-only F1 0.50–0.72 against 0.40–0.48
    for context only.
- **Real browser cues** (`ab05fa5`): Cloud C's two CC BY 4.0 clips, run through the real in-browser perception
  (`data/browser/*.jsonl`), replayed with `valence.eval.emotion_capture --ref Focused`. The results are in
  `artifacts/emotion/browser_captures.json` and model card §9.6.
  - On neutral faces, the per-tick cues are much steadier than simulated. Smile and frown sit at about 0; brow
    lowering stays near 0 for one person and reaches 0.16 at p90 for the other; every tick-to-tick step is smaller
    than simulated. So the simulator's within-state noise is pessimistic.
  - Between-state differences are still unmeasured. The per-state cue means are the untested part.
  - The map reads Focused 93%, and Focused/Calm 50/50, with no negative state.
  - Blinks: about 9–10/min at 6.9 face fps.

Numbers unchanged: `emotion-v1@006a8d7d`, macro-F1 0.681 (CI 0.667–0.695), face only 0.616, ECE 0.007. The pitch
draft quotes them correctly.

Next: self-report taps (the capture works now; it needs a person). A labeled real check of the cues, ideally a short
acted session per state from the user. Any fixes the coordinator reports.

Blockers: none.
