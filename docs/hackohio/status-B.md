# Cloud B status: emotion map

_Times in America/New_York. Newest last._

### 12:58 ET — Spec published (emotion-v1)

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
