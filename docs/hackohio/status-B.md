# Cloud B status: emotion map

_Times in America/New_York. Newest last._

### 13:10 ET — Spec published (emotion-v1)

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
