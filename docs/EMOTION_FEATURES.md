# Emotion map: face cues, tick fields, window features and states (feature version `emotion-v1`)

The contract Cloud C (browser perception), Cloud A (web engine) and Cloud B (emotion map models) build against
during Hack OHI/O. Everything here is **additive**: the existing 123 features (`valence/features/compute.py`),
the existing `TickRecord` fields and the served models are frozen and unchanged. The new values travel in new
optional fields (`valence/contracts.py`, defaults NaN / 0), so old recordings and old producers keep working.

Python reference implementations (the source of truth for parity tests):

| Piece | Python | Parity target |
|---|---|---|
| Per-frame cues from MediaPipe (§1) | `valence/features/face_cues.py` | exact formulas; golden frames in `artifacts/emotion/golden.json` → `frames` |
| Per-tick aggregation (§2) | `valence/runtime/aggregator.py` (`_PersonAcc`, `flush`) | — |
| Window features (§3) | `valence/features/emotion.py` `compute_emotion()` | 1e-6, golden windows in `artifacts/emotion/golden.json` → `windows` |
| Models (§5) | `valence/models/emotion.py`, export in `artifacts/emotion/` | 1e-5 on raw scores, probabilities, valence/arousal and `pred_contrib` |

Privacy is unchanged: only scalars per frame and per tick. No landmarks, no matrix, no frame and no audio is kept
or sent; the per-person neutral baselines live in memory under the ephemeral display id.

---

## 1. Per-frame face cues (browser perception, one value set per tracked face per processed frame)

Inputs per face, from MediaPipe Tasks `FaceLandmarker` with `outputFaceBlendshapes: true` and
`outputFacialTransformationMatrixes: true`:

- `B[name]`: the blendshape score (0–1) for a category name below. A missing category counts as missing.
- `M`: the 4×4 facial transformation matrix (`facialTransformationMatrixes[i].data`, 16 numbers).

Let `avg(a, b)` be the mean of the two sides; if one side is missing (absent, null or NaN) use the other; if both are
missing the value is NaN.

### 1.1 Raw values

| Raw value | Formula | Range |
|---|---|---|
| `brow_raw` | `avg(B.browDownLeft, B.browDownRight)` | 0–1 |
| `closure_raw` | `avg(B.eyeBlinkLeft, B.eyeBlinkRight)` | 0–1 |
| `eye_raw` | `1 − closure_raw` (eye openness) | 0–1 |
| `frown_raw` | `avg(B.mouthFrownLeft, B.mouthFrownRight)` | 0–1 |
| `smile_raw` | `avg(B.mouthSmileLeft, B.mouthSmileRight)` | 0–1 |
| `yaw_raw` | `deg(atan2(f_x, f_z))` | −180..180 |
| `pitch_raw` | `deg(atan2(f_y, sqrt(f_x² + f_z²)))` | −90..90 |

**Head pose.** `f = (f_x, f_y, f_z)` is the face's forward axis in camera space: the third column of the rotation
block of `M` (the image of the canonical model's +Z axis, which points out of the face, toward the camera).

- Layout: MediaPipe packs the matrix **column-major** (`M[row][col] = data[col * 4 + row]`), so
  `f = (data[8], data[9], data[10])` and the translation is `data[12..14]`. To be layout-proof: if
  `|data[14]| < |data[11]|` (the translation sits in the last row, i.e. row-major), use `f = (data[2], data[6], data[10])`.
  A face in front of the camera has a translation z of roughly −20 to −80 (cm), so the test is unambiguous.
- `atan2` is scale-invariant, so a uniform scale in `M` does not matter.
- **Sign convention** (MediaPipe's metric space: right-handed, camera at the origin looking down −Z, +X to the right
  of the image MediaPipe received, +Y up):
  - `yaw > 0`: the nose points toward the **right of the image as given to MediaPipe**. If the page feeds a
    mirrored canvas, the sign flips; the models only use `|yaw|`, so this does not matter for them.
  - `pitch > 0`: the face tilts **up**; `pitch < 0`: it tilts **down** (keyboard, phone, nodding off). The models
    use the sign of pitch.
  - A face looking straight into the lens reads `yaw ≈ 0`, `pitch ≈ 0`.
  - **Check on the fake/real camera:** nodding down must make `pitch_raw` decrease. If it increases, the matrix is
    being read transposed; fix the read, do not flip the formula.

### 1.2 Neutral baseline (one per person, same rule as `face_valence`)

Resting faces and camera placements differ, so the cues are **relative to the person's neutral**, exactly like
`face_valence` (`valence/perception/face_valence.py` `FaceValenceCalibrator`):

- One calibrator per ephemeral display id, in memory only, holding the six raw channels
  `brow_raw, closure_raw, frown_raw, smile_raw, yaw_raw, pitch_raw`.
- Perception feeds the calibrator only frames where the face is visible (a face was found **and** `face_valence` is
  finite); other frames are "not visible" (`FaceCueTracker.update(None, t)`), and so is a frame whose six raw channels
  are all NaN. A NaN in a single channel just skips that channel.
- Visible-face time accrues only between consecutive visible frames: `gap = min(max(t − t_prev, 0), 1 s)` (a clock
  that steps backwards adds 0).
- While visible time < 20 s: append the frame's raw values; the baseline of each channel is the running median of
  its finite samples (0 before any sample).
- At 20 s of visible face the baseline is frozen (per channel, the median at that moment) and never changes again for
  that display id (a new display id starts a new calibrator).
- A frame with no face (or `face_valence` NaN) never enters the baseline.
- Order within a frame (as in `FaceCueTracker.update`): accrue visible time → append the sample (or freeze) →
  read the baseline → blink detector and relative values. While calibrating, the baseline used for a frame
  already includes that frame.

Relative per-frame values (what perception puts into `PersonObs`):

| `PersonObs` field | Formula | Range (clipped) | Unit |
|---|---|---|---|
| `brow_down` | `brow_raw − base.brow` | −1..1 | blendshape score |
| `eye_open` | `base.closure − closure_raw` (= `eye_raw − (1 − base.closure)`) | −1..1 | blendshape score |
| `mouth_frown` | `frown_raw − base.frown` | −1..1 | blendshape score |
| `smile` | `smile_raw − base.smile` | −1..1 | blendshape score |
| `head_yaw` | `wrap(yaw_raw − base.yaw)` with `wrap(d) = ((d + 180) mod 360) − 180`, `mod` floored (result in [0, 360), Python's `%`; in JS `((x % 360) + 360) % 360`); so `d = 180` → −180 → clipped to −90 | −90..90 | degrees, signed as above |
| `head_pitch` | `pitch_raw − base.pitch` | −90..90 | degrees, + up |
| `blink_onsets` | 0 or 1, see §1.3 | int | events |

All of them are NaN (blink 0) on frames where the face is not visible. They are only meaningful on frames where
`face_visible` is true and `face_valence` is finite (the frames `TickRecord.face_visible` counts).

### 1.3 Blink events

A per-person two-state detector on `closure_raw` (not relative), with thresholds relative to the neutral
closure `b = base.closure` (the running median while calibrating):

1. State `open` → `closed` when `closure_raw ≥ b + 0.30`. Remember `t_closed` = the frame's timestamp (ms).
2. State `closed` → `open` when `closure_raw ≤ b + 0.15`. **A blink is counted on this frame**
   (`blink_onsets = 1`) if `t − t_closed ≤ 600 ms`; a longer closure is not a blink (it shows up as low `eye_open`).
3. A frame without a visible face resets the state to `open` without counting. A visible frame whose `closure_raw` is
   NaN leaves the state unchanged.
4. At most one blink per frame. The thresholds are compared exactly as written, in float64 (`closure_raw >= b + 0.30`,
   `closure_raw <= b + 0.15`, `t − t_closed <= 600`); `t − t_closed` may be negative after a backwards clock step and
   then counts.

Blinks last ~100–400 ms, so the detector needs ≥ 15 processed face frames per second to see most of them. At lower
rates the measured blink rate is lower than the true one; report the face fps next to any blink numbers.

---

## 2. Per-tick fields (`TickRecord`, one row per person per 5 s tick)

"Face frames" are the tick's frames with `face_visible` true and `face_valence` finite (the same frames as
`TickRecord.face_visible`). For each channel, frames whose value is NaN are skipped; a channel with no finite value
in the tick is NaN.

| `TickRecord` field | Default | Per-tick rule | Unit |
|---|---|---|---|
| `brow_down` | NaN | median of `brow_down` over face frames | score, relative |
| `eye_open` | NaN | median of `eye_open` over face frames | score, relative |
| `mouth_frown` | NaN | median of `mouth_frown` over face frames | score, relative |
| `smile` | NaN | median of `smile` over face frames | score, relative |
| `head_yaw` | NaN | median of `|head_yaw|` over face frames (unsigned) | degrees ≥ 0 |
| `head_pitch` | NaN | median of `head_pitch` over face frames (signed) | degrees, + up |
| `head_away` | NaN | among the face frames whose `head_yaw` and `head_pitch` are both finite (the denominator; NaN if there are none), the share with `|head_yaw| > 20` or `|head_pitch| > 20` | 0–1 |
| `blink_count` | 0 | sum of `blink_onsets` over **all** of the person's frames in the tick (a null/NaN `blink_onsets` adds 0) | count |

The room row (`person_id 0`) leaves all eight at their defaults. Medians use the usual definition (mean of the two
middle values for an even count).

---

## 3. Window features (`emotion-v1`, `valence/features/emotion.py`)

Computed from the **person's** window only (oldest first, the last ≤ 180 ticks), next to the existing 123. Window
lengths in ticks: `1m = 12`, `5m = 60`, `15m = 180` (a shorter history uses what exists). `x[-n:]` is the last `n`
ticks; `nanmean`, `nanstd` (population, ddof 0) and `nanmax` skip NaN and return NaN when every value is NaN, exactly
like the existing `_nanstat`.

| Feature | Formula | Count |
|---|---|---|
| `{c}_now` | `x[-1]` (NaN when the face was not read on the last tick) | 6 |
| `{c}_mean_1m` | `nanmean(x[-12:])` | 6 |
| `{c}_mean_5m` | `nanmean(x[-60:])` | 6 |
| `{c}_std_5m` | `nanstd(x[-60:])` | 6 |
| `eye_mean_15m` | `nanmean(eye_open[-180:])` | 1 |
| `brow_max_1m` | `nanmax(brow_down[-12:])` | 1 |
| `away_mean_1m`, `away_mean_5m` | `nanmean(head_away[-n:])` | 2 |
| `blink_rate_1m`, `blink_rate_5m`, `blink_rate_15m` | per minute of **cued** visible face: over the last `n` ticks, keep only ticks whose `eye_open` is finite (the cues were produced); `seconds = 5 · Σ face_visible`, `rate = 60 · Σ blink_count / seconds`; NaN if `seconds < 10 − 1e-9`. NaN counts as 0 in both sums. A tick with a visible face but no cues (an older producer) is missing, not "no blinks" | 3 |

where `c` runs over `brow` (`brow_down`), `eye` (`eye_open`), `frown` (`mouth_frown`), `smile` (`smile`), `yaw`
(`head_yaw`), `pitch` (`head_pitch`), in that order. **31 features**, in this exact order (also in
`artifacts/emotion/meta.json` → `emotion_feature_names`):

```
brow_now brow_mean_1m brow_mean_5m brow_std_5m
eye_now eye_mean_1m eye_mean_5m eye_std_5m
frown_now frown_mean_1m frown_mean_5m frown_std_5m
smile_now smile_mean_1m smile_mean_5m smile_std_5m
yaw_now yaw_mean_1m yaw_mean_5m yaw_std_5m
pitch_now pitch_mean_1m pitch_mean_5m pitch_std_5m
eye_mean_15m brow_max_1m away_mean_1m away_mean_5m blink_rate_1m blink_rate_5m blink_rate_15m
```

**Numerics.** Sums, means and standard deviations are numpy's (`np.sum`, `np.nanmean`, `np.nanstd`: pairwise
summation), as in the frozen 123; the web engine's `pairwiseSum` matches them.

**Missing face.** A tick without a readable face carries NaN in the seven continuous channels and contributes
0 visible seconds to the blink rate; window statistics skip it. A window with no readable face at all gives NaN for
every feature (blink rates included). Nothing is imputed and nothing is clipped. All 31 features belong to the
**Facial Surface** explanation layer.

**How the trees read NaN** (LightGBM, per split; `valence/models/emotion_json.py`): first, any input with
`|x| <= 1e-35f` (1.0000000180025095e-35) is read as exactly 0; then a split with `missing_type` `NaN` sends NaN the
`default_left` way, `Zero` reads NaN as 0 and sends 0 the default way, and `None` reads NaN as 0 and compares.
A dump threshold of `±1e300` stands for `±inf` (JSON has no infinity; the text models keep `inf`), so `x = +inf` or
`x > 1e300` goes left at such a split (11 splits, on `blink_rate_1m` and `db_rel_15m`). All 31
emotion features only meet `NaN` splits; 30 of the M2 inputs (e.g. `lux_room_now`, `hours_since_break`,
`face_visible_now`, `fill_ratio`) only meet `None` splits, so for them NaN behaves like 0.

**Model input.** The emotion models read `meta.json` → `input_features`: the served M2 set (`FEATURE_SETS["m2"]`,
101 of the frozen 123, room-relative) followed by the 31 above; 132 values in that order. The face-only comparison
model reads the 7 existing face features plus the 31.

---

## 4. States and the valence–arousal plane

Ten states: the existing six plus Bored, Confused, Calm and Anxious. Class order (also `meta.json` → `states`):

| id | Display | Valence | Arousal | Main cues (simulator, §6 of the model card) |
|---|---|---|---|---|
| `Focused` | Focused | +0.1 | +0.3 | still, eyes on the screen (low `head_away`), low blink rate, slight brow lowering |
| `Frustrated_Masked` | Frustrated / Masked Stress | −0.6 | +0.6 | loud room, fists, a forced smile over lowered brows |
| `Exhausted` | Exhausted | −0.4 | −0.7 | long time since a break, slumped, eyes closing, high blink rate, head down |
| `Collaborative_Happy` | Collaborative / Happy | +0.7 | +0.5 | laughter, compliments, smiling, looking at others |
| `Overwhelmed` | Overwhelmed / Distressed | −0.6 | +0.8 | head in hands, face often hidden, frown |
| `Tense` | Tense / Visible Frustration | −0.4 | +0.5 | fists, loud, lowered brows and frown |
| `Bored` | Bored | −0.3 | −0.5 | looking away or down (phone) a lot, slumped, chin in hand, little motion |
| `Confused` | Confused | −0.2 | +0.3 | strongly lowered brows without a frown, moderate motion, quiet room |
| `Calm` | Calm | +0.4 | −0.4 | relaxed, slight smile, normal blink rate, after breaks |
| `Anxious` | Anxious | −0.5 | +0.6 | fidgeting, high blink rate, glancing around, quiet room, no fists |

**The anchors are assumptions**, taken from the plan's circumplex placement, not measured. In the simulator each
episode's true valence and arousal is the anchor plus a per-episode offset (N(0, 0.10) each, clipped to −1..1) that
also shifts the episode's face cues, so the regressors have within-state signal to find. Frustrated / Tense /
Anxious (and Overwhelmed) share the negative, high-arousal region; they are told apart by context and behavior,
not by valence–arousal.

---

## 5. Model outputs and artifacts (`artifacts/emotion/`)

| File | Content |
|---|---|
| `classifier.json.gz` | `booster.dump_model()` of the 10-state LightGBM classifier (multiclass, raw scores per class) |
| `valence.json.gz`, `arousal.json.gz` | `booster.dump_model()` of the two LightGBM regressors (raw score = prediction) |
| `classifier.txt.gz`, `valence.txt.gz`, `arousal.txt.gz` | the same three models in LightGBM's text format, for Python (`EmotionPredictor`) |
| `meta.json` | `feature_version`, `model_id` (`emotion-v1@<sha8>`, what `Affect.model` carries), `input_features` (exact order), `emotion_feature_names`, `states` (id, display, valence, arousal, in class order), `temperature`, `layers` (feature → explanation layer), training summary and metrics |
| `golden.json` | ~20 input vectors with Python's raw scores, temperature-softmax probabilities, valence/arousal predictions and `pred_contrib`, plus golden windows (tick channels → the 31 features) and golden frames (blendshapes + matrix → per-frame cues) |

Inference:

- `p = softmax(raw / T)` over the 10 classes.
- **Map position (served):** `valence = Σ_k p_k · states[k].valence`, `arousal = Σ_k p_k · states[k].arousal`, the
  probability-weighted anchors. On held-out simulated sessions this beats the regressors on MAE and CCC for both
  axes (the simulator's within-state spread carries little observable signal), and the point always agrees with
  the displayed probabilities.
- Regressors (exported for comparison and for retraining on real taps later):
  `valence = clip(valence_raw, −1, 1)`, `arousal = clip(arousal_raw, −1, 1)`.
- `pred_contrib` is LightGBM's path-dependent TreeSHAP (per class for the classifier: `n + 1` values per class, the
  last being the bias). Golden vectors carry `valence_mix` / `arousal_mix` (map position) and the regressor values.

The optional `Prediction.affect` field (`valence/contracts.py` `Affect`) carries the result to the console:
`valence`, `arousal`, `label`, `label_display`, `confidence`, `probabilities` (10 states) and `model`.

## 6. Self-report sessions (real-data evaluation)

One JSON line per tap, appended by the console's capture; **never media, never a transcript**:

```json
{"t": 1761321600.0, "session_id": "sr-2026-10-24-a", "person_id": 1, "valence": 0.3, "arousal": -0.2,
 "feature_version": "emotion-v1", "features": {"face_now": 0.12, "brow_now": 0.04, "...": null}}
```

- `t`: wall-clock seconds; `valence`, `arousal` in −1..1 (the tap on the grid).
- `features`: the emotion model's input vector at that moment, either an object keyed by name (`null` = NaN) or a
  list in `meta.json` → `input_features` order. Optional `label` (a state id the person picked) is scored too.
- Evaluated by `python -m valence.eval.self_report <files.jsonl>` (MAE, CCC, quadrant agreement, nearest-anchor
  state agreement, per person).

**Session protocol** (what makes the taps usable):

1. Consent first: the person knows that only the feature numbers and their taps are kept, in the browser's memory
   until they press Download, and that they can stop and clear at any time.
2. Sit at the desk for ≥ 1 minute before the first tap, looking at the screen normally: the face cues' neutral
   baseline is the first 20 s of visible face, and the window needs a minute to warm up.
3. Tap every ~2 minutes (the console reminds), from how you feel *right now*: left–right = unpleasant–pleasant,
   down–up = calm/tired–activated. Optionally pick a state name.
4. Aim for ≥ 30 minutes and some variety (a hard task, a break, a chat, a boring stretch); the evaluator needs
   ≥ 3 taps with some spread for CCC, and a constant mood only tests the mean.
5. Download the JSONL and run the evaluator; commit the file only if the person agrees (features and taps only).

## 7. Simulator assumptions for the new cues (`valence/sim/params.py`)

The training data is simulated. These numbers **are** the hypothesis the emotion models are trained and evaluated
under; none of them was measured on people. Values are episode means relative to the person's neutral
(`FACE_CUES`); blinks per minute of visible face; away = share of time with `|yaw|` or `|pitch|` > 20°.

| State | brow | eye | frown | smile | blinks/min | away | pitch (°) | pitch when away |
|---|---|---|---|---|---|---|---|---|
| Focused | +0.06 | +0.02 | +0.01 | 0 | 7 | 0.05 | −2 | — |
| Frustrated_Masked | +0.15 | −0.02 | +0.06 | 0 | 16 | 0.15 | −3 | — |
| Exhausted | 0 | −0.15 | +0.03 | 0 | 24 | 0.20 | −10 | −10 more |
| Collaborative_Happy | −0.03 | −0.05 | 0 | +0.10 | 18 | 0.35 | +2 | — |
| Overwhelmed | +0.12 | −0.05 | +0.10 | 0 | 22 | 0.25 | −12 | −10 more |
| Tense | +0.20 | +0.03 | +0.14 | 0 | 20 | 0.20 | 0 | — |
| Bored | 0 | −0.06 | +0.04 | 0 | 16 | 0.55 | −8 | −18 more (phone) |
| Confused | +0.22 | −0.03 | +0.03 | 0 | 11 | 0.15 | +1 | — |
| Calm | −0.02 | −0.03 | 0 | +0.06 | 13 | 0.25 | +2 | — |
| Anxious | +0.05 | +0.05 | +0.07 | 0 | 26 | 0.35 | −2 | — |

On top of the table (`FaceCueNuisance`):

- smile also follows positive face valence (+0.35 × max(face, 0)) and frown negative face valence
  (+0.25 × max(−face, 0)), so a masked episode's forced smile shows in `smile` while its brows stay lowered;
- each episode draws its true valence/arousal offset (N(0, 0.10) each); +0.25 × the valence offset goes to smile,
  −0.15 × to frown; +0.10 × the arousal offset to eye openness and ×(1 + arousal offset) to the blink rate;
- spread: episode means N(0, 0.08) (blendshape cues), pitch N(0, 4°), blink rate × lognormal(0, 0.3), away share
  ± N(0, 0.10); per-tick noise 0.06 (frown 0.045) and 4° pitch, growing below 200 lux like `face_valence`; each
  person's neutral baseline is off by N(0, 0.06) and N(0, 4°); resting blink rates differ by × U(0.6, 1.6);
- head pose is simulated per frame (10 frames per tick: the tick's share of looking-away frames, away frames at
  |yaw| U(18°, 45°) and the state's extra away pitch, the others near the screen with 4° yaw and 2° pitch jitter around
  the tick's pitch) and aggregated with the live tick rule (§2), so `head_yaw`, `head_pitch` and `head_away` agree
  the way they do live: a head drooping past −20° counts as looking away whatever the state;
- cues lag a state change like the other behaviors (the person's 1–2 min lag), and a hidden face gives NaN cues and
  no blinks.

The cues come from their own random stream (`crc32(session_id + "/face-cues")`), so every existing channel of every
scenario and corpus is unchanged (`tests/test_emotion_features.py` checks hashes against the pre-event simulator).
