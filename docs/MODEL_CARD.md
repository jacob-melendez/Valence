# Model card: Valence M2 (served) and its baselines

> **Every accuracy result in this card comes from simulated data.** Simulator v2 generated it under the behavioral
> assumptions in §3 and [DATASET.md](DATASET.md). The results test the architecture. They say nothing about
> accuracy on real people. The one real-data check (the negative control, §5) fails. Claim wording: *"Under explicit, published behavioral assumptions, a temporal context
> model beats a face snapshot; in-the-wild validation is future work."*

Sources: `reports/metrics.json` (results), `artifacts/model_meta.json` (training run) and the code cited in each row,
as committed at `4e8290f`. Regenerate with `make train && make eval`. A new `artifacts/m2.txt` means a new version
id and new numbers.

## 1. Model details

| Item | Value | Source |
|---|---|---|
| Served model | **M2** = context history + face history. M0 (face snapshot) runs every tick as a shadow model | `valence/models/served.py` |
| Algorithm | LightGBM gradient-boosted trees, `objective=multiclass`, softmax with a fitted temperature | `valence/models/train_ladder.py` |
| Version id | `m2_lgbm@<sha8>`: the first 8 hex characters of the SHA-256 of `artifacts/m2.txt`. Exposed as `served` in `GET /api/models`, as `model` in `/api/health` and as `feature_version` in `hello`. Current: **`m2_lgbm@620f8a67`** | `served.py` |
| Trained | `2026-10-17 23:26:53` (`trained_at`) | `artifacts/model_meta.json` |
| Training run | simulator v2, 300 sessions, seed 7, `noise_scale` 1.0 (untuned), state set `core` | `model_meta.json`, `make train` |
| Classes | Focused, Frustrated_Masked, Exhausted, Collaborative_Happy | `valence/contracts.py` `CORE_STATES` |
| Input | 114 features from one person's rolling window plus the room's window: 5 s ticks, up to 180 ticks (15 min) | `valence/features/compute.py` |
| Output | calibrated probabilities, label, TreeSHAP attributions grouped into 4 layers, face-only reading, mask-override flag | `served.py`, `predictor.py` |
| Hyperparameters | learning rate 0.05, 31 leaves, `min_data_in_leaf` 100, feature and bagging fraction 0.8, at most 800 rounds, early stop after 50, class weights n/(k·count), seed 7 | `train_ladder.py` `PARAMS` |
| Rounds / temperature (current artifact) | M2 407 rounds, T = 1.012. M0 96 rounds, T = 0.871 | `model_meta.json` `summary` |
| Calibration | one temperature per model, fitted by NLL on the calibration half (bounds 0.25–8) | `fit_temperature` |
| Runtime | `predict` and `pred_contrib` with `num_threads=1`, because the torch + LightGBM libomp combination segfaults otherwise | `served.py`, [BENCHMARKS.md](BENCHMARKS.md) |
| License | MIT | `LICENSE` |

**Features (114, order in `model_meta.json` `feature_names`).** Missing values stay NaN, and LightGBM handles
them natively.

| Group | Features | Count | Explanation layer |
|---|---|---|---|
| Room levels: `db`, `lux` | `_now`; `_mean/_std/_slope/_max` over 1, 5 and 15 min; `_ewm_1m/_5m`; `_d1m15m` (1-min mean minus 15-min mean); `_rel_15m` (now minus the 15-min median) | 2 × 17 | Ambient Physics |
| Person levels: `motion`, `slump`, `hih` | same 17 statistics per channel | 3 × 17 | Somatic & Social |
| Events: `fist` (person), `laugh`, `compl` (room) | `_cnt_now`, `_cnt_1m/_5m/_15m`, `_since_s` (capped at 900 s) | 3 × 5 | Somatic & Social |
| Social, time, fill | `withdraw_now`, `withdraw_frac_5m`, `n_people`; `hours_since_break`, `tod_sin`, `tod_cos`; `fill_ratio` | 7 | S&S; Time Dynamics; `fill_ratio` is never narrated |
| Face | `face_now`, `face_visible_now`, `face_mean_1m/_5m/_15m`, `face_std_5m`, `face_visible_frac_5m` | 7 | Facial Surface |

**Model ladder** (feature sets in `compute.py` `FEATURE_SETS`):

| Model | Inputs | # features | Role |
|---|---|---|---|
| M0 | `face_now`, `face_visible_now` | 2 | face-snapshot baseline; served as the shadow model |
| M0t | the 7 face features | 7 | face history |
| M1 | 9 context `_now` values, `n_people`, `hours_since_break`, `tod_sin`, `tod_cos` | 13 | context snapshot |
| M2-noface | all context features, no face | 107 | ablation: how much the face adds |
| M2-noabsdb | M2 minus the 9 absolute dB levels (`db_now`, `_mean`, `_max`, `_ewm`) | 105 | ablation: per-session mic offset |
| M2-notime | M2 minus `hours_since_break`, `tod_sin`, `tod_cos` | 111 | ablation: operator-settable clock |
| **M2** | all | 114 | **served** |
| M3 | GRU(64) + additive attention over 15 raw channels × 180 ticks | — | benchmark only, run on cloud CPU (`valence/models/temporal.py`) |

**Serving path per tick.** (1) Compute the features. (2) Clip each feature to the training p1–p99
(`artifacts/clip_bounds.json`) and list the clipped features as `data_quality.ood_features`. (3) Compute
softmax(z/T). If more than 3 features were clipped, use p ← 0.7·p + 0.3/k. (4) Run M0 when `face_now` is not NaN.
(5) Compute TreeSHAP for the argmax class. (6) Smooth with `PersonSmoother`: an EMA (α 0.5) feeds the *label* only,
and a new label needs 2 consecutive ticks. The displayed confidence is the calibrated, unsmoothed probability of the
shown label, documented as "sim-calibrated" (`contracts.py`). `warming_up` is set for the first 60 s of a window, and the override is
blocked for the first 120 s.

**6-state variant (P1: + Overwhelmed, Tense).** Trained separately into `artifacts/all/` (2026-10-17 23:31:36; M0,
M2-noface, M2; same corpus settings). Frustrated_Masked becomes mask-only (`mask_p` 0.92) and Tense takes unmasked
frustration. **Not yet evaluated** with `evaluate.py`; the trainer's own mixed-length test macro-F1
(`artifacts/all/model_meta.json`) is M0 0.3341, M2-noface 0.6936, M2 0.7861 (the face is what separates Tense from
Frustrated_Masked). Serve with `VALENCE_STATE_SET=all VALENCE_ARTIFACTS_DIR=artifacts/all` (id now `m2_lgbm@1caba6f3`).

## 2. Intended use

| | |
|---|---|
| **Intended** | A research prototype for **consented self-monitoring and personal wellbeing outside workplace and education settings**. It also demonstrates explainable, on-device, features-only state estimation in the hackathon demo, on the builder and volunteers who agree at the table ([PRD.md](PRD.md) §1.4). |
| **Users** | The builder, and judges watching the demo. Replay scenarios show `source = replay`. |
| **Regulatory position** | EU AI Act Art. 5(1)(f) prohibits AI systems that infer the emotions of natural persons **in the workplace and in education institutions**. Only medical and safety reasons are exempt, and **consent does not lift the prohibition**. Outside those settings, emotion recognition is high-risk (Annex III 1(c)) and carries a duty to inform people (Art. 50(3)). Room-only mode (`VALENCE_ROOM_ONLY=1` or `POST /api/mode`; a `--room-only` CLI flag is not built) is the configuration to start from in any compliance review. Perception still runs per person in it, and it does not make workplace or classroom use lawful. |

| Out of scope (do not use for) | Why |
|---|---|
| Emotion recognition in workplaces or schools: monitoring, proctoring, productivity or engagement scoring | Prohibited by Art. 5(1)(f) regardless of consent; also a PRD non-goal |
| Hiring, grading, performance review, discipline, or any decision about a person | Trained only on simulated data and never validated on people; the labels are not ground truth |
| Surveillance, identity or face recognition, tracking people across sessions | Non-goal. There is no identity embedding, ids are ephemeral, and baselines are forgotten when an id is released |
| Clinical, diagnostic or mental-health screening | The states are not diagnoses, and nothing here is clinically validated |
| Inferring demographics or appearance | Non-goal; the narrator validator rejects such language |
| Attributing speech or laughter to one person | Audio is a room-scope signal (§7) |

## 3. Training data

Simulator v2 (`valence/sim/generator.py`): sessions of 60–120 min at 5 s ticks, one room, 1–3 people (p 0.4 / 0.4 /
0.2). Each person has a hidden semi-Markov state with Gamma(shape 3, mean `dwell_min`) dwell times (2-min floor, no
self-transitions). The next state is weighted by `max(0.05, 1 + fatigue_affinity·f(h)) × group/solo affinity`, with
`f(h) = 1 / (1 + exp(−(h − 3) / 0.5))` and h = hours since break. Room dB and light lag the occupants' targets, behavior
lags each person's own state, and events are Poisson counts per tick. Each tick also has a room row (`person_id 0`).
The full spec, the reasons the original mock was rebuilt, and the scenarios: **[DATASET.md](DATASET.md)**.
The tables below are the simulator at commit `4e8290f`, which trained `m2_lgbm@620f8a67`. Rescaling it to the
live-capture measurements ([CONTRACT_REQUESTS.md](CONTRACT_REQUESTS.md) round 3) is not in that commit.

**Per-state behavioral parameters** (`valence/sim/params.py` `STATE_PARAMS`; P1 states are used only with
`state_set=all`):

| Parameter | Focused | Frustrated_Masked | Exhausted | Collaborative_Happy | Overwhelmed (P1) | Tense (P1) |
|---|---|---|---|---|---|---|
| `face_mu`, episode-mean valence | −0.15 | −0.20 | −0.10 | +0.55 | −0.30 | −0.45 |
| `mask_p`, probability of a forced smile (`mask_mu` +0.45) | 0 | 0.70; 0.92 when Tense is in the set (`params_for`) | 0 | 0 | 0 | 0 |
| `db_target`, dB SPL (calibrated mic) | 48 | 66 | 45 | 63 | 66 | 70 |
| `lux_mult`, × session light baseline | 1.00 | 0.85 | 0.65 | 1.15 | 0.85 | 0.95 |
| `fist_rate`, onsets/min (person) | 0 | 0.8 | 0 | 0 | 0.2 | 0.8 |
| `laugh_rate`, onsets/min into the room | 0.03 | 0.10 | 0.02 | 0.50 | 0.02 | 0.05 |
| `compl_rate`, compliments/min into the room | 0.02 | 0.02 | 0.01 | 0.20 | 0.01 | 0.01 |
| `slump_p`, share of time slumped * | 0.15 | 0.35 | 0.70 | 0.05 | 0.60 | 0.20 |
| `hih_p`, share of time head in hands | 0.20 | 0.25 | 0.40 | 0.10 (+0.35 on laughter ticks) | 0.70 | 0.10 |
| `motion`, shoulder-widths/s * | 0.08 | 0.20 | 0.04 | 0.18 | 0.10 | 0.30 |
| `face_hidden_extra` | 0 | 0 | 0 | 0 | 0.55 → hidden share drawn U(0.4, 0.7) per episode | 0 |
| `dwell_min`, mean episode length (min) | 9 | 6 | 8 | 6 | 5 | 5 |
| `fatigue_affinity` | −0.6 | +1.0 | +2.0 | −0.5 | +1.5 | +1.0 |
| `group_affinity` / `solo_affinity` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.8 / 0.4 | 1.0 / 1.0 | 1.5 / 0.6 |
| Break-rate multiplier (`generator.py`) | ×1 | ×1 | ×2 | ×1 | ×2 | ×1 |

\* Provisional scales. Live capture run 1 (one person, built-in camera) measured motion about 3× below these
(typing p50 0.027 sw/s, still 0.014) and a baseline-relative neutral face near 0, not −0.15
([BENCHMARKS.md](BENCHMARKS.md), [CONTRACT_REQUESTS.md](CONTRACT_REQUESTS.md) round 3). Not applied to this simulator.

**Session nuisance** (`params.py` `Nuisance`; ×ns marks terms that scale with `noise_scale`, which is 1.0 here):

| Parameter | Value | ×ns |
|---|---|---|
| `mic_offset_db` | U(−12, +12) dB per session | |
| `db_noise` | σ 4 dB per tick (p90 spread σ 2 dB; +2 dB per laughter onset, at most 2) | ✓ |
| `lux_range` | log-uniform 120–600 lux per session; ×0.85 from 22:00 to 06:00; lognormal tick noise σ 0.08 | noise ✓ |
| `face_hidden_range` | session share U(0.05, 0.40), + 0.3 × head-in-hands fraction, cap 0.98 (Overwhelmed: the per-episode share instead) | |
| `face_episode_sd` / `face_tick_sd` | 0.20 / 0.15 | tick ✓ |
| `low_light_lux` | 200: below it, face σ gains + 0.35 × (200 − lux)/200 | |
| `dropout` | 0.02 per tick, per modality (audio per room, video per person) | |
| `room_tau_min` / `person_tau_min` | U(2, 4) / U(1, 2) min lag | ✓ |
| motion noise | lognormal σ 0.4; ×0.7 while slumped (`generator.py`) | ✓ |
| Fist bursts | multiplier alternates 1.7 / 0.3, switch probability 0.1 per tick (mean ×1.0) (`generator.py`) | |
| `break_per_min` / `break_len_min` | 1/120 per min / U(10, 20) min; after a break: Focused alone, or Focused / Collaborative_Happy 50/50 with ≥ 2 people | |
| `session_min`, `start_hour`, `start_hours_since_break` | U(60, 120) min, U(9, 23) h, U(0, 5) h | |
| Withdrawal (derived, not a parameter) | slump ≥ 0.6 of the tick and motion ≤ 0.06 sw/s; the live aggregator uses the same rule (`valence/features/spec.py`) | |

**Detector false positives (`params.py` `FP`): PLACEHOLDERS** in this simulator. Capture run 1 measured 0.0 fist
onsets/min and 0% slump during 3 min of typing ([BENCHMARKS.md](BENCHMARKS.md)); the keyboard-gaze, slump and audio
segments are being re-recorded. `make negative` (≤ 20% Frustrated/Tense on typing) has run and **fails** (§5).

| `fist_per_min` | `slump_share` | `laugh_per_min` (room) | `compl_per_min` (room) |
|---|---|---|---|
| 0.05 | 0.05 | 0.02 | 0.01 |

**Training examples** (`valence/features/build_dataset.py`). One window ends every 6 ticks per person, with a random
length of 1–180 ticks. The room window is the full 180 ticks 70% of the time and otherwise matches the person
window. The label is the person's `true_state` at the window's last tick. Splits are by session, from a seeded
permutation (seed 7): 70% train, 7.5% early stopping, 7.5% calibration, 15% test.

## 4. Evaluation data and protocol (`valence/eval/evaluate.py`)

| Aspect | Protocol |
|---|---|
| Test set | 45 held-out simulator sessions, rebuilt from the same seed and split. They are never used for training, early stopping or calibration. Scenarios are never part of any split. |
| Headline set | Windows every 3 ticks with up to 15 min of history (`fixed_len=180`; shorter only where the person's data starts later). Reports macro-F1, balanced accuracy, per-class recall, episode age and calibration. |
| History truncation | Windows every 6 ticks, with the person and room windows both cut to exactly 1 / 12 / 24 / 60 / 120 / 180 ticks (0 / 1 / 2 / 5 / 10 / 15 min). Windows with less history are dropped. |
| Bootstrap CIs | 200 resamples of test sessions with replacement; 2.5th–97.5th percentile of macro-F1. Computed only for the history curve. |
| Focused misread as negative | Share of true-Focused windows predicted Frustrated_Masked or Exhausted (the "Focused paradox") |
| Episode age | Accuracy by seconds since the person entered the current true state |
| Calibration | ECE over 10 equal-width bins of max probability, before (T = 1) and after temperature scaling |
| Robustness | Degrades **only the face channel** at tick level; lux and context are untouched. Face dropout hides an extra share p of person-ticks. "Low light" adds face noise σ and an extra hidden share: relative light 0.5 → 0.15 / 0.10, 0.25 → 0.30 / 0.25, 0.1 → 0.45 / 0.45. |
| Legacy mock | The original generator, 2,000 rows: train on the first 1,400 rows, test accuracy on the other 600, LightGBM 100 rounds |
| Negative control (real data) | `valence/eval/negative_control.py`: replay a features-only typing capture through the served model; pass if Frustrated_Masked/Tense ≤ 20% of non-warm-up ticks |
| **Not done** | The `noise_scale` sensitivity sweep (×0.5 / ×1.5; `tune_difficulty.py` not built), labeled live or in-the-wild data, the acted set, self-report labels |

## 5. Metrics (simulated data, `reports/metrics.json`)

> **Which model these numbers describe.** `reports/metrics.json` was generated at 2026-10-17 23:18:00 for the
> **previous artifact, `m2_lgbm@f69de179`** (trained 22:51:51, T = 0.975). That artifact was trained on the
> simulator *before* commit `5a75413`. The served artifact `m2_lgbm@620f8a67` has **no `evaluate.py` run yet**:
> the first attempt crashed on the M3 entry (fixed in `4e8290f`) and `make eval` has not been re-run since. Re-run
> it and refresh this section. The committed figures in `reports/figs/` match `metrics.json`; figures from a
> partial run must not be mixed with it. The only score for `620f8a67` is the trainer's own test macro-F1 on
> mixed-length windows (`model_meta.json` `summary`): M2 0.8163, M0 0.3671. That protocol differs from the
> tables below. Simulator differences between the two artifacts:
>
> | Simulator item | Evaluated artifact (f69de179) | Commit `4e8290f` and served artifact (620f8a67) |
> |---|---|---|
> | Break length | U(5, 15) min | U(10, 20) min |
> | Fist burst multiplier | 2.5 / 0.3 (mean ×1.4) | 1.7 / 0.3 (mean ×1.0) |
> | Face hidden share | session + 0.6 × head-in-hands (+0.55 Overwhelmed), cap 0.98 | session + 0.3 × head-in-hands; Overwhelmed U(0.4, 0.7) per episode |

**Ladder, headline set** (45 test sessions, up to 15 min of history; values exactly as in `metrics.json`):

| Model | Macro-F1 | Balanced acc. | Recall Focused | Recall Frustrated_Masked | Recall Exhausted | Recall Collab./Happy | Focused misread as negative |
|---|---|---|---|---|---|---|---|
| M0 | 0.3733 | 0.4018 | 0.4890 | **0.0478** | 0.5091 | 0.5613 | **0.4761** |
| M0t | 0.4633 | 0.4920 | 0.5359 | 0.1458 | 0.5138 | 0.7727 | 0.4445 |
| M1 | 0.5914 | 0.5925 | 0.5885 | 0.5470 | 0.6811 | 0.5536 | 0.2427 |
| M2-noface | 0.7796 | 0.7812 | 0.8077 | 0.7619 | 0.8380 | 0.7174 | 0.1240 |
| M2-noabsdb | 0.8233 | 0.8253 | 0.8582 | 0.7559 | 0.8585 | 0.8285 | 0.1209 |
| M2-notime | 0.8102 | 0.8117 | 0.8476 | 0.7487 | 0.8504 | 0.8002 | 0.1342 |
| **M2** | **0.8217** | 0.8237 | 0.8588 | **0.7546** | 0.8558 | 0.8257 | **0.1208** |

Headline: Frustrated_Masked recall goes from 0.0478 (M0) to 0.7546 (M2), and "Focused misread as negative" from
0.4761 to 0.1208. M2's 0.82 macro-F1 is below the plan's 0.85–0.92 difficulty band. It is reported as is, not
tuned toward the band.

**Ablations** (no CIs on this set; the history curve's 95% intervals at 15 min span about ±0.013):

| Comparison | Δ macro-F1 vs M2 | Reading |
|---|---|---|
| M2-noface | −0.0421 | The face adds about 0.04. The plan's check (M2-noface ≥ M2 − 0.05) holds. |
| M2-noabsdb | +0.0016 | Dropping the 9 absolute dB levels does not hurt. The remaining features (relative and trend dB among them) carry the signal, which fits the ±12 dB per-session mic offset. |
| M2-notime | −0.0115 | The time layer (including the operator-settable `hours_since_break`) contributes a little. Frustrated_Masked recall drops 0.7546 → 0.7487, and Focused misread rises 0.1208 → 0.1342. |

**Accuracy vs history** (macro-F1, session-bootstrap 95% CI):

| History | M0 | M0t | M1 | M2-noface | **M2** |
|---|---|---|---|---|---|
| 0 min | 0.3777 (0.3620–0.3926) | 0.3588 (0.3459–0.3704) | 0.5880 (0.5695–0.6058) | 0.5750 (0.5602–0.5915) | 0.6215 (0.6078–0.6356) |
| 1 min | 0.3773 (0.3606–0.3927) | 0.4531 (0.4344–0.4729) | 0.5867 (0.5678–0.6043) | 0.7199 (0.7060–0.7333) | 0.7646 (0.7535–0.7731) |
| 2 min | 0.3757 (0.3590–0.3915) | 0.4605 (0.4405–0.4824) | 0.5848 (0.5655–0.6019) | 0.7638 (0.7507–0.7771) | 0.8041 (0.7917–0.8142) |
| 5 min | 0.3759 (0.3589–0.3913) | 0.4599 (0.4383–0.4818) | 0.5833 (0.5632–0.5986) | 0.7761 (0.7634–0.7866) | 0.8184 (0.8058–0.8290) |
| 10 min | 0.3758 (0.3577–0.3924) | 0.4652 (0.4394–0.4918) | 0.5823 (0.5614–0.6014) | 0.7739 (0.7606–0.7863) | 0.8175 (0.8045–0.8286) |
| 15 min | 0.3765 (0.3581–0.3932) | 0.4658 (0.4359–0.4938) | 0.5851 (0.5640–0.6032) | 0.7767 (0.7621–0.7902) | 0.8226 (0.8080–0.8341) |

At 15 min the intervals of M2, M1 and M0 do not overlap. With one tick of history (0 min), M2 (0.6215) is only
0.03 above M1.

**Accuracy by episode age** (share correct, headline set). The first 30 s of an episode are weak for the context
models: context has not caught up yet, and M0 beats both M1 and M2 there.

| Time in current state | n | M0 | M1 | M2 |
|---|---|---|---|---|
| 0–30 s | 1779 | **0.3513** | 0.2389 | **0.2698** |
| 30–60 s | 2102 | 0.3716 | 0.3321 | 0.5005 |
| 1–2 min | 4135 | 0.3927 | 0.4675 | 0.7291 |
| 2–5 min | 9419 | 0.4148 | 0.6468 | 0.9119 |
| 5+ min | 10125 | 0.4375 | 0.7368 | 0.9563 |

**Robustness, face channel only** (macro-F1, full windows; lux and context untouched):

| Extra face dropout | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| M0 | 0.3755 | 0.3467 | 0.2988 | 0.2335 | 0.1236 |
| M0t | 0.4632 | 0.4602 | 0.3920 | 0.3448 | 0.1236 |
| M2-noface | 0.7811 | 0.7811 | 0.7811 | 0.7811 | 0.7811 |
| **M2** | 0.8215 | 0.8180 | 0.8142 | 0.8017 | **0.7643** |

| Relative light (face noise + hiding) | 1.0 | 0.5 | 0.25 | 0.1 |
|---|---|---|---|---|
| M0 | 0.3755 | 0.3539 | 0.3140 | 0.2668 |
| M0t | 0.4632 | 0.4517 | 0.3784 | 0.3334 |
| M2-noface | 0.7811 | 0.7811 | 0.7811 | 0.7811 |
| **M2** | 0.8215 | 0.8115 | 0.7863 | **0.7528** |

M2 degrades gradually where M0 and M0t collapse. However, M2 falls **below M2-noface** with the face fully hidden
(0.7643 vs 0.7811) and at the lowest light level (0.7528 vs 0.7811). Degraded face evidence costs M2 more than
having no face model at all (§8).

**Calibration (M2).** T = 0.9755; ECE 0.0067 before and 0.0031 after temperature scaling. In every reliability
bin from 0.3 to 1.0, accuracy is within 0.02 of confidence. The 0.2–0.3 bin has n = 9.

**M3 (GRU + attention, benchmark).** No current result. Its results (`reports/m3_metrics.json`, trained on the
pre-`5a75413` simulator) were removed in `4e8290f` as stale until M3 is retrained on the current corpus. M2 stays
the served model.

**Negative control (real data, not simulated).** One person, 3 min of quiet typing (`data/real/capture_typing.jsonl`,
0 fist onsets, 0% slump). Gate: Frustrated_Masked/Tense ≤ 20% of ticks. On `f69de179`: Frustrated_Masked 49%,
Exhausted 35%, Collaborative 16%, Focused 0%: **fail** ([CONTRACT_REQUESTS.md](CONTRACT_REQUESTS.md) round 3). On the
served `620f8a67` (cloud re-run at `4e8290f` of `python -m valence.eval.negative_control
data/real/capture_typing.jsonl --expect Focused`, 71 ticks): Frustrated_Masked 63.4%, Exhausted 35.2%, Focused
1.4%: **fail**. The
drivers were absolute motion and face level: real motion and neutral-face valence sit far below the simulator's
Focused values (§3 footnote). Re-run with `make negative` after retraining on a rescaled simulator.

**Legacy mock (why the data was rebuilt).** On the original i.i.d. generator (600 test rows), accuracy is
face-only 0.6667, context-only 1.0000 and both 1.0000. The context columns encode the label, so that 100% says
nothing about the thesis.

## 6. Explainability (as built)

| Element | Implementation |
|---|---|
| Attributions | LightGBM TreeSHAP (`predict(pred_contrib=True)`), one row, raw-score (log-odds) space, computed on the clipped vector. `tests/test_served.py::test_shap_additivity` checks that contributions sum to the raw score (atol 1e-6). |
| Class explained | The per-tick M2 argmax. The shown label comes from the smoother, so during the 2-tick hysteresis the layers can explain a different class than the one displayed. |
| Layers | SHAP summed per layer: **Ambient Physics** (`db_`, `lux_`), **Time Dynamics** (`hours_since_break`, `tod_*`), **Somatic & Social** (motion, slump, head-in-hands, fists, laughter, compliments, withdrawal, `n_people`), **Facial Surface** (`face_*`). Sorted by contribution. `fill_ratio` is excluded (`valence/explain/explainer.py`). |
| Drivers | Per layer, sorted by abs(SHAP), keeping **one driver per sensor channel** (so `motion_now` and `motion_ewm_1m` never both appear), at most 3 per layer. Each driver carries a display string with its only number, a `level` (high / elevated / normal / low, fixed thresholds in `describe.py`), an `effect` (supports / opposes, from the sign) and a `scope` (room / person). |
| Template text | Uses the top 3 supporting drivers. The face layer is excluded when the override is on or no face is visible. The text adds an out-of-range note when more than 3 features were clipped (`valence/explain/template.py`). |
| **Mask override, as built** | Raw condition, evaluated only on ticks with a readable face: M0 reads **Collaborative_Happy**, M2 reads a non-positive state, M0 confidence ≥ 0.45, M2 confidence ≥ 0.70, and the summed face-layer SHAP toward M0's class inside M2 is > 0. Smoother: a tick counts only if M2's raw label equals the shown label. The override turns **on at ≥ 3 of the last 6 face-visible ticks** and **off at ≤ 1**. A tick without a face keeps the current state. The history resets when the shown label changes, and the override is blocked for the first 120 s of the window. Effect: the core-set display changes from "Frustrated" to "Frustrated / Masked Stress", and the template states that context corrected the facial mask. |
| Override tests | `tests/test_served.py` asserts the override fires within 30 ticks of the masked segment in `frustrated_masked` and never in `focused`, `collaborative_happy` or `exhausted`. The plan's target was 6 ticks; the test allows 30. Passed with `620f8a67` at commit `4e8290f` (cloud, 2026-10-17); re-run after any retrain. |
| Counterfactual | `GET /api/explain/{event_id}/counterfactual?feature=&value=` re-runs M2 with one feature changed and tags it in `simulated_fields` |
| Narrator | An optional local LLM rephrases the fact sheet behind a deterministic validator and falls back to the template. It is outside the scope of this card. |

## 7. Ethical considerations and limitations

| Topic | Limitation | Mitigation or status |
|---|---|---|
| **Simulated data, circularity risk** | The simulator encodes the hypothesis (context lags the state, the face is ambiguous), so a context model can win partly by construction. | `noise_scale` stays at 1.0, **untuned**: no knob was turned toward the result, and the one knob scales face and context noise alike. The face gets a built-in head start, because it changes at once while context lags 1–4 min; the episode-age table reports this. The ablations (M2-noface, -noabsdb, -notime) show which inputs carry the result. On the original i.i.d. mock, context-only scores 1.0 accuracy, a perfect score that proves nothing; that is why v2 uses overlapping distributions. M0 and M1 are reported wherever they land. **Not done:** the sensitivity sweep and the live-vs-sim variance check. |
| **Not validated in the wild** | No labeled real data exists. The acted set and self-reports have not been collected, the detector false-positive rates are placeholders, and the one real-data check, the negative control on quiet typing, **fails** (§5): a calm person can read as Frustrated. | Treat outputs as a demonstration of the pipeline, not a measurement of a person. |
| **Upstream detector fairness** | This repo does **not measure** demographic performance for MediaPipe FaceLandmarker (blendshapes → face valence), GestureRecognizer (fists), YAMNet via MediaPipe AudioClassifier (laughter), Ultralytics YOLO26n-pose (people, posture, motion) or Vosk small-en-us 0.15 (compliments; English, restricted grammar). | Fairness evaluations, where they exist, are in the vendors' model cards: the MediaPipe model cards linked from each task's page in the MediaPipe Solutions docs, the Ultralytics docs, and the YAMNet model page. The anti-bias claim is narrowed to "decision stability when face evidence is missing or degraded" (§5), and demographic validation is an explicit gap. |
| **Naive face valence** | Fixed blendshape weights (`mouthSmile` +1.0, `mouthFrown` −0.7, `browDown` −0.5, `noseSneer` −0.3, `mouthPress` −0.2), relative to the person's first 20 s of visible face. Posed and felt smiles read the same, by design (`valence/perception/face_valence.py`). Expression norms differ between people, and the simulator assumes an exact neutral baseline. | The face is one of four layers; the face-only degradations are in §5. |
| **Context lag** | Room dB and light lag 2–4 min; behavior lags 1–2 min. In the first 30 s of a new episode, M2 is right 0.2698 of the time (M0 0.3513). `warming_up` covers only the first 60 s of a window, not a state change in the middle of one. | The shown label also waits for 2-tick hysteresis. Read transitions with caution. |
| **Operator-set fields** | `hours_since_break` can be reset (`POST /api/events/break`) or seeded (`/api/demo/seed`). Lux can be overridden (`/api/sim/lux`), events injected (`/api/sim/event`), and face occlusion or low light overlaid (`/api/sim/{overlay}`). | Every one of these is tagged in `simulated_fields` and audited. M2-notime measures how much the model relies on the clock (§5), and the counterfactual endpoint shows the effect on a single prediction. |
| **Room-scope audio** | dB, laughter and compliments are shared by everyone present, so in a mixed room each person sees the same in-between level. Live, laughter and compliments count only at ≥ room median + 6 dB (near-field gate, `settings.near_field_db`). | Drivers carry `scope: room`, and narration must say "the room". Nothing is attributed to one person. |
| **Light as a proxy** | Room light is linked to state (×0.65–×1.15) as a stand-in for confounds such as late-night work in dim rooms. A person's state does not change the lights. | The link is kept weak, and lux shifts are reported separately from face degradation. |
| **Room-only mode** | Applies the person-trained M2 to the room aggregate row. `build_dataset.py` trains on person rows only. | Not evaluated. |
| **Breaks inside training windows** | `build_dataset.py` groups by person id and does not split windows at breaks. Live, the window slot is keyed by display id: a returning person who gets the same Person #N (the smallest free id, `valence/perception/ids.py`) continues the old slot; one who gets another id starts cold. | Train/live mismatch only in the second case; not measured. |
| **Calibration** | Fitted on simulated data. | The contract calls the confidence "sim-calibrated"; the UI should say so. |

## 8. Caveats and recommendations

1. Read every number as "under the published simulator assumptions"; quote the claim wording above and nothing stronger.
2. Before trusting live output, finish the live captures (keyboard gaze, slump and audio are being re-recorded),
   rescale `FP`, motion, slump and face levels to them, retrain with the same seed, and pass `make negative`
   (it fails today, §5).
3. When the face is mostly missing or very noisy, M2 scores below M2-noface (§5). A fallback to M2-noface based on
   `face_visible_frac_5m` would fix this. **Not built.**
4. Ignore the first minute after a state change; always show `warming_up`, `source`, `simulated_fields`, "sim-calibrated".
5. Run the `noise_scale` sweep (×0.5 / ×1.5) before claiming that the model ordering holds beyond `noise_scale` 1.0.
6. Evaluate `artifacts/all` before demoing Overwhelmed or Tense as anything more than a trained variant.
7. Retrain M3 on the current simulator before comparing it with M2 (§5).
8. Start any compliance review from room-only mode (`POST /api/mode`), which is not compliance; never deploy in
   workplaces or schools.
