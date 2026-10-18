# Model card: Valence M2 (served) and its baselines

> **Every accuracy result in this card comes from simulated data.** Simulator v2 generated it under the behavioral
> assumptions in §3 and [DATASET.md](DATASET.md). The results test the architecture. They say nothing about
> accuracy on real people. The one real-data check (the negative control, §5) is in-sample. Claim wording:
> *"Under explicit, published behavioral assumptions, a temporal context model beats a face snapshot; in-the-wild
> validation is future work."*

Sources: `reports/metrics.json` (results, generated 2026-10-18 00:33:13, M3 merged in), `reports/m3_metrics.json`
(M3), `artifacts/model_meta.json` (training run) and the code cited in each row. Regenerate with `make train`,
`python -m valence.models.temporal`, then `make eval`. A new `artifacts/m2.txt` means a new version id and new
numbers.

## 1. Model details

| Item | Value | Source |
|---|---|---|
| Served model | **M2** = context history + face history. M0 (face snapshot) runs every tick as a shadow model | `valence/models/served.py` |
| Algorithm | LightGBM gradient-boosted trees, `objective=multiclass`, softmax with a fitted temperature | `valence/models/train_ladder.py` |
| Version id | `m2_lgbm@<sha8>`: the first 8 hex characters of the SHA-256 of `artifacts/m2.txt`. Exposed as `served` in `GET /api/models`, as `model` in `/api/health` and as `feature_version` in `hello`. Current: **`m2_lgbm@85874e97`** | `served.py` |
| Trained | `2026-10-18 00:15:34` (`trained_at`), on the simulator refit to live capture run 1 (commit `e0ad762`), with modality dropout (§3) | `artifacts/model_meta.json` |
| Training run | simulator v2, 300 sessions, seed 7, `noise_scale` 1.0 (untuned), state set `core`; M2-family face models get modality-dropout copies | `model_meta.json`, `make train`, `train_ladder.py` `face_dropout` |
| Classes | Focused, Frustrated_Masked, Exhausted, Collaborative_Happy | `valence/contracts.py` `CORE_STATES` |
| Input | 114 features computed from one person's rolling window plus the room's window (5 s ticks, up to 180 ticks = 15 min); M2 uses 110 of them. Served models read their feature lists from `model_meta.json` `feature_sets` | `valence/features/compute.py`, `served.py` |
| Output | calibrated probabilities, label, TreeSHAP attributions grouped into 4 layers, face-only reading, mask-override flag | `served.py`, `predictor.py` |
| Hyperparameters | learning rate 0.05, 31 leaves, `min_data_in_leaf` 100, feature and bagging fraction 0.8, at most 800 rounds, early stop after 50, class weights n/(k·count), seed 7 | `train_ladder.py` `PARAMS` |
| Rounds / temperature | M2 377 rounds, T = 0.954. M0 109 rounds, T = 0.903 | `model_meta.json` `summary` |
| Calibration | one temperature per model, fitted by NLL on the calibration half (bounds 0.25–8) | `fit_temperature` |
| Runtime | `predict` and `pred_contrib` with `num_threads=1`, because the torch + LightGBM libomp combination segfaults otherwise | `served.py`, [BENCHMARKS.md](BENCHMARKS.md) |
| License | MIT | `LICENSE` |

**Features (114, order in `model_meta.json` `feature_names`).** Missing values stay NaN, and LightGBM handles
them natively.

| Group | Features | Count | Explanation layer |
|---|---|---|---|
| Room levels: `db`, `lux` | `_now`; `_mean/_std/_slope/_max` over 1, 5 and 15 min; `_ewm_1m/_5m`; `_d1m15m` (1-min mean minus 15-min mean); `_rel_15m` (now minus the 15-min median). `_slope_1m` and `_std_1m` are computed but used by no model in the ladder (`NOISY_LIVE`: dominated by live transients) | 2 × 17 | Ambient Physics |
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
| M2-noface | context features minus the 4 `NOISY_LIVE` ones, no face | 103 | ablation: how much the face adds |
| M2-noabsdb † | M2 minus the 9 absolute dB levels (`db_now`, `_mean`, `_max`, `_ewm`) | 101 | ablation: per-session mic offset |
| M2-notime † | M2 minus `hours_since_break`, `tod_sin`, `tod_cos` | 107 | ablation: operator-settable clock |
| **M2** † | all 114 minus the 4 `NOISY_LIVE` | 110 | **served** |
| M3 | GRU(64) + additive attention over 15 raw channels × 180 ticks | — | benchmark only (`valence/models/temporal.py`); §5 |

† Trained with modality dropout (§3): the M2-family models whose features include the face block. M0, M0t, M1,
M2-noface and M3 are trained without it.

**Serving path per tick.** (1) Compute the features. (2) Clip each feature to the training p1–p99
(`artifacts/clip_bounds.json`) and list the clipped features as `data_quality.ood_features`. (3) Compute
softmax(z/T). If more than 3 features were clipped, use p ← 0.7·p + 0.3/k. (4) Run M0 when `face_now` is not NaN.
(5) Compute TreeSHAP for the argmax class. (6) Smooth with `PersonSmoother`: an EMA (α 0.5) feeds the *label* only,
and a new label needs 2 consecutive ticks. The displayed confidence is the calibrated, unsmoothed probability of the
shown label, documented as "sim-calibrated" (`contracts.py`). `warming_up` is set for the first 60 s of a window,
and the override is blocked for the first 120 s.

**6-state variant (P1: + Overwhelmed, Tense).** `artifacts/all/` (`m2_lgbm@4a4d8135`, `trained_at` 2026-10-18
00:23:40, on the refit simulator with modality dropout, same 110-feature M2, 247 rounds, T = 0.960). Frustrated_Masked
becomes mask-only (`mask_p` 0.92) and Tense takes unmasked frustration. **Not evaluated** with `evaluate.py` and not
negative-controlled; the trainer's own mixed-length test macro-F1 (`artifacts/all/model_meta.json` `summary`, simulated)
is M0 0.4614, M2-noface 0.5732, M2 0.7682. Serve with `VALENCE_STATE_SET=all VALENCE_ARTIFACTS_DIR=artifacts/all`.

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
Full spec and scenarios: **[DATASET.md](DATASET.md)**. Motion, face and dB/lux noise scales were refit in `e0ad762`
to live capture run 1 (one person at the demo Mac; live-vs-sim comparison in `reports/capture_vs_sim.json`).

**Per-state behavioral parameters** (`valence/sim/params.py` `STATE_PARAMS`; P1 states only with `state_set=all`):

| Parameter | Focused | Frustrated_Masked | Exhausted | Collaborative_Happy | Overwhelmed (P1) | Tense (P1) |
|---|---|---|---|---|---|---|
| `face_mu`, episode-mean valence (0 = personal neutral) | 0.00 | −0.12 | −0.04 | +0.50 | −0.20 | −0.35 |
| `mask_p`, probability of a forced smile (`mask_mu` +0.45) | 0 | 0.70; 0.92 when Tense is in the set (`params_for`) | 0 | 0 | 0 | 0 |
| `db_target`, dB SPL (calibrated mic) | 48 | 66 | 45 | 63 | 66 | 70 |
| `lux_mult`, × session light baseline | 1.00 | 0.85 | 0.65 | 1.15 | 0.85 | 0.95 |
| `fist_rate`, onsets/min (person) | 0 | 0.8 | 0 | 0 | 0.2 | 0.8 |
| `laugh_rate`, onsets/min into the room * | 0.03 | 0.10 | 0.02 | 0.50 | 0.02 | 0.05 |
| `compl_rate`, compliments/min into the room * | 0.02 | 0.02 | 0.01 | 0.20 | 0.01 | 0.01 |
| `slump_p`, share of time slumped * | 0.05 | 0.30 | 0.70 | 0.03 | 0.60 | 0.20 |
| `hih_p`, share of time head in hands * | 0.08 | 0.15 | 0.35 | 0.05 (+0.35 on laughter ticks) | 0.70 | 0.08 |
| `motion`, shoulder-widths/s (live typing p50 0.027) | 0.03 | 0.05 | 0.015 | 0.06 | 0.03 | 0.09 |
| `face_hidden_extra` | 0 | 0 | 0 | 0 | 0.55 → hidden share drawn U(0.4, 0.7) per episode | 0 |
| `dwell_min`, mean episode length (min) | 9 | 6 | 8 | 6 | 5 | 5 |
| `fatigue_affinity` | −0.6 | +1.0 | +2.0 | −0.5 | +1.5 | +1.0 |
| `group_affinity` / `solo_affinity` | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.8 / 0.4 | 1.0 / 1.0 | 1.5 / 0.6 |
| Break-rate multiplier (`generator.py`) | ×1 | ×1 | ×2 | ×1 | ×2 | ×1 |

\* Not yet checked against live data: the keyboard-gaze, slump and audio captures are still wanted
([CONTRACT_RESPONSES.md](CONTRACT_RESPONSES.md) round 4).

**Session nuisance** (`params.py` `Nuisance`; ×ns marks terms that scale with `noise_scale`, which is 1.0 here):

| Parameter | Value | ×ns |
|---|---|---|
| `mic_offset_db` | U(−12, +12) dB per session | |
| dB jitter / drift | σ 1.2 dB per tick; AR(1) drift (×0.95 per tick, σ 0.5 dB); p90 = level + 1.5 + abs(N(0, 1)); +2 dB per laughter onset (at most 2) | ✓ |
| Sound transients | 0.3 per min, +4 to +12 dB (30%: −0.5× that) for 2–8 ticks, state-independent | |
| Lux | log-uniform 120–600 lux per session; ×0.85 from 22:00 to 06:00; lognormal tick jitter σ 0.015; AR(1) log drift (×0.98, σ 0.01) | jitter, drift ✓ |
| Light transients | 0.5 per min, ×U(0.75, 1.3) for 2–12 ticks (auto-exposure), state-independent | |
| `face_hidden_range` | session share U(0, 0.30), + 0.3 × head-in-hands fraction, cap 0.98 (Overwhelmed: the per-episode share instead) | |
| `face_episode_sd` / `face_tick_sd` | 0.12 / 0.03; readable share of a tick's frames U(0.85, 1.0) | tick ✓ |
| `low_light_lux` | 200: below it, face σ gains + 0.35 × (200 − lux)/200 | |
| `dropout` | 0.02 per tick, per modality (audio per room, video per person) | |
| `room_tau_min` / `person_tau_min` | U(2, 4) / U(1, 2) min lag | ✓ |
| Motion | log-AR(1) toward the state mean (pull 0.25 per tick, innovation σ 0.18); ×0.7 while slumped; still phases in every state (share 0.4, switch 0.06, ×0.5 motion) | innovation ✓ |
| Slump value | slumped: U(0.7, 1.0) of the tick; otherwise 0, or U(0, 0.1) on 15% of ticks (`generator.py`) | |
| Fist bursts | multiplier alternates 1.7 / 0.3, switch probability 0.1 per tick (mean ×1.0) (`generator.py`) | |
| `break_per_min` / `break_len_min` | 1/120 per min / U(10, 20) min; after a break: Focused alone, or Focused / Collaborative_Happy 50/50 with ≥ 2 people | |
| `session_min`, `start_hour`, `start_hours_since_break` | U(60, 120) min, U(9, 23) h, U(0, 5) h | |
| Withdrawal (derived, not a parameter) | slump ≥ 0.6 of the tick and motion ≤ 0.02 sw/s; the live aggregator uses the same rule (`valence/features/spec.py`) | |

**Detector false positives (`params.py` `FP`).** Capture run 1 measured 0.0 fist onsets/min and 0% slump during
3 min of typing, and 0 false onsets around 15 cued clenches ([BENCHMARKS.md](BENCHMARKS.md)); small non-zero rates
are kept so one event is never certain. Laughter and compliment rates are not yet measured.

| `fist_per_min` | `slump_share` | `laugh_per_min` (room) | `compl_per_min` (room) |
|---|---|---|---|
| 0.02 | 0.02 | 0.02 | 0.01 |

**Training examples** (`valence/features/build_dataset.py`). One window ends every 6 ticks per person, with a random
length of 1–180 ticks. The room window is the full 180 ticks 70% of the time and otherwise matches the person
window. The label is the person's `true_state` at the window's last tick. Splits are by session, from a seeded
permutation (seed 7): 70% train, 7.5% early stopping, 7.5% calibration, 15% test.

**Modality dropout** (`valence/models/train_ladder.py` `face_dropout`, seed 7; only the † models above). Without
it, a context + face model rarely sees a hidden or noisy face and leans on the face. The training split gets two
sets of extra copies of its own windows, with labels kept:

| Copies | Share of training windows | Change to the face block |
|---|---|---|
| Face hidden | 30% | `face_now`, `face_mean_1m/_5m/_15m`, `face_std_5m` → NaN; `face_visible_now`, `face_visible_frac_5m` → 0 |
| Degraded, as in low light | 20% | `face_now` and the three face means + N(0, 0.25), clipped to [−1, 1]; `face_std_5m` + 0.2; both visibility features × one U(0.3, 0.8) draw per copy |

Class weights are computed on the augmented set. Early-stopping, calibration and test windows are never augmented.

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
| **Not done** | The `noise_scale` sensitivity sweep (×0.5 / ×1.5; `tune_difficulty.py` not built), labeled live or in-the-wild data, an independent negative-control capture, the acted set, self-report labels |

## 5. Metrics (simulated data, `reports/metrics.json`)

These numbers evaluate the served artifact `m2_lgbm@85874e97` (`calibration_m2.temperature` 0.954 matches
`model_meta.json`). The seven figures in `reports/figs/` carry the same 2026-10-18 00:33 timestamp as `metrics.json`.

**Ladder, headline set** (45 test sessions, up to 15 min of history; values exactly as in `metrics.json`):

| Model | Macro-F1 | Balanced acc. | Recall Focused | Recall Frustrated_Masked | Recall Exhausted | Recall Collab./Happy | Focused misread as negative |
|---|---|---|---|---|---|---|---|
| M0 | 0.4255 | 0.4464 | 0.4356 | **0.1169** | 0.5993 | 0.6338 | **0.5587** |
| M0t | 0.4910 | 0.5181 | 0.4964 | 0.1810 | 0.6107 | 0.7844 | 0.4961 |
| M1 | 0.5631 | 0.5651 | 0.5681 | 0.4432 | 0.6949 | 0.5542 | 0.2598 |
| M2-noface | 0.6939 | 0.6956 | 0.6647 | 0.6738 | 0.7796 | 0.6642 | 0.1927 |
| M2-noabsdb † | 0.7992 | 0.8006 | 0.8066 | 0.7014 | 0.8515 | 0.8427 | 0.1817 |
| M2-notime † | 0.7861 | 0.7876 | 0.7795 | 0.6868 | 0.8432 | 0.8408 | 0.2137 |
| **M2** † | **0.8021** | 0.8031 | 0.8014 | **0.7082** | 0.8575 | 0.8452 | **0.1882** |

Headline: Frustrated_Masked recall goes from 0.1169 (M0) to 0.7082 (M2), and "Focused misread as negative" from
0.5587 to 0.1882. M2's 0.80 macro-F1 is below the plan's 0.85–0.92 difficulty band. It is reported as is, not
tuned toward the band. M3 is not on this set (its mixed-history score is below).

**Ablations** (no CIs on this set; the history curve's 95% interval for M2 at 15 min spans about −0.015 / +0.018):

| Comparison | Δ macro-F1 vs M2 | Reading |
|---|---|---|
| M2-noface | −0.1082 | The face adds about 0.11. The plan's check (M2-noface ≥ M2 − 0.05) **fails** on this simulator. |
| M2-noabsdb | −0.0029 | Dropping the 9 absolute dB levels barely matters. The remaining features (relative and trend dB among them) carry the signal, which fits the ±12 dB per-session mic offset. |
| M2-notime | −0.0160 | The time layer (including the operator-settable `hours_since_break`) contributes a little. Frustrated_Masked recall drops 0.7082 → 0.6868, and Focused misread rises 0.1882 → 0.2137. |

**Accuracy vs history** (macro-F1, session-bootstrap 95% CI; M3 is one run without bootstrap):

| History | M0 | M0t | M1 | M2-noface | **M2** | M3 |
|---|---|---|---|---|---|---|
| 0 min | 0.4244 (0.4062–0.4439) | 0.3629 (0.3451–0.3807) | 0.5611 (0.5410–0.5815) | 0.5363 (0.5200–0.5499) | 0.6407 (0.6222–0.6558) | 0.6089 |
| 1 min | 0.4241 (0.4053–0.4435) | 0.4714 (0.4486–0.4957) | 0.5601 (0.5401–0.5806) | 0.6296 (0.6142–0.6470) | 0.7425 (0.7228–0.7604) | 0.7248 |
| 2 min | 0.4232 (0.4043–0.4417) | 0.4849 (0.4621–0.5090) | 0.5583 (0.5387–0.5792) | 0.6684 (0.6550–0.6860) | 0.7787 (0.7623–0.7948) | 0.7499 |
| 5 min | 0.4229 (0.4035–0.4414) | 0.4872 (0.4644–0.5084) | 0.5578 (0.5390–0.5785) | 0.6852 (0.6657–0.7052) | 0.7945 (0.7772–0.8125) | 0.7632 |
| 10 min | 0.4250 (0.4062–0.4430) | 0.4924 (0.4685–0.5160) | 0.5576 (0.5387–0.5780) | 0.6815 (0.6640–0.7013) | 0.7896 (0.7742–0.8049) | 0.7547 |
| 15 min | 0.4246 (0.4059–0.4442) | 0.4922 (0.4684–0.5125) | 0.5599 (0.5419–0.5813) | 0.6784 (0.6593–0.6998) | 0.7888 (0.7736–0.8068) | 0.6992 |

At 15 min the intervals of M2, M1 and M0 do not overlap. With one tick of history (0 min), M2 (0.6407) is 0.08
above M1.

**Accuracy by episode age** (share correct, headline set). The first 30 s of an episode are weak for the context
models: context has not caught up yet, and M0 beats both M1 and M2 there.

| Time in current state | n | M0 | M1 | M2 |
|---|---|---|---|---|
| 0–30 s | 1663 | **0.4402** | 0.2044 | **0.3620** |
| 30–60 s | 1971 | 0.4257 | 0.2714 | 0.4972 |
| 1–2 min | 3887 | 0.4494 | 0.4021 | 0.6771 |
| 2–5 min | 9179 | 0.4484 | 0.6224 | 0.8658 |
| 5+ min | 9795 | 0.4630 | 0.7270 | 0.9385 |

**Robustness, face channel only** (macro-F1, full windows; lux and context untouched; no CIs):

| Extra face dropout | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| M0 | 0.4357 | 0.3902 | 0.3270 | 0.2484 | 0.1208 |
| M0t | 0.4934 | 0.4733 | 0.4565 | 0.4218 | 0.0819 |
| M2-noface | 0.6933 | 0.6933 | 0.6933 | 0.6933 | 0.6933 |
| **M2** | 0.8025 | 0.7946 | 0.7852 | 0.7708 | **0.6870** |

| Relative light (face noise + hiding) | 1.0 | 0.5 | 0.25 | 0.1 |
|---|---|---|---|---|
| M0 | 0.4357 | 0.3807 | 0.3062 | 0.2539 |
| M0t | 0.4934 | 0.4536 | 0.3418 | 0.2886 |
| M2-noface | 0.6933 | 0.6933 | 0.6933 | 0.6933 |
| **M2** | 0.8025 | 0.7702 | 0.7165 | **0.6770** |

M2 degrades gradually where M0 and M0t collapse. Modality dropout was added so that M2 does not fall below
M2-noface when the face disappears. With the face **fully hidden** it is now 0.0063 below M2-noface (0.6870 vs
0.6933), a near tie (this table has no CIs). At relative light 0.25 it stays above M2-noface (0.7165 vs 0.6933,
+0.0232). At 0.1 it is still below (0.6770 vs 0.6933, −0.0163): very noisy face evidence still costs a little more
than no face at all (§8). The training degradation resembles these test conditions (at 1.0 the hidden copies match
the test input exactly), so part of this gain is by construction (§7).

**Calibration (M2).** T = 0.9541; ECE 0.0079 before and 0.0063 after temperature scaling. In every reliability
bin from 0.4 to 1.0, accuracy is within 0.01 of confidence; the 0.3–0.4 bin (n = 310) is within 0.016. The
0.2–0.3 bin has n = 5.

**M3 (GRU + attention, benchmark; `reports/m3_metrics.json`).** Retrained on the current corpus and session split,
on CPU, without modality dropout: 29,777 training windows (step 12, random lengths), at most 15 epochs, early stop on
macro-F1 with patience 3, 343 s. Macro-F1 on mixed-history test windows 0.7532 (early-stopping set 0.7592); the
nearest M2 figure is the trainer's own mixed-length test score, 0.7965 (`model_meta.json` `summary`). On the history
windows above, M3 is below M2 at every length. Its curve peaks at 5 min (0.7632) and then drops to 0.6992 at 15 min,
which suggests it is under-trained on long sequences within this budget; treat it as a lower bound. M2 stays the
served model (its TreeSHAP attributions drive the explanations).

**Negative control (real data, not simulated).** One person, 3 min of quiet typing (`data/real/capture_typing.jsonl`,
0 fist onsets, 0% slump). Gate: Frustrated_Masked/Tense ≤ 20% of ticks. Before the refit (`f69de179`):
Frustrated_Masked 49%, Exhausted 35%, Collaborative 16%, Focused 0%, **fail**
([CONTRACT_REQUESTS.md](CONTRACT_REQUESTS.md) round 3). Served `85874e97`: **Focused on 100%** of the 25 non-warm-up
ticks, stress share 0, 0 mask overrides, **pass** (cloud, 2026-10-18: `python -m valence.eval.negative_control
data/real/capture_typing.jsonl`). The simulator was refit to this same capture run, so the pass is **in-sample**: a
consistency check, not validation.

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
| Override tests | `tests/test_served.py` asserts the override fires within 30 ticks of the masked segment in `frustrated_masked` and never in `focused`, `collaborative_happy` or `exhausted`. The plan's target was 6 ticks; the test allows 30. Passed with `85874e97` (cloud, 2026-10-18; in a replay the override first fired 8 ticks into the masked segment); re-run after any retrain. |
| Counterfactual | `GET /api/explain/{event_id}/counterfactual?feature=&value=` re-runs M2 on the person's current window with one feature changed and tags it in `simulated_fields` |
| Narrator | Optional local LLM (Qwen3-1.7B 4-bit, MLX, 127.0.0.1:8081). The fact sheet is validated strictly against the contract (`extra="forbid"`, else HTTP 422). Each draft must pass a deterministic, driver-bound validator: every number is a sheet number and, when bound to a driver, that driver's own; at least 2 of the top-3 supporting drivers are named and nothing outside the top-5; no other state, demographic or unmeasured signal. Up to 3 attempts within a 2.5 s deadline, else the template. The fixed system + few-shot prefix per state set is prefilled once into a prompt cache (`valence/narrator/engine.py`, `validator.py`, `prompt.py`; `c61bed2`). Otherwise outside the scope of this card. |

## 7. Ethical considerations and limitations

| Topic | Limitation | Mitigation or status |
|---|---|---|
| **Simulated data, circularity risk** | The simulator encodes the hypothesis (context lags the state, the face is ambiguous), so a context model can win partly by construction. | `noise_scale` stays at 1.0, **untuned**: no knob was turned toward the headline, and the one knob scales face and context noise alike. The `e0ad762` refit changed scales to match live measurements, not results. The face gets a built-in head start, because it changes at once while context lags 1–4 min; the episode-age table reports this. The ablations show which inputs carry the result. On the original i.i.d. mock, context-only scores 1.0 accuracy, a perfect score that proves nothing; that is why v2 uses overlapping distributions. M0 and M1 are reported wherever they land. Modality dropout trains on hidden and degraded faces much like the robustness tests apply them, so the proof-3 gain is partly by construction. **Not done:** the sensitivity sweep. |
| **Not validated in the wild** | No labeled real data exists. The acted set and self-reports have not been collected. The negative control passes only on the capture the simulator was refit to, and laughter, compliment, slump and head-in-hands scales are unchecked against live data. | Treat outputs as a demonstration of the pipeline, not a measurement of a person. |
| **Upstream detector fairness** | This repo does **not measure** demographic performance for MediaPipe FaceLandmarker (blendshapes → face valence), GestureRecognizer (fists), YAMNet via MediaPipe AudioClassifier (laughter), Ultralytics YOLO26n-pose (people, posture, motion) or Vosk small-en-us 0.15 (compliments; English, restricted grammar). | Fairness evaluations, where they exist, are in the vendors' model cards: the MediaPipe model cards linked from each task's page in the MediaPipe Solutions docs, the Ultralytics docs, and the YAMNet model page. The anti-bias claim is narrowed to "decision stability when face evidence is missing or degraded" (§5), and demographic validation is an explicit gap. |
| **Naive face valence** | Fixed blendshape weights (`mouthSmile` +1.0, `mouthFrown` −0.7, `browDown` −0.5, `noseSneer` −0.3, `mouthPress` −0.2), relative to the person's first 20 s of visible face. Posed and felt smiles read the same, by design (`valence/perception/face_valence.py`). Expression norms differ between people, and the simulator assumes an exact neutral baseline. | The face is one of four layers; the face-only degradations are in §5. |
| **Context lag** | Room dB and light lag 2–4 min; behavior lags 1–2 min. In the first 30 s of a new episode, M2 is right 0.3620 of the time (M0 0.4402). `warming_up` covers only the first 60 s of a window, not a state change in the middle of one. | The shown label also waits for 2-tick hysteresis. Read transitions with caution. |
| **Operator-set fields** | `hours_since_break` can be reset (`POST /api/events/break`) or seeded (`/api/demo/seed`). Lux can be overridden (`/api/sim/lux`), events injected (`/api/sim/event`), and face occlusion or low light overlaid (`/api/sim/{overlay}`). | Every one of these is tagged in `simulated_fields` and audited. M2-notime measures how much the model relies on the clock (§5), and the counterfactual endpoint shows the effect on a single prediction. |
| **Room-scope audio** | dB, laughter and compliments are shared by everyone present, so in a mixed room each person sees the same in-between level. Live, laughter and compliments count only at ≥ room median + 6 dB (near-field gate, `settings.near_field_db`). | Drivers carry `scope: room`, and narration must say "the room". Nothing is attributed to one person. |
| **Light as a proxy** | Room light is linked to state (×0.65–×1.15) as a stand-in for confounds such as late-night work in dim rooms. A person's state does not change the lights. | The link is kept weak, and lux shifts are reported separately from face degradation. |
| **Room-only mode** | Applies the person-trained M2 to the room aggregate row. `build_dataset.py` trains on person rows only. | Not evaluated. |
| **Breaks inside training windows** | `build_dataset.py` groups by person id and does not split windows at breaks. Live, the window slot is keyed by display id: a returning person who gets the same Person #N (the smallest free id, `valence/perception/ids.py`) continues the old slot; one who gets another id starts cold. | Train/live mismatch only in the second case; not measured. |
| **Calibration** | Fitted on simulated data. | The contract calls the confidence "sim-calibrated"; the UI should say so. |

## 8. Caveats and recommendations

1. Read every number as "under the published simulator assumptions"; quote the claim wording above and nothing stronger.
2. Before trusting live output, record the missing captures (keyboard gaze, slump, audio), check those scales,
   and pass `make negative` on a capture the simulator was **not** fit to.
3. Modality dropout brought M2 with no face to within 0.0063 of M2-noface (0.6870 vs 0.6933), but at relative
   light 0.1 it is still 0.0163 below (§5). A fallback to M2-noface based on `face_visible_frac_5m` would close that. **Not built.**
4. Ignore the first minute after a state change; always show `warming_up`, `source`, `simulated_fields`, "sim-calibrated".
5. Run the `noise_scale` sweep (×0.5 / ×1.5) before claiming that the model ordering holds beyond `noise_scale` 1.0.
6. Evaluate `artifacts/all` (and run its negative control) before demoing Overwhelmed or Tense as anything more
   than a trained variant.
7. Treat M3 as a lower bound: its curve drops at 15 min (§5). Train it longer on full-length windows before
   comparing it with M2 as an equal.
8. Start any compliance review from room-only mode (`POST /api/mode`), which is not compliance; never deploy in
   workplaces or schools.
