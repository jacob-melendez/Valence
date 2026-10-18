# Valence dataset specification (simulator v2)

This document replaces the original "Mock Dataset" script. It is the specification that
`valence/sim/generator.py` implements, together with `valence/sim/params.py` (parameter table),
`valence/sim/scenarios.py` (scripted replays), `valence/sim/legacy.py` (the original generator, kept for
comparison) and `valence/sim/tune_difficulty.py` (difficulty tuning; not written yet). Where code and this
document disagree, this document is the target and the code is changed; the known differences are listed
below.

What this document is and is not:

- Every parameter below is a **published behavioral assumption**. The evaluation tests the thesis *under these
  assumptions*. It is not evidence about real people.
- **No model result appears here.** Accuracy, F1 and calibration numbers are produced later by
  `valence/eval/evaluate.py` into `reports/metrics.json` and reported in `docs/RESULTS.md`.
- Detector false-positive rates, and the motion and slump scales, started as **placeholders**. Live capture run 1
  on the demo Mac replaced them (`e0ad762`, section 10), and capture runs 2–3 checked them without a refit
  (section 11.3). Capture numbers are in `docs/BENCHMARKS.md`.
- Field names follow `valence/contracts.py` (contract version 1.0.0) exactly.

**Spec vs code: resolved.** The differences found on 2026-10-17 were settled as follows (also recorded in the
"Build-time decisions" table of `docs/PLAN.md`); the models in `artifacts/` are trained on this version:

| Topic | Decision (now in code and in this spec) |
|---|---|
| Break onset rate | 1 per 120 min (×2 in Exhausted/Overwhelmed). Kept below the plan's ~1/90 so long stretches without a break occur and the fatigue tilt (~3 h) matters. |
| Break length | U(10, 20) min, so every simulated break is also a break under the live 10-minute absence rule. |
| Fist bursts | burst multiplier switches symmetrically between 1.7 and 0.3 (mean ×1.0), so the per-state rates are the true long-run rates. |
| Overwhelmed face hiding | each Overwhelmed episode draws its own face-hidden share U(0.40, 0.70) (hands over the face); it replaces the session share instead of stacking on it. |
| Head-in-hands while laughing | Collaborative_Happy head-in-hands share rises by +0.35 on ticks with laughter (people cover their faces when they laugh). |
| Difficulty | `noise_scale` = 1.0, **untuned**: criterion A (§12.2) already holds (served M2 macro-F1 0.797 ≤ 0.92, `reports/metrics.json`), so no tuning toward the results was done. The tuning script in §12.2 is not built. |
| Relative features | `*_rel_15m` = current value minus its 15-minute median (window-only, so train/live parity is exact). |
| Capture refit (`e0ad762`) | Face, motion, slump, head-in-hands, dB/lux noise, face-hidden and false-positive values were refit to live capture run 1. The as-built values are in `valence/sim/params.py` and `docs/MODEL_CARD.md` §3; sections 7, 9, 10 and Appendix A below show these as-built values (§10 keeps the replaced placeholders in parentheses). |
| Venue and room baseline (`bbd46a4`, `docs/CONTRACT_REQUESTS.md` round 4) | Each session draws a hall level U(40, 75) dB, independent of state, power-summed with the team's own level, and the arrival baseline an operator would capture (`room_db_base`, `room_lux_base`; sections 4.1, 9.2, 9.3); the lux range widened to 100–800. The served models read the room relative to that baseline: 9 room-relative features (`db_room_*`, `lux_room_*`), 123 features computed, 101 used by M2, no absolute dB or lux level. |

## Contents

1. [At a glance](#1-at-a-glance)
2. [Why the original mock was rebuilt](#2-why-the-original-mock-was-rebuilt)
3. [Unit of data](#3-unit-of-data)
4. [Output schema](#4-output-schema)
5. [Legacy compatibility](#5-legacy-compatibility)
6. [Hidden state process](#6-hidden-state-process)
7. [Per-state parameters](#7-per-state-parameters)
8. [State sets: core 4 and the P1 switch](#8-state-sets-core-4-and-the-p1-switch)
9. [Observation model](#9-observation-model)
10. [Detector false positives (set from the live capture)](#10-detector-false-positives-set-from-the-live-capture)
11. [Units and sim-to-real consistency](#11-units-and-sim-to-real-consistency)
12. [Difficulty and anti-circularity](#12-difficulty-and-anti-circularity)
13. [Scenarios](#13-scenarios)
14. [Splits and training windows](#14-splits-and-training-windows)
15. [Files, reproducibility and privacy](#15-files-reproducibility-and-privacy)
16. [Planned verification](#16-planned-verification)
17. [Limitations](#17-limitations)
18. [Appendix A: full parameter table](#appendix-a-full-parameter-table)

---

## 1. At a glance

| Item | Specification |
|---|---|
| Unit of data | A session: one room, 1–3 people, 60–120 min, one row per present person per 5 s tick, plus one room row per tick |
| Corpus | ~300 sessions, ≈ 500k person-ticks (estimate from the parameters; exact count written to the manifest) |
| Columns | All `TickRecord` fields (`contracts.TICK_FIELDS`) + hidden `true_state`, `episode_age_s` |
| Hidden state | Per person, semi-Markov over the active state set (4 core states, or 6 with P1) |
| Context | Room sound and light lag the occupants' states; events are sparse Poisson counts; sensors are noisy, offset and sometimes missing |
| Face | Valence relative to the person's neutral, per-state mean, masking for `Frustrated_Masked`; missing on a per-session base of 0–30% of ticks (more during head-in-hands; 40–70% in `Overwhelmed`) |
| Difficulty | One `noise_scale`, tuned by a rule that never looks at the headline comparison (section 12) |
| Splits | By session, 70 / 15 / 15; validation split again into an early-stop half and a calibration half |
| Storage | `data/sim/*.parquet` (gitignored), regenerated deterministically from seeds logged in `artifacts/sim_manifest.json` (planned; today the trainer regenerates sessions in memory from the seed and session count in `artifacts/model_meta.json`) |
| Real data | `data/real/*.jsonl`, features only (no images, audio, transcripts or keypoints) |

---

## 2. Why the original mock was rebuilt

The original script (`generate_valence_mock_data`, 500 rows, 5 s apart) picked a state with `random.choice` for
**every row** and then drew each column from a state-specific uniform range:

| Original column | Focused | Frustrated_Masked | Exhausted | Collaborative_Happy |
|---|---|---|---|---|
| `ambient_db` | U(40, 55) | U(68, 85) | U(35, 50) | U(60, 75) |
| `light_lux` | U(350, 500) | U(150, 300) | U(100, 200) | U(400, 600) |
| `hours_since_break` | U(1.0, 3.0) | U(3.5, 6.0) | U(5.0, 8.0) | U(0.2, 1.5) |
| `clenched_fists_count` (per 5 s row) | 0 | randint(1, 4) | 0 | 0 |
| `compliments_count` (per 5 s row) | 0 | 0 | 0 | randint(1, 3) |
| `social_withdrawal_flag` | 0 | 0 or 1 | 1 | 0 |
| `facial_valence_snapshot` | U(−0.4, 0.0) | U(0.4, 0.8) | U(−0.2, 0.1) | U(0.5, 0.9) |

Problems, and what v2 does instead:

| # | Problem in the original | Consequence | v2 design |
|---|---|---|---|
| 1 | **Rows are i.i.d.** The state is redrawn every 5 s, and `hours_since_break` jumps between unrelated values 5 s apart. | A 15-minute window carries no information about the current row, so the thesis ("context over time beats a face snapshot") cannot be tested. | Sessions with a hidden semi-Markov state per person (episodes of minutes), a running break clock, and context that lags the state (sections 6, 9.1). |
| 2 | **Classes are trivially separable.** The count columns encode the label: fists ≥ 1 only in Frustrated_Masked, compliments ≥ 1 only in Collaborative_Happy, withdrawal = 1 with no fists only in Exhausted. | Three if-statements label every row correctly by construction. A ~100% score proves nothing, and judges see through it. | Overlapping state distributions, sparse Poisson events, room channels shared by everyone present, per-session offsets, detector false positives. |
| 3 | **No laughter.** The PRD lists laughter as a Somatic & Social signal (and treated it as a keyword); the mock has no column for it. | One of the PRD's own signals is untestable. | `laughter_count` as an audio event (YAMNet live), room scope, near-field gated live. |
| 4 | **No noise, no missing modalities, no multi-person data.** The face is present on every row; one row per tick with no person or session id. | The anti-bias proof (face hidden or poorly lit) cannot be tested; without session ids any split leaks. | Face missing on 0–30% of ticks (per-session base), extra face noise below 200 lux, 2% per-modality dropout, 1–3 people, session and person ids. |
| 5 | **Unrealistic per-tick fist counts.** randint(1, 4) per 5 s row is 12–48 fists per minute, sustained. | A single row is diagnostic. | ~0.8 fist onsets per minute, bursty. Even in a Frustrated_Masked episode a single 5 s tick holds an onset only ~6% of the time (Poisson arithmetic, before burstiness). |
| 6 | **No sim-to-real link.** No units, absolute dB with no microphone calibration, no posture/head-in-hands/motion columns, no detector errors (typing never looks like a fist). | A model trained on the mock cannot be run on live features. | Every channel in the edge's units (section 11), a ±12 dB per-session mic offset, false-positive rates measured from a live capture (section 10), and the same `TickRecord` for simulation, replay and live. |

The original generator is kept as `valence/sim/legacy.py` (`generate_legacy(num_records=500, seed=0)`, a seeded
port) for the "why we rebuilt the data" slide. The slide's accuracy figure is **TBD — produced by
`valence/sim/legacy.py` with `valence/eval/evaluate.py`**.

---

## 3. Unit of data

| Level | Specification |
|---|---|
| Tick | 5 s (`settings.tick_s`). Per-minute rates become per-tick Poisson counts (rate ÷ 12). |
| Person | Ephemeral id 1..3 within a session, `display_id` "Person #N". No row while the person is on a break. |
| Room | One row per tick with `person_id = 0`, `display_id = "Room"` (section 4.2), emitted even when nobody is present. |
| Session | One room, one continuous recording. Length U(60, 120) min. People: 1, 2 or 3 with probability 0.4 / 0.4 / 0.2. Start time of day U(09:00, 23:00). Each person starts U(0, 5) h since their last break. |
| Corpus | ~300 sessions. Expected size from these parameters: 300 sessions × 1.8 people × ~1,080 ticks × ~0.86 presence (breaks) ≈ 500k person-ticks, plus ≈ 324k room rows. The generator writes the exact counts to the manifest. |
| Training example | A window of 1–180 ticks ending at one person-tick (section 14). |

---

## 4. Output schema

### 4.1 Columns

Column order is `contracts.TICK_FIELDS` followed by the two hidden columns. Floats use NaN for missing values in
Parquet (`TickRecord.to_dict()` turns NaN into `null` for JSON).

| Field | Type | Scope | Unit / range | How the simulator fills it |
|---|---|---|---|---|
| `tick` | int | — | 0-based tick index in the session | counter |
| `ts` | float | — | s | tick × 5 from the session start (the simulator has no real epoch; `tod_h` carries the time of day) |
| `t_mono` | float | — | s | same as `ts` |
| `session_id` | str | — | — | `s{seed}-{i:04d}` (corpus), `scenario-{name}` (replay) |
| `person_id` | int | — | 1..3; 0 = room row | fixed per person within a session |
| `display_id` | str | — | "Person #N" / "Room" | from `person_id` |
| `n_people` | int | room | 0–3 | people present this tick |
| `source` | str | — | `sim`, `replay`, `sim_seed` | `sim` (corpus), `replay` (scenarios), `sim_seed` (`POST /api/demo/seed`) |
| `simulated_fields` | list[str] | — | — | `[]`: `source` already marks the whole row as synthetic; the list is for operator-set values in live or hybrid mode |
| `ambient_db` | float | room | approx. dB SPL (Leq over the tick) | section 9.2; NaN on audio dropout |
| `db_p90` | float | room | dB SPL | section 9.2; NaN on audio dropout |
| `light_lux` | float | room | lux | section 9.3 |
| `lux_quality` | float | room | 0–1 | 1.0 |
| `laughter_count` | int | room | onsets per tick | section 9.5; 0 on audio dropout |
| `compliments_count` | int | room | phrase hits per tick | section 9.5; 0 on audio dropout |
| `clenched_fists_count` | int | person | onsets per tick | section 9.5; 0 on video dropout |
| `posture_slump` | float | person | fraction of the tick, 0–1 | section 9.6; NaN on video dropout |
| `head_in_hands_frac` | float | person | fraction of the tick, 0–1 | section 9.6; NaN on video dropout |
| `social_withdrawal_flag` | int | person | 0 / 1 | derived rule, section 9.6 |
| `motion_energy` | float | person | shoulder-widths per second | section 9.6; NaN on video dropout |
| `face_valence` | float | person | −1..1, relative to the person's neutral | section 9.7; NaN when the face is hidden |
| `face_visible` | float | person | fraction of the tick, 0–1 | U(0.6, 1.0) when readable, else 0 |
| `hours_since_break` | float | person | h | break clock, section 6.4 |
| `tod_h` | float | room | h, [0, 24) | start time + elapsed, wraps at midnight |
| `room_db_base` | float | room | dB SPL | arrival baseline of the session, constant per session (section 9.2); live: the venue baseline, NaN until known |
| `room_lux_base` | float | room | lux | arrival baseline light, constant per session (section 9.3); live: the venue baseline, NaN until known |
| **`true_state`** (hidden) | str | person | a state id from the active set | the hidden state; never a model input |
| **`episode_age_s`** (hidden) | float | person | s | seconds since the person entered the current hidden state; used for `acc_vs_episode_age.png` |

`true_state` and `episode_age_s` are the only columns that do not exist in live data. They are dropped before
feature computation and used only for labels and evaluation.

### 4.2 Room row

Every tick has one room row (`person_id = 0`). It mirrors `valence/runtime/aggregator.room_record` and the room
slot of `valence/features/window.RollingWindow`, which a newly seated person inherits:

| Field group | Room row value |
|---|---|
| Room channels (`ambient_db`, `db_p90`, `light_lux`, `lux_quality`, `room_db_base`, `room_lux_base`, `laughter_count`, `compliments_count`, `tod_h`, `n_people`) | identical to the person rows of that tick |
| `clenched_fists_count` | sum over present people |
| `posture_slump`, `head_in_hands_frac`, `motion_energy`, `face_visible`, `hours_since_break` | mean over present people (NaN-aware) |
| `face_valence` | median over present people with a readable face |
| `social_withdrawal_flag` | 1 if any present person is flagged |
| `true_state` | most common state among present people (ties broken by state order in `contracts.ALL_STATES`); `"Empty"` when nobody is present |
| `episode_age_s` | NaN |

The room row is the training target for `--room-only` mode (one aggregate label, no per-person inference).
`"Empty"` ticks are excluded from training labels.

---

## 5. Legacy compatibility

`to_legacy_csv(df, path)` writes **person rows only** in the original 9-column format, so anything built on the
old CSV still runs:

| Legacy column | Source column | Note |
|---|---|---|
| `timestamp` | `ts` | relative seconds, not epoch |
| `ambient_db` | `ambient_db` | empty on audio dropout |
| `light_lux` | `light_lux` | |
| `hours_since_break` | `hours_since_break` | |
| `clenched_fists_count` | `clenched_fists_count` | per 5 s tick (now mostly 0) |
| `compliments_count` | `compliments_count` | room scope |
| `social_withdrawal_flag` | `social_withdrawal_flag` | derived rule |
| `facial_valence_snapshot` | `face_valence` | **empty when the face is hidden** (the original never had a missing face) |
| `true_emotional_state` | `true_state` | can also be `Overwhelmed` / `Tense` when generated with the P1 state set |

Columns that have no legacy equivalent (`db_p90`, `laughter_count`, `posture_slump`, `head_in_hands_frac`,
`motion_energy`, `face_visible`, `n_people`, ids, `tod_h`) are dropped. `legacy.generate_legacy()` reproduces the
original generator itself (same ranges, one random state per row) with a seed added.

---

## 6. Hidden state process

Each person has an independent hidden state that changes in episodes. People in the same session interact only
through the group-affinity term (6.5) and through the room channels they share (section 9).

### 6.1 Episodes (semi-Markov)

- On entering a state, the person draws a dwell time `D ~ Gamma(shape = 3, scale = dwell_min / 3)` minutes,
  floored at **2 min** (24 ticks). Per-state means `dwell_min` are 5–9 min (section 7), inside the plan's
  4–10 min band. A shape of 3 gives a coefficient of variation of ~0.58: episodes are neither fixed-length nor
  memoryless.
- When the dwell expires, the next state is drawn from the transition weights below. Self-transitions are
  excluded; duration is carried entirely by the dwell time.
- At session start each person's first state is drawn from the same weights at their starting
  `hours_since_break`.

### 6.2 Transition weights

For a person leaving state `s_now`, each candidate state `s ≠ s_now` in the active set gets

```
w_s = max(0.05, 1 + fatigue_affinity_s × f(h)) × (group_affinity_s if n_people_session ≥ 2 else solo_affinity_s)
```

and the next state is drawn with probability `w_s / Σ w`.

### 6.3 Fatigue tilt

`f(h) = 1 / (1 + exp(−(h − 3.0) / 0.5))`, with `h = hours_since_break`. The tilt is negligible before ~2 h,
half-strength at 3 h and near full strength after ~4 h. Positive `fatigue_affinity` (Exhausted +2.0,
Overwhelmed +1.5, Frustrated_Masked +1.0, Tense +1.0) pulls long stretches toward strain; negative values
(Focused −0.6, Collaborative_Happy −0.5) make those states less likely late. Example at `h = 5` (f ≈ 0.98),
core set, one person, before the current state is excluded: weights ≈ Focused 0.41, Frustrated_Masked 1.98, Exhausted 2.96, Collaborative_Happy 0.20
(after the solo multiplier 0.4).

### 6.4 Breaks and the break clock

| Rule | Value |
|---|---|
| Break onset | Poisson, base rate **1 per 120 min** of presence (as built; the plan's ~1/90, see the resolved table above); ×2 while Exhausted or Overwhelmed |
| Break length | U(10, 20) min, at least `settings.break_absent_min` (10 min), so every simulated break also counts as a break for the live aggregator's presence-based clock |
| During a break | The person has no rows; `n_people` drops; the room's sound and light targets are recomputed without them |
| On return | `hours_since_break` resets to 0; the new state is `Focused` (alone) or `Focused` / `Collaborative_Happy` with equal probability (session with ≥ 2 people) |
| Clock | `hours_since_break` increases by 5 s / 3600 per present tick |

### 6.5 Group affinity

`Collaborative_Happy` has `group_affinity` 1.8 and `solo_affinity` 0.4, so it is ~4.5× likelier, relative to the
other states, in sessions with 2–3 people. `Tense` has 1.5 / 0.6 ("often ≥ 2 people"). All other states have
1.0 / 1.0. A lone person in `Collaborative_Happy` is displayed as "Upbeat" (`contracts.display_label`).

---

## 7. Per-state parameters

Values are the per-state behavioral targets the observation model (section 9) moves toward with a lag. Room
quantities (dB, lux link, laughter, compliments) are averaged or summed over everyone present. "sw/s" =
shoulder-widths per second.

| Parameter (`params.py` field) | Focused | Frustrated_Masked | Exhausted | Collaborative_Happy | Overwhelmed (P1) | Tense (P1) |
|---|---|---|---|---|---|---|
| Face valence, episode mean (`face_mu`) | 0.00 (neutral) | −0.12 unmasked; **+0.45 forced smile** with probability `mask_p` | −0.04 (flat) | +0.50 | −0.20 | −0.35 (face agrees) |
| Masking probability per episode (`mask_p`) | 0 | 0.70 core set / 0.92 with P1 | 0 | 0 | 0 | 0 |
| Face hidden share | session base U(0, 0.30) + head-in-hands term | same | same | same | **U(0.40, 0.70) per episode** (hands over the face) | same as Focused |
| Room dB target, calibrated mic (`db_target`) | 48 | 66 | 45 | 63 | 66 | 70 |
| Lux link, × session baseline (`lux_mult`) | ×1.00 | ×0.85 | ×0.65 | ×1.15 | ×0.85 | ×0.95 |
| Fist onsets / min, bursty (`fist_rate`) | 0 | 0.8 | 0 | 0 | 0.2 | 0.8 |
| Laughter onsets / min into the room (`laugh_rate`) | 0.03 | 0.10 | 0.02 | **0.50** | 0.02 | 0.05 |
| Compliments / min into the room (`compl_rate`) | 0.02 | 0.02 | 0.01 | **0.20** | 0.01 | 0.01 |
| Slump, long-run share of time (`slump_p`) * | 0.05 | 0.30 | 0.70 | 0.03 | 0.60 | 0.20 |
| Head in hands, long-run share (`hih_p`) | 0.08 ("thinking") | 0.15 | 0.35 | 0.05 (laughing; bouts start preferentially in ticks with a laughter onset) | 0.70 | 0.08 |
| Social withdrawal | derived per tick from slump and motion (section 9.6); not a free parameter | ← | ← | ← | ← | ← |
| Motion level, sw/s (`motion`) * | 0.03 | 0.05 | 0.015 | 0.06 | 0.03 | 0.09 |
| Mean dwell, min (`dwell_min`) | 9 | 6 | 8 | 6 | 5 | 5 |
| Fatigue affinity (`fatigue_affinity`) | −0.6 | +1.0 | +2.0 | −0.5 | +1.5 | +1.0 |
| Group / solo multiplier | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.8 / 0.4 | 1.0 / 1.0 | 1.5 / 0.6 |
| Break-rate multiplier | ×1 | ×1 | ×2 | ×1 | ×2 | ×1 |
| Plan signature (tendency, not an enforced range) | quiet 40–55 dB, normal light, 1–3 h since break | loud, dim, 3.5–6 h, fists | silent, very dim, 5–8 h, slump / withdrawal | lively 60–75 dB, bright, fresh break, laughter / compliments, ≥ 2 people | head in hands, slump, 3–7 h, 60–75 dB | fists, 65–80 dB, often ≥ 2 people, high motion |

\* Refit to live capture run 1 (`e0ad762`, section 10); Focused motion 0.03 sw/s is the quiet-typing median
measured live.

Design intent of the table (properties built in, not results):

- **The face is informative but not sufficient.** Focused (0.00), unmasked Frustrated_Masked (−0.12),
  Exhausted (−0.04) and Overwhelmed (−0.20) overlap once the episode spread (σ 0.12) and tick noise (σ 0.03) are
  added; a masked Frustrated_Masked episode (+0.45) overlaps Collaborative_Happy (+0.50). This creates the
  "Focused paradox" and the masking dilemma for a face-only model.
- **Context is informative but slow, shared and sparse.** dB and light lag the state by minutes, are shared by
  everyone in the room and carry a per-session offset; events are rare per tick. A snapshot of context is weak;
  a window is needed.
- **The face reacts first.** A new episode's face appears at once, while context follows with a 1–4 min lag, so
  a face model has a deliberate advantage in the first minutes of an episode. `acc_vs_episode_age.png` reports
  this rather than hiding it.
- **Frustrated_Masked and Tense differ mainly in the face** (fists 0.8 / min in both; dB 66 vs 70; motion 0.05
  vs 0.09). The no-face model is expected to confuse this pair; the plan reports it as "the face is informative
  but not sufficient".

---

## 8. State sets: core 4 and the P1 switch

The active set comes from `settings.state_set` (`VALENCE_STATE_SET`): `core` = `CORE_STATES` (4), `all` =
`ALL_STATES` (6). Each set has its own corpus and its own models.

| | Core set (P0) | With P1 (`all`) |
|---|---|---|
| States | Focused, Frustrated_Masked, Exhausted, Collaborative_Happy | + Overwhelmed, Tense |
| Frustrated_Masked covers | masked **and** unmasked frustration: `mask_p` = 0.70, so ~30% of its episodes show the unmasked face (−0.20) | **masked only**: `mask_p` = 0.92 (forced smile in ≥ 90% of episodes) |
| Unmasked, visible frustration | inside Frustrated_Masked | `Tense` (face −0.45, fists, loud, high motion) |
| Display of Frustrated_Masked | "Frustrated", or "Frustrated / Masked Stress" only when `mask_override` fires | always "Frustrated / Masked Stress" |
| Corpus | `data/sim/corpus_core_*.parquet` | `data/sim/corpus_all_*.parquet`, generated only if the H10:30 gate passes |

`params_for(state, state_set)` applies the switch: it returns the table values, except that `mask_p` for
Frustrated_Masked becomes 0.92 when `Tense` is in the active set. Scenarios may override `mask_p` per segment
(for example `mask_p: 1.0` in `frustrated_masked`).

---

## 9. Observation model

All behavior targets come from section 7. `ns` is the global `noise_scale` (section 12); terms marked ×ns grow
with it.

### 9.1 Context lag

| Lagged quantity | Time constant τ | Update per tick |
|---|---|---|
| Room sound level and room light multiplier | τ_room ~ U(2, 4) min per session, ×ns | `x ← x + α (target − x)`, `α = 1 − exp(−5 s / τ)` |
| Each person's behavior targets (fist, laughter and compliment rates, slump and head-in-hands shares, motion level) | τ_person ~ U(1, 2) min per person, ×ns | same |
| Face | none: drawn fresh at each episode start | — |

### 9.2 Room sound (`ambient_db`, `db_p90`)

| Step | Rule |
|---|---|
| Target | Mean `db_target` of the people present; 40 dB when the room is empty |
| Lag | Room level follows the target with τ_room |
| Mic offset | One per session, U(−12, +12) dB, not scaled by `ns` (plan range for the half-width: 12–15 dB). Stands for an uncalibrated microphone and room differences. Scenarios use 0 dB (calibrated) unless they set `mic_offset_db`. |
| Venue (`venue_db_range`) | One hall level per session, U(40, 75) dB, drawn independently of every state: the noise of the hall around the team. Scenarios have none (a quiet team room) unless they set `venue_db` (`loud_venue`: 72 dB). |
| Arrival baseline (`room_db_base`) | `power_sum(venue, 48 dB) + mic_offset + N(0, 1.5·ns)`: what an operator's capture with one neutral occupant (`neutral_db` 48, the Focused target) would measure. Constant per session. |
| Tick value (as built) | `ambient_db = power_sum(venue, room_level) + mic_offset + drift + transient + N(0, 1.2·ns) + 2 dB × min(laughter_count, 2)`, where `power_sum` adds levels as powers (`10·log10(Σ 10^(L/10))`), drift is AR(1) (×0.95, σ 0.5·ns dB per tick) and transients are state-independent bumps (0.3 per min, +4 to +12 dB, 30% of them −0.5× that, for 2–8 ticks) |
| p90 (as built) | `db_p90 = ambient_db + 1.5 + abs(N(0, 1·ns)) + 1.5 dB if laughter_count > 0` |
| Random streams | The venue and baseline draws use their own generator seeded with crc32 of the session id, so adding them left every scripted scenario's other draws unchanged. |
| Sharing | Every person row of the tick carries the same room values. In a mixed room (for example one Collaborative_Happy and one Focused person) both rows see the same, intermediate level, so dB is weaker evidence per person than per room. |

Because the offset (±12 dB) is comparable to the gap between quiet and lively states (~15 dB), absolute dB is
only weakly diagnostic across sessions; relative and trend features (`db_rel_15m`, slopes) carry most of the
signal. The hall level (40–75 dB, power-summed) makes absolute dB weaker still: in a 72 dB hall a Frustrated room
(66 dB) adds only about 1 dB. So the served models read the room relative to the arrival baseline (`db_room_*`
= `ambient_db` − `room_db_base`). The M2-abs ablation (absolute levels, same corpus) and the venue-shift
evaluation (`docs/MODEL_CARD.md` §5) test this.

### 9.3 Light (`light_lux`, `lux_quality`)

| Step | Rule |
|---|---|
| Session baseline | log-uniform in [100, 800] lux |
| Arrival baseline (`room_lux_base`) | session baseline × tod factor at the session start × `exp(N(0, 0.08·ns))`, constant per session; the served models read `lux_room_*` = log2(`light_lux` / `room_lux_base`) |
| Time-of-day drift | ×0.85 between 22:00 and 06:00 (artificial light only), ×1.0 otherwise; sessions start 09:00–23:00, so many cross 22:00 |
| State link | Room light multiplier lags (τ_room) toward the mean `lux_mult` of the people present (1.0 when empty). The link is weak (×0.65–×1.15) next to the 8× spread of baselines (100–800 lux). |
| Tick value (as built) | `light_lux = baseline × tod_factor × room_light_multiplier × transient × exp(drift + N(0, 0.015·ns))`, where drift is an AR(1) on log-lux (×0.98, σ 0.01·ns per tick) and transients are state-independent auto-exposure jumps (0.5 per min, ×U(0.75, 1.3) for 2–12 ticks) |
| `lux_quality` | 1.0 (live: background share × unclipped share × stability, halved while uncalibrated) |

### 9.4 Time (`hours_since_break`, `tod_h`)

`tod_h = (start_hour + elapsed_h) mod 24`. `hours_since_break` is the per-person clock of section 6.4. Neither
channel ever has dropout (they come from clocks).

### 9.5 Events (`clenched_fists_count`, `laughter_count`, `compliments_count`)

| Channel | Per-tick count |
|---|---|
| `clenched_fists_count` (person) | `Poisson((lagged fist_rate × burst + FP_fist_per_min) / 12)` |
| Burst modulator (fists) | Two-state, per person: high ×1.7 / low ×0.3, equal long-run time, switches with probability 0.1 per tick (about every 50 s). The long-run mean stays at the table rate. |
| `laughter_count` (room) | `Poisson((Σ_present lagged laugh_rate + FP_laugh_per_min) / 12)` |
| `compliments_count` (room) | `Poisson((Σ_present lagged compl_rate + FP_compl_per_min) / 12)` |

Laughter and compliments are room-scope: in a multi-person session they cannot be attributed to one person,
matching the live pipeline (one microphone; near-field gating keeps other tables out). By construction, at
0.8 fists / min a single tick holds an onset ~6% of the time, while at least one onset in 5 minutes is expected
~98% of the time (Poisson arithmetic, ignoring burstiness). That gap is what the 1 / 5 / 15-minute count
features exploit.

### 9.6 Posture, head in hands, motion, withdrawal

| Channel | Rule |
|---|---|
| Slump state | Two-state Markov chain per person. With target share `p = lagged slump_p + FP_slump_share`: when off, switch on with probability 0.15·p per tick; when on, switch off with probability 0.15·(1 − p). Stationary share = p; bouts last minutes, not ticks. |
| `posture_slump` | U(0.7, 1.0) while slumped; otherwise 0, or U(0.0, 0.1) on 15% of ticks |
| Head-in-hands state | Same chain with target share = lagged `hih_p` (+0.35 for Collaborative_Happy on ticks with a laughter onset, when the chain also switches with probability 0.35 instead of 0.15) |
| `head_in_hands_frac` | U(0.5, 1.0) while on, 0 otherwise |
| `motion_energy` | `exp(log_motion)`: log-motion follows an AR(1) toward log(lagged `motion` × 0.7 while slumped × 0.5 in a still phase), pull 0.25 per tick (`motion_ar`), innovation N(0, 0.18·ns) (`motion_noise`). Still phases occur in every state: a two-state chain with share 0.4 (`still_share`), switching 0.06 per tick (`still_switch`) |
| `social_withdrawal_flag` | `withdrawal_flag(posture_slump, motion_energy)` from `valence/features/spec.py`: 1 if `posture_slump ≥ 0.6` and `motion_energy ≤ 0.02` sw/s (as built), else 0 (0 when either is NaN). The live aggregator calls the same function. |

### 9.7 Face (`face_valence`, `face_visible`)

| Step | Rule |
|---|---|
| Episode face | At episode start: with probability `mask_p` use the forced smile (+0.45), otherwise `face_mu`; then `episode_face ~ N(mean, 0.12)` |
| Hidden share | Session base U(0, 0.30). Per tick: `p_hidden = min(0.98, base + 0.3 × head_in_hands_frac)`. Overwhelmed episodes instead draw `p_hidden ~ U(0.40, 0.70)` once per episode. |
| Readable tick | Face readable if video is not dropped and `U(0,1) ≥ p_hidden` |
| Value | `face_valence = clip(N(episode_face, 0.03·ns + 0.35 × max(0, (200 − light_lux) / 200)), −1, 1)`: tick noise grows below 200 lux (+0.175 at 100 lux, +0.35 at 0 lux) |
| `face_visible` | U(0.85, 1.0) on a readable tick, 0 otherwise |
| Hidden tick | `face_valence = NaN`, `face_visible = 0` |

Valence is relative to the person's neutral (0 = their neutral). Live, the edge subtracts the median of the
person's first 20 s of visible face; the simulator assumes that baseline is exact (section 17).

### 9.8 Missing data

| Source | Rule | Effect |
|---|---|---|
| Audio dropout | 2% of ticks, per session | `ambient_db`, `db_p90` = NaN; `laughter_count`, `compliments_count` = 0 |
| Video dropout | 2% of ticks, per person | `posture_slump`, `head_in_hands_frac`, `motion_energy`, `face_valence` = NaN; `face_visible` = 0; `clenched_fists_count` = 0; withdrawal flag 0 |
| Face hidden | section 9.7 | `face_valence` = NaN on 0–30% of ticks (40–70% in Overwhelmed) |
| Light | no per-tick dropout (live lux is a 10 s median refreshed at 1 Hz) | — |
| Clocks | never missing | — |

Count fields are integers in the contract, so a dropped tick reads as 0 events (section 17).

---

## 10. Detector false positives (set from the live capture)

Real detectors fire when nothing is happening: typing can look like a fist, looking down at the keyboard can
look like a slump, keyboard clicks can be heard as laughter. The simulator adds these as state-independent
rates. They started as placeholders; capture run 1 replaced them with measurements (as built in `params.FP`:
fist 0.02 / min, slump share 0.02, laughter 0.02 / min, compliments 0.01 / min; small non-zero rates kept so one
event is never certain). The table gives the as-built value, with the placeholder it replaced in parentheses.

| Parameter (`params.FP`) | As built (placeholder) | Meaning | Set from (capture → statistic) |
|---|---|---|---|
| `fist_per_min` | 0.02 / min (0.05) | fist onsets with no deliberate fist | quiet typing, 3 min → fist onsets per minute |
| `slump_share` | 0.02 (0.05) | share of time the slump flag is on while sitting upright | looking at the keyboard and quiet typing, 3 min each → slump duty cycle |
| `laugh_per_min` | 0.02 / min (room; unchanged) | laughter onsets with nobody laughing | quiet typing audio → laughter onsets per minute |
| `compl_per_min` | 0.01 / min (room; unchanged) | compliment hits with nobody speaking | quiet typing audio → compliment hits per minute |
| Motion scale (section 7, `motion`) | Focused 0.03 sw/s, others as in section 7 (Focused 0.08) | typical motion per state | quiet typing → median `motion_energy` anchors Focused; the other states were refit with it |
| Slump scale (section 7, `slump_p`) | as in section 7 (Focused 0.15) | share of time slumped per state | slumping, 3 min → duty cycle of the slump flag while actually slumped; per-state shares are rescaled if the detector sees only part of a real slump |
| Fist hit rate (check only) | — | does the detector see deliberate fists | clenching fists, 3 min → fist onsets per deliberate fist; reported in `docs/BENCHMARKS.md`, not fed into the simulator (per-state fist rates are already observed, post-detector rates) |

Procedure (plan H7:00–10:30):

1. The edge session records the four 3-minute captures (quiet typing, looking at the keyboard, clenching fists,
   slumping) as features-only JSONL in `data/real/capture_*.jsonl` and reports fist FP / min while typing, slump
   duty cycle while typing and the motion distribution in `docs/BENCHMARKS.md`.
2. The captures are replayed through the same aggregator into `TickRecord`s, and the statistics above are
   computed from those ticks.
3. `params.FP` and the motion / slump scales are updated, the corpus is regenerated with the **same seed and the
   frozen `noise_scale`** (changed only by the single, logged re-tune of section 12.2 step 5), and the models are
   retrained.
4. Negative control: replaying the typing capture must yield `Frustrated_Masked` or `Tense` on **≤ 20% of
   ticks** (`valence/eval/negative_control.py`). This is a gate, not a tuning target.

Assumption: quiet typing and keyboard gaze are treated as Focused behavior. The capture is one person (the
builder) in one room with one camera.

---

## 11. Units and sim-to-real consistency

### 11.1 Units per channel (`valence/features/spec.SENSOR_SPEC`)

The simulator generates every channel in the unit the live pipeline produces from the edge observations.

| Channel | Unit | Live source (edge → aggregator) |
|---|---|---|
| `ambient_db` | approx. dB SPL, Leq over the tick | power mean of `AudioObs.db_spl` (dBFS + `db_cal_offset`, default 94, set by `POST /api/calibrate/room` against a phone SPL meter) |
| `db_p90` | dB SPL | 90th percentile of the tick's ~1 s block levels |
| `light_lux` | lux | 10 s median of the webcam lux regression, refreshed at 1 Hz; 2-point calibration against a phone lux meter |
| `laughter_count` | onsets per tick, room | YAMNet laughter onsets, counted only when the block is ≥ the venue room baseline + 6 dB (`settings.near_field_db`; before the baseline is complete, the 15-min room median + 6 dB) |
| `compliments_count` | hits per tick, room | Vosk restricted-grammar phrase hits, same near-field gate |
| `clenched_fists_count` | onsets per tick, person | debounced fist onsets (score ≥ 0.7 held ≥ 0.5 s; hands in the bottom 20% keyboard zone ignored) |
| `posture_slump` | fraction of the tick | share of frames with a debounced slump |
| `head_in_hands_frac` | fraction of the tick | share of frames with head in hands |
| `motion_energy` | shoulder-widths per second | mean of `PersonObs.motion` (EMA α 0.3 of keypoint displacement). Raw unit; per-person normalization happens in the feature layer (`*_rel_15m`). The edge describes ~0.0–0.1 as still and 0.3+ as fidgeting (`docs/CONTRACT_REQUESTS.md`). |
| `face_valence` | −1..1, relative to the person's neutral | median blendshape valence over the tick; neutral = median of the first 20 s of visible face |
| `face_visible` | fraction of the tick | share of frames with a readable face |
| `hours_since_break` | h | presence-based clock: an absence ≥ `break_absent_min` (10 min; ~0.5 for demos) or `POST /api/events/break` resets it |
| `tod_h` | h, [0, 24) | local clock |
| `n_people` | count | most people seen in the tick (≤ `settings.max_tracks` = 3) |
| `room_db_base`, `room_lux_base` | dB SPL, lux | the venue room baseline: median dB and median lux of ~90 s captured on arrival with the person sitting neutral (`POST /api/room/baseline`), or automatically the first 90 s of a live session or replayed recording |

### 11.2 Consistency rules

| Rule | Why |
|---|---|
| One `TickRecord` shape for simulation, replay and live | `compute_features` runs unchanged offline and live (parity test) |
| Withdrawal uses `features.spec.withdrawal_flag` in both the simulator and the aggregator | one definition |
| Room row = `aggregator.room_record` semantics | the room slot and room-only mode see the same thing in both worlds |
| Corpus breaks last ≥ `break_absent_min` | a simulated break is always a live break |
| A person who returns from a break starts a new person window in `build_dataset.py` (room history kept) | live does not re-identify people: after an absence the person normally gets a new display id and a cold person window, while the room slot persists |
| Mic offset and hall level in the simulator; venue calibration and the room baseline live | the served models read dB and lux only relative to the room baseline, never as absolute levels (M2-abs ablation, venue-shift evaluation) |
| Simulated arrival baseline = hall + one neutral occupant + measurement error | the same thing `POST /api/room/baseline` measures live with the person sitting neutral |
| Face valence is relative in both | the simulator's 0 is the person's neutral, as the edge's baseline subtraction makes it live |

### 11.3 Capture runs 2–3: scale check (no refit)

Mac captures of one person at the demo desk (`data/real/capture_{keyboard_gaze,slump,audio}.jsonl`), replayed
through the aggregator and compared with simulated ticks by `valence/eval/capture_stats.py`
(`reports/capture_vs_sim.json`; it also reports `db_room`, laughter and compliments). The simulator was not
changed.

| Capture | Finding |
|---|---|
| Keyboard gaze vs simulated Focused | Slump 0% under the fixed rule (run 1's nose-only rule gave 93%); motion p50 0.022 sw/s (simulated Focused 0.024); face visible 100%; `db_room` p50 0.1 dB. Matches. |
| Slump (cued 20 s slumps with upright gaps) | Slumping is detected (per-tick slump p10 / p50 / p90 0 / 0.27 / 1.0). But motion during a *held* slump is p50 0.101 sw/s, against 0.033 upright and 0.575 in transitions; simulated Exhausted is ~0.012. Face visible drops to 0.33 while slumped. This looks like a perception artifact (head keypoints jitter when the head drops) and is assigned to the edge session; the simulator's still slump was kept. |
| Audio ("great job" ×3, one laugh) | Laughter detected under the new rule (≥ 0.05 for one result, off < 0.02, 4 s refractory; one ~5 s laugh counts 2 onsets); compliments 3/3. Speech and laughter at the desk are +12–17 dB over the quiet baseline. The one laugh tick read face valence 0.977, the others 0.03–0.07: the face-only model reads Collaborative with probability ≥ 0.5 from face valence 0.4 up, so a sustained broad smile should drive the mask override, but a sustained forced-smile capture is still needed to confirm the face scale live. |

---

## 12. Difficulty and anti-circularity

The simulator encodes the hypothesis, so its difficulty could be turned until context wins by any margin. To
avoid that, difficulty is **one global knob, tuned against one criterion that is not the result, then frozen**.

### 12.1 What `noise_scale` scales

| Scaled by `noise_scale` | Not scaled |
|---|---|
| dB tick noise σ (4 dB), p90 spread σ (2 dB), lux noise σ (0.08, log scale), motion noise σ (0.4, log scale), face tick noise σ (0.15), lag time constants τ_room and τ_person | mic offset, face episode σ, low-light face term, face hidden shares, dropout, event rates, false-positive rates, all per-state targets, dwell times, transitions |

It acts on face and context channels alike, so turning it up cannot favor one model family.

### 12.2 Tuning rule (`valence/sim/tune_difficulty.py`)

1. Generate a **tuning corpus** of 100 sessions with its own seed (never reused for training or evaluation) at
   `noise_scale ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0}`.
2. At each value, train M2 with the production LightGBM settings on the tuning corpus's train split and score
   macro-F1 on its validation split.
3. **Criterion A (no ceiling):** choose `ns*` = the smallest grid value with M2 macro-F1 ≤ 0.92.
4. Freeze `ns*` in `artifacts/sim_manifest.json` and generate the 300-session corpus with it.
5. **Criterion B (variance match), once the live capture exists:** for each channel the capture measures
   (`ambient_db`, `light_lux`, `motion_energy`, `posture_slump`, `face_valence` when visible), the live
   tick-to-tick standard deviation within each 3-minute capture must fall inside the 5th–95th percentile of the
   same statistic over 36-tick stretches of simulated Focused episodes at `ns*`. If a channel fails, `ns*` is
   re-picked **once** as the smallest grid value meeting A and B, and the change and its reason are logged in
   the manifest. If no grid value meets both, `ns*` stays at the criterion-A value and the mismatching channels
   are listed as a limitation.

The rule reads only M2's validation macro-F1 and the variance check. It never consults M0, M0t, M1, M2-noface,
the M2-minus-M0 gap, Frustrated_Masked recall, or any test split. M0 and M1 are reported **wherever they land**.
The frozen value is **TBD — recorded by `valence/sim/tune_difficulty.py` in `artifacts/sim_manifest.json`**.

### 12.3 Sensitivity

The main corpus is regenerated with the same seed at **×0.5, ×1 and ×1.5** of `ns*`. The full ladder is retrained
on each and every model's macro-F1 is reported, together with the legacy slide, to show whether the model
ordering is structural or an artifact of one noise level. If the ordering changes, that is reported.
**TBD — produced by `valence/sim/tune_difficulty.py` and `valence/eval/evaluate.py` → `reports/metrics.json`.**

### 12.4 Claim wording

"Under explicit, published behavioral assumptions, a temporal context model beats a face snapshot; in-the-wild
validation is future work." Nothing stronger is claimed from this dataset.

---

## 13. Scenarios

Scenarios are scripted sessions in `scenarios/*.yaml`, rendered by `valence/sim/scenarios.render()` through the
same observation model with `source = "replay"`. Each segment holds the listed state for its whole length (no
Markov transitions); context still lags by τ, so labels are expected to follow segment boundaries with a delay
of minutes. Unset nuisance values are drawn from the scenario's own `seed`; the mic offset defaults to 0 dB, and
there is no hall noise unless the YAML sets `venue_db`. Scenarios are **never** part of the training or evaluation
splits. They drive the demo, `tests/test_scenarios.py` (every scenario, after a 2-min grace: the scripted state on
≥ 80% of ticks per segment and person, the override on masked segments and nowhere else) and validation of the
mask-override detector.

| Scenario | State set | People | Minutes | Script | Setup (start, h since break, lux base, face hidden) | Shows |
|---|---|---|---|---|---|---|
| `focused` | core | 1 | 12 | Focused 12 | 14:30, 1.5 h, 420 lux, 5% | The Focused paradox: the furrowed face (−0.15) reads negative to a face-only model; quiet room, normal light and 1.5 h since break say Focused. Override must not fire. |
| `frustrated_masked` | core | 1 | 14 | Focused 4 → Frustrated_Masked 10 (`mask_p: 1.0`) | 16:00, 3.6 h, 230 lux, 5% | The masking dilemma: forced smile over fists, rising noise and a long stretch without a break. Target (plan): `mask_override` fires within 6 ticks on this scenario and the display flips to "Frustrated / Masked Stress". |
| `exhausted` | core | 1 | 12 | Exhausted 12 | 02:00, 5.2 h, 140 lux, 15% | Late-night silence, dim light, slump and withdrawal; low light adds face noise. |
| `collaborative_happy` | core | 2 | 12 | Collaborative_Happy 12 (both) | 11:00, 0.3 h, 520 lux, 5% | Laughter and compliments in a bright, lively room; displayed "Collaborative / Happy" because 2 people are present. Override must not fire. |
| `overwhelmed` | all (P1) | 1 | 13 | Focused 3 → Overwhelmed 10 | 22:30, 2.5 h, 260 lux, seeded draw | Head in hands and slump in a noisy crunch; the face is hidden 40–70% of the time. Decision stability with missing face evidence. |
| `tense` | all (P1) | 2 | 13 | Focused 3 → Tense 10 (both) | 17:00, 2.5 h, 330 lux, seeded draw | Visible frustration: face, fists and a loud room agree, so no override is expected. |
| `context_blindness` | core | 2 | 18 | Collaborative_Happy 7 (both, `hih_p: 0.45`) → both absent 2 → Person #1 Exhausted 9 (`hih_p: 0.45`, `lux_mult: 0.5`), Person #2 absent | 20:00; 4.4 h (Person #1) / 1.0 h (Person #2); 380 lux; 5% | The same head-in-hands pose in two contexts gets two labels: with laughter and compliments (amused), and in a silent, dim room after hours of work (exhausted). The 2-min absence is shorter than the 5-min scripted-break threshold, so Person #1's clock keeps running past 4.4 h into the Exhausted half; no `hours_since_break` override is needed. |
| `full_demo` | core | 1 | 26 | Focused 5 → Frustrated_Masked 9 (`mask_p: 1.0`) → absent 6 (break) → Collaborative_Happy 6 | 15:00, 2.0 h, 300 lux, 5% | The pitch storyline: focused work, the masked flip, a logged break that resets the clock, recovery. With one person the last segment displays "Upbeat". |
| `loud_venue` | core | 1 | 15 | Focused 6 → Frustrated_Masked 9 (`mask_p: 1.0`) | 13:00, 1.5 h, 450 lux, 5%; `venue_db` 72, seed 109 | A loud hackathon hall around the team. With the venue baseline the hall is this room's normal, so the flip has to come from the person, not the noise. |

Scripted absences reset `hours_since_break` when they last ≥ 5 min (scenarios are short; live demos run with
`break_absent_min` ≈ 0.5 min). The demo seed (`POST /api/demo/seed`) uses the same engine to fill an empty
window with room context only (no face), tagged `source = "sim_seed"`.

---

## 14. Splits and training windows

| Split | Share of sessions | Use |
|---|---|---|
| Train | 70% (~210) | model fitting |
| Validation, early-stop half | 7.5% (~22) | LightGBM early stopping (50 rounds) |
| Validation, calibration half | 7.5% (~23) | temperature scaling, clip bounds check |
| Test | 15% (~45) | every reported number |

- Splits are **by session**: all rows of a session, including its room rows, go to one split. Assignment is a
  seeded permutation of session ids, stored in the manifest (planned; today `train_ladder.split_sessions`
  recomputes it from a seed and `evaluate.py` rebuilds the held-out split the same way). A planned test checks
  that no session appears in two splits.
- The tuning corpus (section 12.2) and the scenarios are separate and never enter these splits.
- Training windows (`valence/features/build_dataset.py`): a window ends every 6 ticks for each person, with a
  random length of 1–180 ticks (up to 15 min), so models see the warm-up regime as well as full windows. The label
  is the person's `true_state` at the window's last tick. That is ~83k windows from ~500k person-ticks (estimate).
- Class balance is not forced; it follows from dwell times and transition weights. LightGBM uses balanced class
  weights. Per-state counts are written to the manifest (**TBD — produced by the generator**).

---

## 15. Files, reproducibility and privacy

| Path (planned names) | Contents | In git | Producer |
|---|---|---|---|
| `data/sim/corpus_core_ns{ns}.parquet` | 300-session corpus, core states | no (`data/sim/` is gitignored) | `valence/sim/generator.py` |
| `data/sim/corpus_all_ns{ns}.parquet` | 300-session corpus, 6 states (only if P1 lands) | no | same |
| `data/sim/sens_{core,all}_ns{ns}.parquet` | sensitivity corpora (×0.5, ×1.5) | no | `valence/sim/tune_difficulty.py` |
| `data/sim/tune_ns{ns}.parquet` | tuning corpora | no | same |
| `data/sim/legacy.csv` | output of the original generator | no | `valence/sim/legacy.py` |
| `artifacts/sim_manifest.json` | per corpus: seed, `n_sessions`, `state_set`, `noise_scale`, params hash, split assignment, row and per-state counts, content hash; the `ns*` tuning log | yes | generator / tuner |
| `data/real/capture_*.jsonl` | live feature capture: run 1 (baseline, covered camera, fists, typing), run 2 (keyboard gaze, slump), run 3 (audio); summaries in `data/real/capture_summary_*.json` | yes | edge session (`python -m valence.edge.run --jsonl`) |
| `data/real/` (other files) | acted set and self-report labels (names set by the edge session) | yes | edge session |

Reproducibility:

- Corpus seeding: `numpy.random.SeedSequence(seed).spawn(n_sessions)` gives one independent generator per
  session, so session `i` depends only on `(seed, i)` and the parameters.
- Same `(seed, n_sessions, state_set, noise_scale, params)` → same rows. The manifest stores a content hash
  computed over the rows in order (`pandas.util.hash_pandas_object`), not over Parquet bytes, which can differ
  between library versions. The plan's simulator check is "same seed → same hash".
- `data/sim/` is never committed. Anyone can regenerate it from the manifest (planned interface:
  `python -m valence.sim.generator --sessions 300 --seed <seed> --state-set core --noise-scale <ns*>`).

Privacy:

- Simulated data describes no real person.
- `data/real/*.jsonl` holds only `VideoObs` / `AudioObs` scalars in the `obs_to_dict` format
  (`{"type": "video" | "audio", ...}`, NaN as `null`): no frames, audio, transcripts or keypoints. It comes from
  the consenting builder. Pseudonymous behavioral data is still personal data, so these files stay local to the
  project and are deleted on request.

---

## 16. Planned verification

None of these checks has been run yet; they are the acceptance criteria for the generator and the data.

| Check | Criterion | Source |
|---|---|---|
| Dwell and transitions | empirical dwell means, 2-min floor and transition frequencies match the configured values | plan, Verification: Simulator |
| Determinism | same seed → same content hash | plan |
| Sensitivity sweep | printed for ×0.5 / ×1 / ×1.5 | plan |
| Units and ranges | every channel inside its `SENSOR_SPEC` range; NaN only where section 9.8 allows | this spec |
| Missingness | per-session base face-hidden share drawn in 0–30%, realized share rising with head in hands; 40–70% in Overwhelmed episodes; dropout ≈ 2% per modality | this spec |
| Splits | no session in two splits | plan |
| Difficulty target | M2 macro-F1 between 0.85 and 0.92 on the test split; a value outside the band is reported, not tuned away | plan |
| Sim-to-real | negative control ≤ 20% Frustrated_Masked / Tense on typing; ≥ 90% of live features inside clip bounds | plan |

---

## 17. Limitations

| Limitation | Consequence |
|---|---|
| The parameters are assumptions carried over from the original PRD and the plan, not measurements of people. | The context-vs-face result holds only under these assumptions. In-the-wild validation is future work. |
| Room light is linked to emotional state (×0.65–×1.15). A person's state does not change the lights; the link stands in for confounds (for example, late-night work in dim rooms). | A model may learn a light–state association that will not transfer. The link is kept weak, and lux shifts are reported separately from face degradation. |
| Hidden states are crisp categories with clean boundaries. | Real states blend and change gradually; self-report labels are noisy. |
| Face valence is a Gaussian around a state mean, with an exact personal-neutral baseline. Individual differences in expression, display norms and baseline estimation error are not modeled. | The face channel is cleaner here than live. The anti-bias claim is narrowed to "decision stability when face evidence is missing or degraded"; demographic validation is an explicit gap. |
| Low light adds face noise but does not raise the missing-face rate. | The live face-detection rate at three light levels (acted set) will show whether it should. |
| The hall is one constant level per session, power-summed with the room, and the arrival baseline is the hall plus one neutral occupant with N(0, 1.5 dB) error. | Real halls fluctuate, and a baseline captured during a loud burst or with an agitated person shifts every room-relative feature. The venue-shift evaluation mixes hall noise in the same way, so its result is partly by construction. |
| Masking is a forced-smile episode with unchanged context. | Real masking may be subtler or only partial. |
| False-positive rates and scales come from one person, one room, one camera, three minutes per condition. | Other people, desks and cameras may differ. |
| People are independent apart from group affinity and shared room channels. There is no emotional contagion, conversation dynamics or person-to-person attribution of laughter. | Multi-person realism is limited to "the room is shared". |
| Count fields are integers, so a dropped tick reads as 0 events. | Missing counts cannot be told apart from true zeros (`fill_ratio` covers window fill, not per-channel dropout). |
| `lux_quality` is always 1.0 and lux has only multiplicative noise. | Live lux is less reliable than simulated lux, especially before calibration. |
| Sessions are 1–2 h slices that start 0–5 h after a break; there is no full-day trajectory or circadian model beyond the night light factor. | Long-horizon fatigue is approximated by the tilt only. |
| Confidence calibration is fit on simulated data. | Displayed confidence is labeled "sim-calibrated". |
| One `noise_scale` cannot represent structured real nuisance (camera angle, occlusion by objects, other rooms). | The sensitivity table bounds the effect of noise level, not of nuisance type. |
| Not clinically validated; the states are not diagnoses. | Valence is a research prototype for consented self-monitoring. EU AI Act Art. 5(1)(f) prohibits emotion recognition in workplaces and education, and nothing here changes that. |

---

## Appendix A: full parameter table

Single source in code: `valence/sim/params.py` (`STATE_PARAMS`, `FP`, `NUISANCE`). This table is meant to be
copied into `docs/MODEL_CARD.md` as is. Per-state values are listed in the order
**Focused / Frustrated_Masked / Exhausted / Collaborative_Happy / Overwhelmed / Tense**.

Basis: **Plan** = `docs/PLAN.md` (state table, simulator section, ⚑ items); **Design** = chosen in this spec to
create overlap or realism; **Placeholder** = replaced from the live feature capture (section 10).

| Group | Parameter (identifier) | Value | Unit | ×ns | Basis |
|---|---|---|---|---|---|
| Session | sessions per corpus | ~300 | — | | Plan |
| Session | people per session | 1 / 2 / 3 with p = 0.4 / 0.4 / 0.2 | people | | Plan (1–3); Design (probabilities) |
| Session | `session_min` | U(60, 120) | min | | Plan |
| Session | tick length (`settings.tick_s`) | 5 | s | | Plan |
| Session | `start_hour` | U(9, 23) | h of day | | Design |
| Session | `start_hours_since_break` | U(0, 5) per person | h | | Design |
| States | active set | core 4, or all 6 with P1 | — | | Plan |
| States | dwell distribution | Gamma(shape 3, mean `dwell_min`), floor 2 min | min | | Plan (gamma, mean 4–10, min 2) |
| States | `dwell_min` | 9 / 6 / 8 / 6 / 5 / 5 | min | | Design within Plan band |
| States | fatigue tilt | `f(h) = 1 / (1 + exp(−(h − 3.0) / 0.5))` | — | | Plan (~3 h) |
| States | `fatigue_affinity` | −0.6 / +1.0 / +2.0 / −0.5 / +1.5 / +1.0 | — | | Design |
| States | `group_affinity` / `solo_affinity` | 1/1 · 1/1 · 1/1 · 1.8/0.4 · 1/1 · 1.5/0.6 | — | | Plan (Happy with ≥ 2; Tense often ≥ 2); Design (values) |
| States | transition weight floor | 0.05 | — | | Design |
| Breaks | `break_per_min` | 1/120, ×2 in Exhausted and Overwhelmed | per min | | Design (below the plan's ~1/90 so long stretches occur); ×2 Design |
| Breaks | `break_len_min` | U(10, 20), ≥ `break_absent_min` | min | | Design (matches live break rule) |
| Breaks | state after a break | Focused (alone); Focused or Collaborative_Happy 50/50 (≥ 2 people) | — | | Design |
| Face | `face_mu` | 0.00 / −0.12 / −0.04 / +0.50 / −0.20 / −0.35 | valence | | Plan (FM unmasked: Design); refit to capture run 1 |
| Face | `mask_mu` (forced smile) | +0.45 | valence | | Plan |
| Face | `mask_p` (Frustrated_Masked) | 0.70 core / 0.92 with P1 | — | | Plan (30% unmasked; ≥ 90% masked) |
| Face | `face_episode_sd` | 0.12 | valence | | Design; refit to capture run 1 |
| Face | `face_tick_sd` | 0.03 | valence | ✓ | Design; refit to capture run 1 |
| Face | low-light term | +0.35 × max(0, (200 − lux) / 200) added to tick σ | valence | | Plan (< 200 lux); Design (slope) |
| Face | `face_hidden_range` (session base) | U(0, 0.30) | share of ticks | | Plan; refit to capture run 1 |
| Face | head-in-hands hiding | +0.3 × `head_in_hands_frac` on the session share, cap 0.98 (not for Overwhelmed, whose episode share already covers it) | share | | Design |
| Face | Overwhelmed hidden share | U(0.40, 0.70) per episode | share of ticks | | Plan |
| Face | `face_visible_range` (`face_visible` when readable) | U(0.85, 1.0) | fraction | | Design; refit to capture run 1 |
| Sound | `db_target` | 48 / 66 / 45 / 63 / 66 / 70 | dB SPL | | Plan signatures (midpoints) |
| Sound | empty-room level | 40 | dB SPL | | Design |
| Sound | `room_tau_min` | U(2, 4) per session | min | ✓ | Plan |
| Sound | `db_noise` | 1.2 | dB (σ per tick) | ✓ | Plan; refit to capture run 1 |
| Sound | `db_drift` | AR(1) ×0.95 per tick, σ 0.5 | dB per tick | ✓ | Design (capture run 1) |
| Sound | transients (`sound_events_per_min`, `sound_event_db`, `sound_event_ticks`) | 0.3 per min, +4 to +12 dB (30%: −0.5× that), 2–8 ticks, state-independent | dB | | Design (capture run 1) |
| Sound | `mic_offset_db` | U(−12, +12) per session | dB | | Plan (±12–15) |
| Sound | laughter bump | +2 dB per laughter onset, at most 2 | dB | | Design |
| Sound | p90 | ambient + 1.5 + abs(N(0, 1)) + 1.5 if laughter | dB | ✓ (σ) | Design |
| Sound | `venue_db_range` (hall level, power-summed with the room) | U(40, 75) per session; scenarios none unless `venue_db` | dB SPL | | Design (`docs/CONTRACT_REQUESTS.md` round 4) |
| Sound | `neutral_db` (one neutral occupant, for the arrival baseline) | 48 | dB SPL | | Design (= Focused `db_target`) |
| Sound | `baseline_db_noise` (arrival capture error) | 1.5 | dB (σ) | ✓ | Design |
| Light | `lux_range` (session baseline) | log-uniform 100–800 | lux | | Design (`docs/CONTRACT_REQUESTS.md` round 4; was 120–600) |
| Light | `baseline_lux_noise` (arrival capture error) | lognormal σ 0.08 | log units | ✓ | Design |
| Light | time-of-day factor | ×0.85 from 22:00 to 06:00 | — | | Plan (drift); Design (value) |
| Light | `lux_mult` | ×1.00 / ×0.85 / ×0.65 / ×1.15 / ×0.85 / ×0.95 | — | | Plan (dim / bright signatures); Design (values) |
| Light | `lux_noise` (tick jitter) | lognormal σ 0.015 | log units | ✓ | Design; refit to capture run 1 |
| Light | `lux_drift` | AR(1) on log-lux, ×0.98 per tick, σ 0.01 | log units | ✓ | Design (capture run 1) |
| Light | transients (`light_events_per_min`, `light_event_mult`, `light_event_ticks`) | 0.5 per min, ×U(0.75, 1.3), 2–12 ticks, state-independent | — | | Design (capture run 1) |
| Events | `fist_rate` | 0 / 0.8 / 0 / 0 / 0.2 / 0.8 | per min | | Plan (0.8 bursty); Design (Overwhelmed 0.2) |
| Events | fist burst modulator | ×1.7 / ×0.3, equal time, switch p = 0.1 per tick | — | | Design (mean-preserving) |
| Events | `laugh_rate` | 0.03 / 0.10 / 0.02 / 0.50 / 0.02 / 0.05 | per min per person | | Plan (Happy 0.5); Design (others) |
| Events | `compl_rate` | 0.02 / 0.02 / 0.01 / 0.20 / 0.01 / 0.01 | per min per person | | Plan (Happy 0.2); Design (others) |
| Body | `slump_p` | 0.05 / 0.30 / 0.70 / 0.03 / 0.60 / 0.20 | share of time | | Design; scale Placeholder |
| Body | `hih_p` | 0.08 / 0.15 / 0.35 / 0.05 / 0.70 / 0.08 | share of time | | Plan; refit to capture run 1 |
| Body | slump / head-in-hands chain switching | 0.15 per tick (head-in-hands 0.35 on ticks with laughter) | — | | Design |
| Body | slump tick value | on U(0.7, 1.0); off 0, or U(0, 0.1) on 15% of ticks | fraction | | Design |
| Body | head-in-hands tick value | on U(0.5, 1.0), off 0 | fraction | | Design |
| Body | `motion` | 0.03 / 0.05 / 0.015 / 0.06 / 0.03 / 0.09 | shoulder-widths/s | | Design; scale Placeholder |
| Body | `motion_ar` / `motion_noise` | log-AR(1) toward the state mean, pull 0.25 per tick, innovation σ 0.18; ×0.7 while slumped | log units | ✓ (innovation) | Design; refit to capture run 1 |
| Body | still phases (`still_share`, `still_switch`, `still_motion_mult`) | share 0.4, switch 0.06 per tick, ×0.5 motion, in every state | — | | Design (capture run 1) |
| Body | `person_tau_min` | U(1, 2) per person | min | ✓ | Design |
| Body | withdrawal rule | slump ≥ 0.6 and motion ≤ 0.02 sw/s, as built (`features/spec.py`) | — | | Design (shared with live) |
| Detectors | `FP["fist_per_min"]` | 0.02 | per min | | Placeholder |
| Detectors | `FP["slump_share"]` | 0.02 | share of time | | Placeholder |
| Detectors | `FP["laugh_per_min"]` | 0.02 (room) | per min | | Placeholder |
| Detectors | `FP["compl_per_min"]` | 0.01 (room) | per min | | Placeholder |
| Missing | `dropout` | 0.02 per tick, per modality (audio per session; video per person) | probability | | Plan |
| Difficulty | `noise_scale` | `ns*` = TBD (section 12; recorded in `artifacts/sim_manifest.json`) | — | — | Plan (one knob, one criterion) |
| Difficulty | sensitivity | ×0.5 / ×1 / ×1.5 of `ns*` | — | — | Plan |
