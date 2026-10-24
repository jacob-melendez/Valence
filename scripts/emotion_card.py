"""Render docs/MODEL_CARD.md section 9 ("Emotion map") from the artifacts, so its numbers cannot drift from them.

    python scripts/emotion_card.py            # rewrites section 9 in place (it is the card's last section)
    python scripts/emotion_card.py --print    # prints it instead

Reads artifacts/emotion/{meta,metrics,robustness,ci,real_captures}.json, golden.json, experiments/*.json, and the
cached training corpus (for the attribution shares and the per-state map positions; ~1 min).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from valence.contracts import AFFECT_ANCHORS, EMOTION_DISPLAY  # noqa: E402
from valence.features.emotion import CONTEXT_FEATURES, EMOTION_FEATURE_NAMES  # noqa: E402
from valence.models.emotion import EmotionPredictor, labels, load_corpus  # noqa: E402
from valence.models.train_ladder import split_sessions  # noqa: E402

ART = ROOT / "artifacts" / "emotion"
CARD = ROOT / "docs" / "MODEL_CARD.md"
HEADER = "## 9. Emotion map (Hack OHI/O)"
SHORT = {"Focused": "Foc", "Frustrated_Masked": "Fru", "Exhausted": "Exh", "Collaborative_Happy": "Col",
         "Overwhelmed": "Ovw", "Tense": "Ten", "Bored": "Bor", "Confused": "Cnf", "Calm": "Clm", "Anxious": "Anx"}
NICE = {"Frustrated_Masked": "Frustrated", "Collaborative_Happy": "Collaborative"}


def j(name: str) -> dict:
    return json.loads((ART / name).read_text())


def f3(v: float) -> str:
    return f"{v:.3f}"


def ci(c: dict, key: str) -> str:
    lo, hi = c[key]["ci95"]
    return f"{c[key]['point']:.3f} ({lo:.3f}–{hi:.3f})"


def sci(c: dict, key: str) -> str:
    lo, hi = c[key]["ci95"]
    return f"{c[key]['point']:+.3f} ({lo:+.3f} to {hi:+.3f})"


def corpus_stats(pred: EmotionPredictor) -> dict:
    t = pred.meta["training"]
    X, m = load_corpus(t["sessions"], t["seed"], t["noise_scale"], Path.home() / ".cache" / "valence-emotion")
    y = labels(m)
    split = split_sessions(m, t["seed"])
    Xt, yt = X.loc[split.test, pred.cols].to_numpy(), y[split.test]
    r = pred.predict(Xt)
    pos = {}
    for k, s in enumerate(pred.states):
        sel = yt == k
        pos[s] = (float(r["valence_mix"][sel].mean()), float(r["arousal_mix"][sel].mean()))
    idx = np.random.default_rng(0).choice(len(Xt), 3000, replace=False)
    c = pred.contrib(Xt[idx])["classifier"][:, :, :-1]
    imp = np.abs(c).mean(axis=(0, 1))
    emo, ctx = set(EMOTION_FEATURE_NAMES), set(CONTEXT_FEATURES)
    share = {"context": imp[[i for i, n in enumerate(pred.cols) if n in ctx]].sum() / imp.sum(),
             "cues": imp[[i for i, n in enumerate(pred.cols) if n in emo]].sum() / imp.sum()}
    share["face"] = 1.0 - share["context"] - share["cues"]
    top = [pred.cols[i] for i in np.argsort(-imp)[:7]]
    k = pred.states.index("Confused")
    conf = [pred.cols[i] for i in np.argsort(-np.abs(c[:, k, :]).mean(axis=0))[:6]]
    return {"pos": pos, "share": share, "top": top, "confused_top": conf}


def render() -> str:
    meta, met, rob, cis, real = j("meta.json"), j("metrics.json"), j("robustness.json"), j("ci.json"), j(
        "real_captures.json")
    golden = j("golden.json")
    mix_exp = j("experiments/noise_mix_378fa16c.json")
    reg_exp = j("experiments/regressor_tuning_378fa16c.json")
    pred = EmotionPredictor(ART)
    mid = meta["model_id"]
    for name, d in (("metrics", met), ("robustness", rob), ("ci", cis), ("real_captures", real), ("golden", golden)):
        got = d.get("model_id", d.get("model"))
        assert got == mid, f"{name}.json is for {got}, meta.json for {mid}: regenerate it"
    st = corpus_stats(pred)
    tr = meta["training"]
    sz = tr["sizes"]
    full, face, ctx = met["full"], met["face_only"], met["context_only"]
    hid, sh_t, sh_tr = met["full_face_hidden"], met["shuffled_context_test"], met["shuffled_context_trained"]
    sw = tr["split_windows"]
    n_test_sessions = cis["test_sessions"]

    def va(m: dict, key: str = "va_from_probabilities") -> str:
        v = m[key]
        return f"{f3(v['valence_mae'])} / {v['valence_ccc']:.2f} | {f3(v['arousal_mae'])} / {v['arousal_ccc']:.2f} | {v['quadrant_acc']:.2f}"

    out = [HEADER, ""]
    out += [
        "> Added during Hack OHI/O (Oct 24–25, 2026). **Every number in this section comes from simulated data**, generated",
        "> under the assumptions in [EMOTION_FEATURES.md](EMOTION_FEATURES.md) §7 and [DATASET.md](DATASET.md). The",
        "> valence–arousal anchors are assumptions, not measurements. No real labeled data has been evaluated yet (§9.6).",
        "> The served desktop models above (M2/M0, `artifacts/`, `artifacts/all/`) are unchanged.",
        "",
        "Sources: `artifacts/emotion/meta.json` (model), `metrics.json` (held-out evaluation, scenario replays),",
        "`robustness.json` (shifted test corpora), `ci.json` (bootstrap intervals), `real_captures.json`, `experiments/`.",
        "This section is generated from them by `python scripts/emotion_card.py`. Regenerate the artifacts with",
        "`python -m valence.models.emotion` (≈ 12 min on 4 cores), `python -m valence.eval.emotion_eval --robustness`,",
        "`python scripts/emotion_export.py`, `python scripts/emotion_tour.py`, `python scripts/emotion_figures.py` and",
        "`python -m valence.eval.emotion_capture data/real/<capture>.jsonl ...`.",
        "",
        "### 9.1 Model details",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Version id | **`{mid}`**: first 8 hex of the SHA-256 of the three JSON dumps (`meta.json` `model_id`) |",
        f"| Models | 10-state LightGBM classifier (`multiclass`, 15 leaves, learning rate 0.1, ≤ 280 rounds → {sz['classifier']['rounds']}; class weights n/(k·count)); LightGBM valence and arousal regressors (L2, 31 leaves, learning rate 0.1, early stop → {sz['valence']['rounds']} and {sz['arousal']['rounds']} rounds, unweighted). `deterministic=True` |",
        f"| Input | {meta['n_features']} features (`meta.json` `input_features`): the served M2 set (101 of the frozen 123, room-relative) + the 31 `emotion-v1` face-cue features (brow lowering, eye openness, frown, smile, head yaw and pitch, looking away, blink rate; `valence/features/emotion.py`). NaN = missing (how the trees read it: EMOTION_FEATURES.md §3); no input clipping |",
        f"| Output | 10 calibrated probabilities (`softmax(raw / T)`, T = {meta['temperature']:.3f} fitted by NLL on the calibration sessions); **map position = probability-weighted anchors** `Σ p_k · anchor_k` (served); regressor valence/arousal (exported for comparison); `pred_contrib` (TreeSHAP) per output |",
        f"| Training data | Simulator, state set `emotion`: {tr['sessions']} sessions (seed {tr['seed']}), {tr['windows']:,} windows (random 1–180 tick history, every 6 ticks); session split 70 / 7.5 / 7.5 / 15 (train / early stop / calibration / test = {sw['train']:,} / {sw['early']:,} / {sw['calib']:,} / {sw['test']:,} windows; {n_test_sessions} test sessions) |",
        "| Modality dropout | as M2: +30% training copies with the whole face block hidden, +20% degraded (face valence ± 0.25, cues ± 0.08, pose ± 6°, blink rate × U(0.5, 1.5), visible share × U(0.3, 0.8)) |",
        f"| Size | `classifier.json.gz` {sz['classifier']['json_gz_bytes'] / 1e6:.2f} MB, `valence.json.gz` {sz['valence']['json_gz_bytes'] / 1e6:.2f} MB, `arousal.json.gz` {sz['arousal']['json_gz_bytes'] / 1e6:.2f} MB (gzipped `dump_model()` JSON); the `.txt.gz` files are the same models in LightGBM's text format, which Python loads and the web engine reads first (the JSON dumps are its fallback) |",
        f"| Parity | `golden.json`: {len(golden['vectors'])} input vectors with Python's raw scores, probabilities, both V/A outputs and `pred_contrib`; {len(golden['windows'])} feature windows (3 of them missing-data edge cases); {len(golden['frames'])} face-cue frames (blink edge cases included). `valence/models/emotion_json.py` reproduces LightGBM's raw scores from the JSON alone, exactly, including NaN, ±inf, huge and near-zero inputs (`tests/test_emotion_models.py`). The web engine's port (Cloud A, `frontend/src/engine/emotion-model.ts`, fixture on this build) reproduces raw scores and regressor outputs bit for bit, probabilities within 3 ulp and the map position within 4e-16, and replays the 600-tick tour end to end within 1e-9 with identical labels |",
        "",
        "**States and anchors** (class order; anchors from the plan's circumplex, **assumptions**):",
        "",
        "| State | Valence | Arousal | | State | Valence | Arousal |",
        "|---|---|---|---|---|---|---|",
    ]
    states = pred.states
    for a, b in zip(states[:5], states[5:]):
        out.append(f"| {EMOTION_DISPLAY[a]} | {AFFECT_ANCHORS[a][0]:+.1f} | {AFFECT_ANCHORS[a][1]:+.1f} | | "
                   f"{EMOTION_DISPLAY[b]} | {AFFECT_ANCHORS[b][0]:+.1f} | {AFFECT_ANCHORS[b][1]:+.1f} |")
    out += [
        "",
        "In the simulator, an episode's true valence/arousal is its state's anchor plus N(0, 0.10) per axis, so the",
        "regression targets are the anchors by construction, with a little within-state spread.",
        "",
        f"### 9.2 Held-out results ({n_test_sessions} test sessions, {sw['test']:,} windows, up to 15 min of history)",
        "",
        "| Model (features) | Macro-F1 | Accuracy | ECE | Valence MAE / CCC | Arousal MAE / CCC | Quadrant |",
        "|---|---|---|---|---|---|---|",
        f"| **Full (132), map position = Σ p·anchor (served)** | **{f3(full['macro_f1'])}** | {f3(full['accuracy'])} | {f3(full['ece'])} ({f3(full['ece_uncalibrated'])} before T) | {va(full)} |",
        f"| Full (132), regressors | — | — | — | {va(full, 'va')} |",
        f"| Face only (38: the 7 face features + the 31 cues) | {f3(face['macro_f1'])} | {f3(face['accuracy'])} | {f3(face['ece'])} | {va(face)} |",
        f"| Context only (94) | {f3(ctx['macro_f1'])} | {f3(ctx['accuracy'])} | {f3(ctx['ece'])} | {va(ctx)} |",
        f"| Full, face hidden on every test window | {f3(hid['macro_f1'])} | {f3(hid['accuracy'])} | {f3(hid['ece'])} | {va(hid)} |",
        f"| Full, context from a random other test window | {f3(sh_t['macro_f1'])} | {f3(sh_t['accuracy'])} | {f3(sh_t['ece'])} | {va(sh_t)} |",
        f"| Trained on shuffled context (sanity check) | {f3(sh_tr['macro_f1'])} | {f3(sh_tr['accuracy'])} | {f3(sh_tr['ece'])} | {va(sh_tr)} |",
        "",
        "Valence/arousal columns are the map position (probability-weighted anchors) except in the regressor row.",
        f"Session-bootstrap 95% intervals (`ci.json`, {n_test_sessions} test sessions, {cis['n_boot']} resamples): full",
        f"macro-F1 {ci(cis, 'full_macro_f1')}, face only {ci(cis, 'face_only_macro_f1')}; paired difference",
        f"**{sci(cis, 'full_minus_face_macro_f1')}**; map-position MAE difference (full − face only) valence",
        f"{sci(cis, 'full_minus_face_valence_mae')}, arousal {sci(cis, 'full_minus_face_arousal_mae')}.",
        "",
        "Reading:",
        "",
        f"- **Context adds {cis['full_minus_face_macro_f1']['point']:+.3f} macro-F1 over the face alone** ({f3(face['macro_f1'])} → "
        f"{f3(full['macro_f1'])}; the interval excludes 0) and improves both axes. The face cues carry most of the signal",
        "  for the four new states, by the simulator's design.",
        f"- **The model really uses context:** with context taken from a random other window it drops to {f3(sh_t['macro_f1'])}",
        f"  and its calibration breaks (ECE {f3(sh_t['ece'])}).",
        f"- The model trained on shuffled context lands at the face-only level ({f3(sh_tr['macro_f1'])} vs {f3(face['macro_f1'])}),",
        "  as it must by construction: shuffling cuts every context–label link in training. It is a pipeline sanity check,",
        "  **not evidence about leakage** (a planted label leak in a context feature leaves it unchanged). The guards",
        "  against leakage are the session split, the no-lookahead tests and the simulator's documented assumptions.",
        f"- **Modality dropout works as intended:** with the face hidden, the full model falls back to the context-only level",
        f"  ({f3(hid['macro_f1'])} vs {f3(ctx['macro_f1'])}).",
        f"- **The regressors lose to the probability-weighted anchors** on both axes and both metrics (valence MAE",
        f"  {f3(full['va']['valence_mae'])} vs {f3(full['va_from_probabilities']['valence_mae'])}). Tuning did not close the gap on the",
        f"  previous build (`experiments/regressor_tuning_378fa16c.json`: class weights off, other learning rates and leaf",
        f"  counts, valence MAE {min(r['valence']['mae'] for r in reg_exp['runs']):.3f}–{max(r['valence']['mae'] for r in reg_exp['runs']):.3f} against",
        f"  {reg_exp['anchor_mixture_same_split']['valence']['mae']:.3f} for the mixture). The within-state spread shows only weakly in the",
        "  simulated cues, so a state-mixture estimate is close to optimal here. The map position is therefore",
        "  `Σ p_k · anchor_k`; the regressors stay exported for real-data retraining.",
        f"- **Calibration** is good in distribution (ECE {f3(full['ece'])}); it degrades under shift (§9.4).",
        f"- Warm-up: macro-F1 {f3(full['by_history']['<=1m'])} with ≤ 1 min of history, {f3(full['by_history']['1-5m'])} at 1–5 min,",
        f"  {f3(full['by_history']['5-15m'])} at 5–15 min.",
        "",
        "![Emotion map on held-out simulated sessions](../artifacts/emotion/figs/affect_map.png)",
        "",
        "*Hollow: assumed anchors. Filled: mean served map position per true state, ±1 SD shaded",
        "(`scripts/emotion_figures.py`).* The probability-weighted position is pulled toward the centre whenever the",
        "classifier is unsure, so the states' means sit between their anchors and the centre;",
    ]
    pos = st["pos"]
    shifts = sorted(((s, pos[s][0] - AFFECT_ANCHORS[s][0], pos[s][1] - AFFECT_ANCHORS[s][1]) for s in states),
                    key=lambda t: -(t[1] ** 2 + t[2] ** 2))
    out.append("largest pulls " + ", ".join(f"{NICE.get(s, s)} ({dv:+.2f} V, {da:+.2f} A)" for s, dv, da in shifts[:4])
               + "; the four negative, high-arousal states overlap in one cloud. A point on the map is a hedged")
    out += ["summary, not a measurement.", ""]
    f1, f1f = full["per_state_f1"], face["per_state_f1"]
    order = sorted(states, key=lambda s: -f1[s])
    out.append("**Per-state F1** (full / face only): " + ", ".join(
        f"{NICE.get(s, s)} {f1[s]:.2f} / {f1f[s]:.2f}" for s in order) + ".")
    conf = full["confusion"]
    rows = np.array(conf["row_normalized"])
    weakest = order[-1]
    wi = states.index(weakest)
    off = sorted(((rows[wi, k], states[k]) for k in range(len(states)) if k != wi), reverse=True)[:2]
    out += [f"The weakest is **{NICE.get(weakest, weakest)}**, mistaken for " + " and ".join(
        f"{NICE.get(s, s)} ({v:.0%})" for v, s in off) + ".", ""]
    out += ["**Confusion** (full model, rows = true state, row-normalized; · < 0.005):", "",
            "| True ↓ / predicted → | " + " | ".join(SHORT[s] for s in states) + " |",
            "|" + "---|" * (len(states) + 1)]
    for i, s in enumerate(states):
        cells = [f"**{v:.2f}**" if k == i else (f"{v:.2f}" if v >= 0.005 else "·") for k, v in enumerate(rows[i])]
        out.append(f"| {s} ({SHORT[s]}) | " + " | ".join(cells) + " |")
    reg = conf["region"]
    REGION = reg["states"]
    face_f1, ctx_f1 = face["per_state_f1"], ctx["per_state_f1"]
    fru_col = rows[states.index("Frustrated_Masked"), states.index("Collaborative_Happy")]
    out += [
        "",
        f"- **The shared negative, high-arousal region (Frustrated / Tense / Anxious / Overwhelmed):** {reg['predicted_inside_region']:.0%} of",
        f"  its windows are predicted inside the region and {reg['exact_within_region']:.0%} of those on the right state. They are",
        "  told apart mostly by the face cues (brow lowering and frown for Tense, the forced smile over lowered brows for",
        "  Frustrated, blink rate and glancing around for Anxious, a hidden face and a lowered head for Overwhelmed; the",
        f"  face-only model's F1 on these four is {', '.join(f'{face_f1[s]:.2f}' for s in REGION)} against "
        f"{', '.join(f'{ctx_f1[s]:.2f}' for s in REGION)} for context only),",
        "  with context adding the rest (fists and a loud room, a quiet room, head in hands); not by valence–arousal:",
        "  their anchors are within 0.3 of each other on each axis.",
        f"- Masked frustration still reads as Collaborative {fru_col:.0%} of the time: the forced smile shows in `smile` as",
        "  intended.",
        "",
    ]
    scen = met["scenarios"]
    low = [(name, s, v) for name, d in scen.items() for s, v in d.items() if v["agreement"] < 0.85]
    out += [
        "**Scripted scenarios** (`metrics.json` `scenarios`; the existing replay scenarios, ticks after a 2-min grace in",
        "each segment): every scripted state reads as scripted on ≥ 0.85 of its ticks except " + "; ".join(
            f"`{name}` {NICE.get(s, s)} {v['agreement']:.2f} ({v['n']} ticks"
            + (f", mostly read as {NICE.get(v['top_label'], v['top_label'])})" if v["top_label"] != s else ")")
            for name, s, v in low) + ". The repo has no replay scenarios for the four new states (`scenarios/` is",
        "outside this track); `artifacts/emotion/tour.json.gz` is a 50-min scripted walk through all ten states.",
        "",
        "### 9.3 Explanations",
        "",
        "`pred_contrib` (TreeSHAP) is exported for all three models, and `meta.json` `layers` maps each input to a layer",
        "(the 31 cues → Facial Surface). As with M2, the attributions explain the model, which learned the simulator's",
        "assumptions; they are not evidence about a person.",
        "",
        f"What the classifier leans on (mean |TreeSHAP| over all classes, 3 000 test windows): the 94 context features carry",
        f"{st['share']['context']:.0%} of the attribution, the 31 new cues {st['share']['cues']:.0%} and the 7 face-valence features"
        f" {st['share']['face']:.0%}. Top inputs: " + ", ".join(f"`{n}`" for n in st["top"]) + ". For Confused: "
        + ", ".join(f"`{n}`" for n in st["confused_top"]) + ".",
        "",
        f"### 9.4 Robustness (`robustness.json`, {rob['_note'].split(' (')[0]})",
        "",
        "| Condition | Macro-F1 | ECE | Valence MAE | Arousal MAE |",
        "|---|---|---|---|---|",
    ]
    labels_ = [("noise_1.0", "Same simulator, new sessions"),
               ("noise_0.5", "Simulator `noise_scale` × 0.5 (per-tick noise, session baselines, lags)"),
               ("noise_1.5", "Simulator `noise_scale` × 1.5 (same terms)"),
               ("face_cues_all_spreads_x2", "Face cues: every spread × 2 (episode, tick, frame, neutral baseline; blendshapes, pose, blinks, looking away)"),
               ("face_cues_between_person_x1.5", "Face cues: between-person and between-episode spreads × 1.5"),
               ("blinks_halved", "Blink rates halved (low face fps in the browser)"),
               ("pitch_sign_flipped", "**Pitch sign flipped** (matrix read transposed)")]
    for key, lab in labels_:
        v = rob[key]
        out.append(f"| {lab} | {f3(v['macro_f1'])} | {f3(v['ece'])} | {f3(v['valence_mae'])} | {f3(v['arousal_mae'])} |")
    base_ece = rob["noise_1.0"]["ece"]
    eces = {k: rob[k]["ece"] for k, _ in labels_[1:]}
    worst = max(eces, key=eces.get)
    allx2 = rob["face_cues_all_spreads_x2"]["macro_f1"]
    out += [
        "",
        "`noise_scale` widens only the per-tick terms, the session baselines and the lags; the episode- and person-level spreads, which make",
        "up most of the cue variance, are stressed by the two face-cue rows. Against the same fresh corpus (ECE",
        f"{f3(base_ece)}), ECE is {f3(min(eces.values()))}–{f3(max(eces.values()))} under these shifts, up to "
        f"{max(eces.values()) / base_ece:.1f}× with {dict(labels_)[worst].split(' (')[0].lower().replace('**', '')}; less noise also",
        "miscalibrates (the model is then underconfident). The model shrugs off under-counted blinks; it is sensitive",
        f"to the pitch sign and to face cues noisier than simulated (macro-F1 {f3(allx2)} with every spread doubled).",
        "Perception must pass the \"nodding down decreases pitch\" check in EMOTION_FEATURES.md §1.1 before the map is",
        "trusted, and real cue spreads should be measured (`valence.eval.emotion_capture`) before the numbers above are",
        "quoted for people.",
        "",
    ]
    sm, mx = mix_exp["served_single_noise"], mix_exp["noise_mix"]
    out += [
        "**Domain randomization, tried and not adopted** (on the previous build `emotion-v1@378fa16c`,",
        "`experiments/noise_mix_378fa16c.json`; not rerun). The same recipe trained on equal shares of `noise_scale` 0.75,",
        f"1.0, 1.25 and 1.5 scored macro-F1 {f3(mx['noise_1.0']['macro_f1'])} / {f3(mx['noise_0.5']['macro_f1'])} / {f3(mx['noise_1.5']['macro_f1'])} at noise",
        f"1.0 / 0.5 / 1.5 (single noise: {f3(sm['noise_1.0']['macro_f1'])} / {f3(sm['noise_0.5']['macro_f1'])} / {f3(sm['noise_1.5']['macro_f1'])}), with ECE",
        f"{f3(mx['noise_1.0']['ece'])} in distribution ({f3(sm['noise_1.0']['ece'])}) and {f3(mx['noise_1.5']['ece'])} at noise 1.5 ({f3(sm['noise_1.5']['ece'])}). About one point",
        "of macro-F1 under shift: kept as an option (`--noise-mix`) for a retrain once real captures show how noisy the",
        "cues are.",
        "",
        "### 9.5 Limits (read these before the numbers)",
        "",
        "- **Simulated training and test data.** The face-cue values per state (EMOTION_FEATURES.md §7) are the hypothesis,",
        "  written before any measurement. The face-only score shows how strongly the cues were set, not how expressive",
        "  people are.",
        "- **The anchors are assumptions**, and the regression targets are built from them. The map position cannot be more",
        "  right than the anchors are; self-reports (§9.6) are the first check.",
        "- **Frustrated, Tense, Anxious and Overwhelmed share a region** of the plane (anchors within 0.3 on each axis); a",
        "  point there says \"negative, aroused\", and the state label comes from the face cues and context.",
        f"- **{NICE.get(weakest, weakest)} is weak** (F1 {f1[weakest]:.2f}).",
        "- **Perception assumptions:** blendshapes relative to a 20 s neutral baseline (taken while the person may not be",
        "  neutral), blink detection needs ≥ 15 face fps, and the pitch sign must be right (§9.4). Head pose is relative to",
        "  the starting pose, so a camera moved mid-session shifts it; a head drooping past 20° counts as looking away.",
        "- **No demographic or fairness evaluation**, as for the rest of the system (§7); MediaPipe's own model cards apply.",
        "- **Calibration is sim-calibrated**; ECE rises under the shifts above.",
        "- Intended use is the same as M2's (§2): a demonstration of context-aware affect estimation with explanations, not",
        "  a measurement of a person, never for workplace or school monitoring.",
        "",
        "### 9.6 Real data",
        "",
        "- **Self-report sessions** (guaranteed track): `python -m valence.eval.self_report <taps.jsonl>` scores taps from",
        "  the console's capture (format and protocol: EMOTION_FEATURES.md §6; features and taps only, never media): MAE,",
        "  CCC and quadrant agreement for the map position, the regressors, a constant (0, 0) guess and the leave-one-out",
        "  mean tap, per person and for taps with a warm window; duplicate taps from repeated downloads are skipped.",
        "  **Status: no real taps yet**; the evaluator has only run on a synthetic simulator fixture",
        "  (`tests/test_emotion_models.py`), whose numbers say nothing about people.",
        "- **Existing real captures** (one person, one desk, recorded before the event, so **without the new face cues**:",
        "  the 31 cue features are NaN, blink rates included, and the model runs on context + face valence;",
        f"  `python -m valence.eval.emotion_capture`, `real_captures.json`). Time of day is fixed at 12:00: recordings carry",
        "  no time zone, and the time features are model inputs, so another hour would move these readings somewhat.",
        "  The first minute is skipped where the recording is longer than that:",
        "",
        "  | Capture | Scored ticks | Emotion map reads | Map position (V, A) |",
        "  |---|---|---|---|",
    ]
    names = {"capture_typing": "Quiet typing (acted Focused)", "capture_keyboard_gaze": "Keyboard gaze (acted Focused)",
             "capture_baseline": "Baseline (acted Focused)", "capture_slump": "Cued slumps", "capture_fists": "Cued fists",
             "capture_audio": "Laughter and compliments"}
    for key, lab in names.items():
        e = real["recordings"].get(key)
        if not e or not e.get("scored_ticks"):
            continue
        reads = ", ".join(f"{NICE.get(s, s)} {v:.0%}" for s, v in e["predicted_share"].items())
        n = f"{e['scored_ticks']}" + ("" if e["warmup_skipped"] else " (all warm-up: shorter than 1 min)")
        mp = e["map_position"]
        out.append(f"  | {lab} | {n} | {reads} | ({mp['valence']:+.2f}, {mp['arousal']:+.2f}) |")
    typing = real["recordings"]["capture_typing"]["predicted_share"]
    stress = sum(v for s, v in typing.items() if s in ("Frustrated_Masked", "Tense", "Anxious", "Overwhelmed"))
    out += [
        "",
        f"  As a negative control, quiet typing reads a negative, high-arousal state on {stress:.0%} of ticks (gate ≤ 20%), but",
        "  this is in-sample for the simulator's refit and says nothing about the four new states or the cues themselves.",
        "- **Public dataset** (stretch): **not attempted.** DAiSEE releases its clips only after a request form, its host is",
        "  not reachable from the event's cloud sessions, and it is raw video, which this track does not write to disk.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    section = render()
    if args.print:
        print(section)
        return
    text = CARD.read_text()
    i = text.index(HEADER)
    CARD.write_text(text[:i] + section)
    print(f"{CARD}: section 9 rewritten ({len(section.splitlines())} lines)")


if __name__ == "__main__":
    main()
