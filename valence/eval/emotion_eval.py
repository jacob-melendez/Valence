"""Evaluation of the emotion map on held-out simulator sessions.

Metrics: macro-F1 and per-state F1, the row-normalized confusion matrix, top-label ECE before and after the
temperature, valence/arousal MAE and CCC (regressors, and the probability-weighted anchor baseline), face-only vs full
vs context-only, and two shuffled-context controls:

* *test-time*: the full model on test windows whose context features (everything but the face block) come from a
  random other test window. If the model really uses context, scores fall toward face-only.
* *trained*: a model trained on rows whose context was shuffled the same way. It can only learn face -> state, so it
  should land at face-only; landing higher would mean the context features leak the label some other way.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from valence.contracts import AFFECT_ANCHORS, EMOTION_STATES
from valence.features.emotion import CONTEXT_FEATURES, FACE_ONLY_FEATURES, INPUT_FEATURES
from valence.models.train_ladder import Split, softmax

STATES = EMOTION_STATES
ANCHORS = np.array([AFFECT_ANCHORS[s] for s in STATES])  # (K, 2)
REGION = ["Frustrated_Masked", "Tense", "Anxious", "Overwhelmed"]  # the shared negative, high-arousal region


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ccc(pred: np.ndarray, true: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mp, mt = pred.mean(), true.mean()
    vp, vt = pred.var(), true.var()
    cov = ((pred - mp) * (true - mt)).mean()
    denom = vp + vt + (mp - mt) ** 2
    return float(2 * cov / denom) if denom > 0 else math.nan


def va_metrics(pred_v: np.ndarray, pred_a: np.ndarray, true_v: np.ndarray, true_a: np.ndarray) -> dict:
    pv, pa = np.clip(pred_v, -1, 1), np.clip(pred_a, -1, 1)
    quad = (np.sign(pv) == np.sign(true_v)) & (np.sign(pa) == np.sign(true_a))
    return {"valence_mae": round(float(np.abs(pv - true_v).mean()), 4), "valence_ccc": round(ccc(pv, true_v), 4),
            "arousal_mae": round(float(np.abs(pa - true_a).mean()), 4), "arousal_ccc": round(ccc(pa, true_a), 4),
            "quadrant_acc": round(float(quad.mean()), 4)}


def nearest_anchor(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    d = (v[:, None] - ANCHORS[None, :, 0]) ** 2 + (a[:, None] - ANCHORS[None, :, 1]) ** 2
    return d.argmin(axis=1)


def cls_metrics(p: np.ndarray, y: np.ndarray, p_uncal: np.ndarray | None = None) -> dict:
    pred = p.argmax(axis=1)
    f1 = f1_score(y, pred, average=None, labels=range(len(STATES)), zero_division=0)
    out = {"macro_f1": round(float(f1.mean()), 4), "accuracy": round(float((pred == y).mean()), 4),
           "ece": round(ece(p, y), 4), "per_state_f1": {s: round(float(v), 4) for s, v in zip(STATES, f1)}}
    if p_uncal is not None:
        out["ece_uncalibrated"] = round(ece(p_uncal, y), 4)
    return out


def confusion(p: np.ndarray, y: np.ndarray) -> dict:
    cm = confusion_matrix(y, p.argmax(axis=1), labels=range(len(STATES))).astype(float)
    rows = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ri = [STATES.index(s) for s in REGION]
    sub = cm[np.ix_(ri, ri)]
    region_true = cm[ri].sum()
    return {"states": list(STATES), "row_normalized": [[round(float(v), 4) for v in r] for r in rows],
            "counts": cm.astype(int).tolist(),
            "region": {"states": REGION,
                       "predicted_inside_region": round(float(sub.sum() / max(region_true, 1)), 4),
                       "exact_within_region": round(float(np.trace(sub) / max(sub.sum(), 1)), 4)}}


def shuffle_context(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Context columns from a random other row (face block kept): breaks the context -> state link."""
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    perm = rng.permutation(len(X))
    Xs[CONTEXT_FEATURES] = X[CONTEXT_FEATURES].to_numpy()[perm]
    return Xs


class Models:
    """A trained classifier (+ temperature) and valence/arousal regressors on one feature set."""

    def __init__(self, cols, cls, temperature, reg_v, reg_a):
        self.cols, self.cls, self.t, self.reg_v, self.reg_a = list(cols), cls, float(temperature), reg_v, reg_a

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.cls.predict(X[self.cols], raw_score=True)

    def proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        return softmax(self.raw(X), self.t if calibrated else 1.0)

    def va(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (np.clip(self.reg_v.predict(X[self.cols]), -1, 1), np.clip(self.reg_a.predict(X[self.cols]), -1, 1))


def train_models(X, meta, y, split: Split, cols, *, augment: bool, seed: int) -> Models:
    from valence.models.emotion import train_classifier, train_regressor

    cls, t = train_classifier(X, y, split, cols, augment=augment, params={"seed": seed})
    reg_v = train_regressor(X, meta["valence"].to_numpy(), y, split, cols, augment=augment, params={"seed": seed})
    reg_a = train_regressor(X, meta["arousal"].to_numpy(), y, split, cols, augment=augment, params={"seed": seed})
    return Models(cols, cls, t, reg_v, reg_a)


def score(models: Models, X: pd.DataFrame, meta: pd.DataFrame, y: np.ndarray) -> dict:
    p = models.proba(X)
    pv, pa = models.va(X)
    tv, ta = meta["valence"].to_numpy(), meta["arousal"].to_numpy()
    anchor = p @ ANCHORS
    out = cls_metrics(p, y, models.proba(X, calibrated=False))
    out["va"] = va_metrics(pv, pa, tv, ta)
    out["va_from_probabilities"] = va_metrics(anchor[:, 0], anchor[:, 1], tv, ta)  # the served map position
    out["nearest_anchor_state_acc"] = round(float((nearest_anchor(pv, pa) == y).mean()), 4)
    return out


def history_breakdown(models: Models, X, meta, y) -> dict:
    """Macro-F1 by history length (warm-up)."""
    out = {}
    h = meta["hist_ticks"].to_numpy()
    for name, lo, hi in (("<=1m", 1, 12), ("1-5m", 13, 60), ("5-15m", 61, 180)):
        m = (h >= lo) & (h <= hi)
        if m.sum() > 50:
            p = models.proba(X[m])
            out[name] = round(float(f1_score(y[m], p.argmax(axis=1), average="macro")), 4)
    return out


def evaluate_all(X, meta, y, split: Split, *, seed: int, full: Models | None = None, log=print) -> tuple[dict, Models]:
    test = split.test
    Xt, mt, yt = X.loc[test].reset_index(drop=True), meta.loc[test].reset_index(drop=True), y[test]
    if full is None:
        full = train_models(X, meta, y, split, INPUT_FEATURES, augment=True, seed=seed)
    report: dict = {"full": score(full, Xt, mt, yt)}
    report["full"]["confusion"] = confusion(full.proba(Xt), yt)
    report["full"]["by_history"] = history_breakdown(full, Xt, mt, yt)
    log(f"  full          macro-F1={report['full']['macro_f1']:.3f}  V MAE={report['full']['va']['valence_mae']:.3f} "
        f"A MAE={report['full']['va']['arousal_mae']:.3f}", flush=True)
    # Face hidden at test time (modality dropout pays off here).
    Xh = Xt.copy()
    from valence.models.emotion import FACE_VIS
    face_cols = [c for c in FACE_ONLY_FEATURES if c not in FACE_VIS]
    Xh[face_cols] = np.nan
    Xh[FACE_VIS] = 0.0
    report["full_face_hidden"] = score(full, Xh, mt, yt)
    report["shuffled_context_test"] = score(full, shuffle_context(Xt, seed), mt, yt)
    log(f"  shuffled ctx  macro-F1={report['shuffled_context_test']['macro_f1']:.3f} (test-time)", flush=True)
    for name, cols in (("face_only", FACE_ONLY_FEATURES), ("context_only", CONTEXT_FEATURES)):
        m = train_models(X, meta, y, split, cols, augment=False, seed=seed)
        report[name] = score(m, Xt, mt, yt)
        log(f"  {name:13s} macro-F1={report[name]['macro_f1']:.3f}  V MAE={report[name]['va']['valence_mae']:.3f} "
            f"A MAE={report[name]['va']['arousal_mae']:.3f}", flush=True)
    Xs = X.copy()
    Xs.loc[split.train] = shuffle_context(X.loc[split.train].reset_index(drop=True), seed + 1).to_numpy()
    m = train_models(Xs, meta, y, split, INPUT_FEATURES, augment=False, seed=seed)
    report["shuffled_context_trained"] = score(m, Xt, mt, yt)
    log(f"  shuffled ctx  macro-F1={report['shuffled_context_trained']['macro_f1']:.3f} (trained)", flush=True)
    return report, full


def scenario_check(predictor, grace_s: float = 120.0) -> dict:
    """The exported models on the scripted replay scenarios: share of ticks (after a 2-min grace in each scripted
    segment) where the 10-state label equals the scripted state, and the mean map position per state."""
    from valence.contracts import CORE_STATES
    from valence.features.compute import PERSON_FIELDS, ROOM_FIELDS, compute
    from valence.features.emotion import EMOTION_FIELDS, compute_emotion, model_input
    from valence.sim.scenarios import SCENARIO_DIR, load_script, render

    out = {}
    fields = sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS))
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        script = load_script(path.stem)
        states = {s for seg in script.segments for s in seg.states if s is not None}
        df = render(path.stem, state_set="core" if states <= set(CORE_STATES) else "all")
        room = df[df["person_id"] == 0].sort_values("tick")
        rarr = {f: room[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
        rpos = {t: i for i, t in enumerate(room["tick"].to_numpy())}
        per_state: dict[str, list] = {}
        for _, p in df[df["person_id"] != 0].groupby("person_id"):
            p = p.sort_values("tick")
            arr = {f: p[f].to_numpy(dtype=float) for f in fields}
            ticks, truth, age = p["tick"].to_numpy(), p["true_state"].to_numpy(), p["episode_age_s"].to_numpy()
            rows, keep = [], []
            for i in range(len(p)):
                if age[i] < grace_s:
                    continue
                lo, rp = max(0, i - 179), rpos[ticks[i]]
                rows.append(model_input(compute({f: arr[f][lo:i + 1] for f in PERSON_FIELDS},
                                                {f: v[max(0, rp - 179):rp + 1] for f, v in rarr.items()}),
                                        compute_emotion({f: arr[f][lo:i + 1] for f in EMOTION_FIELDS})))
                keep.append(truth[i])
            if not rows:
                continue
            r = predictor.predict(np.vstack(rows))
            labels = [predictor.states[k] for k in r["proba"].argmax(axis=1)]
            for s, lab, v, a in zip(keep, labels, r["valence_mix"], r["arousal_mix"]):
                per_state.setdefault(s, []).append((lab == s, v, a, lab))
        out[path.stem] = {s: {"n": len(v), "agreement": round(float(np.mean([x[0] for x in v])), 4),
                              "valence": round(float(np.mean([x[1] for x in v])), 3),
                              "arousal": round(float(np.mean([x[2] for x in v])), 3),
                              "top_label": max(set(x[3] for x in v), key=[x[3] for x in v].count)}
                          for s, v in per_state.items()}
    return out


def robustness(predictor, sessions: int = 60, seed: int = 2027) -> dict:
    """The exported models on fresh test corpora under deployment-like shifts (simulated):

    * ``noise_scale`` 0.5 / 1.0 / 1.5 (every noise term of the simulator, face cues included);
    * face cues twice as noisy (episode, tick and neutral-baseline spread x2) than in training;
    * blink rates halved (browser face tracking below ~15 fps misses blinks);
    * pitch sign flipped (a transposed matrix read in perception).
    """
    from dataclasses import replace

    import valence.sim.generator as gen
    from valence.features.emotion_dataset import build_emotion
    from valence.sim.params import FACE_CUE_NUISANCE

    def run(noise_scale: float, cues=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        saved = gen.FACE_CUE_NUISANCE
        try:
            if cues is not None:
                gen.FACE_CUE_NUISANCE = cues
            df = gen.simulate_corpus(sessions, seed, state_set="emotion", noise_scale=noise_scale)
        finally:
            gen.FACE_CUE_NUISANCE = saved
        return build_emotion(df, seed=seed)

    def metrics(X: pd.DataFrame, meta: pd.DataFrame) -> dict:
        r = predictor.predict(X[predictor.cols].to_numpy())
        y = meta["label"].map({s: i for i, s in enumerate(predictor.states)}).to_numpy()
        tv, ta = meta["valence"].to_numpy(), meta["arousal"].to_numpy()
        return {"macro_f1": round(float(f1_score(y, r["proba"].argmax(axis=1), average="macro")), 4),
                "ece": round(ece(r["proba"], y), 4),
                "valence_mae": round(float(np.abs(r["valence_mix"] - tv).mean()), 4),
                "arousal_mae": round(float(np.abs(r["arousal_mix"] - ta).mean()), 4)}

    out = {}
    base_X, base_meta = run(1.0)
    out["noise_1.0"] = metrics(base_X, base_meta)
    for ns in (0.5, 1.5):
        out[f"noise_{ns}"] = metrics(*run(ns))
    noisy = replace(FACE_CUE_NUISANCE, episode_sd=2 * FACE_CUE_NUISANCE.episode_sd,
                    tick_sd=2 * FACE_CUE_NUISANCE.tick_sd, neutral_sd=2 * FACE_CUE_NUISANCE.neutral_sd,
                    neutral_deg_sd=2 * FACE_CUE_NUISANCE.neutral_deg_sd)
    out["face_cues_x2_noise"] = metrics(*run(1.0, noisy))
    Xb = base_X.copy()
    for c in ("blink_rate_1m", "blink_rate_5m", "blink_rate_15m"):
        Xb[c] = Xb[c] * 0.5
    out["blinks_halved"] = metrics(Xb, base_meta)
    Xp = base_X.copy()
    for c in ("pitch_now", "pitch_mean_1m", "pitch_mean_5m"):
        Xp[c] = -Xp[c]
    out["pitch_sign_flipped"] = metrics(Xp, base_meta)
    out["_note"] = f"{sessions} fresh sessions (seed {seed}), simulated; not real data"
    return out


def main() -> None:
    """``python -m valence.eval.emotion_eval [--robustness]`` writes artifacts/emotion/ci.json (and robustness.json)."""
    import argparse
    import json
    from pathlib import Path

    from valence.models.emotion import EmotionPredictor, clean_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/emotion")
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--robustness", action="store_true", help="also rerun the shifted-corpus checks (~5 min)")
    args = ap.parse_args()
    pred = EmotionPredictor(Path(args.dir))
    ap_ci = {"model_id": pred.model_id, **bootstrap_ci(pred)}
    (Path(args.dir) / "ci.json").write_text(json.dumps(clean_json(ap_ci), indent=1, allow_nan=False))
    print(json.dumps(ap_ci, indent=1))
    if args.robustness:
        rob = {"model_id": pred.model_id, **robustness(pred, sessions=args.sessions)}
        (Path(args.dir) / "robustness.json").write_text(json.dumps(clean_json(rob), indent=1, allow_nan=False))
        print(json.dumps(rob, indent=1))



def bootstrap_ci(predictor, n_boot: int = 1000, seed: int = 0) -> dict:
    """Session-bootstrap 95% intervals on the held-out test sessions: macro-F1 and map-position MAE of the exported
    model, of a face-only model retrained on the same split, and their paired difference."""
    from valence.models.emotion import labels, load_corpus

    info = predictor.meta["training"]
    from pathlib import Path

    X, meta = load_corpus(info["sessions"], info["seed"], info["noise_scale"], Path.home() / ".cache" / "valence-emotion")
    y = labels(meta)
    split = split_from_meta(meta, info["seed"])
    test = split.test
    Xt, mt, yt = X.loc[test].reset_index(drop=True), meta.loc[test].reset_index(drop=True), y[test]
    face = train_models(X, meta, y, split, FACE_ONLY_FEATURES, augment=False, seed=info["seed"])
    full_p = predictor.predict(Xt[predictor.cols].to_numpy())
    preds = {"full": (full_p["proba"].argmax(axis=1), full_p["valence_mix"], full_p["arousal_mix"])}
    fp = face.proba(Xt)
    fmix = fp @ ANCHORS
    preds["face_only"] = (fp.argmax(axis=1), fmix[:, 0], fmix[:, 1])
    tv, ta = mt["valence"].to_numpy(), mt["arousal"].to_numpy()
    sessions = mt["session_id"].to_numpy()
    uniq = np.unique(sessions)
    idx = {s: np.nonzero(sessions == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)

    def stats(rows: np.ndarray) -> dict:
        out = {}
        for name, (lab, v, a) in preds.items():
            out[name] = (f1_score(yt[rows], lab[rows], average="macro", labels=range(len(STATES)), zero_division=0),
                         np.abs(v[rows] - tv[rows]).mean(), np.abs(a[rows] - ta[rows]).mean())
        return out

    point = stats(np.arange(len(yt)))
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        draws.append(stats(rows))

    def ci(get) -> list[float]:
        vals = np.array([get(d) for d in draws])
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    out = {"test_sessions": int(len(uniq)), "n_boot": n_boot}
    for name in preds:
        for j, metric in enumerate(("macro_f1", "valence_mae", "arousal_mae")):
            out[f"{name}_{metric}"] = {"point": round(float(point[name][j]), 4), "ci95": ci(lambda d: d[name][j])}
    for j, metric in enumerate(("macro_f1", "valence_mae", "arousal_mae")):
        out[f"full_minus_face_{metric}"] = {
            "point": round(float(point["full"][j] - point["face_only"][j]), 4),
            "ci95": ci(lambda d: d["full"][j] - d["face_only"][j])}
    return out


def split_from_meta(meta, seed):
    from valence.models.train_ladder import split_sessions

    return split_sessions(meta, seed)


if __name__ == "__main__":
    main()
