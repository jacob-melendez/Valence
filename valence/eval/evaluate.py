"""Evaluation for the judging proofs -> reports/metrics.json and reports/figs/*.png.

    python -m valence.eval.evaluate [--artifacts artifacts] [--bootstrap 200]

Rebuilds the exact held-out test sessions from model_meta.json (same seed, same session split), then:
  * accuracy vs history available (proof 2's key chart), with session-bootstrap 95% CIs
  * confusion matrices M0 vs M2, Frustrated_Masked recall, the Focused paradox, accuracy by episode age
  * robustness to missing / degraded face evidence (proof 3), degrading only the face channel
  * calibration (reliability diagram, ECE before/after temperature)
  * the legacy mock dataset result, to show why the data was rebuilt
Every number here comes from simulator data generated under the assumptions in docs/DATASET.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from valence.eval import style
from valence.features.build_dataset import build
from valence.features.compute import FEATURE_SETS
from valence.models.train_ladder import softmax, split_sessions
from valence.sim.generator import simulate_corpus
from valence.sim.legacy import generate_legacy

HISTORY_TICKS = {0: 1, 1: 12, 2: 24, 5: 60, 10: 120, 15: 180}
LADDER = ["m0", "m0t", "m1", "m2_noface", "m2"]
VARIANTS = ["m2_noabsdb", "m2_notime"]
NEG_STATES = {"Frustrated_Masked", "Exhausted", "Tense", "Overwhelmed"}


def load_models(artifacts: Path, names: list[str]) -> dict[str, lgb.Booster]:
    return {n: lgb.Booster(model_file=str(artifacts / f"{n}.txt")) for n in names if (artifacts / f"{n}.txt").exists()}


def predict(models: dict[str, lgb.Booster], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {n: m.predict(X[FEATURE_SETS[n]], raw_score=True) for n, m in models.items()}


def boot_f1(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n: int, seed: int = 0) -> tuple[float, float, float]:
    point = float(f1_score(y, pred, average="macro"))
    if n <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        vals.append(f1_score(y[idx], pred[idx], average="macro"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
            table.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
                          "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return float(total), table


def degrade_face(df: pd.DataFrame, *, dropout: float = 0.0, extra_sd: float = 0.0, extra_hidden: float = 0.0,
                 seed: int = 0) -> pd.DataFrame:
    """Degrade only the face channel at tick level (lux and every context channel untouched)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    people = out["person_id"] != 0
    face = out.loc[people, "face_valence"].to_numpy(dtype=float).copy()
    vis = out.loc[people, "face_visible"].to_numpy(dtype=float).copy()
    if extra_sd:
        noisy = ~np.isnan(face)
        face[noisy] = np.clip(face[noisy] + rng.normal(0, extra_sd, noisy.sum()), -1, 1)
    hide = rng.random(face.size) < min(1.0, dropout + extra_hidden)
    face[hide], vis[hide] = np.nan, 0.0
    out.loc[people, "face_valence"] = face
    out.loc[people, "face_visible"] = vis
    return out


def legacy_result(seed: int) -> dict[str, float]:
    legacy = generate_legacy(2000, seed)
    y = legacy["true_emotional_state"].astype("category").cat.codes.to_numpy()
    full = ["ambient_db", "light_lux", "hours_since_break", "clenched_fists_count", "compliments_count",
            "social_withdrawal_flag", "facial_valence_snapshot"]
    out = {}
    for name, cols in {"face_only": ["facial_valence_snapshot"], "context_only": full[:-1], "full": full}.items():
        x = legacy[cols].to_numpy(dtype=float)
        m = lgb.train({"objective": "multiclass", "num_class": 4, "verbose": -1, "num_threads": 1},
                      lgb.Dataset(x[:1400], y[:1400]), num_boost_round=100)
        out[name] = round(float((m.predict(x[1400:]).argmax(1) == y[1400:]).mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    artifacts, reports = Path(args.artifacts), Path(args.reports)
    figs = reports / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((artifacts / "model_meta.json").read_text())
    states = meta["states"]
    s2i = {s: i for i, s in enumerate(states)}
    models = load_models(artifacts, LADDER + VARIANTS)

    df = simulate_corpus(meta["sessions"], meta["seed"], state_set=meta["state_set"], noise_scale=meta["noise_scale"])
    _, full_meta = build(df, seed=meta["seed"])  # same windows as training -> same split
    split = split_sessions(full_meta, meta["seed"])
    test_sessions = set(full_meta.loc[split.test, "session_id"])
    test_df = df[df["session_id"].isin(test_sessions)]
    print(f"test sessions: {len(test_sessions)} ({time.time() - t0:.0f}s)")
    metrics: dict = {"states": states, "test_sessions": len(test_sessions), "noise_scale": meta["noise_scale"],
                     "bootstrap": args.bootstrap, "models": {}}

    # 1) accuracy vs history available ------------------------------------------------------------
    curves: dict[str, list] = {n: [] for n in models if n in LADDER}
    for minutes, ticks in HISTORY_TICKS.items():
        X, m = build(test_df, step=6, seed=11, fixed_len=ticks)
        keep = (m["hist_ticks"] == ticks).to_numpy()
        X, m = X[keep], m[keep]
        y = m["label"].map(s2i).to_numpy()
        preds = predict({k: v for k, v in models.items() if k in curves}, X)
        for n, raw in preds.items():
            curves[n].append((minutes, *boot_f1(y, raw.argmax(1), m["session_id"].to_numpy(), args.bootstrap)))
    m3_path = reports / "m3_metrics.json"
    if m3_path.exists():  # M3 is trained separately (valence.models.temporal), on the same test sessions
        m3 = json.loads(m3_path.read_text())
        curves["m3"] = [(p["minutes"], p["macro_f1"], p["macro_f1"], p["macro_f1"]) for p in m3["accuracy_vs_history"]]
        metrics["models"]["m3"] = {"macro_f1_mixed_history": m3["test_macro_f1"], "device": m3["device"]}
    metrics["accuracy_vs_history"] = {n: [{"minutes": c[0], "macro_f1": round(c[1], 4), "ci95": [round(c[2], 4), round(c[3], 4)]}
                                          for c in pts] for n, pts in curves.items()}
    style.history_chart(curves, figs / "accuracy_vs_history.png")

    # 2) full-history test set: headline metrics, confusion, episode age, calibration ------------------
    X, m = build(test_df, step=3, seed=12, fixed_len=180)
    y = m["label"].map(s2i).to_numpy()
    preds = predict(models, X)
    for n, raw in preds.items():
        pred = raw.argmax(1)
        entry = {
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "recall": {s: round(float(r), 4) for s, r in zip(states, recall_score(y, pred, average=None, labels=range(len(states))))},
        }
        if "Focused" in s2i:
            foc = y == s2i["Focused"]
            entry["focused_misread_as_negative"] = round(float(np.isin(pred[foc], [s2i[s] for s in NEG_STATES if s in s2i]).mean()), 4)
        metrics["models"][n] = entry
    if {"m0", "m2"} <= preds.keys():
        cms = {n: confusion_matrix(y, preds[n].argmax(1), labels=range(len(states)), normalize="true") for n in ("m0", "m2")}
        style.confusion_pair(cms, states, figs / "confusion_m0_vs_m2.png")
        ages = m["episode_age_s"].to_numpy()
        bins = [0, 30, 60, 120, 300, np.inf]
        names = ["0-30 s", "30-60 s", "1-2 min", "2-5 min", "5+ min"]
        age_rows = []
        for lo, hi, label in zip(bins[:-1], bins[1:], names):
            sel = (ages >= lo) & (ages < hi)
            age_rows.append({"bin": label, "n": int(sel.sum()),
                             **{n: round(float((preds[n].argmax(1)[sel] == y[sel]).mean()), 4) for n in ("m0", "m1", "m2") if n in preds}})
        metrics["accuracy_by_episode_age"] = age_rows
        style.episode_age_chart(age_rows, figs / "acc_vs_episode_age.png")
        raw2 = preds["m2"]
        e_before, _ = ece(softmax(raw2, 1.0), y)
        e_after, table = ece(softmax(raw2, meta["temperatures"]["m2"]), y)
        metrics["calibration_m2"] = {"temperature": meta["temperatures"]["m2"], "ece_before": round(e_before, 4),
                                     "ece_after": round(e_after, 4), "reliability": table}
        style.reliability_chart(table, e_before, e_after, figs / "reliability.png")
        headline = {
            "frustrated_masked_recall": {n: metrics["models"][n]["recall"].get("Frustrated_Masked") for n in ("m0", "m2")},
            "focused_misread_as_negative": {n: metrics["models"][n].get("focused_misread_as_negative") for n in ("m0", "m2")},
            "macro_f1": {n: metrics["models"][n]["macro_f1"] for n in metrics["models"]},
        }
        metrics["headline"] = headline

    # 3) robustness: degrade only the face channel ----------------------------------------------------
    robust_models = {k: v for k, v in models.items() if k in ("m0", "m0t", "m2_noface", "m2")}
    rob: dict[str, dict[str, list]] = {"face_dropout": {}, "low_light": {}}
    for kind, levels in {"face_dropout": [0.0, 0.25, 0.5, 0.75, 1.0], "low_light": [1.0, 0.5, 0.25, 0.1]}.items():
        for level in levels:
            if kind == "face_dropout":
                ddf = degrade_face(test_df, dropout=level, seed=21)
            else:
                severity = {1.0: (0.0, 0.0), 0.5: (0.15, 0.10), 0.25: (0.30, 0.25), 0.1: (0.45, 0.45)}[level]
                ddf = degrade_face(test_df, extra_sd=severity[0], extra_hidden=severity[1], seed=22)
            Xd, md = build(ddf, step=6, seed=13, fixed_len=180)
            yd = md["label"].map(s2i).to_numpy()
            for n, raw in predict(robust_models, Xd).items():
                rob[kind].setdefault(n, []).append((level, round(float(f1_score(yd, raw.argmax(1), average="macro")), 4)))
    metrics["robustness"] = rob
    style.robustness_chart(rob["face_dropout"], "face_dropout", figs / "robustness_face_dropout.png")
    style.robustness_chart(rob["low_light"], "low_light", figs / "robustness_lux.png")

    # 4) legacy mock data ------------------------------------------------------------------------------
    metrics["legacy_mock"] = legacy_result(meta["seed"])
    metrics["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["disclosure"] = ("All results are on simulator data generated under the published behavioral assumptions "
                             "in docs/DATASET.md; they test the architecture, not real-world accuracy.")
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics.get("headline", {}), indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {reports / 'metrics.json'}")


if __name__ == "__main__":
    main()
