"""Train the model ladder on simulator data and write artifacts/.

    python -m valence.models.train_ladder --sessions 300 --seed 7 --noise-scale 1.0 --state-set core

Ladder (feature sets in valence/features/compute.py): m0 face snapshot, m0t face history, m1 context
snapshot, m2_noface, m2_noabsdb, m2_notime and m2 (served). Splits are by session (70/15/15); the
validation sessions are halved into an early-stopping half and a calibration half, and each model gets
a softmax temperature fitted on the calibration half.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from valence.contracts import STATE_SETS
from valence.features.build_dataset import build
from valence.features.compute import FEATURE_NAMES, FEATURE_SETS
from valence.sim.generator import simulate_corpus

PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4, seed=7)


@dataclass
class Split:
    train: np.ndarray
    early: np.ndarray
    calib: np.ndarray
    test: np.ndarray


def split_sessions(meta: pd.DataFrame, seed: int = 0) -> Split:
    sessions = np.array(sorted(meta["session_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    n = len(sessions)
    a, b, c = int(0.70 * n), int(0.775 * n), int(0.85 * n)
    groups = {"train": sessions[:a], "early": sessions[a:b], "calib": sessions[b:c], "test": sessions[c:]}
    mask = {k: meta["session_id"].isin(v).to_numpy() for k, v in groups.items()}
    return Split(**mask)


def class_weights(y: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(y, minlength=k).astype(float)
    w = counts.sum() / (k * np.maximum(counts, 1))
    return w[y]


def train_one(X: pd.DataFrame, y: np.ndarray, split: Split, cols: list[str], k: int,
              rounds: int = 800, params: dict | None = None) -> lgb.Booster:
    p = {**PARAMS, **(params or {}), "num_class": k}
    dtrain = lgb.Dataset(X.loc[split.train, cols], y[split.train], weight=class_weights(y[split.train], k),
                         free_raw_data=False)
    dval = lgb.Dataset(X.loc[split.early, cols], y[split.early], reference=dtrain)
    return lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(raw: np.ndarray, y: np.ndarray) -> float:
    def nll(t: float) -> float:
        p = softmax(raw, t)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())

    return float(minimize_scalar(nll, bounds=(0.25, 8.0), method="bounded").x)


def macro_f1(booster: lgb.Booster, X: pd.DataFrame, y: np.ndarray, cols: list[str]) -> float:
    pred = booster.predict(X[cols], raw_score=True).argmax(axis=1)
    return float(f1_score(y, pred, average="macro"))


def make_data(sessions: int, seed: int, noise_scale: float, state_set: str):
    df = simulate_corpus(sessions, seed, state_set=state_set, noise_scale=noise_scale)
    X, meta = build(df, seed=seed)
    states = STATE_SETS[state_set]
    y = meta["label"].map({s: i for i, s in enumerate(states)}).to_numpy()
    return df, X, meta, y, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--state-set", default="core", choices=list(STATE_SETS))
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--models", default=",".join(FEATURE_SETS))
    args = ap.parse_args()

    t0 = time.time()
    df, X, meta, y, states = make_data(args.sessions, args.seed, args.noise_scale, args.state_set)
    split = split_sessions(meta, args.seed)
    print(f"data: {len(df)} rows -> {len(X)} windows in {time.time() - t0:.0f}s; "
          f"train/early/calib/test = {split.train.sum()}/{split.early.sum()}/{split.calib.sum()}/{split.test.sum()}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = len(states)
    summary = {}
    temps = {}
    for name in args.models.split(","):
        cols = FEATURE_SETS[name]
        t1 = time.time()
        booster = train_one(X, y, split, cols, k)
        temps[name] = fit_temperature(booster.predict(X.loc[split.calib, cols], raw_score=True), y[split.calib])
        booster.save_model(str(out / f"{name}.txt"))
        summary[name] = {"test_macro_f1": round(macro_f1(booster, X.loc[split.test], y[split.test], cols), 4),
                         "rounds": booster.best_iteration, "features": len(cols), "temperature": round(temps[name], 3)}
        print(f"  {name:11s} macro-F1={summary[name]['test_macro_f1']:.3f} rounds={booster.best_iteration} "
              f"T={temps[name]:.2f} ({time.time() - t1:.0f}s)")
    train_X = X.loc[split.train]
    clip = {c: [float(np.nanpercentile(train_X[c], 1)), float(np.nanpercentile(train_X[c], 99))]
            for c in FEATURE_NAMES if train_X[c].notna().any()}
    meta_out = {
        "states": list(states), "state_set": args.state_set, "sessions": args.sessions, "seed": args.seed,
        "noise_scale": args.noise_scale, "feature_names": FEATURE_NAMES, "feature_sets": FEATURE_SETS,
        "temperatures": temps, "summary": summary, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "model_meta.json").write_text(json.dumps(meta_out, indent=2))
    (out / "clip_bounds.json").write_text(json.dumps(clip, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
