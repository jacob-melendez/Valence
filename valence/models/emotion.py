"""Emotion map models: a 10-state LightGBM classifier and LightGBM valence and arousal regressors.

    python -m valence.models.emotion --sessions 600 --seed 11 --out artifacts/emotion

Inputs are ``valence.features.emotion.INPUT_FEATURES`` (the served M2 set + the 31 emotion-v1 face cues). Splits are
by session (``train_ladder.split_sessions``: 70% train, 7.5% early stopping, 7.5% calibration, 15% test); the
classifier gets a softmax temperature fitted on the calibration sessions. Training rows get modality-dropout copies
(face block hidden or degraded), as the served M2 does. The served desktop models in ``artifacts/`` are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from valence.contracts import EMOTION_STATES
from valence.features.compute import FACE_FEATURES
from valence.features.emotion import EMOTION_FEATURE_NAMES, INPUT_FEATURES
from valence.features.emotion_dataset import build_emotion
from valence.models.train_ladder import Split, class_weights, fit_temperature, split_sessions
from valence.sim.generator import simulate_corpus

STATES = EMOTION_STATES
K = len(STATES)
CLS_PARAMS = dict(objective="multiclass", num_class=K, learning_rate=0.1, num_leaves=15, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=4, seed=11, deterministic=True, force_row_wise=True)
REG_PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=4, seed=11, deterministic=True, force_row_wise=True)
CLS_ROUNDS = 280  # keeps classifier.json.gz under ~3 MB (10 trees per round)
REG_ROUNDS = 800
DROPOUT = {"face_hidden": 0.30, "face_degraded": 0.20}

FACE_LEVELS = ["face_now", "face_mean_1m", "face_mean_5m", "face_mean_15m"]
FACE_VIS = ["face_visible_now", "face_visible_frac_5m"]
CUE_LEVELS = [n for n in EMOTION_FEATURE_NAMES if n.split("_")[0] in ("brow", "eye", "frown", "smile")
              and "_std_" not in n]
POSE_LEVELS = [n for n in EMOTION_FEATURE_NAMES if n.split("_")[0] in ("yaw", "pitch") and "_std_" not in n]


def load_corpus(sessions: int, seed: int, noise_scale: float = 1.0, cache: Path | None = None):
    """(X, meta) for an emotion-state-set corpus; cached as parquet outside the repository."""
    if cache is not None:
        xp, mp = cache / f"emotion_X_{sessions}_{seed}_{noise_scale}.parquet", cache / f"emotion_meta_{sessions}_{seed}_{noise_scale}.parquet"
        if xp.exists() and mp.exists():
            return pd.read_parquet(xp), pd.read_parquet(mp)
    df = simulate_corpus(sessions, seed, state_set="emotion", noise_scale=noise_scale)
    X, meta = build_emotion(df, seed=seed)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        X.to_parquet(xp)
        meta.to_parquet(mp)
    return X, meta


def labels(meta: pd.DataFrame) -> np.ndarray:
    return meta["label"].map({s: i for i, s in enumerate(STATES)}).to_numpy()


def face_dropout(X: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Extra copies of training rows with the whole face block missing (30%) or degraded (20%).

    Returns the augmented frame and, per row, the index of the source row (so targets follow their rows).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    hid_idx = rng.choice(n, int(DROPOUT["face_hidden"] * n), replace=False)
    hid = X.iloc[hid_idx].copy()
    hid[[c for c in FACE_FEATURES if c not in FACE_VIS] + EMOTION_FEATURE_NAMES] = np.nan
    hid[FACE_VIS] = 0.0
    deg_idx = rng.choice(n, int(DROPOUT["face_degraded"] * n), replace=False)
    deg = X.iloc[deg_idx].copy()
    m = len(deg)
    for c in FACE_LEVELS:
        deg[c] = np.clip(deg[c] + rng.normal(0, 0.25, m), -1, 1)
    deg["face_std_5m"] = deg["face_std_5m"] + 0.2
    for c in CUE_LEVELS:
        deg[c] = np.clip(deg[c] + rng.normal(0, 0.08, m), -1, 1)
    for c in POSE_LEVELS:
        deg[c] = deg[c] + rng.normal(0, 6.0, m)
    shrink = rng.uniform(0.3, 0.8, (m, 1))
    deg[FACE_VIS] = deg[FACE_VIS].to_numpy() * shrink
    blink = [c for c in EMOTION_FEATURE_NAMES if c.startswith("blink_rate")]
    deg[blink] = deg[blink].to_numpy() * rng.uniform(0.5, 1.5, (m, 1))  # fewer face frames: noisier blink counts
    src = np.concatenate([np.arange(n), hid_idx, deg_idx])
    return pd.concat([X, hid, deg], ignore_index=True), src


def _train_rows(X: pd.DataFrame, split: Split, cols: list[str], augment: bool, seed: int):
    Xt = X.loc[split.train, cols].reset_index(drop=True)
    src = np.arange(len(Xt))
    if augment:
        Xt, src = face_dropout(Xt, seed)
    return Xt, src


def train_classifier(X: pd.DataFrame, y: np.ndarray, split: Split, cols: list[str], *, augment: bool = True,
                     params: dict | None = None, rounds: int = CLS_ROUNDS) -> tuple[lgb.Booster, float]:
    p = {**CLS_PARAMS, **(params or {})}
    Xt, src = _train_rows(X, split, cols, augment, p["seed"])
    yt = y[split.train][src]
    dtrain = lgb.Dataset(Xt, yt, weight=class_weights(yt, K), free_raw_data=False)
    dval = lgb.Dataset(X.loc[split.early, cols], y[split.early], reference=dtrain)
    booster = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    t = fit_temperature(booster.predict(X.loc[split.calib, cols], raw_score=True), y[split.calib])
    return booster, t


def train_regressor(X: pd.DataFrame, target: np.ndarray, y: np.ndarray, split: Split, cols: list[str], *,
                    augment: bool = True, params: dict | None = None, rounds: int = REG_ROUNDS) -> lgb.Booster:
    p = {**REG_PARAMS, **(params or {})}
    Xt, src = _train_rows(X, split, cols, augment, p["seed"] + 1)
    tt = target[split.train][src]  # unweighted: class weights distort an L2 regression's targets
    dtrain = lgb.Dataset(Xt, tt, free_raw_data=False)
    dval = lgb.Dataset(X.loc[split.early, cols], target[split.early], reference=dtrain)
    return lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def dump_gz(booster: lgb.Booster, path: Path) -> int:
    """Write ``booster.dump_model()`` as gzipped JSON (deterministic bytes); returns the size in bytes."""
    import gzip

    raw = json.dumps(booster.dump_model(), separators=(",", ":")).encode()
    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    return path.stat().st_size


def model_id(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return "emotion-v1@" + h.hexdigest()[:8]


MODEL_FILES = {"classifier": "classifier", "valence": "valence", "arousal": "arousal"}


def save_text_gz(booster: lgb.Booster, path: Path) -> None:
    """LightGBM's own text format, gzipped: what Python loads (``EmotionPredictor``)."""
    import gzip

    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(booster.model_to_string().encode())


class EmotionPredictor:
    """Loads ``artifacts/emotion`` and predicts one input vector (INPUT_FEATURES order, NaN = missing)."""

    def __init__(self, directory: Path) -> None:
        import gzip

        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.cols = self.meta["input_features"]
        self.states = [s["id"] for s in self.meta["states"]]
        self.anchors = np.array([[s["valence"], s["arousal"]] for s in self.meta["states"]])
        self.t = float(self.meta["temperature"])
        self.models = {k: lgb.Booster(model_str=gzip.decompress((self.dir / f"{k}.txt.gz").read_bytes()).decode())
                       for k in MODEL_FILES}
        self.model_id = self.meta["model_id"]

    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.models["classifier"].predict(np.atleast_2d(X), raw_score=True, num_threads=1)

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        raw = self.raw(X)
        z = raw / self.t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        v = self.models["valence"].predict(X, num_threads=1)
        a = self.models["arousal"].predict(X, num_threads=1)
        mix = p @ self.anchors  # the served map position: probability-weighted anchors
        return {"raw": raw, "proba": p, "valence_raw": v, "arousal_raw": a,
                "valence": np.clip(v, -1, 1), "arousal": np.clip(a, -1, 1),
                "valence_mix": mix[:, 0], "arousal_mix": mix[:, 1]}

    def contrib(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = len(self.cols) + 1
        return {"classifier": np.asarray(self.models["classifier"].predict(X, pred_contrib=True, num_threads=1)
                                         ).reshape(len(X), len(self.states), n),
                "valence": np.asarray(self.models["valence"].predict(X, pred_contrib=True, num_threads=1)),
                "arousal": np.asarray(self.models["arousal"].predict(X, pred_contrib=True, num_threads=1))}


def clean_json(v):
    """NaN -> null, recursively, so every file here is strict JSON (``JSON.parse``)."""
    if isinstance(v, dict):
        return {k: clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    if isinstance(v, float) and v != v:
        return None
    return v


def layers_of(cols: list[str]) -> dict[str, str]:
    from valence.features.spec import layer_of

    emo = set(EMOTION_FEATURE_NAMES)
    return {c: ("Facial Surface" if c in emo else layer_of(c)) for c in cols}


def export(full, out: Path, report: dict, train_info: dict) -> dict:
    """Write the dumps (JSON for TypeScript, text for Python) and meta.json."""
    from valence.contracts import AFFECT_ANCHORS, EMOTION_DISPLAY
    from valence.features.emotion import FEATURE_VERSION

    boosters = {"classifier": full.cls, "valence": full.reg_v, "arousal": full.reg_a}
    sizes = {}
    for name, b in boosters.items():
        sizes[name] = {"json_gz_bytes": dump_gz(b, out / f"{name}.json.gz"), "trees": b.num_trees(),
                       "rounds": b.current_iteration()}
        save_text_gz(b, out / f"{name}.txt.gz")
    meta = {
        "feature_version": FEATURE_VERSION,
        "model_id": model_id([out / f"{k}.json.gz" for k in boosters]),
        "input_features": full.cols,
        "emotion_feature_names": EMOTION_FEATURE_NAMES,
        "n_features": len(full.cols),
        "states": [{"id": s, "display": EMOTION_DISPLAY[s], "valence": AFFECT_ANCHORS[s][0],
                    "arousal": AFFECT_ANCHORS[s][1]} for s in STATES],
        "anchors_are_assumptions": True,
        "temperature": full.t,
        "inference": {"probabilities": "softmax(raw / temperature)",
                      "map_position": "valence = sum_k p_k * states[k].valence, arousal = sum_k p_k * states[k].arousal "
                                      "(served: beats the regressors on held-out simulated sessions, see summary)",
                      "regressors": "valence = clip(valence raw, -1, 1), arousal = clip(arousal raw, -1, 1) "
                                    "(exported for comparison)",
                      "missing": "NaN (LightGBM native); no clipping of inputs",
                      "pred_contrib": "path-dependent TreeSHAP, n_features + 1 values (last = bias) per output"},
        "files": {k: {"json": f"{k}.json.gz", "python": f"{k}.txt.gz"} for k in boosters},
        "layers": layers_of(full.cols),
        "training": {**train_info, "sizes": sizes, "classifier_params": CLS_PARAMS, "regressor_params": REG_PARAMS,
                     "modality_dropout": DROPOUT, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "summary": {k: {m: v for m, v in report[k].items() if m != "confusion"}
                    for k in ("full", "face_only", "context_only", "shuffled_context_test",
                              "shuffled_context_trained", "full_face_hidden") if k in report},
    }
    (out / "meta.json").write_text(json.dumps(clean_json(meta), indent=1, allow_nan=False))
    return meta


def main() -> None:
    from valence.eval.emotion_eval import Models, evaluate_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--out", default="artifacts/emotion")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "valence-emotion"))
    ap.add_argument("--no-variants", action="store_true", help="train and export the full models only")
    args = ap.parse_args()
    t0 = time.time()
    X, meta = load_corpus(args.sessions, args.seed, args.noise_scale, Path(args.cache))
    y = labels(meta)
    split = split_sessions(meta, args.seed)
    print(f"data: {len(X)} windows ({time.time() - t0:.0f}s); train/early/calib/test = "
          f"{split.train.sum()}/{split.early.sum()}/{split.calib.sum()}/{split.test.sum()}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cls, t = train_classifier(X, y, split, INPUT_FEATURES, params={"seed": args.seed})
    reg_v = train_regressor(X, meta["valence"].to_numpy(), y, split, INPUT_FEATURES, params={"seed": args.seed})
    reg_a = train_regressor(X, meta["arousal"].to_numpy(), y, split, INPUT_FEATURES, params={"seed": args.seed})
    full = Models(INPUT_FEATURES, cls, t, reg_v, reg_a)
    print(f"trained full models ({time.time() - t0:.0f}s): rounds cls={cls.current_iteration()} "
          f"V={reg_v.current_iteration()} A={reg_a.current_iteration()} T={t:.3f}", flush=True)
    if args.no_variants:
        from valence.eval.emotion_eval import score

        test = split.test
        report = {"full": score(full, X.loc[test].reset_index(drop=True), meta.loc[test].reset_index(drop=True),
                                y[test])}
    else:
        report, _ = evaluate_all(X, meta, y, split, seed=args.seed, full=full)
    info = {"sessions": args.sessions, "seed": args.seed, "noise_scale": args.noise_scale, "windows": int(len(X)),
            "split": "by session, 70/7.5/7.5/15 (train/early-stop/calibration/test)",
            "split_windows": {k: int(getattr(split, k).sum()) for k in ("train", "early", "calib", "test")},
            "data": "simulated (valence.sim, state set 'emotion'); no real data"}
    m = export(full, out, report, info)
    from valence.eval.emotion_eval import scenario_check

    report["scenarios"] = scenario_check(EmotionPredictor(out))
    report["model_id"] = m["model_id"]
    report["sizes"] = m["training"]["sizes"]
    report["training"] = info
    (out / "metrics.json").write_text(json.dumps(clean_json(report), indent=1, allow_nan=False))
    print(json.dumps(m["training"]["sizes"]), f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
