"""Real-data evaluation of the emotion map on consented self-report taps (docs/EMOTION_FEATURES.md section 6).

    python -m valence.eval.self_report data/self_report/*.jsonl [--models artifacts/emotion] [--out report.json]
    python -m valence.eval.self_report --synthetic 6 --write /tmp/sr.jsonl   # simulator fixture, no real people

Each JSONL line is one tap: ``t``, ``person_id``, ``valence`` and ``arousal`` in [-1, 1], the emotion model's input
vector at that moment in ``features`` (an object keyed by feature name, null = NaN, or a list in ``meta.json``
``input_features`` order), optionally ``session_id`` and a picked state ``label``. Lines never carry media or text.

Reported (overall, per person, and for taps with >= 1 min of window): valence/arousal MAE and CCC of the served map
position (probability-weighted anchors), of the regressors and of a constant (0, 0) guess, quadrant agreement, and
agreement between the state nearest the tap and the state nearest the map position (and the classifier's label).
CCC needs >= 3 taps that vary on that axis. Lines with another feature_version, non-numeric or out-of-range taps
are skipped and listed under ``problems``; absent input features are read as missing and listed too.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from valence.eval.emotion_eval import ccc, nearest_anchor

KEYS = ("valence", "arousal")
WARM_FILL = 12 / 180  # fill_ratio of a 1-minute window


def load_taps(paths: list[Path], input_features: list[str]) -> tuple[list[dict], np.ndarray, list[str]]:
    taps, rows, problems = [], [], []
    index = {n: i for i, n in enumerate(input_features)}
    for path in paths:
        for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if d.get("feature_version") not in (None, "emotion-v1"):
                    raise ValueError(f"feature_version {d.get('feature_version')!r} is not emotion-v1")
                if any(isinstance(d[k], bool) or not isinstance(d[k], (int, float)) for k in KEYS):
                    raise ValueError("valence/arousal must be numbers")
                v, a = float(d["valence"]), float(d["arousal"])
                if not (-1 <= v <= 1 and -1 <= a <= 1):
                    raise ValueError("valence/arousal outside [-1, 1]")
                feats = d["features"]
                x = np.full(len(input_features), np.nan)
                if isinstance(feats, dict):
                    unknown = [k for k in feats if k not in index]
                    if unknown:
                        problems.append(f"{path}:{ln}: ignored unknown features {unknown[:3]}")
                    for k, val in feats.items():
                        if k in index and val is not None:
                            x[index[k]] = float(val)
                    absent = [n for n in input_features if n not in feats]
                    if absent:
                        problems.append(f"{path}:{ln}: {len(absent)} input features absent (read as missing), "
                                        f"e.g. {absent[:3]}")
                else:
                    if len(feats) != len(input_features):
                        raise ValueError(f"features list has {len(feats)} values, expected {len(input_features)}")
                    x = np.array([np.nan if val is None else float(val) for val in feats])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                problems.append(f"{path}:{ln}: skipped ({e})")
                continue
            taps.append({"t": d.get("t"), "person": f"{d.get('session_id', Path(path).stem)}/{d.get('person_id', 1)}",
                         "valence": v, "arousal": a, "label": d.get("label")})
            rows.append(x)
    X = np.vstack(rows) if rows else np.empty((0, len(input_features)))
    return taps, X, problems


def _block(tv, ta, pv, pa) -> dict:
    out = {"n": int(len(tv)),
           "valence_mae": round(float(np.abs(pv - tv).mean()), 4),
           "arousal_mae": round(float(np.abs(pa - ta).mean()), 4),
           "quadrant_agreement": round(float(((np.sign(pv) == np.sign(tv)) & (np.sign(pa) == np.sign(ta))).mean()), 4)}
    # CCC per axis, only with >= 3 taps that vary on that axis
    out["valence_ccc"] = round(ccc(pv, tv), 4) if len(tv) >= 3 and tv.std() > 1e-6 else None
    out["arousal_ccc"] = round(ccc(pa, ta), 4) if len(ta) >= 3 and ta.std() > 1e-6 else None
    return out


def evaluate(taps: list[dict], X: np.ndarray, predictor) -> dict:
    if not taps:
        return {"n": 0}
    r = predictor.predict(X)
    tv = np.array([t["valence"] for t in taps])
    ta = np.array([t["arousal"] for t in taps])
    pv, pa = r["valence"], r["arousal"]  # regressors (comparison)
    anchor = r["proba"] @ predictor.anchors  # the served map position
    states = predictor.states
    tap_state = nearest_anchor(tv, ta)
    face_cols = [i for i, c in enumerate(predictor.cols) if c == "face_visible_now"]

    def summary(m: np.ndarray) -> dict:
        out = {"map_position": _block(tv[m], ta[m], anchor[m, 0], anchor[m, 1]),
               "regressors": _block(tv[m], ta[m], pv[m], pa[m]),
               "constant_zero": _block(tv[m], ta[m], np.zeros(m.sum()), np.zeros(m.sum())),
               "nearest_anchor_agreement": round(float(
                   (nearest_anchor(anchor[m, 0], anchor[m, 1]) == tap_state[m]).mean()), 4),
               "nearest_anchor_agreement_regressors": round(float(
                   (nearest_anchor(pv[m], pa[m]) == tap_state[m]).mean()), 4),
               "classifier_vs_tap_region": round(float((r["proba"][m].argmax(axis=1) == tap_state[m]).mean()), 4)}
        labelled = [i for i in np.nonzero(m)[0] if taps[i]["label"] in states]
        if labelled:
            out["classifier_vs_label"] = {"n": len(labelled), "accuracy": round(float(np.mean(
                [states[int(r["proba"][i].argmax())] == taps[i]["label"] for i in labelled])), 4)}
        if face_cols:
            out["face_visible_share"] = round(float((np.nan_to_num(X[m, face_cols[0]]) > 0).mean()), 4)
        return out

    report = {"model": predictor.model_id, "n_taps": len(taps), "overall": summary(np.ones(len(taps), bool)),
              "per_person": {}}
    if "fill_ratio" in predictor.cols:  # taps with >= 1 min of window (the console shows "warming up" before)
        warm = np.nan_to_num(X[:, predictor.cols.index("fill_ratio")]) >= WARM_FILL
        report["warm_only"] = summary(warm) if warm.any() else {"n": 0}
        report["n_warm_taps"] = int(warm.sum())
    for person in sorted({t["person"] for t in taps}):
        m = np.array([t["person"] == person for t in taps])
        report["per_person"][person] = summary(m)
    report["predicted_states"] = {s: int(c) for s, c in zip(states, np.bincount(r["proba"].argmax(axis=1),
                                                                                minlength=len(states))) if c}
    return report


def figure(taps: list[dict], X: np.ndarray, predictor, path: Path) -> None:
    """Taps vs the served map position, one arrow per tap (tap -> prediction), anchors in gray."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = predictor.predict(X)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#e4e3df", lw=1)
    ax.axvline(0, color="#e4e3df", lw=1)
    for (av, aa), s in zip(predictor.anchors, predictor.states):
        ax.scatter([av], [aa], s=40, facecolor="#fcfcfb", edgecolor="#8a8984", lw=1.2, zorder=2)
        ax.text(av + 0.03, aa, s.split("_")[0], fontsize=7, color="#52514e", va="center")
    for t, v, a in zip(taps, r["valence_mix"], r["arousal_mix"]):
        ax.plot([t["valence"], v], [t["arousal"], a], color="#8a8984", lw=0.8, zorder=1)
        ax.scatter([t["valence"]], [t["arousal"]], s=22, color="#eb6834", zorder=3)
        ax.scatter([v], [a], s=22, color="#2a78d6", zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence", color="#52514e")
    ax.set_ylabel("Arousal", color="#52514e")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Self-report taps (orange) vs map position (blue), n = {len(taps)}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")


def synthetic_taps(n_sessions: int, seed: int, every_s: float = 120.0, report_sd: float = 0.15) -> list[dict]:
    """Simulator fixture in the self-report format: a tap every ``every_s`` seconds, the tap = the episode's true
    valence/arousal plus self-report noise. Not real data; for exercising the pipeline before real files arrive."""
    from valence.features.compute import PERSON_FIELDS, ROOM_FIELDS, compute
    from valence.features.emotion import EMOTION_FIELDS, INPUT_FEATURES, compute_emotion, model_input
    from valence.sim.generator import simulate_session

    rng = np.random.default_rng(seed)
    lines = []
    step = int(every_s / 5)
    for k in range(n_sessions):
        sid = f"synthetic-{seed}-{k}"
        df = simulate_session(sid, np.random.default_rng([seed, k]), state_set="emotion")
        room = df[df["person_id"] == 0].sort_values("tick")
        for pid, p in df[df["person_id"] != 0].groupby("person_id"):
            p = p.sort_values("tick")
            fields = sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS))
            arr = {f: p[f].to_numpy(dtype=float) for f in fields}
            rarr = {f: room[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
            rpos = {t: i for i, t in enumerate(room["tick"].to_numpy())}
            ticks = p["tick"].to_numpy()
            for i in range(step, len(p), step):
                lo, rp = max(0, i - 179), rpos[ticks[i]]
                x = model_input(compute({f: arr[f][lo:i + 1] for f in PERSON_FIELDS},
                                        {f: v[max(0, rp - 179):rp + 1] for f, v in rarr.items()}),
                                compute_emotion({f: arr[f][lo:i + 1] for f in EMOTION_FIELDS}))
                v = float(np.clip(p["true_valence"].iat[i] + rng.normal(0, report_sd), -1, 1))
                a = float(np.clip(p["true_arousal"].iat[i] + rng.normal(0, report_sd), -1, 1))
                lines.append({"t": float(ticks[i] * 5.0), "session_id": sid, "person_id": int(pid), "valence": v,
                              "arousal": a, "feature_version": "emotion-v1", "synthetic": True,
                              "features": {n: (None if math.isnan(val) else float(val))
                                           for n, val in zip(INPUT_FEATURES, x)}})
    return lines


def main() -> None:
    from valence.models.emotion import EmotionPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--models", default="artifacts/emotion")
    ap.add_argument("--out")
    ap.add_argument("--synthetic", type=int, help="write a simulator fixture with this many sessions first")
    ap.add_argument("--write", help="where --synthetic writes its JSONL")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--figure", help="write a taps-vs-prediction PNG here")
    args = ap.parse_args()
    files = [Path(f) for f in args.files]
    if args.synthetic:
        path = Path(args.write or "self_report_synthetic.jsonl")
        path.write_text("".join(json.dumps(line) + "\n" for line in synthetic_taps(args.synthetic, args.seed)))
        files.append(path)
    predictor = EmotionPredictor(Path(args.models))
    taps, X, problems = load_taps(files, predictor.cols)
    report = {"files": [str(f) for f in files], **evaluate(taps, X, predictor), "problems": problems[:50]}
    if args.figure and taps:
        figure(taps, X, predictor, Path(args.figure))
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
