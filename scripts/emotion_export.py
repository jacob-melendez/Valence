"""Golden data for the TypeScript emotion runtime: ``artifacts/emotion/golden.json``.

    python scripts/emotion_export.py [--dir artifacts/emotion]

Three parts:

* ``vectors``: ~20 model inputs (``meta.json`` ``input_features`` order, null = NaN) from scenario replays and the
  emotion simulator, including face-hidden and warm-up windows, with Python's raw scores, temperature-softmax
  probabilities, valence/arousal (raw and clipped) and ``pred_contrib`` for all three models;
* ``windows``: per-tick channel arrays (oldest first) and the 31 emotion-v1 features ``compute_emotion`` returns;
* ``frames``: blendshapes + transformation matrix sequences and the per-frame cues ``FaceCueTracker`` returns.

Nothing here is media: simulated tick values and synthetic blendshape numbers only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from valence.features.build_dataset import ROOM  # noqa: E402
from valence.features.compute import PERSON_FIELDS, ROOM_FIELDS, compute  # noqa: E402
from valence.features.emotion import EMOTION_FIELDS, compute_emotion, model_input  # noqa: E402
from valence.features.face_cues import FaceCueTracker, raw_cues  # noqa: E402
from valence.models.emotion import EmotionPredictor  # noqa: E402
from valence.sim.generator import simulate_session  # noqa: E402
from valence.sim.scenarios import render  # noqa: E402


def _j(v):
    """JSON-safe: NaN -> null, numpy -> lists of floats."""
    if isinstance(v, np.ndarray):
        return [_j(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def windows_from(df, picks):
    """(label, person arrays, room arrays) for (person_id, end_tick, history_ticks) picks."""
    room_df = df[df["person_id"] == ROOM].sort_values("tick")
    out = []
    for pid, end, hist, note in picks:
        p = df[(df["person_id"] == pid) & (df["tick"] <= end)].sort_values("tick").tail(hist)
        if p.empty:
            continue
        r = room_df[room_df["tick"] <= end].tail(180)
        fields = sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS))
        person = {f: p[f].to_numpy(dtype=float) for f in fields}
        room = {f: r[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
        out.append((note, p["true_state"].iat[-1], person, room))
    return out


def edge_windows() -> list[dict]:
    """Synthetic windows for the missing-data rules of section 3."""
    rng = np.random.default_rng(9)
    out = []

    def add(note, person):
        out.append({"note": note, "person": {f: _j(np.asarray(person[f], dtype=float)) for f in EMOTION_FIELDS},
                    "expected": _j(compute_emotion(person))})

    n = 24
    cueless = {f: np.full(n, np.nan) for f in EMOTION_FIELDS}
    cueless["face_visible"] = np.ones(n)
    cueless["blink_count"] = np.zeros(n)
    add("face visible, no cues produced (older producer): blink rates NaN", cueless)
    vis = np.array([22, 16, 16, 20, 0, 21, 20, 19, 21, 18, 21, 20], dtype=float) / 107.0  # exactly 10 s
    edge = {f: rng.normal(0, 0.05, 12) for f in EMOTION_FIELDS}
    edge["face_visible"] = vis
    edge["blink_count"] = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    add("visible time exactly 10 s (summation-order independent)", edge)
    mixed = {f: rng.normal(0, 0.05, 40) for f in EMOTION_FIELDS}
    mixed["face_visible"] = rng.uniform(0.5, 1.0, 40)
    mixed["blink_count"] = rng.integers(0, 3, 40).astype(float)
    for f in ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away"):
        mixed[f][::3] = np.nan  # every third tick without cues
    mixed["head_yaw"] = np.abs(mixed["head_yaw"]) * 300
    add("cues missing on every third tick", mixed)
    return out


def golden_frames() -> list[dict]:
    """Synthetic frame sequences through FaceCueTracker: calibration, brow lowering, a blink, a long closure,
    head turns, a missing face, and a row-major matrix."""
    def mat(yaw, pitch, row_major=False, tz=-45.0):
        y, p = math.radians(yaw), math.radians(pitch)
        f = (math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p))
        m = [[1, 0, f[0], 0], [0, 1, f[1], 0], [0, 0, f[2], tz], [0, 0, 0, 1]]
        return [m[r][c] for r in range(4) for c in range(4)] if row_major else [m[r][c] for c in range(4) for r in range(4)]

    base = {"browDownLeft": 0.08, "browDownRight": 0.12, "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.14,
            "mouthFrownLeft": 0.02, "mouthFrownRight": 0.03, "mouthSmileLeft": 0.05, "mouthSmileRight": 0.07}
    rng = np.random.default_rng(5)
    seq = []
    t = 0.0
    for i in range(330):  # 22 s at 15 fps (calibration freezes at 20 s)
        b = {k: float(np.clip(v + rng.normal(0, 0.01), 0, 1)) for k, v in base.items()}
        seq.append((t, b, mat(rng.normal(3, 1), rng.normal(-8, 1))))
        t += 1000 / 15
    events = ([{"browDownLeft": 0.45, "browDownRight": 0.40}] * 5 + [{"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.8}] * 3
              + [{}] * 3 + [{"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}] * 15 + [{}] * 2
              + [{"mouthSmileLeft": 0.6, "mouthSmileRight": 0.55}] * 3 + [{"mouthFrownLeft": 0.3}] * 2)
    for e in events:
        seq.append((t, {**base, **e}, mat(3, -8)))
        t += 1000 / 15
    for yaw, pitch, rm in ((35, -8, False), (-40, -6, False), (3, -35, False), (3, 15, True), (178, 0, False)):
        seq.append((t, base, mat(yaw, pitch, row_major=rm)))
        t += 1000 / 15
    seq.append((t, None, None))  # no face
    t += 1000 / 15
    seq.append((t, {k: v for k, v in base.items() if not k.endswith("Right")}, mat(3, -8)))  # one side missing
    t += 1000 / 15
    seq.append((t, {**base, "browDownRight": float("nan")}, mat(3, -8)))  # one side NaN: use the other
    # Blink edge cases at the exact thresholds (baseline b from the calibration above).
    probe = FaceCueTracker()
    for t_ms, b, m in seq:
        probe.update(None if b is None else raw_cues(b, m), t_ms)
    bc = probe.baseline()["closure"]

    def eyes(c):
        return {**base, "eyeBlinkLeft": c, "eyeBlinkRight": c}

    edge = [(70, eyes(bc + 0.30)), (300, eyes(bc + 0.15)),  # closes at == b+0.30, reopens at == b+0.15: a blink
            (70, eyes(bc + 0.29)), (70, base),  # just below the close threshold: nothing
            (70, eyes(0.95)), (600, base),  # closed exactly 600 ms: counts
            (70, eyes(0.95)), (601, base),  # 601 ms: a closure, not a blink
            (70, eyes(0.95)), (70, None), (70, base),  # closure across a frame without a face: reset, no blink
            (70, eyes(0.95)), (70, {k: v for k, v in base.items() if not k.startswith("eyeBlink")}),
            (70, base),  # NaN closure while closed leaves the state: the reopening counts
            (70, eyes(0.95)), (-50, base)]  # time steps backwards (clamped gap): -50 ms after closing, counts
    for dt, b in edge:
        t += dt
        seq.append((t, b, None if b is None else mat(3, -8)))
    tracker = FaceCueTracker()
    frames = []
    for t_ms, b, m in seq:
        cues = tracker.update(None if b is None else raw_cues(b, m), t_ms)
        frames.append({"t_ms": t_ms, "blendshapes": _j(b), "matrix": m, "expected": _j(cues)})
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "artifacts" / "emotion"))
    args = ap.parse_args()
    out_dir = Path(args.dir)
    pred = EmotionPredictor(out_dir)
    cases = []
    demo = render("full_demo")
    cases += windows_from(demo, [(1, 59, 60, "full_demo focused 5 min"), (1, 60, 60, "full_demo first Frustrated tick"),
                                 (1, 150, 180, "full_demo 12.5 min"),
                                 (1, 2, 3, "full_demo warm-up 3 ticks"), (1, 300, 180, "full_demo after the break")])
    for name, st in (("tense", "all"), ("overwhelmed", "all"), ("exhausted", "core"), ("collaborative_happy", "core")):
        df = render(name, state_set=st)
        end = int(df["tick"].max())
        cases += windows_from(df, [(1, end, 180, f"{name} end"), (1, end // 2, 90, f"{name} middle")])
    emo = simulate_session("golden-emotion", np.random.default_rng(42), state_set="emotion")
    people = emo[emo["person_id"] == 1].sort_values("tick")
    seen = set()
    for _, row in people.iterrows():
        if row["true_state"] not in seen and row["episode_age_s"] >= 300:
            seen.add(row["true_state"])
            cases += windows_from(emo, [(1, int(row["tick"]), 180, f"emotion sim {row['true_state']}")])
    vectors, windows = [], []
    for note, state, person, room in cases:
        x123 = compute({f: person[f] for f in PERSON_FIELDS}, room)
        xe = compute_emotion(person)
        x = model_input(x123, xe)
        variants = [(note, x)]
        if note.startswith("emotion sim") and len(vectors) % 3 == 0:
            xh = x.copy()  # face hidden
            face = [i for i, c in enumerate(pred.cols) if c.startswith(("face_",)) or c in set(pred.meta["emotion_feature_names"])]
            xh[face] = np.nan
            for c in ("face_visible_now", "face_visible_frac_5m"):
                xh[pred.cols.index(c)] = 0.0
            variants.append((note + " (face hidden)", xh))
        for n, v in variants:
            r = pred.predict(v)
            c = pred.contrib(v)
            vectors.append({"note": n, "true_state": state, "x": _j(v), "raw": _j(r["raw"][0]),
                            "probabilities": _j(r["proba"][0]), "valence_raw": _j(r["valence_raw"][0]),
                            "arousal_raw": _j(r["arousal_raw"][0]), "valence": _j(r["valence"][0]),
                            "arousal": _j(r["arousal"][0]), "valence_mix": _j(r["valence_mix"][0]),
                            "arousal_mix": _j(r["arousal_mix"][0]), "label": pred.states[int(np.argmax(r["proba"][0]))],
                            "pred_contrib": {"classifier": _j(c["classifier"][0]), "valence": _j(c["valence"][0]),
                                             "arousal": _j(c["arousal"][0])}})
        if len(windows) < 8:
            windows.append({"note": note, "person": {f: _j(person[f]) for f in EMOTION_FIELDS}, "expected": _j(xe)})
    windows += edge_windows()
    golden = {
        "model_id": pred.model_id, "feature_version": pred.meta["feature_version"],
        "input_features": pred.cols, "states": pred.states, "temperature": pred.t,
        "tolerance": {"features": 1e-6, "scores": 1e-5, "pred_contrib": 1e-5},
        "vectors": vectors, "windows": windows, "frames": golden_frames(),
    }
    path = out_dir / "golden.json"
    path.write_text(json.dumps(golden, separators=(",", ":"), allow_nan=False))  # strict JSON for JSON.parse
    print(f"{path}: {len(vectors)} vectors, {len(windows)} windows, {len(golden['frames'])} frames, "
          f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
