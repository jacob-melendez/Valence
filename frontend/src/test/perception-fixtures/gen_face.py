"""Face parity fixtures for the browser perception: frontend/src/perception/faceValence.ts, faceCues.ts and assign.ts.

Runs the real valence/perception/face_valence.py (valence_raw, blendshape_scores, nose_tip_px,
FaceValenceCalibrator, assign_faces), valence/features/face_cues.py (head_pose_deg, raw_cues, FaceCueTracker) and
scipy.optimize.linear_sum_assignment on seeded inputs and writes what they return to face-*.json next to this file.
src/test/perception-face.test.ts replays every case. Strict JSON: non-finite floats are the strings "NaN",
"Infinity" and "-Infinity". Called by scripts/export_perception_fixtures.py (group "face"), or on its own:

    .venv/bin/python frontend/src/test/perception-fixtures/gen_face.py
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.optimize import linear_sum_assignment  # noqa: E402

from valence.features.face_cues import BLINK_CLOSE, BLINK_OPEN, FaceCueTracker, head_pose_deg, raw_cues  # noqa: E402
from valence.perception.face_valence import (  # noqa: E402
    VALENCE_BLENDSHAPES,
    FaceValenceCalibrator,
    assign_faces,
    blendshape_scores,
    nose_tip_px,
    valence_raw,
)

NAN, INF = math.nan, math.inf
RAW_KEYS = ("brow", "closure", "frown", "smile", "yaw", "pitch")
CUE_KEYS = ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "blink_onsets")
CUE_SHAPES = tuple(f"{stem}{side}" for stem in ("browDown", "eyeBlink", "mouthFrown", "mouthSmile")
                   for side in ("Left", "Right"))
EXTRA_SHAPES = ("cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "jawOpen", "_neutral", "eyeBlinkLeft")


def enc(x: Any) -> Any:
    """Plain JSON values; non-finite floats as strings."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [enc(v) for v in x]
    raise TypeError(f"cannot encode {type(x).__name__}")


def write(out_dir: pathlib.Path, name: str, obj: Any) -> None:
    text = json.dumps(enc(obj), allow_nan=False, separators=(",", ":"))
    (out_dir / f"{name}.json").write_text(text + "\n", encoding="utf-8")


def r4(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------------------------------------------
# face_valence.py: valence_raw, blendshape_scores, nose_tip_px


def valence_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(1101)
    special = [NAN, INF, -INF, 0.0, -0.0, 1.0, 1e-300, -0.25, 1.5]

    def value() -> float:
        u = rng.random()
        if u < 0.06:
            return special[int(rng.integers(len(special)))]
        return round(float(rng.random()), 5) if u < 0.7 else float(rng.random())

    vr = []
    fixed: list[dict[str, float]] = [{}, {"cheekSquintLeft": 1.0, "cheekSquintRight": 1.0}, {"_neutral": 0.3}]
    fixed += [{name: 0.5} for name in VALENCE_BLENDSHAPES]
    fixed += [{"mouthSmileLeft": NAN}, {"mouthSmileRight": INF, "mouthSmileLeft": -INF}, {"browDownLeft": 1e308,
                                                                                         "browDownRight": 1e308}]
    for bs in fixed:
        vr.append({"bs": bs, "out": valence_raw(bs)})
    for k in range(320):
        p = (0.0, 0.15, 0.4, 0.7, 1.0)[k % 5]
        names = [n for n in VALENCE_BLENDSHAPES if rng.random() < p]
        names += [n for n in EXTRA_SHAPES if rng.random() < 0.25]
        order = rng.permutation(len(names))
        bs = {names[int(i)]: value() for i in order}
        vr.append({"bs": bs, "out": valence_raw(bs)})

    bsc = []
    pool = list(VALENCE_BLENDSHAPES) + list(EXTRA_SHAPES) + ["eyeBlinkRight", "mouthSmile", "mouthsmileLeft"]
    for _ in range(60):
        n = int(rng.integers(0, 16))
        cats = [(pool[int(rng.integers(len(pool)))], value()) for _ in range(n)]  # duplicates allowed: last wins
        objs = [SimpleNamespace(category_name=name, score=score) for name, score in cats]
        bsc.append({"cats": [[name, score] for name, score in cats], "out": blendshape_scores(objs)})

    nose = []
    for _ in range(40):
        n = int(rng.integers(2, 8))
        lms = [[value(), value()] for _ in range(n)]
        w, h = [(640, 360), (1280, 720), (320, 240), (1920, 1080)][int(rng.integers(4))]
        pts = [SimpleNamespace(x=x, y=y) for x, y in lms]
        nose.append({"lms": lms, "w": w, "h": h, "out": list(nose_tip_px(pts, w, h))})
    return {"valence_raw": vr, "blendshape_scores": bsc, "nose_tip_px": nose}


# ---------------------------------------------------------------------------------------------------------------
# face_valence.py: FaceValenceCalibrator


def calibrator_case(name: str, v: list[float], t: list[float], params: dict[str, float] | None = None,
                    every: int = 1) -> dict[str, Any]:
    params = params or {}
    cal = FaceValenceCalibrator(**params)
    out, state = [], []
    was = False
    for k, (vk, tk) in enumerate(zip(v, t, strict=True)):
        out.append(cal.update(vk, tk))
        if k % every == 0 or k >= len(v) - 3 or cal.calibrated != was:
            state.append([k, cal.calibrated, cal.baseline, cal.visible_s])
        was = cal.calibrated
    return {"name": name, "params": params, "v": v, "t": t, "out": out, "state": state}


def calibrator_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(2202)
    cases = []

    # Freezes on the sample where visible time reaches exactly 20.0 s (0.5 s steps are exact in binary).
    cases.append(calibrator_case("freeze_at_exactly_20s", [0.2] * 40 + [0.9] * 21,
                                 [0.5 * k for k in range(61)]))
    v = [round(float(x), 3) for x in rng.normal(0.1, 0.2, 60)]
    cases.append(calibrator_case("even_median_at_freeze", v, [0.5 * k for k in range(60)]))

    # Accumulated 0.1 s steps: visible time is a rounded sum, the 20 s comparison sees it bit for bit.
    t, ts = 0.0, []
    for _ in range(230):
        ts.append(t)
        t += 0.1
    cases.append(calibrator_case("accumulated_0p1_steps", [round(float(x), 3) for x in rng.normal(0, 0.3, 230)], ts))

    # Random dropouts (NaN / +-inf valence), jitter, gaps > 1 s, backwards and repeated times, non-finite t.
    for seed_name in ("random_dropouts_a", "random_dropouts_b"):
        v, ts, t = [], [], 0.0
        for _ in range(520):
            u = rng.random()
            v.append(NAN if u < 0.12 else INF if u < 0.14 else -INF if u < 0.16 else r4(rng.normal(0.05, 0.3)))
            w = rng.random()
            if w < 0.03:
                t = round(t + float(rng.uniform(1.2, 6.0)), 3)
            elif w < 0.06:
                t = round(t - float(rng.uniform(0.0, 0.5)), 3)
            elif w < 0.08:
                pass  # repeated time
            else:
                t = round(t + float(rng.uniform(0.05, 0.2)), 3)
            x = rng.random()
            ts.append(NAN if x < 0.02 else INF if x < 0.03 else -INF if x < 0.04 else t)
        cases.append(calibrator_case(seed_name, v, ts))

    # The 4096-sample cap is reached before 20 s of visible face (4 ms steps): the cap freezes the baseline.
    n = 4200
    v = [NAN if rng.random() < 0.01 else round(float(rng.gamma(2.0, 0.1)) - 0.1, 3) for _ in range(n)]
    cases.append(calibrator_case("cap_4096_before_20s", v, [round(0.004 * k, 3) for k in range(n)], every=128))
    # 5/1024 s steps are exact: on sample 4096 visible time is exactly 20.0 s and the cap is reached together.
    v = [round(float(x), 2) for x in rng.normal(0.0, 0.25, 4100)]
    cases.append(calibrator_case("cap_and_20s_together", v, [k * 5 / 1024 for k in range(4100)], every=256))

    # tests/perception/test_face_valence.py sequences.
    v = [0.1] * 21 + [NAN] * 60 + [0.1] * 25
    t = [0.5 * k for k in range(21)] + [10.5 + 0.5 * k for k in range(60)] + [40.0 + 0.5 * k for k in range(25)]
    cases.append(calibrator_case("hidden_face_does_not_accrue", v, t))
    cases.append(calibrator_case("large_gap_capped", [0.0, 0.0, 0.3, NAN, 0.4, 0.1], [0.0, 100.0, 100.5, 101.0,
                                                                                      200.0, 200.2]))
    cases.append(calibrator_case("nonfinite_t", [0.1, 0.1, 0.1, 0.2, 0.3, NAN, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7],
                                 [0.0, NAN, 0.5, INF, 1.0, 1.4, -INF, 2.0, 2.5, NAN, NAN, 3.0]))

    # Parameters, including degenerate ones where Python's min/max differ from Math.min/Math.max on NaN.
    base_v = [r4(x) for x in rng.normal(0.0, 0.4, 40)]
    base_t = [0.25 * k for k in range(40)]
    for params in ({"calibration_s": 0.0}, {"calibration_s": 3.0, "gain": 1.5, "max_gap_s": 0.25},
                   {"gain": INF}, {"gain": 0.0}, {"gain": -2.0}, {"max_gap_s": NAN}, {"max_gap_s": 0.0},
                   {"calibration_s": NAN}, {"calibration_s": INF, "max_gap_s": INF}, {"gain": NAN}):
        name = "params_" + "_".join(f"{k}={v}" for k, v in params.items())
        cases.append(calibrator_case(name, [0.0] + base_v, [0.0] + [x + 0.25 for x in base_t], params))
    # Huge finite values: the median overflows to inf (and gain 0 turns 0 * inf into NaN, which Python clips to 1).
    v = [1e308, 1.7e308, 1.0, -1e308, 1.6e308, 2.0, 1.7e308]
    cases.append(calibrator_case("overflowing_median", v, [0.1 * k for k in range(len(v))]))
    cases.append(calibrator_case("overflowing_median_gain0", v, [0.1 * k for k in range(len(v))], {"gain": 0.0}))
    return {"cases": cases}


# ---------------------------------------------------------------------------------------------------------------
# face_valence.py: assign_faces


def assign_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(3303)
    cases = []

    def add(name: str, faces: list[list[float]], persons: list[list[float]], frame_w: int,
            max_dist_frac: float = 0.15, min_conf: float = 0.3) -> None:
        noses = {int(p[0]): (p[1], p[2], p[3]) for p in persons}
        out = assign_faces([(f[0], f[1]) for f in faces], noses, frame_w, max_dist_frac, min_conf)
        cases.append({"name": name, "faces": faces, "persons": persons, "frame_w": frame_w,
                      "max_dist_frac": max_dist_frac, "min_conf": min_conf, "out": out})

    # Gate boundary where sqrt(dx*dx + dy*dy) and Math.hypot round to different sides of 96 = 0.15 * 640.
    add("gate_rounding_valid", [[82.32989846953481, 49.37395890544011]], [[3, 0.0, 0.0, 0.9]], 640)
    add("gate_rounding_gated", [[47.964392161577464, 83.15898679379401]], [[3, 0.0, 0.0, 0.9]], 640)
    add("gate_exact", [[196.0, 100.0]], [[3, 100.0, 100.0, 0.9]], 640)
    add("conf_exactly_min", [[100.0, 100.0]], [[4, 101.0, 100.0, 0.3], [5, 99.0, 100.0, 0.29999999999999993]], 640)
    add("nan_face_inf_nose", [[NAN, 1.0], [INF, 2.0], [5.0, 5.0]], [[1, 4.0, 4.0, 1.0], [2, INF, 3.0, 1.0],
                                                                    [3, 5.0, NAN, 1.0], [4, 6.0, 6.0, NAN]], 640)
    add("all_equal_distances", [[120.0, 90.0], [120.0, 110.0]], [[1, 100.0, 100.0, 0.9], [2, 140.0, 100.0, 0.9]], 640)
    add("all_equal_distances_rev", [[120.0, 90.0], [120.0, 110.0]], [[2, 140.0, 100.0, 0.9],
                                                                     [1, 100.0, 100.0, 0.9]], 640)

    for k in range(170):  # gate circle: about half are just inside after rounding
        frame_w = (640, 1280, 400)[k % 3]
        frac = (0.15, 0.05, 0.37)[k // 3 % 3]
        gate = frac * frame_w
        nx, ny = round(float(rng.uniform(0, frame_w)), 2), round(float(rng.uniform(0, frame_w * 0.56)), 2)
        th = float(rng.uniform(0, 2 * math.pi))
        faces = [[nx + gate * math.cos(th), ny + gate * math.sin(th)]]
        persons = [[int(rng.integers(1, 50)), nx, ny, 0.9]]
        if k % 2:
            faces.append([nx - gate * math.sin(th), ny + gate * math.cos(th)])
            persons.append([persons[0][0] + 50, nx + 1.0, ny, 0.8])
        add(f"gate_circle_{k}", faces, persons, frame_w, frac)

    for k in range(260):
        frame_w = (640, 1280, 400, 320)[k % 4]
        frame_h = frame_w * 9 // 16
        n_faces, n_persons = int(rng.integers(0, 7)), int(rng.integers(0, 7))
        pids = [int(p) for p in rng.choice(np.arange(1, 40), n_persons, replace=False)]
        grid = k % 3 == 0  # integer grid: many exactly equal distances (ties)
        persons = []
        for pid in pids:
            if grid:
                x, y = float(rng.integers(0, 6) * 20 + 100), float(rng.integers(0, 4) * 20 + 100)
            else:
                x, y = round(float(rng.uniform(0, frame_w)), 3), round(float(rng.uniform(0, frame_h)), 3)
            u = rng.random()
            conf = 0.3 if u < 0.1 else 0.29999999999999993 if u < 0.15 else round(float(rng.uniform(0, 1)), 3)
            if rng.random() < 0.05:
                x = (NAN, INF, -INF)[int(rng.integers(3))]
            persons.append([pid, x, y, conf])
        faces = []
        for _ in range(n_faces):
            if grid:
                f = [float(rng.integers(0, 6) * 20 + 110), float(rng.integers(0, 4) * 20 + 90)]
            elif persons and rng.random() < 0.7:
                p = persons[int(rng.integers(len(persons)))]
                base_x = p[1] if math.isfinite(p[1]) else frame_w / 2
                f = [round(base_x + float(rng.normal(0, 0.08 * frame_w)), 3),
                     round(p[2] + float(rng.normal(0, 0.08 * frame_w)), 3)]
            else:
                f = [round(float(rng.uniform(0, frame_w)), 3), round(float(rng.uniform(0, frame_h)), 3)]
            if rng.random() < 0.05:
                f[int(rng.integers(2))] = (NAN, INF)[int(rng.integers(2))]
            faces.append(f)
        frac = 0.15 if k % 5 else float((0.05, 0.3, 0.6)[k % 3])
        min_conf = 0.3 if k % 7 else 0.1
        add(f"random_{k}", faces, persons, frame_w, frac, min_conf)

    for k in range(60):  # near-ties: mirror images of the faces around the noses, perturbed by a few ulps or not at all
        cx, cy = float(rng.integers(150, 450)), float(rng.integers(100, 250))
        dx, dy = float(rng.integers(5, 40)), float(rng.integers(5, 40))
        eps = (0.0, 1e-13, 1e-9, 1e-6)[k % 4]
        faces = [[cx + eps, cy - dy], [cx, cy + dy]]
        persons = [[7, cx - dx, cy, 0.9], [3, cx + dx, cy, 0.9]]
        if k % 3 == 0:
            faces.append([cx + dx * 0.5, cy])
        if k % 5 == 0:
            persons.append([11, cx, cy + dy * 0.5 + eps, 0.95])
        add(f"near_tie_{k}", faces, persons, 640)
    return {"cases": cases}


# ---------------------------------------------------------------------------------------------------------------
# scipy.optimize.linear_sum_assignment


def seq_total(m: list[list[float]], rows: list[int], cols: list[int]) -> float:
    total = 0.0  # explicit loop: builtin sum() of floats is compensated since Python 3.12
    for r, c in zip(rows, cols, strict=True):
        total += m[r][c]
    return total


def unique_optimum(m: list[list[float]], total: float) -> bool:
    nr, nc = len(m), len(m[0])
    best: list[float] = []
    if nr <= nc:
        for perm in itertools.permutations(range(nc), nr):
            best.append(seq_total(m, list(range(nr)), list(perm)))
    else:
        for perm in itertools.permutations(range(nr), nc):
            pairs = sorted(zip(perm, range(nc)))
            best.append(seq_total(m, [p[0] for p in pairs], [p[1] for p in pairs]))
    tol = 1e-9 * (1 + abs(total)) if math.isfinite(total) else 0.0
    return sum(1 for x in best if x <= total + tol) == 1


def lsa_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(4404)
    cases = []

    def add(m: list[list[float]]) -> None:
        arr = np.asarray(m, dtype=np.float64) if m and m[0] else np.zeros((len(m), 0))
        try:
            rows, cols = linear_sum_assignment(arr)
        except ValueError as e:
            cases.append({"m": m, "error": str(e)})
            return
        rows_l, cols_l = [int(x) for x in rows], [int(x) for x in cols]
        total = seq_total(m, rows_l, cols_l)
        unique = unique_optimum(m, total) if m and m[0] else True
        cases.append({"m": m, "rows": rows_l, "cols": cols_l, "total": total, "unique": unique})

    add([])
    add([[], []])
    add([[1.0]])
    add([[INF]])
    add([[NAN, 1.0], [1.0, 2.0]])
    add([[-INF, 1.0], [1.0, 2.0]])
    add([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    for k in range(720):
        nr, nc = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        kind = k % 6
        if kind == 0:
            m = rng.random((nr, nc))
        elif kind == 1:
            m = rng.integers(0, 3, (nr, nc)).astype(np.float64)  # ties
        elif kind == 2:
            m = np.round(rng.random((nr, nc)) * 4) / 4
        elif kind == 3:
            m = rng.normal(0, 1e6, (nr, nc)) + rng.random((nr, nc)) * 1e-6
        elif kind == 4:
            m = rng.integers(0, 6, (nr, nc)).astype(np.float64)
            m[rng.random((nr, nc)) < 0.35] = INF
        else:
            m = np.round(rng.normal(0, 3, (nr, nc)), 2)
            if rng.random() < 0.15:
                m[int(rng.integers(nr)), int(rng.integers(nc))] = (NAN, -INF)[int(rng.integers(2))]
        add([[float(x) for x in row] for row in m])
    return {"cases": cases}


# ---------------------------------------------------------------------------------------------------------------
# face_cues.py: head_pose_deg, raw_cues


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pack(rot: np.ndarray, trans: tuple[float, float, float], row_major: bool, scale: float = 1.0) -> list[float]:
    m = np.eye(4)
    m[:3, :3] = rot * scale
    m[:3, 3] = trans
    return [float(m[r, c]) for r in range(4) for c in range(4)] if row_major else \
        [float(m[r, c]) for c in range(4) for r in range(4)]


def rot_y(a: float) -> np.ndarray:
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def rot_x(a: float) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def random_matrix(rng: np.random.Generator) -> list[float]:
    trans = (round(float(rng.uniform(-15, 15)), 3), round(float(rng.uniform(-15, 15)), 3),
             round(float(rng.uniform(-80, -20)), 3))
    scale = 1.0 if rng.random() < 0.7 else float(rng.uniform(0.8, 1.2))
    return pack(random_rotation(rng), trans, bool(rng.random() < 0.5), scale)


def pose_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(5505)
    hp = []

    def add(data: list[float], layout: str) -> None:
        yaw, pitch = head_pose_deg(data)
        hp.append({"data": data, "layout": layout, "yaw": yaw, "pitch": pitch})

    for k in range(240):
        trans = (round(float(rng.uniform(-15, 15)), 3), round(float(rng.uniform(-15, 15)), 3),
                 round(float(rng.uniform(-80, -20)), 3))
        scale = 1.0 if k % 4 else float(rng.uniform(0.8, 1.2))
        rot = random_rotation(rng)
        add(pack(rot, trans, False, scale), "col")
        add(pack(rot, trans, True, scale), "row")
    for a in (math.pi, -math.pi, math.pi - 1e-15, -math.pi + 1e-15, math.pi / 2, 0.0):  # facing away / sideways
        for row_major in (False, True):
            add(pack(rot_y(a), (0.0, 0.0, -40.0), row_major), "row" if row_major else "col")
    for a in (math.pi / 2, -math.pi / 2, math.pi / 2 - 1e-12, 1.2):  # straight up / down
        add(pack(rot_x(-a), (1.0, 2.0, -50.0), False), "col")
    signed_zero = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -40.0, 1.0]
    add(signed_zero, "col")
    add([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], "tie")  # |t14| == |t11|
    add([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.2, 0.9, 7.0, 0.0, 0.0, -7.0, 1.0], "tie")
    add([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.2, 0.9, NAN, 0.0, 0.0, -40.0, 1.0], "nan")
    add([1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.1, 0.0, 0.3, 0.2, 0.9, -30.0, 0.0, 0.0, NAN, 1.0], "nan")
    add([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, NAN, 0.2, 0.9, 0.0, 0.0, 0.0, -40.0, 1.0], "nan")
    add([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, INF, 0.2, INF, 0.0, 0.0, 0.0, -40.0, 1.0], "inf")
    add([0.0] * 16, "zeros")
    add([1.0] * 15, "short")
    add([1.0] * 17, "long")
    add([], "empty")

    rc = []
    for k in range(160):
        bs: dict[str, float | None] = {}
        for name in CUE_SHAPES + EXTRA_SHAPES:
            u = rng.random()
            if u < 0.7:
                bs[name] = round(float(rng.random()), 4)
            elif u < 0.75:
                bs[name] = None
            elif u < 0.78:
                bs[name] = NAN
        v = rng.random()
        matrix: list[float] | None = None if v < 0.15 else [] if v < 0.18 else [1.0] * 12 if v < 0.2 else \
            random_matrix(rng)
        out = raw_cues(bs, matrix)
        rc.append({"bs": bs, "matrix": matrix, "out": [out[key] for key in RAW_KEYS]})
    return {"head_pose": hp, "raw_cues": rc}


# ---------------------------------------------------------------------------------------------------------------
# face_cues.py: FaceCueTracker


class TrackerCase:
    """Feeds one FaceCueTracker frame by frame and records inputs and outputs (frames: null, the six raw channels,
    or {"bs", "matrix"} to go through raw_cues)."""

    def __init__(self, name: str, calibration_s: float | None = None) -> None:
        self.tr = FaceCueTracker() if calibration_s is None else FaceCueTracker(calibration_s=calibration_s)
        self.case: dict[str, Any] = {"name": name, "calibration_s": calibration_s, "frames": [], "t_ms": [],
                                     "out": [], "baseline": []}

    @staticmethod
    def raw(frame: Any) -> dict[str, float] | None:
        if frame is None:
            return None
        if isinstance(frame, dict):
            return raw_cues(frame["bs"], frame["matrix"])
        return dict(zip(RAW_KEYS, frame, strict=True))

    def step(self, frame: Any, t_ms: float) -> dict[str, float]:
        out = self.tr.update(self.raw(frame), t_ms)
        self.case["frames"].append(frame)
        self.case["t_ms"].append(t_ms)
        self.case["out"].append([out[key] for key in CUE_KEYS])
        return out

    def checkpoint(self) -> None:
        b = self.tr.baseline()
        self.case["baseline"].append([len(self.case["t_ms"]) - 1, [b[key] for key in RAW_KEYS]])

    def probe(self, frame: list[float], t_ms: float) -> FaceCueTracker:
        """A copy of the tracker after this frame (the tracker itself is untouched)."""
        probe = copy.deepcopy(self.tr)
        probe.update(self.raw(frame), t_ms)
        return probe

    @property
    def frozen(self) -> bool:
        return self.tr._frozen is not None

    @property
    def closed(self) -> bool:
        return self.tr._closed_at is not None


def neutral(rng: np.random.Generator, yaw: float = 3.0, pitch: float = -4.0) -> list[float]:
    return [r4(0.1 + rng.normal(0, 0.03)), r4(0.08 + abs(rng.normal(0, 0.03))), r4(0.02 + abs(rng.normal(0, 0.02))),
            r4(0.05 + abs(rng.normal(0, 0.04))), r4(yaw + rng.normal(0, 4)), r4(pitch + rng.normal(0, 3))]


def with_closure(frame: list[float], c: float) -> list[float]:
    return [frame[0], c, *frame[2:]]


def tracker_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(6606)
    cases = []

    # 1. Calibrate (with natural blinks and dropouts), freeze, then blinks exactly at the frozen thresholds.
    tc = TrackerCase("calibrate_then_threshold_blinks")
    t = 0.0
    k = 0
    while not tc.frozen or k < 30:
        if tc.frozen:
            k += 1
        u = rng.random()
        frame: Any = None if u < 0.05 else neutral(rng)
        if frame is not None and rng.random() < 0.04:
            frame = with_closure(frame, r4(rng.uniform(0.5, 0.9)))
        tc.step(frame, t)
        if len(tc.case["t_ms"]) % 50 == 0:
            tc.checkpoint()
        t = round(t + float(rng.uniform(55, 80)), 1)
    tc.checkpoint()
    t = float(math.ceil(t))
    b = tc.tr.baseline()["closure"]
    close_at, open_at = b + BLINK_CLOSE, b + BLINK_OPEN
    below_close, above_open = math.nextafter(close_at, -INF), math.nextafter(open_at, INF)
    script: list[tuple[Any, float]] = [  # (closure, None for no face or "all_nan"; dt ms before the frame)
        (below_close, 50), (b, 50),  # just under the close threshold: nothing
        (close_at, 50), (above_open, 300), (open_at, 300),  # closed, still closed, opens after exactly 600 ms: blink
        (b, 50), (close_at, 50), (open_at, 600.00000000001),  # 600 ms + a hair: no blink
        (b, 50), (close_at, 50), (0.95, 200), (open_at, 401),  # 601 ms: no blink
        (b, 50), (close_at, 50), (open_at, 599),  # blink
        (close_at, 66), (None, 66), (open_at, 66),  # a frame without a face resets: no blink
        (close_at, 66), (NAN, 66), (open_at, 66),  # NaN closure on a visible frame: detector waits, blink
        (b, 66), (close_at, 66), ("all_nan", 66), (open_at, 66),  # all-NaN raw = not visible: reset, no blink
        (b, 66), (0.99, 66), (0.99, 66), (b, 66), (0.99, 66), (b, 66),  # two blinks in a row
    ]
    for c, dt in script:
        t += dt
        if c is None:
            tc.step(None, t)
        elif c == "all_nan":
            tc.step([NAN] * 6, t)
        else:
            tc.step(with_closure(neutral(rng), c), t)
    # A 1 s closure is not a blink.
    for k in range(17):
        t += 62.5
        tc.step(with_closure(neutral(rng), 0.8 if k < 16 else b), t)
    tc.checkpoint()
    cases.append(tc.case)

    # 2. Blinks while calibrating: thresholds follow the running median, which already includes the frame (and
    # across the freeze). The closure sits exactly on the threshold the frame will see, or one ulp under it.
    tc = TrackerCase("running_baseline_blinks", calibration_s=6.0)
    t = 0.0
    for k in range(100):
        frame = neutral(rng)
        if k >= 5 and k % 9 == 0:
            b_seen = tc.probe(with_closure(frame, 0.9), t).baseline()["closure"]  # same median for any high closure
            c = b_seen + BLINK_CLOSE
            low = math.nextafter(c, -INF)
            assert tc.probe(with_closure(frame, c), t)._closed_at is not None
            assert tc.probe(with_closure(frame, low), t)._closed_at is None
            tc.step(with_closure(frame, low if k % 18 == 0 else c), t)
            t += 66.0
            frame = neutral(rng)
            b_seen = tc.probe(with_closure(frame, 0.3), t).baseline()["closure"]
            c = b_seen + BLINK_OPEN
            if tc.closed:  # opens exactly on the threshold, one ulp above it stays closed
                assert tc.probe(with_closure(frame, c), t)._closed_at is None
                assert tc.probe(with_closure(frame, math.nextafter(c, INF)), t)._closed_at is not None
            tc.step(with_closure(frame, c), t)
        else:
            tc.step(frame, t)
        t += 66.0
        if k % 10 == 0:
            tc.checkpoint()
    tc.checkpoint()
    cases.append(tc.case)

    # 3. Freezes on the frame where visible time reaches exactly 20 s (500 ms steps).
    tc = TrackerCase("freeze_at_exactly_20s")
    for k in range(46):
        tc.step(neutral(rng), 500.0 * k)
        if 38 <= k <= 42:
            tc.checkpoint()
    cases.append(tc.case)

    # 4. Dropouts, gaps > 1 s, backwards and repeated times, partial NaN channels.
    for name, cal_s in (("dropouts_and_gaps", None), ("dropouts_and_gaps_short_calibration", 4.0)):
        tc = TrackerCase(name, cal_s)
        t = 0.0
        for k in range(420):
            u = rng.random()
            if u < 0.08:
                frame = None
            elif u < 0.11:
                frame = [NAN] * 6
            else:
                frame = neutral(rng)
                if rng.random() < 0.15:
                    frame[4] = frame[5] = NAN  # no matrix
                if rng.random() < 0.05:
                    frame[int(rng.integers(4))] = NAN
                if rng.random() < 0.06:
                    frame = with_closure(frame, r4(rng.uniform(0.4, 0.95)))
            tc.step(frame, t)
            w = rng.random()
            t = round(t + (float(rng.uniform(1200, 6000)) if w < 0.03 else -float(rng.uniform(0, 400)) if w < 0.06
                           else 0.0 if w < 0.08 else float(rng.uniform(30, 90))), 1)
            if k % 40 == 0:
                tc.checkpoint()
        tc.checkpoint()
        cases.append(tc.case)

    # 5. Yaw wrap-around: calibrated facing ~175 deg (samples on both sides of +-180), then a sweep across +-180.
    tc = TrackerCase("yaw_wrap_sideways", calibration_s=2.0)
    t = 0.0
    for _ in range(40):
        f = neutral(rng)
        y = 176.0 + float(rng.normal(0, 3))
        f[4] = r4(y - 360.0 if y > 180.0 else y)
        tc.step(f, t)
        t += 66.0
    tc.checkpoint()
    by = tc.tr.baseline()["yaw"]
    sweep = [r4(-180.0 + 7.3 * i) for i in range(50)] + [-180.0, 180.0, by - 180.0, by + 180.0, by - 90.0, by + 90.0,
                                                          math.nextafter(by - 180.0, -INF), math.nextafter(by + 180.0, INF),
                                                          by + 89.99999999999999, INF, -INF]
    for y in sweep:
        f = neutral(rng)
        f[4] = y
        tc.step(f, t)
        t += 66.0
    cases.append(tc.case)

    # 6. Exactly at the wrap: with a tiny baseline, raw - base lands one ulp outside +-180 (Python's % rounds it).
    for base, yaws in ((2.842170943040401e-14, [-180.0, 180.0, -179.99999999999997, 179.99999999999997, 0.0, -0.0]),
                       (-2.842170943040401e-14, [180.0, -180.0, 179.99999999999997, -179.99999999999997]),
                       (180.0, [-180.0, 0.0, -0.0, 90.0, -90.0, 179.99999999999997]),
                       (-180.0, [180.0, 0.0, -90.0, 90.0, -179.99999999999997]),
                       (1e-300, [-180.0, 180.0])):
        tc = TrackerCase(f"yaw_wrap_boundary_base_{base!r}", calibration_s=0.5)
        tc.step([0.1, 0.1, 0.0, 0.0, base, 0.0], 0.0)
        t = 1000.0
        for y in yaws:
            tc.step([0.1, 0.1, 0.0, 0.0, y, 0.0], t)
            t += 66.0
        tc.checkpoint()
        cases.append(tc.case)

    # 7. Non-finite timestamps (no guard in Python: visible time turns NaN and the baseline freezes at once).
    for name, bad in (("nan_time", NAN), ("inf_time", INF), ("neg_inf_time", -INF)):
        tc = TrackerCase(name, calibration_s=3.0)
        t = 0.0
        for k in range(30):
            tc.step(neutral(rng), bad if k in (8, 9, 15) else t)
            t += 100.0
            if k % 5 == 0:
                tc.checkpoint()
        cases.append(tc.case)

    # 8. Channels that never had a sample freeze at 0.0; inf values enter the median.
    tc = TrackerCase("missing_channels_and_inf", calibration_s=1.0)
    t = 0.0
    for k in range(40):
        f = neutral(rng)
        if k < 20:
            f[3] = f[4] = f[5] = NAN
        if k in (2, 5, 9):
            f[0] = INF
        if k in (3, 6):
            f[2] = -INF
        tc.step(f, t)
        t += 100.0
        if k % 4 == 0:
            tc.checkpoint()
    cases.append(tc.case)

    # 9. Full pipeline: blendshapes + matrix through raw_cues.
    tc = TrackerCase("pipeline_from_blendshapes_and_matrix", calibration_s=3.0)
    t = 0.0
    for k in range(150):
        u = rng.random()
        if u < 0.06:
            frame = None
        else:
            bs: dict[str, float | None] = {}
            for name in CUE_SHAPES:
                w = rng.random()
                if w < 0.9:
                    bs[name] = r4(rng.uniform(0, 0.3) if "eyeBlink" not in name or rng.random() > 0.08 else
                                  rng.uniform(0.6, 1.0))
                elif w < 0.95:
                    bs[name] = None
            yaw, pitch = float(rng.normal(0, 0.3)), float(rng.normal(0, 0.2))
            rot = rot_y(yaw) @ rot_x(-pitch)
            v = rng.random()
            matrix = None if v < 0.1 else pack(rot, (round(float(rng.uniform(-10, 10)), 2),
                                                     round(float(rng.uniform(-10, 10)), 2),
                                                     round(float(rng.uniform(-70, -30)), 2)), v < 0.5)
            frame = {"bs": bs, "matrix": matrix}
        tc.step(frame, t)
        t = round(t + float(rng.uniform(40, 90)), 1)
        if k % 25 == 0:
            tc.checkpoint()
    tc.checkpoint()
    cases.append(tc.case)
    return {"cases": cases}


def export(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir, "face-valence", valence_fixture())
    write(out_dir, "face-calibrator", calibrator_fixture())
    write(out_dir, "face-assign", assign_fixture())
    write(out_dir, "face-lsa", lsa_fixture())
    write(out_dir, "face-pose", pose_fixture())
    write(out_dir, "face-tracker", tracker_fixture())


if __name__ == "__main__":
    export(pathlib.Path(__file__).resolve().parent)
