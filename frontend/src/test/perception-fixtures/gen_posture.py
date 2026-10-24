"""Parity fixtures for frontend/src/perception/posture.ts (port of valence/perception/posture.py).

Replays the real valence.perception.posture.PostureTracker over long synthetic sequences of a seated person
(calibration, slumps, keyboard gaze, head in hands, dropouts, hidden ears, L/R swaps, identity jumps, shoulder width
changes, jittered / repeated / decreasing timestamps, non-finite and None inputs, custom parameters, zero widths and
ties that need numpy's fused ddot norm) and records every output per step: the PostureState fields plus baseline /
ear_baseline / shoulder_width after the step.

Writes posture-<part>.json (strict JSON; non-finite floats as "NaN" / "Infinity" / "-Infinity").
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from valence.perception.posture import BODY_PAIRS, PostureTracker  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent
NAN, INF = math.nan, math.inf
HEAD_PAIRS = ((1, 2), (3, 4))
SAFE_INF_KPTS = (1, 2, 3, 4, 7, 8, 9, 10, 11, 12)  # inf here is filtered by isfinite or gated; nose/shoulder inf latches


# --------------------------------------------------------------------------------------------------- encoding


def enc(v: Any) -> Any:
    """Strict-JSON value: non-finite floats become strings; numpy scalars become Python ones."""
    if isinstance(v, np.generic):
        v = v.item()
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if isinstance(v, np.ndarray):
        return enc(v.tolist())
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    raise TypeError(f"cannot encode {type(v)}")


# --------------------------------------------------------------------------------------------------- scenes


def skeleton(cx: float, sy: float, sw: float, hd: float, ed: float) -> np.ndarray:
    """Seated, camera-facing COCO-17 skeleton; shoulders ``sw`` wide at y=``sy`` centred on ``cx``.

    Nose ``hd`` shoulder widths above the shoulder line, ears ``ed`` (plus a little), like tests/perception/test_posture.py
    scaled by ``sw / 200``."""
    s = sw / 200.0
    nose_y = sy - hd * sw
    ear_y = sy - ed * sw - 5 * s
    return np.array([
        (cx, nose_y),
        (cx + 20 * s, nose_y - 15 * s), (cx - 20 * s, nose_y - 15 * s),
        (cx + 40 * s, ear_y), (cx - 40 * s, ear_y),
        (cx + 100 * s, sy), (cx - 100 * s, sy),
        (cx + 120 * s, sy + 120 * s), (cx - 120 * s, sy + 120 * s),
        (cx + 100 * s, sy + 200 * s), (cx - 100 * s, sy + 200 * s),
        (cx + 60 * s, sy + 250 * s), (cx - 60 * s, sy + 250 * s),
        (cx + 60 * s, sy + 400 * s), (cx - 60 * s, sy + 400 * s),
        (cx + 60 * s, sy + 550 * s), (cx - 60 * s, sy + 550 * s),
    ], dtype=float)


PHASE_DEFAULTS: dict[str, Any] = dict(
    n=10, hd=None, ed=None, sw=None, ramp=1, wrists="rest", hide=(), drop_p=0.0, lowband_p=0.0, swap_p=0.0,
    jump_p=0.0, idjump_p=0.0, drift=(0.0, 0.0), body_conf=None, head_artifact=False, thresh_p=0.0, nan_p=0.0,
    inf_p=0.0, hd_noise=0.012, tail_inf=False, tilt=None,
)


def scene(rng: np.random.Generator, phases: list[dict[str, Any]], *, cx: float = 640.0, sy: float = 420.0,
          sw: float = 200.0, hd: float = 0.7) -> list[tuple[np.ndarray, np.ndarray]]:
    """Frames (xy (17, 2), conf (17,)) for a list of phases; targets ramp linearly over ``ramp`` steps."""
    frames = []
    cur = dict(hd=hd, ed=hd, sw=sw, cx=cx, sy=sy, tilt=0.0)
    for spec in phases:
        ph = {**PHASE_DEFAULTS, **spec}
        start = dict(cur)
        hd_t = start["hd"] if ph["hd"] is None else ph["hd"]
        ed_t = hd_t if ph["ed"] is None else ph["ed"]
        sw_t = start["sw"] if ph["sw"] is None else ph["sw"]
        tilt_t = start["tilt"] if ph["tilt"] is None else ph["tilt"]
        for k in range(ph["n"]):
            frac = min(1.0, (k + 1) / ph["ramp"])
            cur["hd"] = start["hd"] + (hd_t - start["hd"]) * frac
            cur["ed"] = start["ed"] + (ed_t - start["ed"]) * frac
            cur["sw"] = start["sw"] + (sw_t - start["sw"]) * frac
            cur["tilt"] = start["tilt"] + (tilt_t - start["tilt"]) * frac
            cur["cx"] += ph["drift"][0] * 0.1
            cur["sy"] += ph["drift"][1] * 0.1
            frames.append(frame(rng, ph, cur))
    return frames


def frame(rng: np.random.Generator, ph: dict[str, Any], cur: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    cx, sy, sw = cur["cx"], cur["sy"], cur["sw"]
    s = sw / 200.0
    hd = cur["hd"] + rng.normal(0.0, ph["hd_noise"])
    ed = cur["ed"] + rng.normal(0.0, ph["hd_noise"])
    xy = skeleton(cx, sy, sw, hd, ed)
    xy[5:13] += rng.normal(0.0, 0.4 * s, (8, 2))
    xy[:5] += rng.normal(0.0, 0.8 * s, (5, 2))
    xy[13:] += rng.normal(0.0, 3.0 * s, (4, 2))
    nose = xy[0].copy()
    mode = ph["wrists"]
    if mode == "rest":
        xy[[7, 8, 9, 10]] += rng.normal(0.0, 1.5 * s, (4, 2))  # typing
    elif mode in ("face", "one_face"):
        xy[9] = nose + (50 * s, 40 * s) + rng.normal(0.0, 3.0 * s, 2)
        if mode == "face":
            xy[10] = nose + (-50 * s, 40 * s) + rng.normal(0.0, 3.0 * s, 2)
    elif mode == "face_edge":  # nearest head point is the ear, ~0.5 sw away: near flickers
        xy[9] = xy[3] + (0.0, 100 * s) + rng.normal(0.0, 4.0 * s, 2)
        xy[10] = xy[4] + (0.0, 100 * s) + rng.normal(0.0, 4.0 * s, 2)
    elif mode == "raised_far":
        xy[9] = (cx + 300 * s, sy - 250 * s) + rng.normal(0.0, 3.0 * s, 2)
        xy[10] = (cx - 300 * s, sy - 250 * s) + rng.normal(0.0, 3.0 * s, 2)
    elif mode == "below":
        xy[9] = (cx + 50 * s, sy + 15 * s) + rng.normal(0.0, 2.0 * s, 2)
        xy[10] = (cx - 50 * s, sy + 15 * s) + rng.normal(0.0, 2.0 * s, 2)
    elif mode == "wave":  # one arm swinging fast
        xy[[7, 9]] += rng.normal(0.0, 25.0 * s, (2, 2))
    else:
        raise ValueError(mode)
    if ph["head_artifact"]:
        xy[:5] += rng.normal(0.0, 0.3 * sw, (5, 2))
    if cur["tilt"]:  # leaning sideways: rotate about the shoulder midpoint (shoulder deltas large in x and y)
        a = math.radians(cur["tilt"])
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        xy = (xy - (cx, sy)) @ rot.T + (cx, sy)

    conf = np.clip(rng.normal(0.88, 0.04, 17), 0.0, 1.0)
    conf[13:] = rng.uniform(0.02, 0.35, 4)
    if ph["head_artifact"]:
        conf[:5] = rng.uniform(0.3, 0.95, 5)
    if ph["body_conf"] is not None:
        conf[5:13] = ph["body_conf"]
    for k in ph["hide"]:
        conf[k] = rng.uniform(0.0, 0.3)
    drops = rng.random(13) < ph["drop_p"]
    conf[:13][drops] = rng.uniform(0.0, 0.49, int(drops.sum()))
    band = rng.random(13) < ph["lowband_p"]  # valid for geometry, below the motion floor
    conf[:13][band] = rng.uniform(0.5, 0.6, int(band.sum()))
    conf = np.round(conf, 2)  # what a JSON fixture stores; the special values below keep their digits

    if rng.random() < ph["swap_p"]:
        pairs = BODY_PAIRS + HEAD_PAIRS
        a, b = pairs[int(rng.integers(len(pairs)))]
        xy[[a, b]] = xy[[b, a]]
        conf[[a, b]] = conf[[b, a]]
    if rng.random() < ph["jump_p"]:
        k = int(rng.integers(5, 13))
        xy[k] = nose if rng.random() < 0.5 else xy[k] + rng.normal(0.0, 1.5 * sw, 2)
    if rng.random() < ph["idjump_p"]:
        xy += (2.5 * sw * (1 if rng.random() < 0.5 else -1), rng.normal(0.0, 0.2 * sw))
    if rng.random() < ph["thresh_p"]:
        conf[int(rng.integers(17))] = (0.5, 0.6, 0.4999, 0.5999, 0.6001, NAN)[int(rng.integers(6))]
    if rng.random() < ph["nan_p"]:
        xy[int(rng.integers(17)), int(rng.integers(2))] = NAN
    if rng.random() < ph["inf_p"]:
        xy[SAFE_INF_KPTS[int(rng.integers(len(SAFE_INF_KPTS)))], int(rng.integers(2))] = (INF, -INF)[
            int(rng.integers(2))]
    if ph["tail_inf"]:  # latching cases: shoulders / nose non-finite with good conf
        k, c = ((0, 1), (5, 0), (6, 1), (5, 1))[int(rng.integers(4))]
        xy[k, c] = (INF, -INF, NAN)[int(rng.integers(3))]
        conf[k] = 0.9
    return xy, conf


def timestamps(rng: np.random.Generator, n: int, *, t0: float = 0.0, jitter: float = 0.01, p_rep: float = 0.01,
               p_back: float = 0.005, p_gap: float = 0.005, special: dict[int, float] | None = None) -> list[float]:
    """~10 Hz with jitter; occasional repeated (dt 0), decreasing and gapped timestamps; rounded to 0.1 ms."""
    out, t = [], t0
    for i in range(n):
        if i:
            u = rng.random()
            if u < p_rep:
                dt = 0.0
            elif u < p_rep + p_back:
                dt = -rng.uniform(0.01, 0.15)
            elif u < p_rep + p_back + p_gap:
                dt = rng.uniform(0.5, 3.0)
            else:
                dt = 0.1 + rng.normal(0.0, jitter)
            t += dt
        out.append(round(t, 4))
    for i, v in (special or {}).items():
        out[i] = v
    return out


def replay(name: str, frames: list[tuple[np.ndarray, np.ndarray]], ts: list[float], params: dict[str, Any] | None = None,
           *, scale: float = 1.0, decimals: int | None = 1,
           nulls: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] | None = None) -> dict[str, Any]:
    """Round the inputs to what the JSON stores, feed them to the real tracker, record every output per step.

    Coordinates keep ``decimals`` digits (the always-hidden knees and ankles ``decimals - 1``; None keeps them exact,
    for constructions that are exact doubles already). ``nulls`` maps a step to (flat xy indices, conf indices) passed
    as None (numpy reads it as NaN; the JSON stores null)."""
    params = params or {}
    tracker = PostureTracker(**params)
    case: dict[str, Any] = {"name": name, "params": params, "t": [], "xy": [], "conf": [], "slump": [],
                            "head_in_hands": [], "motion": [], "head_drop": [], "calibrated": [], "baseline": [],
                            "ear_baseline": [], "shoulder_width": []}
    assert len(frames) == len(ts)
    for i, ((xy, conf), t) in enumerate(zip(frames, ts)):
        xy = xy * scale
        if decimals is not None:
            xy[:13] = np.round(xy[:13], decimals)
            xy[13:] = np.round(xy[13:], decimals - 1)
        flat: list[float | None] = [float(v) for v in xy.reshape(-1)]
        cl: list[float | None] = [float(v) for v in conf]
        xy_in: Any = xy
        conf_in: Any = conf
        if nulls and i in nulls:  # nested lists with None, as a JSON-fed caller would pass them
            for j in nulls[i][0]:
                flat[j] = None
            for j in nulls[i][1]:
                cl[j] = None
            xy_in, conf_in = [flat[2 * k: 2 * k + 2] for k in range(17)], cl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # mean of an empty slice when motion_min_points <= 0
            state = tracker.update(xy_in, conf_in, t)
        case["t"].append(float(t))
        case["xy"].append(flat)
        case["conf"].append(cl)
        case["slump"].append(bool(state.slump))
        case["head_in_hands"].append(bool(state.head_in_hands))
        case["motion"].append(float(state.motion))
        case["head_drop"].append(float(state.head_drop))
        case["calibrated"].append(bool(state.calibrated))
        case["baseline"].append(None if tracker.baseline is None else float(tracker.baseline))
        case["ear_baseline"].append(None if tracker.ear_baseline is None else float(tracker.ear_baseline))
        case["shoulder_width"].append(float(tracker.shoulder_width))
    return case


def sequence(name: str, seed: int, phases: list[dict[str, Any]], params: dict[str, Any] | None = None, *,
             timing: dict[str, Any] | None = None, scene_kw: dict[str, Any] | None = None, scale: float = 1.0,
             decimals: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frames = scene(rng, phases, **(scene_kw or {}))
    ts = timestamps(np.random.default_rng(seed + 1), len(frames), **(timing or {}))
    return replay(name, frames, ts, params, scale=scale, decimals=decimals)


# --------------------------------------------------------------------------------------------------- cases


def calibration_cases() -> list[dict[str, Any]]:
    restart = sequence("restart_then_extended_calibration", 11, [
        dict(n=190, hd=-0.2),  # head below the shoulder line: median <= 0 restarts calibration
        dict(n=230, hd=0.7, ramp=10, hide=(0,)),  # new window passes with the nose hidden (ears still sampled)
        dict(n=30, hd=0.7),  # the first 10 nose samples finish it
        dict(n=50, hd=0.3, ramp=10),
        dict(n=20, hd=0.7, ramp=5),
    ], timing=dict(p_gap=0.0))
    ears_hidden = sequence("ears_hidden_nose_only_rule", 12, [
        dict(n=210, hd=0.7, hide=(3, 4)),  # ears never calibrated
        dict(n=50, hd=0.45, ramp=10, hide=(3, 4)),  # 0.64x: not below the 0.6x nose-only rule
        dict(n=50, hd=0.35, ramp=10, hide=(3, 4)),  # 0.5x: slump
        dict(n=40, hd=0.35),  # ears back, but no ear baseline: still nose-only
        dict(n=20, hd=0.7, ramp=5),
    ])
    ears_rare = sequence("ears_rare_in_window", 13, [
        dict(n=8, hd=0.7),  # 8 ear samples: fewer than 10, so no ear baseline
        dict(n=202, hd=0.7, hide=(3, 4)),
        dict(n=40, hd=0.45, ramp=8),
        dict(n=40, hd=0.38, ramp=8, ed=0.7),
        dict(n=20, hd=0.7, ramp=4),
    ])
    ears_low = sequence("ear_median_not_positive", 16, [
        dict(n=70, hd=0.5, ed=-0.05, ramp=1),  # ears under the shoulder line: ear baseline rejected
        dict(n=40, hd=0.2, ramp=5, ed=-0.05),
        dict(n=30, hd=0.5, ramp=3, ed=0.5),
    ], params=dict(baseline_s=5.0))
    timing = sequence("timestamps_irregular", 14, [
        dict(n=220, hd=0.65, drop_p=0.03),
        dict(n=60, hd=0.3, ramp=15),
        dict(n=50, hd=0.65, ramp=5, wrists="face", hide=(0,)),
        dict(n=40, hd=0.65, drift=(40.0, 0.0)),
        dict(n=40, hd=0.25, ramp=20),
    ], timing=dict(t0=-25.0, jitter=0.02, p_rep=0.05, p_back=0.04, p_gap=0.03,
                   special={3: NAN, 300: NAN, 301: NAN, 340: INF, 380: NAN}))
    nan_t_in_window = sequence("nan_timestamp_ends_window", 15, [
        dict(n=50, hd=0.7),
        dict(n=50, hd=0.3, ramp=10),
        dict(n=30, hd=0.7, ramp=3),
    ], timing=dict(special={30: NAN}))  # a NaN t is outside the window: calibrates on the spot (>= 10 samples)
    return [restart, ears_hidden, ears_rare, ears_low, timing, nan_t_in_window]


def slump_cases() -> list[dict[str, Any]]:
    cycles = sequence("slump_cycles", 21, [
        dict(n=220, hd=0.7),
        dict(n=60, hd=0.3, ramp=25),
        dict(n=30, hd=0.72, ramp=8),
        dict(n=18, hd=0.35, ramp=3),  # ~1.8 s dip: no slump
        dict(n=20, hd=0.72, ramp=3),
        dict(n=26, hd=0.35, ramp=3),  # ~2.6 s
        dict(n=20, hd=0.72, ramp=3),
        dict(n=36, hd=0.35, ramp=3),  # ~3.6 s: slump
        dict(n=20, hd=0.72, ramp=3),
        dict(n=50, hd=0.5, ramp=10, hd_noise=0.03),  # mild lean around 0.7 x baseline
        dict(n=60, hd=0.2, ramp=30),
        dict(n=80, hd=0.3, head_artifact=True),  # held slump: nose/eyes/ears wander and flicker
        dict(n=20, hd=0.7, ramp=4),
    ])
    gaze = sequence("keyboard_gaze_vs_slump", 22, [
        dict(n=215, hd=0.68),
        dict(n=70, hd=0.45, ed=0.68, ramp=10),  # head pitched: nose 0.66x, ears ~1x
        dict(n=40, hd=0.45, ed=0.68, hide=(3, 4)),  # ears hidden: nose-only rule, 0.66x > 0.6x
        dict(n=45, hd=0.38, ed=0.66, hide=(3, 4)),  # 0.56x: slump by the fallback
        dict(n=40, hd=0.38, ed=0.66),  # ears back at ~0.97x: not a slump
        dict(n=60, hd=0.4, ed=0.4, ramp=15),  # whole head sinks
        dict(n=40, hd=0.4, ed=0.4, hide=(4,)),  # one ear is enough
        dict(n=20, hd=0.68, ramp=5),
    ])
    return [cycles, gaze]


def hands_cases() -> list[dict[str, Any]]:
    hands = sequence("head_in_hands_episodes", 31, [
        dict(n=6, hd=0.7, wrists="face", hide=(5, 6)),  # no shoulder width yet: no drop, no hands, no motion
        dict(n=15, hd=0.7),
        dict(n=35, wrists="face", hide=(0,)),  # works before calibration
        dict(n=12),
        dict(n=28, wrists="face", hide=(0, 1, 2)),  # ears only
        dict(n=25, wrists="face", hide=(0, 1, 2, 3, 4)),  # no head point
        dict(n=25, wrists="one_face", hide=(0,)),
        dict(n=25, wrists="raised_far"),
        dict(n=60),
        dict(n=35, hd=0.05, ramp=5, wrists="below"),  # hands near the face but under the shoulder line
        dict(n=60, hd=0.7, ramp=5, wrists="face_edge"),  # distance flickers around 0.5 sw
        dict(n=25, wrists="face", hide=(10,)),  # occluded wrist
        dict(n=35, wrists="face", hide=(0, 6)),  # one shoulder: width reused
        dict(n=20, wrists="face", hide=(5, 6)),  # no shoulder line
        dict(n=60, hd=0.3, ramp=10, wrists="face"),  # in hands while slumped
        dict(n=20, hd=0.7, ramp=5),
        dict(n=35, wrists="face", drop_p=0.1, hide=(0,)),
    ])
    return [hands]


def motion_cases() -> list[dict[str, Any]]:
    swaps = sequence("swaps_jumps_width_changes", 41, [
        dict(n=215, hd=0.7, swap_p=0.05, jump_p=0.03, idjump_p=0.01),
        dict(n=50, sw=260.0, ramp=30, swap_p=0.05, jump_p=0.03),  # leaning in
        dict(n=50, sw=150.0, ramp=40, drift=(-30.0, 5.0), swap_p=0.08),  # leaning back, sliding
        dict(n=40, hd=0.3, ramp=10, swap_p=0.05, idjump_p=0.03),
        dict(n=35, hd=0.3, hide=(7, 8, 9, 10, 11, 12)),  # arms out of frame: torso only
        dict(n=35, sw=200.0, hd=0.7, ramp=10, wrists="wave", jump_p=0.05),
        dict(n=25, swap_p=0.3),
        dict(n=45, tilt=28.0, ramp=15, swap_p=0.05),
        dict(n=20, tilt=-15.0, ramp=10, hd=0.3),
    ])
    paths = sequence("motion_paths", 42, [
        dict(n=25, hd=0.7),
        dict(n=35, drift=(100.0, 0.0)),  # 0.5 sw/s
        dict(n=25, body_conf=0.55),  # below the motion floor: holds
        dict(n=25, hide=(6, 8, 9, 10, 11, 12)),  # left shoulder + elbow only: too few, no torso
        dict(n=25, drift=(0.0, 60.0)),
        dict(n=25, lowband_p=0.4),
        dict(n=25, drift=(400.0, 0.0)),  # ~2 sw/s: some steps over the jump gate
        dict(n=30, wrists="wave"),
        dict(n=25, drop_p=0.45),
    ], timing=dict(p_rep=0.03, p_back=0.03))
    return [swaps, paths]


def dropout_cases() -> list[dict[str, Any]]:
    drops = sequence("random_dropouts", 51, [
        dict(n=215, hd=0.7, drop_p=0.15, lowband_p=0.05),
        dict(n=60, hd=0.3, ramp=10, drop_p=0.15),
        dict(n=15, drop_p=1.0),  # person lost: nothing valid
        dict(n=40, hd=0.7, ramp=5, drop_p=0.3, lowband_p=0.1),
        dict(n=35, wrists="face", drop_p=0.15),
        dict(n=15, drop_p=1.0),
        dict(n=50, hd=0.3, ramp=10, drop_p=0.25, hide=(6,)),
    ])
    nonfinite = sequence("nonfinite_inputs", 52, [
        dict(n=210, hd=0.7, nan_p=0.15, inf_p=0.1, thresh_p=0.2),
        dict(n=50, hd=0.3, ramp=10, nan_p=0.2, inf_p=0.1, thresh_p=0.2),
        dict(n=35, wrists="face", nan_p=0.2, inf_p=0.15, thresh_p=0.2),
        dict(n=25, hd=0.7, ramp=5, thresh_p=0.5),
        dict(n=12, tail_inf=True),  # nose / shoulder inf or NaN with good conf (EMA latches at inf)
    ])
    return [drops, nonfinite]


def param_cases() -> list[dict[str, Any]]:
    fast = sequence("params_fast", 61, [
        dict(n=60, hd=0.7, drop_p=0.05),
        dict(n=35, hd=0.5, ramp=5),
        dict(n=25, hd=0.7, ramp=3),
        dict(n=30, wrists="face_edge", hide=(0,)),
        dict(n=35, drift=(60.0, 0.0), swap_p=0.1, jump_p=0.1),
        dict(n=35, hd=0.5, ed=0.7, ramp=5),
        dict(n=35, hd=0.45, ramp=5, hide=(3, 4)),
        dict(n=25, lowband_p=0.3),
    ], params=dict(baseline_s=5.0, slump_ratio=0.8, slump_hold_s=1.0, hih_hold_s=0.5, hih_radius=0.8,
                   ema_alpha=1.0, ear_ratio=0.97, nose_only_ratio=0.75, motion_jump=0.5, motion_min_points=1,
                   min_conf=0.7, motion_min_conf=0.6))
    zero = sequence("params_zero_holds", 62, [
        dict(n=40, hd=0.7, drop_p=0.2),
        dict(n=40, hd=0.3, ramp=5, drop_p=0.2),
        dict(n=30, wrists="face", drop_p=0.2),
        dict(n=40, hd=0.7, drift=(30.0, 0.0), drop_p=0.4),
    ], params=dict(baseline_s=0.0, slump_hold_s=0.0, hih_hold_s=0.0, ema_alpha=0.05, motion_min_points=0,
                   min_conf=0.3, motion_min_conf=0.9))
    normalized = sequence("normalized_coordinates", 63, [
        dict(n=210, hd=0.7, drop_p=0.05),
        dict(n=40, hd=0.3, ramp=10),
        dict(n=30, wrists="face", hide=(0,)),
        dict(n=30, hd=0.7, drift=(80.0, 0.0), swap_p=0.1),
        dict(n=30, tilt=-22.0, ramp=10),
    ], scale=1 / 1280, decimals=5)
    return [fast, zero, normalized]


def tpose(head_drop: float = 0.7, shift: tuple[float, float] = (0.0, 0.0),
          wrists: tuple[tuple[float, float], tuple[float, float]] | None = None,
          hidden: tuple[int, ...] = (), ear_drop: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """tests/perception/test_posture.py's pose(): integer pixels, shoulders 200 px wide at y=400."""
    nose_y = 400.0 - head_drop * 200.0
    xy = np.array([
        (600, nose_y), (620, nose_y - 15), (580, nose_y - 15), (640, nose_y - 5), (560, nose_y - 5),
        (700, 400), (500, 400), (720, 520), (480, 520), (700, 600), (500, 600), (660, 650), (540, 650),
        (660, 800), (540, 800), (660, 950), (540, 950),
    ], dtype=float)
    if ear_drop is not None:
        xy[3, 1] = xy[4, 1] = 400.0 - ear_drop * 200.0
    if wrists is not None:
        xy[9], xy[10] = wrists
    xy += np.asarray(shift, dtype=float)
    conf = np.full(17, 0.9)
    conf[13:] = 0.0
    conf[list(hidden)] = 0.0
    return xy, conf


def boundary_case() -> dict[str, Any]:
    """Exact constructions: half-open window, radius / shoulder-line / jump-gate equality, float-rounded holds."""
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    values = [0.6, 0.7, 0.8, 0.75, 0.2]
    frames += [tpose(head_drop=values[i % 5]) for i in range(200)]  # t 0..19.9: window
    frames += [tpose()]  # t 20.0: calibrates (median 0.7)
    edge = ((700.0, 335.0), (500.0, 335.0))  # exactly 100 px (0.5 sw) from the ears, 120+ from the rest
    frames += [tpose(wrists=edge, hidden=(0,)) for _ in range(25)]
    frames += [tpose(wrists=((701.0, 335.0), (500.0, 335.0)), hidden=(0,)) for _ in range(10)]
    frames += [tpose(head_drop=0.05, wrists=((650.0, 400.0), (550.0, 400.0))) for _ in range(10)]  # on the line
    frames += [tpose(head_drop=0.05, wrists=((650.0, 399.0), (550.0, 399.0))) for _ in range(25)]
    frames += [tpose() for _ in range(5)]
    frames += [tpose(shift=(200.0, 0.0)) for _ in range(5)]  # exactly 1 sw in one sample: counts
    frames += [tpose(shift=(401.0, 0.0)) for _ in range(5)]  # 201 px: gated, held
    frames += [tpose(shift=(401.0, 0.0), head_drop=0.3) for _ in range(40)]  # slump hold over rounded times
    frames += [tpose(shift=(401.0, 0.0), head_drop=0.45, ear_drop=0.7) for _ in range(30)]
    frames += [tpose(shift=(401.0, 0.0), head_drop=0.49) for _ in range(40)]
    swap_xy, swap_conf = tpose(shift=(401.0, 0.0))
    for i in range(20):
        xy, conf = swap_xy.copy(), swap_conf.copy()
        if i % 4 == 1:
            xy[[9, 10]] = xy[[10, 9]]
        if i % 4 == 2:
            xy[[5, 6]] = xy[[6, 5]]
        frames.append((xy, conf))
    ts = [round(i / 10, 6) for i in range(len(frames))]
    return replay("boundaries_exact", frames, ts)


def rounding_case() -> dict[str, Any]:
    """TIME_EPS at work: on a 0.1 s grid 3.3 - 1.3, 5.6 - 3.6 and 9.2 - 6.2 fall just short of 2, 2 and 3."""
    ks = list(range(13, 125))
    ts = [round(k / 10, 6) for k in ks]
    assert 2.0 - 1e-9 <= 3.3 - 1.3 < 2.0 and 2.0 - 1e-9 <= 5.6 - 3.6 < 2.0 and 3.0 - 1e-9 <= 9.2 - 6.2 < 3.0
    frames = []
    for k in ks:
        if 36 <= k <= 60:  # hands on the head from 3.6: held 2 s at 5.6
            frames.append(tpose(wrists=((650.0, 300.0), (550.0, 300.0)), hidden=(0,)))
        elif 62 <= k <= 95:  # slumped from 6.2: held 3 s at 9.2
            frames.append(tpose(head_drop=0.1))
        else:  # calibration window 1.3 .. 3.3 (2 s, half-open, closes at 3.3)
            frames.append(tpose())
    return replay("float_rounding_holds", frames, ts, dict(baseline_s=2.0, ema_alpha=1.0))


def ddot_fma_norm(dx: float, dy: float) -> float:
    """np.linalg.norm of a 2-vector: sqrt(v.dot(v)), a BLAS ddot that fuses its multiply-adds (posture.ts emulates)."""
    return math.sqrt(math.fma(dy, dy, dx * dx))


def degenerate_case() -> dict[str, Any]:
    """Exact constructions around the shoulder width: a zero / infinite width, an infinite head drop under ema_alpha 1
    (1.0 * v + 0.0 * inf is NaN), null inputs, and ties that only hold with numpy's fused ddot norm.

    Ties: shoulders R = (0, 0), L = (a, b), the ear at (0, 0) the only head point and the wrists at (+-a/2, -b/2), so
    every wrist sits at exactly half the unfused norm of (a, b) while the radius is half the fused one; then the
    shoulders alternate between that pose and one shifted by exactly (a, b), so each step moves them the unfused norm
    against a jump gate of the fused one (and the L/R swap test ties: 0 + 2n vs n + n keeps the labels). Where the
    fused norm is 1 ulp shorter there are no hands and the motion is gated (held); where 1 ulp longer, both count."""
    ties = (((150.0, 97.9), True), ((150.0, 88.9), False), ((150.1, 71.8), True), ((150.2, 85.6), False))
    for (a, b), shorter in ties:
        fused, plain = float(np.linalg.norm(np.array([a, b]))), math.sqrt(a * a + b * b)
        if fused != ddot_fma_norm(a, b) or (fused < plain) != shorter:
            raise SystemExit(f"np.linalg.norm([{a}, {b}]) = {fused!r}: this numpy's BLAS ddot does not fuse "
                             "multiply-adds, so the fused-norm ties in posture-edges would not hold; regenerate on a "
                             "machine whose OpenBLAS uses FMA (x86-64 Haswell+ or arm64)")

    def bare(points: dict[int, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        xy, conf = np.zeros((17, 2)), np.zeros(17)
        for k, p in points.items():
            xy[k], conf[k] = p, 0.9
        return xy, conf

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    nulls: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    same = tpose()
    same[0][6] = same[0][5]  # both shoulders on one point: width 0 is never used
    frames += [(same[0].copy(), same[1].copy()) for _ in range(3)]
    frames += [tpose() for _ in range(15)]  # window [0, 1) holds 7 nose samples: calibrates on the 10th, at t 1.2
    frames += [(same[0].copy(), same[1].copy()) for _ in range(3)]  # width 0 again: the last width is kept
    inf_y = tpose()
    inf_y[0][5, 1] = INF  # infinite shoulder y with good conf: width skipped, head and ear drop inf
    frames += [inf_y, tpose(), tpose()]  # then 1.0 * v + 0.0 * inf = NaN, then v
    inf_x = tpose(head_drop=0.3)
    inf_x[0][6, 0] = -INF
    frames += [inf_x, tpose(head_drop=0.3)]
    nulls[len(frames)] = ((1, 6 * 2, 3 * 2 + 1), (4,))  # nose y, R shoulder x, L ear y as None; R ear conf None
    frames += [tpose(head_drop=0.3)]
    nulls[len(frames)] = ((), (0, 5, 9))
    frames += [tpose(head_drop=0.3)]
    frames += [tpose(head_drop=0.3) for _ in range(6)]
    for (a, b), _ in ties:
        hands = bare({3: (0.0, 0.0), 5: (a, b), 6: (0.0, 0.0), 9: (a / 2, -b / 2), 10: (-a / 2, -b / 2)})
        frames += [(hands[0].copy(), hands[1].copy()) for _ in range(10)]
        for i in range(8):
            frames.append(bare({5: (a, b), 6: (0.0, 0.0)} if i % 2 == 0 else {5: (2 * a, 2 * b), 6: (a, b)}))
    nulls[len(frames)] = (tuple(range(34)), tuple(range(17)))  # all None: nothing valid
    frames += [tpose()]
    ts = [round(i / 10, 6) for i in range(len(frames))]
    return replay("degenerate_widths_and_fused_norm_ties", frames, ts,
                  dict(ema_alpha=1.0, baseline_s=1.0, hih_hold_s=0.5, slump_hold_s=0.3), decimals=None, nulls=nulls)


def edge_cases() -> list[dict[str, Any]]:
    restart = sequence("restart_opens_a_new_window", 64, [
        dict(n=55, hd=-0.2),  # 5 s window, median < 0: restarts with t0 = now
        dict(n=80, hd=0.7, ramp=3),  # calibrates 5 s after the restart, not after 10 more samples
        dict(n=30, hd=0.3, ramp=5),
    ], params=dict(baseline_s=5.0), timing=dict(p_gap=0.0, p_back=0.0))
    return [boundary_case(), rounding_case(), restart, degenerate_case()]


def constructor_cases() -> list[dict[str, Any]]:
    out = []
    for params in [{}, {"ema_alpha": 0.0}, {"ema_alpha": -0.1}, {"ema_alpha": 1.0}, {"ema_alpha": 1.0000001},
                   {"ema_alpha": NAN}, {"ema_alpha": 1e-12}, {"min_conf": 0.7, "motion_min_conf": 0.6},
                   {"min_conf": 0.5, "motion_min_conf": NAN}, {"min_conf": NAN, "motion_min_conf": 0.6},
                   {"min_conf": 0.6, "motion_min_conf": 0.6}, {"min_conf": 0.2, "motion_min_conf": 0.9}]:
        try:
            tracker = PostureTracker(**params)
        except ValueError:
            out.append({"params": params, "error": True, "motion_min_conf": None})
        else:
            out.append({"params": params, "error": False, "motion_min_conf": float(tracker.motion_min_conf)})
    return out


def write(out_dir: Path, part: str, payload: dict[str, Any]) -> None:
    text = json.dumps(enc(payload), allow_nan=False, separators=(",", ":"))
    (out_dir / f"posture-{part}.json").write_text(text + "\n", encoding="utf-8")


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir, "calibration", {"cases": calibration_cases()})
    write(out_dir, "slump", {"cases": slump_cases()})
    write(out_dir, "motion", {"cases": hands_cases() + motion_cases()})
    write(out_dir, "edges", {"cases": dropout_cases() + param_cases() + edge_cases(),
                             "constructor_cases": constructor_cases()})


if __name__ == "__main__":
    export(DEFAULT_OUT)
