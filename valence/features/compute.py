"""The one feature function: rolling window -> Rolling State Vector.

Used unchanged by offline training (DataFrame slices) and live inference (TickRecord lists from the
RollingWindow); ``tests/test_feature_parity.py`` checks both paths produce identical vectors.

Inputs are two windows of per-tick channel arrays, oldest first: the person's own window and the room
slot's window (room history is continuous even when the person just arrived). Missing values stay NaN.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np

from valence.contracts import TickRecord

TICK_S = 5.0
WIN = {"1m": 12, "5m": 60, "15m": 180}
MAX_TICKS = 180

ROOM_CONT = {"db": "ambient_db", "lux": "light_lux"}
PERSON_CONT = {"motion": "motion_energy", "slump": "posture_slump", "hih": "head_in_hands_frac"}
EVENTS = {"fist": ("person", "clenched_fists_count"), "laugh": ("room", "laughter_count"),
          "compl": ("room", "compliments_count")}
PERSON_FIELDS = ("motion_energy", "posture_slump", "head_in_hands_frac", "clenched_fists_count",
                 "social_withdrawal_flag", "n_people", "hours_since_break", "tod_h", "face_valence", "face_visible")
ROOM_FIELDS = ("ambient_db", "light_lux", "laughter_count", "compliments_count")


def _cont_names(ch: str) -> list[str]:
    names = [f"{ch}_now"]
    for w in WIN:
        names += [f"{ch}_mean_{w}", f"{ch}_std_{w}", f"{ch}_slope_{w}", f"{ch}_max_{w}"]
    return names + [f"{ch}_ewm_1m", f"{ch}_ewm_5m", f"{ch}_d1m15m", f"{ch}_rel_15m"]


CONTINUOUS = [*ROOM_CONT, *PERSON_CONT]
FEATURE_NAMES: list[str] = [n for ch in CONTINUOUS for n in _cont_names(ch)]
for _e in EVENTS:
    FEATURE_NAMES += [f"{_e}_cnt_now", f"{_e}_cnt_1m", f"{_e}_cnt_5m", f"{_e}_cnt_15m", f"{_e}_since_s"]
FEATURE_NAMES += ["withdraw_now", "withdraw_frac_5m", "n_people", "hours_since_break", "tod_sin", "tod_cos",
                  "fill_ratio"]
FACE_FEATURES = ["face_now", "face_visible_now", "face_mean_1m", "face_mean_5m", "face_mean_15m", "face_std_5m",
                 "face_visible_frac_5m"]
FEATURE_NAMES += FACE_FEATURES
FEATURE_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}

_ABS_DB = [n for n in FEATURE_NAMES if n.startswith("db_") and n.split("_")[1] in ("now", "mean", "max", "ewm")]
SNAPSHOT = ["db_now", "lux_now", "motion_now", "slump_now", "hih_now", "fist_cnt_now", "laugh_cnt_now",
            "compl_cnt_now", "withdraw_now", "n_people", "hours_since_break", "tod_sin", "tod_cos"]
# 1-minute dB/lux slopes and spreads are dominated by transients live (typing starting, doors, webcam
# auto-exposure): computed and shown, but not used by the served context models (live capture run 1).
NOISY_LIVE = ["db_slope_1m", "db_std_1m", "lux_slope_1m", "lux_std_1m"]
CONTEXT = [n for n in FEATURE_NAMES if n not in FACE_FEATURES and n not in NOISY_LIVE]
_M2 = [n for n in FEATURE_NAMES if n not in NOISY_LIVE]

# Feature sets of the model ladder (docs/PLAN.md "Model ladder").
FEATURE_SETS: dict[str, list[str]] = {
    "m0": ["face_now", "face_visible_now"],
    "m0t": FACE_FEATURES,
    "m1": SNAPSHOT,
    "m2_noface": CONTEXT,
    "m2_noabsdb": [n for n in _M2 if n not in _ABS_DB],
    "m2_notime": [n for n in _M2 if n not in ("hours_since_break", "tod_sin", "tod_cos")],
    "m2": _M2,
}


def arrays_from_records(records: Sequence[TickRecord], fields: Sequence[str]) -> dict[str, np.ndarray]:
    return {f: np.array([getattr(r, f) for r in records], dtype=float) for f in fields}


def _nanstat(fn, x: np.ndarray) -> float:
    if x.size == 0 or np.all(np.isnan(x)):
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(fn(x))


def _slope_per_min(x: np.ndarray) -> float:
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return math.nan
    t = np.arange(x.size, dtype=float)[ok] * (TICK_S / 60.0)
    y = x[ok]
    t = t - t.mean()
    denom = float((t * t).sum())
    return math.nan if denom == 0 else float((t * (y - y.mean())).sum() / denom)


def _ewm(x: np.ndarray, half_life_ticks: float) -> float:
    ok = ~np.isnan(x)
    if not ok.any():
        return math.nan
    age = (x.size - 1 - np.arange(x.size, dtype=float))[ok]
    w = np.power(0.5, age / half_life_ticks)
    return float((w * x[ok]).sum() / w.sum())


def _continuous(x: np.ndarray) -> list[float]:
    now = float(x[-1]) if x.size else math.nan
    out = [now]
    for n in WIN.values():
        seg = x[-n:]
        out += [_nanstat(np.nanmean, seg), _nanstat(np.nanstd, seg), _slope_per_min(seg), _nanstat(np.nanmax, seg)]
    m1, m15 = out[1], out[9]
    med15 = _nanstat(np.nanmedian, x[-WIN["15m"]:])
    out += [_ewm(x, WIN["1m"]), _ewm(x, WIN["5m"]), m1 - m15, now - med15]
    return out


def _events(x: np.ndarray) -> list[float]:
    x = np.nan_to_num(x, nan=0.0)
    now = float(x[-1]) if x.size else 0.0
    counts = [float(x[-n:].sum()) for n in WIN.values()]
    hits = np.nonzero(x > 0)[0]
    since = 900.0 if hits.size == 0 else min(900.0, (x.size - 1 - hits[-1]) * TICK_S)
    return [now, *counts, since]


def compute(person: dict[str, np.ndarray], room: dict[str, np.ndarray]) -> np.ndarray:
    """Person/room channel arrays (oldest first, <= 180 ticks each) -> feature vector in FEATURE_NAMES order."""
    p = {k: v[-MAX_TICKS:] for k, v in person.items()}
    r = {k: v[-MAX_TICKS:] for k, v in room.items()}
    feats: list[float] = []
    for field in ROOM_CONT.values():
        feats += _continuous(r[field])
    for field in PERSON_CONT.values():
        feats += _continuous(p[field])
    for scope, field in EVENTS.values():
        feats += _events((p if scope == "person" else r)[field])
    withdraw = np.nan_to_num(p["social_withdrawal_flag"], nan=0.0)
    tod = float(p["tod_h"][-1])
    feats += [
        float(withdraw[-1]), float(withdraw[-WIN["5m"]:].mean()), float(p["n_people"][-1]),
        float(p["hours_since_break"][-1]),
        math.sin(2 * math.pi * tod / 24) if not math.isnan(tod) else math.nan,
        math.cos(2 * math.pi * tod / 24) if not math.isnan(tod) else math.nan,
        p["n_people"].size / MAX_TICKS,
    ]
    face, vis = p["face_valence"], p["face_visible"]
    feats += [
        float(face[-1]), float(vis[-1]), _nanstat(np.nanmean, face[-WIN["1m"]:]), _nanstat(np.nanmean, face[-WIN["5m"]:]),
        _nanstat(np.nanmean, face[-WIN["15m"]:]), _nanstat(np.nanstd, face[-WIN["5m"]:]),
        _nanstat(np.nanmean, vis[-WIN["5m"]:]),
    ]
    return np.asarray(feats, dtype=float)


def features_from_window(person: Sequence[TickRecord], room: Sequence[TickRecord]) -> np.ndarray:
    """Live path: RollingWindow snapshots -> feature vector."""
    p = arrays_from_records(person, PERSON_FIELDS)
    # Room fields also live in person rows (copied each tick); room slot history takes precedence.
    r = arrays_from_records(room, ROOM_FIELDS) if room else arrays_from_records(person, ROOM_FIELDS)
    return compute(p, r)


assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
