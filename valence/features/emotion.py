"""Emotion-map window features (feature version ``emotion-v1``; spec: docs/EMOTION_FEATURES.md section 3).

A separate module on purpose: ``compute.py`` and its 123 features are frozen while the web engine ports them.
Inputs are the person's per-tick channel arrays (oldest first, <= 180 ticks), the same windows ``compute`` sees.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from valence.contracts import TickRecord
from valence.features.compute import FACE_FEATURES, FEATURE_SETS, MAX_TICKS, TICK_S, WIN, _nanstat

FEATURE_VERSION = "emotion-v1"

# Tick channels (TickRecord fields) the emotion features read; face_visible gives the visible seconds.
CUE_CHANNELS: dict[str, str] = {
    "brow": "brow_down",
    "eye": "eye_open",
    "frown": "mouth_frown",
    "smile": "smile",
    "yaw": "head_yaw",
    "pitch": "head_pitch",
}
EMOTION_FIELDS: tuple[str, ...] = (*CUE_CHANNELS.values(), "head_away", "blink_count", "face_visible")
BLINK_MIN_VISIBLE_S = 10.0

EMOTION_FEATURE_NAMES: list[str] = [f"{c}_{s}" for c in CUE_CHANNELS for s in ("now", "mean_1m", "mean_5m", "std_5m")]
EMOTION_FEATURE_NAMES += ["eye_mean_15m", "brow_max_1m", "away_mean_1m", "away_mean_5m",
                          "blink_rate_1m", "blink_rate_5m", "blink_rate_15m"]
EMOTION_INDEX = {n: i for i, n in enumerate(EMOTION_FEATURE_NAMES)}

# Model inputs: the served M2 set (frozen 123-feature module, room-relative) + the 31 cue features.
INPUT_FEATURES: list[str] = list(FEATURE_SETS["m2"]) + EMOTION_FEATURE_NAMES
FACE_ONLY_FEATURES: list[str] = list(FACE_FEATURES) + EMOTION_FEATURE_NAMES
CONTEXT_FEATURES: list[str] = [n for n in INPUT_FEATURES if n not in FACE_ONLY_FEATURES]


def _blink_rate(blinks: np.ndarray, visible: np.ndarray, n: int) -> float:
    seconds = float(np.nan_to_num(visible[-n:], nan=0.0).sum()) * TICK_S
    if seconds < BLINK_MIN_VISIBLE_S:
        return math.nan
    return 60.0 * float(np.nan_to_num(blinks[-n:], nan=0.0).sum()) / seconds


def compute_emotion(person: dict[str, np.ndarray]) -> np.ndarray:
    """Person channel arrays (oldest first) -> the 31 features in EMOTION_FEATURE_NAMES order."""
    p = {k: np.asarray(person[k], dtype=float)[-MAX_TICKS:] for k in EMOTION_FIELDS}
    feats: list[float] = []
    for field in CUE_CHANNELS.values():
        x = p[field]
        feats += [float(x[-1]) if x.size else math.nan, _nanstat(np.nanmean, x[-WIN["1m"]:]),
                  _nanstat(np.nanmean, x[-WIN["5m"]:]), _nanstat(np.nanstd, x[-WIN["5m"]:])]
    feats += [
        _nanstat(np.nanmean, p["eye_open"][-WIN["15m"]:]),
        _nanstat(np.nanmax, p["brow_down"][-WIN["1m"]:]),
        _nanstat(np.nanmean, p["head_away"][-WIN["1m"]:]),
        _nanstat(np.nanmean, p["head_away"][-WIN["5m"]:]),
    ]
    feats += [_blink_rate(p["blink_count"], p["face_visible"], WIN[w]) for w in ("1m", "5m", "15m")]
    return np.asarray(feats, dtype=float)


def emotion_from_window(person: Sequence[TickRecord]) -> np.ndarray:
    """Live path: a RollingWindow snapshot -> the 31 emotion features."""
    return compute_emotion({f: np.array([getattr(r, f) for r in person], dtype=float) for f in EMOTION_FIELDS})


def model_input(x123: np.ndarray, x_emotion: np.ndarray) -> np.ndarray:
    """The 123 frozen features + the 31 emotion features -> the emotion models' input (INPUT_FEATURES order)."""
    from valence.features.compute import FEATURE_INDEX

    base = np.asarray([x123[FEATURE_INDEX[n]] for n in FEATURE_SETS["m2"]], dtype=float)
    return np.concatenate([base, np.asarray(x_emotion, dtype=float)])


assert len(EMOTION_FEATURE_NAMES) == 31 and not set(EMOTION_FEATURE_NAMES) & set(FEATURE_SETS["m2"])
