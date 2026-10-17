"""Per-person posture signals from COCO-17 pose keypoints.

Turns one person's keypoint stream into three somatic signals: a slump (head sinking
toward the shoulders relative to that person's own calibrated baseline), head in
hands, and overall body motion. Everything is normalized by shoulder width so it is
independent of distance to the camera. The tracker keeps only a handful of scalars
plus the previous sample's 17x2 keypoint coordinates; it never sees pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from valence.perception.debounce import TIME_EPS

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
N_KPTS = 17
MIN_BASELINE_SAMPLES = 10


@dataclass(slots=True, frozen=True)
class PostureState:
    """One sample's posture signals; ``head_drop`` is NaN until a nose and shoulders are seen."""

    slump: bool
    head_in_hands: bool
    motion: float
    head_drop: float
    calibrated: bool


def _ema(prev: float, value: float, alpha: float) -> float:
    """NaN-skipping exponential moving average."""
    if math.isnan(value):
        return prev
    if math.isnan(prev):
        return value
    return alpha * value + (1.0 - alpha) * prev


def _shoulder_line_y(xy: np.ndarray) -> float:
    """Mean y of the visible shoulders (one is enough for an above/below test)."""
    ys = xy[[L_SHOULDER, R_SHOULDER], 1]
    ys = ys[np.isfinite(ys)]
    return float(ys.mean()) if ys.size else math.nan


def _held(since: float | None, condition: bool, t: float, hold_s: float) -> tuple[float | None, bool]:
    """Track how long ``condition`` has been continuously true; True once held ``hold_s``."""
    if not condition:
        return None, False
    since = t if since is None else since
    return since, t - since >= hold_s - TIME_EPS


class PostureTracker:
    """Slump / head-in-hands / motion for one display id.

    ``head_drop`` is the nose height above the shoulder midpoint in shoulder widths
    (EMA). Its baseline is the median of the raw valid samples seen during the first
    ``baseline_s`` after the first update (at least 10 samples; calibration continues
    past the window until it has them, and restarts if the median is not positive).
    ``motion`` is the mean keypoint displacement in shoulder widths per second (EMA),
    so it does not depend on the sampling rate.
    """

    __slots__ = ("baseline_s", "slump_ratio", "slump_hold_s", "hih_hold_s", "hih_radius",
                 "ema_alpha", "min_conf", "_t0", "_samples", "_baseline", "_shoulder_w",
                 "_head_drop", "_motion", "_slump_since", "_hih_since", "_prev_xy", "_prev_t")

    def __init__(self, baseline_s: float = 20.0, slump_ratio: float = 0.7,
                 slump_hold_s: float = 3.0, hih_hold_s: float = 2.0, hih_radius: float = 0.5,
                 ema_alpha: float = 0.3, min_conf: float = 0.5) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.baseline_s, self.slump_ratio = baseline_s, slump_ratio
        self.slump_hold_s, self.hih_hold_s, self.hih_radius = slump_hold_s, hih_hold_s, hih_radius
        self.ema_alpha, self.min_conf = ema_alpha, min_conf
        self._t0: float | None = None
        self._samples: list[float] = []
        self._baseline: float | None = None
        self._shoulder_w = math.nan
        self._head_drop = math.nan
        self._motion = math.nan
        self._slump_since: float | None = None
        self._hih_since: float | None = None
        self._prev_xy: np.ndarray | None = None
        self._prev_t: float | None = None

    @property
    def baseline(self) -> float | None:
        return self._baseline

    @property
    def shoulder_width(self) -> float:
        return self._shoulder_w

    def update(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray, t: float) -> PostureState:
        """Feed one pose sample (pixel coordinates, per-keypoint confidence) at ``t``."""
        xy = self._mask(kpts_xy, kpts_conf)
        t0 = self._t0 = t if self._t0 is None else self._t0

        self._update_shoulder_width(xy)
        raw_drop = self._raw_head_drop(xy)
        self._head_drop = _ema(self._head_drop, raw_drop, self.ema_alpha)
        self._calibrate(raw_drop, t, t0)

        self._slump_since, slump = _held(self._slump_since, self._below_baseline(), t,
                                         self.slump_hold_s)
        self._hih_since, hih = _held(self._hih_since, self._hands_on_head(xy), t,
                                     self.hih_hold_s)
        self._motion = _ema(self._motion, self._raw_motion(xy, t), self.ema_alpha)
        self._prev_xy, self._prev_t = xy, t

        return PostureState(
            slump=slump,
            head_in_hands=hih,
            motion=0.0 if math.isnan(self._motion) else self._motion,
            head_drop=self._head_drop,
            calibrated=self._baseline is not None,
        )

    def _mask(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> np.ndarray:
        xy = np.array(kpts_xy, dtype=float)
        conf = np.asarray(kpts_conf, dtype=float)
        if xy.shape != (N_KPTS, 2) or conf.shape != (N_KPTS,):
            raise ValueError(f"expected (17, 2) keypoints and (17,) conf, got {xy.shape}, {conf.shape}")
        xy[~(conf >= self.min_conf)] = np.nan
        return xy

    def _update_shoulder_width(self, xy: np.ndarray) -> None:
        width = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
        if math.isfinite(width) and width > 0.0:
            self._shoulder_w = _ema(self._shoulder_w, width, self.ema_alpha)

    def _raw_head_drop(self, xy: np.ndarray) -> float:
        mid_y = (xy[L_SHOULDER, 1] + xy[R_SHOULDER, 1]) / 2.0
        if math.isnan(self._shoulder_w):
            return math.nan
        return float((mid_y - xy[NOSE, 1]) / self._shoulder_w)

    def _calibrate(self, raw_drop: float, t: float, t0: float) -> None:
        if self._baseline is not None:
            return
        in_window = t - t0 < self.baseline_s - TIME_EPS
        if math.isfinite(raw_drop) and (in_window or len(self._samples) < MIN_BASELINE_SAMPLES):
            self._samples.append(raw_drop)
        if in_window or len(self._samples) < MIN_BASELINE_SAMPLES:
            return
        baseline = float(np.median(self._samples))
        self._samples = []
        if baseline > 0.0:
            self._baseline = baseline
        else:
            self._t0 = t

    def _below_baseline(self) -> bool:
        if self._baseline is None or math.isnan(self._head_drop):
            return False
        return self._head_drop < self.slump_ratio * self._baseline

    def _hands_on_head(self, xy: np.ndarray) -> bool:
        if math.isnan(self._shoulder_w):
            return False
        wrists = xy[[L_WRIST, R_WRIST]]
        head = xy[list(HEAD_POINTS)]
        head = head[np.isfinite(head).all(axis=1)]
        shoulder_y = _shoulder_line_y(xy)
        if head.size == 0 or not np.isfinite(wrists).all() or math.isnan(shoulder_y):
            return False
        dists = np.linalg.norm(wrists[:, None, :] - head[None, :, :], axis=2).min(axis=1)
        near = bool((dists <= self.hih_radius * self._shoulder_w).all())
        raised = bool((wrists[:, 1] < shoulder_y).all())
        return near and raised

    def _raw_motion(self, xy: np.ndarray, t: float) -> float:
        if self._prev_xy is None or self._prev_t is None or math.isnan(self._shoulder_w):
            return math.nan
        dt = t - self._prev_t
        both = np.isfinite(xy).all(axis=1) & np.isfinite(self._prev_xy).all(axis=1)
        if dt <= 0.0 or not both.any():
            return math.nan
        disp = np.linalg.norm(xy[both] - self._prev_xy[both], axis=1).mean()
        return float(disp / self._shoulder_w / dt)
