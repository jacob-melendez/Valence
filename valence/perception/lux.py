"""Ambient Physics layer: room light (lux) from the webcam, no light sensor needed.

Auto-exposure normalises brightness, so background luminance alone is a weak lux proxy;
in a dim room the camera raises gain and sensor noise rises with it. We measure both on
the background only (person boxes masked, so clothing and skin do not move the reading)
from a single frame, and map them to lux with a log-linear model fitted against a phone
lux meter (``scripts/calibrate_lux.py``). Only scalars leave this module; no frame,
thumbnail or frame difference is ever kept.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

CLIP_LO, CLIP_HI = 5, 250
MAX_WIDTH = 320
MAX_LUX = 200_000.0
DEFAULT_A, DEFAULT_B = 3.59, 0.01463
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_SIGMA_SCALE = math.sqrt(math.pi / 2.0) / 6.0
_DILATE = np.ones((3, 3), dtype=np.uint8)

Box = Sequence[float]


@dataclass(slots=True, frozen=True)
class BgStats:
    """Single-frame background statistics: mean luminance, noise sigma, coverage, clipping."""

    y_bg: float
    sigma: float
    bg_frac: float
    clipped_frac: float

    @property
    def valid(self) -> bool:
        return self.bg_frac > 0.0 and math.isfinite(self.y_bg)


def _person_mask(shape: tuple[int, int], boxes: Iterable[Box]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        c0, c1 = int(np.clip(math.floor(min(x1, x2) * w), 0, w)), int(np.clip(math.ceil(max(x1, x2) * w), 0, w))
        r0, r1 = int(np.clip(math.floor(min(y1, y2) * h), 0, h)), int(np.clip(math.ceil(max(y1, y2) * h), 0, h))
        mask[r0:r1, c0:c1] = 1
    return mask


def _interior_keep(excluded: np.ndarray) -> np.ndarray:
    keep = cv2.dilate(excluded, _DILATE) == 0
    keep[0, :] = keep[-1, :] = False
    keep[:, 0] = keep[:, -1] = False
    return keep


def _immerkaer_sigma(img: np.ndarray, person: np.ndarray) -> float:
    """sigma = sqrt(pi/2) / 6 * mean|I * N| over background pixels whose 3x3 support is
    entirely background (Immerkaer 1996), so person/background edges do not count as noise.

    Hard-clipped pixels (0 or 255) carry no noise and would dilute the estimate, so their
    3x3 neighbourhoods are skipped too, unless that leaves nothing to measure.
    """
    if min(img.shape) < 3:
        return math.nan
    response = cv2.filter2D(img, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REPLICATE)
    saturated = ((img == 0) | (img == 255)).astype(np.uint8)
    keep = _interior_keep(person | saturated)
    if not keep.any():
        keep = _interior_keep(person)
    n = int(np.count_nonzero(keep))
    if n == 0:
        return math.nan
    return float(_SIGMA_SCALE * np.abs(response[keep]).sum(dtype=np.float64) / n)


def background_stats(gray: np.ndarray, person_boxes_norm: Iterable[Box] = (),
                     max_width: int = MAX_WIDTH) -> BgStats:
    """Background luminance and sensor-noise sigma from ONE uint8 HxW luminance frame.

    The frame is decimated (strided, not averaged) to <= ``max_width`` columns for speed:
    decimation keeps per-pixel noise statistics intact, whereas area averaging would
    shrink sigma by the stride. ``person_boxes_norm`` are (x1, y1, x2, y2) in [0, 1].
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("background_stats expects a uint8 HxW luminance image")
    stride = max(1, math.ceil(gray.shape[1] / max_width))
    img = gray[::stride, ::stride]
    person = _person_mask(img.shape, person_boxes_norm)
    background = person == 0
    n_bg = int(np.count_nonzero(background))
    if n_bg == 0:
        return BgStats(math.nan, math.nan, 0.0, 0.0)
    px = img[background]
    clipped = int(np.count_nonzero((px <= CLIP_LO) | (px >= CLIP_HI)))
    return BgStats(
        y_bg=float(px.mean(dtype=np.float64)),
        sigma=_immerkaer_sigma(img, person),
        bg_frac=n_bg / background.size,
        clipped_frac=clipped / n_bg,
    )


@dataclass(slots=True, frozen=True)
class LuxModel:
    """lux = exp(a + b * y_bg + c * sigma). The default (uncalibrated) model maps
    y_bg 30 -> ~55, 128 -> ~240, 220 -> ~900 lux, ignoring noise."""

    a: float = DEFAULT_A
    b: float = DEFAULT_B
    c: float = 0.0
    calibrated: bool = False

    def predict(self, y_bg: float, sigma: float) -> float:
        z = self.a + self.b * y_bg + (self.c * sigma if self.c else 0.0)
        if not math.isfinite(z):
            return math.nan
        return min(math.exp(min(z, 50.0)), MAX_LUX)

    @classmethod
    def fit(cls, readings: Sequence[tuple[float, float, float]], ridge: float = 1e-2,
            use_sigma: bool = True) -> LuxModel:
        """Fit from (y_bg, sigma, lux) readings.

        >= 3 readings: least squares on log(lux) = a + b*y + c*sigma, with a light ridge
        on c (``ridge`` x the centred sigma energy) to tame y/sigma collinearity.
        Exactly 2 readings, or ``use_sigma=False``: linear fit on y only, c = 0.
        """
        rows = np.array(
            [(y, s, lux) for y, s, lux in readings
             if math.isfinite(y) and math.isfinite(lux) and lux > 0 and (math.isfinite(s) or not use_sigma)],
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(rows) < 2:
            raise ValueError("need at least 2 valid (y_bg, sigma, lux) readings")
        y, s, log_lux = rows[:, 0], rows[:, 1], np.log(rows[:, 2])
        if np.ptp(y) == 0:
            raise ValueError("readings must differ in background luminance")
        if len(rows) == 2 or not use_sigma:
            b, a = np.polyfit(y, log_lux, 1)
            return cls(float(a), float(b), 0.0, True)
        yc, sc, lc = y - y.mean(), s - s.mean(), log_lux - log_lux.mean()
        lam = ridge * float(sc @ sc)
        design = np.vstack([np.column_stack([yc, sc]), [0.0, math.sqrt(lam)]])
        (b, c), *_ = np.linalg.lstsq(design, np.append(lc, 0.0), rcond=None)
        a = log_lux.mean() - b * y.mean() - c * s.mean()
        return cls(float(a), float(b), float(c), True)

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "c": self.c, "calibrated": self.calibrated}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LuxModel:
        return cls(
            a=float(d.get("a", DEFAULT_A)),
            b=float(d.get("b", DEFAULT_B)),
            c=float(d.get("c", 0.0)),
            calibrated=bool(d.get("calibrated", False)),
        )


class LuxEstimator:
    """Rolling ``window_s`` median of per-frame lux, plus a [0, 1] quality score:
    bg_frac * (1 - clipped_frac) * stability, stability = 1 / (1 + IQR / median),
    halved while the model is uncalibrated."""

    __slots__ = ("model", "window_s", "_history")

    def __init__(self, model: LuxModel | None = None, window_s: float = 10.0) -> None:
        self.model = model if model is not None else LuxModel()
        self.window_s = window_s
        self._history: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._history.clear()

    def update(self, stats: BgStats, t: float) -> tuple[float, float]:
        """Returns (lux_median_over_window, lux_quality); (NaN, 0.0) before any valid frame.

        A non-finite ``t`` is ignored (the current window is reported unchanged), so it can
        never wedge the eviction of old readings.
        """
        if not math.isfinite(t):
            return self._summary(stats)
        if stats.valid:
            lux = self.model.predict(stats.y_bg, stats.sigma)
            if math.isfinite(lux):
                self._history.append((t, lux))
        while self._history and t - self._history[0][0] > self.window_s:
            self._history.popleft()
        return self._summary(stats)

    def _summary(self, stats: BgStats) -> tuple[float, float]:
        if not self._history:
            return math.nan, 0.0
        values = np.fromiter((v for _, v in self._history), dtype=np.float64, count=len(self._history))
        q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
        stability = 1.0 / (1.0 + (q75 - q25) / median) if median > 0 else 0.0
        quality = stats.bg_frac * (1.0 - stats.clipped_frac) * stability
        if not self.model.calibrated:
            quality *= 0.5
        return float(median), float(np.clip(quality, 0.0, 1.0))
