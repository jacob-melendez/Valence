"""Image-region helpers: head boxes, in-place blackout and box normalization.

``black_out`` backs the ``--obscure-face`` switch: head regions are zeroed in the
live frame before MediaPipe runs, proving the classifier still works when the face
carries no signal at all. Nothing here copies, keeps or returns pixel data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

HEAD_POINTS = (0, 1, 2, 3, 4)
L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
SHOULDER_TO_HEAD = 0.6
EYES_TO_EARS = 2.0  # ear-to-ear span ~ 2x eye distance; keeps the box face-sized when ears are hidden
FALLBACK_HEAD_FRAC = 0.15

BoxPx = tuple[int, int, int, int]


def head_box_px(kpts_xy: np.ndarray, kpts_conf: np.ndarray, frame_w: int, frame_h: int,
                scale: float = 1.8, min_conf: float = 0.3) -> BoxPx | None:
    """Square box ``(x1, y1, x2, y2)`` (x2/y2 exclusive) around a person's head.

    Centred on the mean of the confident nose/eye/ear keypoints. Its side is ``scale``
    times the largest of the head points' spread, ``2 * eye distance`` (both eyes
    confident; covers hidden ears) and ``0.6 * shoulder width`` (both shoulders
    confident); with a single head point and no shoulders it falls back to
    ``0.15 * frame_w``. Clipped to the frame; None if nothing is visible.
    """
    xy = np.asarray(kpts_xy, dtype=float)[:, :2]
    conf = np.asarray(kpts_conf, dtype=float)
    ok = (conf >= min_conf) & np.isfinite(xy).all(axis=1)

    head = xy[list(HEAD_POINTS)][ok[list(HEAD_POINTS)]]
    if head.size == 0:
        return None
    spread = float((head.max(axis=0) - head.min(axis=0)).max())
    if ok[L_EYE] and ok[R_EYE]:
        spread = max(spread, EYES_TO_EARS * float(np.linalg.norm(xy[L_EYE] - xy[R_EYE])))
    if ok[L_SHOULDER] and ok[R_SHOULDER]:
        spread = max(spread, SHOULDER_TO_HEAD * float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER])))
    if spread <= 0.0:
        spread = FALLBACK_HEAD_FRAC * frame_w

    cx, cy = head.mean(axis=0)
    half = scale * spread / 2.0
    box = (max(0, math.floor(cx - half)), max(0, math.floor(cy - half)),
           min(frame_w, math.ceil(cx + half)), min(frame_h, math.ceil(cy + half)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def black_out(frame: np.ndarray, boxes_px: Iterable[BoxPx | None]) -> None:
    """Zero the given pixel boxes of ``frame`` in place (None entries are skipped)."""
    h, w = frame.shape[:2]
    for box in boxes_px:
        if box is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = 0


def norm_bbox(xyxy_px: Iterable[float], frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """Pixel ``(x1, y1, x2, y2)`` -> fractions of the frame, each clipped to [0, 1]."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame size must be positive")
    x1, y1, x2, y2 = (float(v) for v in xyxy_px)
    return _clip01(x1 / frame_w), _clip01(y1 / frame_h), _clip01(x2 / frame_w), _clip01(y2 / frame_h)


def _clip01(v: float) -> float:
    return min(1.0, max(0.0, v))
