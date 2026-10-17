"""Facial Surface layer: valence from MediaPipe FaceLandmarker blendshapes.

This feeds both the full model and the face-only baseline, and it is deliberately naive:
a smile reads as positive valence whether or not it is felt. ``cheekSquint`` (the
Duchenne cue that separates felt from posed smiles) is intentionally NOT used, so the
face-only baseline cannot see through a Frustrated_Masked forced smile; catching that is
the job of the Ambient Physics, Time Dynamics and Somatic & Social layers.

Privacy: functions take only the blendshape scores or one nose-tip point per face. The
478 landmarks are never stored, nothing here identifies anyone, and calibrators are
keyed by ephemeral display ids and live in memory only.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

VALENCE_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "mouthSmile": 1.0,
    "mouthFrown": -0.7,
    "browDown": -0.5,
    "noseSneer": -0.3,
    "mouthPress": -0.2,
})
VALENCE_BLENDSHAPES: tuple[str, ...] = tuple(
    f"{stem}{side}" for stem in VALENCE_WEIGHTS for side in ("Left", "Right")
)
NOSE_TIP_INDEX = 1
CALIBRATION_S = 20.0
GAIN = 2.5
MAX_CALIBRATION_SAMPLES = 4096


def blendshape_scores(categories: Iterable[Any]) -> dict[str, float]:
    """Keep only the scores ``valence_raw`` needs from MediaPipe ``Category`` objects."""
    return {
        c.category_name: float(c.score)
        for c in categories
        if c.category_name in VALENCE_BLENDSHAPES
    }


def nose_tip_px(face_landmarks: Sequence[Any], frame_w: int, frame_h: int) -> tuple[float, float]:
    """Pixel position of the nose-tip landmark; the rest of the mesh is not touched."""
    tip = face_landmarks[NOSE_TIP_INDEX]
    return float(tip.x) * frame_w, float(tip.y) * frame_h


def _bilateral(blendshapes: Mapping[str, float], stem: str) -> float | None:
    values = [float(blendshapes[k]) for k in (f"{stem}Left", f"{stem}Right") if k in blendshapes]
    return sum(values) / len(values) if values else None


def valence_raw(blendshapes: Mapping[str, float]) -> float:
    """smile - 0.7 frown - 0.5 browDown - 0.3 noseSneer - 0.2 mouthPress (each L/R-averaged).

    A missing side falls back to the other side, a missing category counts as 0, and a
    mapping with none of the categories (no face) yields NaN.
    """
    total, seen = 0.0, False
    for stem, weight in VALENCE_WEIGHTS.items():
        value = _bilateral(blendshapes, stem)
        if value is not None:
            total += weight * value
            seen = True
    return total if seen else math.nan


@dataclass(slots=True)
class FaceValenceCalibrator:
    """Per-person neutral baseline: resting faces differ, so valence is measured relative
    to the median raw valence over that person's first ``calibration_s`` seconds of
    visible face (running median while calibrating, frozen afterwards).

    Visible time accrues only between consecutive visible samples, with each gap capped
    at ``max_gap_s``; a non-finite ``t`` accrues nothing. Non-finite input means "face not
    visible": it returns NaN and never enters the baseline. One instance per ephemeral display id, in memory only.
    """

    calibration_s: float = CALIBRATION_S
    gain: float = GAIN
    max_gap_s: float = 1.0
    _samples: list[float] = field(default_factory=list, init=False, repr=False)
    _visible_s: float = field(default=0.0, init=False)
    _prev_t: float | None = field(default=None, init=False, repr=False)
    _prev_visible: bool = field(default=False, init=False, repr=False)
    _frozen: float | None = field(default=None, init=False)

    @property
    def calibrated(self) -> bool:
        return self._frozen is not None

    @property
    def baseline(self) -> float:
        if self._frozen is not None:
            return self._frozen
        return float(statistics.median(self._samples)) if self._samples else 0.0

    @property
    def visible_s(self) -> float:
        return self._visible_s

    def update(self, v_raw: float, t: float) -> float:
        """Feed one raw valence sample; returns clip(tanh(gain * (v_raw - baseline)), -1, 1)."""
        visible = math.isfinite(v_raw)
        if not math.isfinite(t):
            self._prev_t = None
        else:
            if visible and self._prev_visible and self._prev_t is not None:
                self._visible_s += min(max(t - self._prev_t, 0.0), self.max_gap_s)
            self._prev_t = t
        self._prev_visible = visible
        if not visible:
            return math.nan
        if self._frozen is None:
            if self._visible_s < self.calibration_s and len(self._samples) < MAX_CALIBRATION_SAMPLES:
                self._samples.append(float(v_raw))
            else:
                self._frozen = self.baseline
                self._samples.clear()
        return max(-1.0, min(1.0, math.tanh(self.gain * (v_raw - self.baseline))))


def assign_faces(
    face_points_px: Sequence[tuple[float, float]],
    person_noses_px: Mapping[int, tuple[float, float, float]],
    frame_w: int,
    max_dist_frac: float = 0.15,
    min_conf: float = 0.3,
) -> list[int | None]:
    """Match each face (its nose-tip landmark, px) to a tracked person (pose nose, px).

    Optimal one-to-one assignment (Hungarian) on Euclidean distance, gated at
    ``max_dist_frac * frame_w``. Gated pairs cost more than any full set of valid pairs,
    so the solver first maximises the number of valid matches, then minimises distance.
    Person noses below ``min_conf`` are ignored. Returns a person id or None per face.
    """
    out: list[int | None] = [None] * len(face_points_px)
    pids = [
        pid for pid, (x, y, conf) in person_noses_px.items()
        if conf >= min_conf and math.isfinite(x) and math.isfinite(y)
    ]
    if not out or not pids:
        return out
    faces = np.asarray(face_points_px, dtype=np.float64).reshape(-1, 2)
    noses = np.asarray([person_noses_px[pid][:2] for pid in pids], dtype=np.float64)
    dist = np.linalg.norm(faces[:, None, :] - noses[None, :, :], axis=2)
    gate = max_dist_frac * frame_w
    valid = np.isfinite(dist) & (dist <= gate)
    if not valid.any():
        return out
    blocked = gate * (min(dist.shape) + 1) + 1.0
    rows, cols = linear_sum_assignment(np.where(valid, dist, blocked))
    for r, c in zip(rows, cols):
        if valid[r, c]:
            out[r] = pids[c]
    return out
