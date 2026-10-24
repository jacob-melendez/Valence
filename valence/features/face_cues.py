"""Per-frame emotion-map face cues from MediaPipe FaceLandmarker outputs (docs/EMOTION_FEATURES.md section 1).

Reference implementation for the browser perception (TypeScript) and its parity tests. Functions take only the
blendshape scores and the 16 numbers of the facial transformation matrix; landmarks are never touched. One
``FaceCueTracker`` per ephemeral display id, in memory only.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

CALIBRATION_S = 20.0
MAX_GAP_S = 1.0
BLINK_CLOSE = 0.30  # closed when eye closure >= neutral + 0.30
BLINK_OPEN = 0.15  # open again when closure <= neutral + 0.15
BLINK_MAX_MS = 600.0
RAW_CHANNELS = ("brow", "closure", "frown", "smile", "yaw", "pitch")


def _avg(b: Mapping[str, float], stem: str) -> float:
    vals = [float(b[k]) for k in (f"{stem}Left", f"{stem}Right") if k in b and b[k] is not None]
    return sum(vals) / len(vals) if vals else math.nan


def head_pose_deg(data: Sequence[float]) -> tuple[float, float]:
    """(yaw, pitch) in degrees from the 4x4 facial transformation matrix (16 numbers, column-major as packed by
    MediaPipe; a row-major matrix is detected by where the translation sits). yaw > 0: nose toward the image's
    right; pitch > 0: face tilted up."""
    if len(data) != 16:
        return math.nan, math.nan
    if abs(data[14]) >= abs(data[11]):  # column-major: translation in data[12..14]
        fx, fy, fz = data[8], data[9], data[10]
    else:  # row-major: translation in data[3], data[7], data[11]
        fx, fy, fz = data[2], data[6], data[10]
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
    return yaw, pitch


def raw_cues(blendshapes: Mapping[str, float], matrix: Sequence[float] | None) -> dict[str, float]:
    """Raw (not yet neutral-relative) channels for one face."""
    yaw, pitch = head_pose_deg(matrix) if matrix is not None else (math.nan, math.nan)
    return {"brow": _avg(blendshapes, "browDown"), "closure": _avg(blendshapes, "eyeBlink"),
            "frown": _avg(blendshapes, "mouthFrown"), "smile": _avg(blendshapes, "mouthSmile"),
            "yaw": yaw, "pitch": pitch}


def _clip(v: float, lo: float, hi: float) -> float:
    return v if math.isnan(v) else max(lo, min(hi, v))


def _wrap_deg(v: float) -> float:
    return v if math.isnan(v) else (v + 180.0) % 360.0 - 180.0


@dataclass
class FaceCueTracker:
    """Neutral baseline (first 20 s of visible face, running median, then frozen) and the blink detector."""

    calibration_s: float = CALIBRATION_S
    _samples: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in RAW_CHANNELS})
    _frozen: dict[str, float] | None = None
    _visible_s: float = 0.0
    _prev_t: float | None = None
    _prev_visible: bool = False
    _closed_at: float | None = None  # ms

    def baseline(self) -> dict[str, float]:
        if self._frozen is not None:
            return self._frozen
        return {k: float(statistics.median(v)) if v else 0.0 for k, v in self._samples.items()}

    def update(self, raw: dict[str, float] | None, t_ms: float) -> dict[str, float]:
        """One frame. ``raw`` is None (or has NaN closure) when no face is visible. Returns the PersonObs cues."""
        visible = raw is not None and not all(math.isnan(v) for v in raw.values())
        t = t_ms / 1000.0
        if visible and self._prev_visible and self._prev_t is not None:
            self._visible_s += min(max(t - self._prev_t, 0.0), MAX_GAP_S)
        self._prev_t, self._prev_visible = t, visible
        if not visible:
            self._closed_at = None
            return {**{k: math.nan for k in ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw",
                                             "head_pitch")}, "blink_onsets": 0}
        if self._frozen is None:
            if self._visible_s < self.calibration_s:
                for k, v in raw.items():
                    if not math.isnan(v):
                        self._samples[k].append(v)
            else:
                self._frozen = self.baseline()
                self._samples = {k: [] for k in RAW_CHANNELS}
        b = self.baseline()
        blink = 0
        c = raw["closure"]
        if not math.isnan(c):
            if self._closed_at is None and c >= b["closure"] + BLINK_CLOSE:
                self._closed_at = t_ms
            elif self._closed_at is not None and c <= b["closure"] + BLINK_OPEN:
                blink = int(t_ms - self._closed_at <= BLINK_MAX_MS)
                self._closed_at = None
        return {
            "brow_down": _clip(raw["brow"] - b["brow"], -1.0, 1.0),
            "eye_open": _clip(b["closure"] - c, -1.0, 1.0),
            "mouth_frown": _clip(raw["frown"] - b["frown"], -1.0, 1.0),
            "smile": _clip(raw["smile"] - b["smile"], -1.0, 1.0),
            "head_yaw": _clip(_wrap_deg(raw["yaw"] - b["yaw"]), -90.0, 90.0),
            "head_pitch": _clip(raw["pitch"] - b["pitch"], -90.0, 90.0),
            "blink_onsets": blink,
        }
