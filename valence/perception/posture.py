"""Per-person posture signals from COCO-17 pose keypoints.

Turns one person's keypoint stream into three somatic signals: a slump (head sinking
toward the shoulders relative to that person's own calibrated baseline), head in
hands, and torso/arm motion. Everything is normalized by shoulder width so it is
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
BODY_POINTS = tuple(range(L_SHOULDER, 13))  # shoulders, elbows, wrists, hips: motion ignores the head
BODY_PAIRS = ((5, 6), (7, 8), (9, 10), (11, 12))  # left/right twins, which pose models swap
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
    (EMA); ``ear_drop`` is the same for the mean of the visible ears. Baselines are the
    medians of the raw valid samples seen during the first ``baseline_s`` after the first
    update (at least 10 samples; calibration continues past the window until the nose has
    them, and restarts if its median is not positive).

    A slump needs the whole head to sink, not just tip forward: looking down at a keyboard
    pitches the head, which swings the nose down but leaves the ears (near the pitch axis)
    in place. Measured live (docs/BENCHMARKS.md): keyboard gaze nose 0.84x / ears 0.96x of
    baseline, slump nose 0.42x / ears 0.86x; the nose-only rule flagged 93% of a 3-minute
    typing-while-looking-down capture as slumped. So slump = nose < ``slump_ratio`` x
    baseline AND ears < ``ear_ratio`` x baseline, held ``slump_hold_s``; when the ears were
    never calibrated or aren't visible, the nose alone must fall below
    ``nose_only_ratio`` x baseline.
    ``motion`` is the mean displacement of the torso and arm keypoints (COCO 5-12) in
    shoulder widths per second (EMA), so it does not depend on the sampling rate. The head
    is left out: when it drops, the nose/eyes/ears lose confidence and jump, and a held
    slump read 0.101 sw/s live against 0.033 upright. A point counts only if it has
    conf >= ``motion_min_conf`` in both samples and moved at most ``motion_jump`` shoulder
    widths since the last one (a bigger per-sample jump is a keypoint switching identity,
    not a limb: at 5 Hz that is a 5 sw/s, roughly 2 m/s, limb). A left/right twin pair seen in
    both samples is matched whichever way is shorter, so a one-sample L/R label swap (the most
    common switch, and under 1 sw for wrists near each other) is not motion. The survivors are
    averaged (a mean, not a median, so one arm moving still counts). With fewer than
    ``motion_min_points`` survivors, unless both shoulders survive (a measurable torso, e.g. the
    arms dropped out of a laptop frame), the last value is held; motion is NaN only until the
    first measurable pair.
    """

    __slots__ = ("baseline_s", "slump_ratio", "ear_ratio", "nose_only_ratio", "slump_hold_s", "hih_hold_s",
                 "hih_radius", "ema_alpha", "min_conf", "motion_min_conf", "motion_jump", "motion_min_points",
                 "_t0", "_samples", "_ear_samples", "_baseline",
                 "_ear_baseline", "_shoulder_w", "_head_drop", "_ear_drop", "_motion", "_slump_since",
                 "_hih_since", "_prev_xy", "_prev_t")

    def __init__(self, baseline_s: float = 20.0, slump_ratio: float = 0.7,
                 slump_hold_s: float = 3.0, hih_hold_s: float = 2.0, hih_radius: float = 0.5,
                 ema_alpha: float = 0.3, min_conf: float = 0.5, ear_ratio: float = 0.93,
                 nose_only_ratio: float = 0.6, motion_min_conf: float = 0.6, motion_jump: float = 1.0,
                 motion_min_points: int = 3) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.baseline_s, self.slump_ratio = baseline_s, slump_ratio
        self.ear_ratio, self.nose_only_ratio = ear_ratio, nose_only_ratio
        self.slump_hold_s, self.hih_hold_s, self.hih_radius = slump_hold_s, hih_hold_s, hih_radius
        self.ema_alpha, self.min_conf = ema_alpha, min_conf
        self.motion_min_conf = max(min_conf, motion_min_conf)
        self.motion_jump, self.motion_min_points = motion_jump, motion_min_points
        self._t0: float | None = None
        self._samples: list[float] = []
        self._ear_samples: list[float] = []
        self._baseline: float | None = None
        self._ear_baseline: float | None = None
        self._shoulder_w = math.nan
        self._head_drop = math.nan
        self._ear_drop = math.nan
        self._motion = math.nan
        self._slump_since: float | None = None
        self._hih_since: float | None = None
        self._prev_xy: np.ndarray | None = None
        self._prev_t: float | None = None

    @property
    def baseline(self) -> float | None:
        return self._baseline

    @property
    def ear_baseline(self) -> float | None:
        return self._ear_baseline

    @property
    def shoulder_width(self) -> float:
        return self._shoulder_w

    def update(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray, t: float) -> PostureState:
        """Feed one pose sample (pixel coordinates, per-keypoint confidence) at ``t``."""
        xy, body = self._mask(kpts_xy, kpts_conf)
        t0 = self._t0 = t if self._t0 is None else self._t0

        self._update_shoulder_width(xy)
        raw_drop = self._raw_head_drop(xy)
        raw_ear = self._raw_ear_drop(xy)
        self._head_drop = _ema(self._head_drop, raw_drop, self.ema_alpha)
        self._ear_drop = raw_ear if math.isnan(raw_ear) else _ema(self._ear_drop, raw_ear, self.ema_alpha)
        self._calibrate(raw_drop, raw_ear, t, t0)

        self._slump_since, slump = _held(self._slump_since, self._below_baseline(), t,
                                         self.slump_hold_s)
        self._hih_since, hih = _held(self._hih_since, self._hands_on_head(xy), t,
                                     self.hih_hold_s)
        self._motion = _ema(self._motion, self._raw_motion(body, t), self.ema_alpha)
        self._prev_xy, self._prev_t = body, t

        return PostureState(
            slump=slump,
            head_in_hands=hih,
            motion=self._motion,
            head_drop=self._head_drop,
            calibrated=self._baseline is not None,
        )

    def _mask(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(all keypoints at ``min_conf``, torso/arm keypoints at ``motion_min_conf``); NaN = not valid."""
        xy = np.array(kpts_xy, dtype=float)
        conf = np.asarray(kpts_conf, dtype=float)
        if xy.shape != (N_KPTS, 2) or conf.shape != (N_KPTS,):
            raise ValueError(f"expected (17, 2) keypoints and (17,) conf, got {xy.shape}, {conf.shape}")
        xy[~(conf >= self.min_conf)] = np.nan
        body = np.full_like(xy, np.nan)
        keep = np.zeros(N_KPTS, dtype=bool)
        keep[list(BODY_POINTS)] = True
        keep &= conf >= self.motion_min_conf
        body[keep] = xy[keep]
        return xy, body

    def _update_shoulder_width(self, xy: np.ndarray) -> None:
        width = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
        if math.isfinite(width) and width > 0.0:
            self._shoulder_w = _ema(self._shoulder_w, width, self.ema_alpha)

    def _raw_head_drop(self, xy: np.ndarray) -> float:
        mid_y = (xy[L_SHOULDER, 1] + xy[R_SHOULDER, 1]) / 2.0
        if math.isnan(self._shoulder_w):
            return math.nan
        return float((mid_y - xy[NOSE, 1]) / self._shoulder_w)

    def _raw_ear_drop(self, xy: np.ndarray) -> float:
        ears = xy[[L_EAR, R_EAR]]
        ears = ears[np.isfinite(ears).all(axis=1)]
        mid_y = (xy[L_SHOULDER, 1] + xy[R_SHOULDER, 1]) / 2.0
        if ears.size == 0 or math.isnan(self._shoulder_w) or math.isnan(mid_y):
            return math.nan
        return float((mid_y - ears[:, 1].mean()) / self._shoulder_w)

    def _calibrate(self, raw_drop: float, raw_ear: float, t: float, t0: float) -> None:
        if self._baseline is not None:
            return
        in_window = t - t0 < self.baseline_s - TIME_EPS
        if in_window or len(self._samples) < MIN_BASELINE_SAMPLES:
            if math.isfinite(raw_drop):
                self._samples.append(raw_drop)
            if math.isfinite(raw_ear):
                self._ear_samples.append(raw_ear)
        if in_window or len(self._samples) < MIN_BASELINE_SAMPLES:
            return
        baseline = float(np.median(self._samples))
        ear = float(np.median(self._ear_samples)) if len(self._ear_samples) >= MIN_BASELINE_SAMPLES else None
        self._samples, self._ear_samples = [], []
        if baseline > 0.0:
            self._baseline = baseline
            self._ear_baseline = ear if ear is not None and ear > 0.0 else None
        else:
            self._t0 = t

    def _below_baseline(self) -> bool:
        if self._baseline is None or math.isnan(self._head_drop):
            return False
        if self._ear_baseline is not None and not math.isnan(self._ear_drop):
            return (self._head_drop < self.slump_ratio * self._baseline
                    and self._ear_drop < self.ear_ratio * self._ear_baseline)
        return self._head_drop < self.nose_only_ratio * self._baseline

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

    def _raw_motion(self, body: np.ndarray, t: float) -> float:
        """Torso/arm speed in shoulder widths/s for this pair; NaN (so the EMA holds) when unmeasurable."""
        if self._prev_xy is None or self._prev_t is None or math.isnan(self._shoulder_w):
            return math.nan
        dt = t - self._prev_t
        if dt <= 0.0:
            return math.nan
        prev = self._prev_xy
        disp = np.linalg.norm(body - prev, axis=1) / self._shoulder_w  # NaN unless valid in both
        for left, right in BODY_PAIRS:
            swapped = np.linalg.norm(body[[left, right]] - prev[[right, left]], axis=1) / self._shoulder_w
            if swapped.sum() < disp[left] + disp[right]:  # False when any is NaN: keep the labels
                disp[[left, right]] = swapped
        ok = np.isfinite(disp) & (disp <= self.motion_jump)
        torso = bool(ok[L_SHOULDER] and ok[R_SHOULDER])
        if ok.sum() < self.motion_min_points and not torso:
            return math.nan
        return float(disp[ok].mean() / dt)
