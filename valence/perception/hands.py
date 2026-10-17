"""Hands -> people, fist heuristics and per-person fist onsets.

MediaPipe's GestureRecognizer sees hands but not people; YOLO-pose sees people and
their wrists. ``assign_hands`` joins the two by wrist position. ``is_curl_fist`` is a
geometric fallback for when the gesture head is unsure. ``FistTracker`` turns noisy
per-hand ``Closed_Fist`` scores into debounced onsets per display id. Everything here
is coordinates and scalars; no image data is touched.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from valence.perception.debounce import HysteresisDebouncer, fist_debouncer

SIDES = ("L", "R")
WRIST = 0
FINGER_TIP_PIP = ((8, 6), (12, 10), (16, 14), (20, 18))

Wrist = tuple[float, float, float]
HandKey = tuple[int, str]


def assign_hands(hand_wrists_px: Sequence[tuple[float, float]],
                 person_wrists: Mapping[int, Mapping[str, Wrist]],
                 frame_w: int, max_dist_frac: float = 0.12,
                 min_conf: float = 0.5) -> list[HandKey | None]:
    """Match each MediaPipe hand (wrist landmark, in pixels) to a person's wrist.

    ``person_wrists`` maps display id -> ``{"L": (x, y, conf), "R": (x, y, conf)}``
    from YOLO keypoints 9/10. Matching is globally greedy: every hand/wrist pair
    within ``max_dist_frac * frame_w`` is sorted by distance and accepted nearest
    first, so each hand and each YOLO wrist is used at most once. Wrists below
    ``min_conf`` are ignored. Returns ``(display_id, side)`` or None per hand.
    """
    max_dist = max_dist_frac * frame_w
    candidates = [
        (pid, side, wx, wy)
        for pid, sides in person_wrists.items()
        for side, (wx, wy, conf) in sides.items()
        if conf >= min_conf and math.isfinite(wx) and math.isfinite(wy)
    ]
    pairs = sorted(
        (d, h, pid, side)
        for h, (hx, hy) in enumerate(hand_wrists_px)
        for pid, side, wx, wy in candidates
        if (d := math.hypot(hx - wx, hy - wy)) <= max_dist
    )
    result: list[HandKey | None] = [None] * len(hand_wrists_px)
    used: set[HandKey] = set()
    for _, h, pid, side in pairs:
        if result[h] is None and (pid, side) not in used:
            result[h] = (pid, side)
            used.add((pid, side))
    return result


def curled_fingers(landmarks: np.ndarray, aspect: float = 9 / 16) -> int:
    """Count curled fingers (index..pinky) from 21 normalized hand landmarks.

    A finger is curled when its tip is closer to the wrist than its PIP joint.
    Distances use x as-is and y multiplied by ``aspect`` (frame h / w), i.e. frame
    widths, so they are isotropic; z is ignored. Non-finite landmarks never count.
    """
    pts = np.asarray(landmarks, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != 21 or pts.shape[1] < 2:
        raise ValueError(f"expected (21, 2|3) landmarks, got {pts.shape}")
    xy = pts[:, :2] * np.array([1.0, aspect])
    dist = np.linalg.norm(xy - xy[WRIST], axis=1)
    return sum(bool(dist[tip] < dist[pip]) for tip, pip in FINGER_TIP_PIP)


def is_curl_fist(landmarks: np.ndarray, aspect: float = 9 / 16, min_curled: int = 3) -> bool:
    """Fallback fist test: at least ``min_curled`` of the four fingers curled."""
    return curled_fingers(landmarks, aspect) >= min_curled


class FistTracker:
    """One hysteresis debouncer per (display id, side), reported per person.

    ``person_state`` returns the onsets counted since the previous call for that
    person (a delta, so callers can sum it into a rolling window) and whether any of
    their hands is currently clenched.
    """

    __slots__ = ("_factory", "_debouncers", "_pending")

    def __init__(self, factory: Callable[[], HysteresisDebouncer] = fist_debouncer) -> None:
        self._factory = factory
        self._debouncers: dict[HandKey, HysteresisDebouncer] = {}
        self._pending: dict[int, int] = {}

    def update(self, display_id: int, side: str, score: float, t: float) -> bool:
        """Feed one ``Closed_Fist`` score for a hand; True iff an onset was counted."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        key = (display_id, side)
        deb = self._debouncers.get(key)
        if deb is None:
            deb = self._debouncers[key] = self._factory()
        onset = deb.update(score, t)
        if onset:
            self._pending[display_id] = self._pending.get(display_id, 0) + 1
        return onset

    def mark_unobserved(self, t: float, seen: set[HandKey]) -> None:
        """Feed NaN ("not observed") to every known hand not in ``seen`` this tick."""
        for key, deb in self._debouncers.items():
            if key not in seen:
                deb.update(math.nan, t)

    def person_state(self, display_id: int) -> tuple[int, bool]:
        """``(onsets since the last call for this person, any hand active)``."""
        onsets = self._pending.pop(display_id, 0)
        active = any(self._debouncers[(display_id, side)].active
                     for side in SIDES if (display_id, side) in self._debouncers)
        return onsets, active

    def drop(self, display_id: int) -> None:
        """Forget a person entirely (their display id was released)."""
        for side in SIDES:
            self._debouncers.pop((display_id, side), None)
        self._pending.pop(display_id, None)

    def known(self) -> frozenset[HandKey]:
        """Hands currently being tracked."""
        return frozenset(self._debouncers)
