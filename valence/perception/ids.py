"""Stable, ephemeral "Person #N" labels on top of ByteTrack tracker ids.

ByteTrack hands out ever-increasing tracker ids and re-issues a fresh one after an
occlusion, which would turn one person into Person #7, #8, #9 over a session. This
mapper keeps the label small and steady: a track that vanishes keeps its display id
reserved for ``reuse_window_s``, and a new track that appears near where it vanished
inherits it. Only integer ids, normalized box centres and timestamps are kept; there
is no appearance model, embedding or identity, and everything is forgotten once the
reservation expires.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from valence.perception.debounce import TIME_EPS

BBox = tuple[float, float, float, float]
Track = tuple[int, BBox]


@dataclass(slots=True)
class _Lost:
    tracker_id: int
    centre: tuple[float, float]
    lost_t: float


def _centre(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class DisplayIdMapper:
    """Map ByteTrack tracker ids to display ids 1..N.

    ``update`` takes the tracks of one frame as ``(tracker_id, (x1, y1, x2, y2))``
    with boxes normalized to [0, 1]. Distances are measured in frame widths:
    ``dx`` in normalized x and ``dy`` in normalized y times ``aspect`` (frame h / w).
    A new tracker id first reclaims its own reservation (ByteTrack revived it), then
    the closest reserved display id within ``reuse_dist`` (matched globally, nearest
    pairs first), else the smallest display id that is neither active nor reserved.
    """

    __slots__ = ("reuse_window_s", "reuse_dist", "aspect",
                 "_active", "_centres", "_lost", "_released")

    def __init__(self, reuse_window_s: float = 3.0, reuse_dist: float = 0.25,
                 aspect: float = 9 / 16) -> None:
        if reuse_window_s < 0 or reuse_dist < 0 or aspect <= 0:
            raise ValueError("reuse_window_s, reuse_dist must be >= 0 and aspect > 0")
        self.reuse_window_s = reuse_window_s
        self.reuse_dist = reuse_dist
        self.aspect = aspect
        self.reset()

    def reset(self) -> None:
        """Forget all tracks; every display id known so far is reported by ``pop_released``."""
        released: set[int] = set()
        if hasattr(self, "_released"):
            released = self._released | set(self._active.values()) | set(self._lost)
        self._active: dict[int, int] = {}
        self._centres: dict[int, tuple[float, float]] = {}
        self._lost: dict[int, _Lost] = {}
        self._released = released

    def active(self) -> dict[int, int]:
        """Current ``tracker_id -> display_id`` mapping (a copy)."""
        return dict(self._active)

    def reserved(self) -> frozenset[int]:
        """Display ids of recently lost people still held for reuse."""
        return frozenset(self._lost)

    def pop_released(self) -> list[int]:
        """Display ids whose reservation expired since the last call (gone for good)."""
        released, self._released = sorted(self._released), set()
        return released

    def update(self, tracks: Sequence[Track], t: float) -> dict[int, int]:
        """Feed one frame of tracks at time ``t``; returns ``tracker_id -> display_id``."""
        current = self._dedupe(tracks)
        self._expire(t)
        self._mark_lost(current, t)

        new_ids = [tid for tid in current if tid not in self._active]
        for tid in new_ids:
            self._centres[tid] = current[tid]
        unassigned = self._reclaim_own(new_ids)
        unassigned = self._reuse_nearby(unassigned)
        for tid in sorted(unassigned):
            self._active[tid] = self._smallest_free()

        for tid, centre in current.items():
            self._centres[tid] = centre
        return dict(self._active)

    @staticmethod
    def _dedupe(tracks: Sequence[Track]) -> dict[int, tuple[float, float]]:
        current: dict[int, tuple[float, float]] = {}
        for tid, bbox in tracks:
            current.setdefault(int(tid), _centre(bbox))
        return current

    def _expire(self, t: float) -> None:
        horizon = self.reuse_window_s - TIME_EPS
        for did in [d for d, lost in self._lost.items() if t - lost.lost_t >= horizon]:
            del self._lost[did]
            self._released.add(did)

    def _mark_lost(self, current: dict[int, tuple[float, float]], t: float) -> None:
        for tid in [tid for tid in self._active if tid not in current]:
            did = self._active.pop(tid)
            self._lost[did] = _Lost(tid, self._centres.pop(tid), t)

    def _reclaim_own(self, new_ids: list[int]) -> list[int]:
        by_tracker = {lost.tracker_id: did for did, lost in self._lost.items()}
        rest = []
        for tid in new_ids:
            did = by_tracker.get(tid)
            if did is None:
                rest.append(tid)
            else:
                del self._lost[did]
                self._active[tid] = did
        return rest

    def _reuse_nearby(self, new_ids: list[int]) -> list[int]:
        pairs = sorted(
            (d, tid, did)
            for tid in new_ids
            for did, lost in self._lost.items()
            if (d := self._distance(self._centres[tid], lost.centre)) <= self.reuse_dist
        )
        taken: set[int] = set()
        for _, tid, did in pairs:
            if tid in self._active or did in taken:
                continue
            taken.add(did)
            del self._lost[did]
            self._active[tid] = did
        return [tid for tid in new_ids if tid not in self._active]

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], (a[1] - b[1]) * self.aspect)

    def _smallest_free(self) -> int:
        used = set(self._active.values()) | set(self._lost)
        did = 1
        while did in used:
            did += 1
        return did
