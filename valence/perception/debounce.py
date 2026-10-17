"""Hysteresis debouncer: noisy per-sample scores -> stable state + counted onsets.

Shared by fist detection (per person x hand) and laughter detection. Pure logic: it
sees only scalar scores and timestamps, never media.
"""

from __future__ import annotations

import math


class HysteresisDebouncer:
    """Enter when ``score >= on`` for ``on_count`` consecutive samples; exit once
    ``score < off`` has held for ``off_hold_s`` seconds (0 exits on the first low
    sample). A rising edge is counted as an onset only if ``refractory_s`` has passed
    since the previous counted onset. Scores between ``off`` and ``on`` keep the
    current state (and break an entry streak). NaN means "not observed": it breaks
    an entry streak and counts as low for exit.
    """

    __slots__ = ("on", "off", "on_count", "off_hold_s", "refractory_s",
                 "active", "onsets", "_streak", "_low_since", "_last_onset_t")

    def __init__(self, on: float, off: float, on_count: int = 1,
                 off_hold_s: float = 0.0, refractory_s: float = 0.0) -> None:
        if not off <= on:
            raise ValueError("off threshold must be <= on threshold")
        self.on, self.off = on, off
        self.on_count, self.off_hold_s, self.refractory_s = on_count, off_hold_s, refractory_s
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.onsets = 0  # cumulative counted rising edges
        self._streak = 0
        self._low_since: float | None = None
        self._last_onset_t = -math.inf

    def update(self, score: float, t: float) -> bool:
        """Feed one sample; returns True iff an onset was counted at this sample."""
        observed = not math.isnan(score)
        if not self.active:
            self._streak = self._streak + 1 if observed and score >= self.on else 0
            if self._streak < self.on_count:
                return False
            self.active, self._streak, self._low_since = True, 0, None
            if t - self._last_onset_t >= self.refractory_s:
                self._last_onset_t = t
                self.onsets += 1
                return True
            return False

        if observed and score >= self.off:
            self._low_since = None
            return False
        if self._low_since is None:
            self._low_since = t
        if t - self._low_since >= self.off_hold_s:
            self.active, self._low_since = False, None
        return False


def fist_debouncer() -> HysteresisDebouncer:
    """Closed_Fist: enter >= 0.6 x3 samples, exit < 0.4 for 1 s, 3 s refractory."""
    return HysteresisDebouncer(on=0.6, off=0.4, on_count=3, off_hold_s=1.0, refractory_s=3.0)


def laughter_debouncer() -> HysteresisDebouncer:
    """YAMNet laughter family: ON >= 0.30 for 2 results, OFF below 0.15, 3 s refractory."""
    return HysteresisDebouncer(on=0.30, off=0.15, on_count=2, off_hold_s=0.0, refractory_s=3.0)
