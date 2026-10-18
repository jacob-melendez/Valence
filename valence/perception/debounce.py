"""Hysteresis debouncer: noisy per-sample scores -> stable state + counted onsets.

Shared by fist detection (per person x hand) and laughter detection. Pure logic: it
sees only scalar scores and timestamps, never media.
"""

from __future__ import annotations

import math

TIME_EPS = 1e-9  # absorbs float error in timestamp differences (2.3 - 1.3 < 1.0)


class HysteresisDebouncer:
    """Enter when ``score >= on`` for ``on_count`` consecutive samples spanning at least
    ``on_hold_s`` seconds (first to last high sample); exit once
    ``score < off`` has held for ``off_hold_s`` seconds (0 exits on the first low
    sample). A rising edge is counted as an onset only if ``refractory_s`` has passed
    since the previous counted onset. Scores between ``off`` and ``on`` keep the
    current state (and break an entry streak). NaN means "not observed": it breaks
    an entry streak and counts as low for exit.
    """

    __slots__ = ("on", "off", "on_count", "on_hold_s", "off_hold_s", "refractory_s",
                 "active", "onsets", "_streak", "_high_since", "_low_since", "_last_onset_t")

    def __init__(self, on: float, off: float, on_count: int = 1, off_hold_s: float = 0.0,
                 refractory_s: float = 0.0, on_hold_s: float = 0.0) -> None:
        if not off <= on:
            raise ValueError("off threshold must be <= on threshold")
        self.on, self.off = on, off
        self.on_count, self.on_hold_s = on_count, on_hold_s
        self.off_hold_s, self.refractory_s = off_hold_s, refractory_s
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.onsets = 0  # cumulative counted rising edges
        self._streak = 0
        self._high_since: float | None = None
        self._low_since: float | None = None
        self._last_onset_t = -math.inf

    def update(self, score: float, t: float) -> bool:
        """Feed one sample; returns True iff an onset was counted at this sample."""
        observed = not math.isnan(score)
        if not self.active:
            if observed and score >= self.on:
                self._streak += 1
                if self._high_since is None:
                    self._high_since = t
            else:
                self._streak, self._high_since = 0, None
            if self._streak < self.on_count or t - self._high_since < self.on_hold_s - TIME_EPS:
                return False
            self.active, self._streak, self._high_since, self._low_since = True, 0, None, None
            if t - self._last_onset_t >= self.refractory_s - TIME_EPS:
                self._last_onset_t = t
                self.onsets += 1
                return True
            return False

        if observed and score >= self.off:
            self._low_since = None
            return False
        if self._low_since is None:
            self._low_since = t
        if t - self._low_since >= self.off_hold_s - TIME_EPS:
            self.active, self._low_since = False, None
        return False


def fist_debouncer() -> HysteresisDebouncer:
    """Closed_Fist: enter at >= 0.7 held >= 0.5 s, exit < 0.4 for 1 s, 3 s refractory.

    (Raised from 0.6 x 3 samples after live typing produced false fists; the keyboard zone
    is excluded upstream in the video perceiver.)"""
    return HysteresisDebouncer(on=0.7, off=0.4, on_count=1, on_hold_s=0.5, off_hold_s=1.0, refractory_s=3.0)


def laughter_debouncer() -> HysteresisDebouncer:
    """YAMNet laughter family: ON >= 0.05 for 1 result, OFF below 0.02 held 1.5 s, 5.5 s refractory.

    Measured on the laptop mic (docs/BENCHMARKS.md, laugh probe): real laughter scored mostly as
    Speech, with the laughter family at 0.04-0.08 and a single 0.50; quiet and normal talking scored
    exactly 0.00 even at Speech 0.99. The original 0.30 x 2-results rule never fired. The family
    is specific enough for a low threshold; other people's laughter is removed downstream by the
    near-field gate (only events well above the room's level count).

    Inside one laugh the family drops to 0.00 for single seconds, so an instant exit re-armed the
    state and a ~5 s laugh counted twice (capture run 3). The exit hold rides through up to two
    low results (exit on the third); the refractory catches longer dips while laughs 6 blocks
    apart still count separately (replay table in docs/BENCHMARKS.md). Both sit half a block off
    the 1 s grid: live t is time.monotonic() at block end and jitters by milliseconds, and a
    value of exactly 2.0 / 6.0 made the outcome depend on which side of 1.000 s the blocks fell."""
    return HysteresisDebouncer(on=0.05, off=0.02, on_count=1, off_hold_s=1.5, refractory_s=5.5)
