from __future__ import annotations

import math

import pytest

from valence.perception.debounce import HysteresisDebouncer, fist_debouncer, laughter_debouncer


def legacy_fist() -> HysteresisDebouncer:
    """The original fist parameters (0.6 x 3 samples): exercises the generic state machine."""
    return HysteresisDebouncer(on=0.6, off=0.4, on_count=3, off_hold_s=1.0, refractory_s=3.0)

NAN = math.nan


def feed(deb: HysteresisDebouncer, samples: list[tuple[float, float]]) -> list[bool]:
    return [deb.update(score, t) for t, score in samples]


def enter_fist(deb: HysteresisDebouncer, t0: float = 0.0, step: float = 0.25) -> float:
    """Drive a fist debouncer into the active state; returns the time of the onset sample."""
    times = [t0 + i * step for i in range(3)]
    feed(deb, [(t, 0.8) for t in times])
    assert deb.active
    return times[-1]


class TestFistEntry:
    def test_needs_three_consecutive_high_samples(self) -> None:
        deb = legacy_fist()
        assert feed(deb, [(0.0, 0.7), (0.25, 0.7)]) == [False, False]
        assert not deb.active
        assert deb.update(0.7, 0.5) is True
        assert deb.active and deb.onsets == 1

    def test_threshold_is_inclusive(self) -> None:
        deb = legacy_fist()
        assert feed(deb, [(0.0, 0.6), (0.25, 0.6), (0.5, 0.6)]) == [False, False, True]

    def test_band_breaks_entry_streak(self) -> None:
        deb = legacy_fist()
        feed(deb, [(0.0, 0.7), (0.25, 0.7), (0.5, 0.5), (0.75, 0.7), (1.0, 0.7)])
        assert not deb.active and deb.onsets == 0
        assert deb.update(0.7, 1.25) is True

    def test_low_sample_breaks_entry_streak(self) -> None:
        deb = legacy_fist()
        feed(deb, [(0.0, 0.9), (0.25, 0.9), (0.5, 0.1), (0.75, 0.9), (1.0, 0.9)])
        assert not deb.active

    def test_nan_breaks_entry_streak(self) -> None:
        deb = legacy_fist()
        feed(deb, [(0.0, 0.9), (0.25, 0.9), (0.5, NAN), (0.75, 0.9), (1.0, 0.9)])
        assert not deb.active and deb.onsets == 0


class TestFistExit:
    def test_band_holds_active_state(self) -> None:
        deb = legacy_fist()
        t = enter_fist(deb)
        for i in range(1, 41):
            deb.update(0.5, t + i * 0.25)
        assert deb.active
        deb.update(0.4, t + 11.0)
        assert deb.active

    def test_exit_after_low_held_one_second(self) -> None:
        deb = legacy_fist()
        enter_fist(deb)
        feed(deb, [(1.0, 0.3), (1.5, 0.3), (1.75, 0.3)])
        assert deb.active
        deb.update(0.3, 2.0)
        assert not deb.active

    def test_band_sample_restarts_exit_hold(self) -> None:
        deb = legacy_fist()
        enter_fist(deb)
        feed(deb, [(1.0, 0.3), (1.5, 0.5), (1.75, 0.3), (2.5, 0.3)])
        assert deb.active
        deb.update(0.3, 2.75)
        assert not deb.active

    def test_nan_counts_as_low_for_exit(self) -> None:
        deb = legacy_fist()
        enter_fist(deb)
        feed(deb, [(1.0, NAN), (1.5, NAN)])
        assert deb.active
        deb.update(NAN, 2.0)
        assert not deb.active

    def test_nan_and_low_samples_share_exit_hold(self) -> None:
        deb = legacy_fist()
        enter_fist(deb)
        feed(deb, [(1.0, 0.1), (1.5, NAN), (2.0, 0.2)])
        assert not deb.active


class TestRefractory:
    def test_suppresses_counting_but_state_still_toggles(self) -> None:
        deb = legacy_fist()
        assert enter_fist(deb, t0=0.0) == 0.5
        feed(deb, [(0.75, 0.0), (1.75, 0.0)])
        assert not deb.active

        onsets = feed(deb, [(2.0, 0.9), (2.25, 0.9), (2.5, 0.9)])
        assert onsets == [False, False, False]
        assert deb.active and deb.onsets == 1

        feed(deb, [(2.75, 0.0), (3.75, 0.0)])
        assert not deb.active
        assert feed(deb, [(4.0, 0.9), (4.25, 0.9), (4.5, 0.9)]) == [False, False, True]
        assert deb.onsets == 2

    def test_refractory_measured_from_last_counted_onset(self) -> None:
        deb = legacy_fist()
        enter_fist(deb, t0=0.0)
        feed(deb, [(1.0, 0.0), (2.0, 0.0)])
        enter_fist(deb, t0=2.25)
        feed(deb, [(3.0, 0.0), (4.0, 0.0)])
        assert not deb.active and deb.onsets == 1
        feed(deb, [(4.25, 0.9), (4.5, 0.9), (4.75, 0.9)])
        assert deb.onsets == 2

    def test_held_fist_counts_once(self) -> None:
        deb = legacy_fist()
        results = feed(deb, [(i * 0.1, 0.95) for i in range(200)])
        assert sum(results) == 1 and deb.onsets == 1


class TestLaughter:
    def test_on_after_two_results(self) -> None:
        deb = laughter_debouncer()
        assert deb.update(0.30, 0.0) is False
        assert deb.update(0.30, 0.48) is True
        assert deb.active

    def test_off_on_first_low_result(self) -> None:
        deb = laughter_debouncer()
        feed(deb, [(0.0, 0.5), (0.48, 0.5)])
        deb.update(0.15, 0.96)
        assert deb.active
        deb.update(0.14, 1.44)
        assert not deb.active

    def test_band_holds_and_breaks_streak(self) -> None:
        deb = laughter_debouncer()
        feed(deb, [(0.0, 0.4), (0.48, 0.2), (0.96, 0.4)])
        assert not deb.active
        feed(deb, [(1.44, 0.4)])
        assert deb.active
        feed(deb, [(t * 0.48, 0.2) for t in range(4, 20)])
        assert deb.active

    def test_nan_turns_off(self) -> None:
        deb = laughter_debouncer()
        feed(deb, [(0.0, 0.5), (0.48, 0.5), (0.96, NAN)])
        assert not deb.active

    def test_refractory_three_seconds(self) -> None:
        deb = laughter_debouncer()
        feed(deb, [(0.0, 0.5), (0.5, 0.5), (1.0, 0.0), (1.5, 0.5), (2.0, 0.5)])
        assert deb.active and deb.onsets == 1
        feed(deb, [(2.5, 0.0), (3.0, 0.5), (3.5, 0.5)])
        assert deb.active and deb.onsets == 2


class TestGeneral:
    def test_rejects_inverted_thresholds(self) -> None:
        with pytest.raises(ValueError):
            HysteresisDebouncer(on=0.3, off=0.5)

    def test_reset_clears_everything(self) -> None:
        deb = legacy_fist()
        enter_fist(deb)
        deb.reset()
        assert not deb.active and deb.onsets == 0
        assert feed(deb, [(1.0, 0.9), (1.25, 0.9), (1.5, 0.9)])[-1] is True

    def test_defaults_single_sample_entry_and_instant_exit(self) -> None:
        deb = HysteresisDebouncer(on=0.5, off=0.5)
        assert deb.update(0.5, 0.0) is True
        deb.update(0.49, 0.1)
        assert not deb.active
        assert deb.update(0.9, 0.2) is True
        assert deb.onsets == 2


class TestFloatTimestamps:
    """Holds must not slip a sample because ``2.3 - 1.3 == 0.9999999999999998``."""

    def test_exit_hold_tolerates_float_rounding(self) -> None:
        deb = legacy_fist()
        enter_fist(deb)
        deb.update(0.1, 1.3)
        deb.update(0.1, 2.3)
        assert not deb.active

    def test_refractory_tolerates_float_rounding(self) -> None:
        deb = legacy_fist()
        assert feed(deb, [(0.9, 0.9), (1.0, 0.9), (1.1, 0.9)])[-1] is True
        feed(deb, [(1.5, 0.0), (2.5, 0.0)])
        assert not deb.active
        assert feed(deb, [(3.9, 0.9), (4.0, 0.9), (4.1, 0.9)])[-1] is True
        assert deb.onsets == 2


class TestFistRule:
    """Contract addendum: score >= 0.7 held >= 0.5 s; exit < 0.4 for 1 s; 3 s refractory."""

    def test_needs_half_a_second_at_0_7(self):
        deb = fist_debouncer()
        assert feed(deb, [(0.0, 0.7), (0.2, 0.75), (0.4, 0.8), (0.5, 0.8)]) == [False, False, False, True]

    def test_0_65_never_enters(self):
        deb = fist_debouncer()
        assert not any(feed(deb, [(i * 0.2, 0.65) for i in range(20)]))

    def test_dip_restarts_the_hold(self):
        deb = fist_debouncer()
        out = feed(deb, [(0.0, 0.9), (0.2, 0.9), (0.4, 0.5), (0.6, 0.9), (0.8, 0.9), (1.0, 0.9), (1.1, 0.9)])
        assert out == [False, False, False, False, False, False, True]

    def test_nan_restarts_the_hold(self):
        deb = fist_debouncer()
        assert not any(feed(deb, [(0.0, 0.9), (0.2, 0.9), (0.3, float("nan")), (0.4, 0.9), (0.6, 0.9)]))
