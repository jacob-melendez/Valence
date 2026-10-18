from __future__ import annotations

import math

import numpy as np
import pytest

from valence.perception.hands import FistTracker, assign_hands, curled_fingers, is_curl_fist
from valence.perception.debounce import HysteresisDebouncer


def legacy_fist() -> HysteresisDebouncer:
    """Original 0.6 x 3-sample entry: these tests cover FistTracker bookkeeping, not thresholds."""
    return HysteresisDebouncer(on=0.6, off=0.4, on_count=3, off_hold_s=1.0, refractory_s=3.0)

W = 1000
NAN = math.nan


def make_hand(curled: set[int] = frozenset(), wrist: tuple[float, float] = (0.5, 0.8)) -> np.ndarray:
    """Synthetic upright hand (21, 3), normalized. ``curled`` holds finger indices 0..3 (index..pinky)."""
    lm = np.zeros((21, 3))
    wx, wy = wrist
    lm[0, :2] = wrist
    for i, (x, y) in enumerate([(-0.04, -0.03), (-0.06, -0.06), (-0.07, -0.09), (-0.08, -0.11)], start=1):
        lm[i, :2] = (wx + x, wy + y)
    for f in range(4):
        x = wx - 0.03 + 0.02 * f
        base = 5 + 4 * f
        lm[base, :2] = (x, wy - 0.15)
        lm[base + 1, :2] = (x, wy - 0.22)
        if f in curled:
            lm[base + 2, :2] = (x, wy - 0.18)
            lm[base + 3, :2] = (x, wy - 0.12)
        else:
            lm[base + 2, :2] = (x, wy - 0.27)
            lm[base + 3, :2] = (x, wy - 0.31)
    return lm


def person(lx: float, ly: float, rx: float, ry: float, lc: float = 0.9, rc: float = 0.9) -> dict[str, tuple[float, float, float]]:
    return {"L": (lx, ly, lc), "R": (rx, ry, rc)}


class TestAssignHands:
    def test_matches_nearest_wrist(self) -> None:
        people = {1: person(200, 500, 400, 500), 2: person(700, 500, 900, 500)}
        assert assign_hands([(705, 510), (395, 490)], people, W) == [(2, "L"), (1, "R")]

    def test_global_greedy_beats_hand_order(self) -> None:
        people = {1: person(500, 500, 0, 0, rc=0.0), 2: person(0, 0, 612, 500, lc=0.0)}
        hands = [(510, 500), (505, 500)]
        assert assign_hands(hands, people, W) == [(2, "R"), (1, "L")]

    def test_each_wrist_used_once(self) -> None:
        people = {1: person(500, 500, 0, 0, rc=0.0)}
        assert assign_hands([(510, 500), (505, 500)], people, W) == [None, (1, "L")]

    def test_low_confidence_wrists_ignored(self) -> None:
        people = {1: person(500, 500, 520, 500, lc=0.49, rc=0.5)}
        assert assign_hands([(500, 500)], people, W) == [(1, "R")]
        people = {1: person(500, 500, 520, 500, lc=0.2, rc=0.3)}
        assert assign_hands([(500, 500)], people, W) == [None]

    def test_distance_limit_scales_with_frame_width(self) -> None:
        people = {1: person(500, 500, 0, 0, rc=0.0)}
        assert assign_hands([(619, 500)], people, W) == [(1, "L")]
        assert assign_hands([(621, 500)], people, W) == [None]
        assert assign_hands([(621, 500)], people, 2 * W) == [(1, "L")]
        assert assign_hands([(560, 500)], people, W, max_dist_frac=0.05) == [None]

    def test_nan_and_empty_inputs(self) -> None:
        people = {1: person(NAN, NAN, 500, 500)}
        assert assign_hands([(NAN, NAN), (505, 500)], people, W) == [None, (1, "R")]
        assert assign_hands([], people, W) == []
        assert assign_hands([(500, 500)], {}, W) == [None]


class TestCurl:
    def test_open_hand(self) -> None:
        hand = make_hand()
        assert curled_fingers(hand) == 0
        assert not is_curl_fist(hand)

    def test_closed_hand(self) -> None:
        hand = make_hand({0, 1, 2, 3})
        assert curled_fingers(hand) == 4
        assert is_curl_fist(hand)

    @pytest.mark.parametrize(("curled", "fist"), [({0}, False), ({1, 2}, False), ({0, 1, 2}, True),
                                                  ({1, 2, 3}, True)])
    def test_three_of_four_threshold(self, curled: set[int], fist: bool) -> None:
        hand = make_hand(curled)
        assert curled_fingers(hand) == len(curled)
        assert is_curl_fist(hand) is fist

    def test_accepts_2d_landmarks_and_ignores_z(self) -> None:
        hand = make_hand({0, 1, 2})
        assert curled_fingers(hand[:, :2]) == 3
        hand[:, 2] = np.linspace(-1, 1, 21)
        assert curled_fingers(hand) == 3

    def test_aspect_makes_distances_isotropic(self) -> None:
        hand = make_hand()
        wx, wy = hand[0, :2]
        for f in range(4):
            hand[6 + 4 * f, :2] = (wx, wy - 0.20)
            hand[8 + 4 * f, :2] = (wx + 0.15, wy)
        assert curled_fingers(hand, aspect=1.0) == 4
        assert curled_fingers(hand, aspect=9 / 16) == 0

    def test_nan_landmarks_never_count(self) -> None:
        hand = make_hand({0, 1, 2, 3})
        hand[8] = NAN
        hand[12] = NAN
        assert curled_fingers(hand) == 2

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            curled_fingers(np.zeros((20, 2)))
        with pytest.raises(ValueError):
            curled_fingers(np.zeros(42))


def clench(tracker: FistTracker, pid: int, side: str, t0: float, step: float = 0.2) -> list[bool]:
    return [tracker.update(pid, side, 0.9, t0 + i * step) for i in range(3)]


class TestFistTracker:
    def test_onset_reported_as_delta(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        assert clench(ft, 1, "L", 0.0) == [False, False, True]
        assert ft.person_state(1) == (1, True)
        assert ft.person_state(1) == (0, True)

    def test_both_hands_together_are_one_onset(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        clench(ft, 1, "L", 0.0)
        clench(ft, 1, "R", 0.0)
        assert ft.person_state(1) == (1, True)
        for i in range(1, 8):
            ft.update(1, "L", 0.0, 0.4 + i * 0.2)
        assert ft.person_state(1) == (0, True)
        for i in range(1, 8):
            ft.update(1, "R", 0.0, 0.4 + i * 0.2)
        assert ft.person_state(1) == (0, False)

    def test_people_are_independent(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        clench(ft, 1, "L", 0.0)
        assert ft.person_state(2) == (0, False)
        assert ft.person_state(1) == (1, True)

    def test_refractory_is_per_person(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        clench(ft, 1, "L", 0.0)  # onset at 0.4
        for i in range(1, 8):
            ft.update(1, "L", 0.0, 0.4 + i * 0.2)
        assert ft.person_state(1) == (1, False)
        assert clench(ft, 1, "R", 2.0)[-1] is False  # other hand, but within 3 s of the person's onset
        assert ft.person_state(1) == (0, True)
        for i in range(1, 8):
            ft.update(1, "R", 0.0, 2.4 + i * 0.2)
        assert clench(ft, 1, "L", 5.0)[-1] is True  # after the refractory
        assert ft.person_state(1) == (1, True)

    def test_second_hand_while_first_clenched_adds_nothing(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        clench(ft, 1, "L", 0.0)
        ft.person_state(1)
        assert clench(ft, 1, "R", 4.0)[-1] is False  # L still clenched (no low samples fed)
        assert ft.person_state(1) == (0, True)

    def test_mark_unobserved_ends_unseen_fists(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        clench(ft, 1, "L", 0.0)
        clench(ft, 2, "R", 0.0)
        for i in range(1, 8):
            t = 0.4 + i * 0.2
            ft.update(2, "R", 0.9, t)
            ft.mark_unobserved(t, {(2, "R")})
        assert ft.person_state(1) == (1, False)
        assert ft.person_state(2) == (1, True)

    def test_mark_unobserved_breaks_unseen_streaks_only(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        for key in [(1, "L"), (1, "R")]:
            ft.update(*key, 0.9, 0.0)
            ft.update(*key, 0.9, 0.2)
        ft.mark_unobserved(0.3, {(1, "R")})
        assert ft.update(1, "L", 0.9, 0.4) is False
        assert ft.update(1, "R", 0.9, 0.4) is True

    def test_drop_forgets_person(self) -> None:
        ft = FistTracker(factory=legacy_fist)
        clench(ft, 3, "L", 0.0)
        clench(ft, 4, "L", 0.0)
        ft.drop(3)
        assert ft.person_state(3) == (0, False)
        assert ft.known() == frozenset({(4, "L")})
        assert clench(ft, 3, "L", 0.8)[-1] is True
        ft.drop(99)

    def test_rejects_unknown_side(self) -> None:
        with pytest.raises(ValueError):
            FistTracker().update(1, "left", 0.9, 0.0)


def test_default_tracker_uses_the_contract_fist_rule() -> None:
    ft = FistTracker()
    onsets = [ft.update(1, "L", 0.75, t) for t in (0.0, 0.2, 0.4, 0.6)]
    assert onsets == [False, False, False, True]  # >= 0.7 held >= 0.5 s
