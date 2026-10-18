from __future__ import annotations

import math

import numpy as np
import pytest

from valence.perception.posture import PostureState, PostureTracker

SHOULDER_Y = 400.0
SHOULDER_W = 200.0
HIDDEN = 0.0


def pose(head_drop: float = 0.7, shift: tuple[float, float] = (0.0, 0.0),
         wrists: tuple[tuple[float, float], tuple[float, float]] | None = None,
         hidden: tuple[int, ...] = (), ear_drop: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Seated, camera-facing COCO-17 skeleton in pixels; shoulders 200 px wide at y=400.

    The whole head moves with ``head_drop`` (a slump) unless ``ear_drop`` pins the ears
    (a head pitched down: nose low, ears in place)."""
    nose_y = SHOULDER_Y - head_drop * SHOULDER_W
    xy = np.array([
        (600, nose_y),
        (620, nose_y - 15), (580, nose_y - 15),
        (640, nose_y - 5), (560, nose_y - 5),
        (700, SHOULDER_Y), (500, SHOULDER_Y),
        (720, 520), (480, 520),
        (700, 600), (500, 600),
        (660, 650), (540, 650),
        (660, 800), (540, 800), (660, 950), (540, 950),
    ], dtype=float)
    if ear_drop is not None:
        xy[3, 1] = xy[4, 1] = SHOULDER_Y - ear_drop * SHOULDER_W
    if wrists is not None:
        xy[9], xy[10] = wrists
    xy += np.asarray(shift, dtype=float)
    conf = np.full(17, 0.9)
    conf[13:] = HIDDEN
    conf[list(hidden)] = HIDDEN
    return xy, conf


def run(tracker: PostureTracker, t0: float, t1: float, hz: float = 10.0, **pose_kwargs) -> list[tuple[float, PostureState]]:
    """Feed a constant pose on [t0, t1) at ``hz``; returns (t, state) pairs."""
    n = round((t1 - t0) * hz)
    out = []
    for i in range(n):
        t = round(t0 + i / hz, 6)
        out.append((t, tracker.update(*pose(**pose_kwargs), t)))
    return out


def calibrated(head_drop: float = 0.7) -> PostureTracker:
    tracker = PostureTracker()
    run(tracker, 0.0, 20.1, head_drop=head_drop)
    assert tracker.baseline == pytest.approx(head_drop)
    return tracker


def first(states: list[tuple[float, PostureState]], attr: str) -> float | None:
    return next((t for t, s in states if getattr(s, attr)), None)


class TestCalibration:
    def test_calibrates_after_baseline_window(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 20.0, head_drop=0.1)
        assert not any(s.calibrated or s.slump for _, s in states)
        state = tracker.update(*pose(head_drop=0.7), 20.0)
        assert state.calibrated

    def test_baseline_is_median_of_window(self) -> None:
        tracker = PostureTracker()
        values = [0.6, 0.7, 0.8, 0.75, 0.2]
        for i in range(201):
            tracker.update(*pose(head_drop=values[i % len(values)]), i / 10)
        assert tracker.baseline == pytest.approx(0.7)

    def test_samples_after_window_are_ignored(self) -> None:
        tracker = calibrated(0.7)
        run(tracker, 20.1, 60.0, head_drop=0.4)
        assert tracker.baseline == pytest.approx(0.7)

    def test_needs_ten_valid_samples(self) -> None:
        tracker = PostureTracker()
        for t in range(5):
            tracker.update(*pose(), float(t))
        for t in range(5, 25):
            assert not tracker.update(*pose(hidden=(0,)), float(t)).calibrated
        states = [tracker.update(*pose(), float(t)).calibrated for t in range(25, 30)]
        assert states == [False, False, False, False, True]
        assert tracker.baseline == pytest.approx(0.7)

    def test_non_positive_baseline_restarts_calibration(self) -> None:
        tracker = PostureTracker()
        run(tracker, 0.0, 20.1, head_drop=-0.2)
        assert tracker.baseline is None
        states = run(tracker, 20.1, 40.1, head_drop=0.7)
        assert not states[-2][1].calibrated
        assert states[-1][1].calibrated
        assert tracker.baseline == pytest.approx(0.7)


class TestSlump:
    def test_slump_after_three_seconds_below_ratio(self) -> None:
        tracker = calibrated(0.7)
        run(tracker, 20.1, 30.0, head_drop=0.7)
        states = run(tracker, 30.0, 40.0, head_drop=0.3)
        onset = first(states, "slump")
        assert onset is not None
        assert 33.0 <= onset <= 33.5
        assert all(s.slump for t, s in states if t >= onset)

    def test_brief_dip_does_not_slump(self) -> None:
        tracker = calibrated(0.7)
        for k in range(5):
            run(tracker, 25.0 + 6 * k, 27.5 + 6 * k, head_drop=0.3)
            states = run(tracker, 27.5 + 6 * k, 31.0 + 6 * k, head_drop=0.7)
            assert not any(s.slump for _, s in states)

    def test_mild_lean_above_ratio_is_not_slump(self) -> None:
        tracker = calibrated(0.7)
        states = run(tracker, 20.1, 40.0, head_drop=0.52)
        assert not any(s.slump for _, s in states)

    def test_recovers_as_soon_as_head_rises(self) -> None:
        tracker = calibrated(0.7)
        assert run(tracker, 20.1, 30.0, head_drop=0.3)[-1][1].slump
        states = run(tracker, 30.0, 32.0, head_drop=0.7)
        cleared = next(t for t, s in states if not s.slump)
        assert cleared <= 30.3
        assert not any(s.slump for t, s in states if t >= cleared)

    def test_head_drop_is_smoothed(self) -> None:
        tracker = calibrated(0.7)
        state = tracker.update(*pose(head_drop=0.3), 20.1)
        assert state.head_drop == pytest.approx(0.3 * 0.3 + 0.7 * 0.7)

    def test_occluded_nose_holds_head_drop(self) -> None:
        tracker = calibrated(0.7)
        state = tracker.update(*pose(head_drop=0.1, hidden=(0,)), 20.1)
        assert state.head_drop == pytest.approx(0.7)

    def test_scale_invariant(self) -> None:
        near, far = PostureTracker(), PostureTracker()
        for i in range(30):
            xy, conf = pose(head_drop=0.6)
            near.update(xy, conf, i / 10)
            far.update(xy * 0.5, conf, i / 10)
        assert near.update(*pose(0.6), 3.0).head_drop == pytest.approx(
            far.update(pose(0.6)[0] * 0.5, pose(0.6)[1], 3.0).head_drop)


FACE_WRISTS = ((650.0, 300.0), (550.0, 300.0))


class TestHeadInHands:
    def test_held_two_seconds_with_occluded_nose(self) -> None:
        tracker = PostureTracker()
        run(tracker, 0.0, 1.0)
        states = run(tracker, 1.0, 5.0, wrists=FACE_WRISTS, hidden=(0,))
        onset = first(states, "head_in_hands")
        assert onset == pytest.approx(3.0)
        assert all(s.head_in_hands for t, s in states if t >= onset)

    def test_works_with_only_ears_visible(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 3.0, wrists=FACE_WRISTS, hidden=(0, 1, 2))
        assert states[-1][1].head_in_hands

    def test_no_head_points_means_no_detection(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 4.0, wrists=FACE_WRISTS, hidden=(0, 1, 2, 3, 4))
        assert not any(s.head_in_hands for _, s in states)

    def test_breaks_when_hands_drop(self) -> None:
        tracker = PostureTracker()
        run(tracker, 0.0, 3.0, wrists=FACE_WRISTS, hidden=(0,))
        state = tracker.update(*pose(), 3.0)
        assert not state.head_in_hands
        states = run(tracker, 3.1, 4.9, wrists=FACE_WRISTS, hidden=(0,))
        assert not any(s.head_in_hands for _, s in states)

    def test_raised_hands_far_from_head(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 4.0, wrists=((900.0, 150.0), (300.0, 150.0)))
        assert not any(s.head_in_hands for _, s in states)

    def test_one_hand_is_not_enough(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 4.0, wrists=((650.0, 300.0), (500.0, 600.0)), hidden=(0,))
        assert not any(s.head_in_hands for _, s in states)

    def test_hands_below_shoulder_line(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 4.0, head_drop=0.05, wrists=((650.0, 420.0), (550.0, 420.0)))
        assert not any(s.head_in_hands for _, s in states)

    def test_occluded_wrist_is_not_assumed(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 4.0, wrists=FACE_WRISTS, hidden=(0, 10))
        assert not any(s.head_in_hands for _, s in states)

    def test_missing_shoulder_reuses_width(self) -> None:
        tracker = PostureTracker()
        run(tracker, 0.0, 1.0)
        assert tracker.shoulder_width == pytest.approx(SHOULDER_W)
        states = run(tracker, 1.0, 4.0, wrists=FACE_WRISTS, hidden=(0, 6))
        assert tracker.shoulder_width == pytest.approx(SHOULDER_W)
        assert states[-1][1].head_in_hands


class TestMotion:
    def test_still_is_zero(self) -> None:
        tracker = PostureTracker()
        states = run(tracker, 0.0, 2.0)
        assert math.isnan(states[0][1].motion)  # no previous sample yet: undefined, not "still"
        assert all(s.motion == pytest.approx(0.0) for _, s in states[1:])

    @pytest.mark.parametrize("hz", [5.0, 10.0, 30.0])
    def test_units_are_shoulder_widths_per_second(self, hz: float) -> None:
        tracker = PostureTracker()
        speed_px = 100.0
        state = None
        for i in range(int(3 * hz)):
            t = i / hz
            state = tracker.update(*pose(shift=(speed_px * t, 0.0)), t)
        assert state is not None
        assert state.motion == pytest.approx(speed_px / SHOULDER_W)

    def test_motion_smoothed_and_decays(self) -> None:
        tracker = PostureTracker()
        tracker.update(*pose(), 0.0)
        moved = tracker.update(*pose(shift=(20.0, 0.0)), 0.1)
        assert moved.motion == pytest.approx(20.0 / SHOULDER_W / 0.1)
        still = tracker.update(*pose(shift=(20.0, 0.0)), 0.2)
        assert still.motion == pytest.approx(0.7 * moved.motion)

    def test_only_keypoints_valid_in_both_samples(self) -> None:
        tracker = PostureTracker()
        tracker.update(*pose(hidden=(9,)), 0.0)
        state = tracker.update(*pose(wrists=((900.0, 100.0), (500.0, 600.0))), 0.1)
        assert state.motion == pytest.approx(0.0)

    def test_undefined_cases_report_nan(self) -> None:
        """Contract: NaN when undefined, because 0.0 means "still"."""
        tracker = PostureTracker()
        xy, conf = pose()
        assert math.isnan(tracker.update(xy, np.zeros(17), 0.0).motion)
        assert math.isnan(tracker.update(xy, conf, 0.0).motion)
        assert math.isnan(tracker.update(xy + 50, conf, 0.0).motion)


BODY = list(range(5, 13))  # shoulders, elbows, wrists, hips: the points motion uses (8 in the fixture)
HEAD = [0, 1, 2, 3, 4]


def all_keypoint_motion(frames: list[tuple[np.ndarray, np.ndarray, float]], alpha: float = 0.3) -> list[float]:
    """The pre-P1-1 definition (mean over every keypoint valid in both samples), to show the artifact."""
    out, prev, m = [], None, math.nan
    for xy, conf, t in frames:
        xy = np.where((conf >= 0.5)[:, None], xy, np.nan)
        if prev is not None:
            both = np.isfinite(xy).all(axis=1) & np.isfinite(prev[0]).all(axis=1)
            raw = np.linalg.norm(xy[both] - prev[0][both], axis=1).mean() / SHOULDER_W / (t - prev[1])
            m = raw if math.isnan(m) else alpha * raw + (1 - alpha) * m
        prev = (xy, t)
        out.append(m)
    return out


def jittery(head_drop: float, n: int, hz: float, seed: int, head_jitter: bool) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Seated person holding still: 0.3 px body jitter (same per seed; reads ~0.027 sw/s at 10 Hz, the
    live typing p50); optionally the head-drop artifact,
    where nose/eyes/ears wander up to ~0.3 shoulder widths and flicker between conf 0.3 and 0.95."""
    body_rng, head_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    frames = []
    for i in range(n):
        xy, conf = pose(head_drop=head_drop)
        xy[BODY] += body_rng.normal(0.0, 0.3, (len(BODY), 2))
        if head_jitter:
            xy[HEAD] += head_rng.normal(0.0, 0.3 * SHOULDER_W, (len(HEAD), 2))
            conf[HEAD] = head_rng.uniform(0.3, 0.95, len(HEAD))
        frames.append((xy, conf, round(i / hz, 6)))
    return frames


class TestMotionTorsoOnly:
    """Motion = torso/arm keypoints only, identity jumps rejected, last value held when unmeasurable."""

    def test_held_slump_with_head_jitter_reads_as_still(self) -> None:
        upright = jittery(0.7, 300, 10.0, seed=7, head_jitter=False)
        slumped = jittery(0.3, 300, 10.0, seed=7, head_jitter=True)
        up_t, sl_t = PostureTracker(), PostureTracker()
        up = [up_t.update(xy, c, t).motion for xy, c, t in upright][10:]
        sl = [sl_t.update(xy, c, t).motion for xy, c, t in slumped][10:]
        assert sl == pytest.approx(up)  # the head no longer contributes at all
        assert float(np.median(sl)) < 0.033  # at or below the live upright p50
        old_up = float(np.median(all_keypoint_motion(upright)[10:]))
        old_sl = float(np.median(all_keypoint_motion(slumped)[10:]))
        assert old_sl > 10 * old_up  # the old definition reproduces the artifact: a still slump reads agitated

    def test_head_motion_is_ignored(self) -> None:
        tracker = PostureTracker()
        states = [tracker.update(*pose(head_drop=0.7 - 0.02 * i), i / 10) for i in range(30)]
        assert all(s.motion == pytest.approx(0.0) for s in states[1:])

    @pytest.mark.parametrize("hz", [5.0, 10.0, 30.0])
    def test_one_arm_moving_counts_in_shoulder_widths_per_second(self, hz: float) -> None:
        """2 of 8 points move at 0.5 sw/s -> 0.125 sw/s. A median of displacements would read 0.0 here,
        which is why the mean is kept and outliers are gated instead."""
        tracker = PostureTracker()
        speed_px = 100.0
        state = None
        for i in range(int(3 * hz)):
            t = i / hz
            xy, conf = pose()
            xy[[7, 9], 0] += speed_px * t  # left elbow + wrist
            state = tracker.update(xy, conf, t)
        assert state is not None
        assert state.motion == pytest.approx(2 / 8 * speed_px / SHOULDER_W)

    @pytest.mark.parametrize("hz", [5.0, 10.0, 30.0])
    def test_keypoint_identity_switch_is_rejected(self, hz: float) -> None:
        """A wrist snapping to the face for one sample (conf 0.9, 1.8 sw away) and back. Capping it at
        1 sw per sample would still add 1/8 sw per sample (1.25 sw/s raw at 10 Hz) twice; gating drops it."""
        tracker = PostureTracker()
        states = []
        for i in range(int(3 * hz)):
            xy, conf = pose()
            if i == int(hz):
                xy[9] = xy[0]
            states.append(tracker.update(xy, conf, i / hz))
        assert all(s.motion == pytest.approx(0.0) for s in states[1:])

    @pytest.mark.parametrize("wrist_gap_sw", [0.4, 1.0])
    def test_left_right_swap_is_not_motion(self, wrist_gap_sw: float) -> None:
        """YOLO's most common identity switch: the two wrists trade labels for one sample and back. Each
        wrist 'moves' by the wrist gap (<= 1 sw, so the jump gate alone keeps it: 0.51 / 1.27 sw/s before)."""
        tracker = PostureTracker()
        half = wrist_gap_sw * SHOULDER_W / 2
        states = []
        for i in range(30):
            xy, conf = pose(wrists=((600.0 + half, 600.0), (600.0 - half, 600.0)))
            if i == 10:
                xy[[9, 10]] = xy[[10, 9]]
            states.append(tracker.update(xy, conf, i / 10))
        assert all(s.motion == pytest.approx(0.0) for s in states[1:])

    def test_arms_leaving_the_frame_do_not_latch_the_transition(self) -> None:
        """Slumping at a laptop: the arms drop out of frame during the transition, leaving the shoulders.
        Both shoulders are a measurable torso, so the held slump decays to still instead of holding the
        transition's speed for the whole hold."""
        tracker = PostureTracker()
        for i in range(10):  # slumping down: everything moves 0.5 sw/s
            tracker.update(*pose(shift=(0.0, 10.0 * i)), i / 10)
        states = [tracker.update(*pose(shift=(0.0, 100.0), hidden=(7, 8, 9, 10, 11, 12)), i / 10)
                  for i in range(10, 60)]
        assert states[-1].motion == pytest.approx(0.0, abs=1e-6)

    def test_low_confidence_frames_hold_the_last_value(self) -> None:
        tracker = PostureTracker()
        for i in range(20):
            moving = tracker.update(*pose(shift=(10.0 * i, 0.0)), i / 10).motion
        assert moving == pytest.approx(0.5)
        for i in range(20, 30):  # body conf 0.55: fine for slump geometry, below the motion floor
            xy, conf = pose(shift=(10.0 * i + 300.0 * (i % 2), 0.0))
            conf[BODY] = 0.55
            state = tracker.update(xy, conf, i / 10)
            assert state.motion == moving
            assert not math.isnan(state.head_drop)
        for i in range(30, 40):  # only the left shoulder and elbow: too few points, and no torso
            state = tracker.update(*pose(shift=(50.0 * i, 0.0), hidden=(6, 8, 9, 10, 11, 12)), i / 10)
            assert state.motion == moving
        state = tracker.update(*pose(shift=(2000.0, 0.0)), 4.0)
        state = tracker.update(*pose(shift=(2000.0, 0.0)), 4.1)
        assert state.motion == pytest.approx(0.7 * moving)  # measurable again: still, so it decays

    def test_nan_only_before_the_first_valid_pair(self) -> None:
        tracker = PostureTracker()

        def sample(hidden: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
            xy, conf = pose(hidden=hidden)
            conf[[5, 6]] = 0.55  # shoulders give the width but sit below the motion floor
            return xy, conf

        states = [tracker.update(*sample((8, 10, 11, 12)), i / 10) for i in range(5)]
        assert tracker.shoulder_width == pytest.approx(SHOULDER_W)
        assert all(math.isnan(s.motion) for s in states)  # left elbow + wrist only: nothing measured yet
        assert math.isnan(tracker.update(*sample((8, 10, 12)), 0.5).motion)  # 3 now, but only 2 in both
        assert tracker.update(*sample((8, 10, 12)), 0.6).motion == pytest.approx(0.0)
        for i in range(7, 20):
            assert tracker.update(np.zeros((17, 2)), np.zeros(17), i / 10).motion == pytest.approx(0.0)

    def test_motion_floor_is_never_looser_than_the_tracker_floor(self) -> None:
        assert PostureTracker(min_conf=0.7, motion_min_conf=0.6).motion_min_conf == 0.7


class TestStateAndValidation:
    def test_low_confidence_keypoints_are_ignored(self) -> None:
        tracker = PostureTracker()
        xy, conf = pose()
        conf[0] = 0.49
        state = tracker.update(xy, conf, 0.0)
        assert math.isnan(state.head_drop)
        conf[0] = 0.5
        assert tracker.update(xy, conf, 0.1).head_drop == pytest.approx(0.7)

    def test_nothing_seen_yet(self) -> None:
        state = PostureTracker().update(np.zeros((17, 2)), np.zeros(17), 0.0)
        assert isinstance(state, PostureState)
        assert (state.slump, state.head_in_hands, state.calibrated) == (False, False, False)
        assert math.isnan(state.motion)
        assert math.isnan(state.head_drop)

    def test_keeps_only_a_private_copy_of_previous_keypoints(self) -> None:
        tracker = PostureTracker()
        xy, conf = pose()
        tracker.update(xy, conf, 0.0)
        xy += 1000.0
        assert tracker.update(*pose(), 0.1).motion == pytest.approx(0.0)
        arrays = [getattr(tracker, s) for s in PostureTracker.__slots__
                  if isinstance(getattr(tracker, s), np.ndarray)]
        assert [a.shape for a in arrays] == [(17, 2)]

    @pytest.mark.parametrize(("xy_shape", "conf_shape"), [((17, 3), (17,)), ((16, 2), (16,)), ((17, 2), (17, 1))])
    def test_rejects_bad_shapes(self, xy_shape: tuple[int, ...], conf_shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            PostureTracker().update(np.zeros(xy_shape), np.ones(conf_shape), 0.0)


class TestTimingEdges:
    def test_calibration_window_is_half_open(self) -> None:
        tracker = PostureTracker()
        for i, value in enumerate([0.5] * 5 + [0.9] * 5):
            assert not tracker.update(*pose(head_drop=value), 2.0 * i).calibrated
        assert tracker.update(*pose(head_drop=0.9), 20.0).calibrated
        assert tracker.baseline == pytest.approx(0.7)

    def test_head_in_hands_hold_tolerates_float_rounding(self) -> None:
        tracker = PostureTracker()
        tracker.update(*pose(), 1.2)
        assert not tracker.update(*pose(wrists=FACE_WRISTS, hidden=(0,)), 1.3).head_in_hands
        assert tracker.update(*pose(wrists=FACE_WRISTS, hidden=(0,)), 3.3).head_in_hands

    def test_slump_hold_tolerates_float_rounding(self) -> None:
        tracker = PostureTracker(ema_alpha=1.0)
        run(tracker, -25.0, -4.9, head_drop=0.7)
        assert tracker.baseline == pytest.approx(0.7)
        assert not tracker.update(*pose(head_drop=0.1), 1.1).slump
        assert not tracker.update(*pose(head_drop=0.1), 2.6).slump
        assert tracker.update(*pose(head_drop=0.1), 4.1).slump


class TestSlumpVsKeyboardGaze:
    """Looking down pitches the head (nose drops, ears stay); a slump sinks the whole head."""

    def test_keyboard_gaze_is_not_a_slump(self) -> None:
        tracker = calibrated()
        states = run(tracker, 20.1, 30.1, head_drop=0.45, ear_drop=0.7)  # nose 0.64x, ears ~0.97x
        assert first(states, "slump") is None

    def test_whole_head_sinking_is_a_slump(self) -> None:
        tracker = calibrated()
        states = run(tracker, 20.1, 30.1, head_drop=0.45)  # nose 0.64x, ears 0.65x
        t = first(states, "slump")
        assert t is not None and t >= 23.0

    def test_ears_never_visible_falls_back_to_a_stricter_nose_rule(self) -> None:
        tracker = PostureTracker()
        run(tracker, 0.0, 20.1, hidden=(3, 4))
        assert tracker.ear_baseline is None
        assert first(run(tracker, 20.1, 30.1, head_drop=0.45, hidden=(3, 4)), "slump") is None  # 0.64x
        assert first(run(tracker, 30.1, 40.1, head_drop=0.35, hidden=(3, 4)), "slump") is not None  # 0.5x

    def test_ears_hidden_mid_slump_uses_the_nose_fallback(self) -> None:
        tracker = calibrated()
        assert tracker.ear_baseline is not None
        assert first(run(tracker, 20.1, 30.1, head_drop=0.35, hidden=(3, 4)), "slump") is not None
