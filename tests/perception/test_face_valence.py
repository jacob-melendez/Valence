from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from valence.perception.face_valence import (
    FaceValenceCalibrator,
    assign_faces,
    blendshape_scores,
    nose_tip_px,
    valence_raw,
)

BS = {
    "mouthSmileLeft": 0.8, "mouthSmileRight": 0.6,
    "mouthFrownLeft": 0.2, "mouthFrownRight": 0.4,
    "browDownLeft": 0.1, "browDownRight": 0.3,
    "noseSneerLeft": 0.5, "noseSneerRight": 0.1,
    "mouthPressLeft": 0.2, "mouthPressRight": 0.0,
}


def test_valence_formula() -> None:
    expected = 0.7 - 0.7 * 0.3 - 0.5 * 0.2 - 0.3 * 0.3 - 0.2 * 0.1
    assert valence_raw(BS) == pytest.approx(expected)


def test_left_right_averaging() -> None:
    assert valence_raw({"mouthSmileLeft": 1.0, "mouthSmileRight": 0.0}) == pytest.approx(0.5)
    assert valence_raw({"browDownLeft": 0.4, "browDownRight": 0.8}) == pytest.approx(-0.5 * 0.6)


def test_missing_side_uses_other_and_missing_category_is_zero() -> None:
    assert valence_raw({"mouthSmileLeft": 0.6}) == pytest.approx(0.6)
    assert valence_raw({"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0}) == pytest.approx(-0.7)


def test_cheek_squint_and_other_shapes_ignored() -> None:
    noisy = {**BS, "cheekSquintLeft": 1.0, "cheekSquintRight": 1.0, "eyeSquintLeft": 0.9, "jawOpen": 0.7}
    assert valence_raw(noisy) == pytest.approx(valence_raw(BS))
    assert math.isnan(valence_raw({"cheekSquintLeft": 1.0, "cheekSquintRight": 1.0}))


def test_no_face_is_nan() -> None:
    assert math.isnan(valence_raw({}))


def test_blendshape_scores_keeps_only_needed() -> None:
    cats = [SimpleNamespace(category_name=n, score=s) for n, s in
            [("mouthSmileLeft", 0.5), ("cheekSquintLeft", 0.9), ("_neutral", 0.1), ("browDownRight", 0.2)]]
    assert blendshape_scores(cats) == {"mouthSmileLeft": 0.5, "browDownRight": 0.2}


def test_nose_tip_px_reads_only_index_1() -> None:
    lms = [SimpleNamespace(x=0.0, y=0.0), SimpleNamespace(x=0.25, y=0.5)]
    assert nose_tip_px(lms, 640, 360) == (160.0, 180.0)


def test_baseline_freezes_after_20s_of_visible_face() -> None:
    cal = FaceValenceCalibrator()
    t = 0.0
    while t < 20.0:
        assert cal.update(0.2, t) == pytest.approx(0.0)
        assert not cal.calibrated
        t += 0.5
    out = cal.update(0.9, 20.0)
    assert cal.calibrated
    assert cal.baseline == pytest.approx(0.2)
    assert out == pytest.approx(math.tanh(2.5 * 0.7))
    for k in range(100):
        cal.update(0.9, 20.5 + 0.5 * k)
    assert cal.baseline == pytest.approx(0.2)


def test_running_median_while_calibrating() -> None:
    cal = FaceValenceCalibrator()
    assert cal.baseline == 0.0
    cal.update(0.1, 0.0)
    cal.update(0.5, 0.1)
    out = cal.update(0.3, 0.2)
    assert cal.baseline == pytest.approx(0.3)
    assert out == pytest.approx(0.0)
    cal.update(0.9, 0.3)
    assert cal.baseline == pytest.approx(0.4)


def test_tanh_and_clip() -> None:
    cal = FaceValenceCalibrator(calibration_s=0.0)
    assert cal.update(5.0, 0.0) == pytest.approx(1.0)
    assert cal.baseline == 0.0
    assert cal.update(-0.2, 1.0) == pytest.approx(math.tanh(-0.5))
    assert -1.0 <= cal.update(-50.0, 2.0) <= 1.0


def test_nan_returns_nan_and_is_not_added() -> None:
    cal = FaceValenceCalibrator()
    cal.update(0.4, 0.0)
    assert math.isnan(cal.update(math.nan, 0.5))
    assert cal.baseline == pytest.approx(0.4)
    assert math.isnan(cal.update(math.inf, 0.6))


def test_hidden_face_does_not_accrue_calibration_time() -> None:
    cal = FaceValenceCalibrator()
    for k in range(21):
        cal.update(0.1, 0.5 * k)
    for k in range(60):
        cal.update(math.nan, 10.5 + 0.5 * k)
    cal.update(0.1, 40.0)
    assert not cal.calibrated
    assert cal.visible_s == pytest.approx(10.0)
    t = 40.5
    while not cal.calibrated:
        cal.update(0.1, t)
        t += 0.5
    assert t == pytest.approx(50.5)


def test_large_gap_capped() -> None:
    cal = FaceValenceCalibrator(max_gap_s=1.0)
    cal.update(0.0, 0.0)
    cal.update(0.0, 100.0)
    assert cal.visible_s == pytest.approx(1.0)


def test_assign_faces_optimal_not_greedy() -> None:
    faces = [(130.0, 100.0), (100.0, 100.0)]
    persons = {1: (120.0, 100.0, 0.9), 2: (160.0, 100.0, 0.9)}
    assert assign_faces(faces, persons, frame_w=640) == [2, 1]


def test_assign_faces_distance_gate() -> None:
    faces = [(100.0, 100.0)]
    gate = 0.15 * 640
    assert assign_faces(faces, {3: (100.0 + gate - 1, 100.0, 0.9)}, 640) == [3]
    assert assign_faces(faces, {3: (100.0 + gate + 1, 100.0, 0.9)}, 640) == [None]
    assert assign_faces(faces, {3: (150.0, 100.0, 0.9)}, 640, max_dist_frac=0.05) == [None]


def test_assign_faces_low_confidence_ignored() -> None:
    faces = [(100.0, 100.0)]
    assert assign_faces(faces, {4: (101.0, 100.0, 0.2)}, 640) == [None]
    assert assign_faces(faces, {4: (101.0, 100.0, 0.2), 5: (130.0, 100.0, 0.5)}, 640) == [5]
    assert assign_faces(faces, {4: (101.0, 100.0, 0.2)}, 640, min_conf=0.1) == [4]


def test_assign_faces_more_faces_than_people() -> None:
    faces = [(0.0, 0.0), (50.0, 0.0)]
    assert assign_faces(faces, {7: (40.0, 0.0, 0.9)}, frame_w=400) == [None, 7]


def test_assign_faces_maximises_valid_matches() -> None:
    faces = [(0.0, 0.0), (55.0, 0.0)]
    persons = {1: (50.0, 0.0, 0.9), 2: (105.0, 0.0, 0.9)}
    assert assign_faces(faces, persons, frame_w=400) == [1, 2]


def test_assign_faces_empty_and_nan() -> None:
    assert assign_faces([], {1: (0.0, 0.0, 1.0)}, 640) == []
    assert assign_faces([(1.0, 1.0)], {}, 640) == [None]
    assert assign_faces([(math.nan, 1.0)], {1: (1.0, 1.0, 1.0)}, 640) == [None]
    assert assign_faces([(1.0, 1.0)], {1: (math.nan, 1.0, 1.0)}, 640) == [None]


def test_non_finite_time_does_not_poison_calibration() -> None:
    cal = FaceValenceCalibrator()
    cal.update(0.1, 0.0)
    cal.update(0.1, math.nan)
    cal.update(0.1, 0.5)
    assert math.isfinite(cal.visible_s)
    assert cal.visible_s <= 1.0
    assert not cal.calibrated
