from __future__ import annotations

import numpy as np
import pytest

from valence.perception.regions import black_out, head_box_px, norm_bbox

FW, FH = 1280, 720


def kpts(head: dict[int, tuple[float, float]], shoulders: tuple[tuple[float, float], tuple[float, float]] | None = None,
         conf: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((17, 2))
    c = np.zeros(17)
    for i, p in head.items():
        xy[i], c[i] = p, conf
    if shoulders is not None:
        xy[5], xy[6] = shoulders
        c[5] = c[6] = conf
    return xy, c


FACE = {0: (600.0, 260.0), 1: (620.0, 245.0), 2: (580.0, 245.0), 3: (640.0, 255.0), 4: (560.0, 255.0)}


class TestHeadBox:
    def test_box_from_head_spread(self) -> None:
        box = head_box_px(*kpts(FACE), FW, FH)
        assert box is not None
        x1, y1, x2, y2 = box
        assert all(isinstance(v, int) for v in box)
        assert x2 - x1 == pytest.approx(1.8 * 80, abs=2)
        assert y2 - y1 == pytest.approx(1.8 * 80, abs=2)
        assert all(x1 <= x < x2 and y1 <= y < y2 for x, y in FACE.values())

    def test_shoulder_width_sets_minimum_size(self) -> None:
        box = head_box_px(*kpts({0: (600.0, 260.0)}, ((700.0, 400.0), (500.0, 400.0))), FW, FH)
        assert box is not None
        assert box[2] - box[0] == pytest.approx(1.8 * 0.6 * 200, abs=2)
        assert (box[0] + box[2]) / 2 == pytest.approx(600, abs=1)

    def test_larger_of_spread_and_shoulders(self) -> None:
        narrow = ((620.0, 400.0), (580.0, 400.0))
        box = head_box_px(*kpts(FACE, narrow), FW, FH)
        assert box is not None and box[2] - box[0] == pytest.approx(1.8 * 80, abs=2)

    def test_single_point_without_shoulders_uses_fallback(self) -> None:
        box = head_box_px(*kpts({3: (640.0, 300.0)}), FW, FH)
        assert box is not None
        assert box[2] - box[0] == pytest.approx(1.8 * 0.15 * FW, abs=2)

    def test_scale_parameter(self) -> None:
        small = head_box_px(*kpts(FACE), FW, FH, scale=1.0)
        assert small is not None and small[2] - small[0] == pytest.approx(80, abs=2)

    def test_none_when_head_not_visible(self) -> None:
        assert head_box_px(*kpts({}, ((700.0, 400.0), (500.0, 400.0))), FW, FH) is None
        assert head_box_px(*kpts(FACE, conf=0.29), FW, FH) is None
        assert head_box_px(*kpts(FACE, conf=0.3), FW, FH) is not None

    def test_low_confidence_points_do_not_inflate_box(self) -> None:
        xy, c = kpts(FACE)
        xy[3] = (1200.0, 700.0)
        c[3] = 0.1
        box = head_box_px(xy, c, FW, FH)
        assert box is not None and box[2] < 1000

    def test_clipped_to_frame(self) -> None:
        corner = {0: (10.0, 10.0), 3: (40.0, 15.0), 4: (-20.0, 15.0)}
        box = head_box_px(*kpts(corner), FW, FH)
        assert box is not None
        assert box[0] == 0 and box[1] == 0 and box[2] > 0 and box[3] > 0
        edge = {0: (1275.0, 715.0), 3: (1300.0, 720.0), 4: (1250.0, 720.0)}
        box = head_box_px(*kpts(edge), FW, FH)
        assert box is not None and box[2] == FW and box[3] == FH

    def test_entirely_outside_frame_is_none(self) -> None:
        assert head_box_px(*kpts({0: (-500.0, -500.0), 3: (-480.0, -500.0)}), FW, FH) is None

    def test_accepts_xy_with_extra_column(self) -> None:
        xy, c = kpts(FACE)
        assert head_box_px(np.column_stack([xy, c]), c, FW, FH) == head_box_px(xy, c, FW, FH)


class TestBlackOut:
    def test_zeroes_region_in_place(self) -> None:
        frame = np.full((FH, FW, 3), 200, dtype=np.uint8)
        view = frame
        assert black_out(frame, [(100, 50, 300, 150)]) is None
        assert view is frame
        assert not frame[50:150, 100:300].any()
        assert frame.sum() == 200 * 3 * (FH * FW - 100 * 200)

    def test_multiple_boxes_and_none_entries(self) -> None:
        frame = np.ones((100, 100), dtype=np.uint8)
        black_out(frame, [(0, 0, 10, 10), None, (90, 90, 100, 100)])
        assert frame.sum() == 100 * 100 - 200

    def test_out_of_bounds_boxes_are_clipped_not_wrapped(self) -> None:
        frame = np.ones((100, 100), dtype=np.uint8)
        black_out(frame, [(-50, -50, 10, 10)])
        assert frame.sum() == 100 * 100 - 100
        assert frame[-1, -1] == 1
        black_out(frame, [(95, 95, 500, 500), (200, 200, 300, 300), (50, 50, 50, 60)])
        assert frame.sum() == 100 * 100 - 100 - 25

    def test_empty_box_list_is_noop(self) -> None:
        frame = np.ones((10, 10), dtype=np.uint8)
        black_out(frame, [])
        assert frame.all()

    def test_head_box_blacks_out_face(self) -> None:
        frame = np.full((FH, FW, 3), 255, dtype=np.uint8)
        black_out(frame, [head_box_px(*kpts(FACE), FW, FH)])
        assert all(not frame[int(y), int(x)].any() for x, y in FACE.values())
        assert frame[700, 10].all()


class TestNormBbox:
    def test_normalizes(self) -> None:
        assert norm_bbox((128, 72, 640, 360), FW, FH) == pytest.approx((0.1, 0.1, 0.5, 0.5))

    def test_clips_to_unit_range(self) -> None:
        assert norm_bbox((-10, -5, 1400, 800), FW, FH) == (0.0, 0.0, 1.0, 1.0)

    def test_returns_plain_floats(self) -> None:
        out = norm_bbox(np.array([10, 20, 30, 40], dtype=np.int64), 100, 100)
        assert all(type(v) is float for v in out)

    def test_rejects_bad_frame_size(self) -> None:
        with pytest.raises(ValueError):
            norm_bbox((0, 0, 1, 1), 0, 100)


class TestHeadBoxCoverage:
    def test_eyes_only_box_covers_ear_to_ear(self) -> None:
        eyes_nose = {0: FACE[0], 1: FACE[1], 2: FACE[2]}
        box = head_box_px(*kpts(eyes_nose), FW, FH)
        full = head_box_px(*kpts(FACE), FW, FH)
        assert box is not None and full is not None
        assert box[2] - box[0] >= full[2] - full[0] - 2
        assert box[0] <= FACE[4][0] and box[2] > FACE[3][0]
