from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from valence.perception.ids import DisplayIdMapper

REPO = Path(__file__).resolve().parents[2]


def box(cx: float, cy: float, w: float = 0.1, h: float = 0.2) -> tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class TestAllocation:
    def test_first_appearance_gets_sequential_ids(self) -> None:
        m = DisplayIdMapper()
        assert m.update([(11, box(0.7, 0.5)), (10, box(0.3, 0.5))], 0.0) == {10: 1, 11: 2}
        assert m.update([(10, box(0.3, 0.5)), (11, box(0.7, 0.5)), (14, box(0.5, 0.5))], 0.1) == {
            10: 1, 11: 2, 14: 3}

    def test_active_ids_are_stable_even_when_people_move(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.1, 0.5)), (2, box(0.9, 0.5))], 0.0)
        assert m.update([(1, box(0.9, 0.5)), (2, box(0.1, 0.5))], 0.1) == {1: 1, 2: 2}

    def test_smallest_free_after_expiry(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.1, 0.5)), (2, box(0.5, 0.5)), (3, box(0.9, 0.5))], 0.0)
        m.update([(1, box(0.1, 0.5)), (3, box(0.9, 0.5))], 1.0)
        m.update([(1, box(0.1, 0.5)), (3, box(0.9, 0.5))], 4.0)
        assert m.update([(1, box(0.1, 0.5)), (3, box(0.9, 0.5)), (9, box(0.5, 0.1))], 4.1)[9] == 2

    def test_reserved_ids_are_skipped(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.1, 0.5)), (2, box(0.5, 0.5)), (3, box(0.9, 0.5))], 0.0)
        m.update([(1, box(0.1, 0.5)), (3, box(0.9, 0.5))], 1.0)
        assert m.reserved() == frozenset({2})
        mapping = m.update([(1, box(0.1, 0.5)), (3, box(0.9, 0.5)), (9, box(0.2, 0.95, h=0.05))], 1.5)
        assert mapping[9] == 4

    def test_duplicate_tracker_ids_are_ignored(self) -> None:
        m = DisplayIdMapper()
        assert m.update([(5, box(0.2, 0.5)), (5, box(0.8, 0.5))], 0.0) == {5: 1}


class TestReuse:
    def test_reuse_inside_window_and_radius(self) -> None:
        m = DisplayIdMapper()
        m.update([(10, box(0.3, 0.5)), (11, box(0.8, 0.5))], 0.0)
        m.update([(11, box(0.8, 0.5))], 1.0)
        assert m.update([(11, box(0.8, 0.5)), (12, box(0.5, 0.5))], 3.5) == {11: 2, 12: 1}
        assert m.reserved() == frozenset()

    def test_no_reuse_outside_radius(self) -> None:
        m = DisplayIdMapper()
        m.update([(10, box(0.3, 0.5)), (11, box(0.8, 0.5))], 0.0)
        m.update([(11, box(0.8, 0.5))], 1.0)
        assert m.update([(11, box(0.8, 0.5)), (12, box(0.56, 0.5))], 1.5)[12] == 3

    def test_window_boundary_is_exclusive(self) -> None:
        def run(t_back: float) -> dict[int, int]:
            m = DisplayIdMapper()
            m.update([(10, box(0.2, 0.5)), (11, box(0.5, 0.5))], 0.0)
            m.update([(11, box(0.5, 0.5))], 1.0)
            return m.update([(11, box(0.5, 0.5)), (12, box(0.9, 0.5)), (13, box(0.2, 0.5))], t_back)

        assert run(3.75) == {11: 2, 12: 3, 13: 1}
        assert run(4.0) == {11: 2, 12: 1, 13: 3}

    def test_distance_uses_frame_width_units(self) -> None:
        def reuses(dx: float, dy: float, aspect: float = 9 / 16) -> bool:
            m = DisplayIdMapper(aspect=aspect)
            m.update([(1, box(0.5, 0.3)), (2, box(0.05, 0.9))], 0.0)
            m.update([(2, box(0.05, 0.9))], 0.5)
            return m.update([(2, box(0.05, 0.9)), (3, box(0.5 + dx, 0.3 + dy))], 1.0)[3] == 1

        assert reuses(0.0, 0.4)
        assert not reuses(0.4, 0.0)
        assert not reuses(0.0, 0.4, aspect=1.0)
        assert reuses(0.24, 0.0)
        assert not reuses(0.26, 0.0)
        assert reuses(0.15, -0.25)

    def test_two_people_swapping_reclaim_by_position(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0)
        m.update([], 1.0)
        assert m.update([(20, box(0.75, 0.5)), (21, box(0.25, 0.5))], 1.5) == {20: 2, 21: 1}

    def test_matching_is_global_nearest_first(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.40, 0.5)), (2, box(0.60, 0.5))], 0.0)
        m.update([], 1.0)
        assert m.update([(30, box(0.52, 0.5)), (31, box(0.62, 0.5))], 1.2) == {30: 1, 31: 2}

    def test_revived_tracker_reclaims_own_id_even_if_far(self) -> None:
        m = DisplayIdMapper()
        m.update([(7, box(0.1, 0.5)), (8, box(0.5, 0.5))], 0.0)
        m.update([(8, box(0.5, 0.5))], 1.0)
        assert m.update([(8, box(0.5, 0.5)), (7, box(0.95, 0.5))], 2.0) == {7: 1, 8: 2}

    def test_reused_id_is_not_double_assigned(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.5, 0.5))], 0.0)
        m.update([], 1.0)
        mapping = m.update([(2, box(0.52, 0.5)), (3, box(0.48, 0.5))], 1.5)
        assert sorted(mapping.values()) == [1, 2]


class TestLifecycle:
    def test_pop_released_reports_expired_ids_once(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0)
        m.update([(2, box(0.8, 0.5))], 1.0)
        m.update([(2, box(0.8, 0.5))], 3.9)
        assert m.pop_released() == []
        m.update([(2, box(0.8, 0.5))], 4.0)
        assert m.pop_released() == [1]
        assert m.pop_released() == []
        assert m.reserved() == frozenset()

    def test_reused_id_is_never_released(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5))], 0.0)
        m.update([], 1.0)
        m.update([(2, box(0.22, 0.5))], 2.0)
        m.update([(2, box(0.22, 0.5))], 10.0)
        assert m.pop_released() == []

    def test_active_is_a_copy_and_reset_forgets(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0)
        snapshot = m.active()
        snapshot[1] = 99
        assert m.active() == {1: 1, 2: 2}
        m.update([(2, box(0.8, 0.5))], 1.0)
        m.reset()
        assert m.active() == {} and m.reserved() == frozenset()
        assert m.update([(5, box(0.8, 0.5))], 2.0) == {5: 1}

    def test_state_holds_only_ids_centres_and_times(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0)
        m.update([(2, box(0.8, 0.5))], 1.0)

        def leaves(obj: object) -> list[object]:
            if isinstance(obj, dict):
                return [x for k, v in obj.items() for x in leaves(k) + leaves(v)]
            if isinstance(obj, (list, tuple, set, frozenset)):
                return [x for v in obj for x in leaves(v)]
            if hasattr(obj, "__slots__"):
                return [x for s in obj.__slots__ for x in leaves(getattr(obj, s))]
            return [obj]

        assert all(type(x) in (int, float) for x in leaves(m))

    @pytest.mark.parametrize("kwargs", [{"reuse_window_s": -1}, {"reuse_dist": -0.1}, {"aspect": 0}])
    def test_rejects_bad_config(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            DisplayIdMapper(**kwargs)


def test_bytetrack_config_matches_ultralytics_defaults_except_buffer() -> None:
    import ultralytics

    default_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    default = yaml.safe_load(default_path.read_text())
    ours = yaml.safe_load((REPO / "valence" / "perception" / "bytetrack_valence.yaml").read_text())
    assert set(ours) == set(default)
    assert ours["track_buffer"] == 60
    assert {k: v for k, v in ours.items() if k != "track_buffer"} == {
        k: v for k, v in default.items() if k != "track_buffer"}


class TestRobustness:
    def test_window_expiry_tolerates_float_rounding(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0)
        m.update([(2, box(0.8, 0.5))], 1.1)
        assert m.update([(2, box(0.8, 0.5)), (3, box(0.5, 0.95))], 4.1)[3] == 1
        assert m.pop_released() == [1]

    def test_reset_reports_every_known_id_as_released(self) -> None:
        m = DisplayIdMapper()
        m.update([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0)
        m.update([(2, box(0.8, 0.5))], 1.0)
        m.reset()
        assert m.pop_released() == [1, 2]
        assert m.pop_released() == []

    def test_numpy_boxes_stored_as_plain_floats(self) -> None:
        import numpy as np

        m = DisplayIdMapper()
        m.update([(np.int64(4), np.array(box(0.2, 0.5)))], 0.0)
        m.update([], 1.0)
        lost = next(iter(m._lost.values()))
        assert type(lost.tracker_id) is int
        assert all(type(v) is float for v in lost.centre)
