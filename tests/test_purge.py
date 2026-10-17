import gc

import numpy as np
import pytest

from valence.governance.purge import PurgeGate


@pytest.fixture
def gate():
    return PurgeGate()


def test_purge_zeroes_and_finalizer_fires(gate):
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    view = frame  # a second name the caller also drops
    gate.purge(frame, "video")
    assert not frame.any(), "buffer must be zeroed in place"
    assert gate.live_raw_buffers() == 1
    del frame, view
    assert gate.live_raw_buffers() == 0
    stats = gate.end_tick()
    assert stats.purged == {"video": 1}
    assert stats.freed == {"video": 1}
    assert stats.violations == 0


def test_leaked_reference_is_a_violation(gate):
    leaked = []
    block = np.ones(1600, dtype=np.float32)
    leaked.append(block)
    gate.purge(block, "audio")
    del block
    stats = gate.end_tick()
    assert stats.violations == 1
    assert stats.live_raw_buffers == 1
    # Reported once, then no longer tracked.
    assert gate.end_tick().violations == 1
    assert not leaked[0].any(), "even a leaked buffer has been zeroed"


def test_view_zeroes_owning_buffer(gate):
    frame = np.full((10, 10, 3), 7, dtype=np.uint8)
    rgb = frame[:, :, ::-1]  # BGR->RGB view, as perception code commonly does
    gate.purge(rgb, "video")
    assert not frame.any()
    del rgb
    assert gate.live_raw_buffers() == 1  # owner `frame` still referenced
    del frame
    assert gate.live_raw_buffers() == 0


def test_readonly_buffer_counted_unzeroable(gate):
    arr = np.frombuffer(b"\x01\x02\x03\x04", dtype=np.uint8)
    gate.purge(arr, "audio")
    assert gate.stats().unzeroable == {"audio": 1}
    del arr
    gc.collect()


def test_cycle_is_collected_before_violation(gate):
    class Holder:
        pass

    h = Holder()
    h.self_ref = h
    h.frame = np.ones((4, 4), dtype=np.uint8)
    gate.purge(h.frame, "video")
    del h
    stats = gate.end_tick()
    assert stats.violations == 0


class FakeResults:
    """Stands in for ultralytics Results, which keeps a reference to the frame as orig_img."""

    def __init__(self, frame):
        self.orig_img = frame
        self.boxes = np.array([[0.1, 0.1, 0.5, 0.9]])


def test_holding_detector_results_leaks_the_frame(gate):
    frame = np.ones((8, 8, 3), dtype=np.uint8)
    results = FakeResults(frame)
    boxes = results.boxes.copy()
    gate.purge(frame, "video")
    del frame
    assert gate.end_tick().violations == 1  # results.orig_img kept it alive
    del results

    frame = np.ones((8, 8, 3), dtype=np.uint8)
    results = FakeResults(frame)
    boxes = results.boxes.copy()
    del results  # correct order: copy outputs, drop Results, then purge
    gate.purge(frame, "video")
    del frame
    assert gate.end_tick().violations == 1  # unchanged: no new violation
    assert boxes.shape == (1, 4)


def test_transcripts_and_kind_validation(gate):
    gate.note_transcript_purged(3)
    assert gate.stats().transcripts_purged == 3
    with pytest.raises(ValueError):
        gate.purge(np.zeros(3), "text")
    with pytest.raises(TypeError):
        gate.purge(b"raw", "audio")
