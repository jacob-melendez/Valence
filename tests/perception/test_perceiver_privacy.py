"""Privacy properties of the real perceivers (need the downloaded models; skipped otherwise).

- every raw buffer handed to ``process`` is zero-filled by the purge gate before it returns;
- no reference to a frame or audio block survives ``process`` (weakref is dead after ``del``);
- ``obscure_face`` means MediaPipe never sees a face (face_visible is False for everyone);
- only scalars, booleans and normalized boxes come out.
"""

from __future__ import annotations

import gc
import math
import weakref
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("valence.contracts")
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
needs_video = pytest.mark.skipif(not (MODELS / "yolo26n-pose.pt").exists(), reason="models not downloaded")
needs_audio = pytest.mark.skipif(not (MODELS / "yamnet.tflite").exists(), reason="models not downloaded")


def _sample_frame() -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    assets = Path(cv2.__file__).parents[1] / "ultralytics" / "assets"
    img = cv2.imread(str(assets / "zidane.jpg"))
    if img is None:
        pytest.skip("ultralytics sample image missing")
    return cv2.resize(img, (1280, 720))


@pytest.fixture(scope="module")
def frame_template() -> np.ndarray:
    return _sample_frame()


def _run_frames(perceiver, template: np.ndarray, n: int):
    obs = None
    for i in range(n):
        frame = template.copy()
        ref = weakref.ref(frame)
        obs = perceiver.process(frame, 1000.0 + 0.1 * i, 0.0, i)
        assert not frame.any(), "frame was not purged"
        del frame
        gc.collect()
        assert ref() is None, "a reference to the frame survived process()"
    return obs


@needs_video
def test_video_purges_and_retains_no_frame(frame_template):
    from valence.perception.video import VideoPerceiver

    vp = VideoPerceiver()
    try:
        obs = _run_frames(vp, frame_template, 12)
    finally:
        vp.close()
    assert obs.persons, "expected people in the sample image"
    for p in obs.persons:
        assert all(0.0 <= v <= 1.0 for v in p.bbox)
        for v in (p.track_id, p.fist_onsets, p.fist_active, p.slump, p.head_in_hands, p.motion,
                  p.face_visible, p.face_valence):
            assert isinstance(v, (int, float, bool)), type(v)
    assert any(p.face_visible for p in obs.persons), "head-crop face path found no face"
    for name, value in vars(vp).items():
        assert not (isinstance(value, np.ndarray) and value.ndim >= 2 and value.size > 1000), name


@needs_video
def test_obscure_face_hides_every_face(frame_template):
    from valence.perception.video import VideoPerceiver

    vp = VideoPerceiver(obscure_face=True)
    try:
        obs = _run_frames(vp, frame_template, 8)
    finally:
        vp.close()
    assert obs.persons
    assert not any(p.face_visible for p in obs.persons)
    assert all(math.isnan(p.face_valence) for p in obs.persons)


@needs_audio
def test_audio_purges_block_and_reports_scalars():
    from valence.perception.audio import AudioPerceiver

    ap = AudioPerceiver(spl_offset_db=94.0)
    try:
        sr = 16000
        tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        for i in range(3):
            block = tone.copy()
            ref = weakref.ref(block)
            obs = ap.process(block, 10.0 + i, 0.0, i)
            assert not block.any(), "audio block was not purged"
            del block
            gc.collect()
            assert ref() is None
    finally:
        ap.close()
    # 0.1 amplitude sine: mean power 0.005 -> -23 dBFS -> ~71 dB SPL with a +94 offset
    assert obs.db_spl == pytest.approx(10 * math.log10(0.005) + 94.0, abs=1.0)
    assert obs.compliment_hits == 0 and obs.laughter_onsets == 0
    assert 0.0 <= obs.speech_prob <= 1.0
