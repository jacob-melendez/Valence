"""Edge runner sinks: features-only JSONL (NaN -> null, typed lines) and the print readout."""

from __future__ import annotations

import json
import math

import pytest

contracts = pytest.importorskip("valence.contracts")
from valence.edge.run import JsonlSink, PrintSink, EdgeStats, obs_to_dict  # noqa: E402


def _person(**kw):
    base = dict(track_id=1, bbox=(0.1, 0.2, 0.5, 0.9), fist_onsets=1, fist_active=True, slump=False,
                head_in_hands=False, motion=0.12, face_visible=False, face_valence=math.nan)
    base.update(kw)
    return contracts.PersonObs(**base)


def _video():
    return contracts.VideoObs(t=1.0, wall=2.0, frame_seq=7, lux=312.5, lux_quality=0.6, persons=(_person(),))


def _audio():
    return contracts.AudioObs(t=1.0, wall=2.0, block_seq=3, db_spl=55.0, rms_power=1.23456789e-5,
                              speech_prob=0.8, laughter_onsets=1, compliment_hits=2)


def test_obs_to_dict_is_json_safe_and_typed():
    v = obs_to_dict(_video())
    assert v["type"] == "video" and v["persons"][0]["face_valence"] is None
    assert v["persons"][0]["bbox"] == [0.1, 0.2, 0.5, 0.9]
    a = obs_to_dict(_audio())
    assert a["type"] == "audio" and a["rms_power"] == pytest.approx(1.23456789e-5, rel=1e-6)
    json.dumps(v, allow_nan=False)
    json.dumps(a, allow_nan=False)


def test_jsonl_sink_writes_one_text_line_per_obs(tmp_path):
    path = tmp_path / "real" / "scenario.jsonl"
    sink = JsonlSink(path)
    sink(_video())
    sink(_audio())
    sink.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["type"] for ln in lines] == ["video", "audio"]


def test_print_sink_shows_events(capsys):
    sink = PrintSink(lambda: EdgeStats(video_hz=9.9, video_ms=45.0), every=5)
    sink(_video())  # frame_seq 7 is throttled, but a fist onset always prints
    sink(_audio())
    out = capsys.readouterr().out
    assert "P1" in out and "fist" in out
    assert "LAUGH +1" in out and "COMPLIMENT +2" in out
