import json
import math

import pytest

from valence.config import Settings
from valence.contracts import AudioObs, PersonObs, VideoObs
from valence.features.window import RollingWindow
from valence.governance.purge import PurgeGate
from valence.models.predictor import StubPredictor
from valence.runtime.aggregator import Aggregator
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime
from valence.sim.scenarios import render


@pytest.fixture
def runtime(tmp_path):
    bus = EventBus()
    msgs = []
    bus.add_listener(lambda payload: msgs.append(json.loads(payload)))
    rt = Runtime(Settings(), bus, StubPredictor(), gate=PurgeGate(), audit_dir=str(tmp_path))
    rt.msgs = msgs
    yield rt
    rt.close()


def test_replay_scenario_end_to_end(runtime):
    df = render("frustrated_masked")
    runtime.use_scenario(df, "frustrated_masked")
    n_ticks = df["tick"].nunique()
    for _ in range(n_ticks + 2):
        runtime.step()
    types = {m["type"] for m in runtime.msgs}
    assert {"governance", "state_vector", "prediction"} <= types
    preds = [m for m in runtime.msgs if m["type"] == "prediction"]
    assert len(preds) == n_ticks  # one person, one prediction per tick
    assert all(m["source"] == "replay" for m in preds)
    assert all(abs(sum(m["probabilities"].values()) - 1) < 1e-3 for m in preds)
    assert all(m["explanation"]["source"] == "template" and "Person #1" in m["explanation"]["text"] for m in preds)
    assert preds[0]["warming_up"] and not preds[-1]["warming_up"]
    gov = [m for m in runtime.msgs if m["type"] == "governance"]
    assert all(g["violations"] == 0 for g in gov)
    assert runtime.paused  # scenario finished
    ok, n = runtime.audit.verify()
    assert ok and n >= 2


def test_overlays_and_room_only(runtime):
    runtime.use_scenario(render("collaborative_happy"), "collaborative_happy")
    runtime.set_overlay("face_occlusion", True)
    runtime.room_only = True
    for _ in range(5):
        runtime.step()
    preds = [m for m in runtime.msgs if m["type"] == "prediction"]
    assert preds and all(m["person_id"] == 0 for m in preds)
    assert all("face_occlusion" in m["simulated_fields"] for m in preds)
    assert all(m["face_only"]["label"] is None for m in preds)


def test_seed_prefills_room_history(runtime):
    runtime.use_live()
    n = runtime.seed(render("full_demo"), "full_demo", minutes=10, hours_since_break=4.2)
    assert n == 120 and len(runtime.window.room()) == 120
    assert all(r.source == "sim_seed" for r in runtime.window.room())


def test_window_room_slot_survives_person_reset():
    from valence.contracts import TickRecord

    w = RollingWindow(max_ticks=3)
    for t in range(5):
        for pid in (0, 1):
            w.append(TickRecord(tick=t, ts=t, t_mono=t, session_id="s", person_id=pid, display_id="x", n_people=1))
    assert len(w.snapshot(1)) == 3 and w.people() == [1]
    w.reset_people()
    assert w.people() == [] and len(w.room()) == 3


def _video(t, persons, seq):
    return VideoObs(t=t, wall=1_000_000 + t, frame_seq=seq, frames_decoded=seq, lux=300.0, lux_quality=0.8,
                    frame_w=1280, frame_h=720, persons=tuple(persons))


def test_aggregator_tick_and_near_field_gate():
    agg = Aggregator(Settings(), "s")
    for i in range(50):  # 10 Hz for 5 s, one person, two fist onsets, half the frames slumped
        p = PersonObs(track_id=1, bbox=(0.1, 0.1, 0.5, 0.9), fist_onsets=1 if i in (10, 30) else 0,
                      slump=i % 2 == 0, motion=0.05, face_visible=i < 40, face_valence=0.4 if i < 40 else math.nan)
        agg.push(_video(i * 0.1, [p], i + 1))
    for i in range(20):  # quiet room at 50 dB
        agg.push(AudioObs(t=i, wall=1_000_000 + i, block_seq=i, db_spl=50.0, rms_power=1e-5))
    agg.push(AudioObs(t=21, wall=1_000_021, block_seq=21, db_spl=51.0, rms_power=1e-5, laughter_onsets=1))  # far away
    agg.push(AudioObs(t=22, wall=1_000_022, block_seq=22, db_spl=62.0, rms_power=1e-4, compliment_hits=1))  # at the desk
    room, person = agg.flush(wall=1_000_005, t_mono=5, tod_h=15.5)
    assert person.clenched_fists_count == 2
    assert person.posture_slump == pytest.approx(0.5)
    assert person.face_visible == pytest.approx(0.8) and person.face_valence == pytest.approx(0.4)
    assert person.laughter_count == 0 and person.compliments_count == 1 and agg.gated_out == 1
    assert room.person_id == 0 and room.clenched_fists_count == 2
    assert 50 < person.ambient_db < 53


def test_break_clock_and_seed():
    s = Settings(break_absent_min=1)
    agg = Aggregator(s, "s")
    p = PersonObs(track_id=1, bbox=(0.1, 0.1, 0.5, 0.9))
    agg.push(_video(0, [p], 1))
    agg.seed_hours(4.2, wall=1_000_000)
    assert agg.hours_since_break(1, 1_000_000) == pytest.approx(4.2)
    agg.push(VideoObs(t=200, wall=1_000_200, frame_seq=2, frames_decoded=2, persons=(p,)))  # back after 200 s
    assert agg.hours_since_break(1, 1_000_200) == pytest.approx(0.0)


def test_recording_replay_offline(tmp_path, runtime):
    from valence.contracts import obs_to_dict
    from valence.runtime.recording import replay_offline

    path = tmp_path / "capture_typing.jsonl"
    with path.open("w") as f:
        for i in range(600):  # 60 s of video at 10 Hz plus 1 Hz audio
            p = PersonObs(track_id=1, bbox=(0.2, 0.1, 0.6, 0.95), motion=0.07, face_visible=True, face_valence=-0.1)
            f.write(json.dumps(obs_to_dict(_video(i * 0.1, [p], i + 1))) + "\n")
            if i % 10 == 0:
                f.write(json.dumps(obs_to_dict(AudioObs(t=i * 0.1, wall=1_000_000 + i * 0.1, block_seq=i, db_spl=47.0,
                                                        rms_power=1e-5))) + "\n")
    runtime.use_live()
    preds = [m for m in replay_offline(runtime, path) if m["type"] == "prediction"]
    assert 11 <= len(preds) <= 13  # one per 5 s tick
    assert all(m["person_id"] == 1 for m in preds)
