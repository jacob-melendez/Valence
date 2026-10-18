import json
import math

import numpy as np
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


def test_override_resets_when_label_changes():
    from valence.models.predictor import PersonSmoother, PredictionOutput

    states = ("Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy")
    sm = PersonSmoother(states, override_block_s=0)

    def out(label, override):
        probs = {s: (0.85 if s == label else 0.05) for s in states}
        return PredictionOutput(label=label, confidence=0.85, probabilities=probs, face_only_label=None,
                                face_only_confidence=None, layers=[], raw_override=override)

    for _ in range(4):
        r = sm.update(out("Frustrated_Masked", True), 100)
    assert r.mask_override
    for _ in range(4):  # face hidden (None) and the label moves on: the override must not follow it
        r = sm.update(out("Collaborative_Happy", None), 100)
    assert r.label == "Collaborative_Happy" and not r.mask_override


def test_background_tracks_are_not_participants():
    agg = Aggregator(Settings(), "s")
    for i in range(50):
        desk = PersonObs(track_id=1, bbox=(0.2, 0.05, 0.7, 0.98), motion=0.03, face_visible=True, face_valence=0.0)
        background = PersonObs(track_id=3, bbox=(0.8, 0.3, 0.9, 0.62))  # a third of the height, always there
        agg.push(_video(i * 0.1, [desk, background], i + 1))
    room, *people = agg.flush(wall=1_000_005, t_mono=5, tod_h=12.0)
    assert [p.person_id for p in people] == [1]
    assert room.n_people == 1 and people[0].n_people == 1


def test_absent_people_are_evicted_and_reset_clears_history(runtime):
    runtime.use_scenario(render("full_demo"), "full_demo")
    for _ in range(20):
        runtime.step()
    assert runtime.window.people() == [1] and runtime.history
    from valence.contracts import TickRecord

    stale = runtime.window.snapshot(1)[-1]
    runtime.window.append(TickRecord(**{**stale.to_dict(), "person_id": 7, "ts": stale.ts - 3600,
                                        "face_valence": float("nan")}))
    runtime._evict(stale.ts)
    assert 7 not in runtime.window.people()
    runtime.reset_people()
    assert not runtime.history


def test_frame_counter_survives_capture_restart_and_source_change(runtime):
    runtime.use_live()
    p = PersonObs(track_id=1, bbox=(0.2, 0.05, 0.7, 0.98))
    for i in range(1, 6):
        runtime.push_obs(_video(i * 0.1, [p], i))  # first capture: counter 1..5
    for i in range(1, 4):
        runtime.push_obs(_video(1 + i * 0.1, [p], i))  # capture restarted (pause/start): counter 1..3
    assert runtime.aggregator.frames_decoded == 8
    runtime.use_recording  # attribute exists; switching source re-aligns the pair
    runtime.use_live()
    assert runtime.aggregator.frames_decoded == 0


def test_reset_clears_narration_cache(runtime):
    class Narrator:
        cleared = 0

        def clear_cache(self):
            self.cleared += 1

    runtime.narrator = Narrator()
    runtime.reset_people()
    assert runtime.narrator.cleared == 1


def test_replay_offline_uses_the_replay_clock(runtime):
    """Break clocks start when the replayed person appears, not when the capture was recorded."""
    from pathlib import Path

    from valence.runtime.recording import replay_offline

    path = Path(__file__).resolve().parents[1] / "data" / "real" / "capture_typing.jsonl"
    if not path.exists():
        pytest.skip("capture not present")
    runtime.use_live()
    preds = replay_offline(runtime, path)
    assert preds
    hours = [m["hours_since_break"] for m in runtime.msgs
             if m["type"] == "state_vector" and m["person_id"] != 0 and m["hours_since_break"] is not None]
    assert hours and max(hours) < 1.0


def test_narrator_status_needs_two_failures_in_a_row():
    from valence.explain.narrator_client import NarratorClient

    c = NarratorClient("http://127.0.0.1:9", lambda message: None)
    c._mark(True)
    c._mark(False)  # one slow narration past the deadline
    assert c.healthy
    c._mark(False)
    assert not c.healthy
    c._mark(True)
    assert c.healthy


def test_room_baseline_capture_and_relative_gating():
    """An arrival capture sets the venue baseline; records carry it and near-field gating uses it."""
    agg = Aggregator(Settings(room_baseline_s=10), "t")
    assert agg.baseline_status()["mode"] == "auto"
    agg.start_baseline(10)
    for i in range(12):
        agg.push(AudioObs(t=float(i), wall=1000.0 + i, block_seq=i, db_spl=70.0 + (i % 3) - 1, rms_power=1.0))
        agg.push(VideoObs(t=float(i), wall=1000.0 + i, frame_seq=i, lux=300.0, lux_quality=1.0))
    snap = agg.baseline_status()
    assert snap["mode"] == "captured" and snap["db_median"] == 70.0 and snap["lux_median"] == 300.0
    agg.push(AudioObs(t=13.0, wall=1013.0, block_seq=13, db_spl=73.0, rms_power=1.0, laughter_onsets=1))  # < 70 + 6
    agg.push(AudioObs(t=14.0, wall=1014.0, block_seq=14, db_spl=78.0, rms_power=1.0, laughter_onsets=1))
    room = agg.flush(1015.0, 15.0, 12.0)[0]
    assert room.room_db_base == 70.0 and room.room_lux_base == 300.0
    assert room.laughter_count == 1 and agg.gated_out == 1


def test_room_relative_features_follow_the_baseline():
    from valence.features.compute import FEATURE_INDEX, compute

    n = 30
    person = {f: np.full(n, 0.0) for f in ("motion_energy", "posture_slump", "head_in_hands_frac", "clenched_fists_count",
                                          "social_withdrawal_flag", "n_people", "hours_since_break", "tod_h",
                                          "face_valence", "face_visible")}
    room = {"ambient_db": np.full(n, 72.0), "light_lux": np.full(n, 150.0), "laughter_count": np.zeros(n),
            "compliments_count": np.zeros(n), "room_db_base": np.full(n, 66.0), "room_lux_base": np.full(n, 300.0)}
    x = compute(person, room)
    assert x[FEATURE_INDEX["db_room_now"]] == pytest.approx(6.0)
    assert x[FEATURE_INDEX["lux_room_mean_5m"]] == pytest.approx(-1.0)  # half the baseline light
    room["room_db_base"] = np.full(n, np.nan)  # no baseline yet: relative features are missing, not zero
    assert math.isnan(compute(person, room)[FEATURE_INDEX["db_room_now"]])
