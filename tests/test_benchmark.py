"""valence.eval.benchmark: session parsing, the offline replay, block windowing and the scores it writes.

The session here is synthetic (written by the test, one person, 4 minutes); the served and emotion models are the
committed artifacts. Nothing asserts what the models read: only that the replay runs the real runtime and that the
scores are the arithmetic the module docstring defines.
"""

import json
import os
import time

import pytest

from valence.contracts import AudioObs, PersonObs, VideoObs, obs_to_dict
from valence.eval import benchmark as bm
from valence.eval.benchmark import Block, TickReading
from valence.runtime import engine as eng_mod
from valence.runtime.recording import iter_obs

WALL0, MONO0 = 1_792_850_400.0, 1000.0  # Saturday 2026-10-24 14:00:00 UTC (10:00 in New York)
FOCUSED = {"card": ["Focused"], "map": ["Focused", "Calm"]}
RESTLESS = {"card": ["Frustrated_Masked"], "map": ["Frustrated_Masked", "Tense", "Anxious", "Overwhelmed"]}


def _person(t: float, restless: bool, cues: bool = True) -> PersonObs:
    k = int(round((t - MONO0) * 2))
    face = {"brow_down": 0.02, "eye_open": -0.01, "mouth_frown": 0.0, "smile": 0.01, "head_yaw": 2.0,
            "head_pitch": -4.0} if cues else {}
    return PersonObs(track_id=1, bbox=(0.2, 0.1, 0.7, 0.95), fist_onsets=int(restless and k % 10 == 0),
                     fist_active=restless and k % 10 < 3, motion=0.12 if restless else 0.02, face_visible=True,
                     face_valence=-0.05 if restless else 0.02, **face)


def _marker(mono: float, kind: str, **kw) -> dict:
    return {"type": "marker", "wall": WALL0 + mono - MONO0, "mono": mono, "kind": kind, **kw}


def _block(mono: float, kind: str, bid: str, expect: dict, calm: bool) -> dict:
    return _marker(mono, kind, block_id=bid, label=bid.title(), scored=True, calm=calm, expect=expect)


def session_lines(seconds: float = 240.0, gaps: tuple[tuple[float, float], ...] = (), controls=(),
                  predictions: dict[float, list] | None = None, cues: bool = True) -> list[dict]:
    """One person: focused until 1120, then restless; an abort at 1200; a tap at 1102.5. ``gaps`` drop lines."""
    events = []  # (mono, order, line)
    for k in range(int(seconds * 2)):
        t = MONO0 + 0.05 + k * 0.5
        events.append((t, 0, obs_to_dict(VideoObs(t=t, wall=WALL0 + t - MONO0, frame_seq=k, frames_decoded=k + 1,
                                                  lux=300.0, lux_quality=0.2, frame_w=640, frame_h=480,
                                                  persons=(_person(t, t > 1120.0, cues),)))))
    for j in range(int(seconds)):
        t = MONO0 + 0.3 + j
        events.append((t, 0, obs_to_dict(AudioObs(t=t, wall=WALL0 + t - MONO0, block_seq=j, db_spl=45.0 + (j % 3),
                                                  rms_power=1e-5, speech_prob=0.05))))
    for i in range(1, int(seconds / 5) + 1):
        mono = MONO0 + 5.0 * i
        events.append((mono, 2, {"type": "tick", "wall": WALL0 + mono - MONO0, "mono": mono,
                                 "predictions": (predictions or {}).get(mono, [])}))
    events += [(1000.0, 1, _marker(1000.0, "protocol_start", label="valence-protocol/1")),
               (1000.0, 1, _block(1000.0, "block_start", "focused", FOCUSED, True)),
               (1120.0, 1, _block(1120.0, "block_end", "focused", FOCUSED, True)),
               (1120.0, 1, _block(1120.0, "block_start", "restless", RESTLESS, False)),
               (1200.0, 1, _marker(1200.0, "protocol_abort", reason="test")),
               (1102.5, 1, {"type": "self_report", "t": WALL0 + 102.5, "person_id": 1, "valence": 0.3, "arousal": -0.2,
                            "feature_version": "emotion-v1", "features": {}, "label": "Calm",
                            "wall": WALL0 + 102.5, "mono": 1102.5})]
    events += [(mono, 1, _marker(mono, "control", action=action)) for mono, action in controls]
    events = [e for e in events if not any(lo < e[0] < hi for lo, hi in gaps)]
    events.sort(key=lambda e: (e[0], e[1]))
    header = {"type": "session", "format": "valence-session/1", "app": "valence-web", "session_id": "vs-test-0001",
              "started_wall": WALL0, "tz": "America/New_York", "feature_version": None, "emotion_model": None,
              "protocol": "valence-protocol/1"}
    return [header] + [line for *_, line in events]


def write(path, lines) -> str:
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return str(path)


@pytest.fixture(scope="module")
def models():
    return bm.load_models()


@pytest.fixture(scope="module")
def replayed(models, tmp_path_factory):
    path = write(tmp_path_factory.mktemp("bench") / "session.jsonl", session_lines())
    s = bm.load_session(path)
    return s, bm.replay(s, models, tz=s.header["tz"])


# --- parsing --------------------------------------------------------------------------------------------------------

def test_load_session_parses_every_line_type(tmp_path):
    lines = session_lines()
    lines.insert(5, {"type": "future_thing", "x": 1})
    s = bm.load_session(write(tmp_path / "s.jsonl", lines))
    assert s.session_id == "vs-test-0001" and s.header["tz"] == "America/New_York"
    assert len(s.observations) == 480 + 240 and len(s.ticks) == 48
    assert [m["kind"] for m in s.markers] == ["protocol_start", "block_start", "block_end", "block_start",
                                              "protocol_abort"]
    assert len(s.taps) == 1 and s.taps[0]["valence"] == 0.3
    assert s.unknown == {"future_thing": 1} and not s.legacy and s.late_observations == 0


def test_iter_obs_skips_session_lines_but_still_rejects_unknown_types(tmp_path):
    path = write(tmp_path / "s.jsonl", session_lines(60.0))
    obs = list(iter_obs(path))
    assert len(obs) == 120 + 60 and {type(o) for o in obs} == {VideoObs, AudioObs}
    bad = write(tmp_path / "bad.jsonl", session_lines(10.0) + [{"type": "future_thing"}])
    with pytest.raises(ValueError, match="unknown observation kind"):
        list(iter_obs(bad))


def test_mono_order_puts_late_observations_before_their_tick(tmp_path):
    late = obs_to_dict(AudioObs(t=1004.9, wall=WALL0 + 4.9, block_seq=0, db_spl=45.0, rms_power=1e-5))
    lines = [{"type": "tick", "wall": WALL0 + 5, "mono": 1005.0, "predictions": []}, late,
             {"type": "tick", "wall": WALL0 + 10, "mono": 1010.0, "predictions": []}]
    s = bm.load_session(write(tmp_path / "late.jsonl", lines))
    assert s.late_observations == 1
    assert [k for k, _ in bm.ordered_events(s, "mono")] == ["obs", "tick", "tick"]
    assert [k for k, _ in bm.ordered_events(s, "file")] == ["tick", "obs", "tick"]


# --- blocks and scores ---------------------------------------------------------------------------------------------

def test_find_blocks_pairs_markers_and_an_abort_ends_the_open_block():
    markers = [_marker(1000.0, "protocol_start"), _block(1000.0, "block_start", "focused", FOCUSED, True),
               _block(1120.0, "block_end", "focused", FOCUSED, True),
               _block(1120.0, "block_start", "restless", RESTLESS, False),
               _marker(1200.0, "protocol_abort", reason="test")]
    blocks, problems = bm.find_blocks(markers, 1240.0)
    assert [(b.id, b.start, b.end, b.ended_by) for b in blocks] == [
        ("focused", 1000.0, 1120.0, "block_end"), ("restless", 1120.0, 1200.0, "protocol_abort")]
    assert blocks[0].calm and not blocks[1].calm and blocks[1].expect == RESTLESS and not problems
    # an unmatched block_end is ignored; a block left open ends with the recording
    blocks, problems = bm.find_blocks([_block(990.0, "block_end", "x", FOCUSED, True),
                                       _block(1000.0, "block_start", "calm", FOCUSED, True)], 1240.0)
    assert [(b.id, b.end, b.ended_by) for b in blocks] == [("calm", 1240.0, "end_of_recording")] and len(problems) == 2


def _reading(i: int, card: str, m0: str | None = "Focused") -> TickReading:
    return TickReading(i, 5.0 * i, 5.0 * i, {1: {"card": card, "m2": card, "m0": m0, "map": None,
                                                 "valence": None, "arousal": None}}, None)


def test_hit_rate_first_hit_majority_and_grace():
    # 36 ticks at 5..180 s: one hit at 30 s (inside the grace), misses 60..80 s, hits 85..180 s
    cards = ["Frustrated_Masked" if (i == 6 or i >= 17) else "Focused" for i in range(1, 37)]
    ticks = [_reading(i, c, m0=None if 12 <= i <= 16 else "Focused") for i, c in enumerate(cards, 1)]
    ticks.append(TickReading(37, 185.0, 185.0, {}, None))  # after the block: never counted
    ticks.insert(10, TickReading(99, 52.5, 52.5, {}, None))  # the person absent (in the grace)
    block = Block("restless", "Restless", True, False, RESTLESS, 0.0, None, 180.0, "block_end")
    s = bm.score_block(block, ticks, person=1, grace=60.0)
    assert (s["ticks_in_block"], s["absent_in_block"], s["ticks_scored"], s["absent"]) == (37, 1, 25, 0)
    card = s["models"]["card"]
    assert (card["ticks"], card["hits"], card["hit_rate"]) == (25, 20, 0.8)
    assert card["first_hit_s"] == 30.0  # from the block start, grace included
    assert card["labels"] == {"Frustrated_Masked": 20, "Focused": 5}
    assert (card["majority"], card["majority_share"]) == ("Frustrated_Masked", 0.8)
    m0 = s["models"]["m0"]  # five ticks without a face reading: counted, as misses
    assert (m0["ticks"], m0["read"], m0["hits"], m0["hit_rate"], m0["first_hit_s"]) == (25, 20, 0, 0.0, None)
    assert m0["labels"]["none"] == 5
    assert s["models"]["map"]["hit_rate"] is None  # never read
    s0 = bm.score_block(block, ticks, person=1, grace=0.0)
    assert s0["ticks_scored"] == 37 and s0["models"]["card"]["hits"] == 21


def test_session_scores_from_block_scores():
    calm = Block("calm", "Calm", True, True, FOCUSED, 0.0, None, 100.0, "block_end")
    ticks = [_reading(i, "Exhausted" if i % 4 == 0 else "Focused") for i in range(1, 21)]
    blocks = [bm.score_block(calm, ticks, 1, 0.0)]
    fa = bm.calm_false_alarms(blocks)
    assert fa["blocks"] == ["calm"] and (fa["card"]["alarms"], fa["card"]["ticks"], fa["card"]["rate"]) == (5, 20, 0.25)
    assert bm.mean_hit_rates(blocks)["card"] == {"blocks": 0, "mean": None}  # calm blocks are not in the mean


# --- the replay ----------------------------------------------------------------------------------------------------

def test_replay_runs_the_runtime_at_the_recorded_ticks(replayed):
    s, rep = replayed
    assert [t.mono for t in rep.ticks] == [MONO0 + 5.0 * i for i in range(1, 49)]
    rows = [t.people[1] for t in rep.ticks]
    assert all(r["card"] in ("Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy") for r in rows)
    assert [r["warming_up"] for r in rows[:13]] == [True] * 11 + [False] * 2  # the smoother's 60 s warm-up
    assert all(r["m2"] is not None and r["m0"] is not None for r in rows)  # raw argmax; a face every tick
    assert all(r["cues"] and r["map"] is not None and -1 <= r["valence"] <= 1 for r in rows)
    assert rep.tz == "America/New_York" and rep.ticks[0].tod_h == pytest.approx(10.0 + 5 / 3600)
    assert eng_mod.time is time and eng_mod.local_tod.__module__ == "valence.runtime.engine"  # patches undone


def test_time_of_day_matches_the_engine_under_tz():
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX-only")
    old = os.environ.get("TZ")
    try:
        for zone in ("America/New_York", "Asia/Kolkata", "UTC"):
            os.environ["TZ"] = zone
            time.tzset()
            for wall in (WALL0, WALL0 + 12345.678, 1_798_761_599.9):
                assert bm.time_of_day(zone)(wall) == eng_mod.local_tod(wall)
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_score_windows_blocks_matches_taps_and_handles_the_abort(replayed):
    s, rep = replayed
    r = bm.score(s, rep, grace=60.0)
    focused, restless = r["blocks"]
    assert (focused["ticks_in_block"], focused["ticks_scored"]) == (24, 13)  # 1005..1120; 1060..1120
    assert (restless["ticks_in_block"], restless["ticks_scored"], restless["ended_by"]) == (16, 5, "protocol_abort")
    by_mono = {t["mono"]: t["people"][0] for t in r["ticks"]}
    for b in (focused, restless):
        for m in bm.MODELS:
            expect = b["expect"]["map" if m == "map" else "card"]
            labels = [by_mono[mono][m] for mono in by_mono if b["start_mono"] + 60 <= mono <= b["end_mono"]]
            assert b["models"][m]["hits"] == sum(lab in expect for lab in labels)
            assert b["models"][m]["hit_rate"] == round(sum(lab in expect for lab in labels) / len(labels), 3)
    assert [t["block"] for t in r["ticks"]][-8:] == [None] * 8  # after the abort
    sr = r["session"]["self_report"]
    assert (sr["taps"], sr["matched"]) == (1, 1)
    row = sr["rows"][0]
    assert (row["block"], row["lag_s"]) == ("focused", 2.5) and row["map_valence"] == by_mono[1100.0]["valence"]
    assert sr["mae_valence"] == round(abs(0.3 - row["map_valence"]), 3) and sr["zero_mae_arousal"] == 0.2
    fa = r["session"]["calm_false_alarm"]
    assert fa["blocks"] == ["focused"] and fa["card"]["ticks"] == 13
    mh = r["session"]["mean_hit_rate"]
    assert mh["card"] == {"blocks": 1, "mean": restless["models"]["card"]["hit_rate"]}
    assert r["counts"]["taps"] == 1 and r["replay"]["time_of_day"] == "wall clock in America/New_York"


def test_parity_is_complete_on_the_replays_own_labels_and_counts_mismatches(replayed, models, tmp_path):
    s, rep = replayed
    preds = {t.mono: [{"person_id": pid, "label": row["card"], "face_only": {"label": row["m0"]},
                       "affect": {"label": row["map"], "valence": row["valence"], "arousal": row["arousal"]}}
                      for pid, row in t.people.items()] for t in rep.ticks}
    path = write(tmp_path / "p.jsonl", session_lines(predictions=preds))
    s2 = bm.load_session(path)
    par = bm.score(s2, bm.replay(s2, models, tz="America/New_York"))["session"]["parity"]
    assert (par["card"]["ticks"], par["card"]["share"], par["card"]["mismatches"]) == (48, 1.0, 0)
    assert par["m0"]["share"] == 1.0 and par["map"]["share"] == 1.0
    wrong = "Exhausted" if preds[1050.0][0]["label"] != "Exhausted" else "Focused"
    preds[1050.0][0]["label"] = wrong
    preds[1055.0] = []  # the browser showed nobody
    s3 = bm.load_session(write(tmp_path / "q.jsonl", session_lines(predictions=preds)))
    par = bm.score(s3, bm.replay(s3, models, tz="America/New_York"))["session"]["parity"]
    assert (par["card"]["matches"], par["card"]["mismatches"]) == (46, 2)
    assert [(x["mono"], x["browser"]) for x in par["card"]["examples"]] == [(1050.0, wrong), (1055.0, None)]


def test_start_after_pause_forgets_the_people(models, tmp_path):
    lines = session_lines(150.0, gaps=((1100.2, 1110.2),), controls=((1100.2, "pause"), (1110.2, "start")))
    s = bm.load_session(write(tmp_path / "pause.jsonl", lines))
    rep = bm.replay(s, models, tz=None)
    warm = {t.mono: t.people[1]["warming_up"] for t in rep.ticks}
    assert warm[1100.0] is False and warm[1115.0] is True  # the person's window restarted
    assert [c["action"] for c in rep.controls] == ["pause", "start"] and rep.ticks[0].tod_h == 12.0
    assert 1105.0 not in warm  # the browser does not tick while paused


# --- legacy manifests and the CLI -------------------------------------------------------------------------------------

def _legacy_capture(path, seconds: float = 90.0) -> None:
    """Observations only, from before the emotion-v1 cues (their fields null)."""
    write(path, [x for x in session_lines(seconds, cues=False) if x["type"] in ("video", "audio")])


def test_legacy_manifest_gives_each_capture_one_block(models, tmp_path):
    _legacy_capture(tmp_path / "capture_tiny.jsonl")
    manifest = {"format": "valence-legacy-manifest/1", "tz": None, "captures": [
        {"file": "capture_tiny.jsonl", "id": "tiny", "label": "Tiny", "scored": True, "calm": True, "expect": FOCUSED,
         "note": "synthetic"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    sessions, _ = bm.legacy_sessions(tmp_path / "manifest.json")
    (s,) = sessions
    assert s.legacy and s.session_id == "legacy-tiny" and s.note == "synthetic"
    assert [t["mono"] for t in s.ticks] == [MONO0 + 0.05 + 5.0 * i for i in range(1, 18)]  # every 5 s, no partial
    r = bm.score(s, bm.replay(s, models), grace=60.0)
    (b,) = r["blocks"]
    assert (b["id"], b["calm"], b["ticks_in_block"], b["ticks_scored"]) == ("tiny", True, 17, 6)
    assert r["scoring"]["map"].startswith("no emotion-v1 face cues") and b["models"]["map"]["hit_rate"] is None
    assert b["models"]["card"]["hit_rate"] is not None and r["session"]["parity"] is None
    assert r["replay"]["time_of_day"] == "fixed at 12:00" and r["kind"] == "legacy"


def test_cli_writes_json_and_markdown(tmp_path):
    session = write(tmp_path / "valence-session-test.jsonl", session_lines(150.0))
    _legacy_capture(tmp_path / "capture_tiny.jsonl", 70.0)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "valence-legacy-manifest/1", "captures": [
        {"file": "capture_tiny.jsonl", "id": "tiny", "label": "Tiny", "scored": True, "calm": True,
         "expect": FOCUSED}]}))
    out = tmp_path / "out"
    reports = bm.main([session, "--legacy", str(tmp_path / "manifest.json"), "--out", str(out), "--grace", "30"])
    assert [r["session_id"] for r in reports] == ["vs-test-0001", "legacy-tiny"]
    assert sorted(p.name for p in out.iterdir()) == ["legacy-tiny.json", "legacy-tiny.md", "summary.md",
                                                     "vs-test-0001.json", "vs-test-0001.md"]
    data = json.loads((out / "vs-test-0001.json").read_text())
    assert data["scoring"]["grace_s"] == 30.0 and len(data["ticks"]) == 30 and data["blocks"][0]["id"] == "focused"
    md = (out / "vs-test-0001.md").read_text()
    for title in bm.MODEL_TITLES.values():
        assert f"## {title}" in md
    assert "| Focused (`focused`, calm) |" in md
    summary = (out / "summary.md").read_text()
    assert "| vs-test-0001 |" in summary and "| legacy-tiny |" in summary and "(`tiny`, calm)" in summary
