"""Golden-demo tooling (scripts/spikes/golden_capture.py, replay_check.py): no camera, no mic."""

from __future__ import annotations

import ast
import importlib
import json
import math
import sys
import time as real_time
from pathlib import Path

import pytest

from valence.contracts import AudioObs, PersonObs, VideoObs, obs_from_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
SPIKES = REPO_ROOT / "scripts" / "spikes"
DATA = REPO_ROOT / "data" / "real"


def _load(name: str):
    sys.path.insert(0, str(SPIKES))
    try:
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SPIKES))


@pytest.fixture(scope="module")
def gc():
    return _load("golden_capture")


@pytest.fixture(scope="module")
def rc():
    return _load("replay_check")


# --- cue schedule -------------------------------------------------------------------------------------

def test_fist_cues_are_45_to_60_s_apart_seeded_and_only_in_the_masked_act(gc) -> None:
    segs = {s.name: s for s in gc.build_segments(1.0)}
    fists = [c.at for c in segs["masked"].cues if c.kind == "fist"]
    assert len(fists) >= 7  # 8 min / 45-60 s
    gaps = [b - a for a, b in zip([0.0, *fists], fists)]
    assert all(45.0 <= g <= 60.0 for g in gaps), gaps
    assert fists[-1] + gc.FIST_HOLD_S <= segs["masked"].seconds
    again = [c.at for c in gc.build_segments(1.0)[2].cues if c.kind == "fist"]
    assert again == fists  # fixed seed
    other = [c.at for c in gc.build_segments(1.0, seed=gc.SEED + 1)[2].cues if c.kind == "fist"]
    assert other != fists
    for name, seg in segs.items():
        if name != "masked":
            assert not [c for c in seg.cues if c.kind in ("fist", "sigh")], name


def test_sigh_cues_about_every_90_s_and_clear_of_fist_cues(gc) -> None:
    masked = gc.build_segments(1.0)[2]
    fists = [c.at for c in masked.cues if c.kind == "fist"]
    sighs = [c.at for c in masked.cues if c.kind == "sigh"]
    assert 4 <= len(sighs) <= 6
    assert all(abs(s - f) >= 6.0 for s in sighs for f in fists)
    assert masked.before and "PHONE" in masked.before and "STOP" in masked.after


def test_segment_plan_and_dry_run_scale(gc) -> None:
    full = gc.build_segments(1.0)
    assert [s.name for s in full] == ["neutral", "typing", "masked", "break", "laugh"]
    assert [s.seconds for s in full] == [90, 240, 480, 120, 210]
    dry = gc.build_segments(0.1)
    assert [round(s.seconds, 1) for s in dry] == [9.0, 24.0, 48.0, 12.0, 21.0]
    for seg in dry:
        assert all(0 <= c.at and c.at + c.hold <= seg.seconds + 1e-9 for c in seg.cues), seg.name
    fists = [c.at for c in dry[2].cues if c.kind == "fist"]
    assert fists and all(b - a >= 10.0 for a, b in zip(fists, fists[1:]))
    assert sum(c.kind == "compliment" for c in full[4].cues) == 3


# --- summary metrics on a synthetic stream -------------------------------------------------------------

def _person(tid: int, val: float = math.nan, fist: int = 0, visible: bool = True, h: float = 0.7) -> PersonObs:
    return PersonObs(track_id=tid, bbox=(0.2, 0.25, 0.8, 0.25 + h), fist_onsets=fist, motion=0.03,
                     face_visible=visible, face_valence=val if visible else math.nan)


def _stream(t0: float, seconds: float, valence, fists_at: set[float] = frozenset(), db: float = 67.0,
            extra_person: bool = False, laughs_at: set[int] = frozenset()) -> list:
    out = []
    n = int(seconds * 10)
    for i in range(n):
        t = t0 + i / 10
        rel = round(i / 10, 1)
        persons = [_person(1, valence(rel), fist=int(rel in fists_at))]
        if extra_person:
            persons.append(_person(2, 0.6))
        out.append(VideoObs(t=t, wall=1e9 + t, frame_seq=i, frames_decoded=i + 1, lux=250.0, lux_quality=0.5,
                            persons=tuple(persons)))
        if i % 10 == 0:
            s = i // 10
            out.append(AudioObs(t=t + 0.02, wall=1e9 + t, block_seq=s, db_spl=db + (s % 3), rms_power=1e-4,
                                speech_prob=0.2, laughter_onsets=int(s in laughs_at),
                                compliment_hits=int(s == 7)))
    return sorted(out, key=lambda o: o.t)


def test_segment_metrics_face_valence_fists_and_audio(gc) -> None:
    masked = gc.build_segments(1.0)[2]
    cue_t = [c.at for c in masked.cues if c.kind == "fist"][:3]
    hits = [round(cue_t[0] + 1.0, 1), round(cue_t[1] + 0.5, 1)]
    onsets = {*hits, 5.0}  # 2 cued hits + 1 false onset
    obs = _stream(100.0, 180.0, lambda r: 0.30 + 0.002 * (r % 100), onsets)
    m = gc.segment_metrics(obs, 100.0, 280.0, masked.cues)
    assert m["video_obs"] == 1800 and m["audio_obs"] == 180
    assert m["video_hz"] == pytest.approx(10.0, abs=0.01)
    assert m["people_seen"] == [1] and m["person"] == 1 and m["face_visible_frac"] == 1.0
    p10, p50, p90 = m["face_valence_p10_p50_p90"]
    assert 0.30 <= p10 < p50 < p90 <= 0.50
    assert 0.0 < m["face_valence_ge_0_4_frac"] < 1.0
    assert m["face_valence_outside_cues_p10_p50_p90"][1] is not None
    assert m["fist_onsets"] == 3
    cued = [c for c in masked.cues if c.kind == "fist" and c.at < 180.0]
    assert m["fist_cues"] == len(cued)
    assert m["fist_cue_latency_s"][:2] == [pytest.approx(h - c, abs=0.011) for h, c in zip(hits, cue_t)]
    assert all(x is None for x in m["fist_cue_latency_s"][2:])
    assert m["fist_recall"] == round(2 / len(cued), 3)
    assert m["fist_false_onsets"] == 1
    assert m["db_p10_p50_p90"][1] == pytest.approx(68.0)
    assert m["compliment_hits"] == 1 and m["laughter_onsets"] == 0
    json.dumps(m, allow_nan=False)


def test_segment_metrics_time_range_and_people(gc) -> None:
    obs = _stream(0.0, 30.0, lambda r: 0.0) + _stream(30.0, 30.0, lambda r: 0.6, extra_person=True,
                                                     laughs_at={3, 9})
    first = gc.segment_metrics(obs, 0.0, 30.0)
    second = gc.segment_metrics(obs, 30.0, 60.0)
    assert first["people_seen"] == [1] and first["face_valence_p10_p50_p90"] == [0.0, 0.0, 0.0]
    assert second["people_seen"] == [1, 2] and second["max_people_in_frame"] == 2
    assert second["laughter_onsets"] == 2 and second["laughter_s"] == [3.0, 9.0]
    assert "fist_recall" not in second
    empty = gc.segment_metrics(obs, 100.0, 120.0)
    assert empty["video_obs"] == 0 and empty["people_seen"] == []


def test_build_summary_boundaries_relative_to_first_line(gc, tmp_path: Path) -> None:
    segs = gc.build_segments(0.1)
    obs = _stream(50.0, 60.0, lambda r: 0.45)
    records = [{"name": "neutral", "skipped": False, "t_start": 52.0, "t_end": 61.0},
               {"name": "typing", "skipped": True},
               {"name": "masked", "skipped": False, "t_start": 62.0, "t_end": 108.0, "complete": False}]
    s = gc.build_summary(obs, records, segs, jsonl=tmp_path / "g.jsonl", scale=0.1, seed=7)
    assert s["t0"] == 50.0 and s["segments"][0]["start_s"] == 2.0 and s["segments"][0]["end_s"] == 11.0
    assert s["segments"][1] == {"segment": "typing", "planned_s": 24.0, "skipped": True}
    assert s["segments"][2]["complete"] is False
    assert s["held_smile"]["face_valence_p10_p50_p90"] == [0.45, 0.45, 0.45]
    json.dumps(s, allow_nan=False)


# --- overwrite guard + main() without hardware -------------------------------------------------------------

def test_overwrite_guard(gc, tmp_path: Path) -> None:
    jsonl, summary = gc.output_paths(tmp_path)
    assert jsonl.name == "golden_demo.jsonl" and summary.name == "golden_demo_summary.json"
    summary.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gc.output_paths(tmp_path)
    assert gc.output_paths(tmp_path, force=True) == (jsonl, summary)
    jsonl.write_text("", encoding="utf-8")
    assert gc.main(["--out-dir", str(tmp_path), "--yes"]) == 2  # refuses before touching the camera
    assert jsonl.read_text(encoding="utf-8") == ""


class _FakeClock:
    """Stands in for the script's ``time`` module; each sleep advances the clock and emits fake obs."""

    def __init__(self) -> None:
        self.now, self.sink, self.seq, self.bystander = 1000.0, None, 0, False

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1.7e9 + self.now

    def strftime(self, fmt: str) -> str:
        return real_time.strftime(fmt)

    def sleep(self, dt: float) -> None:
        end = self.now + dt
        while self.now < end - 1e-9:
            self.now = min(end, self.now + 0.1)
            if self.sink is not None:
                self.seq += 1
                persons = (_person(1, 0.42),) + ((_person(9, 0.0, h=0.1),) if self.bystander else ())
                self.sink(VideoObs(t=self.now, wall=self.time(), frame_seq=self.seq, frames_decoded=self.seq,
                                   persons=persons))
                if self.seq % 10 == 0:
                    self.sink(AudioObs(t=self.now, wall=self.time(), block_seq=self.seq // 10, db_spl=66.0,
                                       rms_power=1e-4))


def test_main_records_one_continuous_contract_jsonl(gc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import valence.edge.run as edge_run

    clock = _FakeClock()
    stopped: list[bool] = []

    def fake_start(sink, **kw):
        clock.sink = sink

        def stop() -> None:
            clock.sink = None
            stopped.append(True)
        return stop

    monkeypatch.setattr(gc, "time", clock)
    monkeypatch.setattr(edge_run, "start", fake_start)
    assert gc.main(["--out-dir", str(tmp_path), "--yes", "--minutes-scale", "0.05"]) == 0
    assert stopped == [True]
    lines = (tmp_path / "golden_demo.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) > 100
    for line in lines:
        d = json.loads(line)
        assert d["type"] in ("video", "audio") and "segment" not in d
        obs_from_dict(d)
    s = json.loads((tmp_path / "golden_demo_summary.json").read_text(encoding="utf-8"))
    assert [x["segment"] for x in s["segments"]] == ["neutral", "typing", "masked", "break", "laugh"]
    assert all(x["complete"] and x["video_obs"] > 0 for x in s["segments"])
    starts = [x["start_s"] for x in s["segments"]]
    assert starts == sorted(starts) and s["obs"] == len(lines)
    assert s["held_smile"]["face_valence_p10_p50_p90"] == [0.42, 0.42, 0.42]
    assert gc.main(["--out-dir", str(tmp_path), "--yes"]) == 2  # second take needs --force


# --- replay_check ------------------------------------------------------------------------------------------

def _served_available() -> bool:
    return (REPO_ROOT / "artifacts" / "model_meta.json").exists()


needs_served = pytest.mark.skipif(not _served_available() or not (DATA / "capture_typing.jsonl").exists(),
                                  reason="served artifacts or capture_typing.jsonl missing")


@needs_served
def test_replay_check_on_capture_typing(rc) -> None:
    res = rc.check(DATA / "capture_typing.jsonl")
    rows, rep = res["timeline"], res["report"]
    assert rows and set(rows[0]) == set(rc.TIMELINE_KEYS)
    assert all(isinstance(v, (str, int, float, bool, type(None))) for r in rows for v in r.values())
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts) and ts[0] == 5.0 and all(t % 5 == 0 for t in ts)
    assert rep["ticks"] == len(rows) and rep["scored_ticks"] == len(rows) - rep["warmup_ticks"]
    assert abs(sum(rep["labels"].values()) - 1.0) < 0.01
    assert rep["masked_stress_with_override"] in (True, False)
    json.dumps(res, allow_nan=False)


@needs_served
def test_replay_check_segments_from_a_summary(rc, tmp_path: Path) -> None:
    path = DATA / "capture_typing.jsonl"
    first = next(iter(rc.iter_obs(path))).t
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"segments": [
        {"segment": "a", "skipped": False, "t_start": first, "t_end": first + 60.0},
        {"segment": "skipme", "skipped": True},
        {"segment": "b", "skipped": False, "start_s": 70.0, "end_s": 400.0},
    ]}), encoding="utf-8")
    res = rc.check(path, summary)
    segs = res["report"]["segments"]
    assert set(segs) >= {"a", "b"} and "skipme" not in segs
    assert {r["segment"] for r in res["timeline"] if r["t"] <= 60.0} == {"a"}
    assert {r["segment"] for r in res["timeline"] if 62.5 < r["t"] <= 72.5} == {"(between)"}
    assert segs["a"]["start_s"] == 0.0 and segs["b"]["end_s"] == 400.0


@needs_served
def test_replay_check_cli_json(rc, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "t.json"
    assert rc.main([str(DATA / "capture_typing.jsonl"), "--json", str(out), "--no-timeline"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["timeline"] and "labels" in data["report"]
    assert '"labels"' in capsys.readouterr().out


def test_report_detects_masked_stress_with_override(rc) -> None:
    def row(t, label, shown, override, warm=False, seg=None):
        return {"t": t, "segment": seg, "person_id": 1, "person": "Person #1", "label": label, "label_display": shown,
                "confidence": 0.7, "mask_override": override, "face_only_label": "Collaborative_Happy",
                "face_only_confidence": 0.8, "warming_up": warm}
    rows = [row(5.0, "Focused", "Focused", False, True), row(10.0, "Frustrated_Masked", "Frustrated", False),
            row(15.0, "Frustrated_Masked", rc.MASKED_DISPLAY, True), row(20.0, "Focused", "Focused", False)]
    rep = rc.report(rows, [], file="x", model="m")
    assert rep["masked_stress_with_override"] and rep["masked_stress_first_t"] == 15.0
    assert rep["first_mask_override"]["t"] == 15.0
    assert rep["labels"] == {"Frustrated_Masked": 0.667, "Focused": 0.333}
    assert rep["mask_override_share"] == 0.333


# --- privacy -------------------------------------------------------------------------------------------

BANNED_ATTRS = {("cv2", "imwrite"), ("cv2", "VideoWriter"), ("np", "save"), ("np", "savez"),
                ("numpy", "save"), ("numpy", "savez"), ("wave", "open"), ("soundfile", "write"),
                ("pickle", "dump")}


@pytest.mark.parametrize("rel", ["scripts/spikes/golden_capture.py", "scripts/spikes/replay_check.py"])
def test_no_banned_media_writes(rel: str) -> None:
    tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in BANNED_ATTRS, f"{rel}:{node.lineno}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {a.name for a in node.names} | {getattr(node, "module", None) or ""}
            assert not names & {"soundfile", "pickle", "wave", "cv2"}, f"{rel}:{node.lineno}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            raise AssertionError(f"{rel}:{node.lineno} opens a file directly")


# --- adversarial review regressions ----------------------------------------------------------------------

def test_every_cue_is_shown_and_rings_at_its_start(gc) -> None:
    """Overlapping cues (the 0.1-scale laugh segment) must not hide a later cue behind an earlier one."""
    for scale in (1.0, 0.1, 0.05):
        for seg in gc.build_segments(scale):
            for c in seg.cues:
                assert gc.active_cue(seg.cues, c.at + 1e-6) is c, (scale, seg.name, c)


def test_recorder_keeps_participants_only_like_the_jsonl(gc, tmp_path: Path) -> None:
    """The summary is computed from the in-memory obs: a bystander that never reaches the JSONL must not
    reach the summary either (people seen, frames per person)."""
    from valence.edge.run import JsonlSink, participants_only

    sink = JsonlSink(tmp_path / "g.jsonl")
    rec = gc.Recorder(sink, keep=participants_only)
    rec(VideoObs(t=1.0, wall=1e9, frame_seq=0, persons=(_person(1, 0.4, h=0.7), _person(2, 0.1, h=0.1))))
    sink.close()
    assert [p.track_id for p in rec.obs[0].persons] == [1]
    line = json.loads((tmp_path / "g.jsonl").read_text(encoding="utf-8"))
    assert [p["track_id"] for p in line["persons"]] == [1]


def _fake_runner(clock: "_FakeClock", stopped: list[bool], bystander: bool = False):
    def fake_start(sink, **kw):
        clock.sink, clock.bystander = sink, bystander

        def stop() -> None:
            clock.sink = None
            stopped.append(True)
        return stop
    return fake_start


def test_main_summary_excludes_bystanders(gc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import valence.edge.run as edge_run

    clock, stopped = _FakeClock(), []
    monkeypatch.setattr(gc, "time", clock)
    monkeypatch.setattr(edge_run, "start", _fake_runner(clock, stopped, bystander=True))
    assert gc.main(["--out-dir", str(tmp_path), "--yes", "--minutes-scale", "0.05"]) == 0
    s = json.loads((tmp_path / "golden_demo_summary.json").read_text(encoding="utf-8"))
    assert all(x["people_seen"] == [1] and x["max_people_in_frame"] == 1 for x in s["segments"])


def test_failed_edge_start_leaves_nothing_that_blocks_a_retry(gc, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import valence.edge.run as edge_run

    def broken(sink, **kw):
        raise RuntimeError("camera gave no usable frames")
    monkeypatch.setattr(edge_run, "start", broken)
    assert gc.main(["--out-dir", str(tmp_path), "--yes"]) == 3
    assert not (tmp_path / "golden_demo.jsonl").exists()
    assert not list(tmp_path.glob("golden_demo*"))  # no partial file left behind either
    # --force over a good earlier take: a failed start must not destroy it
    (tmp_path / "golden_demo.jsonl").write_text('{"keep": 1}\n', encoding="utf-8")
    assert gc.main(["--out-dir", str(tmp_path), "--yes", "--force"]) == 3
    assert (tmp_path / "golden_demo.jsonl").read_text(encoding="utf-8") == '{"keep": 1}\n'


def test_failed_edge_start_of_any_kind_is_reported(gc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import valence.edge.run as edge_run

    def broken(sink, **kw):
        raise OSError("no such device")
    monkeypatch.setattr(edge_run, "start", broken)
    assert gc.main(["--out-dir", str(tmp_path), "--yes"]) == 3
    assert not (tmp_path / "golden_demo.jsonl").exists()


def test_ctrl_c_during_warmup_still_stops_the_camera(gc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import valence.edge.run as edge_run

    clock, stopped = _FakeClock(), []
    real_sleep = clock.sleep

    def sleep(dt: float) -> None:
        real_sleep(min(dt, 1.0))
        if dt == 4.0:
            raise KeyboardInterrupt
    clock.sleep = sleep
    monkeypatch.setattr(gc, "time", clock)
    monkeypatch.setattr(edge_run, "start", _fake_runner(clock, stopped))
    (tmp_path / "golden_demo.jsonl").write_text('{"keep": 1}\n', encoding="utf-8")
    assert gc.main(["--out-dir", str(tmp_path), "--yes", "--force"]) == 130
    assert stopped == [True]
    # no segment started: the earlier take survives and no summary or side file is written
    assert (tmp_path / "golden_demo.jsonl").read_text(encoding="utf-8") == '{"keep": 1}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["golden_demo.jsonl"]


def test_unexpected_error_mid_session_still_stops_and_summarizes(gc, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import valence.edge.run as edge_run

    clock, stopped = _FakeClock(), []
    monkeypatch.setattr(gc, "time", clock)
    monkeypatch.setattr(edge_run, "start", _fake_runner(clock, stopped))
    calls = {"n": 0}
    real_readout = gc.Recorder.readout

    def flaky(self) -> str:
        calls["n"] += 1
        if calls["n"] == 50:
            raise ValueError("boom")
        return real_readout(self)
    monkeypatch.setattr(gc.Recorder, "readout", flaky)
    assert gc.main(["--out-dir", str(tmp_path), "--yes", "--minutes-scale", "0.05"]) != 0
    assert stopped == [True]
    s = json.loads((tmp_path / "golden_demo_summary.json").read_text(encoding="utf-8"))
    assert s["segments"] and s["segments"][-1]["complete"] is False
    assert (tmp_path / "golden_demo.jsonl").exists()


def test_wait_enter_discards_keystrokes_typed_during_a_segment(gc, monkeypatch: pytest.MonkeyPatch) -> None:
    """Typing segment: lines typed into this Terminal must not start (or "s"-skip) the masked act."""
    order: list[str] = []
    monkeypatch.setattr(gc, "_flush_stdin", lambda: order.append("flush"))
    monkeypatch.setattr("builtins.input", lambda *a: order.append("input") or "")
    rec = gc.Recorder(lambda o: None)
    assert gc.wait_enter(rec, "go", lambda: None) == ""
    assert order == ["flush", "input"]


def test_summary_is_strict_json_even_with_non_finite_values(gc, tmp_path: Path) -> None:
    segs = gc.build_segments(0.1)
    obs = _stream(0.0, 20.0, lambda r: 0.3)
    records = [{"name": "neutral", "skipped": False, "t_start": 0.0, "t_end": 9.0}]
    s = gc.build_summary(obs, records, segs, jsonl=tmp_path / "g.jsonl", scale=0.1, seed=7,
                         edge={"video_hz": float("nan"), "errors": 0})
    json.dumps(s, allow_nan=False)
    assert s["edge"]["video_hz"] is None


def test_report_shares_per_person(rc) -> None:
    def row(t, pid, label, seg="laugh"):
        return {"t": t, "segment": seg, "person_id": pid, "person": f"Person #{pid}", "label": label,
                "label_display": label, "confidence": 0.7, "mask_override": False, "face_only_label": None,
                "face_only_confidence": None, "warming_up": False}
    rows = [row(5.0, 1, "Collaborative_Happy"), row(5.0, 2, "Focused"), row(10.0, 1, "Collaborative_Happy")]
    rep = rc.report(rows, [{"segment": "laugh", "start_s": 0.0, "end_s": 20.0}], file="x", model="m")
    assert rep["persons"]["1"]["labels"] == {"Collaborative_Happy": 1.0}
    assert rep["persons"]["2"]["labels"] == {"Focused": 1.0}
    assert rep["segments"]["laugh"]["persons"]["1"]["scored_ticks"] == 2
