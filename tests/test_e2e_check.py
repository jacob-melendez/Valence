"""The pre-demo check's pure logic (scripts/e2e_check.py), fed synthetic messages: no server."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from valence.contracts import Explanation, Governance, Heartbeat, Hello, Prediction, StateInfo, Status

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e2e_check.py"
spec = importlib.util.spec_from_file_location("e2e_check", SCRIPT)
e2e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e2e  # dataclasses resolve annotations through it
spec.loader.exec_module(e2e)


@pytest.fixture(scope="module")
def validators():
    return e2e.make_validators()


def hello():
    return Hello(mode="replay", states=[StateInfo(id="Focused", display="Focused", tier="P0")],
                 feature_version="stub-rules").model_dump(mode="json")


def prediction(tick, *, pid=1, mask=False):
    return Prediction(
        event_id=f"p{pid}-{tick:06d}", t=1.0 + tick, person_id=pid, label="Focused", label_display="Focused",
        confidence=0.8, probabilities={"Focused": 0.8, "Exhausted": 0.2},
        face_only={"label": "Collaborative_Happy", "confidence": 0.6}, mask_override=mask, warming_up=False,
        layers=[], data_quality={"face_visible_frac": 1.0}, explanation={"text": "x"}, source="replay",
    ).model_dump(mode="json")


def governance(tick, *, violations=0, purge_gap=0):
    return Governance(t=1.0 + tick, tick=tick, frames_decoded=0, frames_purged=0, audio_blocks_purged=0,
                      transcripts_purged=0, violations=violations, live_raw_buffers=0, retention_s=86400,
                      purge_gap=purge_gap).model_dump(mode="json")


def explanation(event_id, pid=1):
    return Explanation(event_id=event_id, person_id=pid, text="Person #1 reads as Focused.", source="llm",
                       latency_ms=500).model_dump(mode="json")


def status(**kw):
    return Status(mode="replay", camera=False, mic=False, fps=0.0, loop_latency_ms=3.0, narrator="fallback",
                  offline=False, uptime_s=10.0, **kw).model_dump(mode="json")


def run(validators, messages, t=10.0):
    c = e2e.Checker(validators, masked_window_s=(15.0, 42.0))
    for m in messages:
        c.feed(m, t)
    return {r.name: r for r in c.results()}, c


def good_stream():
    msgs = [hello()]
    for tick in range(6):
        msgs += [governance(tick), prediction(tick, mask=tick == 4), status()]
    msgs += [explanation("p1-000004"), Heartbeat(t=9.0).model_dump(mode="json")]
    return msgs


def test_schema_map_covers_every_ws_message():
    kinds = set(e2e.schema_by_type())
    assert kinds == {"hello", "tracks", "governance", "state_vector", "prediction", "explanation", "status", "heartbeat"}


def test_good_stream_passes_and_is_satisfied(validators):
    res, c = run(validators, good_stream())
    assert all(r.ok for r in res.values()), [r for r in res.values() if not r.ok]
    assert c.satisfied()
    assert res["explanation pairing"].detail.startswith("1/1")
    assert "first at +10.0 s" in res["mask_override"].detail


def test_no_mask_override_fails(validators):
    msgs = [m for m in good_stream() if m["type"] != "prediction"] + [prediction(9)]
    res, c = run(validators, msgs)
    assert not res["mask_override"].ok and not c.satisfied()


def test_unpaired_explanation_fails_but_early_one_is_ignored(validators):
    res, _ = run(validators, good_stream() + [explanation("p1-000099")])
    assert not res["explanation pairing"].ok and "unknown event_id" in res["explanation pairing"].detail
    res, _ = run(validators, good_stream() + [explanation("p2-000004", pid=2)])
    assert not res["explanation pairing"].ok  # p2-000004 was never predicted
    res, _ = run(validators, [hello(), explanation("p1-000001"), *good_stream()[1:]], t=0.5)  # within grace
    assert res["explanation pairing"].ok and "ignored" in res["explanation pairing"].detail


def test_explanation_before_its_prediction_fails(validators):
    msgs = [hello(), explanation("p1-000001"), *good_stream()[1:]]
    res, _ = run(validators, msgs)
    assert not res["explanation pairing"].ok


@pytest.mark.parametrize("bad", [governance(9, violations=1), governance(9, purge_gap=3), status(purge_violation=True)])
def test_governance_failures(validators, bad):
    res, _ = run(validators, good_stream() + [bad])
    assert not res["governance"].ok and all(r.ok for k, r in res.items() if k != "governance")


def test_purge_gap_of_two_is_allowed(validators):
    res, _ = run(validators, good_stream() + [governance(9, purge_gap=2)])
    assert res["governance"].ok


def test_schema_violations_fail(validators):
    extra = {**prediction(9), "unexpected": 1}  # additionalProperties: false
    res, _ = run(validators, good_stream() + [extra])
    assert not res["ws schema"].ok and "unexpected" in res["ws schema"].detail
    wrong = {**governance(9), "violations": "zero"}
    res, _ = run(validators, good_stream() + [wrong])
    assert not res["ws schema"].ok
    res, _ = run(validators, good_stream() + [{"type": "mystery"}])
    assert not res["ws schema"].ok and "no schema" in res["ws schema"].detail
    c = e2e.Checker(validators)
    c.feed_raw("not json")
    assert c.schema_errors == 1


def test_hello_must_come_first(validators):
    res, _ = run(validators, good_stream()[1:] + [hello()])
    assert not res["ws /ws/events"].ok


def test_empty_stream_fails(validators):
    res, _ = run(validators, [])
    assert not res["predictions"].ok and not res["governance"].ok and not res["ws schema"].ok


def test_rest_checks():
    assert e2e.check_verify(200, {"ok": True, "entries": 4}).ok
    assert not e2e.check_verify(200, {"ok": False, "entries": 2}).ok
    assert not e2e.check_verify(503, {"detail": "runtime not started"}).ok
    assert e2e.check_health(200, {"ok": True, "mode": "replay", "room_baseline": {"mode": "auto"}}).ok
    assert not e2e.check_health(200, {"ok": True, "mode": "replay"}).ok
    assert not e2e.check_health(500, "Internal Server Error").ok
    assert e2e.check_replay(200, {"ok": True, "scenario": "full_demo", "speed": 20.0}, "full_demo").ok
    assert not e2e.check_replay(422, {"detail": "speed"}, "full_demo").ok


def test_table_and_verdict():
    ok = e2e.render_table([e2e.Result("a", True, "fine"), e2e.Result("bb", True, "")])
    assert "e2e: PASS (2/2" in ok
    bad = e2e.render_table([e2e.Result("a", True, "fine"), e2e.Result("bb", False, "broken")])
    assert any("FAIL" in line and "broken" in line for line in bad.splitlines()) and "e2e: FAIL (1/2" in bad


def test_masked_window_from_full_demo():
    start, end = e2e.masked_window_min("full_demo")
    assert start < end
    assert e2e.default_seconds("full_demo", 20.0) == pytest.approx(end * 3 + 20)


def test_jsonschema_and_pydantic_fallback_agree(monkeypatch, validators):
    import builtins

    real_import = builtins.__import__

    def no_jsonschema(name, *a, **kw):
        if name == "jsonschema":
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_jsonschema)
    fallback = e2e.make_validators()
    for msg in [*good_stream(), {**prediction(9), "unexpected": 1}]:
        assert (validators[msg["type"]](msg) is None) == (fallback[msg["type"]](msg) is None), json.dumps(msg)[:80]
