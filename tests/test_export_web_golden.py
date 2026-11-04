"""scripts/export_web_golden.py: the golden-fixture exporter for the TypeScript engine stays runnable and deterministic.

Runs the exporter's building blocks on tiny cases (a few ticks, one short synthetic stream) so a change in the
Python engine that breaks the exporter shows up here, not when someone regenerates the web fixtures.
"""

import gzip
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_web_golden", ROOT / "scripts" / "export_web_golden.py")
golden = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = golden
SPEC.loader.exec_module(golden)

needs_models = pytest.mark.skipif(not (ROOT / "artifacts" / "m2.txt").exists(), reason="no trained core artifacts")


def test_encoding_is_byte_identical_and_nan_safe():
    obj = {"a": math.nan, "b": 0.1, "c": np.float64(1e-05), "d": (1, 2.0), "e": np.array([np.nan, -0.0])}
    data = golden.encode_fixture(obj)
    assert data == golden.encode_fixture(obj)  # gzip mtime=0
    text = gzip.decompress(data).decode()
    assert text == '{"a":null,"b":0.1,"c":1e-05,"d":[1,2.0],"e":[null,-0.0]}'
    with pytest.raises(ValueError):
        golden.encode_fixture({"x": math.inf})


def test_features_scenario_checkpoints_match_a_replay_of_its_records():
    sc = golden.features_scenario("full_demo", "core", False, max_ticks=8)
    assert len(sc["ticks"]) == 8 and sc["ticks"][0]["wall"] == golden.WALL0
    assert all(t["records"][0]["person_id"] == 0 for t in sc["ticks"])  # room row first
    assert [c["tick"] for c in sc["checkpoints"]] == [0, 1, 2, 4, 6, 7]  # fresh window, even ticks, last tick
    from valence.features.compute import features_from_window
    from valence.features.window import RollingWindow

    window = RollingWindow(180)
    for tick in sc["ticks"][:5]:
        for d in tick["records"]:
            window.append(golden.record_from_dict(d))
    x = features_from_window(window.snapshot(1), window.room())
    assert golden._same(x, sc["checkpoints"][3]["features"]) and len(x) == 123


def test_aggregator_replay_is_deterministic_and_utc():
    from valence.config import Settings

    obs = golden.synth_lux_sampling()
    case = golden.aggregator_case("tiny", "synthetic", obs, Settings())
    assert case["ticks"][0]["wall"] == golden.WALL0 + 5.0
    room = case["ticks"][0]["records"][0]
    assert room["person_id"] == 0 and room["tod_h"] == pytest.approx(14.0 + 5 / 3600)  # 14:00:05 UTC
    again = golden.replay_aggregator([golden.obs_from_dict(d) for d in case["obs"]], Settings())
    assert golden._same(again.ticks, case["ticks"]) and golden._same(again.checkpoints, case["checkpoints"])


@needs_models
def test_model_cases_are_additive_and_cover_the_no_face_path():
    served = golden.served_for("core")
    (name, x, tod), *_ = golden.scenario_windows("frustrated_masked", "core", every=10, max_ticks=20)
    case = golden.model_case(served, name, x, tod)
    np.testing.assert_allclose(np.sum(case["m2_contrib"], axis=1), case["m2_raw"], atol=1e-9)
    assert case["k"] == int(np.argmax(case["p"])) and case["m0"] is not None
    perturbed = dict(golden.perturbations(served, name, x))
    blind = golden.model_case(served, "all_nan", perturbed[f"{name}/all_nan"], tod)
    assert blind["m0"] is None and blind["raw_override"] is None


def test_audit_entries_carry_the_hashed_blob_and_float_pointers(tmp_path):
    _, entries = golden.run_audit(golden.AUDIT_CASES["tricky_floats"], str(tmp_path))
    e = entries[0]
    assert hashlib.sha256((e["prev"] + e["blob"]).encode()).hexdigest() == e["hash"]
    assert e["ts"] == round(golden.WALL0 + 1.0, 3) and e["prev"] == golden.GENESIS
    assert {"/ts", "/payload/a", "/payload/d", "/payload/e"} <= set(e["floats"])
    assert "/payload/p" not in e["floats"] and "/payload/y" not in e["floats"]  # int 0 and True
    assert '"a":1e-05' in e["blob"] and '"b":1e+16' in e["blob"] and '"d":2.0' in e["blob"]


@needs_models
def test_runtime_run_captures_messages_in_emission_order():
    run = golden.runtime_run("tiny", "frustrated_masked", "core", False, max_ticks=3)
    assert [m["type"] for m in run["ticks"][0]["messages"]] == ["governance", "state_vector", "prediction"]
    assert run["ticks"][0]["messages"][0]["audit_hash"] == golden.GENESIS[:16]
    assert run["audit"][0]["ts"] == golden.WALL0 and run["audit"][0]["kind"] == "prediction"
    assert run["ticks"][1]["messages"][0]["audit_hash"] == run["audit"][0]["hash"][:16]
    assert run["hello"].feature_version == run["model_id"]
    again = golden.runtime_run("tiny", "frustrated_masked", "core", False, max_ticks=3)
    assert golden.encode_fixture(run) == golden.encode_fixture(again)


def test_describe_rows_and_main_write_reloadable_fixtures(tmp_path, capsys):
    rows = golden.describe_grid(["db_room_mean_5m", "tod_sin"], values=(0.5, math.nan))
    assert {r["feature"] for r in rows} == {"db_room_mean_5m", "tod_sin"}
    assert all(r["out"] is None for r in rows if r["value"] is not None and math.isnan(r["value"]))
    golden.main(["--out", str(tmp_path), "--only", "audit"])
    first = (tmp_path / "audit.json.gz").read_bytes()
    golden.main(["--out", str(tmp_path), "--only", "audit"])
    assert (tmp_path / "audit.json.gz").read_bytes() == first
    fx = json.loads(gzip.decompress(first))
    assert fx["version"] == 1 and fx["cases"] and "check audit" in capsys.readouterr().out
