from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from valence.contracts import ALL_STATES, CORE_STATES, LAYERS, FactSheet, display_label
from valence.explain.describe import describe
from valence.features.spec import layer_of, scope_of
from valence.narrator.fake_fact_sheets import (
    FACE_CONDITIONS,
    FEATURE_NAMES,
    example_fact_sheet,
    make_batch,
    make_fact_sheet,
)
from valence.narrator.schema import iter_drivers, parse_fact_sheet

REPO_ROOT = Path(__file__).resolve().parents[2]
N = 720
BATCHES = {"core": make_batch(N, seed=0, states="core"), "all": make_batch(N, seed=0, states="all")}
STATES = {"core": CORE_STATES, "all": ALL_STATES}


def _shape(display: str) -> str:
    return re.sub(r"[+-]?\d+(?:\.\d+)?", "#", display)


def _condition(sheet: dict) -> str:
    if not sheet["face_visible"]:
        return "hidden"
    return "disagree" if sheet["mask_override"] else "agree"


def test_feature_names_are_the_served_vector():
    from valence.features.compute import FEATURE_NAMES as served
    from valence.features.compute import ROOM_REL

    assert FEATURE_NAMES == tuple(f for f in served if f not in ROOM_REL)  # venue-relative features: CLOUD round 7
    assert {"db_mean_15m", "fist_cnt_5m", "fist_since_s", "face_visible_frac_5m", "tod_sin", "fill_ratio",
            "withdraw_now", "n_people", "lux_rel_15m", "hih_d1m15m", "face_visible_now"} <= set(FEATURE_NAMES)


@pytest.mark.parametrize("states", ["core", "all"])
def test_sheets_are_contract_valid_and_round_trip(states):
    for sheet in BATCHES[states]:
        model = FactSheet.model_validate(sheet)
        assert model.model_dump(mode="json") == sheet
    assert json.loads(json.dumps(BATCHES[states][:20])) == BATCHES[states][:20]
    parse_fact_sheet(example_fact_sheet(states))


@pytest.mark.parametrize("states", ["core", "all"])
def test_drivers_match_the_core_describe(states):
    for sheet in BATCHES[states]:
        for layer, d in iter_drivers(sheet):
            assert d["feature"] in FEATURE_NAMES and d["feature"] != "fill_ratio"
            assert layer == layer_of(d["feature"]) and d["scope"] == scope_of(d["feature"])
            if abs(d["shap"]) > 1e-4:  # effect comes from the raw SHAP sign, shap is stored rounded
                assert d["effect"] == ("supports" if d["shap"] > 0 else "opposes")
            if not d["feature"].startswith("tod_"):
                # build_layers describes the raw value and stores it rounded to 4 decimals, so only
                # compare the wording (numbers masked) and the unit, within that rounding.
                shapes = {(_shape(x.display), x.unit) for v in (d["value"], d["value"] - 1e-4, d["value"] + 1e-4)
                          if (x := describe(d["feature"], v)) is not None}
                assert (_shape(d["display"]), d["unit"]) in shapes, d
        assert [layer["name"] for layer in sheet["layers"]] == sorted(
            (layer["name"] for layer in sheet["layers"]), key=lambda n: -next(
                x["contribution"] for x in sheet["layers"] if x["name"] == n))
        assert all(len(layer["drivers"]) <= 3 for layer in sheet["layers"])
        assert {layer["name"] for layer in sheet["layers"]} <= set(LAYERS)


@pytest.mark.parametrize("states", ["core", "all"])
def test_probabilities_and_labels(states):
    for sheet in BATCHES[states]:
        probs = sheet["probabilities"]
        assert tuple(probs) == STATES[states]
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)
        assert max(probs, key=probs.get) == sheet["label"]
        assert sorted(probs.values())[-1] > sorted(probs.values())[-2]
        assert probs[sheet["label"]] == sheet["confidence"]
        assert sheet["confidence_display"] == f"{round(100 * sheet['confidence'])}%"
        assert sheet["label_display"] == display_label(sheet["label"], mask_override=sheet["mask_override"],
                                                       n_people=sheet["n_people"], states=STATES[states])
        assert 1 <= sheet["n_people"] <= 3
        assert 1 <= int(sheet["person"].removeprefix("Person #")) <= sheet["n_people"]


@pytest.mark.parametrize("states", ["core", "all"])
def test_balance_across_states_and_face_conditions(states):
    combos = Counter((s["label"], _condition(s)) for s in BATCHES[states])
    n_combos = len(STATES[states]) * len(FACE_CONDITIONS)
    assert len(combos) == n_combos and set(combos.values()) == {N // n_combos}


@pytest.mark.parametrize("states", ["core", "all"])
def test_face_conditions(states):
    for sheet in BATCHES[states]:
        face_only = sheet["face_only"]
        features = {d["feature"] for _, d in iter_drivers(sheet)}
        if not sheet["face_visible"]:
            assert not sheet["mask_override"] or sheet["label"] != "Collaborative_Happy"
            assert face_only == {"label": None, "confidence": None, "face_valence": None}
            assert not {f for f in features if f.startswith("face_") and not f.startswith("face_visible")}
            assert sheet["data_quality"]["face_visible_frac"] <= 0.3
        elif sheet["mask_override"]:
            assert face_only["label"] not in (None, sheet["label"])
            face = [d for _, d in iter_drivers(sheet) if d["feature"].startswith("face_")
                    and not d["feature"].startswith("face_visible")]
            assert face and max(face, key=lambda d: abs(d["shap"]))["effect"] == "opposes"
        else:
            assert face_only["label"] in ALL_STATES and face_only["face_valence"] is not None
            if face_only["label"] != sheet["label"]:
                assert face_only["confidence"] <= 0.5


def test_production_quirks_are_covered():
    sheets = BATCHES["core"]
    assert any(not s["face_visible"] and s["mask_override"] for s in sheets)
    assert any(s["face_visible"] and not s["mask_override"] and s["face_only"]["label"] != s["label"] for s in sheets)


def test_ood_features_and_people_vary():
    sheets = BATCHES["all"]
    with_ood = [s for s in sheets if s["data_quality"]["ood_features"]]
    assert 0.05 * N < len(with_ood) < 0.3 * N
    for sheet in with_ood:
        assert set(sheet["data_quality"]["ood_features"]) <= {d["feature"] for _, d in iter_drivers(sheet)}
    assert set(Counter(s["n_people"] for s in sheets)) == {1, 2, 3}
    assert any(s["label_display"] == "Upbeat" for s in sheets)
    assert any(s["label_display"] == "Frustrated" for s in BATCHES["core"])
    assert not any(s["label_display"] == "Frustrated" for s in sheets)


def test_anchor_drivers_lead():
    top_is_support = sum(max(iter_drivers(s), key=lambda x: abs(x[1]["shap"]))[1]["effect"] == "supports"
                         for s in BATCHES["core"])
    assert top_is_support / N > 0.9
    assert sum(any(d["effect"] == "opposes" for _, d in iter_drivers(s)) for s in BATCHES["core"]) > N // 3


def test_batches_are_deterministic_per_seed():
    assert make_batch(20, seed=7) == make_batch(20, seed=7)
    assert make_batch(20, seed=7) != make_batch(20, seed=8)
    assert [s["event_id"] for s in make_batch(5)] == [f"evt_{i:06d}" for i in range(5)]


def test_overrides_steer_generation():
    rng = random.Random(3)
    hidden = make_fact_sheet(rng, label="Exhausted", face_condition="hidden", person_id=1, n_people=1)
    assert hidden["label"] == "Exhausted" and not hidden["face_visible"] and hidden["person"] == "Person #1"
    tense = make_fact_sheet(rng, label="Tense", face_condition="disagree", states="all", ood=True)
    assert tense["mask_override"] and tense["data_quality"]["ood_features"]
    with pytest.raises(ValueError):
        make_fact_sheet(rng, label="Tense", states="core")
    with pytest.raises(ValueError):
        make_fact_sheet(rng, face_condition="sideways")


def test_example_sheet_is_the_prd_case():
    sheet = example_fact_sheet()
    assert sheet["label_display"] == "Frustrated / Masked Stress" and sheet["confidence_display"] == "88%"
    displays = [d["display"] for _, d in iter_drivers(sheet)]
    assert "average noise of 72 dB over the last 15 minutes" in displays
    assert "3 clenched-fist events in the last 5 minutes" in displays
    assert "4.2 hours since the last break" in displays


@pytest.mark.parametrize("states", ["core", "all"])
def test_cli_writes_jsonl(states):
    result = subprocess.run(
        [sys.executable, "-m", "valence.narrator.fake_fact_sheets", "--n", "5", "--seed", "0", "--states", states],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=60,
    )
    assert [json.loads(line) for line in result.stdout.strip().splitlines()] == make_batch(5, 0, states)
