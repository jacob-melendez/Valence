from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from valence.narrator.fake_fact_sheets import (
    ANCHORS,
    STATE_RANGES,
    example_fact_sheet,
    make_batch,
    make_fact_sheet,
)
from valence.narrator.schema import LAYER_NAMES, STATES, face_only_label, iter_drivers, parse_fact_sheet

REPO_ROOT = Path(__file__).resolve().parents[2]
N = 600
SHEETS = make_batch(N, seed=0)


def _face_condition(sheet: dict) -> str:
    if not sheet["face_visible"]:
        return "hidden"
    return "agree" if sheet["face_only"]["label"] == sheet["label"] else "disagree"


def _drivers(sheet: dict) -> dict[str, dict]:
    return {d["feature"]: d for _, d in iter_drivers(sheet)}


def test_sheets_are_schema_valid():
    for sheet in SHEETS:
        parse_fact_sheet(sheet)
    parse_fact_sheet(example_fact_sheet())


def test_sheets_round_trip_through_json():
    assert json.loads(json.dumps(SHEETS[:20])) == SHEETS[:20]


def test_probabilities_are_consistent():
    for sheet in SHEETS:
        probs = sheet["probabilities"]
        assert tuple(probs) == STATES
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.values())
        assert max(probs, key=probs.get) == sheet["label"]
        assert sorted(probs.values())[-1] > sorted(probs.values())[-2]
        assert probs[sheet["label"]] == sheet["confidence"]
        assert sheet["confidence_display"] == f"{round(100 * sheet['confidence'])}%"


def test_mask_override_consistency():
    for sheet in SHEETS:
        face_only = sheet["face_only"]
        if sheet["mask_override"]:
            assert sheet["face_visible"]
            assert face_only["label"] not in (None, sheet["label"])
            assert face_only["face_valence"] > 0
            assert sheet["label"] != "Collaborative_Happy"
        if not sheet["face_visible"]:
            assert not sheet["mask_override"]
            assert face_only == {"label": None, "confidence": None, "face_valence": None}
            assert face_only_label(sheet) is None
            assert [d["feature"] for d in sheet["layers"][3]["drivers"]] == ["face_visible_frac"]
            assert sheet["data_quality"]["face_visible_frac"] <= 0.3
        else:
            assert face_only["label"] in STATES and face_only["face_valence"] is not None


def test_mask_cases_cover_expected_states():
    masked = Counter(s["label"] for s in SHEETS if s["mask_override"])
    disagree_frustrated = [s for s in SHEETS if s["label"] == "Frustrated_Masked" and _face_condition(s) == "disagree"]
    assert all(s["mask_override"] for s in disagree_frustrated)
    assert masked["Exhausted"] > 0 and masked["Collaborative_Happy"] == 0
    for sheet in SHEETS:
        if sheet["mask_override"]:
            assert sheet["face_only"]["label"] == "Collaborative_Happy"


def test_balance_across_states_and_face_conditions():
    assert Counter(s["label"] for s in SHEETS) == {state: N // 4 for state in STATES}
    assert Counter(_face_condition(s) for s in SHEETS) == {"hidden": N // 3, "agree": N // 3, "disagree": N // 3}
    combos = Counter((s["label"], _face_condition(s)) for s in SHEETS)
    assert len(combos) == 12 and set(combos.values()) == {N // 12}


def test_layers_and_contributions():
    for sheet in SHEETS:
        layers = sheet["layers"]
        assert tuple(layer["name"] for layer in layers) == LAYER_NAMES
        for layer in layers[:3]:
            assert 2 <= len(layer["drivers"]) <= 4
        assert 1 <= len(layers[3]["drivers"]) <= 2
        for layer in layers:
            shaps = [abs(d["shap"]) for d in layer["drivers"]]
            assert shaps == sorted(shaps, reverse=True)
            assert abs(layer["contribution"] - sum(shaps)) < 0.006
        total = sum(abs(d["shap"]) for _, d in iter_drivers(sheet))
        assert 0.9 < total < 1.1
        assert len({d["feature"] for _, d in iter_drivers(sheet)}) == len(iter_drivers(sheet))


def test_values_tell_the_state_story():
    for sheet in SHEETS:
        drivers = _drivers(sheet)
        assert set(ANCHORS[sheet["label"]]) <= set(drivers)
        for feature, (low, high) in STATE_RANGES[sheet["label"]].items():
            if feature in drivers:
                assert low <= drivers[feature]["value"] <= high, (sheet["label"], feature)
        if "light_lux" in drivers:
            assert drivers["light_lux"]["value"] == sheet["data_quality"]["lux"]
        if "face_visible_frac" in drivers:
            assert drivers["face_visible_frac"]["value"] == sheet["data_quality"]["face_visible_frac"]
        if "face_valence_mean" in drivers:
            assert drivers["face_valence_mean"]["value"] == sheet["face_only"]["face_valence"]
        if "session_min" in drivers and "hours_since_break" in drivers:
            assert drivers["session_min"]["value"] >= drivers["hours_since_break"]["value"] * 60


def test_anchor_drivers_dominate():
    anchored = 0
    for sheet in SHEETS:
        top = max(iter_drivers(sheet), key=lambda item: abs(item[1]["shap"]))[1]
        assert top["shap"] > 0
        anchored += top["feature"] in ANCHORS[sheet["label"]]
    assert anchored / N > 0.9


def test_some_sheets_carry_opposing_drivers():
    opposing = sum(any(d["shap"] < 0 for _, d in iter_drivers(s)) for s in SHEETS)
    assert opposing > N // 3


def test_face_valence_ranges_by_condition():
    for sheet in SHEETS:
        valence = sheet["face_only"]["face_valence"]
        if sheet["mask_override"]:
            assert 0.3 <= valence <= 0.8
        elif sheet["label"] == "Collaborative_Happy" and _face_condition(sheet) == "agree":
            assert valence >= 0.5


def test_feature_selection_varies():
    layouts = Counter(tuple(d["feature"] for _, d in iter_drivers(s)) for s in SHEETS)
    assert len(layouts) > 100
    seen = {d["feature"] for s in SHEETS for _, d in iter_drivers(s)}
    assert len(seen) >= 12


def test_batches_are_deterministic_per_seed():
    assert make_batch(20, seed=7) == make_batch(20, seed=7)
    assert make_batch(20, seed=7) != make_batch(20, seed=8)
    assert [s["event_id"] for s in make_batch(5)] == [f"evt_{i:06d}" for i in range(5)]


def test_overrides_steer_generation():
    rng = random.Random(3)
    hidden = make_fact_sheet(rng, label="Exhausted", face_visible=False, person="Person #7")
    assert hidden["label"] == "Exhausted" and not hidden["face_visible"] and hidden["person"] == "Person #7"
    masked = make_fact_sheet(rng, label="Exhausted", mask_override=True)
    assert masked["mask_override"] and masked["face_only"]["label"] == "Collaborative_Happy"
    agree = make_fact_sheet(rng, label="Focused", face_condition="agree", window_fill_min=9.5)
    assert agree["face_only"]["label"] == "Focused" and agree["window_fill_min"] == 9.5
    parse_fact_sheet(hidden), parse_fact_sheet(masked), parse_fact_sheet(agree)


def test_invalid_overrides_rejected():
    with pytest.raises(ValueError):
        make_fact_sheet(random.Random(0), label="Collaborative_Happy", face_condition="mask")
    with pytest.raises(ValueError):
        make_fact_sheet(random.Random(0), label="Angry")
    with pytest.raises(ValueError):
        make_fact_sheet(random.Random(0), face_condition="sideways")


def test_cli_writes_jsonl():
    result = subprocess.run(
        [sys.executable, "-m", "valence.narrator.fake_fact_sheets", "--n", "5", "--seed", "0"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=60,
    )
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 5
    assert [json.loads(line) for line in lines] == make_batch(5, seed=0)
