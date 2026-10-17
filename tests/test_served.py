"""Served model behavior (needs trained artifacts: python -m valence.models.train_ladder)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from valence.config import Settings
from valence.features.compute import features_from_window
from valence.features.window import RollingWindow
from valence.governance.purge import PurgeGate
from valence.models.served import load_served
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime, ScenarioSource
from valence.sim.scenarios import render

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
served = load_served(ARTIFACTS, "core")
pytestmark = pytest.mark.skipif(served is None, reason="no trained core artifacts")


def _run(name):
    bus = EventBus()
    msgs = []
    bus.add_listener(lambda p: msgs.append(json.loads(p)))
    rt = Runtime(Settings(), bus, served, gate=PurgeGate(), audit_dir=tempfile.mkdtemp())
    df = render(name)
    rt.use_scenario(df, name)
    for _ in range(df["tick"].nunique() + 1):
        rt.step()
    rt.close()
    return [m for m in msgs if m["type"] == "prediction" and m["person_id"] == 1], df


def test_shap_additivity():
    df = render("frustrated_masked")
    src, w = ScenarioSource(df, "x"), RollingWindow(180)
    for _ in range(100):
        for r in src.next_tick(0, 0):
            w.append(r)
    xc, _ = served._clip(features_from_window(w.snapshot(1), w.room()))
    raw, contrib = served._m2(xc)
    np.testing.assert_allclose(contrib.sum(axis=1), raw, atol=1e-6)


def test_mask_override_fires_on_masked_and_not_elsewhere():
    preds, _ = _run("frustrated_masked")
    first = next(i for i, m in enumerate(preds) if m["mask_override"])
    assert first <= 48 + 30  # masked episode starts at tick 48; detected within 2.5 min
    assert preds[-1]["label_display"] == "Frustrated / Masked Stress"
    for name in ("focused", "collaborative_happy", "exhausted"):
        preds, _ = _run(name)
        assert not any(m["mask_override"] for m in preds), name


def test_scenarios_track_their_script():
    for name in ("focused", "exhausted", "collaborative_happy"):
        preds, df = _run(name)
        truth = df[df.person_id == 1].sort_values("tick").true_state.tolist()
        acc = np.mean([m["label"] == t for m, t in zip(preds, truth)])
        assert acc >= 0.9, (name, acc)


def test_counterfactual_changes_only_the_named_feature():
    df = render("frustrated_masked")
    src, w = ScenarioSource(df, "x"), RollingWindow(180)
    for _ in range(150):
        for r in src.next_tick(0, 0):
            w.append(r)
    out = served.counterfactual(w.snapshot(1), w.room(), "hours_since_break", 0.5)
    assert out["feature"] == "hours_since_break" and out["counterfactual_value"] == 0.5
    assert out["simulated_fields"] == ["hours_since_break"]
    assert abs(sum(out["counterfactual"]["probabilities"].values()) - 1) < 1e-3
