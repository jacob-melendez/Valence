"""Synthetic fact sheets for narrator development, prompt tuning and tests.

Each sheet tells a coherent story for its label: driver values come from per-state ranges,
SHAP mass concentrates on that state's anchor features, layer contributions are the layer's
share of |SHAP|, and the facial layer follows one of three face conditions:

- ``hidden``: ``face_visible: false``; every ``face_only`` field is null (label, confidence,
  face_valence) and the Facial Surface layer carries only the face-visibility driver;
- ``agree``: the face-only baseline predicts the same label;
- ``disagree``: the baseline predicts another state. For Frustrated_Masked (always) and
  Exhausted (half the time) the face reads positive and ``mask_override`` is true; otherwise the
  disagreement is a neutral or flat face and ``mask_override`` stays false.

CLI: ``python -m valence.narrator.fake_fact_sheets --n 100 --seed 0`` writes JSONL to stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from valence.narrator.schema import LAYER_NAMES, STATE_DISPLAY, STATES

AMBIENT, TIME, SOMATIC, FACIAL = LAYER_NAMES
FACE_CONDITIONS: tuple[str, ...] = ("hidden", "agree", "disagree")
MASKABLE_STATES = frozenset({"Frustrated_Masked", "Exhausted", "Focused"})
COUNTER_SIGNAL_RATE = 0.3


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature: str
    display: str
    unit: str
    layer: str
    decimals: int


CATALOGUE: tuple[FeatureSpec, ...] = (
    FeatureSpec("ambient_db", "Ambient noise", "dB", AMBIENT, 1),
    FeatureSpec("db_variability", "Noise variability", "dB", AMBIENT, 1),
    FeatureSpec("light_lux", "Light level", "lux", AMBIENT, 0),
    FeatureSpec("hours_since_break", "Time since last break", "h", TIME, 2),
    FeatureSpec("time_of_day_h", "Time of day", "h", TIME, 2),
    FeatureSpec("noise_trend_db", "Noise trend", "dB", TIME, 1),
    FeatureSpec("session_min", "Time in current session", "min", TIME, 0),
    FeatureSpec("fist_onsets_15m", "Clenched-fist onsets", "count", SOMATIC, 0),
    FeatureSpec("slump_frac", "Posture slump share", "%", SOMATIC, 2),
    FeatureSpec("head_in_hands_frac", "Head-in-hands share", "%", SOMATIC, 2),
    FeatureSpec("motion_per_min", "Movement", "/min", SOMATIC, 2),
    FeatureSpec("laughter_15m", "Laughter events", "count", SOMATIC, 0),
    FeatureSpec("compliments_15m", "Compliments heard", "count", SOMATIC, 0),
    FeatureSpec("speech_frac", "Speech share", "%", SOMATIC, 2),
    FeatureSpec("face_valence_mean", "Facial valence", "", FACIAL, 2),
    FeatureSpec("face_visible_frac", "Face visible share", "%", FACIAL, 2),
)
SPECS = {spec.feature: spec for spec in CATALOGUE}

Range = tuple[float, float]
STATE_RANGES: dict[str, dict[str, Range]] = {
    "Frustrated_Masked": {
        "ambient_db": (68, 85), "db_variability": (4, 9), "light_lux": (150, 300),
        "hours_since_break": (3.5, 6), "time_of_day_h": (13, 18), "noise_trend_db": (1, 6),
        "fist_onsets_15m": (1, 4), "slump_frac": (0.1, 0.3), "head_in_hands_frac": (0.0, 0.1),
        "motion_per_min": (2.5, 5), "laughter_15m": (0, 0), "compliments_15m": (0, 0),
        "speech_frac": (0.05, 0.25),
    },
    "Exhausted": {
        "ambient_db": (35, 50), "db_variability": (1, 3), "light_lux": (100, 200),
        "hours_since_break": (5, 8), "time_of_day_h": (16, 22), "noise_trend_db": (-3, 0.5),
        "fist_onsets_15m": (0, 0), "slump_frac": (0.5, 0.85), "head_in_hands_frac": (0.25, 0.6),
        "motion_per_min": (0.1, 0.8), "laughter_15m": (0, 0), "compliments_15m": (0, 0),
        "speech_frac": (0.0, 0.1),
    },
    "Collaborative_Happy": {
        "ambient_db": (60, 75), "db_variability": (3, 7), "light_lux": (400, 600),
        "hours_since_break": (0.2, 1.5), "time_of_day_h": (10, 16), "noise_trend_db": (0, 3),
        "fist_onsets_15m": (0, 0), "slump_frac": (0.0, 0.15), "head_in_hands_frac": (0.0, 0.03),
        "motion_per_min": (1.5, 4), "laughter_15m": (1, 5), "compliments_15m": (1, 3),
        "speech_frac": (0.4, 0.75),
    },
    "Focused": {
        "ambient_db": (40, 55), "db_variability": (1, 3), "light_lux": (350, 500),
        "hours_since_break": (1, 3), "time_of_day_h": (9, 15), "noise_trend_db": (-1, 1),
        "fist_onsets_15m": (0, 0), "slump_frac": (0.05, 0.25), "head_in_hands_frac": (0.0, 0.05),
        "motion_per_min": (0.1, 0.8), "laughter_15m": (0, 0), "compliments_15m": (0, 0),
        "speech_frac": (0.0, 0.15),
    },
}

STATE_WEIGHTS: dict[str, dict[str, float]] = {
    "Frustrated_Masked": {"ambient_db": 1.0, "fist_onsets_15m": 0.9, "hours_since_break": 0.8,
                          "noise_trend_db": 0.4, "db_variability": 0.4, "motion_per_min": 0.35, "light_lux": 0.3},
    "Exhausted": {"slump_frac": 1.0, "hours_since_break": 0.9, "head_in_hands_frac": 0.85,
                  "time_of_day_h": 0.45, "light_lux": 0.4, "motion_per_min": 0.35, "session_min": 0.3},
    "Collaborative_Happy": {"laughter_15m": 1.0, "compliments_15m": 0.8, "speech_frac": 0.7,
                            "ambient_db": 0.4, "light_lux": 0.35, "motion_per_min": 0.25},
    "Focused": {"motion_per_min": 0.9, "ambient_db": 0.8, "light_lux": 0.6,
                "speech_frac": 0.4, "hours_since_break": 0.4, "db_variability": 0.3},
}
ANCHORS: dict[str, tuple[str, ...]] = {
    state: tuple(sorted(weights, key=weights.get, reverse=True)[:3]) for state, weights in STATE_WEIGHTS.items()
}

AGREE_VALENCE: dict[str, Range] = {
    "Focused": (-0.25, 0.0), "Exhausted": (-0.2, 0.1),
    "Collaborative_Happy": (0.5, 0.9), "Frustrated_Masked": (-0.45, -0.1),
}
MASK_VALENCE: dict[str, Range] = {"Frustrated_Masked": (0.4, 0.8), "Exhausted": (0.3, 0.6), "Focused": (0.3, 0.6)}
DISAGREE_FACE: dict[str, tuple[str, Range]] = {
    "Focused": ("Exhausted", (-0.3, 0.05)),
    "Exhausted": ("Focused", (-0.15, 0.1)),
    "Collaborative_Happy": ("Focused", (-0.1, 0.25)),
    "Frustrated_Masked": ("Focused", (-0.2, 0.1)),
}


@dataclass(slots=True)
class _Face:
    visible: bool
    mask: bool
    label: str | None
    confidence: float | None
    valence: float | None
    visible_frac: float


def _uniform(rng: random.Random, bounds: Range, decimals: int) -> float:
    return round(rng.uniform(*bounds), decimals)


def _sample_world(rng: random.Random, label: str) -> dict[str, float]:
    world: dict[str, float] = {}
    for feature, bounds in STATE_RANGES[label].items():
        spec = SPECS[feature]
        if spec.unit == "count":
            world[feature] = rng.randint(int(bounds[0]), int(bounds[1]))
        else:
            world[feature] = _uniform(rng, bounds, spec.decimals)
    world["session_min"] = float(round(world["hours_since_break"] * 60 + rng.uniform(5, 45)))
    return world


def _resolve_condition(rng: random.Random, label: str, condition: str) -> str:
    if condition == "disagree":
        if label == "Frustrated_Masked" or (label == "Exhausted" and rng.random() < 0.5):
            return "mask"
        return "disagree"
    if condition == "mask" and label not in MASKABLE_STATES:
        raise ValueError(f"a positive-face mask cannot override {label}")
    if condition not in (*FACE_CONDITIONS, "mask"):
        raise ValueError(f"unknown face condition {condition!r}")
    return condition


def _sample_face(rng: random.Random, label: str, condition: str) -> _Face:
    condition = _resolve_condition(rng, label, condition)
    if condition == "hidden":
        return _Face(False, False, None, None, None, _uniform(rng, (0.0, 0.3), 2))
    visible_frac = _uniform(rng, (0.75, 1.0), 2)
    confidence = _uniform(rng, (0.5, 0.9), 2)
    if condition == "agree":
        return _Face(True, False, label, confidence, _uniform(rng, AGREE_VALENCE[label], 2), visible_frac)
    if condition == "mask":
        valence = _uniform(rng, MASK_VALENCE[label], 2)
        return _Face(True, True, "Collaborative_Happy", confidence, valence, visible_frac)
    face_label, bounds = DISAGREE_FACE[label]
    return _Face(True, False, face_label, confidence, _uniform(rng, bounds, 2), visible_frac)


def _pick_features(rng: random.Random, label: str, face: _Face) -> list[str]:
    anchors = set(ANCHORS[label])
    picked: list[str] = []
    for layer in (AMBIENT, TIME, SOMATIC):
        pool = [spec.feature for spec in CATALOGUE if spec.layer == layer]
        required = [f for f in pool if f in anchors]
        optional = [f for f in pool if f not in anchors]
        rng.shuffle(optional)
        k = rng.randint(max(2, len(required)), min(4, len(pool)))
        picked += required + optional[: k - len(required)]
    if not face.visible:
        picked.append("face_visible_frac")
    else:
        picked.append("face_valence_mean")
        if rng.random() < 0.5:
            picked.append("face_visible_frac")
    return picked


def _raw_shap(rng: random.Random, label: str, feature: str, face: _Face) -> float:
    """Unnormalised SHAP: anchors dominate, a masking face pushes against the label."""
    if feature == "face_valence_mean":
        if face.mask:
            return -rng.uniform(0.15, 0.9)
        if face.label == label:
            return rng.uniform(0.3, 0.9) if label == "Collaborative_Happy" else rng.uniform(0.05, 0.25)
        return -rng.uniform(0.05, 0.2)
    if feature == "face_visible_frac":
        magnitude = rng.uniform(0.05, 0.8) if not face.visible else rng.uniform(0.02, 0.12)
        return magnitude if rng.random() < 0.6 else -magnitude
    weight = STATE_WEIGHTS[label].get(feature)
    if weight is not None:
        return weight * (rng.uniform(0.6, 1.3) if feature in ANCHORS[label] else rng.uniform(0.6, 1.8))
    magnitude = rng.uniform(0.02, 0.15)
    return -magnitude if rng.random() < 0.3 else magnitude


def _counter_signal(rng: random.Random, label: str, features: list[str], raw: dict[str, float]) -> None:
    """Sometimes one context feature pushes clearly against the label, as real SHAP often does."""
    pool = [f for f in features if f not in ANCHORS[label] and SPECS[f].layer != FACIAL]
    if pool and rng.random() < COUNTER_SIGNAL_RATE:
        raw[rng.choice(pool)] = -rng.uniform(0.3, 0.8)


def _keep_story(raw: dict[str, float]) -> None:
    """The single strongest driver always supports the label; opposing ones stay below it."""
    strongest = max(raw.values())
    for feature, value in raw.items():
        if value < -0.9 * strongest:
            raw[feature] = -0.9 * strongest


def _build_layers(rng: random.Random, label: str, world: dict[str, float], face: _Face) -> list[dict[str, Any]]:
    features = _pick_features(rng, label, face)
    raw = {f: _raw_shap(rng, label, f, face) for f in features}
    _counter_signal(rng, label, features, raw)
    _keep_story(raw)
    total = sum(abs(v) for v in raw.values())
    layers: list[dict[str, Any]] = []
    for layer in LAYER_NAMES:
        drivers = []
        for feature in (f for f in features if SPECS[f].layer == layer):
            spec = SPECS[feature]
            shap = round(raw[feature] / total, 2)
            if abs(shap) < 0.01:
                shap = 0.01 if raw[feature] >= 0 else -0.01
            drivers.append({
                "feature": feature, "display": spec.display, "value": world[feature],
                "unit": spec.unit, "shap": shap,
            })
        drivers.sort(key=lambda d: -abs(d["shap"]))
        contribution = round(sum(abs(d["shap"]) for d in drivers), 2)
        layers.append({"name": layer, "contribution": contribution, "drivers": drivers})
    return layers


def _sample_probabilities(rng: random.Random, label: str) -> tuple[float, dict[str, float]]:
    confidence = _uniform(rng, (0.55, 0.95), 2)
    others = [s for s in STATES if s != label]
    weights = [rng.random() + 0.05 for _ in others]
    rest = round(1.0 - confidence, 2)
    shares = [round(rest * w / sum(weights), 2) for w in weights]
    largest = max(range(len(shares)), key=shares.__getitem__)
    shares[largest] = round(shares[largest] + rest - sum(shares), 2)
    probabilities = dict(zip(others, shares, strict=True))
    probabilities[label] = confidence
    return confidence, {s: probabilities[s] for s in STATES}


def make_fact_sheet(rng: random.Random, **overrides: Any) -> dict[str, Any]:
    """One schema-valid, internally consistent fact sheet.

    ``label`` and ``face_condition`` ("hidden" | "agree" | "disagree" | "mask") steer the story;
    ``face_visible=False`` implies "hidden" and ``mask_override=True`` implies "mask". Any other
    keyword replaces the top-level field of the same name after generation.
    """
    overrides = dict(overrides)
    label = overrides.pop("label", None) or rng.choice(STATES)
    if label not in STATES:
        raise ValueError(f"unknown state {label!r}")
    condition = overrides.pop("face_condition", None)
    if condition is None and overrides.get("face_visible") is False:
        condition = "hidden"
    if condition is None and overrides.get("mask_override") is True:
        condition = "mask"
    face = _sample_face(rng, label, condition or rng.choice(FACE_CONDITIONS))
    world = _sample_world(rng, label)
    world["face_visible_frac"] = face.visible_frac
    if face.valence is not None:
        world["face_valence_mean"] = face.valence
    layers = _build_layers(rng, label, world, face)
    confidence, probabilities = _sample_probabilities(rng, label)
    fill = 15.0 if rng.random() < 0.8 else _uniform(rng, (5.0, 14.9), 1)
    sheet: dict[str, Any] = {
        "event_id": f"evt_{rng.randrange(1_000_000):06d}",
        "person": f"Person #{rng.randint(1, 4)}",
        "label": label,
        "label_display": STATE_DISPLAY[label],
        "confidence": confidence,
        "confidence_display": f"{round(100 * confidence)}%",
        "probabilities": probabilities,
        "window_fill_min": fill,
        "face_only": {"label": face.label, "confidence": face.confidence, "face_valence": face.valence},
        "face_visible": face.visible,
        "mask_override": face.mask,
        "layers": layers,
        "data_quality": {"face_visible_frac": face.visible_frac, "lux": world["light_lux"]},
    }
    sheet.update(overrides)
    return sheet


def make_batch(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` sheets balanced across states x face conditions (shuffled, deterministic per seed)."""
    rng = random.Random(seed)
    plan = [(STATES[i % len(STATES)], FACE_CONDITIONS[(i // len(STATES)) % len(FACE_CONDITIONS)]) for i in range(n)]
    rng.shuffle(plan)
    return [
        make_fact_sheet(rng, label=label, face_condition=condition, event_id=f"evt_{i:06d}")
        for i, (label, condition) in enumerate(plan)
    ]


def example_fact_sheet() -> dict[str, Any]:
    """The reference sheet from the narrator contract (the PRD's masked-frustration example)."""
    return {
        "event_id": "evt_000123",
        "person": "Person #1",
        "label": "Frustrated_Masked",
        "label_display": "Frustrated (Masked)",
        "confidence": 0.88,
        "confidence_display": "88%",
        "probabilities": {"Focused": 0.04, "Frustrated_Masked": 0.88, "Exhausted": 0.05, "Collaborative_Happy": 0.03},
        "window_fill_min": 15.0,
        "face_only": {"label": "Collaborative_Happy", "confidence": 0.71, "face_valence": 0.62},
        "face_visible": True,
        "mask_override": True,
        "layers": [
            {"name": AMBIENT, "contribution": 0.41, "drivers": [
                {"feature": "ambient_db", "display": "Ambient noise", "value": 72.3, "unit": "dB", "shap": 0.31},
                {"feature": "light_lux", "display": "Light level", "value": 180.0, "unit": "lux", "shap": 0.10},
            ]},
            {"name": TIME, "contribution": 0.22, "drivers": [
                {"feature": "hours_since_break", "display": "Time since last break", "value": 4.2, "unit": "h",
                 "shap": 0.22},
            ]},
            {"name": SOMATIC, "contribution": 0.30, "drivers": [
                {"feature": "fist_onsets_15m", "display": "Clenched-fist onsets", "value": 3, "unit": "count",
                 "shap": 0.27},
                {"feature": "compliments_15m", "display": "Compliments heard", "value": 0, "unit": "count",
                 "shap": 0.03},
            ]},
            {"name": FACIAL, "contribution": 0.07, "drivers": [
                {"feature": "face_valence_mean", "display": "Facial valence", "value": 0.62, "unit": "",
                 "shap": -0.07},
            ]},
        ],
        "data_quality": {"face_visible_frac": 0.93, "lux": 180.0},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic narrator fact sheets as JSONL.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for sheet in make_batch(args.n, args.seed):
        sys.stdout.write(json.dumps(sheet) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
