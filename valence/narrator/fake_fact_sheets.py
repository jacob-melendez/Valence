"""Synthetic contract fact sheets for narrator development, prompt tuning and tests.

Sheets are built the way production builds them: a value and a SHAP attribution for every real
feature name (``valence.features.compute.FEATURE_NAMES``, layered by ``valence.features.spec``),
passed through the core's own ``build_layers`` (which calls
``valence.explain.describe.describe``) and ``build_fact_sheet``. So display strings, units, levels,
scopes, effects (the SHAP sign) and per-layer truncation match production exactly; only the numbers
are invented.

Each sheet tells a coherent story for its label: channel levels come from per-state ranges and the
state's anchor channels carry most of the SHAP mass. Face conditions:

- ``agree``: the face-only baseline predicts the label and face features support it (a quarter of
  these carry a weak face-only disagreement that does not trigger the override, as in production);
- ``disagree``: the baseline predicts another state from the face and ``mask_override`` is true
  (a positive-looking face for every label except Collaborative_Happy, whose baseline reads a
  neutral face as Focused); face features oppose the label;
- ``hidden``: ``face_visible`` false, face features NaN (so no valence drivers) and the face-only
  fields null, as the contract requires; a fifth keep a latched ``mask_override`` (also seen in the
  core's exported sheets).

CLI: ``python -m valence.narrator.fake_fact_sheets --n 100 --seed 0 [--states core|all]`` writes JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from typing import Any

from valence.contracts import CORE_STATES, STATE_SETS, display_label
from valence.explain.explainer import build_fact_sheet, build_layers
from valence.features.compute import FEATURE_NAMES as COMPUTED_FEATURES

SMOOTH_CHANNELS: tuple[str, ...] = ("db", "lux", "motion", "slump", "hih")
EVENT_CHANNELS: tuple[str, ...] = ("fist", "laugh", "compl")
WINDOWS: tuple[str, ...] = ("1m", "5m", "15m")

FEATURE_NAMES: tuple[str, ...] = tuple(COMPUTED_FEATURES)  # the served model's frozen vector
FACE_CONDITIONS: tuple[str, ...] = ("agree", "disagree", "hidden")

Range = tuple[float, float]
# Per-state channel levels: dB SPL, lux, motion (shoulder-widths/s), slump/hih fractions,
# events per 5 minutes, hours since break, time of day, face valence when the face agrees.
STATE_WORLD: dict[str, dict[str, Range]] = {
    "Focused": {"db": (42, 56), "lux": (300, 550), "motion": (0.03, 0.10), "slump": (0.0, 0.2), "hih": (0.0, 0.08),
                "fist": (0, 0.3), "laugh": (0, 0.3), "compl": (0, 0.2), "break": (0.5, 2.5), "tod": (9, 16),
                "valence": (-0.15, 0.1)},
    "Frustrated_Masked": {"db": (60, 78), "lux": (180, 400), "motion": (0.10, 0.22), "slump": (0.05, 0.3),
                          "hih": (0.05, 0.3), "fist": (1, 5), "laugh": (0, 0.3), "compl": (0, 0.2), "break": (2.5, 5.5),
                          "tod": (13, 19), "valence": (-0.45, -0.1)},
    "Exhausted": {"db": (38, 52), "lux": (90, 220), "motion": (0.01, 0.06), "slump": (0.45, 0.9), "hih": (0.25, 0.6),
                  "fist": (0, 0.3), "laugh": (0, 0.2), "compl": (0, 0.2), "break": (3.5, 7.0), "tod": (16, 23.5),
                  "valence": (-0.3, 0.05)},
    "Collaborative_Happy": {"db": (58, 72), "lux": (300, 600), "motion": (0.10, 0.25), "slump": (0.0, 0.15),
                            "hih": (0.0, 0.05), "fist": (0, 0.2), "laugh": (1.5, 5), "compl": (0.5, 3),
                            "break": (0.2, 1.8), "tod": (10, 17), "valence": (0.35, 0.85)},
    "Overwhelmed": {"db": (66, 82), "lux": (150, 350), "motion": (0.15, 0.35), "slump": (0.2, 0.6), "hih": (0.35, 0.75),
                    "fist": (1, 4), "laugh": (0, 0.2), "compl": (0, 0.2), "break": (2.5, 6.0), "tod": (14, 23),
                    "valence": (-0.7, -0.25)},
    "Tense": {"db": (58, 74), "lux": (200, 450), "motion": (0.15, 0.32), "slump": (0.0, 0.25), "hih": (0.0, 0.15),
              "fist": (2, 6), "laugh": (0, 0.3), "compl": (0, 0.2), "break": (2.0, 5.0), "tod": (11, 19),
              "valence": (-0.7, -0.3)},
}
WITHDRAW_P: dict[str, float] = {"Exhausted": 0.5, "Overwhelmed": 0.4}
# Anchor channels and their SHAP weight for each state (log-odds scale, before jitter).
ANCHORS: dict[str, dict[str, float]] = {
    "Focused": {"motion": 0.5, "db": 0.45, "lux": 0.3, "break": 0.25},
    "Frustrated_Masked": {"db": 0.55, "fist": 0.5, "break": 0.4, "motion": 0.2},
    "Exhausted": {"slump": 0.55, "hih": 0.45, "break": 0.45, "tod": 0.25, "motion": 0.25, "withdraw": 0.2},
    "Collaborative_Happy": {"laugh": 0.6, "compl": 0.45, "n_people": 0.35, "db": 0.2},
    "Overwhelmed": {"hih": 0.55, "db": 0.45, "motion": 0.35, "fist": 0.3, "break": 0.25},
    "Tense": {"fist": 0.6, "motion": 0.4, "db": 0.3, "face": 0.3},
}
MASK_VALENCE: Range = (0.35, 0.8)
NEUTRAL_VALENCE: Range = (-0.15, 0.1)
OOD_RATE = 0.15
LATCHED_MASK_RATE = 0.2  # override still latched after the face went out of view (seen in production)
SOFT_DISAGREE_RATE = 0.25  # weak face-only disagreement that does not trigger the override
COUNTER_SIGNAL_RATE = 0.35

_STD = {"db": (1.0, 7.0), "lux": (5.0, 40.0), "motion": (0.01, 0.08), "slump": (0.05, 0.3), "hih": (0.05, 0.3)}
_SLOPE = {"db": 0.6, "lux": 6.0, "motion": 0.02, "slump": 0.05, "hih": 0.05}
_JITTER = {"db": 2.0, "lux": 25.0, "motion": 0.02, "slump": 0.05, "hih": 0.04}
_DELTA = {"db": 3.0, "lux": 30.0, "motion": 0.03, "slump": 0.1, "hih": 0.08}


def channel_of(feature: str) -> str:
    """Anchor channel of a feature name ("db_mean_15m" -> "db", "tod_sin" -> "tod")."""
    if feature == "hours_since_break":
        return "break"
    if feature in ("tod_sin", "tod_cos"):
        return "tod"
    if feature == "n_people":
        return "n_people"
    return feature.split("_", 1)[0]


def _u(rng: random.Random, bounds: Range) -> float:
    return rng.uniform(*bounds)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _smooth_values(rng: random.Random, ch: str, base: float) -> dict[str, float]:
    frac = ch in ("slump", "hih")

    def clip(v: float) -> float:
        if frac:
            return min(max(v, 0.0), 1.0)
        return max(v, 5.0 if ch == "lux" else 0.0)

    jitter = _JITTER[ch]
    out = {f"{ch}_now": clip(base + rng.gauss(0, jitter))}
    for w in WINDOWS:
        out[f"{ch}_mean_{w}"] = clip(base + rng.gauss(0, jitter / 2))
        out[f"{ch}_std_{w}"] = _u(rng, _STD[ch])
        out[f"{ch}_slope_{w}"] = rng.gauss(0, _SLOPE[ch])
        out[f"{ch}_max_{w}"] = clip(base + abs(rng.gauss(0, jitter * 2)))
    for w in ("1m", "5m"):
        out[f"{ch}_ewm_{w}"] = clip(base + rng.gauss(0, jitter / 2))
    out[f"{ch}_d1m15m"] = rng.gauss(0, _DELTA[ch])
    out[f"{ch}_rel_15m"] = rng.gauss(0, _DELTA[ch] * 1.5)
    return out


def _event_values(rng: random.Random, ch: str, per5: float) -> dict[str, float]:
    now = _poisson(rng, per5 / 60)
    c1 = now + _poisson(rng, per5 * 11 / 60)
    c5 = c1 + _poisson(rng, per5 * 4 / 5)
    c15 = c5 + _poisson(rng, per5 * 2)
    if c15 == 0:
        since = 900.0
    elif now:
        since = 0.0
    elif c1:
        since = 5.0 * rng.randint(1, 11)
    elif c5:
        since = 5.0 * rng.randint(12, 59)
    else:
        since = 5.0 * rng.randint(60, 179)
    return {f"{ch}_cnt_now": float(now), f"{ch}_cnt_1m": float(c1), f"{ch}_cnt_5m": float(c5),
            f"{ch}_cnt_15m": float(c15), f"{ch}_since_s": since}


def _sample_values(rng: random.Random, label: str, n_people: int,
                   face: dict[str, Any]) -> tuple[dict[str, float], float]:
    world = STATE_WORLD[label]
    values: dict[str, float] = {}
    for ch in SMOOTH_CHANNELS:
        values.update(_smooth_values(rng, ch, _u(rng, world[ch])))
    for ch in EVENT_CHANNELS:
        values.update(_event_values(rng, ch, _u(rng, world[ch])))
    withdrawn = rng.random() < WITHDRAW_P.get(label, 0.05)
    values["withdraw_now"] = 1.0 if withdrawn else 0.0
    values["withdraw_frac_5m"] = _u(rng, (0.5, 1.0)) if withdrawn else _u(rng, (0.0, 0.2))
    values["n_people"] = float(n_people)
    values["hours_since_break"] = _u(rng, world["break"])
    tod_h = _u(rng, world["tod"]) % 24
    values["tod_sin"] = math.sin(2 * math.pi * tod_h / 24)
    values["tod_cos"] = math.cos(2 * math.pi * tod_h / 24)
    values["fill_ratio"] = 1.0
    valence = face["valence"]
    for f in ("face_now", "face_mean_1m", "face_mean_5m", "face_mean_15m"):
        values[f] = math.nan if valence is None else min(max(valence + rng.gauss(0, 0.05), -1.0), 1.0)
    values["face_std_5m"] = math.nan if valence is None else _u(rng, (0.03, 0.3))
    values["face_visible_now"] = 1.0 if face["visible"] else 0.0
    values["face_visible_frac_5m"] = face["visible_frac"]
    return values, tod_h


def _sample_face(rng: random.Random, label: str, condition: str) -> dict[str, Any]:
    if condition == "hidden":
        latched = label != "Collaborative_Happy" and rng.random() < LATCHED_MASK_RATE
        return {"visible": False, "mask": latched, "label": None, "confidence": None, "valence": None,
                "visible_frac": round(_u(rng, (0.0, 0.3)), 3)}
    face = {"visible": True, "confidence": round(_u(rng, (0.55, 0.9)), 2),
            "visible_frac": round(_u(rng, (0.6, 1.0)), 3)}
    if condition == "agree":
        valence = round(_u(rng, STATE_WORLD[label]["valence"]), 2)
        if rng.random() < SOFT_DISAGREE_RATE:
            other = rng.choice([s for s in STATE_WORLD if s != label and s in CORE_STATES])
            return {**face, "mask": False, "label": other, "confidence": round(_u(rng, (0.25, 0.5)), 2),
                    "valence": valence}
        return {**face, "mask": False, "label": label, "valence": valence}
    if condition == "disagree":
        if label == "Collaborative_Happy":
            return {**face, "mask": True, "label": "Focused", "valence": round(_u(rng, NEUTRAL_VALENCE), 2)}
        return {**face, "mask": True, "label": "Collaborative_Happy", "valence": round(_u(rng, MASK_VALENCE), 2)}
    raise ValueError(f"unknown face condition {condition!r}")


def _is_level(feature: str) -> bool:
    """Level-type statistics (now/mean/max/ewm/counts), the ones a state is anchored on.
    ``*_cnt_now`` is left out: describe() has no wording for it (it shows the raw name)."""
    return not any(tag in feature for tag in ("_std_", "_slope_", "_d1m15m", "_rel_", "_cnt_now"))


def _sample_shap(rng: random.Random, label: str, values: dict[str, float], face: dict[str, Any]) -> dict[str, float]:
    """Small noise on every feature, large mass on 1-3 features of each anchor channel."""
    shap = {f: rng.gauss(0, 0.02) for f in FEATURE_NAMES}
    anchors = dict(ANCHORS[label])
    if values["n_people"] < 2:
        anchors.pop("n_people", None)
    for ch, weight in anchors.items():
        if ch == "face":
            continue
        pool = [f for f in FEATURE_NAMES if channel_of(f) == ch and _is_level(f) and not math.isnan(values[f])]
        for f in rng.sample(pool, k=min(len(pool), rng.randint(1, 3))):
            shap[f] = weight * _u(rng, (0.6, 1.4))
    face_pool = ["face_now", "face_mean_1m", "face_mean_5m", "face_mean_15m", "face_std_5m"]
    if face["visible"]:
        sign = -1.0 if face["mask"] else 1.0
        scale = (0.15, 0.45) if (face["mask"] or label in ("Collaborative_Happy", "Tense")) else (0.04, 0.2)
        for f in rng.sample(face_pool, k=rng.randint(1, 2)):
            shap[f] = sign * _u(rng, scale)
    else:
        mag = _u(rng, (0.02, 0.2))
        shap["face_visible_frac_5m"] = mag if rng.random() < 0.6 else -mag
    if rng.random() < COUNTER_SIGNAL_RATE:
        pool = [f for f in FEATURE_NAMES
                if channel_of(f) not in anchors and not f.startswith("face_") and f != "fill_ratio"]
        shap[rng.choice(pool)] = -_u(rng, (0.1, 0.3))
    return shap


def _sample_probabilities(rng: random.Random, label: str, states: tuple[str, ...]) -> dict[str, float]:
    """Hundredths that sum to exactly 100, with the label a strict argmax."""
    others = [s for s in states if s != label]
    while True:
        conf = rng.randint(45, 95)
        rest = 100 - conf
        weights = [rng.random() + 0.05 for _ in others]
        shares = [int(rest * w / sum(weights)) for w in weights]
        shares[0] += rest - sum(shares)
        if max(shares) < conf:
            break
    probs = dict(zip(others, shares, strict=True))
    probs[label] = conf
    return {s: probs[s] / 100 for s in states}


def make_fact_sheet(
    rng: random.Random,
    *,
    label: str | None = None,
    face_condition: str | None = None,
    states: str | tuple[str, ...] = "core",
    n_people: int | None = None,
    person_id: int | None = None,
    event_id: str | None = None,
    ood: bool | None = None,
) -> dict[str, Any]:
    """One contract-valid fact sheet as a JSON-mode dict (see the module docstring)."""
    state_set = STATE_SETS[states] if isinstance(states, str) else tuple(states)
    label = label or rng.choice(state_set)
    if label not in state_set:
        raise ValueError(f"state {label!r} is not in the state set {state_set}")
    condition = face_condition or rng.choice(FACE_CONDITIONS)
    n_people = n_people or rng.randint(1, 3)
    person_id = person_id or rng.randint(1, n_people)
    face = _sample_face(rng, label, condition)
    values, tod_h = _sample_values(rng, label, n_people, face)
    shap = _sample_shap(rng, label, values, face)
    layers = build_layers(shap, values, tod_h=tod_h)
    probabilities = _sample_probabilities(rng, label, state_set)
    if ood is None:
        ood = rng.random() < OOD_RATE
    drivers = [d.feature for layer in layers for d in layer.drivers]
    ood_features = sorted(rng.sample(drivers, k=min(len(drivers), rng.randint(1, 3)))) if ood else []
    fill = 15.0 if rng.random() < 0.8 else round(_u(rng, (1.0, 14.9)), 1)
    sheet = build_fact_sheet(
        event_id=event_id or f"p{person_id}-{rng.randrange(1_000_000):06d}",
        person=f"Person #{person_id}",
        label=label,
        label_display=display_label(label, mask_override=face["mask"], n_people=n_people, states=state_set),
        confidence=probabilities[label],
        probabilities=probabilities,
        window_fill_min=fill,
        n_people=n_people,
        face_only_label=face["label"],
        face_only_confidence=face["confidence"],
        face_valence=face["valence"],
        face_visible=face["visible"],
        mask_override=face["mask"],
        layers=layers,
        face_visible_frac=face["visible_frac"],
        lux=values["lux_now"],
        ood_features=ood_features,
    )
    return sheet.model_dump(mode="json")


def make_batch(n: int, seed: int = 0, states: str = "core") -> list[dict[str, Any]]:
    """``n`` sheets balanced over states x face conditions (shuffled, deterministic per seed)."""
    rng = random.Random(seed)
    state_set = STATE_SETS[states]
    n_s, n_c = len(state_set), len(FACE_CONDITIONS)
    # Interleaved so that any prefix covers states and face conditions evenly: block index k maps to
    # (state k % S, condition (k // S + k) % C), a bijection on S*C when gcd(S + 1, C) == 1.
    combos = [(state_set[k % n_s], FACE_CONDITIONS[(k // n_s + k) % n_c]) for k in range(n_s * n_c)]
    assert len(set(combos)) == n_s * n_c
    plan = [combos[i % len(combos)] for i in range(n)]
    rng.shuffle(plan)
    return [
        make_fact_sheet(rng, label=label, face_condition=cond, states=states, event_id=f"evt_{i:06d}")
        for i, (label, cond) in enumerate(plan)
    ]


def example_fact_sheet(states: str = "core") -> dict[str, Any]:
    """The PRD's masked-frustration example as a contract sheet (deterministic)."""
    state_set = STATE_SETS[states]
    values = {"db_mean_15m": 72.4, "lux_now": 180.0, "hours_since_break": 4.2, "fist_cnt_5m": 3.0,
              "compl_cnt_15m": 0.0, "face_now": 0.62, "n_people": 2.0}
    contribs = {"db_mean_15m": 0.31, "lux_now": 0.06, "hours_since_break": 0.22, "fist_cnt_5m": 0.27,
                "compl_cnt_15m": 0.03, "face_now": -0.07, "n_people": -0.02}
    rest = {"Focused": 0.04, "Exhausted": 0.05, "Collaborative_Happy": 0.03, "Overwhelmed": 0.0, "Tense": 0.0}
    probabilities = {s: (0.88 if s == "Frustrated_Masked" else rest[s]) for s in state_set}
    if states == "all":
        probabilities.update(Focused=0.03, Exhausted=0.03, Collaborative_Happy=0.02, Overwhelmed=0.02, Tense=0.02)
    sheet = build_fact_sheet(
        event_id="evt_000123", person="Person #1", label="Frustrated_Masked",
        label_display=display_label("Frustrated_Masked", mask_override=True, n_people=2, states=state_set),
        confidence=0.88, probabilities=probabilities, window_fill_min=15.0, n_people=2,
        face_only_label="Collaborative_Happy", face_only_confidence=0.71, face_valence=0.62, face_visible=True,
        mask_override=True, layers=build_layers(contribs, values), face_visible_frac=0.93, lux=180.0,
        ood_features=[],
    )
    return sheet.model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic contract fact sheets as JSONL.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--states", choices=tuple(STATE_SETS), default="core")
    args = parser.parse_args(argv)
    for sheet in make_batch(args.n, args.seed, args.states):
        sys.stdout.write(json.dumps(sheet) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
