"""The served predictor: M2 (context + face history) with M0 (face snapshot) running in its shadow.

Per window: features -> clip to the training range -> M2 calibrated probabilities -> LightGBM TreeSHAP (``pred_contrib``) for the
predicted class -> explanation layers. M0 gives the face-only reading; the mask-override condition
compares the two (docs/PLAN.md, MaskOverrideDetector).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from valence.contracts import TickRecord
from valence.explain.explainer import build_layers
from valence.features.compute import ABS_ROOM, FEATURE_INDEX, FEATURE_NAMES, FEATURE_SETS, features_from_window
from valence.features.spec import layer_of
from valence.models.predictor import PredictionOutput

M0_MIN_CONF = 0.45  # face-only model must clearly read something else (chance is 1/4 or 1/6)
M2_MIN_CONF = 0.70
POSITIVE = {"Collaborative_Happy"}  # a "mask" is a positive-looking face over a non-positive state
OOD_SHRINK_AFTER = 3  # more out-of-distribution features than this -> pull probabilities toward uniform
OOD_SHRINK = 0.3
OOD_MARGIN = 0.5  # unbounded features: OOD only this fraction of the training range beyond p1/p99

# Clipping to the training p1..p99 range is not evidence of a strange input: a face visible on every frame
# (1.0 vs a training p99 of 0.998) or an 08:00 time of day are legitimate. Features with a physical domain
# are OOD only outside it; clock/context features never are; the rest only well beyond the training range.
NEVER_OOD = {"tod_sin", "tod_cos", "hours_since_break", "n_people", "fill_ratio"}
PHYSICAL_DOMAINS: list[tuple[str, tuple[float, float]]] = [
    (r"^face_(now|mean_\w+)$", (-1.0, 1.0)),
    (r"^face_std_5m$", (0.0, 1.0)),
    (r"^(face_visible_\w+|withdraw_\w+)$", (0.0, 1.0)),
    (r"^(slump|hih)_(now|mean_\w+|max_\w+|ewm_\w+)$", (0.0, 1.0)),
    (r"^(slump|hih)_std_\w+$", (0.0, 0.5)),
    (r"^\w+_cnt_\w+$", (0.0, math.inf)),  # event counts are legitimately unbounded
    (r"^\w+_since_s$", (0.0, math.inf)),
]


def ood_band(name: str, lo: float, hi: float) -> tuple[float, float]:
    """Values outside the returned band count as out of distribution."""
    if name in NEVER_OOD:
        return -math.inf, math.inf
    for pattern, domain in PHYSICAL_DOMAINS:
        if re.match(pattern, name):
            return domain
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return -math.inf, math.inf
    margin = OOD_MARGIN * (hi - lo)
    return lo - margin, hi + margin


def _softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t
    e = np.exp(z - z.max())
    return e / e.sum()


class ServedPredictor:
    def __init__(self, artifacts: Path, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.states = tuple(meta["states"])
        m2_path = artifacts / "m2.txt"
        self.model_id = "m2_lgbm@" + hashlib.sha256(m2_path.read_bytes()).hexdigest()[:8]
        self.m2 = lgb.Booster(model_file=str(m2_path))
        self.m0 = lgb.Booster(model_file=str(artifacts / "m0.txt"))
        self.t2 = float(meta["temperatures"]["m2"])
        self.t0 = float(meta["temperatures"]["m0"])
        # Feature sets come from the trained artifact, so code changes cannot silently misalign a model.
        self.cols2 = meta["feature_sets"]["m2"]
        self.cols0 = meta["feature_sets"]["m0"]
        self.idx2 = np.array([FEATURE_INDEX[c] for c in self.cols2])
        self.idx0 = np.array([FEATURE_INDEX[c] for c in self.cols0])
        clip = json.loads((artifacts / "clip_bounds.json").read_text())
        self.lo = np.array([clip.get(n, [-np.inf, np.inf])[0] for n in FEATURE_NAMES])
        self.hi = np.array([clip.get(n, [-np.inf, np.inf])[1] for n in FEATURE_NAMES])
        band = np.array([ood_band(n, lo, hi) for n, lo, hi in zip(FEATURE_NAMES, self.lo, self.hi)])
        self.ood_lo, self.ood_hi = band[:, 0], band[:, 1]
        used = set(self.cols2) | set(self.cols0)  # a feature neither model reads cannot make an input unusual
        self.ood_used = np.array([n in used for n in FEATURE_NAMES])
        self.face_cols = np.array([layer_of(c) == "Facial Surface" for c in self.cols2])

    # --- core -----------------------------------------------------------------------------------
    def _clip(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        ok = ~np.isnan(x)
        clipped = np.where(ok, np.clip(x, self.lo, self.hi), x)
        ood = [FEATURE_NAMES[i] for i in np.nonzero(ok & self.ood_used & ((x < self.ood_lo) | (x > self.ood_hi)))[0]]
        return clipped, ood

    def _m2(self, xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row = xc[self.idx2][None, :]
        raw = self.m2.predict(row, raw_score=True, num_threads=1)[0]
        contrib = np.asarray(self.m2.predict(row, pred_contrib=True, num_threads=1)).reshape(len(self.states), -1)
        return raw, contrib  # contrib[k, :-1] per feature, contrib[k, -1] bias

    def predict_vector(self, x: np.ndarray, tod_h: float | None = None) -> PredictionOutput:
        xc, ood = self._clip(x)
        raw, contrib = self._m2(xc)
        p = _softmax(raw, self.t2)
        if len(ood) > OOD_SHRINK_AFTER:
            p = (1 - OOD_SHRINK) * p + OOD_SHRINK / len(p)
        k = int(np.argmax(p))
        label = self.states[k]
        face_label = face_conf = None
        if not math.isnan(x[FEATURE_INDEX["face_now"]]):
            p0 = _softmax(self.m0.predict(xc[self.idx0][None, :], raw_score=True, num_threads=1)[0], self.t0)
            k0 = int(np.argmax(p0))
            face_label, face_conf = self.states[k0], float(p0[k0])
            face_push_for_m0_class = float(contrib[k0, :-1][self.face_cols].sum())
        else:
            face_push_for_m0_class = 0.0
        raw_override = None if face_label is None else bool(
            face_label in POSITIVE and label not in POSITIVE and face_conf >= M0_MIN_CONF and p[k] >= M2_MIN_CONF
            and face_push_for_m0_class > 0)
        names = self.cols2
        values = {n: float(x[FEATURE_INDEX[n]]) for n in names}
        contribs = {n: float(c) for n, c in zip(names, contrib[k, :-1]) if c != 0.0}
        absolute = {n: float(x[FEATURE_INDEX[n]]) for n in ABS_ROOM}
        layers = build_layers(contribs, values, tod_h=tod_h, absolute=absolute)
        return PredictionOutput(
            label=label, confidence=float(p[k]), probabilities=dict(zip(self.states, map(float, p))),
            face_only_label=face_label, face_only_confidence=face_conf, layers=layers, raw_override=raw_override,
            values=values, ood_features=ood,
        )

    def predict(self, person: list[TickRecord], room: list[TickRecord]) -> PredictionOutput | None:
        if not person:
            return None
        x = features_from_window(person, room or person)
        return self.predict_vector(x, tod_h=person[-1].tod_h)

    # --- counterfactual ("what if this one feature were different?") -------------------------------
    def counterfactual(self, person: list[TickRecord], room: list[TickRecord], feature: str, value: float) -> dict[str, Any]:
        if feature not in FEATURE_INDEX:
            raise KeyError(f"unknown feature {feature}")
        x = features_from_window(person, room or person)
        before = self.predict_vector(x, tod_h=person[-1].tod_h)
        x2 = x.copy()
        x2[FEATURE_INDEX[feature]] = value
        after = self.predict_vector(x2, tod_h=person[-1].tod_h)
        return {
            "feature": feature, "original_value": None if math.isnan(x[FEATURE_INDEX[feature]]) else float(x[FEATURE_INDEX[feature]]),
            "counterfactual_value": value,
            "original": {"label": before.label, "confidence": round(before.confidence, 4)},
            "counterfactual": {"label": after.label, "confidence": round(after.confidence, 4),
                               "probabilities": {s: round(v, 4) for s, v in after.probabilities.items()}},
            "label_changed": before.label != after.label,
            "simulated_fields": [feature],
        }


def load_served(artifacts: Path, state_set: str) -> ServedPredictor | None:
    meta_path = artifacts / "model_meta.json"
    if not meta_path.exists() or not (artifacts / "m2.txt").exists() or not (artifacts / "m0.txt").exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("state_set") != state_set:
        return None
    return ServedPredictor(artifacts, meta)
