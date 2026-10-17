"""Predictor interface, per-person smoothing, and a rule-based stub used until M2 is trained.

``Predictor.predict`` is stateless: one window in, one calibrated distribution out. ``PersonSmoother``
adds the stateful parts the live display needs: EMA + 2-tick hysteresis on the *label* (the shown
confidence stays the calibrated, unsmoothed probability), the 3-tick persistence rule for the mask
override, and the warm-up blocks.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from valence.contracts import CORE_STATES, Layer, TickRecord
from valence.explain.explainer import build_layers


@dataclass
class PredictionOutput:
    label: str
    confidence: float
    probabilities: dict[str, float]
    face_only_label: str | None
    face_only_confidence: float | None
    layers: list[Layer]
    raw_override: bool | None  # this tick's mask-override condition (None = no face this tick)
    values: dict[str, float] = field(default_factory=dict)
    ood_features: list[str] = field(default_factory=list)


class Predictor(Protocol):
    states: tuple[str, ...]
    model_id: str

    def predict(self, person: list[TickRecord], room: list[TickRecord]) -> PredictionOutput | None: ...


@dataclass
class SmoothedPrediction:
    label: str
    mask_override: bool
    warming_up: bool


class PersonSmoother:
    def __init__(self, states: tuple[str, ...], tick_s: float = 5.0, warmup_s: float = 60.0,
                 override_block_s: float = 120.0, alpha: float = 0.5, hold: int = 2, override_on: int = 3,
                 override_off: int = 1, override_window: int = 6) -> None:
        self.states = states
        self.tick_s = tick_s
        self.warmup_ticks = int(warmup_s / tick_s)
        self.override_block_ticks = int(override_block_s / tick_s)
        self.alpha = alpha
        self.hold = hold
        self.override_on_n = override_on
        self.override_off_n = override_off
        self.ema: np.ndarray | None = None
        self.label: str | None = None
        self.candidate: str | None = None
        self.candidate_n = 0
        self.override_hist: deque[bool] = deque(maxlen=override_window)
        self.override_on = False

    def update(self, out: PredictionOutput, window_ticks: int) -> SmoothedPrediction:
        p = np.array([out.probabilities.get(s, 0.0) for s in self.states])
        self.ema = p if self.ema is None else self.alpha * p + (1 - self.alpha) * self.ema
        top = self.states[int(np.argmax(self.ema))]
        if self.label is None:
            self.label = top
        elif top != self.label:
            self.candidate_n = self.candidate_n + 1 if top == self.candidate else 1
            self.candidate = top
            if self.candidate_n >= self.hold:
                self.label, self.candidate, self.candidate_n = top, None, 0
        else:
            self.candidate, self.candidate_n = None, 0
        # Override hysteresis: on when the condition held on >= 3 of the last 6 face-visible ticks, off at <= 1.
        # A tick without a readable face holds the current state instead of resetting it.
        if out.raw_override is not None:
            self.override_hist.append(bool(out.raw_override and out.label == self.label))
            hits = sum(self.override_hist)
            if hits >= self.override_on_n:
                self.override_on = True
            elif hits <= self.override_off_n:
                self.override_on = False
        override = self.override_on and window_ticks >= self.override_block_ticks
        return SmoothedPrediction(label=self.label, mask_override=override,
                                  warming_up=window_ticks < self.warmup_ticks)


# --- rule-based stub ------------------------------------------------------------------------------

def _vals(rows: list[TickRecord], attr: str) -> np.ndarray:
    v = np.array([getattr(r, attr) for r in rows], dtype=float)
    return v[~np.isnan(v)]


def _mean(rows: list[TickRecord], attr: str) -> float:
    v = _vals(rows, attr)
    return float(v.mean()) if v.size else math.nan


def stub_features(person: list[TickRecord], room: list[TickRecord]) -> dict[str, float]:
    last5 = person[-60:]
    room5, room15 = room[-60:], room[-180:]
    face = _vals(person[-1:], "face_valence")
    return {
        "db_mean_5m": _mean(room5, "ambient_db"),
        "lux_mean_5m": _mean(room5, "light_lux"),
        "laugh_cnt_15m": float(sum(r.laughter_count for r in room15)),
        "compl_cnt_15m": float(sum(r.compliments_count for r in room15)),
        "fist_cnt_5m": float(sum(r.clenched_fists_count for r in last5)),
        "slump_mean_5m": _mean(last5, "posture_slump"),
        "hih_mean_5m": _mean(last5, "head_in_hands_frac"),
        "motion_mean_5m": _mean(last5, "motion_energy"),
        "hours_since_break": person[-1].hours_since_break,
        "face_now": float(face[-1]) if face.size else math.nan,
        "face_visible_frac_5m": _mean(last5, "face_visible"),
        "n_people": float(person[-1].n_people),
        "_tod_h": person[-1].tod_h,
    }


# (center, scale) to standardize, then per-state weights. Hand-set to mirror docs/DATASET.md.
_NORM = {
    "db_mean_5m": (56, 8), "lux_mean_5m": (300, 150), "laugh_cnt_15m": (1, 2), "compl_cnt_15m": (0.5, 1),
    "fist_cnt_5m": (1, 2), "slump_mean_5m": (0.3, 0.2), "hih_mean_5m": (0.2, 0.15), "motion_mean_5m": (0.12, 0.08),
    "hours_since_break": (2.5, 1.5), "face_now": (0.1, 0.35),
}
_W = {
    "Focused": {"db_mean_5m": -1.0, "fist_cnt_5m": -0.8, "slump_mean_5m": -0.5, "motion_mean_5m": -0.3,
                "hours_since_break": -0.4, "laugh_cnt_15m": -0.4, "face_now": -0.2},
    "Frustrated_Masked": {"db_mean_5m": 0.8, "fist_cnt_5m": 1.6, "hours_since_break": 0.6, "lux_mean_5m": -0.3,
                          "compl_cnt_15m": -0.4, "face_now": 0.1},
    "Exhausted": {"db_mean_5m": -0.9, "slump_mean_5m": 1.2, "motion_mean_5m": -1.0, "hours_since_break": 1.0,
                  "lux_mean_5m": -0.6, "hih_mean_5m": 0.5},
    "Collaborative_Happy": {"laugh_cnt_15m": 1.3, "compl_cnt_15m": 1.3, "db_mean_5m": 0.4, "lux_mean_5m": 0.6,
                            "hours_since_break": -0.8, "fist_cnt_5m": -0.8, "face_now": 0.6},
    "Overwhelmed": {"hih_mean_5m": 1.5, "slump_mean_5m": 0.8, "db_mean_5m": 0.5, "hours_since_break": 0.6},
    "Tense": {"fist_cnt_5m": 1.3, "db_mean_5m": 0.9, "face_now": -1.0, "motion_mean_5m": 0.8},
}


class StubPredictor:
    """Transparent linear rules standing in for M2 until the trained model is in artifacts/."""

    model_id = "stub-rules"

    def __init__(self, states: tuple[str, ...] = CORE_STATES) -> None:
        self.states = states

    def predict(self, person: list[TickRecord], room: list[TickRecord]) -> PredictionOutput | None:
        if not person:
            return None
        f = stub_features(person, room or person)
        z = {k: (0.0 if math.isnan(f[k]) else (f[k] - c) / s) for k, (c, s) in _NORM.items()}
        scores = np.array([sum(w * z[k] for k, w in _W[s].items()) for s in self.states])
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        i = int(np.argmax(probs))
        label = self.states[i]
        contribs = {k: _W[label].get(k, 0.0) * z[k] - np.mean([_W[s].get(k, 0.0) * z[k] for s in self.states])
                    for k in _NORM if not math.isnan(f[k])}
        contribs["face_visible_frac_5m"] = 0.0
        face_label, face_conf = None, None
        if not math.isnan(f["face_now"]):
            v = f["face_now"]
            face_label = "Collaborative_Happy" if v > 0.25 else ("Focused" if v < -0.1 else "Exhausted")
            face_conf = 0.55 + min(0.35, abs(v) * 0.5)
        raw_override = None if face_label is None else bool(
            face_label == "Collaborative_Happy" and label != face_label and probs[i] >= 0.7
            and contribs.get("face_now", 0) < 0)
        values = {k: v for k, v in f.items() if not k.startswith("_")}
        layers = build_layers({k: v for k, v in contribs.items() if k in values}, values, tod_h=f["_tod_h"])
        return PredictionOutput(
            label=label, confidence=float(probs[i]), probabilities=dict(zip(self.states, map(float, probs))),
            face_only_label=face_label, face_only_confidence=face_conf, layers=layers, raw_override=raw_override,
            values=values,
        )
