"""Fact-sheet schema: the only input the narrator ever sees.

The core API emits one fact sheet per classified event (prediction, face-only baseline, SHAP
drivers grouped by layer, data quality). The models are permissive (``extra="allow"``) so new
core fields never break narration, and every helper takes the plain JSON dict the service
receives. Narration keys off ``display``/``unit``/layer names, never a fixed feature list.

Conventions the rest of the narrator relies on:
- a hidden face is ``face_visible: false`` with ``face_only`` fields null; a ``face_only.label``
  of "Unknown" is accepted and treated the same as null;
- a driver with unit "%" may carry a fraction (0.93) or a percentage (93.0); values with
  ``abs(value) <= 1`` are read as fractions;
- a driver with unit "h" whose display names the time of day is a clock hour (15.5 -> 15:30).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from valence.narrator.lexicon import driver_concepts

STATES: tuple[str, ...] = ("Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy")
StateName = Literal["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"]
STATE_DISPLAY: dict[str, str] = {
    "Focused": "Focused",
    "Frustrated_Masked": "Frustrated (Masked)",
    "Exhausted": "Exhausted",
    "Collaborative_Happy": "Collaborative (Happy)",
}
LAYER_NAMES: tuple[str, ...] = ("Ambient Physics", "Time Dynamics", "Somatic & Social", "Facial Surface")

_UNKNOWN_LABELS = frozenset({"", "unknown", "none", "null", "n/a"})
_PERSON_NUMBER = re.compile(r"(\d+)")


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Driver(_Open):
    feature: str
    display: str
    value: float | None = None
    unit: str = ""
    shap: float


class Layer(_Open):
    name: str
    contribution: float
    drivers: list[Driver] = Field(default_factory=list)


class FaceOnly(_Open):
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_valence: float | None = Field(default=None, ge=-1.0, le=1.0)


class DataQuality(_Open):
    face_visible_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    lux: float | None = None


class FactSheet(_Open):
    event_id: str
    person: str = Field(pattern=r"^Person #\d+$")
    label: StateName
    label_display: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_display: str
    probabilities: dict[str, float]
    window_fill_min: float = Field(ge=0.0)
    face_only: FaceOnly = Field(default_factory=FaceOnly)
    face_visible: bool
    mask_override: bool
    layers: list[Layer]
    data_quality: DataQuality = Field(default_factory=DataQuality)


def as_dict(sheet: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    """Plain-dict view of a fact sheet given as a dict or a pydantic model."""
    if isinstance(sheet, BaseModel):
        return sheet.model_dump()
    return dict(sheet)


def parse_fact_sheet(sheet: Mapping[str, Any] | BaseModel) -> FactSheet:
    return FactSheet.model_validate(as_dict(sheet))


def iter_drivers(sheet: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(layer_name, driver) pairs in fact-sheet order."""
    return [
        (str(layer.get("name", "")), driver)
        for layer in sheet.get("layers") or ()
        for driver in layer.get("drivers") or ()
    ]


def abs_shap(driver: Mapping[str, Any]) -> float:
    shap = driver.get("shap")
    if shap is None or isinstance(shap, bool):
        return 0.0
    value = abs(float(shap))
    return value if math.isfinite(value) else 0.0


def top_drivers(sheet: Mapping[str, Any] | BaseModel, k: int | None = 3) -> list[tuple[str, dict[str, Any]]]:
    """Drivers across all layers sorted by |shap| descending (stable on ties)."""
    ranked = sorted(iter_drivers(as_dict(sheet)), key=lambda item: -abs_shap(item[1]))
    return ranked if k is None else ranked[:k]


def person_number(sheet: Mapping[str, Any]) -> int | None:
    match = _PERSON_NUMBER.search(str(sheet.get("person") or ""))
    return int(match.group(1)) if match else None


def face_hidden(sheet: Mapping[str, Any]) -> bool:
    return not bool(sheet.get("face_visible", True))


def face_only_label(sheet: Mapping[str, Any]) -> str | None:
    """The face-only baseline's state, or None when the face was not read."""
    label = (sheet.get("face_only") or {}).get("label")
    if label is None or str(label).strip().lower() in _UNKNOWN_LABELS:
        return None
    return str(label)


def fixed(value: float, decimals: int) -> str:
    """Fixed-point string without a stray "-0"."""
    text = f"{value:.{decimals}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _trim(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_percent(value: float) -> str:
    """0.88 -> "88%"; values above 1 are taken as already in percent."""
    pct = value * 100 if abs(value) <= 1 else value
    return f"{fixed(pct, 0)}%"


def _format_hours(value: float) -> str:
    text = _trim(fixed(value, 1))
    return f"{text} hour" if text == "1" else f"{text} hours"


def _format_minutes(value: float) -> str:
    text = _trim(fixed(value, 0 if abs(value) >= 10 else 1))
    return f"{text} minute" if text == "1" else f"{text} minutes"


def _format_clock(value: float) -> str:
    hours, minutes = divmod(round(value * 60), 60)
    return f"{hours % 24:02d}:{minutes:02d}"


def _format_plain(value: float) -> str:
    magnitude = abs(value)
    decimals = 0 if magnitude >= 100 else 1 if magnitude >= 10 else 2
    return _trim(fixed(value, decimals)) if decimals else fixed(value, 0)


_UNIT_FORMATS = {
    "dB": lambda v: f"{fixed(v, 0 if abs(v) >= 10 else 1)} dB",
    "lux": lambda v: f"{v:,.0f} lux",
    "h": _format_hours,
    "min": _format_minutes,
    "count": lambda v: f"{round(v):d}",
    "%": format_percent,
    "/min": lambda v: f"{fixed(v, 1)} per minute",
    "": lambda v: fixed(v, 2),
}
_UNIT_ALIASES = {"db": "dB", "hr": "h", "hrs": "h", "hours": "h", "mins": "min", "minutes": "min",
                 "events": "count", "n": "count", "per min": "/min", "pct": "%", "percent": "%"}


def canonical_unit(unit: str) -> str:
    return unit if unit in _UNIT_FORMATS else _UNIT_ALIASES.get(unit.lower(), unit)


def format_value(driver: Mapping[str, Any]) -> str:
    """Human-readable value+unit for a driver, e.g. "72 dB", "4.2 hours", "3", "93%", "15:30"."""
    value = driver.get("value")
    unit = canonical_unit(str(driver.get("unit") or "").strip())
    if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
        return "unavailable"
    number = float(value)
    if unit == "h" and 0 <= number < 24 and "time_of_day" in driver_concepts(driver):
        return _format_clock(number)
    formatter = _UNIT_FORMATS.get(unit)
    if formatter is not None:
        return formatter(number)
    return f"{_format_plain(number)} {unit}".strip()
