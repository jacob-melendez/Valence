"""Fact-sheet input for the narrator, read through the shared contract (``valence.contracts``).

The core API posts one contract ``FactSheet`` per narrated event. The narrator validates it with
the contract model itself (strict, ``extra="forbid"``), then works on the plain JSON dict so every
helper also accepts sheets loaded from JSONL. States, display names, layers and the lexicon all
come from ``valence.contracts``; nothing here keeps its own copy.

Driver ranking mirrors ``valence.explain.explainer.top_drivers`` exactly: drivers are taken in
fact-sheet order (layers as sent, drivers within each layer as sent), optionally filtered by
``effect``, then stably sorted by ``abs(shap)`` descending and cut to ``k``. The validator uses two
views of it: ``key_drivers`` (what the core template cites) and ``allowed_drivers``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from valence.contracts import (
    ALL_STATES,
    CORE_STATES,
    LAYERS,
    STATE_DISPLAY,
    STATE_LEXICON,
    STATE_SETS,
    FactSheet,
    display_label,
)

__all__ = [
    "ALL_STATES", "CORE_STATES", "LAYERS", "STATE_DISPLAY", "STATE_LEXICON", "STATE_SETS", "FactSheet",
    "KEY_K", "ALLOWED_K", "allowed_drivers", "as_dict", "display_label", "face_hidden", "face_only_label",
    "iter_drivers", "key_drivers", "parse_fact_sheet", "person_number", "sheet_states", "state_set_name",
    "top_drivers",
]

KEY_K = 3  # drivers the explanation must draw on (>= 2 of them)
ALLOWED_K = 5  # drivers the explanation may draw on

_PERSON_NUMBER = re.compile(r"^Person #(\d+)$")


def as_dict(sheet: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    """Plain-dict (JSON-mode) view of a fact sheet given as a dict or a pydantic model."""
    if isinstance(sheet, BaseModel):
        return sheet.model_dump(mode="json")
    return dict(sheet)


def parse_fact_sheet(sheet: Mapping[str, Any] | BaseModel) -> FactSheet:
    """Strict contract validation; raises ``pydantic.ValidationError`` on any deviation."""
    if isinstance(sheet, FactSheet):
        return sheet
    return FactSheet.model_validate(as_dict(sheet))


def iter_drivers(sheet: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(layer_name, driver) pairs in fact-sheet order."""
    return [
        (str(layer.get("name", "")), driver)
        for layer in sheet.get("layers") or ()
        for driver in layer.get("drivers") or ()
    ]


def top_drivers(
    sheet: Mapping[str, Any] | BaseModel, k: int | None = ALLOWED_K, effect: str | None = "supports"
) -> list[dict[str, Any]]:
    """Same ranking as ``valence.explain.explainer.top_drivers`` (stable sort on |shap|)."""
    sheet = as_dict(sheet)
    drivers = [d for _, d in iter_drivers(sheet) if effect is None or d.get("effect") == effect]
    ranked = sorted(drivers, key=lambda d: abs(float(d.get("shap") or 0.0)), reverse=True)
    return ranked if k is None else ranked[:k]


def key_drivers(sheet: Mapping[str, Any] | BaseModel) -> list[dict[str, Any]]:
    """The core's own "top-3", exactly what ``valence.explain.template.template_text`` cites:
    ``top_drivers(k=3, effect="supports")``, over the context layers only (Facial Surface left
    out) when ``mask_override`` is set or the face is hidden. If nothing supports the label, the
    top-3 of any effect stand in."""
    sheet = as_dict(sheet)
    if sheet.get("mask_override") or face_hidden(sheet):
        sheet = {**sheet, "layers": [layer for layer in sheet.get("layers") or ()
                                     if layer.get("name") != "Facial Surface"]}
    return top_drivers(sheet, KEY_K, "supports") or top_drivers(sheet, KEY_K, None)


def allowed_drivers(sheet: Mapping[str, Any] | BaseModel) -> list[dict[str, Any]]:
    """Drivers an explanation may mention: ``top_drivers(k=5, effect=None)`` (so the strongest
    opposing evidence can be named as contrast) plus the key drivers, in rank order."""
    out: list[dict[str, Any]] = []
    for driver in [*top_drivers(sheet, ALLOWED_K, None), *key_drivers(sheet)]:
        if not any(driver is seen for seen in out):
            out.append(driver)
    return out


def person_number(sheet: Mapping[str, Any]) -> int | None:
    """N for a "Person #N" sheet; None for the room-only sheet ("Room")."""
    match = _PERSON_NUMBER.match(str(sheet.get("person") or ""))
    return int(match.group(1)) if match else None


def face_hidden(sheet: Mapping[str, Any]) -> bool:
    return not bool(sheet.get("face_visible", False))


def face_only_label(sheet: Mapping[str, Any]) -> str | None:
    """The face-only baseline's state; None when the face was hidden (contract: null)."""
    if face_hidden(sheet):
        return None
    label = (sheet.get("face_only") or {}).get("label")
    return str(label) if label in ALL_STATES else None


def sheet_states(sheet: Mapping[str, Any]) -> tuple[str, ...]:
    """The model's state set, read from the probabilities' keys (contract order)."""
    keys = set((sheet.get("probabilities") or {}).keys())
    states = tuple(s for s in ALL_STATES if s in keys)
    if sheet.get("label") in ALL_STATES and sheet["label"] not in states:
        states = tuple(s for s in ALL_STATES if s in keys | {sheet["label"]})
    return states or CORE_STATES


def state_set_name(states: tuple[str, ...]) -> str:
    """"core" / "all" for the two contract sets, else a stable joined key."""
    for name, members in STATE_SETS.items():
        if tuple(states) == members:
            return name
    return "+".join(states)
