"""Instant template explanation, sent with every prediction before the LLM narration arrives.

It also serves as the narrator's last-resort fallback, so it obeys the same rules: only numbers that
appear in the fact sheet's display strings, masking language only when the override fired, room
drivers phrased as the room, people referred to as "Person #N".
"""

from __future__ import annotations

import hashlib

from valence.contracts import STATE_DISPLAY, FactSheet
from valence.explain.explainer import top_drivers

OOD_NOTE_AFTER = 3  # matches the served model's confidence shrink rule

OPENERS = (
    "Model predicted {label} ({conf}) for {person}.",
    "{person} reads as {label} with {conf} confidence.",
    "{label} ({conf}) is the most likely state for {person}.",
)


def _pick(options: tuple[str, ...], key: str) -> str:
    return options[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(options)]


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def template_text(sheet: FactSheet) -> str:
    opener = _pick(OPENERS, sheet.event_id).format(
        label=sheet.label_display, conf=sheet.confidence_display, person=sheet.person)
    context_layers = [layer for layer in sheet.layers if layer.name != "Facial Surface"]
    drivers = top_drivers(context_layers if sheet.mask_override or not sheet.face_visible else sheet.layers, k=3)
    evidence = _join([d.display for d in drivers])
    if sheet.mask_override and sheet.face_only.label == "Collaborative_Happy":
        face_name = STATE_DISPLAY.get(sheet.face_only.label, sheet.face_only.label)
        body = (f"The face alone suggests {face_name}, but {evidence} point the other way. "
                "Context corrected the facial mask.")
    elif not sheet.face_visible:
        body = f"The face is not readable right now, so the call rests on context: {evidence}."
    elif evidence:
        body = f"Main evidence: {evidence}."
    else:
        body = "The window is still filling, so evidence is limited."
    if len(sheet.data_quality.ood_features) > OOD_NOTE_AFTER:
        body += " Some readings are outside the training range, so confidence is reduced."
    return f"{opener} {body}"
