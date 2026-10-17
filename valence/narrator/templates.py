"""Deterministic template explanations: the narrator's guaranteed fallback.

Six phrasings of the same audit sentence set: name the state via ``label_display``, quote
``confidence_display``, cite the top-3 drivers with their display names and formatted values,
then add a mask-contrast clause only when context overrode a positive-looking face, an
occlusion clause only when the face was hidden, and a neutral closing line otherwise. Every
variant is built so it passes :func:`valence.narrator.validator.validate` for any well-formed
fact sheet; ``render_template`` re-checks and walks to the next variant if one ever fails.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from valence.narrator.lexicon import driver_concepts, text_concepts
from valence.narrator.schema import (
    STATE_DISPLAY,
    as_dict,
    face_hidden,
    face_only_label,
    fixed,
    format_percent,
    format_value,
    top_drivers,
)
from valence.narrator.validator import KEY_DRIVERS, allowed_concepts, validate

N_VARIANTS = 6
_VALUE_EPS = 1e-6


@dataclass(slots=True)
class _Context:
    person: str
    label: str
    confidence: str
    listed: list[dict[str, Any]]
    mask: bool
    hidden: bool
    valence: str | None
    baseline: str | None
    baseline_confidence: str | None
    visibility: str | None
    lead_layer: str | None
    window: str | None


def _lower_first(text: str) -> str:
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _phrase(driver: Mapping[str, Any], style: str) -> str:
    name = _lower_first(str(driver.get("display") or driver.get("feature") or "signal"))
    value = format_value(driver)
    if style == "at" and value != "unavailable":
        return f"{name} at {value}"
    return f"{name} ({value})"


def _close(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < _VALUE_EPS
    except (TypeError, ValueError):
        return False


def _absorb(drivers: list[dict[str, Any]], concept: str, anchor: Any) -> tuple[list[dict[str, Any]], dict | None]:
    """Split off the driver a face clause will cite (the one matching ``anchor``, else the first)."""
    candidates = [d for d in drivers if concept in driver_concepts(d)]
    if not candidates:
        return drivers, None
    chosen = next((d for d in candidates if _close(d.get("value"), anchor)), None)
    if chosen is None and anchor is None:
        chosen = candidates[0]
    if chosen is None:
        return drivers, None
    return [d for d in drivers if d is not chosen], chosen


def _lead_layer(sheet: Mapping[str, Any]) -> str | None:
    layers = [layer for layer in sheet.get("layers") or () if layer.get("name")]
    if not layers:
        return None
    lead = max(layers, key=lambda layer: float(layer.get("contribution") or 0.0))
    name = str(lead["name"])
    if not text_concepts(name) <= allowed_concepts(sheet):
        return None
    contribution = lead.get("contribution")
    if contribution is None:
        return f"the {name} layer"
    share = float(contribution)
    shown = format_percent(share) if 0 <= share <= 1 else fixed(share, 2)
    return f"the {name} layer ({shown})"


def _context(sheet: Mapping[str, Any]) -> _Context:
    face_only = sheet.get("face_only") or {}
    quality = sheet.get("data_quality") or {}
    mask = bool(sheet.get("mask_override"))
    hidden = face_hidden(sheet)
    listed = [d for _, d in top_drivers(sheet, KEY_DRIVERS)]
    valence = visibility = None
    if mask:
        anchor = face_only.get("face_valence")
        listed, face_driver = _absorb(listed, "face_valence", anchor)
        if face_driver is not None:
            valence = f"{_lower_first(str(face_driver.get('display')))} {format_value(face_driver)}"
        elif anchor is not None:
            valence = f"facial valence {fixed(float(anchor), 2)}"
    if hidden:
        anchor = quality.get("face_visible_frac")
        listed, vis_driver = _absorb(listed, "face_visible", anchor)
        if vis_driver is not None and vis_driver.get("value") is not None:
            visibility = format_value(vis_driver)
        elif anchor is not None:
            visibility = format_percent(float(anchor))
    baseline_label = face_only_label(sheet) if mask else None
    baseline_conf = face_only.get("confidence") if baseline_label else None
    fill = sheet.get("window_fill_min")
    return _Context(
        person=str(sheet.get("person") or "the person"),
        label=str(sheet.get("label_display") or STATE_DISPLAY.get(str(sheet.get("label")), sheet.get("label"))),
        confidence=str(sheet.get("confidence_display") or format_percent(float(sheet.get("confidence") or 0.0))),
        listed=listed,
        mask=mask,
        hidden=hidden,
        valence=valence,
        baseline=f"'{STATE_DISPLAY.get(baseline_label, baseline_label)}'" if baseline_label else None,
        baseline_confidence=format_percent(float(baseline_conf)) if baseline_conf is not None else None,
        visibility=visibility,
        lead_layer=_lead_layer(sheet),
        window=format_value({"value": fill, "unit": "min"}) if fill else None,
    )


def _split(ctx: _Context, style: str) -> tuple[list[str], list[str]]:
    support = [_phrase(d, style) for d in ctx.listed if float(d.get("shap") or 0.0) >= 0]
    oppose = [_phrase(d, style) for d in ctx.listed if float(d.get("shap") or 0.0) < 0]
    return support, oppose


def _evidence(ctx: _Context, style: str, against: str) -> str:
    """Supporting drivers, then opposing ones via ``against`` (a format with ``{x}``)."""
    support, oppose = _split(ctx, style)
    if not support and not oppose:
        return "the combined context layers"
    if not support:
        return f"{_join(oppose)}, which pulled the other way"
    tail = against.format(x=_join(oppose)) if oppose else ""
    return _join(support) + tail


def _paren(value: str | None, pattern: str = " ({v})") -> str:
    return pattern.format(v=value) if value else ""


def _v0(ctx: _Context) -> list[str]:
    lines = [
        f"Model predicted '{ctx.label}' with {ctx.confidence} confidence for {ctx.person}.",
        f"The system detected {_evidence(ctx, 'paren', ', while {x} pulled the other way')}.",
    ]
    if ctx.mask:
        seen = f" that a face-only model read as {ctx.baseline}" if ctx.baseline else ""
        lines.append(
            f"While {ctx.person} displays a positive facial expression{_paren(ctx.valence)}, "
            f"context corrected the facial mask{seen}."
        )
    elif ctx.hidden:
        lines.append(
            f"{ctx.person}'s face was not visible for most of the window"
            f"{_paren(ctx.visibility, ' (visible {v} of the time)')}, so this reading relies on context alone."
        )
    else:
        lines.append("These aggregate context signals carry the explanation.")
    return lines


def _v1(ctx: _Context) -> list[str]:
    lines = [
        f"{ctx.person} was classified as '{ctx.label}' at {ctx.confidence} confidence.",
        f"Key drivers: {_evidence(ctx, 'at', '; {x} weighed against it')}.",
    ]
    if ctx.mask:
        seen = f" of {ctx.baseline}" if ctx.baseline else ""
        lines.append(
            f"{ctx.person}'s smiling expression{_paren(ctx.valence)} was outweighed by these signals, "
            f"so context overrode the face-only reading{seen}."
        )
    elif ctx.hidden:
        lines.append(
            f"With {ctx.person}'s face hidden for most of the window{_paren(ctx.visibility, ' ({v} visible)')}, "
            "the call relies on context alone."
        )
    else:
        lines.append("No single signal decided the outcome; the pattern across layers did.")
    return lines


def _v2(ctx: _Context) -> list[str]:
    lead = f", led by {ctx.lead_layer}" if ctx.lead_layer else ""
    lines = [
        f"The context model reads {ctx.person} as '{ctx.label}' with {ctx.confidence} confidence{lead}.",
        f"Its main evidence was {_evidence(ctx, 'paren', ', with {x} pointing the other way')}.",
    ]
    if ctx.mask:
        said = ""
        if ctx.baseline:
            said = f": the face-only baseline said {ctx.baseline}{_paren(ctx.baseline_confidence, ' at {v}')}"
        lines.append(
            f"That outweighs a positive-looking face{_paren(ctx.valence)}{said}, and context corrected the facial mask."
        )
    elif ctx.hidden:
        lines.append(
            f"Because {ctx.person}'s face was obscured for most of the window"
            f"{_paren(ctx.visibility, ' (visible {v} of the time)')}, the estimate uses context signals only."
        )
    else:
        lines.append("Each value comes from aggregate, anonymous measurements in the current window.")
    return lines


def _v3(ctx: _Context) -> list[str]:
    lines = [
        f"Context audit for {ctx.person}: '{ctx.label}' at {ctx.confidence} confidence.",
        f"Top signals were {_evidence(ctx, 'paren', ', while {x} pulled the other way')}.",
    ]
    if ctx.mask:
        behind = f" behind the face-only call of {ctx.baseline}" if ctx.baseline else ""
        lines.append(
            f"{ctx.person}'s expression looked positive{_paren(ctx.valence)}, yet context outweighed it "
            f"and corrected the facial mask{behind}."
        )
    elif ctx.hidden:
        lines.append(
            f"{ctx.person}'s face was out of view for most of the window{_paren(ctx.visibility, ' ({v} visible)')}, "
            "so context alone informed this result."
        )
    else:
        lines.append("The result reflects the combined pattern of these aggregate signals.")
    return lines


def _v4(ctx: _Context) -> list[str]:
    support, oppose = _split(ctx, "paren")
    if support:
        away = f", while {_join(oppose)} pointed away" if oppose else ""
        first = f"{_capitalize(_join(support))} pointed to '{ctx.label}' for {ctx.person}{away}."
    elif oppose:
        first = f"For {ctx.person}, the model chose '{ctx.label}' even though {_join(oppose)} pointed away."
    else:
        first = f"For {ctx.person}, the combined context layers pointed to '{ctx.label}'."
    lines = [first, f"The model holds {ctx.confidence} confidence in this reading."]
    if ctx.mask:
        over = ""
        if ctx.baseline:
            over = f" over a face-only reading of {ctx.baseline}{_paren(ctx.baseline_confidence)}"
        lines.append(
            f"Although {ctx.person}'s expression reads as positive{_paren(ctx.valence)}, "
            f"context corrected the facial mask{over}."
        )
    elif ctx.hidden:
        lines.append(
            f"{ctx.person}'s face was not visible for most of the window"
            f"{_paren(ctx.visibility, ' (visible {v} of the time)')}, so the face played no part."
        )
    else:
        lines.append("Together these context signals explain the prediction.")
    return lines


def _v5(ctx: _Context) -> list[str]:
    opener = f"Over the last {ctx.window}" if ctx.window else "In the current window"
    lines = [
        f"{opener}, {ctx.person}'s context pattern matched '{ctx.label}' with {ctx.confidence} confidence.",
        f"The leading signals were {_evidence(ctx, 'at', ', while {x} pulled the other way')}.",
    ]
    if ctx.mask and ctx.baseline:
        lines.append(
            f"A face-only model saw {ctx.baseline}{_paren(ctx.baseline_confidence)} from a positive expression"
            f"{_paren(ctx.valence)}, but context corrected the facial mask."
        )
    elif ctx.mask:
        lines.append(f"A positive expression{_paren(ctx.valence)} was overruled as context corrected the facial mask.")
    elif ctx.hidden:
        lines.append(
            f"{ctx.person}'s face stayed hidden for most of the window{_paren(ctx.visibility, ' ({v} visible)')}, "
            "so only context signals were used."
        )
    else:
        lines.append("No identity data was used to reach this result.")
    return lines


_VARIANTS: tuple[Callable[[_Context], list[str]], ...] = (_v0, _v1, _v2, _v3, _v4, _v5)


def _pick_variant(sheet: Mapping[str, Any], variant: int | None, rng: random.Random | None) -> int:
    if variant is not None:
        if not 0 <= variant < N_VARIANTS:
            raise ValueError(f"variant must be in [0, {N_VARIANTS}), got {variant}")
        return variant
    if rng is not None:
        return rng.randrange(N_VARIANTS)
    return zlib.crc32(str(sheet.get("event_id", "")).encode()) % N_VARIANTS


def render_variant(sheet: Mapping[str, Any] | BaseModel, variant: int) -> str:
    """One template variant, rendered as-is (no validation)."""
    sheet = as_dict(sheet)
    return " ".join(_VARIANTS[_pick_variant(sheet, variant, None)](_context(sheet)))


def render_template(
    sheet: Mapping[str, Any] | BaseModel,
    variant: int | None = None,
    rng: random.Random | None = None,
    *,
    verify: bool = True,
) -> str:
    """Template explanation for a fact sheet.

    ``variant`` picks a phrasing; otherwise ``rng`` picks one, otherwise the event id does
    (stable across calls). With ``verify`` the text is checked by ``validate`` and the next
    variant is tried if it ever fails; the chosen variant is returned as-is if none passes.
    """
    sheet = as_dict(sheet)
    start = _pick_variant(sheet, variant, rng)
    first = render_variant(sheet, start)
    if not verify:
        return first
    for offset in range(N_VARIANTS):
        text = first if offset == 0 else render_variant(sheet, (start + offset) % N_VARIANTS)
        if validate(text, sheet).ok:
            return text
    return first
