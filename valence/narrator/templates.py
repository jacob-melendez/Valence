"""Deterministic template explanations: the narrator's guaranteed, validator-clean fallback.

Six phrasings of one audit note. Every variant names the state with the verbatim
``label_display`` (so one person always reads "Upbeat", never "Collaborative"), quotes
``confidence_display``, and cites the key drivers (the core's top-3 supporting drivers) by copying
their ``display`` strings verbatim, so every number, level word and unit comes from the core.
Room-scope drivers are phrased as the room ("the room had ..."), person-scope ones after the
person. No qualifier is added to any driver. Optional parts, in order of priority:

- a face clause: the face-only contrast and "context corrected the facial mask" only when
  ``mask_override`` and the face is visible; "the face is not visible" only when the face is hidden
  (also when an override is still latched); otherwise a neutral closing sentence with no face words;
- the strongest opposing driver of the allowed set, framed as contrast ("although X pulled the
  other way", the only place a contrast word appears);
- an out-of-distribution note when ``data_quality.ood_features`` is non-empty.

Optional parts are dropped when the text would exceed the word limit. ``render_template``
re-validates and walks to the next variant if one ever fails; a minimal note is the last resort.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from valence.contracts import display_label
from valence.narrator.lexicon import driver_concept
from valence.narrator.schema import (
    allowed_drivers,
    as_dict,
    face_hidden,
    face_only_label,
    key_drivers,
    sheet_states,
)
from valence.narrator.validator import MAX_WORDS, count_words, validate

N_VARIANTS = 6
WORD_BUDGET = MAX_WORDS - 4


@dataclass(slots=True)
class _Context:
    person: str
    label: str
    confidence: str
    personal: list[str]  # person-scope key driver displays
    room: list[str]  # room-scope key driver displays
    opposing: str | None  # display of the strongest opposing allowed driver
    mask: bool
    hidden: bool
    face_label: str | None  # display of the face-only label (mask case)
    face_confidence: str | None
    face_display: str | None  # an allowed face-valence driver display (mask case)
    face_valence: str | None
    visibility: str | None
    window: str | None
    ood: bool


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _pct(value: float) -> str:
    return f"{round(100 * value)}%"


def _context(sheet: Mapping[str, Any], *, n_key: int = 3, contrast: bool = True, ood: bool = True) -> _Context:
    key = key_drivers(sheet)[:n_key]
    cited: list[str] = []
    personal: list[str] = []
    room: list[str] = []
    for d in key:
        display = str(d["display"])
        if display in cited:
            continue
        cited.append(display)
        (room if d.get("scope") == "room" else personal).append(display)
    hidden = face_hidden(sheet)
    mask = bool(sheet.get("mask_override")) and not hidden  # a latched override with a hidden face reads as hidden
    allowed = allowed_drivers(sheet)
    opposing = None
    if contrast:
        for d in allowed:
            concept = driver_concept(d)
            if d.get("effect") == "opposes" and str(d["display"]) not in cited and not (
                    concept == "face_valence" and not mask) and concept != "face_visible":
                opposing = str(d["display"])
                break
    face_only = sheet.get("face_only") or {}
    states = sheet_states(sheet)
    n_people = int(sheet.get("n_people") or 1)
    face_label = face_only_label(sheet) if mask else None
    face_display = None
    if mask:
        face_display = next((str(d["display"]) for d in allowed
                             if driver_concept(d) == "face_valence" and str(d["display"]) not in cited
                             and str(d["display"]) != opposing), None)
    valence = face_only.get("face_valence")
    quality = sheet.get("data_quality") or {}
    fill = sheet.get("window_fill_min")
    return _Context(
        person=str(sheet.get("person") or "the person"),
        label=str(sheet.get("label_display")),
        confidence=str(sheet.get("confidence_display")),
        personal=personal,
        room=room,
        opposing=opposing,
        mask=mask,
        hidden=hidden,
        face_label=display_label(face_label, n_people=n_people, states=states) if face_label else None,
        face_confidence=_pct(face_only["confidence"]) if mask and face_only.get("confidence") is not None else None,
        face_display=face_display,
        face_valence=f"{valence:+.2f}" if mask and valence is not None else None,
        visibility=_pct(quality["face_visible_frac"]) if hidden and quality.get("face_visible_frac") is not None
        else None,
        window=f"{fill:g} minutes" if fill and fill != 1 else None,
        ood=ood and bool(quality.get("ood_features")),
    )


def _evidence(ctx: _Context, person_lead: str, room_lead: str, both: str) -> str:
    """Key driver displays: person-scope ones after ``person_lead``, room ones after ``room_lead``."""
    person_part = f"{person_lead}{_join(ctx.personal)}" if ctx.personal else ""
    room_part = f"{room_lead}{_join(ctx.room)}" if ctx.room else ""
    if person_part and room_part:
        return both.format(p=person_part, r=room_part)
    return person_part or room_part


def _contrast(ctx: _Context, form: str = ", although {x} pulled the other way") -> str:
    return form.format(x=ctx.opposing) if ctx.opposing else ""


def _paren(value: str | None, form: str = " ({v})") -> str:
    return form.format(v=value) if value else ""


def _face_mask(ctx: _Context, style: int) -> str:
    face = ctx.face_label or "a positive state"
    seen = ctx.face_display or "a positive-looking face"
    if style == 0:
        return f"The face alone suggested {face}{_paren(ctx.face_confidence)}, but context corrected the facial mask."
    if style == 1:
        return f"The face-only reading of {face} came from {seen}, and context corrected the facial mask."
    return (f"A face-only model read {face} from a smile{_paren(ctx.face_valence, ' (valence {v})')}; "
            "context corrected the facial mask.")


def _face_hidden(ctx: _Context, style: int) -> str:
    seen = _paren(ctx.visibility, " (visible {v} of the window)")
    if style == 0:
        return f"The face is not visible right now{seen}, so the call relies on context."
    if style == 1:
        return f"With the face hidden{seen}, only context signals were used."
    return f"The face was out of view{seen}, so no face-only reading was used."


CLOSINGS: tuple[str, ...] = (
    "No single signal decided it; the pattern across layers did.",
    "These are aggregate context measurements, with no identity data involved.",
    "The pattern across the context layers carries this reading.",
)


def _clause(ctx: _Context, style: int) -> str:
    if ctx.mask:
        text = _face_mask(ctx, style % 3)
    elif ctx.hidden:
        text = _face_hidden(ctx, style % 3)
    else:
        text = CLOSINGS[style % 3]
    if ctx.ood:
        text = text[:-1] + "; some readings fall outside the training range."
    return text


def _v0(ctx: _Context) -> list[str]:
    ev = _evidence(ctx, "", "the room had ", "{p}, while {r}")
    return [f"Model predicted {ctx.label} ({ctx.confidence}) for {ctx.person}.",
            f"Main evidence: {ev}{_contrast(ctx)}.", _clause(ctx, 0)]


def _v1(ctx: _Context) -> list[str]:
    ev = _evidence(ctx, "", "in the room, ", "{p}; {r}")
    return [f"{ctx.person} reads as {ctx.label} with {ctx.confidence} confidence.",
            f"The model drew on {ev}{_contrast(ctx)}.", _clause(ctx, 1)]


def _v2(ctx: _Context) -> list[str]:
    ev = _evidence(ctx, f"signals for {ctx.person} were ", "the room had ", "{p}, and {r}")
    return [f"{ctx.label} ({ctx.confidence}) is the most likely state for {ctx.person}.",
            f"{_cap(ev)}{_contrast(ctx, '; {x} pulled the other way')}.", _clause(ctx, 2)]


def _v3(ctx: _Context) -> list[str]:
    ev = _evidence(ctx, "", "room: ", "{p}; {r}")
    return [f"Context audit for {ctx.person}: {ctx.label} at {ctx.confidence} confidence.",
            f"Signals: {ev}{_contrast(ctx, '; {x} argued against it')}.", _clause(ctx, 0)]


def _v4(ctx: _Context) -> list[str]:
    ev = _evidence(ctx, "", "the room had ", "{p}, and {r}")
    lead = f"Although {ctx.opposing} pulled the other way, " if ctx.opposing else ""
    first = f"{lead}the context layers place {ctx.person} at {ctx.label} ({ctx.confidence})."
    return [_cap(first), f"The model weighed {ev}.", _clause(ctx, 1)]


def _v5(ctx: _Context) -> list[str]:
    opener = f"Over the last {ctx.window}" if ctx.window else "In the current window"
    ev = _evidence(ctx, "", "the room had ", "{p}, while {r}")
    return [f"{opener}, {ctx.person} matched {ctx.label} with {ctx.confidence} confidence.",
            f"Key signals: {ev}{_contrast(ctx)}.", _clause(ctx, 2)]


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
    """One template variant (no validation), shortened to fit the word budget if needed."""
    sheet = as_dict(sheet)
    build = _VARIANTS[_pick_variant(sheet, variant, None)]
    text = ""
    for options in ({}, {"contrast": False}, {"contrast": False, "ood": False},
                    {"contrast": False, "ood": False, "n_key": 2}):
        text = " ".join(build(_context(sheet, **options)))
        if count_words(text) <= WORD_BUDGET:
            break
    return text


def minimal_text(sheet: Mapping[str, Any]) -> str:
    """Last resort: label, confidence and the first two key displays, nothing else."""
    displays = list(dict.fromkeys(str(d["display"]) for d in key_drivers(sheet)))[:2]
    evidence = f" Main evidence: {_join(displays)}." if displays else ""
    return (f"Model predicted {sheet.get('label_display')} ({sheet.get('confidence_display')}) for "
            f"{sheet.get('person')}, based on aggregate context signals.{evidence}")


def render_template(
    sheet: Mapping[str, Any] | BaseModel,
    variant: int | None = None,
    rng: random.Random | None = None,
    *,
    verify: bool = True,
) -> str:
    """Template explanation for a fact sheet.

    ``variant`` picks a phrasing; otherwise ``rng`` picks one, otherwise the event id does (stable
    across calls). With ``verify`` the text is validated and the next variant is tried on failure.
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
    return minimal_text(sheet)
