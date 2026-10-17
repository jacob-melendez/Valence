"""Prompt for the local narrator LLM: a strict system prompt, a compact fact-sheet rendering and
three validated few-shot examples.

The fact sheet is rendered as short ``key: value`` lines so prefill stays cheap, and every string
the model should quote (state display, confidence display, driver display names, formatted values)
is written exactly as the validator expects it. Each sheet also carries an ``Avoid`` line (other
states' words, mask words when no mask override) and a ``Note`` line for the face case, because a
1.7B model follows per-request reminders far better than rules buried in the system prompt.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from valence.narrator.schema import (
    STATE_DISPLAY,
    STATES,
    as_dict,
    face_hidden,
    face_only_label,
    fixed,
    format_percent,
    format_value,
    top_drivers,
)
from valence.narrator.validator import KEY_DRIVERS, MAX_SENTENCES, allowed_concepts

Message = dict[str, str]

SYSTEM_PROMPT = """You are the audit narrator for Valence, a privacy-preserving system that infers a person's state from their environment and behaviour rather than their face. Turn one fact sheet into a short audit explanation.

States:
- Focused: steady, quiet, low-movement work.
- Frustrated (Masked): stress building up, often behind a positive-looking face.
- Exhausted: fatigue, such as slumped posture, little movement and a long time without a break.
- Collaborative (Happy): positive group engagement, such as laughter, compliments and conversation.

Layers: Ambient Physics (room noise in dB, light in lux), Time Dynamics (time since the last break, time of day, trends), Somatic & Social (fists, posture, head-in-hands, movement, laughter, compliments, speech), Facial Surface (facial valence read by a face-only baseline model).

Rules:
1. Name the predicted state exactly as written in State, in single quotes, and give the confidence exactly as written.
2. Explain the prediction with at least two of the listed drivers, copying their names and values exactly. A driver marked "argues against" pulled the other way; say so.
3. Use only numbers that appear in the fact sheet. Never invent, convert or calculate numbers, and never count or rank the drivers.
4. Refer to the person only as the given Person #N or "they". Never mention gender, age, race, appearance, clothing, health conditions or diagnoses, and never use he, she, him, her or his.
5. Do not name any other state, and do not use the words in the Avoid line.
6. Mention smiles, expressions or masking only when Mask override is true; then contrast the positive face with the context and say that context corrected the facial mask.
7. If the face is hidden, say so and say the prediction relies on context alone.
8. Talk only about the listed drivers; do not bring in other signals.
9. Write 2 or 3 sentences and 30 to 70 words of plain text: no lists, markdown, headings or preamble."""

STATE_WORDS: dict[str, tuple[str, ...]] = {
    "Focused": ("focus",),
    "Frustrated_Masked": ("frustrated",),
    "Exhausted": ("exhausted",),
    "Collaborative_Happy": ("happy", "collaborative"),
}
MASK_WORDS: tuple[str, ...] = ("smile", "mask", "despite")
CONCEPT_WORDS: dict[str, str] = {
    "noise": "noise",
    "light": "light",
    "break": "breaks",
    "time_of_day": "the time of day",
    "fist": "fists",
    "slump": "posture",
    "head_in_hands": "head-in-hands",
    "withdrawal": "withdrawal",
    "motion": "movement",
    "laughter": "laughter",
    "compliment": "compliments",
    "speech": "speech",
    "face_valence": "the face or expressions",
    "face_visible": "face visibility",
}
NOTE_MASK = "contrast the positive face with the context and say that context corrected the facial mask."
NOTE_HIDDEN = "the face was hidden; say so and say the prediction relies on context alone."
NOTE_NO_FACE = "do not mention the face or expressions."


def _direction(driver: Mapping[str, Any]) -> str:
    effect = str(driver.get("effect") or "").lower()
    if effect in ("supports", "opposes"):
        return "supports" if effect == "supports" else "argues against"
    return "argues against" if float(driver.get("shap") or 0.0) < 0 else "supports"


def _driver_line(rank: int, layer: str, driver: Mapping[str, Any]) -> str:
    display = str(driver.get("display") or driver.get("feature") or "signal")
    level = str(driver.get("level") or "").strip()
    value = format_value(driver) + (f", {level}" if level else "")
    tags = [layer, _direction(driver)] + (["room"] if driver.get("scope") == "room" else [])
    return f"{rank}. {display}: {value} ({', '.join(t for t in tags if t)})"


def _state_display(state: str | None) -> str:
    return STATE_DISPLAY.get(str(state), str(state))


def _face_line(sheet: Mapping[str, Any]) -> str:
    face_only = sheet.get("face_only") or {}
    if face_hidden(sheet):
        frac = (sheet.get("data_quality") or {}).get("face_visible_frac")
        seen = f" (visible {format_percent(float(frac))} of the window)" if frac is not None else ""
        return f"Face: hidden{seen}; no face-only baseline"
    baseline = face_only_label(sheet)
    if not sheet.get("mask_override"):
        verdict = "agrees" if baseline == sheet.get("label") else "differs" if baseline else "unavailable"
        return f"Face: visible; face-only baseline {verdict}"
    parts = [f"'{_state_display(baseline)}'" if baseline else "a positive state"]
    if face_only.get("confidence") is not None:
        parts.append(f"at {format_percent(float(face_only['confidence']))}")
    text = " ".join(parts)
    if face_only.get("face_valence") is not None:
        text += f", facial valence {fixed(float(face_only['face_valence']), 2)}"
    return f"Face: visible; face-only baseline {text}"


def avoid_words(sheet: Mapping[str, Any] | BaseModel) -> list[str]:
    """Words the explanation must not use: other states' names, plus mask words without an override."""
    sheet = as_dict(sheet)
    label = sheet.get("label")
    contrast = face_only_label(sheet) if sheet.get("mask_override") else None
    words = [w for state in STATES if state not in (label, contrast) for w in STATE_WORDS[state]]
    if not sheet.get("mask_override"):
        words += MASK_WORDS
    return words


def face_note(sheet: Mapping[str, Any] | BaseModel) -> str | None:
    sheet = as_dict(sheet)
    if sheet.get("mask_override"):
        return NOTE_MASK
    if face_hidden(sheet):
        return NOTE_HIDDEN
    if "face_valence" not in allowed_concepts(sheet):
        return NOTE_NO_FACE
    return None


def render_fact_sheet(sheet: Mapping[str, Any] | BaseModel, k: int = KEY_DRIVERS) -> str:
    """Compact user-turn text (~120 tokens): the facts the explanation may use, nothing else."""
    sheet = as_dict(sheet)
    confidence = sheet.get("confidence_display") or format_percent(float(sheet.get("confidence") or 0.0))
    label = sheet.get("label_display") or _state_display(sheet.get("label"))
    lines = [
        f"Person: {sheet.get('person')}",
        f"State: '{label}' at {confidence} confidence",
        "Drivers (strongest first):",
        *(_driver_line(i, layer, d) for i, (layer, d) in enumerate(top_drivers(sheet, k), start=1)),
        _face_line(sheet),
        f"Mask override: {'true' if sheet.get('mask_override') else 'false'}",
    ]
    fill = sheet.get("window_fill_min")
    if fill:
        lines.append(f"Window: {format_value({'value': fill, 'unit': 'min'})}")
    avoid = avoid_words(sheet)
    if avoid:
        lines.append(f"Avoid: {', '.join(avoid)}")
    note = face_note(sheet)
    if note:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def _join(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _reason_hints(reasons: Iterable[str], sheet: Mapping[str, Any]) -> list[str]:
    person = str(sheet.get("person") or "the person")
    numbers: list[str] = []
    hints: dict[str, None] = {}
    for reason in reasons:
        kind, _, detail = reason.partition(":")
        if kind == "number":
            numbers.append(detail)
        elif kind == "words":
            short = detail.isdigit() and int(detail) < 30
            hints["Write 30 to 70 words." if short else "Keep it under 70 words."] = None
        elif kind == "sentences":
            hints[f"Use at most {MAX_SENTENCES} sentences."] = None
        elif kind == "state_missing":
            hints[f"Name the state exactly as '{sheet.get('label_display')}'."] = None
        elif kind == "other_state":
            words = [f"'{w}'" for w in STATE_WORDS.get(detail, (detail,))]
            hints[f"Do not use the word{'s' if len(words) > 1 else ''} {_join(words)}."] = None
        elif kind == "concept":
            hints[f"Do not mention {CONCEPT_WORDS.get(detail, detail)}; use only the listed drivers."] = None
        elif kind == "drivers":
            hints["Mention at least two of the listed drivers by name with their values."] = None
        elif kind == "mask_term":
            hints[f"Do not use '{detail}' and do not mention smiles or masking."] = None
        elif kind == "banned":
            term = detail.partition(":")[2] or detail
            hints[f"Do not use '{term}'; refer to the person only as {person} or they."] = None
        elif kind == "person":
            hints[f"Refer to the person only as {person}."] = None
        elif kind in ("empty", "timeout"):
            hints["Write the explanation now, briefly."] = None
    if numbers:
        quoted = _join([f"'{n}'" for n in dict.fromkeys(numbers)])
        hints[f"Do not write {quoted}; copy numbers exactly from the facts."] = None
    return list(hints)


def feedback_hint(reasons: Iterable[str], sheet: Mapping[str, Any] | BaseModel) -> str:
    """One line telling the model what the validator rejected in its previous draft."""
    hints = _reason_hints(reasons, as_dict(sheet))
    return f"Fix: {' '.join(hints)}" if hints else ""


def _driver(feature: str, display: str, value: float, unit: str, shap: float) -> dict[str, Any]:
    return {"feature": feature, "display": display, "value": value, "unit": unit, "shap": shap}


def _layer(name: str, *drivers: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "contribution": round(sum(abs(d["shap"]) for d in drivers), 2), "drivers": list(drivers)}


def _example(
    event_id: str, person: str, label: str, confidence: float, face_only: dict[str, Any], face_visible: bool,
    mask_override: bool, layers: list[dict[str, Any]], face_visible_frac: float, lux: float,
) -> dict[str, Any]:
    rest = round((1.0 - confidence) / 3, 2)
    probabilities = {s: (confidence if s == label else rest) for s in STATES}
    return {
        "event_id": event_id, "person": person, "label": label, "label_display": STATE_DISPLAY[label],
        "confidence": confidence, "confidence_display": format_percent(confidence),
        "probabilities": probabilities, "window_fill_min": 15.0, "face_only": face_only,
        "face_visible": face_visible, "mask_override": mask_override, "layers": layers,
        "data_quality": {"face_visible_frac": face_visible_frac, "lux": lux},
    }


_MASKED = _example(
    "evt_fewshot1", "Person #2", "Frustrated_Masked", 0.81,
    {"label": "Collaborative_Happy", "confidence": 0.66, "face_valence": 0.55}, True, True,
    [
        _layer("Ambient Physics", _driver("ambient_db", "Ambient noise", 76.4, "dB", 0.29),
               _driver("light_lux", "Light level", 210.0, "lux", 0.05)),
        _layer("Time Dynamics", _driver("hours_since_break", "Time since last break", 3.8, "h", 0.21)),
        _layer("Somatic & Social", _driver("fist_onsets_15m", "Clenched-fist onsets", 2, "count", 0.26),
               _driver("motion_per_min", "Movement", 3.1, "/min", 0.05)),
        _layer("Facial Surface", _driver("face_valence_mean", "Facial valence", 0.55, "", -0.14)),
    ],
    0.91, 210.0,
)
_AGREE = _example(
    "evt_fewshot2", "Person #3", "Collaborative_Happy", 0.79,
    {"label": "Collaborative_Happy", "confidence": 0.74, "face_valence": 0.71}, True, False,
    [
        _layer("Ambient Physics", _driver("light_lux", "Light level", 520.0, "lux", 0.06),
               _driver("ambient_db", "Ambient noise", 64.2, "dB", 0.04)),
        _layer("Time Dynamics", _driver("hours_since_break", "Time since last break", 2.6, "h", -0.12)),
        _layer("Somatic & Social", _driver("laughter_15m", "Laughter events", 4, "count", 0.30),
               _driver("speech_frac", "Speech share", 0.58, "%", 0.24),
               _driver("compliments_15m", "Compliments heard", 1, "count", 0.09)),
        _layer("Facial Surface", _driver("face_valence_mean", "Facial valence", 0.71, "", 0.08)),
    ],
    0.95, 520.0,
)
_HIDDEN = _example(
    "evt_fewshot3", "Person #1", "Exhausted", 0.74,
    {"label": None, "confidence": None, "face_valence": None}, False, False,
    [
        _layer("Ambient Physics", _driver("light_lux", "Light level", 140.0, "lux", 0.07),
               _driver("ambient_db", "Ambient noise", 41.5, "dB", 0.03)),
        _layer("Time Dynamics", _driver("hours_since_break", "Time since last break", 6.3, "h", 0.25),
               _driver("time_of_day_h", "Time of day", 19.5, "h", 0.06)),
        _layer("Somatic & Social", _driver("slump_frac", "Posture slump share", 0.71, "%", 0.29),
               _driver("head_in_hands_frac", "Head-in-hands share", 0.38, "%", 0.18),
               _driver("motion_per_min", "Movement", 0.4, "/min", 0.05)),
        _layer("Facial Surface", _driver("face_visible_frac", "Face visible share", 0.14, "%", 0.04)),
    ],
    0.14, 140.0,
)

FEW_SHOTS: tuple[tuple[dict[str, Any], str], ...] = (
    (
        _MASKED,
        "Model predicted 'Frustrated (Masked)' with 81% confidence for Person #2. Although they show a smiling "
        "expression (facial valence 0.55) that a face-only model read as 'Collaborative (Happy)', the system "
        "detected 2 clenched-fist onsets, ambient noise at 76 dB and 3.8 hours since their last break. "
        "Context corrected the facial mask.",
    ),
    (
        _AGREE,
        "Person #3 was classified as 'Collaborative (Happy)' with 79% confidence. The strongest signals were "
        "4 laughter events and a speech share of 58%, which point to lively shared conversation, while "
        "2.6 hours since the last break argued against it.",
    ),
    (
        _HIDDEN,
        "Model predicted 'Exhausted' with 74% confidence for Person #1. Their posture slumped for 71% of the "
        "window, with their head in their hands 38% of the time and 6.3 hours since their last break. Their "
        "face was hidden for most of the window (visible 14% of the time), so the prediction relies on "
        "context alone.",
    ),
)


def user_content(sheet: Mapping[str, Any] | BaseModel, feedback: Iterable[str] | None = None) -> str:
    """The user turn for a sheet, with a ``Fix:`` line when retrying after rejected drafts."""
    content = render_fact_sheet(sheet)
    hint = feedback_hint(feedback, sheet) if feedback else ""
    return f"{content}\n{hint}" if hint else content


def base_messages() -> list[Message]:
    """System prompt plus the few-shot turns: identical for every request (prompt-cacheable)."""
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for sheet, answer in FEW_SHOTS:
        messages.append({"role": "user", "content": render_fact_sheet(sheet)})
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_messages(sheet: Mapping[str, Any] | BaseModel, feedback: Iterable[str] | None = None) -> list[Message]:
    """System prompt, few-shot user/assistant pairs, then the real user turn."""
    return [*base_messages(), {"role": "user", "content": user_content(sheet, feedback)}]


def _merge_system(messages: Sequence[Message]) -> list[Message]:
    """For chat templates without a system role: fold the system prompt into the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return list(messages)
    system, rest = messages[0]["content"], list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
    return rest


def render_chat(tokenizer: Any, messages: Sequence[Message]) -> str:
    """Chat-template text with the generation prompt and thinking disabled (Qwen3 inserts an empty
    ``<think></think>`` block); falls back for templates that reject the kwarg or a system role."""
    kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
    try:
        return tokenizer.apply_chat_template(list(messages), enable_thinking=False, **kwargs)
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(list(messages), **kwargs)
    except Exception as exc:
        if "system" not in str(exc).lower():
            raise
    return tokenizer.apply_chat_template(_merge_system(messages), **kwargs)


def build_prompt(tokenizer: Any, sheet: Mapping[str, Any] | BaseModel, feedback: Iterable[str] | None = None) -> str:
    return render_chat(tokenizer, build_messages(sheet, feedback))


_SENTINEL = "⁣VALENCE_FACT_SHEET⁣"


def prompt_prefix(tokenizer: Any) -> str:
    """Rendered text every prompt starts with (system, few-shots, opening of the user turn)."""
    text = render_chat(tokenizer, [*base_messages(), {"role": "user", "content": _SENTINEL}])
    index = text.find(_SENTINEL)
    return text[:index] if index >= 0 else ""
