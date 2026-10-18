"""Prompt for the local narrator LLM: a strict system prompt, a compact fact-sheet rendering and
three validated few-shot examples.

The fact sheet goes to the model as compact ``key: display`` lines (~120 tokens): the person, the
state's ``label_display`` and ``confidence_display``, a face line only when the face is hidden or
overridden, the allowed drivers (top-5,
key drivers first) each with scope, level and effect followed by its verbatim ``display``, an
``avoid`` line and an optional out-of-range note. Everything the model may quote is written exactly
as the validator expects it; SHAP values, feature names and layer sums are never shown.

The system prompt depends only on the state set (the sheet's probability keys), so the system turn
plus the few-shot turns form a fixed prefix per state set that the engine prefills once into an
mlx_lm prompt cache (``prompt_prefix``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from valence.contracts import CORE_STATES, LAYERS, STATE_SETS, display_label
from valence.explain.explainer import build_fact_sheet, build_layers
from valence.narrator.schema import (
    allowed_drivers,
    as_dict,
    face_hidden,
    face_only_label,
    key_drivers,
    sheet_states,
)
from valence.narrator.validator import MAX_SENTENCES

Message = dict[str, str]

STATE_MEANINGS: dict[str, str] = {
    "Focused": "steady, low-movement work",
    "Frustrated_Masked": "stress building up, often behind a positive-looking face",
    "Exhausted": "fatigue: slumped posture, little movement, a long stretch without a break",
    "Collaborative_Happy": "positive shared engagement such as laughter and compliments in the room",
    "Overwhelmed": "distress such as head in hands and agitation",
    "Tense": "visible frustration such as clenched fists and restlessness",
}
LAYER_MEANINGS: dict[str, str] = {
    "Ambient Physics": "room noise and light",
    "Time Dynamics": "time since the last break, time of day",
    "Somatic & Social": "fists, posture, head in hands, movement, withdrawal, laughter, compliments, people present",
    "Facial Surface": "expression and face visibility, read by a separate face-only model",
}
# Words that would name each state (shown on the per-request avoid line for the other states).
STATE_WORDS: dict[str, tuple[str, ...]] = {
    "Focused": ("focus",),
    "Frustrated_Masked": ("frustrated",),
    "Exhausted": ("exhausted",),
    "Collaborative_Happy": ("happy", "upbeat"),
    "Overwhelmed": ("overwhelmed",),
    "Tense": ("tense",),
}
MASK_WORDS: tuple[str, ...] = ("smile", "mask")

_RULES = """Rules:
1. Name the state exactly as written after "state:" and give its confidence.
2. Use at least two of the first three drivers and no other signals. Copy each driver's wording exactly, numbers included.
3. Never write a number that is not in the facts. Never count, rank, convert or round anything.
4. A "room" driver describes the room: write "the room had ...", never "their noise" or "Person #N's light".
5. A driver marked "against" pulled the other way: introduce it with "although" or "despite". Never use "although" or "despite" for a driver marked "for".
6. Do not add words like high, low, loud, quiet, long or short unless they match the driver's level.
7. Mention smiles, expressions or masks only when the face line says "mask override", and then say that context corrected the facial mask. If the face is hidden, say so. If there is no face line, do not mention the face at all.
8. Refer to the person only as the given Person #N or "they". Never mention gender, age, race, appearance or health conditions, and never use he, she, him, her or his.
9. Do not name any other state and do not use the words on the "avoid" line.
10. Write {sentences} plain sentences, 30 to 70 words in total, with no lists, markdown or preamble."""


@lru_cache(maxsize=8)
def system_prompt(states: tuple[str, ...] = CORE_STATES) -> str:
    """Fixed system prompt for one state set (cacheable)."""
    lines = [
        "You write one short audit note explaining why Valence, a privacy-preserving context model, predicted a "
        "person's state. Restate only the facts you are given.",
        "",
        "States this model can predict:",
    ]
    for state in states:
        name = display_label(state, mask_override=True, n_people=2, states=states)
        if state == "Frustrated_Masked" and "Tense" not in states:
            name = f"Frustrated (or {name} when the face hides it)"
        alone = " (shown as Upbeat when the person is alone)" if state == "Collaborative_Happy" else ""
        lines.append(f"- {name}: {STATE_MEANINGS.get(state, state)}{alone}.")
    lines += ["", "Layers: " + "; ".join(f"{layer} ({LAYER_MEANINGS[layer]})" for layer in LAYERS) + ".", "",
              _RULES.format(sentences="2 or 3" if MAX_SENTENCES >= 3 else str(MAX_SENTENCES))]
    return "\n".join(lines)


SYSTEM_PROMPT = system_prompt(CORE_STATES)


def _effect(driver: Mapping[str, Any]) -> str:
    return "against" if driver.get("effect") == "opposes" else "for"


def prompt_drivers(sheet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Drivers shown to the model: key drivers first, then the rest of the allowed set."""
    key = key_drivers(sheet)
    rest = [d for d in allowed_drivers(sheet) if not any(d is k for k in key)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in [*key, *rest]:
        if str(d.get("display")) not in seen:
            seen.add(str(d.get("display")))
            out.append(d)
    return out


def _face_line(sheet: Mapping[str, Any]) -> str | None:
    """Only when the face is part of the story (hidden, or overridden); otherwise no line at all,
    since a 1.7B model tends to narrate any face line it is shown."""
    if face_hidden(sheet):
        frac = (sheet.get("data_quality") or {}).get("face_visible_frac")
        seen = f" (visible {round(100 * frac)}% of the window)" if frac is not None else ""
        return f"face: hidden{seen}, say so"
    if not sheet.get("mask_override"):
        return None
    face_only = sheet.get("face_only") or {}
    baseline = face_only_label(sheet)
    name = (display_label(baseline, n_people=int(sheet.get("n_people") or 1), states=sheet_states(sheet))
            if baseline else "a positive state")
    conf = face_only.get("confidence")
    return f"face: mask override, face-only model read {name}" + (f" ({round(100 * conf)}%)" if conf else "")


def avoid_words(sheet: Mapping[str, Any] | BaseModel) -> list[str]:
    """Words the explanation must not use: other states, mask words without an override, and
    "collaborative" for one person."""
    sheet = as_dict(sheet)
    label = sheet.get("label")
    contrast = face_only_label(sheet) if sheet.get("mask_override") else None
    words = [w for s in sheet_states(sheet) if s not in (label, contrast) for w in STATE_WORDS.get(s, ())]
    if label == "Tense" and "frustrated" not in words:
        words.append("frustrated")
    if int(sheet.get("n_people") or 0) == 1 and "collaborative" not in words:
        words.append("collaborative")
    if not sheet.get("mask_override"):
        words += MASK_WORDS
    return words


def render_fact_sheet(sheet: Mapping[str, Any] | BaseModel) -> str:
    """Compact user-turn text: the facts the explanation may use, nothing else."""
    sheet = as_dict(sheet)
    face = _face_line(sheet)
    lines = [
        f"person: {sheet.get('person')}",
        f"state: {sheet.get('label_display')}, {sheet.get('confidence_display')}",
        *([face] if face else []),
        "drivers (scope/level/effect: wording):",
    ]
    for i, d in enumerate(prompt_drivers(sheet), start=1):
        lines.append(f"{i}. {d.get('scope')}/{d.get('level')}/{_effect(d)}: {d.get('display')}")
    avoid = avoid_words(sheet)
    if avoid:
        lines.append(f"avoid: {', '.join(avoid)}")
    if (sheet.get("data_quality") or {}).get("ood_features"):
        lines.append("note: some readings out of training range")
    return "\n".join(lines)


def _join(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _reason_hints(reasons: Iterable[str], sheet: Mapping[str, Any]) -> list[str]:
    person = str(sheet.get("person") or "the person")
    numbers: list[str] = []
    hints: dict[str, None] = {}
    for reason in reasons:
        kind, _, detail = reason.partition(":")
        if kind in ("number", "sign"):
            numbers.append(detail)
        elif kind == "words":
            short = detail.isdigit() and int(detail) < 30
            hints["Write 30 to 70 words." if short else "Keep it under 70 words."] = None
        elif kind == "sentences":
            hints[f"Use at most {MAX_SENTENCES} sentences."] = None
        elif kind in ("state_missing", "state_negated"):
            hints[f"Name the state exactly as {sheet.get('label_display')}."] = None
        elif kind == "face_state":
            hints[f"The prediction is {sheet.get('label_display')}; name the face-only reading only as what the "
                  "face alone suggested."] = None
        elif kind == "direction":
            hints["Keep each driver's direction words (above, below, rising, falling) exactly as written."] = None
        elif kind == "time_of_day":
            hints["Do not describe the time of day with words like morning or evening; copy the time."] = None
        elif kind == "unmeasured":
            hints[f"Do not mention '{detail}'; use only the listed drivers."] = None
        elif kind == "other_state":
            words = [f"'{w}'" for w in STATE_WORDS.get(detail, (detail,))]
            hints[f"Do not use {_join(words)}."] = None
        elif kind == "collaborative_solo":
            hints["Do not use 'collaborative'; the person is alone."] = None
        elif kind == "outside_top5":
            hints["Use only the listed drivers."] = None
        elif kind == "drivers":
            hints["Copy at least two of the first three drivers word for word."] = None
        elif kind in ("qualifier", "effect"):
            hints["Copy the driver wording without adding words like high, low, loud or quiet."] = None
        elif kind == "contrast":
            hints["Use 'although' or 'despite' only for drivers marked against."] = None
        elif kind == "room_scope":
            hints["Describe room drivers as the room, not as the person's."] = None
        elif kind == "mask_term":
            hints[f"Do not use '{detail}' and do not mention smiles, expressions or masks."] = None
        elif kind == "banned":
            term = detail.partition(":")[2] or detail
            hints[f"Do not use '{term}'; refer to the person only as {person} or they."] = None
        elif kind == "person":
            hints[f"Refer to the person only as {person}."] = None
        elif kind in ("empty", "timeout", "length"):
            hints["Write the note now, briefly."] = None
    if numbers:
        quoted = _join([f"'{n}'" for n in dict.fromkeys(numbers)])
        hints[f"Do not write {quoted}; copy numbers exactly from the facts."] = None
    return list(hints)


def feedback_hint(reasons: Iterable[str], sheet: Mapping[str, Any] | BaseModel) -> str:
    """One line telling the model what the validator rejected in its previous draft."""
    hints = _reason_hints(reasons, as_dict(sheet))
    return f"fix: {' '.join(hints)}" if hints else ""


# ---------------------------------------------------------------------------------------------
# Few-shot examples: contract sheets built with the core's own explainer
# ---------------------------------------------------------------------------------------------


def _shot_sheet(*, event_id: str, person: str, label: str, confidence: float, n_people: int, mask: bool,
                face_visible: bool, face_label: str | None, face_conf: float | None, face_valence: float | None,
                values: dict[str, float], contribs: dict[str, float], face_visible_frac: float, lux: float,
                states: tuple[str, ...], tod_h: float | None = None) -> dict[str, Any]:
    others = [s for s in states if s != label]
    rest = round(1.0 - confidence, 2)
    shares = [round(rest / len(others), 2)] * len(others)
    shares[0] = round(rest - sum(shares[1:]), 2)
    probabilities = {**dict(zip(others, shares, strict=True)), label: confidence}
    sheet = build_fact_sheet(
        event_id=event_id, person=person, label=label,
        label_display=display_label(label, mask_override=mask, n_people=n_people, states=states),
        confidence=confidence, probabilities={s: probabilities[s] for s in states}, window_fill_min=15.0,
        n_people=n_people, face_only_label=face_label, face_only_confidence=face_conf, face_valence=face_valence,
        face_visible=face_visible, mask_override=mask, layers=build_layers(contribs, values, tod_h=tod_h),
        face_visible_frac=face_visible_frac, lux=lux, ood_features=[],
    )
    return sheet.model_dump(mode="json")


@lru_cache(maxsize=8)
def _few_shots(states: tuple[str, ...]) -> tuple[tuple[dict[str, Any], str], ...]:
    masked = _shot_sheet(
        event_id="shot-mask", person="Person #2", label="Frustrated_Masked", confidence=0.81, n_people=2, mask=True,
        face_visible=True, face_label="Collaborative_Happy", face_conf=0.66, face_valence=0.55,
        values={"db_mean_15m": 76.4, "fist_cnt_5m": 2.0, "hours_since_break": 3.8, "lux_now": 210.0, "face_now": 0.55},
        contribs={"db_mean_15m": 0.29, "fist_cnt_5m": 0.26, "hours_since_break": 0.21, "lux_now": 0.05,
                  "face_now": -0.14},
        face_visible_frac=0.91, lux=210.0, states=states,
    )
    agree = _shot_sheet(
        event_id="shot-agree", person="Person #3", label="Collaborative_Happy", confidence=0.79, n_people=3,
        mask=False, face_visible=True, face_label="Collaborative_Happy", face_conf=0.74, face_valence=0.71,
        values={"laugh_cnt_15m": 4.0, "compl_cnt_15m": 2.0, "n_people": 3.0, "hours_since_break": 2.6,
                "face_mean_5m": 0.71, "db_mean_5m": 64.0},
        contribs={"laugh_cnt_15m": 0.30, "compl_cnt_15m": 0.24, "n_people": 0.09, "hours_since_break": -0.12,
                  "face_mean_5m": 0.08, "db_mean_5m": 0.04},
        face_visible_frac=0.95, lux=520.0, states=states,
    )
    hidden = _shot_sheet(
        event_id="shot-hidden", person="Person #1", label="Exhausted", confidence=0.74, n_people=1, mask=False,
        face_visible=False, face_label=None, face_conf=None, face_valence=None,
        values={"slump_mean_5m": 0.71, "hours_since_break": 6.3, "hih_mean_15m": 0.38, "lux_now": 140.0,
                "face_visible_frac_5m": 0.14, "tod_sin": -0.97},
        contribs={"slump_mean_5m": 0.29, "hours_since_break": 0.25, "hih_mean_15m": 0.18, "lux_now": 0.07,
                  "face_visible_frac_5m": 0.04, "tod_sin": 0.03},
        face_visible_frac=0.14, lux=140.0, states=states, tod_h=19.5,
    )
    return (
        (masked,
         "Model predicted Frustrated / Masked Stress (81%) for Person #2. The room had average noise of 76 dB over "
         "15 minutes, and they showed 2 clenched-fist events in the last 5 minutes after 3.8 hours since the last "
         "break. Despite a smiling expression (valence +0.55) that the face-only model read as Collaborative / "
         "Happy, context corrected the facial mask."),
        (agree,
         "Person #3 reads as Collaborative / Happy with 79% confidence. The strongest signals were 4 laughter events "
         "in the room in the last 15 minutes and 2 compliments in the room in the last 15 minutes, with 3 people "
         "present. Although 2.6 hours since the last break pulled the other way, the shared mood in the room carried "
         "the call."),
        (hidden,
         "Model predicted Exhausted (74%) for Person #1. They were slumped 71% of the last 5 minutes, with head in "
         "hands 38% of the last 15 minutes and 6.3 hours since the last break. The face was hidden for most of the "
         "window, so the call relies on context."),
    )


def few_shots(states: tuple[str, ...] = CORE_STATES) -> tuple[tuple[dict[str, Any], str], ...]:
    """(sheet, answer) pairs: masked face, face agrees, face hidden (all pass ``validate``)."""
    return _few_shots(tuple(states))


FEW_SHOTS = few_shots(CORE_STATES)


def user_content(sheet: Mapping[str, Any] | BaseModel, feedback: Iterable[str] | None = None) -> str:
    """The user turn for a sheet, with a ``fix:`` line when retrying after rejected drafts."""
    content = render_fact_sheet(sheet)
    hint = feedback_hint(feedback, sheet) if feedback else ""
    return f"{content}\n{hint}" if hint else content


def base_messages(states: tuple[str, ...] = CORE_STATES) -> list[Message]:
    """System prompt plus the few-shot turns: identical for every request of a state set."""
    messages: list[Message] = [{"role": "system", "content": system_prompt(tuple(states))}]
    for sheet, answer in few_shots(tuple(states)):
        messages.append({"role": "user", "content": render_fact_sheet(sheet)})
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_messages(sheet: Mapping[str, Any] | BaseModel, feedback: Iterable[str] | None = None) -> list[Message]:
    """System prompt, few-shot user/assistant pairs, then the real user turn."""
    sheet = as_dict(sheet)
    return [*base_messages(sheet_states(sheet)), {"role": "user", "content": user_content(sheet, feedback)}]


def _merge_system(messages: Sequence[Message]) -> list[Message]:
    """For chat templates without a system role: fold the system prompt into the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return list(messages)
    system, rest = messages[0]["content"], list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
    return rest


def render_chat(tokenizer: Any, messages: Sequence[Message]) -> str:
    """Chat-template text with the generation prompt and thinking disabled (Qwen3 then emits an
    empty ``<think></think>`` block); falls back for templates that reject the kwarg or a system role."""
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


def prompt_prefix(tokenizer: Any, states: tuple[str, ...] = CORE_STATES) -> str:
    """Rendered text every prompt of a state set starts with (system, few-shots, opening of the
    final user turn): what the engine prefills into the prompt cache."""
    text = render_chat(tokenizer, [*base_messages(tuple(states)), {"role": "user", "content": _SENTINEL}])
    index = text.find(_SENTINEL)
    return text[:index] if index >= 0 else ""


STANDARD_STATE_SETS: tuple[tuple[str, ...], ...] = tuple(STATE_SETS.values())
