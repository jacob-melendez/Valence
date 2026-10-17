"""Deterministic gate every narrator explanation must pass before it is shown or logged.

The local LLM is fluent but not trusted: an explanation is accepted only if every number it
quotes comes from the fact sheet, it names the predicted state (and no other state as the
prediction), it talks only about the drivers that actually moved the model, it uses mask
language only when context overrode a positive-looking face, it contains no identity,
demographic, appearance or diagnostic language, and it stays short. Each failure appends a
short machine-readable reason ("number:57", "other_state:Exhausted", ...), so the engine can
retry with feedback or fall back to a template.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel

from valence.narrator.lexicon import (
    banned_hits,
    driver_concepts,
    mask_hits,
    names_state,
    remove_verbatim,
    text_concepts,
)
from valence.narrator.schema import (
    STATE_DISPLAY,
    STATES,
    as_dict,
    canonical_unit,
    face_hidden,
    face_only_label,
    person_number,
    top_drivers,
)

MIN_WORDS = 20
MAX_WORDS = 90
MAX_SENTENCES = 3
CONTEXT_DRIVERS = 6
KEY_DRIVERS = 3
REQUIRED_KEY_MENTIONS = 2

_EPS = 1e-9
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]{1,40}\|>")
_PERSON_REF = re.compile(r"\bperson\s*#?\s*(\d+)\b", re.IGNORECASE)
_CLOCK = re.compile(r"(?<![\d.:])(\d{1,2}):([0-5]\d)(?![\d:])(?:\s*([ap])\.?m\b\.?)?", re.IGNORECASE)
_HOUR_AMPM = re.compile(r"(?<![\d.:,])(\d{1,2})\s*([ap])\.?m\b\.?", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\d.,])(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_PERCENT_SUFFIX = re.compile(r"\s*(?:%|percent\b|per\s+cent\b|pct\b)", re.IGNORECASE)
_WORD = re.compile(r"\S+")
_ALNUM = re.compile(r"[A-Za-z0-9]")
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9#]|\s*$)")
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_ABBREVIATIONS = frozenset({"e.g", "i.e", "vs", "approx", "etc", "cf", "no", "fig", "avg", "est", "dr", "st"})
_PREAMBLE = re.compile(
    r"^(?:explanation|audit(?: (?:log|note|explanation|entry))?|summary|answer|output|response)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»", "`": "`"}
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)])\s+")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_INLINE_MARKUP = (
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)"), r"\1"),
    (re.compile(r"`+"), ""),
    (re.compile(r"\*+"), ""),
)


@dataclass(frozen=True, slots=True)
class NumberToken:
    """A number quoted in text; ``step`` is the precision it was written at."""

    raw: str
    value: float
    step: float
    percent: bool = False

    @property
    def tolerance(self) -> float:
        return self.step / 2 + _EPS


@dataclass(frozen=True, slots=True)
class SheetNumber:
    """A number the fact sheet contains (absolute value); ``percent`` if it is on a 0-100 scale."""

    value: float
    percent: bool = False


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    reasons: list[str]
    text: str
    word_count: int
    mentioned_concepts: list[str]
    sentence_count: int = 0
    mentioned_drivers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def strip_think(text: str) -> str:
    """Remove <think> blocks, an orphan closing tag's preamble, and an unterminated block."""
    text = _THINK_BLOCK.sub(" ", text or "")
    lower = text.lower()
    close = lower.rfind("</think>")
    if close != -1:
        text = text[close + len("</think>"):]
        lower = text.lower()
    start = lower.find("<think>")
    if start != -1:
        text = text[:start]
    return _SPECIAL_TOKEN.sub(" ", text).strip()


def _strip_markdown(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if _FENCE.match(line) or _RULE.match(line) or _HEADING.match(line):
            continue
        line = re.sub(r"^\s*>\s?", "", line)
        listed = bool(_LIST_MARKER.match(line))
        line = _LIST_MARKER.sub("", line)
        for pattern, repl in _INLINE_MARKUP:
            line = pattern.sub(repl, line)
        line = line.strip()
        if listed and line and line[-1] not in ".!?;:,":
            line += "."
        if line:
            lines.append(line)
    return " ".join(lines)


def _strip_quotes(text: str) -> str:
    while len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    return text


def clean_text(text: str) -> str:
    """The explanation as it will be shown: no thinking, markdown, preamble or wrapping quotes."""
    text = _strip_markdown(strip_think(text))
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_quotes(text)
    text = _PREAMBLE.sub("", text)
    return _strip_quotes(text).strip()


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _split_number(raw: str) -> list[tuple[str, str]]:
    """Split a digit run on commas unless they are thousands separators; returns (raw, numeric)."""
    raw = raw.rstrip(",")
    if raw.startswith("."):
        return [(raw, "0" + raw)]
    whole, dot, frac = raw.partition(".")
    groups = whole.split(",")
    if len(groups) == 1 or (len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
        return [(raw, "".join(groups) + dot + frac)]
    pieces = [g for g in groups if g]
    tail = pieces[-1] + dot + frac
    return [(p, p) for p in pieces[:-1]] + [(tail, tail)]


def extract_numbers(text: str) -> list[NumberToken]:
    """Every number in the text (person references excluded), with its written precision.

    Handles decimals, thousands separators, numbers glued to units ("72dB", "4.2h"), percentages
    ("88%", "88 percent") and clock times ("15:30", "3:30 pm", "4 pm" -> fractional hours).
    Signs are ignored: "-0.07" is 0.07.
    """
    found: list[tuple[int, NumberToken]] = []
    masked = text
    for match in _PERSON_REF.finditer(masked):
        masked = _blank(masked, *match.span())
    for match in _CLOCK.finditer(masked):
        hour, minute = int(match.group(1)), int(match.group(2))
        hour = _apply_meridiem(hour, match.group(3))
        found.append((match.start(), NumberToken(match.group(0).strip(), hour + minute / 60, 1 / 60)))
        masked = _blank(masked, *match.span())
    for match in _HOUR_AMPM.finditer(masked):
        hour = _apply_meridiem(int(match.group(1)), match.group(2))
        found.append((match.start(), NumberToken(match.group(0).strip(), float(hour), 1.0)))
        masked = _blank(masked, *match.span())
    for match in _NUMBER.finditer(masked):
        percent = bool(_PERCENT_SUFFIX.match(masked, match.end()))
        pieces = _split_number(match.group(0))
        for index, (raw, numeric) in enumerate(pieces):
            last = index == len(pieces) - 1
            decimals = len(numeric.partition(".")[2])
            token = NumberToken(
                raw + ("%" if last and percent else ""),
                float(numeric),
                10.0 ** -decimals,
                percent and last,
            )
            found.append((match.start(), token))
    return [token for _, token in sorted(found, key=lambda item: item[0])]


def _apply_meridiem(hour: int, meridiem: str | None) -> int:
    if not meridiem:
        return hour
    hour %= 12
    return hour + 12 if meridiem.lower() == "p" else hour


def _is_percent_leaf(key: str | None, parent: Mapping[str, Any] | None) -> bool:
    return key == "value" and parent is not None and canonical_unit(str(parent.get("unit") or "")) == "%"


def allowed_numbers(sheet: Mapping[str, Any] | BaseModel) -> list[SheetNumber]:
    """Every number the fact sheet contains: numeric leaves plus numbers inside its strings.

    Includes the person's own number. Values are absolute; ``percent`` marks 0-100 scale values
    (unit "%" drivers, "88%" strings) so fractions and percentages can match both ways.
    """
    sheet = as_dict(sheet)
    numbers: dict[SheetNumber, None] = {}

    def walk(node: Any, key: str | None, parent: Mapping[str, Any] | None) -> None:
        if node is None or isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            if math.isfinite(float(node)):
                numbers[SheetNumber(abs(float(node)), _is_percent_leaf(key, parent))] = None
        elif isinstance(node, str):
            for token in extract_numbers(node):
                numbers[SheetNumber(token.value, token.percent)] = None
        elif isinstance(node, Mapping):
            for child_key, child in node.items():
                walk(child, str(child_key), node)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, key, parent)

    walk(sheet, None, None)
    own = person_number(sheet)
    if own is not None:
        numbers[SheetNumber(float(own))] = None
    return list(numbers)


def number_allowed(token: NumberToken, allowed: list[SheetNumber]) -> bool:
    """True if the token is some sheet number rounded to the token's precision (or finer)."""
    tolerance = token.tolerance
    for number in allowed:
        if token.percent:
            candidates = ([number.value * 100] if number.value <= 1 else []) + ([number.value] if number.percent else [])
        else:
            candidates = [number.value] + ([number.value / 100] if number.percent else [])
        if any(abs(token.value - c) <= tolerance + _EPS * max(1.0, c) for c in candidates):
            return True
    return False


def split_sentences(text: str) -> list[str]:
    """Sentences, ignoring decimals ("4.2"), "#1", and common abbreviations ("e.g.", "vs.")."""
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        trailing = _TRAILING_WORD.search(text[start:match.start()])
        if match.group(0)[0] == "." and trailing and trailing.group(1).lower().rstrip(".") in _ABBREVIATIONS:
            continue
        parts.append(text[start:match.end()].strip())
        start = match.end()
    parts.append(text[start:].strip())
    return [part for part in parts if _ALNUM.search(part)]


def count_words(text: str) -> int:
    return sum(1 for token in _WORD.findall(text) if _ALNUM.search(token))


def allowed_concepts(sheet: Mapping[str, Any] | BaseModel) -> set[str]:
    """Concepts an explanation may mention: the top-6 drivers' concepts, plus face visibility
    when the face was hidden and facial valence when context overrode the face."""
    sheet = as_dict(sheet)
    allowed: set[str] = set()
    for _, driver in top_drivers(sheet, CONTEXT_DRIVERS):
        allowed |= driver_concepts(driver)
    if face_hidden(sheet):
        allowed.add("face_visible")
    if sheet.get("mask_override"):
        allowed.add("face_valence")
    return allowed


def driver_mentioned(driver: Mapping[str, Any], text: str, concepts: set[str] | None = None) -> bool:
    """A driver is mentioned if any of its concepts or its display string appears in the text."""
    concepts = text_concepts(text) if concepts is None else concepts
    if driver_concepts(driver) & concepts:
        return True
    display = str(driver.get("display") or "").strip()
    return bool(display) and display.lower() in text.lower()


def _check_length(text: str) -> tuple[list[str], int, int]:
    words = count_words(text)
    sentences = len(split_sentences(text))
    reasons = []
    if not MIN_WORDS <= words <= MAX_WORDS:
        reasons.append(f"words:{words}")
    if sentences > MAX_SENTENCES:
        reasons.append(f"sentences:{sentences}")
    return reasons, words, sentences


def _check_numbers(text: str, sheet: Mapping[str, Any]) -> list[str]:
    allowed = allowed_numbers(sheet)
    return [f"number:{token.raw}" for token in extract_numbers(text) if not number_allowed(token, allowed)]


def _check_states(text: str, sheet: Mapping[str, Any]) -> list[str]:
    label = str(sheet.get("label"))
    display = sheet.get("label_display")
    reasons = [] if names_state(text, label, display) else ["state_missing"]
    contrast = face_only_label(sheet) if sheet.get("mask_override") else None
    scan = remove_verbatim(text, display)
    for state in STATES:
        if state != label and state != contrast and names_state(scan, state, STATE_DISPLAY.get(state)):
            reasons.append(f"other_state:{state}")
    return reasons


def _check_drivers(text: str, sheet: Mapping[str, Any]) -> tuple[list[str], set[str], list[str]]:
    concepts = text_concepts(text)
    reasons = [f"concept:{name}" for name in sorted(concepts - allowed_concepts(sheet))]
    key = top_drivers(sheet, KEY_DRIVERS)
    mentioned = [str(d.get("feature")) for _, d in key if driver_mentioned(d, text, concepts)]
    required = min(REQUIRED_KEY_MENTIONS, len(key))
    if len(mentioned) < required:
        reasons.append(f"drivers:{len(mentioned)}<{required}")
    return reasons, concepts, mentioned


def _check_mask(text: str, sheet: Mapping[str, Any]) -> list[str]:
    if sheet.get("mask_override"):
        return []
    scan = remove_verbatim(text, sheet.get("label_display"))
    return [f"mask_term:{term}" for term in mask_hits(scan)]


def _check_identity(text: str, sheet: Mapping[str, Any]) -> list[str]:
    reasons = [f"banned:{category}:{term}" for category, term in banned_hits(text)]
    own = person_number(sheet)
    for match in _PERSON_REF.finditer(text):
        if int(match.group(1)) != own:
            reasons.append(f"person:{match.group(1)}")
    return reasons


def validate(text: str, sheet: Mapping[str, Any] | BaseModel) -> ValidationResult:
    """Check an explanation against its fact sheet; ``ok`` only if no rule fails."""
    sheet = as_dict(sheet)
    cleaned = clean_text(text)
    if not cleaned:
        return ValidationResult(False, ["empty"], "", 0, [])
    reasons, words, sentences = _check_length(cleaned)
    reasons += _check_numbers(cleaned, sheet)
    reasons += _check_states(cleaned, sheet)
    driver_reasons, concepts, mentioned = _check_drivers(cleaned, sheet)
    reasons += driver_reasons
    reasons += _check_mask(cleaned, sheet)
    reasons += _check_identity(cleaned, sheet)
    reasons = list(dict.fromkeys(reasons))
    return ValidationResult(
        ok=not reasons,
        reasons=reasons,
        text=cleaned,
        word_count=words,
        mentioned_concepts=sorted(concepts),
        sentence_count=sentences,
        mentioned_drivers=mentioned,
    )
