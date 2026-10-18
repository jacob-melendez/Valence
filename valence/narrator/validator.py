"""Deterministic gate every narrator explanation must pass before it is shown (contract 1.0).

The local LLM is fluent but not trusted. ``validate(text, sheet)`` cleans the text (``<think>``
blocks, markdown, preambles, wrapping quotes) and accepts it only if no rule fails. Each failure
appends a short machine-readable reason so the engine can retry with feedback:

======================  ===================================================================
reason                  rule
======================  ===================================================================
``empty``               nothing left after cleaning
``words:N``             20-90 words (tokens containing a letter or digit)
``sentences:N``         at most 3 sentences (decimals, "#1", "e.g." do not split; a lower-case
                        start after ". " does)
``number:X``            every number is a sheet number at the text's precision, and a number
                        bound to a driver is one of that driver's own numbers (below)
``sign:X``              an explicit sign ("-0.31") agrees with the sheet's sign
``state_missing``       the predicted state is named (lexicon stem or display phrase)
``state_negated``       ... and not negated ("is not Frustrated")
``other_state:S``       no other state is named
``face_state:S``        the face-only label (allowed only with ``mask_override``) is named as what
                        the face showed, never as the prediction ("they are really Upbeat")
``drivers:n<k``         at least 2 of the key drivers (top-3) are mentioned
``outside_top5:X``      nothing outside the allowed drivers (top-5) is mentioned
``qualifier:F:w``       a level word within 6 tokens of a mention agrees with its ``level``
                        (a negated one, "not high", with the levels it excludes)
``direction:F:w``       above/below, rising/falling, brighter/darker agree with the display
``time_of_day:w``       "morning", "evening", ... fit the sheet's clock time
``contrast:F:w``        despite/although-style contrast only for ``effect == "opposes"``
``effect:F:w``          "drove"/"supports"-style words only for ``effect == "supports"``
``room_scope:F``        room drivers are never attributed to the person ("their noise", "noise
                        from Person #1", "Person #1 generated noise")
``collaborative_solo``  "Collaborative" never appears when ``n_people == 1``
``mask_term:t``         mask/smile language only with ``mask_override``
``banned:cat:t``        no gendered, demographic, appearance or diagnostic terms
``unmeasured:t``        no signal Valence never measures ("heart rate", "typing")
``person:N``            "Person #N" only for the sheet's own number; no "Participant #2" style
======================  ===================================================================

Definitions:

* **Key / allowed drivers** (``schema.key_drivers`` / ``schema.allowed_drivers``) use the core's
  ranking (``valence.explain.explainer.top_drivers``: fact-sheet order, stable sort on |shap|).
  Key = ``top_drivers(k=3, effect="supports")``, what the core template cites (like the core
  template, over the context layers only when ``mask_override`` is set or the face is hidden). Allowed =
  ``top_drivers(k=5, effect=None)`` plus the key drivers, so the strongest opposing evidence may be
  named as contrast. The face concepts are also allowed when the sheet makes them the topic: facial
  expression with ``mask_override``, face visibility when the face is hidden.
* **Mentions**: a verbatim (case-insensitive) driver ``display`` binds to that driver; otherwise a
  concept word ("noise", "lux", "slumped", ...) binds to every sheet driver of that concept,
  narrowed to the allowed ones and then to those whose numbers appear within 6 tokens after it.
  A key driver also counts as mentioned when its display's lead number (the first one, unless the
  display reads "no ...") appears in a sentence that mentions its concept ("slumped 62% ... and 65%
  of the last 5 minutes"). Otherwise each concept word credits at most one key driver; displays
  shared by two drivers (tod_sin/tod_cos) count once.
* **Qualifiers**: tokens are words (hyphenated words are one token) and numbers. Tokens inside a
  verbatim display or the verbatim ``label_display`` are never read as qualifiers, mask terms or
  contrast (they come from the core and are correct by construction). A level word binds to the
  nearest mention within 6 tokens in the same sentence (ties go to the following mention, as
  adjectives precede nouns) and must be compatible with one of its drivers' levels
  (``lexicon.QUALIFIERS``; duration words invert for ``*_since_s`` features, whose level is high
  when the event was recent); a level word directly followed by "confidence"/"probability" describes
  the prediction and is skipped. A before-contrast ("despite", "although") binds to the first
  mention after it in the same clause (up to the next comma/semicolon), an after-contrast ("pulled
  the other way", "argued against") to the nearest mention before it; both need ``opposes``. With
  ``mask_override``, a sentence naming the face-only state is exempt from after-contrasts (the core
  template's "the face alone suggests X, but A and B point the other way" means away from the face).
* **Numbers**: allowed values are the numeric fields a reader sees (confidence, probabilities,
  window_fill_min, n_people, face_only confidence/valence, data_quality face_visible_frac/lux,
  driver values) and every number inside the sheet's strings (confidence_display, label_display,
  driver displays, the person's own number). SHAP values, layer contributions and the event id are
  internal and not quotable. A text number matches when it equals an allowed value rounded to the
  text's precision ("72" for 72.4, "4.2" for 4.25), and a number the sheet only shows rounded
  (a display's "72") is never quoted more precisely ("72.0"); fractions and percents convert both ways
  ("88%" <-> 0.88); clock times match as fractional hours; "the last minute" / "per minute" also
  allow 1 ("one minute"); number words count ("fourteen", "twenty-five", "a dozen", "a hundred",
  "half", "twice", and "one" when it counts something measured: "one clenched-fist event",
  "one hour", not "one signal" or "a positive one").
* **Number binding**: a number followed by a unit ("dB", "lux") must be the lead number or value of
  a driver of that concept (allowed drivers first). Otherwise it binds to the nearest driver mention
  or prediction anchor (a state name, "confidence") within 6 tokens in the same sentence, preferring
  its own clause, never across a colon, semicolon, dash or bracket; a window length ("the last 15
  minutes", "15-minute") binds to the mention before it (12 tokens). Bound to a mention, it must be
  one of the mention's candidate drivers' numbers (just the lead number or value when it directly
  precedes the mention: "14 clenched-fist events"), plus the face-only reading for face words, the
  measured lux for light, the head count for people; a percent next to a driver with no shares is
  about the prediction. Bound to a prediction anchor, or unbound, the sheet-wide list applies. An
  explicit sign is rejected when the bound driver states the opposite one.
* **Negation**: not/never/isn't/no longer/rather than/... up to two tokens before the predicted
  state's name rejects it; before a level word it inverts the word ("not high" fits normal/low).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from valence.narrator.lexicon import (
    COLLABORATIVE_RE,
    CONCEPT_RES,
    CONTRAST_AFTER,
    CONTRAST_BEFORE,
    DIRECTION_WORDS,
    FACE_CONTEXT,
    MASK_RE,
    NEGATORS,
    NOT_A_LEVEL_AFTER,
    NOT_A_SIGNAL,
    POSSESSIVES,
    PREDICTION_CUES,
    QUALIFIERS,
    ROOM_ACT_RE,
    ROOM_SOURCE,
    SINCE_NEGATORS,
    SINCE_QUALIFIERS,
    SUPPORT_WORDS,
    TIME_WORDS,
    banned_hits,
    driver_concept,
    state_mentions,
    unmeasured_hits,
)
from valence.narrator.schema import (
    allowed_drivers,
    as_dict,
    face_hidden,
    face_only_label,
    iter_drivers,
    key_drivers,
    person_number,
)

MIN_WORDS = 20
MAX_WORDS = 90
MAX_SENTENCES = 3
WINDOW = 6  # tokens
REQUIRED_KEY_MENTIONS = 2

_EPS = 1e-9
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]{1,40}\|>")
_PERSON_REF = re.compile(r"\bperson\s*#?\s*(\d+)\b", re.IGNORECASE)
_PERSON_NO_REF = re.compile(r"\bperson\s+(?:no\.?|number|num\.?)\s*#?\s*(\d+)\b", re.IGNORECASE)
_OTHER_REF = re.compile(
    r"\b(?:participant|user|subject|individual|occupant|attendee|member|student|worker|employee)s?\s*#\s*\d+\b"
    r"|\b(?:participant|occupant|attendee)s?\s+(?:no\.?\s*|number\s*)?\d+\b",
    re.IGNORECASE,
)
_CLOCK = re.compile(r"(?<![\d.:])(\d{1,2}):([0-5]\d)(?![\d:])(?:\s*([ap])\.?m\b\.?)?", re.IGNORECASE)
_HOUR_AMPM = re.compile(r"(?<![\d.:,])(\d{1,2})\s*([ap])\.?m\b\.?", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\d.,])(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_PERCENT_SUFFIX = re.compile(r"\s*(?:%|percent\b|per\s+cent\b|pct\b)", re.IGNORECASE)
_UNIT_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS_WORDS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())
               if w != "_"}
_SCALE_WORDS = {"hundred": 100, "thousand": 1000, "dozen": 12}
_LONE_WORDS = {"twice": (2.0, 1.0), "half": (0.5, 0.1)}  # value, step ("once" is mostly an adverb)
_ANY_NUMBER_WORD = "|".join(sorted([*_UNIT_WORDS, *_TENS_WORDS, *_SCALE_WORDS], key=len, reverse=True))
_NUMBER_WORD_RUN = re.compile(
    rf"\b(?:an?\s+(?=(?:hundred|thousand|dozen)\b))?(?:{_ANY_NUMBER_WORD})(?:[\s-]+(?:{_ANY_NUMBER_WORD}))*\b"
    r"|\b(?:twice|half)\b",
    re.IGNORECASE,
)
_NUMBER_WORD_PART = re.compile(rf"\b(?:an?|{_ANY_NUMBER_WORD}|twice|half)\b", re.IGNORECASE)
# "one" is a number only when it counts something the sheet measures ("one clenched-fist event",
# "one hour"); "one signal pulled the other way" or "a positive one" is ordinary prose.
_ONE_COUNTS = re.compile(
    r"^[\s-]+(?:(?:more|single|recent|last|full)\s+)?(?:clench\w*|fists?|laugh\w*|giggl\w*|chuckl\w*|compliment\w*"
    r"|prais\w*|events?|times?|person|people|occupants?|hours?|hrs?|minutes?|mins?|seconds?|secs?|breaks?|db|lux"
    r"|percent|per\s*cent|pct)\b",
    re.IGNORECASE,
)
_ONE_MINUTE = re.compile(r"\b(?:last|per|a|one) minute\b|/min\b", re.IGNORECASE)
_SIGNS = {"+": 1, "-": -1, "−": -1, "–": -1}
_WORD = re.compile(r"\S+")
_ALNUM = re.compile(r"[A-Za-z0-9]")
_TOKEN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,:]\d+)*%?")
_CLAUSE_BREAK = re.compile(r"[,;—–]")
# A sentence ends at .!? followed by whitespace and any word start (lower case too, so "that is
# all. nothing else." counts as two sentences), or at the end of the text.
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Za-z0-9#]|\s*$)")
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_ABBREVIATIONS = frozenset({"e.g", "i.e", "vs", "approx", "etc", "cf", "no", "fig", "avg", "est", "dr", "st"})
_LOWER_ONLY_ABBREVIATIONS = frozenset({"a.m", "p.m", "min", "mins", "hr", "hrs", "sec", "secs"})
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
_POSSESSIVE_BEFORE = re.compile(POSSESSIVES + r"\s+(?:[\w-]+\s+){0,2}$", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# Cleaning, sentences, words
# ---------------------------------------------------------------------------------------------


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


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, ignoring decimals, "#1" and common abbreviations."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        trailing = _TRAILING_WORD.search(text[start:match.start()])
        word = trailing.group(1).lower().rstrip(".") if trailing and match.group(0)[0] == "." else ""
        if word in _ABBREVIATIONS:
            continue
        if word in _LOWER_ONLY_ABBREVIATIONS and text[match.end():].lstrip()[:1].islower():
            continue
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if _ALNUM.search(text[s:e])]


def split_sentences(text: str) -> list[str]:
    return [text[s:e].strip() for s, e in sentence_spans(text)]


def count_words(text: str) -> int:
    return sum(1 for token in _WORD.findall(text) if _ALNUM.search(token))


# ---------------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NumberToken:
    """A number quoted in text; ``step`` is the precision it was written at; ``sign`` is +1/-1 when
    the text writes an explicit sign ("+0.31", "-2"), else 0."""

    raw: str
    value: float
    step: float
    percent: bool = False
    sign: int = field(default=0, compare=False)

    @property
    def tolerance(self) -> float:
        return self.step / 2 + _EPS


@dataclass(frozen=True, slots=True)
class SheetNumber:
    """A number the fact sheet shows (absolute value); ``percent`` if it is on a 0-100 scale;
    ``step`` is the precision it was shown at (0 for exact numeric fields)."""

    value: float
    percent: bool = False
    step: float = 0.0


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


def _apply_meridiem(hour: int, meridiem: str | None) -> int:
    if not meridiem:
        return hour
    hour %= 12
    return hour + 12 if meridiem.lower() == "p" else hour


def _word_numbers(run: str) -> list[tuple[int, int, float, float]]:
    """(start, end, value, step) for a run of number words: "twenty-five" (25), "a hundred" (100),
    "a dozen" (12), "half" (0.5), "twice" (2); words that do not combine become separate numbers."""
    groups: list[list[Any]] = []  # [start, end, value, step, last_kind]
    for part in _NUMBER_WORD_PART.finditer(run):
        word = part.group(0).lower()
        if word in _LONE_WORDS:
            groups.append([part.start(), part.end(), *_LONE_WORDS[word], "lone"])
            continue
        cur = groups[-1] if groups else None
        if word in ("a", "an"):
            groups.append([part.start(), part.end(), 1.0, 1.0, "article"])
        elif word in _SCALE_WORDS:
            if cur is not None and cur[4] in ("article", "unit", "tens", "compound"):
                cur[1], cur[2], cur[4] = part.end(), cur[2] * _SCALE_WORDS[word], "scale"
            else:
                groups.append([part.start(), part.end(), float(_SCALE_WORDS[word]), 1.0, "scale"])
        elif word in _TENS_WORDS:
            if cur is not None and cur[4] == "scale":
                cur[1], cur[2], cur[4] = part.end(), cur[2] + _TENS_WORDS[word], "tens"
            else:
                groups.append([part.start(), part.end(), float(_TENS_WORDS[word]), 1.0, "tens"])
        else:
            value = _UNIT_WORDS[word]
            if cur is not None and (cur[4] == "tens" and 1 <= value <= 9 or cur[4] == "scale"):
                cur[1], cur[2], cur[4] = part.end(), cur[2] + value, "compound"
            else:
                groups.append([part.start(), part.end(), float(value), 1.0, "unit"])
    return [(g[0], g[1], g[2], g[3]) for g in groups if g[4] != "article"]


def number_spans(text: str, *, words: bool = True) -> list[tuple[int, int, NumberToken]]:
    """(start, end, token) for every number in the text except person references (see
    ``extract_numbers``)."""
    found: list[tuple[int, int, NumberToken]] = []
    masked = text
    for match in _PERSON_REF.finditer(masked):
        masked = _blank(masked, *match.span())
    for match in _CLOCK.finditer(masked):
        hour = _apply_meridiem(int(match.group(1)), match.group(3))
        found.append((*match.span(), NumberToken(match.group(0).strip(), hour + int(match.group(2)) / 60, 1 / 60)))
        masked = _blank(masked, *match.span())
    for match in _HOUR_AMPM.finditer(masked):
        hour = _apply_meridiem(int(match.group(1)), match.group(2))
        found.append((*match.span(), NumberToken(match.group(0).strip(), float(hour), 1.0)))
        masked = _blank(masked, *match.span())
    for match in _NUMBER.finditer(masked):
        percent = bool(_PERCENT_SUFFIX.match(masked, match.end()))
        pieces = _split_number(match.group(0))
        before = match.start() - 1
        sign = _SIGNS.get(masked[before], 0) if before >= 0 else 0
        if sign and before > 0 and masked[before - 1].isalnum():
            sign = 0  # a range or hyphenated word ("3-5"), not a sign
        for index, (raw, numeric) in enumerate(pieces):
            last = index == len(pieces) - 1
            decimals = len(numeric.partition(".")[2])
            found.append((match.start(), match.end(), NumberToken(
                raw + ("%" if last and percent else ""), float(numeric), 10.0 ** -decimals, percent and last,
                sign if index == 0 else 0)))
    if words:
        for run in _NUMBER_WORD_RUN.finditer(masked):
            for start, end, value, step in _word_numbers(run.group(0)):
                start, end = run.start() + start, run.start() + end
                raw = masked[start:end]
                if raw.lower() == "one" and not _ONE_COUNTS.match(masked[end:end + 24]):
                    continue
                percent = bool(_PERCENT_SUFFIX.match(masked, end))
                found.append((start, end, NumberToken(raw, value, step, percent)))
    return sorted(found, key=lambda item: item[0])


def extract_numbers(text: str, *, words: bool = True) -> list[NumberToken]:
    """Every number in the text except person references, with its written precision.

    Handles decimals, thousands separators, numbers glued to units ("72dB"), percentages ("88%",
    "88 percent"), clock times ("15:30", "3:30 pm", "4 pm" -> fractional hours) and, with
    ``words``, number words: zero..nineteen ("one" only before something measured: "one hour",
    "one clenched-fist event"), the tens, compounds ("twenty-five"), "a hundred", "a dozen",
    "half", "twice". Values are magnitudes
    ("-0.07" is 0.07); an explicit sign is kept in ``sign``.
    """
    return [token for _, _, token in number_spans(text, words=words)]


def allowed_numbers(sheet: Mapping[str, Any] | BaseModel) -> list[SheetNumber]:
    """The numbers a reader of the sheet sees (see the module docstring for the exact list)."""
    sheet = as_dict(sheet)
    numbers: dict[SheetNumber, None] = {}

    def add(value: Any, percent: bool = False, step: float = 0.0) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            numbers[SheetNumber(abs(float(value)), percent, step)] = None

    def add_text(text: Any) -> None:
        for token in extract_numbers(str(text or ""), words=False):
            add(token.value, token.percent, token.step)

    add(sheet.get("confidence"))
    add_text(sheet.get("confidence_display"))
    add_text(sheet.get("label_display"))
    for value in (sheet.get("probabilities") or {}).values():
        add(value)
    add(sheet.get("window_fill_min"))
    add(sheet.get("n_people"))
    face_only = sheet.get("face_only") or {}
    add(face_only.get("confidence"))
    add(face_only.get("face_valence"))
    quality = sheet.get("data_quality") or {}
    add(quality.get("face_visible_frac"))
    add(quality.get("lux"))
    for _, driver in iter_drivers(sheet):
        add(driver.get("value"))
        add_text(driver.get("display"))
        if _ONE_MINUTE.search(str(driver.get("display") or "")):
            add(1)  # "the last minute" / "per minute" may be written "one minute"
    own = person_number(sheet)
    if own is not None:
        add(own)
    return list(numbers)


def number_allowed(token: NumberToken, allowed: Sequence[SheetNumber]) -> bool:
    """True if the token is some sheet number rounded to the token's precision.

    A number the sheet shows rounded ("72" in a display) cannot be quoted more precisely than
    shown ("72.0"), since that would claim a precision the sheet does not support.
    """
    for number in allowed:
        if token.percent:
            candidates = ([(number.value * 100, number.step * 100)] if number.value <= 1 else []) + (
                [(number.value, number.step)] if number.percent else [])
        else:
            candidates = [(number.value, number.step)] + (
                [(number.value / 100, number.step / 100)] if number.percent else [])
        for value, step in candidates:
            if token.step + _EPS >= step and abs(token.value - value) <= token.tolerance + _EPS * max(1.0, value):
                return True
    return False


# ---------------------------------------------------------------------------------------------
# Mentions and token windows
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Mention:
    start: int
    end: int
    concept: str | None
    drivers: list[dict[str, Any]]  # every sheet driver this mention can refer to
    display: bool  # verbatim display string
    first: int = 0  # token indices (inclusive)
    last: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    exempt: bool = False  # face topic the sheet itself raises (mask override / hidden face)

    @property
    def label(self) -> str:
        if self.display and self.drivers:
            return str(self.drivers[0].get("feature"))
        return str(self.concept or (self.drivers[0].get("feature") if self.drivers else "?"))


@dataclass(slots=True)
class _Tokens:
    spans: list[tuple[int, int]]
    words: list[str]
    sentence: list[int]
    clause: list[int]
    protected: list[bool]

    def index_range(self, start: int, end: int) -> tuple[int, int] | None:
        inside = [i for i, (s, e) in enumerate(self.spans) if s < end and e > start]
        return (inside[0], inside[-1]) if inside else None


def _tokenize(text: str, protected: list[tuple[int, int]]) -> _Tokens:
    sentences = sentence_spans(text)
    breaks = [m.start() for m in _CLAUSE_BREAK.finditer(text)]
    spans, words, sent, clause, prot = [], [], [], [], []
    for match in _TOKEN.finditer(text):
        s, e = match.span()
        spans.append((s, e))
        words.append(match.group(0).lower())
        sent.append(next((i for i, (a, b) in enumerate(sentences) if a <= s < b), len(sentences)))
        clause.append(sum(1 for b in breaks if b < s) + 1000 * sent[-1])
        prot.append(any(a <= s and e <= b for a, b in protected))
    return _Tokens(spans, words, sent, clause, prot)


def _find_all(lower: str, needle: str) -> list[tuple[int, int]]:
    out, start = [], lower.find(needle)
    while needle and start != -1:
        end = start + len(needle)
        before_ok = start == 0 or not lower[start - 1].isalnum()
        after_ok = end == len(lower) or not lower[end].isalnum()
        if before_ok and after_ok:
            out.append((start, end))
        start = lower.find(needle, start + 1)
    return out


def _overlaps(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < e and span[1] > s for s, e in spans)


def _driver_numbers(driver: Mapping[str, Any]) -> list[SheetNumber]:
    display = str(driver.get("display") or "")
    out = [SheetNumber(t.value, t.percent, t.step) for t in extract_numbers(display, words=False)]
    if _ONE_MINUTE.search(display):
        out.append(SheetNumber(1.0))
    value = driver.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        out.append(SheetNumber(abs(float(value)), False))
    return out


def find_mentions(text: str, sheet: Mapping[str, Any]) -> tuple[list[Mention], list[tuple[int, int]]]:
    """Driver mentions in the text plus the protected (verbatim core string) spans."""
    lower = text.lower()
    drivers = [d for _, d in iter_drivers(sheet)]
    allowed = allowed_drivers(sheet)
    mentions: list[Mention] = []
    protected: list[tuple[int, int]] = []
    label_display = str(sheet.get("label_display") or "").lower()
    if label_display:
        protected += _find_all(lower, label_display)
    by_display: dict[str, list[dict[str, Any]]] = {}
    for d in drivers:
        by_display.setdefault(str(d.get("display") or "").lower(), []).append(d)
    for display in sorted(by_display, key=len, reverse=True):
        for span in _find_all(lower, display):
            if not _overlaps(span, [(m.start, m.end) for m in mentions]):
                group = by_display[display]
                mentions.append(Mention(*span, driver_concept(group[0]), group, True))
                protected.append(span)
    taken = [(m.start, m.end) for m in mentions] + protected
    for concept, regex in CONCEPT_RES.items():
        for match in regex.finditer(text):
            if _overlaps(match.span(), taken):
                continue
            group = [d for d in drivers if driver_concept(d) == concept]
            mentions.append(Mention(match.start(), match.end(), concept, group, False))
    mentions.sort(key=lambda m: m.start)
    hidden, masked = face_hidden(sheet), bool(sheet.get("mask_override"))
    for m in mentions:
        in_allowed = [d for d in m.drivers if any(d is a for a in allowed)]
        m.candidates = in_allowed or list(m.drivers)
        m.exempt = (m.concept == "face_valence" and masked) or (m.concept == "face_visible" and hidden)
    return mentions, protected


def _bind_tokens(mentions: list[Mention], tokens: _Tokens) -> list[Mention]:
    bound = []
    for m in mentions:
        rng = tokens.index_range(m.start, m.end)
        if rng is not None:
            m.first, m.last = rng
            bound.append(m)
    return bound


def _narrow_by_numbers(m: Mention, tokens: _Tokens) -> None:
    """For a bare concept word with several candidates, prefer those whose numbers follow it."""
    if m.display or len(m.candidates) < 2:
        return
    near = [tokens.words[i] for i in range(m.last + 1, min(len(tokens.words), m.last + 1 + WINDOW))
            if tokens.sentence[i] == tokens.sentence[m.last]]
    quoted = [t for w in near for t in extract_numbers(w, words=False)]
    if not quoted:
        return
    narrowed = [d for d in m.candidates if any(number_allowed(q, _driver_numbers(d)) for q in quoted)]
    if narrowed:
        m.candidates = narrowed


@lru_cache(maxsize=8)
def _phrase_index(phrases: tuple[str, ...]) -> dict[str, list[tuple[str, list[str]]]]:
    """First word -> [(phrase, words)], longest phrase first."""
    index: dict[str, list[tuple[str, list[str]]]] = {}
    for phrase in sorted(phrases, key=lambda p: -len(p.split())):
        parts = phrase.split()
        index.setdefault(parts[0], []).append((phrase, parts))
    return index


def _phrase_hits(tokens: _Tokens, phrases: Sequence[str]) -> list[tuple[int, int, str]]:
    """(first, last, phrase) token ranges for (multi-word) phrases, skipping protected tokens."""
    index = _phrase_index(tuple(phrases))
    hits: list[tuple[int, int, str]] = []
    i, n = 0, len(tokens.words)
    while i < n:
        for phrase, parts in index.get(tokens.words[i], ()):
            j = i + len(parts) - 1
            if j < n and tokens.words[i:j + 1] == parts and not any(tokens.protected[i:j + 1]):
                hits.append((i, j, phrase))
                i = j
                break
        i += 1
    return hits


def _distance(m: Mention, first: int, last: int) -> int:
    if m.first <= last and first <= m.last:
        return 0
    return m.first - last if m.first > last else first - m.last


def _nearest(mentions: list[Mention], tokens: _Tokens, first: int, last: int, *, side: str = "any",
             same_clause: bool = False) -> Mention | None:
    best: tuple[int, int, Mention] | None = None
    for m in mentions:
        if tokens.sentence[m.first] != tokens.sentence[first]:
            continue
        after = m.first > last
        before = m.last < first
        if side == "after" and not after or side == "before" and not before:
            continue
        if same_clause and len({tokens.clause[k] for k in range(min(first, m.first), max(last, m.last) + 1)}) > 1:
            continue
        dist = _distance(m, first, last)
        if dist > WINDOW:
            continue
        key = (dist, 0 if after or dist == 0 else 1)
        if best is None or key < best[:2]:
            best = (*key, m)
    return best[2] if best else None


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    reasons: list[str]
    text: str
    word_count: int = 0
    sentence_count: int = 0
    mentioned_drivers: list[str] = field(default_factory=list)
    mentioned_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_length(text: str) -> tuple[list[str], int, int]:
    words, sentences = count_words(text), len(sentence_spans(text))
    reasons = [] if MIN_WORDS <= words <= MAX_WORDS else [f"words:{words}"]
    if sentences > MAX_SENTENCES:
        reasons.append(f"sentences:{sentences}")
    return reasons, words, sentences


_UNIT_CONCEPTS = {"db": "noise", "dba": "noise", "decibel": "noise", "decibels": "noise", "lux": "light"}
_TIME_UNITS = frozenset({"minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "second", "seconds",
                         "sec", "secs", "s"})
_WINDOW_LEADS = frozenset({"last", "past", "over", "previous", "within", "first", "recent"})
_PREDICTION_ANCHORS = NOT_A_SIGNAL | {"likely", "certain"}
_FACE_EXTRA_CONCEPTS = ("face_valence", "face_visible")
_NUMBER_TOKEN = re.compile(r"^\d")
_BIND_BREAK = re.compile(r"[;—–()]|:(?!\d)|(?<!\d):")  # a number never binds across these


def _extra_numbers(concept: str | None, sheet: Mapping[str, Any]) -> list[SheetNumber]:
    """Non-driver sheet numbers that belong to a concept (face-only reading, measured lux, head count)."""
    face_only = sheet.get("face_only") or {}
    quality = sheet.get("data_quality") or {}
    values: list[Any] = []
    if concept in _FACE_EXTRA_CONCEPTS:
        values = [face_only.get("confidence"), face_only.get("face_valence"), quality.get("face_visible_frac")]
    elif concept == "light":
        values = [quality.get("lux")]
    elif concept == "people":
        values = [sheet.get("n_people")]
    return [SheetNumber(abs(float(v))) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]


def _prediction_numbers(sheet: Mapping[str, Any]) -> list[SheetNumber]:
    """Numbers about the prediction itself: confidence, probabilities, face-only confidence, face share."""
    out = [SheetNumber(t.value, t.percent, t.step)
           for t in extract_numbers(str(sheet.get("confidence_display") or ""), words=False)]
    values = [sheet.get("confidence"), *(sheet.get("probabilities") or {}).values(),
              (sheet.get("face_only") or {}).get("confidence"), (sheet.get("data_quality") or {}).get("face_visible_frac")]
    out += [SheetNumber(abs(float(v))) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return out


def _lead_numbers(driver: Mapping[str, Any]) -> list[SheetNumber]:
    """What a count or a unit quotes for a driver: its display's lead number and its value (0 for a
    display that reads "no ...")."""
    out: list[SheetNumber] = []
    lead = _lead_number(driver)
    if lead is not None:
        out.append(SheetNumber(lead.value, lead.percent, lead.step))
    if str(driver.get("display") or "").lower().startswith("no "):
        out.append(SheetNumber(0.0))
    value = driver.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        out.append(SheetNumber(abs(float(value))))
    return out


def _signed_numbers(driver: Mapping[str, Any]) -> list[tuple[SheetNumber, int]]:
    """Driver numbers whose sign the sheet states: signed display numbers and the (signed) value."""
    out = [(SheetNumber(t.value, t.percent, t.step), t.sign)
           for t in extract_numbers(str(driver.get("display") or ""), words=False) if t.sign]
    value = driver.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and value:
        out.append((SheetNumber(abs(float(value))), 1 if value > 0 else -1))
    return out


@dataclass(slots=True)
class _Binding:
    """What a quoted number is about: driver mentions (``mentions``) or the prediction (empty)."""

    mentions: list[Mention]
    unit: str | None = None


def _separated(text: str, tokens: _Tokens, a: int, b: int) -> bool:
    """Is there a colon, semicolon, dash or bracket between tokens a and b?"""
    lo, hi = sorted((a, b))
    return bool(_BIND_BREAK.search(text, tokens.spans[lo][1], tokens.spans[hi][0])) if lo != hi else False


def _bind_number(text: str, first: int, last: int, mentions: list[Mention], anchors: list[int],
                 tokens: _Tokens) -> _Binding | None:
    """The nearest driver mention (or prediction anchor) within the window of a number, same sentence.

    A unit right after the number ("dB", "lux") binds it to that concept. A window length ("the last
    15 minutes") binds to the mention before it. A candidate in the number's own clause beats one
    across a comma; no binding crosses a colon, semicolon, dash or bracket. Ties keep every nearest
    candidate, and a tie with a prediction anchor makes the number about the prediction.
    """
    sentence = tokens.sentence[first]
    nxt = tokens.words[last + 1] if last + 1 < len(tokens.words) and tokens.sentence[last + 1] == sentence else ""
    if nxt in _UNIT_CONCEPTS:
        return _Binding([], _UNIT_CONCEPTS[nxt])
    hyphenated = text[tokens.spans[last][1]:tokens.spans[last][1] + 1] == "-"  # "the 15-minute average"
    window_spec = nxt in _TIME_UNITS and (hyphenated or first > 0 and tokens.words[first - 1] in _WINDOW_LEADS
                                          and tokens.clause[first - 1] == tokens.clause[first])
    reach = 2 * WINDOW if window_spec else WINDOW  # "slumped 55% of the time over the last 5 minutes"
    scored: list[tuple[tuple[int, int], Mention | None]] = []

    def consider(near: int, dist: int, item: Mention | None) -> None:
        if dist <= reach and (dist == 0 or not _separated(text, tokens, first, near)):
            scored.append(((0 if dist == 0 or tokens.clause[near] == tokens.clause[first] else 1, dist), item))

    for m in mentions:
        if tokens.sentence[m.first] != sentence or (window_spec and m.last >= first) or m.display:
            continue  # a verbatim display already carries its own numbers
        if m.first == m.last and tokens.words[m.first] in _UNIT_CONCEPTS and m.first > 0 and m.first - 1 != last \
                and _NUMBER_TOKEN.match(tokens.words[m.first - 1]):
            continue  # the unit of another number ("62 dB and 2.8 hours") describes that number only
        consider(m.first if m.first > last else m.last, _distance(m, first, last), m)
    if window_spec:
        if not scored:
            return None  # a window length with nothing before it to describe: any sheet number
    else:
        for i in anchors:
            if tokens.sentence[i] == sentence:
                consider(i, _distance_index(i, first, last), None)
    if not scored:
        return None
    best = min(key for key, _ in scored)
    tied = [item for key, item in scored if key == best]
    if any(item is None for item in tied):
        return None
    return _Binding([m for m in tied if m is not None])


def _distance_index(i: int, first: int, last: int) -> int:
    if first <= i <= last:
        return 0
    return i - last if i > last else first - i


def _prediction_anchors(text: str, tokens: _Tokens) -> list[int]:
    """Token indices that make a nearby number about the prediction: state names, "confidence"."""
    out = [i for i, w in enumerate(tokens.words) if w in _PREDICTION_ANCHORS]
    for start, end, _ in state_mentions(text):
        rng = tokens.index_range(start, end)
        if rng is not None:
            out += list(range(rng[0], rng[1] + 1))
    return out


def _check_numbers(text: str, sheet: Mapping[str, Any], mentions: list[Mention], tokens: _Tokens,
                   protected: list[tuple[int, int]]) -> list[str]:
    """Every number is a sheet number; one bound to a driver (nearest mention, or a unit) must be
    one of that driver's own numbers, and an explicit sign must agree with the sheet's."""
    allowed = allowed_numbers(sheet)
    anchors = _prediction_anchors(text, tokens)
    drivers = [d for _, d in iter_drivers(sheet)]
    allowed_set = allowed_drivers(sheet)
    reasons: list[str] = []
    for start, end, token in number_spans(text):
        if not number_allowed(token, allowed):
            reasons.append(f"number:{token.raw}")
            continue
        if _overlaps((start, end), protected):
            continue  # verbatim core string
        rng = tokens.index_range(start, end)
        if rng is None:
            continue
        binding = _bind_number(text, *rng, mentions, anchors, tokens)
        if binding is None:
            continue
        pool: list[SheetNumber] = []
        signed: list[tuple[SheetNumber, int]] = []
        if binding.unit is not None:
            group = [d for d in drivers if driver_concept(d) == binding.unit]
            group = [d for d in group if any(d is a for a in allowed_set)] or group
            if not group:
                continue
            for d in group:
                pool += _lead_numbers(d)
                signed += _signed_numbers(d)
            pool += _extra_numbers(binding.unit, sheet)
        else:
            for m in binding.mentions:
                if not m.candidates:
                    pool = []
                    break
                count = m.first == rng[1] + 1 and not m.display
                for d in m.candidates:
                    pool += _lead_numbers(d) if count else _driver_numbers(d)
                    signed += _signed_numbers(d)
                    if str(d.get("display") or "").lower().startswith("no "):
                        pool.append(SheetNumber(0.0))
                pool += _extra_numbers(m.concept, sheet)
                if m.concept in _FACE_EXTRA_CONCEPTS:
                    valence = (sheet.get("face_only") or {}).get("face_valence")
                    if isinstance(valence, (int, float)) and valence:
                        signed.append((SheetNumber(abs(float(valence))), 1 if valence > 0 else -1))
            if not pool:
                continue
            if token.percent and not any(n.percent or n.value <= 1 for n in pool):
                pool = _prediction_numbers(sheet)  # a share next to a non-share driver is about the call
        if not number_allowed(token, pool):
            reasons.append(f"number:{token.raw}")
        elif token.sign:
            same = any(sign == token.sign and number_allowed(token, [n]) for n, sign in signed)
            if not same and any(sign == -token.sign and number_allowed(token, [n]) for n, sign in signed):
                reasons.append(f"sign:{'+' if token.sign > 0 else '-'}{token.raw}")
    return reasons


def _negated(tokens: _Tokens, first: int, negators: Sequence[str] = NEGATORS) -> bool:
    """A negator ends right before token ``first`` or one token earlier, in the same sentence."""
    for negator in negators:
        parts = negator.split()
        for gap in (0, 1):
            end = first - 1 - gap
            begin = end - len(parts) + 1
            if begin >= 0 and tokens.words[begin:end + 1] == parts and tokens.sentence[begin] == tokens.sentence[first]:
                return True
    return False


def _check_states(text: str, sheet: Mapping[str, Any], tokens: _Tokens, label_spans: list[tuple[int, int]]) -> list[str]:
    """The predicted state is named (and not negated); no other state is; the face-only state (mask
    override) only as what the face showed."""
    label = sheet.get("label")
    contrast = face_only_label(sheet) if sheet.get("mask_override") else None
    named = state_mentions(text)
    reasons = [] if any(state == label for _, _, state in named) else ["state_missing"]
    for start, end, state in named:
        rng = tokens.index_range(start, end)
        if state == label:
            if rng is not None and _negated(tokens, rng[0]):
                reasons.append("state_negated")
        elif state == contrast:
            if rng is not None and not _face_attributed(tokens, *rng, label_spans):
                reasons.append(f"face_state:{state}")
        else:
            reasons.append(f"other_state:{state}")
    if int(sheet.get("n_people") or 0) == 1 and COLLABORATIVE_RE.search(text):
        reasons.append("collaborative_solo")
    return reasons


def _face_attributed(tokens: _Tokens, first: int, last: int, label_spans: list[tuple[int, int]]) -> bool:
    """Is a face-only state tied to the face ("the face alone suggested X", "a smile the face-only
    model read as X")? A face word must be near it in the sentence, and after a prediction cue
    ("predicted", "reads as", "they are really") it must be in the same clause."""
    sentence = tokens.sentence[first]

    def face_word(i: int) -> bool:
        if tokens.sentence[i] != sentence or tokens.words[i] not in FACE_CONTEXT:
            return False
        a, b = tokens.spans[i]
        return not any(s <= a and b <= e for s, e in label_spans)  # "Masked" inside the label itself

    near = [i for i in range(max(0, first - 12), min(len(tokens.words), last + 7)) if face_word(i)]
    if not near:
        return False
    cue = any(tokens.words[i:i + len(c.split())] == c.split()
              for c in PREDICTION_CUES for i in range(max(0, first - 4), first)
              if tokens.sentence[i] == sentence)
    return not cue or any(tokens.clause[i] == tokens.clause[first] for i in near)


def _lead_number(driver: Mapping[str, Any]) -> NumberToken | None:
    """The number a display is about (its first one), unless the display reads "no ..."."""
    display = str(driver.get("display") or "")
    if display.lower().startswith("no "):
        return None
    found = extract_numbers(display, words=False)
    return found[0] if found else None


def _check_coverage(mentions: list[Mention], sheet: Mapping[str, Any], text: str,
                    tokens: _Tokens) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    allowed = allowed_drivers(sheet)
    for m in mentions:
        if m.exempt:
            continue
        if not m.drivers or not any(any(d is a for a in allowed) for d in m.drivers):
            reasons.append(f"outside_top5:{m.label}")
    key = key_drivers(sheet)
    credited: list[dict[str, Any]] = []
    displays: set[str] = set()

    def credit(d: dict[str, Any]) -> None:
        if str(d.get("display")) not in displays:
            displays.add(str(d.get("display")))
            credited.append(d)

    # 1. verbatim displays
    for m in mentions:
        if m.display:
            for d in m.drivers:
                if any(d is k for k in key):
                    credit(d)
    # 2. paraphrases: the display's lead number in a sentence that mentions the driver's concept
    sentences = sentence_spans(text)
    quoted = [(t, i) for i, (a, b) in enumerate(sentences) for t in extract_numbers(text[a:b], words=False)]
    concept_sentences = {(m.concept, tokens.sentence[m.first]) for m in mentions}
    for k in key:
        lead = _lead_number(k)
        if lead is None or str(k.get("display")) in displays:
            continue
        lead_number = [SheetNumber(lead.value, lead.percent, lead.step)]
        if any(number_allowed(t, lead_number) and (driver_concept(k), i) in concept_sentences for t, i in quoted):
            credit(k)
    # 3. a bare concept word credits at most one key driver of that concept
    used: set[str] = {str(driver_concept(d)) for d in credited}
    for m in mentions:
        if m.display or m.concept in used:
            continue
        pool = [k for k in key if driver_concept(k) == m.concept and str(k.get("display")) not in displays]
        preferred = [d for d in pool if any(d is c for c in m.candidates)] or pool
        if preferred:
            credit(preferred[0])
            used.add(str(m.concept))
    required = min(REQUIRED_KEY_MENTIONS, len({str(k.get("display")) for k in key}))
    if len(credited) < required:
        reasons.append(f"drivers:{len(credited)}<{required}")
    mentioned = [str(d.get("feature")) for m in mentions for d in m.candidates if not m.exempt]
    mentioned += [str(d.get("feature")) for d in credited]
    return reasons, list(dict.fromkeys(mentioned))


def _is_since(driver: Mapping[str, Any]) -> bool:
    return str(driver.get("feature") or "").endswith("_since_s")


def _level_ok(word: str, driver: Mapping[str, Any], negated: bool = False) -> bool:
    """Is a level word compatible with a driver's level? Durations invert for "time since the
    last event" features, whose level is high when the event was recent. A negated word ("not
    high") fits exactly the levels the word itself does not."""
    if _is_since(driver) and word in SINCE_QUALIFIERS:
        fits = driver.get("level") in SINCE_QUALIFIERS[word]
    elif word in QUALIFIERS:
        fits = driver.get("level") in QUALIFIERS[word]
    else:
        return True  # "recent" says nothing about a non-time-since driver's level
    return fits != negated


_DIRECTION_IN_DISPLAY = re.compile(r"\b(above|below|rising|falling|brighter|darker)\b")
_DISPLAY_DIRECTION = {"above": "up", "rising": "up", "brighter": "up", "below": "down", "falling": "down",
                      "darker": "down"}


def driver_direction(driver: Mapping[str, Any]) -> str | None:
    """"up"/"down" when the display states a direction ("noise 1 dB above its median")."""
    match = _DIRECTION_IN_DISPLAY.search(str(driver.get("display") or "").lower())
    return _DISPLAY_DIRECTION[match.group(1)] if match else None


def _face_contrast_sentences(text: str, sheet: Mapping[str, Any], tokens: _Tokens) -> set[int]:
    """Sentences that set the context against the face-only reading (mask override only): there
    "X points the other way" means away from the face, not against the predicted label."""
    contrast = face_only_label(sheet) if sheet.get("mask_override") else None
    if contrast is None:
        return set()
    out: set[int] = set()
    for start, end, state in state_mentions(text):
        rng = tokens.index_range(start, end)
        if state == contrast and rng is not None:
            out.add(tokens.sentence[rng[0]])
    return out


_LEVEL_WORDS = tuple(dict.fromkeys([*QUALIFIERS, *SINCE_QUALIFIERS]))


def _check_qualifiers(mentions: list[Mention], tokens: _Tokens, face_sentences: set[int] = frozenset()) -> list[str]:
    reasons: list[str] = []
    for first, last, word in _phrase_hits(tokens, _LEVEL_WORDS):
        following = tokens.words[last + 1] if last + 1 < len(tokens.words) else ""
        if following in NOT_A_SIGNAL or following in NOT_A_LEVEL_AFTER.get(word, ()):
            continue
        m = _nearest(mentions, tokens, first, last)
        if m is None or m.exempt or not m.candidates:
            continue
        if word in DIRECTION_WORDS and any(driver_direction(d) for d in m.candidates):
            continue  # "higher"/"darker" on a delta driver states its direction (checked below)
        negated = _negated(tokens, first) or (
            word in SINCE_QUALIFIERS and any(_is_since(d) for d in m.candidates)
            and _negated(tokens, first, SINCE_NEGATORS))
        if not any(_level_ok(word, d, negated) for d in m.candidates):
            reasons.append(f"qualifier:{m.label}:{'not ' if negated else ''}{word}")
    for first, last, word in _phrase_hits(tokens, tuple(DIRECTION_WORDS)):
        m = _nearest(mentions, tokens, first, last)
        if m is None or m.exempt:
            continue
        directed = [d for d in m.candidates if driver_direction(d)]
        if directed and not any(driver_direction(d) == DIRECTION_WORDS[word] for d in directed):
            reasons.append(f"direction:{directed[0].get('feature')}:{word}")
    for first, last, word in _phrase_hits(tokens, CONTRAST_BEFORE):
        m = _nearest(mentions, tokens, first, last, side="after", same_clause=True)
        if m is not None and not m.exempt and m.candidates and not any(
                d.get("effect") == "opposes" for d in m.candidates):
            reasons.append(f"contrast:{m.label}:{word}")
    for first, last, word in _phrase_hits(tokens, CONTRAST_AFTER):
        if tokens.sentence[first] in face_sentences:
            continue
        m = _nearest(mentions, tokens, first, last, side="before")
        if m is not None and not m.exempt and m.candidates and not any(
                d.get("effect") == "opposes" for d in m.candidates):
            reasons.append(f"contrast:{m.label}:{word}")
    for first, last, word in _phrase_hits(tokens, SUPPORT_WORDS):
        m = _nearest(mentions, tokens, first, last)
        if m is not None and not m.exempt and m.candidates and all(
                d.get("effect") == "opposes" for d in m.candidates):
            reasons.append(f"effect:{m.label}:{word}")
    return reasons


def _check_room_scope(text: str, mentions: list[Mention]) -> list[str]:
    reasons: list[str] = []
    for m in mentions:
        if m.exempt or not m.candidates or not all(d.get("scope") == "room" for d in m.candidates):
            continue
        head = text[max(0, m.start - 40):m.start]
        hit = _POSSESSIVE_BEFORE.search(head)
        if hit and "room" not in hit.group(0).lower() or ROOM_SOURCE.match(text[m.end:m.end + 40]):
            reasons.append(f"room_scope:{m.label}")
    if ROOM_ACT_RE.search(text):
        reasons.append("room_scope:act")
    return reasons


def _check_mask(text: str, sheet: Mapping[str, Any], protected: list[tuple[int, int]]) -> list[str]:
    if sheet.get("mask_override"):
        return []
    scan = text
    for span in protected:
        scan = _blank(scan, *span)
    return [f"mask_term:{m.group(0).lower()}" for m in MASK_RE.finditer(scan)]


def _check_identity(text: str, sheet: Mapping[str, Any]) -> list[str]:
    reasons = [f"banned:{category}:{term}" for category, term in banned_hits(text)]
    own = person_number(sheet)
    reasons += [f"person:{m.group(1)}" for m in _PERSON_REF.finditer(text) if int(m.group(1)) != own]
    reasons += [f"person:{m.group(1)}" for m in _PERSON_NO_REF.finditer(text) if int(m.group(1)) != own]
    reasons += ["person:ref" for _ in _OTHER_REF.finditer(text)]
    return reasons


def _check_unmeasured(text: str, protected: list[tuple[int, int]]) -> list[str]:
    scan = text
    for span in protected:
        scan = _blank(scan, *span)
    return [f"unmeasured:{term}" for term in unmeasured_hits(scan)]


_CLOCK_DISPLAY = re.compile(r"\((\d{1,2}):(\d{2})\)")


def _check_time_words(sheet: Mapping[str, Any], tokens: _Tokens) -> list[str]:
    """"morning"/"evening"/... must fit the sheet's clock time (when the sheet shows one)."""
    clock = next((m for _, d in iter_drivers(sheet) if (m := _CLOCK_DISPLAY.search(str(d.get("display") or "")))),
                 None)
    if clock is None:
        return []
    hour = int(clock.group(1)) + int(clock.group(2)) / 60
    reasons = []
    for i, word in enumerate(tokens.words):
        if word in TIME_WORDS and not tokens.protected[i]:
            start, end = TIME_WORDS[word]
            inside = start <= hour < end if start < end else (hour >= start or hour < end)
            if not inside:
                reasons.append(f"time_of_day:{word}")
    return reasons


def validate(text: str, sheet: Mapping[str, Any] | BaseModel) -> ValidationResult:
    """Check an explanation against its (current) fact sheet; ``ok`` only if no rule fails."""
    sheet = as_dict(sheet)
    cleaned = clean_text(text)
    if not cleaned:
        return ValidationResult(False, ["empty"], "")
    reasons, words, sentences = _check_length(cleaned)
    mentions, protected = find_mentions(cleaned, sheet)
    tokens = _tokenize(cleaned, protected)
    mentions = _bind_tokens(mentions, tokens)
    for m in mentions:
        _narrow_by_numbers(m, tokens)
    label_display = str(sheet.get("label_display") or "").lower()
    label_spans = _find_all(cleaned.lower(), label_display) if label_display else []
    reasons += _check_numbers(cleaned, sheet, mentions, tokens, protected)
    reasons += _check_states(cleaned, sheet, tokens, label_spans)
    coverage, mentioned = _check_coverage(mentions, sheet, cleaned, tokens)
    reasons += coverage
    reasons += _check_qualifiers(mentions, tokens, _face_contrast_sentences(cleaned, sheet, tokens))
    reasons += _check_time_words(sheet, tokens)
    reasons += _check_room_scope(cleaned, mentions)
    reasons += _check_mask(cleaned, sheet, protected)
    reasons += _check_unmeasured(cleaned, protected)
    reasons += _check_identity(cleaned, sheet)
    reasons = list(dict.fromkeys(reasons))
    return ValidationResult(
        ok=not reasons,
        reasons=reasons,
        text=cleaned,
        word_count=words,
        sentence_count=sentences,
        mentioned_drivers=mentioned,
        mentioned_concepts=sorted({m.concept for m in mentions if m.concept}),
    )
