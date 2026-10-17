"""Regex lexicon the narrator uses to read its own (or the LLM's) prose.

Each concept maps to case-insensitive, word-bounded patterns that detect when a sentence talks
about a signal. Patterns are tuned so everyday words do not false-trigger: bare "still",
"late", "light" in "in light of", and "rest" in "the rest of" never count. The same lexicon
also lists the words that name each state, the words that imply a facial mask, and the terms a
privacy-safe explanation must never use (gender, demographics, appearance, diagnoses).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

CONCEPTS: dict[str, list[str]] = {
    "noise": [
        r"\bnois\w*",
        r"\bloud(?:er|est|ness|ly)?\b",
        r"\bquiet(?:er|est|ness)?\b",
        r"(?<![a-z])db\b",
        r"\bdecibels?\b",
        r"\bvolume\b",
        r"\bsound levels?\b",
        r"\bacoustic\w*",
    ],
    "light": [
        r"(?<!in )\blight(?:ing|s)?\b"
        r"(?!\s+(?:of|movement|motion|activity|fidget\w*|chatter|conversation|laughter|talk\w*|speech|touch))",
        r"\blux\b",
        r"\bdim(?:ly|mer|med|ness)?\b",
        r"\bbright\w*",
        r"\bdark\w*",
        r"\billuminat\w*",
        r"\bluminance\b",
    ],
    "break": [
        r"\bbreaks?\b",
        r"\brest(?:ed|ing)?\b(?!\s+of\b)",
        r"\bsince (?:the |their |a )?last\b",
        r"\bpause[sd]?\b",
        r"\bbreather\b",
        r"\btime off\b",
    ],
    "time_of_day": [
        r"\btime of (?:the )?day\b",
        r"\bhour of (?:the )?day\b",
        r"\bo'?clock\b",
        r"\b(?:morning|afternoon|evening|night|nighttime|midday|noon|midnight|overnight|tonight)\b",
        r"\blate (?:in the )?(?:day|afternoon|evening|night|hour|shift)\b",
        r"\bearly (?:in the )?(?:day|morning|shift)\b",
        r"\bend of (?:the )?(?:day|workday|shift)\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d{1,2}\s*[ap]\.?m\b",
    ],
    "fist": [r"\bfists?\b", r"\bclench\w*"],
    "slump": [r"\bslump\w*", r"\bposture\w*", r"\bslouch\w*", r"\bhunch(?:ed|ing)\b"],
    "head_in_hands": [
        r"\bheads?[- ]in[- ](?:(?:the|their|both)[- ])?hands?\b",
        r"\bheads?(?:\s+[a-z]+){0,2}\s+(?:in|on|into|against)\s+(?:(?:their|the|both)\s+)?hands?\b",
        r"\bface in (?:their |the |both )?hands\b",
        r"\bhands? (?:on|over|covering|supporting) (?:their |the )?(?:head|face)\b",
        r"\bhead[- ]down\b",
    ],
    "withdrawal": [
        r"\bwithdr[ae]w\w*",
        r"\bdisengag\w*",
        r"\bisolat\w*",
        r"\bretreat\w*",
        r"\bwent (?:quiet|silent)\b",
    ],
    "motion": [
        r"\bmovements?\b",
        r"\bmotion\w*",
        r"\bfidget\w*",
        r"\brestless\w*",
        r"\bmoving\b",
        r"\bstillness\b",
        r"\b(?:sat|sitting|kept|keeping|remained|remaining|stayed|staying|held|holding)"
        r" (?:very |completely |mostly )?still\b",
        r"\bpacing\b",
    ],
    "laughter": [r"\blaugh\w*", r"\bgiggl\w*", r"\bchuckl\w*"],
    "compliment": [r"\bcompliment\w*", r"\bprais\w*", r"\bkind words\b"],
    "speech": [
        r"\bspeech\b",
        r"\btalk(?:s|ed|ing|ative)?\b",
        r"\bconversation\w*",
        r"\bspeak(?:s|ing)?\b",
        r"\bchat(?:s|ted|ting|ter)?\b",
        r"\bverbal\w*",
        r"\bvoices?\b",
        r"\bdiscussion\w*",
    ],
    "face_valence": [
        r"\bfacial\b",
        r"\bface valence\b",
        r"\bexpressions?\b",
        r"\bsmil\w*",
        r"\bfrown\w*",
        r"\bgrin(?:s|ned|ning)?\b",
        r"\bscowl\w*",
        r"\bface[- ]only\b",
    ],
    "face_visible": [
        r"\bface\b(?:\s+(?:was|is|were|remained|stayed|being|became|went))?"
        r"(?:\s+(?:mostly|largely|partly|partially|often|frequently|entirely|completely|fully))?"
        r"\s+(?:hidden|obscured|occluded|covered|not visible|invisible|out of (?:camera )?view"
        r"|turned away|unavailable|off[- ]camera|not detected)\b",
        r"\b(?:hidden|obscured|occluded|covered|invisible) face\b",
        r"\bocclu\w*",
        r"\bobscur\w*",
        r"\bface[- ]visib\w*",
        r"\bvisibility\b",
        r"\bout of (?:camera )?view\b",
        r"\boff[- ]camera\b",
        r"\bno face\b",
    ],
}

STATE_TERMS: dict[str, list[str]] = {
    "Focused": [r"\bfocus\w*"],
    "Frustrated_Masked": [r"\bfrustrat\w*"],
    "Exhausted": [r"\bexhaust\w*"],
    "Collaborative_Happy": [r"\bcollaborat\w*", r"\bhapp(?:y|ier|iest|ily|iness)\b"],
}

MASK_TERMS: list[str] = [
    r"\b(?:un)?mask\w*",
    r"\bsmil\w*",
    r"\bdespite\b",
    r"\bsuperficial\w*",
    r"\bforced\b",
    r"\bconceal\w*",
    r"\bhiding\b",
    r"\bfa[cç]ade\b",
    r"\bpolite\w*",
]

BANNED_TERMS: dict[str, list[str]] = {
    "gendered": [
        r"\b(?:he|she|him|her|his|hers|himself|herself)\b",
        r"\b(?:man|men|woman|women|guys?|girls?|boys?|lad(?:y|ies)|gentlem[ae]n|males?|females?)\b",
        r"\b(?:sir|madam|ma'am|dude|gals?)\b",
        r"\b(?:husband|wife|boyfriend|girlfriend|mother|father|son|daughter)s?\b",
        r"\b(?:masculin|feminin|gender)\w*",
        r"(?-i:\b(?:Mr|Mrs|Ms|MR|MRS|MS)\b)",
    ],
    "demographic": [
        r"\bag(?:e|ed|es|ing|eing)\b",
        r"\bold(?:er|est)?\b",
        r"\byoung(?:er|est)?\b",
        r"\byouth\w*",
        r"\belderly\b",
        r"\bteen\w*",
        r"\b(?:millennials?|boomers?|gen[- ]?z)\b",
        r"\brac(?:e|es|ial|ially|ism|ist)\b",
        r"\bethnic\w*",
        r"\bskin\w*",
        r"\bcomplexion\w*",
        r"\bnationalit\w*",
        r"\breligio\w*",
        r"\b(?:asian|african|caucasian|hispanic|latin[aox]s?|latine|arab|indian|european"
        r"|middle[- ]eastern|native american|indigenous|aboriginal|pacific islander"
        r"|jewish|muslim|christian|hindu|buddhist|sikh)\b",
    ],
    "appearance": [
        r"\bhair\w*",
        r"\bbeard\w*",
        r"\bmo?ustache\w*",
        r"\b(?:eye)?glasses\b",
        r"\bspectacles\b",
        r"\b(?:un)?attractive\w*",
        r"\b(?:handsome|beautiful|ugly|good[- ]looking)\b",
        r"\b(?:over|under)weight\b",
        r"\bbody[- ]?weight\b",
        r"\b(?:their|his|her) weight\b",
        r"\bobes(?:e|ity)\b",
        r"\b(?:skinny|chubby|heavyset|plump|petite)\b",
        r"\btall(?:er|est)?\b",
        r"\bshort(?:er|est)? (?:in stature|person|individual|stature|build|frame)\b",
        r"\b(?:blue|green|brown|hazel|grey|gray)[- ]eyed\b",
        r"\beye colou?r\b",
        r"\b(?:blue|green|brown|hazel|grey|gray) eyes\b",
        r"\bcloth(?:es|ing)\b",
        r"\bwearing\b",
        r"\boutfits?\b",
        r"\b(?:shirt|t-shirt|hoodie|jacket|sweater|dress)(?:es|s)?\b",
        r"\btattoo\w*",
        r"\bpiercings?\b",
        r"\bbald\w*",
        r"\bfreckl\w*",
        r"\bwrinkl\w*",
    ],
    "diagnostic": [
        r"\bdepress\w*",
        r"\banxi(?:ety|eties|ous|ously)\b",
        r"\badhd\b",
        r"\bautis\w*",
        r"\bbipolar\b",
        r"\bptsd\b",
        r"\bocd\b",
        r"\bdisorder\w*",
        r"\bdiagnos\w*",
        r"\bclinical\w*",
        r"\bsyndrome\w*",
        r"\bmental(?:ly)? ill(?:ness)?\b",
        r"\bpanic attacks?\b",
        r"\bburn(?:ed|t)?[- ]?out\b",
        r"\bschizo\w*",
        r"\bpsychiatr\w*",
        r"\binsomnia\w*",
        r"\bneurodivergen\w*",
        r"\bpatholog\w*",
        r"\btrauma\w*",
        r"\bdisabled\b",
        r"\bdisabilit\w*",
        r"\bdiseases?\b",
        r"\billness\w*",
    ],
}


def _compile(patterns: Iterable[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_CONCEPT_RES = {name: _compile(patterns) for name, patterns in CONCEPTS.items()}
_MASK_RE = _compile(MASK_TERMS)
_BANNED_RES = {category: _compile(patterns) for category, patterns in BANNED_TERMS.items()}


def text_concepts(text: str) -> set[str]:
    """Concepts the text talks about."""
    return {name for name, regex in _CONCEPT_RES.items() if regex.search(text)}


def driver_concepts(driver: Mapping[str, Any]) -> set[str]:
    """Concepts a fact-sheet driver stands for, read from its display, feature key and unit."""
    feature = str(driver.get("feature") or "").replace("_", " ")
    parts = (str(driver.get("display") or ""), feature, str(driver.get("unit") or ""))
    return text_concepts(" ".join(parts))


@lru_cache(maxsize=64)
def state_regex(state: str, label_display: str | None = None) -> re.Pattern[str]:
    """Pattern matching any way of naming ``state``: lexicon stems, its key, its display string."""
    patterns = list(STATE_TERMS.get(state, ()))
    patterns.append(re.escape(state))
    if label_display:
        patterns.append(re.escape(label_display))
    return _compile(patterns)


def names_state(text: str, state: str, label_display: str | None = None) -> bool:
    return bool(state_regex(state, label_display).search(text))


def mask_hits(text: str) -> list[str]:
    """Mask-implying words in the text, lower-cased, in order of appearance."""
    return [m.group(0).lower() for m in _MASK_RE.finditer(text)]


def banned_hits(text: str) -> list[tuple[str, str]]:
    """(category, matched term) for every banned term in the text."""
    hits: list[tuple[str, str]] = []
    for category, regex in _BANNED_RES.items():
        hits.extend((category, m.group(0).lower()) for m in regex.finditer(text))
    return hits


def remove_verbatim(text: str, phrase: str | None) -> str:
    """Blank out case-insensitive occurrences of ``phrase`` (used to exempt label_display)."""
    if not phrase:
        return text
    return re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
