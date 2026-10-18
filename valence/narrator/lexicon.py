"""Word lists the validator uses to read narrator prose. All matching is case-insensitive.

* ``CONCEPTS``: word-bounded patterns for each signal a driver can stand for. A driver's concept
  comes from its feature channel (``FEATURE_CONCEPTS``, real feature names from
  ``valence.features.spec``), falling back to its display text for unknown features. Everyday
  words do not trigger: "in light of", "the rest of", bare "still" or "late".
* State naming: ``STATE_LEXICON`` stems from ``valence.contracts`` plus the display phrases
  ("Visible Frustration", "Masked Stress", "Upbeat", ...). Phrases are matched longest first and
  each match consumes its span, so "visibly frustrat" (Tense) is claimed before "frustrat"
  (Frustrated_Masked) can see it.
* ``QUALIFIERS``: level words and the driver levels each one is compatible with;
  ``CONTRAST_BEFORE`` / ``CONTRAST_AFTER`` / ``SUPPORT_WORDS``: words that state a driver's
  effect (contrast only fits ``effect == "opposes"``).
* ``DIRECTION_WORDS``: up/down words ("above", "falling", "darker") that must agree with the
  direction a delta driver's display states.
* ``NEGATORS``: words that negate the state or level word right after them.
* ``FACE_CONTEXT`` / ``PREDICTION_CUES``: what makes a face-only state a face contrast (allowed
  with ``mask_override``) rather than the prediction.
* ``TIME_WORDS``: clock ranges for "morning", "evening", ... (checked against the sheet's clock).
* ``MASK_TERMS`` (only with ``mask_override``), ``BANNED_TERMS`` (never), ``UNMEASURED_TERMS``
  (signals Valence never measures, never), possessives and verbs that would attribute a room-scope
  signal to the person.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from valence.contracts import ALL_STATES, CORE_STATES, STATE_DISPLAY, STATE_LEXICON, display_label

CONCEPTS: dict[str, list[str]] = {
    "noise": [
        r"\bnois\w*", r"\bloud(?:er|est|ness|ly)?\b", r"\bquiet(?:er|est|ness|ly)?\b", r"(?<![a-z])db\b",
        r"\bdecibels?\b", r"\bvolume\b", r"\bsound levels?\b", r"\bacoustic\w*",
    ],
    "light": [
        r"(?<!in )\blight(?:ing|s)?\b"
        r"(?!\s+(?:of|movement|motion|activity|fidget\w*|chatter|conversation|laughter|talk\w*|speech|touch))",
        r"\blux\b", r"\bdim(?:ly|mer|med|ness)?\b", r"\bbright\w*", r"\bdark\w*", r"\billuminat\w*",
        r"\bluminance\b",
    ],
    "break": [
        r"\bbreaks?\b", r"\brest(?:ed|ing)?\b(?!\s+of\b)", r"\bsince (?:the |their |a )?last\b", r"\bpause[sd]?\b",
        r"\bbreather\b", r"\btime off\b",
    ],
    "time_of_day": [
        r"\btime of (?:the )?day\b", r"\bhour of (?:the )?day\b", r"\bo'?clock\b",
        r"\b(?:morning|afternoon|evening|night|nighttime|midday|noon|midnight|overnight|tonight)\b",
        r"\b\d{1,2}:\d{2}\b", r"\b\d{1,2}\s*[ap]\.?m\b",
    ],
    "fist": [r"\bfists?\b", r"\bclench\w*"],
    "slump": [r"\bslump\w*", r"\bposture\w*", r"\bslouch\w*", r"\bhunch(?:ed|ing)\b"],
    "head_in_hands": [
        r"\bheads?[- ]in[- ](?:(?:the|their|both)[- ])?hands?\b",
        r"\bface in (?:their |the |both )?hands\b",
        r"\bhands? (?:on|over|covering|supporting) (?:their |the )?(?:head|face)\b",
        r"\bhead[- ]down\b",
    ],
    "withdrawal": [r"\bwithdr[ae]w\w*", r"\bdisengag\w*", r"\bisolat\w*", r"\bretreat\w*"],
    "motion": [
        r"\bmovements?\b", r"\bmotion\w*", r"\bfidget\w*", r"\brestless\w*", r"\bmoving\b", r"\bstillness\b",
        r"\b(?:nearly|almost|sat|sitting|kept|keeping|remained|remaining|stayed|staying|held|holding)"
        r" (?:very |completely |mostly )?still\b",
        r"\bpacing\b",
    ],
    "laughter": [r"\blaugh\w*", r"\bgiggl\w*", r"\bchuckl\w*"],
    "compliment": [r"\bcompliment\w*", r"\bprais\w*", r"\bkind words\b"],
    "people": [r"\bpeople\b", r"\boccupan\w*", r"\bpeople present\b"],
    "speech": [r"\bspeech\b", r"\bconversation\w*", r"\bchat(?:s|ted|ting|ter)?\b", r"\bvoices?\b"],
    "face_valence": [
        r"\bfacial\b", r"\bvalence\b", r"\bexpressions?\b", r"\bsmil\w*", r"\bfrown\w*", r"\bgrin(?:s|ned|ning)?\b",
        r"\bscowl\w*",
    ],
    "face_visible": [
        r"\bface\b(?:\s+(?:was|is|were|remained|stayed|being|became|went))?"
        r"(?:\s+(?:mostly|largely|partly|partially|often|frequently|entirely|completely|fully))?"
        r"\s+(?:hidden|obscured|occluded|covered|not visible|invisible|out of (?:camera )?view|not readable"
        r"|turned away|unavailable|off[- ]camera|not detected|visible)\b",
        r"\b(?:hidden|obscured|occluded|covered|invisible|unreadable) face\b",
        r"\bocclu\w*", r"\bobscur\w*", r"\bface[- ]visib\w*", r"\bvisibility\b", r"\bout of (?:camera )?view\b",
        r"\boff[- ]camera\b", r"\bno face\b",
    ],
}

FEATURE_CONCEPTS: dict[str, str] = {
    "db": "noise", "lux": "light", "motion": "motion", "slump": "slump", "hih": "head_in_hands", "fist": "fist",
    "laugh": "laughter", "compl": "compliment", "withdraw": "withdrawal", "n_people": "people",
    "hours_since_break": "break", "tod_sin": "time_of_day", "tod_cos": "time_of_day",
}
FACE_CONCEPTS = ("face_valence", "face_visible")

# Level words -> driver levels they agree with. "high" also fits "elevated"; "normal"/"moderate"
# also fit "elevated" (describe() itself calls elevated light "normal light").
_UP = frozenset({"high", "elevated"})
_TOP = frozenset({"high"})
_MID = frozenset({"normal", "elevated"})
_DOWN = frozenset({"low"})
_CALM = frozenset({"low", "normal"})
# "recent" is not a level word ("recent noise" is the core's own wording for an EWM driver); it only
# matters for the time-since features below. "steady" describes variation, not level.
QUALIFIERS: dict[str, frozenset[str]] = {
    **{w: _TOP for w in ("very high", "extremely high", "highest", "extreme", "extremely", "intense", "severe",
                         "very loud", "very bright", "peak level", "late", "very restless")},
    **{w: _UP for w in ("high", "higher", "elevated", "raised", "increased", "heightened", "above-average",
                        "loud", "louder", "noisy", "bright", "brighter", "heavy", "frequent", "many", "lots",
                        "numerous", "repeated", "several", "restless", "long", "longer", "extended", "prolonged",
                        "agitated", "busy")},
    **{w: _MID for w in ("normal", "typical", "ordinary", "moderate", "moderately", "neutral")},
    **{w: _CALM for w in ("quiet", "calm", "short", "brief")},
    **{w: _DOWN for w in ("low", "lower", "lowest", "minimal", "little", "few", "reduced", "scarce", "rare",
                          "dim", "dimmer", "dark", "darker", "sparse", "negligible", "barely")},
}
# "*_since_s" features (time since the last fist/laugh/compliment) are "high" when the event was
# recent, so duration words read the other way round for them.
SINCE_QUALIFIERS: dict[str, frozenset[str]] = {
    **{w: _DOWN for w in ("long", "longer", "extended", "prolonged")},
    **{w: _UP for w in ("short", "brief", "recent")},
}
# Words after which a level word describes the prediction, not a signal ("high confidence").
NOT_A_SIGNAL = frozenset({"confidence", "confident", "probability", "likelihood", "certainty", "chance", "odds"})
# "late" in "late afternoon" places the clock time; it does not claim a late (night) level.
NOT_A_LEVEL_AFTER = {"late": frozenset({"morning", "afternoon", "evening"})}

# Negators (token sequences) that flip the state or level word up to two tokens after them. "no"
# only flips time-since words ("no recent laughter").
NEGATORS: tuple[str, ...] = ("not", "never", "isn't", "wasn't", "aren't", "weren't", "isnt", "wasnt", "hardly",
                             "no longer", "rather than", "instead of", "nor")
SINCE_NEGATORS: tuple[str, ...] = ("no",)

# Direction words: they must agree with the direction a delta driver's display states
# ("noise 1 dB above its 15-minute median", "light 40 lux darker than ...", "noise rising ...").
DIRECTION_WORDS: dict[str, str] = {
    **{w: "up" for w in ("above", "rising", "rose", "risen", "rises", "climbing", "climbed", "increasing",
                         "brighter", "higher", "louder")},
    **{w: "down" for w in ("below", "falling", "fell", "fallen", "falls", "dropping", "dropped", "declining",
                           "decreasing", "darker", "dimmer", "lower", "quieter")},
}

# A face-only state is a face contrast (allowed with mask_override) only when its sentence ties it
# to the face; after one of these cues, with no face word in the clause, it reads as the prediction.
FACE_CONTEXT: tuple[str, ...] = ("face", "faces", "facial", "face-only", "expression", "expressions", "valence",
                                 "smile", "smiling", "smiled", "smiles", "surface", "outward", "outwardly", "looked",
                                 "looks", "looking", "appeared", "appears", "mask", "masked")
PREDICTION_CUES: tuple[str, ...] = ("predicted", "prediction", "predicts", "classified", "classifies", "really",
                                    "actually", "truly", "in fact", "most likely", "reads as", "read as", "is", "are",
                                    "was", "were", "labelled", "labeled", "rated")

# Clock words and the hours (inclusive start, exclusive end, wrapping past midnight) they fit,
# with a little slack for rounding. Checked against the sheet's time-of-day display.
TIME_WORDS: dict[str, tuple[float, float]] = {
    "morning": (4.5, 12.5), "midday": (11.0, 13.5), "noon": (11.5, 12.5), "afternoon": (11.75, 18.5),
    "evening": (16.5, 23.5), "night": (19.5, 6.5), "nighttime": (19.5, 6.5), "tonight": (17.0, 6.0),
    "overnight": (21.0, 7.0), "midnight": (23.0, 1.0),
}

# Contrast markers. BEFORE: the contrasted signal follows in the same clause ("despite X",
# "although X was ..."). AFTER: the signal precedes ("X pulled the other way", "X argued against").
CONTRAST_BEFORE: tuple[str, ...] = ("despite", "although", "though", "even though", "in spite of",
                                    "notwithstanding", "whereas", "even with")
CONTRAST_AFTER: tuple[str, ...] = ("the other way", "against", "opposed", "opposing", "opposes", "countered",
                                   "counter", "offset", "offsetting", "weighed against", "argued against")
# Words that frame a driver as a cause of the label (only fit ``effect == "supports"``). The noun
# "signals" is deliberately absent ("Signals: ..." heads a list of drivers).
SUPPORT_WORDS: tuple[str, ...] = (
    "supports", "supported", "supporting", "reinforced", "reinforcing", "reinforces", "drove", "driven", "driving",
    "drives", "pointed to", "points to", "pointing to", "pointed toward", "points toward", "pointed towards",
    "points towards", "rests", "rested", "resting on", "based on", "due to", "because of", "caused", "causes",
    "causing", "led to", "leads to", "leading to", "explains", "explained", "explaining", "indicates", "indicated",
    "indicating", "suggests", "suggested", "suggesting", "signaled", "signalled", "signaling", "signalling",
    "reflects", "reflected", "reflecting", "consistent with", "contributed", "contributes", "contributing",
    "pushed", "pushes", "pushing", "fuelled", "fueled", "behind the call", "behind the label",
)

MASK_TERMS: list[str] = [
    r"\b(?:un)?mask\w*", r"\bsmil\w*", r"\bsuperficial\w*", r"\bforced\b", r"\bconceal\w*", r"\bhiding\b",
    r"\bfa[cç]ade\b", r"\bput[- ]on\b", r"\bfake\w*", r"\bpositive[- ]looking\b", r"\bpretend\w*",
    r"\b(?:brave|poker|happy|cheerful|straight|composed)[- ](?:face|front)\b", r"\bcheerful (?:expression|look)\b",
    r"\bexteriors?\b", r"\bveneer\b", r"\bon the surface\b", r"\bbeneath\b", r"\bunderneath\b",
    r"\bbehind (?:a|an|the|their)\b", r"\bcover(?:s|ed|ing)?[- ]up\b", r"\bcovering (?:up|for)\b",
    r"\boutward(?:ly)?\b", r"\bputting on\b", r"\bkeeping up appearances\b",
]

POSSESSIVES = r"(?:their|theirs|own|person\s*#\s*\d+['’]s|person['’]s)"
_WHO = r"(?:they|them|person\s*#\s*\d+)"
ROOM_ACTS = [
    rf"\b{_WHO}\s+(?:\w+\s+)?(?:laugh\w*|giggl\w*|chuckl\w*|complimented|prais\w*)\b",
    rf"\b{_WHO}\s+(?:were|was|are|is|been)\s+(?:very\s+)?(?:loud|noisy)\b",
    rf"\b{_WHO}\s+(?:(?:was|were|is|are|kept|keeps|been)\s+)?(?:made|making|makes|make|caused|causing|causes"
    r"|generated|generating|generates|produced|producing|produces|created|creating|creates|emitted|emitting)"
    r"\s+(?:(?:the|a|an|some|much|more|most|lots of|a lot of|all the)\s+)?(?:\w+\s+)?(?:noise|sound|din|racket)\b",
]
# Right after a room-scope mention: "noise from Person #1", "light by them".
ROOM_SOURCE = re.compile(rf"^\W*(?:[\w-]+\W+){{0,2}}?(?:from|by)\s+{_WHO}\b", re.IGNORECASE)

BANNED_TERMS: dict[str, list[str]] = {
    "gendered": [
        r"\b(?:he|she|him|her|his|hers|himself|herself)\b",
        r"\b(?:man|men|woman|women|guys?|girls?|boys?|lad(?:s|y|ies|dies)?|gentlem[ae]n|males?|females?)\b",
        r"\b(?:sir|madam|ma'am|dude|gals?|blokes?|chaps?|fellas?|lasse?s?)\b",
        r"\b(?:husband|wife|boyfriend|girlfriend|mother|father|son|daughter)s?\b",
        r"\b(?:masculin|feminin|gender)\w*",
        r"(?-i:\b(?:Mr|Mrs|Ms|MR|MRS|MS|Mx|Miss)\b)",
    ],
    "demographic": [
        r"\bag(?:e|ed|es|eing|ing)\b",
        r"(?<!minutes )(?<!minute )(?<!hours )(?<!hour )(?<!seconds )(?<!second )(?<!days )\bold(?:er|est)?\b",
        r"\byoung(?:er|est)?\b", r"\byouth\w*", r"\belderly\b", r"\bkids?\b", r"\bkiddos?\b",
        r"\bchild(?:ren|ish)?\b", r"\btoddlers?\b", r"\bseniors?\b", r"\bretirees?\b", r"\badolescen\w*",
        r"\bjuveniles?\b", r"\bpensioners?\b",
        r"\bteen\w*", r"\b(?:millennials?|boomers?|gen[- ]?z)\b", r"\brac(?:e|es|ial|ially|ism|ist)\b",
        r"\bethnic\w*", r"\bskin\w*", r"\bcomplexion\w*", r"\bnationalit\w*", r"\breligio\w*",
        r"\b(?:asian|african|caucasian|hispanic|latin[aox]s?|latine|arab|indian|european"
        r"|middle[- ]eastern|native american|indigenous|aboriginal|pacific islander"
        r"|jewish|muslim|christian|hindu|buddhist|sikh)\b",
    ],
    "appearance": [
        r"\bhair\w*", r"\bbeard\w*", r"\bmo?ustache\w*", r"\b(?:eye)?glasses\b", r"\bspectacles\b",
        r"\b(?:un)?attractive\w*", r"\b(?:handsome|beautiful|pretty|ugly|good[- ]looking)\b",
        r"\b(?:over|under)weight\b", r"\bbody[- ]?weight\b", r"\b(?:their|his|her) weight\b", r"\bobes(?:e|ity)\b",
        r"\b(?:skinny|chubby|heavyset|plump|petite)\b", r"\btall(?:er|est)?\b",
        r"\bshort(?:er|est)? (?:in stature|person|individual|stature|build|frame)\b",
        r"\b(?:blue|green|brown|hazel|grey|gray)[- ]eyed\b", r"\beye colou?r\b",
        r"\b(?:blue|green|brown|hazel|grey|gray) eyes\b", r"\bcloth(?:es|ing)\b",
        r"\bwearing\b(?!\s+(?:on|down|off|thin|away)\b)", r"\boutfits?\b", r"\bblond(?:e|es)?\b", r"\bbrunettes?\b",
        r"\bredheads?\b", r"\bred-haired\b", r"\bmake-?up\b", r"\bhijabs?\b", r"\bturbans?\b", r"\bheadscar(?:f|ves)\b",
        r"\bveil(?:ed)?\b", r"\bjewel(?:le)?ry\b",
        r"\b(?:shirt|t-shirt|hoodie|jacket|sweater|dress)(?:es|s)?\b", r"\btattoo\w*", r"\bpiercings?\b",
        r"\bbald\w*", r"\bfreckl\w*", r"\bwrinkl\w*",
    ],
    "diagnostic": [
        r"\bdepress\w*", r"\banxi(?:ety|eties|ous|ously)\b", r"\badhd\b", r"\bautis\w*", r"\bbipolar\b",
        r"\bptsd\b", r"\bocd\b", r"\bdisorder\w*", r"\bdiagnos\w*", r"\bclinical\w*", r"\bsyndrome\w*",
        r"\bmental(?:ly)? ill(?:ness)?\b", r"\bpanic attacks?\b", r"\bburn(?:ed|t)?[- ]?out\b", r"\bschizo\w*",
        r"\bpsychiatr\w*", r"\binsomnia\w*", r"\bneurodivergen\w*", r"\bpatholog\w*", r"\btrauma\w*",
        r"\bdisabled\b", r"\bdisabilit\w*", r"\bdiseases?\b", r"\billness\w*", r"\bsymptom\w*",
        r"\bmanic\b", r"\bmania\b", r"\bhyperactiv\w*", r"\bneurotic\w*", r"\bneuros[ie]s\b", r"\bpsychotic\b",
        r"\bpsychos[ie]s\b", r"\bmigraines?\b", r"\bconcuss\w*", r"\bmedicat\w*", r"\bmedicine\b",
        r"\btherap\w*", r"\bsick\w*", r"\bpathologic\w*", r"\bhypochondr\w*", r"\bparanoi\w*",
    ],
}

# Signals Valence never measures: naming one invents evidence.
UNMEASURED_TERMS: list[str] = [
    r"\bheart[- ]?rates?\b", r"\bheart ?beats?\b", r"\bpulse\b", r"\bbreath(?:e|es|ed|ing|s)?\b", r"\bsweat\w*",
    r"\bperspir\w*", r"\btemperature\b", r"\bblood pressure\b", r"\btyping\b", r"\btyped\b", r"\bkeystrokes?\b",
    r"\bkeyboards?\b", r"\bmouse\b", r"\bscreens?\b", r"\bgaze\w*", r"\beye[- ]contact\b", r"\bstar(?:e|ed|es|ing)\b",
    r"\bblink\w*", r"\bsigh(?:s|ed|ing)?\b", r"\btone of (?:voice|speech)\b", r"\bvocal\b", r"\bpitch\b",
    r"\btears\b", r"\bcry(?:ing)?\b", r"\bcried\b", r"\bdeadlines?\b", r"\bworkload\b", r"\bemails?\b",
]


def _compile(patterns: Iterable[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


CONCEPT_RES: dict[str, re.Pattern[str]] = {name: _compile(patterns) for name, patterns in CONCEPTS.items()}
MASK_RE = _compile(MASK_TERMS)
ROOM_ACT_RE = _compile(ROOM_ACTS)
BANNED_RES: dict[str, re.Pattern[str]] = {c: _compile(p) for c, p in BANNED_TERMS.items()}
UNMEASURED_RE = _compile(UNMEASURED_TERMS)
COLLABORATIVE_RE = re.compile(r"collaborat", re.IGNORECASE)


def text_concepts(text: str) -> set[str]:
    """Concepts the text talks about."""
    return {name for name, regex in CONCEPT_RES.items() if regex.search(text)}


def driver_concept(driver: Mapping[str, Any]) -> str | None:
    """The concept a driver stands for: by feature channel, else read from its display text."""
    feature = str(driver.get("feature") or "")
    if feature.startswith("face_"):
        return "face_visible" if feature.startswith("face_visible") else "face_valence"
    concept = FEATURE_CONCEPTS.get(feature) or FEATURE_CONCEPTS.get(feature.split("_", 1)[0])
    if concept:
        return concept
    found = text_concepts(f"{driver.get('display') or ''} {feature.replace('_', ' ')}")
    return min(found) if found else None


def mask_hits(text: str) -> list[str]:
    return [m.group(0).lower() for m in MASK_RE.finditer(text)]


def banned_hits(text: str) -> list[tuple[str, str]]:
    """(category, matched term) for every banned term in the text."""
    return [(c, m.group(0).lower()) for c, regex in BANNED_RES.items() for m in regex.finditer(text)]


def unmeasured_hits(text: str) -> list[str]:
    """Signals the text names that Valence never measures ("heart rate", "typing")."""
    return [m.group(0).lower() for m in UNMEASURED_RE.finditer(text)]


# Everyday words that contain a state stem but do not name a state.
STATE_EXEMPT = re.compile(
    r"\boverwhelmingly\b"
    r"|\b(?:main|primary|key|chief|its|model['’]s|explanation['’]s|audit['’]s)\s+focus\b"
    r"|\b(?:model|it|analysis|audit|explanation|note)\s+(?:was\s+|is\s+)?focus(?:ed|es|ing)?\s+on\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=16)
def state_phrases(states: tuple[str, ...] = ALL_STATES) -> tuple[tuple[str, str], ...]:
    """(lower-cased phrase, state) for every way of naming a state, longest phrase first.

    Phrases: contract lexicon stems, the display name, its " / " parts and display_label's short
    forms ("Frustrated", "Upbeat"). Longest-first order makes "visibly frustrat" and "visible
    frustration" (Tense) win over "frustrat" (Frustrated_Masked).
    """
    phrases: dict[str, str] = {}

    def add(phrase: str, state: str) -> None:
        phrases.setdefault(phrase.lower().strip(), state)

    for state in ALL_STATES:
        for stem in STATE_LEXICON[state]:
            add(stem, state)
        add(STATE_DISPLAY[state], state)
        for part in STATE_DISPLAY[state].split(" / "):
            add(part, state)
        add(state.replace("_", " "), state)
        for n_people in (1, 2):
            for override in (False, True):
                add(display_label(state, mask_override=override, n_people=n_people, states=CORE_STATES), state)
    return tuple(sorted(phrases.items(), key=lambda item: (-len(item[0]), item[0])))


def state_mentions(text: str) -> list[tuple[int, int, str]]:
    """(start, end, state) for every state named in the text, claimed longest phrase first."""
    lower = text.lower()
    taken = [False] * len(lower)
    for match in STATE_EXEMPT.finditer(lower):
        for i in range(*match.span()):
            taken[i] = True
    found: list[tuple[int, int, str]] = []
    for phrase, state in state_phrases():
        start = lower.find(phrase)
        while start != -1:
            end = start + len(phrase)
            if not any(taken[start:end]) and (start == 0 or not lower[start - 1].isalpha()):
                found.append((start, end, state))
                for i in range(start, end):
                    taken[i] = True
            start = lower.find(phrase, start + 1)
    return sorted(found)
