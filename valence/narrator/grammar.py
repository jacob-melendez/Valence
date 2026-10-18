"""Deterministic grammar for LoRA narrator targets: varied, natural explanations, faithful by construction.

``realize(sheet, rng)`` writes one explanation from exactly what the compact prompt shows the model
(``prompt.render_fact_sheet``): the person, ``label_display`` and ``confidence_display``, the face
line (mask override with the face-only reading, or hidden with the visible share), the prompt
drivers with their scope/level/effect and verbatim ``display``, and the out-of-range note. Nothing
else from the sheet (window fill, SHAP, raw face valence) is ever used, so a target never asks the
model for a fact it cannot see.

Grammar:

- **Openers** (8 plain + 3 per confidence band: >= 85% "clearly", 60-85% "most likely", < 60%
  "tentatively") always name the state with the verbatim ``label_display`` (so one person reads
  "Upbeat") and quote ``confidence_display``.
- **Drivers**: 2 or 3 of the key drivers (top-3), occasionally one more listed supporting driver
  (with no face line in the prompt, face-expression drivers are never cited: rule 7 of the system
  prompt says not to mention the face then, and two non-face key drivers always exist there),
  each realized by display shape (movement, posture share, counts, time since an event, room noise
  and light, break, clock, head count, ...) as a noun phrase or a clause. Displays are copied
  verbatim or re-quoted with their exact numbers ("noise averaging 72 dB over the last 15
  minutes"); a level word is added only when it agrees with the driver's level and the driver is
  the only allowed one of its concept ("a loud room with ..."). Room-scope drivers are phrased as
  the room, person-scope ones after "they" / Person #N.
- **Contrast**: the strongest listed opposing driver (never a face one) may appear only as contrast
  ("although X pulled the other way", "despite X", "X argued against it").
- **Face**: a face-vs-context clause only with ``mask_override`` and a visible face (smile wording
  only when the face-only model read a positive state); an occlusion clause only, and always, when
  the face is hidden (about the moment of the call, never "mostly hidden", when the face was
  visible for at least half the window); otherwise no face words at all.
- **Style guards**: gerund phrases ("being slumped ...") only in frames where they read well (never
  after a copula like "the main signals were"); "repeated" / "frequent" only for counts >= 2; at
  most one semicolon per sentence; the out-of-range note exactly once.
- **Orderings**: opener-first, evidence-first ("Based on ..., the model reads ..."), and
  contrast/face-first ("Although ..., ..."); then an optional closer (10) or the out-of-range note
  (always mentioned when the prompt shows it). Targets keep to what the system prompt asks: 2 or 3
  sentences and, on the preferred path, 30-72 words.

Every candidate goes through ``validate()``; a rejected one is resampled (with fewer optional parts
as tries run out), and the template is the last resort. ``TrigramDeduper`` rejects a target that
shares too many word trigrams with any single accepted one, keeping the dataset's distinct-2 high.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from valence.contracts import display_label
from valence.narrator.lexicon import driver_concept
from valence.narrator.prompt import prompt_drivers
from valence.narrator.schema import (
    allowed_drivers,
    as_dict,
    face_hidden,
    face_only_label,
    key_drivers,
    sheet_states,
)
from valence.narrator.templates import render_template
from valence.narrator.validator import count_words, validate

__all__ = ["Facts", "TrigramDeduper", "facts_of", "realize", "realize_candidate", "realize_many", "trigrams"]

TARGET_WORDS = (30, 72)  # preferred length (the prompt asks for 30-70; the validator allows 20-90)
MAX_TRIES = 40
FACE_CONCEPTS = ("face_valence", "face_visible")


# ---------------------------------------------------------------------------------------------
# What the prompt shows
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Facts:
    """The facts ``render_fact_sheet`` puts in front of the model, nothing more."""

    person: str
    label: str
    confidence: str
    band: str  # high | mid | low
    key: list[dict[str, Any]]
    extra: list[dict[str, Any]]  # listed supporting non-key drivers (no face concepts)
    opposing: list[dict[str, Any]]  # listed opposing drivers usable as contrast (no face concepts)
    allowed: list[dict[str, Any]]
    mask: bool  # mask override with a visible face
    hidden: bool
    face_name: str | None
    face_conf: str | None
    positive_face: bool
    face_agrees: bool  # mask override, yet the face-only model read the predicted state itself
    face_display: str | None  # a listed positive face-valence display (mask case only)
    visibility: str | None
    ood: bool


def _optional_ok(d: Mapping[str, Any]) -> bool:
    """May this driver be cited when it is optional (extra support, contrast)? Not a face driver,
    and not a raw feature-name display ("laugh_cnt_now = 0") the core could not describe."""
    return driver_concept(d) not in FACE_CONCEPTS and not _RAW_DISPLAY.match(str(d.get("display")))


def facts_of(sheet: Mapping[str, Any] | BaseModel) -> Facts:
    sheet = as_dict(sheet)
    listed = prompt_drivers(sheet)
    key_displays: list[str] = []
    for d in key_drivers(sheet):
        if str(d.get("display")) not in key_displays:
            key_displays.append(str(d.get("display")))
    key = [d for d in listed if str(d.get("display")) in key_displays]
    rest = [d for d in listed if str(d.get("display")) not in key_displays]
    hidden = face_hidden(sheet)
    mask = bool(sheet.get("mask_override")) and not hidden
    face_only = sheet.get("face_only") or {}
    baseline = face_only_label(sheet) if mask else None
    face_name = None
    if mask:
        face_name = (display_label(baseline, n_people=int(sheet.get("n_people") or 1), states=sheet_states(sheet))
                     if baseline else "a positive state")
    conf = face_only.get("confidence") if mask else None
    frac = (sheet.get("data_quality") or {}).get("face_visible_frac") if hidden else None
    positive = mask and (baseline in (None, "Collaborative_Happy"))
    face_display = None
    if positive:
        face_display = next((str(d["display"]) for d in rest if driver_concept(d) == "face_valence"
                             and str(d["display"]).startswith(("smiling", "slightly positive"))), None)
    match = re.search(r"\d+", str(sheet.get("confidence_display") or ""))
    pct = int(match.group(0)) if match else round(100 * float(sheet.get("confidence") or 0))
    return Facts(
        person=str(sheet.get("person") or "the person"),
        label=str(sheet.get("label_display")),
        confidence=str(sheet.get("confidence_display")),
        band="high" if pct >= 85 else "mid" if pct >= 60 else "low",
        key=key,
        extra=[d for d in rest if d.get("effect") == "supports" and _optional_ok(d)],
        opposing=[d for d in listed if d.get("effect") == "opposes" and _optional_ok(d)
                  and str(d.get("display")) not in key_displays],
        allowed=allowed_drivers(sheet),
        mask=mask,
        hidden=hidden,
        face_name=face_name,
        face_conf=f"{round(100 * conf)}%" if conf else None,
        positive_face=positive,
        face_agrees=mask and baseline == sheet.get("label"),
        face_display=face_display,
        visibility=f"{round(100 * frac)}%" if frac is not None else None,
        ood=bool((sheet.get("data_quality") or {}).get("ood_features")),
    )


# ---------------------------------------------------------------------------------------------
# Driver phrases
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Phrase:
    """One driver in two grammatical roles: noun phrases ("dim light (90 lux)") and clauses with a
    subject ("the room had dim light (90 lux)"). ``{who}`` in a clause is the person subject.
    ``gerunds`` ("being slumped ...") read well after "despite" or "driven by" but not after a
    copula ("the main signals were being ..."), so frames ask for them explicitly."""

    nps: list[str]
    cls: list[str]
    gerunds: list[str] = field(default_factory=list)


_MOTION_ADJ = re.compile(r"^(?:nearly still|moving normally|very restless|restless) \(motion")
_SHARE = re.compile(r"^(slumped|withdrawn|head in hands) (\d+%) of (the last .+)$")
_HOURS = re.compile(r"^(\d+(?:\.\d+)?) hours since the last break$")
_FIST_COUNT = re.compile(r"^(\d+) clenched-fist events? in (the last .+)$")
_SINCE = re.compile(r"^last (clenched-fist event|laughter event|compliment) (\d+) s ago$")
_PEOPLE = re.compile(r"^(\d+) people present$")
_LEAD_COUNT = re.compile(r"^(\d+) ")
_RAW_DISPLAY = re.compile(r"^[a-z_]+ = ")  # a feature name the core could not describe ("compl_cnt_now = 3")
_NOISE_LEVEL = re.compile(r"^(average|peak|current|recent) noise of (\d+) dB(?: over (the last .+))?$")
_LIGHT_LEVEL = re.compile(r"^(?:dim|normal|bright) light \(\d+ lux\)$")
_CLOCK = re.compile(r"\((\d{1,2}):(\d{2})\)")
_UP = ("high", "elevated")


def _time_word(display: str) -> str | None:
    """A clock word safely inside the validator's ranges for this clock time."""
    match = _CLOCK.search(display)
    if not match:
        return None
    hour = int(match.group(1)) + int(match.group(2)) / 60
    if 6 <= hour < 11.5:
        return "morning"
    if 12.5 <= hour < 17.5:
        return "afternoon"
    if 18 <= hour < 22.5:
        return "evening"
    if hour >= 22.5 or hour < 5:
        return "night"
    return None


def _sole(d: Mapping[str, Any], facts: Facts) -> bool:
    """Is this the only allowed driver of its concept? (A bare level word then binds to it alone.)"""
    concept = driver_concept(d)
    return sum(1 for a in facts.allowed if driver_concept(a) == concept) == 1


def phrase(d: Mapping[str, Any], facts: Facts, rng: random.Random) -> Phrase:
    """Noun-phrase and clause realizations of one driver whose qualifiers agree with its level,
    effect and scope. Paraphrases re-quote the display's own numbers."""
    text = str(d.get("display"))
    concept = driver_concept(d)
    level = str(d.get("level"))
    room = d.get("scope") == "room"
    sole = _sole(d, facts)
    nps: list[str] = []
    cls: list[str] = []
    gerunds: list[str] = []
    paraphrase = rng.random() < 0.3

    if concept == "motion" and _MOTION_ADJ.match(text):
        if text.startswith("moving normally"):
            gerunds += [text]
            nps += [f"normal movement{text[len('moving normally'):]}"]
        else:
            nps += [f"movement that was {text}"]
            gerunds += [f"being {text}"]
        cls += [f"{{who}} {{be}} {text}", f"{{whose}} movement was {text}"]
    elif (share := _SHARE.match(text)) is not None:
        what, pct, span = share.groups()
        if what == "head in hands":
            nps += [text, f"time with {text}"]
            cls += [f"{{who}} sat with {text}", f"{{who}} {{be}} seen with {text}"]
            if paraphrase:
                nps.append(f"head in hands for {pct} of {span}")
        else:
            gerunds += [f"being {text}"]
            nps += [f"a posture {text}" if what == "slumped" else f"withdrawal for {pct} of {span}"]
            cls += [f"{{who}} {{be}} {text}"] + ([f"{{who}} sat {text}"] if what == "slumped" else [])
            if paraphrase:
                cls.append(f"{{who}} {{be}} {what} for {pct} of {span}")
            if what == "slumped" and level == "high" and sole:
                nps.append(f"heavy slumping ({text})")
    elif text in ("withdrawn right now", "not withdrawn right now"):
        gerunds += [f"being {text}"]
        nps += ["withdrawal right now" if text.startswith("withdrawn") else "no withdrawal right now"]
        cls += [f"{{who}} {{is}} {text}"]
    elif concept == "break" and (hours := _HOURS.match(text)) is not None:
        h = hours.group(1)
        nps += [text]
        cls += [f"it had been {text}"]
        if level in _UP:
            if paraphrase:
                nps.append(f"{h} hours without a break")
                cls.append(f"{{who}} had gone {h} hours without a break")
            if sole:
                nps.append(f"{text} (a long stretch)")  # no comma: it sits inside driver lists
        elif level == "low":
            nps.append(f"only {text}")
    elif concept == "time_of_day":
        word = _time_word(text)
        nps += [f"the {text}"]
        cls += [f"the {text} played a part"]
        if word:
            nps.append(f"the {text} {'at' if word == 'night' else 'in the'} {word}")
            cls.append(f"it was the {word}, with the {text}")
        if level == "high" and sole:
            nps.append(f"a late {text}")
    elif concept in ("fist", "laughter", "compliment"):
        since = _SINCE.match(text)
        if since is not None:
            what, seconds = since.groups()
            nps += [f"the {text}"]
            cls += [f"the room had its {text}" if room else f"{{who}} had their {text}"]
            if paraphrase:
                nps.append({"clenched-fist event": f"a clenched fist {seconds} s ago",
                            "laughter event": f"laughter in the room {seconds} s ago",
                            "compliment": f"a compliment in the room {seconds} s ago"}[what])
        else:
            singular = text.startswith(("1 ", "no clenched-fist event ", "no compliment ", "no laughter event "))
            nps += [text]
            cls += [f"there {'was' if singular else 'were'} {text}"]
            if not room and not text.startswith("no "):
                cls.append(f"{{who}} had {text}")
            count = _FIST_COUNT.match(text)
            if count is not None and paraphrase and int(count.group(1)) >= 2:
                nps.append(f"{count.group(1)} clenched fists in {count.group(2)}")
            lead = _LEAD_COUNT.match(text)
            if level in _UP and sole and lead is not None and int(lead.group(1)) >= 2:
                nps.append({"fist": f"repeated fist clenching ({text})", "laughter": f"frequent laughter ({text})",
                            "compliment": f"repeated compliments ({text})"}[concept])
    elif concept == "people":
        people = _PEOPLE.match(text)
        nps += [text]
        cls += [f"there {'were' if people else 'was'} {text}"]
        if people is not None:
            if paraphrase:
                nps.append(f"{people.group(1)} people in the room")
            if level in _UP and sole and int(people.group(1)) >= 3:
                nps.append(f"a room busy with {text}")
    elif concept == "face_valence":
        article = "" if text.startswith("expression varying") else "a "
        nps += [f"{article}{text}"]
        cls += [f"{{who}} showed {article}{text}"]
    elif concept == "face_visible":
        nps += [text]
        cls += [f"{{who}} had their {text}"]
    elif concept in ("noise", "light") and room:
        nps += [text, f"{text} in the room", f"the room's {text}"]
        cls += [f"the room had {text}", f"the room showed {text}"]
        noise = _NOISE_LEVEL.match(text) if concept == "noise" else None
        if noise is not None and paraphrase:
            kind, db, span = noise.groups()
            tail = f" over {span}" if span else ""
            nps.append({"average": f"noise averaging {db} dB{tail}", "peak": f"noise peaking at {db} dB{tail}",
                        "current": f"noise at {db} dB right now", "recent": f"recent noise around {db} dB{tail}"}[kind])
        if sole and (noise is not None or _LIGHT_LEVEL.match(text)):
            if concept == "noise":
                adjective = rng.choice(("loud", "noisy")) if level in _UP else "quiet"
                nps.append(f"a {adjective} room with {text}")
            elif level == "low" and not text.startswith("dim"):
                nps.append(f"a dim room with {text}")
            elif level == "high" and not text.startswith("bright"):
                nps.append(f"a bright room with {text}")
    else:
        nps += [text]
        cls += [f"the room had {text}"] if room else [f"the data showed {text}", f"the readings showed {text}"]
    return Phrase(nps, cls, gerunds)


# ---------------------------------------------------------------------------------------------
# Sentence parts
# ---------------------------------------------------------------------------------------------


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: Sequence[str], rng: random.Random) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    comma = "," if rng.random() < 0.5 else ""
    return f"{', '.join(items[:-1])}{comma} and {items[-1]}"


def _fill(text: str, who: str) -> str:
    """Fill the person subject of a clause ("they were" / "Person #1 was")."""
    they = who == "they"
    return text.format(**{"who": who, "be": "were" if they else "was", "is": "are" if they else "is",
                          "whose": "their" if they else f"{who}'s"})


def _np(p: Phrase, rng: random.Random, gerund: bool = True) -> str:
    """A noun phrase; gerunds ("being slumped ...") only where the frame reads well with one."""
    options = [*p.nps, *p.gerunds] if gerund or not p.nps else p.nps
    return rng.choice(options or p.gerunds)


def _cl(p: Phrase, rng: random.Random, who: str = "they") -> str:
    return _fill(rng.choice(p.cls), who)


def _np_list(phrases: Sequence[Phrase], rng: random.Random, gerund: bool = True) -> str:
    return _join([_np(p, rng, gerund) for p in phrases], rng)


def _cl_list(phrases: Sequence[Phrase], rng: random.Random, first_who: str = "they") -> str:
    clauses = [_cl(p, rng, first_who if i == 0 else "they") for i, p in enumerate(phrases)]
    for i in range(1, len(clauses)):  # "the room had A and the room had B" -> "... and it had B"
        if clauses[i].startswith("the room ") and clauses[i - 1].startswith(("the room ", "it ")):
            clauses[i] = "it " + clauses[i][len("the room "):]
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return rng.choice((f"{clauses[0]}, and {clauses[1]}", f"{clauses[0]}, while {clauses[1]}",
                           f"{clauses[0]} and {clauses[1]}"))
    return rng.choice((f"{clauses[0]}, {clauses[1]}, and {clauses[2]}",
                       f"{clauses[0]} and {clauses[1]}, while {clauses[2]}"))


def _hedged(f: Facts, rng: random.Random) -> bool:
    return rng.random() < (0.65 if f.band == "low" else 0.45)


def _verdict(f: Facts, rng: random.Random, named: bool = False) -> str:
    """A clause stating the prediction, hedged by confidence band about half the time. With
    ``named`` the person was just named in the same sentence, so the clause says "they"/"them"."""
    s, c = f.label, f.confidence
    if named:
        plain = [f"the model reads them as {s} ({c})", f"they read as {s} with {c} confidence",
                 f"the model places them at {s} ({c})", f"they come out as {s} at {c}",
                 f"the prediction is {s} ({c})", f"the model settles on {s} ({c})"]
        hedged = {
            "high": [f"they clearly read as {s} ({c})", f"the model is confident they read as {s} ({c})",
                     f"{s} is a firm call at {c}"],
            "mid": [f"they most likely read as {s} ({c})", f"the model leans toward {s} ({c})",
                    f"they probably fit {s}, at {c}"],
            "low": [f"the model tentatively reads them as {s}, at only {c} confidence",
                    f"{s} is only a tentative call ({c})", f"the model weakly favors {s} ({c})"],
        }[f.band]
        return rng.choice(hedged if _hedged(f, rng) else plain)
    p = f.person
    plain = [
        f"the model reads {p} as {s} ({c})",
        f"{p} reads as {s} with {c} confidence",
        f"the model places {p} at {s} ({c})",
        f"{p} comes out as {s} at {c}",
        f"the context model rates {p} as {s} ({c})",
        f"the prediction for {p} is {s} ({c})",
        f"{s} is the call for {p}, at {c} confidence",
        f"the model settles on {s} for {p} ({c})",
    ]
    hedged = {
        "high": [f"{p} clearly reads as {s} ({c})", f"the model is confident that {p} reads as {s} ({c})",
                 f"{p} reads strongly as {s} ({c})", f"{s} is a firm call for {p} ({c})"],
        "mid": [f"{p} most likely reads as {s} ({c})", f"the model leans toward {s} for {p} ({c})",
                f"{s} is the likely state for {p} ({c})", f"{p} probably fits {s}, at {c}"],
        "low": [f"the model tentatively reads {p} as {s}, at only {c} confidence",
                f"{p} leans toward {s}, with only {c} confidence", f"{s} is a tentative call for {p} ({c})",
                f"the model only weakly favors {s} for {p} ({c})"],
    }[f.band]
    return rng.choice(hedged if _hedged(f, rng) else plain)


def _opener(f: Facts, rng: random.Random) -> str:
    p, s, c = f.person, f.label, f.confidence
    plain = [
        f"Model predicted {s} ({c}) for {p}.",
        f"{p} reads as {s} with {c} confidence.",
        f"The context model places {p} at {s} ({c}).",
        f"For {p}, the most likely state is {s} at {c}.",
        f"{s} is the call for {p}, at {c} confidence.",
        f"The model's read on {p} is {s}, with {c} confidence.",
        f"With {c} confidence, {p} is classified as {s}.",
        f"In the current window, {p} matches {s} ({c}).",
        f"The context audit rates {p} as {s} ({c}).",
        f"The audit for {p} shows {s} at {c} confidence.",
        f"Right now {p} is predicted {s} ({c}).",
    ]
    hedged = {
        "high": [f"{p} clearly reads as {s} ({c}).", f"The model is confident: {p} reads as {s} at {c}.",
                 f"{s} is a strong call for {p} ({c}).", f"With a firm {c}, {p} comes out as {s}."],
        "mid": [f"{p} most likely reads as {s} ({c}).",
                f"The model leans toward {s} for {p}, at {c} confidence.",
                f"{s} is the probable state for {p} ({c}).", f"{p} probably fits {s}, at {c} confidence."],
        "low": [f"The model tentatively reads {p} as {s}, at only {c} confidence.",
                f"{s} is a low-confidence call for {p} ({c}).",
                f"At just {c}, {s} is only a tentative reading for {p}.",
                f"The model only weakly favors {s} for {p}, at {c}."],
    }[f.band]
    return rng.choice(hedged if _hedged(f, rng) else plain)


def _contrast_tail(p: Phrase, rng: random.Random) -> str:
    """An opposing driver appended to a sentence, framed only as contrast."""
    return rng.choice((
        lambda: f", although {_np(p, rng)} pulled the other way",
        lambda: f", despite {_np(p, rng)}",
        lambda: f"; {_np(p, rng)} argued against it",
        lambda: f", though {_np(p, rng)} weighed against it",
        lambda: f", although {_cl(p, rng)}",
        lambda: f", even though {_cl(p, rng)}",
        lambda: f", whereas {_cl(p, rng)}",
        lambda: f"; {_np(p, rng)} pointed the other way",
        lambda: f", even with {_np(p, rng, gerund=False)}",
    ))()


def _contrast_lead(p: Phrase, f: Facts, rng: random.Random) -> str:
    verdict = _verdict(f, rng)
    return rng.choice((
        lambda: f"Although {_np(p, rng)} pulled the other way, {verdict}.",
        lambda: f"Despite {_np(p, rng)}, {verdict}.",
        lambda: f"Although {_cl(p, rng)}, {verdict}.",
        lambda: f"Even though {_cl(p, rng)}, {verdict}.",
        lambda: f"In spite of {_np(p, rng)}, {verdict}.",
    ))()


def _contrast_sentence(p: Phrase, rng: random.Random) -> str:
    return rng.choice((
        lambda: f"Only {_np(p, rng)} pulled the other way.",
        lambda: f"{_cap(_np(p, rng))} argued against it, but not enough to change the call.",
        lambda: f"Although {_cl(p, rng)}, the overall pattern held.",
        lambda: f"Despite {_np(p, rng)}, the pattern held.",
        lambda: f"The one counterweight was {_np(p, rng, gerund=False)}, which pointed the other way.",
        lambda: f"Even though {_cl(p, rng)}, that was not enough to shift the reading.",
    ))()


def _evidence(phrases: Sequence[Phrase], rng: random.Random, tail: str = "") -> str:
    """One sentence citing the drivers (after an opener that already named the person)."""
    forms: list[Callable[[], str]] = [
        lambda: f"The main signals were {_np_list(phrases, rng, gerund=False)}{tail}.",
        lambda: f"{_cap(_cl_list(phrases, rng))}{tail}.",
        lambda: f"This rests on {_np_list(phrases, rng)}{tail}.",
        lambda: f"The model weighed {_np_list(phrases, rng, gerund=False)}{tail}.",
        lambda: f"Most of the weight came from {_np_list(phrases, rng)}{tail}.",
        lambda: f"Key evidence: {_np_list(phrases, rng)}{tail}.",
        lambda: f"It is driven by {_np_list(phrases, rng)}{tail}.",
        lambda: f"{_cap(_np_list(phrases, rng))} pointed that way{tail}.",
        lambda: f"The evidence includes {_np_list(phrases, rng, gerund=False)}{tail}.",
        lambda: f"The strongest cues were {_np_list(phrases, rng, gerund=False)}{tail}.",
        lambda: f"What tipped it was {_np_list(phrases, rng)}{tail}.",
        lambda: f"The reading draws on {_np_list(phrases, rng)}{tail}.",
        lambda: f"In the data, {_cl_list(phrases, rng)}{tail}.",
        lambda: f"Behind this call are {_np_list(phrases, rng, gerund=False)}{tail}.",
    ]
    if not tail:
        forms.append(lambda: f"{_cap(_cl_list(phrases, rng))}, which fits that reading.")
        forms.append(lambda: f"{_cap(_np_list(phrases, rng))} carried most of the weight.")
    return rng.choice(forms)()


def _evidence_lead(phrases: Sequence[Phrase], f: Facts, rng: random.Random, tail: str = "") -> str:
    """Evidence first, then the prediction it supports."""
    forms: list[Callable[[], str]] = [
        lambda: f"{_cap(_cl_list(phrases, rng, f.person))}, so {_verdict(f, rng, named=True)}{tail}.",
        lambda: f"Based on {_np_list(phrases, rng)}, {_verdict(f, rng)}{tail}.",
        lambda: f"With {_np_list(phrases, rng, gerund=False)}, {_verdict(f, rng)}{tail}.",
        lambda: f"Given {_np_list(phrases, rng, gerund=False)}, {_verdict(f, rng)}{tail}.",
        lambda: f"{_cap(_np_list(phrases, rng))} put {f.person} at {f.label}, with {f.confidence} confidence{tail}.",
        lambda: f"Weighing {_np_list(phrases, rng, gerund=False)}, {_verdict(f, rng)}{tail}.",
    ]
    if not tail.startswith(";"):  # one semicolon per sentence at most
        forms.append(lambda: f"{_cap(_cl_list(phrases, rng, f.person))}; on that basis "
                             f"{_verdict(f, rng, named=True)}{tail}.")
    return rng.choice(forms)()


def _paren(value: str | None, rng: random.Random, form: str = " ({v})", p: float = 0.75) -> str:
    return form.format(v=value) if value and rng.random() < p else ""


def _mask_sentence(f: Facts, rng: random.Random) -> str:
    face, fc = f.face_name, _paren(f.face_conf, rng)
    if f.face_agrees:  # nothing was corrected: say the face reading points the same way
        return rng.choice((
            f"The face-only model read {face}{fc} as well, so face and context line up here.",
            f"The face alone also suggested {face}{fc}, in line with the context.",
            f"A face-only reading of {face}{fc} points the same way.",
        ))
    options = [
        f"The face alone suggested {face}{fc}, but context corrected the facial mask.",
        f"Although the face-only model read {face}{fc}, context corrected the facial mask.",
        f"The face-only reading was {face}{fc}; the context layers overrode that facial mask.",
        f"On the surface the face read as {face}{fc}, but context corrected the mask.",
        f"A face-only reading of {face}{fc} was set aside, since the context signals outweighed the facial mask.",
        f"The face pointed to {face}{fc}, yet context treated that as a mask and overruled it.",
        f"Context overrode the face-only reading of {face}{fc}, which it treats as a mask.",
    ]
    if f.positive_face:
        options += [f"A smile led the face-only model to {face}{fc}, but context saw past the mask.",
                    f"Their smiling face pointed to {face}{fc}, yet the context layers flagged it as a mask.",
                    f"The outward smile read as {face}{fc} to the face-only model, but context corrected it."]
        if f.face_display:
            options.append(f"A {f.face_display} led the face-only model to {face}{fc}, but context corrected "
                           "the facial mask.")
    return rng.choice(options)


def _mask_lead(f: Facts, rng: random.Random) -> str:
    face, fc = f.face_name, _paren(f.face_conf, rng)
    if f.face_agrees:
        return rng.choice((
            lambda: f"In line with a face-only reading of {face}{fc}, {_verdict(f, rng)}.",
            lambda: f"The face-only model also read {face}{fc}, and {_verdict(f, rng)}.",
        ))()
    return rng.choice((
        lambda: f"The face alone suggested {face}{fc}, but context places {f.person} at {f.label} ({f.confidence}).",
        lambda: f"Although the face-only model read {face}{fc}, context puts {f.person} at {f.label} ({f.confidence}).",
        lambda: f"Context overrode the facial mask: the face-only model read {face}{fc}, but {_verdict(f, rng)}.",
        lambda: f"Behind a face that read as {face}{fc}, {_verdict(f, rng)}.",
        lambda: f"The face-only model saw {face}{fc}; looking past that mask, {_verdict(f, rng)}.",
    ))()


_VIS_FORMS = (" (visible {v} of the window)", " (visible only {v} of the window)", " (seen {v} of the window)",
              ", visible {v} of the window,")


def _vis(f: Facts, rng: random.Random, inline: bool = True) -> str:
    forms = _VIS_FORMS if inline else _VIS_FORMS[:3]
    if f.visibility == "0%":  # "visible only 0%" reads oddly
        forms = tuple(form for form in forms if "only" not in form)
    return _paren(f.visibility, rng, rng.choice(forms))


def _mostly_visible(f: Facts) -> bool:
    """The face is hidden now but was visible for at least half the window: occlusion wording must
    then be about the moment of the call, never "mostly hidden"."""
    return f.visibility is not None and int(f.visibility.rstrip("%")) >= 50


def _hidden_sentence(f: Facts, rng: random.Random) -> str:
    vis = _vis(f, rng, inline=False)
    if _mostly_visible(f):
        return rng.choice((
            f"The face was hidden at prediction time{vis}, so the reading relies on context.",
            f"With the face hidden at the moment of the call{vis}, only context signals were used.",
            f"Their face was out of view when the prediction was made{vis}, so no face-only reading was used.",
            f"The face was not visible at prediction time{vis}, so context carried the reading.",
            f"No usable face reading was available, as the face was hidden at prediction time{vis}.",
            f"Because the face was out of view at the moment of the call{vis}, context signals carried it.",
        ))
    return rng.choice((
        f"The face was hidden{vis}, so the call relies on context.",
        f"With the face hidden{vis}, only context signals were used.",
        f"Their face was out of view{vis}, so no face-only reading was used.",
        f"The face stayed mostly hidden{vis}, leaving the reading to context.",
        f"The face was not visible{vis}, so context carried the call.",
        f"The face was obscured{vis}, so the estimate leans on context alone.",
        f"No usable face reading was available, as the face was hidden{vis}.",
        f"Because the face was out of view{vis}, the call comes from context signals only.",
    ))


def _hidden_lead(f: Facts, rng: random.Random) -> str:
    vis = _vis(f, rng, inline=False)
    now = " at prediction time" if _mostly_visible(f) else ""
    return rng.choice((
        lambda: f"With the face hidden{now}{vis}, {_verdict(f, rng)}.",
        lambda: f"The face was out of view{now}{vis}, so context alone puts {f.person} at {f.label} ({f.confidence}).",
        lambda: f"Since the face was hidden{now}{vis}, {_verdict(f, rng)} from context alone.",
        lambda: f"Reading context only, because the face was obscured{now}{vis}, {_verdict(f, rng)}.",
    ))()


CLOSERS: tuple[str, ...] = (
    "No single signal decided it; the pattern across layers did.",
    "These are aggregate context measurements, with no identity data involved.",
    "The pattern across the context layers carries this reading.",
    "Taken together, the context layers make this the most likely state.",
    "It is a reading of the context, not a judgment about the person.",
    "The combination matters more than any single reading.",
    "Each of these is a measured context signal, not a guess about identity.",
    "No one cue settles it on its own; together they do.",
    "The estimate comes from aggregate signals in the current window.",
    "Together these cues make the call.",
)
OOD_SENTENCES: tuple[str, ...] = (
    "Some readings fall outside the training range, so treat this with care.",
    "Note that some readings are outside the training range.",
    "Some of these readings sit outside the training range.",
    "Some inputs lie beyond what the model saw in training, so hold this loosely.",
    "Some values are unusual for the model, outside its training range.",
    "Treat this with some caution: several inputs are out of the training range.",
)
OOD_TAILS: tuple[str, ...] = (
    "; some readings fall outside the training range.",
    ", and some readings are outside the training range.",
    "; note that some readings are outside the training range.",
    ", and some inputs lie beyond the training range.",
    "; some values are unusual for the model.",
)


def _with_ood(sentence: str, rng: random.Random) -> str:
    """Append the out-of-range note; never a second semicolon in one sentence."""
    tails = [t for t in OOD_TAILS if not t.startswith(";")] if ";" in sentence else list(OOD_TAILS)
    return sentence[:-1] + rng.choice(tails)


# ---------------------------------------------------------------------------------------------
# Realization
# ---------------------------------------------------------------------------------------------


def _pick_drivers(f: Facts, rng: random.Random, lean: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Key drivers to cite (>= 2 of the top-3 when there are that many), an occasional extra
    supporting one, and an optional opposing one. ``lean`` > 0 drops optional parts."""
    key = list(f.key)
    if not f.mask and not f.hidden:  # no face line in the prompt: rule 7 says not to mention the face
        plain = [d for d in key if driver_concept(d) not in FACE_CONCEPTS]
        key = plain if len(plain) >= 2 else key
    readable = [d for d in key if not _RAW_DISPLAY.match(str(d.get("display")))]
    key = readable if len(readable) >= 2 else key
    if len(key) >= 3:
        n = 3 if rng.random() < (0.5 if lean == 0 else 0.15) else 2
        if n == 2:
            faces = [d for d in key if driver_concept(d) in FACE_CONCEPTS]
            if faces and rng.random() < 0.7:
                key = [d for d in key if d is not faces[0]][:3]
            chosen = key[:2] if rng.random() < 0.6 else rng.sample(key[:3], 2)
            chosen.sort(key=lambda d: f.key.index(d))
        else:
            chosen = key[:3]
    else:
        chosen = key
    if lean == 0 and f.extra and len(chosen) < 3 and rng.random() < 0.15:
        chosen.append(f.extra[0] if rng.random() < 0.6 else rng.choice(f.extra))
    if len(chosen) > 1 and rng.random() < 0.25:
        rng.shuffle(chosen)
    opposing = None
    if f.opposing and lean < 2 and rng.random() < 0.75:
        opposing = f.opposing[0] if rng.random() < 0.7 else rng.choice(f.opposing)
    return chosen, opposing


def realize_candidate(sheet: Mapping[str, Any] | BaseModel, rng: random.Random, lean: int = 0) -> str:
    """One unvalidated grammar sample (``lean`` 0-2: higher drops optional parts)."""
    f = facts_of(sheet)
    chosen, opposing = _pick_drivers(f, rng, lean)
    phrases = [phrase(d, f, rng) for d in chosen]
    against = phrase(opposing, f, rng) if opposing is not None else None
    face = "mask" if f.mask else "hidden" if f.hidden else None
    order = rng.choice(("opener", "evidence", "lead")) if (face or against) else rng.choice(("opener", "evidence"))
    sentences: list[str] = []
    if order == "opener":
        tail = _contrast_tail(against, rng) if against else ""
        sentences = [_opener(f, rng), _evidence(phrases, rng, tail)]
        if face == "mask":
            sentences.append(_mask_sentence(f, rng))
        elif face == "hidden":
            sentences.append(_hidden_sentence(f, rng))
    elif order == "evidence":
        tail = _contrast_tail(against, rng) if against and face else ""
        sentences = [_evidence_lead(phrases, f, rng, tail)]
        if face == "mask":
            sentences.append(_mask_sentence(f, rng))
        elif face == "hidden":
            sentences.append(_hidden_sentence(f, rng))
        elif against:
            sentences.append(_contrast_sentence(against, rng))
    else:
        if face == "mask":
            lead = _mask_lead(f, rng)
        elif face == "hidden":
            lead = _hidden_lead(f, rng)
        else:
            lead, against = _contrast_lead(against, f, rng), None
        tail = _contrast_tail(against, rng) if against else ""
        sentences = [lead, _evidence(phrases, rng, tail)]
    noted = False  # the out-of-range note is said exactly once
    if len(sentences) < 2:  # the system prompt asks for 2 or 3 sentences
        sentences.append(rng.choice(OOD_SENTENCES if f.ood else CLOSERS))
        noted = f.ood
    elif len(sentences) < 3 and lean < 2:
        if f.ood and rng.random() < 0.6:
            sentences.append(rng.choice(OOD_SENTENCES))
            noted = True
        elif rng.random() < 0.55:
            sentences.append(rng.choice(CLOSERS))
    if f.ood and not noted:
        sentences[-1] = _with_ood(sentences[-1], rng)
    return " ".join(sentences)


def realize(sheet: Mapping[str, Any] | BaseModel, rng: random.Random,
            accept: Callable[[str], bool] | None = None, tries: int = MAX_TRIES) -> str:
    """A validator-clean grammar explanation for the sheet; resamples on any rejection (and when
    ``accept`` says no, e.g. a near-duplicate), dropping optional parts as tries run out. Falls
    back to the best rejected-by-``accept`` candidate, then to the template."""
    sheet = as_dict(sheet)
    fallback: str | None = None
    for attempt in range(tries):
        lean = 0 if attempt < tries // 2 else 1 if attempt < 3 * tries // 4 else 2
        text = realize_candidate(sheet, rng, lean)
        words = count_words(text)
        if attempt < tries // 2 and not TARGET_WORDS[0] <= words <= TARGET_WORDS[1]:
            continue
        if not validate(text, sheet).ok:
            continue
        if accept is None or accept(text):
            return text
        fallback = fallback or text
    return fallback or render_template(sheet)


# ---------------------------------------------------------------------------------------------
# Dataset-level dedupe
# ---------------------------------------------------------------------------------------------


_TOKEN = re.compile(r"[a-z0-9]+(?:[.'%-][a-z0-9]+)*%?")  # same tokens as evaluate.distinct_n


def trigrams(text: str) -> set[tuple[str, ...]]:
    tokens = _TOKEN.findall(text.lower())
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


class TrigramDeduper:
    """Rejects a text whose trigrams overlap any single accepted text by more than ``max_overlap``
    (shared trigrams / the candidate's trigrams)."""

    def __init__(self, max_overlap: float = 0.4) -> None:
        self.max_overlap = max_overlap
        self._index: dict[tuple[str, ...], list[int]] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def overlap(self, text: str) -> float:
        grams = trigrams(text)
        if not grams:
            return 0.0
        shared: Counter[int] = Counter(i for g in grams for i in self._index.get(g, ()))
        return max(shared.values(), default=0) / len(grams)

    def accepts(self, text: str) -> bool:
        return self.overlap(text) <= self.max_overlap

    def add(self, text: str) -> None:
        for gram in trigrams(text):
            self._index.setdefault(gram, []).append(self._count)
        self._count += 1


def sheet_rng(sheet: Mapping[str, Any], seed: int) -> random.Random:
    """Per-sheet generator: the same (seed, event id) always gives the same target."""
    return random.Random(f"valence-grammar:{seed}:{sheet.get('event_id')}")


def realize_many(sheets: Iterable[Mapping[str, Any]], seed: int = 0, max_overlap: float | None = 0.4,
                 tries: int = MAX_TRIES) -> tuple[list[str], dict[str, int]]:
    """Targets for a sequence of sheets, deduplicated across the whole sequence. Returns the
    targets and counts (``dedupe_fallback``: no candidate passed the dedupe within ``tries``;
    ``template``: the grammar never validated and the template was used)."""
    deduper = TrigramDeduper(max_overlap) if max_overlap is not None else None
    targets: list[str] = []
    stats = Counter({"dedupe_fallback": 0, "template": 0})
    for sheet in sheets:
        sheet = as_dict(sheet)
        text = realize(sheet, sheet_rng(sheet, seed), deduper.accepts if deduper else None, tries)
        if deduper is not None:
            if not deduper.accepts(text):
                stats["dedupe_fallback"] += 1
            deduper.add(text)
        if text == render_template(sheet):
            stats["template"] += 1
        targets.append(text)
    return targets, dict(stats)
