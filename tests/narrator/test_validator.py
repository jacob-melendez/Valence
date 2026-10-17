from __future__ import annotations

import pytest

from valence.narrator.fake_fact_sheets import example_fact_sheet
from valence.narrator.lexicon import text_concepts
from valence.narrator.schema import parse_fact_sheet
from valence.narrator.validator import (
    allowed_numbers,
    clean_text,
    extract_numbers,
    number_allowed,
    split_sentences,
    strip_think,
    validate,
)

MASKED = (
    "Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1. "
    "Ambient noise (72 dB), clenched-fist onsets (3) and time since last break (4.2 hours) drove the call. "
    "Although the expression reads as positive (facial valence 0.62), context corrected the facial mask."
)
PLAIN = (
    "Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1. "
    "Ambient noise (72 dB), clenched-fist onsets (3) and time since last break (4.2 hours) drove the call. "
    "These context signals carry the explanation."
)
PRD = (
    "Model predicted 'Frustrated (Masked)' with 88% confidence. While Person #1 is currently displaying a "
    "superficial smiling expression (high facial valence), the system detected a somatic trigger (clenched fists) "
    "paired with a negative environmental history layer: high ambient noise (72dB) and an extended duration since "
    "the last communal room break (4.2 hours). Context corrected the facial mask."
)


@pytest.fixture()
def sheet() -> dict:
    return example_fact_sheet()


@pytest.fixture()
def unmasked() -> dict:
    sheet = example_fact_sheet()
    sheet["mask_override"] = False
    sheet["face_only"] = {"label": "Frustrated_Masked", "confidence": 0.66, "face_valence": -0.2}
    sheet["layers"][3]["drivers"][0]["value"] = -0.2
    return sheet


def _driver(sheet: dict, feature: str) -> dict:
    return next(d for layer in sheet["layers"] for d in layer["drivers"] if d["feature"] == feature)


def reasons(text: str, sheet: dict) -> list[str]:
    return validate(text, sheet).reasons


def test_reference_texts_pass(sheet, unmasked):
    for text, target in ((MASKED, sheet), (PRD, sheet), (PRD.replace("'Frustrated (Masked)'", "'Frustrated'"), sheet),
                         (PLAIN, unmasked)):
        result = validate(text, target)
        assert result.ok, (result.reasons, text)


def test_prd_example_reports_concepts_and_drivers(sheet):
    result = validate(PRD, sheet)
    assert result.ok
    assert {"noise", "fist", "break", "face_valence"} <= set(result.mentioned_concepts)
    assert result.mentioned_drivers == ["ambient_db", "fist_onsets_15m", "hours_since_break"]
    assert result.sentence_count == 3


def test_validate_accepts_pydantic_model(sheet):
    assert validate(MASKED, parse_fact_sheet(sheet)).ok


@pytest.mark.parametrize("quote", ["88%", "88 percent", "0.88", "88.0%", "0.9"])
def test_confidence_fraction_and_percent_match(sheet, quote):
    assert validate(MASKED.replace("88%", quote), sheet).ok


@pytest.mark.parametrize("quote", ["89%", "87%", "0.87", "88.4%"])
def test_confidence_off_by_more_than_rounding_rejected(sheet, quote):
    assert reasons(MASKED.replace("88%", quote), sheet) == [f"number:{quote}"]


def test_fraction_leaves_match_percentages(sheet):
    text = MASKED.replace("context corrected", "with the face visible 93% of the time and a 4% Focused odds, context corrected")
    assert "number:93%" not in reasons(text, sheet)
    assert "number:4%" not in reasons(text, sheet)


@pytest.mark.parametrize("quote,ok", [("72dB", True), ("72.3dB", True), ("72 dB", True), ("73dB", False),
                                      ("72.4 dB", False), ("72.30 dB", True)])
def test_numbers_glued_to_units(sheet, quote, ok):
    result = validate(MASKED.replace("72 dB", quote), sheet)
    assert result.ok is ok, result.reasons


def test_hours_glued_unit(sheet):
    assert validate(MASKED.replace("4.2 hours", "4.2h"), sheet).ok


@pytest.mark.parametrize("quote,ok", [("4.2", True), ("4.3", True), ("4", True), ("4.25", True),
                                      ("4.4", False), ("4.35", False), ("5", False), ("4.20", False)])
def test_rounding_boundaries(sheet, quote, ok):
    _driver(sheet, "hours_since_break")["value"] = 4.25
    result = validate(MASKED.replace("4.2 hours", f"{quote} hours"), sheet)
    assert result.ok is ok, result.reasons


@pytest.mark.parametrize("quote,ok", [("88%", True), ("88.4%", True), ("88.5%", False), ("89%", False)])
def test_percent_rounding_from_fraction(sheet, quote, ok):
    sheet["confidence"] = 0.884
    sheet["confidence_display"] = "88%"
    result = validate(MASKED.replace("88%", quote), sheet)
    assert result.ok is ok, result.reasons


def test_unknown_number_rejected(sheet):
    assert "number:57" in reasons(MASKED.replace("(3)", "(57)"), sheet)


def test_clock_times_match_fractional_hours(sheet):
    sheet["layers"][1]["drivers"].append(
        {"feature": "time_of_day_h", "display": "Time of day", "value": 15.5, "unit": "h", "shap": 0.2})
    base = MASKED.replace("drove the call", "at {} drove the call")
    assert validate(base.format("15:30"), sheet).ok
    assert validate(base.format("3:30 pm"), sheet).ok
    assert "number:16:30" in reasons(base.format("16:30"), sheet)


def test_person_number_rules(sheet):
    assert validate(MASKED.replace("for Person #1", "for Person 1"), sheet).ok
    assert "person:2" in reasons(MASKED.replace("for Person #1", "for Person #2"), sheet)
    assert "number:2" not in reasons(MASKED.replace("for Person #1", "for Person #2"), sheet)


def test_sentence_splitting_handles_decimals_and_person_numbers():
    text = "Noise hit 72.3 dB for Person #1. Breaks were 4.2 h apart, e.g. a long stretch. Light was 180 lux."
    assert len(split_sentences(text)) == 3
    assert len(split_sentences("Person #1. Then 4.5 h passed. Done.")) == 3


def test_too_many_sentences_rejected(sheet):
    text = MASKED + " This is a fourth sentence."
    assert "sentences:4" in reasons(text, sheet)


def test_word_bounds(sheet):
    short = "Model predicted 'Frustrated (Masked)' at 88%: ambient noise (72 dB), fists (3)."
    assert any(r.startswith("words:") for r in reasons(short, sheet))
    long = MASKED.replace("drove the call", "drove the call " + "and that pattern held " * 15)
    assert any(r.startswith("words:") for r in reasons(long, sheet))


def test_think_blocks_are_stripped(sheet):
    assert validate("<think>The user wants 57 words about Person #2.</think>\n" + MASKED, sheet).ok
    assert validate("reasoning about 57 things\n</think>\n\n" + MASKED, sheet).ok
    assert validate("<think>\n\n</think>\n\n" + MASKED, sheet).text == MASKED
    unterminated = validate("<think>still deciding what to say about " + MASKED, sheet)
    assert not unterminated.ok and unterminated.reasons == ["empty"]
    assert strip_think("Answer.<think>trailing") == "Answer."


def test_markdown_and_quotes_are_cleaned(sheet):
    wrapped = '"**Model predicted** \'Frustrated (Masked)\' with *88%* confidence for Person #1. ' + MASKED.split(". ", 1)[1] + '"'
    result = validate(wrapped, sheet)
    assert result.ok, result.reasons
    assert result.text.startswith("Model predicted 'Frustrated (Masked)'")
    assert clean_text("## Audit\nExplanation: " + MASKED) == MASKED


def test_state_must_be_named(sheet):
    text = MASKED.replace("'Frustrated (Masked)'", "a state")
    assert "state_missing" in reasons(text, sheet)


def test_other_state_as_prediction_rejected(sheet, unmasked):
    assert "other_state:Exhausted" in reasons(MASKED.replace("drove the call", "suggest they are exhausted"), sheet)
    contrast = PLAIN.replace("These context signals", "A face-only reading of 'Collaborative (Happy)' lost; these signals")
    assert "other_state:Collaborative_Happy" in reasons(contrast, unmasked)


def test_face_only_label_allowed_in_mask_contrast(sheet):
    text = MASKED.replace("context corrected the facial mask", "context corrected the face-only 'Collaborative (Happy)' mask")
    assert validate(text, sheet).ok


def test_masked_label_display_exempt_without_override(unmasked):
    assert validate(PLAIN, unmasked).ok


@pytest.mark.parametrize("word", ["smiling", "despite", "forced", "masking"])
def test_mask_terms_need_override(unmasked, word):
    text = PLAIN.replace("These context signals", f"With {word} cues, these context signals")
    assert f"mask_term:{word}" in reasons(text, unmasked)


@pytest.mark.parametrize("insert,reason", [
    ("She showed", "banned:gendered:she"),
    ("His desk showed", "banned:gendered:his"),
    ("Depressed signs showed", "banned:diagnostic:depressed"),
    ("An elderly user showed", "banned:demographic:elderly"),
    ("Wearing headphones, they showed", "banned:appearance:wearing"),
])
def test_banned_terms_rejected(unmasked, insert, reason):
    text = PLAIN.replace("These context signals carry", f"{insert} these context signals, which carry")
    assert reason in reasons(text, unmasked)


def test_short_break_is_not_appearance(unmasked):
    text = PLAIN.replace("drove the call", "with no short break drove the call")
    assert validate(text, unmasked).ok


def test_concepts_outside_top_drivers_rejected(sheet, unmasked):
    assert "concept:laughter" in reasons(MASKED.replace("drove the call", "and no laughter drove the call"), sheet)
    hidden_face = PLAIN.replace("These context signals", "With the face hidden, these context signals")
    assert "concept:face_visible" in reasons(hidden_face, unmasked)
    unmasked["face_visible"] = False
    unmasked["face_only"] = {"label": None, "confidence": None, "face_valence": None}
    assert validate(hidden_face, unmasked).ok


def test_ordinary_words_do_not_trigger_concepts():
    text = "In light of the rest of the window, they still worked late; a short summary of the call follows."
    assert text_concepts(text) == set()


def test_at_least_two_top_drivers_required(sheet):
    text = (
        "Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1, mainly from ambient noise at 72 dB "
        "in the room. Although the expression reads as positive, context corrected the facial mask."
    )
    assert reasons(text, sheet) == ["drivers:1<2"]


def test_display_string_counts_as_mention(sheet):
    _driver(sheet, "ambient_db").update(display="Room SPL", feature="spl", unit="")
    result = validate(MASKED.replace("Ambient noise (72 dB)", "Room SPL (72.3)"), sheet)
    assert result.ok, result.reasons
    assert result.mentioned_drivers[0] == "spl"


def test_extract_numbers_variants():
    values = [(t.value, t.percent) for t in extract_numbers("1,200 lux, .88 share, 3,4 counts, 93% seen, 4 pm, Person #7")]
    assert values == [(1200.0, False), (0.88, False), (3.0, False), (4.0, False), (93.0, True), (16.0, False)]
    clock = extract_numbers("at 3:30 p.m. and 15:30")
    assert [round(t.value, 4) for t in clock] == [15.5, 15.5]
    assert extract_numbers("-0.07")[0].value == 0.07


def test_allowed_numbers_cover_sheet(sheet):
    allowed = allowed_numbers(sheet)
    values = {n.value for n in allowed}
    assert {0.88, 88.0, 72.3, 4.2, 0.62, 0.71, 0.93, 15.0, 1.0, 123.0} <= values
    assert all(number_allowed(t, allowed) for t in extract_numbers("88% 0.88 72 4.2 93% 71% 15 3 0"))
