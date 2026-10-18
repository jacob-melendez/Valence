from __future__ import annotations

import copy

import pytest

from valence.contracts import FactSheet
from valence.narrator.fake_fact_sheets import example_fact_sheet
from valence.narrator.validator import (
    allowed_numbers,
    clean_text,
    extract_numbers,
    number_allowed,
    split_sentences,
    strip_think,
    validate,
)

# example_fact_sheet(): Frustrated_Masked 88%, Person #1, n_people 2, mask_override, face visible.
# key (top-3 supporting): db_mean_15m (room, high), fist_cnt_5m (person, high), hours_since_break (high)
# allowed adds: face_now (opposes, "smiling expression (valence +0.62)"), lux_now (room, low).
GOOD = (
    "Model predicted Frustrated / Masked Stress (88%) for Person #1. The room had average noise of 72 dB over "
    "15 minutes, and they showed 3 clenched-fist events in the last 5 minutes after 4.2 hours since the last "
    "break. Despite a smiling expression (valence +0.62), context corrected the facial mask."
)
PLAIN = (
    "Model predicted Frustrated (88%) for Person #1. The room had average noise of 72 dB over 15 minutes, and "
    "they showed 3 clenched-fist events in the last 5 minutes after 4.2 hours since the last break. "
    "These aggregate context signals carry the call."
)


@pytest.fixture()
def sheet() -> dict:
    return example_fact_sheet()


@pytest.fixture()
def unmasked() -> dict:
    """Same facts, but the face agrees, so no mask override (core states: label reads "Frustrated")."""
    s = example_fact_sheet()
    s.update(mask_override=False, label_display="Frustrated",
             face_only={"label": "Frustrated_Masked", "confidence": 0.66, "face_valence": -0.2})
    return s


def driver(sheet: dict, feature: str) -> dict:
    return next(d for layer in sheet["layers"] for d in layer["drivers"] if d["feature"] == feature)


def reasons(text: str, sheet: dict) -> list[str]:
    return validate(text, sheet).reasons


def test_reference_texts_pass(sheet, unmasked):
    for text, target in ((GOOD, sheet), (PLAIN, unmasked)):
        result = validate(text, target)
        assert result.ok, (result.reasons, text)
    assert validate(GOOD, FactSheet.model_validate(sheet)).ok


def test_result_reports_mentions(sheet):
    result = validate(GOOD, sheet)
    assert {"db_mean_15m", "fist_cnt_5m", "hours_since_break"} <= set(result.mentioned_drivers)
    assert result.sentence_count == 3 and 20 <= result.word_count <= 90


# --- PLAN.md adversarial strings --------------------------------------------------------------


def test_unknown_number_91_percent_rejected(sheet):
    assert reasons(GOOD.replace("88%", "91%"), sheet) == ["number:91%"]


def test_she_rejected(sheet):
    assert "banned:gendered:she" in reasons(GOOD.replace("they showed", "she showed"), sheet)


def test_depressed_rejected(sheet):
    assert "banned:diagnostic:depressed" in reasons(GOOD.replace("Despite a smiling", "Despite a depressed, smiling"),
                                                   sheet)


def test_high_noise_45_db_with_normal_level_rejected(unmasked):
    d = driver(unmasked, "db_mean_15m")
    d.update(value=45.0, display="average noise of 45 dB over 15 minutes", level="normal")
    text = PLAIN.replace("average noise of 72 dB over 15 minutes", "high noise (45 dB)")
    assert "qualifier:noise:high" in reasons(text, unmasked)
    assert validate(PLAIN.replace("72 dB", "45 dB"), unmasked).ok
    d["level"] = "high"
    assert "qualifier:noise:high" not in reasons(text, unmasked)


def test_mask_language_when_face_agrees_rejected(unmasked):
    text = PLAIN.replace("These aggregate context signals carry the call.",
                         "Despite the smile, context corrected the facial mask.")
    found = reasons(text, unmasked)
    assert "mask_term:smile" in found and "mask_term:mask" in found


def test_label_display_and_driver_displays_exempt_from_mask_terms(unmasked):
    unmasked["label_display"] = "Frustrated / Masked Stress"
    text = PLAIN.replace("Frustrated (88%)", "Frustrated / Masked Stress (88%)").replace(
        "These aggregate context signals carry the call.", "A smiling expression (valence +0.62) was also seen.")
    driver(unmasked, "face_now")["effect"] = "supports"
    assert validate(text, unmasked).ok, reasons(text, unmasked)


def test_stale_cached_text_rejected_when_numbers_change(sheet):
    assert validate(GOOD, sheet).ok
    newer = copy.deepcopy(sheet)
    driver(newer, "db_mean_15m").update(value=75.1, display="average noise of 75 dB over 15 minutes")
    driver(newer, "hours_since_break").update(value=4.6, display="4.6 hours since the last break")
    found = reasons(GOOD, newer)
    assert "number:72" in found and "number:4.2" in found


def test_tense_stems_are_claimed_before_frustrated(sheet):
    tense = copy.deepcopy(sheet)
    tense.update(label="Tense", label_display="Tense / Visible Frustration", mask_override=False,
                 probabilities={"Focused": 0.03, "Frustrated_Masked": 0.03, "Exhausted": 0.02,
                                "Collaborative_Happy": 0.02, "Overwhelmed": 0.02, "Tense": 0.88},
                 face_only={"label": "Tense", "confidence": 0.7, "face_valence": -0.5})
    driver(tense, "face_now").update(display="negative expression (valence -0.50)", value=-0.5, level="low")
    body = PLAIN.split(". ", 1)[1]
    for name in ("visibly frustrated", "Tense / Visible Frustration", "tense", "showing visible frustration"):
        text = f"Model predicted {name} (88%) for Person #1. {body}"
        assert validate(text, tense).ok, (name, reasons(text, tense))
    plain_frustrated = f"Model predicted frustrated (88%) for Person #1. {body}"
    assert reasons(plain_frustrated, tense) == ["state_missing", "other_state:Frustrated_Masked"]
    fm = copy.deepcopy(tense)
    fm.update(label="Frustrated_Masked", label_display="Frustrated / Masked Stress")
    assert "other_state:Tense" in reasons(f"Model predicted visibly frustrated (88%) for Person #1. {body}", fm)
    assert validate(f"Model predicted frustrated (88%) for Person #1. {body}", fm).ok


def test_collaborative_with_one_person_rejected(sheet):
    solo = copy.deepcopy(sheet)
    solo.update(label="Collaborative_Happy", label_display="Upbeat", n_people=1, mask_override=False,
                face_only={"label": "Collaborative_Happy", "confidence": 0.7, "face_valence": 0.6})
    body = PLAIN.split(". ", 1)[1]
    assert validate(f"Model predicted Upbeat (88%) for Person #1. {body}", solo).ok
    assert reasons(f"Model predicted Collaborative (88%) for Person #1. {body}", solo) == ["collaborative_solo"]
    solo["n_people"] = 2
    assert validate(f"Model predicted Collaborative (88%) for Person #1. {body}", solo).ok


@pytest.mark.parametrize("phrase", ["Their average noise of 72 dB over 15 minutes",
                                    "Person #1's average noise of 72 dB over 15 minutes",
                                    "Their noise of 72 dB"])
def test_room_driver_attributed_to_person_rejected(sheet, phrase):
    text = GOOD.replace("The room had average noise of 72 dB over 15 minutes", phrase)
    assert any(r.startswith("room_scope:") for r in reasons(text, sheet)), reasons(text, sheet)


def test_room_act_and_room_light_attribution_rejected(sheet):
    text = GOOD.replace("after 4.2 hours since the last break", "in their dim light (180 lux)")
    assert "room_scope:lux_now" in reasons(text, sheet)
    laugh = GOOD.replace("Despite a smiling expression", "They laughed little, and despite a smiling expression")
    assert "room_scope:act" in reasons(laugh, sheet)


# --- qualifiers and effect -------------------------------------------------------------------------


def test_contrast_only_for_opposing_drivers(sheet):
    driver(sheet, "n_people")["shap"] = -0.065  # rank 5: allowed, and lux_now drops out of the top-5
    assert "contrast:hours_since_break:despite" in reasons(GOOD.replace("after 4.2 hours", "despite 4.2 hours"), sheet)
    text = GOOD.replace("Despite a smiling expression (valence +0.62), context",
                        "Although 2 people present pulled the other way, context")
    assert validate(text, sheet).ok, reasons(text, sheet)
    sheet = example_fact_sheet()
    text = text.replace("2 people present pulled the other way", "dim light (180 lux) pulled the other way")
    found = reasons(text, sheet)
    assert "contrast:lux_now:although" in found and "contrast:lux_now:the other way" in found


def test_contrast_binds_within_its_clause(sheet):
    text = ("Although the confidence is 88%, Model predicted Frustrated / Masked Stress for Person #1, with the room "
            "at average noise of 72 dB over 15 minutes and 3 clenched-fist events in the last 5 minutes. Context "
            "corrected the facial mask.")
    assert validate(text, sheet).ok, reasons(text, sheet)


def test_support_word_on_opposing_driver_rejected(sheet):
    text = GOOD.replace("Despite a smiling expression (valence +0.62), context corrected the facial mask.",
                        "Context corrected the facial mask, which 2 people present supported.")
    assert "effect:n_people:supported" in reasons(text, sheet)


@pytest.mark.parametrize("phrase,ok", [
    ("loud average noise of 72 dB over 15 minutes", True),
    ("quiet average noise of 72 dB over 15 minutes", False),
    ("low noise of 72 dB", False),
    ("very high noise of 72 dB", True),
    ("normal noise of 72 dB", False),
])
def test_level_words_against_level(sheet, phrase, ok):
    text = GOOD.replace("average noise of 72 dB over 15 minutes", phrase)
    assert validate(text, sheet).ok is ok, reasons(text, sheet)


def test_high_confidence_is_not_a_signal_qualifier(sheet):
    driver(sheet, "db_mean_15m")["level"] = "normal"
    text = GOOD.replace("(88%) for Person #1", "with high confidence (88%) for Person #1")
    assert validate(text, sheet).ok, reasons(text, sheet)


def test_qualifier_binds_nearest_mention(sheet):
    text = GOOD.replace("The room had average noise of 72 dB over 15 minutes",
                        "The room had loud noise (72 dB) and dim light (180 lux)")
    assert validate(text, sheet).ok, reasons(text, sheet)
    text = text.replace("dim light", "bright light")
    assert "qualifier:light:bright" in reasons(text, sheet)


def test_qualifier_inside_display_is_trusted(sheet):
    text = GOOD.replace("after 4.2 hours since the last break", "in dim light (180 lux)")
    assert validate(text, sheet).ok, reasons(text, sheet)


# --- coverage --------------------------------------------------------------------------------------


def test_at_least_two_key_drivers_required(sheet):
    text = ("Model predicted Frustrated / Masked Stress (88%) for Person #1, mainly because the room had average "
            "noise of 72 dB over 15 minutes. Despite a smiling expression (valence +0.62), context corrected the "
            "facial mask.")
    assert reasons(text, sheet) == ["drivers:1<2"]


def test_concept_words_count_as_mentions(sheet):
    text = ("Model predicted Frustrated / Masked Stress (88%) for Person #1. The room was loud at 72 dB and they "
            "clenched their fists 3 times in 5 minutes. Despite the smile, context corrected the facial mask.")
    assert validate(text, sheet).ok, reasons(text, sheet)


def test_mentions_outside_top5_rejected(sheet):
    text = GOOD.replace("after 4.2 hours", "with no compliments in the room in the last 15 minutes after 4.2 hours")
    assert "outside_top5:compl_cnt_15m" in reasons(text, sheet)
    assert "outside_top5:laughter" in reasons(GOOD.replace("after 4.2 hours", "with no laughter after 4.2 hours"),
                                              sheet)
    assert "outside_top5:speech" in reasons(GOOD.replace("Despite a", "Amid conversation and despite a"), sheet)


def test_face_topics_allowed_only_when_the_sheet_raises_them(sheet, unmasked):
    hidden_clause = " The face is hidden, so the call relies on context."
    text = PLAIN.replace(" These aggregate context signals carry the call.", hidden_clause)
    assert "outside_top5:face_visible" in reasons(text, unmasked)
    unmasked.update(face_visible=False, face_only={"label": None, "confidence": None, "face_valence": None})
    assert validate(text, unmasked).ok, reasons(text, unmasked)


def test_other_state_rejected_except_mask_contrast(sheet, unmasked):
    exhausted = GOOD.replace("context corrected", "they look exhausted; context corrected")
    assert "other_state:Exhausted" in reasons(exhausted, sheet)
    contrast = GOOD.replace("context corrected", "read as Collaborative / Happy by the face, context corrected")
    assert validate(contrast, sheet).ok, reasons(contrast, sheet)
    plain = PLAIN.replace("These aggregate", "A happy face aside, these aggregate")
    assert "other_state:Collaborative_Happy" in reasons(plain, unmasked)


def test_state_must_be_named(sheet):
    assert "state_missing" in reasons(GOOD.replace("Frustrated / Masked Stress", "a state"), sheet)


def test_person_number_rules(sheet):
    assert "person:2" in reasons(GOOD.replace("for Person #1", "for Person #2"), sheet)
    assert "number:2" not in reasons(GOOD.replace("for Person #1", "for Person #2"), sheet)
    room = dict(sheet, person="Room")
    assert "person:1" in reasons(GOOD, room)


# --- numbers -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("quote", ["88%", "88 percent", "0.88", "88.0%", "0.9"])
def test_confidence_fraction_and_percent_match(sheet, quote):
    assert validate(GOOD.replace("88%", quote), sheet).ok


@pytest.mark.parametrize("quote", ["89%", "87%", "0.87", "88.4%"])
def test_confidence_off_by_more_than_rounding_rejected(sheet, quote):
    assert reasons(GOOD.replace("88%", quote), sheet) == [f"number:{quote}"]


@pytest.mark.parametrize("quote,ok", [("72", True), ("72.4", True), ("72.0", False), ("73", False), ("70", False)])
def test_numbers_round_to_the_texts_precision(sheet, quote, ok):
    result = validate(GOOD.replace("of 72 dB", f"of {quote} dB"), sheet)
    assert result.ok is ok, result.reasons


def test_display_numbers_and_percent_leaves(sheet):
    text = GOOD.replace("context corrected", "with the face visible 93% of the time and a 4% Focused share, context "
                                             "corrected")
    found = reasons(text, sheet)
    assert not any(r.startswith("number:") for r in found), found


def test_shap_values_are_not_quotable(sheet):
    assert "number:0.31" in reasons(GOOD.replace("(88%)", "(88%, weight 0.31)"), sheet)


def test_number_words(sheet):
    assert validate(GOOD.replace("3 clenched-fist", "three clenched-fist"), sheet).ok
    assert "number:seven" in reasons(GOOD.replace("3 clenched-fist", "seven clenched-fist"), sheet)


def test_clock_times_match_display(sheet):
    sheet["layers"][1]["drivers"].append({"feature": "tod_sin", "display": "time of day (15:30)", "value": -0.71,
                                          "unit": "", "shap": 0.2, "level": "normal", "effect": "supports",
                                          "scope": "person"})
    base = GOOD.replace("after 4.2 hours", "at {} after 4.2 hours")
    assert validate(base.format("15:30"), sheet).ok
    assert validate(base.format("3:30 pm"), sheet).ok
    assert "number:16:30" in reasons(base.format("16:30"), sheet)


def test_extract_numbers_variants():
    values = [(t.value, t.percent) for t in extract_numbers("1,200 lux, .88 share, 93% seen, 4 pm, Person #7, two")]
    assert values == [(1200.0, False), (0.88, False), (93.0, True), (16.0, False), (2.0, False)]
    assert [round(t.value, 4) for t in extract_numbers("at 3:30 p.m. and 15:30")] == [15.5, 15.5]
    assert extract_numbers("-0.07")[0].value == 0.07


def test_allowed_numbers_cover_sheet(sheet):
    values = {n.value for n in allowed_numbers(sheet)}
    assert {0.88, 88.0, 72.4, 72.0, 15.0, 4.2, 3.0, 180.0, 0.62, 0.71, 0.93, 2.0, 1.0} <= values
    assert 0.31 not in values and 0.27 not in values
    assert all(number_allowed(t, allowed_numbers(sheet)) for t in extract_numbers("88% 0.88 72 4.2 93% 71% 15 3 0"))


# --- length and cleaning -------------------------------------------------------------------------------


def test_too_many_sentences_rejected(sheet):
    assert "sentences:4" in reasons(GOOD + " This is a fourth sentence.", sheet)


def test_word_bounds(sheet):
    short = "Frustrated / Masked Stress (88%): average noise of 72 dB over 15 minutes, 4.2 hours since the last break."
    assert any(r.startswith("words:") for r in reasons(short, sheet))
    long = GOOD.replace("context corrected", "context " + "and the same pattern " * 12 + "corrected")
    assert any(r.startswith("words:") for r in reasons(long, sheet))


def test_sentence_splitting_handles_decimals_and_person_numbers():
    assert len(split_sentences("Noise hit 72.3 dB for Person #1. Breaks were 4.2 h apart, e.g. long. Light 180.")) == 3


def test_think_blocks_are_stripped(sheet):
    assert validate("<think>The user wants 57 words about Person #2.</think>\n" + GOOD, sheet).ok
    assert validate("reasoning about 57 things\n</think>\n\n" + GOOD, sheet).ok
    assert validate("<think>\n\n</think>\n\n" + GOOD, sheet).text == GOOD
    unterminated = validate("<think>still deciding what to say about " + GOOD, sheet)
    assert not unterminated.ok and unterminated.reasons == ["empty"]
    assert strip_think("Answer.<think>trailing") == "Answer."


def test_markdown_and_quotes_are_cleaned(sheet):
    result = validate('"**Model predicted** ' + GOOD[len("Model predicted "):] + '"', sheet)
    assert result.ok, result.reasons
    assert clean_text("## Audit\nExplanation: " + GOOD) == GOOD
