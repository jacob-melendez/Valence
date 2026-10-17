from __future__ import annotations

import random

import pytest

from valence.narrator import templates
from valence.narrator.fake_fact_sheets import example_fact_sheet, make_batch
from valence.narrator.lexicon import driver_concepts, mask_hits, remove_verbatim, text_concepts
from valence.narrator.schema import format_value, parse_fact_sheet, top_drivers
from valence.narrator.templates import N_VARIANTS, render_template, render_variant
from valence.narrator.validator import count_words, validate

SHEETS = make_batch(500, seed=0)


def test_six_variants():
    assert N_VARIANTS == 6


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_every_variant_passes_validation_on_fake_sheets(variant):
    failures = []
    for sheet in SHEETS:
        text = render_variant(sheet, variant)
        result = validate(text, sheet)
        if not result.ok or not 25 <= result.word_count <= 80:
            failures.append((sheet["event_id"], result.reasons, result.word_count, text))
    assert not failures, failures[:3]


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_every_variant_passes_on_example_sheet(variant):
    sheet = example_fact_sheet()
    text = render_variant(sheet, variant)
    result = validate(text, sheet)
    assert result.ok, (result.reasons, text)
    assert "Frustrated (Masked)" in text and "88%" in text
    assert "corrected the facial mask" in text
    assert result.mentioned_drivers == ["ambient_db", "fist_onsets_15m", "hours_since_break"]


def test_example_variant_zero_reads_like_the_prd():
    text = render_variant(example_fact_sheet(), 0)
    assert text.startswith("Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1.")
    for fragment in ("ambient noise (72 dB)", "clenched-fist onsets (3)", "time since last break (4.2 hours)",
                     "facial valence 0.62", "'Collaborative (Happy)'"):
        assert fragment in text


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_clauses_appear_only_when_warranted(variant):
    for sheet in SHEETS[:200]:
        text = render_variant(sheet, variant)
        assert bool(mask_hits(remove_verbatim(text, sheet["label_display"]))) == sheet["mask_override"]
        hidden = not sheet["face_visible"]
        visible_driver = any("face_visible" in driver_concepts(d) for _, d in top_drivers(sheet, 3))
        if hidden:
            assert "face_visible" in text_concepts(text)
        elif not visible_driver:
            assert "face_visible" not in text_concepts(text)


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_variants_quote_label_confidence_and_top_drivers(variant):
    for sheet in SHEETS[:100]:
        text = render_variant(sheet, variant)
        assert sheet["label_display"] in text
        assert sheet["confidence_display"] in text
        for _, driver in top_drivers(sheet, 3):
            absorbed = (sheet["mask_override"] and "face_valence" in driver_concepts(driver)) or (
                not sheet["face_visible"] and "face_visible" in driver_concepts(driver))
            assert format_value(driver) in text or absorbed, (driver, text)


def test_variants_differ():
    sheet = SHEETS[0]
    assert len({render_variant(sheet, v) for v in range(N_VARIANTS)}) == N_VARIANTS


def test_selection_is_deterministic():
    sheet = SHEETS[1]
    assert render_template(sheet) == render_template(sheet)
    assert render_template(sheet, rng=random.Random(4)) == render_template(sheet, rng=random.Random(4))
    assert render_template(sheet, 3) == render_variant(sheet, 3)


def test_accepts_pydantic_model():
    sheet = example_fact_sheet()
    assert render_template(parse_fact_sheet(sheet), 2) == render_variant(sheet, 2)


def test_invalid_variant_rejected():
    with pytest.raises(ValueError):
        render_template(SHEETS[0], N_VARIANTS)


def test_verify_falls_through_to_next_variant(monkeypatch):
    broken = (lambda ctx: ["She was tired."],) + templates._VARIANTS[1:]
    monkeypatch.setattr(templates, "_VARIANTS", broken)
    sheet = example_fact_sheet()
    assert render_template(sheet, 0, verify=False) == "She was tired."
    assert render_template(sheet, 0) == render_variant(sheet, 1)


def test_face_driver_in_top_three_is_cited_by_the_face_clause():
    sheet = example_fact_sheet()
    sheet["layers"][3]["drivers"][0]["shap"] = -0.29
    for variant in range(N_VARIANTS):
        text = render_variant(sheet, variant)
        assert validate(text, sheet).ok
        assert "facial valence 0.62" in text
        assert text.count("0.62") == 1


def test_hidden_face_with_visibility_driver():
    sheet = example_fact_sheet()
    sheet.update(face_visible=False, mask_override=False, face_only={"label": None, "confidence": None, "face_valence": None})
    sheet["data_quality"]["face_visible_frac"] = 0.12
    sheet["layers"][3]["drivers"] = [
        {"feature": "face_visible_frac", "display": "Face visible share", "value": 0.12, "unit": "%", "shap": 0.25}]
    for variant in range(N_VARIANTS):
        text = render_variant(sheet, variant)
        result = validate(text, sheet)
        assert result.ok, (result.reasons, text)
        assert "12%" in text and not mask_hits(remove_verbatim(text, sheet["label_display"]))


def test_opposing_driver_is_framed_as_counter_evidence():
    sheet = example_fact_sheet()
    sheet["layers"][1]["drivers"][0]["shap"] = -0.22
    for variant in range(N_VARIANTS):
        text = render_variant(sheet, variant)
        assert validate(text, sheet).ok
        assert any(phrase in text for phrase in ("other way", "against it", "pointed away"))


def test_word_counts_stay_compact():
    counts = [count_words(render_variant(sheet, v)) for sheet in SHEETS for v in range(N_VARIANTS)]
    assert 25 <= min(counts) and max(counts) <= 80


@pytest.mark.parametrize("driver,expected", [
    ({"value": 72.3, "unit": "dB"}, "72 dB"),
    ({"value": 4.06, "unit": "dB"}, "4.1 dB"),
    ({"value": 180.0, "unit": "lux"}, "180 lux"),
    ({"value": 1234.0, "unit": "lux"}, "1,234 lux"),
    ({"value": 4.2, "unit": "h"}, "4.2 hours"),
    ({"value": 1.0, "unit": "h"}, "1 hour"),
    ({"value": 4.0, "unit": "h"}, "4 hours"),
    ({"value": 35.0, "unit": "min"}, "35 minutes"),
    ({"value": 7.5, "unit": "min"}, "7.5 minutes"),
    ({"value": 3, "unit": "count"}, "3"),
    ({"value": 0.93, "unit": "%"}, "93%"),
    ({"value": 93.0, "unit": "%"}, "93%"),
    ({"value": 0.62, "unit": ""}, "0.62"),
    ({"value": -0.07, "unit": ""}, "-0.07"),
    ({"value": 2.43, "unit": "/min"}, "2.4 per minute"),
    ({"value": 812.4, "unit": "ppm"}, "812 ppm"),
    ({"value": None, "unit": "dB"}, "unavailable"),
    ({"feature": "time_of_day_h", "display": "Time of day", "value": 15.5, "unit": "h"}, "15:30"),
    ({"feature": "hour", "display": "Hour of day", "value": 9.999, "unit": "h"}, "10:00"),
])
def test_format_value(driver, expected):
    assert format_value(driver) == expected


def test_top_drivers_sorted_by_absolute_shap():
    ranked = top_drivers(example_fact_sheet(), None)
    assert [d["feature"] for _, d in ranked] == [
        "ambient_db", "fist_onsets_15m", "hours_since_break", "light_lux", "face_valence_mean", "compliments_15m"]
    assert ranked[0][0] == "Ambient Physics"
    assert len(top_drivers(example_fact_sheet(), 3)) == 3
