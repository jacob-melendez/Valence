from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from valence.contracts import FactSheet

from valence.narrator import templates
from valence.narrator.fake_fact_sheets import example_fact_sheet, make_batch
from valence.narrator.lexicon import MASK_RE, text_concepts
from valence.narrator.schema import key_drivers
from valence.narrator.templates import N_VARIANTS, minimal_text, render_template, render_variant
from valence.narrator.validator import count_words, split_sentences, validate

SHEETS = {name: make_batch(1000, seed=11, states=name) for name in ("core", "all")}


def _without(text: str, sheet: dict) -> str:
    """Text with the verbatim core strings (label_display, driver displays) blanked out."""
    for phrase in [sheet["label_display"], *(d["display"] for layer in sheet["layers"] for d in layer["drivers"])]:
        text = text.replace(phrase, " ")
    return text


def test_at_least_six_variants():
    assert N_VARIANTS >= 6


@pytest.mark.parametrize("states", ["core", "all"])
@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_every_variant_passes_validation_on_generated_sheets(states, variant):
    failures = []
    for sheet in SHEETS[states]:
        text = render_variant(sheet, variant)
        result = validate(text, sheet)
        if not result.ok:
            failures.append((sheet["event_id"], result.reasons, text))
    assert not failures, failures[:3]


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_example_sheet(variant):
    sheet = example_fact_sheet()
    text = render_variant(sheet, variant)
    assert validate(text, sheet).ok
    assert "Frustrated / Masked Stress" in text and "88%" in text
    assert "corrected the facial mask" in text
    for d in key_drivers(sheet):
        assert d["display"] in text


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_display_strings_are_copied_verbatim(variant):
    for sheet in SHEETS["all"][:200]:
        text = render_variant(sheet, variant)
        assert sheet["label_display"] in text and sheet["confidence_display"] in text
        cited = [d["display"] for d in key_drivers(sheet) if d["display"] in text]
        assert len(set(cited)) >= min(2, len({d["display"] for d in key_drivers(sheet)}))


@pytest.mark.parametrize("variant", range(N_VARIANTS))
def test_clauses_appear_only_when_warranted(variant):
    for sheet in SHEETS["core"][:300] + SHEETS["all"][:300]:
        text = render_variant(sheet, variant)
        free = _without(text, sheet)
        assert bool(MASK_RE.search(free)) == (sheet["mask_override"] and sheet["face_visible"]), text
        assert ("face_visible" in text_concepts(free)) == (not sheet["face_visible"]), text
        contrast = any(w in free.lower() for w in ("although", "the other way", "against it"))
        if contrast:
            assert any(d["effect"] == "opposes" and d["display"] in text for layer in sheet["layers"]
                       for d in layer["drivers"])
        if sheet["data_quality"]["ood_features"] and "training range" not in text:
            assert count_words(text) > templates.WORD_BUDGET - 12


def test_upbeat_for_one_person_and_no_collaborative():
    solos = [s for s in SHEETS["all"] if s["n_people"] == 1 and s["label"] == "Collaborative_Happy"]
    assert solos
    for sheet in solos:
        assert sheet["label_display"] == "Upbeat"
        for variant in range(N_VARIANTS):
            assert "ollaborat" not in render_variant(sheet, variant)


def test_room_drivers_are_phrased_as_the_room():
    for sheet in SHEETS["core"][:200]:
        for variant in range(N_VARIANTS):
            text = render_variant(sheet, variant)
            for d in key_drivers(sheet):
                if d["scope"] == "room" and d["display"] in text:
                    sentence = next(x for x in split_sentences(text) if d["display"] in x)
                    assert "room" in sentence[:sentence.index(d["display"])].lower(), (d["display"], text)


def test_variants_differ_and_selection_is_deterministic():
    sheet = SHEETS["core"][0]
    assert len({render_variant(sheet, v) for v in range(N_VARIANTS)}) == N_VARIANTS
    assert render_template(sheet) == render_template(dict(sheet))
    assert render_template(sheet, rng=random.Random(4)) == render_template(sheet, rng=random.Random(4))
    assert render_template(sheet, 3) == render_variant(sheet, 3)
    with pytest.raises(ValueError):
        render_template(sheet, N_VARIANTS)


def test_verify_falls_through_to_next_variant(monkeypatch):
    broken = (lambda ctx: ["She was tired."],) + templates._VARIANTS[1:]
    monkeypatch.setattr(templates, "_VARIANTS", broken)
    sheet = example_fact_sheet()
    assert render_template(sheet, 0, verify=False) == "She was tired."
    assert render_template(sheet, 0) == render_variant(sheet, 1)


def test_minimal_text_is_the_last_resort(monkeypatch):
    monkeypatch.setattr(templates, "_VARIANTS", tuple(lambda ctx: ["She was tired."] for _ in range(N_VARIANTS)))
    sheet = example_fact_sheet()
    assert render_template(sheet) == minimal_text(sheet)
    assert validate(minimal_text(sheet), sheet).ok


def test_word_counts_stay_in_bounds():
    counts = [count_words(render_variant(s, v)) for s in SHEETS["all"][:300] for v in range(N_VARIANTS)]
    assert 20 <= min(counts) and max(counts) <= 90


EXPORTED = Path(__file__).resolve().parents[2] / "data/narrator/fact_sheets_sim.jsonl"


@pytest.mark.skipif(not EXPORTED.exists(), reason="core's exported fact sheets not present")
def test_every_variant_passes_on_the_cores_exported_sheets():
    """Real sheets from the served model (scripts/export_fact_sheets.py), including production quirks
    such as a latched mask override with a hidden face."""
    sheets = [json.loads(line) for line in EXPORTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = []
    for sheet in sheets[:1200]:
        FactSheet.model_validate(sheet)
        for variant in range(N_VARIANTS):
            result = validate(render_variant(sheet, variant), sheet)
            if not result.ok:
                failures.append((sheet["event_id"], variant, result.reasons))
    assert not failures, failures[:5]
