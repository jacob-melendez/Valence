from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from valence.contracts import STATE_SETS, FactSheet
from valence.narrator.build_lora_data import (
    P1_STATES,
    PRODUCTION,
    build_splits,
    cell_of,
    check_disjoint,
    is_production,
    load_jsonl,
    rows_for,
    synthetic_sheets,
)
from valence.narrator.engine import Narrator, resolve_style
from valence.narrator.fake_fact_sheets import make_batch
from valence.narrator.grammar import (
    TrigramDeduper,
    facts_of,
    realize,
    realize_candidate,
    realize_many,
    sheet_rng,
    trigrams,
)
from valence.narrator.lexicon import CONCEPT_RES, driver_concept, mask_hits
from valence.narrator.prompt import (
    avoid_words,
    base_messages,
    build_messages,
    build_prompt,
    prompt_drivers,
    prompt_prefix,
    render_chat,
    render_fact_sheet,
    system_prompt,
    training_messages,
)
from valence.narrator.schema import face_hidden, sheet_states
from valence.narrator.templates import render_template
from valence.narrator.validator import count_words, extract_numbers, sentence_spans, validate

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/llm/Qwen3-1.7B-4bit"
HAS_TOKENIZER = (MODEL_DIR / "tokenizer_config.json").exists()
HAS_PRODUCTION = PRODUCTION.exists()
DATA_DIR = ROOT / "data/narrator"


def _production() -> list[dict]:
    return load_jsonl(PRODUCTION) if HAS_PRODUCTION else []


@pytest.fixture(scope="module")
def corpus() -> list[tuple[dict, str]]:
    """(sheet, target) over every production sheet plus synthetic sheets of both state sets."""
    sheets = _production() + make_batch(600, seed=41, states="core") + make_batch(600, seed=42, states="all")
    return [(sheet, realize(sheet, sheet_rng(sheet, 0))) for sheet in sheets]


def _blank_core_strings(text: str, sheet: dict) -> str:
    """The text with verbatim driver displays and the label removed (core strings, correct by construction)."""
    out = text
    for d in prompt_drivers(sheet):
        out = out.replace(str(d["display"]), " ")
    return out.replace(str(sheet["label_display"]), " ")


# --- faithfulness -----------------------------------------------------------------------------


def test_every_target_passes_the_validator_for_both_state_sets(corpus):
    n_production = sum(1 for line in PRODUCTION.open() if line.strip()) if HAS_PRODUCTION else 0
    assert len(corpus) >= 1200 + n_production
    assert {name for sheet, _ in corpus for name in [("all" if len(sheet_states(sheet)) == 6 else "core")]} == {
        "core", "all"}
    failures = [(s["event_id"], validate(t, s).reasons) for s, t in corpus if not validate(t, s).ok]
    assert not failures, failures[:5]
    templates = sum(t == render_template(s) for s, t in corpus)
    assert templates <= len(corpus) // 200  # the grammar itself, not the fallback, writes them


def test_targets_follow_the_prompt_shape(corpus):
    for sheet, text in corpus:
        assert 30 <= count_words(text) <= 72, text
        assert 2 <= len(sentence_spans(text)) <= 3, text
        assert sheet["label_display"] in text and sheet["confidence_display"] in text


def test_targets_only_quote_numbers_the_prompt_shows(corpus):
    """Every number in a target appears in the rendered prompt (the model can always copy it)."""
    for sheet, text in corpus:
        prompt = render_fact_sheet(sheet)
        shown = {t.raw for t in extract_numbers(prompt, words=False)}
        quoted = {t.raw for t in extract_numbers(text, words=False)}
        assert quoted <= shown, (quoted - shown, text)


def test_mask_language_only_with_a_visible_mask_override(corpus):
    for sheet, text in corpus:
        facts = facts_of(sheet)
        free = _blank_core_strings(text, sheet)
        if facts.mask:
            assert mask_hits(free) or "face" in free.lower(), text
            assert facts.face_name in text
        else:
            assert not mask_hits(free), text
            assert "face-only model" not in free and "face alone" not in free


def test_occlusion_clause_exactly_when_the_face_is_hidden(corpus):
    seen_hidden = 0
    for sheet, text in corpus:
        mentions_hidden = bool(CONCEPT_RES["face_visible"].search(_blank_core_strings(text, sheet)))
        assert mentions_hidden == face_hidden(sheet), text
        seen_hidden += face_hidden(sheet)
    assert seen_hidden > 300


def test_upbeat_for_one_person_and_room_drivers_as_the_room(corpus):
    upbeat = [(s, t) for s, t in corpus if s["label_display"] == "Upbeat"]
    assert upbeat and all("Upbeat" in t and "ollaborat" not in t for _, t in upbeat)
    for sheet, text in corpus:
        lower = text.lower()
        assert "their noise" not in lower and "their light" not in lower


def test_contrast_words_only_for_opposing_drivers(corpus):
    """The validator enforces this per mention; here: a contrast appears only on sheets that list an
    opposing driver (or the face-only reading with a mask override)."""
    for sheet, text in corpus:
        facts = facts_of(sheet)
        free = _blank_core_strings(text, sheet).lower()
        if any(w in free for w in ("although", "despite", "pulled the other way", "argued against")):
            assert facts.opposing or facts.mask, text


def test_at_least_two_key_drivers_are_cited(corpus):
    for sheet, text in corpus[:800]:
        key = facts_of(sheet).key
        mentioned = set(validate(text, sheet).mentioned_drivers)
        assert len({str(d["feature"]) for d in key} & mentioned) >= min(2, len(key)), text


# --- style the adapter must not learn (review findings) ----------------------------------------


_COPULA_GERUND = re.compile(
    r"\b(?:signals were|call are|evidence includes|model weighed|cues were|counterweight was|even with|With|Given"
    r"|Weighing) being\b")
_FACE_WORDS = re.compile(r"\b(?:face|facial|smil\w*|expressions?|valence|mask\w*)\b", re.IGNORECASE)
_OOD_NOTE = re.compile(r"training range|saw in training|unusual for the model", re.IGNORECASE)


def test_targets_never_use_an_avoid_word_or_a_raw_feature_name(corpus):
    """Not even inside a copied display: the prompt's avoid line and the target must agree."""
    for sheet, text in corpus:
        lower = text.lower()
        for word in avoid_words(sheet):
            assert not re.search(rf"\b{word}", lower), (word, text)
        assert not re.search(r"\b[a-z]+_[a-z_]+ = ", text), text


def test_no_face_words_without_a_face_line(corpus):
    """Rule 7: with no face line in the prompt the target never mentions the face, so face-expression
    key drivers are skipped, unless fewer than two non-face key drivers exist (6 of ~1,600 sheets)."""
    checked = 0
    for sheet, text in corpus:
        facts = facts_of(sheet)
        plain = [d for d in facts.key if driver_concept(d) not in ("face_valence", "face_visible")]
        if not facts.mask and not face_hidden(sheet) and len(plain) >= 2:
            assert not _FACE_WORDS.search(text.replace(str(sheet["label_display"]), " ")), text
            checked += 1
    assert checked > 1000


def test_phrasing_guards(corpus):
    for sheet, text in corpus:
        facts = facts_of(sheet)
        assert not _COPULA_GERUND.search(text), text
        for match in re.finditer(r"\b(?:repeated|frequent) [a-z ]+\((\d+) ", text):
            assert int(match.group(1)) >= 2, text
        assert "busy with 2 " not in text, text
        assert all(sentence.count(";") <= 1 for sentence in re.split(r"(?<=[.!?]) ", text)), text
        assert len(_OOD_NOTE.findall(text.replace("unusual for the model, outside its training range", "training range"))) \
            == int(facts.ood), text
        assert not re.search(r"the room (?:had|showed) [^.;]*\band the room (?:had|showed)\b", text), text
        if facts.hidden and facts.visibility and int(facts.visibility[:-1]) >= 50:
            assert "mostly hidden" not in text and "not visible (" not in text, text
        if facts.face_agrees:
            assert not re.search(r"corrected|overr|set aside|past th", text), text


def test_agreeing_face_reading_is_not_called_a_correction():
    sheet = next(s for s in _production() + make_batch(800, seed=5)
                 if s.get("mask_override") and not face_hidden(s) and s["face_only"]["label"] == s["label"]) \
        if HAS_PRODUCTION else None
    if sheet is None:
        pytest.skip("no agreeing mask sheet")
    facts = facts_of(sheet)
    assert facts.face_agrees
    texts = {realize(sheet, random.Random(i)) for i in range(20)}
    assert all(facts.face_name in t and not re.search(r"corrected|overr", t) for t in texts)


# --- determinism and dedupe -------------------------------------------------------------------


def test_realize_is_deterministic_per_seed():
    sheets = make_batch(30, seed=7) + make_batch(30, seed=8, states="all")
    first = [realize(s, random.Random(5)) for s in sheets]
    assert first == [realize(s, random.Random(5)) for s in sheets]
    assert realize_many(sheets, seed=3) == realize_many(sheets, seed=3)
    assert realize_many(sheets, seed=3)[0] != realize_many(sheets, seed=4)[0]


def test_candidates_vary():
    sheet = make_batch(1, seed=9)[0]
    rng = random.Random(0)
    assert len({realize_candidate(sheet, rng) for _ in range(40)}) >= 30


def test_trigram_deduper_rejects_near_duplicates():
    dedupe = TrigramDeduper(max_overlap=0.5)
    text = "Model predicted Focused (80%) for Person #1. The main signals were moving normally (motion 0.08)."
    assert dedupe.overlap(text) == 0.0 and dedupe.accepts(text)
    dedupe.add(text)
    assert len(dedupe) == 1 and dedupe.overlap(text) == 1.0 and not dedupe.accepts(text)
    near = text.replace("80%", "81%")
    assert dedupe.overlap(near) > 0.5 and not dedupe.accepts(near)
    other = "With the face hidden, Exhausted is a tentative call for Person #2 (55%), slumped 71% of the last minute."
    assert dedupe.accepts(other)
    assert trigrams("a b c d") == {("a", "b", "c"), ("b", "c", "d")}


def test_realize_many_keeps_overlap_under_the_threshold():
    sheets = make_batch(240, seed=11) + make_batch(120, seed=12, states="all")
    targets, stats = realize_many(sheets, seed=0, max_overlap=0.4)
    for sheet, text in zip(sheets, targets, strict=True):
        assert validate(text, sheet).ok
    check = TrigramDeduper(1.0)
    overlaps = []
    for text in targets:
        overlaps.append(check.overlap(text))
        check.add(text)
    assert max(overlaps) <= 0.4 + 1e-9 or stats["dedupe_fallback"] > 0
    loose, _ = realize_many(sheets, seed=0, max_overlap=None)
    loose_check, loose_overlaps = TrigramDeduper(1.0), []
    for text in loose:
        loose_overlaps.append(loose_check.overlap(text))
        loose_check.add(text)
    assert max(overlaps) <= max(loose_overlaps)


# --- splits -----------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PRODUCTION, reason="production fact sheets not exported")
def test_splits_are_disjoint_balanced_and_mostly_production_in_test():
    splits = build_splits(_production(), synthetic_sheets(0), seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"train": 1500, "valid": 150, "test": 300}
    ids = [{s["event_id"] for s in v} for v in splits.values()]
    assert all(len(a & b) == 0 for i, a in enumerate(ids) for b in ids[i + 1:])
    assert sum(len(i) for i in ids) == 1950  # no duplicates inside a split either
    check_disjoint(splits)
    assert sum(is_production(s) for s in splits["test"]) >= 0.75 * 300
    for name in ("train", "test"):
        labels = {s["label"] for s in splits[name]}
        assert set(P1_STATES) <= labels, name
    train_cells = {cell_of(s) for s in splits["train"]}
    for state_set, states in STATE_SETS.items():
        for label in states:
            for case in ("agree", "mask", "hidden"):
                assert (state_set, label, case) in train_cells
    for sheet in splits["train"][:50] + splits["test"][:50]:
        FactSheet.model_validate(sheet)


def test_synthetic_ids_never_collide():
    sheets = synthetic_sheets(0)
    ids = [s["event_id"] for s in sheets]
    assert len(ids) == len(set(ids)) and all(i.startswith(("syn-core-", "syn-all-")) for i in ids)
    with pytest.raises(ValueError):
        check_disjoint({"train": sheets[:2], "test": sheets[1:3]})


# --- compact prompt: training rows render exactly like inference ---------------------------------


def test_training_messages_are_the_compact_inference_messages():
    for sheet in make_batch(10, seed=13) + make_batch(10, seed=14, states="all"):
        target = realize(sheet, sheet_rng(sheet, 0))
        row = training_messages(sheet, target)
        assert row[:-1] == build_messages(sheet, style="compact")
        assert row[-1] == {"role": "assistant", "content": target}
        assert [m["role"] for m in row] == ["system", "user", "assistant"]
        assert row[0]["content"] == system_prompt(sheet_states(sheet))
        assert row[1]["content"] == render_fact_sheet(sheet)
    assert base_messages(STATE_SETS["core"], "compact") == base_messages(STATE_SETS["core"])[:1]
    with pytest.raises(ValueError):
        build_messages(make_batch(1, seed=1)[0], style="zero_shot")


def test_rows_for_writes_the_sheet_only_in_test():
    sheets = make_batch(3, seed=15)
    targets = [realize(s, sheet_rng(s, 0)) for s in sheets]
    assert all(set(r) == {"messages"} for r in rows_for("train", sheets, targets))
    assert all(r["sheet"] == s for r, s in zip(rows_for("test", sheets, targets), sheets, strict=True))
    with pytest.raises(ValueError):
        rows_for("train", sheets[:1], ["Focused, maybe."])


def test_engine_picks_the_prompt_style_from_the_adapter(monkeypatch):
    monkeypatch.delenv("VALENCE_NARRATOR_PROMPT_STYLE", raising=False)
    monkeypatch.delenv("VALENCE_NARRATOR_ADAPTER", raising=False)
    assert resolve_style(None) == "few_shot" and resolve_style("adapters/x") == "compact"
    assert resolve_style("adapters/x", "few_shot") == "few_shot"
    monkeypatch.setenv("VALENCE_NARRATOR_PROMPT_STYLE", "compact")
    assert resolve_style(None) == "compact"
    monkeypatch.delenv("VALENCE_NARRATOR_PROMPT_STYLE")
    sheet = make_batch(1, seed=16)[0]
    for adapter, style in ((None, "few_shot"), ("adapters/narrator", "compact")):
        seen: list[list[dict]] = []

        def fake(messages, params, deadline, seen=seen):
            seen.append(list(messages))
            return realize(sheet, sheet_rng(sheet, 0))

        with Narrator(adapter_path=adapter, generate_fn=fake, reuse_texts=False) as narrator:
            out = narrator.narrate(sheet)
            assert narrator.prompt_style == style and narrator.health()["prompt_style"] == style
        assert out["source"] == "llm" and seen[0] == build_messages(sheet, style=style)
    with Narrator(adapter_path="adapters/narrator", prompt_style="few_shot", generate_fn=lambda *a: "") as narrator:
        assert narrator.prompt_style == "few_shot"


@pytest.fixture(scope="module")
def qwen():
    if not HAS_TOKENIZER:
        pytest.skip("Qwen3 tokenizer not downloaded")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))


def test_compact_training_row_renders_as_inference_prompt_plus_target(qwen):
    """mlx-lm renders a chat row with the tokenizer's template; its prompt part must be exactly the
    engine's compact prompt (thinking disabled), text and tokens, then the target and end of turn."""
    encode = lambda text: qwen.encode(text, add_special_tokens=False)  # noqa: E731
    for sheet in make_batch(40, seed=17) + make_batch(40, seed=18, states="all"):
        target = realize(sheet, sheet_rng(sheet, 0))
        trained = qwen.apply_chat_template(training_messages(sheet, target), tokenize=False)
        prompt = build_prompt(qwen, sheet, style="compact")
        assert prompt == render_chat(qwen, build_messages(sheet, style="compact"))
        assert trained == prompt + target + "<|im_end|>\n"
        full, head = encode(trained), encode(prompt)
        assert full[:len(head)] == head
        prefix = prompt_prefix(qwen, sheet_states(sheet), "compact")
        assert prompt.startswith(prefix) and encode(prompt)[:len(encode(prefix))] == encode(prefix)


def test_compact_prefix_is_much_shorter(qwen):
    encode = lambda text: qwen.encode(text, add_special_tokens=False)  # noqa: E731
    for states in STATE_SETS.values():
        few, compact = (len(encode(prompt_prefix(qwen, states, style))) for style in ("few_shot", "compact"))
        assert compact < 0.5 * few and compact <= 600


# --- the written dataset ----------------------------------------------------------------------


@pytest.mark.skipif(not (DATA_DIR / "test.jsonl").exists(), reason="LoRA data not built")
def test_written_dataset_matches_the_current_prompt_and_validator():
    rows = {name: load_jsonl(DATA_DIR / f"{name}.jsonl") for name in ("train", "valid", "test")}
    assert {k: len(v) for k, v in rows.items()} == {"train": 1500, "valid": 150, "test": 300}
    users = [{r["messages"][1]["content"] for r in v} for v in rows.values()]
    assert all(len(a & b) == 0 for i, a in enumerate(users) for b in users[i + 1:])
    for row in rows["test"]:
        sheet, target = row["sheet"], row["messages"][-1]["content"]
        assert row["messages"] == training_messages(sheet, target)
        assert validate(target, sheet).ok
    for row in rows["train"][:200] + rows["valid"]:
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] in {system_prompt(s) for s in STATE_SETS.values()}
    assert json.loads(json.dumps(rows["test"][0]))["sheet"]["event_id"]
