from __future__ import annotations

from pathlib import Path

import pytest

from valence.contracts import ALL_STATES, CORE_STATES, LAYERS, STATE_SETS, FactSheet
from valence.narrator.fake_fact_sheets import example_fact_sheet, make_batch
from valence.narrator.prompt import (
    SYSTEM_PROMPT,
    avoid_words,
    base_messages,
    build_messages,
    build_prompt,
    feedback_hint,
    few_shots,
    prompt_drivers,
    prompt_prefix,
    render_chat,
    render_fact_sheet,
    system_prompt,
)
from valence.narrator.schema import allowed_drivers, key_drivers
from valence.narrator.validator import validate

MODEL_DIR = Path(__file__).resolve().parents[2] / "models/llm/Qwen3-1.7B-4bit"
HAS_TOKENIZER = (MODEL_DIR / "tokenizer_config.json").exists()


class RecordingTokenizer:
    """Stands in for a HF tokenizer: joins turns and records the kwargs it was given."""

    def __init__(self, reject_thinking: bool = False, reject_system: bool = False) -> None:
        self.reject_thinking = reject_thinking
        self.reject_system = reject_system
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        if self.reject_thinking and "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        if self.reject_system and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


@pytest.fixture(scope="module")
def qwen():
    if not HAS_TOKENIZER:
        pytest.skip("Qwen3 tokenizer not downloaded")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))


@pytest.mark.parametrize("states", list(STATE_SETS.values()), ids=list(STATE_SETS))
def test_few_shots_are_contract_valid_and_pass_validation(states):
    shots = few_shots(states)
    assert len(shots) == 3
    for sheet, answer in shots:
        FactSheet.model_validate(sheet)
        assert tuple(sheet["probabilities"]) == states
        result = validate(answer, sheet)
        assert result.ok, result.reasons
        assert 30 <= result.word_count <= 70


def test_few_shots_cover_mask_agree_and_hidden():
    cases = [(s["mask_override"], s["face_visible"], s["face_only"]["label"] == s["label"]) for s, _ in few_shots()]
    assert cases == [(True, True, False), (False, True, True), (False, False, False)]
    assert len({s["label"] for s, _ in few_shots()}) == 3


def test_messages_structure():
    sheet = example_fact_sheet()
    messages = build_messages(sheet)
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1]["content"] == render_fact_sheet(sheet)
    for (shot, answer), user, assistant in zip(few_shots(), messages[1:-1:2], messages[2:-1:2], strict=True):
        assert user["content"] == render_fact_sheet(shot) and assistant["content"] == answer


def test_prefix_is_fixed_per_state_set():
    core = [build_messages(s)[:-1] for s in make_batch(4, seed=3)]
    assert all(m == core[0] for m in core) and core[0] == base_messages(CORE_STATES)
    full = build_messages(make_batch(1, seed=3, states="all")[0])
    assert full[:-1] == base_messages(ALL_STATES) and full[0]["content"] != SYSTEM_PROMPT


@pytest.mark.parametrize("states", list(STATE_SETS.values()), ids=list(STATE_SETS))
def test_system_prompt_lists_the_sheet_states_layers_and_rules(states):
    text = system_prompt(states)
    for state in states:
        assert {"Frustrated_Masked": "Masked Stress", "Collaborative_Happy": "Collaborative / Happy",
                "Overwhelmed": "Overwhelmed / Distressed", "Tense": "Tense / Visible Frustration"}.get(state,
                                                                                                    state) in text
    if states == CORE_STATES:
        assert "Tense" not in text and "Overwhelmed" not in text
    for layer in LAYERS:
        assert layer in text
    for phrase in ("Person #N", "gender, age, race, appearance", "health conditions", "mask override",
                   "the room had", "although", "30 to 70 words", "Never write a number", "Upbeat",
                   "at least two of the first three drivers", "level", "avoid"):
        assert phrase in text, phrase


def test_render_contains_display_strings_verbatim():
    sheet = example_fact_sheet()
    text = render_fact_sheet(sheet)
    assert "person: Person #1" in text and "state: Frustrated / Masked Stress, 88%" in text
    assert "face: mask override, face-only model read Collaborative / Happy (71%)" in text
    lines = text.splitlines()
    for i, d in enumerate(key_drivers(sheet), start=1):
        assert f"{i}. {d['scope']}/{d['level']}/for: {d['display']}" in lines
    assert "4. person/high/against: smiling expression (valence +0.62)" in lines
    for word in ("shap", "db_mean_15m", "0.31", "contribution"):
        assert word not in text


def test_rendered_drivers_are_exactly_the_allowed_set():
    for sheet in make_batch(200, seed=1, states="all"):
        shown = prompt_drivers(sheet)
        assert {d["display"] for d in shown} == {d["display"] for d in allowed_drivers(sheet)}
        key = list(dict.fromkeys(d["display"] for d in key_drivers(sheet)))
        assert [d["display"] for d in shown[:len(key)]] == key


def test_render_face_lines_and_avoid_words():
    hidden = next(s for s in make_batch(30, seed=4) if not s["face_visible"])
    assert "face: hidden (visible" in render_fact_sheet(hidden) and render_fact_sheet(hidden).count("say so") == 1
    agree = next(s for s in make_batch(30, seed=4) if s["face_visible"] and not s["mask_override"])
    assert "face:" not in render_fact_sheet(agree)
    assert {"smile", "mask"} <= set(avoid_words(agree))
    assert avoid_words(example_fact_sheet()) == ["focus", "exhausted"]
    solo = next(s for s in make_batch(200, seed=4) if s["n_people"] == 1 and s["label"] == "Collaborative_Happy")
    assert "collaborative" in avoid_words(solo) and "upbeat" not in avoid_words(solo)
    tense = next(s for s in make_batch(60, seed=4, states="all") if s["label"] == "Tense")
    assert "frustrated" in avoid_words(tense) and "tense" not in avoid_words(tense)


def test_render_is_compact():
    for sheet in make_batch(200, seed=1, states="all"):
        text = render_fact_sheet(sheet)
        assert len(text) < 700, text
        assert text.count("\n") <= 10


def test_feedback_hint_translates_reasons():
    sheet = example_fact_sheet()
    hint = feedback_hint(["number:57", "number:16:30", "other_state:Exhausted", "mask_term:smile",
                          "banned:gendered:she", "words:12", "outside_top5:laughter", "state_missing", "person:2",
                          "drivers:1<2", "sentences:4", "qualifier:noise:high", "contrast:db:despite",
                          "room_scope:db_mean_15m", "collaborative_solo"], sheet)
    assert hint.startswith("fix: ")
    for fragment in ("'57' and '16:30'", "'exhausted'", "'smile'", "'she'", "30 to 70 words", "listed drivers",
                     "Frustrated / Masked Stress", "Person #1", "first three drivers", "at most 3 sentences",
                     "high, low", "marked against", "as the room", "'collaborative'"):
        assert fragment in hint, fragment
    assert feedback_hint([], sheet) == ""


def test_retry_messages_carry_feedback_in_the_user_turn_only():
    sheet = example_fact_sheet()
    plain, retry = build_messages(sheet), build_messages(sheet, ["number:57"])
    assert plain[:-1] == retry[:-1]
    assert retry[-1]["content"].startswith(plain[-1]["content"])
    assert retry[-1]["content"].endswith("Do not write '57'; copy numbers exactly from the facts.")


def test_build_prompt_disables_thinking_and_falls_back():
    tok = RecordingTokenizer()
    assert build_prompt(tok, example_fact_sheet()).endswith("<assistant>")
    assert tok.calls == [{"enable_thinking": False, "add_generation_prompt": True, "tokenize": False}]
    tok = RecordingTokenizer(reject_thinking=True)
    assert SYSTEM_PROMPT in build_prompt(tok, example_fact_sheet()) and "enable_thinking" not in tok.calls[-1]
    tok = RecordingTokenizer(reject_thinking=True, reject_system=True)
    assert build_prompt(tok, example_fact_sheet()).startswith(f"<user>{SYSTEM_PROMPT}\n\n")


def test_prompt_prefix_is_a_prefix_of_every_prompt():
    tok = RecordingTokenizer()
    for name, states in STATE_SETS.items():
        prefix = prompt_prefix(tok, states)
        assert prefix.endswith("<user>")
        for sheet in make_batch(5, seed=2, states=name):
            assert build_prompt(tok, sheet).startswith(prefix)


# --- real Qwen3 tokenizer (no model weights loaded) ------------------------------------------------


def test_real_qwen3_template_has_one_empty_think_block(qwen):
    prompt = build_prompt(qwen, example_fact_sheet())
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert prompt.count("<think>") == 1
    prefix = prompt_prefix(qwen)
    assert prompt.startswith(prefix) and "<think>" not in prefix
    assert render_chat(qwen, build_messages(example_fact_sheet())) == prompt


@pytest.mark.parametrize("name", list(STATE_SETS))
def test_cached_prefix_tokens_equal_uncached_prompt_tokens(qwen, name):
    """The cached path feeds encode(prefix) from the cache and the rest live; together they must be
    token-for-token the uncached prompt."""
    encode = lambda text: qwen.encode(text, add_special_tokens=False)  # noqa: E731
    prefix_tokens = encode(prompt_prefix(qwen, STATE_SETS[name]))
    for sheet in make_batch(150, seed=5, states=name):
        for feedback in (None, ["number:57", "qualifier:noise:high"]):
            full = encode(build_prompt(qwen, sheet, feedback))
            assert full[:len(prefix_tokens)] == prefix_tokens
            assert prefix_tokens + full[len(prefix_tokens):] == full


def test_prompt_token_budget(qwen):
    encode = lambda text: qwen.encode(text, add_special_tokens=False)  # noqa: E731
    for name in STATE_SETS:
        counts = sorted(len(encode(render_fact_sheet(s))) for s in make_batch(200, seed=6, states=name))
        assert counts[len(counts) // 2] <= 170 and counts[-1] <= 200, counts[-5:]
        assert len(encode(prompt_prefix(qwen, STATE_SETS[name]))) <= 1400
