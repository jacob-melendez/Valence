from __future__ import annotations

import random
from pathlib import Path

import pytest

from valence.narrator.fake_fact_sheets import example_fact_sheet, make_batch, make_fact_sheet
from valence.narrator.prompt import (
    FEW_SHOTS,
    SYSTEM_PROMPT,
    avoid_words,
    build_messages,
    build_prompt,
    feedback_hint,
    prompt_prefix,
    render_chat,
    render_fact_sheet,
)
from valence.narrator.schema import LAYER_NAMES, STATE_DISPLAY, format_value, parse_fact_sheet, top_drivers
from valence.narrator.validator import validate

MODEL_DIR = Path(__file__).resolve().parents[2] / "models/llm/Qwen3-1.7B-4bit"


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
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + "<assistant>"


@pytest.mark.parametrize("index", range(len(FEW_SHOTS)))
def test_few_shot_answers_pass_validation(index):
    sheet, answer = FEW_SHOTS[index]
    parse_fact_sheet(sheet)
    result = validate(answer, sheet)
    assert result.ok, result.reasons
    assert 30 <= result.word_count <= 70


def test_few_shots_cover_mask_agree_and_hidden():
    cases = [(s["mask_override"], s["face_visible"], s["face_only"]["label"] == s["label"]) for s, _ in FEW_SHOTS]
    assert (True, True, False) in cases
    assert (False, True, True) in cases
    assert (False, False, False) in cases
    assert len({s["label"] for s, _ in FEW_SHOTS}) == 3
    assert len({s["person"] for s, _ in FEW_SHOTS}) == 3


def test_few_shot_answers_are_not_template_text():
    for sheet, answer in FEW_SHOTS:
        assert answer.startswith(("Model predicted", "Person #"))
        assert "'" + sheet["label_display"] + "'" in answer
        assert sheet["confidence_display"] in answer


def test_messages_structure():
    sheet = example_fact_sheet()
    messages = build_messages(sheet)
    roles = [m["role"] for m in messages]
    assert roles == ["system"] + ["user", "assistant"] * len(FEW_SHOTS) + ["user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1]["content"] == render_fact_sheet(sheet)
    for (shot, answer), user, assistant in zip(FEW_SHOTS, messages[1:-1:2], messages[2:-1:2], strict=True):
        assert user["content"] == render_fact_sheet(shot)
        assert assistant["content"] == answer
    assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in messages)


def test_prefix_is_identical_across_sheets():
    first, second = build_messages(make_batch(2, seed=3)[0]), build_messages(make_batch(2, seed=3)[1])
    assert first[:-1] == second[:-1]


def test_system_prompt_defines_states_layers_and_rules():
    for display in STATE_DISPLAY.values():
        assert display in SYSTEM_PROMPT
    for layer in LAYER_NAMES:
        assert layer in SYSTEM_PROMPT
    for phrase in ("Person #N", "gender, age, race, appearance", "diagnoses", "Mask override is true",
                   "relies on context", "30 to 70 words", "plain text", "at least two of the listed drivers",
                   "Never invent"):
        assert phrase in SYSTEM_PROMPT


def test_render_contains_display_strings_verbatim():
    sheet = example_fact_sheet()
    text = render_fact_sheet(sheet)
    assert "Person: Person #1" in text
    assert "'Frustrated (Masked)' at 88% confidence" in text
    for rank, (layer, driver) in enumerate(top_drivers(sheet, 3), start=1):
        assert f"{rank}. {driver['display']}: {format_value(driver)} ({layer}, supports)" in text
    assert "face-only baseline 'Collaborative (Happy)' at 71%, facial valence 0.62" in text
    assert "Mask override: true" in text
    assert "Window: 15 minutes" in text
    assert "corrected the facial mask" in text
    assert "ambient_db" not in text and "shap" not in text.lower()


def test_render_is_compact():
    for sheet in make_batch(200, seed=1):
        text = render_fact_sheet(sheet)
        assert len(text) < 750, text
        assert text.count("\n") <= 11


def test_render_marks_opposing_drivers_and_respects_effect_field():
    sheet = example_fact_sheet()
    sheet["layers"][2]["drivers"][0]["shap"] = -0.27
    assert "Clenched-fist onsets: 3 (Somatic & Social, argues against)" in render_fact_sheet(sheet)
    sheet["layers"][2]["drivers"][0].update(effect="supports", level="high", scope="room")
    assert "Clenched-fist onsets: 3, high (Somatic & Social, supports, room)" in render_fact_sheet(sheet)


def test_render_hidden_face():
    sheet = make_fact_sheet(random.Random(4), label="Exhausted", face_condition="hidden")
    text = render_fact_sheet(sheet)
    assert "Face: hidden" in text and "no face-only baseline" in text
    assert "relies on context alone" in text
    assert "Mask override: false" in text
    for word in ("smile", "mask", "despite", "happy", "collaborative", "focus", "frustrated"):
        assert word in avoid_words(sheet)


def test_render_visible_face_without_mask_never_names_the_baseline_state():
    rng = random.Random(7)
    sheet = make_fact_sheet(rng, label="Focused", face_condition="disagree")
    assert not sheet["mask_override"]
    text = render_fact_sheet(sheet)
    assert "face-only baseline differs" in text
    assert STATE_DISPLAY[sheet["face_only"]["label"]] not in text


def test_avoid_words_allow_the_mask_contrast_state():
    words = avoid_words(example_fact_sheet())
    assert words == ["focus", "exhausted"]


def test_render_accepts_pydantic_model():
    sheet = example_fact_sheet()
    assert render_fact_sheet(parse_fact_sheet(sheet)) == render_fact_sheet(sheet)


def test_feedback_hint_translates_reasons():
    sheet = example_fact_sheet()
    hint = feedback_hint(["number:57", "number:16:30", "other_state:Exhausted", "mask_term:smile",
                          "banned:gendered:she", "words:12", "concept:laughter", "state_missing",
                          "person:2", "drivers:1<2", "sentences:4"], sheet)
    assert hint.startswith("Fix: ")
    for fragment in ("'57' and '16:30'", "'exhausted'", "'smile'", "'she'", "30 to 70 words", "laughter",
                     "'Frustrated (Masked)'", "Person #1", "two of the listed drivers", "at most 3 sentences"):
        assert fragment in hint
    assert feedback_hint([], sheet) == ""


def test_retry_messages_carry_feedback_in_the_user_turn_only():
    sheet = example_fact_sheet()
    plain, retry = build_messages(sheet), build_messages(sheet, ["number:57"])
    assert plain[:-1] == retry[:-1]
    assert retry[-1]["content"].startswith(plain[-1]["content"])
    assert retry[-1]["content"].endswith("Do not write '57'; copy numbers exactly from the facts.")


def test_build_prompt_disables_thinking():
    tok = RecordingTokenizer()
    prompt = build_prompt(tok, example_fact_sheet())
    assert tok.calls == [{"enable_thinking": False, "add_generation_prompt": True, "tokenize": False}]
    assert prompt.endswith("<assistant>")


def test_build_prompt_falls_back_when_template_rejects_kwarg():
    tok = RecordingTokenizer(reject_thinking=True)
    prompt = build_prompt(tok, example_fact_sheet())
    assert "enable_thinking" not in tok.calls[-1]
    assert SYSTEM_PROMPT in prompt


def test_build_prompt_folds_system_prompt_when_unsupported():
    tok = RecordingTokenizer(reject_thinking=True, reject_system=True)
    prompt = build_prompt(tok, example_fact_sheet())
    assert prompt.startswith(f"<user>{SYSTEM_PROMPT}\n\n")


def test_prompt_prefix_is_a_prefix_of_every_prompt():
    tok = RecordingTokenizer()
    prefix = prompt_prefix(tok)
    assert prefix.endswith("<user>")
    for sheet in make_batch(5, seed=2):
        assert build_prompt(tok, sheet).startswith(prefix)


@pytest.mark.skipif(not (MODEL_DIR / "tokenizer_config.json").exists(), reason="Qwen3 tokenizer not downloaded")
def test_real_qwen3_template_has_empty_think_block_and_no_thinking_in_shots():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))
    prompt = build_prompt(tok, example_fact_sheet())
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert prompt.count("<think>") == 1
    prefix = prompt_prefix(tok)
    assert prompt.startswith(prefix) and "<think>" not in prefix
    assert render_chat(tok, build_messages(example_fact_sheet())) == prompt
