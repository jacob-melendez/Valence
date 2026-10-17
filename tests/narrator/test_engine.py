from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from valence.narrator.engine import (
    SCHEDULE,
    Generation,
    Narrator,
    SamplingParams,
    TemplateOnlyNarrator,
    complete_sentences,
    event_seed,
    salvage,
)
from valence.narrator.fake_fact_sheets import example_fact_sheet, make_batch
from valence.narrator.prompt import FEW_SHOTS
from valence.narrator.templates import render_template
from valence.narrator.validator import validate

SHEET = example_fact_sheet()
GOOD = (
    "Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1. While they show a smiling "
    "expression (facial valence 0.62), the system detected 3 clenched-fist onsets, ambient noise at 72 dB "
    "and 4.2 hours since their last break. Context corrected the facial mask."
)
BAD = "She seems exhausted, with noise at 57 dB."


class FakeLLM:
    """Scripted generate_fn: returns the queued outputs in order and records every call."""

    def __init__(self, *outputs: str | Generation, delay_s: float = 0.0) -> None:
        self.outputs = list(outputs)
        self.delay_s = delay_s
        self.calls: list[tuple[list[dict], SamplingParams, float]] = []
        self.threads: set[int] = set()

    def __call__(self, messages, params, deadline):
        self.calls.append((list(messages), params, deadline))
        self.threads.add(threading.get_ident())
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.outputs[min(len(self.calls), len(self.outputs)) - 1]


@pytest.fixture
def make_narrator():
    made: list[Narrator] = []

    def factory(*outputs, **kwargs) -> tuple[Narrator, FakeLLM]:
        delay = kwargs.pop("delay_s", 0.0)
        fake = FakeLLM(*outputs, delay_s=delay)
        narrator = Narrator(generate_fn=fake, **kwargs)
        made.append(narrator)
        return narrator, fake

    yield factory
    for narrator in made:
        narrator.close()


def test_good_answer_passes_validation():
    assert validate(GOOD, SHEET).ok


def test_first_try_pass_is_llm(make_narrator):
    narrator, fake = make_narrator(GOOD)
    out = narrator.narrate(SHEET)
    assert out["source"] == "llm"
    assert out["attempts"] == 1
    assert out["text"] == GOOD
    assert out["reasons"] == []
    assert set(out) >= {"text", "source", "latency_ms", "attempts", "reasons"}
    assert fake.calls[0][1].temp == 0.7 and fake.calls[0][1].top_p == 0.8 and fake.calls[0][1].top_k == 20
    assert fake.calls[0][1].repetition_penalty == 1.05 and fake.calls[0][1].max_tokens == 110


def test_retry_pass_is_llm_retry_and_carries_feedback(make_narrator):
    narrator, fake = make_narrator(BAD, GOOD)
    out = narrator.narrate(SHEET)
    assert out["source"] == "llm_retry"
    assert out["attempts"] == 2
    assert out["text"] == GOOD
    assert "number:57" in out["reasons"] and "banned:gendered:she" in out["reasons"]
    assert [call[1].temp for call in fake.calls] == [0.7, 0.4]
    first_user, retry_user = fake.calls[0][0][-1]["content"], fake.calls[1][0][-1]["content"]
    assert "Fix:" not in first_user
    assert "Fix:" in retry_user and "'57'" in retry_user and "'she'" in retry_user


def test_always_failing_falls_back_to_template_after_three_attempts(make_narrator):
    narrator, fake = make_narrator(BAD)
    out = narrator.narrate(SHEET)
    assert out["source"] == "template"
    assert out["attempts"] == 3
    assert out["text"] == render_template(SHEET)
    assert validate(out["text"], SHEET).ok
    assert [call[1].temp for call in fake.calls] == [p.temp for p in SCHEDULE] == [0.7, 0.4, 0.0]
    assert len(out["attempt_reasons"]) == 3


def test_template_fallback_is_deterministic_per_event(make_narrator):
    narrator, _ = make_narrator(BAD)
    first, second = narrator.narrate(SHEET), narrator.narrate(dict(SHEET))
    assert first["text"] == second["text"]


def test_slow_backend_exceeding_deadline_returns_template_in_time(make_narrator):
    narrator, fake = make_narrator(GOOD, delay_s=4.0)
    started = time.monotonic()
    out = narrator.narrate(SHEET)
    elapsed = time.monotonic() - started
    assert out["source"] == "template"
    assert elapsed < 2.65
    assert out["latency_ms"] < 2650
    assert out["attempts"] == 1
    assert "timeout" in out["reasons"]
    assert validate(out["text"], SHEET).ok


def test_backend_that_hits_the_deadline_mid_answer_is_not_shown(make_narrator):
    cut = Generation("Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1 because", "deadline")
    narrator, fake = make_narrator(cut, deadline_s=0.3)
    out = narrator.narrate(SHEET)
    assert out["source"] == "template"
    assert "timeout" in out["attempt_reasons"][0]


def test_retries_stop_when_too_little_time_remains(make_narrator):
    narrator, fake = make_narrator(BAD, delay_s=0.25, deadline_s=0.6, min_attempt_s=0.4)
    out = narrator.narrate(SHEET)
    assert out["source"] == "template"
    assert out["attempts"] == 1
    assert len(fake.calls) == 1


def test_backend_exception_falls_back_to_template(make_narrator):
    def broken(messages, params, deadline):
        raise RuntimeError("metal exploded")

    narrator = Narrator(generate_fn=broken)
    try:
        out = narrator.narrate(SHEET)
    finally:
        narrator.close()
    assert out["source"] == "template" and out["text"] == render_template(SHEET)


def test_truncated_answer_is_salvaged_to_complete_sentences(make_narrator):
    tail = " Ambient noise kept rising while"
    narrator, _ = make_narrator(Generation(GOOD + tail, "length", 110))
    out = narrator.narrate(SHEET)
    assert out["source"] == "llm"
    assert out["text"] == GOOD
    assert out["salvaged"] is True
    assert "length" in out["attempt_reasons"][0]


def test_salvage_drops_a_bad_trailing_sentence_only():
    result = salvage(GOOD + " She looked exhausted at 99 dB.", SHEET)
    assert result is not None and result.text == GOOD
    assert salvage(BAD, SHEET) is None


def test_complete_sentences():
    assert complete_sentences("One is 4.2 hours. Two ends here! Three is cut") == ["One is 4.2 hours.", "Two ends here!"]
    assert complete_sentences("A. B. C. D.") == ["A.", "B.", "C."]


def test_llm_output_is_cleaned_before_it_is_shown(make_narrator):
    narrator, _ = make_narrator(f"<think>\n\n</think>\n\nExplanation: **{GOOD}**")
    out = narrator.narrate(SHEET)
    assert out["source"] == "llm" and out["text"] == GOOD


def test_all_generation_runs_on_one_worker_thread(make_narrator):
    narrator, fake = make_narrator(GOOD)
    sheets = make_batch(6, seed=5)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(narrator.narrate, sheets))
    assert len(fake.threads) == 1
    assert threading.get_ident() not in fake.threads


def test_seed_is_deterministic_per_event_and_attempt(make_narrator):
    narrator, fake = make_narrator(BAD)
    narrator.narrate(SHEET)
    seeds = [call[1].seed for call in fake.calls]
    assert seeds == [event_seed("evt_000123", i) for i in range(3)]
    assert len(set(seeds)) == 3


def test_template_only_narrator():
    narrator = TemplateOnlyNarrator()
    narrator.load()
    out = narrator.narrate(SHEET)
    assert out["source"] == "template" and out["attempts"] == 0
    health = narrator.health()
    assert health["mode"] == "template" and health["model"] is None and health["warm"] is True
    assert health["by_source"] == {"llm": 0, "llm_retry": 0, "template": 1}
    narrator.close()


def test_unloaded_narrator_without_model_serves_templates(tmp_path):
    narrator = Narrator(model_path=str(tmp_path / "missing-model"))
    with pytest.raises(FileNotFoundError):
        narrator.load()
    out = narrator.narrate(SHEET)
    health = narrator.health()
    narrator.close()
    assert out["source"] == "template"
    assert health["warm"] is False and "FileNotFoundError" in health["error"]


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("VALENCE_NARRATOR_MODEL", "models/llm/Other")
    monkeypatch.setenv("VALENCE_NARRATOR_ADAPTER", "adapters/v1")
    narrator = Narrator()
    assert narrator.model_path == "models/llm/Other" and narrator.adapter_path == "adapters/v1"
    assert narrator.deadline_s == 2.5 and narrator.max_tokens == 110
    narrator.close()
    monkeypatch.delenv("VALENCE_NARRATOR_MODEL")
    monkeypatch.delenv("VALENCE_NARRATOR_ADAPTER")
    narrator = Narrator()
    assert narrator.model_path == "models/llm/Qwen3-1.7B-4bit" and narrator.adapter_path is None
    narrator.close()


def test_health_counts_requests_by_source(make_narrator):
    narrator, _ = make_narrator(GOOD)
    narrator.load()
    narrator.narrate(SHEET)
    narrator.narrate(FEW_SHOTS[2][0])
    health = narrator.health()
    assert health["warm"] is True and health["requests"] == 2
    assert health["by_source"]["llm"] == 1 and health["by_source"]["template"] == 1


def test_logs_carry_no_narration_text_or_sheet_facts(make_narrator, caplog):
    narrator, _ = make_narrator(BAD, GOOD)
    with caplog.at_level(logging.DEBUG, logger="valence.narrator"):
        out = narrator.narrate(SHEET)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "evt_000123" in logged and "llm_retry" in logged
    for secret in (out["text"], "Ambient noise", "Frustrated (Masked)", "smiling", "She seems"):
        assert secret not in logged
    info = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert all("number:" not in message for message in info)
