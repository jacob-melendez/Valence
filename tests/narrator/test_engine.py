from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from valence.contracts import NarrationResult
from valence.narrator.engine import (
    SCHEDULE,
    Generation,
    MLXBackend,
    Narrator,
    SamplingParams,
    TemplateOnlyNarrator,
    _Prefix,
    complete_sentences,
    event_seed,
    salvage,
    situation_key,
)
from valence.narrator.fake_fact_sheets import example_fact_sheet, make_batch
from valence.narrator.prompt import few_shots
from valence.narrator.templates import render_template
from valence.narrator.validator import validate

SHEET = example_fact_sheet()
GOOD = (
    "Model predicted Frustrated / Masked Stress (88%) for Person #1. The room had average noise of 72 dB over "
    "15 minutes, and they showed 3 clenched-fist events in the last 5 minutes after 4.2 hours since the last "
    "break. Despite a smiling expression (valence +0.62), context corrected the facial mask."
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
        fake = FakeLLM(*outputs, delay_s=kwargs.pop("delay_s", 0.0))
        narrator = Narrator(generate_fn=fake, **kwargs)
        made.append(narrator)
        return narrator, fake

    yield factory
    for narrator in made:
        narrator.close()


def test_good_answer_passes_validation():
    assert validate(GOOD, SHEET).ok


def test_first_try_pass_is_llm_with_contract_sampling(make_narrator):
    narrator, fake = make_narrator(GOOD)
    out = narrator.narrate(SHEET)
    assert (out["source"], out["attempts"], out["text"], out["reasons"]) == ("llm", 1, GOOD, [])
    assert isinstance(out["latency_ms"], int)
    params = fake.calls[0][1]
    assert (params.temp, params.top_p, params.top_k, params.repetition_penalty, params.max_tokens) == (
        0.7, 0.8, 20, 1.05, 90)
    result = narrator.narrate_result(dict(SHEET, event_id="evt_other"))
    assert isinstance(result, NarrationResult) and result.source == "llm"


def test_retry_pass_is_llm_retry_and_carries_feedback(make_narrator):
    narrator, fake = make_narrator(BAD, GOOD)
    out = narrator.narrate(SHEET)
    assert (out["source"], out["attempts"], out["text"]) == ("llm_retry", 2, GOOD)
    assert "number:57" in out["reasons"] and "banned:gendered:she" in out["reasons"]
    assert [call[1].temp for call in fake.calls] == [0.7, 0.4]
    first_user, retry_user = fake.calls[0][0][-1]["content"], fake.calls[1][0][-1]["content"]
    assert "fix:" not in first_user
    assert "fix:" in retry_user and "'57'" in retry_user and "'she'" in retry_user


def test_always_failing_falls_back_to_template_after_three_attempts(make_narrator):
    narrator, fake = make_narrator(BAD)
    out = narrator.narrate(SHEET)
    assert (out["source"], out["attempts"]) == ("template", 3)
    assert out["text"] == render_template(SHEET) and validate(out["text"], SHEET).ok
    assert [call[1].temp for call in fake.calls] == [p.temp for p in SCHEDULE] == [0.7, 0.4, 0.0]
    assert len(out["attempt_reasons"]) == 3


def test_slow_backend_exceeding_deadline_returns_template_in_time(make_narrator):
    narrator, _ = make_narrator(GOOD, delay_s=4.0)
    started = time.monotonic()
    out = narrator.narrate(SHEET)
    assert time.monotonic() - started <= 2.51 and out["latency_ms"] <= 2510
    assert (out["source"], out["attempts"]) == ("template", 1)
    assert "timeout" in out["reasons"] and validate(out["text"], SHEET).ok


def test_backend_ignoring_the_deadline_still_answers_within_it(make_narrator):
    """Review F2: three 0.9 s attempts used to end at ~2557 ms; the template now fits inside 2.5 s."""
    narrator, fake = make_narrator(BAD, delay_s=0.9)
    started = time.monotonic()
    out = narrator.narrate(SHEET)
    assert out["source"] == "template" and out["latency_ms"] <= 2510, out["latency_ms"]
    assert all(call[2] <= started + 2.5 - 0.04 for call in fake.calls)  # generation stops early to validate


def test_answer_cut_at_the_deadline_is_not_shown(make_narrator):
    cut = Generation("Model predicted Frustrated / Masked Stress (88%) for Person #1 because", "deadline")
    narrator, _ = make_narrator(cut, deadline_s=0.3)
    out = narrator.narrate(SHEET)
    assert out["source"] == "template" and "timeout" in out["attempt_reasons"][0]


def test_retries_stop_when_too_little_time_remains(make_narrator):
    narrator, fake = make_narrator(BAD, delay_s=0.25, deadline_s=0.6, min_attempt_s=0.4)
    out = narrator.narrate(SHEET)
    assert (out["source"], out["attempts"], len(fake.calls)) == ("template", 1, 1)


def test_backend_exception_falls_back_to_template():
    def broken(messages, params, deadline):
        raise RuntimeError("metal exploded")

    narrator = Narrator(generate_fn=broken)
    try:
        out = narrator.narrate(SHEET)
    finally:
        narrator.close()
    assert out["source"] == "template" and out["text"] == render_template(SHEET)
    # review F3: a crash is reported as an error, not as a timeout
    assert out["reasons"] == ["error:RuntimeError"] and out["attempts"] == 1


def test_queued_request_with_too_little_time_left_does_not_start_an_attempt(make_narrator):
    """Review F5: a request queued behind a slow one used to start an attempt with ~0 s left."""
    narrator, fake = make_narrator(GOOD, delay_s=2.3, reuse_texts=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(narrator.narrate, SHEET)
        time.sleep(0.05)
        second = pool.submit(narrator.narrate, dict(SHEET, event_id="evt_queued"))
        a, b = first.result(), second.result()
    assert a["source"] == "llm"
    assert b["source"] == "template" and b["attempts"] == 0 and len(fake.calls) == 1


def test_truncated_answer_is_salvaged_to_complete_sentences(make_narrator):
    narrator, _ = make_narrator(Generation(GOOD + " Ambient noise kept rising while", "length", 90))
    out = narrator.narrate(SHEET)
    assert (out["source"], out["text"], out["salvaged"]) == ("llm", GOOD, True)
    assert "length" in out["attempt_reasons"][0]


def test_salvage_and_complete_sentences():
    result = salvage(GOOD + " She looked exhausted at 99 dB.", SHEET)
    assert result is not None and result.text == GOOD
    assert salvage(BAD, SHEET) is None
    assert complete_sentences("One is 4.2 hours. Two ends here! Three is cut") == [
        "One is 4.2 hours.", "Two ends here!"]
    assert complete_sentences("A. B. C. D.") == ["A.", "B.", "C."]


def test_llm_output_is_cleaned_before_it_is_shown(make_narrator):
    narrator, _ = make_narrator(f"<think>\n\n</think>\n\nExplanation: **{GOOD}**")
    out = narrator.narrate(SHEET)
    assert out["source"] == "llm" and out["text"] == GOOD


def test_all_generation_runs_on_one_worker_thread(make_narrator):
    narrator, fake = make_narrator(GOOD)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(narrator.narrate, make_batch(6, seed=5)))
    assert len(fake.threads) == 1 and threading.get_ident() not in fake.threads


def test_seed_is_deterministic_per_event_and_attempt(make_narrator):
    narrator, fake = make_narrator(BAD)
    narrator.narrate(SHEET)
    assert [call[1].seed for call in fake.calls] == [event_seed("evt_000123", i) for i in range(3)]


# --- reuse of remembered texts --------------------------------------------------------------------


def test_remembered_text_is_reused_only_after_revalidation(make_narrator):
    narrator, fake = make_narrator(GOOD)
    assert narrator.narrate(SHEET)["source"] == "llm"
    again = narrator.narrate(dict(SHEET, event_id="evt_000124"))
    assert again["reused"] and again["text"] == GOOD and again["attempts"] == 0 and len(fake.calls) == 1


def test_stale_remembered_text_is_rejected_and_regenerated(make_narrator):
    narrator, fake = make_narrator(GOOD, BAD)
    narrator.narrate(SHEET)
    newer = copy.deepcopy(SHEET)
    newer["event_id"] = "evt_000200"
    for layer in newer["layers"]:
        for d in layer["drivers"]:
            if d["feature"] == "db_mean_15m":
                d.update(value=75.2, display="average noise of 75 dB over 15 minutes")
    assert situation_key(newer) == situation_key(SHEET)
    assert not validate(GOOD, newer).ok
    out = narrator.narrate(newer)
    assert not out["reused"] and len(fake.calls) > 1
    assert out["source"] == "template" and "72 dB" not in out["text"] and validate(out["text"], newer).ok


def test_reuse_can_be_disabled(make_narrator):
    narrator, fake = make_narrator(GOOD, reuse_texts=False)
    narrator.narrate(SHEET)
    narrator.narrate(dict(SHEET, event_id="evt_000124"))
    assert len(fake.calls) == 2


# --- service-facing behaviour -------------------------------------------------------------------------


def test_template_only_narrator():
    narrator = TemplateOnlyNarrator()
    narrator.load()
    out = narrator.narrate(SHEET)
    assert (out["source"], out["attempts"]) == ("template", 0)
    health = narrator.health()
    assert health["mode"] == "template" and health["model"] is None and health["warm"] is True
    assert health["by_source"] == {"llm": 0, "llm_retry": 0, "template": 1}
    narrator.close()


def test_unloaded_narrator_without_model_serves_templates(tmp_path):
    narrator = Narrator(model_path=str(tmp_path / "missing-model"))
    with pytest.raises(FileNotFoundError):
        narrator.load()
    out, health = narrator.narrate(SHEET), narrator.health()
    narrator.close()
    assert out["source"] == "template"
    assert health["warm"] is False and "FileNotFoundError" in health["error"]


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("VALENCE_NARRATOR_MODEL", "models/llm/Other")
    monkeypatch.setenv("VALENCE_NARRATOR_ADAPTER", "adapters/v1")
    narrator = Narrator()
    assert (narrator.model_path, narrator.adapter_path) == ("models/llm/Other", "adapters/v1")
    assert narrator.deadline_s == 2.5 and narrator.max_tokens == 128  # compact style with an adapter
    narrator.close()
    monkeypatch.delenv("VALENCE_NARRATOR_MODEL")
    monkeypatch.delenv("VALENCE_NARRATOR_ADAPTER")
    narrator = Narrator()
    assert narrator.model_path == "models/llm/Qwen3-1.7B-4bit" and narrator.adapter_path is None
    assert narrator.max_tokens == 90
    with Narrator(max_tokens=64) as explicit:
        assert explicit.max_tokens == 64
    narrator.close()


def test_health_counts_requests_by_source(make_narrator):
    narrator, _ = make_narrator(GOOD)
    narrator.load()
    narrator.narrate(SHEET)
    narrator.narrate(few_shots()[2][0])
    health = narrator.health()
    assert health["warm"] is True and health["requests"] == 2
    assert health["by_source"]["llm"] == 1 and health["by_source"]["template"] == 1


def test_logs_carry_no_narration_text_or_sheet_facts(make_narrator, caplog):
    narrator, _ = make_narrator(BAD, GOOD)
    with caplog.at_level(logging.DEBUG, logger="valence.narrator"):
        out = narrator.narrate(SHEET)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "evt_000123" in logged and "llm_retry" in logged
    for secret in (out["text"], "average noise", "Frustrated / Masked Stress", "smiling", "She seems"):
        assert secret not in logged
    assert all("number:" not in r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    # review F4: even DEBUG logs carry only reason kinds, never quoted numbers or words
    assert "number:57" not in logged and "she" not in logged.lower().split()
    assert "rejected: words, number, state_missing" in logged


# --- prompt cache mechanics (tiny mlx arrays, no model) ---------------------------------------------


def test_request_cache_deep_copies_and_never_mutates_the_base():
    mx = pytest.importorskip("mlx.core")
    cache_mod = pytest.importorskip("mlx_lm.models.cache")
    base = [cache_mod.KVCache() for _ in range(2)]
    for c in base:
        c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
    backend = MLXBackend(None, prompt_cache=True)
    backend._prefixes["SYS"] = _Prefix("core", [1, 2, 3, 4, 5], base)
    snapshot = [mx.array(c.keys) for c in base]

    cache, cached = backend.request_cache([1, 2, 3, 4, 5, 9, 9], "SYS")
    assert cached == 5 and cache is not base and all(a is not b for a, b in zip(cache, base, strict=True))
    for c in cache:
        c.update_and_fetch(mx.zeros((1, 2, 2, 4)), mx.zeros((1, 2, 2, 4)))
    assert [c.offset for c in base] == [5, 5] and [c.offset for c in cache] == [7, 7]
    assert all(mx.array_equal(c.keys, s).item() for c, s in zip(base, snapshot, strict=True))

    cache, cached = backend.request_cache([1, 2, 3, 7, 7, 7], "SYS")  # diverges inside the prefix
    assert cached == 3 and [c.offset for c in cache] == [3, 3] and [c.offset for c in base] == [5, 5]
    cache, cached = backend.request_cache([1, 2, 3, 4, 5], "SYS")  # leaves one token to feed
    assert cached == 4
    assert backend.request_cache([8, 8], "SYS") == (None, 0)
    assert backend.request_cache([1, 2, 3], "UNKNOWN SYSTEM PROMPT") == (None, 0)
    assert backend.prefix_tokens == {"core": 5}


def test_request_cache_disabled():
    backend = MLXBackend(None, prompt_cache=False)
    assert backend.request_cache([1, 2, 3], "SYS") == (None, 0)


def test_ship_switch(monkeypatch, tmp_path):
    from valence.narrator import engine

    monkeypatch.delenv("VALENCE_NARRATOR_ADAPTER", raising=False)
    monkeypatch.delenv("VALENCE_NARRATOR_SHIP", raising=False)
    monkeypatch.setattr(engine, "SHIP", "prompt")
    assert engine.ship_choice() == "prompt" and engine.startup_adapter() is None
    adapter = tmp_path / "narrator"
    adapter.mkdir()
    monkeypatch.setattr(engine, "SHIPPED_ADAPTER", str(adapter))
    monkeypatch.setenv("VALENCE_NARRATOR_SHIP", "lora")
    assert engine.startup_adapter() is None  # files missing: degrade to prompt-only
    for name in ("adapters.safetensors", "adapter_config.json"):
        (adapter / name).write_text("{}")
    assert engine.startup_adapter() == str(adapter)
    assert engine.startup_adapter("adapters/x") == "adapters/x"
    assert engine.startup_adapter("none") is None
    monkeypatch.setenv("VALENCE_NARRATOR_ADAPTER", "none")
    assert engine.startup_adapter() is None
    with Narrator() as narrator:  # "none" in the env is no adapter, not a path
        assert narrator.adapter_path is None and narrator.prompt_style == "few_shot"
    monkeypatch.setenv("VALENCE_NARRATOR_SHIP", "bogus")
    monkeypatch.delenv("VALENCE_NARRATOR_ADAPTER")
    with pytest.raises(ValueError):
        engine.ship_choice()
