"""Narrator engine: a validated local-LLM explanation inside a hard deadline, else a template.

Each request gets up to three sampled attempts (temperature 0.7, then 0.4, then greedy; top_p 0.8,
top_k 20, repetition penalty 1.05, max 90 tokens), each checked by :func:`validate`; the first one
that passes is shown (source ``llm`` on the first attempt, ``llm_retry`` after). A retry carries a
``fix:`` line built from the previous draft's rejection reasons. When nothing passes before the
2.5 s deadline, the deterministic template for the event is returned (source ``template``), so the
caller always gets a faithful explanation on time.

The whole request, template included, fits inside the deadline: generation stops
``RESERVE_S`` early (time to validate the last draft), the caller stops waiting
``TEMPLATE_RESERVE_S`` early (time to render the template, ~2 ms), and an attempt only starts
with enough time left for a prefill and a few tokens (``min_first_attempt_s`` / ``min_attempt_s``),
so a request queued behind a slow one does not tie up the worker.

Recently accepted texts are kept per situation (person, label, face case, key drivers) and reused
only after re-validation against the *current* sheet, so a stale number never survives a reuse.

MLX is not thread-safe, so model loading, prompt-cache prefill, warm-up and every generation run
on one dedicated worker thread; ``narrate`` only submits work and waits up to the deadline. The
backend is injectable (``generate_fn``) so tests and template-only deployments need no model. Logs
carry the event id, source, attempt count and latency, never narration text or sheet facts.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import threading
import time
import zlib
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from valence.contracts import NarrationResult
from valence.narrator.prompt import (
    STANDARD_STATE_SETS,
    Message,
    build_messages,
    few_shots,
    prompt_prefix,
    render_chat,
    system_prompt,
)
from valence.narrator.schema import as_dict, key_drivers, state_set_name
from valence.narrator.templates import minimal_text, render_template
from valence.narrator.validator import MAX_SENTENCES, ValidationResult, clean_text, split_sentences, validate

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "models/llm/Qwen3-1.7B-4bit"
DEADLINE_S = 2.5
MAX_TOKENS = 90
MIN_ATTEMPT_S = 0.4  # a retry needs a prefill plus most of a short answer
MIN_FIRST_ATTEMPT_S = 0.3  # a first attempt needs at least the (cached) prefill and a few tokens
RESERVE_S = 0.05  # generation stops this long before the deadline, leaving time to validate
TEMPLATE_RESERVE_S = 0.02  # the caller stops waiting this long before the deadline, to render the template
TEXT_CACHE_SIZE = 256
SOURCES = ("llm", "llm_retry", "template")
_SENTENCE_DONE = re.compile(r"[.!?][\"'”’)\]]*$")


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temp: float
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_tokens: int = MAX_TOKENS
    seed: int | None = None


SCHEDULE: tuple[SamplingParams, ...] = (SamplingParams(temp=0.7), SamplingParams(temp=0.4), SamplingParams(temp=0.0))


@dataclass(slots=True)
class Generation:
    """Raw model output; ``finish`` is "stop" (EOS), "length" (max_tokens) or "deadline"."""

    text: str
    finish: str = "stop"
    tokens: int = 0


GenerateFn = Callable[[Sequence[Message], SamplingParams, float], "str | Generation"]


@dataclass(slots=True)
class Attempt:
    temp: float
    ok: bool
    reasons: list[str]
    finish: str
    salvaged: bool = False
    ms: float = 0.0
    text: str = ""


@dataclass(slots=True)
class _Run:
    """Shared between the waiting caller and the worker; the caller reads it once at the end."""

    started: int = 0
    attempts: list[Attempt] = field(default_factory=list)
    result: tuple[str, str] | None = None
    error: str | None = None  # exception type if a generation crashed
    cancel: threading.Event = field(default_factory=threading.Event)


def complete_sentences(text: str, limit: int = MAX_SENTENCES) -> list[str]:
    """Leading complete sentences of a (cleaned) text, dropping a trailing fragment."""
    prefix: list[str] = []
    for sentence in split_sentences(clean_text(text)):
        if not _SENTENCE_DONE.search(sentence) or len(prefix) == limit:
            break
        prefix.append(sentence)
    return prefix


def salvage(text: str, sheet: Mapping[str, Any]) -> ValidationResult | None:
    """Longest prefix of complete sentences that passes validation, if any.

    Only drops trailing sentences (never edits words), so a salvaged text is still the model's own
    wording and still fully checked against the sheet.
    """
    sentences = complete_sentences(text)
    for count in range(len(sentences), 0, -1):
        result = validate(" ".join(sentences[:count]), sheet)
        if result.ok:
            return result
    return None


def event_seed(event_id: str, attempt: int) -> int:
    return (zlib.crc32(event_id.encode()) + attempt) & 0xFFFFFFFF


def situation_key(sheet: Mapping[str, Any]) -> tuple:
    """What a reusable text is about. Deliberately excludes the numbers: a reused text is always
    re-validated, which is what rejects it when a number it quotes has changed."""
    return (sheet.get("person"), sheet.get("label"), sheet.get("label_display"), bool(sheet.get("mask_override")),
            bool(sheet.get("face_visible")), sheet.get("n_people"),
            tuple(str(d.get("feature")) for d in key_drivers(sheet)))


def reason_kind(reason: str) -> str:
    """The category of a validator reason ("number:57" -> "number"): safe to log, as it never
    carries words or numbers from the draft."""
    return reason.partition(":")[0]


def _as_generation(output: str | Generation) -> Generation:
    return output if isinstance(output, Generation) else Generation(str(output or ""))


def _resolve_local(path: str | None, what: str) -> str | None:
    """Local path only: the narrator never downloads weights or adapters."""
    if not path:
        return None
    candidate = Path(path).expanduser()
    for option in (candidate, REPO_ROOT / candidate):
        if option.exists():
            return str(option)
    raise FileNotFoundError(f"narrator {what} not found locally: {path}")


class Narrator:
    """Validated LLM narration with a template fallback, all generation on one worker thread."""

    mode = "llm"

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
        deadline_s: float = DEADLINE_S,
        max_tokens: int = MAX_TOKENS,
        *,
        generate_fn: GenerateFn | None = None,
        schedule: Sequence[SamplingParams] = SCHEDULE,
        min_attempt_s: float = MIN_ATTEMPT_S,
        min_first_attempt_s: float = MIN_FIRST_ATTEMPT_S,
        prompt_cache: bool = True,
        reuse_texts: bool = True,
    ) -> None:
        self.model_path = model_path or os.environ.get("VALENCE_NARRATOR_MODEL") or DEFAULT_MODEL
        self.adapter_path = adapter_path or os.environ.get("VALENCE_NARRATOR_ADAPTER") or None
        self.deadline_s = float(deadline_s)
        self.max_tokens = int(max_tokens)
        self.schedule = tuple(replace(p, max_tokens=self.max_tokens) for p in schedule)
        self.min_attempt_s = float(min_attempt_s)
        # capped at half the deadline so that short test deadlines still get one attempt
        self.min_first_attempt_s = min(float(min_first_attempt_s), self.deadline_s / 2)
        self.prompt_cache = prompt_cache
        self.reuse_texts = reuse_texts
        self.warm = generate_fn is not None
        self.load_error: str | None = None
        self.backend: MLXBackend | None = None
        self._generate = generate_fn
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="valence-narrator")
        self._load_lock = threading.Lock()
        self._lock = threading.Lock()
        self._texts: OrderedDict[tuple, tuple[str, str]] = OrderedDict()
        self._requests = 0
        self._reused = 0
        self._by_source: Counter[str] = Counter({source: 0 for source in SOURCES})

    # -- lifecycle ---------------------------------------------------------------------------

    def load(self) -> None:
        """Load the model, prefill the prompt caches and warm up, on the worker thread. Idempotent."""
        with self._load_lock:
            if self._generate is not None:
                self.warm = True
                return
            try:
                self._executor.submit(self._load_on_worker).result()
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise
            self.load_error = None

    def _load_on_worker(self) -> None:
        started = time.monotonic()
        backend = MLXBackend(_resolve_local(self.model_path, "model"), _resolve_local(self.adapter_path, "adapter"),
                             prompt_cache=self.prompt_cache)
        backend.load()
        loaded = time.monotonic()
        backend.warm_up()
        self.backend, self._generate, self.warm = backend, backend.generate, True
        logger.info("narrator model ready: load %.0f ms, warm-up %.0f ms, cached prefixes %s",
                    (loaded - started) * 1e3, (time.monotonic() - loaded) * 1e3, backend.prefix_tokens)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Narrator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- narration ---------------------------------------------------------------------------

    def narrate(self, sheet: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
        """Explanation for one fact sheet: the NarrationResult fields plus diagnostics
        (``reasons``, ``attempt_reasons``, ``salvaged``, ``reused``)."""
        started = time.monotonic()
        sheet = as_dict(sheet)
        event_id = str(sheet.get("event_id", "?"))
        reused = self._reuse(sheet)
        run = _Run()
        if reused is None and self._generate is not None:
            self._wait_for_llm(sheet, started + self.deadline_s, run)
        if reused is not None:
            text, source = reused
        elif run.result is not None:
            text, source = run.result
            self._remember(sheet, text, source)
        else:
            text, source = _fallback_text(sheet), "template"
        failed = [reason for attempt in run.attempts if not attempt.ok for reason in attempt.reasons]
        if run.error is not None:
            failed.append(f"error:{run.error}")
        elif run.result is None and run.started > len(run.attempts):
            failed.append("timeout")
        latency_ms = int(round((time.monotonic() - started) * 1e3))
        self._record(source, reused is not None)
        logger.info("narrate %s source=%s attempts=%d reused=%s latency_ms=%d", event_id, source, run.started,
                    reused is not None, latency_ms)
        if failed:
            logger.debug("narrate %s rejected: %s", event_id, ", ".join(dict.fromkeys(map(reason_kind, failed))))
        return {
            "text": text,
            "source": source,
            "latency_ms": latency_ms,
            "attempts": run.started,
            "reasons": list(dict.fromkeys(failed)),
            "attempt_reasons": [list(attempt.reasons) for attempt in run.attempts],
            "salvaged": any(attempt.salvaged and attempt.ok for attempt in run.attempts),
            "reused": reused is not None,
        }

    def narrate_result(self, sheet: Mapping[str, Any] | BaseModel) -> NarrationResult:
        out = self.narrate(sheet)
        return NarrationResult(text=out["text"], source=out["source"], latency_ms=out["latency_ms"],
                               attempts=out["attempts"])

    def _reuse(self, sheet: dict[str, Any]) -> tuple[str, str] | None:
        """A remembered text for this situation, only if it passes against the current sheet."""
        if not self.reuse_texts:
            return None
        key = situation_key(sheet)
        with self._lock:
            hit = self._texts.get(key)
        if hit is None:
            return None
        result = validate(hit[0], sheet)
        with self._lock:
            if result.ok:
                self._texts.move_to_end(key)
                return result.text, hit[1]
            self._texts.pop(key, None)
        logger.debug("narrate %s: remembered text is stale (%s)", sheet.get("event_id", "?"),
                     ", ".join(dict.fromkeys(map(reason_kind, result.reasons))))
        return None

    def _remember(self, sheet: dict[str, Any], text: str, source: str) -> None:
        if not self.reuse_texts:
            return
        with self._lock:
            self._texts[situation_key(sheet)] = (text, source)
            while len(self._texts) > TEXT_CACHE_SIZE:
                self._texts.popitem(last=False)

    def _wait_for_llm(self, sheet: dict[str, Any], deadline: float, run: _Run) -> None:
        """Run the attempts on the worker; stop waiting early enough to render the template in time."""
        try:
            future = self._executor.submit(self._run_attempts, sheet, deadline - RESERVE_S, run)
        except RuntimeError:
            logger.warning("narrator worker is shut down; using template")
            return
        try:
            future.result(timeout=max(0.0, deadline - TEMPLATE_RESERVE_S - time.monotonic()))
        except FutureTimeout:
            run.cancel.set()
            future.cancel()
        except Exception as exc:  # _run_attempts records its own errors; this is a last resort
            run.cancel.set()
            run.error = run.error or type(exc).__name__
            logger.warning("narrator generation failed for %s: %s", sheet.get("event_id", "?"), type(exc).__name__)

    def _run_attempts(self, sheet: dict[str, Any], deadline: float, run: _Run) -> None:
        event_id = str(sheet.get("event_id", ""))
        feedback: list[str] = []
        for index, params in enumerate(self.schedule):
            remaining = deadline - time.monotonic()
            floor = self.min_attempt_s if index else self.min_first_attempt_s
            if run.cancel.is_set() or remaining <= 0 or remaining < floor:
                return
            run.started += 1
            try:
                attempt = self._attempt(sheet, replace(params, seed=event_seed(event_id, index)), deadline, feedback)
            except Exception as exc:
                run.error = type(exc).__name__
                logger.warning("narrator generation failed for %s: %s", event_id or "?", run.error)
                return
            run.attempts.append(attempt)
            if attempt.ok:
                run.result = (attempt.text, "llm" if index == 0 else "llm_retry")
                return
            feedback = attempt.reasons

    def _attempt(self, sheet: dict[str, Any], params: SamplingParams, deadline: float, feedback: list[str]) -> Attempt:
        started = time.monotonic()
        generation = _as_generation(self._generate(build_messages(sheet, feedback or None), params, deadline))
        result = validate(generation.text, sheet)
        reasons = list(result.reasons)
        if generation.finish in ("length", "deadline"):
            reasons.append("timeout" if generation.finish == "deadline" else "length")
        accepted = result if result.ok and generation.finish == "stop" else salvage(generation.text, sheet)
        if accepted is not None and time.monotonic() > deadline:
            accepted = None
            reasons.append("timeout")
        return Attempt(temp=params.temp, ok=accepted is not None, reasons=reasons, finish=generation.finish,
                       salvaged=accepted is not None and accepted is not result,
                       ms=round((time.monotonic() - started) * 1e3, 1),
                       text=accepted.text if accepted is not None else "")

    # -- reporting ---------------------------------------------------------------------------

    def _record(self, source: str, reused: bool) -> None:
        with self._lock:
            self._requests += 1
            self._reused += int(reused)
            self._by_source[source] += 1

    @property
    def ready(self) -> bool:
        """The LLM is loaded and warmed up, so explanations can come from the model."""
        return self.mode == "llm" and self._generate is not None and self.warm and self.load_error is None

    def health(self) -> dict[str, Any]:
        """``ok`` is true only when the LLM is ready (the service answers 503 otherwise, including in
        template-only mode and after a failed load, when every explanation is a template)."""
        ready = self.ready
        with self._lock:
            return {
                "ok": ready,
                "mode": "llm" if ready else "template",
                "model": self.model_path if self.backend is not None else None,
                "adapter": self.adapter_path,
                "warm": self.warm,
                "requests": self._requests,
                "reused": self._reused,
                "by_source": dict(self._by_source),
                "deadline_s": self.deadline_s,
                "prompt_cache": self.backend.prefix_tokens if self.backend is not None else None,
                "error": self.load_error,
            }


class TemplateOnlyNarrator(Narrator):
    """No model at all: every request gets the deterministic template (demo fallback, tests)."""

    mode = "template"

    def __init__(self, deadline_s: float = DEADLINE_S) -> None:
        super().__init__(model_path="", adapter_path="", deadline_s=deadline_s, reuse_texts=False)
        self.model_path, self.adapter_path, self.warm = None, None, True

    def load(self) -> None:
        self.warm = True


def _fallback_text(sheet: Mapping[str, Any]) -> str:
    try:
        return render_template(sheet)
    except Exception as exc:
        logger.warning("template failed for %s: %s", sheet.get("event_id", "?"), type(exc).__name__)
        return minimal_text(sheet)


@dataclass(slots=True)
class _Prefix:
    name: str
    tokens: list[int]
    cache: list[Any]  # never mutated after prefill; each request works on a deep copy


class MLXBackend:
    """mlx-lm generation with a KV cache of the fixed prompt prefix (system prompt + few-shots).

    One prefix per state set is prefilled at load. Each request deep-copies that cache (the way
    mlx_lm's own ``LRUPromptCache`` hands out cached prefixes) and feeds only the tokens after the
    longest common prefix, so the base cache is never mutated. Must be used from a single thread.
    """

    def __init__(self, model_path: str | None, adapter_path: str | None = None, *, prompt_cache: bool = True) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.use_prompt_cache = prompt_cache
        self.model: Any = None
        self.tokenizer: Any = None
        self._prefixes: dict[str, _Prefix] = {}
        self.last_prompt_tokens = 0
        self.last_cached_tokens = 0

    @property
    def prefix_tokens(self) -> dict[str, int]:
        return {p.name: len(p.tokens) for p in self._prefixes.values()}

    def load(self) -> None:
        from mlx_lm import load

        if self.model_path is None:
            raise FileNotFoundError("no narrator model path")
        self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)
        if self.use_prompt_cache:
            for states in STANDARD_STATE_SETS:
                self._prefix_for(system_prompt(states), states)

    def warm_up(self) -> None:
        """Short generations with the real sampling settings (sampled and greedy) for every cached
        state set, so Metal kernels and the cache-copy path are compiled before the first request."""
        for states in STANDARD_STATE_SETS:
            messages = build_messages(few_shots(states)[1][0])
            for params in (SCHEDULE[0], SCHEDULE[-1]):
                self.generate(messages, replace(params, max_tokens=8, seed=0), time.monotonic() + 60)

    def encode(self, text: str) -> list[int]:
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special = bos is None or not text.startswith(bos)
        return list(self.tokenizer.encode(text, add_special_tokens=add_special))

    def _prefix_for(self, system: str, states: tuple[str, ...] | None = None) -> _Prefix | None:
        """Prefilled prefix for a system prompt, built on first use."""
        if not self.use_prompt_cache:
            return None
        if system in self._prefixes:
            return self._prefixes[system]
        if states is None:
            states = next((s for s in STANDARD_STATE_SETS if system_prompt(s) == system), None)
            if states is None:
                return None
        import mlx.core as mx
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

        tokens = self.encode(prompt_prefix(self.tokenizer, states))
        cache = make_prompt_cache(self.model)
        if not tokens or not can_trim_prompt_cache(cache):
            return None
        for start in range(0, len(tokens), 512):
            self.model(mx.array(tokens[start:start + 512])[None], cache=cache)
            mx.eval([c.state for c in cache])
        self._prefixes[system] = prefix = _Prefix(state_set_name(states), tokens, cache)
        return prefix

    def request_cache(self, tokens: list[int], system: str) -> tuple[list[Any] | None, int]:
        """A private cache copy for this prompt and how many leading tokens it already holds."""
        prefix = self._prefix_for(system)
        if prefix is None:
            return None, 0
        from mlx_lm.models.cache import trim_prompt_cache

        common, limit = 0, min(len(prefix.tokens), len(tokens) - 1)
        while common < limit and prefix.tokens[common] == tokens[common]:
            common += 1
        if common == 0:
            return None, 0
        cache = copy.deepcopy(prefix.cache)
        if common < len(prefix.tokens):
            trim_prompt_cache(cache, len(prefix.tokens) - common)
        return cache, common

    def generate(self, messages: Sequence[Message], params: SamplingParams, deadline: float) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        tokens = self.encode(render_chat(self.tokenizer, messages))
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        cache, cached = self.request_cache(tokens, system)
        self.last_prompt_tokens, self.last_cached_tokens = len(tokens), cached
        if params.seed is not None:
            mx.random.seed(params.seed)
        stream = stream_generate(
            self.model, self.tokenizer, mx.array(tokens[cached:]), max_tokens=params.max_tokens,
            sampler=make_sampler(temp=params.temp, top_p=params.top_p, top_k=params.top_k),
            logits_processors=make_logits_processors(repetition_penalty=params.repetition_penalty),
            prompt_cache=cache,
        )
        pieces: list[str] = []
        finish, count = "length", 0
        try:
            for response in stream:
                pieces.append(response.text)
                count = response.generation_tokens
                if response.finish_reason:
                    finish = response.finish_reason
                elif time.monotonic() >= deadline:
                    finish = "deadline"
                    break
        finally:
            stream.close()
        return Generation("".join(pieces), finish, count)
