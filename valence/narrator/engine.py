"""Narrator engine: a local LLM explanation inside a hard deadline, or the event's template.

Each request gets up to three sampled attempts (temperature 0.7, then 0.4, then greedy), each
checked by :func:`validate`; the first one that passes is shown. A retry carries a ``Fix:`` line
built from the previous draft's rejection reasons. When nothing passes before the deadline, the
deterministic template for the event is returned instead, so the caller always gets a faithful
explanation on time.

MLX is not thread-safe across threads, so model loading, warm-up and every generation run on one
dedicated worker thread; ``narrate`` only submits work to it and waits up to the deadline. The
backend is injectable (``generate_fn``) so tests and template-only deployments need no model.
Logs carry the event id, source, attempt count and latency, never narration text or sheet facts.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from valence.narrator.prompt import FEW_SHOTS, Message, build_messages, prompt_prefix, render_chat
from valence.narrator.schema import as_dict, format_percent
from valence.narrator.templates import render_template
from valence.narrator.validator import MAX_SENTENCES, ValidationResult, clean_text, split_sentences, validate

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "models/llm/Qwen3-1.7B-4bit"
DEADLINE_S = 2.5
MAX_TOKENS = 110
MIN_ATTEMPT_S = 0.4
GRACE_S = 0.05
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


SCHEDULE: tuple[SamplingParams, ...] = (
    SamplingParams(temp=0.7),
    SamplingParams(temp=0.4),
    SamplingParams(temp=0.0),
)


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
    cancel: threading.Event = field(default_factory=threading.Event)


def complete_sentences(text: str, limit: int = MAX_SENTENCES) -> list[str]:
    """Leading complete sentences of a (cleaned) text, dropping a trailing fragment."""
    sentences = split_sentences(clean_text(text))
    prefix: list[str] = []
    for sentence in sentences:
        if not _SENTENCE_DONE.search(sentence) or len(prefix) == limit:
            break
        prefix.append(sentence)
    return prefix


def salvage(text: str, sheet: Mapping[str, Any]) -> ValidationResult | None:
    """Longest prefix of complete sentences that passes validation, if any.

    Only drops trailing sentences (never edits words), so a salvaged text is still the model's
    own wording and still fully checked against the sheet.
    """
    sentences = complete_sentences(text)
    for count in range(len(sentences), 0, -1):
        result = validate(" ".join(sentences[:count]), sheet)
        if result.ok:
            return result
    return None


def event_seed(event_id: str, attempt: int) -> int:
    return (zlib.crc32(event_id.encode()) + attempt) & 0xFFFFFFFF


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
        prompt_cache: bool = True,
    ) -> None:
        self.model_path = model_path or os.environ.get("VALENCE_NARRATOR_MODEL") or DEFAULT_MODEL
        self.adapter_path = adapter_path or os.environ.get("VALENCE_NARRATOR_ADAPTER") or None
        self.deadline_s = float(deadline_s)
        self.max_tokens = int(max_tokens)
        self.schedule = tuple(replace(p, max_tokens=self.max_tokens) for p in schedule)
        self.min_attempt_s = float(min_attempt_s)
        self.prompt_cache = prompt_cache
        self.warm = generate_fn is not None
        self.load_error: str | None = None
        self._generate = generate_fn
        self._backend: MLXBackend | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="valence-narrator")
        self._load_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._requests = 0
        self._by_source: Counter[str] = Counter({source: 0 for source in SOURCES})

    def load(self) -> None:
        """Load the model once and warm it up, on the worker thread. Idempotent."""
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
        backend = MLXBackend(
            _resolve_local(self.model_path, "model"),
            _resolve_local(self.adapter_path, "adapter"),
            prompt_cache=self.prompt_cache,
        )
        backend.load()
        loaded = time.monotonic()
        backend.warm_up()
        self._backend, self._generate, self.warm = backend, backend.generate, True
        logger.info(
            "narrator model ready: load %.0f ms, warm-up %.0f ms, cached prefix %d tokens",
            (loaded - started) * 1e3, (time.monotonic() - loaded) * 1e3, backend.prefix_tokens,
        )

    def narrate(self, sheet: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
        """Explanation for one fact sheet: {"text", "source", "latency_ms", "attempts", "reasons"}."""
        started = time.monotonic()
        sheet = as_dict(sheet)
        run = _Run()
        if self._generate is not None:
            self._wait_for_llm(sheet, started + self.deadline_s, run)
        result = run.result
        if result is None:
            text, source = _fallback_text(sheet), "template"
        else:
            text, source = result
        attempts = list(run.attempts)
        latency_ms = round((time.monotonic() - started) * 1e3, 1)
        failed = [reason for attempt in attempts if not attempt.ok for reason in attempt.reasons]
        if result is None and run.started > len(attempts):
            failed.append("timeout")
        self._record(source)
        event_id = str(sheet.get("event_id", "?"))
        logger.info("narrate %s source=%s attempts=%d latency_ms=%.0f", event_id, source, run.started, latency_ms)
        if failed:
            logger.debug("narrate %s rejected: %s", event_id, ", ".join(dict.fromkeys(failed)))
        return {
            "text": text,
            "source": source,
            "latency_ms": latency_ms,
            "attempts": run.started,
            "reasons": list(dict.fromkeys(failed)),
            "attempt_reasons": [list(attempt.reasons) for attempt in attempts],
            "salvaged": any(attempt.salvaged and attempt.ok for attempt in attempts),
        }

    def _wait_for_llm(self, sheet: dict[str, Any], deadline: float, run: _Run) -> None:
        try:
            future = self._executor.submit(self._run_attempts, sheet, deadline, run)
        except RuntimeError:
            logger.warning("narrator worker is shut down; using template")
            return
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()) + GRACE_S)
        except FutureTimeout:
            run.cancel.set()
            future.cancel()
        except Exception as exc:
            run.cancel.set()
            logger.warning("narrator generation failed for %s: %s", sheet.get("event_id", "?"), type(exc).__name__)

    def _run_attempts(self, sheet: dict[str, Any], deadline: float, run: _Run) -> None:
        event_id = str(sheet.get("event_id", ""))
        feedback: list[str] = []
        for index, params in enumerate(self.schedule):
            remaining = deadline - time.monotonic()
            if run.cancel.is_set() or remaining <= 0 or (index and remaining < self.min_attempt_s):
                return
            run.started += 1
            attempt = self._attempt(sheet, replace(params, seed=event_seed(event_id, index)), deadline, feedback)
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
        return Attempt(
            temp=params.temp,
            ok=accepted is not None,
            reasons=reasons,
            finish=generation.finish,
            salvaged=accepted is not None and accepted is not result,
            ms=round((time.monotonic() - started) * 1e3, 1),
            text=accepted.text if accepted is not None else "",
        )

    def _record(self, source: str) -> None:
        with self._stats_lock:
            self._requests += 1
            self._by_source[source] += 1

    def health(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "ok": True,
                "mode": self.mode if self._generate is not None else "template",
                "model": self.model_path if self.mode == "llm" else None,
                "adapter": self.adapter_path,
                "warm": self.warm,
                "requests": self._requests,
                "by_source": dict(self._by_source),
                "deadline_s": self.deadline_s,
                "error": self.load_error,
            }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Narrator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TemplateOnlyNarrator(Narrator):
    """No model at all: every request gets the deterministic template (demo fallback, tests)."""

    mode = "template"

    def __init__(self, deadline_s: float = DEADLINE_S) -> None:
        super().__init__(model_path="", adapter_path="", deadline_s=deadline_s)
        self.model_path, self.adapter_path, self.warm = None, None, True

    def load(self) -> None:
        self.warm = True


def _fallback_text(sheet: Mapping[str, Any]) -> str:
    try:
        return render_template(sheet)
    except Exception as exc:
        logger.warning("template failed for %s: %s", sheet.get("event_id", "?"), type(exc).__name__)
        confidence = sheet.get("confidence_display") or format_percent(float(sheet.get("confidence") or 0.0))
        return (f"Model predicted '{sheet.get('label_display') or sheet.get('label')}' with {confidence} "
                f"confidence for {sheet.get('person') or 'this person'}, based on aggregate context signals.")


class MLXBackend:
    """mlx-lm generation with a KV cache of the fixed prompt prefix (system prompt + few-shots).

    The prefix is prefilled once at load; each request feeds only its own user turn and trims the
    cache back to the prefix afterwards. Must be used from a single thread.
    """

    def __init__(self, model_path: str | None, adapter_path: str | None = None, *, prompt_cache: bool = True) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.use_prompt_cache = prompt_cache
        self.model: Any = None
        self.tokenizer: Any = None
        self._cache: list[Any] | None = None
        self._prefix: list[int] = []

    @property
    def prefix_tokens(self) -> int:
        return len(self._prefix) if self._cache is not None else 0

    def load(self) -> None:
        from mlx_lm import load

        if self.model_path is None:
            raise FileNotFoundError("no narrator model path")
        self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)
        if self.use_prompt_cache:
            self._build_prefix_cache()

    def warm_up(self) -> None:
        """One short greedy generation so Metal kernels are compiled before the first request."""
        sheet = FEW_SHOTS[0][0]
        self.generate(build_messages(sheet), SamplingParams(temp=0.0, max_tokens=24), time.monotonic() + 60)

    def _encode(self, text: str) -> list[int]:
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special = bos is None or not text.startswith(bos)
        return list(self.tokenizer.encode(text, add_special_tokens=add_special))

    def _build_prefix_cache(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

        prefix = self._encode(prompt_prefix(self.tokenizer))
        cache = make_prompt_cache(self.model)
        if not prefix or not can_trim_prompt_cache(cache):
            return
        for start in range(0, len(prefix), 1024):
            self.model(mx.array(prefix[start:start + 1024])[None], cache=cache)
            mx.eval([c.state for c in cache])
        self._cache, self._prefix = cache, prefix

    def _cache_for(self, tokens: list[int]) -> tuple[list[Any] | None, int]:
        """Reusable cache and how many leading prompt tokens it already holds."""
        if self._cache is None:
            return None, 0
        common = 0
        limit = min(len(self._prefix), len(tokens) - 1)
        while common < limit and self._prefix[common] == tokens[common]:
            common += 1
        if common < len(self._prefix):
            self._trim_to(common)
            self._prefix = self._prefix[:common]
        return self._cache, common

    def _trim_to(self, length: int) -> None:
        from mlx_lm.models.cache import trim_prompt_cache

        if self._cache is None:
            return
        extra = self._cache[0].offset - length
        if extra > 0:
            trim_prompt_cache(self._cache, extra)

    def generate(self, messages: Sequence[Message], params: SamplingParams, deadline: float) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        tokens = self._encode(render_chat(self.tokenizer, messages))
        cache, cached = self._cache_for(tokens)
        if params.seed is not None:
            mx.random.seed(params.seed)
        stream = stream_generate(
            self.model,
            self.tokenizer,
            mx.array(tokens[cached:]),
            max_tokens=params.max_tokens,
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
            self._trim_to(len(self._prefix))
        return Generation("".join(pieces), finish, count)
