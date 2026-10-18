"""LoRA training rows tokenized exactly as the engine prompts the model, with the right loss mask.

mlx_lm's stock ``ChatDataset`` (0.31.3) tokenizes ``apply_chat_template(messages)`` and masks
``apply_chat_template(messages[:-1], add_generation_prompt=True)`` *without* ``enable_thinking=False``.
For Qwen3 that masked prompt ends at ``<|im_start|>assistant\\n``, while the full render (and the
engine's inference prompt) continues with the empty ``<think>\\n\\n</think>\\n\\n`` block, so those 4
tokens land in the trained completion although at inference they are part of the prompt.

``NarratorChatDataset`` fixes that by construction. For each chat row ``{"messages": [system, user,
assistant]}`` (``prompt.training_messages``):

- ``prompt = prompt.render_chat(tokenizer, messages[:-1])``: the engine's own rendering (generation
  prompt, thinking disabled), tokenized with ``prompt.encode_text`` like ``MLXBackend.encode``;
- ``tokens = encode_text(prompt + target + turn_suffix(tokenizer))``, where the suffix is how the
  chat template itself closes an assistant turn (``<|im_end|>\\n`` for Qwen3);
- ``offset = len(encode_text(prompt))``: the first trained token is the first target token.

It raises if the prompt tokens are not a prefix of the full tokens (a BPE merge across the
boundary) or, with ``check_template``, if the template's own render of the row differs from
``prompt + target + suffix``. Items are ``(tokens, offset)``, the shape mlx_lm's
``iterate_batches`` reads; pass the dataset straight to ``mlx_lm.tuner.trainer.train`` (wrapping
it in ``CacheDataset`` also works through the identity ``process`` but loses length sorting).

``completion_loss`` is mlx_lm's ``default_loss`` with the mask's upper bound made exclusive: the
stock mask (``steps <= length``) also trains one padding token (id 0) after the final ``\\n``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from valence.narrator.prompt import Message, encode_text, render_chat

__all__ = ["NarratorChatDataset", "completion_loss", "encode_row", "inference_prompt", "load_splits", "turn_suffix"]

SPLITS: tuple[str, ...] = ("train", "valid", "test")
_SENTINEL = "⁣VALENCE_TARGET⁣"


def turn_suffix(tokenizer: Any) -> str:
    """What the chat template writes after an assistant message's content (Qwen3: ``<|im_end|>\\n``)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": _SENTINEL}], tokenize=False)
    index = text.rfind(_SENTINEL)
    if index < 0:
        raise ValueError("chat template drops the assistant content")
    return text[index + len(_SENTINEL):]


def inference_prompt(tokenizer: Any, messages: Sequence[Message]) -> str:
    """The engine's prompt text for these (system + user) messages."""
    return render_chat(tokenizer, messages)


def encode_row(tokenizer: Any, messages: Sequence[Message], suffix: str | None = None,
               check_template: bool = False) -> tuple[list[int], int]:
    """(tokens, offset) for one chat row whose last message is the assistant target."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the last message must be the assistant target")
    suffix = turn_suffix(tokenizer) if suffix is None else suffix
    prompt = inference_prompt(tokenizer, messages[:-1])
    text = prompt + str(messages[-1]["content"]) + suffix
    if check_template:
        rendered = tokenizer.apply_chat_template(list(messages), tokenize=False)
        if rendered != text:
            raise ValueError("chat template renders the row differently from inference prompt + target + suffix")
    head = encode_text(tokenizer, prompt)
    tokens = encode_text(tokenizer, text)
    if tokens[:len(head)] != head:
        raise ValueError("prompt tokens are not a prefix of the row's tokens (merge across the boundary)")
    return tokens, len(head)


class NarratorChatDataset:
    """Pre-tokenized ``(tokens, offset)`` items for mlx_lm training (see the module docstring)."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, check_template: bool = True,
                 chat_key: str = "messages") -> None:
        suffix = turn_suffix(tokenizer)
        self.items: list[tuple[list[int], int]] = [
            encode_row(tokenizer, row[chat_key], suffix, check_template) for row in rows]

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Any, **kwargs: Any) -> NarratorChatDataset:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([json.loads(line) for line in lines if line.strip()], tokenizer, **kwargs)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.items[idx]

    def __iter__(self) -> Iterator[tuple[list[int], int]]:
        return iter(self.items)

    def itemlen(self, idx: int) -> int:
        return len(self.items[idx][0])

    def process(self, item: tuple[list[int], int]) -> tuple[list[int], int]:
        """Identity: items are already processed (lets mlx_lm's ``CacheDataset`` wrap this)."""
        return item

    def stats(self) -> dict[str, int]:
        prompts = [offset for _, offset in self.items]
        completions = [len(tokens) - offset for tokens, offset in self.items]
        return {"n": len(self.items), "max_tokens": max((len(t) for t, _ in self.items), default=0),
                "prompt_min": min(prompts, default=0), "prompt_max": max(prompts, default=0),
                "completion_min": min(completions, default=0), "completion_max": max(completions, default=0),
                "completion_total": sum(completions)}


def load_splits(data_dir: str | Path, tokenizer: Any, **kwargs: Any) -> dict[str, NarratorChatDataset]:
    """train / valid / test datasets from ``data_dir`` (missing files are skipped)."""
    data_dir = Path(data_dir)
    return {name: NarratorChatDataset.from_jsonl(data_dir / f"{name}.jsonl", tokenizer, **kwargs)
            for name in SPLITS if (data_dir / f"{name}.jsonl").exists()}


def completion_loss(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
    """mlx_lm ``default_loss`` training only tokens ``offset .. length-1`` (no trailing pad token)."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks
