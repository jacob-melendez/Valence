from __future__ import annotations

import json
from pathlib import Path

import pytest

from valence.narrator.engine import MLXBackend
from valence.narrator.lora_data import NarratorChatDataset, completion_loss, encode_row, turn_suffix
from valence.narrator.prompt import build_messages, render_chat

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/llm/Qwen3-1.7B-4bit"
DATA_DIR = ROOT / "data/narrator"


def _rows(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        pytest.skip("LoRA data not built")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def qwen():
    """The tokenizer object the engine gets from ``mlx_lm.load`` (mlx_lm's TokenizerWrapper)."""
    if not (MODEL_DIR / "tokenizer_config.json").exists():
        pytest.skip("Qwen3 tokenizer not downloaded")
    tokenizer_utils = pytest.importorskip("mlx_lm.tokenizer_utils")
    return tokenizer_utils.load(MODEL_DIR)


def test_prompt_tokens_equal_the_engine_prompt_tokens(qwen):
    """For 50 test rows: tokens[:offset] is exactly what MLXBackend feeds for that sheet (compact
    style), and tokens[offset:] is exactly the target plus the template's end of turn."""
    backend = MLXBackend(None, prompt_style="compact")
    backend.tokenizer = qwen
    suffix = turn_suffix(qwen)
    assert suffix == "<|im_end|>\n"
    rows = _rows("test")[:50]
    assert len(rows) == 50
    for row in rows:
        tokens, offset = encode_row(qwen, row["messages"], check_template=True)
        engine_tokens = backend.encode(render_chat(qwen, build_messages(row["sheet"], style=backend.prompt_style)))
        assert tokens[:offset] == engine_tokens
        assert qwen.decode(tokens[offset:]) == row["messages"][-1]["content"] + suffix
        assert qwen.decode(tokens[offset - 4:offset]) == "<think>\n\n</think>\n\n"  # prompt side, not trained


def test_stock_chat_dataset_would_train_the_think_block(qwen):
    """Documents why the helper exists: mlx_lm's ChatDataset masks 4 tokens fewer."""
    datasets = pytest.importorskip("mlx_lm.tuner.datasets")
    row = _rows("train")[0]
    stock_tokens, stock_offset = datasets.ChatDataset([row], qwen, mask_prompt=True).process(row)
    tokens, offset = encode_row(qwen, row["messages"])
    assert stock_tokens == tokens
    assert offset - stock_offset == 4
    assert qwen.decode(tokens[stock_offset:offset]) == "<think>\n\n</think>\n\n"


def test_every_train_and_valid_row_tokenizes_cleanly(qwen):
    for name in ("train", "valid"):
        dataset = NarratorChatDataset(_rows(name), qwen)
        stats = dataset.stats()
        assert stats["n"] == {"train": 1500, "valid": 150}[name]
        assert stats["max_tokens"] < 1024 and stats["completion_min"] > 20
        assert dataset.itemlen(0) == len(dataset[0][0]) and dataset.process(dataset[0]) is dataset[0]


def test_batches_and_loss_train_exactly_the_target_tokens(qwen):
    mx = pytest.importorskip("mlx.core")
    trainer = pytest.importorskip("mlx_lm.tuner.trainer")
    dataset = NarratorChatDataset(_rows("valid")[:8], qwen)
    batch, lengths = next(trainer.iterate_batches(dataset, batch_size=8, max_seq_length=2048))
    pairs = [tuple(x) for x in lengths.tolist()]
    assert sorted(pairs) == sorted((offset, len(tokens)) for tokens, offset in dataset)
    vocab = 8
    model = lambda inputs: mx.zeros((*inputs.shape, vocab))  # noqa: E731
    _, ntoks = completion_loss(model, batch % vocab, lengths)
    _, stock = trainer.default_loss(model, batch % vocab, lengths)
    targets = sum(length - offset for offset, length in pairs)
    assert int(ntoks) == targets and int(stock) == targets + len(pairs)
