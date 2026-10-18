"""train_lora wiring without a GPU: CLI defaults, adapter config, LoRA keys on a tiny CPU Qwen3, offsets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from valence.narrator import train_lora
from valence.narrator.lora_data import NarratorChatDataset, encode_row, turn_suffix
from valence.narrator.prompt import encode_text, render_chat

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/llm/Qwen3-1.7B-4bit"
DATA_DIR = ROOT / "data/narrator"


def _args(*argv: str):
    return train_lora.resolve_args(train_lora.build_parser().parse_args(list(argv)))


def test_defaults_match_the_training_plan():
    args = _args()
    assert (args.iters, args.batch_size, args.num_layers, args.learning_rate) == (600, 4, 16, 1e-4)
    assert (args.rank, args.scale, args.dropout) == (8, 20.0, 0.05)
    assert (args.steps_per_eval, args.val_batches, args.grad_checkpoint) == (100, -1, True)
    assert args.out == "adapters/narrator" and args.model == "models/llm/Qwen3-1.7B-4bit"
    assert args.max_seq_length >= 1024


def test_smoke_shrinks_the_run_and_explicit_flags_win():
    smoke = _args("--smoke")
    assert (smoke.iters, smoke.val_batches, smoke.out) == (10, 2, "adapters/narrator-smoke")
    assert smoke.steps_per_eval <= 5 and smoke.save_every >= smoke.iters
    custom = _args("--smoke", "--iters", "4", "--out", "adapters/x", "--batch-size", "2", "--num-layers", "8")
    assert (custom.iters, custom.out, custom.batch_size, custom.num_layers) == (4, "adapters/x", 2, 8)
    assert not _args("--no-grad-checkpoint").grad_checkpoint


def test_adapter_config_has_what_load_adapters_reads(tmp_path):
    config = train_lora.adapter_config(_args())
    assert config["fine_tune_type"] == "lora" and config["num_layers"] == 16
    assert config["lora_parameters"] == {"rank": 8, "scale": 20.0, "dropout": 0.05, "keys": list(train_lora.LORA_KEYS)}
    assert set(train_lora.LORA_KEYS) == {f"self_attn.{p}_proj" for p in "qkvo"} | {
        f"mlp.{p}_proj" for p in ("gate", "up", "down")}
    assert config["mask_prompt"] and config["prompt_style"] == "compact"
    json.dumps(config)  # serializable as adapter_config.json


def test_training_args_wiring(tmp_path):
    pytest.importorskip("mlx_lm.tuner.trainer")
    args = _args()
    targs = train_lora.training_args(args, tmp_path / "adapters.safetensors")
    assert (targs.batch_size, targs.iters, targs.steps_per_eval, targs.steps_per_save) == (4, 600, 100, 100)
    assert targs.grad_checkpoint and targs.val_batches == -1 and targs.max_seq_length == args.max_seq_length
    assert targs.adapter_file == str(tmp_path / "adapters.safetensors")


@pytest.fixture()
def tiny_qwen3():
    """A 4-layer Qwen3 with the real module names, on the CPU."""
    mx = pytest.importorskip("mlx.core")
    qwen3 = pytest.importorskip("mlx_lm.models.qwen3")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield lambda: qwen3.Model(qwen3.ModelArgs(
        model_type="qwen3", hidden_size=32, num_hidden_layers=4, intermediate_size=64, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=2, max_position_embeddings=128, rope_theta=1e6,
        head_dim=8, tie_word_embeddings=True))
    mx.set_default_device(previous)


def test_lora_keys_exist_and_round_trip_through_load_adapters(tiny_qwen3, tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

    import mlx.core as mx

    args = _args("--num-layers", "2")
    config = train_lora.adapter_config(args)
    model = tiny_qwen3()
    model.freeze()
    train_lora.check_lora_keys(model, train_lora.LORA_KEYS, args.num_layers)
    with pytest.raises(ValueError, match="no module"):
        train_lora.check_lora_keys(model, ("self_attn.qkv_proj",), 2)
    with pytest.raises(ValueError, match="model has 4"):
        train_lora.check_lora_keys(model, train_lora.LORA_KEYS, 5)

    linear_to_lora_layers(model, args.num_layers, config["lora_parameters"])
    converted = sorted(name for name, module in model.named_modules() if isinstance(module, LoRALinear))
    assert len(converted) == 2 * len(train_lora.LORA_KEYS)
    assert all(name.startswith(("model.layers.2.", "model.layers.3.")) for name in converted)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all(name.endswith(("lora_a", "lora_b")) for name in trainable)

    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    weights = {name: mx.ones_like(value) for name, value in trainable.items()}
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"), weights)
    reloaded = load_adapters(tiny_qwen3(), str(tmp_path))
    loaded = dict(tree_flatten(reloaded.parameters()))
    assert all(mx.array_equal(loaded[name], value).item() for name, value in weights.items())


def test_loss_log_keeps_the_curve():
    log = train_lora.LossLog()
    log.on_val_loss_report({"iteration": 0, "val_loss": 4.9, "val_time": 1.0})
    log.on_train_loss_report({"iteration": 10, "train_loss": 1.5, "tokens_per_second": 80.0, "peak_memory": 4.3,
                              "learning_rate": 1e-4})
    assert log.val == [{"iteration": 0.0, "val_loss": 4.9, "val_time": 1.0}]
    assert log.train[0]["train_loss"] == 1.5 and "learning_rate" not in log.train[0]


@pytest.fixture(scope="module")
def qwen():
    if not (MODEL_DIR / "tokenizer_config.json").exists():
        pytest.skip("Qwen3 tokenizer not downloaded")
    return pytest.importorskip("mlx_lm.tokenizer_utils").load(MODEL_DIR)


def test_dataset_offsets_and_length_checks(qwen):
    path = DATA_DIR / "valid.jsonl"
    if not path.exists():
        pytest.skip("LoRA data not built")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:12]]
    dataset = NarratorChatDataset(rows, qwen)
    suffix = turn_suffix(qwen)
    for row, (tokens, offset) in zip(rows, dataset):
        assert offset == len(encode_text(qwen, render_chat(qwen, row["messages"][:-1])))
        assert qwen.decode(tokens[offset:]) == row["messages"][-1]["content"] + suffix
        assert (tokens, offset) == encode_row(qwen, row["messages"])
    train_lora.check_lengths({"valid": dataset}, max_seq_length=1024, batch_size=4)
    with pytest.raises(ValueError, match="max_seq_length"):
        train_lora.check_lengths({"valid": dataset}, max_seq_length=600, batch_size=4)
    with pytest.raises(ValueError, match="batch size"):
        train_lora.check_lengths({"valid": dataset}, max_seq_length=1024, batch_size=64)
