"""QLoRA-specialize the local narrator model on the grammar dataset (mlx_lm tuner, Apple GPU).

    python -m valence.narrator.train_lora [--smoke] [--iters 600] [--out adapters/narrator] \\
        [--batch-size 4] [--num-layers 16] [--model models/llm/Qwen3-1.7B-4bit] [--data data/narrator]

Mirrors ``mlx_lm.lora.train_model`` with two deliberate differences:

- data comes from :mod:`valence.narrator.lora_data` (``NarratorChatDataset``), so the masked prompt
  is exactly the engine's compact inference prompt, empty ``<think></think>`` block included, and
  only the target plus ``<|im_end|>\\n`` is trained (``completion_loss``, no trailing pad token);
  the datasets go to ``trainer.train`` directly, not through ``CacheDataset`` / the stock CLI;
- the LoRA keys are explicit (all attention and MLP projections) and checked against the loaded
  model before any layer is converted.

The output directory holds ``adapter_config.json`` + ``adapters.safetensors`` (what
``mlx_lm.load(model, adapter_path=...)`` and ``Narrator(adapter_path=...)`` read), the trainer's
``NNNNNNN_adapters.safetensors`` checkpoints, and ``train_log.json`` (loss curve, wall time, peak
memory). ``--smoke`` runs 10 iterations with a tiny eval to check the wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from valence.narrator.lora_data import NarratorChatDataset, completion_loss, load_splits

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "models/llm/Qwen3-1.7B-4bit"
DEFAULT_DATA = "data/narrator"
DEFAULT_OUT = "adapters/narrator"
SMOKE_OUT = "adapters/narrator-smoke"
LORA_KEYS: tuple[str, ...] = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
SMOKE_ITERS = 10
SMOKE_VAL_BATCHES = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA-train the Valence narrator adapter.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local mlx model directory")
    parser.add_argument("--data", default=DEFAULT_DATA, help="directory with train.jsonl and valid.jsonl")
    parser.add_argument("--out", default=None, help=f"adapter directory (default {DEFAULT_OUT}, {SMOKE_OUT} with --smoke)")
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_ITERS} iterations and a tiny eval")
    parser.add_argument("--iters", type=int, default=None, help="iterations (default 600, 10 with --smoke)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=16, help="last N transformer blocks get LoRA")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps-per-eval", type=int, default=100)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=-1, help="-1 = the whole valid split")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the ``--smoke`` defaults (explicit flags still win)."""
    if args.iters is None:
        args.iters = SMOKE_ITERS if args.smoke else 600
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    if args.smoke:
        args.val_batches = SMOKE_VAL_BATCHES if args.val_batches == -1 else args.val_batches
        args.steps_per_eval = min(args.steps_per_eval, max(1, args.iters // 2))
        args.steps_per_report = min(args.steps_per_report, 2)
        args.save_every = max(args.save_every, args.iters)
    return args


def _repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() or candidate.exists() else REPO_ROOT / candidate


def adapter_config(args: argparse.Namespace) -> dict[str, Any]:
    """``adapter_config.json``: the keys ``mlx_lm.tuner.utils.load_adapters`` reads plus provenance."""
    return {
        "model": args.model,
        "data": args.data,
        "fine_tune_type": "lora",
        "num_layers": args.num_layers,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": args.dropout, "keys": list(LORA_KEYS)},
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "steps_per_eval": args.steps_per_eval,
        "val_batches": args.val_batches,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": args.grad_checkpoint,
        "mask_prompt": True,
        "loss": "valence.narrator.lora_data.completion_loss",
        "dataset": "valence.narrator.lora_data.NarratorChatDataset",
        "prompt_style": "compact",
        "seed": args.seed,
    }


def training_args(args: argparse.Namespace, adapter_file: Path) -> Any:
    from mlx_lm.tuner.trainer import TrainingArgs

    return TrainingArgs(
        batch_size=args.batch_size, iters=args.iters, val_batches=args.val_batches,
        steps_per_report=args.steps_per_report, steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every, adapter_file=str(adapter_file), max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
    )


def check_lora_keys(model: Any, keys: Sequence[str], num_layers: int) -> None:
    """Every key must name a linear module in each of the last ``num_layers`` blocks."""
    layers = model.layers
    if num_layers > len(layers):
        raise ValueError(f"requested {num_layers} LoRA layers but the model has {len(layers)}")
    for index, layer in enumerate(layers[-max(num_layers, 0):], start=len(layers) - num_layers):
        names = dict(layer.named_modules())
        missing = [key for key in keys if key not in names]
        if missing:
            raise ValueError(f"layer {index} has no module(s) {missing}; available: {sorted(names)}")


def check_lengths(datasets: dict[str, NarratorChatDataset], max_seq_length: int, batch_size: int) -> None:
    """No row may be truncated (that would cut the target), and each split must fill a batch."""
    for name, data in datasets.items():
        stats = data.stats()
        if stats["max_tokens"] > max_seq_length:
            raise ValueError(f"{name}: a row has {stats['max_tokens']} tokens > max_seq_length {max_seq_length}")
        if stats["n"] < batch_size:
            raise ValueError(f"{name}: {stats['n']} rows < batch size {batch_size}")


class LossLog:
    """mlx_lm ``TrainingCallback`` that keeps the loss curve (duck-typed, so no import at module load)."""

    def __init__(self) -> None:
        self.train: list[dict[str, float]] = []
        self.val: list[dict[str, float]] = []

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        self.train.append({k: float(info[k]) for k in ("iteration", "train_loss", "tokens_per_second", "peak_memory")})

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        self.val.append({"iteration": float(info["iteration"]), "val_loss": float(info["val_loss"]),
                         "val_time": float(info["val_time"])})


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train and save the adapter; returns the summary that is also written to ``train_log.json``."""
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.trainer import train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    started = time.monotonic()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    out = _repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(str(_repo_path(args.model)))
    datasets = load_splits(_repo_path(args.data), tokenizer)
    for name in ("train", "valid"):
        if name not in datasets:
            raise FileNotFoundError(f"{name}.jsonl missing in {args.data}")
    datasets.pop("test", None)
    check_lengths(datasets, args.max_seq_length, args.batch_size)
    stats = {name: data.stats() for name, data in datasets.items()}
    print(f"datasets: {json.dumps(stats)}", flush=True)

    config = adapter_config(args)
    model.freeze()
    check_lora_keys(model, LORA_KEYS, args.num_layers)
    linear_to_lora_layers(model, args.num_layers, config["lora_parameters"])
    print_trainable_parameters(model)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=4, sort_keys=True) + "\n", encoding="utf-8")

    log = LossLog()
    mx.reset_peak_memory()
    train_started = time.monotonic()
    train(model=model, optimizer=optim.Adam(learning_rate=args.learning_rate),
          train_dataset=datasets["train"], val_dataset=datasets["valid"],
          args=training_args(args, out / "adapters.safetensors"), loss=completion_loss, training_callback=log)
    summary = {
        "config": config,
        "datasets": stats,
        "wall_s": round(time.monotonic() - started, 1),
        "train_s": round(time.monotonic() - train_started, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        "train": log.train,
        "val": log.val,
        "adapter_path": str(out),
    }
    (out / "train_log.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    summary = run(args)
    val = ", ".join(f"{int(v['iteration'])}:{v['val_loss']:.3f}" for v in summary["val"])
    print(f"done in {summary['wall_s']} s (train {summary['train_s']} s), peak {summary['peak_memory_gb']} GB; "
          f"val loss {val}; adapter at {summary['adapter_path']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
