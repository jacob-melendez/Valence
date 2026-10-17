"""Spike 6: mlx_lm generation speed for the narrator LLM (thinking disabled).

    python scripts/spikes/bench_llm.py [--model models/llm/Qwen3-1.7B-4bit] [--runs 5]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = ("You write one short audit explanation (20-90 words, at most 3 sentences) for an emotion-inference "
          "prediction. Use only the facts given; refer to the person as Person #1 or they.")
USER = ("State: Frustrated (Masked), confidence 88%. Top drivers: Ambient noise 72 dB (Ambient Physics), "
        "Clenched-fist onsets 3 (Somatic & Social), Time since last break 4.2 hours (Time Dynamics). "
        "Face-only baseline: Collaborative_Happy 71%, facial valence 0.62. mask_override: true.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/llm/Qwen3-1.7B-4bit"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=110)
    a = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load(a.model)
    print(f"model {Path(a.model).name}: load {time.perf_counter() - t0:.2f} s")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    print(f"template ends with: {prompt[-40:]!r}")
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
    procs = make_logits_processors(repetition_penalty=1.05)

    gen_tps, prompt_tps, wall, ntoks = [], [], [], []
    for i in range(a.runs + 1):
        t0 = time.perf_counter()
        text, last = "", None
        for last in stream_generate(model, tok, prompt, max_tokens=a.max_tokens, sampler=sampler, logits_processors=procs):
            text += last.text
        dt = time.perf_counter() - t0
        if i == 0:
            print(f"warm-up {dt * 1e3:.0f} ms; sample output ({len(text.split())} words):\n  {text.strip()}\n")
            continue
        gen_tps.append(last.generation_tps); prompt_tps.append(last.prompt_tps)
        wall.append(dt * 1e3); ntoks.append(last.generation_tokens)
    print(f"prompt tokens {last.prompt_tokens}: prompt {statistics.median(prompt_tps):.0f} tok/s")
    print(f"generation {statistics.median(gen_tps):.1f} tok/s; {statistics.median(ntoks)} tokens; "
          f"wall p50 {statistics.median(wall):.0f} ms, max {max(wall):.0f} ms; peak mem {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
