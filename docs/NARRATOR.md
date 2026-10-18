# Narrator

The local narrator (Qwen3-1.7B 4-bit, mlx-lm 0.31.3, on-device) writes a two-to-three-sentence
audit note for each emotion prediction from its contract `FactSheet`. The LLM is fluent but not
trusted: `valence/narrator/validator.py` is the faithfulness gate for every explanation, and a
deterministic template is always the last resort, so every response is faithful and on time.

## Pipeline

```
FactSheet (valence.contracts, POST /narrate)
  -> prompt.render_fact_sheet: compact fact lines (person, state + confidence, face line,
     ranked drivers with scope/level/effect and the core's verbatim display, notes)
  -> prompt.build_messages: system prompt [+ 3 few-shot pairs in the few_shot style] + the sheet
  -> Qwen3 (thinking disabled; cached prefix; sampled attempt, 128/90-token budget)
  -> validator.validate(text, sheet): pass -> source "llm" (first try) / "llm_retry"
       fail -> retry with a "fix:" line built from the reasons (temp 0.7 -> 0.4 -> greedy)
       a draft cut by the token budget is salvaged only by dropping trailing sentences
  -> nothing passed before the 2.5 s deadline -> templates.render_template (source "template")
```

All generation runs on one worker thread (MLX is single-threaded); the caller waits up to the
deadline and renders the template in the remaining ~20 ms. Accepted texts are remembered per
situation and reused only after re-validation against the current sheet. Logs carry event id,
source, attempts and latency, never text or facts.

## Validator rules (brief)

Each rule yields a machine-readable reason that feeds the retry's `fix:` line (full table in the
module docstring):

- Shape: 20-90 words, at most 3 sentences, not empty after cleaning (think blocks, markdown).
- State: the predicted state is named and not negated; no other state is named; the face-only
  label is named only as what the face showed, and only with `mask_override`.
- Drivers: at least 2 of the top-3 key drivers are mentioned; nothing outside the top-5 allowed
  drivers; level words, above/below, time-of-day words agree with the sheet; contrast words
  ("despite", "although") only for opposing drivers, "drove"/"supports" only for supporting ones;
  room drivers are never attributed to a person.
- Numbers: every number is a sheet number at the text's precision and bound to the right driver;
  signs agree; SHAP values and layer contributions are not quotable.
- Safety: mask/smile language only with `mask_override`; no gendered, demographic, appearance or
  diagnostic terms; no unmeasured signals (heart rate, typing); people are only "Person #N" with
  the sheet's own number; "Collaborative" never for a person alone.

## Prompt cache

`MLXBackend` prefills one KV cache per standard state set (core, all) for the active prompt style
at load. Each request deep-copies the cache, trims it to the longest common token prefix and feeds
only the remaining ~150 sheet tokens, so the base cache is never mutated. Prefix sizes: few-shot
1234 / 1272 tokens, compact 515 / 541 (core / all). Warm-up runs sampled and greedy generations per
state set so Metal kernels are compiled before the first request; `/health` is 200 only after that.

## Stage A baseline: prompt-only (few-shot style, prefix cache, YOLO at 10 Hz)

| metric | 100 synthetic core sheets | 100 production sheets |
|---|---|---|
| final pass | 100% | 100% |
| LLM first-try | 85% | 61% |
| LLM any attempt | 94% | 83% |
| template fallback | 6% | 17% |
| latency p50 / p95 / max (ms) | 710 / 1751 / 2469 | 726 / 2109 / – |
| distinct-2 | 0.276 | 0.305 |
| hallucinated-number rate (raw attempts) | 0.106 | – |
| false-mask rate (raw attempts) | 0.08 | – |

Top attempt failures on production sheets: outside_top5 54, number 40, mask_term 37, sentences 34,
drivers 31, contrast 28, other_state 27. Production sheets are harder than the synthetic ones,
which is what motivated Stage B.

# Stage B: LoRA adapter

## Grammar targets (no LLM outputs)

`valence/narrator/grammar.py` writes each target deterministically from exactly what the prompt
shows the model (`prompt.render_fact_sheet`): person, `label_display`, `confidence_display`, the face
line, the listed drivers (scope/level/effect + verbatim display) and the out-of-range note.

- Openers: 11 plain + 4 per confidence band (>= 85% / 60-85% / < 60%).
- Drivers: 2-3 of the top-3 key drivers (sometimes one more supporting driver). Each display shape
  has noun-phrase and clause forms. Displays are copied verbatim or re-quoted with their exact
  numbers. A level word is added only when it agrees with the driver's level and the driver is the
  only allowed one of its concept. Room drivers are phrased as the room.
- Opposing drivers appear only as contrast ("although ... pulled the other way", "despite ...").
- A face clause appears only with `mask_override` and a visible face (when the face-only model read
  the predicted state itself, it says the face agrees rather than "context corrected the mask"). An
  occlusion clause appears always, and only, when the face is hidden; when the face was visible for
  at least half the window it is about the moment of the call, never "mostly hidden". With no face
  line in the prompt (rule 7) no face words appear at all: face-expression key drivers are skipped
  whenever two non-face key drivers exist (all but 6 of ~1,600 such sheets). "Upbeat" comes from
  `label_display`.
- Style guards: no avoid-line word anywhere in a target (the avoid line no longer lists "mask" when
  the state's own display is "Frustrated / Masked Stress"); no raw feature-name displays
  ("laugh_cnt_now = 0") as optional drivers; gerunds ("being slumped ...") only where they read well;
  "repeated" / "frequent" only for counts >= 2; at most one semicolon per sentence; the out-of-range
  note exactly once.
- Three orderings (opener-first, evidence-first, contrast/face-first) plus closers. Targets have
  2-3 sentences and 30-72 words.
- Every target passes `validate()`. Any target that does not is resampled, with the template as the
  last resort (never needed on the current data). `TrigramDeduper` rejects a target that shares more
  than 40% of its word trigrams with any single accepted target (1 valid row falls back to its best
  candidate, overlap 0.73).

## Dataset

```
.venv/bin/python -m valence.narrator.build_lora_data [--seed 0] [--dry-run]
```

The builder writes `data/narrator/{train,valid,test}.jsonl` (1500 / 150 / 300 rows). The sheets
come from two sources:

- the production export (`fact_sheets_sim.jsonl`; 1,150 of its 2,223 sheets are used);
- synthetic `make_batch` sheets for both state sets, renamed `syn-core-*` / `syn-all-*` so their
  ids never collide.

Train is balanced over state x {agree, mask override, hidden}: 90 per core cell; for the "all"
set, 40 per P1 cell and 15 per other cell. Test is 240 production + 60 synthetic, and its
synthetic part includes 30 Overwhelmed/Tense sheets. No event id or rendered prompt appears in
two splits. Rows are `{"messages": prompt.training_messages(sheet, target)}`, and test rows also
carry `"sheet"` for the A/B evaluation. The build is deterministic for a given seed.

Target corpus (seed 0): distinct-1 0.012 and distinct-2 0.075 over all 1,950 targets. Over
100-target samples, distinct-2 is 0.32 (the prompt-only LLM scored 0.28-0.31 on 100 sheets).
Targets average 54 words (30-72), 565 with 2 sentences and 1,385 with 3.

## Prompt styles

| style | used when | messages | cached prefix (Qwen3 tokens, core / all) |
|---|---|---|---|
| `few_shot` | no adapter | system + 3 few-shot pairs + sheet | 1234 / 1272 |
| `compact` | adapter loaded | system + sheet | 515 / 541 |

`Narrator` picks the style from adapter presence. `prompt_style=` or
`VALENCE_NARRATOR_PROMPT_STYLE` overrides it, and the `--prompt-style` flag on
`evaluate` / `service` does the same. Each style prefills its own prefix cache.

The training row and the inference prompt render identically as text: the template's render of a
row is the engine's compact prompt (thinking disabled, so it ends in an empty
`<think>\n\n</think>\n\n`), then the target, then `<|im_end|>\n`.

The compact style cuts the cached prefix about 2.4x and the whole prompt about 2.1x (1381 to 662
tokens, core). With the prefix cache on, the per-request live prefill is the same in both styles:
about 150 tokens of fact sheet.

## Training: use `lora_data.NarratorChatDataset`, not mlx-lm's `ChatDataset`

mlx-lm 0.31.3's `ChatDataset(mask_prompt=True)` computes the masked prompt with
`apply_chat_template(messages[:-1], add_generation_prompt=True)` and no `enable_thinking=False`
(its `TokenizerWrapper` even defaults `enable_thinking=True` for Qwen3). That prompt ends at
`<|im_start|>assistant\n`, so the 4 think-block tokens, which at inference are part of the prompt,
would be trained as completion (offset 698 instead of 702 on the first train row).

`valence/narrator/lora_data.py` builds `(tokens, offset)` items directly: `tokens =
encode_text(render_chat(tok, messages[:-1]) + target + turn_suffix(tok))` and `offset =
len(encode_text(render_chat(tok, messages[:-1])))`, using the engine's own `render_chat` and
`encode_text`. It raises on a BPE merge across the boundary or a template render that differs.
`tests/narrator/test_lora_data.py` checks, on 50 test rows with mlx-lm's tokenizer, that
`tokens[:offset]` equals `MLXBackend`'s prompt tokens and `tokens[offset:]` decodes to target +
`<|im_end|>\n`.

```python
from mlx_lm import load
from mlx_lm.tuner.trainer import train
from valence.narrator.lora_data import completion_loss, load_splits

model, tok = load("models/llm/Qwen3-1.7B-4bit")
data = load_splits("data/narrator", tok)          # {"train", "valid", "test"} -> NarratorChatDataset
train(model, optimizer, data["train"], data["valid"], args=..., loss=completion_loss)
```

Pass the datasets to `train` directly (they length-sort themselves through `__getitem__`).
`completion_loss` is `default_loss` with an exclusive upper bound: the stock mask
(`steps <= length`) also trains one padding token (id 0) after the final `\n`.

## Training run: `python -m valence.narrator.train_lora`

`train_lora.py` wires the above into `mlx_lm.tuner.trainer.train`: LoRA rank 8, scale 20.0,
dropout 0.05 on `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` of the last 16 of 28
blocks (4.98M trainable params, 0.29%), Adam at constant lr 1e-4, batch 4, 600 iterations,
gradient checkpointing, eval on the whole valid split every 100 iterations. `--smoke` runs 10
iterations into `adapters/narrator-smoke/`; `--iters`, `--out`, `--batch-size`, `--num-layers`
override. Output in `adapters/narrator/` (gitignored): `adapter_config.json`,
`adapters.safetensors` (load with `mlx_lm.load(model, adapter_path=...)` or
`Narrator(adapter_path="adapters/narrator")`), checkpoints every 100 iterations, `train_log.json`.

Run of 2026-10-17 (M3 Max, peak memory 4.39 GB): val loss 5.48 → 0.381 (100) → 0.282 (200) →
0.261 (300) → 0.246 (400) → 0.242 (500) → 0.239 (600); train loss 0.218 at 600. About 0.245 it/s
(~4.1 s/iteration, ~48 min for 600 including evals) on AC power; this run took 66 min because the
laptop dropped into low-power battery mode near the end.

With an adapter the engine uses the compact prompt and a 128-token budget per attempt
(`MAX_TOKENS_COMPACT`): adapter targets run up to ~118 tokens, so the few-shot budget of 90 would
cut about a quarter of them.

# A/B: what ships

`python -m valence.narrator.evaluate` runs one arm over fact sheets. `--sheets` accepts the dataset
rows of `data/narrator/test.jsonl` (it uses each row's stored `"sheet"`); `--deadline` and
`--max-tokens` override the engine defaults; every llm-mode report also records a greedy
decode-throughput probe before and after the run (`decode_tok_s`) and the power source
(`pmset -g batt`), so a throttled run is visible in the report itself.

```
.venv/bin/python scripts/spikes/gpu_load.py --seconds 2400 &      # camera-free YOLO load
.venv/bin/python -m valence.narrator.evaluate --n 300 --sheets data/narrator/test.jsonl --out A.json
.venv/bin/python -m valence.narrator.evaluate --n 300 --sheets data/narrator/test.jsonl \
    --adapter adapters/narrator --out B.json
.venv/bin/python -m valence.narrator.evaluate --n 300 --sheets data/narrator/test.jsonl --mode template --out C.json
```

## Results of 2026-10-17 (throttled laptop, see below)

The A/B phase ran with the laptop on battery at 3% -> 1% in low-power mode. Under that throttling
the GPU decoded 10-25 tokens/s (about 140 tokens/s unthrottled, measured with the same probe at
the start of the phase) and the YOLO load reached only 4.4 Hz instead of 10 Hz. At the real 2.5 s
deadline nothing can finish, so the faithfulness comparison was run with the deadline lifted to
60 s on a 60-sheet subset of the test split (every 5th row: 50 production + 10 synthetic; 23 mask,
14 hidden face, 23 visible), without the YOLO load. Latencies in that run are throttled and are
not deadline evidence.

| metric | A prompt-only (few-shot) | B LoRA (compact) | C template |
|---|---|---|---|
| sheets / deadline | 60 / 60 s | 60 / 60 s | 300 / 2.5 s |
| final pass | 100% | 100% | 100% |
| LLM first-try pass | 76.7% | **95.0%** | – |
| LLM any-attempt pass | 91.7% | **100%** | – |
| template fallback | 8.3% (3 attempts failed) | **0%** | 100% |
| hallucinated-number rate (raw attempts) | 0.122 | **0.000** | – |
| false-mask rate (raw unmasked attempts) | 0.160 | **0.000** | – |
| banned/demographic-term rate (raw attempts) | 0.000 | 0.000 | – |
| qualifier/contrast mismatch rate (raw attempts) | 0.146 | **0.016** | – |
| mean attempts per request | 1.37 | 1.07 | – |
| latency p50 / p95 / max (ms), throttled | 4700 / 14272 / 14551 | 7651 / 10906 / 16532 | 1 / 2 / 2 |
| decode probe (tokens/s, start / end) | 17.0 / 16.9 | 10.5 / 10.7 | – |
| distinct-1 / distinct-2 | 0.144 / 0.381 | 0.133 / 0.355 | 0.036 / 0.136 |
| mean words (final text) | 40.2 | 50.8 | 57.8 |

First-try by case (A / B): mask 73.9% / 95.7%, hidden face 85.7% / 100%, visible 73.9% / 91.3%.
Top raw-attempt failures: A sentences 23, outside_top5 23, number 16, contrast 12, drivers 10,
mask_term 8; B drivers 3, sentences 1, contrast 1.

The full 300-sheet arm A was also run at the real 2.5 s deadline with the YOLO load (4.4 Hz,
decode 16 tokens/s): 0/300 from the LLM, 300/300 template, p50 2494 / p95 2521 / max 2530 ms, i.e.
on the throttled machine every request times out and the template keeps the response on time and
faithful. Arm B at the real deadline was not run: under the same throttling it can only produce
the same all-template result, and the battery ran out. The two lifted-deadline arms ran at
different throttled speeds (17 vs 10.5 tokens/s), so their latencies are not comparable to each
other either.

## Decision: ship prompt-only (for now)

Ship rule: LoRA ships only if its final **and** first-try pass rates are >= prompt-only's **and**
its p95 latency is <= 2.5 s measured with YOLO running. The faithfulness half favours the
adapter (see the table), but the latency half could not be measured on this machine state, so by
the rule the shipped arm is **prompt-only** (`engine.SHIP = "prompt"`).

The switch is one constant, read by the service at startup:

- `valence/narrator/engine.py`: `SHIP = "prompt"` (or `"lora"`), `SHIPPED_ADAPTER = "adapters/narrator"`.
- `VALENCE_NARRATOR_SHIP=lora|prompt` overrides `SHIP` without a code change.
- With `lora`, the service loads `adapters/narrator` (compact prompt, 128-token budget) only if
  `adapters.safetensors` and `adapter_config.json` exist; otherwise it logs a warning and serves
  prompt-only. `adapters/` is gitignored, so a fresh clone always degrades to prompt-only.
- `--adapter PATH` / `VALENCE_NARRATOR_ADAPTER=PATH` override the ship choice; `--no-adapter` or
  `VALENCE_NARRATOR_ADAPTER=none` force prompt-only. `--template-only` skips the LLM.
- The service logs `narrator arm: prompt|lora (ship=..., adapter=..., prompt_style=...)` at startup
  and `/health` reports `adapter` and `prompt_style`.

To flip to LoRA: on AC power (not low-power mode), rerun arms A and B above with the YOLO load; if
B's `pass_rate` and `llm_first_try_pass_rate` are >= A's and B's `latency_ms.p95` <= 2500, set
`SHIP = "lora"` (or export `VALENCE_NARRATOR_SHIP=lora` for the demo).

```
.venv/bin/python -m valence.narrator.service                 # shipped arm (prompt-only today)
VALENCE_NARRATOR_SHIP=lora .venv/bin/python -m valence.narrator.service   # LoRA arm
bash scripts/spikes/run_demo.sh                               # demo launcher uses the shipped arm
```

## Limitations

- Deadline latency for the adapter is unmeasured. The only unthrottled adapter numbers come from
  the smoke run (first try ~1 s). The compact prompt prefills the same ~150 sheet tokens as the
  few-shot style, but adapter outputs are longer (median ~80 tokens, budget 128 vs 90), so its
  generation time per attempt is somewhat higher than prompt-only's.
- The lifted-deadline comparison is 60 sheets, not 300, and without the YOLO load; first-try
  rates on 60 sheets carry about +/-6 points of sampling noise.
- Grammar targets are deterministic templates with variation, not human writing; distinct-n of
  the adapter is bounded by the grammar's variety, and the adapter may sound more uniform than
  the few-shot model over a long session.
- The validator is the faithfulness authority, so every faithfulness metric here is "passes the
  validator". A gap seen once in the 10-iteration smoke run ("context corrected the facial mask"
  on a hidden-face sheet without `mask_override` passed) was not seen after full training, but it
  is a validator gap, not something training fixes.
- Template fallback is always faithful but repetitive (distinct-2 0.136 on the test split).
