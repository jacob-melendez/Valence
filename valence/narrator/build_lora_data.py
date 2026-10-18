"""Build the narrator LoRA dataset from the deterministic grammar (no LLM outputs anywhere).

    python -m valence.narrator.build_lora_data [--seed 0] [--production data/narrator/fact_sheets_sim.jsonl] \\
        [--out-dir data/narrator] [--max-overlap 0.4] [--dry-run]

Sheets: the core's production export (core state set) plus synthetic ``make_batch`` sheets for both
state sets ("core" and "all", so the P1 states Overwhelmed and Tense appear), re-identified
``syn-core-NNNNN`` / ``syn-all-NNNNN`` so event ids never collide. Every sheet falls in a cell
(state set, label, face case) with face case ``agree`` (face visible, no override), ``mask``
(``mask_override`` with a visible face) or ``hidden`` (face not visible, override latched or not).

Splits (no event id, and no rendered prompt, in two splits):

- test 300: 240 production sheets (100 agree / 80 hidden / 60 mask, labels round-robin) plus 30
  synthetic core sheets over the core cells and 30 synthetic P1 sheets (Overwhelmed / Tense);
- valid 150: 90 production (30 per face case) plus 30 synthetic core and 30 synthetic "all";
- train 1500: balanced cells, production first and synthetic to fill: 90 per core cell (12 cells)
  and, for the "all" set, 40 per P1 cell and 15 per other cell.

Rows are chat JSONL ``{"messages": prompt.training_messages(sheet, target)}`` (exactly the
compact-style inference messages plus the target); test rows also carry ``"sheet"`` for the A/B
evaluation. Every target passes ``validate()`` (checked again before writing) and targets are
trigram-deduplicated across the whole dataset. Prints a balance table and corpus statistics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from valence.contracts import STATE_SETS
from valence.narrator.evaluate import distinct_n
from valence.narrator.fake_fact_sheets import make_batch
from valence.narrator.grammar import realize_many
from valence.narrator.prompt import render_fact_sheet, training_messages
from valence.narrator.schema import face_hidden, parse_fact_sheet, sheet_states, state_set_name
from valence.narrator.validator import count_words, sentence_spans, validate

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = REPO_ROOT / "data/narrator/fact_sheets_sim.jsonl"
OUT_DIR = REPO_ROOT / "data/narrator"
CASES: tuple[str, ...] = ("agree", "mask", "hidden")
P1_STATES: tuple[str, ...] = tuple(s for s in STATE_SETS["all"] if s not in STATE_SETS["core"])
N_SYNTHETIC = {"core": 1500, "all": 1200}

Cell = tuple[str, str, str]  # (state set, label, face case)


def face_case(sheet: Mapping[str, Any]) -> str:
    if face_hidden(sheet):
        return "hidden"
    return "mask" if sheet.get("mask_override") else "agree"


def cell_of(sheet: Mapping[str, Any]) -> Cell:
    return state_set_name(sheet_states(sheet)), str(sheet["label"]), face_case(sheet)


def is_production(sheet: Mapping[str, Any]) -> bool:
    return not str(sheet.get("event_id", "")).startswith("syn-")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def synthetic_sheets(seed: int) -> list[dict[str, Any]]:
    """Balanced synthetic sheets for both state sets, with collision-free event ids."""
    out: list[dict[str, Any]] = []
    for offset, (states, n) in enumerate(N_SYNTHETIC.items()):
        for i, sheet in enumerate(make_batch(n, seed=1000 * seed + 17 + offset, states=states)):
            out.append({**sheet, "event_id": f"syn-{states}-{i:05d}"})
    return out


class Pool:
    """Unused sheets by cell; ``take`` hands each sheet out at most once."""

    def __init__(self, sheets: Iterable[dict[str, Any]], rng: random.Random) -> None:
        self.cells: dict[Cell, list[dict[str, Any]]] = defaultdict(list)
        for sheet in sheets:
            self.cells[cell_of(sheet)].append(sheet)
        for cell in sorted(self.cells):
            rng.shuffle(self.cells[cell])

    def take(self, cells: Sequence[Cell], n: int) -> list[dict[str, Any]]:
        """``n`` sheets round-robin over ``cells`` (cells that run dry are skipped)."""
        out: list[dict[str, Any]] = []
        live = [c for c in cells if self.cells.get(c)]
        while len(out) < n and live:
            for cell in list(live):
                if len(out) == n:
                    break
                out.append(self.cells[cell].pop())
                if not self.cells[cell]:
                    live.remove(cell)
        return out

    def fill(self, cell: Cell, n: int) -> list[dict[str, Any]]:
        return self.take([cell], n)


def _cells(state_set: str, labels: Sequence[str] | None = None, cases: Sequence[str] = CASES) -> list[Cell]:
    return [(state_set, label, case) for case in cases for label in (labels or STATE_SETS[state_set])]


def build_splits(production: Sequence[dict[str, Any]], synthetic: Sequence[dict[str, Any]], seed: int = 0,
                 n_train: int = 1500, n_valid: int = 150, n_test: int = 300) -> dict[str, list[dict[str, Any]]]:
    """Disjoint test / valid / train sheet lists (see the module docstring for the recipe)."""
    rng = random.Random(f"valence-lora-split:{seed}")
    prod, syn = Pool(production, rng), Pool(synthetic, rng)
    core_labels = STATE_SETS["core"]

    def prod_by_case(counts: Mapping[str, int]) -> list[dict[str, Any]]:
        return [s for case, n in counts.items() for s in prod.take(_cells("core", core_labels, (case,)), n)]

    test = prod_by_case({"agree": 100, "hidden": 80, "mask": 60})
    test += syn.take(_cells("core"), 30) + syn.take(_cells("all", P1_STATES), n_test - len(test) - 30)
    valid = prod_by_case({"agree": 30, "hidden": 30, "mask": 30})
    valid += syn.take(_cells("core"), 30) + syn.take(_cells("all"), n_valid - len(valid) - 30)

    quota: dict[Cell, int] = {cell: 90 for cell in _cells("core")}
    quota.update({cell: (40 if cell[1] in P1_STATES else 15) for cell in _cells("all")})
    scale = n_train / sum(quota.values())
    train: list[dict[str, Any]] = []
    for cell, n in quota.items():
        n = round(n * scale)
        picked = prod.fill(cell, n) if cell[0] == "core" else []
        train += picked + syn.fill(cell, n - len(picked))
    if len(train) < n_train:  # a starved cell: top up from the largest remaining production cells
        train += prod.take(sorted(prod.cells, key=lambda c: -len(prod.cells[c])), n_train - len(train))
    train = train[:n_train]
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    splits = {"train": train, "valid": valid, "test": test}
    check_disjoint(splits)
    return splits


def check_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """No event id and no rendered prompt appears in two splits (or twice in one)."""
    seen_ids: dict[str, str] = {}
    seen_prompts: dict[str, str] = {}
    for name, sheets in splits.items():
        for sheet in sheets:
            event_id, prompt = str(sheet["event_id"]), render_fact_sheet(sheet)
            if event_id in seen_ids:
                raise ValueError(f"event id {event_id} in both {seen_ids[event_id]} and {name}")
            if prompt in seen_prompts:
                raise ValueError(f"identical prompt in {seen_prompts[prompt]} and {name} ({event_id})")
            seen_ids[event_id], seen_prompts[prompt] = name, name


def rows_for(split: str, sheets: Sequence[dict[str, Any]], targets: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for sheet, target in zip(sheets, targets, strict=True):
        result = validate(target, sheet)
        if not result.ok:
            raise ValueError(f"target for {sheet['event_id']} fails validation: {result.reasons}")
        row: dict[str, Any] = {"messages": training_messages(sheet, target)}
        if split == "test":
            row["sheet"] = sheet
        rows.append(row)
    return rows


def balance_table(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Counts per (state set, label) x face case for each split; "n/p" = total / production."""
    lines = [f"{'split':<6} {'set':<5} {'label':<20} " + " ".join(f"{c:>9}" for c in CASES) + f" {'total':>9}"]
    for name, sheets in splits.items():
        counts: Counter[Cell] = Counter(cell_of(s) for s in sheets)
        prods: Counter[Cell] = Counter(cell_of(s) for s in sheets if is_production(s))
        for state_set in STATE_SETS:
            for label in STATE_SETS[state_set]:
                cells = [(state_set, label, case) for case in CASES]
                if not any(counts[c] for c in cells):
                    continue
                parts = [f"{counts[c]}/{prods[c]}" for c in cells]
                total = f"{sum(counts[c] for c in cells)}/{sum(prods[c] for c in cells)}"
                lines.append(f"{name:<6} {state_set:<5} {label:<20} " + " ".join(f"{p:>9}" for p in parts)
                             + f" {total:>9}")
        n_prod = sum(is_production(s) for s in sheets)
        lines.append(f"{name:<6} {'all':<5} {'(total n/production)':<20} "
                     + " ".join(f"{sum(v for c, v in counts.items() if c[2] == case):>9}" for case in CASES)
                     + f" {f'{len(sheets)}/{n_prod}':>9}")
    return "\n".join(lines)


def corpus_stats(targets: Sequence[str], seed: int = 0) -> dict[str, Any]:
    """distinct-1/2 over the corpus and averaged over random 100-text samples (the eval's batch size)."""
    rng = random.Random(seed)
    samples = [rng.sample(list(targets), min(100, len(targets))) for _ in range(20)]
    words = [count_words(t) for t in targets]
    return {
        "n": len(targets),
        "distinct_1": distinct_n(targets, 1),
        "distinct_2": distinct_n(targets, 2),
        "distinct_2_per_100": round(sum(distinct_n(s, 2) for s in samples) / len(samples), 4),
        "words_mean": round(sum(words) / len(words), 1),
        "words_min": min(words),
        "words_max": max(words),
        "sentences": dict(sorted(Counter(len(sentence_spans(t)) for t in targets).items())),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(seed: int = 0, production_path: str | Path = PRODUCTION, max_overlap: float = 0.4,
          ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], dict[str, Any]]:
    """Sheets, targets and stats per split (nothing written)."""
    production = load_jsonl(production_path)
    synthetic = synthetic_sheets(seed)
    for sheet in synthetic:
        parse_fact_sheet(sheet)  # contract-valid with the new ids
    splits = build_splits(production, synthetic, seed)
    ordered = [s for name in ("train", "valid", "test") for s in splits[name]]
    targets, dedupe = realize_many(ordered, seed=seed, max_overlap=max_overlap)
    out: dict[str, list[str]] = {}
    start = 0
    for name in ("train", "valid", "test"):
        out[name] = targets[start:start + len(splits[name])]
        start += len(splits[name])
    stats = {"all": corpus_stats(targets, seed), **{name: corpus_stats(t, seed) for name, t in out.items()},
             "dedupe": dedupe, "production_used": sum(is_production(s) for s in ordered),
             "production_total": len(production)}
    return splits, out, stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the narrator LoRA chat dataset from the grammar.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--production", default=str(PRODUCTION))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-overlap", type=float, default=0.4,
                        help="reject a target sharing more than this share of its trigrams with any accepted one")
    parser.add_argument("--dry-run", action="store_true", help="print the tables, write nothing")
    args = parser.parse_args(argv)

    splits, targets, stats = build(args.seed, args.production, args.max_overlap)
    if not args.dry_run:
        out_dir = Path(args.out_dir)
        for name in ("train", "valid", "test"):
            write_jsonl(out_dir / f"{name}.jsonl", rows_for(name, splits[name], targets[name]))
    sys.stdout.write(balance_table(splits) + "\n\n" + json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
