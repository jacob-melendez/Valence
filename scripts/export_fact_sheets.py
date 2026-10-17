"""Export realistic narrator fact sheets from the served model -> data/narrator/fact_sheets_sim.jsonl.

Runs fresh simulator sessions (a seed not used for training) and every scenario through the real
runtime, captures each FactSheet the runtime would send to the narrator, then balances them across
states x face condition (agrees / disagrees / hidden) so the LoRA sees every combination.

    python scripts/export_fact_sheets.py --sessions 40 --per-cell 120
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from valence.config import Settings
from valence.governance.purge import PurgeGate
from valence.models.served import load_served
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime
from valence.sim.generator import simulate_session
from valence.sim.scenarios import list_scenarios, render


class Capture:
    def __init__(self) -> None:
        self.sheets = []

    def request(self, sheet, person_id) -> None:
        self.sheets.append(sheet)


def face_condition(sheet) -> str:
    if not sheet.face_visible or sheet.face_only.label is None:
        return "hidden"
    return "agrees" if sheet.face_only.label == sheet.label else "disagrees"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--per-cell", type=int, default=120)
    ap.add_argument("--state-set", default="core")
    ap.add_argument("--out", default="data/narrator/fact_sheets_sim.jsonl")
    args = ap.parse_args()
    served = load_served(Path("artifacts"), args.state_set)
    if served is None:
        raise SystemExit("train the model first: python -m valence.models.train_ladder")
    cap = Capture()
    rt = Runtime(Settings(state_set=args.state_set), EventBus(), served, gate=PurgeGate(), narrator=cap,
                 audit_dir=tempfile.mkdtemp())
    root = np.random.SeedSequence(args.seed)
    runs = [(f"sim-{i}", simulate_session(f"fs{i}", np.random.default_rng(c), state_set=args.state_set))
            for i, c in enumerate(root.spawn(args.sessions))]
    runs += [(s["name"], render(s["name"], state_set=args.state_set)) for s in list_scenarios()
             if args.state_set == "all" or s["name"] not in ("overwhelmed", "tense")]
    for name, df in runs:
        rt.use_scenario(df, name)
        rt.paused = False
        for _ in range(df["tick"].nunique()):
            rt.step()
    rt.close()
    cells = defaultdict(list)
    for sheet in cap.sheets:
        cells[(sheet.label, face_condition(sheet), sheet.mask_override)].append(sheet)
    rnd = random.Random(args.seed)
    chosen = []
    for key, sheets in sorted(cells.items()):
        rnd.shuffle(sheets)
        chosen += sheets[: args.per_cell]
    rnd.shuffle(chosen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sheet in chosen:
            f.write(sheet.model_dump_json() + "\n")
    print(f"captured {len(cap.sheets)} sheets; wrote {len(chosen)} balanced sheets to {out}")
    for key, sheets in sorted(cells.items()):
        print(f"  {key[0]:20s} face {key[1]:9s} mask_override={key[2]!s:5s} available={len(sheets):5d} kept={min(len(sheets), args.per_cell)}")


if __name__ == "__main__":
    main()
