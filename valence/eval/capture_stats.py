"""Compare live feature captures with the simulator, tick by tick (criterion B: variance match).

    python -m valence.eval.capture_stats data/real/capture_typing.jsonl:Focused data/real/capture_baseline.jsonl:Focused

Each capture is replayed through the real aggregator (the same 5 s TickRecords the model sees live); the
per-channel distribution and tick-to-tick spread are printed next to the simulator's rows for the state
the capture was acted as. Writes reports/capture_vs_sim.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from valence.config import Settings
from valence.runtime.aggregator import Aggregator
from valence.runtime.recording import iter_obs
from valence.sim.generator import simulate_corpus

CHANNELS = ["ambient_db", "db_room", "light_lux", "motion_energy", "posture_slump", "head_in_hands_frac",
            "clenched_fists_count", "laughter_count", "compliments_count", "face_valence", "face_visible", "n_people"]


def with_room_db(df: pd.DataFrame) -> pd.DataFrame:
    """dB relative to the venue baseline (what the served models see), next to the absolute level."""
    return df.assign(db_room=df["ambient_db"] - df["room_db_base"])


def capture_ticks(path: Path, tick_s: float = 5.0) -> pd.DataFrame:
    agg = Aggregator(Settings(), "capture")
    rows, boundary = [], None
    for obs in iter_obs(path):
        if boundary is None:
            boundary = obs.t + tick_s
        while obs.t >= boundary:
            rows += [r.to_dict() for r in agg.flush(boundary, boundary, 12.0)]
            boundary += tick_s
        agg.push(obs)
    df = with_room_db(pd.DataFrame(rows))
    return df.astype({c: float for c in CHANNELS if c in df}, errors="ignore")


def describe(values: pd.Series) -> dict:
    v = values.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0}
    diffs = np.abs(np.diff(v)) if v.size > 1 else np.array([math.nan])
    return {"n": int(v.size), "p10": round(float(np.percentile(v, 10)), 4), "p50": round(float(np.percentile(v, 50)), 4),
            "p90": round(float(np.percentile(v, 90)), 4), "tick_to_tick": round(float(np.nanmedian(diffs)), 4)}


def main() -> None:
    specs = [a.split(":") for a in sys.argv[1:]] or [["data/real/capture_typing.jsonl", "Focused"]]
    sim = simulate_corpus(30, 99)
    sim_people = with_room_db(sim[sim["person_id"] != 0])
    report = {}
    for path, state in specs:
        cap = capture_ticks(Path(path))
        person = cap[cap["person_id"] == 1]
        ref = sim_people[sim_people["true_state"] == state]
        report[Path(path).stem] = {"state": state, "ticks": int(len(person)),
                                   "channels": {c: {"live": describe(person[c]), "sim": describe(ref[c])} for c in CHANNELS}}
        print(f"\n{Path(path).stem} vs simulated {state} ({len(person)} live ticks)")
        print(f"  {'channel':20s} {'live p10/p50/p90 (step)':34s} {'sim p10/p50/p90 (step)'}")
        for c in CHANNELS:
            lv, sv = report[Path(path).stem]["channels"][c]["live"], report[Path(path).stem]["channels"][c]["sim"]
            fmt = lambda d: "-" if not d.get("n") else f"{d['p10']:.3f}/{d['p50']:.3f}/{d['p90']:.3f} ({d['tick_to_tick']:.3f})"
            print(f"  {c:20s} {fmt(lv):34s} {fmt(sv)}")
    out = Path("reports/capture_vs_sim.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
