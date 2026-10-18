"""The original mock dataset generator, kept to show why it was rebuilt.

Rows are independent draws (``random.choice`` per row) with non-overlapping context ranges, so a tree
on the context columns scores ~100% and the 15-minute window has nothing to exploit. Ported from
the team's original script (see docs/DATASET.md) with a seed added for reproducibility.
"""

from __future__ import annotations

import random

import pandas as pd

STATES = ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"]


def generate_legacy(num_records: int = 500, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for i in range(num_records):
        vibe = rnd.choice(STATES)
        if vibe == "Frustrated_Masked":
            row = (rnd.uniform(68, 85), rnd.uniform(150, 300), rnd.uniform(3.5, 6.0), rnd.randint(1, 4), 0,
                   rnd.choice([0, 1]), rnd.uniform(0.4, 0.8))
        elif vibe == "Exhausted":
            row = (rnd.uniform(35, 50), rnd.uniform(100, 200), rnd.uniform(5.0, 8.0), 0, 0, 1, rnd.uniform(-0.2, 0.1))
        elif vibe == "Collaborative_Happy":
            row = (rnd.uniform(60, 75), rnd.uniform(400, 600), rnd.uniform(0.2, 1.5), 0, rnd.randint(1, 3), 0,
                   rnd.uniform(0.5, 0.9))
        else:
            row = (rnd.uniform(40, 55), rnd.uniform(350, 500), rnd.uniform(1.0, 3.0), 0, 0, 0, rnd.uniform(-0.4, 0.0))
        db, lux, hrs, fists, compl, withdraw, face = row
        rows.append({
            "timestamp": i * 5, "ambient_db": round(db, 1), "light_lux": round(lux, 1),
            "hours_since_break": round(hrs, 2), "clenched_fists_count": fists, "compliments_count": compl,
            "social_withdrawal_flag": withdraw, "facial_valence_snapshot": round(face, 2),
            "true_emotional_state": vibe,
        })
    return pd.DataFrame(rows)
