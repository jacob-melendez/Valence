"""Offline dataset builder: simulator sessions -> (X, meta) using the same ``compute`` as live inference.

Training windows end every ``step`` ticks and are cut to a random history length (1-180 ticks) so the
model sees warm-up windows like the ones it meets live. The room window is the full room history 70% of
the time (a person sits down in a running system) and as short as the person's window otherwise (the
system itself just started).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from valence.features.compute import FEATURE_NAMES, MAX_TICKS, PERSON_FIELDS, ROOM_FIELDS, compute

ROOM = 0


def _session_windows(args) -> tuple[np.ndarray, pd.DataFrame]:
    sdf, step, seed, fixed_len, room_full_p = args
    rng = np.random.default_rng(seed)
    room_df = sdf[sdf["person_id"] == ROOM].sort_values("tick")
    room = {f: room_df[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
    room_tick_pos = {t: i for i, t in enumerate(room_df["tick"].to_numpy())}
    rows, meta = [], []
    for pid, pdf in sdf[sdf["person_id"] != ROOM].groupby("person_id"):
        pdf = pdf.sort_values("tick")
        person = {f: pdf[f].to_numpy(dtype=float) for f in PERSON_FIELDS}
        ticks = pdf["tick"].to_numpy()
        labels = pdf["true_state"].to_numpy()
        ages = pdf["episode_age_s"].to_numpy(dtype=float)
        start = int(rng.integers(0, step))
        for i in range(start, len(pdf), step):
            if fixed_len is None:
                length = int(rng.integers(1, MAX_TICKS + 1))
                room_len = MAX_TICKS if rng.random() < room_full_p else length
            else:
                length = room_len = fixed_len
            lo = max(0, i - length + 1)
            rp = room_tick_pos[ticks[i]]
            rlo = max(0, rp - room_len + 1)
            x = compute({f: v[lo:i + 1] for f, v in person.items()}, {f: v[rlo:rp + 1] for f, v in room.items()})
            rows.append(x)
            meta.append((sdf["session_id"].iat[0], int(pid), int(ticks[i]), labels[i], i + 1 - lo, ages[i]))
    X = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)))
    m = pd.DataFrame(meta, columns=["session_id", "person_id", "tick", "label", "hist_ticks", "episode_age_s"])
    return X, m


def build(df: pd.DataFrame, *, step: int = 6, seed: int = 0, fixed_len: int | None = None,
          room_full_p: float = 0.7, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (X as a DataFrame with FEATURE_NAMES columns, meta with labels and groups)."""
    sessions = [g for _, g in df.groupby("session_id", sort=True)]
    jobs = [(g, step, seed + i, fixed_len, room_full_p) for i, g in enumerate(sessions)]
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 4:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_session_windows, jobs, chunksize=4))
    else:
        parts = [_session_windows(j) for j in jobs]
    X = np.vstack([p[0] for p in parts])
    meta = pd.concat([p[1] for p in parts], ignore_index=True)
    return pd.DataFrame(X, columns=FEATURE_NAMES), meta
