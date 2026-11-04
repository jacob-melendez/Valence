"""Offline dataset for the emotion map: simulator sessions -> the 123 frozen features + the 31 emotion features.

Same windowing as ``build_dataset.build`` (windows every ``step`` ticks, random 1-180 tick history, the room window
full 70% of the time), plus the emotion features from the same person window and the true valence/arousal targets.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from valence.features.compute import FEATURE_NAMES, MAX_TICKS, PERSON_FIELDS, ROOM_FIELDS, compute
from valence.features.emotion import EMOTION_FEATURE_NAMES, EMOTION_FIELDS, compute_emotion

ROOM = 0
META_COLUMNS = ["session_id", "person_id", "tick", "label", "valence", "arousal", "hist_ticks", "episode_age_s"]


def _session_windows(args) -> tuple[np.ndarray, pd.DataFrame]:
    sdf, step, seed, room_full_p = args
    rng = np.random.default_rng(seed)
    room_df = sdf[sdf["person_id"] == ROOM].sort_values("tick")
    room = {f: room_df[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
    room_tick_pos = {t: i for i, t in enumerate(room_df["tick"].to_numpy())}
    fields = sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS))
    rows, meta = [], []
    for pid, pdf in sdf[sdf["person_id"] != ROOM].groupby("person_id"):
        pdf = pdf.sort_values("tick")
        person = {f: pdf[f].to_numpy(dtype=float) for f in fields}
        ticks = pdf["tick"].to_numpy()
        labels = pdf["true_state"].to_numpy()
        val, aro = pdf["true_valence"].to_numpy(dtype=float), pdf["true_arousal"].to_numpy(dtype=float)
        ages = pdf["episode_age_s"].to_numpy(dtype=float)
        start = int(rng.integers(0, step))
        for i in range(start, len(pdf), step):
            length = int(rng.integers(1, MAX_TICKS + 1))
            room_len = MAX_TICKS if rng.random() < room_full_p else length
            lo = max(0, i - length + 1)
            rp = room_tick_pos[ticks[i]]
            rlo = max(0, rp - room_len + 1)
            pw = {f: v[lo:i + 1] for f, v in person.items()}
            x = compute({f: pw[f] for f in PERSON_FIELDS}, {f: v[rlo:rp + 1] for f, v in room.items()})
            rows.append(np.concatenate([x, compute_emotion(pw)]))
            meta.append((sdf["session_id"].iat[0], int(pid), int(ticks[i]), labels[i], val[i], aro[i], i + 1 - lo,
                         ages[i]))
    X = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES) + len(EMOTION_FEATURE_NAMES)))
    return X, pd.DataFrame(meta, columns=META_COLUMNS)


def build_emotion(df: pd.DataFrame, *, step: int = 6, seed: int = 0, room_full_p: float = 0.7,
                  workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (X with FEATURE_NAMES + EMOTION_FEATURE_NAMES columns, meta with labels, targets and groups)."""
    sessions = [g for _, g in df.groupby("session_id", sort=True)]
    jobs = [(g, step, seed + i, room_full_p) for i, g in enumerate(sessions)]
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 4:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_session_windows, jobs, chunksize=4))
    else:
        parts = [_session_windows(j) for j in jobs]
    X = np.vstack([p[0] for p in parts])
    meta = pd.concat([p[1] for p in parts], ignore_index=True)
    return pd.DataFrame(X, columns=FEATURE_NAMES + EMOTION_FEATURE_NAMES), meta
