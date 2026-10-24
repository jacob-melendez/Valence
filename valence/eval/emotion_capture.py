"""Real captures through the emotion map: face-cue distributions vs the simulator, and what the model reads.

    python -m valence.eval.emotion_capture rec.jsonl:Focused [rec2.jsonl:Bored ...] [--out report.json]

Each recording is a features-only observation JSONL (``obs_to_dict`` lines from the Python edge or the browser
perception, PersonObs carrying the emotion-v1 cues). It is replayed through the real aggregator (5 s TickRecords),
then through the emotion features and models, exactly as live. Reported per recording: p10/p50/p90 and the
tick-to-tick step of each cue channel next to the simulator's rows for the acted state, the share of ticks per
predicted state, agreement with the acted state, and the mean map position. Nothing here reads or writes media.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from valence.contracts import AFFECT_ANCHORS
from valence.eval.capture_stats import capture_ticks, describe
from valence.features.compute import PERSON_FIELDS, ROOM_FIELDS, compute
from valence.features.emotion import EMOTION_FIELDS, compute_emotion, model_input
from valence.sim.generator import simulate_corpus

CUES = ["brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away", "blink_count",
        "face_valence", "face_visible"]


def read_capture(df, predictor) -> dict:
    """Per-tick emotion readings for person 1 of an aggregated capture (oldest first)."""
    person = df[df["person_id"] == 1].sort_values("tick")
    room = df[df["person_id"] == 0].sort_values("tick")
    if person.empty:
        return {"ticks": 0}
    fields = sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS))
    parr = {f: person[f].to_numpy(dtype=float) for f in fields}
    rarr = {f: room[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
    rpos = {t: i for i, t in enumerate(room["tick"].to_numpy())}
    rows = []
    for i, t in enumerate(person["tick"].to_numpy()):
        lo, rp = max(0, i - 179), rpos[t]
        rows.append(model_input(compute({f: parr[f][lo:i + 1] for f in PERSON_FIELDS},
                                        {f: v[max(0, rp - 179):rp + 1] for f, v in rarr.items()}),
                                compute_emotion({f: parr[f][lo:i + 1] for f in EMOTION_FIELDS})))
    r = predictor.predict(np.vstack(rows))
    labels = [predictor.states[k] for k in r["proba"].argmax(axis=1)]
    return {"ticks": len(labels), "labels": labels, "valence": r["valence_mix"], "arousal": r["arousal_mix"]}


def main() -> None:
    from valence.models.emotion import EmotionPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="recording.jsonl[:ActedState]")
    ap.add_argument("--models", default="artifacts/emotion")
    ap.add_argument("--warmup-ticks", type=int, default=12, help="ignore the first minute (baselines settle)")
    ap.add_argument("--out")
    args = ap.parse_args()
    predictor = EmotionPredictor(Path(args.models))
    sim = simulate_corpus(30, 99, state_set="emotion")
    sim_people = sim[sim["person_id"] != 0]
    report = {"model": predictor.model_id, "recordings": {}}
    for spec in args.specs:
        path, _, state = spec.partition(":")
        cap = capture_ticks(Path(path))
        person = cap[cap["person_id"] == 1] if len(cap) else cap
        read = read_capture(cap, predictor) if len(cap) else {"ticks": 0}
        entry = {"acted": state or None, "ticks": int(len(person))}
        if read["ticks"]:
            sl = slice(args.warmup_ticks, None) if read["ticks"] > args.warmup_ticks else slice(None)
            labels = read["labels"][sl]
            n = len(labels)
            entry["scored_ticks"] = n
            entry["predicted_share"] = {k: round(v / n, 3) for k, v in Counter(labels).most_common()}
            entry["map_position"] = {"valence": round(float(np.mean(read["valence"][sl])), 3),
                                     "arousal": round(float(np.mean(read["arousal"][sl])), 3)}
            if state:
                entry["agreement"] = round(sum(lab == state for lab in labels) / n, 3)
                entry["acted_anchor"] = AFFECT_ANCHORS.get(state)
        ref = sim_people[sim_people["true_state"] == state] if state else sim_people
        entry["channels"] = {c: {"live": describe(person[c]) if c in person else {"n": 0}, "sim": describe(ref[c])}
                             for c in CUES}
        report["recordings"][Path(path).stem] = entry
    text = json.dumps(report, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
