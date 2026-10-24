"""A scripted 10-state replay for the emotion map: ``artifacts/emotion/tour.json.gz``.

    python scripts/emotion_tour.py [--dir artifacts/emotion]

One simulated person (and the room) walks through all ten states, 5 minutes each, in an order that crosses the
valence-arousal plane. Each tick carries the ``TickRecord`` fields (person and room rows, null = NaN) and Python's
outputs for the person at that tick: the 31 emotion features, the 10-state probabilities, the label and the map
position (probability-weighted anchors). The console can replay it to show the affect map and its trail without a
camera, and the web engine can use it as an end-to-end parity check (ticks -> features -> models).
Simulated data only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from valence.contracts import TICK_FIELDS  # noqa: E402
from valence.features.compute import PERSON_FIELDS, ROOM_FIELDS, compute  # noqa: E402
from valence.features.emotion import EMOTION_FIELDS, compute_emotion, model_input  # noqa: E402
from valence.models.emotion import EmotionPredictor, clean_json  # noqa: E402
from valence.sim.generator import Script, Segment, simulate_session  # noqa: E402

ORDER = ["Focused", "Confused", "Frustrated_Masked", "Tense", "Anxious", "Overwhelmed", "Exhausted", "Bored", "Calm",
         "Collaborative_Happy"]
MINUTES = 5.0


def tour_script() -> Script:
    return Script(name="emotion_tour", people=1, segments=[Segment(minutes=MINUTES, states=[s]) for s in ORDER],
                  seed=2026, description="Scripted walk through the ten emotion-map states, 5 min each",
                  start_hour=14.0, start_hours_since_break=[1.0], lux_base=350.0, mic_offset_db=0.0,
                  face_hidden=0.05, venue_db=55.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "artifacts" / "emotion"))
    args = ap.parse_args()
    out_dir = Path(args.dir)
    pred = EmotionPredictor(out_dir)
    script = tour_script()
    df = simulate_session("scenario-emotion_tour", np.random.default_rng(script.seed), state_set="emotion",
                          script=script)
    room = df[df["person_id"] == 0].sort_values("tick")
    person = df[df["person_id"] == 1].sort_values("tick")
    fields = sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS))
    parr = {f: person[f].to_numpy(dtype=float) for f in fields}
    rarr = {f: room[f].to_numpy(dtype=float) for f in ROOM_FIELDS}
    ticks = []
    for i in range(len(person)):
        lo = max(0, i - 179)
        x123 = compute({f: parr[f][lo:i + 1] for f in PERSON_FIELDS}, {f: v[lo:i + 1] for f, v in rarr.items()})
        xe = compute_emotion({f: parr[f][lo:i + 1] for f in EMOTION_FIELDS})
        r = pred.predict(model_input(x123, xe))
        p = r["proba"][0]
        rows = [room.iloc[i], person.iloc[i]]
        ticks.append({
            "records": [{k: row[k] for k in TICK_FIELDS} for row in rows],
            "true_state": person["true_state"].iat[i],
            "true_valence": float(person["true_valence"].iat[i]), "true_arousal": float(person["true_arousal"].iat[i]),
            "emotion_features": xe.tolist(),
            "probabilities": p.tolist(), "label": pred.states[int(np.argmax(p))],
            "valence": float(r["valence_mix"][0]), "arousal": float(r["arousal_mix"][0]),
            "valence_regressor": float(r["valence"][0]), "arousal_regressor": float(r["arousal"][0]),
        })
    doc = clean_json({"model_id": pred.model_id, "feature_version": pred.meta["feature_version"],
                      "description": script.description, "order": ORDER, "minutes_per_state": MINUTES,
                      "tick_s": 5.0, "states": pred.states, "ticks": ticks,
                      "note": "Simulated. The room window equals the person's window here (both start at tick 0)."})
    for t in doc["ticks"]:
        for rec in t["records"]:
            rec["simulated_fields"] = list(rec["simulated_fields"])
            for k, v in rec.items():
                if isinstance(v, (np.integer,)):
                    rec[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    rec[k] = None if np.isnan(v) else float(v)
    raw = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    path = out_dir / "tour.json.gz"
    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    agree = np.mean([t["label"] == t["true_state"] for t in ticks])
    print(f"{path}: {len(ticks)} ticks, label = scripted state on {agree:.0%} of ticks, {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
