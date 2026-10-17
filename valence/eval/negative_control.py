"""Negative control and acted-set scoring on features-only recordings.

    python -m valence.eval.negative_control data/real/capture_typing.jsonl --expect Focused
    python -m valence.eval.negative_control data/real/*.jsonl            # label distribution per file

The gate the plan sets: replaying quiet typing must not predict Frustrated (or Tense) on more than 20%
of ticks. If it does, detector false positives (fists while typing, slump while looking at the keyboard)
are leaking into the context model: update ``valence/sim/params.py`` FP rates from the capture and retrain.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from valence.config import Settings
from valence.governance.purge import PurgeGate
from valence.models.predictor import StubPredictor
from valence.models.served import load_served
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime
from valence.runtime.recording import replay_offline

STRESS = {"Frustrated_Masked", "Tense"}
GATE = 0.20


def score(path: Path, state_set: str = "core") -> dict:
    served = load_served(Path("artifacts"), state_set) or StubPredictor()
    rt = Runtime(Settings(state_set=state_set), EventBus(), served, gate=PurgeGate(), audit_dir=tempfile.mkdtemp())
    rt.use_live()
    rt.aggregator.source = "replay"
    preds = [m for m in replay_offline(rt, path) if not m["warming_up"]]
    rt.close()
    labels = Counter(m["label"] for m in preds)
    n = max(1, len(preds))
    return {
        "file": str(path), "model": getattr(served, "model_id", "?"), "ticks": len(preds),
        "labels": {k: round(v / n, 3) for k, v in labels.most_common()},
        "stress_share": round(sum(labels[s] for s in STRESS) / n, 3),
        "mask_overrides": sum(m["mask_override"] for m in preds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expect", help="state the recording was acted as (reports agreement)")
    ap.add_argument("--state-set", default="core")
    args = ap.parse_args()
    failed = False
    for f in args.files:
        res = score(Path(f), args.state_set)
        if args.expect:
            res["agreement"] = res["labels"].get(args.expect, 0.0)
        if "typing" in Path(f).stem:
            res["negative_control_pass"] = res["stress_share"] <= GATE
            failed |= not res["negative_control_pass"]
        print(json.dumps(res))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
