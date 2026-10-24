"""Drive valence/runtime/engine.py Runtime through the edge-case specs (specs.py) with a stub predictor.

Python's side of "runtime against Python on inputs the golden runs never produce" in
frontend/src/test/engine-runtime.test.ts: SPEC_EXPECT (canonical-text digest, audit chain head, end state) and the
narrator calls, times of day, rounded state vector, break payloads and other values asserted on the sv / room / op
runs. The predictor is a stub whose outputs are keyed by "tick:person_id" (missing = predict() returns None); the
purge gate is a stub; the audit / aggregator / engine clocks follow the spec; np.random.normal draws from the
spec's "draws"; the narrator records its calls.

    .venv/bin/python -B scripts/golden_runtime/edge_cases.py                # every spec of specs.py, a line each
    .venv/bin/python -B scripts/golden_runtime/edge_cases.py --out DIR      # also DIR/<name>.json: ticks, audit, state
    .venv/bin/python -B scripts/golden_runtime/edge_cases.py SPEC.json ...  # specs from files (specs.py --out DIR)

Spec numbers: "NaN" / "Infinity" / "-Infinity" strings decode to floats; null in a record field = NaN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import types
from pathlib import Path
from typing import Any

from _common import Bus, Clock, Gate, canon, out_dir

import numpy as np  # noqa: E402

import valence.governance.audit as audit_mod  # noqa: E402
import valence.runtime.aggregator as agg_mod  # noqa: E402
import valence.runtime.engine as eng_mod  # noqa: E402
from valence.config import Settings  # noqa: E402
from valence.contracts import STATE_SETS, TICK_FIELDS, Driver, Layer, TickRecord  # noqa: E402
from valence.models.predictor import PredictionOutput  # noqa: E402
from valence.runtime.engine import Runtime  # noqa: E402

STATE_KEYS = ("latest", "smoothers", "history", "gap_ticks", "paused", "tick", "people")  # SPEC_EXPECT[*].state


def dec(v):
    if isinstance(v, str) and v in ("NaN", "Infinity", "-Infinity"):
        return float(v.replace("Infinity", "inf"))
    if isinstance(v, list):
        return [dec(x) for x in v]
    if isinstance(v, dict):
        return {k: dec(x) for k, x in v.items()}
    return v


class Stub:
    """Outputs looked up by (last.tick, last.person_id); missing -> None."""

    def __init__(self, states, outs):
        self.states = states
        self.model_id = "stub-edge"
        self.outs = outs

    def predict(self, person, room):
        last = person[-1]
        o = self.outs.get(f"{last.tick}:{last.person_id}")
        if o is None:
            return None
        layers = [Layer(name=l["name"], contribution=l["contribution"],
                        drivers=[Driver(**d) for d in l.get("drivers", [])]) for l in o.get("layers", [])]
        return PredictionOutput(label=o["label"], confidence=o["confidence"], probabilities=o["probabilities"],
                                face_only_label=o.get("face_only_label"),
                                face_only_confidence=o.get("face_only_confidence"),
                                layers=layers, raw_override=o.get("raw_override"),
                                ood_features=o.get("ood_features", []))


def record(d, session):
    d = dict(d)
    d.setdefault("session_id", session)
    d.setdefault("display_id", f"Person #{d.get('person_id', 0)}")
    out = {}
    for k in TICK_FIELDS:
        if k in d:
            v = d[k]
            out[k] = math.nan if v is None else dec(v)
    return TickRecord(**out)


def run_spec(spec: dict) -> dict[str, Any]:
    """{ticks: [canonical text per tick], audit: [{kind, hash, payload}], state: {...}, messages: [[dict]]}."""
    states = STATE_SETS[spec.get("state_set", "core")]
    settings = Settings(**spec.get("settings", {}))
    clock = Clock()
    clock.wall, clock.mono = spec.get("wall_init", 0.0), spec.get("mono_init", 0.0)
    gate = Gate()
    outs = {}
    for st in spec["steps"]:
        for k, o in st.get("outs", {}).items():
            outs[k] = dec(o)
    stub = Stub(states, outs)
    bus = Bus()
    draws = iter(spec.get("draws", []))
    orig_time = (audit_mod.time, agg_mod.time, eng_mod.time)
    audit_mod.time = agg_mod.time = eng_mod.time = clock
    orig_normal = np.random.normal
    np.random.normal = lambda loc, scale: loc + scale * next(draws)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            narr = types.SimpleNamespace(
                request=lambda sheet, pid: calls.append([sheet.event_id, sheet.label_display, sheet.confidence]),
                clear_cache=lambda: calls.append(["clear"]))
            rt = Runtime(settings, bus, stub, gate=gate, narrator=narr, audit_dir=tmp)
            rt.session_id = rt.aggregator.session_id = "session-edge"
            rt.room_only = spec.get("room_only", False)
            src = types.SimpleNamespace(label=spec.get("label", "replay"), recs=None)
            src.next_tick = lambda wall, mono: src.recs
            rt.source = src
            ticks, messages = [], []
            for st in spec["steps"]:
                for act in st.get("pre", []):
                    op = act[0]
                    if op == "clock":
                        clock.wall = act[1]
                    elif op == "overlay":
                        rt.set_overlay(act[1], act[2])
                    elif op == "room_only":
                        if act[1] != rt.room_only:
                            rt.latest.clear()
                        rt.room_only = act[1]
                        rt._audit("operator", {"action": "mode", "room_only": rt.room_only})
                    elif op == "reset":
                        rt.reset_people(keep_room=act[1])
                    elif op == "break":
                        rt.aggregator.log_break(act[1])
                        rt._audit("operator", {"action": "break", "person_id": act[1]})
                    elif op == "purge":
                        gate.c.update(act[1])
                    elif op == "seed":
                        import pandas as pd
                        rows = [record(r, "seed").to_dict() for r in act[1]]
                        df = pd.DataFrame(rows)
                        rt.seed(df, act[2], dec(act[3]), dec(act[4]))
                    elif op == "audit":
                        rt._audit(act[1], dec(act[2]))
                    else:
                        raise ValueError(op)
                clock.wall, clock.mono = st["wall"], st["mono"]
                src.recs = None if st.get("records") is None else [record(r, "session-edge") for r in st["records"]]
                bus.dumps.clear()
                rt.step(wall=st["wall"], mono=st["mono"])
                ticks.append(canon(list(bus.dumps)))
                messages.append(list(bus.dumps))
            entries = rt.audit.entries()
            ok, n = rt.audit.verify()
            state = {"latest": sorted(rt.latest), "smoothers": sorted(rt.smoothers), "history": len(rt.history),
                     "gap_ticks": rt._gap_ticks, "paused": rt.paused, "tick": rt._tick_i,
                     "people": rt.window.people(), "verify": [ok, n], "narrator": calls}
            rt.close()
    finally:
        audit_mod.time, agg_mod.time, eng_mod.time = orig_time
        np.random.normal = orig_normal
    return {"ticks": ticks, "audit": [{"kind": e["kind"], "hash": e["hash"], "payload": e["payload"]} for e in entries],
            "state": state, "messages": messages}


def summary(name: str, r: dict[str, Any]) -> dict[str, Any]:
    """What SPEC_EXPECT pastes (digest, auditN, auditLast, state), plus the chain check and the narrator calls."""
    return {"name": name, "digest": hashlib.sha256("\n".join(r["ticks"]).encode()).hexdigest(),
            "auditN": len(r["audit"]), "auditLast": r["audit"][-1]["hash"] if r["audit"] else "",
            "state": {k: r["state"][k] for k in STATE_KEYS}, "verify": r["state"]["verify"],
            "narrator": r["state"]["narrator"]}


def load_specs(paths: list[str]) -> dict[str, dict]:
    """Specs from files, or every spec of specs.py; either way through JSON, as the TS test reads them."""
    if paths:
        return {Path(p).stem: json.loads(Path(p).read_text()) for p in paths}
    from specs import build_specs
    return {name: json.loads(json.dumps(spec)) for name, spec in build_specs().items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("specs", nargs="*", metavar="SPEC.json", help="spec files (default: every spec of specs.py)")
    ap.add_argument("--out", metavar="DIR", help="also write DIR/<name>.json: ticks, audit and end state")
    args = ap.parse_args()
    out = out_dir(args.out)
    for name, spec in load_specs(args.specs).items():
        r = run_spec(spec)
        if out is not None:
            full = {"ticks": r["ticks"], "audit": r["audit"], "state": r["state"]}
            (out / f"{name}.json").write_text(json.dumps(full, indent=1) + "\n")
        print(json.dumps(summary(name, r)))


if __name__ == "__main__":
    main()
