"""Golden runtime records, perturbed by a xorshift32 stream, through Python's Runtime with the served predictor.

Python's side of PERTURBED in frontend/src/test/engine-runtime.test.ts ("runtime against Python on perturbed golden
runs"): the test applies the same perturbation (perturbRecord, perturbedTicks) to the same fixture records and must
reproduce the digest of the canonical message text and the audit chain head. Reads the runs of
frontend/src/test/fixtures/engine/runtime.json.gz (scripts/export_web_golden.py) and the served models under
artifacts/.

    .venv/bin/python -B scripts/golden_runtime/perturb.py             # one JSON line per run on stdout
    .venv/bin/python -B scripts/golden_runtime/perturb.py --out DIR   # also DIR/<run>.txt, the canonical text
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tempfile
import types

from _common import ARTIFACTS, FIXTURES, Bus, Clock, Gate, canon, out_dir

import valence.governance.audit as audit_mod  # noqa: E402
import valence.runtime.aggregator as agg_mod  # noqa: E402
import valence.runtime.engine as eng_mod  # noqa: E402
from valence.config import Settings  # noqa: E402
from valence.contracts import TICK_FIELDS, TickRecord  # noqa: E402
from valence.models.served import load_served  # noqa: E402
from valence.runtime.engine import Runtime  # noqa: E402

FX = FIXTURES / "runtime.json.gz"
FLOATS = ["ambient_db", "light_lux", "posture_slump", "head_in_hands_frac", "motion_energy", "face_valence",
          "face_visible", "hours_since_break", "tod_h", "room_db_base", "smile", "head_yaw"]
SPECIAL = [0.0, -0.0, 1e-300, 1e6, -1e6, 1e300, 0.5, 2.5, 0.0005, 0.00049999999999999, 23.999999, -0.01]
MULT = [0.5, 1.5, -1.0, 10.0]
ADD = [0.0, 0.001, -0.25]
INTS = ["laughter_count", "compliments_count", "clenched_fists_count", "social_withdrawal_flag", "n_people"]
INT_CHOICES = [0, 1, 2, 3, 7, 40]
TAGS = ["low_light", "room_history", "été"]
RUNS = (("full_demo", 200), ("overwhelmed", 156), ("context_blindness", 120))


class XorShift32:
    def __init__(self, seed: int) -> None:
        self.x = seed

    def next(self) -> float:
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.x = x
        return x / 4294967296.0

    def pick(self, xs):
        return xs[int(self.next() * len(xs))]


def perturb(rng: XorShift32, d: dict) -> dict:
    d = dict(d)
    d["simulated_fields"] = list(d["simulated_fields"])
    for k in FLOATS:
        u = rng.next()
        if u < 0.04:
            d[k] = math.nan
        elif u < 0.055:
            d[k] = rng.pick(SPECIAL)
        elif u < 0.10 and not math.isnan(d[k]):
            d[k] = d[k] * rng.pick(MULT) + rng.pick(ADD)
    for k in INTS:
        if rng.next() < 0.03:
            d[k] = rng.pick(INT_CHOICES)
    if rng.next() < 0.02:
        d["simulated_fields"].append(rng.pick(TAGS))
    return d


def ticks_for(run: dict, n: int):
    rng = XorShift32(0x9E3779B9)
    for i, t in enumerate(run["ticks"][:n]):
        recs = [perturb(rng, {k: (math.nan if v is None else v) for k, v in r.items()}) for r in t["records"]]
        if 40 <= i % 97 < 60:  # person 1 away for 20 ticks
            recs = [r for r in recs if r["person_id"] != 1]
        if i % 50 > 35:  # a second person: person 1's row under id 3
            recs += [{**r, "person_id": 3, "display_id": "Person #3", "n_people": 2,
                      "simulated_fields": list(r["simulated_fields"])} for r in recs if r["person_id"] == 1]
        yield t["wall"], t["mono"], [TickRecord(**{k: r[k] for k in TICK_FIELDS}) for r in recs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", metavar="DIR", help="also write each run's canonical text to DIR/<run>.txt")
    out = out_dir(ap.parse_args().out)
    fx = json.loads(gzip.decompress(FX.read_bytes()))
    for name, n in RUNS:
        run = next(r for r in fx["runs"] if r["name"] == name)
        state_set = run["state_set"]
        art = ARTIFACTS / ("all" if state_set == "all" else "")
        served = load_served(art, state_set)
        clock = Clock()
        orig = (audit_mod.time, agg_mod.time, eng_mod.time)
        audit_mod.time = agg_mod.time = eng_mod.time = clock
        bus = Bus()
        texts, preds = [], 0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rt = Runtime(Settings(state_set=state_set), bus, served, gate=Gate(), narrator=None, audit_dir=tmp)
                holder = types.SimpleNamespace(label="replay", recs=None)
                holder.next_tick = lambda wall, mono: holder.recs
                rt.source, rt.mode = holder, "replay"
                for wall, mono, recs in ticks_for(run, n):
                    holder.recs = recs
                    clock.wall, clock.mono = wall, mono
                    bus.dumps.clear()
                    rt.step(wall=wall, mono=mono)
                    preds += sum(1 for m in bus.dumps if m["type"] == "prediction")
                    texts.append(canon(list(bus.dumps)))
                entries = rt.audit.entries()
                rt.close()
        finally:
            audit_mod.time, agg_mod.time, eng_mod.time = orig
        text = "\n".join(texts)
        if out is not None:
            (out / f"{name}.txt").write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(json.dumps({"name": name, "ticks": len(texts), "predictions": preds, "digest": digest,
                          "auditN": len(entries), "auditLast": entries[-1]["hash"] if entries else ""}))


if __name__ == "__main__":
    main()
