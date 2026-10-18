"""Every scripted scenario reads as scripted: per segment and person, the served label matches the script.

Runs each ``scenarios/*.yaml`` offline through ``Runtime.step`` with the served models (core artifacts, or
``artifacts/all`` for scenarios using the P1 states). The first minutes of each segment are a grace
period (the windows need time to fill); after it, at least 80% of ticks must carry the scripted state,
masked segments must show MASK OVERRIDE on at least half of them, and no other segment may show it.
"""

import json
import tempfile
from pathlib import Path

import pytest

from valence.config import Settings
from valence.contracts import CORE_STATES
from valence.governance.purge import PurgeGate
from valence.models.served import load_served
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime
from valence.sim.scenarios import SCENARIO_DIR, load_script, render

ROOT = Path(__file__).resolve().parents[1]
MIN_ACC = 0.8
GRACE_TICKS = 24  # 2 min, or half the segment if shorter


def _state_set(script) -> str:
    states = {s for seg in script.segments for s in seg.states if s is not None}
    return "core" if states <= set(CORE_STATES) else "all"


def score(name: str) -> list[dict]:
    script = load_script(name)
    state_set = _state_set(script)
    served = load_served(ROOT / ("artifacts" if state_set == "core" else "artifacts/all"), state_set)
    if served is None:
        pytest.skip(f"no trained {state_set} artifacts")
    df = render(name, state_set=state_set)
    bus = EventBus()
    msgs: list[dict] = []
    bus.add_listener(lambda p: msgs.append(json.loads(p)))
    rt = Runtime(Settings(state_set=state_set), bus, served, gate=PurgeGate(), audit_dir=tempfile.mkdtemp())
    rt.use_scenario(df, name)
    preds: dict[tuple[int, int], dict] = {}
    for tick in sorted(df["tick"].unique()):
        msgs.clear()
        rt.step()
        for m in msgs:
            if m["type"] == "prediction" and m["person_id"] != 0 and not m["warming_up"]:
                preds[(m["person_id"], int(tick))] = m
    rt.close()
    out, start = [], 0
    for seg in script.segments:
        n = int(round(seg.minutes * 60 / Settings().tick_s))
        for pid, state in enumerate(seg.states, start=1):
            ticks = [t for t in range(start + min(GRACE_TICKS, n // 2), start + n) if (pid, t) in preds]
            if state is None or not ticks:
                continue
            got = [preds[(pid, t)] for t in ticks]
            out.append({"person": pid, "state": state, "n": len(got),
                        "acc": sum(m["label"] == state for m in got) / len(got),
                        "override": sum(m["mask_override"] for m in got) / len(got)})
        start += n
    return out


@pytest.mark.parametrize("name", sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")))
def test_scenario_reads_as_scripted(name):
    segments = score(name)
    assert segments, "no scored segments"
    for s in segments:
        assert s["acc"] >= MIN_ACC, s
        if s["state"] == "Frustrated_Masked":
            assert s["override"] >= 0.5, s
        else:
            assert s["override"] == 0, s
