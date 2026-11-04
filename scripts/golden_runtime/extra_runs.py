"""Extra Runtime expectations for frontend/src/test/engine-runtime.test.ts (EXTRA, beyond runtime.json.gz).

Runs valence/runtime/engine.py Runtime on: the live source (recording.replay_offline over captures from
aggregator.json.gz) with use_live, an operator baseline capture and a purge gate stub; use_recording on a
synthetic stream with a break and drops; the SIMULATED overlays (np.random.normal patched to a fixed draw
sequence, the test's fixedDraws()); eviction (person_drop_min=3); and a sequence of operator actions (seed, mode,
break, reset, baseline capture/reset, use_scenario). Each run's messages are reduced to a canonical text (numbers
as repr(float(x)), strings as json.dumps, dict keys in insertion order) and its sha256; the test rebuilds the same
text in TypeScript. Clocks are the WALL0/MONO0 of scripts/export_web_golden.py, TZ is UTC.

    .venv/bin/python -B scripts/golden_runtime/extra_runs.py             # JSON summary of the five runs on stdout
    .venv/bin/python -B scripts/golden_runtime/extra_runs.py --out DIR   # also DIR/<run>.txt, the canonical text

Reads frontend/src/test/fixtures/engine/aggregator.json.gz (scripts/export_web_golden.py writes it) and the served
models under artifacts/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from _common import ROOT, canon, out_dir

sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import export_web_golden as g  # noqa: E402
import valence.runtime.aggregator as agg_mod  # noqa: E402
import valence.runtime.engine as engine_mod  # noqa: E402
from valence.config import Settings  # noqa: E402
from valence.contracts import obs_from_dict  # noqa: E402
from valence.governance.purge import PurgeGate, PurgeStats  # noqa: E402
from valence.runtime.engine import LiveSource, Runtime, ScenarioSource  # noqa: E402

WALL0, MONO0 = g.WALL0, g.MONO0
OUT: Path | None = None  # --out DIR


class StubGate:
    """PurgeGate stand-in: end_tick() returns counters computed by ``fn(i)`` for the i-th step."""

    def __init__(self, fn):
        self.fn = fn
        self.i = 0
        self.cur = PurgeStats(purged={}, freed={}, unzeroable={}, transcripts_purged=0, live_raw_buffers=0,
                              violations=0)

    def end_tick(self):
        video, audio, transcripts, violations, live = self.fn(self.i)
        self.cur = PurgeStats(purged={"video": video, "audio": audio}, freed={}, unzeroable={},
                              transcripts_purged=transcripts, live_raw_buffers=live, violations=violations)
        self.i += 1
        return self.cur

    def stats(self):
        return self.cur


def draws():
    """The test's fixedDraws(): ((k * 7919) % 1000) / 250 - 2 for k = 0, 1, 2, ..."""
    k = 0
    while True:
        yield ((k * 7919) % 1000) / 250 - 2
        k += 1


class Harness:
    def __init__(self, state_set: str, settings: Settings, gate=None):
        self.served = g.served_for(state_set)
        self.settings = settings
        self.bus = g._CaptureBus()
        self.clock = g._Clock(WALL0 - 10.0, MONO0 - 10.0)
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [g.patched_clock(g.audit_mod, self.clock), g.patched_clock(engine_mod, self.clock),
                        g.patched_clock(agg_mod, self.clock)]
        for p in self.patches:
            p.__enter__()
        self.rt = Runtime(settings, self.bus, self.served, gate=gate or PurgeGate(), narrator=None,
                          audit_dir=self.tmp.name)
        self.rt.session_id = self.rt.aggregator.session_id = "session-golden"
        self.ticks: list[list[dict]] = []

    def at(self, wall: float) -> None:
        self.clock.wall = wall

    def step(self, wall: float, mono: float) -> None:
        self.clock.wall, self.clock.mono = wall, mono
        self.bus.dumps.clear()
        Runtime.step(self.rt, wall=wall, mono=mono)
        self.ticks.append([d for d in self.bus.dumps if d.get("type") != "tracks"])

    def finish(self, name: str) -> dict:
        ok, n = self.rt.audit.verify()
        assert ok
        entries = self.rt.audit.entries()
        hello = g.hello_for(self.rt, self.served)
        text = "\n".join(canon(t) for t in self.ticks)
        if OUT is not None:
            (OUT / f"{name}.txt").write_text(text)
        for p in reversed(self.patches):
            p.__exit__(None, None, None)
        self.rt.close()
        self.tmp.cleanup()
        msgs = [m for t in self.ticks for m in t]
        return {
            "name": name,
            "ticks": len(self.ticks),
            "messages": len(msgs),
            "predictions": sum(m["type"] == "prediction" for m in msgs),
            "digest": hashlib.sha256(text.encode()).hexdigest(),
            "audit_n": n,
            "audit_kinds": [e["kind"] + ":" + str(e["payload"].get("action", "")) for e in entries],
            "audit_last": entries[-1]["hash"] if entries else None,
            "hello": canon(g.jsonable(hello)),
            "tick_index": self.rt._tick_i,
            "gap_ticks": self.rt._gap_ticks,
            "latest": sorted(self.rt.latest),
            "smoothers": sorted(self.rt.smoothers),
        }


def aggregator_obs(name: str):
    fx = json.loads(gzip.decompress((g.OUT_DIR / "aggregator.json.gz").read_bytes()))
    case = next(c for c in fx["cases"] if c["name"] == name)
    return [obs_from_dict(d) for d in case["obs"]]


def replay_live(h: Harness, obs, pre=None) -> None:
    """recording.replay_offline(h.rt) with the clock at (WALL0, MONO0); pre(i) runs 2.5 s before step i."""
    def step(wall=None, mono=None):
        i = len(h.ticks)
        if pre is not None:
            h.at(wall - 2.5)
            pre(i)
        h.step(wall, mono)
        return []

    h.rt.step = step
    rec_clock = g._Clock(WALL0, MONO0)
    with g.utc_tz(), g.patched_clock(g.recording, rec_clock), g.patched(g.recording, "iter_obs", lambda _: iter(obs)):
        g.recording.replay_offline(h.rt, "<memory>", tick_s=h.settings.tick_s)


def live_counters(rt):
    def fn(i):
        d = rt.aggregator.frames_decoded
        lag = (0, 1, 2, 3, 5, 9, -4)[i % 7]
        return (d - lag + (777 if i >= 10 else 0), 3 * i, i // 4, 1 if i >= 20 else 0, i % 2)
    return fn


def run_live_capture() -> dict:
    gate = StubGate(lambda i: (0, 0, 0, 0, 0))
    h = Harness("core", Settings(state_set="core"), gate=gate)
    gate.fn = live_counters(h.rt)
    h.at(WALL0 - 6.0)
    h.rt.use_live()
    h.at(WALL0 - 5.0)
    h.rt.aggregator.start_baseline(10.0)  # POST /api/room/baseline {"seconds": 10}
    h.rt._audit("operator", {"action": "room_baseline", "seconds": 10.0})

    def pre(i):
        if i == 10:
            h.rt._new_source_counters()

    replay_live(h, aggregator_obs("capture_fists"), pre)
    return h.finish("live_capture")


def run_recording() -> dict:
    h = Harness("core", Settings(state_set="core"))
    h.at(WALL0 - 6.0)
    rt = h.rt
    # use_recording(path, speed=4.0) without the feeder thread
    rt._new_source_counters()
    rt.reset_people(keep_room=False)
    rt.source = LiveSource(rt.aggregator, label="replay")
    rt.aggregator.source = "replay"
    rt.aggregator.start_baseline(mode="auto")
    rt.mode = "replay"
    rt.speed = 4.0
    rt._audit("operator", {"action": "replay_recording", "recording": "synth_break_and_drop"})

    def pre(i):
        if i == 50:
            rt.aggregator.log_break(1)  # POST /api/events/break {"person_id": 1}
            rt._audit("operator", {"action": "break", "person_id": 1})

    replay_live(h, aggregator_obs("synth_break_and_drop"), pre)
    return h.finish("recording_break_and_drop")


def scenario_harness(scenario: str, state_set: str, settings: Settings, gate=None):
    h = Harness(state_set, settings, gate=gate)
    src = ScenarioSource(g._rendered(scenario, state_set), scenario)
    src.session_id = f"replay-{scenario}-golden"
    h.rt.source = src
    h.rt.mode = "replay"
    return h


def run_overlays() -> dict:
    h = scenario_harness("frustrated_masked", "core", Settings(state_set="core"),
                         gate=StubGate(lambda i: (20 * i + i % 3, i, i // 4, 1 if i >= 20 else 0, i % 2)))
    gen = draws()
    old = np.random.normal
    np.random.normal = lambda loc=0.0, scale=1.0: loc + scale * next(gen)
    try:
        toggles = {5: ("low_light", True), 12: ("face_occlusion", True), 20: ("low_light", False),
                   28: ("face_occlusion", False), 33: ("low_light", True)}
        for i in range(40):
            wall, mono = WALL0 + i * 5.0, MONO0 + i * 5.0
            if i in toggles:
                h.at(wall - 2.5)
                h.rt.set_overlay(*toggles[i])
            h.step(wall, mono)
    finally:
        np.random.normal = old
    return h.finish("overlays")


def run_evict() -> dict:
    s = Settings(state_set="core", person_drop_min=3.0)
    h = scenario_harness("full_demo", "core", s)
    for i in range(g.n_scenario_ticks("full_demo", "core")):
        h.step(WALL0 + i * 5.0, MONO0 + i * 5.0)
    return h.finish("evict")


def run_operator() -> dict:
    h = scenario_harness("context_blindness", "core", Settings(state_set="core"))
    rt = h.rt
    h.clock.mono = MONO0 - 1.0
    h.at(WALL0 - 1.0)
    rt.seed(g._rendered("full_demo", "core"), "full_demo", 2.0, 3.5)
    for i in range(70):
        wall, mono = WALL0 + i * 5.0, MONO0 + i * 5.0
        h.at(wall - 2.5)
        if i == 8:
            rt.room_only = True if True != rt.room_only else rt.room_only
            rt.latest.clear()
            rt._audit("operator", {"action": "mode", "room_only": rt.room_only})
        if i == 14:
            rt.aggregator.log_break(1)
            rt._audit("operator", {"action": "break", "person_id": 1})
        if i == 16:
            rt.aggregator.log_break(None)
            rt._audit("operator", {"action": "break", "person_id": None})
        if i == 20:
            if rt.room_only:
                rt.latest.clear()
            rt.room_only = False
            rt._audit("operator", {"action": "mode", "room_only": rt.room_only})
        if i == 25:
            rt.reset_people(keep_room=True)
        if i == 30:
            rt.aggregator.start_baseline(mode="auto")
            rt._audit("operator", {"action": "room_baseline_reset"})
        if i == 35:
            rt.aggregator.start_baseline(30.0)
            rt._audit("operator", {"action": "room_baseline", "seconds": 30.0})
        if i == 40:
            rt.use_scenario(g._rendered("context_blindness", "core"), "context_blindness", speed=4.0)
            rt.source.session_id = "replay-context_blindness-golden"
        h.step(wall, mono)
    return h.finish("operator")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", metavar="DIR", help="also write each run's canonical text to DIR/<run>.txt")
    OUT = out_dir(ap.parse_args().out)
    out = [run_live_capture(), run_recording(), run_overlays(), run_evict(), run_operator()]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
