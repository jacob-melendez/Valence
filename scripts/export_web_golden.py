"""Golden fixtures for the TypeScript engine port (Valence Web) -> frontend/src/test/fixtures/engine/*.json.gz.

The Python engine is the reference. This script drives it deterministically and writes gzipped JSON that the
vitest parity tests (frontend/src/test/engine-*.test.ts, loaded through engine-fixtures.ts) replay through
frontend/src/engine/:

    .venv/bin/python -B scripts/export_web_golden.py                 # all six fixtures, then sanity checks
    .venv/bin/python -B scripts/export_web_golden.py --only models,audit
    .venv/bin/python -B scripts/export_web_golden.py --check-only    # re-derive values from existing fixtures

num.json.gz is written by another exporter; this one never touches it.

Determinism
-----------
* TZ=UTC (os.environ + time.tzset()) around everything that computes a time of day.
* Fixed clocks: WALL0 = 1792850400.0 (Saturday 2026-10-24 14:00:00 UTC), MONO0 = 1000.0. Scenario tick i is
  stamped wall = WALL0 + i * tick_s, mono = MONO0 + i * tick_s.
* Recordings go through valence.runtime.recording.replay_offline with its clock patched to (WALL0, MONO0).
* The audit module's clock is patched: an entry's ts is round(current tick wall, 3).
* Seeded RNGs, fixed session ids, gzip mtime=0: reruns are byte-identical. time.time() never reaches a fixture.

Encoding (all files)
--------------------
JSON with "version": 1. NaN -> null; +-inf never appears (the exporter refuses it). Floats are written by
json.dumps (Python repr), so JSON.parse yields the identical double. Dict key order is the Python insertion
order (it matters for dict-iteration code such as build_layers). TickRecordDict = TickRecord.to_dict(): every
dataclass field, snake_case, NaN -> null. Settings dicts are dataclasses.asdict(Settings(...)) unless noted.

Schemas
-------
features.json.gz
  {version, wall0, mono0,
   scenarios: [{name, state_set, room_only, session_id,
                settings: {person_drop_min, tick_s, window_ticks},
                ticks: [{wall, mono, records: [TickRecordDict]}],      # ScenarioSource.next_tick order, room row first
                checkpoints: [{tick, person_id, features: [123]}]}],   # tick = index into ticks
   edge_cases: [{name, person: {field: [..]}, room: {field: [..]}, features: [123]}]}
  Replay rule (= Runtime.step): append every record of tick i to RollingWindow(window_ticks); evict people whose
  last row is older than person_drop_min minutes (Runtime._evict, `wall - rows[-1].ts > person_drop_min * 60`);
  targets = [0] when room_only else the sorted non-room person ids in tick i's records;
  features = features_from_window(window.snapshot(pid), window.room() or snapshot).
  Checkpoints: every target on even ticks, while its window holds <= 3 rows, and on the last tick.
  edge_cases call compute(person, room) directly; arrays may differ in length, exceed 180 ticks, and the room
  dict may omit room_db_base / room_lux_base (compute reads a missing key as NaN).

aggregator.json.gz
  {version, settings: SettingsDict,
   cases: [{name, kind: "recording"|"synthetic", wall0, mono0, tz: "UTC", tick_s, session_id, source,
            settings: SettingsDict,            # always equal to the top-level settings (Settings() defaults)
            obs: [ObsDict],                    # obs_to_dict() of the stream, file order, original clocks
            ticks: [{wall, mono, records: [TickRecordDict]}],
            checkpoints: [{tick, person_id, features: [123]}]}]}
  Cases: every data/real/capture_*.jsonl (trimmed to 6 minutes; all are shorter today) and synthetic streams
  (synth_*): emotion-map face cues, near-field gate before and after the room baseline completes, background /
  low-presence tracks, break clock + absence drop + window eviction, lux 1 Hz sampling, withdrawal and body cues.
  Produced by recording.replay_offline (clock fixed to wall0/mono0) driving a stand-in runtime whose push_obs
  is Aggregator.push and whose step(wall, mono) is: records = Aggregator.flush(wall, mono, local_tod(wall));
  append to RollingWindow; Runtime._evict(wall); checkpoint every target (sorted non-room ids in records) on
  every tick, features_from_window(snapshot, room or snapshot). Aggregator(settings, session_id, source=source).
  replay_offline arithmetic (left to right, doubles): tick wall = wall0 + next_boundary - t0, restamped obs
  wall = wall0 + obs.t - t0 and t = mono0 + obs.t - t0, where t0 = first obs.t and boundaries start at t0 + tick_s.
  NOTE local_tod uses datetime.fromtimestamp, which first rounds the timestamp to whole microseconds (half
  even) and carries into the seconds: a wall of ...04.99999976 reads as second 05.

models.json.gz
  {version, sets: [{state_set, model_id, states, artifacts, temperatures: {m2, m0}, m2_features: [names],
                    m0_features: [names],
                    cases: [{name, tod_h, x: [123] (unclipped), xc: [123] (clipped), m2_raw: [K],
                             m2_contrib: [K][nM2 + 1] (last = bias), p: [K], ood: [names], k,
                             m0: null | {raw: [K], p: [K], k0, conf, face_push},
                             raw_override: true|false|null, label, confidence, layers: [LayerDict]}]}]}
  ServedPredictor.predict_vector(x, tod_h); m0 is null when x[face_now] is NaN; p includes the OOD shrink.
  artifacts is the directory relative to the repo root ("artifacts" or "artifacts/all").

explain.json.gz
  {version,
   describe: [{feature, value, tod_h, absolute, out: null | {display, unit, level, scope, layer}}],
   layers: [{name, contribs: {feature: shap}, x: {feature: value}, absolute: null | {feature: value},
             tod_h, drivers_per_layer, out: [LayerDict]}],               # build_layers(contribs, x, ...)
   top_drivers: [{layers: [LayerDict], k, effect: "supports"|"opposes"|null, out: [DriverDict]}],
   channel_of: [{feature, out}], band: [{confidence, out}],
   fact_sheets: [{args: {build_fact_sheet kwargs; layers as LayerDicts}, out: FactSheetDict}],
   template: [{sheet: FactSheetDict, out: text}]}
  describe value/absolute null mean None (describe treats None and NaN alike). The grid covers all 123
  features (plus a few synthetic names that reach the `{value:g}` fallbacks) over rounding ties, signed zero,
  negatives, huge values (1e21, 1e300) and NaN.

audit.json.gz
  {version, genesis,
   cases: [{name, entries: [{seq, ts, kind, payload, prev, hash, clock, blob, floats}], verify: [ok, n]}],
   tampered: [{name, entries: [entry without clock/blob/floats], verify: [ok, n]}],
   rejects: [{name, kind, payload, error}]}
  Entries are read back from the AuditLog file. clock = the patched time.time() (ts = round(clock, 3));
  blob = json.dumps({seq, ts, kind, payload}, sort_keys=True, separators=(",", ":")), the exact text hashed:
  hash = sha256(prev + blob). floats = JSON pointers (RFC 6901, relative to the entry) of every value that is
  a Python float, e.g. "/ts", "/payload/confidence"; every other number is a Python int. Ints in payloads are
  exactly representable as doubles. verify = AuditLog.verify() on that file.

runtime.json.gz
  {version, runs: [{name, scenario, state_set, room_only, model_id, settings: SettingsDict, wall0, mono0,
                    session_id, hello: HelloDict,
                    ticks: [{wall, mono, records: [TickRecordDict],
                             messages: [message dicts in emission order]}],
                    audit: [entry dicts as in audit.json.gz (with blob/floats; clock = tick wall)]}]}
  A fresh Runtime(Settings(state_set=...), EventBus(), served, gate=PurgeGate(), narrator=None) whose source is
  set directly to a ScenarioSource (no use_scenario, so the audit chain holds only the 'prediction' entries
  the ticks write and starts at the genesis hash); runtime.room_only set as listed; step(wall, mono) per tick.
  messages are model_dump() of every model published on the EventBus during that step (governance, then per
  target state_vector and prediction; no explanation messages: there is no narrator). hello mirrors
  valence/api/app.py hello() for this runtime (feature_version = served.model_id).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import valence.governance.audit as audit_mod  # noqa: E402
import valence.runtime.recording as recording  # noqa: E402
from valence.config import Settings  # noqa: E402
from valence.contracts import (  # noqa: E402
    STATE_DISPLAY,
    STATE_TIER,
    AudioObs,
    FactSheet,
    Hello,
    Layer,
    PersonObs,
    StateInfo,
    TickRecord,
    VideoObs,
    obs_from_dict,
    obs_to_dict,
)
from valence.explain.describe import describe  # noqa: E402
from valence.explain.explainer import build_fact_sheet, build_layers, channel_of, top_drivers  # noqa: E402
from valence.explain.narrator_client import band  # noqa: E402
from valence.explain.template import template_text  # noqa: E402
from valence.features.compute import (  # noqa: E402
    ABS_ROOM,
    FEATURE_INDEX,
    FEATURE_NAMES,
    PERSON_FIELDS,
    ROOM_FIELDS,
    compute,
    features_from_window,
)
from valence.features.window import ROOM, RollingWindow  # noqa: E402
from valence.governance.audit import GENESIS, AuditLog, AuditPayloadError  # noqa: E402
from valence.governance.purge import PurgeGate  # noqa: E402
from valence.models.served import ServedPredictor, _softmax, load_served  # noqa: E402
from valence.runtime.aggregator import Aggregator  # noqa: E402
from valence.runtime.bus import EventBus  # noqa: E402
from valence.runtime.engine import Runtime, ScenarioSource, local_tod  # noqa: E402
from valence.sim.scenarios import render  # noqa: E402

VERSION = 1
WALL0 = 1792850400.0  # Saturday 2026-10-24 14:00:00 UTC
MONO0 = 1000.0
OUT_DIR = ROOT / "frontend" / "src" / "test" / "fixtures" / "engine"
FIXTURES = ("features", "aggregator", "models", "explain", "audit", "runtime")
SIZE_BUDGET = 6 * 1024 * 1024  # all six files together, gzipped

# ---------- encoding ----------


def jsonable(v: Any) -> Any:
    """Plain JSON types: NaN -> None, numpy -> Python, tuples -> lists, models -> model_dump(). Refuses inf."""
    if isinstance(v, BaseModel):
        return jsonable(v.model_dump())
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return jsonable(dataclasses.asdict(v))
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key {k!r}")
            out[k] = jsonable(x)
        return out
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            raise ValueError("inf cannot be written to a fixture")
        return f
    if v is None or isinstance(v, (int, str)):
        return v
    raise TypeError(f"cannot encode {type(v).__name__}")


def encode_fixture(obj: dict[str, Any]) -> bytes:
    """Compact JSON (repr floats, NaN refused after conversion), gzip with mtime=0 (byte-identical reruns)."""
    text = json.dumps(jsonable(obj), allow_nan=False, separators=(",", ":"))
    return gzip.compress(text.encode("ascii"), compresslevel=9, mtime=0)


def write_fixture(path: Path, obj: dict[str, Any]) -> int:
    data = encode_fixture(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(path.read_bytes()))


def nn(v: Any) -> Any:
    """null -> NaN, recursively (the inverse of the NaN encoding for numeric arrays)."""
    if v is None:
        return math.nan
    if isinstance(v, list):
        return [nn(x) for x in v]
    return v


def record_dicts(records: list[TickRecord]) -> list[dict[str, Any]]:
    return [r.to_dict() for r in records]


def record_from_dict(d: dict[str, Any]) -> TickRecord:
    return TickRecord(**{k: (math.nan if v is None else v) for k, v in d.items()})


# ---------- clocks ----------


@contextlib.contextmanager
def utc_tz() -> Iterator[None]:
    """TZ=UTC for local_tod (datetime.fromtimestamp); restores the caller's zone afterwards."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


class _Clock:
    """Stands in for the ``time`` module inside one valence module: fixed time() / monotonic()."""

    def __init__(self, wall: float, mono: float = 0.0) -> None:
        self.wall = wall
        self.mono = mono

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@contextlib.contextmanager
def patched(module: Any, attr: str, value: Any) -> Iterator[Any]:
    old = getattr(module, attr)
    setattr(module, attr, value)
    try:
        yield value
    finally:
        setattr(module, attr, old)


def patched_clock(module: Any, clock: _Clock):
    """Replace the ``time`` module a valence module imported (its time.time() / time.monotonic())."""
    return patched(module, "time", clock)


# ---------- scenario replay (Runtime.step without the predictor) ----------


@lru_cache(maxsize=None)
def _rendered(name: str, state_set: str):
    return render(name, state_set=state_set)


def scenario_source(name: str, state_set: str) -> ScenarioSource:
    src = ScenarioSource(_rendered(name, state_set), name)
    src.session_id = f"replay-{name}-golden"  # the real id carries a random suffix
    return src


def targets_of(records: list[TickRecord], room_only: bool) -> list[int]:
    return [ROOM] if room_only else sorted({r.person_id for r in records if r.person_id != ROOM})


def evict(window: RollingWindow, settings: Settings, wall: float) -> None:
    """Runtime._evict itself, on a stand-in runtime that only has the window."""
    Runtime._evict(SimpleNamespace(window=window, s=settings, smoothers={}, latest={}), wall)


def window_inputs(window: RollingWindow, pid: int) -> tuple[list[TickRecord], list[TickRecord]] | None:
    person = window.snapshot(pid)
    if not person:
        return None
    room = window.room()
    return person, (room or person)


def replay_scenario(name: str, state_set: str, settings: Settings,
                    max_ticks: int | None = None) -> Iterator[tuple[int, float, float, list[TickRecord], RollingWindow]]:
    """Yield (i, wall, mono, records, window) after the window update and eviction of tick i."""
    src = scenario_source(name, state_set)
    window = RollingWindow(settings.window_ticks)
    i = 0
    while max_ticks is None or i < max_ticks:
        wall, mono = WALL0 + i * settings.tick_s, MONO0 + i * settings.tick_s
        records = src.next_tick(wall, mono)
        if records is None:
            return
        for r in records:
            window.append(r)
        evict(window, settings, wall)
        yield i, wall, mono, records, window
        i += 1


def n_scenario_ticks(name: str, state_set: str) -> int:
    return int(_rendered(name, state_set)["tick"].nunique())


# ---------- features ----------

FEATURE_SCENARIOS: tuple[tuple[str, str, bool], ...] = (  # (scenario, state_set, room_only)
    ("collaborative_happy", "core", False),
    ("context_blindness", "core", False),
    ("exhausted", "core", False),
    ("focused", "core", False),
    ("frustrated_masked", "core", False),
    ("full_demo", "core", False),
    ("loud_venue", "core", False),
    ("overwhelmed", "all", False),
    ("tense", "all", False),
    ("collaborative_happy", "core", True),
    ("context_blindness", "core", True),  # includes ticks with nobody present
)


def features_scenario(name: str, state_set: str, room_only: bool, settings: Settings | None = None,
                      max_ticks: int | None = None, every: int = 2) -> dict[str, Any]:
    settings = settings or Settings(state_set=state_set)
    last = (max_ticks or n_scenario_ticks(name, state_set)) - 1
    ticks, checkpoints = [], []
    for i, wall, mono, records, window in replay_scenario(name, state_set, settings, max_ticks):
        ticks.append({"wall": wall, "mono": mono, "records": record_dicts(records)})
        for pid in targets_of(records, room_only):
            inputs = window_inputs(window, pid)
            if inputs is None:
                continue
            person, room = inputs
            if i % every == 0 or len(person) <= 3 or i == last:
                checkpoints.append({"tick": i, "person_id": pid, "features": features_from_window(person, room)})
    return {"name": name, "state_set": state_set, "room_only": room_only,
            "session_id": f"replay-{name}-golden",
            "settings": {"person_drop_min": settings.person_drop_min, "tick_s": settings.tick_s,
                         "window_ticks": settings.window_ticks},
            "ticks": ticks, "checkpoints": checkpoints}


def _edge(name: str, person: dict[str, np.ndarray], room: dict[str, np.ndarray]) -> dict[str, Any]:
    return {"name": name, "person": person, "room": room, "features": compute(person, room)}


def compute_edge_cases() -> list[dict[str, Any]]:
    """compute() on hand-made windows: degenerate lengths, NaN patterns, truncation, guards and extremes."""
    rng = np.random.default_rng(20261024)
    nan = math.nan

    def person(n: int, nan_frac: float = 0.0) -> dict[str, np.ndarray]:
        p = {
            "motion_energy": rng.gamma(2.0, 0.03, n), "posture_slump": rng.random(n), "head_in_hands_frac": rng.random(n),
            "clenched_fists_count": rng.poisson(0.4, n).astype(float), "social_withdrawal_flag": (rng.random(n) < 0.2) * 1.0,
            "n_people": rng.integers(1, 4, n).astype(float), "hours_since_break": np.linspace(0.5, 3.5, n),
            "tod_h": (14.0 + np.arange(n) * 5 / 3600) % 24, "face_valence": rng.uniform(-1, 1, n), "face_visible": rng.random(n),
        }
        if nan_frac:
            for k in ("motion_energy", "posture_slump", "head_in_hands_frac", "face_valence"):
                p[k][rng.random(n) < nan_frac] = nan
        return p

    def room(n: int, nan_frac: float = 0.0, baseline: bool = True) -> dict[str, np.ndarray]:
        r = {"ambient_db": rng.normal(55, 6, n), "light_lux": rng.normal(320, 60, n),
             "laughter_count": rng.poisson(0.2, n).astype(float), "compliments_count": rng.poisson(0.1, n).astype(float)}
        if baseline:
            r["room_db_base"] = np.full(n, 52.25)
            r["room_lux_base"] = np.full(n, 300.0)
        if nan_frac:
            for k in ("ambient_db", "light_lux"):
                r[k][rng.random(n) < nan_frac] = nan
        return r

    def full(n: int, value: float, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {f: np.full(n, value) for f in fields}

    cases = [
        _edge("single_nan_tick", full(1, nan, PERSON_FIELDS), full(1, nan, ROOM_FIELDS)),
        _edge("single_tick", person(1), room(1)),
        _edge("two_ticks", person(2), room(2)),
        _edge("three_ticks_slope_minimum", person(3), room(3)),
        _edge("four_ticks_with_nans", person(4, 0.5), room(4, 0.5)),
        _edge("constant_180", full(180, 0.25, PERSON_FIELDS), full(180, 50.0, ROOM_FIELDS)),
        _edge("random_240_truncated_to_180", person(240, 0.1), room(240, 0.1)),
        _edge("person_60_room_180", person(60, 0.05), room(180, 0.05)),
        _edge("person_13_room_61", person(13), room(61)),
        _edge("no_baseline_keys", person(30), room(30, baseline=False)),
        _edge("all_nan_200", full(200, nan, PERSON_FIELDS), full(200, nan, ROOM_FIELDS)),
    ]
    # Events: counts at the last tick, at the oldest kept tick, beyond the 180-tick cut, NaN counts.
    p, r = person(200), room(200)
    p["clenched_fists_count"][:] = 0.0
    p["clenched_fists_count"][20] = 2.0  # oldest kept tick is index 20 -> since = 179 * 5 = 895
    r["laughter_count"][:] = 0.0
    r["laughter_count"][-1] = 3.0
    r["compliments_count"][:] = 0.0
    r["compliments_count"][5] = 1.0  # dropped by the 180-tick cut -> since = 900
    r["compliments_count"][-3] = nan
    p["social_withdrawal_flag"][-2:] = nan
    cases.append(_edge("events_edges_and_nan_counts", p, r))
    # Lux guards: zero / negative light, zero baseline, NaN last baseline (the last finite one is used).
    p, r = person(40), room(40)
    r["light_lux"][-1] = 0.0
    r["light_lux"][-5:-1] = -3.0
    r["room_lux_base"][:] = 0.0
    r["room_db_base"][-3:] = nan
    r["room_db_base"][:10] = 48.5
    cases.append(_edge("lux_guards_and_last_finite_baseline", p, r))
    p, r = person(25), room(25)
    r["room_lux_base"][:] = nan
    r["room_db_base"][:] = nan
    cases.append(_edge("baseline_all_nan", p, r))
    # Even counts of finite values (median midpoint), signed zero, NaN / wrapped time of day.
    p, r = person(12), room(12)
    r["ambient_db"][:] = [50.0, 51.0, nan, 53.0, 54.0, 55.5, nan, 57.0, 58.0, 59.0, 60.0, 61.0]
    p["motion_energy"][:] = -0.0
    p["tod_h"][-1] = nan
    cases.append(_edge("even_median_signed_zero_tod_nan", p, r))
    p, r = person(70), room(70)
    p["tod_h"][:] = 23.999
    p["face_valence"][:] = nan
    p["face_visible"][:] = 0.0
    cases.append(_edge("late_tod_no_face", p, r))
    p, r = person(50), room(50)
    for k in ("ambient_db", "light_lux"):
        r[k][:] = rng.choice([1e100, -1e100, 3.0], 50)
    p["motion_energy"][:] = rng.choice([1e100, 0.0], 50)
    cases.append(_edge("huge_values", p, r))
    p, r = person(9), room(9)
    for f in p:
        p[f][:] = nan
    p["n_people"][:] = 1.0
    cases.append(_edge("person_all_nan_but_n_people", p, r))
    return cases


def export_features(max_ticks: int | None = None, scenarios=FEATURE_SCENARIOS) -> dict[str, Any]:
    return {"version": VERSION, "wall0": WALL0, "mono0": MONO0,
            "scenarios": [features_scenario(n, s, ro, max_ticks=max_ticks) for n, s, ro in scenarios],
            "edge_cases": compute_edge_cases()}


# ---------- aggregator ----------

SYNTH_T0 = 20_000.0  # monotonic clock of the synthetic streams
SYNTH_WALL = 1_790_000_000.0  # their original wall clock (replay re-stamps it)


class _Stream:
    """Builds a synthetic observation stream; obs are kept in time order (stable for equal t)."""

    def __init__(self) -> None:
        self.obs: list[tuple[float, int, Any]] = []
        self.frame_seq = 0
        self.block_seq = 0

    def video(self, t: float, persons: list[PersonObs], lux: float = 300.0, lux_quality: float = 0.8,
              decoded: int | None = None) -> None:
        self.frame_seq += 1
        o = VideoObs(t=SYNTH_T0 + t, wall=SYNTH_WALL + t, frame_seq=self.frame_seq,
                     frames_decoded=self.frame_seq if decoded is None else decoded, lux=lux, lux_quality=lux_quality,
                     frame_w=1280, frame_h=720, mirrored=True, persons=tuple(persons))
        self.obs.append((t, len(self.obs), o))

    def audio(self, t: float, db: float, laugh: int = 0, compl: int = 0, speech: float = 0.0) -> None:
        self.block_seq += 1
        o = AudioObs(t=SYNTH_T0 + t, wall=SYNTH_WALL + t, block_seq=self.block_seq, db_spl=db,
                     rms_power=math.nan if math.isnan(db) else 10.0 ** ((db - 94.0) / 10.0), speech_prob=speech,
                     laughter_onsets=laugh, compliment_hits=compl)
        self.obs.append((t, len(self.obs), o))

    def done(self) -> list[Any]:
        return [o for _, _, o in sorted(self.obs, key=lambda e: (e[0], e[1]))]


def _desk(track: int = 1, x: float = 0.2, **kw: Any) -> PersonObs:
    return PersonObs(track_id=track, bbox=(x, 0.06, min(1.0, x + 0.45), 0.98), **kw)


def synth_near_field_running_median() -> list[Any]:
    """No completed baseline: < 10 blocks always count, then the 15-min median + 3 dB rule (loudest of 3)."""
    rng = np.random.default_rng(11)
    s = _Stream()
    events = {3: (55.0, 1, 0), 15: (50.5, 1, 0), 20: (56.0, 0, 1), 30: (60.0, 0, 0), 31: (50.0, 1, 0),
              40: (math.nan, 1, 0), 50: (54.0, 2, 1), 58: (49.0, 0, 2), 66: (70.0, 1, 1)}
    for b in range(75):
        db, laugh, compl = events.get(b, (float(rng.normal(50.0, 0.7)), 0, 0))
        s.audio(b + 0.05, db, laugh, compl, speech=float(rng.random()))
    for i in range(375):  # 5 Hz
        t = i * 0.2
        s.video(t, [_desk(motion=float(rng.gamma(2, 0.02)), face_visible=True, face_valence=float(rng.uniform(-0.3, 0.6)))],
                lux=float(rng.normal(310, 5)))
    return s.done()


def synth_near_field_baseline() -> list[Any]:
    """The room baseline completes after 90 blocks; laughter then needs baseline + 3 dB; lux sampled at 1 Hz."""
    rng = np.random.default_rng(12)
    s = _Stream()
    for b in range(150):
        db = float(rng.normal(66.0, 1.0))
        laugh = compl = 0
        if b == 40:
            db = math.nan  # a dropped block: not a baseline sample
        if b in (100, 101):
            db, laugh = 68.0, 1  # within 3 dB of the baseline: another table
        if b == 110:
            db, compl = 71.0, 1
        if b == 120:
            db = 75.0  # the user's laugh ...
        if b == 122:
            db, laugh = 66.0, 1  # ... reported two blocks late
        if b == 130:
            db, laugh, compl = 64.0, 1, 1
        s.audio(b + 0.5, db, laugh, compl)
    t = 0.0
    while t < 150.0:
        lux = math.nan if rng.random() < 0.1 else float(rng.normal(300.0, 20.0))
        people = [_desk(motion=0.03, face_visible=True, face_valence=0.1)] if 20.0 <= t < 120.0 else []
        s.video(t, people, lux=lux, lux_quality=float(rng.uniform(0.2, 1.0)))
        t = round(t + (0.1 if t < 30.0 else 0.2), 10)
    return s.done()


def synth_background_and_two_people() -> list[Any]:
    """Two desk participants, a small background figure, a tall figure seen in 20% of frames, a leaver."""
    rng = np.random.default_rng(13)
    s = _Stream()
    for i in range(400):  # 40 s at 10 Hz
        t = i * 0.1
        persons = [_desk(1, 0.05, motion=float(rng.gamma(2, 0.02)), fist_onsets=int(i % 37 == 5), slump=bool(i % 3 == 0),
                         face_visible=i % 10 != 0, face_valence=float(rng.uniform(-0.5, 0.5)) if i % 10 else math.nan)]
        if i < 250 or i % 4 == 0:  # person 2 leaves after 25 s (still in 25% of frames: below the presence share)
            persons.append(_desk(2, 0.5, motion=math.nan if i % 5 == 0 else 0.015, slump=True, head_in_hands=i % 2 == 0,
                                 face_visible=False))
        persons.append(PersonObs(track_id=3, bbox=(0.85, 0.3, 0.95, 0.62), motion=0.2))  # background: a third of the height
        if i % 5 == 0:
            persons.append(_desk(4, 0.3, motion=0.05))  # tall but present in 20% of frames
        s.video(t, persons, lux=float(rng.normal(450, 10)))
    for b in range(40):
        s.audio(b + 0.3, float(rng.normal(58, 2)), laugh=int(b in (7, 33)), compl=int(b == 21))
    return s.done()


def synth_break_and_drop() -> list[Any]:
    """Break clock: an 11-min absence restarts it; a 16.7-min absence drops the clock and the window slot."""
    rng = np.random.default_rng(14)
    s = _Stream()
    present = [(0.0, 120.0), (780.0, 900.0), (1900.0, 1950.0)]
    t = 0.0
    while t < 1950.0:
        here = any(a <= t < b for a, b in present)
        persons = [_desk(1, motion=float(rng.gamma(2, 0.02)), face_visible=True, face_valence=0.2)] if here else []
        if 60.0 <= t < 90.0:
            persons.append(_desk(2, 0.5, motion=0.04))
        s.video(t, persons, lux=float(rng.normal(280, 8)))
        t = round(t + (0.5 if here else 2.0), 10)
    for k in range(975):
        s.audio(k * 2.0 + 0.7, float(rng.normal(45, 1.5)))
    return s.done()


def synth_lux_sampling() -> list[Any]:
    """Irregular frame spacing around the 1 s lux sampling step, NaN-lux frames, a capture restart."""
    rng = np.random.default_rng(15)
    s = _Stream()
    t = 0.0
    n = restart = 0
    while t < 30.0:
        lux = math.nan if n % 7 == 3 else float(200.0 + 10.0 * t)
        if t >= 15.0 and not restart:
            restart = n  # the capture restarts: its frame counter starts again at 1
        s.video(t, [_desk(motion=0.02)] if t > 4.0 else [], lux=lux, decoded=n + 1 - restart)
        t = round(t + float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.25])), 10)
        n += 1
    for b in range(30):
        s.audio(b + 0.9, float(rng.normal(52, 1)))
    return s.done()


def synth_withdrawal_and_body() -> list[Any]:
    """Withdrawal (slumped and still), fists, head in hands, a faceless tick, a tick with undefined motion."""
    rng = np.random.default_rng(16)
    s = _Stream()
    for i in range(300):  # 60 s at 5 Hz
        t = i * 0.2
        tick = int(t // 5)
        slump = tick in (2, 3, 4) or (tick == 5 and i % 2 == 0)
        motion = math.nan if tick == 7 else (0.01 if tick in (2, 3) else float(rng.gamma(2, 0.03)))
        face = tick != 6
        s.video(t, [_desk(fist_onsets=int(tick == 8 and i % 4 == 0), slump=slump, head_in_hands=tick in (4, 9),
                          motion=motion, face_visible=face, face_valence=float(rng.uniform(-0.8, 0.2)) if face else math.nan)],
                lux=float(rng.normal(150, 4)))
    for b in range(60):
        s.audio(b + 0.1, float(rng.normal(41, 0.5)))
    return s.done()


def synth_face_cues() -> list[Any]:
    """Emotion-map face cues: per-tick medians over face frames, |yaw| median, head-away share, blinks."""
    rng = np.random.default_rng(17)
    s = _Stream()

    def cue(p: float) -> float:
        return math.nan if rng.random() < p else float(np.clip(rng.normal(0.0, 0.3), -1.0, 1.0))

    for i in range(500):  # 50 s at 10 Hz, two people
        t = i * 0.1
        tick = int(t // 5)
        persons = []
        for track, x in ((1, 0.05), (2, 0.5)):
            visible = tick != 3 and i % 9 != 0  # tick 3: the face is never read (cues NaN, blinks still count)
            valence = float(rng.uniform(-0.6, 0.6)) if visible and i % 13 else math.nan  # visible, valence unread
            no_pose = tick == 5 or i % 11 == 0  # tick 5: pose never read -> head_away NaN
            yaw = math.nan if no_pose else float(rng.choice([-35.0, -21.0, -20.0, -3.5, 0.0, 4.25, 20.0, 27.5]))
            pitch = math.nan if no_pose else float(rng.choice([-25.0, -8.0, 0.5, 12.0, 20.0, 22.5]))
            persons.append(_desk(track, x, motion=float(rng.gamma(2, 0.02)), face_visible=visible, face_valence=valence,
                                 brow_down=cue(0.1), eye_open=cue(0.1), mouth_frown=cue(0.2), smile=cue(0.15) if track == 1 else 0.25,
                                 head_yaw=yaw, head_pitch=pitch, blink_onsets=int(rng.random() < 0.06)))
        s.video(t, persons, lux=float(rng.normal(330, 6)))
    for b in range(50):
        s.audio(b + 0.4, float(rng.normal(47, 1)))
    return s.done()


SYNTHETIC_STREAMS = {
    "synth_face_cues": synth_face_cues,
    "synth_near_field_running_median": synth_near_field_running_median,
    "synth_near_field_baseline": synth_near_field_baseline,
    "synth_background_and_two_people": synth_background_and_two_people,
    "synth_break_and_drop": synth_break_and_drop,
    "synth_lux_sampling": synth_lux_sampling,
    "synth_withdrawal_and_body": synth_withdrawal_and_body,
}
RECORDINGS = ("capture_audio", "capture_baseline", "capture_covered", "capture_fists", "capture_keyboard_gaze",
              "capture_slump", "capture_typing")
MAX_RECORDING_S = 360.0  # trim long recordings to ~6 minutes


class _AggregatorRuntime:
    """What replay_offline needs from a Runtime: push_obs -> Aggregator.push, step -> LiveSource tick."""

    def __init__(self, settings: Settings, session_id: str, source: str) -> None:
        self.s = settings
        self.aggregator = Aggregator(settings, session_id, source=source)
        self.window = RollingWindow(settings.window_ticks)
        self.ticks: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []

    def push_obs(self, obs: Any) -> None:
        self.aggregator.push(obs)

    def step(self, wall: float, mono: float) -> list[dict[str, Any]]:
        records = self.aggregator.flush(wall, mono, local_tod(wall))
        for r in records:
            self.window.append(r)
        evict(self.window, self.s, wall)
        i = len(self.ticks)
        self.ticks.append({"wall": wall, "mono": mono, "records": record_dicts(records)})
        for pid in targets_of(records, False):
            inputs = window_inputs(self.window, pid)
            if inputs is not None:
                self.checkpoints.append({"tick": i, "person_id": pid, "features": features_from_window(*inputs)})
        return []


def replay_aggregator(obs: list[Any], settings: Settings, session_id: str = "golden-live",
                      source: str = "replay") -> _AggregatorRuntime:
    """recording.replay_offline with its clock fixed to (WALL0, MONO0), driving the aggregator stand-in."""
    rt = _AggregatorRuntime(settings, session_id, source)
    # replay_offline reads a JSONL path through iter_obs; hand it the stream from memory instead.
    with utc_tz(), patched_clock(recording, _Clock(WALL0, MONO0)), patched(recording, "iter_obs", lambda _: iter(obs)):
        recording.replay_offline(rt, "<memory>", tick_s=settings.tick_s)
    return rt


def load_recording(name: str) -> list[Any]:
    obs = list(recording.iter_obs(ROOT / "data" / "real" / f"{name}.jsonl"))
    if obs and obs[-1].t - obs[0].t > MAX_RECORDING_S:
        obs = [o for o in obs if o.t - obs[0].t <= MAX_RECORDING_S]
    return obs


def aggregator_case(name: str, kind: str, obs: list[Any], settings: Settings) -> dict[str, Any]:
    session_id, source = "golden-live", "replay"
    rt = replay_aggregator(obs, settings, session_id, source)
    return {"name": name, "kind": kind, "wall0": WALL0, "mono0": MONO0, "tz": "UTC", "tick_s": settings.tick_s,
            "session_id": session_id, "source": source, "settings": dataclasses.asdict(settings),
            "obs": [obs_to_dict(o) for o in obs], "ticks": rt.ticks, "checkpoints": rt.checkpoints}


def export_aggregator(recordings=RECORDINGS, synthetic=tuple(SYNTHETIC_STREAMS)) -> dict[str, Any]:
    settings = Settings()
    cases = [aggregator_case(n, "recording", load_recording(n), settings) for n in recordings]
    cases += [aggregator_case(n, "synthetic", SYNTHETIC_STREAMS[n](), settings) for n in synthetic]
    return {"version": VERSION, "settings": dataclasses.asdict(settings), "cases": cases}


# ---------- models ----------

MODEL_SETS = (("core", "artifacts"), ("all", "artifacts/all"))
MODEL_SCENARIOS = {  # state_set -> (scenario, render state_set) windows sampled for cases
    "core": (("focused", "core"), ("frustrated_masked", "core"), ("full_demo", "core"), ("collaborative_happy", "core"),
             ("exhausted", "core"), ("context_blindness", "core"), ("loud_venue", "core")),
    "all": (("overwhelmed", "all"), ("tense", "all"), ("frustrated_masked", "core"), ("exhausted", "core")),
}


@lru_cache(maxsize=None)
def served_for(state_set: str) -> ServedPredictor:
    rel = dict(MODEL_SETS)[state_set]
    served = load_served(ROOT / rel, state_set)
    if served is None:
        raise SystemExit(f"no trained {state_set} artifacts in {rel}")
    return served


def scenario_windows(name: str, state_set: str, every: int, offset: int = 3,
                     max_ticks: int | None = None) -> list[tuple[str, np.ndarray, float]]:
    """(case name, unclipped x, tod_h) for every person every ``every`` ticks (as the runtime would predict)."""
    out = []
    settings = Settings()
    for i, _, _, records, window in replay_scenario(name, state_set, settings, max_ticks):
        if i % every != offset % every:
            continue
        for pid in targets_of(records, False):
            inputs = window_inputs(window, pid)
            if inputs is not None:
                out.append((f"{name}/t{i:03d}/p{pid}", features_from_window(*inputs), inputs[0][-1].tod_h))
    return out


def split_info(served: ServedPredictor) -> tuple[dict[str, set[str]], dict[str, list[float]]]:
    """Per M2 feature: the missing types of its splits and its thresholds (from dump_model)."""
    missing: dict[str, set[str]] = {}
    thresholds: dict[str, list[float]] = {}
    names = served.m2.feature_name()

    def walk(node: dict[str, Any]) -> None:
        if "split_feature" not in node:
            return
        f = names[node["split_feature"]]
        missing.setdefault(f, set()).add(node["missing_type"])
        thresholds.setdefault(f, []).append(float(node["threshold"]))
        walk(node["left_child"])
        walk(node["right_child"])

    for tree in served.m2.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    return missing, thresholds


def perturbations(served: ServedPredictor, base_name: str, x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    fi = FEATURE_INDEX
    missing, thresholds = split_info(served)
    none_type = [f for f in served.cols2 if missing.get(f) == {"None"}]
    nan_type = [f for f in served.cols2 if missing.get(f) == {"NaN"}]
    finite = [i for i in range(len(FEATURE_NAMES)) if math.isfinite(served.lo[i]) and math.isfinite(served.hi[i])]
    out: list[tuple[str, np.ndarray]] = []

    def case(name: str, fn) -> None:
        y = x.copy()
        fn(y)
        out.append((f"{base_name}/{name}", y))

    def set_nan(names):
        def fn(y):
            for n in names:
                y[fi[n]] = math.nan
        return fn

    def no_face(y):
        for n in ("face_now", "face_mean_1m", "face_mean_5m", "face_mean_15m", "face_std_5m"):
            y[fi[n]] = math.nan
        y[fi["face_visible_now"]] = 0.0
        y[fi["face_visible_frac_5m"]] = 0.0

    def tiny(y):
        for j, n in enumerate(none_type[:6] + nan_type[:6]):
            y[fi[n]] = (1e-36, -1e-36, -0.0, 0.0, 1e-35, -1e-35)[j % 6]

    def at_bound(which):
        def fn(y):
            for i in finite:
                y[i] = served.lo[i] if which == "lo" else served.hi[i]
        return fn

    def beyond(y):  # clipped but inside the OOD band (margin is 0.5 x range)
        for i in finite[::2]:
            y[i] = served.hi[i] + 0.25 * (served.hi[i] - served.lo[i])
        for i in finite[1::2]:
            y[i] = served.lo[i] - 0.25 * (served.hi[i] - served.lo[i])

    def ood(names):
        def fn(y):
            for n in names:
                i = fi[n]
                y[i] = served.hi[i] + 0.6 * (served.hi[i] - served.lo[i]) + 1.0
        return fn

    def huge(y):
        y[:] = 1e300
        y[fi["face_now"]] = -1e300

    def ties(nextafter: bool):
        def fn(y):
            for n in sorted(thresholds)[:: max(1, len(thresholds) // 16)]:
                thr = thresholds[n][0]
                y[fi[n]] = np.nextafter(thr, math.inf) if nextafter else thr
        return fn

    def face_extremes(y):
        y[fi["face_now"]] = 1.0
        y[fi["face_mean_1m"]] = 1.0
        y[fi["face_visible_now"]] = 1.0
        y[fi["face_visible_frac_5m"]] = 1.0

    case("nan_on_none_missing", set_nan(none_type[:10]))
    case("nan_on_nan_missing", set_nan(nan_type[:12]))
    case("all_nan", lambda y: y.fill(math.nan))
    case("no_face", no_face)
    case("zero_and_tiny", tiny)
    case("clip_lo_exact", at_bound("lo"))
    case("clip_hi_exact", at_bound("hi"))
    case("clip_beyond_in_band", beyond)
    case("ood_one", ood(["db_room_now"]))
    case("ood_three", ood(["db_room_now", "db_room_mean_1m", "motion_now"]))
    case("ood_four_shrinks", ood(["db_room_now", "db_room_mean_1m", "motion_now", "motion_max_1m"]))
    case("huge_all", huge)
    case("threshold_ties", ties(False))
    case("threshold_next_up", ties(True))
    case("face_extremes", face_extremes)
    return out


def model_case(served: ServedPredictor, name: str, x: np.ndarray, tod_h: float | None) -> dict[str, Any]:
    """Every intermediate of ServedPredictor.predict_vector, checked against the method's own output."""
    out = served.predict_vector(x, tod_h=tod_h)
    xc, ood = served._clip(x)
    raw, contrib = served._m2(xc)
    p = _softmax(raw, served.t2)
    if len(ood) > 3:
        p = 0.7 * p + 0.3 / len(p)
    k = int(np.argmax(p))
    m0 = None
    if not math.isnan(x[FEATURE_INDEX["face_now"]]):
        raw0 = served.m0.predict(xc[served.idx0][None, :], raw_score=True, num_threads=1)[0]
        p0 = _softmax(raw0, served.t0)
        k0 = int(np.argmax(p0))
        m0 = {"raw": raw0, "p": p0, "k0": k0, "conf": float(p0[k0]),
              "face_push": float(contrib[k0, :-1][served.face_cols].sum())}
        assert out.face_only_label == served.states[k0] and out.face_only_confidence == float(p0[k0])
    assert out.label == served.states[k] and out.confidence == float(p[k]) and out.ood_features == ood
    return {"name": name, "tod_h": tod_h, "x": x, "xc": xc, "m2_raw": raw, "m2_contrib": contrib, "p": p, "ood": ood,
            "k": k, "m0": m0, "raw_override": out.raw_override, "label": out.label, "confidence": out.confidence,
            "layers": out.layers}


def model_set(state_set: str, every: int = 15, max_ticks: int | None = None, perturb: bool = True,
              scenarios: tuple[tuple[str, str], ...] | None = None) -> dict[str, Any]:
    served = served_for(state_set)
    windows: list[tuple[str, np.ndarray, float | None]] = []
    for name, render_set in scenarios or MODEL_SCENARIOS[state_set]:
        windows += scenario_windows(name, render_set, every, max_ticks=max_ticks)
    cases = [model_case(served, n, x, tod) for n, x, tod in windows]
    if perturb:
        bases = [w for w in windows if w[0].startswith(("frustrated_masked/t1", "focused/t09", "overwhelmed/t12"))][:2]
        bases = bases or windows[:1]
        for base_name, x, tod in bases:
            cases += [model_case(served, n, y, tod) for n, y in perturbations(served, base_name, x)]
    temps = {"m2": served.t2, "m0": served.t0}
    return {"state_set": state_set, "model_id": served.model_id, "states": list(served.states),
            "artifacts": dict(MODEL_SETS)[state_set], "temperatures": temps, "m2_features": served.cols2,
            "m0_features": served.cols0, "cases": cases}


def export_models(every: int = 15) -> dict[str, Any]:
    return {"version": VERSION, "sets": [model_set(s, every=every) for s, _ in MODEL_SETS]}


# ---------- explain ----------

DESCRIBE_VALUES = (
    0.0, -0.0, 1e-05, 0.0001, 1.234e-05, 0.0005, 0.005, 0.015, 0.02, 0.035, 0.1, 0.125, 0.25, 0.3, 0.35, 0.375,
    0.45, 0.5, 0.6, 0.75, 0.8, 0.99, 1.0, 1.05, 1.5, 2.0, 2.5, 2.675, 3.0, 3.5, 4.0, 6.0, 7.5, 8.0, 40.0, 44.5, 45.0,
    57.5, 58.0, 64.5, 65.0, 72.25, 100.0, 199.5, 200.0, 250.0, 449.5, 450.0, 899.5, 900.0, 1234.5678, 123456.0,
    1234567.0, -0.2, -0.3, -0.5, -1.5, -2.5, -3.0, -3.5, -40.0, -250.5, 1e16, 1e21, 1e300, -1e300, math.nan,
)
TOD_VALUES = (None, 0.0, 5.999, 6.0, 9.508333333333333, 13.5, 14.999999, 15.9999, 21.99, 22.0, 23.99, 23.999999)
ABSOLUTE_VALUES = (None, math.nan, 0.0, 72.4, 72.5, 73.5, 299.5, 1e21)
EXTRA_DESCRIBE_FEATURES = ("db_p90", "db_p90_5m", "custom_metric", "X-Y")  # fallbacks and the p90 branch


def describe_out(feature: str, value: float | None, tod_h: float | None, absolute: float | None) -> dict[str, Any] | None:
    d = describe(feature, value, tod_h=tod_h, absolute=absolute)
    return None if d is None else dataclasses.asdict(d)


def describe_grid(features: list[str] | None = None, values=DESCRIBE_VALUES) -> list[dict[str, Any]]:
    out = []
    for f in features or (list(FEATURE_NAMES) + list(EXTRA_DESCRIBE_FEATURES)):
        if f in ("tod_sin", "tod_cos"):
            for tod in TOD_VALUES:
                for v in (0.5, math.nan):
                    out.append({"feature": f, "value": v, "tod_h": tod, "absolute": None, "out": describe_out(f, v, tod, None)})
            continue
        absolutes = ABSOLUTE_VALUES if "_room_" in f else (None,)
        for a in absolutes:
            for v in values:
                out.append({"feature": f, "value": v, "tod_h": 15.5, "absolute": a, "out": describe_out(f, v, 15.5, a)})
    return out


def _layers_case(name: str, contribs: dict[str, float], values: dict[str, float], tod_h: float | None,
                 absolute: dict[str, float] | None = None, drivers_per_layer: int = 3) -> dict[str, Any]:
    out = build_layers(contribs, values, tod_h=tod_h, drivers_per_layer=drivers_per_layer, absolute=absolute)
    return {"name": name, "contribs": contribs, "x": values, "absolute": absolute, "tod_h": tod_h,
            "drivers_per_layer": drivers_per_layer, "out": out}


def _prediction_inputs(served: ServedPredictor, x: np.ndarray) -> tuple[dict, dict, dict]:
    """The exact contribs / values / absolute dicts predict_vector hands to build_layers."""
    xc, _ = served._clip(x)
    raw, contrib = served._m2(xc)
    p = _softmax(raw, served.t2)
    k = int(np.argmax(p))
    values = {n: float(x[FEATURE_INDEX[n]]) for n in served.cols2}
    contribs = {n: float(c) for n, c in zip(served.cols2, contrib[k, :-1]) if c != 0.0}
    absolute = {n: float(x[FEATURE_INDEX[n]]) for n in ABS_ROOM}
    return contribs, values, absolute


def layers_cases(windows: list[tuple[str, np.ndarray, float]]) -> list[dict[str, Any]]:
    served = served_for("core")
    cases = []
    for name, x, tod in windows:
        contribs, values, absolute = _prediction_inputs(served, x)
        cases.append(_layers_case(f"served/{name}", contribs, values, tod, absolute))
    nan = math.nan
    cases += [
        _layers_case("ties_same_channel_keep_first",
                     {"motion_now": 0.25, "motion_ewm_1m": -0.25, "motion_mean_5m": 0.25, "slump_now": 0.1},
                     {"motion_now": 0.04, "motion_ewm_1m": 0.05, "motion_mean_5m": 0.06, "slump_now": 0.5}, 15.0),
        _layers_case("missing_values_and_fill_ratio_skipped",
                     {"fill_ratio": 2.0, "hih_now": 0.3, "fist_cnt_5m": -0.2, "db_room_now": 0.1},
                     {"fill_ratio": 1.0, "fist_cnt_5m": nan, "db_room_now": 4.5}, 15.0),
        _layers_case("tod_without_clock_skipped", {"tod_sin": 0.4, "tod_cos": 0.3, "hours_since_break": -0.1},
                     {"tod_sin": 0.5, "tod_cos": -0.8, "hours_since_break": 2.25}, None),
        _layers_case("tod_with_clock_one_channel", {"tod_sin": 0.4, "tod_cos": 0.3, "hours_since_break": -0.1},
                     {"tod_sin": 0.5, "tod_cos": -0.8, "hours_since_break": 2.25}, 23.999999),
        _layers_case("zero_and_negative_zero_shap", {"slump_now": 0.0, "hih_now": -0.0, "withdraw_now": -1e-05},
                     {"slump_now": 0.7, "hih_now": 0.2, "withdraw_now": 1.0}, 12.0),
        _layers_case("more_channels_than_drivers",
                     {"slump_now": 0.11, "hih_now": -0.12, "motion_now": 0.13, "fist_cnt_1m": 0.14, "withdraw_now": -0.15,
                      "laugh_cnt_15m": 0.05, "compl_since_s": -0.02, "n_people": 0.01},
                     {"slump_now": 0.7, "hih_now": 0.2, "motion_now": 0.01, "fist_cnt_1m": 2.0, "withdraw_now": 1.0,
                      "laugh_cnt_15m": 3.0, "compl_since_s": 900.0, "n_people": 2.0}, 12.0, drivers_per_layer=2),
        _layers_case("value_rounding_ties",
                     {"motion_now": 0.3, "face_now": -0.2, "db_slope_5m": 0.1, "lux_std_15m": 0.05},
                     {"motion_now": 0.00005, "face_now": 1.00005, "db_slope_5m": 2.67455, "lux_std_15m": -0.00015}, 9.0),
        _layers_case("room_relative_absolute_default_to_values",
                     {"db_room_mean_5m": 0.2, "lux_room_now": -0.1, "db_mean_5m": 0.05},
                     {"db_room_mean_5m": 3.5, "lux_room_now": -0.75, "db_mean_5m": 61.5, "lux_now": 120.0}, 16.0),
        _layers_case("room_relative_absolute_nan",
                     {"db_room_mean_5m": 0.2, "lux_room_now": -0.1},
                     {"db_room_mean_5m": 3.5, "lux_room_now": -0.75}, 16.0, absolute={"db_mean_5m": nan, "lux_now": 450.5}),
        _layers_case("empty", {}, {}, 10.0),
        _layers_case("contribution_sum_order",
                     {"db_room_now": 0.1, "db_room_mean_1m": 0.2, "db_room_mean_5m": 0.3, "lux_room_mean_15m": -0.6000000000000001},
                     {"db_room_now": 1.0, "db_room_mean_1m": 2.0, "db_room_mean_5m": 3.0, "lux_room_mean_15m": 0.1}, 10.0,
                     absolute={"db_now": 50.0}),
    ]
    return cases


def _sheet_args(served: ServedPredictor, name: str, x: np.ndarray, tod_h: float, **over: Any) -> dict[str, Any]:
    out = served.predict_vector(x, tod_h=tod_h)
    face = float(x[FEATURE_INDEX["face_now"]])
    lux = float(x[FEATURE_INDEX["lux_now"]])
    args = dict(
        event_id=name, person="Person #1", label=out.label, label_display=STATE_DISPLAY[out.label],
        confidence=out.confidence, probabilities=out.probabilities, window_fill_min=15.0, n_people=1,
        face_only_label=out.face_only_label, face_only_confidence=out.face_only_confidence,
        face_valence=None if math.isnan(face) else face, face_visible=not math.isnan(face),
        mask_override=False, layers=out.layers, face_visible_frac=float(x[FEATURE_INDEX["face_visible_frac_5m"]]),
        lux=None if math.isnan(lux) else lux, ood_features=out.ood_features,
    )
    args.update(over)
    return args


def fact_sheet_cases(windows: list[tuple[str, np.ndarray, float]]) -> list[dict[str, Any]]:
    served = served_for("core")
    cases = []
    for j, (name, x, tod) in enumerate(windows):
        ev = f"p{1 + j % 3}-{j * 37:06d}"
        cases.append(_sheet_args(served, ev, x, tod))
        cases.append(_sheet_args(served, ev + "m", x, tod, mask_override=True, face_only_label="Collaborative_Happy",
                                 face_only_confidence=0.6125, face_valence=0.555, face_visible=True))
        cases.append(_sheet_args(served, ev + "h", x, tod, face_visible=False, face_valence=None))
    some = windows[0]
    cases += [
        _sheet_args(served, "p0-000000", some[1], some[2], person="Room", n_people=3, layers=[], window_fill_min=0.25,
                    confidence=0.125, face_visible_frac=0.0005, lux=None),
        _sheet_args(served, "p2-000123", some[1], some[2], confidence=0.995, window_fill_min=0.35, face_valence=0.375,
                    face_visible_frac=0.0625, lux=299.95, ood_features=["db_room_now", "motion_now", "slump_now", "hih_now"]),
        _sheet_args(served, "p1-999999", some[1], some[2], face_only_label=None, face_only_confidence=None,
                    face_visible=True, face_valence=-0.125, ood_features=["a", "b", "c"]),
        _sheet_args(served, "p3-000042", some[1], some[2], mask_override=True, face_only_label="Focused",
                    face_only_confidence=0.7, face_visible=True, face_valence=0.2, n_people=2,
                    label="Collaborative_Happy", label_display="Collaborative / Happy"),
    ]
    return [{"args": args, "out": build_fact_sheet(**args)} for args in cases]


def top_driver_cases(layer_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for case in layer_cases[:12]:
        layers = case["out"]
        for k, effect in ((5, "supports"), (3, "opposes"), (4, None), (1, "supports")):
            out.append({"layers": layers, "k": k, "effect": effect, "out": top_drivers(layers, k=k, effect=effect)})
    return out


def export_explain(every: int = 40) -> dict[str, Any]:
    windows = []
    for name in ("frustrated_masked", "focused", "context_blindness", "exhausted", "collaborative_happy"):
        windows += scenario_windows(name, "core", every)
    layer_cases = layers_cases(windows)
    sheets = fact_sheet_cases(windows[:: max(1, len(windows) // 12)])
    extra = ["tod_x", "face_visible_frac_5m", "face_now", "x", "_lead", "db"]
    return {
        "version": VERSION,
        "describe": describe_grid(),
        "layers": layer_cases,
        "top_drivers": top_driver_cases(layer_cases),
        "channel_of": [{"feature": f, "out": channel_of(f)} for f in list(FEATURE_NAMES) + extra],
        "band": [{"confidence": c, "out": band(c)} for c in (0.0, 0.25, 0.5999999999999999, 0.6, 0.8499999999999999,
                                                            0.85, 0.9, 1.0)],
        "fact_sheets": sheets,
        "template": [{"sheet": c["out"], "out": template_text(c["out"])} for c in sheets],
    }


# ---------- audit ----------


def float_paths(v: Any, path: str = "") -> list[str]:
    """JSON pointers (RFC 6901) of every Python float inside ``v``."""
    if isinstance(v, bool):
        return []
    if isinstance(v, float):
        return [path]
    if isinstance(v, dict):
        out = []
        for k, x in v.items():
            out += float_paths(x, f"{path}/{k.replace('~', '~0').replace('/', '~1')}")
        return out
    if isinstance(v, list):
        return [p for i, x in enumerate(v) for p in float_paths(x, f"{path}/{i}")]
    return []


def annotate_entries(entries: list[dict[str, Any]], clocks: list[float]) -> list[dict[str, Any]]:
    """Add clock, the exact hashed blob and the float pointers to entries read back from an audit file."""
    out = []
    for e, clock in zip(entries, clocks, strict=True):
        body = {k: e[k] for k in ("seq", "ts", "kind", "payload")}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256((e["prev"] + blob).encode()).hexdigest() == e["hash"]
        out.append({**e, "clock": clock, "blob": blob, "floats": float_paths(e)})
    return out


AUDIT_CASES: dict[str, list[tuple[float, str, dict[str, Any]]]] = {
    "basic": [
        (WALL0, "operator", {"action": "reset", "keep_room": False}),
        (WALL0 + 0.25, "operator", {"action": "replay", "scenario": "full_demo", "speed": 1.0}),
        (WALL0 + 5.0, "prediction", {"person": "Person #1", "label": "Focused", "confidence": 0.8123,
                                     "mask_override": False, "source": "replay",
                                     "layers": {"Ambient Physics": 0.4211, "Facial Surface": -0.0532},
                                     "simulated_fields": []}),
        (WALL0 + 10.0, "operator", {"action": "demo_seed", "scenario": "full_demo", "minutes": 10,
                                    "hours_since_break": None, "simulated_fields": ["room_history"]}),
    ],
    "tricky_floats": [
        (WALL0 + 1.0, "numbers", {"a": 1e-05, "b": 1e16, "c": 0.1, "d": 2.0, "e": -0.0, "f": 2 ** 53, "g": -(2 ** 53),
                                  "h": 10 ** 20, "i": 1.5e-07, "j": 123456789.123, "k": 1e22, "l": 5e-324,
                                  "m": 1.7976931348623157e308, "n": 0.0001, "o": 1e15 + 0.3, "p": 0, "q": -7,
                                  "r": 1e-7, "s": 9007199254740993.0, "t": 0.30000000000000004, "u": 100.0,
                                  "v": 1e+21, "w": 12345678901234567890.0, "x": 2 ** 63, "y": True, "z": None}),
        (WALL0 + 2.0, "nested", {"list": [1, 1.0, -0.0, [2.5, [3]], {"k": 4.0}], "empty": {}, "none": [], "neg": -1e-300}),
    ],
    "unicode_and_escapes": [
        (WALL0 + 3.0, "text", {"emoji": "smile \U0001F600 ok", "accents": "café naïve", "cjk": "情绪",
                               "controls": "tab\there\nnew\x00nul\x1fus\x7fdel", "quotes": "say \"hi\" \\ back/slash",
                               "bmp_high": "￿﻿", "line_sep": "  "}),
        (WALL0 + 4.0, "keys", {"z": 1, "Z": 2, "é": 3, "\U0001F600": 4, "￿": 5, "a b": 6, "": 7, "10": 8, "9": 9,
                               "~/": 10}),
    ],
    "ts_rounding": [
        (WALL0 + 0.0005, "tick", {"i": 0}),
        (WALL0 + 0.0015, "tick", {"i": 1}),
        (WALL0 + 0.1235, "tick", {"i": 2}),
        (WALL0 + 0.9995, "tick", {"i": 3}),
        (1792850404.9999998, "tick", {"i": 4}),
        (WALL0 + 7.123456789, "tick", {"i": 5}),
    ],
}

AUDIT_REJECTS: list[tuple[str, str, Any]] = [
    ("long_string", "text", {"t": "x" * 201}),
    ("long_list", "list", {"l": list(range(65))}),
    ("too_deep", "deep", {"a": {"b": {"c": {"d": {"e": 1}}}}}),
    ("bad_type", "bytes_like", {"v": (1, object)}),
    ("long_kind", "k" * 201, {}),
]


def run_audit(appends: list[tuple[float, str, dict[str, Any]]], directory: str) -> tuple[AuditLog, list[dict]]:
    log = AuditLog(directory, "golden")
    clock = _Clock(0.0)
    with patched_clock(audit_mod, clock):
        for wall, kind, payload in appends:
            clock.wall = wall
            log.append(kind, payload)
    entries = annotate_entries(log.entries(), [w for w, _, _ in appends])
    return log, entries


def _verify_rows(rows: list[dict[str, Any]], directory: str) -> list[Any]:
    log = AuditLog(directory, "verify")
    log.path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    ok, n = log.verify()
    log.close(delete=True)
    return [ok, n]


def export_audit() -> dict[str, Any]:
    cases, tampered, rejects = [], [], []
    for name, appends in AUDIT_CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            log, entries = run_audit(appends, tmp)
            cases.append({"name": name, "entries": entries, "verify": list(log.verify())})
    base = [{k: v for k, v in e.items() if k not in ("clock", "blob", "floats")} for e in cases[0]["entries"]]

    def tamper(name: str, fn) -> None:
        rows = json.loads(json.dumps(base))
        rows = fn(rows) or rows
        with tempfile.TemporaryDirectory() as tmp:
            tampered.append({"name": name, "entries": rows, "verify": _verify_rows(rows, tmp)})

    tamper("payload_edited", lambda rows: rows[2]["payload"].update(confidence=0.99))
    tamper("entry_deleted", lambda rows: rows.pop(1) and None)
    tamper("ts_edited", lambda rows: rows[3].update(ts=rows[3]["ts"] + 1))
    tamper("reordered", lambda rows: [rows[1], rows[0], *rows[2:]])
    tamper("untouched", lambda rows: None)
    for name, kind, payload in AUDIT_REJECTS:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                AuditLog(tmp, "x").append(kind, payload)
            except AuditPayloadError as e:
                rejects.append({"name": name, "kind": kind, "payload": _reject_payload(payload), "error": str(e)})
            else:
                raise AssertionError(f"audit accepted {name}")
    return {"version": VERSION, "genesis": GENESIS, "cases": cases, "tampered": tampered, "rejects": rejects}


def _reject_payload(payload: Any) -> Any:
    """Rejected payloads as JSON (a non-JSON value is written as its type name)."""
    if isinstance(payload, dict):
        return {k: _reject_payload(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        if any(isinstance(v, type) for v in payload):
            return f"<{type(payload).__name__} containing a {type(payload[-1]).__name__}>"
        return [_reject_payload(v) for v in payload]
    return payload


# ---------- runtime ----------

RUNTIME_RUNS: tuple[tuple[str, str, str, bool], ...] = (  # (run name, scenario, state_set, room_only)
    ("frustrated_masked", "frustrated_masked", "core", False),  # mask override
    ("full_demo", "full_demo", "core", False),  # break: the person leaves for 6 minutes
    ("context_blindness", "context_blindness", "core", False),  # two people, empty room ticks
    ("collaborative_happy_room_only", "collaborative_happy", "core", True),
    ("overwhelmed", "overwhelmed", "all", False),  # 6-state model
)


class _CaptureBus(EventBus):
    """EventBus that also keeps model_dump() of every published message (the wire carries the same values)."""

    def __init__(self) -> None:
        super().__init__()
        self.dumps: list[dict[str, Any]] = []

    def publish(self, msg: Any) -> None:
        d = msg.model_dump() if isinstance(msg, BaseModel) else dict(msg)
        wire = json.loads(msg.model_dump_json()) if isinstance(msg, BaseModel) else d
        assert jsonable(d) == wire, "model_dump() and the wire JSON disagree"
        self.dumps.append(d)
        super().publish(msg)


class _RecordingSource(ScenarioSource):
    """ScenarioSource that remembers the records of the last tick (the fixture stores them)."""

    last: list[TickRecord] | None = None

    def next_tick(self, wall: float, t_mono: float) -> list[TickRecord] | None:
        self.last = super().next_tick(wall, t_mono)
        return self.last


def hello_for(rt: Runtime, served: ServedPredictor) -> Hello:
    """valence/api/app.py hello() for this runtime."""
    agg = rt.aggregator
    return Hello(
        mode=rt.mode if rt.mode in ("live", "replay", "hybrid") else "replay", room_only=rt.room_only,
        states=[StateInfo(id=s, display=STATE_DISPLAY[s], tier=STATE_TIER[s]) for s in served.states],
        feature_version=served.model_id, frame_w=agg.frame_w, frame_h=agg.frame_h, mirrored=agg.mirrored,
    )


def runtime_run(name: str, scenario: str, state_set: str, room_only: bool, max_ticks: int | None = None) -> dict[str, Any]:
    served = served_for(state_set)
    settings = Settings(state_set=state_set)
    bus = _CaptureBus()
    clock = _Clock(WALL0)
    with tempfile.TemporaryDirectory() as tmp, patched_clock(audit_mod, clock):
        rt = Runtime(settings, bus, served, gate=PurgeGate(), narrator=None, audit_dir=tmp)
        rt.session_id = rt.aggregator.session_id = "session-golden"
        src = _RecordingSource(_rendered(scenario, state_set), scenario)
        src.session_id = f"replay-{scenario}-golden"
        rt.source = src
        rt.mode = "replay"
        rt.room_only = room_only
        n = n_scenario_ticks(scenario, state_set) if max_ticks is None else max_ticks
        ticks, audit_clocks = [], []
        for i in range(n):
            wall, mono = WALL0 + i * settings.tick_s, MONO0 + i * settings.tick_s
            clock.wall = wall
            bus.dumps.clear()
            before = rt.audit._seq
            rt.step(wall=wall, mono=mono)
            audit_clocks += [wall] * (rt.audit._seq - before)
            assert src.last is not None
            ticks.append({"wall": wall, "mono": mono, "records": record_dicts(src.last), "messages": list(bus.dumps)})
        ok, n_entries = rt.audit.verify()
        assert ok and n_entries == len(audit_clocks)
        audit = annotate_entries(rt.audit.entries(), audit_clocks)
        hello = hello_for(rt, served)
        rt.close()
    return {"name": name, "scenario": scenario, "state_set": state_set, "room_only": room_only,
            "model_id": served.model_id, "settings": dataclasses.asdict(settings), "wall0": WALL0, "mono0": MONO0,
            "session_id": f"replay-{scenario}-golden", "hello": hello, "ticks": ticks, "audit": audit}


def export_runtime(runs=RUNTIME_RUNS, max_ticks: int | None = None) -> dict[str, Any]:
    return {"version": VERSION, "runs": [runtime_run(*r, max_ticks=max_ticks) for r in runs]}


# ---------- sanity checks (reload each fixture and re-derive values from the Python code) ----------


def _same(a: Any, b: Any) -> bool:
    """Bit-exact equality of encoded values (NaN == NaN via the null encoding)."""
    return json.dumps(jsonable(a)) == json.dumps(jsonable(b))


def check_features(fx: dict[str, Any]) -> str:
    checked = 0
    for sc in fx["scenarios"][:: max(1, len(fx["scenarios"]) // 3)]:
        settings = Settings(**sc["settings"])
        window = RollingWindow(settings.window_ticks)
        want = {(c["tick"], c["person_id"]): c["features"] for c in sc["checkpoints"]}
        picks = sorted(want)[:: max(1, len(want) // 3)][:3]
        for i, tick in enumerate(sc["ticks"]):
            records = [record_from_dict(d) for d in tick["records"]]
            for r in records:
                window.append(r)
            evict(window, settings, tick["wall"])
            for t, pid in picks:
                if t == i:
                    got = features_from_window(*window_inputs(window, pid))
                    assert _same(got, want[(t, pid)]), (sc["name"], t, pid)
                    checked += 1
    for case in fx["edge_cases"]:
        person = {k: np.array(nn(v), dtype=float) for k, v in case["person"].items()}
        room = {k: np.array(nn(v), dtype=float) for k, v in case["room"].items()}
        assert _same(compute(person, room), case["features"]), case["name"]
    return f"features: {checked} checkpoints re-derived from fixture records, {len(fx['edge_cases'])} edge cases"


def check_aggregator(fx: dict[str, Any]) -> str:
    for case in fx["cases"]:
        obs = [obs_from_dict(d) for d in case["obs"]]
        rt = replay_aggregator(obs, Settings(**case["settings"]), case["session_id"], case["source"])
        assert _same(rt.ticks, case["ticks"]) and _same(rt.checkpoints, case["checkpoints"]), case["name"]
    return f"aggregator: re-replayed all {len(fx['cases'])} cases from the fixture's obs dicts"


def check_models(fx: dict[str, Any]) -> str:
    n = 0
    for s in fx["sets"]:
        served = served_for(s["state_set"])
        assert s["model_id"] == served.model_id
        for case in s["cases"]:
            contrib = np.array(case["m2_contrib"], dtype=float)
            raw = np.array(case["m2_raw"], dtype=float)
            np.testing.assert_allclose(contrib.sum(axis=1), raw, rtol=0, atol=1e-9)
            p = _softmax(raw, served.t2)
            if len(case["ood"]) > 3:
                p = 0.7 * p + 0.3 / len(p)
            assert _same(p, case["p"]) and int(np.argmax(p)) == case["k"]
            n += 1
        for case in s["cases"][::7]:
            again = model_case(served, case["name"], np.array(nn(case["x"]), dtype=float), case["tod_h"])
            assert _same(again, case), case["name"]
    return f"models: {n} cases additive (sum(contrib) == raw), softmax re-derived, every 7th re-predicted"


def check_explain(fx: dict[str, Any]) -> str:
    for e in fx["describe"][::97]:
        v = math.nan if e["value"] is None else e["value"]
        a = None if e["absolute"] is None else e["absolute"]
        assert _same(describe_out(e["feature"], v, e["tod_h"], a), e["out"]), e
    for c in fx["layers"][::5]:
        values = {k: (math.nan if v is None else v) for k, v in c["x"].items()}
        absolute = None if c["absolute"] is None else {k: (math.nan if v is None else v) for k, v in c["absolute"].items()}
        got = build_layers(c["contribs"], values, tod_h=c["tod_h"], drivers_per_layer=c["drivers_per_layer"], absolute=absolute)
        assert _same(got, c["out"]), c["name"]
    for t in fx["template"]:
        assert template_text(FactSheet(**t["sheet"])) == t["out"]
    for f in fx["fact_sheets"][::4]:
        args = dict(f["args"], layers=[Layer(**layer) for layer in f["args"]["layers"]])
        assert _same(build_fact_sheet(**args), f["out"])
    return f"explain: {len(fx['describe'])} describe rows (every 97th re-derived), templates re-rendered"


def _check_chain(entries: list[dict[str, Any]]) -> None:
    prev = GENESIS
    for e in entries:
        assert e["prev"] == prev
        assert hashlib.sha256((prev + e["blob"]).encode()).hexdigest() == e["hash"]
        prev = e["hash"]


def check_audit(fx: dict[str, Any]) -> str:
    for case in fx["cases"]:
        _check_chain(case["entries"])
        assert case["verify"] == [True, len(case["entries"])]
    assert [t["verify"][0] for t in fx["tampered"]] == [False, False, False, False, True]
    return f"audit: {len(fx['cases'])} chains re-hashed, {len(fx['tampered'])} tamper cases"


def check_runtime(fx: dict[str, Any]) -> str:
    for run in fx["runs"]:
        _check_chain(run["audit"])
        heads = [GENESIS] + [e["hash"] for e in run["audit"]]
        seen = 0
        for i, tick in enumerate(run["ticks"]):
            gov = tick["messages"][0]
            assert gov["type"] == "governance" and gov["tick"] == i
            while seen < len(run["audit"]) and run["audit"][seen]["ts"] < tick["wall"]:
                seen += 1
            assert gov["audit_hash"] == heads[seen][:16], (run["name"], i)
            preds = [m for m in tick["messages"] if m["type"] == "prediction"]
            for m in preds:
                assert abs(sum(m["probabilities"].values()) - 1) < 1e-3
        assert run["hello"]["feature_version"] == run["model_id"]
    return f"runtime: {len(fx['runs'])} runs, governance audit_hash follows the chain on every tick"


CHECKS = {"features": check_features, "aggregator": check_aggregator, "models": check_models,
          "explain": check_explain, "audit": check_audit, "runtime": check_runtime}
EXPORTS = {"features": export_features, "aggregator": export_aggregator, "models": export_models,
           "explain": export_explain, "audit": export_audit, "runtime": export_runtime}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--only", default=",".join(FIXTURES), help="comma-separated subset of " + ",".join(FIXTURES))
    ap.add_argument("--check-only", action="store_true", help="only re-derive values from existing fixtures")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = set(names) - set(FIXTURES)
    if unknown:
        raise SystemExit(f"unknown fixture(s): {sorted(unknown)}")
    with utc_tz():
        if not args.check_only:
            for name in names:
                t0 = time.perf_counter()
                size = write_fixture(args.out / f"{name}.json.gz", EXPORTS[name]())
                print(f"{name}.json.gz  {size / 1024:8.1f} KiB  ({time.perf_counter() - t0:.1f} s)")
        for name in names:
            print("check", CHECKS[name](load_fixture(args.out / f"{name}.json.gz")))
    total = sum((args.out / f"{n}.json.gz").stat().st_size for n in FIXTURES if (args.out / f"{n}.json.gz").exists())
    print(f"total {total / 1024 / 1024:.2f} MiB gzipped (budget {SIZE_BUDGET / 1024 / 1024:.0f} MiB)")
    if total > SIZE_BUDGET:
        raise SystemExit("fixtures exceed the size budget")


if __name__ == "__main__":
    main()
