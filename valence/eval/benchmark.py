"""Real-data benchmark: replay recorded sessions through the Python runtime and score each model per protocol block.

    python -m valence.eval.benchmark sessions/*.jsonl [--grace 60] [--out reports/benchmark]
    python -m valence.eval.benchmark --legacy data/real/manifest.json [--out reports/benchmark]

Input (docs/PROTOCOL.md section 3): a ``valence-session/1`` file from the web console (header, ``video``/``audio``
observations, ``tick`` lines with what the browser showed, ``marker`` lines for blocks and operator controls,
``self_report`` taps; other line types are counted and skipped), or, with ``--legacy``, a manifest that gives each
observation-only capture (``data/real/capture_*.jsonl``) one block.

Replay. A fresh ``valence.runtime.engine.Runtime`` (the served M2, smoother and mask override: what the card shows,
on its LiveSource) runs offline: the ``time`` modules of the engine, aggregator and audit log are patched with a clock
set from the session (the pattern of scripts/golden_runtime), and the engine's ``local_tod`` is patched with the
header's time zone (``zoneinfo``: the same wall clock ``datetime.fromtimestamp`` reads under ``TZ=<zone>``) or, when
no zone is known, a fixed hour (12:00 by default, as ``valence.eval.emotion_capture``). Observations are pushed and
``step(wall, mono)`` is called at each recorded tick's (wall, mono); with ``--order file`` (default) the lines run in
file order, which is the order the browser engine took them (the recorder writes each line as the engine takes it),
and with ``--order mono`` they are interleaved by the page clock (an observation with ``t <= mono`` goes before that
step, markers by their ``mono``). A frame that reaches the engine just after a tick it precedes is one the two orders
place differently (``late_observations``): file order reproduces the browser's numbers exactly, mono order moves those
frames into the earlier tick, so confidences can differ slightly (and labels, near a decision boundary).
Control markers are mirrored as browser-source.ts does: ``stop`` forgets everyone (room too), ``start`` after
``pause`` forgets the people (the room stays), ``demo_reset`` forgets the people, ``break`` restarts the break clocks,
``room_only_on|off``, ``sim_<overlay>_on|off`` and ``baseline_<seconds>|default|reset`` set the same engine state.
A file without tick lines (legacy) is stepped every 5 s from its first observation. Read per tick and person: the card
label (``prediction.label``), the raw M2 argmax before smoothing (``PredictionOutput.label``, kept by a thin wrapper
around the served predictor; the engine is unchanged), the face-only M0 label, and the emotion map (emotion-v1 label,
valence and arousal from ``affect_for_window`` on the runtime's window, as the browser computes it). The map is
skipped while the person's window holds no emotion-v1 face cue (recordings made before the cues existed, a covered
face) or when ``artifacts/emotion`` is missing.

Scoring (person 1 unless ``--person``). Blocks come from ``block_start``/``block_end`` markers; ``protocol_abort``
(or ``protocol_end``, the next ``block_start``, the end of the file) ends an open block there. A tick belongs to a
block when ``start < mono <= end``; the first ``--grace`` seconds of each block are not scored. Per block and model:
hit rate (scored ticks whose label is in the block's ``expect`` list, ``card``/``m2``/``m0`` against
``expect.card`` and the map against ``expect.map``; a tick where the person was predicted but that model had no
reading, such as M0 without a face, counts as a miss; ticks where the person was not predicted at all are counted
as ``absent`` and left out), time to first hit (seconds from the block start, over all its ticks, grace included;
null if never), the label distribution and majority label, and tick counts. Per session: the calm false-alarm rate
(scored calm blocks: share of read ticks outside {Focused}, {Focused, Calm} for the map), the mean hit rate over the
scored non-calm blocks, self-report agreement (map valence/arousal MAE against each tap, at the latest tick at or
before it, next to a constant (0, 0) guess) and parity (ticks where the replayed card label for the person equals the
browser's recorded one; also M0 and map labels).

Outputs: ``<out>/<session_id>.json`` (everything, per tick included), ``<out>/<session_id>.md`` (one table per model)
and ``<out>/summary.md`` across the sessions given. Nothing here reads or writes media.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time as _time
import types
from bisect import bisect_right
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import valence.governance.audit as audit_mod
import valence.runtime.aggregator as agg_mod
import valence.runtime.engine as eng_mod
from valence.config import Settings
from valence.contracts import Obs, obs_from_dict
from valence.models.served import ServedPredictor, load_served
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "valence-session/1"
MANIFEST_FORMAT = "valence-legacy-manifest/1"
TICK_S = 5.0
MODELS = ("card", "m2", "m0", "map")
MODEL_TITLES = {"card": "Card (served M2: smoothed label with mask override)", "m2": "Raw M2 argmax (before smoothing)",
                "m0": "Face-only M0", "map": "Emotion map (emotion-v1)"}
MODEL_SHORT = {"card": "Card", "m2": "Raw M2", "m0": "M0", "map": "Map"}
CALM_LABELS = {"card": {"Focused"}, "m2": {"Focused"}, "m0": {"Focused"}, "map": {"Focused", "Calm"}}
CUE_FIELDS = agg_mod.CUE_FIELDS  # the per-tick emotion-v1 face cues; all NaN in recordings made before them
ORDER_PRIORITY = {"obs": 0, "marker": 1, "tick": 2}  # at equal times: observations, then markers, then the step
GRACE_EPS = 1e-6  # synthesized ticks fall exactly on the grace boundary: float error must not drop that tick


# --- session files -----------------------------------------------------------------------------------------------

@dataclass
class Session:
    path: Path
    session_id: str
    header: dict[str, Any] | None
    events: list[tuple[str, Any]]  # file order: ("obs", Obs) | ("tick", dict) | ("marker", dict) | ("tap", dict)
    unknown: dict[str, int] = field(default_factory=dict)  # skipped line types -> count
    legacy: bool = False  # no tick lines: the ticks are synthesized every 5 s
    late_observations: int = 0  # observations written after a tick whose mono is >= their t
    note: str | None = None  # the manifest's note for a legacy capture

    def of(self, kind: str) -> list[Any]:
        return [item for k, item in self.events if k == kind]

    @property
    def ticks(self) -> list[dict[str, Any]]:
        return self.of("tick")

    @property
    def markers(self) -> list[dict[str, Any]]:
        return self.of("marker")

    @property
    def taps(self) -> list[dict[str, Any]]:
        return self.of("tap")

    @property
    def observations(self) -> list[Obs]:
        return self.of("obs")


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_session(path: str | Path, tick_s: float = TICK_S) -> Session:
    """Read a session (or an observation-only recording); a file without tick lines gets ticks every ``tick_s``."""
    path = Path(path)
    header: dict[str, Any] | None = None
    events: list[tuple[str, Any]] = []
    unknown: Counter[str] = Counter()
    late, tick_mono = 0, -math.inf
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: not JSON ({e})") from None
            if not isinstance(d, dict):
                raise ValueError(f"{path}:{ln}: a line must be a JSON object")
            kind = d.get("type", d.get("kind"))  # obs_from_dict also reads the older "kind" key
            if kind in ("video", "audio"):
                try:
                    obs = obs_from_dict(d)
                except (TypeError, ValueError) as e:  # a field contracts.py does not know, or out of range
                    raise ValueError(f"{path}:{ln}: not a valid {kind} observation ({e})") from None
                late += int(obs.t <= tick_mono)
                events.append(("obs", obs))
            elif kind == "session":
                if header is None:
                    header = d
                else:
                    unknown["session (repeated header)"] += 1
            elif kind == "tick":
                if not (_number(d.get("wall")) and _number(d.get("mono"))):
                    raise ValueError(f"{path}:{ln}: a tick needs numeric wall and mono")
                tick_mono = max(tick_mono, float(d["mono"]))
                events.append(("tick", d))
            elif kind == "marker":
                events.append(("marker", d))
            elif kind == "self_report":
                events.append(("tap", d))
            else:
                unknown[str(kind)] += 1
    session_id = str((header or {}).get("session_id") or path.stem)
    s = Session(path=path, session_id=session_id, header=header, events=events, unknown=dict(unknown),
                late_observations=late)
    if not s.ticks:
        s.legacy = True
        s.events += [("tick", t) for t in synthetic_ticks(s.observations, tick_s)]
    return s


def synthetic_ticks(observations: list[Obs], tick_s: float = TICK_S) -> list[dict[str, Any]]:
    """Ticks every ``tick_s`` from the first observation, on the recording's own clock (no partial last tick)."""
    if not observations:
        return []
    t0 = observations[0].t
    offset = observations[0].wall - t0  # wall time of the recording's monotonic clock
    last = max(o.t for o in observations)
    ticks, k = [], 1
    while t0 + k * tick_s <= last:
        mono = t0 + k * tick_s
        ticks.append({"type": "tick", "wall": mono + offset, "mono": mono, "predictions": None, "synthetic": True})
        k += 1
    return ticks


def legacy_sessions(manifest_path: str | Path, tick_s: float = TICK_S) -> tuple[list[Session], dict[str, Any]]:
    """Each capture in a legacy manifest as a Session with one block spanning the whole file."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{manifest_path}: format must be {MANIFEST_FORMAT!r}")
    sessions = []
    for entry in manifest["captures"]:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = manifest_path.parent / path  # entries are relative to the manifest
        s = load_session(path, tick_s)
        s.session_id = str(entry.get("session_id") or f"legacy-{entry['id']}")
        s.note = entry.get("note")
        if s.observations and s.ticks:
            o0, last = s.observations[0], s.ticks[-1]
            block = {"block_id": entry["id"], "label": entry.get("label", entry["id"]),
                     "scored": bool(entry.get("scored", True)), "calm": bool(entry.get("calm", False)),
                     "expect": {"card": list(entry.get("expect", {}).get("card", [])),
                                "map": list(entry.get("expect", {}).get("map", []))}}
            s.events.insert(0, ("marker", {"type": "marker", "kind": "block_start", "wall": o0.wall, "mono": o0.t,
                                           **block}))
            s.events.append(("marker", {"type": "marker", "kind": "block_end", "wall": last["wall"],
                                        "mono": last["mono"], **block}))
        sessions.append(s)
    return sessions, manifest


# --- models --------------------------------------------------------------------------------------------------------

@dataclass
class Models:
    served: ServedPredictor
    emotion: Any | None  # valence.models.emotion.EmotionPredictor
    emotion_missing: str | None = None  # why the map is not available


def load_models(artifacts: str | Path = ROOT / "artifacts", emotion: str | Path | None = ROOT / "artifacts" / "emotion",
                state_set: str = "core") -> Models:
    served = load_served(Path(artifacts), state_set)
    if served is None:
        raise SystemExit(f"no served {state_set} models in {artifacts} (m2.txt, m0.txt, model_meta.json)")
    if emotion is None:
        return Models(served, None, "disabled")
    from valence.models.emotion import EmotionPredictor

    try:
        return Models(served, EmotionPredictor(Path(emotion)))
    except (OSError, KeyError, ValueError) as e:  # missing directory or files, or a broken meta.json
        return Models(served, None, f"{emotion}: {type(e).__name__}: {e}"[:200])


class _RawM2:
    """The served predictor, also keeping each person's last raw output (M2 argmax before the smoother)."""

    def __init__(self, served: ServedPredictor) -> None:
        self.served = served
        self.states = served.states
        self.model_id = served.model_id
        self.last: dict[int, Any] = {}

    def predict(self, person, room):
        out = self.served.predict(person, room)
        if out is not None and person:
            self.last[person[-1].person_id] = out
        return out


# --- offline runtime ---------------------------------------------------------------------------------------------

class _Clock:
    """Stands in for the ``time`` module of the engine, aggregator and audit log: time() and monotonic() as set."""

    def __init__(self, wall: float = 0.0, mono: float = 0.0) -> None:
        self.wall = wall
        self.mono = mono

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def __getattr__(self, name: str) -> Any:  # perf_counter, sleep, ...
        return getattr(_time, name)


class _Gate:
    """PurgeGate stand-in: a replay decodes no frames, so there is nothing to purge (governance counters stay 0)."""

    def _stats(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(purged={"video": 0, "audio": 0}, transcripts_purged=0, violations=0,
                                     live_raw_buffers=0)

    def end_tick(self) -> types.SimpleNamespace:
        return self._stats()

    def stats(self) -> types.SimpleNamespace:
        return self._stats()


class _Bus(EventBus):
    """Drops every message (the readings come from step()'s return value and the runtime's window)."""

    def publish(self, msg: Any) -> None:
        self.published += 1


def time_of_day(tz: str | None, fixed: float = 12.0) -> Callable[[float], float]:
    """The engine's ``local_tod`` for a replay: hours on the wall clock in ``tz``, or ``fixed`` without a zone."""
    if tz is None:
        return lambda wall: fixed
    zone = ZoneInfo(tz)

    def tod(wall: float) -> float:
        dt = datetime.fromtimestamp(wall, zone)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    return tod


@contextmanager
def offline_runtime(clock: _Clock, tod: Callable[[float], float]) -> Iterator[None]:
    """Patch the runtime modules' clocks and time of day for an offline replay; restored afterwards."""
    saved = (eng_mod.time, agg_mod.time, audit_mod.time, eng_mod.local_tod)
    eng_mod.time = agg_mod.time = audit_mod.time = clock
    eng_mod.local_tod = tod
    try:
        yield
    finally:
        eng_mod.time, agg_mod.time, audit_mod.time, eng_mod.local_tod = saved


def ordered_events(session: Session, order: str = "file") -> list[tuple[str, Any]]:
    """The events the runtime takes (taps excluded), in file order or interleaved by the page clock."""
    events = [(k, item) for k, item in session.events if k != "tap"]
    if order == "file" and not session.legacy:
        return events
    if order not in ("file", "mono"):
        raise ValueError(f"order must be 'mono' or 'file', not {order!r}")
    keyed, last = [], -math.inf
    for seq, (kind, item) in enumerate(events):
        key = item.t if kind == "obs" else item.get("mono")
        key = float(key) if _number(key) else last  # a marker without mono stays where the file put it
        last = key
        keyed.append((key, ORDER_PRIORITY[kind], seq, kind, item))
    keyed.sort(key=lambda e: e[:3])
    return [(kind, item) for *_, kind, item in keyed]


def _event_clock(kind: str, item: Any) -> tuple[float, float] | None:
    if kind == "obs":
        return item.wall, item.t
    if _number(item.get("wall")) and _number(item.get("mono")):
        return float(item["wall"]), float(item["mono"])
    return None


class _Controls:
    """The browser's operator controls (frontend/src/lib/browser-source.ts ``command``) applied to the replay."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.state = "running"  # a recording starts with the engine, running
        self.applied: list[dict[str, Any]] = []
        self.ignored: Counter[str] = Counter()

    def apply(self, marker: dict[str, Any]) -> None:
        rt, action = self.rt, str(marker.get("action"))
        was, effect = self.state, None
        if action == "start":
            self.state = "running"
            if was == "paused":  # a new perception numbers people from 1 again: forget them, keep the room
                rt.reset_people(keep_room=True)
                effect = "reset_people(keep_room=True)"
        elif action == "pause":
            self.state = "paused"
        elif action == "stop":
            self.state = "stopped"
            rt.reset_people(keep_room=False)
            effect = "reset_people(keep_room=False)"
        elif action == "break":
            rt.aggregator.log_break(None, wall=marker.get("wall"))
            effect = "log_break(everyone)"
        elif action == "demo_reset":
            rt.reset_people(keep_room=True)
            effect = "reset_people(keep_room=True)"
        elif action in ("room_only_on", "room_only_off"):
            room_only = action == "room_only_on"
            if room_only != rt.room_only:
                rt.latest.clear()
            rt.room_only = room_only
            effect = f"room_only={room_only}"
        elif (m := re.fullmatch(r"sim_(face_occlusion|low_light)_(on|off)", action)):
            rt.set_overlay(m[1], m[2] == "on")
            effect = f"overlay {m[1]}={m[2]}"
        elif action == "baseline_reset":
            rt.aggregator.start_baseline(mode="auto")
            effect = "room baseline reset (auto)"
        elif (m := re.fullmatch(r"baseline_(default|\d+(?:\.\d+)?)", action)):
            seconds = None if m[1] == "default" else float(m[1])
            rt.aggregator.start_baseline(seconds)
            effect = f"room baseline capture ({m[1]})"
        else:
            self.ignored[action] += 1
            return
        self.applied.append({"mono": marker.get("mono"), "action": action, "effect": effect})


@dataclass
class TickReading:
    i: int
    wall: float
    mono: float
    people: dict[int, dict[str, Any]]  # replayed readings by person id
    recorded: dict[int, dict[str, Any]] | None  # the browser's tick line by person id (None: synthesized tick)
    tod_h: float = math.nan  # the time of day the tick was stamped with (room record)


@dataclass
class Replay:
    ticks: list[TickReading]
    tz: str | None  # the zone used for the time of day (None: fixed hour)
    tod_fixed: float | None
    order: str
    served_model: str
    emotion_model: str | None
    emotion_missing: str | None  # why the map was not read at all (None: it was loaded)
    controls: list[dict[str, Any]]
    warnings: list[str]


def _r(v: Any, nd: int = 4) -> Any:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), nd)


def _has_cues(window: list) -> bool:
    return any(not math.isnan(getattr(r, f)) for r in window for f in CUE_FIELDS)


def _recorded(predictions: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]] | None:
    if predictions is None:
        return None
    out = {}
    for p in predictions:
        if isinstance(p, dict) and isinstance(p.get("person_id"), int):
            out[p["person_id"]] = p
    return out


def replay(session: Session, models: Models, *, tz: str | None = None, tod: float = 12.0, order: str = "file",
           settings: Settings | None = None) -> Replay:
    """Drive a fresh Runtime through the session offline; one TickReading per tick."""
    from valence.models.emotion import affect_for_window

    warnings: list[str] = []
    tod_fn = None
    if tz is not None:
        try:
            tod_fn = time_of_day(tz)
        except (ZoneInfoNotFoundError, ValueError):
            warnings.append(f"unknown time zone {tz!r}: time of day fixed at {_hhmm(tod)}")
            tz = None
    if tod_fn is None:
        tod_fn = time_of_day(None, tod)
    settings = settings or Settings(state_set=models.served.meta.get("state_set", "core"))
    predictor = _RawM2(models.served)
    events = ordered_events(session, order)
    first = next((c for k, item in events if (c := _event_clock(k, item))), (0.0, 0.0))
    clock = _Clock(*first)
    readings: list[TickReading] = []
    with tempfile.TemporaryDirectory() as tmp, offline_runtime(clock, tod_fn):
        rt = Runtime(settings, _Bus(), predictor, gate=_Gate(), narrator=None, audit_dir=tmp)
        controls = _Controls(rt)
        try:
            rt.use_live()  # the browser engine runs on its live source (createBrowserEngine: useLive)
            for kind, item in events:
                c = _event_clock(kind, item)
                if c is not None:
                    clock.wall, clock.mono = c
                if kind == "obs":
                    rt.push_obs(item)
                elif kind == "marker":
                    if item.get("kind") == "control":
                        controls.apply(item)
                elif kind == "tick":
                    predictor.last.clear()
                    wall, mono = float(item["wall"]), float(item["mono"])
                    people = {}
                    for p in rt.step(wall=wall, mono=mono):
                        pid = p["person_id"]
                        out = predictor.last.get(pid)
                        row = {"card": p["label"], "card_display": p["label_display"],
                               "confidence": _r(p["confidence"]), "warming_up": p["warming_up"],
                               "mask_override": p["mask_override"],
                               "m2": out.label if out else None, "m2_confidence": _r(out.confidence) if out else None,
                               "m0": p["face_only"]["label"], "m0_confidence": _r(p["face_only"]["confidence"]),
                               "map": None, "map_confidence": None, "valence": None, "arousal": None, "cues": False}
                        if models.emotion is not None:
                            person = rt.window.snapshot(pid)
                            row["cues"] = _has_cues(person)
                            if row["cues"]:
                                a = affect_for_window(models.emotion, person, rt.window.room())
                                if a is not None:
                                    row.update(map=a.label, map_confidence=_r(a.confidence), valence=a.valence,
                                               arousal=a.arousal)  # unrounded: parity compares them exactly
                        people[pid] = row
                    room = rt.window.room()
                    readings.append(TickReading(len(readings), wall, mono, people, _recorded(item.get("predictions")),
                                                room[-1].tod_h if room else math.nan))
        finally:
            rt.close()
    if any(c["action"] == "sim_low_light_on" for c in controls.applied):
        warnings.append("the low_light overlay was on: it draws random noise, so parity cannot be exact there")
    for action, n in controls.ignored.items():
        warnings.append(f"control {action!r} ({n}x) is not mirrored by the replay")
    return Replay(readings, tz, None if tz else tod, "mono" if session.legacy else order, models.served.model_id,
                  models.emotion.model_id if models.emotion is not None else None, models.emotion_missing,
                  controls.applied, warnings)


# --- blocks and scores -------------------------------------------------------------------------------------------

@dataclass
class Block:
    id: str
    label: str
    scored: bool
    calm: bool
    expect: dict[str, list[str]]
    start: float  # mono
    start_wall: float | None
    end: float = math.inf
    ended_by: str = "open"


def find_blocks(markers: list[dict[str, Any]], session_end: float) -> tuple[list[Block], list[str]]:
    """Blocks from the markers (file order): each block_start until its block_end, an abort, or the next start."""
    blocks: list[Block] = []
    problems: list[str] = []
    cur: Block | None = None

    def close(at: float, why: str) -> None:
        nonlocal cur
        if cur is not None:
            cur.end, cur.ended_by = at, why
            blocks.append(cur)
            cur = None

    for m in markers:
        kind, mono = m.get("kind"), m.get("mono")
        if kind not in ("block_start", "block_end", "protocol_abort", "protocol_end"):
            continue
        if not _number(mono):
            problems.append(f"{kind} marker without a numeric mono: ignored")
            continue
        if kind == "block_start":
            if cur is not None:
                problems.append(f"block {cur.id!r} has no block_end: it ends at the next block_start")
                close(mono, "next_block_start")
            expect = m.get("expect") or {}
            cur = Block(id=str(m.get("block_id") or f"block{len(blocks) + 1}"), label=str(m.get("label") or ""),
                        scored=bool(m.get("scored", False)), calm=bool(m.get("calm", False)),
                        expect={"card": list(expect.get("card") or []), "map": list(expect.get("map") or [])},
                        start=float(mono), start_wall=m.get("wall") if _number(m.get("wall")) else None)
        elif kind == "block_end":
            if cur is None or m.get("block_id") not in (None, cur.id):
                problems.append(f"block_end for {m.get('block_id')!r} without its block_start: ignored")
                continue
            close(float(mono), "block_end")
        else:
            close(float(mono), kind)
    if cur is not None:
        problems.append(f"block {cur.id!r} has no block_end: it ends with the recording")
        close(session_end, "end_of_recording")
    return blocks, problems


def model_score(model: str, expect: list[str], inside: list[TickReading], scored: list[TickReading], person: int,
                start: float) -> dict[str, Any]:
    """One model's numbers for one block (see the module docstring for the definitions)."""
    rows = [t.people[person] for t in scored if person in t.people]
    labels = [r[model] for r in rows]
    read = [lab for lab in labels if lab is not None]
    counts = Counter("none" if lab is None else lab for lab in labels)
    out: dict[str, Any] = {"ticks": len(rows), "read": len(read), "hits": None, "hit_rate": None, "first_hit_s": None,
                           "labels": dict(counts.most_common()), "majority": None, "majority_share": None}
    if read:
        top, n = Counter(read).most_common(1)[0]
        out["majority"], out["majority_share"] = top, _r(n / len(rows), 3)
    if expect:
        hits = sum(lab in expect for lab in read)
        out["hits"] = hits
        out["hit_rate"] = _r(hits / len(rows), 3) if read else None
        first = next((t.mono - start for t in inside if person in t.people and t.people[person][model] in expect),
                     None)
        out["first_hit_s"] = _r(first, 1)
    return out


def score_block(block: Block, ticks: list[TickReading], person: int, grace: float) -> dict[str, Any]:
    inside = [t for t in ticks if block.start < t.mono <= block.end]
    scored = [t for t in inside if t.mono - block.start >= grace - GRACE_EPS]
    return {
        "id": block.id, "label": block.label, "scored": block.scored, "calm": block.calm, "expect": block.expect,
        "start_mono": _r(block.start, 3), "end_mono": _r(block.end, 3), "start_wall": _r(block.start_wall, 3),
        "duration_s": _r(block.end - block.start, 1), "ended_by": block.ended_by,
        "ticks_in_block": len(inside), "absent_in_block": sum(person not in t.people for t in inside),
        "ticks_scored": len(scored), "absent": sum(person not in t.people for t in scored),
        "models": {m: model_score(m, block.expect["map" if m == "map" else "card"], inside, scored, person,
                                  block.start) for m in MODELS},
    }


def _block_of(blocks: list[Block], mono: float) -> str | None:
    return next((b.id for b in blocks if b.start < mono <= b.end), None)


def calm_false_alarms(block_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Per model, over the scored calm blocks: share of read ticks outside the calm labels."""
    calm = [b for b in block_scores if b["calm"] and b["scored"]]
    out: dict[str, Any] = {"blocks": [b["id"] for b in calm]}
    for m in MODELS:
        counts: Counter[str] = Counter()
        for b in calm:
            counts.update(b["models"][m]["labels"])
        read = sum(n for lab, n in counts.items() if lab != "none")
        alarms = sum(n for lab, n in counts.items() if lab != "none" and lab not in CALM_LABELS[m])
        out[m] = {"ticks": read, "alarms": alarms, "rate": _r(alarms / read, 3) if read else None}
    return out


def mean_hit_rates(block_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Per model, the mean of the block hit rates over the scored non-calm blocks that have one."""
    out = {}
    for m in MODELS:
        rates = [b["models"][m]["hit_rate"] for b in block_scores
                 if b["scored"] and not b["calm"] and b["models"][m]["hit_rate"] is not None]
        out[m] = {"blocks": len(rates), "mean": _r(sum(rates) / len(rates), 3) if rates else None}
    return out


def self_report_agreement(taps: list[dict[str, Any]], ticks: list[TickReading], blocks: list[Block],
                          person: int) -> dict[str, Any] | None:
    """Map valence/arousal against each tap, read at the latest tick at or before the tap (by mono)."""
    if not taps:
        return None
    monos = [t.mono for t in ticks]
    rows, problems = [], []
    for tap in taps:
        v, a, mono = tap.get("valence"), tap.get("arousal"), tap.get("mono")
        if not (_number(v) and _number(a) and -1 <= v <= 1 and -1 <= a <= 1 and _number(mono)):
            problems.append(f"tap at mono {mono!r}: needs mono and valence/arousal in [-1, 1]")
            continue
        pid = tap.get("person_id") if isinstance(tap.get("person_id"), int) else person
        k = bisect_right(monos, mono) - 1
        reading = ticks[k].people.get(pid) if k >= 0 else None
        if reading is None or reading["valence"] is None:
            problems.append(f"tap at mono {mono:.1f} (person {pid}): no map reading at or before it")
            continue
        rows.append({"mono": _r(mono, 3), "person_id": pid, "block": _block_of(blocks, mono),
                     "lag_s": _r(mono - ticks[k].mono, 2), "tap_valence": _r(v, 3), "tap_arousal": _r(a, 3),
                     "tap_label": tap.get("label"), "map_valence": reading["valence"],
                     "map_arousal": reading["arousal"], "map_label": reading["map"]})
    out: dict[str, Any] = {"taps": len(taps), "matched": len(rows), "problems": problems, "rows": rows}
    if rows:
        def mae(key: str, ref: str | None) -> float:
            return _r(sum(abs(r[key] - (r[ref] if ref else 0.0)) for r in rows) / len(rows), 3)

        out.update(mae_valence=mae("tap_valence", "map_valence"), mae_arousal=mae("tap_arousal", "map_arousal"),
                   zero_mae_valence=mae("tap_valence", None), zero_mae_arousal=mae("tap_arousal", None))
    return out


def parity(ticks: list[TickReading], person: int, examples: int = 10) -> dict[str, Any] | None:
    """Replayed vs recorded labels on the session's own tick lines (None for a file without them)."""
    recorded = [t for t in ticks if t.recorded is not None]
    if not recorded:
        return None
    out: dict[str, Any] = {}
    for model in ("card", "m0", "map"):
        compared = matches = 0
        mismatched: list[dict[str, Any]] = []
        worst_va = worst_conf = 0.0
        for t in recorded:
            rec, rep = t.recorded.get(person), t.people.get(person)
            if model == "card":
                if rec is None and rep is None:
                    continue
                a, b = (rec or {}).get("label"), (rep or {}).get("card")
                if rec is not None and rep is not None and _number(rec.get("confidence")):
                    worst_conf = max(worst_conf, abs(rec["confidence"] - rep["confidence"]))
            elif rec is None or rep is None:
                continue
            elif model == "m0":
                a, b = (rec.get("face_only") or {}).get("label"), rep["m0"]
                if _number((rec.get("face_only") or {}).get("confidence")) and rep["m0_confidence"] is not None:
                    worst_conf = max(worst_conf, abs(rec["face_only"]["confidence"] - rep["m0_confidence"]))
            else:
                affect = rec.get("affect")
                if not affect or rep["map"] is None:  # the replay skips the map without face cues
                    continue
                a, b = affect.get("label"), rep["map"]
                for k in ("valence", "arousal"):
                    if _number(affect.get(k)) and rep[k] is not None:
                        worst_va = max(worst_va, abs(affect[k] - rep[k]))
            compared += 1
            if a == b:  # card: a missing side is a mismatch (both missing was skipped); M0: both None agree
                matches += 1
            elif len(mismatched) < examples:
                mismatched.append({"tick": t.i, "mono": _r(t.mono, 3), "browser": a, "replay": b})
        out[model] = {"ticks": compared, "matches": matches, "mismatches": compared - matches,
                      "share": _r(matches / compared, 4) if compared else None, "examples": mismatched}
        if model == "map":
            out[model]["max_abs_valence_arousal_diff"] = _r(worst_va, 6) if compared else None
        else:  # both sides carry the message's 4-decimal confidence
            out[model]["max_abs_confidence_diff"] = _r(worst_conf, 6) if compared else None
    return out


def score(session: Session, rep: Replay, *, grace: float = 60.0, person: int = 1) -> dict[str, Any]:
    """The full report for one replayed session (what <session_id>.json holds)."""
    ticks = rep.ticks
    first = next((c[1] for k, item in session.events if (c := _event_clock(k, item))), None)
    end = max([t.mono for t in ticks] + [first or 0.0])
    blocks, problems = find_blocks(session.markers, end)
    block_scores = [score_block(b, ticks, person, grace) for b in blocks]
    warnings = list(rep.warnings) + problems
    header = session.header or {}
    if session.header is not None and header.get("format") != FORMAT:
        warnings.append(f"header format {header.get('format')!r}, expected {FORMAT!r}")
    if header.get("feature_version") and header["feature_version"] != rep.served_model:
        warnings.append(f"recorded with {header['feature_version']}, replayed with {rep.served_model}: "
                        "parity is not expected")
    if header.get("emotion_model") and rep.emotion_model and header["emotion_model"] != rep.emotion_model:
        warnings.append(f"emotion map recorded with {header['emotion_model']}, replayed with {rep.emotion_model}")
    for kind, n in session.unknown.items():
        warnings.append(f"skipped {n} line(s) of unknown type {kind!r}")
    if not blocks:
        warnings.append("no blocks (no block_start markers): nothing is scored")
    rows = [r for t in ticks for r in t.people.values()]
    map_note = None
    if rep.emotion_model is None:
        map_note = f"emotion model not available ({rep.emotion_missing})"
    elif not rows:
        map_note = "nobody was predicted, so the map was not read"
    elif not any(r["map"] is not None for r in rows):
        map_note = "no emotion-v1 face cues in this recording: the map was not read"
    par = parity(ticks, person)
    if par and session.late_observations and rep.order == "mono" and (
            par["card"]["mismatches"] or par["card"]["max_abs_confidence_diff"]):
        warnings.append(f"{session.late_observations} observation(s) reached the browser engine after a tick they "
                        "precede; mono order pushes them before it, so the replay differs from the browser "
                        f"(card confidence by up to {par['card']['max_abs_confidence_diff']:g}): --order file replays "
                        "the engine's own order")
    readings = []
    for t in ticks:
        people = []
        for pid, row in sorted(t.people.items()):
            entry = {"person_id": pid, **row}
            if t.recorded is not None and pid in t.recorded:
                rec = t.recorded[pid]
                entry["browser"] = {"card": rec.get("label"), "m0": (rec.get("face_only") or {}).get("label"),
                                    "map": (rec.get("affect") or {}).get("label")}
            people.append(entry)
        readings.append({"i": t.i, "t": _r(t.mono - (first or 0.0), 3), "wall": _r(t.wall, 3), "mono": _r(t.mono, 3),
                         "tod_h": _r(t.tod_h), "block": _block_of(blocks, t.mono), "people": people})
    started = header.get("started_wall") if _number(header.get("started_wall")) else (
        session.observations[0].wall if session.observations else None)
    return {
        "session_id": session.session_id,
        "file": str(session.path),
        "kind": "legacy" if session.legacy else "session",
        "note": session.note,
        "header": session.header,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if started else None,
        "replay": {"served_model": rep.served_model, "emotion_model": rep.emotion_model, "order": rep.order,
                   "time_of_day": f"wall clock in {rep.tz}" if rep.tz else f"fixed at {_hhmm(rep.tod_fixed)}",
                   "ticks": "synthesized every 5 s from the first observation" if session.legacy else "recorded",
                   "controls": rep.controls},
        "scoring": {"person": person, "grace_s": grace, "map": map_note},
        "counts": {"observations": len(session.observations), "ticks": len(ticks), "markers": len(session.markers),
                   "taps": len(session.taps), "late_observations": session.late_observations,
                   "duration_s": _r(end - first, 1) if first is not None and ticks else 0.0},
        "warnings": warnings,
        "blocks": block_scores,
        "session": {"mean_hit_rate": mean_hit_rates(block_scores), "calm_false_alarm": calm_false_alarms(block_scores),
                    "self_report": self_report_agreement(session.taps, ticks, blocks, person), "parity": par},
        "ticks": readings,
    }


def run_session(session: Session, models: Models, *, grace: float = 60.0, person: int = 1, tz: str | None = None,
                tod: float = 12.0, order: str = "file") -> dict[str, Any]:
    """Replay and score one session: the time of day in ``tz`` if given, else the header's zone, else ``tod``."""
    zone = tz if tz is not None else ((session.header or {}).get("tz") or None)
    return score(session, replay(session, models, tz=zone, tod=tod, order=order), grace=grace, person=person)


# --- reports -------------------------------------------------------------------------------------------------------

def _hhmm(hours: float) -> str:
    h, m = divmod(round(hours * 60), 60)
    return f"{h % 24:02d}:{m:02d}"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def _diff(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _share(v: float | None) -> str:
    """A parity share, rounded down to 0.1% so that anything short of every tick never reads 100%."""
    return "—" if v is None else f"{math.floor(1000 * v) / 10:g}%"


def _secs(v: float | None, expect: list[str]) -> str:
    if not expect:
        return "—"
    return "never" if v is None else f"{v:.0f} s"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "session"


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def session_markdown(report: dict[str, Any]) -> str:
    rp, sc, c = report["replay"], report["scoring"], report["counts"]
    kind = "legacy observation recording, no tick lines" if report["kind"] == "legacy" else f"`{FORMAT}` session"
    lines = [f"# Benchmark: {report['session_id']}", "",
             f"- File: `{report['file']}` ({kind})"]
    if report.get("note"):
        lines.append(f"- Manifest note: {report['note']}")
    lines += [f"- Started: {report['started_utc'] or '—'} UTC; time of day: {rp['time_of_day']}",
              f"- Replay: {c['observations']} observations, {c['ticks']} ticks ({rp['ticks']}), "
              f"{c['duration_s']:g} s, order `{rp['order']}`; served `{rp['served_model']}`, emotion map "
              f"`{rp['emotion_model'] or 'not loaded'}`",
              f"- Scored: person {sc['person']}, the first {sc['grace_s']:g} s of each block skipped; "
              "hit rate = scored ticks whose label is expected (no reading counts as a miss); "
              "first hit = seconds from the block start, grace included"]
    if rp["controls"]:
        lines.append("- Controls mirrored: " + ", ".join(f"{x['action']} ({x['effect']})" for x in rp["controls"]))
    for w in report["warnings"]:
        lines.append(f"- Warning: {w}")
    blocks = report["blocks"]
    for m in MODELS:
        lines += ["", f"## {MODEL_TITLES[m]}", ""]
        if m == "map" and sc["map"]:
            lines.append(f"Not scored: {sc['map']}.")
            continue
        if not blocks:
            lines.append("No blocks.")
            continue
        lines += ["| Block | Expect | In block | Scored | Hit rate | First hit | Majority |",
                  "|---|---|---|---|---|---|---|"]
        for b in blocks:
            s = b["models"][m]
            expect = b["expect"]["map" if m == "map" else "card"]
            tags = ", ".join(x for x in (("calm" if b["calm"] else ""), ("" if b["scored"] else "unscored"),
                                         (b["ended_by"] if b["ended_by"] not in ("block_end",) else "")) if x)
            name = f"{b['label'] or b['id']} (`{b['id']}`{', ' + tags if tags else ''})"
            in_block = f"{b['ticks_in_block']}" + (f" ({b['absent_in_block']} absent)" if b["absent_in_block"] else "")
            ticks = f"{s['ticks']}" + (f" ({s['read']} read)" if s["read"] != s["ticks"] else "")
            majority = "—" if s["majority"] is None else f"{s['majority']} ({_pct(s['majority_share'])})"
            lines.append(f"| {name} | {', '.join(expect) or '—'} | {in_block} | {ticks} | {_pct(s['hit_rate'])} | "
                         f"{_secs(s['first_hit_s'], expect)} | {majority} |")
    ss = report["session"]
    lines += ["", "## Session", "", "| Model | Mean hit rate (scored non-calm blocks) | Calm false-alarm rate |",
              "|---|---|---|"]
    for m in MODELS:
        mh, fa = ss["mean_hit_rate"][m], ss["calm_false_alarm"][m]
        lines.append(f"| {MODEL_SHORT[m]} | {_pct(mh['mean'])} ({mh['blocks']} blocks) | "
                     f"{_pct(fa['rate'])} ({fa['alarms']}/{fa['ticks']} ticks) |")
    sr = ss["self_report"]
    lines.append("")
    if sr is None:
        lines.append("Self-report: no taps.")
    elif not sr["matched"]:
        lines.append(f"Self-report: {sr['taps']} taps, none matched to a map reading.")
    else:
        lines.append(f"Self-report: {sr['matched']}/{sr['taps']} taps matched; map MAE valence {sr['mae_valence']:.3f} "
                     f"(constant 0: {sr['zero_mae_valence']:.3f}), arousal {sr['mae_arousal']:.3f} "
                     f"(constant 0: {sr['zero_mae_arousal']:.3f}).")
    par = ss["parity"]
    if par is None:
        lines.append("Parity: no recorded ticks (nothing to compare).")
    else:
        pc = par["card"]
        lines.append(f"Parity (card, person {sc['person']}): {_share(pc['share'])} ({pc['matches']}/{pc['ticks']} "
                     f"ticks), {pc['mismatches']} mismatches, largest confidence difference "
                     f"{_diff(pc['max_abs_confidence_diff'])}; M0 {_share(par['m0']['share'])} "
                     f"({par['m0']['mismatches']} mismatches); map {_share(par['map']['share'])} "
                     f"({par['map']['mismatches']} mismatches, largest valence/arousal difference "
                     f"{_diff(par['map']['max_abs_valence_arousal_diff'])}).")
    return "\n".join(lines) + "\n"


def summary_markdown(reports: list[dict[str, Any]], grace: float) -> str:
    lines = ["# Benchmark summary", "",
             f"{len(reports)} session(s); the first {grace:g} s of each block are not scored. Hit rate = share of "
             "scored ticks whose label is in the block's expect list (`python -m valence.eval.benchmark`; "
             "definitions in the module docstring and docs/PROTOCOL.md section 4).", "",
             "## Sessions", "",
             "Mean hit rate over the scored non-calm blocks, calm false-alarm rate over the scored calm blocks.", "",
             "| Session | Ticks | Card | Raw M2 | M0 | Map | Card calm FA | Map calm FA | Parity (card) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        ss = r["session"]
        par = ss["parity"]
        mh = [_pct(ss["mean_hit_rate"][m]["mean"]) for m in MODELS]
        fa = ss["calm_false_alarm"]
        lines.append(f"| {r['session_id']} | {r['counts']['ticks']} | {' | '.join(mh)} | {_pct(fa['card']['rate'])} | "
                     f"{_pct(fa['map']['rate'])} | {'—' if par is None else _share(par['card']['share'])} |")
    pooled: dict[str, dict[str, Any]] = {}
    for r in reports:
        for b in r["blocks"]:
            if not b["scored"]:
                continue
            p = pooled.setdefault(b["id"], {"label": b["label"], "calm": b["calm"], "sessions": 0, "ticks": 0,
                                            "expect": b["expect"],
                                            **{m: {"hits": 0, "ticks": 0, "read": 0} for m in MODELS}})
            p["sessions"] += 1
            p["ticks"] += b["ticks_scored"] - b["absent"]
            for m in MODELS:
                s = b["models"][m]
                if s["hits"] is not None:
                    p[m]["hits"] += s["hits"]
                    p[m]["ticks"] += s["ticks"]
                    p[m]["read"] += s["read"]
    lines += ["", "## Blocks (pooled over sessions)", "",
              "Hits / ticks over every session's scored ticks of the block (— : no reading, or no scored tick).", "",
              "| Block | Expect (card) | Sessions | Ticks | Card | Raw M2 | M0 | Map |",
              "|---|---|---|---|---|---|---|---|"]
    for bid, p in pooled.items():
        rates = [_pct(p[m]["hits"] / p[m]["ticks"] if p[m]["ticks"] and p[m]["read"] else None) for m in MODELS]
        name = f"{p['label'] or bid} (`{bid}`{', calm' if p['calm'] else ''})"
        lines.append(f"| {name} | {', '.join(p['expect']['card']) or '—'} | {p['sessions']} | {p['ticks']} | "
                     f"{' | '.join(rates)} |")
    notes = sorted({r["scoring"]["map"] for r in reports if r["scoring"]["map"]})
    if notes:
        lines += ["", "Map: " + "; ".join(f"{n} ({sum(r['scoring']['map'] == n for r in reports)} session(s))"
                                          for n in notes) + "."]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    stem = _safe(report["session_id"])
    jp, mp = out / f"{stem}.json", out / f"{stem}.md"
    jp.write_text(json.dumps(_clean(report), indent=1, allow_nan=False) + "\n")
    mp.write_text(session_markdown(report))
    return jp, mp


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sessions", nargs="*", help="valence-session/1 files (or observation recordings)")
    ap.add_argument("--legacy", metavar="MANIFEST", help="legacy captures manifest (data/real/manifest.json)")
    ap.add_argument("--grace", type=float, default=60.0, help="seconds at each block's start that are not scored")
    ap.add_argument("--out", default="reports/benchmark")
    ap.add_argument("--person", type=int, default=1, help="person id to score (the protocol is one person)")
    ap.add_argument("--tz", help="time zone for the time of day, instead of the header's (or the manifest's)")
    ap.add_argument("--tod", type=float, default=12.0, help="fixed time of day (hours) when no time zone is known")
    ap.add_argument("--order", choices=("file", "mono"), default="file",
                    help="replay in file order, the browser engine's own (default), or interleave by the page clock")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"), help="served models (m2.txt, m0.txt, meta)")
    ap.add_argument("--emotion", default=str(ROOT / "artifacts" / "emotion"), help="emotion map models")
    args = ap.parse_args(argv)
    if not args.sessions and not args.legacy:
        ap.error("give session files and/or --legacy MANIFEST")
    models = load_models(args.artifacts, args.emotion)
    out = Path(args.out)
    reports = []
    for path in args.sessions:
        s = load_session(path)
        reports.append(run_session(s, models, grace=args.grace, person=args.person, tz=args.tz, tod=args.tod,
                                   order=args.order))
    if args.legacy:
        sessions, manifest = legacy_sessions(args.legacy)
        zone = args.tz or manifest.get("tz")
        for s in sessions:
            reports.append(score(s, replay(s, models, tz=zone, tod=args.tod, order="mono"), grace=args.grace,
                                 person=args.person))
    for r in reports:
        jp, mp = write_report(r, out)
        mh = r["session"]["mean_hit_rate"]
        print(f"{r['session_id']}: {r['counts']['ticks']} ticks, {len(r['blocks'])} block(s); mean hit rate card "
              f"{_pct(mh['card']['mean'])}, m0 {_pct(mh['m0']['mean'])} -> {jp}, {mp}")
    (out / "summary.md").write_text(summary_markdown(reports, args.grace))
    print(f"summary -> {out / 'summary.md'}")
    return reports


if __name__ == "__main__":
    main()
