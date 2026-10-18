"""Runtime engine: one ticker thread drives the 5 s state-vector tick.

Each tick: take TickRecords from the active source (live aggregator or scripted replay), apply any
SIMULATED overlays, append to the rolling window, close the purge-gate tick, predict per person (or
for the room in room-only mode), and publish governance, state_vector and prediction messages.
Narration is requested asynchronously; its text arrives later as an ``explanation`` message.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from valence.config import Settings
from valence.contracts import (
    STATE_SETS,
    TICK_FIELDS,
    Governance,
    Obs,
    Prediction,
    StateVector,
    TickRecord,
    TrackPerson,
    Tracks,
    VideoObs,
    display_label,
)
from valence.explain.explainer import build_fact_sheet
from valence.explain.narrator_client import NarratorClient, band
from valence.explain.template import template_text
from valence.features.window import ROOM, RollingWindow
from valence.governance.audit import AuditLog
from valence.governance.purge import PurgeGate
from valence.governance.purge import gate as default_gate
from valence.models.predictor import PersonSmoother, Predictor
from valence.runtime.aggregator import Aggregator
from valence.runtime.bus import EventBus

log = logging.getLogger(__name__)


def local_tod(wall: float) -> float:
    dt = datetime.fromtimestamp(wall)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


class TickSource(Protocol):
    label: str

    def next_tick(self, wall: float, t_mono: float) -> list[TickRecord] | None: ...


class LiveSource:
    """Ticks come from the aggregator, fed by the edge (or by a features-only JSONL replay)."""

    def __init__(self, aggregator: Aggregator, label: str = "live") -> None:
        self.aggregator = aggregator
        self.label = label

    def next_tick(self, wall: float, t_mono: float) -> list[TickRecord]:
        return self.aggregator.flush(wall, t_mono, local_tod(wall))


class ScenarioSource:
    """Plays a rendered scenario (simulator output) one tick at a time, re-stamped to the current clock."""

    def __init__(self, df: pd.DataFrame, name: str, loop: bool = False) -> None:
        self.name = name
        self.label = "replay"
        self.loop = loop
        self._ticks = [g for _, g in df.groupby("tick", sort=True)]
        self._i = 0
        self.session_id = f"replay-{name}-{uuid.uuid4().hex[:6]}"

    @property
    def done(self) -> bool:
        return self._i >= len(self._ticks) and not self.loop

    def next_tick(self, wall: float, t_mono: float) -> list[TickRecord] | None:
        if self._i >= len(self._ticks):
            if not self.loop:
                return None
            self._i = 0
        g = self._ticks[self._i]
        self._i += 1
        out = []
        for row in g.to_dict("records"):
            fields = {k: row[k] for k in TICK_FIELDS if k in row}
            fields.update(tick=self._i - 1, ts=wall, t_mono=t_mono, session_id=self.session_id, source="replay",
                          simulated_fields=list(row.get("simulated_fields") or []))
            for k, v in fields.items():
                if isinstance(v, (np.floating, np.integer)):
                    fields[k] = v.item()
            out.append(TickRecord(**fields))
        return out


class Runtime:
    def __init__(self, settings: Settings, bus: EventBus, predictor: Predictor, *,
                 gate: PurgeGate = default_gate, narrator: NarratorClient | None = None,
                 audit_dir: str | None = None) -> None:
        self.s = settings
        self.bus = bus
        self.predictor = predictor
        self.gate = gate
        self.narrator = narrator
        self.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.window = RollingWindow(settings.window_ticks)
        self.aggregator = Aggregator(settings, self.session_id)
        self.source: TickSource = LiveSource(self.aggregator)
        self.mode = settings.mode
        self.room_only = settings.room_only
        self.overlays = {"face_occlusion": False, "low_light": False}
        self.speed = 1.0
        self.paused = False
        self.smoothers: dict[int, PersonSmoother] = {}
        self.latest: dict[int, dict[str, Any]] = {}
        self.history: deque[dict[str, Any]] = deque(maxlen=settings.window_ticks * 8)
        self._last_narrated: dict[int, tuple[str, str, float]] = {}
        self._last_audited: dict[int, tuple[str, float]] = {}
        self._tick_i = 0
        self._last_tracks = 0.0
        self.loop_latency_ms = 0.0
        self.started = time.time()
        self.camera = False
        self.mic = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.audit = AuditLog(audit_dir or settings.audit_dir, self.session_id, ttl_s=settings.audit_ttl_s)
        self._fps_frames = (time.monotonic(), 0)
        self.fps = 0.0
        self.purge_gap = 0
        self._gap_ticks = 0
        self._purged_base = 0  # purged video frames before the current source began
        self.frames_purged = 0

    # --- lifecycle --------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ticker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2 * self.s.tick_s)

    def close(self) -> None:
        self._stop_feeder()
        self.stop()
        self.audit.close(delete=True)

    def _run(self) -> None:
        period = self.s.tick_s / self.speed
        next_t = time.monotonic() + period
        while not self._stop.is_set():
            delay = next_t - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            next_t += self.s.tick_s / self.speed
            if self.paused:
                continue
            try:
                self.step()
            except Exception:  # keep ticking; the status message reports degraded state
                log.exception("tick failed")

    # --- sources and operator actions ----------------------------------------------------------------
    def use_scenario(self, df: pd.DataFrame, name: str, *, speed: float = 1.0, loop: bool = False) -> None:
        self._stop_feeder()
        self._new_source_counters()
        self.source = ScenarioSource(df, name, loop=loop)
        self.mode = "replay"
        self.speed = speed
        self.reset_people(keep_room=False)
        self._audit("operator", {"action": "replay", "scenario": name, "speed": speed})

    def use_live(self) -> None:
        self._stop_feeder()
        self._new_source_counters()
        self.source = LiveSource(self.aggregator)
        self.aggregator.source = "live"
        self.mode = "live"
        self.speed = 1.0
        self._audit("operator", {"action": "live"})

    def use_recording(self, path, *, speed: float = 1.0, loop: bool = False) -> None:
        """Replay a features-only JSONL recording through the live aggregator (stage fallback)."""
        from valence.runtime.recording import JsonlFeeder

        self._stop_feeder()
        self._new_source_counters()
        self.reset_people(keep_room=False)
        self.source = LiveSource(self.aggregator, label="replay")
        self.aggregator.source = "replay"
        self.mode = "replay"
        self.speed = 1.0
        self.feeder = JsonlFeeder(path, self.push_obs, speed=speed, loop=loop)
        self.feeder.start()
        self._audit("operator", {"action": "replay_recording", "recording": str(getattr(path, "name", path))[:80]})

    def _new_source_counters(self) -> None:
        """Align the decoded/purged frame pair to the source that is starting now."""
        self.aggregator.reset_frame_counter()
        self._purged_base = self.gate.stats().purged.get("video", 0)
        self._gap_ticks = 0

    def _stop_feeder(self) -> None:
        feeder = getattr(self, "feeder", None)
        if feeder is not None:
            feeder.stop()
            self.feeder = None

    def reset_people(self, keep_room: bool = True) -> None:
        if keep_room:
            self.window.reset_people()
        else:
            self.window.clear()
        self.aggregator.reset_people()
        self.smoothers.clear()
        self.latest.clear()
        self.history.clear()
        if self.narrator is not None:
            self.narrator.clear_cache()
        self._audit("operator", {"action": "reset", "keep_room": keep_room})

    def set_overlay(self, name: str, enabled: bool) -> None:
        self.overlays[name] = enabled
        self._audit("operator", {"action": "sim", "overlay": name, "enabled": enabled})

    def seed(self, df: pd.DataFrame, scenario: str, minutes: float, hours_since_break: float | None) -> int:
        """Pre-fill room history from a scenario (context only, tagged sim_seed)."""
        room = df[(df["person_id"] == ROOM) & (df["tick"] < minutes * 60 / self.s.tick_s)]
        now, mono = time.time(), time.monotonic()
        n = len(room)
        for i, row in enumerate(room.to_dict("records")):
            fields = {k: row[k] for k in TICK_FIELDS if k in row}
            offset = (n - i) * self.s.tick_s
            fields.update(ts=now - offset, t_mono=mono - offset, source="sim_seed",
                          simulated_fields=["room_history"], session_id=self.session_id)
            for k, v in fields.items():
                if isinstance(v, (np.floating, np.integer)):
                    fields[k] = v.item()
            self.window.append(TickRecord(**fields))
        if hours_since_break is not None:
            self.aggregator.seed_hours(hours_since_break)
        self._audit("operator", {"action": "demo_seed", "scenario": scenario, "minutes": minutes,
                                 "hours_since_break": hours_since_break, "simulated_fields": ["room_history"]})
        return n

    def _audit(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self.audit.append(kind, payload)
        except Exception:
            log.exception("audit append failed")

    # --- perception sink ----------------------------------------------------------------------------
    def push_obs(self, obs: Obs) -> None:
        self.aggregator.push(obs)
        if isinstance(obs, VideoObs):
            self.camera = True
            t0, n = self._fps_frames
            n += 1
            now = time.monotonic()
            if now - t0 >= 2.0:
                self.fps = n / (now - t0)
                t0, n = now, 0
            self._fps_frames = (t0, n)
            if now - self._last_tracks >= 0.1:
                self._last_tracks = now
                tallest = max((p.bbox[3] - p.bbox[1] for p in obs.persons), default=0.0)
                shown = [p for p in obs.persons
                         if p.bbox[3] - p.bbox[1] >= self.s.participant_rel_height * tallest]
                self.bus.publish(Tracks(
                    t=obs.wall, capture_ts_ms=int(obs.wall * 1000), frame_w=obs.frame_w, frame_h=obs.frame_h,
                    mirrored=obs.mirrored,
                    persons=[TrackPerson(id=p.track_id, label=f"Person #{p.track_id}", bbox=p.bbox,
                                         fist_active=p.fist_active, slump=p.slump) for p in shown],
                ))
        else:
            self.mic = True

    # --- the tick -------------------------------------------------------------------------------------
    def step(self) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        wall, mono = time.time(), time.monotonic()
        records = self.source.next_tick(wall, mono)
        if records is None:  # scenario finished
            self.paused = True
            return []
        records = [self._overlay(r) for r in records]
        for r in records:
            self.window.append(r)
        purge = self.gate.end_tick()
        # Both counters cover the current source only. Replayed recordings carry the original capture's
        # counter, and this process decodes no frames for them, so replay reports its own purged count.
        self.frames_purged = purge.purged.get("video", 0) - self._purged_base
        decoded = self.aggregator.frames_decoded if self.source.label == "live" else self.frames_purged
        # Every decoded frame must be purged; allow the capture's in-flight slot (<= 2 frames).
        self.purge_gap = max(0, decoded - self.frames_purged)
        self._gap_ticks = self._gap_ticks + 1 if self.purge_gap > 2 else 0
        self.bus.publish(Governance(
            t=wall, tick=self._tick_i, frames_decoded=decoded,
            frames_purged=self.frames_purged, audio_blocks_purged=purge.purged.get("audio", 0),
            transcripts_purged=purge.transcripts_purged, violations=purge.violations,
            live_raw_buffers=purge.live_raw_buffers, audit_hash=self.audit.last_hash[:16],
            retention_s=self.s.audit_ttl_s, purge_gap=self.purge_gap,
        ))
        self._evict(wall)
        targets = [ROOM] if self.room_only else sorted({r.person_id for r in records if r.person_id != ROOM})
        emitted = []
        for pid in targets:
            msg = self._predict(pid, wall)
            if msg:
                emitted.append(msg)
        self._tick_i += 1
        self.loop_latency_ms = (time.perf_counter() - t0) * 1000
        return emitted

    def _evict(self, wall: float) -> None:
        """Forget people absent longer than ``person_drop_min`` (window slot, smoother, latest state)."""
        for pid in self.window.people():
            rows = self.window.snapshot(pid)
            if rows and wall - rows[-1].ts > self.s.person_drop_min * 60:
                self.window.drop(pid)
                self.smoothers.pop(pid, None)
                self.latest.pop(pid, None)

    def _overlay(self, r: TickRecord) -> TickRecord:
        if not any(self.overlays.values()):
            return r
        sim = list(r.simulated_fields)
        if self.overlays["low_light"]:
            noisy = r.face_valence if math.isnan(r.face_valence) else float(
                np.clip(r.face_valence + np.random.normal(0, 0.35), -1, 1))
            r = replace(r, light_lux=r.light_lux * 0.15 if not math.isnan(r.light_lux) else 30.0, face_valence=noisy)
            sim.append("low_light")
        if self.overlays["face_occlusion"]:
            r = replace(r, face_valence=math.nan, face_visible=0.0)
            sim.append("face_occlusion")
        return replace(r, simulated_fields=sim)

    def _predict(self, pid: int, wall: float) -> dict[str, Any] | None:
        person = self.window.snapshot(pid)
        if not person:
            return None
        room = self.window.room()
        out = self.predictor.predict(person, room)
        if out is None:
            return None
        sm = self.smoothers.setdefault(pid, PersonSmoother(
            self.predictor.states, self.s.tick_s, self.s.warmup_s, self.s.override_block_s))
        smooth = sm.update(out, len(person))
        last = person[-1]
        states = self.predictor.states
        label = smooth.label
        conf = out.probabilities.get(label, out.confidence)
        n_people = max(1, last.n_people)
        shown = display_label(label, mask_override=smooth.mask_override, n_people=n_people, states=states)
        name = "Room" if pid == ROOM else f"Person #{pid}"
        event_id = f"p{pid}-{self._tick_i:06d}"
        face_visible = last.face_visible > 0 and not math.isnan(last.face_valence)
        sheet = build_fact_sheet(
            event_id=event_id, person=name, label=label, label_display=shown, confidence=conf,
            probabilities=out.probabilities, window_fill_min=len(person) * self.s.tick_s / 60, n_people=n_people,
            face_only_label=out.face_only_label, face_only_confidence=out.face_only_confidence,
            face_valence=None if math.isnan(last.face_valence) else last.face_valence, face_visible=face_visible,
            mask_override=smooth.mask_override, layers=out.layers,
            face_visible_frac=float(np.mean([r.face_visible for r in person[-60:]])),
            lux=None if math.isnan(last.light_lux) else last.light_lux, ood_features=out.ood_features,
        )
        text = template_text(sheet)
        sim_fields = sorted(set(last.simulated_fields) | {f for r in room[-180:] for f in r.simulated_fields})
        pred = Prediction(
            event_id=event_id, t=wall, person_id=pid, label=label, label_display=shown, confidence=round(conf, 4),
            probabilities=sheet.probabilities,
            face_only={"label": sheet.face_only.label, "confidence": sheet.face_only.confidence},
            mask_override=smooth.mask_override, warming_up=smooth.warming_up, layers=out.layers,
            data_quality=sheet.data_quality, explanation={"text": text, "source": "template"},
            source=last.source, simulated_fields=sim_fields,
        )
        sv = self._state_vector(pid, person, room, wall, sim_fields)
        self.bus.publish(sv)
        self.bus.publish(pred)
        sv_d, pred_d = sv.model_dump(), pred.model_dump()
        self.history.append(sv_d)
        self.history.append(pred_d)
        self.latest[pid] = {"state_vector": sv_d, "prediction": pred_d}
        self._maybe_narrate(pid, sheet, wall)
        self._maybe_audit(pid, pred, wall)
        return pred_d

    def _state_vector(self, pid: int, person: list[TickRecord], room: list[TickRecord], wall: float,
                      sim_fields: list[str]) -> StateVector:
        last = person[-1]
        room15 = room[-180:] or person[-180:]

        def nz(v: float) -> float | None:
            return None if math.isnan(v) else round(float(v), 3)

        h = int(last.tod_h) if not math.isnan(last.tod_h) else 0
        m = int(round((last.tod_h - h) * 60)) % 60 if not math.isnan(last.tod_h) else 0
        return StateVector(
            t=wall, person_id=pid, ambient_db=nz(last.ambient_db), light_lux=nz(last.light_lux),
            hours_since_break=nz(last.hours_since_break), time_of_day=f"{h % 24:02d}:{m:02d}",
            clenched_fists_5m=int(sum(r.clenched_fists_count for r in person[-60:])),
            compliments_15m=int(sum(r.compliments_count for r in room15)),
            laughter_15m=int(sum(r.laughter_count for r in room15)),
            posture_slump=nz(last.posture_slump), head_in_hands=nz(last.head_in_hands_frac),
            social_withdrawal=int(last.social_withdrawal_flag), face_visible=bool(last.face_visible > 0),
            facial_valence=nz(last.face_valence), window_fill_min=round(len(person) * self.s.tick_s / 60, 2),
            source=last.source, simulated_fields=sim_fields,
        )

    def _maybe_narrate(self, pid: int, sheet, wall: float) -> None:
        if self.narrator is None:
            return
        prev = self._last_narrated.get(pid)
        key = (sheet.label_display, band(sheet.confidence))
        if prev is None or prev[:2] != key or wall - prev[2] >= 30:
            self._last_narrated[pid] = (*key, wall)
            self.narrator.request(sheet, pid)

    def _maybe_audit(self, pid: int, pred: Prediction, wall: float) -> None:
        prev = self._last_audited.get(pid)
        if prev is None or prev[0] != pred.label_display or wall - prev[1] >= 30:
            self._last_audited[pid] = (pred.label_display, wall)
            self._audit("prediction", {
                "person": "Room" if pid == ROOM else f"Person #{pid}", "label": pred.label,
                "confidence": pred.confidence, "mask_override": pred.mask_override, "source": pred.source,
                "layers": {layer.name: layer.contribution for layer in pred.layers},
                "simulated_fields": pred.simulated_fields[:8],
            })

    # --- read models for the API ------------------------------------------------------------------------
    def history_since(self, minutes: float) -> list[dict[str, Any]]:
        cutoff = time.time() - minutes * 60
        return [m for m in list(self.history) if m["t"] >= cutoff]

    def states(self) -> tuple[str, ...]:
        return self.predictor.states


__all__ = ["Runtime", "LiveSource", "ScenarioSource", "STATE_SETS"]
