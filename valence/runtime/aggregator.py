"""Aggregator: perception observations (~10 Hz video, ~1 Hz audio) -> one TickRecord per person per tick.

Also owns the presence-based break clock and near-field gating of room audio events:
laughter and compliments only count when their audio block is clearly louder than the room's
recent median, so other tables in a noisy hall do not register as social events at this desk.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from valence.config import Settings
from valence.contracts import AudioObs, Obs, TickRecord, VideoObs
from valence.features.spec import withdrawal_flag

ROOM = 0
NEAR_FIELD_MIN_BLOCKS = 10


@dataclass
class _PersonAcc:
    frames: int = 0
    fists: int = 0
    slump: int = 0
    hih: int = 0
    face_frames: int = 0
    motion: list[float] = field(default_factory=list)
    face: list[float] = field(default_factory=list)


@dataclass
class _PersonClock:
    first_seen: float
    last_seen: float
    break_start: float
    seeded: bool = False


class Aggregator:
    def __init__(self, settings: Settings, session_id: str, source: str = "live") -> None:
        self.s = settings
        self.session_id = session_id
        self.source = source
        self._lock = threading.Lock()
        self._tick = 0
        self._reset_tick()
        self._block_db: deque[tuple[float, float]] = deque()  # (t, db) for the 15-min room median
        self._lux = math.nan
        self._lux_quality = 0.0
        self.frames_decoded = 0
        self.frame_w = 0
        self.frame_h = 0
        self.mirrored = False
        self._clocks: dict[int, _PersonClock] = {}
        self._room_empty_since: float | None = None
        self.gated_out = 0
        self.latest_boxes: dict[int, tuple[float, float, float, float]] = {}
        self.video_frames_total = 0
        self.audio_blocks_total = 0
        self.db_correction = 0.0  # venue recalibration (POST /api/calibrate/room)
        self.lux_scale = 1.0
        self.lux_override: float | None = None  # POST /api/sim/lux (SIMULATED)
        self.last_video_t = 0.0
        self.last_audio_t = 0.0

    def _reset_tick(self) -> None:
        self._acc: dict[int, _PersonAcc] = {}
        self._db: list[float] = []
        self._laugh = 0
        self._compl = 0
        self._max_people = 0
        self._video_frames = 0
        self._audio_blocks = 0
        self._sim_fields: set[str] = set()

    # --- ingestion (perception threads) ---------------------------------------------------------
    def push(self, obs: Obs) -> None:
        with self._lock:
            if isinstance(obs, VideoObs):
                self._push_video(obs)
            elif isinstance(obs, AudioObs):
                self._push_audio(obs)

    def _push_video(self, obs: VideoObs) -> None:
        self._video_frames += 1
        self.video_frames_total += 1
        self.frames_decoded = max(self.frames_decoded, obs.frames_decoded)
        self.frame_w, self.frame_h, self.mirrored = obs.frame_w, obs.frame_h, obs.mirrored
        self.last_video_t = time.monotonic()
        if not math.isnan(obs.lux):
            self._lux, self._lux_quality = obs.lux * self.lux_scale, obs.lux_quality
        self._max_people = max(self._max_people, len(obs.persons))
        boxes = {}
        for p in obs.persons:
            acc = self._acc.setdefault(p.track_id, _PersonAcc())
            acc.frames += 1
            acc.fists += p.fist_onsets
            acc.slump += int(p.slump)
            acc.hih += int(p.head_in_hands)
            if not math.isnan(p.motion):
                acc.motion.append(p.motion)
            if p.face_visible and not math.isnan(p.face_valence):
                acc.face_frames += 1
                acc.face.append(p.face_valence)
            boxes[p.track_id] = p.bbox
            self._seen(p.track_id, obs.wall)
        self.latest_boxes = boxes

    def _push_audio(self, obs: AudioObs) -> None:
        self._audio_blocks += 1
        self.audio_blocks_total += 1
        self.last_audio_t = time.monotonic()
        if not math.isnan(obs.db_spl):
            obs = replace(obs, db_spl=obs.db_spl + self.db_correction) if self.db_correction else obs
            self._db.append(obs.db_spl)
        if obs.laughter_onsets or obs.compliment_hits:
            if self._near_field(obs.db_spl):
                self._laugh += obs.laughter_onsets
                self._compl += obs.compliment_hits
            else:
                self.gated_out += obs.laughter_onsets + obs.compliment_hits
        if not math.isnan(obs.db_spl):
            self._block_db.append((obs.t, obs.db_spl))
            while self._block_db and self._block_db[0][0] < obs.t - 900:
                self._block_db.popleft()

    def _near_field(self, db: float) -> bool:
        if len(self._block_db) < NEAR_FIELD_MIN_BLOCKS or math.isnan(db):
            return True
        median = statistics.median(v for _, v in self._block_db)
        return db >= median + self.s.near_field_db

    # --- break clock ------------------------------------------------------------------------------
    def _seen(self, pid: int, wall: float) -> None:
        clock = self._clocks.get(pid)
        if clock is None:
            self._clocks[pid] = _PersonClock(first_seen=wall, last_seen=wall, break_start=wall)
            return
        if wall - clock.last_seen >= self.s.break_absent_min * 60:
            clock.break_start = wall  # came back from a break
            clock.seeded = False
        clock.last_seen = wall

    def hours_since_break(self, pid: int, wall: float) -> float:
        clock = self._clocks.get(pid)
        return math.nan if clock is None else max(0.0, (wall - clock.break_start) / 3600.0)

    def log_break(self, pid: int | None = None, wall: float | None = None) -> None:
        wall = time.time() if wall is None else wall
        with self._lock:
            for p, clock in self._clocks.items():
                if pid is None or p == pid:
                    clock.break_start = wall
                    clock.seeded = False

    def seed_hours(self, hours: float, wall: float | None = None) -> None:
        """Operator-set clock for every known person (logged SIMULATED)."""
        wall = time.time() if wall is None else wall
        with self._lock:
            for clock in self._clocks.values():
                clock.break_start = wall - hours * 3600.0
                clock.seeded = True

    def recent_level(self, seconds: float = 10.0) -> float:
        """Leq of the last ``seconds`` of audio blocks (for venue recalibration)."""
        with self._lock:
            if not self._block_db:
                return math.nan
            t_end = self._block_db[-1][0]
            vals = [v for t, v in self._block_db if t >= t_end - seconds]
        return 10.0 * math.log10(float(np.mean(np.power(10.0, np.asarray(vals) / 10.0))))

    def calibrate(self, spl_reference: float | None = None, lux_reference: float | None = None) -> dict[str, float]:
        out = {}
        if spl_reference is not None:
            level = self.recent_level()
            if not math.isnan(level):
                self.db_correction += spl_reference - level
            out["db_correction"] = self.db_correction
        if lux_reference is not None and not math.isnan(self._lux) and self._lux > 0:
            self.lux_scale *= lux_reference / self._lux
            out["lux_scale"] = self.lux_scale
        return out

    def inject(self, *, laughter: int = 0, compliments: int = 0, fists: dict[int, int] | None = None) -> None:
        """SIMULATED events (fallback when a detector fails on stage). Tagged in simulated_fields."""
        with self._lock:
            if laughter:
                self._laugh += laughter
                self._sim_fields.add("laughter_count")
            if compliments:
                self._compl += compliments
                self._sim_fields.add("compliments_count")
            for pid, n in (fists or {}).items():
                self._acc.setdefault(pid, _PersonAcc()).fists += n
                self._sim_fields.add("clenched_fists_count")

    def reset_people(self) -> None:
        with self._lock:
            self._clocks.clear()
            self._acc.clear()
            self.latest_boxes = {}

    # --- flush (ticker thread) --------------------------------------------------------------------
    def flush(self, wall: float, t_mono: float, tod_h: float) -> list[TickRecord]:
        with self._lock:
            tick = self._tick
            self._tick += 1
            if self._db:
                leq = 10.0 * math.log10(float(np.mean(np.power(10.0, np.asarray(self._db) / 10.0))))
                p90 = float(np.percentile(self._db, 90))
            else:
                leq = p90 = math.nan
            lux = self._lux
            if self.lux_override is not None:
                lux = self.lux_override
                self._sim_fields.add("light_lux")
            sim_tick = sorted(self._sim_fields)
            room = dict(
                tick=tick, ts=wall, t_mono=t_mono, session_id=self.session_id, n_people=self._max_people,
                source=self.source, ambient_db=leq, db_p90=p90, light_lux=lux,
                lux_quality=self._lux_quality, laughter_count=self._laugh, compliments_count=self._compl, tod_h=tod_h,
            )
            people: list[TickRecord] = []
            for pid, acc in sorted(self._acc.items()):
                if acc.frames == 0:
                    continue
                slump = acc.slump / acc.frames
                motion = float(np.mean(acc.motion)) if acc.motion else math.nan
                clock = self._clocks.get(pid)
                people.append(TickRecord(
                    **room, person_id=pid, display_id=f"Person #{pid}",
                    simulated_fields=sim_tick + (["hours_since_break"] if clock and clock.seeded else []),
                    clenched_fists_count=acc.fists, posture_slump=slump, head_in_hands_frac=acc.hih / acc.frames,
                    social_withdrawal_flag=withdrawal_flag(slump, motion), motion_energy=motion,
                    face_valence=float(np.median(acc.face)) if acc.face else math.nan,
                    face_visible=acc.face_frames / acc.frames,
                    hours_since_break=self.hours_since_break(pid, wall),
                ))
            self._drop_absent(wall)
            self._reset_tick()
        return [room_record(room, people, sim_tick)] + people

    def _drop_absent(self, wall: float) -> None:
        for pid in [p for p, c in self._clocks.items() if wall - c.last_seen > self.s.person_drop_min * 60]:
            del self._clocks[pid]


def room_record(room: dict, people: list[TickRecord], sim_tick: list[str] = ()) -> TickRecord:
    """Room slot: room channels plus aggregates of whoever is present (also used by room-only mode)."""

    def mean(attr: str) -> float:
        vals = [getattr(p, attr) for p in people if not math.isnan(getattr(p, attr))]
        return float(np.mean(vals)) if vals else math.nan

    faces = [p.face_valence for p in people if not math.isnan(p.face_valence)]
    sim = sorted({f for p in people for f in p.simulated_fields} | set(sim_tick))
    return TickRecord(
        **room, person_id=ROOM, display_id="Room", simulated_fields=sim,
        clenched_fists_count=sum(p.clenched_fists_count for p in people),
        posture_slump=mean("posture_slump"), head_in_hands_frac=mean("head_in_hands_frac"),
        social_withdrawal_flag=int(any(p.social_withdrawal_flag for p in people)),
        motion_energy=mean("motion_energy"), face_valence=float(np.median(faces)) if faces else math.nan,
        face_visible=mean("face_visible") if people else 0.0, hours_since_break=mean("hours_since_break"),
    )
