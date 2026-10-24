"""Simulator v2: session-level time series of the Rolling State Vector with a hidden emotional state.

One engine serves three uses: the training corpus (semi-Markov states), scripted replay scenarios
(``scenarios/*.yaml``) and the demo seed. Output rows use the ``TickRecord`` fields from
``valence.contracts`` plus the hidden label (``true_state``) and ``episode_age_s``. Every tick also has
a room row (``person_id == 0``) so room channels keep their history while people come and go, exactly
like the live aggregator's room slot.

Spec: docs/DATASET.md. Parameters: valence/sim/params.py.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from valence.contracts import AFFECT_ANCHORS, CORE_STATES, EMOTION_STATES, STATE_SETS, TICK_FIELDS
from valence.features.spec import withdrawal_flag
from valence.sim.params import (FACE_CUE_NUISANCE, FACE_CUES, FP, NUISANCE, FaceCueNuisance, Nuisance, StateParams,
                                params_for)

TICK_S = 5.0
TICKS_PER_MIN = 60.0 / TICK_S
EMPTY_ROOM_DB = 40.0
ROOM_ID = 0
# The simulator also knows the emotion map's ten states ("emotion"); contracts.STATE_SETS stays as served.
SIM_STATE_SETS: dict[str, tuple[str, ...]] = {**STATE_SETS, "emotion": EMOTION_STATES}
FACE_CUE_FIELDS = ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away")
HIDDEN_COLUMNS = ["true_state", "episode_age_s", "true_valence", "true_arousal"]

LEGACY_COLUMNS = [
    "timestamp", "ambient_db", "light_lux", "hours_since_break", "clenched_fists_count",
    "compliments_count", "social_withdrawal_flag", "facial_valence_snapshot", "true_emotional_state",
]


@dataclass
class Segment:
    minutes: float
    states: list[str | None]  # one per person; None = absent (on a break)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Script:
    """A scripted session (replay scenarios). Unset fields fall back to random nuisance draws."""

    name: str
    people: int
    segments: list[Segment]
    seed: int = 0
    description: str = ""
    start_hour: float | None = None
    start_hours_since_break: list[float] | None = None
    lux_base: float | None = None
    mic_offset_db: float | None = None
    face_hidden: float | None = None
    venue_db: float | None = None  # hall noise; None = a quiet team room (no venue term)

    @property
    def ticks(self) -> int:
        return int(round(sum(s.minutes for s in self.segments) * TICKS_PER_MIN))


def power_sum(*levels_db: float) -> float:
    """Level of incoherent sources together (dB)."""
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in levels_db))


def _alpha(tau_min: float) -> float:
    return 1.0 - math.exp(-TICK_S / (tau_min * 60.0))


class _Person:
    """Hidden-state process and lagged behavior for one person."""

    def __init__(self, rng: np.random.Generator, states: tuple[str, ...], n_people: int, h0: float,
                 nz: Nuisance, noise_scale: float) -> None:
        self.rng = rng
        self.states = states
        self.n_people = n_people
        self.hours = h0
        self.nz = nz
        self.ns = noise_scale
        self.alpha = _alpha(rng.uniform(*nz.person_tau_min) * noise_scale)
        self.state: str | None = None
        self.params: StateParams | None = None
        self.dwell_left = 0
        self.episode_age = 0
        self.episode_face = 0.0
        self.absent_left = 0
        self.lag = None  # lagged behavior vector
        self.slump_on = 0
        self.hih_on = 0
        self.log_motion: float | None = None
        self.still = 0
        self.burst = 1.0
        self.episode_hidden: float | None = None
        # Emotion-map face cues: their own random stream (``setup_cues``), so the channels above are unchanged.
        self.frng: np.random.Generator | None = None
        self.true_v = math.nan
        self.true_a = math.nan

    def setup_cues(self, frng: np.random.Generator, cn: FaceCueNuisance) -> None:
        self.frng = frng
        self.cn = cn
        self.blink_mult = float(frng.uniform(*cn.blink_mult))
        self.neutral = frng.normal(0.0, cn.neutral_sd, 4)  # error of the 20 s neutral: brow, eye, frown, smile
        self.neutral_deg = frng.normal(0.0, cn.neutral_deg_sd, 2)  # yaw, pitch
        self.cue_target: np.ndarray | None = None
        self.cue_lag: np.ndarray | None = None
        self.away_on = 0

    # --- hidden state -------------------------------------------------------------------------
    def _fatigue(self) -> float:
        return 1.0 / (1.0 + math.exp(-(self.hours - 3.0) / 0.5))

    def _next_state(self, after_break: bool) -> str:
        if after_break:
            options = ["Focused"] + (["Collaborative_Happy"] if self.n_people >= 2 else [])
            options += ["Calm"] if "Calm" in self.states else []
            return options[int(self.rng.integers(len(options)))]
        f = self._fatigue()
        weights = []
        for s in self.states:
            p = params_for(s, self.states)
            w = max(0.05, 1.0 + p.fatigue_affinity * f)
            w *= p.group_affinity if self.n_people >= 2 else p.solo_affinity
            weights.append(0.0 if s == self.state else w)
        w = np.asarray(weights)
        return self.states[int(self.rng.choice(len(self.states), p=w / w.sum()))]

    def enter(self, state: str, overrides: dict[str, Any] | None = None, dwell_ticks: int | None = None) -> None:
        self.state = state
        p = params_for(state, self.states)
        if overrides:
            p = replace(p, **{k: v for k, v in overrides.items() if hasattr(p, k)})
        self.params = p
        if dwell_ticks is None:
            mean = p.dwell_min * TICKS_PER_MIN
            dwell_ticks = max(int(2 * TICKS_PER_MIN), int(self.rng.gamma(3.0, mean / 3.0)))
        self.dwell_left = dwell_ticks
        self.episode_age = 0
        mu = p.mask_mu if self.rng.random() < p.mask_p else p.face_mu
        self.episode_face = float(self.rng.normal(mu, self.nz.face_episode_sd))
        self.episode_hidden = float(self.rng.uniform(0.4, 0.7)) if p.face_hidden_extra > 0 else None
        if self.frng is not None:
            self._enter_cues(state)

    def _enter_cues(self, state: str) -> None:
        """Episode draws for the face cues and the episode's true valence/arousal (anchor + offset)."""
        c, cn, r = FACE_CUES[state], self.cn, self.frng
        dv, da = r.normal(0.0, cn.va_sd, 2)
        av, aa = AFFECT_ANCHORS[state]
        self.true_v = float(np.clip(av + dv, -1.0, 1.0))
        self.true_a = float(np.clip(aa + da, -1.0, 1.0))
        jit = r.normal(0.0, cn.episode_sd, 4)
        blink_jit = float(np.exp(r.normal(0.0, cn.blink_episode_sd)))
        away = float(np.clip(c.away_p + r.normal(0.0, cn.away_episode_sd), 0.0, 0.95))
        self.cue_target = np.array([
            c.brow + jit[0],
            c.eye + jit[1] + cn.eye_from_arousal * da,
            c.frown + jit[2] - cn.frown_from_valence * dv,
            c.smile + jit[3] + cn.smile_from_valence * dv,
            c.blink_per_min * blink_jit * max(0.2, 1.0 + cn.blink_from_arousal * da),
            away, c.pitch + r.normal(0.0, cn.pitch_episode_sd), c.away_pitch,
        ])

    def face_cues(self, face_ok: bool, face: float, face_vis: float, lux: float) -> dict[str, float]:
        """One tick of the emotion-map face cues (call once per tick while present, after ``behave``)."""
        r, cn = self.frng, self.cn
        tgt = self.cue_target
        self.cue_lag = tgt.copy() if self.cue_lag is None else self.cue_lag + self.alpha * (tgt - self.cue_lag)
        brow, eye, frown, smile, blink_rate, away_p, pitch, away_pitch = self.cue_lag
        self.away_on = _two_state(r, self.away_on, away_p, switch=cn.away_switch)
        if not face_ok:
            return {**{k: math.nan for k in FACE_CUE_FIELDS}, "blink_count": 0}
        noise = 1.0 + 3.0 * max(0.0, (self.nz.low_light_lux - lux) / self.nz.low_light_lux) * 0.35
        sd = cn.tick_sd * self.ns * noise
        psd = cn.pitch_tick_sd * self.ns * noise
        f_pos, f_neg = max(face, 0.0), max(-face, 0.0)
        out = {
            "brow_down": brow + self.neutral[0] + r.normal(0.0, sd),
            "eye_open": eye + self.neutral[1] + r.normal(0.0, sd),
            "mouth_frown": frown + cn.frown_from_face * f_neg + self.neutral[2] + r.normal(0.0, 0.75 * sd),
            "smile": smile + cn.smile_from_face * f_pos + self.neutral[3] + r.normal(0.0, sd),
        }
        out = {k: float(np.clip(v, -1.0, 1.0)) for k, v in out.items()}
        if self.away_on:
            away = r.uniform(0.5, 1.0)
            yaw = r.uniform(*cn.away_yaw) * (1.0 if r.random() < 0.5 else -1.0)
            pitch_t = pitch + away_pitch + r.normal(0.0, 2.0 * psd)
        else:
            away = r.uniform(0.0, 0.12) if r.random() < 0.3 else 0.0
            yaw = r.normal(0.0, 4.0 * self.ns)
            pitch_t = pitch + r.normal(0.0, psd)
        out["head_yaw"] = float(abs(yaw + self.neutral_deg[0]))
        out["head_pitch"] = float(np.clip(pitch_t + self.neutral_deg[1], -90.0, 90.0))
        out["head_away"] = float(away)
        out["blink_count"] = int(r.poisson(blink_rate * self.blink_mult / 60.0 * TICK_S * face_vis))
        return out

    def step_markov(self) -> None:
        """Advance the semi-Markov process by one tick (training corpus)."""
        if self.absent_left > 0:
            self.absent_left -= 1
            if self.absent_left == 0:
                self.hours = 0.0
                self.enter(self._next_state(after_break=True))
            return
        if self.state is None:
            self.enter(self._next_state(after_break=False))
        rate = self.nz.break_per_min * (2.0 if self.state in ("Exhausted", "Overwhelmed") else 1.0)
        if self.rng.random() < rate / TICKS_PER_MIN:
            self.absent_left = int(self.rng.uniform(*self.nz.break_len_min) * TICKS_PER_MIN)
            self.state = None
            return
        self.dwell_left -= 1
        if self.dwell_left <= 0:
            self.enter(self._next_state(after_break=False))

    @property
    def present(self) -> bool:
        return self.state is not None

    # --- observable behavior --------------------------------------------------------------------
    def behave(self) -> dict[str, float]:
        """Lagged behavior targets for this tick (call only while present)."""
        p = self.params
        target = np.array([p.fist_rate, p.slump_p, p.hih_p, p.motion, p.laugh_rate, p.compl_rate])
        self.lag = target.copy() if self.lag is None else self.lag + self.alpha * (target - self.lag)
        self.hours += TICK_S / 3600.0
        self.episode_age += 1
        fist_rate, slump_p, hih_p, motion, laugh_rate, compl_rate = self.lag
        return {"fist_rate": fist_rate, "slump_p": slump_p, "hih_p": hih_p, "motion": motion,
                "laugh_rate": laugh_rate, "compl_rate": compl_rate}


def _two_state(rng: np.random.Generator, on: int, share: float, switch: float = 0.15) -> int:
    share = min(max(share, 0.0), 1.0)
    if on:
        return 0 if rng.random() < switch * (1.0 - share) else 1
    return 1 if rng.random() < switch * share else 0


def simulate_session(
    session_id: str,
    rng: np.random.Generator,
    *,
    state_set: str = "core",
    noise_scale: float = 1.0,
    nuisance: Nuisance = NUISANCE,
    script: Script | None = None,
) -> pd.DataFrame:
    states = SIM_STATE_SETS[state_set]
    nz = nuisance
    ns = noise_scale
    if script is not None:
        n_people = script.people
        n_ticks = script.ticks
    else:
        n_people = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
        n_ticks = int(rng.uniform(*nz.session_min) * TICKS_PER_MIN)

    def pick(value, draw):
        return draw() if value is None else value

    start_hour = pick(script.start_hour if script else None, lambda: rng.uniform(*nz.start_hour))
    lux_base = pick(script.lux_base if script else None,
                    lambda: float(np.exp(rng.uniform(*np.log(nz.lux_range)))))
    mic_offset = pick(script.mic_offset_db if script else None,
                      lambda: rng.uniform(-nz.mic_offset_db, nz.mic_offset_db))
    face_hidden = pick(script.face_hidden if script else None, lambda: rng.uniform(*nz.face_hidden_range))
    h0s = (script.start_hours_since_break if script and script.start_hours_since_break
           else [rng.uniform(*nz.start_hours_since_break) for _ in range(n_people)])
    room_alpha = _alpha(rng.uniform(*nz.room_tau_min) * ns)
    # Venue and arrival baseline come from their own stream, so adding them left every scripted scenario's
    # other draws unchanged.
    vrng = np.random.default_rng(zlib.crc32(session_id.encode()))
    venue_db = (script.venue_db if script else float(vrng.uniform(*nz.venue_db_range)))
    venue_terms = () if venue_db is None else (venue_db,)
    base_db = power_sum(*venue_terms, nz.neutral_db) + mic_offset + float(vrng.normal(0.0, nz.baseline_db_noise * ns))
    night0 = 1.0 if (start_hour >= 22 or start_hour < 6) else 0.0
    base_lux = lux_base * (1.0 - 0.15 * night0) * float(np.exp(vrng.normal(0.0, nz.baseline_lux_noise * ns)))

    people = [_Person(rng, states, n_people, h0s[i % len(h0s)], nz, ns) for i in range(n_people)]
    frng = np.random.default_rng(zlib.crc32(f"{session_id}/face-cues".encode()))
    for person in people:
        person.setup_cues(frng, FACE_CUE_NUISANCE)

    # Scripted state timeline: (tick -> per-person (state, overrides, segment_start))
    timeline: list[tuple[list[str | None], dict[str, Any], bool]] = []
    if script is not None:
        for seg in script.segments:
            seg_ticks = int(round(seg.minutes * TICKS_PER_MIN))
            for k in range(seg_ticks):
                timeline.append((seg.states, seg.overrides, k == 0))

    room_db = EMPTY_ROOM_DB
    room_lux_mult = 1.0
    db_drift = 0.0
    lux_drift = 0.0
    sound_bump, sound_left = 0.0, 0
    light_mult, light_left = 1.0, 0
    rows: list[dict[str, Any]] = []
    for t in range(n_ticks):
        # 1) hidden states
        if script is not None:
            seg_states, overrides, seg_start = timeline[t]
            for i, person in enumerate(people):
                s = seg_states[i] if i < len(seg_states) else seg_states[-1]
                if s is None:
                    if person.present:
                        person.state = None
                    person.absent_left += 1
                    continue
                if not person.present:
                    if person.absent_left >= nz.break_len_min[0] * TICKS_PER_MIN:
                        person.hours = 0.0
                    person.absent_left = 0
                if seg_start or person.state != s:
                    person.enter(s, overrides, dwell_ticks=10**9)
        else:
            for person in people:
                person.step_markov()

        present = [p for p in people if p.present]
        n_present = len(present)
        behaviors = [p.behave() for p in present]

        # 2) room channels
        db_target = (np.mean([p.params.db_target for p in present]) if present else EMPTY_ROOM_DB)
        room_db += room_alpha * (db_target - room_db)
        lux_target = np.mean([p.params.lux_mult for p in present]) if present else 1.0
        room_lux_mult += room_alpha * (lux_target - room_lux_mult)
        tod = (start_hour + t * TICK_S / 3600.0) % 24.0
        night = 1.0 if (tod >= 22 or tod < 6) else 0.0
        audio_ok = rng.random() >= nz.dropout
        laugh_rate = sum(b["laugh_rate"] for b in behaviors) + FP["laugh_per_min"]
        compl_rate = sum(b["compl_rate"] for b in behaviors) + FP["compl_per_min"]
        laughs = int(rng.poisson(laugh_rate / TICKS_PER_MIN)) if audio_ok else 0
        compls = int(rng.poisson(compl_rate / TICKS_PER_MIN)) if audio_ok else 0
        db_drift = 0.95 * db_drift + rng.normal(0.0, nz.db_drift * ns)
        if sound_left > 0:
            sound_left -= 1
        elif rng.random() < nz.sound_events_per_min / TICKS_PER_MIN:
            sound_bump = rng.uniform(*nz.sound_event_db) * (1 if rng.random() < 0.7 else -0.5)
            sound_left = int(rng.integers(nz.sound_event_ticks[0], nz.sound_event_ticks[1] + 1))
        if sound_left == 0:
            sound_bump = 0.0
        if light_left > 0:
            light_left -= 1
        elif rng.random() < nz.light_events_per_min / TICKS_PER_MIN:
            light_mult = rng.uniform(*nz.light_event_mult)
            light_left = int(rng.integers(nz.light_event_ticks[0], nz.light_event_ticks[1] + 1))
        if light_left == 0:
            light_mult = 1.0
        lux_drift = 0.98 * lux_drift + rng.normal(0.0, nz.lux_drift * ns)
        if audio_ok:
            ambient = (power_sum(*venue_terms, room_db) + mic_offset + db_drift + sound_bump
                       + rng.normal(0.0, nz.db_noise * ns) + 2.0 * min(laughs, 2))
            p90 = ambient + 1.5 + abs(rng.normal(0.0, 1.0 * ns)) + 1.5 * (laughs > 0)
        else:
            ambient = p90 = math.nan
        lux = (lux_base * (1.0 - 0.15 * night) * room_lux_mult * light_mult
               * float(np.exp(lux_drift + rng.normal(0.0, nz.lux_noise * ns))))
        base = {
            "tick": t, "ts": t * TICK_S, "t_mono": t * TICK_S, "session_id": session_id,
            "n_people": n_present, "source": "sim", "simulated_fields": [],
            "ambient_db": ambient, "db_p90": p90, "light_lux": lux, "lux_quality": 1.0,
            "room_db_base": base_db, "room_lux_base": base_lux,
            "laughter_count": laughs, "compliments_count": compls, "tod_h": tod,
        }

        # 3) person channels
        person_rows = []
        for person, b in zip(present, behaviors):
            pp = person.params
            video_ok = rng.random() >= nz.dropout
            person.burst = (1.7 if person.burst < 1 else 0.3) if rng.random() < 0.1 else person.burst  # mean x1.0
            fists = int(rng.poisson((b["fist_rate"] * person.burst + FP["fist_per_min"]) / TICKS_PER_MIN))
            person.slump_on = _two_state(rng, person.slump_on, b["slump_p"] + FP["slump_share"])
            # People cover their faces when they laugh: laughter in the room pulls head-in-hands up.
            hih_share = b["hih_p"] + (0.35 if laughs and person.state == "Collaborative_Happy" else 0.0)
            person.hih_on = _two_state(rng, person.hih_on, hih_share, switch=0.35 if laughs else 0.15)
            slump = rng.uniform(0.7, 1.0) if person.slump_on else (rng.uniform(0.0, 0.1) if rng.random() < 0.15 else 0.0)
            hih = person.hih_on * rng.uniform(0.5, 1.0)
            person.still = _two_state(rng, person.still, nz.still_share, switch=nz.still_switch)
            mult = (0.7 if person.slump_on else 1.0) * (nz.still_motion_mult if person.still else 1.0)
            target = math.log(max(b["motion"] * mult, 1e-4))
            if person.log_motion is None:
                person.log_motion = target
            person.log_motion += nz.motion_ar * (target - person.log_motion) + rng.normal(0.0, nz.motion_noise * ns)
            motion = float(math.exp(person.log_motion))
            if person.episode_hidden is not None:  # the episode's share already is "hands over the face"
                hidden = person.episode_hidden
            else:
                hidden = min(0.98, face_hidden + 0.3 * hih)
            face_ok = video_ok and rng.random() >= hidden
            if face_ok:
                low_light = max(0.0, (nz.low_light_lux - lux) / nz.low_light_lux) * 0.35
                face = float(np.clip(rng.normal(person.episode_face, nz.face_tick_sd * ns + low_light), -1, 1))
                face_vis = float(rng.uniform(*nz.face_visible_range))
            else:
                face, face_vis = math.nan, 0.0
            if not video_ok:
                fists, slump, hih, motion = 0, math.nan, math.nan, math.nan
            cues = person.face_cues(face_ok, face, face_vis, lux)
            pid = people.index(person) + 1
            person_rows.append({
                **base,
                "person_id": pid, "display_id": f"Person #{pid}",
                "clenched_fists_count": fists, "posture_slump": slump, "head_in_hands_frac": hih,
                "social_withdrawal_flag": withdrawal_flag(slump, motion), "motion_energy": motion,
                "face_valence": face, "face_visible": face_vis, "hours_since_break": person.hours,
                "true_state": person.state, "episode_age_s": person.episode_age * TICK_S,
                **cues, "true_valence": person.true_v, "true_arousal": person.true_a,
            })

        # 4) room row: aggregates of present people (used by the room slot and room-only mode)
        def agg(key: str, fn) -> float:
            vals = [r[key] for r in person_rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
            return float(fn(vals)) if vals else math.nan

        labels = [r["true_state"] for r in person_rows]
        room_state = max(set(labels), key=labels.count) if labels else "Empty"
        rows.append({
            **base,
            "person_id": ROOM_ID, "display_id": "Room",
            "clenched_fists_count": int(sum(r["clenched_fists_count"] for r in person_rows)),
            "posture_slump": agg("posture_slump", np.mean), "head_in_hands_frac": agg("head_in_hands_frac", np.mean),
            "social_withdrawal_flag": int(any(r["social_withdrawal_flag"] for r in person_rows)),
            "motion_energy": agg("motion_energy", np.mean), "face_valence": agg("face_valence", np.median),
            "face_visible": agg("face_visible", np.mean) if person_rows else 0.0,
            "hours_since_break": agg("hours_since_break", np.mean),
            "true_state": room_state, "episode_age_s": math.nan,
            **{k: math.nan for k in FACE_CUE_FIELDS}, "blink_count": 0, "true_valence": math.nan,
            "true_arousal": math.nan,
        })
        rows.extend(person_rows)

    df = pd.DataFrame(rows)
    return df[list(TICK_FIELDS) + HIDDEN_COLUMNS]


def simulate_corpus(n_sessions: int, seed: int, *, state_set: str = "core", noise_scale: float = 1.0) -> pd.DataFrame:
    """Many semi-Markov sessions. Deterministic for a given (n_sessions, seed, state_set, noise_scale)."""
    root = np.random.SeedSequence(seed)
    frames = []
    for i, child in enumerate(root.spawn(n_sessions)):
        frames.append(simulate_session(f"s{seed}-{i:04d}", np.random.default_rng(child),
                                       state_set=state_set, noise_scale=noise_scale))
    return pd.concat(frames, ignore_index=True)


def to_legacy_csv(df: pd.DataFrame, path: str) -> None:
    """Write person rows in the original mock dataset's 9-column format."""
    people = df[df["person_id"] != ROOM_ID]
    legacy = pd.DataFrame({
        "timestamp": people["ts"], "ambient_db": people["ambient_db"], "light_lux": people["light_lux"],
        "hours_since_break": people["hours_since_break"], "clenched_fists_count": people["clenched_fists_count"],
        "compliments_count": people["compliments_count"], "social_withdrawal_flag": people["social_withdrawal_flag"],
        "facial_valence_snapshot": people["face_valence"], "true_emotional_state": people["true_state"],
    })
    legacy.to_csv(path, index=False)


__all__ = ["Script", "Segment", "simulate_session", "simulate_corpus", "to_legacy_csv", "CORE_STATES", "LEGACY_COLUMNS"]
