"""Behavioral assumptions of the simulator, in one table.

These numbers ARE the hypothesis the evaluation tests under; they are published as-is in
docs/DATASET.md and MODEL_CARD.md. Detector false-positive rates (``FP``) are placeholders until the
live feature capture measures them on the demo Mac.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StateParams:
    face_mu: float  # episode-mean facial valence
    db_target: float  # room level this state pulls toward (dB SPL, calibrated mic)
    lux_mult: float  # weak pull on room light (dimmer when stressed/tired)
    fist_rate: float  # clenched-fist onsets per minute (person)
    laugh_rate: float  # laughter onsets per minute contributed to the room
    compl_rate: float  # compliments per minute contributed to the room
    slump_p: float  # long-run share of time slumped
    hih_p: float  # long-run share of time with head in hands
    motion: float  # typical motion, shoulder-widths/s
    face_hidden_extra: float = 0.0  # > 0: each episode draws its own face-hidden share U(0.4, 0.7) (hands over face)
    mask_p: float = 0.0  # probability an episode wears a forced smile instead of its true face
    mask_mu: float = 0.45  # valence of the forced smile
    group_affinity: float = 1.0  # transition weight multiplier when 2+ people are present
    solo_affinity: float = 1.0  # transition weight multiplier when alone
    fatigue_affinity: float = 0.0  # how strongly long stretches without a break pull toward it
    dwell_min: float = 6.0  # mean episode length in minutes


STATE_PARAMS: dict[str, StateParams] = {
    "Focused": StateParams(
        face_mu=0.0, db_target=48, lux_mult=1.0, fist_rate=0.0, laugh_rate=0.03, compl_rate=0.02,
        slump_p=0.05, hih_p=0.08, motion=0.03, dwell_min=9.0, fatigue_affinity=-0.6,
    ),
    "Frustrated_Masked": StateParams(
        face_mu=-0.12, db_target=66, lux_mult=0.85, fist_rate=0.8, laugh_rate=0.10, compl_rate=0.02,
        slump_p=0.30, hih_p=0.15, motion=0.05, mask_p=0.70, dwell_min=6.0, fatigue_affinity=1.0,
    ),
    "Exhausted": StateParams(
        face_mu=-0.04, db_target=45, lux_mult=0.65, fist_rate=0.0, laugh_rate=0.02, compl_rate=0.01,
        slump_p=0.70, hih_p=0.35, motion=0.015, dwell_min=8.0, fatigue_affinity=2.0,
    ),
    "Collaborative_Happy": StateParams(
        face_mu=0.50, db_target=63, lux_mult=1.15, fist_rate=0.0, laugh_rate=0.50, compl_rate=0.20,
        slump_p=0.03, hih_p=0.05, motion=0.06, dwell_min=6.0, group_affinity=1.8, solo_affinity=0.4,
        fatigue_affinity=-0.5,
    ),
    # P1 states (enabled with state_set="all")
    "Overwhelmed": StateParams(
        face_mu=-0.20, db_target=66, lux_mult=0.85, fist_rate=0.2, laugh_rate=0.02, compl_rate=0.01,
        slump_p=0.60, hih_p=0.70, motion=0.03, face_hidden_extra=0.55, dwell_min=5.0, fatigue_affinity=1.5,
    ),
    "Tense": StateParams(
        face_mu=-0.35, db_target=70, lux_mult=0.95, fist_rate=0.8, laugh_rate=0.05, compl_rate=0.01,
        slump_p=0.20, hih_p=0.08, motion=0.09, dwell_min=5.0, group_affinity=1.5, solo_affinity=0.6,
        fatigue_affinity=1.0,
    ),
}

# Detector false positives that happen regardless of state (typing read as fists, looking at the
# keyboard read as slump). Live capture run 1 (3 min typing, 3 min still, 15 cued fists): 0 false fist
# onsets and 0% slump; small non-zero rates are kept so the model never treats one event as certain.
FP = {
    "fist_per_min": 0.02,
    "slump_share": 0.02,
    "laugh_per_min": 0.02,
    "compl_per_min": 0.01,
}


def params_for(state: str, state_set: tuple[str, ...]) -> StateParams:
    """With Tense available, Frustrated_Masked is mask-only (forced smile >= 90%)."""
    p = STATE_PARAMS[state]
    if state == "Frustrated_Masked" and "Tense" in state_set:
        return replace(p, mask_p=0.92)
    return p


@dataclass(frozen=True)
class Nuisance:
    """Session-level nuisance ranges; ``noise_scale`` widens every noise term."""

    mic_offset_db: float = 12.0  # per-session uniform(-x, x): device/room differences
    db_noise: float = 1.2  # per-tick dB jitter (live capture: ~0.5-1.0 dB tick-to-tick in a quiet room)
    db_drift: float = 0.5  # slow AR(1) drift of the room level, dB per tick
    lux_noise: float = 0.015  # per-tick multiplicative lux jitter (live: ~1-2% tick-to-tick)
    lux_drift: float = 0.01  # slow AR(1) drift of log-lux per tick
    motion_ar: float = 0.25  # pull of log-motion toward its target per tick (motion is EMA-smoothed live)
    motion_noise: float = 0.18  # log-motion innovation per tick (live typing p90 ~0.045 at p50 0.03)
    # People alternate active and still phases in every state (typing vs reading/listening): stillness alone
    # is not diagnostic (live: a still, upright, alert person moves ~0.014 sw/s, like a tired one).
    still_share: float = 0.4
    still_switch: float = 0.06
    still_motion_mult: float = 0.5
    # State-independent transients seen live: typing starts, doors, speech bursts; webcam auto-exposure jumps.
    sound_events_per_min: float = 0.3
    sound_event_db: tuple[float, float] = (4.0, 12.0)
    sound_event_ticks: tuple[int, int] = (2, 8)
    light_events_per_min: float = 0.5
    light_event_mult: tuple[float, float] = (0.75, 1.3)
    light_event_ticks: tuple[int, int] = (2, 12)
    lux_range: tuple[float, float] = (120.0, 600.0)  # session baseline, log-uniform
    face_hidden_range: tuple[float, float] = (0.0, 0.30)  # per-session share of unreadable ticks
    face_episode_sd: float = 0.12  # between-episode spread of the episode's mean expression
    face_tick_sd: float = 0.03  # live neutral face: tick-to-tick ~0.001-0.01 (baseline-relative median)
    face_visible_range: tuple[float, float] = (0.85, 1.0)  # share of a tick's frames with a readable face
    low_light_lux: float = 200.0  # below this, face readings get noisier
    dropout: float = 0.02  # per-tick, per-modality
    room_tau_min: tuple[float, float] = (2.0, 4.0)  # room dB lag behind occupants' states
    person_tau_min: tuple[float, float] = (1.0, 2.0)  # behavior lag behind a new state
    break_per_min: float = 1 / 120
    break_len_min: tuple[float, float] = (10.0, 20.0)
    session_min: tuple[float, float] = (60.0, 120.0)
    start_hour: tuple[float, float] = (9.0, 23.0)
    start_hours_since_break: tuple[float, float] = (0.0, 5.0)


NUISANCE = Nuisance()
