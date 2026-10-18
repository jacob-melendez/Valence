"""Sensor units, tick channels and the rules shared by the simulator and the live aggregator.

Anything that must mean the same thing in simulation, replay and live lives here, so the
simulator and the runtime cannot drift apart.
"""

from __future__ import annotations

import math

# Units of every per-tick channel. The simulator generates values in these units and the live
# aggregator produces them from the edge observations (see docs/CONTRACT_RESPONSES.md).
SENSOR_SPEC: dict[str, str] = {
    "ambient_db": "Leq over the tick, approx dB SPL (calibrated mic dBFS + offset)",
    "db_p90": "90th percentile of 1 s block levels in the tick, dB SPL",
    "light_lux": "estimated lux (10 s median of the webcam regression, or simulated)",
    "laughter_count": "near-field laughter onsets in the tick (room scope)",
    "compliments_count": "near-field compliment phrase hits in the tick (room scope)",
    "clenched_fists_count": "debounced fist onsets in the tick (person scope)",
    "posture_slump": "fraction of the tick's frames with a debounced slump [0, 1]",
    "head_in_hands_frac": "fraction of the tick's frames with head-in-hands [0, 1]",
    "motion_energy": "mean keypoint displacement, shoulder-widths per second",
    "face_valence": "median blendshape valence relative to the person's neutral, [-1, 1]",
    "face_visible": "fraction of the tick's frames with a readable face [0, 1]",
    "hours_since_break": "hours since the person's last break (presence-based clock)",
    "tod_h": "local time of day in hours [0, 24)",
}

# Raw per-tick channels the context models see (face is a separate block).
ROOM_CHANNELS: tuple[str, ...] = ("ambient_db", "light_lux", "laughter_count", "compliments_count")
PERSON_CHANNELS: tuple[str, ...] = (
    "motion_energy",
    "posture_slump",
    "head_in_hands_frac",
    "clenched_fists_count",
    "social_withdrawal_flag",
    "n_people",
)
TIME_CHANNELS: tuple[str, ...] = ("hours_since_break", "tod_h")
FACE_CHANNELS: tuple[str, ...] = ("face_valence", "face_visible")

# Withdrawal: mostly slumped and nearly still for the tick. Same rule in sim and live.
WITHDRAW_SLUMP_MIN = 0.6
WITHDRAW_MOTION_MAX = 0.02  # shoulder-widths/s; live capture: still ~0.014, typing ~0.03


def withdrawal_flag(posture_slump: float, motion_energy: float) -> int:
    if math.isnan(posture_slump) or math.isnan(motion_energy):
        return 0
    return int(posture_slump >= WITHDRAW_SLUMP_MIN and motion_energy <= WITHDRAW_MOTION_MAX)


# --- Feature names -> explanation layer / scope -------------------------------------------------
# Feature names follow "<channel>_<stat>_<window>" (e.g. db_mean_15m, fist_cnt_5m, face_now).
ROOM_PREFIXES = ("db_", "lux_", "laugh_", "compl_")
TIME_FEATURES = ("hours_since_break", "tod_sin", "tod_cos")
NEVER_NARRATED = ("fill_ratio",)


def layer_of(feature: str) -> str:
    if feature.startswith(("db_", "lux_")):
        return "Ambient Physics"
    if feature in TIME_FEATURES:
        return "Time Dynamics"
    if feature.startswith("face_"):
        return "Facial Surface"
    return "Somatic & Social"


def scope_of(feature: str) -> str:
    return "room" if feature.startswith(ROOM_PREFIXES) or feature == "n_people" else "person"
