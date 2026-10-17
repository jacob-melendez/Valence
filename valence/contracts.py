"""Shared contracts for Valence.

Single source of truth used by every part of the system:

* LOCAL session (Mac): perception emits ``VideoObs`` / ``AudioObs`` through a ``Sink`` and
  implements ``valence.edge.run.start``; the narrator consumes ``FactSheet`` and returns
  ``NarrationResult``.
* CLOUD session: runtime aggregates observations into ``TickRecord`` rows, models predict,
  the explainer builds ``FactSheet`` objects and the API streams the WebSocket messages below.

Change this file only through ``docs/CONTRACT_REQUESTS.md``. Additive changes (new optional
fields) are safe; renames and removals are not.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Emotional states
# ---------------------------------------------------------------------------

CORE_STATES: tuple[str, ...] = ("Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy")
P1_STATES: tuple[str, ...] = ("Overwhelmed", "Tense")
ALL_STATES: tuple[str, ...] = CORE_STATES + P1_STATES
STATE_SETS: dict[str, tuple[str, ...]] = {"core": CORE_STATES, "all": ALL_STATES}

STATE_TIER: dict[str, str] = {**{s: "P0" for s in CORE_STATES}, **{s: "P1" for s in P1_STATES}}

STATE_DISPLAY: dict[str, str] = {
    "Focused": "Focused",
    "Frustrated_Masked": "Frustrated / Masked Stress",
    "Exhausted": "Exhausted",
    "Collaborative_Happy": "Collaborative / Happy",
    "Overwhelmed": "Overwhelmed / Distressed",
    "Tense": "Tense / Visible Frustration",
}

# Word stems the narrator validator uses to check that a text names a state. Matching is
# case-insensitive substring matching. "visibly frustrat" (Tense) contains "frustrat"
# (Frustrated_Masked), so validators must test Tense's stems before Frustrated_Masked's.
STATE_LEXICON: dict[str, tuple[str, ...]] = {
    "Focused": ("focus", "concentrat"),
    "Frustrated_Masked": ("frustrat", "masked stress"),
    "Exhausted": ("exhaust", "fatigu"),
    "Collaborative_Happy": ("collaborat", "upbeat", "happy"),
    "Overwhelmed": ("overwhelm", "distress"),
    "Tense": ("tense", "visibly frustrat"),
}


def display_label(
    label: str,
    *,
    mask_override: bool = False,
    n_people: int = 2,
    states: tuple[str, ...] = CORE_STATES,
) -> str:
    """Human-readable label for a prediction.

    With only the core states, Frustrated_Masked also covers unmasked frustration, so it reads
    "Frustrated" unless the mask-override detector fired. Once Tense is in the state set,
    Frustrated_Masked is mask-only and always shows its full name. A lone person is never
    called "Collaborative".
    """
    if label == "Frustrated_Masked" and "Tense" not in states and not mask_override:
        return "Frustrated"
    if label == "Collaborative_Happy" and n_people < 2:
        return "Upbeat"
    return STATE_DISPLAY[label]


# ---------------------------------------------------------------------------
# Explanation layers (PRD's three context layers + the face surface)
# ---------------------------------------------------------------------------

LAYERS: tuple[str, ...] = ("Ambient Physics", "Time Dynamics", "Somatic & Social", "Facial Surface")
LayerName = Literal["Ambient Physics", "Time Dynamics", "Somatic & Social", "Facial Surface"]

# ---------------------------------------------------------------------------
# Perception -> runtime observations (LOCAL emits, CLOUD consumes)
# Only scalars, booleans, flags and normalized boxes. Never pixels, audio or keypoints.
# ---------------------------------------------------------------------------

NAN = float("nan")


def _check_unit_interval(name: str, value: float) -> None:
    if not (-1e-6 <= value <= 1 + 1e-6):
        raise ValueError(f"{name}={value} is outside [0, 1]")


@dataclass(frozen=True, slots=True)
class PersonObs:
    """One tracked person in one processed video frame (~10 Hz)."""

    track_id: int  # ephemeral display id 1..N ("Person #N"); never linked to identity
    bbox: tuple[float, float, float, float]  # normalized x1, y1, x2, y2 in [0, 1]
    fist_onsets: int = 0  # debounced fist rising edges in this frame
    fist_active: bool = False
    slump: bool = False  # debounced posture slump
    head_in_hands: bool = False
    motion: float = NAN  # EMA of mean keypoint displacement, shoulder-widths/s; NaN when undefined
    face_visible: bool = False
    face_valence: float = NAN  # [-1, 1] relative to the person's neutral baseline; NaN when not visible

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (x1, y1, x2, y2)")
        for v in self.bbox:
            _check_unit_interval("bbox", v)
        if not math.isnan(self.face_valence) and not (-1 - 1e-6 <= self.face_valence <= 1 + 1e-6):
            raise ValueError(f"face_valence={self.face_valence} is outside [-1, 1]")


@dataclass(frozen=True, slots=True)
class VideoObs:
    """Result of processing one video frame. ``t`` is time.monotonic(), ``wall`` is time.time()."""

    t: float
    wall: float
    frame_seq: int
    frames_decoded: int  # cumulative frames retrieved from the camera; must equal frames purged
    lux: float = NAN  # estimated lux; NaN if not updated this frame
    lux_quality: float = 0.0  # 0..1
    frame_w: int = 0
    frame_h: int = 0
    mirrored: bool = False
    persons: tuple[PersonObs, ...] = ()


@dataclass(frozen=True, slots=True)
class AudioObs:
    """Result of one ~1 s audio analysis step. Transcript text never leaves perception."""

    t: float
    wall: float
    block_seq: int
    db_spl: float  # calibrated approximate dB SPL (Leq over the step)
    rms_power: float  # linear mean(x**2) over the step, for power-domain averaging
    speech_prob: float = 0.0
    laughter_onsets: int = 0  # debounced laughter rising edges
    compliment_hits: int = 0  # lexicon phrase matches


Obs = Union[VideoObs, AudioObs]
Sink = Callable[[Obs], None]


class EdgeStart(Protocol):
    """Signature of ``valence.edge.run.start`` (LOCAL). Returns a ``stop()`` callable."""

    def __call__(
        self, sink: Sink, *, video: bool = True, audio: bool = True, obscure_face: bool = False
    ) -> Callable[[], None]: ...


def _nan_to_none(v: Any) -> Any:
    return None if isinstance(v, float) and math.isnan(v) else v


def _none_to_nan(v: Any) -> Any:
    return NAN if v is None else v


def obs_to_dict(obs: Obs) -> dict[str, Any]:
    """JSON-safe dict for features-only JSONL recordings: ``{"type": "video"|"audio", ...}``, NaN -> null."""
    if isinstance(obs, VideoObs):
        d = {f.name: _nan_to_none(getattr(obs, f.name)) for f in fields(obs) if f.name != "persons"}
        d["persons"] = [{k: _nan_to_none(v) for k, v in asdict(p).items()} for p in obs.persons]
        return {"type": "video", **d}
    return {"type": "audio", **{k: _nan_to_none(v) for k, v in asdict(obs).items()}}


def obs_from_dict(d: dict[str, Any]) -> Obs:
    """Inverse of ``obs_to_dict``. Also accepts the older ``"kind"`` key."""
    d = dict(d)
    kind = d.pop("type", None) or d.pop("kind")
    if kind == "video":
        persons = tuple(
            PersonObs(
                **{
                    **{k: _none_to_nan(v) for k, v in p.items()},
                    "bbox": tuple(p["bbox"]),
                }
            )
            for p in d.pop("persons", [])
        )
        return VideoObs(**{k: _none_to_nan(v) for k, v in d.items()}, persons=persons)
    if kind == "audio":
        return AudioObs(**{k: _none_to_nan(v) for k, v in d.items()})
    raise ValueError(f"unknown observation kind {kind!r}")


# ---------------------------------------------------------------------------
# Per-tick record (5 s, one per person). Identical for simulator, replay and live.
# Room-level fields (dB, lux, laughter, compliments, time) are copied into each person's row.
# ---------------------------------------------------------------------------

Source = Literal["live", "replay", "sim", "sim_seed"]


@dataclass(slots=True)
class TickRecord:
    tick: int
    ts: float  # wall clock
    t_mono: float
    session_id: str
    person_id: int
    display_id: str  # "Person #N"
    n_people: int
    source: str = "live"
    simulated_fields: list[str] = field(default_factory=list)
    # Ambient Physics (room)
    ambient_db: float = NAN
    db_p90: float = NAN
    light_lux: float = NAN
    lux_quality: float = 0.0
    # Somatic & Social
    laughter_count: int = 0  # room, near-field gated
    compliments_count: int = 0  # room, near-field gated
    clenched_fists_count: int = 0
    posture_slump: float = NAN  # fraction of the tick slumped
    head_in_hands_frac: float = NAN
    social_withdrawal_flag: int = 0
    motion_energy: float = NAN
    # Facial Surface
    face_valence: float = NAN  # median over the tick; NaN when never visible
    face_visible: float = 0.0  # fraction of the tick with a visible face
    # Time Dynamics
    hours_since_break: float = NAN
    tod_h: float = NAN  # local time of day in hours [0, 24)

    def to_dict(self) -> dict[str, Any]:
        return {k: _nan_to_none(v) for k, v in asdict(self).items()}


TICK_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TickRecord))

# ---------------------------------------------------------------------------
# Narrator contract (CLOUD -> POST http://127.0.0.1:8081/narrate -> LOCAL)
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Driver(_Model):
    feature: str
    display: str  # pre-formatted, e.g. "average noise of 72 dB over 15 minutes"
    value: float | None
    unit: str
    shap: float
    level: Literal["high", "elevated", "normal", "low"]
    effect: Literal["supports", "opposes"]
    scope: Literal["room", "person"]


class Layer(_Model):
    name: LayerName
    contribution: float
    drivers: list[Driver]


class FaceOnly(_Model):
    """Face-only baseline (M0). ``label``/``confidence`` are null when no face was visible."""

    label: str | None
    confidence: float | None
    face_valence: float | None = None


class DataQuality(_Model):
    face_visible_frac: float
    lux: float | None = None
    ood_features: list[str] = Field(default_factory=list)


class FactSheet(_Model):
    event_id: str
    person: str  # "Person #1"
    label: str
    label_display: str
    confidence: float  # calibrated, unsmoothed
    confidence_display: str  # "88%"
    probabilities: dict[str, float]
    window_fill_min: float
    n_people: int
    face_only: FaceOnly
    face_visible: bool
    mask_override: bool
    layers: list[Layer]
    data_quality: DataQuality


class NarrationResult(_Model):
    text: str
    source: Literal["llm", "llm_retry", "template"]
    latency_ms: int
    attempts: int = 1


# ---------------------------------------------------------------------------
# WebSocket messages (/ws/events). Consumed by the frontend; see docs/API.md.
# ---------------------------------------------------------------------------


class StateInfo(_Model):
    id: str
    display: str
    tier: str


class Hello(_Model):
    type: Literal["hello"] = "hello"
    contract_version: str = CONTRACT_VERSION
    mode: Literal["live", "replay", "hybrid"]
    room_only: bool = False
    states: list[StateInfo]
    feature_version: str
    frame_w: int = 0
    frame_h: int = 0
    mirrored: bool = False


class TrackPerson(_Model):
    id: int
    label: str  # "Person #1"
    bbox: tuple[float, float, float, float]  # normalized x1, y1, x2, y2
    fist_active: bool = False
    slump: bool = False


class Tracks(_Model):
    type: Literal["tracks"] = "tracks"
    t: float
    capture_ts_ms: int
    frame_w: int
    frame_h: int
    mirrored: bool
    persons: list[TrackPerson]


GOVERNANCE_LINE = "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline."


class Governance(_Model):
    type: Literal["governance"] = "governance"
    t: float
    tick: int
    message: str = GOVERNANCE_LINE
    frames_decoded: int
    frames_purged: int
    audio_blocks_purged: int
    transcripts_purged: int
    violations: int
    live_raw_buffers: int
    audit_hash: str | None = None
    retention_s: int


class StateVector(_Model):
    type: Literal["state_vector"] = "state_vector"
    t: float
    person_id: int
    ambient_db: float | None
    light_lux: float | None
    hours_since_break: float | None
    time_of_day: str  # "15:40"
    clenched_fists_5m: int
    compliments_15m: int
    laughter_15m: int
    posture_slump: float | None
    head_in_hands: float | None
    social_withdrawal: int
    face_visible: bool
    facial_valence: float | None
    window_fill_min: float
    source: str = "live"
    simulated_fields: list[str] = Field(default_factory=list)


class InlineExplanation(_Model):
    text: str
    source: Literal["llm", "llm_retry", "template"] = "template"


class FaceOnlyBrief(_Model):
    label: str | None  # null when no face was visible
    confidence: float | None


class Prediction(_Model):
    type: Literal["prediction"] = "prediction"
    event_id: str
    t: float
    person_id: int
    label: str
    label_display: str
    confidence: float  # sim-calibrated, unsmoothed
    probabilities: dict[str, float]
    face_only: FaceOnlyBrief
    mask_override: bool
    warming_up: bool
    layers: list[Layer]
    data_quality: DataQuality
    explanation: InlineExplanation
    source: str = "live"
    simulated_fields: list[str] = Field(default_factory=list)


class Explanation(_Model):
    type: Literal["explanation"] = "explanation"
    event_id: str
    person_id: int
    text: str
    source: Literal["llm", "llm_retry", "template"]
    latency_ms: int


class Status(_Model):
    type: Literal["status"] = "status"
    mode: Literal["live", "replay", "hybrid"]
    camera: bool
    mic: bool
    fps: float
    loop_latency_ms: float
    narrator: Literal["ready", "fallback", "down"]
    offline: bool
    uptime_s: float
    degraded: list[str] = Field(default_factory=list)
    purge_violation: bool = False


class Heartbeat(_Model):
    type: Literal["heartbeat"] = "heartbeat"
    t: float


WS_MESSAGES: tuple[type[BaseModel], ...] = (
    Hello,
    Tracks,
    Governance,
    StateVector,
    Prediction,
    Explanation,
    Status,
    Heartbeat,
)


def json_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema for every WebSocket message plus the narrator contract."""
    models = (*WS_MESSAGES, FactSheet, NarrationResult)
    return {m.__name__: m.model_json_schema() for m in models}
