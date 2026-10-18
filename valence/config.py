"""Runtime configuration. Every value can be overridden with a VALENCE_* environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Settings:
    # Rolling state vector
    tick_s: float = 5.0
    window_ticks: int = 180  # 15 minutes at 5 s
    perception_hz: float = 10.0
    warmup_s: float = 60.0
    override_block_s: float = 120.0
    # People / breaks
    max_tracks: int = 3
    min_box_height: float = 0.25  # fraction of frame height
    participant_rel_height: float = 0.55  # a track counts as a participant if its box is >= this x the largest
    participant_min_presence: float = 0.3  # ... and it is present in >= this share of the tick's frames
    break_absent_min: float = 10.0  # absence that counts as a break (set ~0.5 for demos)
    person_drop_min: float = 15.0
    # Audio
    db_cal_offset: float = 94.0  # dBFS -> approx dB SPL; replaced by calibration
    near_field_db: float = 6.0  # laughter/compliments count only this far above the room median
    # Model / states
    state_set: str = "core"  # "core" (4 states) or "all" (6 states)
    artifacts_dir: str = "artifacts"
    # Services
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    narrator_url: str = "http://127.0.0.1:8081"
    narrator_deadline_s: float = 2.5
    mode: str = "replay"  # live | replay | hybrid
    room_only: bool = False
    # Governance
    audit_dir: str = "var/audit"
    audit_ttl_s: int = 86_400

    @classmethod
    def from_env(cls) -> "Settings":
        overrides = {}
        for f in fields(cls):
            raw = os.environ.get(f"VALENCE_{f.name.upper()}")
            if raw is None:
                continue
            if f.type in ("bool", bool):
                overrides[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif f.type in ("int", int):
                overrides[f.name] = int(raw)
            elif f.type in ("float", float):
                overrides[f.name] = float(raw)
            else:
                overrides[f.name] = raw
        return cls(**overrides)


settings = Settings.from_env()
