"""Load scripted replay scenarios from ``scenarios/*.yaml`` into simulator ``Script`` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from valence.sim.generator import Script, Segment, simulate_session

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def load_script(name: str, directory: Path = SCENARIO_DIR) -> Script:
    data = yaml.safe_load((directory / f"{name}.yaml").read_text(encoding="utf-8"))
    people = int(data.get("people", 1))
    segments = []
    for seg in data["segments"]:
        states = seg.get("states", seg.get("state"))
        if not isinstance(states, list):
            states = [states] * people
        segments.append(Segment(minutes=float(seg["minutes"]), states=states, overrides=seg.get("overrides", {})))
    h0 = data.get("start_hours_since_break")
    if h0 is not None and not isinstance(h0, list):
        h0 = [float(h0)] * people
    return Script(
        name=name, people=people, segments=segments, seed=int(data.get("seed", 0)),
        description=data.get("description", ""), start_hour=data.get("start_hour"),
        start_hours_since_break=h0, lux_base=data.get("lux_base"), mic_offset_db=data.get("mic_offset_db", 0.0),
        face_hidden=data.get("face_hidden"), venue_db=data.get("venue_db"),
    )


def list_scenarios(directory: Path = SCENARIO_DIR) -> list[dict[str, str]]:
    out = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        out.append({"name": path.stem, "description": data.get("description", ""),
                    "minutes": sum(float(s["minutes"]) for s in data["segments"])})
    return out


def render(name: str, *, state_set: str = "core", directory: Path = SCENARIO_DIR) -> pd.DataFrame:
    script = load_script(name, directory)
    rng = np.random.default_rng(script.seed)
    df = simulate_session(f"scenario-{name}", rng, state_set=state_set, script=script)
    df["source"] = "replay"
    return df
