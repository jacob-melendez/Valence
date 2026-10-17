"""Replay features-only JSONL recordings (``data/real/*.jsonl``, written by ``valence.edge.run --jsonl``).

Two ways to use a recording:
* ``JsonlFeeder`` pushes observations into a running runtime with their original timing (scaled by
  ``speed``): the demo fallback when the camera or mic fails on stage, tagged ``source="replay"``.
* ``replay_offline`` drives the runtime deterministically without threads (tick boundaries come from
  the observation timestamps): used for the negative-control check and acted-set scoring.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

from valence.contracts import Obs, obs_from_dict

RECORDINGS = Path(__file__).resolve().parents[2] / "data" / "real"


def iter_obs(path: str | Path) -> Iterator[Obs]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield obs_from_dict(json.loads(line))


def list_recordings(directory: Path = RECORDINGS) -> list[dict]:
    return [{"name": p.stem, "bytes": p.stat().st_size} for p in sorted(directory.glob("*.jsonl"))]


class JsonlFeeder:
    def __init__(self, path: str | Path, push: Callable[[Obs], None], speed: float = 1.0, loop: bool = False) -> None:
        self.path = Path(path)
        self.push = push
        self.speed = speed
        self.loop = loop
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jsonl-feeder", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            prev_t = None
            for obs in iter_obs(self.path):
                if self._stop.is_set():
                    return
                if prev_t is not None and obs.t > prev_t:
                    if self._stop.wait((obs.t - prev_t) / self.speed):
                        return
                prev_t = obs.t
                # Re-stamp wall time so break clocks and time of day follow the replay, not the recording.
                self.push(obs)
            if not self.loop:
                return


def replay_offline(runtime, path: str | Path, tick_s: float = 5.0) -> list[dict]:
    """Feed a recording through ``runtime`` (which must use its LiveSource) and step once per tick."""
    emitted: list[dict] = []
    next_boundary = None
    for obs in iter_obs(path):
        if next_boundary is None:
            next_boundary = obs.t + tick_s
        while obs.t >= next_boundary:
            emitted += runtime.step()
            next_boundary += tick_s
        runtime.push_obs(obs)
    emitted += runtime.step()
    return emitted


__all__ = ["iter_obs", "list_recordings", "JsonlFeeder", "replay_offline", "time"]
