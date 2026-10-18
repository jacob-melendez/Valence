"""Replay features-only JSONL recordings (``data/real/*.jsonl``, written by ``valence.edge.run --jsonl``).

Two ways to use a recording:
* ``JsonlFeeder`` pushes observations into a running runtime with their original timing (scaled by
  ``speed``): the demo fallback when the camera or mic fails on stage, tagged ``source="replay"``.
* ``replay_offline`` drives the runtime deterministically without threads (tick boundaries come from
  the observation timestamps): used for the negative-control check and acted-set scoring.
"""

from __future__ import annotations

import dataclasses
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
        # Join so no stale recorded frame is pushed after the next source re-aligns the frame counters.
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

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
                # Re-stamp both clocks so break clocks and time of day follow the replay, not the recording.
                self.push(restamp(obs, time.time(), time.monotonic()))
            if not self.loop:
                return


def restamp(obs: Obs, wall: float, t: float) -> Obs:
    """The same observation on another clock (observations are frozen)."""
    return dataclasses.replace(obs, wall=wall, t=t)


def replay_offline(runtime, path: str | Path, tick_s: float = 5.0) -> list[dict]:
    """Feed a recording through ``runtime`` (which must use its LiveSource) and step once per tick.

    The recording is mapped onto one synthetic clock starting now, so observations and ticks agree:
    the break clock starts when the replayed person first appears instead of when the capture was made.
    """
    emitted: list[dict] = []
    wall0, mono0 = time.time(), time.monotonic()
    t0 = next_boundary = None
    for obs in iter_obs(path):
        if t0 is None:
            t0, next_boundary = obs.t, obs.t + tick_s
        while obs.t >= next_boundary:
            emitted += runtime.step(wall=wall0 + next_boundary - t0, mono=mono0 + next_boundary - t0)
            next_boundary += tick_s
        runtime.push_obs(restamp(obs, wall0 + obs.t - t0, mono0 + obs.t - t0))
    if next_boundary is not None:
        emitted += runtime.step(wall=wall0 + next_boundary - t0, mono=mono0 + next_boundary - t0)
    return emitted


__all__ = ["iter_obs", "list_recordings", "JsonlFeeder", "replay_offline", "restamp", "time"]
