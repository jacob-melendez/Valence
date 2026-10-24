"""Pieces shared by the runtime golden generators in this folder (see README.md).

Importing this module puts the repo root on sys.path and pins the process to TZ=UTC, so the generators give the
same output from any working directory and on any machine.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TZ"] = "UTC"
time.tzset()

from pydantic import BaseModel  # noqa: E402

from valence.runtime.bus import EventBus  # noqa: E402

FIXTURES = ROOT / "frontend" / "src" / "test" / "fixtures" / "engine"
ARTIFACTS = ROOT / "artifacts"


def canon(v: Any) -> str:
    """The canonical text the TS test rebuilds: numbers as repr(float(x)), strings as json.dumps, keys in order."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v))
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{json.dumps(k)}:{canon(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


class Clock:
    """Stands in for the ``time`` module of a valence module: settable time(), monotonic(), perf_counter()."""

    def __init__(self, wall: float = 0.0, mono: float = 0.0) -> None:
        self.wall = wall
        self.mono = mono
        self.perf = 0.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def perf_counter(self) -> float:
        return self.perf


class Gate:
    """PurgeGate stand-in whose counters are set by hand (``gate.c.update({...})``)."""

    def __init__(self) -> None:
        self.c = {"video_purged": 0, "audio_purged": 0, "transcripts_purged": 0, "violations": 0,
                  "live_raw_buffers": 0}

    def _s(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(purged={"video": self.c["video_purged"], "audio": self.c["audio_purged"]},
                                     transcripts_purged=self.c["transcripts_purged"],
                                     violations=self.c["violations"], live_raw_buffers=self.c["live_raw_buffers"])

    def end_tick(self) -> types.SimpleNamespace:
        return self._s()

    def stats(self) -> types.SimpleNamespace:
        return self._s()


class Bus(EventBus):
    """EventBus that only keeps model_dump() of every published message."""

    def __init__(self) -> None:
        super().__init__()
        self.dumps: list[dict[str, Any]] = []

    def publish(self, msg: Any) -> None:
        self.dumps.append(msg.model_dump() if isinstance(msg, BaseModel) else dict(msg))


def out_dir(path: str | None) -> Path | None:
    """--out DIR, created if missing (None when the option was not given)."""
    if path is None:
        return None
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d
