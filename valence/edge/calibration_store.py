"""Writes calibration constants (dB SPL offset, lux fit) to ``data/real/calibration.json``.

Lives in ``valence/edge`` because ``valence/perception`` never writes files. Text-mode JSON of
scalar constants only; reading them is ``valence.perception.calibration.load``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from typing import Any

from valence.perception import calibration
from valence.perception.calibration import _check_constants, _merge, _resolve, load


def save(updates: Mapping[str, Any], path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Deep-merge ``updates`` into the stored constants and write atomically; returns the result."""
    _check_constants(updates)
    target = _resolve(path) if path is not None else calibration.CALIBRATION_PATH
    merged = _merge(load(target), updates)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return merged
