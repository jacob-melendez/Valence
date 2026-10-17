"""Persisted sensor calibration constants: SPL offset and the lux model.

Stored in ``data/real/calibration.json``. Only scalar constants are accepted, never media
or derived signals, and writes are atomic (temp file in the same directory, fsync, then
``os.replace``), so a crash mid-save leaves the previous file intact.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from valence.perception.lux import LuxModel

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "data" / "real" / "calibration.json"
DEFAULTS: Mapping[str, Any] = {
    "spl_offset_db": 94.0,
    "spl_calibrated": False,
    "lux": LuxModel().to_dict(),
}
_SCALARS = (bool, int, float, str, type(None))


def _resolve(path: str | os.PathLike[str] | None) -> Path:
    return Path(path) if path is not None else Path(CALIBRATION_PATH)


def _merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping):
            out[key] = _merge(out[key] if isinstance(out.get(key), Mapping) else {}, value)
        else:
            out[key] = value
    return out


def _check_constants(obj: Any, where: str = "updates") -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"calibration keys must be str (at {where})")
            _check_constants(value, f"{where}.{key}")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"calibration value at {where} is not finite")
    elif not isinstance(obj, _SCALARS):
        raise TypeError(f"calibration values must be scalar constants, got {type(obj).__name__} at {where}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _conforms(default: Any, value: Any) -> bool:
    """A stored value may replace a default only if it has the same kind (and is finite)."""
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(default, bool) and isinstance(value, bool)
    if isinstance(default, (int, float)):
        return _is_number(value)
    if isinstance(default, str):
        return isinstance(value, str)
    return False


def _sanitize(defaults: Mapping[str, Any], stored: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the stored entries that fit the defaults' shape; drop the rest (and non-scalars)."""
    out: dict[str, Any] = {}
    for key, value in stored.items():
        if not isinstance(key, str):
            continue
        default = defaults.get(key)
        if isinstance(default, Mapping) or isinstance(value, Mapping):
            if isinstance(default, Mapping) and isinstance(value, Mapping):
                out[key] = _sanitize(default, value)
            elif default is None and isinstance(value, Mapping):
                out[key] = _sanitize({}, value)
        elif isinstance(value, _SCALARS) and (default is None or _conforms(default, value)):
            if not isinstance(value, float) or math.isfinite(value):
                out[key] = value
    return out


def load(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Stored constants merged over the defaults. A missing or malformed file gives the
    defaults, and a stored value of the wrong kind (e.g. a string offset) keeps its default."""
    try:
        stored = json.loads(_resolve(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    return _merge(DEFAULTS, _sanitize(DEFAULTS, stored) if isinstance(stored, Mapping) else {})


def save(updates: Mapping[str, Any], path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Deep-merge ``updates`` into the stored constants and write atomically; returns the result."""
    _check_constants(updates)
    target = _resolve(path)
    merged = _merge(load(target), updates)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return merged


def lux_model(cfg: Mapping[str, Any] | None = None) -> LuxModel:
    """The configured lux model (defaults to the uncalibrated one)."""
    return LuxModel.from_dict((cfg if cfg is not None else load())["lux"])


def spl_offset_db(cfg: Mapping[str, Any] | None = None) -> float:
    """dB SPL = dBFS + this offset."""
    return float((cfg if cfg is not None else load())["spl_offset_db"])
