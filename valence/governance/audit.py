"""Hash-chained, features-free audit log.

Records *decisions*, not data: labels, confidences, SHAP layer summaries, purge counters and
operator actions (seeding, simulated inputs, calibration). Each entry carries the SHA-256 of the
previous entry, so any edit or deletion breaks ``verify()``.

Retention: one file per session under ``var/audit/``. The file is deleted when the session closes
(``close(delete=True)``), and files older than ``ttl_s`` are removed at startup, so a crash cannot
leave behavioral timelines on disk indefinitely.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
MAX_STR = 200  # long strings are refused so free text (e.g. a transcript) can never be logged
MAX_DEPTH = 4


class AuditPayloadError(ValueError):
    pass


def _check(value: Any, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise AuditPayloadError("payload nested too deeply")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise AuditPayloadError("NaN/inf not allowed; use null")
        return
    if isinstance(value, str):
        if len(value) > MAX_STR:
            raise AuditPayloadError(f"string longer than {MAX_STR} chars refused")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise AuditPayloadError("keys must be strings")
            _check(v, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise AuditPayloadError("lists longer than 64 items refused")
        for v in value:
            _check(v, depth + 1)
        return
    raise AuditPayloadError(f"type {type(value).__name__} not allowed in audit payloads")


def _digest(prev: str, body: dict[str, Any]) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev + blob).encode()).hexdigest()


class AuditLog:
    def __init__(self, directory: str | Path, session_id: str, ttl_s: int = 86_400) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.ttl_s = ttl_s
        self.path = self.directory / f"{session_id}.jsonl"
        self._lock = threading.Lock()
        self._seq = 0
        self._last = GENESIS
        self.expire_old()

    @property
    def last_hash(self) -> str:
        return self._last

    def append(self, kind: str, payload: dict[str, Any]) -> str:
        _check(kind)
        _check(payload)
        with self._lock:
            body = {"seq": self._seq, "ts": round(time.time(), 3), "kind": kind, "payload": payload}
            h = _digest(self._last, body)
            entry = {**body, "prev": self._last, "hash": h}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
            self._seq += 1
            self._last = h
            return h

    def entries(self, since_seq: int = 0) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r for r in rows if r["seq"] >= since_seq]

    def verify(self) -> tuple[bool, int]:
        """Recompute the chain. Returns (ok, number of entries checked)."""
        prev = GENESIS
        n = 0
        for n, entry in enumerate(self.entries(), start=1):
            body = {k: entry[k] for k in ("seq", "ts", "kind", "payload")}
            if entry["prev"] != prev or _digest(prev, body) != entry["hash"]:
                return False, n
            prev = entry["hash"]
        return True, n

    def expire_old(self) -> int:
        """Delete audit files older than the TTL. Returns how many were removed."""
        cutoff = time.time() - self.ttl_s
        removed = 0
        for p in self.directory.glob("*.jsonl"):
            if p != self.path and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        return removed

    def close(self, delete: bool = True) -> None:
        if delete:
            self.path.unlink(missing_ok=True)
