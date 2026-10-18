"""Thread-safe 15-minute rolling window of TickRecords, one slot per person plus the room slot.

The room slot (``person_id == 0``) is appended every tick, whoever is present, so a person who sits
down inherits 15 minutes of room history (noise, light, laughter) and only their own body and face
channels start cold.
"""

from __future__ import annotations

import threading
from collections import deque

from valence.contracts import TickRecord

ROOM = 0


class RollingWindow:
    def __init__(self, max_ticks: int = 180) -> None:
        self.max_ticks = max_ticks
        self._lock = threading.Lock()
        self._slots: dict[int, deque[TickRecord]] = {}

    def append(self, rec: TickRecord) -> None:
        with self._lock:
            slot = self._slots.get(rec.person_id)
            if slot is None:
                slot = self._slots[rec.person_id] = deque(maxlen=self.max_ticks)
            slot.append(rec)

    def snapshot(self, person_id: int) -> list[TickRecord]:
        with self._lock:
            return list(self._slots.get(person_id, ()))

    def room(self) -> list[TickRecord]:
        return self.snapshot(ROOM)

    def people(self) -> list[int]:
        with self._lock:
            return sorted(pid for pid in self._slots if pid != ROOM)

    def drop(self, person_id: int) -> None:
        with self._lock:
            self._slots.pop(person_id, None)

    def reset_people(self) -> None:
        """Forget every person (between judge groups); keep the room history."""
        with self._lock:
            for pid in [p for p in self._slots if p != ROOM]:
                del self._slots[pid]

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()

    def fill_minutes(self, person_id: int, tick_s: float = 5.0) -> float:
        with self._lock:
            return len(self._slots.get(person_id, ())) * tick_s / 60.0
