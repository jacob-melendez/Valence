"""Biometric purge gate.

Every raw buffer (video frame, audio block) goes through ``gate.purge(buf, kind)`` immediately
after feature extraction, and the caller then drops its own reference::

    feats = perceive(frame)          # returns scalars / normalized boxes only
    gate.purge(frame, "video")
    del frame

``purge`` overwrites the buffer with zeros in place and registers a weak reference whose
callback fires when the array is actually freed. At the end of every tick the runtime calls
``end_tick()``: any purged buffer that is still alive (something kept a reference) is a
PURGE_VIOLATION.

Honest limits (also in docs/GOVERNANCE.md): ``del`` removes a name, not memory; freed pages may
stay in the allocator. Zeroing first means the bytes are gone even if the pages linger. Copies
made inside native libraries (GPU allocator caches, AVFoundation pixel buffers, MediaPipe's
internal ``mp.Image``) cannot be reached from Python and are not zeroed by this gate.
"""

from __future__ import annotations

import gc
import logging
import threading
import weakref
from collections import Counter
from dataclasses import dataclass
from functools import partial

import numpy as np

log = logging.getLogger(__name__)

KINDS = ("video", "audio")


@dataclass(frozen=True)
class PurgeStats:
    purged: dict[str, int]
    freed: dict[str, int]
    unzeroable: dict[str, int]
    transcripts_purged: int
    live_raw_buffers: int
    violations: int


class PurgeGate:
    def __init__(self) -> None:
        # RLock: a weakref callback can run on this thread while the lock is held, if a
        # tracked array happens to be freed inside purge().
        self._lock = threading.RLock()
        self._live: dict[int, tuple[str, weakref.ref]] = {}
        self._purged: Counter[str] = Counter()
        self._freed: Counter[str] = Counter()
        self._unzeroable: Counter[str] = Counter()
        self._transcripts = 0
        self._violations = 0

    def purge(self, buf: np.ndarray, kind: str) -> None:
        """Zero ``buf`` in place and track it until it is freed.

        If ``buf`` is a view, the array that owns the memory is zeroed and tracked, because the
        whole underlying buffer is raw media.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if not isinstance(buf, np.ndarray):
            raise TypeError(f"purge() expects a numpy array, got {type(buf).__name__}")
        owner = buf
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        if owner.flags.writeable:
            owner.fill(0)
        else:
            self._unzeroable[kind] += 1
        key = id(owner)
        ref = weakref.ref(owner, partial(self._on_free, key, kind))
        with self._lock:
            self._live[key] = (kind, ref)
            self._purged[kind] += 1

    def _on_free(self, key: int, kind: str, _ref: weakref.ref) -> None:
        with self._lock:
            entry = self._live.get(key)
            # The id may already belong to a newer array; only drop the entry this ref owns.
            if entry is not None and entry[1] is _ref:
                del self._live[key]
                self._freed[kind] += 1

    def note_transcript_purged(self, n: int = 1) -> None:
        """Count ephemeral transcript text discarded inside perception (str cannot be zeroed)."""
        with self._lock:
            self._transcripts += n

    def live_raw_buffers(self) -> int:
        with self._lock:
            return len(self._live)

    def end_tick(self) -> PurgeStats:
        """Check that no purged buffer survived the tick; count survivors as violations."""
        if self.live_raw_buffers():
            gc.collect()  # reference cycles are allowed one collection before we call it a leak
        with self._lock:
            survivors = len(self._live)
            if survivors:
                self._violations += survivors
                kinds = Counter(kind for kind, _ in self._live.values())
                log.warning("PURGE_VIOLATION: %d raw buffer(s) still referenced: %s", survivors, dict(kinds))
                # Report each leaked buffer once, then stop tracking it.
                self._live.clear()
            return self._stats(live=survivors)

    def stats(self) -> PurgeStats:
        with self._lock:
            return self._stats(live=len(self._live))

    def _stats(self, live: int) -> PurgeStats:
        return PurgeStats(
            purged=dict(self._purged),
            freed=dict(self._freed),
            unzeroable=dict(self._unzeroable),
            transcripts_purged=self._transcripts,
            live_raw_buffers=live,
            violations=self._violations,
        )

    def reset(self) -> None:
        with self._lock:
            self._live.clear()
            self._purged.clear()
            self._freed.clear()
            self._unzeroable.clear()
            self._transcripts = 0
            self._violations = 0


gate = PurgeGate()
