"""Single import point for the biometric purge gate (``valence/governance/purge.py``).

``purge(buf, kind)`` zero-fills the buffer in place and tracks it with a weakref until it is
freed; a buffer still alive at the end of a tick is a PURGE_VIOLATION. Perception calls it right
after feature extraction, then ``del``s its own references.

Two flavours, so the governance counters stay meaningful:

- ``purge(buf, kind)``: a raw capture buffer (a decoded camera frame, a mic callback block).
  ``frames_decoded == purged["video"]`` holds because only decoded frames are purged this way.
- ``purge_derived(buf, kind)``: a copy derived from raw media inside perception (RGB conversion,
  head crop, grayscale, concatenated or int16 audio). Zeroed and tracked the same way, counted
  under ``<kind>_derived`` when the gate supports that kind.
"""

from __future__ import annotations

import numpy as np

from valence.governance.purge import KINDS, gate


def purge(buf: np.ndarray, kind: str) -> None:
    gate.purge(buf, kind)


def purge_derived(buf: np.ndarray, kind: str) -> None:
    derived = f"{kind}_derived"
    gate.purge(buf, derived if derived in KINDS else kind)


def note_transcript_purged(n: int = 1) -> None:
    """Recognizer result strings discarded inside perception (str cannot be zeroed)."""
    gate.note_transcript_purged(n)
