"""Single import point for the biometric purge gate.

The gate itself lives in ``valence/governance/purge.py`` (owned by the core session):
``purge(buf, kind)`` zero-fills the buffer in place, registers a weakref finalizer and
counts the purge. Everything in perception/edge calls ``purge`` from here right after
feature extraction, then ``del``s its own references.
"""

from __future__ import annotations

import sys

import numpy as np

try:
    from valence.governance.purge import gate as _gate

    def purge(buf: np.ndarray, kind: str) -> None:
        _gate.purge(buf, kind)

    GATE_AVAILABLE = True
except ImportError:  # governance module not pulled yet: still zero-fill, but say so loudly
    GATE_AVAILABLE = False
    print("[GOVERNANCE] valence.governance.purge missing; using zero-fill fallback", file=sys.stderr)

    def purge(buf: np.ndarray, kind: str) -> None:
        buf.fill(0)
