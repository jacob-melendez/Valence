"""Logging filter that refuses to emit raw media.

Any log record whose message or arguments carry a numpy array or a bytes-like object is dropped,
so a stray ``log.debug("frame %s", frame)`` can never put pixels or audio into a log file.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

_RAW_TYPES = (np.ndarray, bytes, bytearray, memoryview)


def _is_raw(value: object) -> bool:
    if isinstance(value, _RAW_TYPES):
        return True
    if isinstance(value, dict):
        return any(_is_raw(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_is_raw(v) for v in value)
    return False


class RawMediaFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.dropped = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if _is_raw(record.msg) or (record.args and _is_raw(record.args)):
            self.dropped += 1
            return False
        return True


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time (servers and test runners replace it)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def install(logger: logging.Logger | None = None) -> RawMediaFilter:
    """Filter raw media out of logging.

    With an explicit ``logger``: attach to it and its handlers. Without: attach to every existing logger and
    handler (root, uvicorn, ...) and to ``logging.lastResort``, and give the ``valence`` logger a filtered
    handler of its own with propagation off, so ``valence.*`` records are filtered even when the host has
    configured no root handler.
    """
    flt = RawMediaFilter()
    if logger is not None:
        logger.addFilter(flt)
        for handler in logger.handlers:
            handler.addFilter(flt)
        return flt
    loggers = [logging.getLogger()] + [lg for lg in logging.Logger.manager.loggerDict.values()
                                       if isinstance(lg, logging.Logger)]
    for lg in loggers:
        lg.addFilter(flt)
        for handler in lg.handlers:
            handler.addFilter(flt)
    if logging.lastResort is not None:
        logging.lastResort.addFilter(flt)
    valence = logging.getLogger("valence")
    if not any(getattr(h, "valence_filtered", False) for h in valence.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler.valence_filtered = True
        valence.addHandler(handler)
        valence.propagate = False
    for handler in valence.handlers:
        handler.addFilter(flt)
    return flt
