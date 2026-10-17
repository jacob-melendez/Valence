"""Logging filter that refuses to emit raw media.

Any log record whose message or arguments carry a numpy array or a bytes-like object is dropped,
so a stray ``log.debug("frame %s", frame)`` can never put pixels or audio into a log file.
"""

from __future__ import annotations

import logging

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


def install(logger: logging.Logger | None = None) -> RawMediaFilter:
    """Attach the filter to every handler of ``logger`` (root by default) and to the logger itself."""
    logger = logger or logging.getLogger()
    flt = RawMediaFilter()
    logger.addFilter(flt)
    for handler in logger.handlers:
        handler.addFilter(flt)
    return flt
