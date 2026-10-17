"""Event bus: worker threads publish, every WebSocket client gets its own bounded queue.

A slow or absent client never blocks the pipeline: when a client's queue is full the oldest message
is dropped. With no clients connected, messages are simply not queued.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from pydantic import BaseModel

CLIENT_QUEUE = 200


def encode(msg: BaseModel | dict[str, Any]) -> str:
    if isinstance(msg, BaseModel):
        return msg.model_dump_json()
    return json.dumps(msg, separators=(",", ":"))


class EventBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue[str]] = set()
        self._lock = threading.Lock()
        self.published = 0
        self.dropped = 0
        self._listeners: list = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add_listener(self, fn) -> None:
        """Synchronous in-process listener (tests, console); called on the publishing thread."""
        self._listeners.append(fn)

    def publish(self, msg: BaseModel | dict[str, Any]) -> None:
        """Thread-safe. Encodes once, then fans out on the event loop."""
        payload = encode(msg)
        self.published += 1
        for fn in list(self._listeners):
            fn(payload)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._fanout, payload)
        except RuntimeError:  # loop shutting down
            pass

    def _fanout(self, payload: str) -> None:
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=CLIENT_QUEUE)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._clients.discard(q)

    @property
    def n_clients(self) -> int:
        return len(self._clients)
