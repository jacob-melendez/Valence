"""Client for the local narrator service (POST :8081/narrate). Never blocks the tick loop.

One worker thread; a latest-only slot per person (stale requests are dropped); a small cache keyed on
the label, confidence band and the exact driver display strings, so a cached text can only be reused
when every number it could mention is identical.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable

import httpx

from valence.contracts import Explanation, FactSheet, NarrationResult


def band(confidence: float) -> str:
    return "high" if confidence >= 0.85 else ("medium" if confidence >= 0.6 else "low")


def cache_key(sheet: FactSheet) -> tuple:
    drivers = tuple(d.display for layer in sheet.layers for d in layer.drivers)
    return (sheet.person, sheet.label, sheet.label_display, band(sheet.confidence), sheet.confidence_display,
            sheet.mask_override, sheet.face_visible, drivers)


class NarratorClient:
    def __init__(self, url: str, publish: Callable, deadline_s: float = 2.5, cache_size: int = 256) -> None:
        self.url = url.rstrip("/")
        self.publish = publish
        self.deadline_s = deadline_s
        self._pending: dict[str, tuple[FactSheet, int]] = {}
        self._cv = threading.Condition()
        self._cache: OrderedDict[tuple, NarrationResult] = OrderedDict()
        self._cache_size = cache_size
        self._stop = threading.Event()
        self.healthy = False
        self.failures = 0
        self._thread = threading.Thread(target=self._run, name="narrator-client", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def request(self, sheet: FactSheet, person_id: int) -> None:
        with self._cv:
            self._pending[sheet.person] = (sheet, person_id)  # replaces any stale request for this person
            self._cv.notify()

    def clear_cache(self) -> None:
        self._cache.clear()
        with self._cv:
            self._pending.clear()

    def check_health(self) -> bool:
        try:
            self.healthy = httpx.get(f"{self.url}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            self.healthy = False
        return self.healthy

    def _run(self) -> None:
        with httpx.Client(timeout=self.deadline_s + 0.5) as client:
            while not self._stop.is_set():
                with self._cv:
                    while not self._pending and not self._stop.is_set():
                        self._cv.wait(timeout=1.0)
                    if self._stop.is_set():
                        return
                    _, (sheet, person_id) = self._pending.popitem()
                self._narrate(client, sheet, person_id)

    def _narrate(self, client: httpx.Client, sheet: FactSheet, person_id: int) -> None:
        key = cache_key(sheet)
        result = self._cache.get(key)
        if result is None:
            t0 = time.perf_counter()
            try:
                resp = client.post(f"{self.url}/narrate", content=sheet.model_dump_json(),
                                   headers={"content-type": "application/json"})
                resp.raise_for_status()
                result = NarrationResult.model_validate(resp.json())
                self.healthy = True
            except (httpx.HTTPError, ValueError):
                self.failures += 1
                self.healthy = False
                return  # the prediction already carries template text
            if result.source != "template":
                self._cache[key] = result
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            latency = int((time.perf_counter() - t0) * 1000)
        else:
            latency = 0
        self.publish(Explanation(event_id=sheet.event_id, person_id=person_id, text=result.text,
                                 source=result.source, latency_ms=latency or result.latency_ms))
