"""Listen to the API's /ws/events for a while and summarize what arrives (Step 4 / Step 8 checks).

    python scripts/spikes/ws_probe.py [--url ws://127.0.0.1:8000/ws/events] [--seconds 90]

Prints message counts by type, track ids seen, predictions (label, confidence, source), the last
governance counters (frames decoded vs purged, violations, live raw buffers) and narrator sources.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import time


async def listen(url: str, seconds: float) -> dict:
    import websockets

    counts: collections.Counter[str] = collections.Counter()
    tracks: set[int] = set()
    preds: list[tuple] = []
    sources: collections.Counter[str] = collections.Counter()
    last: dict[str, dict] = {}
    t_end = time.monotonic() + seconds
    async with websockets.connect(url, max_size=2**22) as ws:
        while (left := t_end - time.monotonic()) > 0:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=left))
            except asyncio.TimeoutError:
                break
            kind = msg.get("type", "?")
            counts[kind] += 1
            last[kind] = msg
            if kind == "tracks":
                tracks.update(p["id"] for p in msg.get("persons", []))
            elif kind == "prediction":
                preds.append((msg.get("person_id"), msg.get("label"), round(msg.get("confidence", 0), 2),
                              msg.get("warming_up"), msg.get("explanation", {}).get("source")))
            elif kind == "explanation":
                sources[msg.get("source", "?")] += 1
    return {"counts": dict(counts), "track_ids": sorted(tracks), "predictions_tail": preds[-6:],
            "explanation_sources": dict(sources), "last_governance": last.get("governance"),
            "last_status": last.get("status"), "last_explanation": last.get("explanation")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/events")
    ap.add_argument("--seconds", type=float, default=90.0)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(listen(a.url, a.seconds)), indent=2, default=str))


if __name__ == "__main__":
    main()
