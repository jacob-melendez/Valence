"""Terminal view of /ws/events: the PRD's governance log line, labels and explanations.

Backend tooling that works even if no frontend is running:
    python scripts/console.py [--url ws://127.0.0.1:8000/ws/events]
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets
from rich.console import Console

console = Console()


def render(msg: dict) -> None:
    kind = msg.get("type")
    if kind == "hello":
        states = ", ".join(s["display"] for s in msg["states"])
        console.print(f"[bold]Connected[/] mode={msg['mode']} room_only={msg['room_only']} states: {states}")
    elif kind == "governance":
        console.print(f"[green]{msg['message']}[/] [dim](frames purged {msg['frames_purged']}, audio blocks "
                      f"{msg['audio_blocks_purged']}, violations {msg['violations']}, live buffers {msg['live_raw_buffers']})[/]")
    elif kind == "prediction":
        who = "Room" if msg["person_id"] == 0 else f"Person #{msg['person_id']}"
        flag = " [bold red]MASK OVERRIDE[/]" if msg["mask_override"] else ""
        warm = " [yellow](warming up)[/]" if msg["warming_up"] else ""
        sim = f" [magenta]SIMULATED: {', '.join(msg['simulated_fields'])}[/]" if msg["simulated_fields"] else ""
        face = msg["face_only"]["label"] or "no face"
        console.print(f"[bold cyan]{who}: {msg['label_display']}[/] {round(100 * msg['confidence'])}% "
                      f"[dim](face-only: {face}, source: {msg['source']})[/]{flag}{warm}{sim}")
        console.print(f"  [dim]{msg['explanation']['text']}[/]")
    elif kind == "explanation":
        console.print(f"  [italic]{msg['text']}[/] [dim]({msg['source']}, {msg['latency_ms']} ms)[/]")
    elif kind == "status" and (msg["degraded"] or msg["purge_violation"]):
        console.print(f"[yellow]status: degraded={msg['degraded']} purge_violation={msg['purge_violation']}[/]")


async def main(url: str) -> None:
    async for ws in websockets.connect(url):
        try:
            async for raw in ws:
                render(json.loads(raw))
        except websockets.ConnectionClosed:
            console.print("[red]disconnected, retrying...[/]")
            continue


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/events")
    asyncio.run(main(ap.parse_args().url))
