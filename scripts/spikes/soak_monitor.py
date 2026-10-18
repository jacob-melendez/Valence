"""Soak monitor for a running demo (P1-3): samples the API + narrator every minute.

    python scripts/spikes/soak_monitor.py --minutes 60 --out /path/soak.jsonl

Per sample: RSS of the API and narrator processes, video FPS (frames decoded per second since the last sample),
purge counters (violations, live raw buffers, purge gap), narrator sources, and the files the API process holds
open for writing (only the audit log is expected). Scalars only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def pid_on(port: int) -> int | None:
    out = subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    return int(out.split()[0]) if out.split() else None


def rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return round(int(out) / 1024, 1) if out else None


def write_files(pid: int | None) -> list[str]:
    """Regular files the process has open for writing (FD mode w or u)."""
    if pid is None:
        return []
    out = subprocess.run(["lsof", "-nP", "-p", str(pid)], capture_output=True, text=True).stdout.splitlines()[1:]
    files = []
    for line in out:
        cols = line.split(None, 8)
        if len(cols) >= 9 and cols[4] == "REG" and cols[3][-1:] in ("w", "u"):
            files.append(cols[8])
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--narrator", default="http://127.0.0.1:8081")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=60)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    api_pid, nar_pid = pid_on(8000), pid_on(8081)
    t0 = time.monotonic()
    last = None
    with a.out.open("w", encoding="utf-8") as f:
        while time.monotonic() - t0 < a.minutes * 60:
            now = time.monotonic()
            row: dict = {"min": round((now - t0) / 60, 1)}
            try:
                gov = get(f"{a.api}/api/governance/stats")
                row.update({k: gov.get(k) for k in ("frames_decoded", "frames_purged", "violations", "live_raw_buffers")})
                if last is not None and gov.get("frames_decoded") is not None:
                    row["fps"] = round((gov["frames_decoded"] - last[1]) / (now - last[0]), 2)
                last = (now, gov.get("frames_decoded") or 0)
                health = get(f"{a.api}/api/health")
                row["edge_error"], row["mic_error"] = health.get("edge_error"), health.get("mic_error")
            except Exception as e:  # the API is down: record it, keep sampling
                row["api_error"] = type(e).__name__
            try:
                row["narrator_by_source"] = get(f"{a.narrator}/health").get("by_source")
            except Exception as e:
                row["narrator_error"] = type(e).__name__
            row["rss_api_mb"], row["rss_narrator_mb"] = rss_mb(api_pid), rss_mb(nar_pid)
            row["api_write_files"] = write_files(api_pid)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
            time.sleep(max(0.0, a.every - (time.monotonic() - now)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
