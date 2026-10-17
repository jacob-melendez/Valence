"""Background perception load without a camera: loops the real VideoPerceiver on the bundled
ultralytics sample images at the live rate, so narrator latency can be measured "with YOLO running".

    python scripts/spikes/gpu_load.py --seconds 600 [--hz 10]

Prints achieved Hz and process() latency every 30 s. Frames are copies of a public sample image
and are purged like live frames.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2

from valence.perception.video import VideoPerceiver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--hz", type=float, default=10.0)
    a = ap.parse_args()
    assets = Path(cv2.__file__).parents[1] / "ultralytics" / "assets"
    images = [cv2.resize(cv2.imread(str(assets / n)), (1280, 720)) for n in ("zidane.jpg", "bus.jpg")]
    vp = VideoPerceiver()
    period, t_start = 1.0 / a.hz, time.monotonic()
    next_t, last_report, ms, n = t_start, t_start, [], 0
    try:
        while (now := time.monotonic()) - t_start < a.seconds:
            frame = images[(n // 50) % 2].copy()  # switch scene every 5 s
            t0 = time.perf_counter()
            vp.process(frame, now, time.time(), n)
            ms.append((time.perf_counter() - t0) * 1e3)
            del frame
            n += 1
            if now - last_report >= 30:
                print(f"[load] {n / (now - t_start):.1f} Hz, process p50 {statistics.median(ms):.0f} ms "
                      f"p95 {sorted(ms)[int(0.95 * (len(ms) - 1))]:.0f} ms", flush=True)
                last_report, ms = now, []
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        vp.close()


if __name__ == "__main__":
    main()
