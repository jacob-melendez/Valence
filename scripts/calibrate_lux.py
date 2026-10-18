"""Interactive lux calibration against a phone lux meter.

    python scripts/calibrate_lux.py [--camera 0] [--setups 3] [--seconds 3] [--two-point]
                                    [--width 1280 --height 720]

For each lighting setup: set the light, hold the phone lux meter where people sit and
type its reading, then keep everyone out of the camera's view while ~3 s of frames are
reduced to (y_bg, sigma). Each frame is purged right after that reduction (also when it
fails). Capture at the edge runner's resolution (1280x720 by default): sensor-noise sigma
depends on it. Only the fitted constants are written, to data/real/calibration.json.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from valence.edge import calibration_store  # noqa: E402
from valence.perception import calibration  # noqa: E402
from valence.perception.gate import purge  # noqa: E402
from valence.perception.lux import LuxModel, background_stats  # noqa: E402

WARMUP_FRAMES = 20
DELAY_S = 4.0
BLACK_Y = 1.0


def ask_lux(prompt: str) -> float | None:
    """Phone lux reading from stdin; None when the user stops."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"q", "quit", "done"}:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  enter a lux value (or q to stop)")
            continue
        if math.isfinite(value) and value > 0:
            return value
        print("  lux must be > 0")


def frame_scalars(cap: cv2.VideoCapture) -> tuple[float, float] | None:
    """Read one frame, reduce it to (y_bg, sigma), purge it."""
    ok, frame = cap.read()
    if frame is None:
        return None
    try:
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            stats = background_stats(gray, ())
        finally:
            purge(gray, "video")
            del gray
    finally:
        purge(frame, "video")
        del frame
    return stats.y_bg, stats.sigma


def flush(cap: cv2.VideoCapture, n: int) -> None:
    """Drop stale buffered frames and let auto-exposure settle."""
    for _ in range(n):
        frame_scalars(cap)


def capture(cap: cv2.VideoCapture, seconds: float) -> tuple[float, float, int]:
    """Median (y_bg, sigma) over ``seconds`` of frames, and the frame count."""
    ys: list[float] = []
    sigmas: list[float] = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        scalars = frame_scalars(cap)
        if scalars is not None and all(math.isfinite(v) for v in scalars):
            ys.append(scalars[0])
            sigmas.append(scalars[1])
    if not ys:
        return math.nan, math.nan, 0
    return statistics.median(ys), statistics.median(sigmas), len(ys)


def report(model: LuxModel, readings: list[tuple[float, float, float]]) -> None:
    print(f"\nfit: lux = exp({model.a:.4f} + {model.b:.5f} * y_bg + {model.c:.4f} * sigma)")
    for y, s, lux in readings:
        pred = model.predict(y, s)
        print(f"  y_bg {y:6.1f}  sigma {s:5.2f}  meter {lux:8.1f} lux  model {pred:8.1f} lux"
              f"  residual {100.0 * (pred / lux - 1.0):+5.1f}%")


def collect(cap: cv2.VideoCapture, n_setups: int, seconds: float) -> list[tuple[float, float, float]]:
    readings: list[tuple[float, float, float]] = []
    flush(cap, WARMUP_FRAMES)
    print("Keep people out of the camera's view during every capture.")
    for k in range(1, n_setups + 1):
        lux = ask_lux(f"\n[{k}/{n_setups}] set the lighting, then type the phone lux reading (q to stop): ")
        if lux is None:
            break
        for n in range(int(DELAY_S), 0, -1):  # time to lean out of view after pressing Enter
            print(f"\a  lean OUT of the camera's view: capturing in {n}...", flush=True)
            time.sleep(1.0)
        flush(cap, WARMUP_FRAMES)
        y, s, n_frames = capture(cap, seconds)
        if n_frames == 0:
            print("  no frames read; setup skipped")
            continue
        if y < BLACK_Y:
            print("  black frames: camera permission denied or lens covered; setup skipped")
            continue
        print(f"\a  {n_frames} frames: median y_bg {y:.1f}, sigma {s:.2f}  (you can come back)")
        readings.append((y, s, lux))
    return readings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fit the webcam lux model against a phone lux meter.")
    ap.add_argument("--camera", type=int, default=0, help="AVFoundation camera index")
    ap.add_argument("--setups", type=int, default=3, help="lighting setups to record (3-4 recommended)")
    ap.add_argument("--seconds", type=float, default=3.0, help="capture length per setup")
    ap.add_argument("--width", type=int, default=1280, help="capture width; must match the edge runner")
    ap.add_argument("--height", type=int, default=720, help="capture height; must match the edge runner")
    ap.add_argument("--two-point", action="store_true", help="two setups (dim, bright); fit on y_bg only")
    ap.add_argument("--yes", action="store_true", help="save without asking")
    args = ap.parse_args(argv)
    n_setups = 2 if args.two_point else max(2, args.setups)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.camera} did not open (check index and camera permission)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"camera {args.camera} at {got[0]}x{got[1]}")
    if got != (args.width, args.height):
        print(f"  WARNING: asked for {args.width}x{args.height}; sigma is resolution-dependent, so "
              "calibrate at the resolution the edge runner captures")
    try:
        readings = collect(cap, n_setups, args.seconds)
    finally:
        cap.release()

    try:
        model = LuxModel.fit(readings, use_sigma=not args.two_point)
    except ValueError as e:
        raise SystemExit(f"cannot fit: {e}") from None
    report(model, readings)
    if args.yes or input(f"\nsave to {calibration.CALIBRATION_PATH}? [Y/n] ").strip().lower() in {"", "y", "yes"}:
        calibration_store.save({"lux": model.to_dict()})
        print("saved")
    else:
        print("not saved")


if __name__ == "__main__":
    main()
