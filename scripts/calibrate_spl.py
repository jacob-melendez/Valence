"""Interactive SPL calibration: map microphone dBFS to dB SPL with a phone meter.

    python scripts/calibrate_spl.py [--mic "MacBook Pro Microphone"] [--levels 3] [--seconds 5]

Play 2-3 steady, broadband sounds at clearly different levels (a fan, or pink noise from
a speaker; not a pure tone, since the phone reads dBA and we measure unweighted power).
For each level, type the phone SPL app reading, then stay quiet for the capture. The
offset is reading - dBFS; the median offset is saved as spl_offset_db. Each audio block
is reduced to its mean power and purged at once (also on error); nothing else is kept.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from valence.perception import calibration  # noqa: E402
from valence.perception.audio_features import mean_power, power_to_db, resample_48k_to_16k  # noqa: E402
from valence.perception.gate import purge  # noqa: E402

TARGET_SR = 16_000
FALLBACK_SR = 48_000
BLOCK_S = 0.5
WARMUP_BLOCKS = 1
OFFSET_RANGE = (80.0, 110.0)
SILENCE_DBFS = -90.0
MAX_SPREAD_DB = 3.0


def ask_spl(prompt: str) -> float | None:
    """Phone SPL reading (dBA) from stdin; None when the user stops."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"q", "quit", "done"}:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  enter a dB value (or q to stop)")
            continue
        if math.isfinite(value) and 20.0 <= value <= 130.0:
            return value
        print("  expected a reading between 20 and 130 dB")


def pick_device(sd: Any, name: str | None) -> tuple[int, str]:
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        raise SystemExit("no input devices")
    if name:
        for i, d in inputs:
            if name.lower() in d["name"].lower():
                return i, d["name"]
        available = ", ".join(d["name"] for _, d in inputs)
        raise SystemExit(f"no input device matching {name!r}; available: {available}")
    idx = sd.default.device[0]
    if idx is None or idx < 0:
        idx = inputs[0][0]
    return idx, devices[idx]["name"]


def supported_rate(sd: Any, device: int) -> int:
    """16 kHz if the device accepts it, else 48 kHz (resampled to 16 kHz per block)."""
    for sr in (TARGET_SR, FALLBACK_SR):
        try:
            sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=sr)
            return sr
        except Exception as e:
            print(f"  {sr} Hz rejected ({e.__class__.__name__})")
    raise SystemExit("microphone supports neither 16 kHz nor 48 kHz")


def block_power(stream: Any, sr: int) -> float:
    """Read one block, reduce it to mean power at 16 kHz, purge it."""
    block, _ = stream.read(int(sr * BLOCK_S))
    try:
        if sr == TARGET_SR:
            return mean_power(block[:, 0])
        mono16 = resample_48k_to_16k(block[:, 0])
        try:
            return mean_power(mono16)
        finally:
            purge(mono16, "audio")
            del mono16
    finally:
        purge(block, "audio")
        del block


def measure_dbfs(sd: Any, device: int, sr: int, seconds: float) -> float:
    """dBFS of the power-domain mean over ``seconds`` (never an average of dB values)."""
    n_blocks = max(1, round(seconds / BLOCK_S))
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=device,
                        blocksize=int(sr * BLOCK_S)) as stream:
        for _ in range(WARMUP_BLOCKS):
            block_power(stream, sr)
        powers = [block_power(stream, sr) for _ in range(n_blocks)]
    return power_to_db(float(np.mean(powers)))


def summarize(offsets: list[float]) -> float:
    offset = statistics.median(offsets)
    print(f"\noffsets: {', '.join(f'{o:+.1f}' for o in offsets)} dB -> median {offset:+.2f} dB")
    if len(offsets) > 1 and max(offsets) - min(offsets) > MAX_SPREAD_DB:
        print(f"  WARNING: offsets spread > {MAX_SPREAD_DB:.0f} dB; sound not steady or mic AGC active")
    lo, hi = OFFSET_RANGE
    if not lo <= offset <= hi:
        print(f"  WARNING: offset outside +{lo:.0f}..+{hi:.0f} dB; check the mic, its input gain, and the phone app")
    return offset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Calibrate microphone dBFS to dB SPL with a phone SPL meter.")
    ap.add_argument("--mic", default=os.environ.get("VALENCE_MIC"), help="input device name substring")
    ap.add_argument("--levels", type=int, default=3, help="number of sound levels (2-3)")
    ap.add_argument("--seconds", type=float, default=5.0, help="capture length per level")
    ap.add_argument("--yes", action="store_true", help="save without asking")
    args = ap.parse_args(argv)

    import sounddevice as sd

    device, name = pick_device(sd, args.mic)
    sr = supported_rate(sd, device)
    print(f"mic: {name!r} at {sr} Hz")
    n_levels = min(3, max(2, args.levels))
    offsets: list[float] = []
    for k in range(1, n_levels + 1):
        reading = ask_spl(f"\n[{k}/{n_levels}] play a steady sound, then type the phone SPL reading in dBA (q to stop): ")
        if reading is None:
            break
        dbfs = measure_dbfs(sd, device, sr, args.seconds)
        if dbfs < SILENCE_DBFS:
            print(f"  {dbfs:.1f} dBFS is near-silence: microphone permission denied? level skipped")
            continue
        offsets.append(reading - dbfs)
        print(f"  {dbfs:.1f} dBFS -> offset {reading - dbfs:+.1f} dB")
    if not offsets:
        raise SystemExit("no usable levels recorded")

    offset = summarize(offsets)
    if args.yes or input(f"\nsave to {calibration.CALIBRATION_PATH}? [Y/n] ").strip().lower() in {"", "y", "yes"}:
        calibration.save({"spl_offset_db": round(offset, 2), "spl_calibrated": True})
        print("saved")
    else:
        print("not saved")


if __name__ == "__main__":
    main()
