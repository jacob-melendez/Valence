"""Hardware self-test: camera + microphone must deliver real signal, not silent zeros.

macOS grants camera/mic to the *app that launched Python* (Terminal, iTerm, VS Code,
Claude), not to Python. A denied camera yields black frames and a denied mic yields
exact zeros, so this test fails loudly on both instead of passing silently.

    python scripts/spikes/selftest.py [--camera N] [--mic "MacBook Pro Microphone"]

Nothing is written to disk; frames and audio are zero-filled and dropped after the
scalar checks.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np


def _wipe(buf: np.ndarray | None) -> None:
    if buf is not None:
        buf.fill(0)


def camera_names() -> list[str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    return [ln.strip().rstrip(":") for ln in out.splitlines() if ln.startswith("    ") and ln.strip().endswith(":")]


def test_camera(indices: list[int], wait_s: float = 0.0) -> int:
    print(f"cameras (system_profiler): {camera_names() or '?'}")
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return _probe_cameras(indices)
        except SystemExit:
            if time.monotonic() >= deadline:
                raise
            print("  waiting for camera permission (click OK on the macOS prompt)...", flush=True)
            time.sleep(3.0)


def _probe_cameras(indices: list[int]) -> int:
    import cv2

    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"  [cam {i}] not opened")
            continue
        frame = None
        try:
            for _ in range(10):  # auto-exposure warm-up
                ok, frame = cap.read()
                _wipe(frame if ok else None)
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  [cam {i}] opened but read() failed")
                continue
            h, w = frame.shape[:2]
            brightness = float(frame.mean())
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            _wipe(frame)
            del frame
            print(f"  [cam {i}] {w}x{h} @ {fps:.0f} fps, mean brightness {brightness:.1f} "
                  f"(read {1e3 * (time.monotonic() - t0):.0f} ms)")
            if brightness > 5:
                return i
            print(f"  [cam {i}] black frames -> camera permission denied or lens covered")
        finally:
            cap.release()
    raise SystemExit(
        "CAMERA FAIL: no camera gave non-black frames.\n"
        "  System Settings -> Privacy & Security -> Camera: enable the app running this\n"
        "  (Claude / Terminal / iTerm / VS Code), then fully quit and reopen that app."
    )


def pick_mic(name: str | None) -> tuple[int, dict]:
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    print("input devices:")
    for i, d in inputs:
        print(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch, {d['default_samplerate']:.0f} Hz)")
    if name:
        for i, d in inputs:
            if name.lower() in d["name"].lower():
                return i, d
        raise SystemExit(f"MIC FAIL: no input device matching {name!r}")
    default_in = sd.default.device[0]
    for i, d in inputs:  # prefer the built-in mic over virtual/continuity devices
        if "macbook" in d["name"].lower() and "microphone" in d["name"].lower():
            return i, d
    return default_in, devices[default_in]


def test_mic(name: str | None) -> str:
    import sounddevice as sd

    idx, dev = pick_mic(name)
    rates = [16000, int(dev["default_samplerate"])]
    for sr in rates:
        try:
            x = sd.rec(int(sr * 1.0), samplerate=sr, channels=1, dtype="float32", device=idx, blocking=True)
        except Exception as e:  # rate rejected -> try native
            print(f"  [{dev['name']}] {sr} Hz rejected: {e}")
            continue
        peak = float(np.abs(x).max())
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        dbfs = 20 * np.log10(max(rms, 1e-12))
        _wipe(x)
        del x
        print(f"  [{dev['name']}] {sr} Hz: peak {peak:.2e}, {dbfs:.1f} dBFS")
        if peak > 1e-5:
            return dev["name"]
        raise SystemExit(
            "MIC FAIL: exact silence -> microphone permission denied.\n"
            "  System Settings -> Privacy & Security -> Microphone: enable the app running this\n"
            "  (Claude / Terminal / iTerm / VS Code), then fully quit and reopen that app."
        )
    raise SystemExit("MIC FAIL: no supported sample rate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="AVFoundation index (default: try 0..2)")
    ap.add_argument("--mic", default=os.environ.get("VALENCE_MIC"), help="input device name substring")
    ap.add_argument("--skip-camera", action="store_true")
    ap.add_argument("--skip-mic", action="store_true")
    ap.add_argument("--wait", type=float, default=0.0, help="keep retrying the camera this many seconds "
                    "(first run: macOS shows the permission prompt asynchronously)")
    a = ap.parse_args()

    if platform.machine() != "arm64":
        raise SystemExit(f"PYTHON FAIL: {platform.machine()} interpreter (Rosetta?). Use an arm64 Python 3.12.")
    print(f"python {sys.version.split()[0]} {platform.machine()}  parent app: {os.environ.get('__CFBundleIdentifier', '?')}")
    if os.environ.get("__CFBundleIdentifier", "").startswith("com.anthropic"):
        print("  note: Claude Code's shell helper has no camera usage string, so macOS can't grant it the camera;"
              " run camera steps from Terminal.app / iTerm / VS Code")

    if not a.skip_camera:
        cam = test_camera([a.camera] if a.camera is not None else [0, 1, 2], a.wait)
        print(f"CAMERA OK: index {cam}")
    if not a.skip_mic:
        mic = test_mic(a.mic)
        print(f"MIC OK: {mic!r}")
    print("SELFTEST PASS")


if __name__ == "__main__":
    main()
