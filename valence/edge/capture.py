"""Camera and microphone capture threads.

The camera thread keeps only the latest frame (a 1-slot handoff); a frame that is replaced
before perception takes it is purged. The mic callback only copies the block into a queue.
Both fail loudly on the symptoms of a denied macOS permission (black frames, exact silence).
"""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np
import sounddevice as sd

from valence.perception.audio_features import resample_48k_to_16k
from valence.perception.gate import purge

SR = 16000
PERMISSION_HINT = ("System Settings -> Privacy & Security -> {what}: enable the app that launched Python "
                   "(Claude / Terminal / iTerm / VS Code), then retry.")


class CameraCapture:
    def __init__(self, index: int | None = None, width: int = 1280, height: int = 720) -> None:
        self.cap = self._open(index, width, height)
        self._cond = threading.Condition()
        self._latest: tuple[np.ndarray, float, float] | None = None
        self._stop = threading.Event()
        self.frames_read = 0
        self.frames_dropped = 0
        self._thread = threading.Thread(target=self._run, name="valence-camera", daemon=True)

    @staticmethod
    def _open(index: int | None, width: int, height: int) -> cv2.VideoCapture:
        for i in [index] if index is not None else [0, 1, 2]:
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            brightness = 0.0
            for _ in range(10):  # auto-exposure warm-up
                ok, f = cap.read()
                if ok:
                    brightness = float(f.mean())
                    purge(f, "video")
                    del f
            if brightness > 5:
                return cap
            cap.release()
        raise RuntimeError("camera gave no usable frames (black = permission denied). "
                           + PERMISSION_HINT.format(what="Camera"))

    def start(self) -> CameraCapture:
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            item = (frame, time.monotonic(), time.time())
            del frame
            self.frames_read += 1
            with self._cond:
                old, self._latest = self._latest, item
                self._cond.notify()
            if old is not None:
                self.frames_dropped += 1
                purge(old[0], "video")
            del item, old

    def take(self, timeout: float = 1.0) -> tuple[np.ndarray, float, float] | None:
        """Hand the newest frame (and its ownership) to the caller."""
        with self._cond:
            if self._latest is None:
                self._cond.wait(timeout)
            item, self._latest = self._latest, None
        return item

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()
        with self._cond:
            if self._latest is not None:
                purge(self._latest[0], "video")
            self._latest = None


def find_input_device(name: str | None) -> int:
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if name:
        for i, d in inputs:
            if name.lower() in d["name"].lower():
                return i
        raise RuntimeError(f"no input device matching {name!r}; have {[d['name'] for _, d in inputs]}")
    for i, d in inputs:  # the built-in mic, not a Bluetooth headset that happens to be default
        if "macbook" in d["name"].lower() and "microphone" in d["name"].lower():
            return i
    return int(sd.default.device[0])


class MicCapture:
    """16 kHz mono float32 in 100 ms blocks (48 kHz + polyphase resample if 16 kHz is rejected)."""

    def __init__(self, name: str | None = None, block_s: float = 0.1) -> None:
        self.device = find_input_device(name)
        self.device_name = sd.query_devices(self.device)["name"]
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.overflows = 0
        self.native_sr = SR
        try:
            self.stream = self._open(SR, block_s)
        except sd.PortAudioError:
            self.native_sr = 48000
            self.stream = self._open(48000, block_s)

    def _open(self, sr: int, block_s: float) -> sd.InputStream:
        return sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=int(sr * block_s),
                              device=self.device, callback=self._callback)

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.overflows += 1

    def start(self) -> MicCapture:
        self.stream.start()
        return self

    def read(self, seconds: float = 1.0, timeout: float = 2.0) -> tuple[np.ndarray, float, float]:
        """Concatenate blocks into one ~``seconds`` 16 kHz block; the pieces are purged."""
        need = int(self.native_sr * seconds)
        parts, n = [], 0
        while n < need:
            part = self._q.get(timeout=timeout)
            parts.append(part)
            n += len(part)
        t, wall = time.monotonic(), time.time()
        block = np.concatenate(parts)
        for part in parts:
            purge(part, "audio")
        del parts
        if self.native_sr != SR:
            native, block = block, resample_48k_to_16k(block)
            purge(native, "audio")
            del native
        return block, t, wall

    def check_not_silent(self, seconds: float = 1.0) -> None:
        block, _, _ = self.read(seconds)
        peak = float(np.abs(block).max())
        purge(block, "audio")
        if peak == 0.0:
            raise RuntimeError(f"mic {self.device_name!r} returned exact silence (permission denied). "
                               + PERMISSION_HINT.format(what="Microphone"))

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()
        while not self._q.empty():
            purge(self._q.get_nowait(), "audio")
