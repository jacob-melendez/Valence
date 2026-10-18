"""Edge runner: camera + mic -> perception threads -> ``sink(VideoObs | AudioObs)``.

Library entry point (used by the API in live mode)::

    stop = start(sink, video=True, audio=True, obscure_face=False)
    ...
    stop()

CLI (features only; never media)::

    python -m valence.edge.run --video --audio --print
    python -m valence.edge.run --jsonl data/real/focused.jsonl --duration 180
    python -m valence.edge.run --video --print --tracemalloc --duration 600

Threads: camera capture (latest frame only), one video perception thread that owns YOLO and
every MediaPipe instance, one audio thread (mic queue -> 1 s blocks). The sink is called from
the perception threads and must be thread-safe and fast.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

for _k, _v in (("YOLO_OFFLINE", "1"), ("YOLO_AUTOINSTALL", "false"), ("HF_HUB_OFFLINE", "1"),
               ("TRANSFORMERS_OFFLINE", "1")):
    os.environ.setdefault(_k, _v)  # fully offline after the one-time model downloads

# libomp: torch and LightGBM (Homebrew libomp) each bring an OpenMP runtime. Two runtimes in one
# process segfault as soon as either runs a multi-threaded region (YOLO's conv/bn fusion, LightGBM
# parsing a model file). Measured on the demo Mac (docs/BENCHMARKS.md): torch imported first plus
# OMP_NUM_THREADS=1 is stable; DYLD_LIBRARY_PATH=<torch>/lib at launch is better still (LightGBM then
# shares torch's runtime). Set before torch/LightGBM initialize OpenMP, i.e. import this module first.
_omp = os.environ.setdefault("OMP_NUM_THREADS", "1")
if _omp != "1" and "torch/lib" not in os.environ.get("DYLD_LIBRARY_PATH", ""):
    logging.getLogger("valence.edge").warning("OMP_NUM_THREADS=%s with torch + LightGBM: expect a segfault", _omp)
if "lightgbm" in sys.modules and "torch" not in sys.modules:
    logging.getLogger("valence.edge").warning(
        "lightgbm was imported before torch: expect a segfault; import valence.edge.run first")
import torch  # noqa: E402,F401

from valence.config import settings  # noqa: E402
from valence.contracts import AudioObs, VideoObs, obs_to_dict  # noqa: E402

log = logging.getLogger("valence.edge")
Sink = Callable[[VideoObs | AudioObs], None]


@dataclasses.dataclass(slots=True)
class EdgeStats:
    video_frames: int = 0
    video_hz: float = 0.0
    video_ms: float = 0.0
    camera_dropped: int = 0
    frames_decoded: int = 0
    audio_blocks: int = 0
    mic_overflows: int = 0
    sink_errors: int = 0
    errors: list[str] = dataclasses.field(default_factory=list)


class EdgeRunner:
    """Owns the capture and perception threads; calling the instance stops everything."""

    def __init__(self, sink: Sink, *, video: bool, audio: bool, obscure_face: bool, camera: int | None,
                 mic: str | None, fps: float, imgsz: int, min_person_h: float) -> None:
        self.sink, self.fps, self.imgsz, self.obscure_face = sink, fps, imgsz, obscure_face
        self.min_person_h = min_person_h
        self.stats = EdgeStats()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.camera = self.mic = None
        if video:
            from valence.edge.capture import CameraCapture

            self.camera = CameraCapture(camera, fps=fps)  # raises on black frames (permission)
        if audio:
            from valence.edge.capture import MicCapture

            self.mic = MicCapture(mic).start()
            try:
                self.mic.check_not_silent()  # raises on exact silence (permission)
            except Exception:
                self.mic.close()
                if self.camera is not None:
                    self.camera.close()
                raise
        if self.camera is not None:
            self.camera.start()
            self._spawn("valence-video", self._video_loop)
        if self.mic is not None:
            self._spawn("valence-audio", self._audio_loop)

    def _spawn(self, name: str, target: Callable[[], None]) -> None:
        def guarded() -> None:
            try:
                target()
            except Exception as e:  # surface, never swallow silently
                log.exception("%s crashed", name)
                self.stats.errors.append(f"{name}: {type(e).__name__}: {e}")

        th = threading.Thread(target=guarded, name=name, daemon=True)
        th.start()
        self._threads.append(th)

    def _emit(self, obs: VideoObs | AudioObs) -> None:
        try:
            self.sink(obs)
        except Exception:
            self.stats.sink_errors += 1
            if self.stats.sink_errors <= 3:
                log.exception("sink failed")

    def _video_loop(self) -> None:
        from valence.perception.video import VideoPerceiver

        perceiver = VideoPerceiver(imgsz=self.imgsz, obscure_face=self.obscure_face,  # this thread owns MediaPipe
                                   min_person_h=self.min_person_h)
        period, seq = 1.0 / self.fps, 0
        next_t, last_t = time.monotonic(), None
        try:
            while not self._stop.is_set():
                item = self.camera.take(timeout=1.0)
                if item is None:
                    continue
                frame, t, wall, decoded = item
                del item
                t0 = time.perf_counter()
                obs = perceiver.process(frame, t, wall, seq, frames_decoded=decoded)
                del frame
                s = self.stats
                s.video_ms = 0.9 * s.video_ms + 0.1 * (time.perf_counter() - t0) * 1e3 if seq else (time.perf_counter() - t0) * 1e3
                if last_t is not None and t > last_t:
                    s.video_hz = 0.9 * s.video_hz + 0.1 / (t - last_t) if s.video_hz else 1.0 / (t - last_t)
                last_t, seq = t, seq + 1
                s.video_frames, s.camera_dropped = seq, self.camera.frames_dropped
                s.frames_decoded = self.camera.frames_decoded
                self._emit(obs)
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_t = time.monotonic()  # behind: don't accumulate debt
        finally:
            perceiver.close()

    def _audio_loop(self) -> None:
        from valence.perception.audio import AudioPerceiver

        perceiver = AudioPerceiver()
        seq = 0
        try:
            while not self._stop.is_set():
                try:
                    block, t, wall = self.mic.read(1.0)
                except Exception:
                    if self._stop.is_set():
                        break
                    raise
                obs = perceiver.process(block, t, wall, seq)
                del block
                seq += 1
                self.stats.audio_blocks, self.stats.mic_overflows = seq, self.mic.overflows
                self._emit(obs)
        finally:
            perceiver.close()

    def __call__(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=15.0)  # long enough for a final process() + MediaPipe close; daemon threads otherwise
        if self.camera is not None:
            self.camera.close()
        if self.mic is not None:
            self.mic.close()


def start(sink: Sink, *, video: bool = True, audio: bool = True, obscure_face: bool = False,
          camera: int | None = None, mic: str | None = None, fps: float | None = None, imgsz: int = 640,
          min_person_h: float | None = None) -> Callable[[], None]:
    """Start live perception; returns ``stop()``. Raises if the camera/mic deliver no real signal.

    Env defaults: VALENCE_CAMERA (index), VALENCE_MIC (name substring); fps and the minimum
    person height come from ``valence.config.settings`` (VALENCE_PERCEPTION_HZ, VALENCE_MIN_BOX_HEIGHT).
    """
    camera = camera if camera is not None else _env_int("VALENCE_CAMERA")
    mic = mic or os.environ.get("VALENCE_MIC")
    fps = settings.perception_hz if fps is None else fps
    if min_person_h is None:
        min_person_h = settings.min_box_height
    return EdgeRunner(sink, video=video, audio=audio, obscure_face=obscure_face, camera=camera, mic=mic,
                      fps=fps, imgsz=imgsz, min_person_h=min_person_h)


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None


def participants_only(obs: VideoObs, rel_height: float | None = None) -> VideoObs:
    """Drop bystanders: keep people whose box is at least ``rel_height`` x the tallest box in the
    frame (the aggregator's participant rule, ``settings.participant_rel_height``)."""
    if len(obs.persons) < 2:
        return obs
    rel = settings.participant_rel_height if rel_height is None else rel_height
    tallest = max(p.bbox[3] - p.bbox[1] for p in obs.persons)
    keep = tuple(p for p in obs.persons if p.bbox[3] - p.bbox[1] >= rel * tallest)
    return obs if len(keep) == len(obs.persons) else dataclasses.replace(obs, persons=keep)


class JsonlSink:
    """Features-only recorder: one ``contracts.obs_to_dict`` per line, text mode, never media.

    Recordings keep only the people at the desk (``participants_only``): a bystander's features
    never reach disk, even though perception tracks them so they can mask the lux background."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self.lines = 0

    def __call__(self, obs: VideoObs | AudioObs) -> None:
        if isinstance(obs, VideoObs):
            obs = participants_only(obs)
        line = json.dumps(obs_to_dict(obs), separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._f.write(line + "\n")
            self.lines += 1

    def close(self) -> None:
        with self._lock:
            self._f.close()


class PrintSink:
    """Compact live readout; events (fist/laugh/compliment) always print, the rest is throttled."""

    def __init__(self, stats: Callable[[], EdgeStats], every: int = 5) -> None:
        self.stats, self.every = stats, every
        self._lock = threading.Lock()

    def __call__(self, obs: VideoObs | AudioObs) -> None:
        if isinstance(obs, VideoObs):
            event = any(p.fist_onsets for p in obs.persons)
            if not event and obs.frame_seq % self.every:
                return
            s = self.stats()
            people = " | ".join(_fmt_person(p) for p in obs.persons) or "no one"
            line = (f"[V {obs.frame_seq:06d} {s.video_hz:4.1f} Hz {s.video_ms:3.0f} ms] "
                    f"lux {_f(obs.lux, '.0f')} (q {obs.lux_quality:.2f}) | {people}")
        else:
            events = []
            if obs.laughter_onsets:
                events.append(f"LAUGH +{obs.laughter_onsets}")
            if obs.compliment_hits:
                events.append(f"COMPLIMENT +{obs.compliment_hits}")
            line = (f"[A {obs.block_seq:06d}] {obs.db_spl:5.1f} dB  speech {obs.speech_prob:.2f}  "
                    + "  ".join(events))
        with self._lock:
            print(line, flush=True)


def _f(x: float, fmt: str) -> str:
    return "  nan" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _fmt_person(p) -> str:
    flags = [f"fist{'*' if p.fist_active else ''}{'+' + str(p.fist_onsets) if p.fist_onsets else ''}"
             if (p.fist_active or p.fist_onsets) else "", "SLUMP" if p.slump else "", "HEAD-IN-HANDS" if p.head_in_hands else ""]
    face = f"face {p.face_valence:+.2f}" if p.face_visible else "face --"
    return f"P{p.track_id} {face} m {p.motion:.2f} {' '.join(f for f in flags if f)}".rstrip()


def _rss_mb() -> float:
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                                  text=True, timeout=2).stdout.strip()) / 1024
    except Exception:
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m valence.edge.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", action="store_true", help="enable camera (default: both if neither given)")
    ap.add_argument("--audio", action="store_true", help="enable mic (default: both if neither given)")
    ap.add_argument("--print", dest="print_", action="store_true", help="live readout to stdout")
    ap.add_argument("--print-every", type=int, default=5, help="print every Nth video obs (events always)")
    ap.add_argument("--jsonl", type=Path, help="record features-only obs to this JSONL (e.g. data/real/focused.jsonl)")
    ap.add_argument("--obscure-face", action="store_true", help="black out faces before MediaPipe (anti-bias proof)")
    ap.add_argument("--camera", type=int, default=None, help="AVFoundation index (env VALENCE_CAMERA)")
    ap.add_argument("--mic", default=None, help="input device name substring (env VALENCE_MIC)")
    ap.add_argument("--fps", type=float, default=None, help="perception rate (default settings.perception_hz)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-person-h", type=float, default=None,
                    help="ignore people shorter than this share of the frame (e.g. 0.35: near the camera only)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--tracemalloc", action="store_true", help="report Python heap + RSS every 30 s")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    video, audio = (a.video, a.audio) if (a.video or a.audio) else (True, True)

    if a.tracemalloc:
        import tracemalloc

        tracemalloc.start()
    sinks: list[Callable[[Any], None]] = []
    jsonl = JsonlSink(a.jsonl) if a.jsonl else None
    if jsonl:
        sinks.append(jsonl)
    runner_ref: list[EdgeRunner] = []
    if a.print_:
        sinks.append(PrintSink(lambda: runner_ref[0].stats if runner_ref else EdgeStats(), every=a.print_every))

    def fanout(obs: Any) -> None:
        for s in sinks:
            s(obs)

    try:
        runner = start(fanout, video=video, audio=audio, obscure_face=a.obscure_face, camera=a.camera,
                       mic=a.mic, fps=a.fps, imgsz=a.imgsz, min_person_h=a.min_person_h)
    except RuntimeError as e:
        print(f"EDGE FAIL: {e}", file=sys.stderr)
        return 2
    runner_ref.append(runner)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    t0 = last_mem = time.monotonic()
    try:
        while not stop.is_set():
            stop.wait(0.5)
            now = time.monotonic()
            if runner.stats.errors:
                print(f"EDGE FAIL: {runner.stats.errors}", file=sys.stderr)
                return 3
            if a.tracemalloc and now - last_mem >= 30:
                import tracemalloc

                cur, peak = tracemalloc.get_traced_memory()
                s = runner.stats
                print(f"[MEM t={now - t0:6.0f}s] py heap {cur / 1e6:.1f} MB (peak {peak / 1e6:.1f})  rss {_rss_mb():.0f} MB  "
                      f"video {s.video_frames} @ {s.video_hz:.1f} Hz  dropped {s.camera_dropped}  audio {s.audio_blocks}",
                      flush=True)
                last_mem = now
            if a.duration and now - t0 >= a.duration:
                break
    finally:
        runner()
        if jsonl:
            jsonl.close()
            print(f"wrote {jsonl.lines} obs -> {a.jsonl}", file=sys.stderr)
    s = runner.stats
    print(f"done: video {s.video_frames} frames @ {s.video_hz:.1f} Hz ({s.video_ms:.0f} ms), audio {s.audio_blocks} blocks, "
          f"sink errors {s.sink_errors}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
