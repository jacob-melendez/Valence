"""Runtime.push_obs: the Tracks messages it publishes, its fps, mic and camera flags.

Python's side of "publishes tracks and measures fps exactly as push_obs does" in
frontend/src/test/engine-runtime.test.ts: a Runtime built at monotonic 10.0 (wall fixed at WALL0) is pushed one
VideoObs per row of SEQ (frame_seq = frames_decoded = i, mirrored = i odd, wall = WALL0 + mono * 1.001), then an
AudioObs at monotonic 15.0. The predictor is never asked.

    .venv/bin/python -B scripts/golden_runtime/tracks.py   # JSON: {fps, tracks} per push, then {fps, mic, camera}

The test pastes, per push, [fps, null or [tracks.t, tracks.capture_ts_ms, tracks.mirrored, person ids]], then
[fps, mic, camera].
"""

from __future__ import annotations

import json
import tempfile
import types

from _common import Bus, Clock

import valence.runtime.aggregator as agg_mod  # noqa: E402
import valence.runtime.engine as eng_mod  # noqa: E402
from valence.config import Settings  # noqa: E402
from valence.contracts import AudioObs, PersonObs, VideoObs  # noqa: E402
from valence.runtime.engine import Runtime  # noqa: E402

WALL0 = 1792850400.0


def P(tid, y1, y2, **kw):
    return PersonObs(track_id=tid, bbox=(0.1, y1, 0.2, y2), **kw)


# (monotonic time, persons) per push
SEQ = [
    (10.05, [P(1, 0.1, 0.6), P(2, 0.5, 0.7, fist_active=True), P(3, 0.2, 0.5, slump=True), P(4, 0.0, 0.275)]),
    (10.1, [P(1, 0.1, 0.6)]),
    (10.15, [P(1, 0.1, 0.6)]),  # 10.15 - 10.05 = 0.0999999...: still throttled
    (10.1500001, [P(1, 0.1, 0.6)]),
    (12.0, []),
    (12.05, [P(5, 0.3, 0.3), P(6, 0.3, 0.3)]),
    (12.2, [P(7, 0.0, 1.0), P(8, 0.0, 0.55), P(9, 0.0, 0.5499999)]),
    (14.2, [P(1, 0.1, 0.2)]),
]


def main() -> None:
    clock = Clock(WALL0, 10.0)
    orig = (eng_mod.time, agg_mod.time)
    eng_mod.time = agg_mod.time = clock
    stub = types.SimpleNamespace(states=("Focused",), model_id="x", predict=lambda p, r: None)
    out = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bus = Bus()
            rt = Runtime(Settings(), bus, stub, audit_dir=tmp)
            for i, (m, persons) in enumerate(SEQ):
                clock.mono = m
                bus.dumps.clear()
                rt.push_obs(VideoObs(t=m, wall=WALL0 + m * 1.001, frame_seq=i, frames_decoded=i, frame_w=640,
                                     frame_h=480, mirrored=bool(i % 2), persons=tuple(persons)))
                out.append({"fps": rt.fps, "tracks": bus.dumps[0] if bus.dumps else None})
            clock.mono = 15.0
            rt.push_obs(AudioObs(t=15.0, wall=WALL0 + 15.0, block_seq=0, db_spl=50.0, rms_power=1e-3))
            out.append({"fps": rt.fps, "mic": rt.mic, "camera": rt.camera})
    finally:
        eng_mod.time, agg_mod.time = orig
    print(json.dumps(out))


if __name__ == "__main__":
    main()
