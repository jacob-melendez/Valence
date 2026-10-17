"""Guided live feature capture (Step 3.5) + perception probes. Features only, never media.

Run from Terminal.app (Claude Code's shell has no camera permission):

    python scripts/spikes/live_capture.py                 # full session, ~14 min
    python scripts/spikes/live_capture.py --only typing   # one segment
    python scripts/spikes/live_capture.py --seconds 60    # shorter segments

One edge runner spans the whole session so per-person baselines (posture, face) are set once
during the first segment. Each segment is written to ``data/real/capture_<name>.jsonl`` and a
metrics summary to ``data/real/capture_summary.json``:

- fist onsets per minute while typing / looking at the keyboard (false positives),
- fist recall on cued clenches, slump duty cycle (and agreement with cued slumps),
- motion distribution, face visibility, video Hz, audio dB / speech / laughter / compliments.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from valence.contracts import AudioObs, VideoObs
from valence.edge.run import JsonlSink, start

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "real"


@dataclass
class Segment:
    name: str
    seconds: float
    intro: str
    cues: list[tuple[float, float, str]] = field(default_factory=list)  # (start, duration, text)
    idle: str = ""


def fist_cues(total: float, every: float = 12.0, hold: float = 2.5) -> list[tuple[float, float, str]]:
    return [(t, hold, ">>> CLENCH BOTH FISTS <<<") for t in _arange(6.0, total - hold, every)]


def slump_cues(total: float, down: float = 20.0, up: float = 10.0) -> list[tuple[float, float, str]]:
    return [(t, down, ">>> SLUMP: head down, shoulders forward <<<") for t in _arange(5.0, total - 5.0, down + up)]


def _arange(a: float, b: float, step: float) -> list[float]:
    out, t = [], a
    while t <= b:
        out.append(t)
        t += step
    return out


def segments(sec: float) -> list[Segment]:
    return [
        Segment("baseline", 25, "Sit UPRIGHT, NEUTRAL face, look at the screen, hands visible. Sets your baselines.",
                idle="hold still, neutral face"),
        Segment("typing", sec, "Type normally on the keyboard (any text, in any window). Quiet room.",
                idle="keep typing normally"),
        Segment("keyboard_gaze", sec, "Look DOWN at the keyboard while typing or reading (not slumping on purpose).",
                idle="eyes on the keyboard"),
        Segment("fists", sec, "Sit normally, hands visible ABOVE the keyboard. Clench both fists ONLY when cued.",
                cues=fist_cues(sec), idle="relax hands (open)"),
        Segment("slump", sec, "Slump when cued (head down, shoulders forward); sit upright otherwise.",
                cues=slump_cues(sec), idle="sit upright"),
        Segment("covered", 15, "COVER the camera with your hand or a sticky note for the whole segment.",
                idle="keep the camera covered"),
        Segment("audio", 45, "Say 'great job' three times (pause between), then LAUGH for ~5 s, then stay quiet.",
                idle="'great job' x3, then laugh"),
    ]


class Recorder:
    """Routes observations to the current segment's sink and keeps them for the summary."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sink: JsonlSink | None = None
        self.obs: list = []
        self.t0 = 0.0

    def begin(self, path: Path) -> None:
        with self.lock:
            self.sink, self.obs, self.t0 = JsonlSink(path), [], time.monotonic()

    def end(self) -> list:
        with self.lock:
            sink, obs, self.sink = self.sink, self.obs, None
        if sink:
            sink.close()
        return obs

    def __call__(self, o) -> None:
        with self.lock:
            if self.sink is not None:
                self.sink(o)
                self.obs.append(o)


def main_person(video: list[VideoObs]) -> int | None:
    counts: dict[int, int] = {}
    for v in video:
        for p in v.persons:
            counts[p.track_id] = counts.get(p.track_id, 0) + 1
    return max(counts, key=counts.get) if counts else None


def pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else math.nan


def summarize(seg: Segment, obs: list, t0: float) -> dict:
    video = [o for o in obs if isinstance(o, VideoObs)]
    audio = [o for o in obs if isinstance(o, AudioObs)]
    s: dict = {"segment": seg.name, "seconds": seg.seconds, "video_obs": len(video), "audio_obs": len(audio)}
    if len(video) > 1:
        s["video_hz"] = round((len(video) - 1) / (video[-1].t - video[0].t), 2)
    pid = main_person(video)
    s["people_seen"] = sorted({p.track_id for v in video for p in v.persons})
    if pid is not None:
        ps = [(v.t - t0, p) for v in video for p in v.persons if p.track_id == pid]
        minutes = seg.seconds / 60
        fist_t = [t for t, p in ps for _ in range(p.fist_onsets)]
        s.update({
            "person": pid, "present_frac": round(len(ps) / max(1, len(video)), 3),
            "fist_onsets": len(fist_t), "fist_onsets_per_min": round(len(fist_t) / minutes, 2),
            "slump_duty": round(sum(p.slump for _, p in ps) / len(ps), 3),
            "head_in_hands_frac": round(sum(p.head_in_hands for _, p in ps) / len(ps), 3),
            "face_visible_frac": round(sum(p.face_visible for _, p in ps) / len(ps), 3),
            "face_valence_p10_p50_p90": [round(pct([p.face_valence for _, p in ps], q), 3) for q in (0.1, 0.5, 0.9)],
            "motion_p10_p50_p90": [round(pct([p.motion for _, p in ps], q), 3) for q in (0.1, 0.5, 0.9)],
        })
        if seg.name == "fists" and seg.cues:
            hit = sum(any(c <= t <= c + d + 1.5 for t in fist_t) for c, d, _ in seg.cues)
            fp = sum(not any(c <= t <= c + d + 1.5 for c, d, _ in seg.cues) for t in fist_t)
            s.update({"cued_clenches": len(seg.cues), "fist_recall": round(hit / len(seg.cues), 3),
                      "fist_false_onsets": fp})
        if seg.name == "slump" and seg.cues:
            def cued(t: float) -> bool:
                return any(c + 3.5 <= t <= c + d for c, d, _ in seg.cues)  # allow the 3 s hold
            inside = [p.slump for t, p in ps if cued(t)]
            outside = [p.slump for t, p in ps if not any(c <= t <= c + d + 2 for c, d, _ in seg.cues)]
            s.update({"slump_hit_rate_in_cue": round(sum(inside) / max(1, len(inside)), 3),
                      "slump_false_rate_upright": round(sum(outside) / max(1, len(outside)), 3)})
    if audio:
        s.update({
            "db_p10_p50_p90": [round(pct([a.db_spl for a in audio], q), 1) for q in (0.1, 0.5, 0.9)],
            "speech_p50": round(pct([a.speech_prob for a in audio], 0.5), 3),
            "laughter_onsets": sum(a.laughter_onsets for a in audio),
            "compliment_hits": sum(a.compliment_hits for a in audio),
        })
    if video:
        s["lux_p50"] = round(pct([v.lux for v in video], 0.5), 1)
        s["frames_decoded_last"] = video[-1].frames_decoded
    return s


def run_segment(seg: Segment, rec: Recorder, out_dir: Path, ask: bool = True) -> dict:
    print(f"\n=== {seg.name.upper()} ({seg.seconds:.0f} s) ===\n{seg.intro}")
    if ask and input("Press Enter to start (s + Enter to skip): ").strip().lower() == "s":
        return {"segment": seg.name, "skipped": True}
    for n in (3, 2, 1):
        print(f"  starting in {n}...", flush=True)
        time.sleep(1)
    path = out_dir / f"capture_{seg.name}.jsonl"
    rec.begin(path)
    t0 = rec.t0
    last = ""
    while (el := time.monotonic() - t0) < seg.seconds:
        cue = next((txt for c, d, txt in seg.cues if c <= el < c + d), None)
        msg = cue or seg.idle
        line = f"\r  [{seg.name} {int(el) // 60}:{int(el) % 60:02d} / {int(seg.seconds) // 60}:{int(seg.seconds) % 60:02d}]  {msg:<52}"
        if cue and cue != last:
            sys.stdout.write("\a")
        last = cue or ""
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(0.1)
    obs = rec.end()
    summary = summarize(seg, obs, t0)
    print(f"\n  done -> {path}  ({len(obs)} obs)")
    print("  " + json.dumps({k: v for k, v in summary.items() if k != "segment"}))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0, help="length of the 3-min segments")
    ap.add_argument("--only", nargs="*", help="segment names to run (baseline always runs first)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--yes", action="store_true", help="start every segment without waiting for Enter")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where the JSONL + summary go")
    a = ap.parse_args()

    segs = segments(a.seconds)
    if a.only:
        segs = [s for s in segs if s.name == "baseline" or s.name in a.only]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder()
    print("Starting camera + mic (models load ~5 s)...")
    stop = start(rec, video=True, audio=not a.no_audio)
    time.sleep(4.0)
    results = []
    try:
        for seg in segs:
            results.append(run_segment(seg, rec, a.out_dir, ask=not a.yes))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        rec.end()
        stop()
    out = a.out_dir / "capture_summary.json"
    out.write_text(json.dumps({"captured_at": time.strftime("%Y-%m-%d %H:%M"), "segments": results}, indent=2) + "\n")
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
