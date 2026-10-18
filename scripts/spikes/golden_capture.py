"""Golden demo recording (P0-3): one continuous features-only capture with cued segments.

Run from Terminal.app (Claude Code's shell has no camera permission):

    python scripts/spikes/golden_capture.py                          # full session, ~20 min
    python scripts/spikes/golden_capture.py --minutes-scale 0.1 \\
        --out-dir /tmp/golden_dry --yes                              # 2-min dry run, no prompts
    python scripts/spikes/golden_capture.py --force                  # overwrite an earlier take

One edge runner and ONE ``JsonlSink`` span the whole session, so ``<out-dir>/golden_demo.jsonl``
holds contract obs lines only (participants only, never media). Segments:

1. neutral (90 s): the baseline;
2. typing (4 min): focused typing;
3. masked (8 min): phone hall noise at ~65-70 dB, a broad smile HELD throughout, a two-handed fist
   clench on each cue (seeded random 45-60 s apart), a sigh on each sigh cue (~every 90 s);
4. break (2 min): away from the desk;
5. laugh (3.5 min): laughing and chatting, 2-3 compliments ("great job", "nice work", "well done").

Segment boundaries and cues live only in ``<out-dir>/golden_demo_summary.json``, with per-segment
metrics computed from the in-memory obs by time range (face valence p10/p50/p90, fist onsets vs
cues, dB, people seen, laughter, compliments, video Hz). Replay it with
``scripts/spikes/replay_check.py <jsonl> --summary <summary>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from valence.contracts import AudioObs, VideoObs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "real"
NAME = "golden_demo"
SEED = 7
FIST_HOLD_S = 3.0
SIGH_HOLD_S = 4.0
CUE_GRACE_S = 2.5  # a fist onset this long after the cue ends still counts as the cued clench
SMILE_COLLAB_VALENCE = 0.4  # face-only model reads Collaborative >= 0.5 only from about here up


@dataclass(frozen=True)
class Cue:
    at: float  # seconds from segment start
    hold: float
    kind: str  # "fist" | "sigh" | "compliment" | "laugh" | "return"
    text: str


@dataclass
class Segment:
    name: str
    seconds: float
    title: str
    intro: str
    idle: str
    cues: list[Cue] = field(default_factory=list)
    before: str = ""  # prompt shown (with a bell) before the segment's Enter
    after: str = ""  # prompt shown (with a bell) when the segment ends


def fist_schedule(seconds: float, *, seed: int = SEED, scale: float = 1.0, lo: float = 45.0,
                  hi: float = 60.0, hold: float = FIST_HOLD_S, min_gap: float = 10.0) -> list[float]:
    """Cue times for two-handed clenches: seeded random gaps in [lo, hi] (x scale, never under ``min_gap``
    so a shrunken dry run keeps cues apart), each cue ending inside the segment."""
    rng = random.Random(seed)
    out, t = [], max(min_gap, rng.uniform(lo, hi) * scale)
    while t + hold <= seconds:
        out.append(round(t, 2))
        t += max(min_gap, rng.uniform(lo, hi) * scale)
    return out


def sigh_schedule(seconds: float, fists: list[float], *, scale: float = 1.0, every: float = 90.0,
                  first: float = 30.0, hold: float = SIGH_HOLD_S, clear: float = 6.0) -> list[float]:
    """Sigh cues about every ``every`` s, nudged later so they never overlap a fist cue (+/- ``clear``)."""
    out, t, step = [], first * scale, max(20.0, every * scale)
    while t + hold <= seconds:
        s = t
        while any(abs(s - f) < clear for f in fists):
            s += clear
        if s + hold <= seconds:
            out.append(round(s, 2))
        t += step
    return out


def active_cue(cues: list[Cue], el: float) -> Cue | None:
    """The cue to show at ``el`` s: the latest-started one still running, so an overlapping later cue is
    never hidden behind an earlier one (it rings and shows at its own start)."""
    live = [c for c in cues if c.at <= el < c.at + c.hold]
    return max(live, key=lambda c: c.at) if live else None


def _fit(at: float, hold: float, seconds: float) -> float:
    """A cue start that keeps the whole cue inside the segment."""
    return round(max(0.0, min(at, seconds - hold)), 2)


def build_segments(scale: float = 1.0, seed: int = SEED) -> list[Segment]:
    """The five cued segments; ``scale`` shrinks every duration and cue gap (0.1 = dry run)."""
    masked_s = 480 * scale
    fists = fist_schedule(masked_s, seed=seed, scale=scale)
    sighs = sigh_schedule(masked_s, fists, scale=scale)
    masked_cues = sorted(
        [Cue(t, FIST_HOLD_S, "fist", ">>> CLENCH BOTH FISTS (keep smiling) <<<") for t in fists]
        + [Cue(t, SIGH_HOLD_S, "sigh", ">>> SIGH, audibly (keep smiling) <<<") for t in sighs],
        key=lambda c: c.at)
    laugh_s = 210 * scale
    laugh_cues = sorted(
        [Cue(_fit(f * laugh_s, 6.0, laugh_s), 6.0, "laugh", ">>> LAUGH together <<<") for f in (0.12, 0.42, 0.72)]
        + [Cue(_fit(f * laugh_s, 4.0, laugh_s), 4.0, "compliment", f">>> SAY '{p}' <<<")
           for f, p in ((0.27, "great job"), (0.57, "nice work"), (0.85, "well done"))],
        key=lambda c: c.at)
    break_s = 120 * scale
    return [
        Segment("neutral", 90 * scale, "NEUTRAL BASELINE",
                "Sit UPRIGHT, NEUTRAL face, look at the screen, hands visible. Sets your baselines.",
                "hold still, neutral face"),
        Segment("typing", 240 * scale, "FOCUSED TYPING",
                "Type normally on the keyboard, in ANOTHER window (not this Terminal). Quiet room.", "keep typing, focused"),
        Segment("masked", masked_s, "MASKED ACT",
                "BROAD SMILE held the WHOLE time. Hands visible above the keyboard. Clench both fists and sigh "
                "ONLY when cued; otherwise relax your hands.",
                "keep the broad smile", masked_cues,
                before="PUT THE PHONE HALL-NOISE ON NOW (at the desk, aim for 65-70 dB on the readout)",
                after="STOP THE PHONE HALL-NOISE NOW"),
        Segment("break", break_s, "BREAK: LEAVE THE DESK",
                "Get up and walk OUT of the camera's view. Come back when the bell rings.",
                "stay away from the desk",
                [Cue(_fit(break_s - 5.0, 5.0, break_s), 5.0, "return", ">>> COME BACK to the desk <<<")]),
        Segment("laugh", laugh_s, "LAUGHING AND CHATTING",
                "Chat and laugh (with a second person if you can). Say the compliments when cued.",
                "chat and laugh naturally", laugh_cues),
    ]


def output_paths(out_dir: Path, name: str = NAME, force: bool = False) -> tuple[Path, Path]:
    """The JSONL and summary paths; refuses (FileExistsError) to overwrite either unless ``force``."""
    jsonl, summary = out_dir / f"{name}.jsonl", out_dir / f"{name}_summary.json"
    existing = [str(p) for p in (jsonl, summary) if p.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite {', '.join(existing)} (pass --force)")
    return jsonl, summary


# --- metrics ------------------------------------------------------------------------------------------

def pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else math.nan


def _q3(xs: list[float], nd: int = 3) -> list[float | None]:
    return [_r(pct(xs, q), nd) for q in (0.1, 0.5, 0.9)]


def _r(x: float, nd: int = 3) -> float | None:
    return None if x is None or math.isnan(x) else round(x, nd)


def _in_cue(t: float, cues: list[Cue], grace: float = 0.0) -> bool:
    return any(c.at <= t <= c.at + c.hold + grace for c in cues)


def segment_metrics(obs: list, t_start: float, t_end: float, cues: list[Cue] | None = None) -> dict[str, Any]:
    """Per-segment metrics from obs with ``t_start <= t < t_end`` (perception monotonic clock)."""
    cues = [c for c in cues or [] if c.at < t_end - t_start]  # an interrupted segment never showed later cues
    sel = [o for o in obs if t_start <= o.t < t_end]
    video = [o for o in sel if isinstance(o, VideoObs)]
    audio = [o for o in sel if isinstance(o, AudioObs)]
    m: dict[str, Any] = {"video_obs": len(video), "audio_obs": len(audio)}
    if len(video) > 1 and video[-1].t > video[0].t:
        m["video_hz"] = round((len(video) - 1) / (video[-1].t - video[0].t), 2)
    frames = Counter(p.track_id for v in video for p in v.persons)
    m["people_seen"] = sorted(frames)
    m["frames_per_person"] = {str(k): v for k, v in sorted(frames.items())}
    m["max_people_in_frame"] = max((len(v.persons) for v in video), default=0)
    m["present_frac"] = round(sum(bool(v.persons) for v in video) / len(video), 3) if video else 0.0
    if frames:
        pid = frames.most_common(1)[0][0]
        ps = [(v.t - t_start, p) for v in video for p in v.persons if p.track_id == pid]
        vis = [(t, p.face_valence) for t, p in ps if p.face_visible and not math.isnan(p.face_valence)]
        vals = [x for _, x in vis]
        fist_t = [round(t, 2) for t, p in ps for _ in range(p.fist_onsets)]
        minutes = max(1e-9, (t_end - t_start) / 60)
        m.update({
            "person": pid,
            "face_visible_frac": round(len(vis) / len(ps), 3),
            "face_valence_p10_p50_p90": _q3(vals),
            "face_valence_ge_0_4_frac": round(sum(x >= SMILE_COLLAB_VALENCE for x in vals) / len(vals), 3)
            if vals else None,
            "motion_p10_p50_p90": _q3([p.motion for _, p in ps]),
            "slump_duty": round(sum(p.slump for _, p in ps) / len(ps), 3),
            "head_in_hands_frac": round(sum(p.head_in_hands for _, p in ps) / len(ps), 3),
            "fist_onsets": len(fist_t),
            "fist_onsets_per_min": round(len(fist_t) / minutes, 2),
            "fist_onsets_s": fist_t,
        })
        if cues:
            m["face_valence_outside_cues_p10_p50_p90"] = _q3(
                [x for t, x in vis if not _in_cue(t, cues, CUE_GRACE_S)])
        fist_cues = [c for c in cues if c.kind == "fist"]
        if fist_cues:
            lat = []
            for c in fist_cues:
                hit = [t for t in fist_t if c.at <= t <= c.at + c.hold + CUE_GRACE_S]
                lat.append(round(hit[0] - c.at, 2) if hit else None)
            m.update({
                "fist_cues": len(fist_cues),
                "fist_cue_latency_s": lat,
                "fist_recall": round(sum(x is not None for x in lat) / len(fist_cues), 3),
                "fist_false_onsets": sum(not _in_cue(t, fist_cues, CUE_GRACE_S) for t in fist_t),
            })
    if audio:
        m.update({
            "db_p10_p50_p90": _q3([a.db_spl for a in audio], 1),
            "speech_p50": _r(pct([a.speech_prob for a in audio], 0.5)),
            "laughter_onsets": sum(a.laughter_onsets for a in audio),
            "laughter_s": [round(a.t - t_start, 1) for a in audio for _ in range(a.laughter_onsets)],
            "compliment_hits": sum(a.compliment_hits for a in audio),
            "compliment_s": [round(a.t - t_start, 1) for a in audio for _ in range(a.compliment_hits)],
        })
    if video:
        m["lux_p50"] = _r(pct([v.lux for v in video], 0.5), 1)
    return m


def build_summary(obs: list, records: list[dict[str, Any]], segs: list[Segment], *, jsonl: Path,
                  scale: float, seed: int, edge: dict[str, Any] | None = None) -> dict[str, Any]:
    """Session summary. ``records``: {"name", "skipped", "t_start", "t_end"} per segment (monotonic t).

    ``start_s``/``end_s`` are relative to the first JSONL line, which is where offline replays start."""
    by_name = {s.name: s for s in segs}
    t0 = obs[0].t if obs else None
    out_segs = []
    for r in records:
        seg = by_name[r["name"]]
        d: dict[str, Any] = {"segment": seg.name, "planned_s": round(seg.seconds, 1), "skipped": r["skipped"]}
        if not r["skipped"]:
            d.update({
                "t_start": r["t_start"], "t_end": r["t_end"],
                "start_s": _r(r["t_start"] - t0, 2) if t0 is not None else None,
                "end_s": _r(r["t_end"] - t0, 2) if t0 is not None else None,
                "seconds": round(r["t_end"] - r["t_start"], 1),
                "complete": r.get("complete", True),
                "cues": [{"kind": c.kind, "at_s": c.at, "hold_s": c.hold} for c in seg.cues
                         if c.at < r["t_end"] - r["t_start"]],
                **segment_metrics(obs, r["t_start"], r["t_end"], seg.cues),
            })
        out_segs.append(d)
    video = [o for o in obs if isinstance(o, VideoObs)]
    summary: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M"),
        "file": _rel(jsonl), "obs": len(obs), "video_obs": len(video), "audio_obs": len(obs) - len(video),
        "duration_s": round(obs[-1].t - t0, 1) if obs else 0.0,
        "video_hz": round((len(video) - 1) / (video[-1].t - video[0].t), 2)
        if len(video) > 1 and video[-1].t > video[0].t else None,
        "minutes_scale": scale, "seed": seed, "t0": t0,
        "segments": out_segs,
    }
    masked = next((s for s in out_segs if s["segment"] == "masked" and not s["skipped"]), None)
    if masked:
        summary["held_smile"] = {k: masked.get(k) for k in (
            "face_valence_p10_p50_p90", "face_valence_outside_cues_p10_p50_p90", "face_valence_ge_0_4_frac",
            "face_visible_frac", "fist_recall", "fist_false_onsets", "db_p10_p50_p90")}
    if edge is not None:
        summary["edge"] = edge
    return _finite(summary)


def _finite(x: Any) -> Any:
    """Strict-JSON copy: NaN/inf become None, so the end-of-session summary write can never fail."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# --- live session ---------------------------------------------------------------------------------------

class Recorder:
    """Sink: writes every obs to one JSONL sink and keeps it in memory, plus a tiny live readout.

    ``keep`` (``participants_only`` in a live run) filters video obs before they are kept, so the summary
    sees exactly the people the JSONL does: a bystander's features never reach either file."""

    def __init__(self, sink, keep: Callable[[VideoObs], VideoObs] | None = None) -> None:
        self.lock = threading.Lock()
        self.sink = sink
        self.keep = keep
        self.obs: list = []
        self.db = math.nan
        self.valence = math.nan
        self.people = 0

    def __call__(self, o) -> None:
        if self.keep is not None and isinstance(o, VideoObs):
            o = self.keep(o)
        with self.lock:
            self.sink(o)
            self.obs.append(o)
            if isinstance(o, AudioObs):
                self.db = o.db_spl
            elif isinstance(o, VideoObs):
                self.people = len(o.persons)
                big = max(o.persons, key=lambda p: p.bbox[3] - p.bbox[1], default=None)
                self.valence = big.face_valence if big is not None and big.face_visible else math.nan

    def readout(self) -> str:
        with self.lock:
            db, val, n = self.db, self.valence, self.people
        db_s = "  --" if math.isnan(db) else f"{db:4.0f}"
        val_s = "  --" if math.isnan(val) else f"{val:+.2f}"
        return f"room {db_s} dB | face {val_s} | people {n}"


def bell() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


def banner(text: str) -> None:
    bar = "#" * max(48, len(text) + 8)
    print(f"\n{bar}\n##  {text.upper():<{len(bar) - 8}}  ##\n{bar}", flush=True)


def _flush_stdin() -> None:
    """Drop keystrokes typed during a segment (the typing segment), so a buffered line can't start or skip
    the next segment."""
    try:
        import termios

        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, OSError, ValueError):
        pass


def wait_enter(rec: Recorder, prompt: str, check) -> str:
    """Wait for Enter while showing the live readout (so noise level / smile can be set up). Returns the line."""
    got: list[str] = []
    done = threading.Event()
    _flush_stdin()

    def reader() -> None:
        try:
            got.append(input())
        except EOFError:
            got.append("")
        done.set()

    print(prompt, flush=True)
    threading.Thread(target=reader, name="enter", daemon=True).start()
    while not done.wait(0.5):
        check()
        sys.stdout.write(f"\r  (waiting)  {rec.readout():<44}")
        sys.stdout.flush()
    print()
    return got[0].strip().lower()


def run_segment(seg: Segment, rec: Recorder, ask: bool, check, records: list[dict[str, Any]]) -> None:
    """Show the cues for one segment; appends its record first so an interrupted segment is still summarized."""
    if seg.before:
        bell()
        banner(seg.before)
    banner(f"{seg.title}  ({seg.seconds / 60:.1f} min)")
    print(seg.intro, flush=True)
    if ask and wait_enter(rec, "Press Enter to start (s + Enter to skip):", check) == "s":
        if seg.after:
            bell()
            banner(seg.after)
        records.append({"name": seg.name, "skipped": True})
        return
    for n in (3, 2, 1):
        print(f"  starting in {n}...", flush=True)
        time.sleep(1)
    bell()
    t_start = time.monotonic()
    rec_d: dict[str, Any] = {"name": seg.name, "skipped": False, "t_start": t_start, "t_end": t_start,
                             "complete": False}
    records.append(rec_d)
    active: Cue | None = None
    try:
        while (el := time.monotonic() - t_start) < seg.seconds:
            check()
            cue = active_cue(seg.cues, el)
            if cue is not None and cue is not active:
                bell()
            active = cue
            left = max(0, int(seg.seconds - el))
            msg = cue.text if cue else seg.idle
            sys.stdout.write(f"\r  [{seg.name} {left // 60}:{left % 60:02d} left]  {msg:<44} {rec.readout()}  ")
            sys.stdout.flush()
            time.sleep(0.1)
        rec_d["complete"] = True
    finally:
        rec_d["t_end"] = time.monotonic()
    print()
    if seg.after:
        bell()
        banner(seg.after)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes-scale", type=float, default=1.0, help="scale every duration (0.1 = dry run)")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where the JSONL + summary go (default data/real)")
    ap.add_argument("--name", default=NAME, help="file stem (default golden_demo)")
    ap.add_argument("--seed", type=int, default=SEED, help="fist-cue schedule seed")
    ap.add_argument("--yes", action="store_true", help="start every segment without waiting for Enter")
    ap.add_argument("--force", action="store_true", help="overwrite an existing recording/summary")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)
    if a.minutes_scale <= 0:
        ap.error("--minutes-scale must be > 0")
    try:
        jsonl, summary_path = output_paths(a.out_dir, a.name, a.force)
    except FileExistsError as e:
        print(f"golden_capture: {e}", file=sys.stderr)
        return 2
    segs = build_segments(a.minutes_scale, a.seed)
    total = sum(s.seconds for s in segs)
    print(f"Golden demo capture: {len(segs)} segments, {total / 60:.1f} min of cued time -> {jsonl}")

    from valence.edge.run import JsonlSink, participants_only, start  # torch first (libomp), only when capturing

    # Record into a side file and move it into place at the end: a failed start never leaves a file that
    # blocks the retry, and never clobbers an earlier take when --force is passed.
    partial = jsonl.with_name(jsonl.name + ".partial")
    sink = JsonlSink(partial)
    rec = Recorder(sink, keep=participants_only)
    print("Starting camera + mic (models load ~5 s)...", flush=True)
    try:
        runner = start(rec, video=True, audio=not a.no_audio)
    except BaseException as e:
        sink.close()
        partial.unlink(missing_ok=True)  # only this run's own empty side file
        if not isinstance(e, Exception):
            raise
        print(f"EDGE FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    def check() -> None:
        errs = getattr(getattr(runner, "stats", None), "errors", None)
        if errs:
            raise RuntimeError(f"edge failed: {errs}")

    records: list[dict[str, Any]] = []
    code = 0
    try:
        time.sleep(4.0)
        for seg in segs:
            run_segment(seg, rec, not a.yes, check, records)
    except KeyboardInterrupt:
        print("\ninterrupted: writing what was captured")
        code = 130
    except RuntimeError as e:
        print(f"\nEDGE FAIL: {e}", file=sys.stderr)
        code = 3
    except Exception as e:  # never lose a long take to a display bug: stop, then summarize what we have
        print(f"\nFAIL: {type(e).__name__}: {e}", file=sys.stderr)
        code = 1
    finally:
        runner()
        sink.close()
        if records:
            os.replace(partial, jsonl)
        else:  # no segment ever started: keep any earlier take, drop this run's warm-up-only side file
            partial.unlink(missing_ok=True)
    if not records:
        print("nothing recorded (no segment started)", file=sys.stderr)
        return code
    with rec.lock:
        obs = list(rec.obs)
    st = getattr(runner, "stats", None)
    edge = None if st is None else {k: getattr(st, k) for k in (
        "video_frames", "video_hz", "camera_dropped", "frames_decoded", "audio_blocks", "mic_overflows",
        "sink_errors") if hasattr(st, k)}
    if edge is not None:
        edge["video_hz"] = round(edge.get("video_hz", 0.0), 2)
        edge["errors"] = len(getattr(st, "errors", []))
    summary = build_summary(obs, records, segs, jsonl=jsonl, scale=a.minutes_scale, seed=a.seed, edge=edge)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nwrote {sink.lines} obs -> {jsonl}\nsummary -> {summary_path}")
    for s in summary["segments"]:
        keep = ("seconds", "video_hz", "people_seen", "face_valence_p10_p50_p90", "fist_recall", "fist_onsets",
                "db_p10_p50_p90", "laughter_onsets", "compliment_hits")
        print(f"  {s['segment']:<8} " + ("skipped" if s["skipped"] else json.dumps({k: s[k] for k in keep if k in s})))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
