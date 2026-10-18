"""Offline replay of a features-only recording through the core runtime, with a per-tick timeline.

    python scripts/spikes/replay_check.py data/real/golden_demo.jsonl --summary data/real/golden_demo_summary.json
    python scripts/spikes/replay_check.py data/real/capture_fists.jsonl --no-timeline
    python scripts/spikes/replay_check.py data/real/capture_typing.jsonl --json /tmp/typing_timeline.json

Drives the runtime exactly like ``valence.eval.negative_control`` (served model, PurgeGate, LiveSource,
``replay_offline``), then reports: when ``mask_override`` first fired, whether "Frustrated / Masked Stress"
with ``mask_override`` true appeared, and the share of ticks per label (overall, and per segment when a
``golden_capture`` summary with segment boundaries is given). Warm-up ticks are listed in the timeline
but left out of the shares.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from valence.contracts import STATE_DISPLAY
from valence.runtime.recording import iter_obs

ROOT = Path(__file__).resolve().parents[2]
MASKED_DISPLAY = STATE_DISPLAY["Frustrated_Masked"]
TIMELINE_KEYS = ("t", "segment", "person_id", "person", "label", "label_display", "confidence", "mask_override",
                 "face_only_label", "face_only_confidence", "warming_up")
BETWEEN = "(between)"


def replay(path: str | Path, state_set: str = "core") -> tuple[list[dict[str, Any]], str]:
    """Raw prediction messages for ``path`` and the served model id (stub if no artifacts)."""
    from valence.config import Settings
    from valence.governance.purge import PurgeGate
    from valence.models.predictor import StubPredictor
    from valence.models.served import load_served
    from valence.runtime.bus import EventBus
    from valence.runtime.engine import Runtime
    from valence.runtime.recording import replay_offline

    served = load_served(ROOT / "artifacts", state_set) or StubPredictor()
    s = Settings(state_set=state_set)
    rt = Runtime(s, EventBus(), served, gate=PurgeGate(), audit_dir=tempfile.mkdtemp())
    rt.use_live()
    rt.aggregator.source = "replay"
    try:
        wall0 = time.time()  # replay_offline maps the first obs onto (just after) this instant
        preds = replay_offline(rt, path, tick_s=s.tick_s)
    finally:
        rt.close()
    for m in preds:  # tick boundaries sit on exact multiples of tick_s from the first obs
        m["_t_rel"] = round(round((m["t"] - wall0) / s.tick_s) * s.tick_s, 1)
    return preds, getattr(served, "model_id", "?")


def load_segments(summary: str | Path | None, path: str | Path) -> list[dict[str, Any]]:
    """[{"segment", "start_s", "end_s"}] relative to the recording's first line (skipped segments dropped)."""
    if not summary:
        return []
    data = json.loads(Path(summary).read_text(encoding="utf-8"))
    first_t = next(iter(iter_obs(path))).t
    out = []
    for seg in data.get("segments", []):
        if seg.get("skipped"):
            continue
        if seg.get("t_start") is not None and seg.get("t_end") is not None:
            start, end = seg["t_start"] - first_t, seg["t_end"] - first_t
        elif seg.get("start_s") is not None and seg.get("end_s") is not None:
            start, end = seg["start_s"], seg["end_s"]
        else:
            continue
        out.append({"segment": seg["segment"], "start_s": round(start, 2), "end_s": round(end, 2)})
    return out


def segment_of(t_rel: float, tick_s: float, segments: list[dict[str, Any]]) -> str | None:
    """The segment a tick belongs to, by the midpoint of the window it closes."""
    if not segments:
        return None
    mid = t_rel - tick_s / 2
    return next((s["segment"] for s in segments if s["start_s"] <= mid < s["end_s"]), BETWEEN)


def timeline(preds: list[dict[str, Any]], segments: list[dict[str, Any]], tick_s: float = 5.0) -> list[dict[str, Any]]:
    rows = []
    for m in preds:
        fo = m.get("face_only") or {}
        pid = m["person_id"]
        rows.append({
            "t": m["_t_rel"], "segment": segment_of(m["_t_rel"], tick_s, segments), "person_id": pid,
            "person": "Room" if pid == 0 else f"Person #{pid}", "label": m["label"],
            "label_display": m["label_display"], "confidence": round(float(m["confidence"]), 3),
            "mask_override": bool(m["mask_override"]), "face_only_label": fo.get("label"),
            "face_only_confidence": None if fo.get("confidence") is None else round(float(fo["confidence"]), 3),
            "warming_up": bool(m["warming_up"]),
        })
    return rows


def _shares(rows: list[dict[str, Any]], per_person: bool = True) -> dict[str, Any]:
    """Label shares over scored ticks; with several people (laugh segment, a new track after the break)
    also per person, since the pooled shares mix them."""
    live = [r for r in rows if not r["warming_up"]]
    n = max(1, len(live))
    labels = Counter(r["label"] for r in live)
    out: dict[str, Any] = {"scored_ticks": len(live), "labels": {k: round(v / n, 3) for k, v in labels.most_common()},
                           "mask_override_share": round(sum(r["mask_override"] for r in live) / n, 3)}
    pids = sorted({r["person_id"] for r in rows})
    if per_person and len(pids) > 1:
        out["persons"] = {str(p): _shares([r for r in rows if r["person_id"] == p], False) for p in pids}
    return out


def report(rows: list[dict[str, Any]], segments: list[dict[str, Any]], *, file: str, model: str) -> dict[str, Any]:
    first = next((r for r in rows if r["mask_override"]), None)
    masked = [r for r in rows if r["mask_override"] and r["label_display"] == MASKED_DISPLAY]
    out: dict[str, Any] = {
        "file": file, "model": model, "ticks": len(rows), "warmup_ticks": sum(r["warming_up"] for r in rows),
        "people": sorted({r["person_id"] for r in rows}),
        **_shares(rows),
        "first_mask_override": None if first is None else
        {k: first[k] for k in ("t", "segment", "person", "label_display", "warming_up")},
        "masked_stress_with_override": bool(masked),
        "masked_stress_with_override_ticks": len(masked),
        "masked_stress_first_t": masked[0]["t"] if masked else None,
    }
    if segments:
        by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_seg[r["segment"]].append(r)
        order = [s["segment"] for s in segments] + [BETWEEN]
        out["segments"] = {name: {"start_s": next((s["start_s"] for s in segments if s["segment"] == name), None),
                                  "end_s": next((s["end_s"] for s in segments if s["segment"] == name), None),
                                  **_shares(by_seg[name])}
                           for name in order if name in by_seg or name != BETWEEN}
    return out


def check(path: str | Path, summary: str | Path | None = None, state_set: str = "core") -> dict[str, Any]:
    """Replay ``path``; returns {"report": ..., "timeline": [scalar rows]}."""
    from valence.config import Settings

    tick_s = Settings(state_set=state_set).tick_s
    preds, model = replay(path, state_set)
    segments = load_segments(summary, path)
    rows = timeline(preds, segments, tick_s)
    return {"report": report(rows, segments, file=str(path), model=model), "timeline": rows}


def _fmt(r: dict[str, Any]) -> str:
    fo = f"{r['face_only_label'] or '-'}:{r['face_only_confidence'] if r['face_only_confidence'] is not None else '-'}"
    flags = ("OVERRIDE " if r["mask_override"] else "") + ("warm" if r["warming_up"] else "")
    seg = f"{r['segment']:<10}" if r["segment"] else ""
    return (f"{r['t']:7.1f}s {seg}{r['person']:<10} {r['label']:<20} {r['label_display']:<28} "
            f"{r['confidence']:.2f}  face {fo:<26} {flags}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="features-only JSONL recordings")
    ap.add_argument("--summary", type=Path, help="golden_capture summary with segment boundaries (one file only)")
    ap.add_argument("--json", type=Path, dest="json_out", help="write report + timeline here (one file only)")
    ap.add_argument("--state-set", default="core")
    ap.add_argument("--no-timeline", action="store_true", help="print only the report")
    a = ap.parse_args(argv)
    if (a.summary or a.json_out) and len(a.files) > 1:
        ap.error("--summary and --json take a single recording")
    for f in a.files:
        res = check(f, a.summary, a.state_set)
        if not a.no_timeline:
            print(f"# {f}")
            for r in res["timeline"]:
                print(_fmt(r))
        print(json.dumps(res["report"], indent=None if a.no_timeline else 2))
        if a.json_out:
            a.json_out.parent.mkdir(parents=True, exist_ok=True)
            a.json_out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
            print(f"timeline -> {a.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
