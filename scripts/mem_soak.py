"""Memory soak: loop the full_demo replay offline at full speed and watch the process stay flat.

The runtime is built exactly as the API builds it (served predictor when artifacts exist, the
event bus, the purge gate, the hash-chained audit log) and stepped directly, as fast as it goes, on a
simulated clock that advances one tick (5 s) per step. The scenario loops the way ``make replay``
loops it (``use_scenario(..., loop=True)``). Every ``--interval`` seconds of wall time the script
samples the process: resident memory, open file descriptors, the runtime's history buffer, the
rolling window and the audit chain length.

Nothing but the JSON summary and the figure is written: the audit log goes to a temporary directory
that is deleted at the end, and no frames, audio or transcripts exist on this path at all.

    python scripts/mem_soak.py --minutes 10            # -> reports/mem_soak.json, reports/figs/memory_flat.png
    python scripts/mem_soak.py --minutes 0.05 --interval 1 --out /tmp/x.json --fig /tmp/x.png
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FLAT_MB_PER_MIN = 0.25  # second-half slope below this counts as flat (15 MB over an hour)

try:
    import psutil  # optional

    _PROC = psutil.Process()
except Exception:  # not installed: read /proc directly
    psutil = None
    _PROC = None


def rss_mb() -> float:
    if _PROC is not None:
        return _PROC.memory_info().rss / 2**20
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("no RSS source (install psutil)")


def open_fds() -> int:
    if _PROC is not None and hasattr(_PROC, "num_fds"):
        return _PROC.num_fds()
    return len(os.listdir("/proc/self/fd"))


def build_runtime(predictor_kind: str, audit_dir: str):
    from valence.config import Settings
    from valence.contracts import STATE_SETS
    from valence.governance.purge import PurgeGate
    from valence.models.predictor import StubPredictor
    from valence.runtime.bus import EventBus
    from valence.runtime.engine import Runtime

    settings = Settings()
    if not Path(settings.artifacts_dir).is_absolute():  # relative to the repo, not the caller's cwd
        settings = dataclasses.replace(settings, artifacts_dir=str(REPO / settings.artifacts_dir))
    if predictor_kind == "stub":
        predictor = StubPredictor(STATE_SETS[settings.state_set])
    else:
        from valence.api.app import load_predictor  # same choice the server makes (served M2, else stub)

        predictor = load_predictor(settings)
        if isinstance(predictor, StubPredictor):
            raise SystemExit(f"no served model for state_set={settings.state_set!r} in {settings.artifacts_dir}; "
                             "train one or pass --predictor stub")
    bus = EventBus()  # no event loop bound and no clients: messages are encoded, then dropped
    rt = Runtime(settings, bus, predictor, gate=PurgeGate(), audit_dir=audit_dir)
    return settings, bus, predictor, rt


def _slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or max(xs) - min(xs) <= 0:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def summarize(samples: list[dict], history_cap: int) -> dict:
    first, last = samples[0], samples[-1]
    rss = [s["rss_mb"] for s in samples]
    t_end = last["elapsed_s"]
    half = [s for s in samples if s["elapsed_s"] >= t_end / 2]
    slope = _slope([s["elapsed_s"] / 60 for s in half], [s["rss_mb"] for s in half])
    warm = samples[1] if len(samples) > 1 else first  # first interval fills the 15-minute window
    hist = [s["history_len"] for s in samples]
    secs = max(t_end, 1e-9)
    return {
        "elapsed_s": round(t_end, 1),
        "ticks": last["ticks"],
        "scenario_loops": last["loops"],
        "simulated_hours": round(last["sim_minutes"] / 60, 1),
        "ticks_per_s": round(last["ticks"] / secs, 1),
        "rss_start_mb": round(first["rss_mb"], 1),
        "rss_after_first_interval_mb": round(warm["rss_mb"], 1),
        "rss_end_mb": round(last["rss_mb"], 1),
        "rss_max_mb": round(max(rss), 1),
        "rss_growth_mb": round(last["rss_mb"] - first["rss_mb"], 2),
        "rss_growth_after_first_interval_mb": round(last["rss_mb"] - warm["rss_mb"], 2),
        "rss_slope_mb_per_min_second_half": None if slope is None else round(slope, 4),
        "flat": slope is not None and abs(slope) < FLAT_MB_PER_MIN,
        "flat_threshold_mb_per_min": FLAT_MB_PER_MIN,
        "fds_start": first["fds"],
        "fds_end": last["fds"],
        "fds_max": max(s["fds"] for s in samples),
        "history_len_max": max(hist),
        "history_len_end": last["history_len"],
        "history_cap": history_cap,
        "history_plateaued": last["history_len"] == history_cap,  # reached the cap and stayed bounded by it
        "window_records_max": max(s["window_records"] for s in samples),
        "audit_entries_end": last["audit_entries"],
        "audit_bytes_end": last["audit_bytes"],
        "py_objects_start": first["py_objects"],
        "py_objects_end": last["py_objects"],
    }


def soak(minutes: float, interval: float, scenario: str, predictor_kind: str, trace: bool) -> dict:
    from valence.sim.scenarios import render

    if trace:
        tracemalloc.start(10)
    with tempfile.TemporaryDirectory(prefix="valence-soak-audit-") as audit_dir:
        settings, bus, predictor, rt = build_runtime(predictor_kind, audit_dir)
        df = render(scenario, state_set=settings.state_set)
        rt.use_scenario(df, scenario, speed=1.0, loop=True)  # the server's replay path loops like this
        n_scenario_ticks = len(rt.source._ticks)
        tick_s = settings.tick_s
        wall0, mono0 = time.time(), time.monotonic()
        ticks = restarts = 0

        def sample(elapsed: float) -> dict:
            audit = rt.audit
            return {
                "elapsed_s": round(elapsed, 2),
                "ticks": ticks,
                "loops": ticks // n_scenario_ticks + restarts,
                "sim_minutes": round(ticks * tick_s / 60, 2),
                "rss_mb": round(rss_mb(), 2),
                "fds": open_fds(),
                "history_len": len(rt.history),
                "window_records": len(rt.window.room()) + sum(len(rt.window.snapshot(p)) for p in rt.window.people()),
                "people_tracked": len(rt.window.people()),
                "audit_entries": audit._seq,
                "audit_bytes": audit.path.stat().st_size if audit.path.exists() else 0,
                "bus_published": bus.published,
                "py_objects": len(gc.get_objects()),
            }

        samples = [sample(0.0)]
        snap0 = None
        t0 = time.monotonic()
        deadline = t0 + minutes * 60
        next_sample = t0 + interval
        while True:
            now = time.monotonic()
            if now >= next_sample or now >= deadline:
                samples.append(sample(now - t0))
                if trace and snap0 is None:
                    snap0 = tracemalloc.take_snapshot()
                if now >= deadline:
                    break
                next_sample += interval
            out = rt.step(wall=wall0 + ticks * tick_s, mono=mono0 + ticks * tick_s)
            ticks += 1
            if not out and rt.paused:  # defensive: a non-looping source ended, so start it again
                rt.use_scenario(df, scenario, speed=1.0, loop=True)
                rt.paused = False
                restarts += 1
        top = []
        if trace and snap0 is not None:
            diff = tracemalloc.take_snapshot().compare_to(snap0, "lineno")
            top = [{"where": str(d.traceback[0]), "size_diff_kb": round(d.size_diff / 1024, 1),
                    "count_diff": d.count_diff} for d in diff[:15]]
            tracemalloc.stop()
        rt.close()  # deletes the audit file; the temporary directory goes with the context manager
    config = {
        "scenario": scenario,
        "minutes": minutes,
        "interval_s": interval,
        "mode": "offline: Runtime.step on a simulated clock (one tick per step), scenario loop=True",
        "predictor": type(predictor).__name__,
        "model_id": getattr(predictor, "model_id", None),
        "narrator": None,
        "tick_s": tick_s,
        "window_ticks": settings.window_ticks,
        "history_cap": rt.history.maxlen,
        "scenario_ticks": n_scenario_ticks,
        "rss_source": "psutil" if _PROC is not None else "/proc/self/status VmRSS",
        "fds_source": "psutil" if _PROC is not None else "/proc/self/fd",
        "audit_dir": "temporary directory, deleted after the run",
        "tracemalloc": trace,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall0)),
    }
    report = {"config": config, "summary": summarize(samples, rt.history.maxlen), "samples": samples}
    if trace:
        report["tracemalloc_top"] = top
    return report


def figure(report: dict, path: Path) -> None:
    from valence.eval import style

    plt = style.plt
    samples, summ, cfg = report["samples"], report["summary"], report["config"]
    xs = [s["elapsed_s"] / 60 for s in samples]
    mins = cfg["minutes"]
    soak_label = f"{mins:g}-minute"
    slope = summ["rss_slope_mb_per_min_second_half"]
    if slope is None:  # fewer than two samples in the second half
        title = f"Memory under a {soak_label} soak (too short to fit a slope)"
    elif summ["flat"]:
        title = f"Memory stays flat under a {soak_label} soak"
    else:
        title = f"Memory grows {slope:+.2f} MB/min under a {soak_label} soak"
    subtitle = (f"{cfg['scenario']} replay looped offline as fast as it runs: {summ['ticks']:,} ticks, "
                f"{summ['scenario_loops']:,} loops, {summ['simulated_hours']:,.1f} simulated hours "
                f"({cfg['predictor']}); sampled every {cfg['interval_s']:g} s")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.4), dpi=160, sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1, 1], "hspace": 0.55})
    fig.patch.set_facecolor(style.SURFACE)
    for ax in axes:
        ax.set_facecolor(style.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(style.GRID)
            ax.spines[side].set_linewidth(1)
        ax.tick_params(colors=style.TEXT_2, labelsize=9, length=0)
        ax.grid(axis="y", color=style.GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.text(0.02, 0.975, title, fontsize=13, fontweight="bold", color=style.TEXT, va="top")
    fig.text(0.02, 0.935, subtitle, fontsize=8.5, color=style.TEXT_2, va="top")

    def panel(ax, ys, key, heading, end_text, ymin=None, ymax=None):
        color = style.SERIES[key]
        many = len(xs) > 60
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker=None if many else "o", markersize=8, markerfacecolor=color,
                markeredgecolor=style.SURFACE, markeredgewidth=2)
        ax.annotate(end_text, (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=style.TEXT, annotation_clip=False)
        ax.set_title(heading, loc="left", fontsize=9.5, color=style.TEXT, pad=6)
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * 0.25, 1.0)
        ax.set_ylim(lo - pad if ymin is None else ymin, hi + pad if ymax is None else ymax)

    rss = [s["rss_mb"] for s in samples]
    ax = axes[0]
    top = max(rss)
    panel(ax, rss, "m2", "Resident memory (MB)", f"{rss[-1]:.0f} MB", ymin=0, ymax=max(top * 1.25, top + 20))
    half = [(x, s["rss_mb"]) for x, s in zip(xs, samples) if s["elapsed_s"] >= summ["elapsed_s"] / 2]
    if slope is not None and len(half) >= 2:
        hx = [p[0] for p in half]
        my = sum(p[1] for p in half) / len(half)
        mx = sum(hx) / len(hx)
        ax.plot([hx[0], hx[-1]], [my + slope * (hx[0] - mx), my + slope * (hx[-1] - mx)], color=style.TEXT_2,
                linewidth=1, linestyle=(0, (4, 3)), alpha=0.8)
        ax.annotate(f"second-half fit {slope:+.3f} MB/min", (hx[0], my), xytext=(0, -16),
                    textcoords="offset points", fontsize=8.5, color=style.TEXT_2)
    ax.annotate(f"start {rss[0]:.0f} MB, max {top:.0f} MB, growth {summ['rss_growth_mb']:+.1f} MB",
                (0, 0), xytext=(2, 6), textcoords="offset points", fontsize=8.5, color=style.TEXT_2,
                xycoords=("data", "axes fraction"))

    hist = [s["history_len"] for s in samples]
    cap = cfg["history_cap"]
    panel(axes[1], hist, "m1", f"History buffer (entries; capped at {cap:,} = 15-minute window x 8)",
          f"{hist[-1]:,}", ymin=0, ymax=cap * 1.3)
    axes[1].axhline(cap, color=style.TEXT_2, linewidth=1, alpha=0.35)

    fds = [s["fds"] for s in samples]
    panel(axes[2], fds, "m0", "Open file descriptors", f"{fds[-1]}", ymin=0, ymax=max(fds) * 1.5 + 2)
    axes[2].set_xlabel("minutes of wall-clock soak", color=style.TEXT_2, fontsize=9)
    axes[2].set_xlim(0, max(xs[-1], 1e-3))
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.08, right=0.86)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=style.SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=10.0, help="wall-clock soak length")
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between samples")
    ap.add_argument("--scenario", default="full_demo")
    ap.add_argument("--predictor", choices=("served", "stub"), default="served",
                    help="served: what the API loads (trained M2 if artifacts exist); stub: rule predictor")
    ap.add_argument("--tracemalloc", action="store_true", help="also record the top allocation diffs (slower)")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "mem_soak.json")
    ap.add_argument("--fig", type=Path, default=REPO / "reports" / "figs" / "memory_flat.png")
    ap.add_argument("--replot", action="store_true", help="redraw --fig from an existing --out JSON, no soak")
    args = ap.parse_args(argv)
    if args.minutes <= 0 or args.interval <= 0:
        ap.error("--minutes and --interval must be positive")
    if args.replot:
        figure(json.loads(args.out.read_text(encoding="utf-8")), args.fig)
        print(f"wrote {args.fig}")
        return 0

    report = soak(args.minutes, args.interval, args.scenario, args.predictor, args.tracemalloc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure(report, args.fig)
    s = report["summary"]
    print(f"soak {s['elapsed_s']:.0f} s: {s['ticks']:,} ticks ({s['scenario_loops']} loops, "
          f"{s['simulated_hours']} simulated h) | RSS {s['rss_start_mb']} -> {s['rss_end_mb']} MB "
          f"(max {s['rss_max_mb']}, second-half slope {s['rss_slope_mb_per_min_second_half']} MB/min, "
          f"flat={s['flat']}) | fds {s['fds_start']} -> {s['fds_end']} | history max {s['history_len_max']} "
          f"| audit entries {s['audit_entries_end']}")
    print(f"wrote {args.out} and {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
