"""The memory soak script runs end to end on a few seconds and writes a well-formed report."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "mem_soak.py"


def _module():
    spec = importlib.util.spec_from_file_location("mem_soak", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_short_soak_writes_report_and_figure(tmp_path):
    out, fig = tmp_path / "soak.json", tmp_path / "soak.png"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--minutes", "0.05", "--interval", "1", "--predictor", "stub",
         "--out", str(out), "--fig", str(fig)],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(out.read_text())
    assert set(report) >= {"config", "summary", "samples"}
    assert report["config"]["scenario"] == "full_demo" and report["config"]["interval_s"] == 1
    samples = report["samples"]
    assert len(samples) >= 3
    keys = {"elapsed_s", "ticks", "loops", "rss_mb", "fds", "history_len", "audit_entries"}
    assert all(keys <= set(s) for s in samples)
    assert [s["elapsed_s"] for s in samples] == sorted(s["elapsed_s"] for s in samples)
    assert samples[-1]["ticks"] > 0 and all(s["rss_mb"] > 0 and s["fds"] > 0 for s in samples)
    summary = report["summary"]
    for k in ("rss_start_mb", "rss_end_mb", "rss_max_mb", "rss_growth_mb", "rss_slope_mb_per_min_second_half",
              "fds_start", "fds_end", "history_len_max", "audit_entries_end", "flat"):
        assert k in summary
    assert summary["rss_max_mb"] >= max(summary["rss_start_mb"], summary["rss_end_mb"]) - 1e-6
    assert 0 < summary["history_len_max"] <= report["config"]["history_cap"]
    assert summary["audit_entries_end"] >= 2  # replay start + reset, then predictions
    assert fig.exists() and fig.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not list(tmp_path.glob("*.jsonl"))  # the audit log lived in a deleted temporary directory


def test_summary_slope_uses_the_second_half():
    mod = _module()
    samples = [{"elapsed_s": t * 60.0, "ticks": t * 100, "loops": t, "sim_minutes": t * 500, "rss_mb": rss,
                "fds": 4, "history_len": min(t * 400, 1440), "window_records": 360, "audit_entries": t * 10,
                "audit_bytes": 0, "py_objects": 1} for t, rss in enumerate([100, 150, 150, 152, 154])]
    s = mod.summarize(samples, 1440)
    assert s["rss_growth_mb"] == 54 and s["rss_max_mb"] == 154
    assert abs(s["rss_slope_mb_per_min_second_half"] - 2.0) < 1e-9  # warm-up jump excluded
    assert not s["flat"] and s["history_len_max"] == 1440 and s["history_plateaued"]
