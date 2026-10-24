"""Parity fixtures for the browser perception (frontend/src/perception, TypeScript ports of valence/perception).

Runs the real Python functions on shared, seeded inputs and writes what they return, so the TypeScript tests
(frontend/src/test/perception-*.test.ts) can replay the same inputs and must match. One generator module per group
lives next to the fixtures (frontend/src/test/perception-fixtures/gen_<group>.py, each with ``export(out_dir)``):

    audio    audio_features.py (mean_power, power_to_db, PowerSmoother, laughter) and scipy resample_poly
    posture  posture.py (PostureTracker)
    hands    hands.py (assign_hands, curled_fingers, FistTracker), ids.py (DisplayIdMapper), debounce.py
    lux      lux.py (background_stats, LuxEstimator), regions.py (head_box_px, norm_bbox), cv2 BGR2GRAY
    face     face_valence.py, features/face_cues.py, scipy linear_sum_assignment

Fixtures are strict JSON (non-finite floats as the strings "NaN", "Infinity", "-Infinity"), gzipped with a fixed
mtime so a re-run is byte-identical. Needs numpy, scipy and opencv-python-headless:

    .venv/bin/python scripts/export_perception_fixtures.py [group ...]
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "frontend" / "src" / "test" / "perception-fixtures"
GROUPS = ("audio", "posture", "hands", "lux", "face")


def run_group(group: str) -> list[Path]:
    path = FIXTURES / f"gen_{group}.py"
    spec = importlib.util.spec_from_file_location(f"perception_fixtures_{group}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    before = {p: p.stat().st_mtime_ns for p in FIXTURES.glob("*.json")}
    module.export(FIXTURES)
    return [p for p in FIXTURES.glob("*.json") if before.get(p) != p.stat().st_mtime_ns]


def gzip_fixed(path: Path) -> Path:
    """path.json -> path.json.gz (mtime 0, level 9: deterministic), then remove the plain file."""
    out = path.with_name(path.name + ".gz")
    with path.open("rb") as src, out.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            gz.write(src.read())
    path.unlink()
    return out


def main(argv: list[str]) -> int:
    groups = argv or list(GROUPS)
    for group in groups:
        if group not in GROUPS:
            raise SystemExit(f"unknown group {group!r}; choose from {GROUPS}")
        if not (FIXTURES / f"gen_{group}.py").exists():
            print(f"{group:8s} skipped: no generator yet")
            continue
        written = run_group(group)
        for p in sorted(written):
            gz = gzip_fixed(p)
            print(f"{group:8s} {gz.relative_to(ROOT)} ({gz.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
