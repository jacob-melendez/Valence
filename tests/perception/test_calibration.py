from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from valence.perception import calibration
from valence.perception.lux import LuxModel

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cal_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "real" / "calibration.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", path)
    return path


def test_default_path_is_repo_data_real() -> None:
    assert calibration.CALIBRATION_PATH == REPO_ROOT / "data" / "real" / "calibration.json"


def test_load_missing_gives_defaults(cal_path: Path) -> None:
    cfg = calibration.load()
    assert cfg["spl_offset_db"] == 94.0
    assert cfg["spl_calibrated"] is False
    assert cfg["lux"]["calibrated"] is False
    assert LuxModel.from_dict(cfg["lux"]) == LuxModel()
    cfg["lux"]["a"] = 99.0
    assert calibration.load()["lux"]["a"] != 99.0
    assert not cal_path.exists()


def test_save_load_round_trip_and_merge(cal_path: Path) -> None:
    calibration.save({"spl_offset_db": 101.5, "spl_calibrated": True})
    assert cal_path.exists()
    calibration.save({"lux": {"a": 1.25, "calibrated": True}})
    cfg = calibration.load()
    assert cfg["spl_offset_db"] == 101.5 and cfg["spl_calibrated"] is True
    assert cfg["lux"]["a"] == 1.25 and cfg["lux"]["calibrated"] is True
    assert cfg["lux"]["b"] == LuxModel().b
    assert json.loads(cal_path.read_text(encoding="utf-8")) == cfg


def test_lux_model_and_spl_helpers(cal_path: Path) -> None:
    assert calibration.lux_model() == LuxModel()
    fitted = LuxModel(2.0, 0.02, -0.05, True)
    calibration.save({"lux": fitted.to_dict(), "spl_offset_db": 97.0})
    assert calibration.lux_model() == fitted
    assert calibration.spl_offset_db() == 97.0
    assert calibration.lux_model({"lux": {"a": 1.0}}).a == 1.0


def test_atomic_write_leaves_no_temp_files(cal_path: Path) -> None:
    for k in range(3):
        calibration.save({"spl_offset_db": 90.0 + k})
    assert [p.name for p in cal_path.parent.iterdir()] == ["calibration.json"]


def test_malformed_file_gives_defaults(cal_path: Path) -> None:
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text("{not json", encoding="utf-8")
    assert calibration.load()["spl_offset_db"] == 94.0
    cal_path.write_text("[1, 2]", encoding="utf-8")
    assert calibration.load()["lux"]["calibrated"] is False
    calibration.save({"spl_offset_db": 99.0})
    assert calibration.load()["spl_offset_db"] == 99.0


@pytest.mark.parametrize("bad", [
    {"lux": [1.0, 2.0]},
    {"frame": np.zeros((2, 2), dtype=np.uint8)},
    {"audio": b"\x00\x01"},
    {"nested": {"x": np.float32(1.0)}},
    {1: 2.0},
])
def test_rejects_non_constants(cal_path: Path, bad: dict) -> None:
    calibration.save({"spl_offset_db": 95.0})
    before = cal_path.read_text(encoding="utf-8")
    with pytest.raises(TypeError):
        calibration.save(bad)
    assert cal_path.read_text(encoding="utf-8") == before


def test_rejects_non_finite(cal_path: Path) -> None:
    with pytest.raises(ValueError):
        calibration.save({"spl_offset_db": math.nan})
    assert not cal_path.exists()


def test_explicit_path_argument(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    calibration.save({"spl_offset_db": 88.0}, path=path)
    assert calibration.load(path)["spl_offset_db"] == 88.0


@pytest.mark.parametrize("script", ["calibrate_lux", "calibrate_spl"])
def test_scripts_import_without_hardware(script: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    sys.modules.pop(script, None)
    module = importlib.import_module(script)
    assert callable(module.main)
    assert "sounddevice" not in vars(module)


def test_load_rejects_wrongly_typed_stored_values(cal_path: Path) -> None:
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text(json.dumps({
        "spl_offset_db": "loud", "spl_calibrated": "false",
        "lux": {"a": None, "b": "x", "c": 0.1, "calibrated": "false"},
    }), encoding="utf-8")
    cfg = calibration.load()
    assert cfg["spl_offset_db"] == 94.0 and cfg["spl_calibrated"] is False
    assert cfg["lux"]["calibrated"] is False and cfg["lux"]["c"] == 0.1
    assert calibration.lux_model() == LuxModel(c=0.1)
    cal_path.write_text('{"lux": 5, "spl_offset_db": NaN}', encoding="utf-8")
    assert calibration.lux_model() == LuxModel()
    assert calibration.spl_offset_db() == 94.0
    cal_path.write_text('{"spl_offset_db": 97, "spl_calibrated": true}', encoding="utf-8")
    assert calibration.spl_offset_db() == 97.0 and calibration.load()["spl_calibrated"] is True


OWNED = [
    "valence/perception/face_valence.py", "valence/perception/lux.py",
    "valence/perception/calibration.py", "valence/perception/audio_features.py",
    "scripts/calibrate_lux.py", "scripts/calibrate_spl.py",
]
BANNED_ATTRS = {("cv2", "imwrite"), ("cv2", "VideoWriter"), ("np", "save"), ("np", "savez"),
                ("numpy", "save"), ("numpy", "savez"), ("wave", "open"), ("soundfile", "write"),
                ("pickle", "dump")}


@pytest.mark.parametrize("rel", OWNED)
def test_no_banned_media_writes(rel: str) -> None:
    import ast

    tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in BANNED_ATTRS, f"{rel}:{node.lineno}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {a.name for a in node.names} | {getattr(node, "module", None) or ""}
            assert not names & {"soundfile", "pickle", "wave"}, f"{rel}:{node.lineno}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
            modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
            assert not any("b" in str(m) for m in modes), f"{rel}:{node.lineno}"


def _fake_cap(frames: list[np.ndarray]) -> object:
    class Cap:
        def read(self) -> tuple[bool, np.ndarray | None]:
            return (True, frames.pop(0)) if frames else (False, None)
    return Cap()


def test_lux_script_purges_frame_even_when_reduction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    import calibrate_lux

    good = np.full((36, 64, 3), 120, dtype=np.uint8)
    y, s = calibrate_lux.frame_scalars(_fake_cap([good]))
    assert y == pytest.approx(120.0, abs=1.0) and not good.any()
    odd = np.full((36, 64, 2), 120, dtype=np.uint8)
    with pytest.raises(Exception):
        calibrate_lux.frame_scalars(_fake_cap([odd]))
    assert not odd.any()


def test_spl_script_purges_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    import calibrate_spl

    blocks: list[np.ndarray] = []

    class Stream:
        def read(self, n: int) -> tuple[np.ndarray, bool]:
            blocks.append(np.full((n, 1), 0.5, dtype=np.float32))
            return blocks[-1], False

    for sr in (16000, 48000):
        assert calibrate_spl.block_power(Stream(), sr) == pytest.approx(0.25, rel=0.02)
    assert all(not b.any() for b in blocks)
    monkeypatch.setattr(calibrate_spl, "mean_power", lambda _x: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        calibrate_spl.block_power(Stream(), 16000)
    assert not blocks[-1].any()
