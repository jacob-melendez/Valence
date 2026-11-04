"""Parity fixtures for frontend/src/perception/lux.ts, regions.ts and luma.ts (ports of valence/perception/lux.py,
valence/perception/regions.py and the cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) in valence/perception/video.py).

Calls the real Python functions on seeded inputs and records what they return (or the exception they raise):

    lux-stats.json      background_stats on uint8 images x person-box sets x max_width (random, structured,
                        saturated, flat, tiny and 1280x720 frames; boxes outside / NaN / swapped / covering all)
    lux-estimator.json  LuxModel.predict and LuxModel.from_dict cases, LuxEstimator step sequences
    lux-regions.json    head_box_px, norm_bbox, np.linalg.norm of 2-vectors, math.fma
    lux-luma.json       cv2 BGR2GRAY on RGB triples (edges, random, 14-bit/15-bit disagreements), a small RGBA image

Before writing, asserts that luma.ts's formula, (9798 R + 19235 G + 3735 B + 2^14) >> 15, equals cv2 on all 256^3
colours. Images are base64 of the raw row-major bytes; a "grid" image stores only the pixels at multiples of `grid`
and fills the rest with (31 x + 17 y) % 256. Strict JSON, non-finite floats as "NaN" / "Infinity" / "-Infinity".

    .venv/bin/python frontend/src/test/perception-fixtures/gen_lux.py   (writes next to this file)
"""

from __future__ import annotations

import base64
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from valence.perception.lux import BgStats, LuxEstimator, LuxModel, background_stats  # noqa: E402
from valence.perception.regions import head_box_px, norm_bbox  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent
NAN, INF = math.nan, math.inf


# --------------------------------------------------------------------------------------------------- encoding


def enc(v: Any) -> Any:
    """Strict-JSON value: non-finite floats become strings; numpy scalars become Python ones."""
    if isinstance(v, np.generic):
        v = v.item()
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if isinstance(v, np.ndarray):
        return enc(v.tolist())
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    raise TypeError(f"cannot encode {type(v)}")


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def write(out_dir: Path, name: str, payload: dict[str, Any]) -> None:
    text = json.dumps(enc(payload), allow_nan=False, separators=(",", ":"))
    (out_dir / f"{name}.json").write_text(text + "\n", encoding="utf-8")


def attempt(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, str | None]:
    try:
        return fn(*args, **kwargs), None
    except (ValueError, OverflowError, TypeError, ZeroDivisionError) as e:
        return None, type(e).__name__


# --------------------------------------------------------------------------------------------------- images


def noisy(rng: np.random.Generator, base: np.ndarray | float, sigma: float, shape: tuple[int, int]) -> np.ndarray:
    img = np.broadcast_to(np.asarray(base, dtype=np.float64), shape) + rng.normal(0.0, sigma, shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def grid_fill(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((31 * xx + 17 * yy) % 256).astype(np.uint8)


def full_image(name: str, img: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    return {"name": name, "h": int(img.shape[0]), "w": int(img.shape[1]), "b64": b64(img)}, img


def grid_image(name: str, h: int, w: int, grid: int, sub: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    img = grid_fill(h, w)
    assert img[::grid, ::grid].shape == sub.shape
    img[::grid, ::grid] = sub
    return {"name": name, "h": h, "w": w, "grid": grid, "b64": b64(sub)}, img


def scene(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Room-like frame: smooth gradient + sensor noise, a blown-out window, a black shelf, a flat poster and a bright,
    textured 'person'."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = 60.0 + 110.0 * xx / max(w - 1, 1) + 20.0 * np.sin(yy / max(h, 1) * 3.0)
    img = noisy(rng, base, 6.0, (h, w))
    img[int(0.05 * h):int(0.3 * h), int(0.7 * w):int(0.95 * w)] = 255
    img[int(0.75 * h):, :int(0.2 * w)] = 0
    img[int(0.1 * h):int(0.25 * h), int(0.05 * w):int(0.2 * w)] = 140
    py0, py1, px0, px1 = int(0.3 * h), h, int(0.35 * w), int(0.65 * w)
    img[py0:py1, px0:px1] = noisy(rng, 200.0, 25.0, (py1 - py0, px1 - px0))
    return img


STANDARD_BOXES: list[tuple[str, list[list[float]]]] = [
    ("none", []),
    ("person", [[0.35, 0.3, 0.65, 1.0]]),
    ("two", [[0.1, 0.2, 0.4, 0.9], [0.6, 0.1, 0.8, 0.95]]),
    ("overlap", [[0.2, 0.2, 0.6, 0.6], [0.4, 0.4, 0.8, 0.8]]),
    ("all", [[0.0, 0.0, 1.0, 1.0]]),
    ("all_outside", [[-0.5, -0.5, 1.5, 1.5]]),
    ("swapped", [[0.5, 0.5, 0.0, 0.0]]),
    ("half_swapped", [[0.9, 0.1, 0.3, 0.7]]),
    ("nan", [[NAN, 0.0, 1.0, 1.0]]),
    ("nan_and_real", [[0.1, NAN, 0.2, 0.3], [0.5, 0.5, 0.75, 0.75]]),
    ("inf", [[0.0, 0.0, INF, 1.0]]),
    ("neg_inf", [[-INF, 0.2, 0.5, 0.6]]),
    ("outside_right", [[1.2, 0.1, 1.5, 0.9]]),
    ("outside_left", [[-0.5, -0.5, -0.1, 0.5]]),
    ("outside_below", [[0.2, 1.0, 0.8, 1.7]]),
    ("partly_outside", [[0.8, 0.5, 1.4, 1.3]]),
    ("zero_width", [[0.3, 0.3, 0.3, 0.7]]),
    ("signed_zero", [[-0.0, 0.0, 0.5, -0.0]]),
    ("tiny", [[0.5, 0.5, 0.5000001, 0.5000001]]),
    ("thirds", [[1 / 3, 1 / 7, 2 / 3, 6 / 7]]),
    ("pixel_edges", [[0.25, 0.25, 0.75, 0.75]]),
    ("most", [[0.0, 0.0, 1.0, 0.97], [0.0, 0.0, 0.97, 1.0]]),
    ("ring_only", [[0.02, 0.02, 0.98, 0.98]]),
    ("many", [[0.05 * k, 0.1, 0.05 * k + 0.03, 0.9] for k in range(12)]),
    ("big_finite", [[0.0, 0.0, 1e306, 1.0]]),
    ("huge_negative", [[-1e308, 0.0, 0.5, 1.0]]),
    ("three_values", [[0.1, 0.2, 0.3]]),
    ("five_values", [[0.1, 0.2, 0.3, 0.4, 0.5]]),
]
SHORT_BOXES = ["none", "person", "two", "all", "nan", "partly_outside", "swapped", "most"]


def run_stats(img: np.ndarray, boxes: list[list[float]], max_width: float | None) -> dict[str, Any]:
    kwargs = {} if max_width is None else {"max_width": max_width}
    st, err = attempt(background_stats, img, boxes, **kwargs)
    if err is not None:
        return {"error": err}
    return {"y_bg": st.y_bg, "sigma": st.sigma, "bg_frac": st.bg_frac, "clipped_frac": st.clipped_frac,
            "valid": st.valid}


def stats_cases() -> dict[str, Any]:
    rng = np.random.default_rng(20261024)
    every = [n for n, _ in STANDARD_BOXES]
    table = dict(STANDARD_BOXES)
    images: list[dict[str, Any]] = []
    n_cases = 0

    def add(entry: tuple[dict[str, Any], np.ndarray], box_names: list[str], widths: list[float | None],
            extra: list[list[float]] | None = None) -> None:
        """Run every box set at the default max_width; the other widths only with a few box sets."""
        nonlocal n_cases
        meta, img = entry
        sets: list[tuple[dict[str, Any], list[list[float]]]] = [({"boxes": n}, table[n]) for n in box_names]
        if extra is not None:
            sets.append(({"box_list": extra}, extra))
        runs = []
        for ref, boxes in sets:
            for mw in widths:
                if mw is not None and ref.get("boxes", "none") not in ("none", "person", "all"):
                    continue
                runs.append({**ref, "max_width": mw, "expect": run_stats(img, boxes, mw)})
        n_cases += len(runs)
        images.append({**meta, "runs": runs})

    add(full_image("noise_64x48", rng.integers(0, 256, (48, 64), dtype=np.uint8)), every, [None, 32, 21, 64])
    add(full_image("scene_333x177", scene(rng, 177, 333)), every,
        [None, 333, 332, 111, 100, 1, 1000, -5, 320.5, INF, 1e-300, 0, NAN])
    add(grid_image("scene_1280x720", 720, 1280, 4, scene(rng, 180, 320)), every, [None, 640, 1280, 100])
    add(grid_image("noise_1000x563", 563, 1000, 4, noisy(rng, 90.0, 9.0, (141, 250))), SHORT_BOXES, [None, 500])
    ramp = np.tile(np.linspace(60.0, 190.0, 160), (120, 1))
    add(full_image("gradient_noise_160x120", noisy(rng, ramp, 8.0, (120, 160))), SHORT_BOXES, [None])

    add(full_image("flat_128", np.full((30, 40), 128, dtype=np.uint8)), SHORT_BOXES, [None])
    add(full_image("all_black", np.zeros((30, 40), dtype=np.uint8)), SHORT_BOXES, [None])
    add(full_image("all_white", np.full((30, 40), 255, dtype=np.uint8)), SHORT_BOXES, [None])
    checker = np.where((np.add.outer(np.arange(17), np.arange(21)) % 2) == 0, 0, 255).astype(np.uint8)
    add(full_image("checker_0_255", checker), SHORT_BOXES, [None])
    half = noisy(rng, 128.0, 5.0, (40, 50))
    half[:, :25] = 0
    add(full_image("half_black", half), SHORT_BOXES, [None])
    half_w = noisy(rng, 128.0, 5.0, (40, 50))
    half_w[:, 25:] = 255
    add(full_image("half_white", half_w), SHORT_BOXES + ["pixel_edges"], [None])
    bands = np.repeat(np.array([4, 5, 6, 7, 248, 249, 250, 251, 0, 255, 1, 254], dtype=np.uint8), 3)[None, :]
    add(full_image("clip_bands", np.repeat(bands, 20, axis=0)), SHORT_BOXES, [None])
    islands = np.zeros((25, 25), dtype=np.uint8)
    islands[5:8, 5:8] = noisy(rng, 100.0, 10.0, (3, 3))
    islands[12:20, 10:22] = noisy(rng, 100.0, 10.0, (8, 12))
    add(full_image("noise_islands_in_black", islands), SHORT_BOXES, [None])
    add(full_image("ring_10x10", noisy(rng, 120.0, 7.0, (10, 10))), ["none", "ring_only", "all", "pixel_edges"], [None])

    for h, w in [(0, 0), (0, 5), (5, 0), (1, 1), (2, 2), (1, 7), (7, 1), (2, 9), (9, 2), (3, 3), (3, 4), (4, 3),
                 (3, 700), (700, 3)]:
        img = rng.integers(0, 256, (h, w), dtype=np.uint8)
        add(full_image(f"tiny_{h}x{w}", img), ["none", "person", "all", "nan", "big_finite"], [None, 2])

    for k in range(40):
        h, w = int(rng.integers(1, 25)), int(rng.integers(1, 25))
        kind = k % 4
        if kind == 0:
            img = rng.integers(0, 256, (h, w), dtype=np.uint8)
        elif kind == 1:
            img = noisy(rng, float(rng.uniform(0, 255)), float(rng.uniform(0, 30)), (h, w))
        elif kind == 2:
            img = rng.choice(np.array([0, 3, 5, 6, 128, 249, 250, 255], dtype=np.uint8), (h, w))
        else:
            img = noisy(rng, 128.0, 4.0, (h, w))
            img[: h // 2] = 255
        fuzz_boxes = []
        for _ in range(int(rng.integers(0, 4))):
            b = rng.uniform(-0.3, 1.3, 4)
            if rng.random() < 0.1:
                b[int(rng.integers(0, 4))] = rng.choice([NAN, INF, -INF])
            fuzz_boxes.append([float(v) for v in b])
        add(full_image(f"fuzz_{k}", img), ["none"], [None, float(rng.integers(1, 30))], fuzz_boxes)

    return {"boxes": table, "images": images, "n_cases": n_cases}


# --------------------------------------------------------------------------------------------------- estimator


def model_dict(m: LuxModel) -> dict[str, Any]:
    return {"a": m.a, "b": m.b, "c": m.c, "calibrated": m.calibrated}


def stats_dict(s: BgStats) -> list[float]:
    return [s.y_bg, s.sigma, s.bg_frac, s.clipped_frac]


def predict_cases() -> list[dict[str, Any]]:
    models = [LuxModel(), LuxModel(1.0, 0.02, -0.05, True), LuxModel(c=NAN), LuxModel(c=-0.0), LuxModel(c=INF),
              LuxModel(b=10.0), LuxModel(a=NAN), LuxModel(a=INF), LuxModel(a=-800.0), LuxModel(0.0, 1.0, 0.0, True),
              LuxModel(a=60.0), LuxModel(2.0, 0.03, 0.1, False)]
    inputs = [(0.0, 0.0), (30.0, 5.0), (128.0, 3.0), (220.0, 5.0), (255.0, 0.0), (NAN, 1.0), (100.0, NAN),
              (INF, 1.0), (-INF, 1.0), (100.0, INF), (100.0, -INF), (1e308, 1.0), (-1e308, 2.0), (12.5, 1e308)]
    return [{"model": model_dict(m), "y": y, "sigma": s, "expect": m.predict(y, s)} for m in models for y, s in inputs]


def from_dict_cases() -> list[dict[str, Any]]:
    dicts: list[dict[str, Any]] = [
        {}, {"a": 1.0, "b": 0.02, "c": -0.05, "calibrated": True}, {"a": 2}, {"a": "1.5"}, {"a": " 2e-1 "},
        {"b": "nan"}, {"c": "-inf"}, {"c": " +Infinity\n"}, {"a": "1_000.5"}, {"a": "1e1_0"}, {"a": True},
        {"b": False}, {"calibrated": 1}, {"calibrated": 0}, {"calibrated": 0.0}, {"calibrated": NAN},
        {"calibrated": ""}, {"calibrated": "no"}, {"calibrated": []}, {"calibrated": [0]}, {"calibrated": {}},
        {"calibrated": {"x": 1}}, {"calibrated": None}, {"a": None}, {"a": "abc"}, {"a": [1]}, {"a": {}},
        {"a": "\u00851 "}, {"a": "　1.25 "}, {"a": "1__0"}, {"a": "_1"}, {"a": "1_"}, {"a": "."},
        {"a": "+.5e1"}, {"a": "5."}, {"a": "-.5"}, {"a": "0x10"}, {"a": "1e"}, {"a": "iNfInItY"}, {"a": "infinit"},
        {"a": "\x1c1"}, {"a": "1e400"}, {"a": "-0"}, {"extra": 1, "c": 0.5}, {"a": INF, "b": NAN, "c": -INF},
        {"a": "\u0661\u0662.5"}, {"b": "\uff11\uff10e-2"}, {"c": "-\U0001d7cf\U0001d7d0"}, {"a": "1\u00a02"},
    ]
    out = []
    for d in dicts:
        m, err = attempt(LuxModel.from_dict, d)
        out.append({"d": d, "expect": {"error": err} if err else model_dict(m)})
    return out


def replay(name: str, model: LuxModel, window_s: float, steps: list[Any]) -> dict[str, Any]:
    """steps: (BgStats, t) pairs or the string 'reset'."""
    est = LuxEstimator(model, window_s)
    rows: list[Any] = []
    for step in steps:
        if step == "reset":
            est.reset()
            rows.append("reset")
            continue
        st, t = step
        lux, q = est.update(st, t)
        rows.append({"s": stats_dict(st), "t": t, "lux": lux, "q": q, "n": len(est._history)})
    return {"name": name, "model": model_dict(model), "window_s": window_s, "steps": rows}


def bg(y: float = 128.0, bg_frac: float = 0.8, clipped: float = 0.1, sigma: float = 3.0) -> BgStats:
    return BgStats(y_bg=y, sigma=sigma, bg_frac=bg_frac, clipped_frac=clipped)


INVALID = BgStats(NAN, NAN, 0.0, 0.0)


def estimator_cases() -> list[dict[str, Any]]:
    cal = LuxModel(1.0, 0.03, 0.0, True)
    cases = [
        replay("constant_calibrated", cal, 10.0, [(bg(), 0.5 * k) for k in range(20)]),
        replay("uncalibrated_default", LuxModel(), 10.0, [(bg(), 0.0), (bg(140.0), 1.0), (bg(90.0, 1.0, 0.0), 2.0)]),
        replay("median_window_stability", LuxModel(0.0, 1.0, 0.0, True), 10.0,
               [(bg(y, 1.0, 0.0), float(k)) for k, y in enumerate([5.0, 5.0, 5.0, 6.0, 7.0])]
               + [(bg(7.0, 1.0, 0.0), 13.5)]),
        replay("invalid_stats", LuxModel(calibrated=True), 10.0,
               [(INVALID, 0.0), (bg(), 1.0), (INVALID, 2.0), (bg(100.0, 0.0, 0.0), 3.0), (bg(INF), 4.0),
                (bg(NAN, 0.5), 5.0), (bg(120.0, NAN), 6.0), (bg(120.0, -0.1), 7.0), (bg(130.0, 1e-300), 8.0)]),
        replay("non_finite_t", cal, 10.0,
               [(bg(100.0), NAN), (bg(100.0), 0.0), (bg(150.0), INF), (bg(160.0), 5.0), (bg(170.0), -INF),
                (bg(180.0), 12.0), (INVALID, NAN), (bg(190.0, 0.3, 0.5), NAN), (bg(200.0), 30.0)]),
        replay("boundary_exact", cal, 10.0,
               [(bg(100.0), 0.0), (bg(110.0), 5.0), (bg(120.0), 10.0), (bg(130.0), math.nextafter(10.0, INF)),
                (bg(140.0), 15.0), (bg(150.0), 20.0), (bg(150.0), 20.0), (bg(160.0), 30.0)]),
        replay("boundary_float_rounding", cal, 10.0,
               [(bg(100.0), 0.3), (bg(110.0), 5.3), (bg(120.0), 10.3), (bg(130.0), 0.1 * 3 + 10.0),
                (bg(140.0), 20.6), (bg(150.0), 30.6)]),
        replay("boundary_tenths", cal, 1.0, [(bg(100.0 + k), 0.1 * k) for k in range(40)]),
        replay("c_model", LuxModel(1.0, 0.02, -0.05, True), 10.0,
               [(bg(100.0, sigma=s), float(k)) for k, s in enumerate([1.0, 4.0, NAN, 2.0, INF, -INF, 0.0, 12.0])]),
        replay("c_model_uncalibrated", LuxModel(2.0, 0.01, 0.07, False), 6.0,
               [(bg(80.0 + 7 * k, sigma=1.0 + k), 1.5 * k) for k in range(12)]),
        replay("clamp_max_lux", LuxModel(b=10.0, calibrated=True), 10.0,
               [(bg(255.0), 0.0), (bg(1.0), 1.0), (bg(200.0), 2.0)]),
        replay("underflow_zero_lux", LuxModel(a=-800.0, calibrated=True), 10.0,
               [(bg(10.0), 0.0), (bg(20.0), 1.0), (bg(30.0), 2.0)]),
        replay("decreasing_t", cal, 5.0,
               [(bg(100.0), 10.0), (bg(110.0), 8.0), (bg(120.0), 3.0), (bg(130.0), 14.0), (bg(140.0), 15.5),
                (bg(150.0), 2.0), (bg(160.0), 9.0)]),
        replay("window_zero", cal, 0.0,
               [(bg(100.0), 1.0), (bg(110.0), 1.0), (bg(120.0), 2.0), (bg(130.0), 2.0), (bg(140.0), 2.5)]),
        replay("window_negative", cal, -1.0, [(bg(100.0), 1.0), (bg(110.0), 2.0)]),
        replay("window_nan", cal, NAN, [(bg(100.0 + k), 100.0 * k) for k in range(6)]),
        replay("reset", cal, 10.0, [(bg(100.0), 0.0), (bg(120.0), 1.0), "reset", (INVALID, 2.0), (bg(140.0), 3.0)]),
        replay("quality_clip", cal, 10.0,
               [(bg(128.0, 1.5, -1.0), 0.0), (bg(128.0, 0.5, 2.0), 1.0), (bg(128.0, NAN, 0.0), 2.0),
                (bg(128.0, 0.0, 2.0), 3.0), (bg(128.0, 1.0, NAN), 4.0), (bg(128.0, INF, 0.0), 5.0)]),
        replay("long_window_eviction", cal, 10.0,
               [(bg(100.0), NAN), (bg(100.0), 0.0)] + [(bg(200.0), 0.5 * k) for k in range(1, 400)]
               + [(bg(200.0), 200.0)]),
    ]
    rng = np.random.default_rng(424242)
    for seed_i, (model, window_s) in enumerate([(LuxModel(), 10.0), (LuxModel(0.5, 0.025, 0.0, True), 3.3),
                                                 (LuxModel(1.0, 0.02, -0.04, True), 0.7)]):
        steps: list[Any] = []
        t = 0.0
        for _ in range(300):
            r = rng.random()
            if r < 0.05:
                tt = float(rng.choice([NAN, INF, -INF]))
            else:
                if r < 0.1:
                    pass
                elif r < 0.13:
                    t -= float(rng.uniform(0.0, 2.0))
                else:
                    t += float(rng.uniform(0.0, 1.5))
                tt = t
            if rng.random() < 0.08:
                st = INVALID
            else:
                bgf = 0.0 if rng.random() < 0.05 else float(rng.uniform(0.0, 1.0))
                st = bg(float(rng.uniform(0.0, 255.0)), bgf, float(rng.uniform(0.0, 1.0)), float(rng.uniform(0, 20)))
            steps.append((st, tt))
            if rng.random() < 0.01:
                steps.append("reset")
        cases.append(replay(f"random_{seed_i}", model, window_s, steps))
    return cases


# --------------------------------------------------------------------------------------------------- regions

FW, FH = 1280, 720
FACE = {0: (600.0, 260.0), 1: (620.0, 245.0), 2: (580.0, 245.0), 3: (640.0, 255.0), 4: (560.0, 255.0)}
SHOULDERS = ((700.0, 400.0), (500.0, 400.0))


def kpts(head: dict[int, tuple[float, float]], shoulders: Any = None,
         conf: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((17, 2))
    c = np.zeros(17)
    for i, p in head.items():
        xy[i], c[i] = p, conf
    if shoulders is not None:
        xy[5], xy[6] = shoulders
        c[5] = c[6] = conf
    return xy, c


def head_case(name: str, xy: np.ndarray, conf: np.ndarray, fw: float = FW, fh: float = FH,
              **kwargs: float) -> dict[str, Any]:
    box, err = attempt(head_box_px, xy, conf, fw, fh, **kwargs)
    return {"name": name, "xy": xy.ravel().tolist(), "conf": conf.tolist(), "frame": [fw, fh],
            "scale": kwargs.get("scale", 1.8), "min_conf": kwargs.get("min_conf", 0.3),
            "expect": {"error": err} if err else (None if box is None else list(box))}


def clipped_box(cx: float, cy: float, half: float) -> tuple[int, int, int, int] | None:
    """head_box_px's last step for a given centre and half side, in a 1280 x 720 frame."""
    box = (max(0, math.floor(cx - half)), max(0, math.floor(cy - half)),
           min(FW, math.ceil(cx + half)), min(FH, math.ceil(cy + half)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def fma_sensitive_cases(rng: np.random.Generator, pair: tuple[int, int], factor: float, want: int) -> list[dict]:
    """Boxes whose integer edges change if np.linalg.norm (BLAS ddot: fma(dy, dy, dx * dx)) is replaced by
    sqrt(dx * dx + dy * dy); found by nudging `scale` so that an edge lands within an ulp of an integer."""
    out: list[dict] = []
    tries = 0
    while len(out) < want:
        tries += 1
        assert tries < 100_000, "no fma-sensitive head box found"
        xy, c = np.zeros((17, 2)), np.zeros(17)
        a = rng.uniform([300.0, 200.0], [900.0, 450.0])
        b = a + rng.uniform(-60.0, 60.0, 2) * (4.0 if pair[0] == 5 else 1.0)
        xy[pair[0]], xy[pair[1]] = a, b
        c[pair[0]] = c[pair[1]] = 0.9
        if pair[0] == 5:
            xy[0], c[0] = (a + b) / 2.0 - np.array([0.0, 150.0]), 0.9
        d = xy[pair[0]] - xy[pair[1]]
        n_fma, n_naive = float(np.linalg.norm(d)), math.sqrt(d[0] * d[0] + d[1] * d[1])
        if n_fma == n_naive:
            continue
        head = xy[[i for i in range(5) if c[i] > 0]]
        cx, cy = head.mean(axis=0)
        span = float((head.max(axis=0) - head.min(axis=0)).max())
        sp_f, sp_n = max(span, factor * n_fma), max(span, factor * n_naive)
        base = 2.0 * (cx - math.floor(cx - sp_f / 2.0)) / sp_f  # puts cx - half on an integer
        for k in (0, 1, -1, 2, -2, 3, -3):
            sc = base
            for _ in range(abs(k)):
                sc = math.nextafter(sc, INF if k > 0 else -INF)
            if clipped_box(cx, cy, sc * sp_f / 2.0) != clipped_box(cx, cy, sc * sp_n / 2.0):
                out.append(head_case(f"fma_{pair[0]}{pair[1]}_{len(out)}", xy, c, scale=sc))
                break
    return out


def regions_cases() -> dict[str, Any]:
    rng = np.random.default_rng(1729)
    heads = [
        head_case("face", *kpts(FACE)),
        head_case("nose_shoulders", *kpts({0: (600.0, 260.0)}, SHOULDERS)),
        head_case("face_narrow_shoulders", *kpts(FACE, ((620.0, 400.0), (580.0, 400.0)))),
        head_case("face_wide_shoulders", *kpts(FACE, ((900.0, 420.0), (300.0, 380.0)))),
        head_case("single_ear", *kpts({3: (640.0, 300.0)})),
        head_case("single_nose_int", *kpts({0: (600.0, 300.0)})),
        head_case("face_scale_1", *kpts(FACE), scale=1.0),
        head_case("face_scale_0", *kpts(FACE), scale=0.0),
        head_case("half_px_scale_0", *kpts({0: (600.5, 260.25)}), scale=0.0),
        head_case("face_scale_negative", *kpts(FACE), scale=-1.0),
        head_case("face_scale_nan", *kpts(FACE), scale=NAN),
        head_case("face_scale_inf", *kpts(FACE), scale=INF),
        head_case("shoulders_only", *kpts({}, SHOULDERS)),
        head_case("nothing", *kpts({})),
        head_case("conf_0.29", *kpts(FACE, conf=0.29)),
        head_case("conf_0.3", *kpts(FACE, conf=0.3)),
        head_case("min_conf_0", *kpts(FACE), min_conf=0.0),
        head_case("min_conf_nan", *kpts(FACE), min_conf=NAN),
        head_case("eyes_nose", *kpts({0: FACE[0], 1: FACE[1], 2: FACE[2]})),
        head_case("eyes_only", *kpts({1: FACE[1], 2: FACE[2]})),
        head_case("one_eye_nose", *kpts({0: FACE[0], 1: FACE[1]})),
        head_case("ears_only", *kpts({3: FACE[3], 4: FACE[4]})),
        head_case("corner", *kpts({0: (10.0, 10.0), 3: (40.0, 15.0), 4: (-20.0, 15.0)})),
        head_case("edge", *kpts({0: (1275.0, 715.0), 3: (1300.0, 720.0), 4: (1250.0, 720.0)})),
        head_case("outside", *kpts({0: (-500.0, -500.0), 3: (-480.0, -500.0)})),
        head_case("outside_far", *kpts({0: (5000.0, 300.0)})),
        head_case("small_frame", *kpts(FACE), fw=64, fh=48),
        head_case("float_frame", *kpts({0: (630.0, 700.0), 3: (660.0, 705.0)}), fw=640.5, fh=710.25),
        head_case("nan_frame_fallback", *kpts({0: (600.0, 300.0)}), fw=NAN, fh=720),
        head_case("nan_frame_spread", *kpts(FACE), fw=NAN, fh=720),
        head_case("nan_frame_h", *kpts(FACE), fw=FW, fh=NAN),
        head_case("huge_coords", *kpts({0: (1e308, 0.0), 3: (-1e308, 0.0)})),
        head_case("huge_eyes", *kpts({1: (1e308, 5.0), 2: (-1e308, 5.0)})),
        head_case("huge_mean", *kpts({0: (1.5e308, 10.0), 1: (1.5e308, 10.0)})),
    ]
    xy, c = kpts(FACE)
    xy[3], c[3] = (1200.0, 700.0), 0.1
    heads.append(head_case("low_conf_outlier", xy, c))
    for name, (k, v) in {"nan_nose": (0, (NAN, 260.0)), "inf_ear": (3, (INF, 255.0)),
                         "neg_inf_eye": (1, (620.0, -INF)), "nan_shoulder": (5, (NAN, 400.0))}.items():
        xy, c = kpts(FACE, SHOULDERS)
        xy[k] = v
        heads.append(head_case(name, xy, c))
    xy, c = kpts(FACE)
    c[1] = NAN
    heads.append(head_case("nan_conf_eye", xy, c))
    xy, c = kpts(FACE, SHOULDERS)
    c[6] = 0.2
    heads.append(head_case("one_shoulder", xy, c))
    xy, c = kpts({0: (600.0, 260.0)}, ((603.0, 400.0), (600.0, 404.0)))
    heads.append(head_case("tiny_shoulders_3_4_5", xy, c))
    for k in range(250):
        xy = np.column_stack([rng.uniform(-100.0, FW + 100.0, 17), rng.uniform(-100.0, FH + 100.0, 17)])
        if k % 3 == 0:  # a compact face so boxes are realistic
            centre = rng.uniform([0.0, 0.0], [FW, FH])
            xy[:5] = centre + rng.normal(0.0, rng.uniform(1.0, 40.0), (5, 2))
            xy[5:7] = centre + np.array([[90.0, 140.0], [-90.0, 140.0]]) * rng.uniform(0.2, 2.0)
        if k % 7 == 0:
            xy = np.round(xy)
        c = rng.uniform(0.0, 1.0, 17)
        c[rng.random(17) < 0.3] = 0.0
        fw, fh = (FW, FH) if k % 5 else (int(rng.integers(32, 2000)), int(rng.integers(32, 2000)))
        heads.append(head_case(f"random_{k}", xy, c, fw=fw, fh=fh, scale=float(rng.uniform(0.3, 3.0)),
                               min_conf=float(rng.choice([0.3, 0.5, 0.05]))))
    heads += fma_sensitive_cases(rng, (1, 2), 2.0, 6)
    heads += fma_sensitive_cases(rng, (5, 6), 0.6, 4)

    norms_in = [(128.0, 72.0, 640.0, 360.0, FW, FH), (-10.0, -5.0, 1400.0, 800.0, FW, FH), (10, 20, 30, 40, 100, 100),
                (NAN, 0.0, INF, -INF, FW, FH), (-0.0, 0.0, 1280.0, 720.0, FW, FH), (5.0, 5.0, 6.0, 6.0, NAN, 10),
                (5.0, INF, 6.0, 6.0, INF, 10), (1.0, 1.0, 2.0, 2.0, 0, 100), (1.0, 1.0, 2.0, 2.0, 100, 0),
                (1.0, 1.0, 2.0, 2.0, -3, 100), (1.0, 1.0, 2.0, 2.0, 100, -0.0), (1e308, -1e308, 3.0, 1e-320, 1e-10, 3),
                (0.1, 0.2, 0.3, 0.7, 0.3, 0.7)]
    norms = []
    for x1, y1, x2, y2, fw, fh in norms_in:
        out, err = attempt(norm_bbox, (x1, y1, x2, y2), fw, fh)
        norms.append({"box": [x1, y1, x2, y2], "frame": [fw, fh], "expect": {"error": err} if err else list(out)})
    for bad in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]):
        _, err = attempt(norm_bbox, bad, 100, 100)
        norms.append({"box": bad, "frame": [100, 100], "expect": {"error": err}})
    for _ in range(150):
        box = rng.uniform(-200.0, 2200.0, 4)
        fw, fh = float(rng.choice([1280, 640, 1920, 333])), float(rng.choice([720, 480, 1080, 177]))
        norms.append({"box": box.tolist(), "frame": [fw, fh], "expect": list(norm_bbox(box, fw, fh))})

    vecs = rng.normal(0.0, 300.0, (400, 2)) * rng.choice([1e-3, 1.0, 1e3], (400, 1))
    vecs = np.vstack([vecs, [[0.0, 0.0], [3.0, 4.0], [-0.0, 5.0], [1e-320, 1e-320], [1e200, 1e200], [1e154, 1e154],
                             [INF, 1.0], [1.0, -INF]]])
    norm_cases = [[float(d[0]), float(d[1]), float(np.linalg.norm(d))] for d in vecs]

    return {"head_box": heads, "norm_bbox": norms, "norm2": norm_cases, "fma": fma_cases(rng)}


def ieee_fma(a: float, b: float, c: float) -> float:
    """math.fma, with IEEE results where Python raises (overflow -> +-inf, invalid -> NaN)."""
    try:
        return math.fma(a, b, c)
    except OverflowError:
        return INF if Fraction(a) * Fraction(b) + Fraction(c) > 0 else -INF
    except ValueError:
        return NAN


def fma_cases(rng: np.random.Generator) -> list[list[float]]:
    out = []
    for _ in range(600):
        e = rng.integers(-1100, 1000, 3)
        a, b, c = (float(np.ldexp(rng.uniform(-2.0, 2.0), int(k))) for k in e)
        out.append((a, b, c))
    for _ in range(300):  # near-cancellation: c ~ -a*b
        a, b = float(rng.normal(0, 100)), float(rng.normal(0, 100))
        out.append((a, b, -(a * b) * (1.0 + float(rng.choice([0.0, 1e-16, -1e-16, 2.2e-16])))))
    for _ in range(200):
        dx, dy = (float(v) for v in rng.normal(0, 300, 2))
        out.append((dy, dy, dx * dx))
    tiny, big = 5e-324, 1.7976931348623157e308
    special = [0.0, -0.0, 1.0, -1.0, tiny, -tiny, 2.2250738585072014e-308, big, -big, INF, -INF, NAN, 1e-160, 3.0]
    for a in special:
        for b in special:
            for c in (0.0, -0.0, tiny, -1.0, big, INF, -INF, NAN):
                out.append((a, b, c))
    out += [(1.0 + 2.0 ** -52, 1.0 - 2.0 ** -53, -1.0), (2.0 ** 512, 2.0 ** 512, -big), (big, 2.0, -big),
            (2.0 ** -537, 2.0 ** -537, 0.0), (2.0 ** -537, -(2.0 ** -537), 0.0), (2.0 ** -1000, 2.0 ** -75, tiny)]
    return [[a, b, c, ieee_fma(a, b, c)] for a, b, c in out]


# --------------------------------------------------------------------------------------------------- luma


def gray15(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    r, g, b = (v.astype(np.uint32) for v in (r, g, b))
    return ((9798 * r + 19235 * g + 3735 * b + (1 << 14)) >> 15).astype(np.uint8)


def cv_gray(rgb: np.ndarray) -> np.ndarray:
    """cv2 BGR2GRAY of an (n, 3) array of RGB triples."""
    bgr = np.ascontiguousarray(rgb[:, ::-1]).reshape(1, -1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).reshape(-1)


def check_all_triples() -> np.ndarray:
    """Assert the luma.ts formula on every 8-bit colour; return the triples where the 14-bit formula disagrees."""
    v = np.arange(256, dtype=np.uint8)
    rr, gg, bb = (a.reshape(-1) for a in np.meshgrid(v, v, v, indexing="ij"))
    bgr = np.stack([bb, gg, rr], axis=-1).reshape(4096, 4096, 3)
    ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).reshape(-1)
    mismatches = int(np.count_nonzero(gray15(rr, gg, bb) != ref))
    assert mismatches == 0, f"luma.ts formula differs from cv2 {cv2.__version__} on {mismatches} colours"
    r32, g32, b32 = (a.astype(np.uint32) for a in (rr, gg, bb))
    old = ((4899 * r32 + 9617 * g32 + 1868 * b32 + (1 << 13)) >> 14).astype(np.uint8)
    idx = np.flatnonzero(old != ref)
    return np.stack([rr[idx], gg[idx], bb[idx]], axis=-1)


def luma_cases() -> dict[str, Any]:
    disagree = check_all_triples()
    rng = np.random.default_rng(31337)
    levels = np.array([0, 1, 2, 64, 127, 128, 129, 253, 254, 255], dtype=np.uint8)
    edges_ = np.stack([a.reshape(-1) for a in np.meshgrid(levels, levels, levels, indexing="ij")], axis=-1)
    rand = rng.integers(0, 256, (3000, 3), dtype=np.uint8)
    picks = disagree[rng.choice(len(disagree), 600, replace=False)]
    rgb = np.vstack([edges_, rand, picks]).astype(np.uint8)
    gray = cv_gray(rgb)
    h, w = 7, 13
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    img_gray = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, 2::-1]), cv2.COLOR_BGR2GRAY)
    assert np.array_equal(img_gray, cv2.cvtColor(np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]]),
                                                 cv2.COLOR_BGRA2GRAY))
    return {"cv2_version": cv2.__version__, "n": len(rgb), "rgb_b64": b64(rgb), "gray_b64": b64(gray),
            "rgba": {"h": h, "w": w, "rgba_b64": b64(rgba), "gray_b64": b64(img_gray)}}


# --------------------------------------------------------------------------------------------------- main


def export(out_dir: Path) -> None:
    with np.errstate(all="ignore"):  # overflow cases are deliberate
        luma = luma_cases()  # first: it asserts the formula on all 256^3 colours
        write(out_dir, "lux-stats", stats_cases())
        write(out_dir, "lux-estimator", {"predict": predict_cases(), "from_dict": from_dict_cases(),
                                         "sequences": estimator_cases()})
        write(out_dir, "lux-regions", regions_cases())
        write(out_dir, "lux-luma", luma)


if __name__ == "__main__":
    export(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
