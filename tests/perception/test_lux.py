from __future__ import annotations

import math
import time

import numpy as np
import pytest

from valence.perception.lux import BgStats, LuxEstimator, LuxModel, background_stats

H, W = 360, 640


def noisy(base: np.ndarray | float, sigma: float, seed: int = 0, shape: tuple[int, int] = (H, W)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.broadcast_to(np.asarray(base, dtype=np.float64), shape) + rng.normal(0.0, sigma, shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("sigma", [2.0, 5.0, 10.0, 20.0])
def test_sigma_recovered_flat(sigma: float) -> None:
    stats = background_stats(noisy(128.0, sigma), [])
    assert stats.sigma == pytest.approx(sigma, rel=0.2)
    assert stats.y_bg == pytest.approx(128.0, abs=1.0)
    assert stats.bg_frac == 1.0
    assert stats.clipped_frac < 0.01


@pytest.mark.parametrize("sigma", [2.0, 8.0, 20.0])
def test_sigma_ignores_smooth_gradient(sigma: float) -> None:
    ramp = np.tile(np.linspace(60.0, 190.0, W), (H, 1))
    stats = background_stats(noisy(ramp, sigma, seed=1), [])
    assert stats.sigma == pytest.approx(sigma, rel=0.2)


def test_sigma_at_native_resolution_small_image() -> None:
    stats = background_stats(noisy(100.0, 6.0, shape=(120, 160)), [])
    assert stats.sigma == pytest.approx(6.0, rel=0.2)


def test_person_boxes_masked() -> None:
    img = noisy(50.0, 3.0, seed=2)
    img[90:270, 160:480] = noisy(220.0, 25.0, seed=3, shape=(180, 320))
    box = (160 / W, 90 / H, 480 / W, 270 / H)
    stats = background_stats(img, [box])
    assert stats.y_bg == pytest.approx(50.0, abs=1.0)
    assert stats.sigma == pytest.approx(3.0, rel=0.2)
    assert stats.bg_frac == pytest.approx(1.0 - 0.25, abs=0.02)
    unmasked = background_stats(img, [])
    assert unmasked.y_bg > 80.0
    assert unmasked.sigma > 5.0


def test_clipped_fraction_on_background_only() -> None:
    img = np.full((H, W), 128, dtype=np.uint8)
    img[:, : W // 2] = 0
    stats = background_stats(img, [])
    assert stats.clipped_frac == pytest.approx(0.5, abs=0.01)
    img[:, W // 2:] = 255
    assert background_stats(img, []).clipped_frac == pytest.approx(1.0)
    masked = np.full((H, W), 128, dtype=np.uint8)
    masked[:, : W // 4] = 252
    assert background_stats(masked, [(0.0, 0.0, 0.25, 1.0)]).clipped_frac == pytest.approx(0.0)


def test_all_person_and_degenerate_boxes() -> None:
    img = noisy(100.0, 5.0)
    full = background_stats(img, [(0.0, 0.0, 1.0, 1.0)])
    assert full.bg_frac == 0.0 and math.isnan(full.y_bg) and not full.valid
    flipped = background_stats(img, [(0.5, 0.5, 0.0, 0.0)])
    assert flipped.bg_frac == pytest.approx(0.75, abs=0.01)
    ignored = background_stats(img, [(math.nan, 0.0, 1.0, 1.0)])
    assert ignored.bg_frac == 1.0


def test_rejects_non_uint8_gray() -> None:
    with pytest.raises(ValueError):
        background_stats(np.zeros((10, 10), dtype=np.float32), [])
    with pytest.raises(ValueError):
        background_stats(np.zeros((10, 10, 3), dtype=np.uint8), [])


def test_background_stats_fast() -> None:
    img = noisy(120.0, 6.0)
    boxes = [(0.1, 0.2, 0.4, 0.9), (0.6, 0.1, 0.8, 0.95)]
    background_stats(img, boxes)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        background_stats(img, boxes)
        times.append(time.perf_counter() - t0)
    assert min(times) < 2e-3


def readings_from(a: float, b: float, c: float, n: int = 6, seed: int = 0) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    ys = np.linspace(40.0, 200.0, n)
    sigmas = rng.uniform(1.0, 12.0, n)
    return [(float(y), float(s), float(math.exp(a + b * y + c * s))) for y, s in zip(ys, sigmas)]


def test_fit_recovers_known_coefficients() -> None:
    a, b, c = 2.0, 0.02, -0.08
    exact = LuxModel.fit(readings_from(a, b, c), ridge=0.0)
    assert (exact.a, exact.b, exact.c) == pytest.approx((a, b, c), abs=1e-9)
    model = LuxModel.fit(readings_from(a, b, c))
    assert model.calibrated
    assert model.a == pytest.approx(a, rel=0.05)
    assert model.b == pytest.approx(b, rel=0.05)
    assert model.c == pytest.approx(c, rel=0.05)


def test_fit_three_readings() -> None:
    model = LuxModel.fit(readings_from(1.5, 0.03, -0.1, n=3, seed=4), ridge=0.0)
    assert (model.a, model.b, model.c) == pytest.approx((1.5, 0.03, -0.1), abs=1e-6)


def test_fit_constant_sigma_gives_zero_c() -> None:
    rd = [(y, 4.0, math.exp(1.0 + 0.02 * y)) for y in (40.0, 120.0, 200.0)]
    model = LuxModel.fit(rd)
    assert model.c == pytest.approx(0.0, abs=1e-9)
    assert model.b == pytest.approx(0.02)


def test_two_point_fit() -> None:
    model = LuxModel.fit([(40.0, 9.0, 60.0), (180.0, 2.0, 600.0)])
    assert model.c == 0.0 and model.calibrated
    assert model.predict(40.0, 123.0) == pytest.approx(60.0)
    assert model.predict(180.0, math.nan) == pytest.approx(600.0)


def test_y_only_least_squares() -> None:
    model = LuxModel.fit(readings_from(2.0, 0.02, 0.0, n=5), use_sigma=False)
    assert model.c == 0.0
    assert model.b == pytest.approx(0.02)


def test_fit_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        LuxModel.fit([(50.0, 2.0, 100.0)])
    with pytest.raises(ValueError):
        LuxModel.fit([(50.0, 2.0, 100.0), (50.0, 3.0, 200.0)])
    with pytest.raises(ValueError):
        LuxModel.fit([(50.0, 2.0, 100.0), (90.0, 3.0, -1.0)])


def test_default_model_rough_mapping() -> None:
    model = LuxModel()
    assert not model.calibrated
    assert 35 <= model.predict(30.0, 5.0) <= 75
    assert 200 <= model.predict(128.0, 5.0) <= 400
    assert 600 <= model.predict(220.0, 5.0) <= 1100


def test_predict_handles_nan_and_overflow() -> None:
    assert math.isnan(LuxModel().predict(math.nan, 1.0))
    assert math.isnan(LuxModel(c=0.1).predict(100.0, math.nan))
    assert LuxModel(b=10.0).predict(255.0, 0.0) == 200_000.0


def test_dict_round_trip() -> None:
    model = LuxModel(1.0, 0.02, -0.05, True)
    assert LuxModel.from_dict(model.to_dict()) == model
    assert LuxModel.from_dict({}) == LuxModel()


def stats(y: float = 128.0, bg: float = 0.8, clipped: float = 0.1) -> BgStats:
    return BgStats(y_bg=y, sigma=3.0, bg_frac=bg, clipped_frac=clipped)


def test_estimator_constant_light() -> None:
    model = LuxModel(1.0, 0.03, 0.0, calibrated=True)
    est = LuxEstimator(model)
    for k in range(20):
        lux, q = est.update(stats(), 0.5 * k)
    assert lux == pytest.approx(model.predict(128.0, 3.0))
    assert q == pytest.approx(0.8 * 0.9)


def test_estimator_uncalibrated_penalty() -> None:
    lux, q = LuxEstimator().update(stats(), 0.0)
    assert lux == pytest.approx(LuxModel().predict(128.0, 3.0))
    assert q == pytest.approx(0.5 * 0.8 * 0.9)


def test_estimator_median_window_and_stability() -> None:
    model = LuxModel(0.0, 1.0, 0.0, calibrated=True)
    est = LuxEstimator(model, window_s=10.0)
    for k, y in enumerate([5.0, 5.0, 5.0, 6.0, 7.0]):
        lux, q = est.update(stats(y, bg=1.0, clipped=0.0), float(k))
    values = np.exp([5.0, 5.0, 5.0, 6.0, 7.0])
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    assert lux == pytest.approx(med)
    assert q == pytest.approx(1.0 / (1.0 + (q75 - q25) / med))
    assert q < 1.0
    lux, q = est.update(stats(7.0, bg=1.0, clipped=0.0), 13.5)
    assert lux == pytest.approx(math.exp(7.0))
    assert q == pytest.approx(1.0)


def test_estimator_invalid_stats() -> None:
    est = LuxEstimator(LuxModel(calibrated=True))
    lux, q = est.update(BgStats(math.nan, math.nan, 0.0, 0.0), 0.0)
    assert math.isnan(lux) and q == 0.0
    est.update(stats(), 1.0)
    lux, q = est.update(BgStats(math.nan, math.nan, 0.0, 0.0), 2.0)
    assert math.isfinite(lux) and q == 0.0


@pytest.mark.parametrize("level", [0, 255])
def test_sigma_not_diluted_by_hard_clipped_background(level: int) -> None:
    img = noisy(128.0, 5.0, seed=5)
    img[:, : W // 2] = level
    stats = background_stats(img, [])
    assert stats.clipped_frac == pytest.approx(0.5, abs=0.01)
    assert stats.sigma == pytest.approx(5.0, rel=0.2)


def test_sigma_fully_clipped_background_is_zero_not_nan() -> None:
    stats = background_stats(np.zeros((H, W), dtype=np.uint8), [])
    assert stats.sigma == pytest.approx(0.0)
    assert stats.clipped_frac == 1.0


def test_estimator_ignores_non_finite_time() -> None:
    model = LuxModel(1.0, 0.03, 0.0, calibrated=True)
    est = LuxEstimator(model, window_s=10.0)
    est.update(stats(100.0), math.nan)
    est.update(stats(100.0), 0.0)
    for k in range(1, 400):
        est.update(stats(200.0), 0.5 * k)
    assert len(est._history) <= 21
    lux, _ = est.update(stats(200.0), 200.0)
    assert lux == pytest.approx(model.predict(200.0, 3.0))
