import math

import numpy as np

from valence.config import Settings
from valence.features.build_dataset import build
from valence.features.compute import FEATURE_NAMES, FEATURE_SETS, compute, features_from_window
from valence.features.window import RollingWindow
from valence.runtime.engine import ScenarioSource
from valence.sim.scenarios import render


def test_feature_names_and_sets():
    assert len(FEATURE_NAMES) == 114 and len(FEATURE_SETS["m2"]) == 110
    for name, cols in FEATURE_SETS.items():
        assert set(cols) <= set(FEATURE_NAMES), name
    assert not set(FEATURE_SETS["m2_noface"]) & set(FEATURE_SETS["m0t"])


def test_live_window_matches_offline_builder():
    """Feed a scenario through the live RollingWindow and compare with the offline builder (full history)."""
    df = render("full_demo")
    src = ScenarioSource(df, "full_demo")
    window = RollingWindow(Settings().window_ticks)
    live = {}
    while True:
        recs = src.next_tick(0.0, 0.0)
        if recs is None:
            break
        for r in recs:
            window.append(r)
        for r in recs:
            if r.person_id != 0:
                live[r.tick] = features_from_window(window.snapshot(r.person_id), window.room())
    X, meta = build(df, step=1, fixed_len=180, workers=1)
    # fixed_len=180 means "as much history as exists, up to 15 min" for both person and room windows.
    # The live room window is continuous, so compare ticks where the person was present from the start.
    checked = 0
    for (_, row), x in zip(meta.iterrows(), X.to_numpy()):
        if row["tick"] in live and row["tick"] < 14 * 12:  # before the break in full_demo
            np.testing.assert_allclose(live[row["tick"]], x, rtol=0, atol=1e-9, equal_nan=True)
            checked += 1
    assert checked > 100


def test_no_lookahead():
    """Changing the future must not change features computed at an earlier tick."""
    rng = np.random.default_rng(0)
    n = 120
    person = {f: rng.random(n) for f in ("motion_energy", "posture_slump", "head_in_hands_frac", "clenched_fists_count",
                                         "social_withdrawal_flag", "n_people", "hours_since_break", "tod_h",
                                         "face_valence", "face_visible")}
    room = {f: rng.random(n) * 60 for f in ("ambient_db", "light_lux", "laughter_count", "compliments_count")}
    at = 80
    before = compute({k: v[:at] for k, v in person.items()}, {k: v[:at] for k, v in room.items()})
    for v in list(person.values()) + list(room.values()):
        v[at:] = 999.0
    after = compute({k: v[:at] for k, v in person.items()}, {k: v[:at] for k, v in room.items()})
    np.testing.assert_array_equal(before, after)


def test_short_and_missing_windows_do_not_crash():
    nan = math.nan
    person = {f: np.array([nan]) for f in ("motion_energy", "posture_slump", "head_in_hands_frac", "clenched_fists_count",
                                           "social_withdrawal_flag", "n_people", "hours_since_break", "tod_h",
                                           "face_valence", "face_visible")}
    room = {f: np.array([nan]) for f in ("ambient_db", "light_lux", "laughter_count", "compliments_count")}
    x = compute(person, room)
    assert x.shape == (114,)
