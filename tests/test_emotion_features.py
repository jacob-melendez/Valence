"""Emotion map (docs/EMOTION_FEATURES.md): simulator cues, per-frame cues, tick aggregation and window features."""

import hashlib
import math

import numpy as np
import pytest

from valence.config import Settings
from valence.contracts import AFFECT_ANCHORS, EMOTION_STATES, PersonObs, TickRecord, VideoObs
from valence.features.compute import FEATURE_NAMES, features_from_window
from valence.features.emotion import (EMOTION_FEATURE_NAMES, EMOTION_FIELDS, INPUT_FEATURES, compute_emotion,
                                      emotion_from_window, model_input)
from valence.features.face_cues import FaceCueTracker, head_pose_deg, raw_cues
from valence.features.window import RollingWindow
from valence.runtime.aggregator import Aggregator
from valence.runtime.engine import ScenarioSource
from valence.sim.generator import simulate_corpus, simulate_session
from valence.sim.scenarios import render

OLD_COLUMNS = ["tick", "ts", "t_mono", "session_id", "person_id", "display_id", "n_people", "source", "ambient_db",
               "db_p90", "light_lux", "lux_quality", "room_db_base", "room_lux_base", "laughter_count",
               "compliments_count", "clenched_fists_count", "posture_slump", "head_in_hands_frac",
               "social_withdrawal_flag", "motion_energy", "face_valence", "face_visible", "hours_since_break", "tod_h",
               "true_state", "episode_age_s"]


def _hash(df) -> str:
    df = df.copy()
    df.loc[df["person_id"] == 0, "true_state"] = ""  # room-row ties are broken by set order (hash seed)
    return hashlib.sha256(df[OLD_COLUMNS].to_csv(index=False, float_format="%.12g").encode()).hexdigest()[:16]


# Hashes of the pre-event simulator's output (commit 09ea054) on the original columns.
BASELINE = {"full_demo": "14353508323d54b9", "tense": "f2c3b3b8828c3218", "corpus_core": "379179a72bb9be7e",
            "corpus_all": "177b331b595a8cff"}


def test_existing_simulator_output_unchanged():
    assert _hash(render("full_demo")) == BASELINE["full_demo"]
    assert _hash(render("tense", state_set="all")) == BASELINE["tense"]
    assert _hash(simulate_corpus(3, 11, state_set="core")) == BASELINE["corpus_core"]
    assert _hash(simulate_corpus(3, 11, state_set="all")) == BASELINE["corpus_all"]


def test_emotion_state_set_simulates_all_ten_states_with_anchors():
    df = simulate_corpus(12, 3, state_set="emotion")
    people = df[df["person_id"] != 0]
    assert set(people["true_state"].dropna()) == set(EMOTION_STATES)
    for state, g in people.groupby("true_state"):
        v, a = AFFECT_ANCHORS[state]
        assert abs(g["true_valence"].mean() - v) < 0.12 and abs(g["true_arousal"].mean() - a) < 0.12, state
    # The new cues move the way the spec says (simulator assumptions, docs/EMOTION_FEATURES.md section 4).
    m = people.groupby("true_state")[["brow_down", "eye_open", "head_away", "blink_count"]].mean()
    assert m.loc["Confused", "brow_down"] > m.loc["Focused", "brow_down"] + 0.08
    assert m.loc["Exhausted", "eye_open"] < m.loc["Focused", "eye_open"] - 0.08
    assert m.loc["Bored", "head_away"] > m.loc["Focused", "head_away"] + 0.2
    assert m.loc["Anxious", "blink_count"] > 2 * m.loc["Focused", "blink_count"]


def test_hidden_face_gives_nan_cues():
    df = simulate_session("nan-check", np.random.default_rng(0), state_set="emotion")
    people = df[df["person_id"] != 0]
    hidden = people["face_visible"] == 0
    assert hidden.any()
    for c in ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away"):
        assert people.loc[hidden, c].isna().all() and people.loc[~hidden, c].notna().all(), c
    assert (people.loc[hidden, "blink_count"] == 0).all()
    room = df[df["person_id"] == 0]
    assert room["brow_down"].isna().all() and (room["blink_count"] == 0).all()


def test_head_pose_signs_and_layout():
    def matrix(yaw, pitch, row_major=False):
        y, p = math.radians(yaw), math.radians(pitch)
        f = (math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p))
        m = [[1, 0, f[0], 0], [0, 1, f[1], 0], [0, 0, f[2], -50.0], [0, 0, 0, 1]]
        return [m[r][c] for r in range(4) for c in range(4)] if row_major else [m[r][c] for c in range(4) for r in range(4)]

    for row_major in (False, True):
        yaw, pitch = head_pose_deg(matrix(30, -10, row_major))
        assert yaw == pytest.approx(30) and pitch == pytest.approx(-10)
    assert head_pose_deg([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -40, 1]) == pytest.approx((0.0, 0.0))


def test_tracker_baseline_and_blinks():
    tr = FaceCueTracker()
    neutral = {"browDownLeft": 0.1, "browDownRight": 0.1, "eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1,
               "mouthFrownLeft": 0.0, "mouthFrownRight": 0.0, "mouthSmileLeft": 0.05, "mouthSmileRight": 0.05}
    ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -40, 1]
    t = 0.0
    for _ in range(25 * 15):  # 25 s at 15 fps: calibrates and freezes
        out = tr.update(raw_cues(neutral, ident), t)
        t += 1000 / 15
    assert out["brow_down"] == pytest.approx(0.0) and out["eye_open"] == pytest.approx(0.0)
    closed = {**neutral, "eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.8}
    blinks = []
    for frame in [closed, closed, neutral, closed] + [closed] * 12 + [neutral]:  # a blink, then a 1 s closure
        blinks.append(tr.update(raw_cues(frame, ident), t)["blink_onsets"])
        t += 1000 / 15
    assert sum(blinks) == 1 and blinks[2] == 1
    lowered = {**neutral, "browDownLeft": 0.4, "browDownRight": 0.3}
    assert tr.update(raw_cues(lowered, ident), t)["brow_down"] == pytest.approx(0.25)
    assert math.isnan(tr.update(None, t + 70)["brow_down"])


def test_aggregator_cue_medians_and_away_share():
    agg = Aggregator(Settings(), "agg-cues")
    yaws = [5.0, -30.0, 10.0, -8.0]
    for i, yaw in enumerate(yaws):
        p = PersonObs(track_id=1, bbox=(0.2, 0.1, 0.8, 0.9), face_visible=True, face_valence=0.1, brow_down=0.1 * i,
                      eye_open=-0.02, mouth_frown=0.01, smile=0.2, head_yaw=yaw, head_pitch=-5.0 if i else -25.0,
                      blink_onsets=int(i == 2))
        agg.push(VideoObs(t=i * 0.1, wall=1000.0 + i * 0.1, frame_seq=i, frames_decoded=i + 1, persons=(p,)))
    agg.push(VideoObs(t=0.5, wall=1000.5, frame_seq=5, frames_decoded=5,
                      persons=(PersonObs(track_id=1, bbox=(0.2, 0.1, 0.8, 0.9)),)))
    room, rec = agg.flush(1001.0, 1.0, 12.0)
    assert rec.brow_down == pytest.approx(0.15) and rec.head_yaw == pytest.approx(9.0)
    assert rec.head_pitch == pytest.approx(-5.0) and rec.head_away == pytest.approx(0.5) and rec.blink_count == 1
    assert rec.face_visible == pytest.approx(0.8)
    assert math.isnan(room.brow_down) and room.blink_count == 0
    # Old producers (no cues) still give NaN cues and the same existing fields.
    agg.push(VideoObs(t=2.0, wall=1002.0, frame_seq=6, frames_decoded=6,
                      persons=(PersonObs(track_id=1, bbox=(0.2, 0.1, 0.8, 0.9), face_visible=True, face_valence=0.2),)))
    _, rec2 = agg.flush(1003.0, 3.0, 12.0)
    assert math.isnan(rec2.brow_down) and math.isnan(rec2.head_away) and rec2.face_valence == pytest.approx(0.2)


def test_window_features_formulas_and_missing_face():
    n = 70
    vis = np.full(n, 1.0)
    vis[-6:] = 0.0  # last 30 s: no face
    person = {f: np.full(n, 0.1) for f in EMOTION_FIELDS}
    person["face_visible"] = vis
    person["blink_count"] = np.where(vis > 0, 1.0, 0.0)
    for f in ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away"):
        person[f][-6:] = np.nan
    x = dict(zip(EMOTION_FEATURE_NAMES, compute_emotion(person)))
    assert math.isnan(x["brow_now"]) and x["brow_mean_1m"] == pytest.approx(0.1) and x["brow_std_5m"] == pytest.approx(0.0, abs=1e-12)
    assert x["blink_rate_1m"] == pytest.approx(60 * 6 / 30)  # 6 visible ticks of 5 s, 1 blink each
    assert x["blink_rate_5m"] == pytest.approx(12.0)
    none = {f: np.full(3, np.nan) for f in EMOTION_FIELDS}
    none["face_visible"] = np.zeros(3)
    none["blink_count"] = np.zeros(3)
    assert np.isnan(compute_emotion(none)).all()
    one = {f: np.array([0.2]) for f in EMOTION_FIELDS}
    out = dict(zip(EMOTION_FEATURE_NAMES, compute_emotion(one)))
    assert out["smile_now"] == 0.2 and math.isnan(out["blink_rate_1m"])  # 1 s of visible face < 10 s


def test_blink_rate_is_missing_when_no_cues_were_produced():
    n = 24
    person = {f: np.full(n, np.nan) for f in EMOTION_FIELDS}
    person["face_visible"] = np.ones(n)  # an older producer: face visible, no cues, blink_count 0
    person["blink_count"] = np.zeros(n)
    x = dict(zip(EMOTION_FEATURE_NAMES, compute_emotion(person)))
    assert all(math.isnan(x[f"blink_rate_{w}"]) for w in ("1m", "5m", "15m"))
    person["eye_open"][-3:] = 0.0  # cues on the last 3 ticks only: 15 s of cued face, 0 blinks
    x = dict(zip(EMOTION_FEATURE_NAMES, compute_emotion(person)))
    assert x["blink_rate_1m"] == 0.0
    vis = np.array([22, 16, 16, 20, 0, 21, 20, 19, 21, 18, 21, 20], dtype=float) / 107.0  # 10 s, pairwise sum < 10
    edge = {f: np.zeros(12) for f in EMOTION_FIELDS}
    edge["face_visible"] = vis
    edge["blink_count"] = np.r_[np.ones(6), np.zeros(6)]
    assert dict(zip(EMOTION_FEATURE_NAMES, compute_emotion(edge)))["blink_rate_1m"] == pytest.approx(36.0)


def test_one_sided_blendshapes_and_null_blinks():
    out = raw_cues({"browDownLeft": 0.2, "browDownRight": float("nan"), "eyeBlinkLeft": 0.1}, None)
    assert out["brow"] == pytest.approx(0.2) and out["closure"] == pytest.approx(0.1) and math.isnan(out["yaw"])
    agg = Aggregator(Settings(), "null-blink")
    p = PersonObs(track_id=1, bbox=(0.2, 0.1, 0.8, 0.9), face_visible=True, face_valence=0.0, eye_open=0.0,
                  blink_onsets=float("nan"))
    agg.push(VideoObs(t=0.0, wall=1000.0, frame_seq=0, frames_decoded=1, persons=(p,)))
    _, rec = agg.flush(1005.0, 5.0, 12.0)
    assert rec.blink_count == 0


def test_no_lookahead():
    rng = np.random.default_rng(0)
    person = {f: rng.random(120) for f in EMOTION_FIELDS}
    before = compute_emotion({k: v[:80] for k, v in person.items()})
    for v in person.values():
        v[80:] = 999.0
    np.testing.assert_array_equal(before, compute_emotion({k: v[:80] for k, v in person.items()}))


def test_live_window_matches_offline_arrays():
    df = render("full_demo")
    src = ScenarioSource(df, "full_demo")
    window = RollingWindow(Settings().window_ticks)
    people = df[df["person_id"] == 1].sort_values("tick")
    checked = 0
    while True:
        recs = src.next_tick(0.0, 0.0)
        if recs is None:
            break
        for r in recs:
            window.append(r)
        for r in recs:
            if r.person_id == 1 and r.tick < 14 * 12:
                live = emotion_from_window(window.snapshot(1))
                hist = people[people["tick"] <= r.tick].tail(180)
                off = compute_emotion({f: hist[f].to_numpy(dtype=float) for f in EMOTION_FIELDS})
                np.testing.assert_allclose(live, off, rtol=0, atol=1e-12, equal_nan=True)
                x = model_input(features_from_window(window.snapshot(1), window.room()), live)
                assert x.shape == (len(INPUT_FEATURES),)
                checked += 1
    assert checked > 100


def test_tick_record_defaults_keep_old_rows_valid():
    rec = TickRecord(tick=0, ts=0.0, t_mono=0.0, session_id="s", person_id=1, display_id="Person #1", n_people=1)
    d = rec.to_dict()
    assert d["brow_down"] is None and d["blink_count"] == 0 and len(FEATURE_NAMES) == 123
