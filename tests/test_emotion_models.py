"""Emotion map artifacts (artifacts/emotion): contract, sizes, golden parity, JSON evaluator and self-report eval."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from valence.contracts import AFFECT_ANCHORS, EMOTION_STATES
from valence.features.emotion import EMOTION_FEATURE_NAMES, EMOTION_FIELDS, INPUT_FEATURES, compute_emotion
from valence.features.face_cues import FaceCueTracker, raw_cues
from valence.models.emotion import EmotionPredictor, face_dropout
from valence.models.emotion_json import load_dump, raw_scores

ART = Path(__file__).resolve().parents[1] / "artifacts" / "emotion"
MODELS = ("classifier", "valence", "arousal")
needs_artifacts = pytest.mark.skipif(not (ART / "meta.json").exists(), reason="no trained emotion artifacts")


def _x(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


@pytest.fixture(scope="module")
def golden():
    return json.loads((ART / "golden.json").read_text())


@pytest.fixture(scope="module")
def predictor():
    return EmotionPredictor(ART)


@needs_artifacts
def test_meta_matches_the_contract():
    meta = json.loads((ART / "meta.json").read_text())
    assert meta["feature_version"] == "emotion-v1" and meta["input_features"] == INPUT_FEATURES
    assert meta["emotion_feature_names"] == EMOTION_FEATURE_NAMES
    assert [s["id"] for s in meta["states"]] == list(EMOTION_STATES)
    for s in meta["states"]:
        assert (s["valence"], s["arousal"]) == AFFECT_ANCHORS[s["id"]]
    assert meta["anchors_are_assumptions"] is True and 0.25 <= meta["temperature"] <= 8
    assert set(meta["layers"]) == set(INPUT_FEATURES)


@needs_artifacts
def test_model_files_stay_small():
    for name in MODELS:
        assert (ART / f"{name}.json.gz").stat().st_size < 3_000_000, name


@needs_artifacts
def test_golden_vectors_reproduce(golden, predictor):
    assert golden["model_id"] == predictor.model_id and len(golden["vectors"]) >= 20
    for v in golden["vectors"]:
        x = _x(v["x"])
        r = predictor.predict(x)
        np.testing.assert_allclose(r["raw"][0], v["raw"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(r["proba"][0], v["probabilities"], rtol=0, atol=1e-9)
        assert r["valence_raw"][0] == pytest.approx(v["valence_raw"], abs=1e-9)
        assert r["arousal"][0] == pytest.approx(v["arousal"], abs=1e-9)
        c = predictor.contrib(x)
        np.testing.assert_allclose(c["classifier"][0], np.array(v["pred_contrib"]["classifier"]), atol=1e-9)
        # TreeSHAP is additive: contributions + bias = raw score
        np.testing.assert_allclose(c["classifier"][0].sum(axis=1), v["raw"], atol=1e-6)
        assert c["valence"][0].sum() == pytest.approx(v["valence_raw"], abs=1e-6)


@needs_artifacts
def test_json_dumps_alone_reproduce_scores(golden):
    dumps = {k: load_dump(ART / f"{k}.json.gz") for k in MODELS}
    for v in golden["vectors"]:
        x = _x(v["x"])
        np.testing.assert_allclose(raw_scores(dumps["classifier"], x), v["raw"], rtol=0, atol=1e-9)
        assert raw_scores(dumps["valence"], x)[0] == pytest.approx(v["valence_raw"], abs=1e-9)
        assert raw_scores(dumps["arousal"], x)[0] == pytest.approx(v["arousal_raw"], abs=1e-9)
    assert any(any(e is None for e in v["x"]) for v in golden["vectors"])  # NaN paths are exercised


@needs_artifacts
def test_golden_windows_and_frames(golden):
    for w in golden["windows"]:
        got = compute_emotion({f: _x(w["person"][f]) for f in EMOTION_FIELDS})
        np.testing.assert_allclose(got, _x(w["expected"]), rtol=0, atol=1e-12)
    tracker = FaceCueTracker()
    blinks = 0
    for f in golden["frames"]:
        cues = tracker.update(None if f["blendshapes"] is None else raw_cues(f["blendshapes"], f["matrix"]), f["t_ms"])
        for k, v in f["expected"].items():
            assert (math.isnan(cues[k]) if v is None else cues[k] == pytest.approx(v, abs=1e-12)), k
        blinks += cues["blink_onsets"]
    assert blinks == 5  # the short blink, the threshold-equality, 600 ms, NaN-closure and backwards-clock cases


def test_json_evaluator_zeroes_tiny_inputs_like_lightgbm():
    """LightGBM reads |x| <= 1e-35f as 0 before any split; the dumps split at exactly +-1e-35f."""
    if not (ART / "meta.json").exists():
        pytest.skip("no trained emotion artifacts")
    pred = EmotionPredictor(ART)
    dump = load_dump(ART / "classifier.json.gz")
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 0.5, len(pred.cols))
    for _ in range(100):
        x = x0.copy()
        j = rng.choice(len(x), 12, replace=False)
        x[j] = rng.choice([0.0, -1.0000000180025095e-35, 1.0000000180025095e-35, 5e-36, -3e-40, np.nan, 1e-30], 12)
        np.testing.assert_allclose(raw_scores(dump, x), pred.raw(x)[0], rtol=0, atol=1e-12)


def test_face_dropout_keeps_targets_aligned():
    X = pd.DataFrame(np.random.default_rng(0).random((50, len(INPUT_FEATURES))), columns=INPUT_FEATURES)
    Xa, src = face_dropout(X, seed=1)
    assert len(Xa) == len(src) == 50 + 15 + 10
    np.testing.assert_array_equal(Xa.iloc[:50].to_numpy(), X.to_numpy())
    hidden = Xa.iloc[50:65]
    assert hidden[EMOTION_FEATURE_NAMES].isna().all().all() and (hidden["face_visible_now"] == 0).all()
    np.testing.assert_array_equal(hidden["db_room_mean_5m"].to_numpy(), X["db_room_mean_5m"].to_numpy()[src[50:65]])


@needs_artifacts
def test_self_report_eval_on_a_synthetic_fixture(tmp_path, predictor):
    from valence.eval.self_report import evaluate, load_taps, synthetic_taps

    lines = synthetic_taps(1, seed=3)
    assert lines and all("features" in d and d["synthetic"] for d in lines)
    as_list = dict(lines[0], features=[lines[0]["features"][n] for n in INPUT_FEATURES], session_id="list-form")
    bad = {"t": 1.0, "person_id": 1, "valence": 3.0, "arousal": 0.0, "features": {}}
    path = tmp_path / "taps.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in [*lines, as_list, bad]) + "\n")
    taps, X, problems = load_taps([path], predictor.cols)
    assert len(taps) == len(lines) + 1 and len(problems) == 1 and "outside" in problems[0]
    odd = [{"valence": True, "arousal": 0.0, "features": {}}, {"valence": 0.1, "arousal": 0.0, "features": {},
                                                              "feature_version": "emotion-v2"},
           {"valence": 0.3, "arousal": -0.2, "features": {"face_now": 0.1}}]
    path2 = tmp_path / "odd.jsonl"
    path2.write_text("".join(json.dumps(d) + "\n" for d in odd))
    taps2, X2, problems2 = load_taps([path2], predictor.cols)
    assert len(taps2) == 1 and np.isnan(X2[0]).sum() == len(predictor.cols) - 1
    assert any("numbers" in p for p in problems2) and any("emotion-v2" in p for p in problems2)
    assert any("absent" in p for p in problems2)
    # CCC is per axis: no arousal spread still gives a valence CCC
    from valence.eval.self_report import _block
    flat = _block(np.array([-0.5, 0.0, 0.5]), np.zeros(3), np.array([-0.4, 0.1, 0.4]), np.zeros(3))
    assert flat["valence_ccc"] is not None and flat["arousal_ccc"] is None
    np.testing.assert_array_equal(np.isnan(X[0]), np.isnan(X[-1]))
    report = evaluate(taps, X, predictor)
    o = report["overall"]["map_position"]
    assert report["n_taps"] == len(taps) and 0 <= o["valence_mae"] <= 2 and "list-form/1" in report["per_person"]
    # The fixture is the simulator itself, so the models must beat a constant guess on it.
    assert o["valence_mae"] < report["overall"]["constant_zero"]["valence_mae"]


@needs_artifacts
def test_tour_replays_through_the_python_pipeline(predictor):
    from valence.contracts import TICK_FIELDS
    from valence.features.compute import PERSON_FIELDS, ROOM_FIELDS, compute
    from valence.features.emotion import model_input

    doc = json.loads(gzip.decompress((ART / "tour.json.gz").read_bytes()))
    assert doc["model_id"] == predictor.model_id and len(doc["ticks"]) == 600
    assert set(doc["order"]) == set(EMOTION_STATES)
    rows = [t["records"] for t in doc["ticks"]]
    assert all(set(r[0]) == set(TICK_FIELDS) and r[0]["person_id"] == 0 and r[1]["person_id"] == 1 for r in rows)

    def arrays(recs, fields):
        return {f: np.array([np.nan if r[f] is None else r[f] for r in recs], dtype=float) for f in fields}

    for i in (0, 5, 59, 250, 599):
        lo = max(0, i - 179)
        person = arrays([rows[j][1] for j in range(lo, i + 1)], sorted(set(PERSON_FIELDS) | set(EMOTION_FIELDS)))
        room = arrays([rows[j][0] for j in range(lo, i + 1)], ROOM_FIELDS)
        xe = compute_emotion(person)
        np.testing.assert_allclose(xe, _x(doc["ticks"][i]["emotion_features"]), atol=1e-12)
        r = predictor.predict(model_input(compute({f: person[f] for f in PERSON_FIELDS}, room), xe))
        np.testing.assert_allclose(r["proba"][0], doc["ticks"][i]["probabilities"], atol=1e-9)
        assert r["valence_mix"][0] == pytest.approx(doc["ticks"][i]["valence"], abs=1e-9)


@needs_artifacts
def test_capture_replay_reads_cues_from_observations(tmp_path, predictor):
    """A features-only recording with emotion-v1 cues goes through the aggregator, features and models."""
    from valence.contracts import PersonObs, VideoObs, obs_to_dict
    from valence.eval.capture_stats import capture_ticks
    from valence.eval.emotion_capture import read_capture
    from valence.models.emotion import affect_for_window
    from valence.runtime.aggregator import Aggregator
    from valence.config import Settings

    rng = np.random.default_rng(0)
    lines = []
    for i in range(10 * 90):  # 90 s at 10 fps
        p = PersonObs(track_id=1, bbox=(0.2, 0.1, 0.8, 0.95), motion=0.02, face_visible=True, face_valence=0.02,
                      brow_down=0.25 + rng.normal(0, 0.02), eye_open=0.0, mouth_frown=0.02, smile=0.0,
                      head_yaw=rng.normal(0, 3), head_pitch=rng.normal(0, 3), blink_onsets=int(i % 50 == 0))
        lines.append(obs_to_dict(VideoObs(t=i / 10, wall=1000 + i / 10, frame_seq=i, frames_decoded=i + 1,
                                          persons=(p,))))
    path = tmp_path / "rec.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in lines))
    df = capture_ticks(path)
    person = df[df["person_id"] == 1]
    assert len(person) >= 17 and person["brow_down"].median() == pytest.approx(0.25, abs=0.02)
    assert person["blink_count"].sum() >= 16
    read = read_capture(df, predictor)
    assert read["ticks"] == len(person) and set(read["labels"]) <= set(EMOTION_STATES)
    agg = Aggregator(Settings(), "live")
    from valence.contracts import obs_from_dict

    for d in lines[:60]:
        agg.push(obs_from_dict(d))
    recs = agg.flush(1006.0, 6.0, 12.0)
    a = affect_for_window(predictor, [recs[1]], [recs[0]])
    assert a.label in EMOTION_STATES and -1 <= a.valence <= 1 and abs(sum(a.probabilities.values()) - 1) < 1e-9
