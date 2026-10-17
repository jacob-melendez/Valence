import math

import pytest

from valence import contracts as c


def test_states_are_consistent():
    for s in c.ALL_STATES:
        assert s in c.STATE_DISPLAY and s in c.STATE_LEXICON and s in c.STATE_TIER
    assert c.STATE_SETS["core"] == c.CORE_STATES and len(c.ALL_STATES) == 6


def test_display_label_rules():
    assert c.display_label("Frustrated_Masked") == "Frustrated"
    assert c.display_label("Frustrated_Masked", mask_override=True) == "Frustrated / Masked Stress"
    assert c.display_label("Frustrated_Masked", states=c.ALL_STATES) == "Frustrated / Masked Stress"
    assert c.display_label("Collaborative_Happy", n_people=1) == "Upbeat"
    assert c.display_label("Collaborative_Happy", n_people=3) == "Collaborative / Happy"


def test_obs_roundtrip_with_nan():
    p = c.PersonObs(track_id=1, bbox=(0.1, 0.2, 0.4, 0.9), fist_onsets=1, face_visible=False)
    v = c.VideoObs(t=1.0, wall=2.0, frame_seq=3, frames_decoded=3, frame_w=1280, frame_h=720, persons=(p,))
    a = c.AudioObs(t=1.5, wall=2.5, block_seq=1, db_spl=61.2, rms_power=1e-3, laughter_onsets=1)
    for obs in (v, a):
        d = c.obs_to_dict(obs)
        back = c.obs_from_dict(d)
        assert type(back) is type(obs)
    back_v = c.obs_from_dict(c.obs_to_dict(v))
    assert math.isnan(back_v.persons[0].face_valence) and back_v.persons[0].bbox == p.bbox


def test_person_obs_validation():
    with pytest.raises(ValueError):
        c.PersonObs(track_id=1, bbox=(0.1, 0.2, 1.4, 0.9))
    with pytest.raises(ValueError):
        c.PersonObs(track_id=1, bbox=(0.1, 0.2, 0.4, 0.9), face_valence=2.0)


def test_json_schemas_export():
    schemas = c.json_schemas()
    assert {"Prediction", "Governance", "Tracks", "FactSheet", "NarrationResult"} <= set(schemas)


def test_prediction_message_matches_frontend_example():
    msg = c.Prediction(
        event_id="p1-000348", t=1740.0, person_id=1, label="Frustrated_Masked",
        label_display="Frustrated / Masked Stress", confidence=0.88,
        probabilities={"Focused": 0.04, "Frustrated_Masked": 0.88, "Exhausted": 0.05, "Collaborative_Happy": 0.03},
        face_only={"label": "Collaborative_Happy", "confidence": 0.71}, mask_override=True, warming_up=False,
        layers=[{"name": "Ambient Physics", "contribution": 0.22, "drivers": [{
            "feature": "db_mean_15m", "display": "average noise of 72 dB over 15 minutes", "value": 72.4,
            "unit": "dB", "shap": 0.22, "level": "high", "effect": "supports", "scope": "room"}]}],
        data_quality={"face_visible_frac": 0.9, "lux": 210.0},
        explanation={"text": "template text"},
    )
    d = msg.model_dump()
    assert d["type"] == "prediction" and d["explanation"]["source"] == "template"
