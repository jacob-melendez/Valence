"""Reproducers for the validator review (findings F1-F15, F-a, F-b): unfaithful texts that used to
pass, and faithful phrasings that used to be rejected."""

from __future__ import annotations

import random

import pytest

from valence.narrator.fake_fact_sheets import make_fact_sheet
from valence.narrator.schema import allowed_drivers, key_drivers
from valence.narrator.validator import extract_numbers, split_sentences, validate


def sheet(seed: int, **kwargs) -> dict:
    return make_fact_sheet(random.Random(seed), **kwargs)


@pytest.fixture(scope="module")
def s1() -> dict:
    """Frustrated 69%, Person #1 of 2. Key: db_ewm_5m (high, room) "recent noise of 69 dB over the last
    5 minutes", fist_cnt_15m (high) "14 clenched-fist events in the last 15 minutes", hours_since_break
    (high) "4.4 hours since the last break"; opposing allowed: laugh_since_s "no laughter event in the
    last 15 minutes". Other sheet numbers: lux_std 13.158, face-only confidence 0.6, window fill 15."""
    s = sheet(1, label="Frustrated_Masked", face_condition="agree", states="core", n_people=2, person_id=1)
    assert [d["display"] for d in key_drivers(s)] == [
        "recent noise of 69 dB over the last 5 minutes", "14 clenched-fist events in the last 15 minutes",
        "4.4 hours since the last break"]
    assert s["confidence_display"] == "69%" and not s["mask_override"]
    return s


BASE = "Person #1 reads as Frustrated with 69% confidence."
GOOD = BASE + " Room noise of 69 dB, with 14 clenched-fist events and 4.4 hours since the last break."


def reasons(text: str, s: dict) -> list[str]:
    return validate(text, s).reasons


def test_baseline_passes(s1):
    assert validate(GOOD, s1).ok, reasons(GOOD, s1)


# --- F1: numbers bind to their driver / unit ---------------------------------------------------------


@pytest.mark.parametrize("evidence, bad", [
    ("Room noise of 13 dB, with 14 clenched-fist events and 4.4 hours since the last break.", "number:13"),
    ("Room noise of 69 dB, with 4 clenched-fist events and 4.4 hours since the last break.", "number:4"),
    ("Room noise of 69 dB, with 1 clenched-fist event and 4.4 hours since the last break.", "number:1"),
    ("Room noise of 69 dB, with 14 clenched-fist events and 15 hours since the last break.", "number:15"),
    ("Room noise of 69 dB, with 14 clenched-fist events and 69 hours since the last break.", "number:69"),
])
def test_f1_number_must_belong_to_its_driver(s1, evidence, bad):
    assert bad in reasons(f"{BASE} {evidence}", s1)


def test_f1_faithful_number_placements_still_pass(s1):
    for text in (
        GOOD,
        BASE + " Noise in the room averaged 69 dB recently, and they logged 14 clenched-fist events plus 4.4 hours "
               "since the last break.",
        "Person #1 reads as Frustrated (69%), with 14 clenched-fist events in the last 15 minutes, recent noise of "
        "69 dB over the last 5 minutes and 4.4 hours since the last break.",
        "At 69% confidence Person #1 reads as Frustrated: 14 clenched-fist events over the last 15 minutes and "
        "4.4 hours since the last break, in a room at 69 dB.",
    ):
        assert validate(text, s1).ok, (reasons(text, s1), text)


# --- F2: number words ------------------------------------------------------------------------------


def test_f2_number_words_are_parsed():
    values = [t.value for t in extract_numbers("one event, twenty-five, a hundred, thirty one, twice, half, "
                                               "a dozen, ninety")]
    assert values == [1, 25, 100, 31, 2, 0.5, 12, 90]
    assert extract_numbers("one of them, no one, someone, one another, one signal, a positive one") == []
    assert [t.value for t in extract_numbers("one hour, one-minute window, one clenched fist")] == [1, 1, 1]


def test_f2_one_minute_matches_the_last_minute():
    """Real-model run: three drafts were rejected only for "number:one"; "the last minute" is one minute."""
    s = sheet(1101, states="core")
    s = {**s, "layers": [dict(layer, drivers=[dict(d) for d in layer["drivers"]]) for layer in s["layers"]]}
    drivers = [d for layer in s["layers"] for d in layer["drivers"]]
    slump = next(d for d in drivers if d["feature"].startswith("slump"))
    slump["display"] = "slumped 56% of the last minute"
    text = ("Person #1 reads as Exhausted at 84%: slumped 56% of the last minute and head in hands 29% of the last "
            "5 minutes. One signal, noise 1 dB above its 15-minute median, pulled the other way.")
    assert validate(text, s).ok, reasons(text, s)
    assert validate(text.replace("of the last minute", "over one minute"), s).ok


@pytest.mark.parametrize("evidence", [
    "Room noise of 69 dB, with only one clenched-fist event and 4.4 hours since the last break.",
    "Room noise of 69 dB, with thirty clenched-fist events and 4.4 hours since the last break.",
    "Room noise of 69 dB, with a dozen clenched-fist events and a hundred hours since the last break.",
])
def test_f2_wrong_number_words_rejected(s1, evidence):
    assert any(r.startswith("number:") for r in reasons(f"{BASE} {evidence}", s1))


def test_f2_right_number_words_pass(s1):
    text = BASE + " Room noise of sixty-nine dB, with fourteen clenched-fist events and 4.4 hours since the last break."
    assert validate(text, s1).ok, reasons(text, s1)


# --- F3: direction words and signs --------------------------------------------------------------------


def test_f3_direction_word_flip_rejected():
    s = sheet(1101, states="core")
    opposing = next(d for d in allowed_drivers(s) if d["feature"] == "db_rel_15m")
    assert opposing["display"] == "noise 1 dB above its 15-minute median"
    good = ("Person #1 reads as Exhausted at 84%, based on slumped 56% of the last 5 minutes and head in hands 29% "
            "of the last 5 minutes, although noise 1 dB above its 15-minute median.")
    assert validate(good, s).ok, reasons(good, s)
    paraphrase = good.replace("noise 1 dB above its 15-minute median", "the noise sat 1 dB above the median")
    assert validate(paraphrase, s).ok, reasons(paraphrase, s)
    assert "direction:db_rel_15m:below" in reasons(good.replace("above", "below"), s)


def test_f3_sign_flip_rejected():
    s = sheet(6, label="Exhausted", face_condition="disagree", states="core")
    face = next(d["display"] for d in allowed_drivers(s) if "valence +" in d["display"])
    assert face == "smiling expression over the last 15 minutes (valence +0.31)"
    text = ("Person #1 reads as Exhausted (63%), from slumped 87% of the last 5 minutes and 4.8 hours since the last "
            "break, with a smiling expression over the last 15 minutes (valence -0.31) face reading.")
    assert "sign:-0.31" in reasons(text, s)
    assert "sign:-0.31" not in reasons(text.replace("-0.31", "+0.31"), s)


# --- F4: negation ---------------------------------------------------------------------------------


def test_f4_negated_state_rejected(s1):
    text = ("Person #1 is not Frustrated (69%). Room noise of 69 dB and 14 clenched-fist events along with 4.4 hours "
            "since the last break.")
    assert "state_negated" in reasons(text, s1)


def test_f4_negated_level_word_is_inverted(s1):
    text = BASE + " The noise was not high (69 dB), with 14 clenched-fist events and 4.4 hours since the last break."
    assert "qualifier:noise:not high" in reasons(text, s1)
    # "not low" for high noise is true, so it passes
    assert validate(text.replace("not high", "not low"), s1).ok


# --- F5: face-only state as the prediction (mask override) ---------------------------------------------


@pytest.fixture(scope="module")
def masked() -> dict:
    s = sheet(0, label="Frustrated_Masked", face_condition="disagree", states="all")
    assert s["mask_override"] and s["face_only"]["label"] == "Collaborative_Happy" and s["person"] == "Person #2"
    return s


@pytest.mark.parametrize("text", [
    "The model predicted Collaborative / Happy for Person #2, not Frustrated / Masked Stress (66%), given 3 "
    "clenched-fist events in the last 5 minutes and 2.6 hours since the last break.",
    "Person #2 reads as Frustrated / Masked Stress (66%) on paper, but they are really Collaborative / Happy, with 3 "
    "clenched-fist events in the last 5 minutes and 2.6 hours since the last break.",
])
def test_f5_face_state_as_prediction_rejected(masked, text):
    assert "face_state:Collaborative_Happy" in reasons(text, masked)


def test_f5_face_state_as_face_contrast_passes(masked):
    text = ("Person #2 reads as Frustrated / Masked Stress (66%), from 3 clenched-fist events in the last 5 minutes "
            "and 2.6 hours since the last break. The face alone suggested Collaborative / Happy, but context "
            "corrected the facial mask.")
    assert validate(text, masked).ok, reasons(text, masked)


# --- F6 / F7: invented signals, causal framing of opposing drivers ---------------------------------------


def test_f6_unmeasured_signals_rejected(s1):
    text = (BASE + " Room noise of 69 dB, 14 clenched-fist events, an elevated heart rate and rapid typing, plus "
            "4.4 hours since the last break.")
    got = reasons(text, s1)
    assert "unmeasured:heart rate" in got and "unmeasured:typing" in got


@pytest.mark.parametrize("cue", ["rests mainly on", "is based on", "is due to", "is explained by", "reflects"])
def test_f7_opposing_driver_framed_as_cause_rejected(s1, cue):
    text = (f"{BASE} The call {cue} no laughter event in the last 15 minutes and 14 clenched-fist events, plus "
            "4.4 hours since the last break.")
    assert "effect:laugh_since_s:" in " ".join(reasons(text, s1))


# --- F8: concept-specific qualifiers ----------------------------------------------------------------


def test_f8_noise_word_binds_to_noise_itself():
    """Skipped as a code change: "loud"/"quiet"/"bright"/"dim" are concept words themselves, so they
    always bind to their own concept's drivers (distance 0), never to a neighbouring light mention."""
    s = sheet(2, states="core")
    assert {d["feature"]: d["level"] for d in allowed_drivers(s)}["db_mean_15m"] == "normal"
    text = ("Person #1 reads as Focused at 51%. The noise and light were loud and bright, with average noise of "
            "50 dB over the last 15 minutes and moving normally (motion 0.08).")
    assert "qualifier:noise:loud" in reasons(text, s)


# --- F9 / F10 / F11: room scope, mask paraphrase, banned terms -------------------------------------


@pytest.mark.parametrize("lead", [
    "Person #1 was making noise of 69 dB,", "Noise from Person #1 reached 69 dB,", "Person #1 generated noise of 69 dB,",
])
def test_f9_room_noise_attributed_to_person_rejected(s1, lead):
    text = f"{BASE} {lead} with 14 clenched-fist events and 4.4 hours since the last break."
    assert any(r.startswith("room_scope") for r in reasons(text, s1))


@pytest.mark.parametrize("lead", [
    "Behind a brave face and pretending to be fine,", "Beneath a cheerful exterior,",
])
def test_f10_mask_paraphrase_rejected_without_override(s1, lead):
    text = f"{BASE} {lead} they logged 14 clenched-fist events, 4.4 hours since the last break and room noise of 69 dB."
    assert any(r.startswith("mask_term:") for r in reasons(text, s1))


@pytest.mark.parametrize("phrase, term", [
    ("The kid, a hyperactive student, logged", "kid"),
    ("This blonde senior logged", "blonde"),
    ("The bloke logged", "bloke"),
    ("The aging redhead logged", "aging"),
    ("The lad logged", "lad"),
    ("Person #1, perhaps on medication, logged", "medication"),
    ("Person #1, maybe with a migraine, logged", "migraine"),
])
def test_f11_banned_terms(s1, phrase, term):
    text = f"{BASE} {phrase} 14 clenched-fist events, 4.4 hours since the last break and room noise of 69 dB."
    assert any(r.startswith("banned:") and r.endswith(":" + term) for r in reasons(text, s1)), reasons(text, s1)


# --- F-a / F-b: other people, sentence count -------------------------------------------------------


@pytest.mark.parametrize("who", ["Participant #2", "Person No. 2", "Person number 2"])
def test_fa_other_person_references_rejected(s1, who):
    text = f"{BASE} {who} nearby logged 14 clenched-fist events, 4.4 hours since the last break and room noise of 69 dB."
    assert any(r.startswith("person:") for r in reasons(text, s1))


def test_fb_lowercase_sentence_starts_are_counted(s1):
    text = (BASE + " room noise was 69 dB. fists clenched 14 times. 4.4 hours since the last break. motion was "
            "restless. that is all.")
    assert "sentences:6" in reasons(text, s1)
    assert len(split_sentences("Noise was 69 dB, e.g. the room, approx. 5 minutes at 3 p.m. today")) == 1


# --- F12-F15: faithful phrasings that used to be rejected -------------------------------------------


@pytest.mark.parametrize("text", [
    BASE + " The recent noise of 69 dB in the room, 14 clenched-fist events and 4.4 hours since the last break all fit.",
    BASE + " Steady room noise of 69 dB, 14 clenched-fist events and 4.4 hours since the last break all fit.",
    BASE + " The evidence overwhelmingly fits: room noise of 69 dB, 14 clenched-fist events and 4.4 hours since the "
           "last break.",
    BASE + " The model's main focus was room noise of 69 dB, 14 clenched-fist events and 4.4 hours since the last "
           "break.",
    BASE + " The long session was wearing on them: room noise of 69 dB, 14 clenched-fist events and 4.4 hours since "
           "the last break.",
    BASE + " Readings are 15 minutes old: room noise of 69 dB, 14 clenched-fist events and 4.4 hours since the last "
           "break.",
])
def test_f12_f15_faithful_phrasings_pass(s1, text):
    assert validate(text, s1).ok, reasons(text, s1)


def test_f12_recent_still_checked_for_time_since_features(s1):
    text = BASE + " Recent laughter, 14 clenched-fist events and 4.4 hours since the last break with room noise of 69 dB."
    assert "qualifier:laughter:recent" in reasons(text, s1)


def test_f13_late_afternoon_and_clock_words():
    s = sheet(34, states="core")
    tod = next(d for d in key_drivers(s) if d["feature"].startswith("tod"))
    assert tod["display"] == "time of day (17:05)" and tod["level"] == "normal"
    good = "Person #1 reads as Exhausted at 61%: it is late afternoon (17:05) and the sheet shows slumped 70% of the last minute."
    assert validate(good, s).ok, reasons(good, s)
    assert "time_of_day:morning" in reasons(good.replace("afternoon", "morning"), s)
