from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from valence.perception import audio_features as af
from valence.perception.audio_features import (
    COMPLIMENT_PHRASES,
    LAUGH_FAMILY,
    PHRASE_RISE_DB,
    YAMNET_ALLOWLIST,
    LaughterTracker,
    PhraseLoudness,
    PowerSmoother,
    count_compliments,
    mean_power,
    power_to_db,
    resample_48k_to_16k,
    vosk_grammar,
)

SECRET = "zanzibar"


def sine(freq: float = 1000.0, sr: int = 16000, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def vosk(*words: tuple[str, float], text: str | None = None) -> str:
    result = [{"word": w, "conf": c, "start": 0.3 * i, "end": 0.3 * i + 0.25} for i, (w, c) in enumerate(words)]
    return json.dumps({"result": result, "text": text if text is not None else " ".join(w for w, _ in words)})


def test_full_scale_sine_is_minus_3_dbfs() -> None:
    p = mean_power(sine())
    assert p == pytest.approx(0.5, rel=1e-4)
    assert power_to_db(p) == pytest.approx(-3.01, abs=0.01)
    assert power_to_db(p, 94.0) == pytest.approx(90.99, abs=0.01)


def test_int16_pcm_scaled() -> None:
    pcm = np.rint(sine() * 32767).astype(np.int16)
    assert power_to_db(mean_power(pcm)) == pytest.approx(-3.01, abs=0.02)


def test_mean_power_float64_and_edges() -> None:
    x = np.full(1_000_000, 1e-3, dtype=np.float32)
    assert mean_power(x) == pytest.approx(1e-6, rel=1e-6)
    assert math.isnan(mean_power(np.array([], dtype=np.float32)))


def test_power_to_db_floor_and_nan() -> None:
    assert power_to_db(0.0) == pytest.approx(-120.0)
    assert power_to_db(0.0, 100.0) == pytest.approx(-20.0)
    assert math.isnan(power_to_db(math.nan))


def test_smoother_averages_power_not_db() -> None:
    sm = PowerSmoother(alpha=0.3)
    sm.update(1e-2)
    sm.update(1e-6)
    expected_power = 0.3 * 1e-6 + 0.7 * 1e-2
    assert sm.value == pytest.approx(expected_power)
    assert sm.db() == pytest.approx(10 * math.log10(expected_power))
    db_average = 0.3 * -60.0 + 0.7 * -20.0
    assert abs(sm.db() - db_average) > 10.0


def test_smoother_first_sample_nan_skip_reset() -> None:
    sm = PowerSmoother()
    assert math.isnan(sm.db())
    assert sm.update(0.25) == 0.25
    assert sm.update(math.nan) == 0.25
    assert sm.update(-1.0) == 0.25
    sm.reset()
    assert math.isnan(sm.value)
    with pytest.raises(ValueError):
        PowerSmoother(alpha=0.0)


def test_allowlist_constants() -> None:
    assert YAMNET_ALLOWLIST[0] == "Speech"
    assert list(LAUGH_FAMILY) == ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]


def _legacy():
    """Original 0.30 x 2 rule: these tests cover LaughterTracker bookkeeping, not thresholds."""
    from valence.perception.debounce import HysteresisDebouncer

    return HysteresisDebouncer(on=0.30, off=0.15, on_count=2, off_hold_s=0.0, refractory_s=3.0)


def test_default_tracker_uses_the_measured_laughter_rule() -> None:
    tr = LaughterTracker()
    assert tr.update({"Speech": 0.98, "Laughter": 0.0}, 0.0)[2] is False
    assert tr.update({"Speech": 0.15, "Laughter": 0.08, "Snicker": 0.06}, 1.0)[2] is True
    assert tr.take_onsets() == 1


def test_default_tracker_counts_a_laugh_with_a_silent_second_once() -> None:
    tr = LaughterTracker()
    seq = [{"Speech": 0.41, "Laughter": 0.04}, {"Speech": 0.11, "Laughter": 0.08},
           {"Speech": 0.94, "Laughter": 0.0}, {"Speech": 0.15, "Laughter": 0.08, "Snicker": 0.06}]
    onsets = []
    for t, s in enumerate(seq):
        onsets.append(tr.update(s, float(t))[2])
        if t >= 1:
            assert tr.active, t  # the instant-exit rule dropped out at the silent second (t=2)
    assert onsets == [False, True, False, False]
    assert tr.take_onsets() == 1


def test_default_tracker_counts_a_laugh_with_two_silent_seconds_once() -> None:
    # Capture run 3's shape: the instant-exit, 4 s-refractory rule gave 2 onsets 4 s apart.
    tr = LaughterTracker()
    seq = [{"Laughter": 0.08}, {"Laughter": 0.0}, {"Laughter": 0.0}, {"Laughter": 0.04}, {"Laughter": 0.08}]
    onsets = [tr.update(s, float(t))[2] for t, s in enumerate(seq)]
    assert onsets == [True, False, False, False, False]
    assert tr.take_onsets() == 1
    from valence.perception.debounce import HysteresisDebouncer

    legacy = LaughterTracker(debouncer=HysteresisDebouncer(on=0.05, off=0.02, on_count=1, refractory_s=4.0))
    assert sum(legacy.update(s, float(t))[2] for t, s in enumerate(seq)) == 2


def test_laughter_onset_semantics() -> None:
    tr = LaughterTracker(debouncer=_legacy())
    assert tr.update({"Laughter": 0.35, "Speech": 0.2}, 0.0) == (0.35, 0.2, False)
    laugh, speech, onset = tr.update({"Laughter": 0.1, "Giggle": 0.4}, 0.5)
    assert (laugh, onset) == (0.4, True)
    assert tr.active
    assert tr.take_onsets() == 1
    assert tr.take_onsets() == 0
    tr.update({"Laughter": 0.05}, 1.0)
    assert not tr.active
    tr.update({"Snicker": 0.5}, 1.5)
    assert tr.update({"Snicker": 0.5}, 2.0)[2] is False
    assert tr.take_onsets() == 0
    tr.update({"Laughter": 0.0}, 2.5)
    tr.update({"Belly laugh": 0.9}, 3.5)
    assert tr.update({"Chuckle, chortle": 0.9}, 4.0)[2] is True
    assert tr.take_onsets() == 1


def test_speech_alone_never_counts_as_laughter() -> None:
    tr = LaughterTracker(debouncer=_legacy())
    for k in range(10):
        laugh, speech, onset = tr.update({"Speech": 0.95}, 0.5 * k)
        assert laugh == 0.0 and speech == 0.95 and not onset
    assert tr.take_onsets() == 0


def test_onsets_accumulate_between_takes() -> None:
    tr = LaughterTracker(debouncer=_legacy())
    t = 0.0
    for _ in range(3):
        tr.update({"Laughter": 0.6}, t)
        tr.update({"Laughter": 0.6}, t + 0.5)
        tr.update({"Laughter": 0.0}, t + 1.0)
        t += 4.0
    assert tr.take_onsets() == 3


def test_vosk_grammar() -> None:
    grammar = json.loads(vosk_grammar())
    assert grammar == [*COMPLIMENT_PHRASES, "[unk]"]


def test_count_consecutive_high_confidence() -> None:
    assert count_compliments(vosk(("great", 0.95), ("job", 0.9))) == Counter({"great job": 1})
    got = count_compliments(vosk(("awesome", 1.0), ("[unk]", 1.0), ("thank", 0.8), ("you", 0.75),
                                 ("thank", 0.9), ("you", 0.9), ("amazing", 0.7)))
    assert got == Counter({"thank you": 2, "awesome": 1, "amazing": 1})


def test_low_confidence_rejected() -> None:
    assert count_compliments(vosk(("great", 0.95), ("job", 0.69))) == Counter()
    assert count_compliments(vosk(("awesome", 0.5))) == Counter()
    assert count_compliments(vosk(("great", 0.95), ("job", 0.69)), min_conf=0.6) == Counter({"great job": 1})


def test_words_must_be_consecutive() -> None:
    assert count_compliments(vosk(("great", 0.9), ("[unk]", 1.0), ("job", 0.9))) == Counter()
    assert count_compliments(vosk(("well", 0.9), ("nice", 0.9), ("done", 0.9))) == Counter()
    assert count_compliments(vosk(("job", 0.9), ("great", 0.9))) == Counter()


def test_unk_never_counts() -> None:
    assert count_compliments(vosk(("[unk]", 1.0), ("[unk]", 1.0))) == Counter()


def test_malformed_entry_breaks_phrase() -> None:
    doc = json.dumps({"result": [{"word": "nice", "conf": 0.9}, {"conf": 0.9}, {"word": "work", "conf": 0.9}]})
    assert count_compliments(doc) == Counter()
    doc = json.dumps({"result": [{"word": "nice", "conf": "high"}, {"word": "work", "conf": 0.9}]})
    assert count_compliments(doc) == Counter()


@pytest.mark.parametrize("doc", [
    "", "   ", "{", "null", "[]", "42", '"great job"',
    '{"partial": "great job"}',
    '{"text": "great job"}',
    '{"result": 5, "text": "great job"}',
    '{"result": ["great", "job"]}',
    b"\xff\xfe",
    None,
    12,
])
def test_partial_empty_malformed_give_empty_counter(doc: object) -> None:
    assert count_compliments(doc) == Counter()  # type: ignore[arg-type]


def test_bytes_input_accepted() -> None:
    assert count_compliments(vosk(("good", 0.9), ("idea", 0.9)).encode()) == Counter({"good idea": 1})


@pytest.mark.parametrize("doc", [
    SECRET,
    "{" + SECRET,
    json.dumps({"result": [{"word": SECRET, "conf": SECRET}]}),
    json.dumps({"result": [{"word": SECRET, "conf": 0.99}, {"word": "awesome", "conf": 0.99}]}),
    json.dumps({"result": [{"word": SECRET, "conf": [SECRET]}], "text": SECRET}),
    json.dumps({"partial": SECRET}),
    "[" * 100_000 + SECRET,
])
def test_no_text_escapes(doc: str, capsys: pytest.CaptureFixture[str]) -> None:
    try:
        result = count_compliments(doc)
    except BaseException as exc:  # pragma: no cover - the assertion is that this never happens
        pytest.fail(f"count_compliments raised {type(exc).__name__}; secret leaked: {SECRET in repr(exc.args)}")
    assert set(result) <= set(COMPLIMENT_PHRASES)
    assert SECRET not in repr(result)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_internal_failure_swallowed_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError(f"recognized {SECRET}")

    monkeypatch.setattr(af, "_count", boom)
    assert count_compliments(vosk(("awesome", 0.9))) == Counter()


def timed(*words: tuple[str, float, float, float]) -> str:
    """Vosk result with explicit word times: (word, conf, start s, end s)."""
    return json.dumps({"result": [{"word": w, "conf": c, "start": a, "end": b} for w, c, a, b in words]})


def test_near_check_gets_phrase_span_and_can_reject() -> None:
    spans: list[tuple[float, float]] = []

    def near(a: float, b: float) -> bool:
        spans.append((a, b))
        return a > 5.0

    doc = timed(("great", 0.9, 1.0, 1.3), ("job", 0.9, 1.35, 1.7), ("[unk]", 1.0, 2.0, 2.5),
                ("thank", 0.9, 6.0, 6.2), ("you", 0.9, 6.2, 6.4))
    assert count_compliments(doc, near=near) == Counter({"thank you": 1})
    assert spans == [(1.0, 1.7), (6.0, 6.4)]
    assert count_compliments(doc) == Counter({"great job": 1, "thank you": 1})  # no check: unchanged behaviour


def test_near_check_rejects_phrase_without_word_times() -> None:
    doc = vosk(("awesome", 0.9))
    no_times = json.dumps({"result": [{"word": "awesome", "conf": 0.9}]})
    assert count_compliments(doc, near=lambda a, b: True) == Counter({"awesome": 1})
    assert count_compliments(no_times, near=lambda a, b: True) == Counter()
    assert count_compliments(no_times) == Counter({"awesome": 1})


def noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    """White noise whose mean power is ``db`` dBFS."""
    x = np.random.default_rng(seed).standard_normal(int(16000 * seconds))
    return (x * 10 ** (db / 20)).astype(np.float32)


def feed(pl: PhraseLoudness, x: np.ndarray, block: int = 16000) -> None:
    for i in range(0, len(x), block):
        pl.add(x[i:i + block])


def test_loudness_rise_over_floor() -> None:
    pl = PhraseLoudness()
    feed(pl, np.concatenate([noise(30, -40), noise(1, -30, 1), noise(2, -40, 2)]))
    assert pl.rise_db(30.0, 31.0) == pytest.approx(10.0, abs=0.3)
    assert pl.is_near(30.2, 30.8)
    assert pl.rise_db(31.5, 32.5) == pytest.approx(0.0, abs=0.5)
    assert not pl.is_near(31.5, 32.5)


def test_crowd_level_phrase_is_rejected_and_threshold_is_4_db() -> None:
    assert PHRASE_RISE_DB == 4.0
    rng = np.random.default_rng(3)
    # babble-like floor: 100 ms slots wandering +-3 dB around -40 dBFS
    slots = [noise(0.1, -40 + rng.uniform(-3, 3), i) for i in range(400)]
    pl = PhraseLoudness()
    feed(pl, np.concatenate(slots + [noise(0.6, -38, 99), noise(0.6, -33, 98)]))
    assert not pl.is_near(40.0, 40.6)  # a phrase at the crowd's level
    assert pl.is_near(40.6, 41.2)      # +7 dB over the crowd


def test_floor_is_the_gaps_while_the_person_talks_most_of_the_time() -> None:
    # 30 s of the person's own talk: 0.7 s at -30 dBFS, 0.3 s gaps at the -50 dBFS room; then a phrase at -30
    talk = [np.concatenate([noise(0.7, -30, 2 * i), noise(0.3, -50, 2 * i + 1)]) for i in range(30)]
    pl = PhraseLoudness()
    feed(pl, np.concatenate(talk + [noise(0.7, -30, 999)]))
    assert pl.rise_db(30.0, 30.7) == pytest.approx(20.0, abs=1.0)
    assert pl.is_near(30.0, 30.7)


def test_loudness_block_size_does_not_matter() -> None:
    x = np.concatenate([noise(5, -45), noise(1, -35, 1)])
    a, b = PhraseLoudness(), PhraseLoudness()
    feed(a, x, 16000)
    feed(b, x, 1234)  # slots straddle blocks
    assert b.rise_db(5.0, 6.0) == pytest.approx(a.rise_db(5.0, 6.0), abs=1e-9)
    assert a.rise_db(5.0, 6.0) == pytest.approx(10.0, abs=0.5)


def test_loudness_edges() -> None:
    pl = PhraseLoudness(keep_s=10.0)
    feed(pl, noise(0.5, -40))
    assert pl.rise_db(0.1, 0.4) == math.inf and pl.is_near(0.1, 0.4)  # < 1 s of floor yet: let it through
    feed(pl, noise(20, -40, 1))
    assert math.isnan(pl.rise_db(2.0, 2.5)) and not pl.is_near(2.0, 2.5)  # older than what is kept
    assert math.isnan(pl.rise_db(25.0, 26.0))  # not heard yet
    assert math.isnan(pl.rise_db(math.nan, 19.0)) and math.isnan(pl.rise_db(19.0, 18.0))
    assert np.isfinite(pl.rise_db(19.0, 19.0))  # a zero-length span still covers its slot


def test_loudness_keeps_scalars_only() -> None:
    pl = PhraseLoudness()
    x = noise(2, -40)
    pl.add(x)
    x[:] = 0.0
    assert pl.rise_db(1.5, 2.0) == pytest.approx(0.0, abs=1.5)  # derived from powers taken before the purge
    assert len(pl._p) == 20 and all(isinstance(v, float) for v in pl._p)


_YAMNET = Path(__file__).resolve().parents[2] / "models" / "yamnet.tflite"


@pytest.mark.skipif(not _YAMNET.exists(), reason="models not downloaded")
@pytest.mark.parametrize("phrase_db, expected", [(-40.0, 0), (-28.0, 1)])
def test_perceiver_applies_the_loudness_check_on_the_recognizer_clock(phrase_db: float, expected: int) -> None:
    """Wiring: AudioPerceiver feeds PhraseLoudness the blocks Vosk gets and passes is_near to count_compliments.
    A final at block 6 reports "great job" over 5.2-5.8 s (recognizer clock); the room floor is -40 dBFS."""
    from valence.perception.audio import AudioPerceiver

    class FakeRecognizer:
        def __init__(self) -> None:
            self.n = 0

        def AcceptWaveform(self, view) -> bool:
            self.n += 1
            return self.n == 7  # final arrives one block after the phrase, as Vosk's endpoint does

        def Result(self) -> str:
            return timed(("great", 0.95, 5.2, 5.5), ("job", 0.95, 5.5, 5.8))

    ap = AudioPerceiver(spl_offset_db=94.0, compliments=False)
    ap.recognizer = FakeRecognizer()
    try:
        hits = []
        for i in range(8):
            blk = noise(1, -40, i)
            if i == 5:
                blk[3200:12800] = noise(0.6, phrase_db, 50)
            hits.append(ap.process(blk, float(i + 1), 0.0, i).compliment_hits)
    finally:
        ap.close()
    assert hits == [0] * 6 + [expected, 0]


def test_resample_length_dtype_and_content() -> None:
    x = sine(1000.0, sr=48000)
    y = resample_48k_to_16k(x)
    assert y.dtype == np.float32 and len(y) == 16000
    assert power_to_db(mean_power(y[100:-100])) == pytest.approx(-3.01, abs=0.1)
    assert len(resample_48k_to_16k(np.zeros(4801, dtype=np.float32))) == 1601
    assert resample_48k_to_16k(np.zeros((4800, 1), dtype=np.float32)).shape == (1600, 1)


def test_resample_removes_content_above_nyquist() -> None:
    y = resample_48k_to_16k(sine(12000.0, sr=48000))
    assert power_to_db(mean_power(y[100:-100])) < -40.0
