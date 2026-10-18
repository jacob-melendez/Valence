from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest

from valence.perception import audio_features as af
from valence.perception.audio_features import (
    COMPLIMENT_PHRASES,
    LAUGH_FAMILY,
    YAMNET_ALLOWLIST,
    LaughterTracker,
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
