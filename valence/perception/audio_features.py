"""Audio scalars for the Ambient Physics and Somatic & Social layers.

Callers reduce each raw block to a mean power, YAMNet allowlist scores and compliment
counts, then purge it. Loudness is smoothed in the power domain and converted to dB last:
averaging dB values would understate bursts of sound. Recognised words exist only inside
``count_compliments`` and never leave it.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from valence.perception.debounce import HysteresisDebouncer, laughter_debouncer

POWER_FLOOR = 1e-12
YAMNET_ALLOWLIST: tuple[str, ...] = (
    "Speech", "Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
)
LAUGH_FAMILY: tuple[str, ...] = YAMNET_ALLOWLIST[1:]
COMPLIMENT_PHRASES: tuple[str, ...] = (
    "great job", "nice work", "well done", "thank you", "love it", "good idea", "awesome", "amazing",
)
UNK = "[unk]"
# Near-field check for compliments (docs/BENCHMARKS.md, "Audio detectors in crowd noise"): a phrase counts only if
# the mean power over its own words is >= PHRASE_RISE_DB above the 20th percentile of the 100 ms powers in the
# 30 s before it. Restricted-grammar Vosk turns hall babble into phrases at 1-6 per minute, all at the crowd's level.
PHRASE_RISE_DB = 4.0
FLOOR_WINDOW_S = 30.0
FLOOR_PERCENTILE = 20.0
SUB_BLOCK_S = 0.1
_PHRASE_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    sorted(((p, tuple(p.split())) for p in COMPLIMENT_PHRASES), key=lambda item: -len(item[1]))
)


def mean_power(x: np.ndarray) -> float:
    """float64 mean of squares; integer PCM is scaled to [-1, 1). NaN for an empty block."""
    a = np.asarray(x)
    if a.size == 0:
        return math.nan
    p = float(np.mean(np.square(a, dtype=np.float64)))
    if np.issubdtype(a.dtype, np.integer):
        p /= (float(np.iinfo(a.dtype).max) + 1.0) ** 2
    return p


def power_to_db(p: float, offset: float = 0.0) -> float:
    """10 log10(max(p, 1e-12)) + offset (offset = spl_offset_db gives dB SPL)."""
    if math.isnan(p):
        return math.nan
    return 10.0 * math.log10(max(p, POWER_FLOOR)) + offset


@dataclass(slots=True)
class PowerSmoother:
    """Exponential moving average of mean power (never of dB). Non-finite input is skipped."""

    alpha: float = 0.3
    value: float = math.nan

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def update(self, p: float) -> float:
        if math.isfinite(p) and p >= 0.0:
            self.value = p if math.isnan(self.value) else self.alpha * p + (1.0 - self.alpha) * self.value
        return self.value

    def db(self, offset: float = 0.0) -> float:
        return power_to_db(self.value, offset)

    def reset(self) -> None:
        self.value = math.nan


def _family_max(scores: Mapping[str, float]) -> float:
    values = [float(scores.get(k, 0.0)) for k in LAUGH_FAMILY]
    finite = [v for v in values if not math.isnan(v)]
    return max(finite) if finite else math.nan


class LaughterTracker:
    """YAMNet allowlist scores -> (laugh, speech, onset) with debounced laughter onsets.

    laugh = max over LAUGH_FAMILY; a missing class counts as 0. Onsets accumulate until
    ``take_onsets`` hands out the delta since its previous call.
    """

    __slots__ = ("_debouncer", "_pending")

    def __init__(self, debouncer: HysteresisDebouncer | None = None) -> None:
        self._debouncer = debouncer if debouncer is not None else laughter_debouncer()
        self._pending = 0

    @property
    def active(self) -> bool:
        return self._debouncer.active

    def update(self, scores: Mapping[str, float], t: float) -> tuple[float, float, bool]:
        laugh = _family_max(scores)
        speech = float(scores.get("Speech", 0.0))
        onset = self._debouncer.update(laugh, t)
        if onset:
            self._pending += 1
        return laugh, speech, onset

    def take_onsets(self) -> int:
        n, self._pending = self._pending, 0
        return n


def vosk_grammar() -> str:
    """Grammar for ``KaldiRecognizer(model, 16000, vosk_grammar())``: phrases + [unk]."""
    return json.dumps([*COMPLIMENT_PHRASES, UNK])


class PhraseLoudness:
    """100 ms mean powers of the audio the recognizer has heard, on the recognizer's own clock.

    ``add`` is called with every block before it goes to Vosk; Vosk's word ``start``/``end`` are seconds
    since the recognizer started, so a phrase maps onto the 100 ms slots it spans. Only power scalars
    (10 per second, ``keep_s`` of them) are kept: no samples. ``rise_db`` compares the phrase's mean power
    with the ``FLOOR_PERCENTILE`` of the ``FLOOR_WINDOW_S`` before it; a low percentile keeps the floor at the
    room's level even while the person is talking most of the time (the gaps between words are the room).
    """

    __slots__ = ("_n", "_p", "_count", "_acc", "_acc_n")

    def __init__(self, sr: int = 16000, keep_s: float = 90.0) -> None:
        self._n = int(round(sr * SUB_BLOCK_S))
        self._p: deque[float] = deque(maxlen=int(keep_s / SUB_BLOCK_S))
        self._count = 0  # slots completed since the recognizer started
        self._acc, self._acc_n = 0.0, 0  # partial slot carried across blocks (sum of squares, samples)

    def add(self, block: np.ndarray) -> None:
        x = np.asarray(block, dtype=np.float64).ravel()
        i = 0
        while i < len(x):
            take = min(self._n - self._acc_n, len(x) - i)
            self._acc += float(np.sum(np.square(x[i:i + take])))
            self._acc_n += take
            i += take
            if self._acc_n == self._n:
                self._p.append(self._acc / self._n)
                self._count += 1
                self._acc, self._acc_n = 0.0, 0

    def rise_db(self, start: float, end: float) -> float:
        """dB of the span's mean power above the floor before it; +inf with < 1 s of floor, NaN if unknown."""
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            return math.nan
        first = self._count - len(self._p)
        a = int(math.floor(start / SUB_BLOCK_S + 1e-6))
        b = min(self._count, max(a + 1, int(math.ceil(end / SUB_BLOCK_S - 1e-6))))
        if a < first or a >= b:
            return math.nan
        p = list(self._p)
        span = p[a - first:b - first]
        hist = p[max(0, a - first - int(FLOOR_WINDOW_S / SUB_BLOCK_S)):a - first]
        if len(hist) < int(1.0 / SUB_BLOCK_S):
            return math.inf  # nothing to compare with yet (stream start): let it through
        floor = float(np.percentile(hist, FLOOR_PERCENTILE))
        return power_to_db(float(np.mean(span))) - power_to_db(floor)

    def is_near(self, start: float, end: float) -> bool:
        return self.rise_db(start, end) >= PHRASE_RISE_DB


Token = tuple[str, float, float, float]  # word, conf, start s, end s


def _num(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _tokens(result: object) -> list[Token]:
    if not isinstance(result, dict) or not isinstance(result.get("result"), list):
        return []
    tokens: list[Token] = []
    for item in result["result"]:
        get = item.get if isinstance(item, dict) else (lambda _k: None)
        word, conf = get("word"), _num(get("conf"))
        start, end = _num(get("start")), _num(get("end"))
        if isinstance(word, str) and conf is not None:
            tokens.append((word.lower(), conf, math.nan if start is None else start, math.nan if end is None else end))
        else:
            tokens.append((UNK, 0.0, math.nan, math.nan))
    return tokens


def _count(tokens: list[Token], min_conf: float,
           near: Callable[[float, float], bool] | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    i = 0
    while i < len(tokens):
        for phrase, words in _PHRASE_TOKENS:
            window = tokens[i:i + len(words)]
            if len(window) == len(words) and all(t[0] == p and t[1] >= min_conf for t, p in zip(window, words)):
                start, end = window[0][2], window[-1][3]
                if near is None or (math.isfinite(start) and math.isfinite(end) and near(start, end)):
                    counts[phrase] += 1
                i += len(words)
                break
        else:
            i += 1
    return counts


def count_compliments(result_json: str | bytes, min_conf: float = 0.7,
                      near: Callable[[float, float], bool] | None = None) -> Counter[str]:
    """Count compliment phrases in a final Vosk result (recognizer with ``SetWords(True)``).

    A phrase counts when its words appear consecutively, each with conf >= ``min_conf``, and, when
    ``near`` is given, ``near(start of its first word, end of its last word)`` is true (the live
    perceiver passes ``PhraseLoudness.is_near``; a phrase without word times then never counts).
    ``[unk]`` never counts. Partial results, results without per-word confidences, and
    empty or malformed JSON give an empty Counter. This is the only place recognised words
    exist: they are never logged, returned or carried by an exception (every failure is
    swallowed into an empty Counter), and the returned keys are our own phrase constants.
    """
    try:
        return _count(_tokens(json.loads(result_json)), min_conf, near)
    except Exception:
        return Counter()


def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    """Polyphase 48 kHz -> 16 kHz along axis 0 (anti-aliased), float32. Caller purges both."""
    return resample_poly(np.asarray(x, dtype=np.float32), 1, 3, axis=0).astype(np.float32, copy=False)
