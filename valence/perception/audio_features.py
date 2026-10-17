"""Audio scalars for the Ambient Physics and Somatic & Social layers.

Callers reduce each raw block to a mean power, YAMNet allowlist scores and compliment
counts, then purge it. Loudness is smoothed in the power domain and converted to dB last:
averaging dB values would understate bursts of sound. Recognised words exist only inside
``count_compliments`` and never leave it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
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


def _tokens(result: object) -> list[tuple[str, float]]:
    if not isinstance(result, dict) or not isinstance(result.get("result"), list):
        return []
    tokens: list[tuple[str, float]] = []
    for item in result["result"]:
        word = item.get("word") if isinstance(item, dict) else None
        conf = item.get("conf") if isinstance(item, dict) else None
        valid = isinstance(word, str) and isinstance(conf, (int, float)) and not isinstance(conf, bool)
        tokens.append((word.lower(), float(conf)) if valid else (UNK, 0.0))
    return tokens


def _count(tokens: list[tuple[str, float]], min_conf: float) -> Counter[str]:
    counts: Counter[str] = Counter()
    i = 0
    while i < len(tokens):
        for phrase, words in _PHRASE_TOKENS:
            window = tokens[i:i + len(words)]
            if len(window) == len(words) and all(w == p and c >= min_conf for (w, c), p in zip(window, words)):
                counts[phrase] += 1
                i += len(words)
                break
        else:
            i += 1
    return counts


def count_compliments(result_json: str | bytes, min_conf: float = 0.7) -> Counter[str]:
    """Count compliment phrases in a final Vosk result (recognizer with ``SetWords(True)``).

    A phrase counts when its words appear consecutively, each with conf >= ``min_conf``;
    ``[unk]`` never counts. Partial results, results without per-word confidences, and
    empty or malformed JSON give an empty Counter. This is the only place recognised words
    exist: they are never logged, returned or carried by an exception (every failure is
    swallowed into an empty Counter), and the returned keys are our own phrase constants.
    """
    try:
        return _count(_tokens(json.loads(result_json)), min_conf)
    except Exception:
        return Counter()


def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    """Polyphase 48 kHz -> 16 kHz along axis 0 (anti-aliased), float32. Caller purges both."""
    return resample_poly(np.asarray(x, dtype=np.float32), 1, 3, axis=0).astype(np.float32, copy=False)
