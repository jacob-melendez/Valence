"""Audio perception: one ~1 s mono 16 kHz float32 block in, one ``AudioObs`` out.

Loudness is smoothed in the power domain and converted to dB SPL with the calibrated offset.
YAMNet (AUDIO_STREAM, allowlisted classes) scores laughter and speech; laughter onsets are
debounced. Compliments come from Vosk with a restricted grammar (the phrases + "[unk]"), so
no free transcript ever exists; recognized words never leave ``count_compliments``. Every
block and the int16 copy Vosk reads are purged before ``process`` returns.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import BaseOptions, audio as mp_audio
from mediapipe.tasks.python.components.containers.audio_data import AudioData

from valence.contracts import AudioObs
from valence.perception import calibration
from valence.perception.audio_features import (
    YAMNET_ALLOWLIST, LaughterTracker, PowerSmoother, count_compliments, mean_power, vosk_grammar)
from valence.perception.gate import purge

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SR = 16000


class AudioPerceiver:
    def __init__(self, models_dir: Path = MODELS, *, spl_offset_db: float | None = None,
                 compliments: bool = True) -> None:
        self.offset = calibration.spl_offset_db() if spl_offset_db is None else spl_offset_db
        self.smoother = PowerSmoother()
        self.laughter = LaughterTracker()
        self._lock = threading.Lock()
        self._results: list[dict[str, float]] = []
        self._speech = 0.0
        self._ts_ms = 0
        self.classifier = mp_audio.AudioClassifier.create_from_options(mp_audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "yamnet.tflite")),
            running_mode=mp_audio.RunningMode.AUDIO_STREAM, category_allowlist=list(YAMNET_ALLOWLIST),
            max_results=len(YAMNET_ALLOWLIST), result_callback=self._on_yamnet))
        self.recognizer = None
        if compliments:
            import vosk

            vosk.SetLogLevel(-1)
            model = vosk.Model(str(models_dir / "vosk-model-small-en-us-0.15"))
            self.recognizer = vosk.KaldiRecognizer(model, SR, vosk_grammar())
            self.recognizer.SetWords(True)

    def close(self) -> None:
        self.classifier.close()

    def _on_yamnet(self, result, timestamp_ms: int) -> None:  # MediaPipe's thread: copy scores, nothing else
        scores = {c.category_name: float(c.score) for cl in result.classifications for c in cl.categories}
        with self._lock:
            self._results.append(scores)

    def process(self, block: np.ndarray, t: float, wall: float, seq: int) -> AudioObs:
        power = mean_power(block)
        self.smoother.update(power)

        self.classifier.classify_async(AudioData.create_from_array(block, SR), self._ts_ms)
        self._ts_ms += max(1, round(1000 * len(block) / SR))
        with self._lock:
            results, self._results = self._results, []
        if results:
            self._speech = 0.0
        for scores in results:  # async: usually the previous block's result
            _, speech, _ = self.laughter.update(scores, t)
            self._speech = max(self._speech, speech)

        hits = 0
        if self.recognizer is not None:
            from vosk import _ffi

            pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
            view = _ffi.from_buffer(pcm)  # zero-copy, so the int16 copy can be purged too
            if self.recognizer.AcceptWaveform(view):
                hits = sum(count_compliments(self.recognizer.Result()).values())
            del view
            purge(pcm, "audio")
            del pcm
        purge(block, "audio")
        del block

        return AudioObs(t=t, wall=wall, block_seq=seq, db_spl=self.smoother.db(self.offset), rms_power=power,
                        speech_prob=self._speech, laughter_onsets=self.laughter.take_onsets(), compliment_hits=hits)
