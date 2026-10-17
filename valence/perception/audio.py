"""Audio perception: one ~1 s mono 16 kHz float32 block in, one ``AudioObs`` out.

Loudness is smoothed in the power domain and converted to dB SPL with the calibrated offset.
YAMNet (AUDIO_CLIPS, allowlisted classes) runs synchronously on the block's last 0.975 s, so its
laughter/speech scores belong to exactly this block; laughter onsets are debounced. Compliments come from Vosk with a restricted grammar (the phrases + "[unk]"), so
no free transcript ever exists; recognized words never leave ``count_compliments``. Every
block and the int16 copy Vosk reads are purged before ``process`` returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mediapipe.tasks.python import BaseOptions, audio as mp_audio
from mediapipe.tasks.python.components.containers.audio_data import AudioData

from valence.contracts import AudioObs
from valence.perception import calibration
from valence.perception.audio_features import (
    YAMNET_ALLOWLIST, LaughterTracker, PowerSmoother, count_compliments, mean_power, vosk_grammar)
from valence.perception.gate import note_transcript_purged, purge_derived

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SR = 16000
YAMNET_WINDOW = 15600  # 0.975 s at 16 kHz


class AudioPerceiver:
    def __init__(self, models_dir: Path = MODELS, *, spl_offset_db: float | None = None,
                 compliments: bool = True) -> None:
        self.offset = calibration.spl_offset_db() if spl_offset_db is None else spl_offset_db
        self.smoother = PowerSmoother()
        self.laughter = LaughterTracker()
        self.classifier = mp_audio.AudioClassifier.create_from_options(mp_audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "yamnet.tflite")),
            running_mode=mp_audio.RunningMode.AUDIO_CLIPS, category_allowlist=list(YAMNET_ALLOWLIST),
            max_results=len(YAMNET_ALLOWLIST)))
        self.recognizer = None
        if compliments:
            import vosk

            vosk.SetLogLevel(-1)
            model = vosk.Model(str(models_dir / "vosk-model-small-en-us-0.15"))
            self.recognizer = vosk.KaldiRecognizer(model, SR, vosk_grammar())
            self.recognizer.SetWords(True)

    def close(self) -> None:
        self.classifier.close()

    def process(self, block: np.ndarray, t: float, wall: float, seq: int) -> AudioObs:
        power = mean_power(block)
        self.smoother.update(power)

        clip = np.ascontiguousarray(block[-YAMNET_WINDOW:]) if len(block) >= YAMNET_WINDOW else block.copy()
        results = self.classifier.classify(AudioData.create_from_array(clip, SR))
        purge_derived(clip, "audio")
        del clip
        scores: dict[str, float] = {}
        for res in results:
            for cl in res.classifications:
                for c in cl.categories:
                    scores[c.category_name] = max(scores.get(c.category_name, 0.0), float(c.score))
        del results
        _, speech, _ = self.laughter.update(scores, t)

        hits = 0
        if self.recognizer is not None:
            from vosk import _ffi

            pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
            view = _ffi.from_buffer(pcm)  # zero-copy, so the int16 copy can be purged too
            if self.recognizer.AcceptWaveform(view):
                hits = sum(count_compliments(self.recognizer.Result()).values())
                note_transcript_purged()
            del view
            purge_derived(pcm, "audio")
            del pcm
        purge_derived(block, "audio")
        del block

        return AudioObs(t=t, wall=wall, block_seq=seq, db_spl=self.smoother.db(self.offset), rms_power=power,
                        speech_prob=speech, laughter_onsets=self.laughter.take_onsets(), compliment_hits=hits)
