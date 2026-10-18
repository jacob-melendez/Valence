"""Audio perception: one ~1 s mono 16 kHz float32 block in, one ``AudioObs`` out.

Loudness is the block's Leq: mean power -> dB SPL with the calibrated offset (the aggregator
averages ``rms_power`` in the power domain; dB values are never averaged).
YAMNet (AUDIO_CLIPS, allowlisted classes) runs synchronously on the block's last 0.975 s, so its
laughter/speech scores belong to exactly this block; laughter onsets are debounced. Compliments come from Vosk with a restricted grammar (the phrases + "[unk]"), so
no free transcript ever exists; recognized words never leave ``count_compliments``. A phrase counts only if
its own words are louder than the room's recent floor (``PhraseLoudness``: 100 ms power scalars on the
recognizer's clock), because in crowd babble the restricted grammar finds phrases at the crowd's level. Every
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
    YAMNET_ALLOWLIST, LaughterTracker, PhraseLoudness, count_compliments, mean_power, power_to_db,
    vosk_grammar)
from valence.perception.gate import note_transcript_purged, purge_derived

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SR = 16000
YAMNET_WINDOW = 15600  # 0.975 s at 16 kHz


class AudioPerceiver:
    def __init__(self, models_dir: Path = MODELS, *, spl_offset_db: float | None = None,
                 compliments: bool = True) -> None:
        self.offset = calibration.spl_offset_db() if spl_offset_db is None else spl_offset_db
        self.laughter = LaughterTracker()
        self.classifier = mp_audio.AudioClassifier.create_from_options(mp_audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "yamnet.tflite")),
            running_mode=mp_audio.RunningMode.AUDIO_CLIPS, category_allowlist=list(YAMNET_ALLOWLIST),
            max_results=len(YAMNET_ALLOWLIST)))
        self.recognizer = None
        self.loudness = PhraseLoudness(SR)  # on the recognizer's clock: fed exactly the samples Vosk gets
        if compliments:
            import vosk

            vosk.SetLogLevel(-1)
            model = vosk.Model(str(models_dir / "vosk-model-small-en-us-0.15"))
            self.recognizer = vosk.KaldiRecognizer(model, SR, vosk_grammar())
            self.recognizer.SetWords(True)

    def close(self) -> None:
        try:
            self.classifier.close()
        except RuntimeError:  # MediaPipe's dispatcher already shut down at interpreter exit
            pass

    def process(self, block: np.ndarray, t: float, wall: float, seq: int) -> AudioObs:
        power = mean_power(block)

        # A tail slice is a view: AudioData copies it, and purging the view would zero the whole block
        # before Vosk reads it, so the block itself is purged once, at the end.
        results = self.classifier.classify(AudioData.create_from_array(block[-YAMNET_WINDOW:], SR))
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

            self.loudness.add(block)
            pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
            view = _ffi.from_buffer(pcm)  # zero-copy, so the int16 copy can be purged too
            if self.recognizer.AcceptWaveform(view):
                hits = sum(count_compliments(self.recognizer.Result(), near=self.loudness.is_near).values())
                note_transcript_purged()
            del view
            purge_derived(pcm, "audio")
            del pcm
        purge_derived(block, "audio")
        del block

        return AudioObs(t=t, wall=wall, block_seq=seq, db_spl=power_to_db(power, self.offset), rms_power=power,
                        speech_prob=speech, laughter_onsets=self.laughter.take_onsets(), compliment_hits=hits)
