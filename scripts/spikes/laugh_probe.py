"""Laughter probe: per-second YAMNet scores for the laughter family while the user laughs on cue.

    python scripts/spikes/laugh_probe.py --out /path/to/laugh_probe.jsonl

Cued phases: quiet / LAUGH / talk normally / LAUGH / quiet. Writes one line per second with the
phase and the six allowlisted class scores (scalars only; the audio is purged block by block).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import BaseOptions, audio as mp_audio
from mediapipe.tasks.python.components.containers.audio_data import AudioData

from valence.edge.capture import MicCapture
from valence.perception.audio_features import YAMNET_ALLOWLIST
from valence.perception.gate import purge

ROOT = Path(__file__).resolve().parents[2]
PHASES = [("quiet", 6, "stay quiet"), ("laugh", 6, ">>> LAUGH (a real laugh) <<<"), ("talk", 8, "talk normally"),
          ("laugh", 6, ">>> LAUGH again <<<"), ("quiet", 6, "stay quiet"), ("laugh", 6, ">>> LAUGH once more <<<"),
          ("quiet", 4, "quiet")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mic", default=None)
    a = ap.parse_args()
    clf = mp_audio.AudioClassifier.create_from_options(mp_audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models/yamnet.tflite")),
        running_mode=mp_audio.RunningMode.AUDIO_CLIPS, category_allowlist=list(YAMNET_ALLOWLIST),
        max_results=len(YAMNET_ALLOWLIST)))
    mic = MicCapture(a.mic).start()
    input(f"mic: {mic.device_name}. ~{sum(p[1] for p in PHASES)} s of cued phases. Press Enter to start: ")
    with a.out.open("w", encoding="utf-8") as f:
        for phase, secs, text in PHASES:
            print(f"\a\n{text}", flush=True)
            for _ in range(secs):
                block, t, _ = mic.read(1.0)
                p = float(np.mean(block.astype(np.float64) ** 2))
                res = clf.classify(AudioData.create_from_array(block[-15600:], 16000))
                scores = {c.category_name: round(float(c.score), 3) for r in res for cl in r.classifications
                          for c in cl.categories}
                purge(block, "audio")
                del block
                laugh = max([v for k, v in scores.items() if k != "Speech"] or [0.0])
                print(f"  {phase:5} laugh {laugh:.2f}  speech {scores.get('Speech', 0):.2f}  dBFS {10 * np.log10(max(p, 1e-12)):.0f}",
                      flush=True)
                f.write(json.dumps({"phase": phase, "dbfs": round(10 * np.log10(max(p, 1e-12)), 1), **scores}) + "\n")
    mic.close()
    clf.close()
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
