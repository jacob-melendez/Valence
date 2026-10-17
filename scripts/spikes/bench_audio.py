"""Spikes 4-5: mic dBFS, YAMNet AudioClassifier (AUDIO_STREAM) laughter/speech scores, and a
Vosk restricted-grammar compliment recognizer, all on the live mic.

    python scripts/spikes/bench_audio.py --seconds 20 [--mic "MacBook Pro Microphone"] [--say]

--say plays compliment phrases through the speakers with macOS `say` so the recognizer is
exercised end to end (speaker -> mic) without any audio file. Only scalars and phrase counts
are printed: no transcript exists (the grammar only knows the phrases + [unk]).
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sounddevice as sd
from mediapipe.tasks.python import BaseOptions, audio
from mediapipe.tasks.python.components.containers.audio_data import AudioData

from valence.perception.gate import purge

ROOT = Path(__file__).resolve().parents[2]
SR = 16000
PHRASES = ["great job", "nice work", "well done", "thank you", "love it", "good idea", "awesome", "amazing"]
ALLOW = ["Speech", "Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]
SAY = ["great job", "thank you", "nice work", "that is amazing", "well done everyone"]


def find_device(name: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    raise SystemExit(f"no input device matching {name!r}")


def phrase_hits(result_json: str) -> Counter:
    """Vosk result -> phrase counts; words never leave this function."""
    try:
        words = [w for w in json.loads(result_json).get("result", []) if w.get("conf", 0) >= 0.7]
    except (ValueError, AttributeError):
        return Counter()
    seq = [w["word"] for w in words]
    hits: Counter = Counter()
    for p in PHRASES:
        toks = p.split()
        for i in range(len(seq) - len(toks) + 1):
            if seq[i:i + len(toks)] == toks:
                hits[p] += 1
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--mic", default="MacBook Pro Microphone")
    ap.add_argument("--say", action="store_true")
    a = ap.parse_args()

    import vosk
    vosk.SetLogLevel(-1)
    vmodel = vosk.Model(str(ROOT / "models/vosk-model-small-en-us-0.15"))
    rec = vosk.KaldiRecognizer(vmodel, SR, json.dumps(PHRASES + ["[unk]"]))
    rec.SetWords(True)

    yam_results: list[tuple[int, dict]] = []
    yam_lat: list[float] = []
    sent_at: dict[int, float] = {}

    def on_result(res, ts_ms: int) -> None:
        scores = {c.category_name: c.score for cl in res.classifications for c in cl.categories}
        yam_results.append((ts_ms, scores))
        if ts_ms in sent_at:
            yam_lat.append((time.perf_counter() - sent_at.pop(ts_ms)) * 1e3)

    clf = audio.AudioClassifier.create_from_options(audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models/yamnet.tflite")),
        running_mode=audio.RunningMode.AUDIO_STREAM, category_allowlist=ALLOW, max_results=6,
        result_callback=on_result))

    q: queue.Queue = queue.Queue(maxsize=200)

    def callback(indata, frames, t, status) -> None:
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    dev = find_device(a.mic)
    print(f"mic: {sd.query_devices(dev)['name']}  {SR} Hz mono float32 blocksize 1600")
    if a.say:
        def speak() -> None:
            time.sleep(2.0)
            for s in SAY:
                subprocess.run(["say", "-a", "MacBook Pro Speakers", s])
                time.sleep(1.5)
        threading.Thread(target=speak, daemon=True).start()

    total_hits: Counter = Counter()
    vosk_ms: list[float] = []
    acc: list[np.ndarray] = []
    t_start = time.monotonic()
    ts_ms = 0
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1600, device=dev, callback=callback):
        while time.monotonic() - t_start < a.seconds:
            block = q.get(timeout=2.0)
            acc.append(block)
            t0 = time.perf_counter()
            if rec.AcceptWaveform((np.clip(block, -1, 1) * 32767).astype(np.int16).tobytes()):
                total_hits += phrase_hits(rec.Result())
            vosk_ms.append((time.perf_counter() - t0) * 1e3)
            if len(acc) == 10:  # ~1 s
                sec = np.concatenate(acc)
                p = float(np.mean(sec.astype(np.float64) ** 2))
                sent_at[ts_ms] = time.perf_counter()
                clf.classify_async(AudioData.create_from_array(sec, SR), ts_ms)
                ts_ms += 1000
                latest = yam_results[-1][1] if yam_results else {}
                laugh = max([latest.get(k, 0.0) for k in ALLOW[1:]] or [0.0])
                print(f"  t={time.monotonic() - t_start:5.1f}s  {10 * np.log10(max(p, 1e-12)):6.1f} dBFS  "
                      f"speech {latest.get('Speech', 0.0):.2f}  laugh {laugh:.2f}  hits {dict(total_hits)}")
                for b in acc:
                    purge(b, "audio")
                purge(sec, "audio")
                acc.clear()
                del sec
            purge(block, "audio")
            del block
    total_hits += phrase_hits(rec.FinalResult())
    clf.close()
    print(f"\nYAMNet results {len(yam_results)}, callback latency p50 "
          f"{statistics.median(yam_lat) if yam_lat else float('nan'):.1f} ms")
    print(f"Vosk AcceptWaveform per 100 ms block: p50 {statistics.median(vosk_ms):.2f} ms, "
          f"max {max(vosk_ms):.1f} ms")
    print(f"compliment hits: {dict(total_hits)}  (spoken via --say: {SAY if a.say else 'none'})")


if __name__ == "__main__":
    main()
