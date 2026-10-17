"""Spikes 1-3: YOLO26n-pose + ByteTrack on MPS, MediaPipe GestureRecognizer, FaceLandmarker
(+ PoseLandmarker lite fallback). Prints per-model latency p50/p95 and a 10 Hz loop budget.

    python scripts/spikes/bench_video.py                 # bundled ultralytics sample images
    python scripts/spikes/bench_video.py --camera 0      # live webcam frames (purged after use)

Only timings and counts are printed; frames never leave this process.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from valence.perception.gate import purge

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def report(name: str, ms: list[float]) -> dict:
    r = {"name": name, "n": len(ms), "p50": statistics.median(ms), "p95": pct(ms, 0.95), "mean": statistics.fmean(ms)}
    print(f"  {name:<34} p50 {r['p50']:6.1f} ms   p95 {r['p95']:6.1f} ms   (n={r['n']})")
    return r


class Frames:
    """Sample images (resized to 1280x720) or live camera frames."""

    def __init__(self, camera: int | None) -> None:
        self.cap = None
        if camera is not None:
            self.cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                raise SystemExit(f"camera {camera} not available (permission?)")
            for _ in range(10):
                ok, f = self.cap.read()
                if ok:
                    purge(f, "video")
        else:
            assets = Path(cv2.__file__).parents[1] / "ultralytics" / "assets"
            self.samples = [cv2.resize(cv2.imread(str(assets / n)), (1280, 720)) for n in ("zidane.jpg", "bus.jpg")]
        self.i = 0

    def next(self) -> np.ndarray:
        if self.cap is not None:
            ok, f = self.cap.read()
            if not ok:
                raise SystemExit("camera read failed")
            return f
        self.i += 1
        return self.samples[self.i % len(self.samples)].copy()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}  mediapipe {mp.__version__}")
    frames = Frames(a.camera)
    results: list[dict] = []
    tracker = str(ROOT / "valence/perception/bytetrack_valence.yaml")
    if not os.path.exists(tracker):
        tracker = "bytetrack.yaml"

    # 1) YOLO26n-pose + ByteTrack
    for imgsz in (640, 480):
        model = YOLO(str(MODELS / "yolo26n-pose.pt"))
        ms, persons = [], []
        for k in range(a.n + 10):
            f = frames.next()
            t0 = time.perf_counter()
            r = model.track(f, persist=True, classes=[0], conf=0.4, imgsz=imgsz, device="mps",
                            tracker=tracker, verbose=False)[0]
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            dt = (time.perf_counter() - t0) * 1e3
            purge(f, "video")
            del f
            if k >= 10:
                ms.append(dt)
                persons.append(0 if kp is None else kp.shape[0])
        results.append(report(f"YOLO26n-pose+ByteTrack mps {imgsz}", ms))
        print(f"    persons/frame median {statistics.median(persons)}")
        del model

    # 2) GestureRecognizer, 3) FaceLandmarker, fallback PoseLandmarker lite
    V = vision.RunningMode.VIDEO
    gr = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "gesture_recognizer.task")), running_mode=V, num_hands=8))
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")), running_mode=V, num_faces=4,
        output_face_blendshapes=True))
    pl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")), running_mode=V, num_poses=4))

    tms = 0
    g_ms, f_ms, p_ms, cvt_ms = [], [], [], []
    hands, faces, poses, fist_scores = [], [], [], []
    for k in range(a.n + 5):
        f = frames.next()
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t1 = time.perf_counter()
        tms += 100
        g = gr.recognize_for_video(img, tms)
        t2 = time.perf_counter()
        fr = fl.detect_for_video(img, tms)
        t3 = time.perf_counter()
        pr = pl.detect_for_video(img, tms)
        t4 = time.perf_counter()
        n_faces = len(fr.face_landmarks)
        has_bs = bool(fr.face_blendshapes)
        fists = [c.score for cats in g.gestures for c in cats if c.category_name == "Closed_Fist"]
        purge(f, "video"); purge(rgb, "video")
        del f, rgb, img, fr, pr
        if k >= 5:
            cvt_ms.append((t1 - t0) * 1e3); g_ms.append((t2 - t1) * 1e3)
            f_ms.append((t3 - t2) * 1e3); p_ms.append((t4 - t3) * 1e3)
            hands.append(len(g.hand_landmarks)); faces.append(n_faces); fist_scores += fists
    results += [report("BGR->RGB + mp.Image", cvt_ms), report("GestureRecognizer (8 hands)", g_ms),
                report("FaceLandmarker (4 faces, blendshapes)", f_ms), report("PoseLandmarker lite (4 poses)", p_ms)]
    print(f"    hands/frame median {statistics.median(hands)}, faces/frame median {statistics.median(faces)}, "
          f"blendshapes={has_bs}, Closed_Fist scores seen {len(fist_scores)}")
    gr.close(); fl.close(); pl.close()
    frames.close()

    by = {r["name"]: r for r in results}
    yolo = by["YOLO26n-pose+ByteTrack mps 640"]["p50"]
    alt = max(by["GestureRecognizer (8 hands)"]["p50"], by["FaceLandmarker (4 faces, blendshapes)"]["p50"])
    loop = yolo + alt + by["BGR->RGB + mp.Image"]["p50"]
    print(f"\n10 Hz loop estimate (YOLO640 + max(gesture, face) + convert): {loop:.1f} ms "
          f"-> {'OK' if loop < 100 else 'OVER BUDGET: use imgsz=480 or PoseLandmarker fallback'}")


if __name__ == "__main__":
    main()
