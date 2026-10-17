"""Posture probe: which keypoint geometry separates "looking at the keyboard" from "slumping"?

Cued cycles of UPRIGHT / LOOK DOWN at keyboard / SLUMP; YOLO26n-pose only. Per frame it records
scale-free geometry ratios of the main person (no keypoints, no pixels):

- ``drop_{nose,eye,ear}`` = (shoulder_mid_y - point_y) / shoulder_width,
- ``sh_y`` = shoulder_mid_y / frame_h, ``sh_w`` = shoulder_width / frame_w,
- keypoint confidences of nose / eyes / ears / shoulders.

    python scripts/spikes/posture_probe.py --out /path/to/probe.jsonl [--cycles 3] [--phase-s 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from valence.edge.capture import CameraCapture
from valence.perception.gate import purge

ROOT = Path(__file__).resolve().parents[2]
PHASES = [("upright", "sit UPRIGHT, look at the screen"), ("gaze", "LOOK DOWN at the keyboard (stay upright)"),
          ("slump", "SLUMP: shoulders forward, head sinks")]


def ratios(kxy: np.ndarray, kc: np.ndarray, w: int, h: int) -> dict:
    ok = kc >= 0.5
    if not (ok[5] and ok[6]):
        return {}
    mid_y = (kxy[5, 1] + kxy[6, 1]) / 2
    sw = float(np.linalg.norm(kxy[5] - kxy[6]))
    out = {"sh_y": mid_y / h, "sh_w": sw / w, "c_nose": float(kc[0]), "c_eyes": float(kc[1:3].min()),
           "c_ears": float(kc[3:5].min()), "c_ear_max": float(kc[3:5].max())}
    for name, idx in (("nose", [0]), ("eye", [1, 2]), ("ear", [3, 4])):
        sel = [i for i in idx if ok[i]]
        out[f"drop_{name}"] = float((mid_y - kxy[sel, 1].mean()) / sw) if sel else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--phase-s", type=float, default=15.0)
    ap.add_argument("--camera", type=int, default=None)
    a = ap.parse_args()
    from ultralytics import YOLO

    yolo = YOLO(str(ROOT / "models/yolo26n-pose.pt"))
    cam = CameraCapture(a.camera).start()
    plan = [p for _ in range(a.cycles) for p in PHASES]
    input(f"{len(plan)} phases of {a.phase_s:.0f} s. Follow the on-screen phase. Press Enter to start: ")
    n = 0
    with a.out.open("w", encoding="utf-8") as f:
        for i, (phase, text) in enumerate(plan):
            print(f"\a\n[{i + 1}/{len(plan)}] {phase.upper()}: {text}", flush=True)
            t_end = time.monotonic() + a.phase_s
            t_phase = time.monotonic()
            while time.monotonic() < t_end:
                item = cam.take(1.0)
                if item is None:
                    continue
                frame, t, _, _ = item
                hh, ww = frame.shape[:2]
                r = yolo.track(frame, persist=True, classes=[0], conf=0.4, device="mps", verbose=False)[0]
                rows = []
                if r.keypoints is not None and r.boxes is not None and len(r.boxes):
                    areas = ((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])).cpu().numpy()
                    j = int(areas.argmax())
                    rows.append(ratios(r.keypoints.xy[j].cpu().numpy().copy(), r.keypoints.conf[j].cpu().numpy().copy(), ww, hh))
                del r
                yolo.predictor.batch = yolo.predictor.results = yolo.predictor.dataset = None
                purge(frame, "video")
                del frame, item
                if rows and rows[0]:
                    f.write(json.dumps({"phase": phase, "cycle": i // 3, "t_in_phase": round(t - t_phase, 2),
                                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[0].items()}}) + "\n")
                    n += 1
    cam.close()
    print(f"\nwrote {n} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
