"""Video perception: one BGR frame in, one ``VideoObs`` out.

Per call (target 10 Hz): YOLO26n-pose + ByteTrack every time; MediaPipe gestures on even
and face landmarks on odd iterations (5 Hz each); background lux at 1 Hz. Faces: one full-frame
FaceLandmarker pass, matched to people by the YOLO nose keypoint; anyone left unmatched gets a
second try on a head crop (box from YOLO head keypoints), because the short-range face detector
misses faces that fill little of the frame. The output is
scalars, booleans and normalized boxes only. The frame and every derived buffer go through
the purge gate before ``process`` returns, and nothing image-shaped is kept on ``self``:
not the frame, not landmarks, not YOLO's cached batch. Display ids are ephemeral; per-person
baselines live in memory only and die with the track.

All MediaPipe instances belong to the thread that constructs the perceiver; construct and
call it from one perception thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.processors import ClassifierOptions

from valence.config import settings
from valence.contracts import PersonObs, VideoObs
from valence.perception import calibration
from valence.perception.face_valence import (
    FaceValenceCalibrator, assign_faces, blendshape_scores, nose_tip_px, valence_raw)
from valence.perception.gate import purge, purge_derived
from valence.perception.hands import FistTracker, assign_hands, is_curl_fist
from valence.perception.ids import DisplayIdMapper
from valence.perception.lux import LuxEstimator, background_stats
from valence.perception.posture import PostureTracker
from valence.perception.regions import black_out, head_box_px, norm_bbox

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
TRACKER_CFG = str(Path(__file__).with_name("bytetrack_valence.yaml"))

NOSE, L_EYE, R_EYE, L_WRIST, R_WRIST = 0, 1, 2, 9, 10
LUX_PERIOD_S = 1.0
FACE_CROP_SCALE = 2.2
FACE_CROP_PX = 256
FACE_HOLD_S = 0.45  # a face seen on the previous (odd) iteration still counts as visible
KEYBOARD_ZONE = 0.8  # hands whose wrist is in the bottom 20% of the frame are typing, not fists


@dataclass(slots=True)
class _Det:
    track_id: int
    box_px: np.ndarray  # (4,) x1 y1 x2 y2
    kxy: np.ndarray     # (17, 2) pixels
    kconf: np.ndarray   # (17,)


@dataclass(slots=True)
class _Person:
    """Ephemeral per-display-id state: scalars and keypoint history only."""

    posture: PostureTracker = field(default_factory=PostureTracker)
    face: FaceValenceCalibrator = field(default_factory=FaceValenceCalibrator)
    face_valence: float = math.nan
    face_seen_t: float = -math.inf


class VideoPerceiver:
    def __init__(self, models_dir: Path = MODELS, *, imgsz: int = 640, device: str = "mps",
                 conf: float = 0.4, obscure_face: bool = False, min_person_h: float | None = None,
                 max_tracks: int | None = None) -> None:
        """People shorter than ``min_person_h`` of the frame height are ignored, and only the
        ``max_tracks`` largest are kept (defaults: settings, 25% and 3); ignored people still mask
        the lux background."""
        from ultralytics import YOLO

        self.imgsz, self.device, self.conf, self.obscure_face = imgsz, device, conf, obscure_face
        self.min_person_h = settings.min_box_height if min_person_h is None else min_person_h
        self.max_tracks = settings.max_tracks if max_tracks is None else max_tracks
        self.yolo = YOLO(str(models_dir / "yolo26n-pose.pt"))
        video = vision.RunningMode.VIDEO
        self.gestures = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "gesture_recognizer.task")),
            running_mode=video, num_hands=8,
            canned_gesture_classifier_options=ClassifierOptions(max_results=8)))
        face_task = BaseOptions(model_asset_path=str(models_dir / "face_landmarker.task"))
        self.faces_full = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=face_task, running_mode=vision.RunningMode.IMAGE, num_faces=4,
            output_face_blendshapes=True))
        self.faces_crop = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=face_task, running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=True))
        self.ids = DisplayIdMapper()
        self.fists = FistTracker()
        self.lux = LuxEstimator(calibration.lux_model())
        self.people: dict[int, _Person] = {}
        self._iter = 0
        self._gesture_ts = -1
        self._last_lux_t = -math.inf
        self._lux, self._lux_q = math.nan, 0.0

    def close(self) -> None:
        self.gestures.close()
        self.faces_full.close()
        self.faces_crop.close()

    def process(self, frame_bgr: np.ndarray, t: float, wall: float, seq: int, frames_decoded: int = 0) -> VideoObs:
        """``frames_decoded``: the capture's cumulative decoded-frame count (governance check)."""
        h, w = frame_bgr.shape[:2]
        it, self._iter = self._iter, self._iter + 1

        all_dets = self._track(frame_bgr, w, h)
        dets = sorted((d for d in all_dets if d.box_px[3] - d.box_px[1] >= self.min_person_h * h),
                      key=lambda d: -float((d.box_px[2] - d.box_px[0]) * (d.box_px[3] - d.box_px[1])))
        dets = dets[: self.max_tracks]
        id_map = self.ids.update([(d.track_id, norm_bbox(d.box_px, w, h)) for d in dets], t)
        for gone in self.ids.pop_released():
            self.people.pop(gone, None)
            self.fists.drop(gone)
        by_id = {id_map[d.track_id]: d for d in dets if d.track_id in id_map}
        postures = {did: self.people.setdefault(did, _Person()).posture.update(d.kxy, d.kconf, t)
                    for did, d in by_id.items()}

        if self.obscure_face:  # anti-bias proof: MediaPipe never sees a face
            black_out(frame_bgr, [head_box_px(d.kxy, d.kconf, w, h) for d in all_dets])

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if it % 2 == 0:
            self._update_fists(mp_image, by_id, w, h, t)
        else:
            self._update_faces(mp_image, rgb, by_id, w, h, t)
        del mp_image
        purge_derived(rgb, "video")
        del rgb

        boxes_norm = [norm_bbox(d.box_px, w, h) for d in all_dets]
        if t - self._last_lux_t >= LUX_PERIOD_S:
            self._last_lux_t = t
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            self._lux, self._lux_q = self.lux.update(background_stats(gray, boxes_norm), t)
            purge_derived(gray, "video")
            del gray

        purge(frame_bgr, "video")
        del frame_bgr

        persons = []
        for did, d in sorted(by_id.items()):
            p, ps = self.people[did], postures[did]
            onsets, active = self.fists.person_state(did)
            visible = t - p.face_seen_t <= FACE_HOLD_S
            persons.append(PersonObs(
                track_id=did, bbox=norm_bbox(d.box_px, w, h), fist_onsets=onsets, fist_active=active,
                slump=ps.slump, head_in_hands=ps.head_in_hands, motion=float(ps.motion),
                face_visible=visible, face_valence=p.face_valence if visible else math.nan))
        return VideoObs(t=t, wall=wall, frame_seq=seq, frames_decoded=frames_decoded, lux=float(self._lux),
                        lux_quality=float(self._lux_q), frame_w=w, frame_h=h, mirrored=False,
                        persons=tuple(persons))

    def _track(self, frame_bgr: np.ndarray, w: int, h: int) -> list[_Det]:
        r = self.yolo.track(frame_bgr, persist=True, classes=[0], conf=self.conf, imgsz=self.imgsz,
                            device=self.device, tracker=TRACKER_CFG, verbose=False)[0]
        dets: list[_Det] = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy()
            kconf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
            dets = [_Det(int(i), boxes[n].copy(), kxy[n].copy(), kconf[n].copy()) for n, i in enumerate(ids)]
        del r
        self._scrub_yolo()
        return dets

    def _scrub_yolo(self) -> None:
        """Ultralytics caches the input image on its predictor (batch, results, dataset); drop it."""
        p = getattr(self.yolo, "predictor", None)
        if p is not None:
            p.batch = p.results = p.dataset = p.plotted_img = None

    def _ts_ms(self, t: float) -> int:
        self._gesture_ts = max(int(t * 1000), self._gesture_ts + 1)  # VIDEO mode: strictly increasing ms
        return self._gesture_ts

    def _update_fists(self, image: mp.Image, by_id: dict[int, _Det], w: int, h: int, t: float) -> None:
        res = self.gestures.recognize_for_video(image, self._ts_ms(t))
        wrists_px, scores = [], []
        for n, lms in enumerate(res.hand_landmarks):
            if lms[0].y >= KEYBOARD_ZONE:
                continue  # keyboard zone: never scored, so the hand counts as unobserved
            wrists_px.append((lms[0].x * w, lms[0].y * h))
            cats = res.gestures[n] if n < len(res.gestures) else []
            fist = next((c.score for c in cats if c.category_name == "Closed_Fist"), None)
            if fist is None:
                pts = np.array([(lm.x, lm.y) for lm in lms], dtype=np.float32)
                fist = 1.0 if is_curl_fist(pts, aspect=h / w) else 0.0
                del pts
            scores.append(float(fist))
        del res
        person_wrists = {did: {"L": (*d.kxy[L_WRIST], d.kconf[L_WRIST]), "R": (*d.kxy[R_WRIST], d.kconf[R_WRIST])}
                         for did, d in by_id.items()}
        seen = set()
        for key, score in zip(assign_hands(wrists_px, person_wrists, w), scores):
            if key is not None:
                self.fists.update(key[0], key[1], score, t)
                seen.add(key)
        self.fists.mark_unobserved(t, seen)

    def _update_faces(self, image: mp.Image, rgb: np.ndarray, by_id: dict[int, _Det], w: int, h: int,
                      t: float) -> None:
        res = self.faces_full.detect(image)
        points = [nose_tip_px(lms, w, h) for lms in res.face_landmarks]
        raws = [valence_raw(blendshape_scores(bs)) for bs in res.face_blendshapes]
        del res  # 478 landmarks per face: never stored
        matched = assign_faces(points, {did: self._nose(d) for did, d in by_id.items()}, w)
        found = {did: raws[i] for i, did in enumerate(matched) if did is not None and i < len(raws)}
        for did, d in by_id.items():
            if did not in found and (raw := self._face_from_crop(rgb, d, w, h)) is not None:
                found[did] = raw
        for did, raw in found.items():
            p = self.people[did]
            p.face_valence = p.face.update(raw, t)
            p.face_seen_t = t

    def _face_from_crop(self, rgb: np.ndarray, d: _Det, w: int, h: int) -> float | None:
        box = head_box_px(d.kxy, d.kconf, w, h, scale=FACE_CROP_SCALE)
        if box is None or box[2] - box[0] < 16 or box[3] - box[1] < 16:
            return None
        x1, y1, x2, y2 = box
        crop = cv2.resize(rgb[y1:y2, x1:x2], (FACE_CROP_PX, FACE_CROP_PX))
        res = self.faces_crop.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        purge_derived(crop, "video")
        del crop
        raw = valence_raw(blendshape_scores(res.face_blendshapes[0])) if res.face_blendshapes else None
        del res
        return raw

    @staticmethod
    def _nose(d: _Det) -> tuple[float, float, float]:
        if d.kconf[NOSE] >= 0.3:
            return float(d.kxy[NOSE, 0]), float(d.kxy[NOSE, 1]), float(d.kconf[NOSE])
        eyes = d.kxy[[L_EYE, R_EYE]].mean(axis=0)
        return float(eyes[0]), float(eyes[1]), float(min(d.kconf[L_EYE], d.kconf[R_EYE]))
