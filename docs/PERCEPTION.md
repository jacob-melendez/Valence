# Edge perception

Webcam + microphone in, **only scalars, booleans and normalized boxes out** (`VideoObs` ~10 Hz, `AudioObs` ~1 Hz,
see `valence/contracts.py`). Everything runs on the Mac, offline after the one-time model downloads
(`scripts/download_models.sh`). Code: `valence/perception/` (pure perception, no file writes at all),
`valence/edge/` (capture threads, runner, features-only recorder).

```
camera ─grab() 30 fps─► retrieve() 10 Hz ─► 1-slot handoff ─► VideoPerceiver.process ─► VideoObs ─► sink
                          (displaced frame: purged)            │ YOLO26n-pose + ByteTrack (every frame)
                                                               │ GestureRecognizer (even) / FaceLandmarker (odd)
                                                               │ lux (1 Hz)      ─► purge(frame), purge_derived(copies)
mic ─16 kHz float32, 100 ms callback (copy only)─► 1 s block ─► AudioPerceiver.process ─► AudioObs ─► sink
                                                               │ power/dB, YAMNet clip, Vosk grammar ─► purge
```

## What is discarded, and when

| Raw data | Where it exists | How long | What leaves |
|---|---|---|---|
| Camera frame (BGR) | capture thread → perception thread | < 1 frame period (~50 ms of processing) | nothing; zero-filled by `gate.purge(frame, "video")` |
| Frames grabbed but not needed | camera driver | never decoded (`grab()` only) | nothing |
| Frame displaced from the slot | capture thread | until the next decode | nothing; purged |
| RGB copy, head crops, grayscale | inside `process()` | same call | nothing; `purge_derived(..., "video")` |
| YOLO results (boxes, 17 keypoints) | `process()` locals | same call; arrays copied out, `Results` deleted, ultralytics' cached batch cleared | normalized box; keypoints never leave perception |
| Previous keypoints (17x2 floats) | `PostureTracker` | one sample, per display id | motion scalar |
| Face mesh (478 landmarks) + blendshapes | `_update_faces` locals | same call | one valence scalar |
| Hand landmarks (21) | `_update_fists` locals | same call | fist score -> debounced onsets |
| Mic callback block (100 ms) | queue | < 1 s | nothing; purged |
| 1 s block, int16 PCM for Vosk, YAMNet clip | `AudioPerceiver.process` | same call | dB, power, speech/laughter scores |
| Recognized words | inside `count_compliments` only | that function | an integer count; result string counted via `note_transcript_purged()` |

Per-person baselines (neutral face, upright posture) are in-memory medians tied to an ephemeral display id; when the
id is released (person gone > 3 s) they are forgotten. No embeddings, no re-identification, no identity.
Recordings (`--jsonl`, `data/real/*.jsonl`) contain only the contract fields (`contracts.obs_to_dict`) and only
**participants**: a person whose box is shorter than 0.55x the tallest in the frame (a bystander behind the desk) is
dropped before the line is written (`valence.edge.run.participants_only`, same rule as the aggregator). Perception
still tracks bystanders so their boxes can mask the lux background, but their features never reach disk.

## Signals

### People and ids (`ids.py`, `video.py`)
YOLO26n-pose on MPS (imgsz 640, conf 0.4, persons only) + ByteTrack (track_buffer 60 ≈ 6 s). Boxes shorter than
25% of the frame height are ignored and at most the 3 largest are kept (`settings.min_box_height`, `max_tracks`),
so background people in a busy room don't become tracks. ByteTrack ids map to display ids 1..N by first appearance;
a new track within 3 s and 0.25 frame-widths of a lost one reuses its id.

### Fists (`hands.py`, `debounce.py`)
MediaPipe GestureRecognizer (VIDEO mode, 8 hands, all 8 categories scored) at 5 Hz. Each hand is assigned to the
nearest YOLO wrist (keypoints 9/10, conf ≥ 0.5) within 0.12 frame-widths (globally greedy). Hands whose wrist is in the
**bottom 20% of the frame** (keyboard zone) are not scored. Per (person, hand): enter at `Closed_Fist` ≥ **0.7 held
≥ 0.5 s**, exit below 0.4 held 1 s, onsets counted on rising edges with a 3 s refractory. If the gesture head returns
no categories, a curl heuristic stands in (≥ 3 fingers with tip-to-wrist < PIP-to-wrist). `fist_onsets` is a
per-frame delta; `fist_active` = any hand active.

### Slump and head-in-hands (`posture.py`)
Keypoints with conf < 0.5 are NaN. `head_drop = (shoulder_mid_y − nose_y) / shoulder_width` (EMA α 0.3).
Baseline = median over the person's first 20 s (≥ 10 samples). **Slump** when `head_drop < 0.7 × baseline` for ≥ 3 s;
clears on recovery. **Head-in-hands** when both wrists are within 0.5 shoulder-widths of the nearest visible head
point (nose, eyes or ears, since hands hide the nose) and above the shoulder line for ≥ 2 s.
**Motion** = mean displacement of the torso and arm keypoints (COCO 5-12: shoulders, elbows, wrists, hips) with
conf ≥ 0.6 in both consecutive samples / shoulder width / dt, i.e. shoulder-widths per second (EMA α 0.3). The head is
excluded because a dropped head's nose/eyes/ears lose confidence and jump: a held slump read 0.101 sw/s live under the
old all-keypoint mean, against 0.033 upright. A point that moved > 1 shoulder width in one sample is dropped as a
keypoint identity switch, and each left/right pair is matched whichever way is shorter, so a one-sample L/R label swap
is not motion (a mean of the rest, not a median, so one moving arm still counts). With fewer than 3 surviving points,
unless both shoulders survive (arms out of a laptop frame), the last value is held; NaN only before the first
measurable pair (0.0 means still).
`scripts/spikes/live_capture.py --only slump_hold` measures held-slump vs upright motion.

### Face valence (`face_valence.py`)
FaceLandmarker (IMAGE mode, blendshapes) at 5 Hz: one full-frame pass matched to people by the YOLO nose keypoint
(Hungarian, ≤ 0.15 frame-widths), then a per-person head-crop retry (scale 2.2, 256 px) for anyone unmatched.
Blendshapes are averaged left/right: `v_raw = smile − 0.7·frown − 0.5·browDown − 0.3·noseSneer − 0.2·mouthPress`,
`v = clip(tanh(2.5·(v_raw − b)), −1, 1)` with `b` the person's median over their first 20 s of visible face.
**`cheekSquint` is deliberately excluded**: it is the Duchenne cue that separates real from forced smiles, and the
face-only baseline must stay a naive "smile = happy" reader so the context model's correction is measurable.
No face → `face_visible=False`, `face_valence=NaN`. `--obscure-face` blacks out every head box before MediaPipe runs
(anti-bias proof: the pipeline still runs on context alone).

### Light (`lux.py`)
Once per second: mean luminance of the background (person boxes masked) and a single-frame sensor-noise estimate
(Immerkær Laplacian, on the background, ignoring clipped pixels; no previous frame is kept).
`lux = exp(a + b·Y_bg + c·σ)` fitted from 3-4 phone lux-meter readings (`scripts/calibrate_lux.py`; 2-point fallback).
Reported as a 10 s median; `lux_quality = background share × unclipped share × stability`, halved while uncalibrated.

### Loudness (`audio_features.py`, `audio.py`)
`rms_power = mean(x²)` per 1 s block; smoothed in the power domain (EMA α 0.3) and converted
`db_spl = 10·log10(power) + offset`, offset from `scripts/calibrate_spl.py` against a phone SPL app (default +94).
dB values are never averaged.

### Laughter and speech
YAMNet (AUDIO_CLIPS, synchronous on the block's last 0.975 s, allowlist Speech / Laughter / Giggle / Snicker /
Belly laugh / Chuckle, chortle). `laugh` = max of the laughter family: ON at ≥ 0.05 for one result, OFF once it has stayed below 0.02
for 1.5 s (the third low result), onsets with a 5.5 s refractory. (Measured: the laptop mic hears a real laugh mostly as Speech, laughter family
0.04-0.08 with one 0.50, while quiet and normal talking score exactly 0.00; the planned 0.30 x 2 rule
never fired. Inside one laugh the family drops to 0.00 for single seconds. With an instant exit and a 4 s refractory,
a 5 s laugh counted twice in capture run 3. The exit hold and the longer refractory make it count once, and laughs 6 s or more
apart still count separately. Both values sit half a block off the 1 s grid so millisecond jitter in the
block timestamps cannot flip the outcome; see the replay in BENCHMARKS.md. Other people's laughter is removed downstream by the near-field gate.) `speech_prob` = the Speech score. Near-field gating happens downstream.

### Compliments
Vosk small-en-us with a **restricted grammar**: "great job", "nice work", "well done", "thank you", "love it",
"good idea", "awesome", "amazing" + `[unk]`. Anything else decodes as `[unk]`, so no free transcript ever exists.
A phrase counts when its words are consecutive with confidence ≥ 0.7.

## Calibration
`data/real/calibration.json` holds constants only (`spl_offset_db`, lux `a, b, c`), written by
`valence/edge/calibration_store.py`. Per-person baselines are never stored.

## Limitations (honest)
- `del` removes a name, not memory. Zero-filling first means the bytes are gone from Python-owned buffers, but copies
  inside native code (AVFoundation pixel buffers, MediaPipe's `mp.Image`, MPS tensors in the GPU allocator) can't be
  reached from Python and are not zeroed.
- Display ids are per-session and can swap when two people cross; nothing links an id to a person.
- Fists: typing, gripping a mug or resting a chin on a hand can look like a fist; the 0.7/0.5 s rule and the keyboard
  zone reduce this (rates in `docs/BENCHMARKS.md`). Hands far from the camera (> ~2 m) are too small for the palm detector.
- Slump is relative to the first 20 s: someone already slumped at the start gets a low baseline.
- Face valence is a smile/frown reader, not an emotion detector, by design (see cheekSquint above). Glasses, masks,
  profile views and poor light reduce face visibility; the system degrades to context-only.
- Webcam auto-exposure flattens luminance, so lux comes from luminance + noise and needs per-venue calibration.
- dB SPL from a laptop mic after a one-offset calibration is approximate (±3 dB target), unweighted.
- Vosk's small model misses accented or fast speech; the grammar trades recall for "no transcript".
- macOS camera permission belongs to the launching app; Claude Code's shell cannot get it (run from Terminal).
