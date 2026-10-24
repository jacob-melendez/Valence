# Browser perception observations on freely licensed clips

Features-only observation streams written by the **in-browser perception** (`frontend/src/perception`, the real
`createPerception()` in headless Chromium, fake camera playing the clip). Each line is one observation in
`valence.contracts.obs_to_dict` form, so `obs_from_dict` reads it:

- `VideoObs`: `PersonObs` fields only, emotion-v1 cues included. The browser-only `raw` extras are left out.
- `AudioObs`: the microphone played synthetic quiet room noise (pink, −50 dBFS), not the clip's sound. Ignore the
  audio lines for anything but the format.

There are no frames, images, landmarks or audio in these files.

| File | Clip | Source | License |
|---|---|---|---|
| `female.jsonl` | `head-pose-face-detection-female.mp4` (768x432, 12 fps, 134.6 s) | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) | CC BY 4.0 (the repository's LICENSE) |
| `male.jsonl` | `head-pose-face-detection-male.mp4` (768x432, 12 fps, 133.7 s) | same | CC BY 4.0 |

The clips show one person framed head and shoulders, turning and pitching their head, mostly with a neutral
expression; they are standing, not seated at a desk. Each file covers one pass of 132 s from the first processed
frame.

How they were made (from `frontend/`):

```
ffmpeg -i head-pose-face-detection-female.mp4 -r 10 -pix_fmt yuv420p female.y4m
node e2e/run.mjs --scenario female --video female.y4m --audio quiet.wav --seconds 140 \
  --record ../data/browser/female.jsonl --record-seconds 132
```

Settings: MediaPipe 0.10.35, CPU delegate in a Web Worker, faces on every frame, on a 4-core cloud container
without a GPU (blink counts depend on the face rate):

| File | Video obs | Rate (= face passes/s) | Face visible | Blinks | Spread (pstdev) of brow / eye / smile / yaw / pitch |
|---|---|---|---|---|---|
| `female.jsonl` | 909 in 132 s | 6.9 Hz | 97% | 22 | 0.14 / 0.14 / 0.15 / 12.4° / 10.5° |
| `male.jsonl` | 900 in 132 s | 6.8 Hz | 96% | 20 | 0.11 / 0.13 / 0.20 / 16.9° / 6.5° |

(The people in the clips deliberately turn and nod their heads, so the yaw and pitch spreads are larger than at a
desk.)
