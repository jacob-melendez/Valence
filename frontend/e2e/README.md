# Perception end-to-end test (headless Chromium, fake camera and mic)

Runs the real in-browser perception (`frontend/src/perception`, MediaPipe Tasks Web) in Playwright's Chromium with a
fake camera and microphone fed from files, behind a strict CSP, and checks that:

- faces are detected and observations flow (video ~10 Hz, audio 1 Hz), with all 52 blendshapes;
- YAMNet hears speech in the speech clip;
- every frame taken is purged (`frames_decoded == frames_purged`);
- **no request leaves the origin**, **nothing violates the CSP** and the page throws no error.

It also records fps, model load time and download size in `.out/report-<scenario>.json`.

```
cd frontend
node e2e/run.mjs                          # one person, 25 s, CSP from the brief
node e2e/run.mjs --scenario two           # two people side by side
node e2e/run.mjs --scenario roll_cw       # portrait rotated 15 degrees clockwise (head roll sign check)
node e2e/run.mjs --csp deploy             # the CSP the Cloudflare deploy writes (scripts/security-headers.mjs)
node e2e/run.mjs --delegate CPU           # force the CPU delegate
```

`run.mjs` makes sure the MediaPipe assets exist (`scripts/mediapipe-assets.mjs`), builds the media with
`make_media.py` (repo venv: numpy, Pillow, imageio-ffmpeg) if `.media/` is missing, builds this standalone page with
`vite.config.ts` (it bundles only the perception module, not the console) and serves it with the CSP header.

The page under test is `index.html` + `harness.ts`: it opens the camera and mic with `openMediaStream()` exactly as
the console does, starts `createPerception()`, and exposes compact summaries on `window.__e2e`.

## Test media and licenses

Nobody is recorded. The fake devices play files derived from:

| File | Source | License |
|---|---|---|
| `media/portrait.jpg` | NASA portrait of astronaut Eileen Collins, as shipped in scikit-image (`skimage.data.astronaut()`), re-encoded as JPEG | Public domain ("No known copyright restrictions, released into the public domain": `skimage.data.astronaut` docstring; from the NASA Great Images database, <https://flic.kr/p/r9qvLn>) |
| `media/speech.ogg` | LibriSpeech `5703-47212-0000` ("The Ashiel Mystery", read by Garth Comira), as distributed in librosa's example data (`librosa/data`, `audio/5703-47212-0000.ogg`) | CC BY 4.0 (LibriSpeech, <https://www.openslr.org/12/>) |

The scenes (`.media/*.y4m`, 1280x720, 10 fps, 3 s loops) place the portrait in a plain room: one person drifting a
few pixels, two people side by side (the right one mirrored), and the one-person scene rotated 15 degrees either way.
