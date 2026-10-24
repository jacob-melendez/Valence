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
node e2e/run-all.mjs                      # every scenario + cross checks (roll sign, two people) -> .out/summary.json
node e2e/run.mjs                          # one person, 25 s, CSP from the brief
node e2e/run.mjs --scenario two           # two people side by side
node e2e/run.mjs --scenario roll_cw       # portrait rotated 15 degrees clockwise (head roll sign check)
node e2e/run.mjs --csp deploy             # the CSP the Cloudflare deploy writes (scripts/security-headers.mjs)
node e2e/run.mjs --delegate GPU           # the GPU delegate (on headless Chromium's software WebGL: < 1 fps, fails)
node e2e/run.mjs --deny camera            # the user refuses the camera (getUserMedia: NotAllowedError): audio alone
node e2e/run.mjs --deny mic               # the user refuses the mic: video alone
node e2e/run.mjs --python-cadence         # faces on every other frame, as the Python edge
node e2e/run.mjs --main-thread            # no Web Worker: the video models on the page's main thread
node e2e/run.mjs --no-build               # reuse e2e/dist (rebuild after changing src/perception)
```

`run.mjs` makes sure the MediaPipe assets exist (`scripts/mediapipe-assets.mjs`), builds the media with
`make_media.py` (repo venv: numpy, Pillow, imageio-ffmpeg) if `.media/` is missing, builds this standalone page with
`vite.config.ts` (it bundles only the perception module, not the console) and serves it with the CSP header.

From a fresh clone it needs, besides `npm install`:
- **Playwright and its Chromium**, which are not frontend dependencies: `npm i --no-save playwright` then
  `npx playwright install chromium` (or a global install on `NODE_PATH`);
- the first time, the repo's `.venv` with **imageio-ffmpeg** (not in `requirements.txt`): `.venv/bin/pip install
  imageio-ffmpeg`;
- internet access the first time, for the MediaPipe models.

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

## The whole console on the test media

`scripts/check-static.mjs` (the deploy check) can feed the same files to the production build:

```
npm run build && node scripts/check-static.mjs --path "/?source=browser" --fake-video e2e/.media/one.y4m \
  --fake-audio e2e/.media/speech.wav --wait 45000 --expect "Top drivers by layer" --full --screenshot out.png
```

`console-in-browser.jpg` is that run (15:14 ET, Oct 24): the portrait tracked as Person #1, a prediction with its
layers and template text, the emotion map, and the governance panel with frames decoded 236 / purged 235 (one in
flight), 428 audio blocks purged, 0 violations; 0 off-origin requests and 0 CSP violations.

## Soak

`node e2e/soak.mjs --minutes 15` runs the real perception for 15 minutes and samples the RSS of every Chromium process
every 30 s. Result at 15:50 ET, Oct 24 (worker, CPU delegate):
- 6.7 fps sustained;
- 5,646 frames decoded and 5,645 purged (one in flight), 8,981 audio chunks purged, 0 violations, 0 page errors;
- 3 main-thread long tasks (234 ms in total);
- total Chromium RSS in a sawtooth between 1.60 and 1.65 GB, falling back after collections (at 4.5 and 12 min):
  no leak trend.
