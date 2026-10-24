# Real-camera check after deploy (Mac or phone, ~15 minutes)

The cloud tests use a fake camera playing a portrait. This list checks the browser perception on a real camera and
microphone. Write down the numbers in **bold** and send them back (or paste them into `docs/hackohio/status-C.md`).

In the browser's DevTools console, `__valencePerception.status()` (fps, load time, where the models run, warnings),
`__valencePerception.counters()` (governance) and `__valencePerception.last()` (the latest video and audio
observation: per person `face_valence`, `head_pitch`, `slump`, `fist_onsets`...) show what perception sees. They hold
scalars only.

## 1. Load (Chrome on the Mac first, then Safari on the iPhone)

1. Open the deployed site over **https** and switch the source to **IN-BROWSER** (or add `?source=browser`).
2. Allow the camera and the microphone.
3. In DevTools → Network: every request goes to the site's own origin. The first load fetches `/mediapipe/wasm/*`
   (about 18 MB) and `/mediapipe/models/*` (about 21 MB); a reload serves them from the cache.
   **Load time** = `status().load_ms`.
4. `status().video_backend` should read `worker/GPU` on a Mac (`worker/CPU` or `main/CPU` on a phone is fine).
   **fps** = `status().fps` (target 10; note what you get) and **video_ms** = `status().video_ms`.
5. Console: no red errors, no "Content Security Policy" messages.

## 2. Face (sit 50–80 cm from the camera, face lit)

1. Sit still and neutral for **20 s**: that is the calibration window, so the face valence reads about 0.
2. Smile broadly for 5 s: the face valence goes clearly positive (> 0.3). Frown / lower the brows: it goes negative.
3. Nod down (look at the keyboard): `head_pitch` goes **negative**. Turn your head toward your left shoulder:
   `head_yaw` changes sign consistently; note which way.
4. Cover the camera or leave: the face drops out (face_visible false) within half a second.

## 3. Body and hands

1. Sit upright for 20 s (posture calibration), then slump down in the chair for > 3 s: **slump** turns on; sit up:
   it turns off.
2. Head in both hands for > 2 s: **head in hands**.
3. Raise a clenched fist to chest height (not over the keyboard) for ~1 s: one **fist onset**. Typing on the
   keyboard must not count fists.
4. A second person sits next to you: **Person #2** appears (up to 3 people).

## 4. Room

1. Talk normally: speech probability high; the dB level follows the room.
2. Laugh out loud for 2–3 s: one **laughter onset** (another needs a pause of ~6 s).
3. Turn a lamp off or on: the lux reading moves within ~10 s (10 s median).

## 5. Robustness and governance

1. Deny the microphone (site settings) and reload: video still runs; status shows "no microphone".
2. Switch to another tab for 10 s and back: video pauses ("tab hidden: video paused") and resumes; audio continues.
3. Governance panel: frames decoded = frames purged at every tick; audio blocks purged grows by ~10 per second.
4. Optional: unplug an external camera: "camera stopped".

## What to send back

`load_ms`, `fps`, `video_ms`, `video_backend` and the browser/device for each device tried, plus anything in the list
that did not behave as described.
