# Real-camera check after deploy (Mac or phone, ~15 minutes)

The cloud tests use a fake camera playing a portrait. This list checks the browser perception on a real camera and
microphone. Write down the numbers in **bold** and send them back (or paste them into `docs/hackohio/status-C.md`).

In the browser's DevTools console, `__valencePerception.status()` (fps, time to first observation, where the models
run, warnings), `__valencePerception.counters()` (governance) and `__valencePerception.last()` (the latest video and
audio observation: per person `face_valence`, `head_pitch`, `slump`, `fist_active`...) show what perception sees. They
hold scalars only. The `*_onsets` fields (fists, blinks, laughter) count edges in that one observation, so `last()`
nearly always shows 0: read those from the console's State vector instead.

On the iPhone the DevTools console comes from a Mac: turn on iPhone Settings → Safari → Advanced → Web Inspector, then
in Mac Safari open Develop → [iPhone] → the page. Without a Mac, read FPS and Video models in the System status panel
and leave `load_ms` / `video_ms` blank.

## 1. Load (Chrome on the Mac first, then Safari on the iPhone)

1. Open the deployed site over **https** at `/?source=browser`, or press **Use my camera (on-device)** in the welcome
   card (or **Live (camera + mic)** in the Source panel). Reloading the bare URL lands on the tour, so use
   `/?source=browser` for any step that reloads.
2. Allow the camera and the microphone, then **sit upright, still and neutral, facing the camera, for the first 20 s**
   before opening DevTools. Each person's face baseline (their first 20 s of visible face) and posture baseline (their
   first 20 s in view) are taken then and frozen; sections 2 and 3 read against them.
3. In DevTools → Network: every request goes to the site's own origin. The first load fetches `/mediapipe/wasm/*`
   (about 18 MB) and `/mediapipe/models/*` (about 22 MB); a reload revalidates them (each shows as 304 Not Modified,
   a few hundred bytes) rather than downloading them again. **Load time** = `status().load_ms`: from start to the
   first observation of either kind, so with the mic on it is usually the first 1 s audio block, before the vision
   models are up; also note roughly when `status().fps` first turns non-zero.
4. `status().video_backend` should read `worker/CPU` (the default delegate; `main/CPU` means the worker could not run
   MediaPipe: no module worker or OffscreenCanvas, or no WebGL in a worker, as on Safari before 17). **fps** =
   `status().fps` (target 10; note what you get) and **video_ms** = `status().video_ms`.
5. Console: no red errors, no "Content Security Policy" messages.

## 2. Face (sit 50–80 cm from the camera, face lit)

1. After the neutral start in 1.2, the face valence reads about 0.
2. Smile broadly for 5 s: the face valence goes clearly positive (> 0.3). Frown / lower the brows: it goes negative.
3. Nod down (look at the keyboard): `head_pitch` goes **negative**. Turn your head toward your left shoulder:
   `head_yaw` changes sign consistently; note which way.
4. Cover your face with a hand while your body stays in view: `face_visible` goes false (`face_valence` NaN) within
   about half a second. Covering the camera or leaving the frame instead removes the person from
   `last().video.persons`; away for more than 3 s, the Person number is released and the 20 s baselines start again
   when you come back, so sit upright and neutral for 20 s before section 3.

## 3. Body and hands

1. Slump down in the chair for > 3 s: **slump** turns on; sit up: it turns off. (The baseline is the upright start
   in 1.2.)
2. Look down at the keyboard and type for 15 s, sitting upright: slump must **stay off**. (On identical frames
   BlazePose's ears drop more than YOLO's when the head pitches down, so the keyboard-gaze rule protects less in the
   browser; note if it fires.)
3. Sit still for 30 s: `motion` should read below ~0.1 (the desktop edge reads ~0.03 still, ~0.1 slumped); wave or
   shift in the chair: it rises.
4. Head in both hands for > 2 s: **head in hands**.
5. Raise a clenched fist to chest height (not over the keyboard) for ~1 s: `fist_active` reads true while you hold
   it, and the State vector's "Clenched fists (5 min)" goes up by **one**. Typing on the keyboard must not count
   fists.
6. A second person sits next to you: **Person #2** appears (up to 3 people).

## 4. Room

1. Talk normally: speech probability high; the dB level follows the room.
2. Laugh out loud for 2–3 s: the State vector's "Laughter (15 min)" goes up by **one** (another needs a pause of
   ~6 s).
3. Turn a lamp off or on: the lux reading moves within ~10 s (10 s median).

## 5. Robustness and governance

1. Deny the microphone (site settings) and reload `/?source=browser`: the console asks once and shows "Camera and
   microphone blocked: allow them for this site…, then press Start", and neither device runs (it does not retry the
   camera alone after a refusal). Video-only ("no microphone" under Degraded) happens when the mic is missing or busy
   instead, e.g. no mic attached or another app holding it.
2. Switch to another tab for 10 s and back: video pauses ("tab hidden: video paused") and resumes; audio continues.
3. Governance panel: frames decoded − frames purged (the purge gap) stays 0 or 1 (one frame in flight when the tick
   reads the counters) and Violations stays 0; audio blocks purged grows by ~10 per second (~50 per 5 s tick).
4. Optional: unplug an external camera: "camera stopped".
5. Safari: start the camera in another tab (e.g. a second copy of the site) and come back: while the other tab has
   it, status shows "camera muted: video paused" (and "microphone muted") rather than frozen readings. Note whether
   it resumes on its own when you return.

## 6. GPU vs CPU (Mac, 3 minutes)

The default is the CPU delegate, where the browser matches the Python edge to ~1e-4 on identical frames. On
software WebGL the GPU delegate drifted more (face valence p95 0.06, landmarks ~1 px), and a real GPU has not been
measured. Open the console once as is and once with `delegate=GPU` added to the URL (`/?source=browser&delegate=GPU`;
`video_backend` should then read `worker/GPU`). For each, sit still for 30 s
and note `status().fps`, `video_backend`, and how much `face_valence` wobbles in `__valencePerception.last()` (a few
readings). If CPU is too slow (fps well under 10) and GPU looks as steady, say so: the default can switch.

## What to send back

`load_ms`, `fps`, `video_ms`, `video_backend` and the browser/device for each device tried (`load_ms` and `video_ms`
may be blank for a phone tested without Web Inspector), plus anything in the list that did not behave as described.
