# Hack OHI/O 2026: plan (written before the event; intent only, no code)

24 hours, virtual. Two tracks, built by parallel sessions, all logged in `docs/HACKOHIO_LOG.md`. The commit-subject
tags `[web]`, `[emotion]`, `[eval]`, `[infra]`, `[docs]` and `[pitch]` are required.

## Goals and definition of done

| Track | Done means |
|---|---|
| **Valence Web** `[web]` | A hosted link where anyone can run Valence live in their browser. Camera and mic are processed in-page; no request leaves except loading the site's own static files (provable in DevTools and by a strict CSP). The console shows labels, the four layers, explanation text, the venue baseline and governance counters from the in-browser engine. Features and model outputs match the Python reference on golden windows. |
| **Emotion map** `[emotion]` | Continuous valence and arousal output, plus about 10 named states mapped onto that plane. Trained and evaluated (simulator, session splits), drawn in the console as an affect map with a trail. First real labeled evaluation: consented self-report sessions (guaranteed), and a public dataset if access arrives (stretch). Honest model-card section. |

**Out of scope for the 24 hours:** a cloud server receiving video, Python edge changes beyond what the tracks need,
and new hardware.

## Architecture: Valence Web

The console (`frontend/`) already consumes backend-shaped messages (`hello`, `tracks`, `state_vector`,
`prediction`, `explanation`, `governance`, `status`) from either the WebSocket or a mock engine. Valence Web adds a
third source, **in-browser**, that produces the same messages, so every panel works unchanged.

```
camera/mic (getUserMedia)
  -> perception (MediaPipe Tasks Web: FaceLandmarker with blendshapes, GestureRecognizer, PoseLandmarker lite;
     AudioClassifier YAMNet; WebAudio RMS for level; frame luminance for light)
  -> frames dropped after each step (canvas cleared; no recording APIs)
  -> aggregator: 5 s TickRecords, participants, venue baseline, near-field gate      (TS port of aggregator.py subset)
  -> rolling window -> features (123, same definitions)                              (TS port of features/compute.py)
  -> LightGBM M2 + M0 from JSON dumps, temperature softmax, path-dependent TreeSHAP  (TS)
  -> smoother + mask override, layers, describe() text, template narration           (TS ports)
  -> store -> existing panels; hash-chained audit in memory (WebCrypto SHA-256)
```

- **The room-relative design pays off here.** Browser audio levels and camera luminance are uncalibrated, but the
  served model only sees noise and light *relative to the venue baseline*, so no absolute calibration is needed.
- **Known gaps, to say out loud:**
  - Pose comes from MediaPipe Pose, not YOLO, so slump and motion need a quick side-by-side check against the
    Python edge on the Mac.
  - Compliment keywords (Vosk) are left out at first, so the count stays 0.
  - Narration uses the template; the LLM narrator stays desktop-only.
- **Parity:** a Python exporter writes golden windows (inputs, the 123 features, raw scores, probabilities,
  `pred_contrib`) from scenario replays. TS tests must match to 1e-6 for features and 1e-5 for scores and SHAP.
- **Hosting:** a static build on Vercel or Cloudflare Pages.
  - MediaPipe WASM and model files are self-hosted (no third-party CDN).
  - CSP: `connect-src 'self'`.
  - Model JSON is gzipped and lazy-loaded.

## Architecture: emotion map

- **Valence–arousal anchors** per state, on the circumplex. These are initial values, recorded in the model card as
  assumptions:
  - Focused (+0.1, +0.3), Collaborative / Happy (+0.7, +0.5), Calm (+0.4, −0.4);
  - Bored (−0.3, −0.5), Exhausted (−0.4, −0.7);
  - Confused (−0.2, +0.3), Frustrated (−0.6, +0.6), Tense (−0.4, +0.5), Anxious (−0.5, +0.6),
    Overwhelmed (−0.6, +0.8).

  States that share a region (Frustrated / Tense / Anxious) are told apart by context and behavior, not by
  valence–arousal alone.
- **New sensing for the new states** (the browser gives all 52 blendshapes and head pose for free):
  - brow lowering (confusion cue);
  - blink rate and eye openness (fatigue, boredom);
  - head yaw/pitch away from the screen (disengagement);
  - mouth frown and smile, separately.

  All of them are added to the simulator, the features and the browser perception. The Python edge follows after
  the event.
- **Models:**
  - the 10-state classifier (LightGBM, session splits, modality dropout as today);
  - two LightGBM regressors for valence and arousal;
  - calibration;
  - the existing 4-state and 6-state models kept as-is for the desktop demo.
- **Real data:**
  - *Guaranteed:* consented self-report sessions. Every 2 minutes the user taps a valence–arousal grid in the
    console; only the features and the taps are stored, never media.
  - *Stretch:* a public engagement dataset (e.g. DAiSEE: boredom, confusion, engagement, frustration) if access is
    granted. It is 10 s webcam clips, so it tests short-term face cues, not the 15-minute context model.

## Sessions and ownership (no two sessions edit the same files)

| Session | Owns | Delivers |
|---|---|---|
| **Cloud A: web engine** | `frontend/src/engine/**` (new), the console's source switch, `scripts/export_web_golden.py`, `frontend/src/test/engine*`, deploy config | TS ports (aggregator subset, features, model runtime, TreeSHAP, smoother, describe, template), golden parity tests, the "in-browser" source, hosting |
| **Mac A: browser perception** | `frontend/src/perception/**` (new) | MediaPipe Tasks Web perception producing observations for the engine; real-camera checks; side-by-side feature comparison with the Python edge; performance (fps, load size); self-report sessions with the user |
| **Cloud B: emotion map** | `valence/sim/**`, `valence/features/**`, `valence/models/**`, `artifacts/emotion/` (new), `valence/eval/**`, `tests/test_emotion*`, model JSON export | Valence–arousal anchors, new states and face features in the simulator, the 10-state model plus regressors, evaluation, JSON artifacts for the web engine, the model-card section |
| **Cloud #1 (coordinator)** | `docs/HACKOHIO_LOG.md`, reviews, the integration gate | Log entries every 2 h, merges, full test runs, the affect-map panel spec, the pitch |

The new face features change the feature list, so Cloud B publishes the list first (hour 3). Cloud A ports features
against that list. The 6 states in today's served models stay untouched until the new model passes its gates.

## Timeline (H = hours from the start)

| Hours | Valence Web | Emotion map | Coordinator / you |
|---|---|---|---|
| H0–1 | Sessions start; skeleton dirs | Anchors + new-state behaviors drafted | Log entry; confirm the start time |
| H1–6 | Features + model runtime in TS, golden exporter, parity tests (Cloud A). MediaPipe perception running on the Mac (Mac A) | New face features (spec at H3) + simulator states; corpus | First integration check |
| H6–12 | "In-browser" source wired into the console; venue baseline in the browser; first end-to-end run on the Mac | 10-state + valence/arousal training; evaluation; JSON export | Self-report session 1 (you, ~30 min) |
| H12–17 | Deploy (static host, CSP, self-hosted assets); performance numbers | Affect-map data in messages; model card; real-data evaluation on self-reports | **Your sleep window**: long jobs run |
| H17–21 | Polish, tests, fallbacks (replay mode in the browser) | Stretch: public dataset; error analysis | Self-report session 2; demo recording |
| H21–24 | Freeze at H22 | Freeze at H22 | `make hacklog`, final tag `hackohio-2026-final`, pitch, submit |

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| Browser pose features differ from YOLO's | Side-by-side check; if they drift, drop slump/motion from the browser model and say so |
| TreeSHAP port runs late | Saabas contributions (path attribution) for explanations, labeled as approximate |
| Model JSON too large | Prune to the M2 + M0 trees used; quantize thresholds; lazy load |
| MediaPipe audio in browser unstable | Level (RMS) only; laughter count 0 |
| New states don't separate in evaluation | Keep them in the valence–arousal map only; ship the classifier with the states that pass |
| Public dataset access doesn't arrive | Self-report evaluation only, stated honestly |

## Session prompts (paste at the start)

**Cloud A: web engine.** "You are building Valence Web on jacob-melendez/Valence, branch
claude/happy-ride-7w3ng3, during Hack OHI/O (24 h). Read docs/HACKOHIO_PLAN.md and docs/HACKOHIO_LOG.md first.
Every commit subject starts with `[web]` or `[eval]`. You own `frontend/src/engine/**`, the console's source
switch, `scripts/export_web_golden.py`, engine tests and the deploy config. Build, in TypeScript:
- the aggregator subset, rolling window and 123-feature function, ported from `valence/runtime/aggregator.py` and
  `valence/features/compute.py`;
- LightGBM M2/M0 evaluation from `booster.dump_model()` JSON, with temperature softmax;
- path-dependent TreeSHAP;
- PersonSmoother and the mask override, ported from `valence/models/predictor.py` and `served.py`;
- describe() and template text, ported from `valence/explain/describe.py` and `template.py`;
- a hash-chained in-memory audit.

Prove parity against golden windows exported from Python: features to 1e-6, scores and SHAP to 1e-5. Then add an
'in-browser' source that emits the same messages the WebSocket does, and deploy statically with self-hosted
assets and CSP `connect-src 'self'`. Push small verified commits; pull --rebase first."

**Mac A: browser perception.** "You are building the browser perception for Valence Web during Hack OHI/O. Read
docs/HACKOHIO_PLAN.md. Commit subjects start with `[web]`. You own `frontend/src/perception/**`:
- MediaPipe Tasks Web, self-hosted: FaceLandmarker with blendshapes and the transformation matrix,
  GestureRecognizer, PoseLandmarker lite, AudioClassifier YAMNet;
- WebAudio RMS level and frame luminance;
- observations matching `valence/contracts.py` PersonObs/VideoObs/AudioObs semantics, plus the new face features
  Cloud B specifies;
- no frame or audio is ever kept, recorded or sent.

Check on the real camera next to the Python edge (`run_demo.sh`), comparing slump, motion, face valence and fists
per tick. Report fps and load size, and run the self-report sessions with the user."

**Cloud B: emotion map.** "You are building Valence's emotion map during Hack OHI/O. Read docs/HACKOHIO_PLAN.md.
Commit subjects start with `[emotion]` or `[eval]`. You own `valence/sim/**`, `valence/features/**`,
`valence/models/**`, `valence/eval/**`, `artifacts/emotion/` and `tests/test_emotion*`. Steps:
1. By hour 3, publish the new face-feature list (brow lowering, blink rate, eye openness, head yaw/pitch, frown,
   smile) and add it to the simulator and features.
2. Add Bored, Confused, Calm and Anxious, with behaviors and valence–arousal anchors (plan table; record them as
   assumptions).
3. Train the 10-state LightGBM plus valence and arousal regressors with session splits; evaluate, including
   calibration, the per-state confusion matrix and valence–arousal error.
4. Export JSON for the web engine.
5. Write the model-card section, honestly.

Never change the served desktop models in `artifacts/` or `artifacts/all/`."
