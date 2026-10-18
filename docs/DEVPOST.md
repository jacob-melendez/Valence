# Valence: Devpost submission draft

TrojanHacks @ USC 2026, AI Data Governance track.

Paste the sections between the two horizontal rules into Devpost. The **claim map** at the bottom ties every
number to a claim ID and its source; do not paste it. Every model accuracy number in the body is on simulated
data and says so. Keep that wording if you edit.

---

## Tagline

The environment carries the signal, not the face: context-aware emotional-state inference that keeps no raw media
and explains every call.

## Inspiration

Emotion AI mostly reads faces, and faces fail in two predictable ways. A forced, polite smile reads as happy while
the frustration underneath is missed. A furrowed "resting focus face" reads as anger, and head in hands can mean
weeping, laughing or deep concentration. Context resolves both: how loud the room is, how long since a break,
whether someone is clenching their fists or slumping, whether people around them are laughing.

Face-first pipelines also tend to store frames and identity data and return labels with no reasons. We wanted the
opposite: read context over time, keep none of the raw media, and explain every prediction.

The dataset we started from showed why this needs care. Trained on the original mock rows, context columns alone
scored 100% accuracy and the face alone 66.7%. That looks like a win for our idea, but it meant the labels were a
lookup table with no time structure, so it could not test a 15-minute window at all. We replaced it with our own
simulator.

## What it does

Valence watches a webcam and microphone on a laptop and, every 5 seconds, labels each participant's likely state
(Focused, Frustrated_Masked, Exhausted or Collaborative_Happy in the core model) from a 15-minute rolling window of
context:

- **Ambient Physics:** noise in dB and light in lux, read relative to a ~90-second baseline of the venue itself.
  Together with training on simulated venues from 40 to 75 dB, this makes a loud hall less likely to be read as a
  negative state (a modest gain, on simulated data).
- **Time Dynamics:** hours since a break, time of day.
- **Somatic and Social:** fist clenches, slumping, head in hands, motion, laughter, a small set of spoken
  compliments, number of people.
- **Facial Surface:** face valence from blendshapes, as one input among four.

Alongside the context model, a face-only model runs as a shadow. When the face clearly reads happy but the context
model is at least 70% confident in any other state (Focused included), and the face features are what pull toward
happy, the console shows **MASK OVERRIDE: context corrected the facial mask**. A warm-up and a 2-minute block keep
that rule conservative.

Every prediction comes with:
- its top drivers, grouped into the four layers with signed contributions, so the face's weight is always visible;
- a one-to-three-sentence explanation written by a small LLM running on the laptop, with every number checked
  against the prediction's fact sheet and a deterministic template as the fallback;
- a counterfactual endpoint ("what if they had just had a break?").

**Results on simulated data** (45 held-out simulator sessions, untuned noise. The simulator encodes our behavioral
assumptions, such as forced smiles and context that lags the state, so part of the context model's lead is by
construction; these results test the architecture, not real-world accuracy):

- Masked frustration caught **68.0%** of the time vs **11.2%** for the face-only model.
- Focused people misread as a negative state: **17.9%** vs **56.3%**.
- Macro-F1 **0.797** vs **0.422**.
- With the face hidden on every tick: macro-F1 **0.678** vs **0.121**. With face readings degraded as at 10% light:
  **0.679** vs **0.252**.
- Under simulated hall noise up to +75 dB, the Focused-misread rate moves from 17.7% to 18.7% for the served model
  vs 20.8% to 28.1% for our previous model: a real but modest gain.
- Calibration error (ECE) 0.013, 0.0045 after temperature scaling.

The console (React) shows each person's box, the context label next to the face-only label, the explanation, the
layer drivers, governance counters, a provenance badge (REPLAY, SIMULATED: <fields>, or LIVE), and an evaluation
panel with the metrics and figures. The backend never receives camera pixels.

## How we built it

**Edge perception (Python, Apple Silicon)**
- **YOLO26n-pose + ByteTrack** (Ultralytics, on MPS) for people, tracking and posture (slump, head in hands,
  wrists). 18.4 ms p50 per frame on sample images on an M3 Max.
- **MediaPipe** FaceLandmarker **blendshapes** for face valence, calibrated per person to their own neutral face,
  and the MediaPipe **GestureRecognizer** for fist clenches, matched to people through YOLO wrist keypoints.
- **YAMNet** (through MediaPipe's audio classifier, allowlisted classes) for laughter and speech.
- **Vosk** small English model with a restricted grammar of 8 phrases plus `[unk]` for compliments, so a free
  transcript never exists. 0.84 ms p50 per 100 ms audio block.
- Calibrated sound level (dB) and a light estimate (lux) from the camera.
- The camera is grabbed every frame but decoded only at 10 Hz, so about two thirds of a 30 fps stream is never
  decoded. The whole edge runner takes 45-57 ms per frame at 10 Hz on an M3 Max.

**Purge gate.** Every decoded frame and captured audio block, and the derived copies handed to the gate (RGB,
grayscale, head crops, resampled audio, int16 PCM), is zero-filled in place right after feature extraction and
tracked with a weak reference. Any buffer still alive at the end of the 5-second tick counts as a
`PURGE_VIOLATION`. Short-lived numpy temporaries, MediaPipe's own image and audio copies and mic blocks dropped on
queue overflow are freed without zeroing (`docs/GOVERNANCE.md` §3.5).

**State vector and models**
- An aggregator turns per-frame observations into one numeric tick per participant every 5 s, kept in a 180-tick
  (15-minute) rolling window in RAM.
- **Simulator v2** generates training data with time structure, overlapping states, noise, missing face or light,
  and a per-session venue noise level (40-75 dB). We fit its per-channel noise levels to summary statistics of
  one real capture run; the global difficulty knob (`noise_scale` 1.0) was never tuned toward the results.
- **LightGBM** context model (M2, 101 of 123 computed features, trained with face-hidden and face-degraded
  copies), plus a face-only LightGBM model (M0) as the shadow. Temperature-scaled probabilities.
- A model ladder (face snapshot, face history, context snapshot, context history without face, served) and a GRU
  with attention as a benchmark. We serve LightGBM because **TreeSHAP** (LightGBM `pred_contrib`) gives exact
  per-feature attributions for the explanations.
- Content-hashed model ids (`m2_lgbm@8a78efc7`), with rollback to the previous model through one environment
  variable.

**Narrator**
- **Qwen3-1.7B, 4-bit, on MLX** (`mlx-lm`), served locally on 127.0.0.1.
- A **LoRA** adapter (rank 8, last 16 of 28 blocks, 0.29% of parameters) trained on the laptop (about 48 minutes
  on AC power; the logged run took 66 minutes after the laptop dropped to battery) from deterministic grammar
  targets, with no LLM-written training text.
- A **validator** checks length, sentence count, every number against the fact sheet at its precision, banned
  demographic, appearance and diagnostic terms, and mask language only when the override fired. Anything that
  does not pass within 2.5 s falls back to a template.
- We decided to ship the LoRA with a rule written before the A/B test. Against prompt-only on 300 test fact sheets
  with YOLO running: first-try pass 93.3% vs 75.3%, template fallback 1.3% vs 10.3%, p95 latency 1904 ms.

**Serving and console**
- **FastAPI + WebSocket** API (uvicorn, Pydantic contracts) with live, replay and scripted-scenario modes, operator
  simulation overlays (face hidden, low light), counterfactuals, and governance endpoints.
- **React** console (TypeScript, Vite, TanStack, Tailwind, Recharts), first generated with Lovable and then wired to
  the contracts.
- 586 tests pass (20 skipped in the cloud container), including 23 governance tests.

## Governance

We built Valence for the AI Data Governance track, so the data lifecycle is part of the design, not an
afterthought. Details, with code references and known gaps, are in `docs/GOVERNANCE.md`.

- **Never stored:** video frames, audio blocks, transcripts, pose keypoints, face images or embeddings.
  `GET /api/retention` reports this policy, and the purge gate, a static scan and a log filter back it.
- **What is written to disk.** By default the live pipeline writes one file, the audit log below. Self-report
  labels, if used, go to a local file deleted at the next startup after 24 hours (not yet at session end). Our
  real-data checks use features-only recordings of the builder (boxes, face valence, posture, motion, sound and
  light levels; no media) committed in `data/real/`; early commits also hold feature rows of background people,
  which we have not rewritten out of the history (`docs/GOVERNANCE.md` §8 gap 8).
- **Purge gate with counters.** Frames decoded, frames and audio blocks purged, violations and live raw buffers are
  published every 5-second tick, and a growing purge gap marks the system as degraded. Limits: copies inside native
  libraries (GPU allocator, AVFoundation, MediaPipe, PortAudio) cannot be reached or zeroed, and in replay mode the
  counters prove nothing.
- **Static scan:** no file writes at all in the perception code, and no binary or media-save calls in the capture
  (edge) code, which writes only text: optional features-only JSONL recordings and calibration scalars. It is a
  lint against accidents, not a proof.
- **Log filter against raw media:** a filter on every logger and handler present at startup drops records whose
  message or arguments contain arrays or bytes (handlers added later are not covered).
- **Tamper-evident audit log.** It records decisions, not data: label, confidence, whether the override fired,
  provenance, and layer sums. Each entry carries the SHA-256 of the previous one, and `GET /api/governance/verify`
  recomputes the chain. It is deleted at session end; after a crash, the next startup deletes it once it is older
  than 24 hours. Tamper-evident, not tamper-proof.
- **Provenance everywhere.** Every prediction carries its source (live, replay, sim) and which fields were
  simulated. The console shows a badge, and operator simulation actions are written to the audit chain.
- **Kill switch:** pause releases the camera and microphone; stop also clears all windows and history.
- **Room-only mode** gives one aggregate prediction and no per-person predictions. It reduces exposure, but
  perception still runs per person, so it still processes biometric data.
- **No recorded person in training:** the models are trained only on simulator output.
- **Positioning.** The EU AI Act Art. 5(1)(f) prohibits emotion recognition in workplaces and education
  institutions, except for medical or safety reasons, and there is no consent exception. Valence is an emotion
  recognition system in the Act's sense (Art. 3(39)); outside those settings it would be high-risk (Annex III
  1(c)), with a duty to inform the people exposed (Art. 50(3)). The Act's territorial scope does not reach a demo
  at a US event, so we use it as our design bar. Valence is a research prototype for consented self-monitoring and
  personal wellbeing outside workplace and education settings. It is not for employers, schools, proctoring or
  productivity scoring. A hackathon on a university campus is close to the education setting, so at the event it
  analyzes only the builder and volunteers who opt in. That opt-in is a data-protection courtesy, not an
  Art. 5(1)(f) exemption: if the prohibition applied, consent would not lift it. This is design framing for a
  hackathon prototype, not legal advice.

## Challenges we ran into

- **The data could not test the idea.** The original mock dataset was perfectly separable from context and had no
  time order. We wrote a simulator with time structure, overlap, noise and missing modalities, and published its
  assumptions.
- **Real data caught real bugs.** Typing while looking down at the keyboard was flagged as a slump 93% of the time
  under our first nose-only rule; the new rule (nose and ears against a personal baseline) brought that to 0%. An
  early "fists make it Frustrated" result turned out to be a replay-clock bug.
- **Context is slow at first.** In the first 30 seconds of a new emotional state, the context model is less
  accurate than the face alone (32.8% vs 44.3%, simulated). It overtakes after about 30 s. We added warm-up and
  hysteresis to the label and the override instead of hiding it.
- **Loud venues.** A hackathon hall is loud for everyone. We trained on simulated venues of 40-75 dB and added a
  ~90-second room baseline with room-relative features, then measured the effect under simulated hall noise: real
  but modest, and most of it comes from the venue training (at +75 dB, 19.9% with absolute dB vs 18.7%
  room-relative, against 28.1% for the previous model).
- **A small LLM that does not make things up.** Prompt-only, the narrator invented numbers in 17.1% of raw attempts.
  We trained a LoRA on grammar-generated targets and put a validator in front of everything it says.
- **Latency on one laptop.** YOLO, MediaPipe, YAMNet, Vosk and the LLM share one GPU. On battery the narrator decoded
  at 10-17 tokens/s and every request timed out to the template, so the demo runs plugged in.
- **Honest purging in Python.** `del` removes a name, not memory. We zero-fill the array that owns the memory and
  track it with a weak reference, and we document what native libraries keep out of our reach.

## Accomplishments that we're proud of

- On simulated data, the context model catches masked frustration about six times as often as the face alone
  (68.0% vs 11.2%) and cuts "resting focus face" misreads from 56.3% to 17.9%.
- With the face hidden on every tick it keeps a macro-F1 of 0.678 while the face-only model drops to 0.121
  (simulated, face channel only; partly by construction, since the served model was trained on face-hidden copies).
- The whole perception stack runs on one laptop at 10 Hz (45-57 ms per frame). We found that MediaPipe embeds usage
  telemetry to Google with no opt-out, so the launcher runs the camera/mic processes in a macOS sandbox that denies all
  outbound network (loopback only); `status.offline` reports a live outbound probe, not just a setting.
- The narrator's LoRA passes the faithfulness validator on the first try 93.3% of the time, and its raw attempts
  invented no numbers and made no false mask claims, at a p95 of 1904 ms with YOLO running.
- A governance story we can demonstrate: 23 governance tests, a zero-violation persistence scan, a hash-chained
  audit log that detects a one-character edit, and an audit file that disappears at session end.
- Negative controls on real features-only captures (one person at one desk, about 25 ticks each): quiet typing
  reads Focused on 25 of 25 ticks (in-sample, since the simulator was fit to that capture), cued fist clenches alone
  still read Focused on 24 of 25, and typing while looking down, which was not used for the fit, reads Focused on
  25 of 25.

## What we learned

- Context over time carries information a face snapshot cannot. On simulated data the context model climbs from
  0.649 macro-F1 with no history to 0.791 at five minutes and then plateaus, while the face-only model stays flat.
  Even without the face, context history beats the face snapshot.
- Mock data that makes your idea look perfect is a warning sign.
- Write the ship rule down before the A/B test. It made the LoRA decision easy to defend.
- Real captures are small but sharp. A few minutes of one person found a 93% false-positive rule that the simulator
  never would have.
- Governance claims need precise wording: "zero-filled" not "erased", "tamper-evident" not "tamper-proof",
  "pseudonymous" not "anonymous", "the backend never receives pixels" not "nothing is ever seen".

## What's next for Valence

- Collect consented, labeled real data from more people, and re-evaluate on it before claiming anything about
  real-world accuracy.
- Validate across demographics; so far we have only tested stability when the face is missing or degraded.
- Confirm the live mask flip on real data, and rehearse the full demo with Wi-Fi off.
- A 60-minute memory soak, and a socket check on the API process as well as perception.
- Close the open governance gaps listed in `docs/GOVERNANCE.md` §8, such as deleting self-report labels at session
  end and a race in the purge counters.
- Bring the 6-state model (adds Overwhelmed and Tense; macro-F1 0.767 on simulated data under its own scoring)
  through the same evaluation as the core model.

## Built with

python, fastapi, uvicorn, websockets, pydantic, numpy, pandas, scikit-learn, lightgbm, shap, ultralytics, yolo,
bytetrack, pytorch, opencv, mediapipe, yamnet, vosk, sounddevice, mlx, mlx-lm, qwen, lora, matplotlib,
typescript, react, vite, tanstack, tailwindcss, recharts, lovable, apple-silicon

---

## Claim map (do not paste)

All IDs refer to the claims inventory used for `docs/PITCH.md` (sources: `reports/metrics.json`,
`docs/RESULTS.md`, `docs/NARRATOR.md`, `docs/BENCHMARKS.md`, `docs/GOVERNANCE.md`, `artifacts/*/model_meta.json`).

| Number or statement in the body | ID | Caveat carried in the text |
|---|---|---|
| Mock data: context 100%, face 66.7% | LM1 | LightGBM accuracy on 600 held-out regenerated legacy rows; not a Valence model |
| ~90-second venue baseline | V5 | Body describes the mechanism only; not yet rehearsed in the real venue |
| MASK OVERRIDE rule, warm-up, 2-minute block | M1 | Thresholds tuned on simulated data |
| Explanation validator, template fallback, 2.5 s | P11 | The validator is the faithfulness authority |
| 68.0% vs 11.2% | H2 | Simulated |
| 17.9% vs 56.3% | H3 | Simulated |
| 0.797 vs 0.422 | H1 | 45 held-out simulated sessions, untuned noise; partly by construction (stated) |
| 0.678 vs 0.121 | R1 | Face channel only; not fairness; partly by construction (stated) |
| 0.679 vs 0.252 ("as at 10% light") | R3 | Face readings only; lux unchanged |
| 17.7 → 18.7% vs 20.8 → 28.1% | V1 | Simulated hall noise; modest; gap mostly venue-varied training, m2_abs 19.9% at +75 dB (stated) |
| ECE 0.013 → 0.0045 | C1 | Simulated |
| Backend never receives pixels | G11 | The console's live preview is browser-local |
| YOLO26n-pose 18.4 ms p50 | P2 | M3 Max, sample images |
| 8 phrases, 0.84 ms p50 | P4 | M3 Max |
| Two thirds never decoded | P3 | Assumes 30 fps |
| 45-57 ms per frame at 10 Hz | P1 | M3 Max, plugged in |
| Purge gate, violations | G1 | Native-library copies not zeroed (stated) |
| Venue 40-75 dB in the simulator | V6 | Modeling assumption |
| Noise scales fit to one capture run | D10, L5 | Makes the typing control in-sample |
| 101 of 123 features, modality dropout | L5 | Simulated training only |
| GRU benchmark; LightGBM served for TreeSHAP | L7 | Benchmark only |
| `m2_lgbm@8a78efc7`, one-variable rollback | L5, L6 | |
| LoRA rank 8, last 16 of 28 blocks, 0.29%, ~48 min on AC (logged run 66 min on battery) | P10 | Grammar targets, simulated fact sheets |
| 93.3% vs 75.3%, 1.3% vs 10.3%, p95 1904 ms, pre-stated rule | P6, P8, P9 | 300 sheets, ±3 points; AC power |
| 586 passed, 20 skipped; 23 governance tests | G6 | Cloud container; the 20 skips need Mac-only packages (4) or downloaded models/tokenizer (16) |
| Never-stored list, `/api/retention` | G10 | Declared policy backed by G1, G5, G7 |
| What is written to disk: audit log, labels, `data/real/` captures | G4, G8 | GOVERNANCE.md §2 rows 13-15, §8 gaps 7-8 (stated) |
| Counters every 5 s; replay counters prove nothing | G2 | Stated |
| Zero violations in the scan | G5 | Lint, not proof (stated) |
| Log filter | G7 | Handlers added after startup not filtered |
| Audit chain, verify, deletion, 24 h | G3, G4 | Tamper-evident (stated) |
| Provenance badges | G12 | |
| Kill switch | G13 | Pause keeps history in RAM |
| Room-only mode | G9 | Still processes biometric data (stated) |
| No recorded person in training | G14 | Simulator scales fit to one capture (stated in "How we built it") |
| EU AI Act positioning | G15 | Design framing, not legal advice (stated) |
| Slump 93% → 0% | D6 | One person, 3-min segments |
| Fists replay-clock bug | D2 | |
| First 30 s: 32.8% vs 44.3% | E1 | Simulated, per-window accuracy |
| Narrator 17.1% invented numbers (prompt-only) | P7 | Raw attempts, measured by the validator |
| Battery: 10-17 tok/s, all timeouts | P8 | |
| Camera/mic processes sandboxed, outbound denied; MediaPipe telemetry found and blocked | P12, BENCHMARKS live smoke | verified with EPERM on the Mac; Wi-Fi-off rehearsal not done |
| Typing 25/25, fists 24/25, keyboard gaze 25/25 | D1, D2, D3 | One person, ~25 ticks; typing in-sample (stated) |
| 0.649 → 0.791 at 5 min, plateau; face flat | A1 | Simulated, macro-F1 by history |
| Context history without face beats face snapshot | A2 | Simulated |
| 6-state macro-F1 0.767 | L8 | Own scoring; compare with 0.791, not 0.797 |

**Do not add** to the Devpost text: a live MASK OVERRIDE on real data (D11, not confirmed), "fully offline" or
"Wi-Fi off" (P13), any real-world accuracy claim, "fair" or "unbiased", "anonymous", "tamper-proof", "clinically
validated", or any use for employers, schools, proctoring or productivity scoring. If the live mask flip is
confirmed before submission, add one sentence: "On one person acting a forced smile with cued fists and hall
noise, live, the override fired as designed; this shows the pipeline working, not accuracy."
