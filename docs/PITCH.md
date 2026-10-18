# Valence pitch: 10-minute deck content

TrojanHacks @ USC 2026, AI Data Governance track. Talk at 1:45 pm. **Slides and videos final by 12:00.**

This file is the content for the deck. A second Mac session builds the slides (16:9, 12 slides) and records the
replay-mode videos from the shot list in section 5. The user presents.

**Rules for whoever builds or presents the deck**

- Every number below has a claim ID, such as `[H1]`. Section 3 lists the numbers to say out loud. A number with
  no claim ID does not go on a slide.
- Every results slide has this footer: *"All metrics on simulated data: 45 held-out simulator sessions,
  untuned noise. The simulator encodes our assumptions, so part of the context model's lead is by construction.
  They test the architecture, not real-world accuracy."*
- Every video has a visible caption. Replay clips say **"REPLAY: scripted simulated scenario"**.
- Planned talk time is 9:30, which leaves 30 s of buffer. The `Time` column shows elapsed time when the slide ends.
- Use one macro-F1 for the headline: **0.797 vs 0.422** (whole test set). Do not also show 0.784 vs 0.419
  (15 min of history) or the 6-state 0.767.
- Source files: `reports/metrics.json`, `docs/RESULTS.md`, `docs/NARRATOR.md`, `docs/BENCHMARKS.md`,
  `docs/GOVERNANCE.md`, `artifacts/*/model_meta.json`. Figures are in `reports/figs/`.

---

## 1. Slide plan at a glance

| # | Slide | Visual | Length | Time |
|---|---|---|---|---|
| 1 | Valence: the room carries the signal | Title, still from Clip 1 | 0:30 | 0:30 |
| 2 | The face lies, and the face is not enough | Two-panel problem graphic | 0:45 | 1:15 |
| 3 | The original mock data had the answer baked in | Three-bar chart (build it) | 0:30 | 1:45 |
| 4 | Thesis: context, over time | Four-layer graphic + Clip 3 loop | 0:35 | 2:20 |
| 5 | How it works, on one laptop | Pipeline diagram (build it) | 1:00 | 3:20 |
| 6 | Demo: the face says happy, the room says otherwise | Clip 1 + Clip 2 (or LIVE clip) | 1:30 | 4:50 |
| 7 | Context beats the face snapshot | `accuracy_vs_history.png` | 1:00 | 5:50 |
| 8 | Stable when the face fails or the room is loud | `robustness_face_dropout.png` + `venue_shift.png` | 1:00 | 6:50 |
| 9 | Every prediction explains itself | Person-card still + counterfactual | 0:45 | 7:35 |
| 10 | Governance: decisions kept, data not | Lifecycle graphic + Clip 5 excerpt | 1:00 | 8:35 |
| 11 | What this does not show yet | Plain text | 0:35 | 9:10 |
| 12 | Close | Tagline | 0:20 | 9:30 |

The speaker notes total about 1,470 words, roughly 9:50 at 150 words per minute. Most of slide 6 is spoken over
the video. The disclosures added to slides 8, 10 and 11 make them run about 13, 25 and 20 s long; slides 2, 4, 6,
7 and 12 run about 30 s short together, so the talk lands near 10:00 and uses the whole buffer. Cut these before
rehearsal (about 87 words, 35 s), and never cut a limitation or disclosure sentence: the Clip 3 sentence on slide 4,
the counterfactual sentence on slide 9 and its ", and its 95th percentile stayed at 1.9 seconds with YOLO running"
clause, and on slide 5 "We decode only about a third of camera frames in the first place" and "Perception runs at
ten frames a second in 45 to 57 milliseconds per frame".

Backup slides after the close, not presented: B1 calibration (`reliability.png`), B2 the first 30 seconds of an
episode (`acc_vs_episode_age.png`), B3 model ladder and confusion matrices (`confusion_m0_vs_m2.png`), B4 Clip 6
(full storyline), B5 narrator A/B table.

---

## 2. Slides

### Slide 1: Valence (0:30, ends 0:30)

**On slide**
- Headline: **Valence**
- Subhead: *The environment carries the signal, not the face.*
- One line: Context-aware emotional-state inference that keeps no raw media and explains every call.
- Background: still from Clip 1 at about 48 s (person card with the MASK OVERRIDE banner). Caption it "REPLAY".

**Visual:** still frame from Clip 1. Crop out the Timeline panel.

**Speaker notes**
> "Most emotion AI looks at a face and guesses. We think that's backwards. A face is one input. The room around
> you, how long you've been working, and what your body is doing over the last fifteen minutes say more. Valence
> reads that context on a laptop, throws the raw video and audio away, and explains every call it makes. Every
> accuracy number in this talk is on simulated data, and I'll show you exactly where that matters."

**Claims:** none (no numbers).

---

### Slide 2: The face lies, and the face is not enough (0:45, ends 1:15)

**On slide**
- Headline: **Face-only emotion AI fails in two ways**
- **The masking dilemma:** a forced, polite smile reads as happy, and the frustration underneath is missed.
- **Context blindness:** head in hands can mean weeping, laughing or deep focus. A furrowed "resting focus face"
  reads as anger.
- **The black box:** frames are stored, and labels come with no reason.

**Visual:** two panels. Left: a smiling face icon with the caption "reads: happy". Right: one head-in-hands
silhouette with three captions: "focused?", "laughing?", "exhausted?". No real faces, no photos of people.

**Speaker notes**
> "Three problems. First, masking. People smile through frustration all the time, at work, in class, at a
> hackathon at hour twenty. A face model reads the smile. Second, context blindness. Head in hands could be
> exhaustion or laughter. A focused frown gets read as anger. Third, most pipelines keep the frames and can't tell
> you why they said what they said. In our simulator, the face-only baseline misses masked frustration almost nine
> times out of ten. But first, why we had to rebuild the data to measure that honestly."

**Claims:** "almost nine times out of ten" = Frustrated_Masked recall 11.2% for face-only `[H2]` (simulated).
This line is optional. If you use it, say "in our simulator".

---

### Slide 3: The original mock data had the answer baked in (0:30, ends 1:45)

**On slide**
- Headline: **The original mock dataset could not test our idea**
- Bars, trained on the original mock rows: face only **66.7%**, context only **100.0%**, everything **100.0%**
- One line: Context alone was perfectly separable, and the rows had no time order, so a 15-minute window meant
  nothing. We built Simulator v2: time structure, overlapping states, noise, and missing face or light.

**Visual:** build a 3-bar chart from `reports/metrics.json` → `legacy_mock` (face_only 0.6667, context_only 1.0,
full 1.0). Label the y-axis "accuracy on the original mock rows".

**Speaker notes**
> "The dataset we were handed scored a hundred percent from context alone. That sounds like a win for our thesis,
> but it's a flaw. The labels were a lookup table, with no time structure, so it couldn't test a fifteen-minute
> window at all. So we threw it out and wrote a simulator with time, overlap, noise and missing data, and we
> published its assumptions in DATASET.md. Everything after this slide is on that simulator."

**Claims:** `[LM1]` (LightGBM accuracy on the 600 held-out rows of 2,000 regenerated with the original i.i.d. mock
generator, trained on the other 1,400; a check of the old data, not a Valence model).

---

### Slide 4: Thesis: context, over time (0:35, ends 2:20)

**On slide**
- Headline: **Read the room, over 15 minutes, not one frame**
- Four layers, each a column or tile:
  - **Ambient Physics:** noise (dB), light (lux), relative to the room's own baseline
  - **Time Dynamics:** hours since a break, time of day
  - **Somatic and Social:** fists, slump, head in hands, motion, laughter, compliments
  - **Facial Surface:** face valence, as one input among four

**Visual:** four tiles. Optional muted loop of Clip 3 (same head-in-hands posture, different room, different
label) in a corner, captioned "REPLAY: scripted simulated scenario. Same posture, different room".

**Speaker notes**
> "Our thesis: the environment carries the signal, not the face. We build a fifteen-minute rolling state vector
> per person, in four layers: the physics of the room, time, the body and social cues, and the face, which is
> just one layer, so you can always see how much it counted. In this clip the posture is the same, head in hands,
> but one room is loud with laughter and compliments and the other is quiet and dim. Different
> context, different call."

**Claims:** none (no numbers). If you mention the clip's posture share (~72% vs ~79% when present), that is a demo
observation from the shot list, not a metric. Do **not** say "same face": the face reading changes too.

---

### Slide 5: How it works, on one laptop (1:00, ends 3:20)

**On slide**
- Headline: **Raw media never leaves the perception step**
- Pipeline (left to right): Camera + mic → **Edge perception** → **Purge gate** → **Features only (5 s ticks,
  15-min window in RAM)** → **LightGBM context model + face-only shadow** → **Explanation layers (TreeSHAP)** →
  **Local LLM narrator + validator** → **Console**
- Footer line: On-device. The camera/mic processes run in a macOS sandbox with outbound network denied (loopback only).

**Visual:** build the pipeline diagram. Under each box, in small type:
- Edge perception: YOLO26n-pose + ByteTrack (people, posture), MediaPipe face blendshapes and gestures, YAMNet
  (laughter, speech), Vosk with an 8-phrase grammar (compliments), dB and lux
- Purge gate: zero-fill every frame and audio block in place
- Features only: counts, levels, flags, boxes
- Models: M2 context model; M0 face-only model runs beside it as a shadow
- Narrator: Qwen3-1.7B 4-bit on MLX + LoRA; every number checked; template fallback
- Console: FastAPI + WebSocket → React

**Speaker notes**
> "Everything runs on this MacBook. The camera and mic feed a perception step: YOLO pose with ByteTrack tracking,
> MediaPipe face blendshapes and gestures, YAMNet for laughter, and Vosk restricted to eight phrases, so a free
> transcript never exists. The moment features are extracted, a purge gate zero-fills the frame or audio block in
> memory. We decode only about a third of camera frames in the first place. Perception runs at ten frames a
> second in 45 to 57 milliseconds per frame. What moves on is numbers: every five seconds, one tick per person,
> into a fifteen-minute window held in RAM. A LightGBM model reads that window. Next to it, a face-only model runs
> as a shadow so we can show where they disagree. TreeSHAP splits each decision into the four layers, and a small
> local LLM turns that into a sentence. Every number in that sentence is checked before you see it."

**Claims:** two thirds never decoded `[P3]` (assumes a 30 fps camera); 45-57 ms/frame at 10 Hz `[P1]` (M3 Max,
plugged in, one machine); 8 phrases `[P4]`, `[G10]`; camera/mic processes sandboxed with outbound network denied; MediaPipe's built-in usage telemetry is blocked by it
(unsandboxed, it held 4 HTTPS connections to Google; `docs/BENCHMARKS.md`). Say "sandboxed, no outbound network", not
"fully offline" `[P13]`. **If** the Wi-Fi-off rehearsal is done and passes
before 12:00, you may change the footer to "Demo rehearsed with Wi-Fi off".

---

### Slide 6: Demo (1:30, ends 4:50)

**On slide**
- Headline: **The face says happy. The room says otherwise.**
- Video: Clip 1 (cut to about 32 s) then Clip 2 (about 30 s). Caption both "REPLAY: scripted simulated scenario".
- After the video, three bullets:
  - Face-only: Collaborative / Happy. Context: Frustrated (masked).
  - MASK OVERRIDE fires only when face and context clearly disagree.
  - Hide the face and the label holds.

**Visual:** Clip 1 → Clip 2 (section 5). **LIVE clip option:** use the LIVE clip only if the main Mac session
delivers a recording with MASK OVERRIDE on real data before 11:30 (claim `[D11]` is not confirmed yet). Caption
it "Live, one person acting on cue: the pipeline running, not accuracy". If it arrives, play 20 s of it after Clip 2 and cut
Clip 1 to 25 s. Otherwise use the replay clips only.

**Speaker notes (Clip 1 + Clip 2; talk over the video)**
> "This is the console in replay mode, running a scripted simulated scenario. The person is typing. The context
> model says Focused. The face-only model, on the left, says Exhausted: that's the resting focus face. Now the
> scenario switches. The person smiles, but the room is loud, they've clenched their fists a few times, and it's
> been almost four hours since a break. The face-only model says happy. The context model says frustrated, and the
> banner says MASK OVERRIDE: context corrected the facial mask.
> Now we simulate the face being hidden. Watch the badge: SIMULATED: face occlusion. The face-only model has
> nothing to say. The context model keeps its call, and the explanation says it is resting on context."

**After the video, over the bullet "MASK OVERRIDE fires only when face and context clearly disagree":**
> "That's not the model 'knowing' the smile is fake. It's a conservative rule: it fires only when the face model
> clearly reads happy, the context model is at least 70% sure of a state other than happy, and the face features
> are what pull toward happy. It never fires in the first two minutes."

**If the LIVE clip is used, add:**
> "And this is live, on me, at our table. I'm acting the smile and the fists, and it's one person, so it shows
> the pipeline working, not accuracy."

**Claims:** override rule `[M1]` (thresholds tuned on simulated data); scenario reads as scripted `[S4]`, `[S1]`
(in-distribution); label holds when the face is hidden (Clip 2 observation; the population-level figure is `[R1]`
on slide 8). Do not say the live mask flip works on real data unless the LIVE clip shows it `[D11]`.

---

### Slide 7: Context beats the face snapshot (1:00, ends 5:50)

**On slide**
- Headline: **Context over time beats a face snapshot (simulated data)**
- **Masked frustration caught: 68.0% vs 11.2%** (context vs face-only)
- **Focused people misread as negative: 17.9% vs 56.3%**
- **Macro-F1: 0.797 vs 0.422**

**Visual:** `reports/figs/accuracy_vs_history.png` (left two thirds); the three numbers stacked on the right.
Footer: the simulated-data line.

**Speaker notes**
> "Here's the core result, on forty-five held-out simulated sessions. The x-axis is how many minutes of history
> the model gets. The face-only model is flat: it reads one snapshot, so history can't help it. Our context model
> climbs from 0.65 macro-F1 with no history to about 0.79 at five minutes, and then it plateaus. On the whole test
> set: masked frustration caught 68 percent of the time versus 11 percent for the face alone, about six times as
> often. Focused people misread as negative drops from 56 percent to 18. Macro-F1 0.797 versus 0.422. One honest
> catch: in the first thirty seconds of a new emotional state, context is behind the face: 33 versus 44 percent
> accuracy. It overtakes after about thirty seconds. That's why the label and the override have warm-up and
> hysteresis built in."

**Claims:** 68.0% vs 11.2% `[H2]`; 17.9% vs 56.3% `[H3]`; 0.797 vs 0.422 `[H1]`; curve 0.649 → 0.791 at 5 min
(plateau), face-only flat 0.417-0.419 `[A1]` (macro-F1 by minutes of history; the claim text says "accuracy" but
the metric in `metrics.json` is `macro_f1`); first 30 s 32.8% vs 44.3% accuracy `[E1]`. All simulated.
Do not put 17.7% (venue pass) on this slide `[H3]`.

---

### Slide 8: Stable when the face fails or the room is loud (1:00, ends 6:50)

**On slide**
- Headline: **When the face fails, context holds (simulated data)**
- **Face hidden on every tick:** macro-F1 0.678 vs 0.121 for face-only
- **Face readings degraded (as at 10% light):** 0.679 vs 0.252
- **Loud hall (+75 dB):** Focused misread as negative rises 17.7 → 18.7% (served) vs 20.8 → 28.1% (previous
  model); most of the gain comes from training on simulated venues, the room baseline adds about 1 point

**Visual:** `reports/figs/robustness_face_dropout.png` (left) and `reports/figs/venue_shift.png` (right).
Optional: muted 10 s loop of Clip 4 (loud_venue) under the venue figure, captioned "REPLAY: 72 dB hall,
scripted simulated scenario".

**Speaker notes**
> "Two failure modes. First, the face disappears. With the face hidden on every tick, the face-only model
> collapses to 0.12 macro-F1. Ours keeps 0.68: it falls back to the context-only floor and stays there. With face
> readings degraded as at ten percent light, 0.68 versus 0.25. That degrades the face channel only; it's not a
> camera test. Second, venues. A loud hall is not a frustrated person. The operator captures about ninety seconds
> of the room's noise and light, and the model reads noise above that baseline. With hall noise mixed into the
> test sessions, our Focused-misread rate goes from 17.7 to 18.7 percent at plus 75 dB; the previous model drifts
> from 20.8 to 28.1. Real, but modest. Most of that comes from training on simulated venues of 40 to 75 dB;
> reading noise relative to the baseline adds about one point on top, 18.7 versus 19.9 percent for the same model
> on absolute dB. On our one real typing capture with 72 dB mixed in, both models still read
> Focused, 25 of 25, so that capture shows no drift either way."

**Claims:** 0.678 vs 0.121 `[R1]` (face channel only; not demographic fairness; partly by construction: the served
model was trained on face-hidden and face-degraded copies close to these tests (MODEL_CARD §5, §7); if asked, say
so); 0.679 vs 0.252 `[R3]`, `[R4]`
(face degraded only, lux held fixed); venue 17.7/17.8/18.4/18.7% vs 20.8/22.3/25.9/28.1% `[V1]` (simulated, constant
hall noise, modest gain; the gap is mostly venue-varied training: the same training with absolute dB/lux gives
18.1/18.1/19.1/19.9%, the middle line in `venue_shift.png` (RESULTS.md venue table, `m2_abs`), so room-relative vs
absolute dB at +75 dB is 18.7% vs 19.9%, about 1 point); ~90 s baseline `[V5]` (not yet rehearsed in the real
venue); real typing capture
25/25 `[V4]` (one person, synthetic noise, shows no drift for either model). The 17.7% here is a separate pass
from the 17.9% on slide 7 `[H3]`; if asked, say so.

---

### Slide 9: Every prediction explains itself (0:45, ends 7:35)

**On slide**
- Headline: **Every call comes with its reasons, and every number in them is checked**
- **Four layers:** TreeSHAP contributions summed per layer, so the face's weight is always visible
- **Local narrator:** a 1.7B model on the laptop; every number must match the prediction's fact sheet, or the
  text falls back to a template
- **Counterfactual:** "what if they'd had a break?" Change one input and re-run

**Visual:** still from Clip 1 at about 48 s: the person card with "Top drivers by layer" and the explanation with
its source badge (`template` or `llm`). Optional terminal overlay of the counterfactual call from Clip 1.

**Speaker notes**
> "Every prediction comes with its drivers, grouped into the four layers, so you can see how much the face
> counted. A small local language model writes the sentence, and a validator checks every number in it against
> the facts behind the prediction. Anything that fails falls back to a template. We trained a LoRA adapter and
> decided to ship it by a rule written before the test: it passed the validator on the first try 93 percent of
> the time versus 75 for prompt-only, invented no numbers in raw attempts, and its 95th percentile stayed at 1.9
> seconds with YOLO running. And you can ask what-if: here, setting the break clock to zero flips the call to
> Collaborative."

**Claims:** validator rules `[P11]`; first-try pass 93.3% vs 75.3% `[P6]` (300 test sheets, about ±3 points);
0 invented numbers and 0 false mask claims in raw attempts `[P7]` (measured by the validator); p95 1904 ms under
YOLO load, 2.5 s deadline `[P8]` (AC power only); pre-stated ship rule `[P9]`. The counterfactual flip is a demo
observation from the shot list (Clip 1, optional overlay), not a metric. **Check the badge in the recorded
clip:** if it says `template`, say "this clip shows the template text; live, the LLM writes it."

---

### Slide 10: Governance: decisions kept, data not (1:00, ends 8:35)

**On slide**
- Headline: **We keep decisions, not data**
- **Never stored:** video, audio, transcripts, pose keypoints, face images or embeddings. The backend never
  receives camera pixels.
- **Purge gate + counters:** every frame and audio block is zero-filled after feature extraction; leftovers count
  as violations. **Tamper-evident audit chain** (SHA-256), deleted at session end.
- **Position:** a research prototype for consented self-monitoring and personal wellbeing, outside workplace and
  education settings. Not for employers, schools, proctoring or productivity scoring.

**Visual:** a lifecycle strip: frame → features → zero-filled (with a "PURGED" stamp) | tick → 15-min RAM window →
evicted | decision → audit chain → deleted at session end. Optional 15-20 s excerpt of Clip 5: the governance tests
passing, `persistence_scan` "0 violation(s)", `verify` → `ok: true`, one edited line → `ok: false`.

**Speaker notes**
> "Governance is the track, so here's exactly what we keep. Never stored: video, audio, transcripts, keypoints,
> face images, embeddings. The backend has no endpoint that accepts an image. Every raw frame and audio block is
> zero-filled in memory right after feature extraction, and any buffer that survives the tick counts as a
> violation. Copies inside native libraries, like the GPU or MediaPipe, are out of reach, and we document that. A
> static scan finds no file writes at all in the perception code and no media-save or binary writes in the
> capture code. What we keep is a log of decisions, hash-chained:
> in this clip one character is changed and verification fails. Tamper-evident, not tamper-proof, and deleted at
> session end. On regulation: the EU AI Act bans emotion recognition in workplaces and education institutions,
> except for medical or safety reasons, with no consent exception. The Act doesn't reach a US event, so it's our
> design bar. Valence is a research prototype for consented self-monitoring outside work and education, not for
> employers, schools, proctoring or productivity scoring. A campus hackathon is close to the education setting, so
> at our table it analyzes only me and volunteers who opt in, and that opt-in is a data-protection courtesy, not
> an exemption. This is not legal advice."

**Claims:** never-stored list `[G10]`; backend never receives pixels `[G11]` (the console's own live camera
preview is browser-local, so do not say "the UI never shows pixels"); purge gate `[G1]` (native-library copies
are not zeroed); counters every 5 s `[G2]` (replay counters prove nothing: do not show replay counters as proof);
audit chain `[G3]` (tamper-evident); deletion at session end + 24 h TTL `[G4]`; 0 violations `[G5]`; 23 governance
tests passed `[G6]` (cloud run; GOVERNANCE.md §6 still says 22); positioning `[G15]`, wording taken from
GOVERNANCE.md §7. Optional extras if asked: room-only mode `[G9]` (still processes biometric data), kill switch
`[G13]`, provenance badges `[G12]`, bystanders dropped from recordings `[G8]`.

---

### Slide 11: What this does not show yet (0:35, ends 9:10)

**On slide**
- Headline: **Honest limits**
- **Trained and scored on simulated data only.** Real-data checks are one person at one desk, about 25 ticks
  each, one of them in-sample, and all are negative controls: every capture so far reads Focused (cued slumps
  too, 26 of 26).
- **Not clinical, not a diagnosis.** Not validated across demographics.
- **Context needs time:** the first 30 s of a new state are weaker than the face alone.

**Visual:** plain text. No figure.

**Speaker notes**
> "What this doesn't show. The models never trained on a recorded person: they're trained and scored on our
> simulator. That simulator encodes our own assumptions, like forced smiles and context that lags the state, so
> part of the gap is by construction. Our real-data checks are one person at one desk, about twenty-five ticks
> each. Quiet typing reads Focused twenty-five out of twenty-five, but the simulator was fit to that capture, so it's in-sample.
> Cued fist clenches alone in a quiet room still read Focused, twenty-four of twenty-five, so fists alone don't
> trigger frustration. So far every real capture reads Focused; we have not yet shown a negative state on a real
> person. It isn't clinical and doesn't diagnose anything. We haven't validated it across
> demographics. And context needs about thirty seconds before it beats the face."

**Claims:** simulator-only training `[G14]`, `[L5]`; typing 25/25 in-sample `[D1]`; fists 24/25 `[D2]`; cued
slump 26/26 (MODEL_CARD §5); first
30 s `[E1]`. Add one bullet **only if still true at 12:00**: "The live mask flip on real data is still being
recorded" `[D11]`. If the golden recording or the LIVE clip shows the override before 12:00, change the
"every capture so far reads Focused" bullet and note to say that one acted capture showed the override.
Optionally: Wi-Fi-off rehearsal not yet run `[P13]`; native-library copies not zeroed `[G1]`.

---

### Slide 12: Close (0:20, ends 9:30)

**On slide**
- Headline: **The environment carries the signal, not the face.**
- Three words across: **Context. Ephemeral. Explained.**
- Repo link and team name.

**Visual:** title-slide background, dimmed.

**Speaker notes**
> "Valence reads the room, keeps none of the raw media, and explains every call. Emotion AI should be built to
> notice what the face hides, and to forget what it saw. Thank you."

**Claims:** none.

---

### Backup slides (not presented; for Q&A)

| Slide | Visual | Line to say | Claims |
|---|---|---|---|
| B1 Calibration | `reports/figs/reliability.png` | "ECE 0.013, 0.0045 after temperature scaling. When it says 95%, it's right about 95% of the time, on simulated test windows." | `[C1]`, `[C2]` |
| B2 First 30 seconds | `reports/figs/acc_vs_episode_age.png` | "0-30 s: 32.8% vs 44.3%; 5+ min: 94.0% vs 46.1%. Hence the 60 s warm-up and the 120 s override block." | `[E1]`, `[M1]` |
| B3 Model ladder | `reports/figs/confusion_m0_vs_m2.png` | "Face snapshot 0.422, face history 0.490, context snapshot 0.583, context history without face 0.677, served 0.797. One context snapshot already beats a face model with history." | `[L1]`, `[L2]` |
| B4 Full storyline | Clip 6 | "Focused, then masked frustration with the override, then a break, then Upbeat. Scripted simulated replay." | `[S2]` |
| B5 Narrator A/B | Table from NARRATOR.md | "First try 93.3% vs 75.3%; fallback 1.3% vs 10.3%; final pass 100% both; p95 1904 vs 1950 ms." | `[P6]`, `[P8]`, `[P9]` |
| B6 Why not a neural net? | `reports/figs/m3_attention.png` | "A GRU with attention scored 0.759. We serve LightGBM because TreeSHAP gives exact per-feature attributions for the explanations." | `[L7]` |

---

## 3. Numbers to say out loud (cheat sheet)

Say "on simulated data" at least once per slide that has a model number. Round only the way shown in the
"Say" column.

| Say | Exact value | Caveat to carry | ID |
|---|---|---|---|
| "Masked frustration caught 68% vs 11%, about six times as often" | 68.0% vs 11.2% | Simulated; the simulator writes the mask in | H2 |
| "Focused misread as negative: 56% down to 18%" | 56.3% → 17.9% | Simulated | H3 |
| "Macro-F1 0.797 vs 0.422" | 0.797 vs 0.4224 | 45 held-out simulated sessions | H1 |
| "0.65 with no history, about 0.79 at five minutes, then it plateaus" | 0.649 → 0.791 (5 min); M0 flat 0.417-0.419 | Simulated; macro-F1 by history | A1 |
| "With no face at all, context history still beats the face snapshot" | 0.675 vs 0.417 (both at 5 min of history) | Simulated | A2 |
| "First 30 seconds: 33% vs 44%; it overtakes after about 30 s" | 32.8% vs 44.3% (n = 1,663) | Accuracy per window | E1 |
| "Face hidden every tick: 0.68 vs 0.12" | 0.678 vs 0.121 | Face channel only; not fairness; partly by construction (modality dropout) | R1 |
| "Face degraded as at 10% light: 0.68 vs 0.25" | 0.679 vs 0.252 | Face readings only; lux unchanged; not a camera test; partly by construction (modality dropout) | R3 |
| "Loud hall: 17.7 to 18.7% for us, 20.8 to 28.1% for the old model" | 17.7/17.8/18.4/18.7 vs 20.8/22.3/25.9/28.1% | Simulated hall noise; modest gain; gap is mostly venue-varied training; room-relative vs absolute dB at +75 dB: 18.7% vs 19.9% (RESULTS.md venue table, m2_abs) | V1 |
| "About ninety seconds of room baseline" | 90 s default, up to 600 | Not yet rehearsed in the venue | V5 |
| "Real typing capture with 72 dB mixed in: Focused 25 of 25, both models" | 25/25 | One person; shows no drift either way | V4 |
| "Quiet typing reads Focused 25 of 25" | 25/25 | In-sample; one person | D1 |
| "Fists alone: Focused 24 of 25" | 24/25 | One person, quiet room | D2 |
| "Calibration error 0.013, 0.0045 after scaling" | ECE 0.013 → 0.0045, T = 0.934 | Simulated | C1 |
| "Perception 45 to 57 ms per frame at 10 Hz" | 45-57 ms, 10.0 Hz | M3 Max, plugged in | P1 |
| "Two thirds of frames are never decoded" | ~2/3 of 30 fps | Assumes 30 fps | P3 |
| "Narrator first-try pass 93% vs 75%" | 93.3% vs 75.3% | 300 sheets, ±3 points; "passes our validator" | P6 |
| "No invented numbers in raw attempts" | 0.000 vs 0.171 | Measured by the validator | P7 |
| "p95 1.9 s with YOLO running; deadline 2.5 s" | 1904 ms | AC power only | P8 |
| "No file writes in perception, no media saves in capture" | 0 violations | Scans perception and edge only; edge may write text (features-only JSONL, calibration scalars) | G5 |
| "23 governance tests pass; 586 tests pass in all" | 23 passed; 586 passed, 20 skipped | Cloud run; the 20 skips need Mac-only packages (4) or downloaded models/tokenizer (16) | G6 |
| "Audit log deleted at session end, 24 h backstop" | unlink on shutdown; TTL 86,400 s | Unlink is not a secure erase | G4 |
| "Override needs face ≥ 0.45 happy, context ≥ 0.70 on any state other than happy, no override in the first two minutes" | 0.45 / 0.70 / 60 s warm-up / 120 s block | Tuned on simulated data; context ≥ 0.70 on any non-happy state (Focused included; served.py POSITIVE) | M1 |

---

## 4. Do not say

| Do not say | Say instead | Why |
|---|---|---|
| "Consent makes it legal" / "opt-in exempts us from Article 5" | "The opt-in is a data-protection courtesy, not an Art. 5(1)(f) exemption." | No consent exception (G15) |
| "Room-only mode makes workplace or classroom use compliant" | "Room-only reduces exposure; it still processes biometric data." | G9 caveat |
| "Valence is EU AI Act compliant" | "The Act is our design bar; this is not legal advice." | G15 |
| "It's not emotion recognition" / "no biometric data" | "It is an emotion recognition system in the Act's sense." | GOVERNANCE.md §7 |
| Any use for employers, schools, proctoring, productivity scoring | "Consented self-monitoring outside workplace and education." | G15 |
| Any accuracy number without "simulated", or "accurate in the real world" | "On simulated data, 45 held-out sessions." | H1 caveat |
| "Fair", "unbiased", "prevents bias" | "Stable when the face is missing; demographics not validated." | R1 caveat |
| "Clinically validated", "diagnoses", "detects stress" as a diagnosis | "Infers a likely state from context." | Not clinical |
| "Anonymous" | "Pseudonymous, which is still personal data; so it's RAM-only and deleted." | GOVERNANCE.md §7 |
| "Frames are erased everywhere" / "`del` deletes the frame" | "Our buffers are zero-filled; copies inside native libraries are not reachable." | G1 caveat |
| "Nothing is ever written to disk" | "Raw media is never written. By default the live pipeline writes one file, the decision log, deleted at session end. Self-report labels, if used, expire after 24 h, and our own features-only test captures (no media) are in the repo." | G4, G8; GOVERNANCE.md §2 rows 14-15 |
| "Tamper-proof audit log" | "Tamper-evident." | G3 |
| "The live demo proves accuracy" | "The live clip shows the pipeline working on one person." | D11 |
| "Real data validates the model" | "Small real-data checks, one person, one in-sample." | D1-D4 |
| "Clenching your fists makes it read Frustrated" | "Fists alone read Focused 24 of 25; the masked state needs the whole context." | D2 |
| "The console never shows camera pixels" | "The backend never receives pixels." | G11 |
| "Low light doesn't change the label" | "With face readings degraded as at 10% light, 0.68 vs 0.25." | R3; the console's low-light overlay also changes lux |
| "The venue baseline solves loud rooms" | "A real but modest gain, on simulated hall noise." | V1 |
| "The narrator can't make things up" | "Every number is checked against the fact sheet; failures fall back to a template." | P11 |
| Showing replay governance counters as proof of purging | Show the tests, the scan and the chain check instead | G2: replay counters are equal by construction |
| "The model knows the smile is fake" | "Face and context disagree, and a conservative rule flags it." | M1 |
| "It only fires on negative states" | "It fires when context confidently reads anything but happy and the face pulls toward happy." | M1, served.py POSITIVE |
| "Same face, different label" (Clip 3) | "Same posture, different room." | The face reading changes too |
| "The break reset the clock" (Clip 6) | Say nothing about the clock | The clock keeps running in that scenario |
| "Fully offline" / "Wi-Fi off" (unless rehearsed today) / "nothing ever touches the network" | "The camera and mic processes run in a sandbox with outbound network denied; we found and blocked MediaPipe's built-in telemetry." | P13 |
| "MASK OVERRIDE works live on real data" (unless the LIVE clip shows it) | "Shown here in replay; the live flip is being recorded." | D11 |
| "The 6-state model scores 0.767" next to 0.797 | Leave it out, or compare with 0.791 under the same scoring | L8 caveat |
| "Exhausted, live" | Show Exhausted only in replay | D4: slump cannot be acted live |
| "The model learned from real people" | "Trained only on simulator output; the simulator's noise was fit to one capture." | G14 |

---

## 5. Video shot list (second Mac session, replay mode, no camera)

All clips are recorded from the console in replay mode. No camera, no git commands. Keep off ports 8000, 8081
and 8080 (the main session's API, narrator and frontend).

### 5.1 One-time setup

```bash
# Terminal S1: API on :8100
cd ~/Valence            # the same checkout
[ -f .env ] && { set -a; source .env; set +a; }
VALENCE_MODE=replay VALENCE_AUDIT_DIR=var/pitch-audit VALENCE_NARRATOR_URL=http://127.0.0.1:8199 \
VALENCE_SCENARIO=focused VALENCE_SPEED=1 .venv/bin/python -m uvicorn valence.api.app:app --port 8100

# Any terminal
R=http://127.0.0.1:8100; J='content-type: application/json'
curl -s $R/api/health | jq '{mode, model}'      # expect "replay", "m2_lgbm@8a78efc7"

# Terminal S2: frontend on :5180
cd ~/Valence/frontend && npm run dev -- --host 127.0.0.1 --port 5180
```

- **Narrator.** Never leave `VALENCE_NARRATOR_URL` at its default, or this API sends narration requests to the
  main session's narrator on :8081.
  - With port 8199 (nothing listening) you get template text, Narrator "fallback" and Degraded "narrator". That is
    acceptable for the clips; the slide 9 notes cover it.
  - For LLM text, start a second narrator only while the main session is **not** recording live (they compete for
    the GPU): `.venv/bin/python -m valence.narrator.service --port 8181`, and restart S1 with
    `VALENCE_NARRATOR_URL=http://127.0.0.1:8181`. LLM text shows best at speed 10. Keep the Mac plugged in: on
    battery every request times out to the template (P8).
- **Browser.** Open `http://127.0.0.1:5180/?api=http://127.0.0.1:8100`. Click **Block** on the camera prompt.
  Click **LIVE** in the header. Expect "LIVE · CONNECTED" plus a **REPLAY** badge. (The connection label says LIVE
  whenever the WebSocket is open, even in replay; the REPLAY badge is the provenance.) The console boots in MOCK
  mode with "PLACEHOLDER DATA": never record it in that state.
- **Before each clip:**
  1. Turn both overlays off:
     ```bash
     curl -s -X POST $R/api/sim/face_occlusion -H "$J" -d '{"enabled": false}'
     curl -s -X POST $R/api/sim/low_light -H "$J" -d '{"enabled": false}'
     ```
  2. Click LIVE again to clear stale cards.
  3. Start the screen recording. **t0** is when the replay `curl` returns.
- **Timing rule:** tick k appears about (k + 1) × 5 / speed seconds after t0.
- **Crop** out the 15-minute Timeline panel (it plots wall-clock time, so a fast replay fills only its right edge).
- **Caption** every clip "REPLAY: scripted simulated scenario".

### 5.2 Clips

| Clip | Slide | Command | Record | Keep |
|---|---|---|---|---|
| 1 | 6, 1, 9 | `curl -s -X POST $R/api/replay/frustrated_masked -H "$J" -d '{"speed": 10}'` | 0-84 s | 12-24 s and 38-58 s (~32 s total) |
| 2 | 6 | A fresh run of the Clip 1 command (never the Clip 1 take, which stays overlay-free); face overlay at ~45 s | 40-72 s | ~30 s |
| 3 | 4 (optional loop) | `curl -s -X POST $R/api/replay/context_blindness -H "$J" -d '{"speed": 20}'` | 0-40 s | 10-20 s and 28-38 s |
| 4 | 8 (optional loop) | `curl -s -X POST $R/api/replay/loud_venue -H "$J" -d '{"speed": 20}'` | 0-35 s | 4-17 s and 22-32 s |
| 5 | 10 (excerpt) | Terminal sequence below | ~50 s | 15-20 s excerpt |
| 6 | B4, fallback | `curl -s -X POST $R/api/replay/full_demo -H "$J" -d '{"speed": 20}'` | 0-78 s | cut 42-61 s (break) |
| 7 | 6 (optional) | `curl -s -X POST $R/api/replay_recording/golden_demo -H "$J" -d '{"speed": 4, "loop": false}'` | 0-150 s | only if MASK OVERRIDE appears |

**Clip 1: face says happy, context says otherwise** (speed 10)

| Time after t0 | On screen |
|---|---|
| 0-5.5 s | Warming up |
| 6-24 s | Focused 86-99% (mostly 95-99% from 9 s); Face-only mostly **Exhausted** (resting focus face) |
| 24.5-27.5 s | Masked segment starts in the data (minute 4). Card still says Focused but at 15-21% (one 86% tick) while smoothing catches up; Face-only Collaborative / Happy. Do not use |
| 29-39 s | Flicker: first MASK OVERRIDE at 29-32 s, then Upbeat, then Frustrated |
| **40.5 s** | Stable "Frustrated / Masked Stress" 87-99% (one 33% tick at 40.0 s), Face-only "Collaborative / Happy" (brief "no face visible" ticks), banner on, mask sentence. Fists per 5 min: 1 at 40.5 s, 3 at 44.5 s, 4-5 from 48.5 s; noise ~60 dB, mostly 62-68 dB from ~45 s; 3.7-3.8 h since break |
| 84 s | Scenario ends |

Optional counterfactual overlay for slide 9 (terminal, any time after 45 s):
```bash
EID=$(curl -s $R/api/state | jq -r '.people["1"].prediction.event_id')
curl -s "$R/api/explain/$EID/counterfactual?feature=hours_since_break&value=0.5" | jq '{original, cf: .counterfactual.label, label_changed}'
curl -s "$R/api/explain/$EID/counterfactual?feature=hours_since_break&value=0" | jq '{original, cf: .counterfactual.label, label_changed}'
```
Expected (cloud run): at 0.5 h it stays Frustrated_Masked (`label_changed` false); at 0 h it becomes
Collaborative_Happy (`label_changed` true).

**Clip 2: hide the face** (fresh run of the Clip 1 command; same timings)
- ~45 s: click "Simulate face hidden" (and turn it off by clicking it again, not with curl; the toggle only
  reflects its own clicks. If it is left lit, reload the page and click LIVE), or
  `curl -s -X POST $R/api/sim/face_occlusion -H "$J" -d '{"enabled": true}'`.
  Within one tick: Face-only shows "no face visible"; the explanation switches to "The face is not readable right
  now, so the call rests on context"; the badge shows "SIMULATED: face_occlusion"; **the label holds**
  (confidence holds ~95% for about 5 s, then drifts to ~60-89% as the last fist ages).
- ~57 s: turn it off the same way (click again, or the same curl call with `false`). The SIMULATED badge stays for
  180 ticks (90 s at speed 10); that is intended provenance.
- Optional low light at 60-72 s: `/api/sim/low_light` true, then false. Do not narrate it as "low light doesn't
  change the label" (that overlay also cuts lux).
- Stop before 84 s.
- Quirk: the MASK OVERRIDE banner stays on while the face is hidden (the flag updates only on face-visible ticks).
  Don't call it re-detection.

**Clip 3: same posture, different room** (speed 20)

| Time after t0 | On screen |
|---|---|
| 3-21 s | Persons #1 and #2 Collaborative / Happy. Person #1: head in hands on ~60% of ticks (~72% when present), laughs and compliments, ~57-60 dB (peaks near 70), ~420 lux, 4.5 h since break |
| 21.25 s | Both leave; cards freeze. Person #2 never returns: crop to Person #1 |
| 27.25 s | Person #1 returns: 2 ticks of "Upbeat" (smoothing) |
| **27.75 s** | **Exhausted** ~91-97% until 54 s: head in hands on ~80% of ticks in 28-38 s (~79% when present), ~48 dB, ~220-290 lux, 4.6 h |

**Clip 4: a 72 dB hall is this room's normal** (speed 20)

| Time after t0 | On screen |
|---|---|
| 3-18 s | Focused 87-95% with noise 71-73 dB; Face-only mostly Exhausted |
| 7.75-9.25 s | Honest blip: Exhausted for a few ticks (cut it or leave it; don't hide it in narration if asked) |
| 18.25 s | Masked segment starts |
| **21.75 s** | Stable MASK OVERRIDE ~96% until 45 s |

**Clip 5: governance without a camera** (record LAST: it ends with stopping the API)
```bash
curl -s -X POST $R/api/replay/focused -H "$J" -d '{"speed": 20}'      # creates a "speed": 20.0 audit entry
.venv/bin/python -m pytest -q tests/test_purge.py tests/test_no_persistence_ast.py tests/test_logfilter.py tests/test_audit.py   # 23 passed
.venv/bin/python -m valence.governance.persistence_scan --exceptions   # 0 violation(s), 0 documented exception(s)
curl -s $R/api/retention | jq '{raw_media_stored, never_kept}'
curl -s "$R/api/governance/audit?since=0" | jq -c '.[] | {seq, kind, action: .payload.action, label: .payload.label}' | head -12
curl -s $R/api/governance/verify                                        # {"ok":true,...}
F=$(ls -t var/pitch-audit/*.jsonl | head -1); sed -i '' 's/"scenario": "focused", "speed": 20.0}/"scenario": "focused", "speed": 21.0}/' "$F"
curl -s $R/api/governance/verify                                        # {"ok":false,"entries":<line edited>}
```
Then press Ctrl-C in S1 and run `ls var/pitch-audit/`: it is empty, because the file is deleted at session end.
Restart S1 if you need more clips. Use a large terminal font. Do not show the console's governance counters in
this clip as purge proof (replay counters are 0 and equal by construction).

**Clip 6: full storyline / backup** (speed 20)

| Time after t0 | On screen |
|---|---|
| 0-15 s | Focused ~98% |
| 15.25 s | Masked segment begins (minute 5 of 26) |
| 21.25 s | First MASK OVERRIDE; flickers off at 24-26.5 s and 29.5-31.25 s |
| **31.5 s** | Stable override until 42 s; the text quotes "current noise of 66 dB, 18 dB above this room's baseline" |
| 42.25-61 s | Person #1 on a break; card frozen, then a stale override card at 60.25-60.75 s (70%, then 32-34%) (cut all of this) |
| 61 s | **Upbeat** (median ~81%, mostly 62-93%) until 78 s |

This is also the backup screen recording that DEMO.md asks for.

**Clip 7: golden replay with real boxes** (optional). Only once `data/real/golden_demo.jsonl` exists (do not
commit it), and only if MASK OVERRIDE actually appears. 10 min of recording plays in 150 s; the first ~15 s are
warm-up. Caption: "Replay of a real features-only capture: one person acting the masked scenario on cue". If it
shows the override, slide 11's D11 bullet changes to "shown on one person's acted, replayed capture".

**LIVE clip** (main session only; needs the user and the camera). Record during a `make live` /
`scripts/spikes/run_demo.sh` run, not during a standalone `valence.edge.run --jsonl` recording (the camera cannot
be opened twice). Block the browser camera for :8080 so the clip shows boxes, not a face. Caption: "Live, one
person acting on cue: the pipeline running, not accuracy". If no override appears, do not fake it; use Clip 1.

### 5.3 Export

- 1920×1080, 30 fps, H.264 `.mp4`, no audio track (the presenter talks over it).
- Name files `clip1_masked.mp4`, `clip2_occlusion.mp4`, `clip3_context.mp4`, `clip4_loud_venue.mp4`,
  `clip5_governance.mp4`, `clip6_full_demo.mp4`.
- Embed videos in the deck **and** keep the `.mp4` files next to the deck for the fallback.
- Save clips, the deck and its PDF export in `~/Desktop/Projects/valence-pitch/` (outside git: never commit video or the deck).
- Export stills for the fallback: Clip 1 at ~18 s (Focused vs Exhausted), ~48 s (MASK OVERRIDE card), Clip 2 at
  ~50 s (face hidden, label held).

---

## 6. Sixty-second fallback if a video fails

Do not troubleshoot on stage. Advance to the hidden backup slide **6b** (three stills from section 5.3, side by
side, captioned "REPLAY: scripted simulated scenario") and say:

> "The video isn't cooperating, so here are three frames from it. This is a scripted simulated replay. Left: the
> person is typing. The context model says Focused; the face-only model says Exhausted. That's the resting focus
> face. Middle: the person is smiling, but the room is loud, they've clenched their fists, and it's been nearly
> four hours since a break. The face-only model says happy. The context model says frustrated, and MASK OVERRIDE
> fires. That's a conservative rule: the face model has to clearly read happy and the context model has to be
> confident before it fires, never in the first two minutes. Right: we hide the face. The face-only model has
> nothing. The context model keeps its call and says it's resting on context. Every explanation you see there is
> checked number by number before it's shown. On forty-five held-out simulated sessions, that kind of masked
> frustration gets caught 68 percent of the time versus 11 percent from the face alone."

Claims: `[M1]`, `[P11]`, `[H2]` (simulated). Time: about 65 s (162 words at 150 wpm). Then go to slide 7 and take 15 s off slide 9.

If the projector or laptop fails completely: deliver slides 7, 8 and 10 from the cheat sheet in section 3, verbally.

---

## 7. Hand-off checklist for the deck session (by 12:00)

- [ ] 12 slides + backups B1-B6 + hidden 6b, 16:9.
- [ ] Footer with the simulated-data line on slides 7 and 8 and backups B1-B3.
- [ ] Every video captioned; no MOCK or PLACEHOLDER DATA visible in any frame.
- [ ] Clip 1 shows the REPLAY badge at least once.
- [ ] Slide 5 footer matches whether the Wi-Fi-off rehearsal happened (P13).
- [ ] Slide 6 uses the LIVE clip only if it shows MASK OVERRIDE (D11); slide 11 D11 bullet updated to match.
- [ ] Slide 9 notes match the narrator badge in the recorded clip (`template` or `llm`).
- [ ] No assistant or vendor model names other than the stack named in slide 5.
- [ ] Speaker-notes rehearsal with a timer: 9:30 or less.
