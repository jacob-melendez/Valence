# Valence: judge Q&A cheat sheet

TrojanHacks @ USC 2026, AI Data Governance track: 5 min talk, 5 min Q&A. Claim IDs: `docs/PITCH.md` §3; "Avoid"
follows its §4.

> **If you only remember six things**
> 1. Every accuracy number is **simulated** (45 held-out sessions); part of the lead is by construction.
> 2. Masked frustration caught **68% vs 11%**; Focused misread **18% vs 56%** (simulated) `[H2]` `[H3]`.
> 3. It **is** emotion recognition in the Act's sense; the opt-in is "a data-protection courtesy, not an Art. 5(1)(f) exemption".
> 4. Raw media is never written; the decision log is deleted at session end `[G4]` `[G10]`.
> 5. "**Sandboxed, no outbound network**" (we found and blocked MediaPipe's telemetry), never "fully offline" `[P13]`.
> 6. Real data: one person, every capture so far reads Focused. Demographics not validated. Not clinical.

## Legality and ethics

**1. "Isn't emotion recognition banned under the EU AI Act?"**
In the workplace and in education institutions, yes, "except for medical or safety reasons. There is no consent
exception." Valence "is an emotion recognition system in the Act's sense", so it is "a research prototype for
consented self-monitoring and personal wellbeing outside workplace and education settings."
*Proof:* "The Act's territorial scope (Art. 2) does not reach a demo at a US event; it is used as the design bar." (`docs/GOVERNANCE.md` §7)
*Avoid:* "compliant"; "not emotion recognition".

**2. "A campus is close to education. Does your opt-in fix that?"**
No. "A hackathon on a university campus is close to the education setting, so the demo analyzes only the builder and
volunteers who opt in at the table. That opt-in is a data-protection courtesy, not an Art. 5(1)(f) exemption: if the
prohibition applied, consent would not lift it."
*Avoid:* "consent makes it legal".

**3. "Is this surveillance?"**
Built against it: no identity, embeddings or matching; ephemeral ids; frames dropped after feature extraction. "Not for employers, schools, proctoring or productivity scoring." The timeline is "pseudonymous, not
anonymous", so it stays in RAM and is deleted.
*Avoid:* "anonymous"; "room-only mode makes work use OK": "It reduces exposure; it does not make workplace or
classroom use lawful." `[G9]`

## Validity

**4. "How do you know it works on real people?"**
Not yet; it's our first limit. Trained and scored on simulated data only; real-data checks are one person at
one desk, about 25 ticks each, and every capture so far reads Focused.
*Proof:* typing 25/25, in-sample `[D1]`; fists alone 24/25 `[D2]`; cued slump 26/26 (`docs/MODEL_CARD.md` §5).
*Avoid:* "real data validates it"; "the live demo proves accuracy" `[D11]`.

**5. "You wrote the simulator. Doesn't context win by construction?"**
Partly; every results slide says so. The assumptions are published, the one noise knob stayed untuned at 1.0
across face and context, and context still loses to the face in a new state's first 30 seconds. We dropped the original
mock because context alone scored 100%.
*Proof:* 32.8% vs 44.3% `[E1]`; `[LM1]`. Sensitivity sweep not done.

**6. "What's your ground truth?"**
In simulation, the hidden state the generator drew at the window's last tick: exact but crisp, where real states
blend. No labeled real data exists, and self-report labels never enter the model.

## Privacy mechanics

**7. "What exactly is stored?"**
"Raw media is never written. By default the live pipeline writes one file, the decision log, deleted at session end."
Self-report labels, if used, expire after 24 h; features-only test captures are in the repo. The rest is 15 minutes
of numbers in RAM.
*Proof:* never stored: video, audio, transcripts, keypoints, face images or embeddings `[G10]`.
*Avoid:* "nothing is ever written to disk".

**8. "How can I verify that?"**
Run the scan and governance tests, then edit one digit in the audit log: verification fails. Live, frames
decoded equal frames purged, zero violations.
*Proof:* `make scan` → 0 violations `[G5]`; 23 governance tests pass `[G6]`;
`curl -s 127.0.0.1:8000/api/governance/verify`; live smoke 1099 = 1099 (`docs/BENCHMARKS.md`).
*Avoid:* replay counters as proof `[G2]`; the scan is a lint, not a proof.

**9. "Does anything leave the laptop?"**
The launcher runs the API, which owns camera and mic, and the narrator in a macOS sandbox that denies outbound
network except loopback. That's how we caught MediaPipe's built-in telemetry: unsandboxed, 4 HTTPS connections to Google;
sandboxed, none.
*Proof:* `docs/BENCHMARKS.md` "Live demo smoke".
*Avoid:* "fully offline", "Wi-Fi off" `[P13]`. Outside the launcher, the gap reopens.

**10. "`del frame` isn't erasure."**
Agreed: every frame and audio block is zero-filled in place after feature extraction, and survivors count as
violations. Copies inside native libraries, like GPU caches, are out of reach; we document that.
*Proof:* `[G1]`, `docs/GOVERNANCE.md` §3.5. *Avoid:* "frames are erased everywhere".

**11. "Isn't the emotion log the sensitive part?"**
Yes: inferred states "may count as health data (Art. 9); Valence treats them as sensitive." The log holds labels,
confidences and layer sums, no transcripts or arrays, hash-chained, deleted at session end.
*Proof:* `[G3]`; 24 h crash backstop, unlink is not a secure erase `[G4]`. *Avoid:* "tamper-proof"; say "tamper-evident".

## Model

**12. "Why LightGBM?"**
101 tabular features, often with the face or mic missing: boosted trees handle missing values natively, run on one
laptop thread, and give exact TreeSHAP attributions, summed into four layers.
*Proof:* a GRU-with-attention benchmark scored 0.759, below LightGBM at every history length (simulated; likely under-trained) `[L7]`.

**13. "Why not a bigger model, or a better face model?"**
A forced smile is a smile; a better face model narrows the gap but needs more face data, which we minimize. A bigger
model needs labeled real data we don't have.
*Proof:* simulated macro-F1: face snapshot 0.422, context without face 0.677, served 0.797 `[L1]` `[L2]`.

**14. "What is the mask override?"**
A conservative rule: face-only reads happy at 0.45 or more, context is at least 70% sure of any state other than
happy, and face features pull toward happy. Never in the first two minutes.
*Proof:* `[M1]`, tuned on simulated data; `tests/test_served.py`.
*Avoid:* "the model knows the smile is fake"; "it only fires on negative states".

**15. "Doesn't context lag?"**
Yes: in a new state's first 30 seconds context trails the face, then overtakes; hence the 60-second warm-up and
hysteresis. History helps up to about five minutes, then plateaus.
*Proof:* 32.8% vs 44.3% at 0–30 s, 94.0% vs 46.1% at 5+ min `[E1]`; macro-F1 0.649 → 0.791 `[A1]`; simulated.

**16. "What happens in a loud hall?"**
The model never sees absolute decibels, only the room against a 90-second baseline captured on arrival. On simulated
hall noise, Focused misread goes 17.7 → 18.7% at +75 dB, versus 20.8 → 28.1% for the previous model: real but modest.
*Proof:* `[V1]` (mostly venue-varied training); `[V5]` not rehearsed here; real typing + 72 dB: Focused 25/25 `[V4]`.
*Avoid:* "the venue baseline solves loud rooms".

**17. "What if the face is hidden or the light is bad?"**
Simulated, face hidden every tick: 0.68 macro-F1 versus 0.12 face-only; face degraded as at 10% light: 0.68 versus
0.25. Partly by construction: we trained on hidden and degraded copies.
*Proof:* `[R1]`, `[R3]`; face channel only, not a camera test. *Avoid:* "low light doesn't change the label".

## Fairness and limits

**18. "Demographic bias?"**
Not validated; it's a listed gap. The face-hidden result is decision stability, not fairness. Face valence is
relative to each person's own neutral, and the narrator may not mention demographics.
*Avoid:* "fair", "unbiased".

**19. "Someone with a flat affect?"**
Face valence is relative to your own first 20 seconds, and the face is one layer of four. But we haven't tested
atypical affect; the simulator assumes an exact neutral baseline. Not clinical either.
*Proof:* context-only floor 0.68, simulated `[R1]`. *Avoid:* "detects stress" as a diagnosis.

## The LLM narrator

**20. "Can it hallucinate?"**
Its drafts can, so they aren't trusted. A small local LLM sees only the fact sheet, never pixels; a validator checks
every number, state and driver, and failures fall back to a template within 2.5 s.
*Proof:* first try 93.3% vs 75.3% `[P6]` (300 sheets, ±3 points); 0 invented numbers in raw attempts `[P7]`; p95 1.9 s, AC power `[P8]`.
*Avoid:* "it can't make things up" (newer wording: 0.009 raw invented-number rate, `docs/NARRATOR.md`); say "passes our validator".

## Product and next steps

**21. "Who is this for?"**
Consented self-monitoring outside work and education: your laptop, your data, a reason for every call. The reusable
part: features-only boundary, purge counters, hash-chained log, sandbox.
*Avoid:* employers, schools, proctoring, productivity scoring.

**22. "What's next?"**
Labeled real data from other people and rooms, starting with a sustained forced-smile capture to confirm the override
live; then demographic validation, a noise-sensitivity sweep, a Wi-Fi-off rehearsal and a 60-minute soak.
