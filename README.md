# Valence Context Console

Build a single-page web app called "Valence Console": the judge-facing display for Valence, a privacy-preserving AI system that infers a person's emotional state from room context (noise, light, time since last break) and body behavior (clenched fists, slumping, laughter, compliments) instead of relying on their face. A local Python backend does all the AI; this app only displays what the backend streams and sends a few control commands. You choose the layout and visual style; it must be legible from a few meters away during a live demo.

DATA SOURCE
- A Mock mode (default ON) that generates realistic fake messages matching the schemas below: cycle through all four states over ~2 minutes, include a "mask override" moment, and stream governance purge events every 5 s.
- A Live mode that connects to a configurable backend base URL (default http://localhost:8000; WebSocket ws://localhost:8000/ws/events). Auto-reconnect and show connection state.
- Put all message types (TypeScript types) and handling in one module so switching mock -> live changes nothing else.

THE FOUR EMOTIONAL STATES (always show all four probabilities)
Focused -> "Focused"; Frustrated_Masked -> "Frustrated / Masked Stress"; Exhausted -> "Exhausted"; Collaborative_Happy -> "Collaborative / Happy".

WHAT THE APP MUST DISPLAY
1. Live webcam view from the browser's own camera (getUserMedia; preview only, never recorded, stored, or uploaded) with per-person bounding boxes and labels ("Person #1") from `tracks` messages (normalized 0-1 [x1,y1,x2,y2]). With no camera or in replay mode, draw the boxes on a blank canvas.
2. Governance purge log: a scrolling terminal-style log of `governance` messages (e.g. "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.") plus running counters: frames purged, audio blocks purged, transcripts purged, violations.
3. Each person's detected state + confidence (e.g. "Person #1: Frustrated / Masked Stress — 88%").
4. Probability of all four states per person.
5. The face-only baseline prediction next to the context prediction, and a clear flag when mask_override is true ("Context corrected the facial mask").
6. The audit explanation. A `prediction` message carries a template explanation; a later `explanation` message with the same event_id replaces it with the LLM-written text. Show the source (llm / llm_retry / template).
7. Top drivers grouped by layer (Ambient Physics, Time Dynamics, Somatic & Social, Facial Surface), each with its display text and contribution (positive supports the prediction, negative opposes it).
8. Current state-vector values per person: noise (dB), light (lux), hours since break, time of day, clenched fists (last 5 min), posture slump, social withdrawal, compliments (15 min), laughter (15 min), face visible, facial valence.
9. A 15-minute timeline of state-vector metrics and predicted states, loaded from GET /api/history?minutes=15 and then extended live, with window warm-up/fill status.
10. System status from `status` messages: camera, mic, mode (live/replay), FPS, loop latency, narrator status, offline indicator.
11. Data-retention summary from GET /api/retention.
12. Evaluation results from GET /api/eval/summary and images at GET /api/eval/figures/{name}.png: accuracy vs history chart, robustness charts (face hidden / low light), confusion matrices, calibration plot, metrics table. Use placeholder data in mock mode.

CONTROLS (sent to the backend)
- Start / Stop / Pause (privacy kill switch): POST /api/control/{start|stop|pause}
- Log a break: POST /api/events/break
- Replay scenario picker: POST /api/replay/{focused|frustrated_masked|exhausted|collaborative_happy|full_demo}
- Toggles "Simulate face hidden" and "Simulate low light": POST /api/sim/face_occlusion and POST /api/sim/low_light with {"enabled": true|false}

WEBSOCKET MESSAGES (JSON, discriminated by "type")
{"type":"tracks","t":1739.2,"persons":[{"id":1,"label":"Person #1","bbox":[0.12,0.20,0.41,0.95]}]}
{"type":"governance","t":1740.0,"tick":348,"message":"[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.","frames_purged":17402,"audio_blocks_purged":3480,"transcripts_purged":212,"violations":0,"live_raw_buffers":0}
{"type":"state_vector","t":1740.0,"person_id":1,"ambient_db":72.4,"light_lux":210,"hours_since_break":4.2,"time_of_day":"15:40","clenched_fists_5m":3,"compliments_15m":0,"laughter_15m":0,"posture_slump":0.35,"social_withdrawal":0,"face_visible":true,"facial_valence":0.62,"window_fill_min":12.5}
{"type":"prediction","event_id":"p1-000348","t":1740.0,"person_id":1,"label":"Frustrated_Masked","label_display":"Frustrated / Masked Stress","confidence":0.88,"probabilities":{"Focused":0.04,"Frustrated_Masked":0.88,"Exhausted":0.05,"Collaborative_Happy":0.03},"face_only":{"label":"Collaborative_Happy","confidence":0.71},"mask_override":true,"warming_up":false,"layers":[{"name":"Somatic & Social","contribution":0.41,"drivers":[{"feature":"clenched_fists_count_5m","display":"3 clenched-fist events in the last 5 minutes","value":3,"unit":"events","shap":0.38}]},{"name":"Ambient Physics","contribution":0.22,"drivers":[{"feature":"ambient_db_mean_15m","display":"average noise of 72 dB over 15 minutes","value":72.4,"unit":"dB","shap":0.22}]},{"name":"Time Dynamics","contribution":0.15,"drivers":[{"feature":"hours_since_break","display":"4.2 hours since the last break","value":4.2,"unit":"h","shap":0.15}]},{"name":"Facial Surface","contribution":-0.18,"drivers":[{"feature":"face_last","display":"smiling expression (valence +0.62)","value":0.62,"unit":"","shap":-0.18}]}],"explanation":{"text":"Model predicted Frustrated / Masked Stress (88%). Person #1 appears to smile, but 3 clenched-fist events, 72 dB of sustained noise and 4.2 hours without a break point to masked stress.","source":"template"}}
{"type":"explanation","event_id":"p1-000348","person_id":1,"text":"…LLM-written paragraph…","source":"llm","latency_ms":1420}
{"type":"status","mode":"live","camera":true,"mic":true,"fps":9.6,"loop_latency_ms":84,"narrator":"ready","offline":true,"uptime_s":1820}

CONSTRAINTS
No authentication, no database, no user accounts, no analytics/telemetry, and never store or transmit camera frames.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/2916895f-5304-40be-adf9-40735a8d2a74).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
