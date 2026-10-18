# Lovable frontend prompt (v2)

Paste everything below the line into Lovable. v2 changes from the chat version: states come from
`GET /api/states` (4 or 6), box overlay metadata (`frame_w`, `frame_h`, `mirrored`), data provenance
(`source`, `simulated_fields`), nullable face-only result, and reset / room-only controls.

Run the exported app locally next to the backend for the demo (a hosted page calling `localhost` triggers
Chrome's local-network prompt and may be blocked by Safari).

---

Build a single-page web app called "Valence Audit Console": the judge-facing display for Valence, a privacy-preserving AI system that infers a person's emotional state from room context (noise, light, time since last break) and body behavior (clenched fists, slumping, head in hands, laughter, compliments) instead of relying on their face. A local Python backend does all the AI; this app only displays what the backend streams and sends a few control commands. You choose the layout and visual style; it must be legible from a few meters away during a live demo.

DATA SOURCE
- A Mock mode (default ON) that generates realistic fake messages matching the schemas below: cycle through all states over ~2 minutes, include a "mask override" moment, a moment where the face is hidden, and stream governance purge events every 5 s.
- A Live mode that connects to a configurable backend base URL (default http://127.0.0.1:8000, not localhost, because macOS resolves localhost to ::1 first; overridable with `?api=` and remembered; WebSocket ws://127.0.0.1:8000/ws/events). Auto-reconnect and show connection state.
- Put all message types (TypeScript types) and handling in one module so switching mock -> live changes nothing else.

EMOTIONAL STATES
Do not hardcode them. Load the list from GET /api/states (and the `hello` message): [{"id":"Focused","display":"Focused","tier":"P0"}, ...]. There are 4 or 6 states. Always show the probability of every state. Use `label_display` from each prediction as the visible name (it changes, e.g. "Frustrated" vs "Frustrated / Masked Stress", "Upbeat" for one person).

WHAT THE APP MUST DISPLAY
1. Live webcam view from the browser's own camera (getUserMedia; preview only, never recorded, stored, or uploaded) with per-person bounding boxes and labels ("Person #1") from `tracks` messages. Boxes are normalized 0-1 [x1,y1,x2,y2] relative to a frame of `frame_w` x `frame_h`; if `mirrored` is true, mirror the boxes to match. With no camera or in replay mode, draw the boxes on a blank canvas.
2. Governance purge log: a scrolling terminal-style log of `governance` messages (e.g. "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.") plus running counters: frames decoded, frames purged, audio blocks purged, transcripts purged, violations, live raw buffers, retention.
3. Each person's detected state + confidence (e.g. "Person #1: Frustrated / Masked Stress — 88%").
4. Probability of every state per person.
5. The face-only baseline prediction next to the context prediction (it may be null = "no face visible"), and a clear flag when mask_override is true ("Context corrected the facial mask").
6. The audit explanation. A `prediction` message carries a template explanation; a later `explanation` message with the same event_id replaces it with the LLM-written text. Show the source (llm / llm_retry / template).
7. Top drivers grouped by layer (Ambient Physics, Time Dynamics, Somatic & Social, Facial Surface), each with its display text and contribution (positive supports the prediction, negative opposes it).
8. Current state-vector values per person: noise (dB), light (lux), hours since break, time of day, clenched fists (last 5 min), posture slump, head in hands, social withdrawal, compliments (15 min), laughter (15 min), face visible, facial valence.
9. A 15-minute timeline of state-vector metrics and predicted states, loaded from GET /api/history?minutes=15 and then extended live, with window warm-up/fill status.
10. Data provenance: when a message has `source` other than "live" or a non-empty `simulated_fields`, show it clearly (e.g. "REPLAY", "SIMULATED: hours_since_break").
11. System status from `status` messages: camera, mic, mode (live/replay/hybrid), FPS, loop latency, narrator status, offline indicator, degraded sensors, purge violation alarm.
12. Data-retention summary from GET /api/retention.
13. Evaluation results from GET /api/eval/summary and images at GET /api/eval/figures/{name}.png: accuracy vs history chart, robustness charts (face hidden / low light), confusion matrices, calibration plot, metrics table. Use placeholder data in mock mode.
14. Room baseline status: the venue baseline the model reads the room against, from `room_baseline` in GET /api/health or from GET /api/room/baseline: {"mode":"captured","complete":true,"target_samples":90,"db_samples":90,"lux_samples":90,"db_median":68.4,"db_iqr":2.1,"lux_median":410.0}. Show the mode (auto / capturing / captured), the dB median and the lux median, with capture progress (db_samples of target_samples) while capturing.

CONTROLS (sent to the backend)
- Start / Stop / Pause (privacy kill switch): POST /api/control/{start|stop|pause}
- Log a break: POST /api/events/break
- Reset people between visitors: POST /api/demo/reset
- Replay scenario picker (list from GET /api/scenarios): POST /api/replay/{scenario}
- Toggles "Simulate face hidden" and "Simulate low light": POST /api/sim/face_occlusion and POST /api/sim/low_light with {"enabled": true|false}
- Toggle "Room-only mode" (one aggregate label, no per-person inference): POST /api/mode with {"room_only": true|false}
- Button "Capture room baseline (90 s)": POST /api/room/baseline with {"seconds": 90}; tell the operator to sit neutral and quiet until the mode reads "captured". Button "Reset": POST /api/room/baseline/reset (back to automatic).

WEBSOCKET MESSAGES (JSON, discriminated by "type")
{"type":"hello","contract_version":"1.0.0","mode":"live","room_only":false,"states":[{"id":"Focused","display":"Focused","tier":"P0"},{"id":"Frustrated_Masked","display":"Frustrated / Masked Stress","tier":"P0"},{"id":"Exhausted","display":"Exhausted","tier":"P0"},{"id":"Collaborative_Happy","display":"Collaborative / Happy","tier":"P0"}],"feature_version":"v1","frame_w":1280,"frame_h":720,"mirrored":false}
{"type":"tracks","t":1739.2,"capture_ts_ms":1739200,"frame_w":1280,"frame_h":720,"mirrored":false,"persons":[{"id":1,"label":"Person #1","bbox":[0.12,0.20,0.41,0.95],"fist_active":false,"slump":false}]}
{"type":"governance","t":1740.0,"tick":348,"message":"[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.","frames_decoded":17402,"frames_purged":17402,"audio_blocks_purged":3480,"transcripts_purged":212,"violations":0,"live_raw_buffers":0,"audit_hash":"9f2c…","retention_s":86400}
{"type":"state_vector","t":1740.0,"person_id":1,"ambient_db":72.4,"light_lux":210,"hours_since_break":4.2,"time_of_day":"15:40","clenched_fists_5m":3,"compliments_15m":0,"laughter_15m":0,"posture_slump":0.35,"head_in_hands":0.0,"social_withdrawal":0,"face_visible":true,"facial_valence":0.62,"window_fill_min":12.5,"source":"live","simulated_fields":[]}
{"type":"prediction","event_id":"p1-000348","t":1740.0,"person_id":1,"label":"Frustrated_Masked","label_display":"Frustrated / Masked Stress","confidence":0.88,"probabilities":{"Focused":0.04,"Frustrated_Masked":0.88,"Exhausted":0.05,"Collaborative_Happy":0.03},"face_only":{"label":"Collaborative_Happy","confidence":0.71},"mask_override":true,"warming_up":false,"layers":[{"name":"Somatic & Social","contribution":0.41,"drivers":[{"feature":"fist_cnt_5m","display":"3 clenched-fist events in the last 5 minutes","value":3,"unit":"events","shap":0.38,"level":"high","effect":"supports","scope":"person"}]},{"name":"Ambient Physics","contribution":0.22,"drivers":[{"feature":"db_mean_15m","display":"average noise of 72 dB over 15 minutes","value":72.4,"unit":"dB","shap":0.22,"level":"high","effect":"supports","scope":"room"}]},{"name":"Time Dynamics","contribution":0.15,"drivers":[{"feature":"hours_since_break","display":"4.2 hours since the last break","value":4.2,"unit":"h","shap":0.15,"level":"high","effect":"supports","scope":"person"}]},{"name":"Facial Surface","contribution":-0.18,"drivers":[{"feature":"face_now","display":"smiling expression (valence +0.62)","value":0.62,"unit":"","shap":-0.18,"level":"high","effect":"opposes","scope":"person"}]}],"data_quality":{"face_visible_frac":0.9,"lux":210,"ood_features":[]},"explanation":{"text":"Model predicted Frustrated / Masked Stress (88%). Person #1 appears to smile, but 3 clenched-fist events, 72 dB of sustained noise in the room and 4.2 hours without a break point to masked stress.","source":"template"},"source":"live","simulated_fields":[]}
{"type":"explanation","event_id":"p1-000348","person_id":1,"text":"…LLM-written paragraph…","source":"llm","latency_ms":540}
{"type":"status","mode":"live","camera":true,"mic":true,"fps":9.6,"loop_latency_ms":45,"narrator":"ready","offline":true,"uptime_s":1820,"degraded":[],"purge_violation":false}
{"type":"heartbeat","t":1741.0}

CONSTRAINTS
No authentication, no database, no user accounts, no analytics/telemetry, and never store or transmit camera frames.
