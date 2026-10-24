# Cloud C status: browser perception (`[web]`)

Owner: Cloud C. Scope: `frontend/src/perception/**`, `frontend/src/test/perception*`, `frontend/public/mediapipe/**`,
`frontend/e2e/**`, `scripts/export_perception_fixtures.py`, the MediaPipe dependencies. Times are America/New_York.
Newest entry last.

## Decisions

| Decision | Why |
|---|---|
| **MediaPipe Tasks Web 0.10.35** (`@mediapipe/tasks-vision`, `@mediapipe/tasks-audio`), pinned exactly | Published 2026-04-27, well past the 24 h guard. Every 1.x release (1.0.0 to 1.1.1) bundles a usage logger that POSTs to `odml.pa.googleapis.com/v1/log` every 60 s, created unconditionally in `createFromOptions`. Under our CSP it would be blocked, but it would still be a third-party attempt and a CSP violation. 0.10.35 has no such code: its bundle contains no external URL. The Python edge runs mediapipe 1.1.1 with the same `.task` models. |
| Observations keep NaN for "no reading" | Same as `contracts.py`; `obsToDict` writes null like `obs_to_dict`, and `obsFromDict` reads the Python JSONL recordings unchanged (tested on `data/real/capture_fists.jsonl`). |
| A `type` tag on each observation | The same key `obs_to_dict` writes, so TypeScript gets a discriminated union and recordings parse as-is. |
| `audio_blocks_purged` counts ~100 ms raw mic buffers | Same meaning as the Python edge, where the gate counts each capture part. 1 s blocks and resampled copies count under `derived_purged`. |
| `frontend/package-lock.json` re-synced | `npm ci` failed on the committed lock (ajv 6/8 entries missing). `npm install` fixed it while adding MediaPipe; `bun.lock` got the two MediaPipe entries by hand (bun can't reach the private registry its lock names for `@lovable.dev/vite-tanstack-config`). |

## Log

### 12:40 — Interface pushed

- `frontend/src/perception/`: `types.ts` (PersonObs / VideoObs / AudioObs mirrors, raw extras: 52 blendshapes and
  head yaw/pitch/roll in degrees, counters, status), `createPerception()` (stub: reports a "stub" degradation and
  emits nothing), `createMockPerception()` (10 Hz video, 1 s audio, deterministic per seed, a 180 s script per
  person: focused, upbeat, frustrated, worn out), `openMediaStream()` (echo cancellation, noise suppression and AGC
  off; falls back to camera-only or mic-only).
- `faceValence.ts`: port of `face_valence.py` (formula, calibrator, Hungarian face assignment), used by the mock.
- Tests: `src/test/perception-interface.test.ts` (types, JSONL round trip, mock cadence and content, media fallbacks).
- Next: the pure ports with parity fixtures, then the MediaPipe pipeline and the headless-Chromium end-to-end test.
