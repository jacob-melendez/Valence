# Valence Web: the engine in the browser

Valence Web runs the whole Valence pipeline in a web page:

1. camera and microphone;
2. perception;
3. the 15-minute rolling window;
4. the 123 context features;
5. the served LightGBM models;
6. TreeSHAP explanations;
7. the template narration;
8. the hash-chained audit log;
9. the emotion-v1 map.

Frames and audio are processed in memory and dropped, and the deployed page's CSP (`connect-src 'self'`) forbids
sending data anywhere. Built during Hack OHI/O 2026. The work log is `docs/HACKOHIO_LOG.md`.

| Link | What it does |
|---|---|
| `/?source=browser` | Live: the camera and microphone run through the full engine in the page. |
| `/?demo=tour` | No camera: Cloud B's scripted, simulated 50-minute walk through the 10 states, replayed through the same engine at 10× speed. |
| `/?source=browser&perception=mock` | No camera: Cloud C's deterministic mock perception, for testing. |

## How it is put together

```
getUserMedia (camera 16:9, raw mic)                    lib/browser-source.ts   (one stream, shared with the preview)
  -> perception (MediaPipe Tasks Web in a Worker)       perception/**           (Cloud C; PersonObs/VideoObs/AudioObs)
  -> Runtime.pushObs -> Aggregator (5 s TickRecords)    engine/runtime.ts, aggregator.ts
  -> RollingWindow -> 123 features (+ 31 emotion-v1)    engine/window.ts, features.ts, emotion-features.ts
  -> ServedPredictor: clip, M2 + M0, T-softmax, OOD,    engine/served.ts, lgbm.ts, treeshap.ts, smoother.ts
     TreeSHAP layers, mask override, smoother
  -> describe / explainer / template text               engine/describe.ts, explainer.ts, template.ts
  -> audit chain (SHA-256, Python json.dumps)           engine/audit.ts, sha.ts
  -> emotion map: 10-state model, map position          engine/emotion-model.ts (Cloud B's export)
  -> governance / state_vector / prediction messages    engine/browser-engine.ts -> lib/valence.ts store -> panels
```

- **Python is the reference.** Every engine module is a port of the Python module it names in its JSDoc. Golden
  fixtures exported from the Python engine check the ports in vitest.
- **Messages.** The console's panels are unchanged: the in-browser source emits the same messages the desktop
  backend sends over its WebSocket (`docs/API.md` §3).
- **Models.** The model files are served at `/models/` straight from `artifacts/` (`scripts/model-files.mjs`).
  MediaPipe's WASM and models are self-hosted at `/mediapipe/` (Cloud C's prebuild).

## Parity with the Python engine

The figures below are the largest absolute error against Python, from the vitest suite (12 engine test files, about
400 tests; 860 frontend tests in total).

| Quantity | Max abs error | Notes |
|---|---|---|
| numpy / CPython numerics, 9,244 cases | 0 | pairwise sum, nan statistics, percentile, `round`, float formatting, `repr` |
| SHA-1 / SHA-256 | 0 | |
| 123 features, 145,017 values | 1.1e-16 (1 ulp) | 118 features bit-exact. The rest (tod_sin/cos, lux_room_*) differ by 1 ulp: V8 vs glibc / numpy SIMD `sin`, `cos`, `log2` |
| Aggregator TickRecords, 14 cases | 7.1e-15 on ambient_db | 0 on the other 29 fields. The difference is the pow/log10 Leq |
| LightGBM raw scores (M2, M0; 4 and 6 states) | 0 | |
| TreeSHAP contributions, face_push | 0 | path weights from row counts, as LightGBM does |
| Served probabilities | 1.1e-16 (2 ulp) | `Math.exp` vs `np.exp`. No label or rounding flip in 984 predictions |
| Explanations | 0 | 12,644 describe strings, layers, drivers, fact sheets and template texts, all identical |
| Audit chain | 0 | 191 entries: identical JSON blobs and hashes |
| Runtime messages | 0 | 2,964 messages over 996 ticks |
| Emotion-v1 features | 0 | 1,454 checkpoints and a 2,196-window fuzz |
| Emotion model | raw 0, p 1.1e-16 | through both of Cloud B's model formats |
| Emotion tour, end to end | 1e-9 | 600 ticks: labels identical on 600/600 |

**Speed (node, this container).** M2 plus SHAP takes 6.1 ms per vector. A 3-person tick averages 26 ms (p95 43 ms)
against a 5 s tick. The emotion model adds about 0.6 ms per person.

## Reproducing the fixtures

From the repo root, with the project's Python:

```
.venv/bin/python -B scripts/export_web_golden.py          # features, aggregator, models, explain, audit, runtime
.venv/bin/python -B scripts/export_web_golden_num.py      # numerics
.venv/bin/python -B scripts/export_web_golden_smoother.py # smoother (reads the runtime and models fixtures)
.venv/bin/python -B scripts/export_web_golden_lgbm.py     # the m0 dump_model fixture
.venv/bin/python -B scripts/export_web_golden_emotion.py  # emotion-v1 windows, inputs and outputs
.venv/bin/python -B scripts/golden_runtime/check.py       # the runtime test's pasted literals (extra runs, edge cases, tracks)
cd frontend && npx vitest run
```

Reruns are byte-identical: the timezone is UTC, the clocks are fixed and gzip uses mtime 0. The fixtures go stale
when the simulator or scenarios change, so rerun the exporters after any change under `valence/` or `scenarios/` and
check `git diff`.

## Audit log: verifiable across languages

In browser mode the governance panel shows the tab's audit chain: the number of entries, the head hash, and whether
every hash recomputes. **Download audit log** saves it as the desktop backend's JSONL. The file holds labels,
confidences, layer sums and operator actions, never media. The desktop's own verifier checks it offline:

```
python scripts/audit_verify.py valence-audit-<time>.jsonl --head <first 16 hex of the head>
```

Checked on a 40-tick export from the TypeScript engine:
- the untouched file: "15 entries checked, chain intact";
- the same file with one confidence edited: "line 5: seq 4: hash mismatch (the entry was edited)", exit code 1.

## Known differences and limits

- **Last-bit math.** V8's `Math.sin/cos/log2/log10/pow/exp` can differ from glibc and from numpy's SIMD kernels in
  the last bit. This is visible only at ulp level and changes no label in any fixture.
- **Perception is not the Python edge.** Pose comes from BlazePose, not YOLO, so Cloud C smooths the keypoints and
  computes motion from the shoulders only. Lux comes from the camera's auto-exposed luma. Six served lux features
  depend on absolute lux scale, which an uncalibrated camera does not give.
- **No speech recognition.** The browser has no speech recognizer, so the compliments count stays 0. The narration
  is always the template; the LLM narrator stays desktop-only.
- **Simulated training data.** The models are trained on simulated sessions. Real-camera checks are listed in
  `frontend/e2e/README.md` (Cloud C).
