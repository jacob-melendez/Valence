# Runtime golden generators

These scripts produce the Python-side expectations that `frontend/src/test/engine-runtime.test.ts` pastes as
literals: digests, audit chain heads, end states and a few exact values. Every generator except `specs.py`
(which only builds input specs) drives the real `valence/runtime/engine.py` Runtime. The fixtures in
`frontend/src/test/fixtures/engine/` come from `scripts/export_web_golden.py` and are a separate step.

| Script | Produces | Test block |
| --- | --- | --- |
| `extra_runs.py` | For each of the five extra runs (`live_capture`, `recording_break_and_drop`, `overlays`, `evict`, `operator`): tick, message and prediction counts, the sha256 of the canonical text, audit size and head, operator entries, gap ticks, latest ids and hello | `EXTRA` ("runtime against extra Python runs (live source, overlays, eviction, operator actions)") |
| `specs.py` | The six edge-case specs (`sv`, `room`, `op`, `w0`, `drop0`, `noroom`), built the same way as the test's `SPECS` | input to `edge_cases.py` |
| `edge_cases.py` | For each spec, run with a stub predictor: the digest, audit size and head, end state, and narrator calls on stdout; `--out` adds every tick's canonical text and the audit entries, where the other asserted values are | `SPEC_EXPECT` and the narrator, time-of-day, state-vector, confidence, simulated-field and break assertions in "runtime against Python on inputs the golden runs never produce" |
| `perturb.py` | Golden runs perturbed by a xorshift32 stream and run with the served predictor: ticks, predictions, digest, audit size and head | `PERTURBED` ("runtime against Python on perturbed golden runs (served predictor)") |
| `tracks.py` | `push_obs` results: the Tracks message and fps after each video frame, then fps, mic and camera after an audio block | "publishes tracks and measures fps exactly as push_obs does" |
| `check.py` | Runs all of the above and compares their output with the literals in the test file | — |
| `_common.py` | Shared code: repo root on `sys.path`, `TZ=UTC`, `canon()`, stub clock, gate and bus | — |

The canonical text is the same in every script: one line per tick, each line `canon(list of that tick's messages)`.
Numbers are written as `repr(float(x))`, strings with `json.dumps`, and dict keys stay in insertion order. The
test builds the same text in TypeScript and compares sha256 digests.

## Rerun and compare

From the repo root, using the project's Python (the scripts find the repo from their own path, so with absolute
paths they also run from any other directory):

```
.venv/bin/python -B scripts/golden_runtime/check.py   # every generator against the test's literals; exit 1 on a mismatch
cd frontend && npx vitest run src/test/engine-runtime.test.ts
```

Or run one script and read its output:

```
.venv/bin/python -B scripts/golden_runtime/extra_runs.py [--out DIR]   # JSON summary; --out writes DIR/<run>.txt
.venv/bin/python -B scripts/golden_runtime/specs.py [--out DIR]        # specs as JSON; --out writes DIR/<name>.json
.venv/bin/python -B scripts/golden_runtime/edge_cases.py [--out DIR] [SPEC.json ...]
.venv/bin/python -B scripts/golden_runtime/perturb.py [--out DIR]      # one JSON line per run; --out writes DIR/<run>.txt
.venv/bin/python -B scripts/golden_runtime/tracks.py
```

The scripts print to stdout and write files only when you pass `--out`; the one exception is the throwaway audit
directory each Runtime gets from Python's `tempfile` (under `$TMPDIR`), which is deleted when the run ends. When a
digest differs, write the canonical text with `--out` and diff it line by line (one line per tick; `edge_cases.py`
puts the lines in the `"ticks"` array of its JSON) against the TypeScript side to find the first tick that
differs. The `op` spec deliberately logs an `audit append failed ... NaN/inf not allowed` traceback on stderr:
it tests a refused audit payload.

Reruns give identical output. The process runs in `TZ=UTC`. The audit, aggregator and engine clocks are patched to
fixed clocks built on the scenario clocks (`WALL0 = 1792850400.0`, `MONO0 = 1000.0`, as in
`scripts/export_web_golden.py`; `tracks.py` starts the monotonic clock at 10.0, as the test does). The
`np.random.normal` draws are fixed. `extra_runs.py` and `edge_cases.py` fix the session ids; `perturb.py` and
`tracks.py` keep the Runtime's time-stamped default, which never reaches their output.

## When to rerun

Rerun after any change under `valence/` (runtime, aggregator, audit, models, scenarios) or to `artifacts/`. Also
rerun after `scripts/export_web_golden.py` regenerates `runtime.json.gz` or `aggregator.json.gz`, because
`perturb.py` and `extra_runs.py` read those fixtures. If `check.py` reports a difference and the Python change was
intended, paste the new values into the test. If the change was not intended, the difference is a regression.
