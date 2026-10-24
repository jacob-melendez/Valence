/**
 * runtime.ts against the Python runtime (valence/runtime/engine.py Runtime, valence/api/app.py hello and the
 * operator endpoints).
 *
 * - runtime.json.gz (scripts/export_web_golden.py): five scenario runs stepped tick by tick with the served
 *   predictor (core and the 6-state set, room-only, mask override, a break, two people). Every published
 *   message must equal Python's model_dump() exactly: same keys in the same order, strings identical, numbers
 *   Object.is-equal (so -0.0 counts). The audit chain must match entry for entry (hashes and hashed text), and
 *   hello must match too.
 * - Extra runs (EXTRA below): the live source (use_live, an operator baseline capture that completes, and
 *   purge-gate counters with a source switch mid-run), use_recording on a synthetic stream with a break and
 *   drops, the SIMULATED overlays, eviction, and a sequence of operator actions (seed, mode, break, reset,
 *   baseline reset/capture, use_scenario). Python ran the same sequence; each run's messages are reduced to a
 *   canonical text (numbers as repr(float(x)), strings as json.dumps, keys in emission order) whose sha256 is
 *   pasted below together with the audit chain head. The expected values come from
 *   scripts/golden_runtime/extra_runs.py (README.md there; check.py compares every generator with this file),
 *   which drives the Python Runtime through exactly the steps of EXTRA_RUNS, on the WALL0/MONO0 clocks of
 *   runtime.json.gz, with a purge-gate stub and np.random.normal patched to the draws of fixedDraws(). Rerun it
 *   whenever valence/ changes.
 * - Behaviour without a Python counterpart (null records, onMessage, record copies, refused audit payloads,
 *   the tracks throttle) and a 3-person timing.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  auditBlob,
  entryBody,
  pyJsonDumps,
  verifyChain,
  type AuditEntry,
  type AuditPayload,
} from "@/engine/audit";
import {
  STATE_SETS,
  makeSettings,
  makePersonObs,
  makeVideoObs,
  obsFromDict,
  tickRecordFromDict,
  tickRecordToDict,
  type Layer,
  type Obs,
  type Settings,
  type TickRecord,
} from "@/engine/contracts";
import { pyRepr } from "@/engine/num";
import { replayInto } from "@/engine/replay";
import {
  Runtime,
  ScenarioSource,
  type PurgeCounters,
  type RuntimeOptions,
  type TickMessage,
} from "@/engine/runtime";
import {
  ServedPredictor,
  type ClipBounds,
  type PredictionOutput,
  type Predictor,
  type ServedMeta,
} from "@/engine/served";
import { sha256Hex } from "@/engine/sha";
import { artifactBytes, artifactJson, artifactText, loadFixture } from "./engine-fixtures";

// ---------- Fixtures ----------

type StateSet = "core" | "all";

interface FixtureEntry {
  seq: number;
  ts: number;
  kind: string;
  payload: unknown;
  prev: string;
  hash: string;
  clock: number;
  blob: string;
  floats: string[];
}

interface FixtureTick {
  wall: number;
  mono: number;
  records: Record<string, unknown>[];
  messages: Record<string, unknown>[];
}

interface RuntimeRun {
  name: string;
  scenario: string;
  state_set: StateSet;
  room_only: boolean;
  model_id: string;
  settings: Settings;
  wall0: number;
  mono0: number;
  session_id: string;
  hello: Record<string, unknown>;
  ticks: FixtureTick[];
  audit: FixtureEntry[];
}

const runtime = loadFixture<{ version: number; runs: RuntimeRun[] }>("runtime");
const aggregatorFx = loadFixture<{ cases: { name: string; obs: Record<string, unknown>[] }[] }>(
  "aggregator",
);

const WALL0 = 1792850400.0; // scripts/export_web_golden.py WALL0 (Saturday 14:00 UTC)
const MONO0 = 1000.0;
const ARTIFACT_DIR: Record<StateSet, string> = { core: "", all: "all/" };

function servedFor(stateSet: StateSet): ServedPredictor {
  const d = ARTIFACT_DIR[stateSet];
  return new ServedPredictor({
    m2: artifactBytes(`${d}m2.txt`),
    m0: artifactText(`${d}m0.txt`),
    meta: artifactJson<ServedMeta>(`${d}model_meta.json`),
    clipBounds: artifactJson<ClipBounds>(`${d}clip_bounds.json`),
  });
}

const SERVED: Record<StateSet, ServedPredictor> = {
  core: servedFor("core"),
  all: servedFor("all"),
};

function runNamed(name: string): RuntimeRun {
  const run = runtime.runs.find((r) => r.name === name);
  if (run === undefined) throw new Error(`no runtime run ${name}`);
  return run;
}

function obsOf(caseName: string): Obs[] {
  const c = aggregatorFx.cases.find((x) => x.name === caseName);
  if (c === undefined) throw new Error(`no aggregator case ${caseName}`);
  return c.obs.map((d) => obsFromDict(d));
}

// ---------- Comparison helpers ----------

/**
 * Differences between `got` and Python's value `want` (parsed JSON): key order, types, strings, and numbers
 * with Object.is (JSON.parse keeps "-0.0" as -0). A field missing on either side is reported.
 */
function strictDiff(got: unknown, want: unknown, path = ""): string[] {
  if (typeof want === "number") {
    return typeof got === "number" && Object.is(got, want)
      ? []
      : [`${path}: ${String(got)} != ${want}`];
  }
  if (want === null || typeof want !== "object") {
    return got === want ? [] : [`${path}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`];
  }
  if (Array.isArray(want)) {
    if (!Array.isArray(got)) return [`${path}: not a list`];
    if (got.length !== want.length) return [`${path}: length ${got.length} != ${want.length}`];
    return want.flatMap((w, i) => strictDiff(got[i], w, `${path}[${i}]`));
  }
  if (got === null || typeof got !== "object" || Array.isArray(got))
    return [`${path}: not an object`];
  const gk = Object.keys(got);
  const wk = Object.keys(want);
  if (gk.join("\u0000") !== wk.join("\u0000")) return [`${path}: keys [${gk}] != [${wk}]`];
  return wk.flatMap((k) =>
    strictDiff(
      (got as Record<string, unknown>)[k],
      (want as Record<string, unknown>)[k],
      `${path}.${k}`,
    ),
  );
}

/** Objects with their keys sorted, recursively (json.dumps(sort_keys=True) order). */
function sortedKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortedKeys);
  if (v === null || typeof v !== "object") return v;
  const o = v as Record<string, unknown>;
  return Object.fromEntries(
    Object.keys(o)
      .sort()
      .map((k) => [k, sortedKeys(o[k])]),
  );
}

/**
 * An audit entry with only the fields Python's file holds (the fixture adds clock/blob/floats). Python reads
 * entries back from a file written with sort_keys=True, so payload keys are compared sorted; the hashed text,
 * checked separately, covers the order json.dumps uses.
 */
function entryFields(e: AuditEntry | FixtureEntry): Record<string, unknown> {
  const payload = sortedKeys(e.payload);
  return { seq: e.seq, ts: e.ts, kind: e.kind, payload, prev: e.prev, hash: e.hash };
}

/**
 * The canonical text the extra-run generator writes for a message: numbers as Python repr(float(x)) (so ints
 * and floats print alike), strings as json.dumps, dict keys in their (emission) order.
 */
function canon(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return pyRepr(v);
  if (typeof v === "string") return pyJsonDumps(v);
  if (Array.isArray(v)) return `[${v.map(canon).join(",")}]`;
  if (typeof v === "object")
    return `{${Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${pyJsonDumps(k)}:${canon(x)}`)
      .join(",")}}`;
  throw new TypeError(`cannot canonicalise ${typeof v}`);
}

// ---------- Golden runs ----------

const perf = { steps: 0, ms: 0, messages: 0, notes: [] as string[] };

interface Replayed {
  rt: Runtime;
  got: TickMessage[][];
}

/** Step a fixture run through a fresh Runtime the way the exporter drove Python's (source set directly). */
function replayRun(run: RuntimeRun): Replayed {
  let now = run.wall0;
  const rt = new Runtime(makeSettings(run.settings), SERVED[run.state_set], {
    sessionId: "session-golden",
    clock: () => now, // the exporter patched the audit clock to each tick's wall
    tz: "UTC",
  });
  // rt.source = ScenarioSource(...); rt.mode = "replay"; rt.room_only = ... (no use_scenario: no audit entries)
  rt.sourceLabel = "replay";
  rt.mode = "replay";
  rt.roomOnly = run.room_only;
  const got: TickMessage[][] = [];
  for (const tick of run.ticks) {
    now = tick.wall;
    const records = tick.records.map(tickRecordFromDict);
    const t0 = performance.now();
    got.push(rt.step(records, tick.wall, tick.mono));
    perf.ms += performance.now() - t0;
    perf.steps += 1;
  }
  perf.messages += got.reduce((n, m) => n + m.length, 0);
  return { rt, got };
}

describe("runtime against the golden Python runs", () => {
  it("covers every kind of message and branch the runs were chosen for", () => {
    expect(runtime.version).toBe(1);
    expect(runtime.runs.map((r) => r.name)).toEqual([
      "frustrated_masked",
      "full_demo",
      "context_blindness",
      "collaborative_happy_room_only",
      "overwhelmed",
    ]);
    const preds = runtime.runs.flatMap((r) =>
      r.ticks.flatMap((t) => t.messages.filter((m) => m["type"] === "prediction")),
    );
    expect(preds.some((p) => p["mask_override"] === true)).toBe(true);
    expect(preds.some((p) => p["warming_up"] === true)).toBe(true);
    expect(preds.some((p) => p["person_id"] === 0)).toBe(true); // room-only
    expect(preds.some((p) => p["person_id"] === 2)).toBe(true); // two people
    // ticks where nobody is present publish only the governance message
    expect(runtime.runs.some((r) => r.ticks.some((t) => t.messages.length === 1))).toBe(true);
  });

  for (const run of runtime.runs) {
    describe(run.name, () => {
      let r: Replayed;
      beforeAll(() => {
        r = replayRun(run);
      }, 120_000);

      it("publishes every message of every tick exactly as Python did, in order", () => {
        expect(r.rt.predictor.modelId).toBe(run.model_id);
        const bad: string[] = [];
        run.ticks.forEach((tick, i) => {
          const mine = r.got[i] ?? [];
          if (mine.length !== tick.messages.length) {
            bad.push(`tick ${i}: ${mine.length} messages != ${tick.messages.length}`);
            return;
          }
          tick.messages.forEach((want, j) => {
            const where = `tick ${i} ${String(want["type"])}#${j}`;
            for (const d of strictDiff(mine[j], want)) bad.push(`${where} ${d}`);
          });
        });
        expect(bad.slice(0, 20)).toEqual([]);
      });

      it("writes the same audit chain (entries, hashed text, hashes)", () => {
        const entries = r.rt.audit.entries();
        expect(entries.length).toBe(run.audit.length);
        const bad: string[] = [];
        entries.forEach((e, i) => {
          const want = run.audit[i] as FixtureEntry;
          for (const d of strictDiff(entryFields(e), entryFields(want)))
            bad.push(`entry ${i} ${d}`);
          if (auditBlob(entryBody(e)) !== want.blob) bad.push(`entry ${i}: hashed text differs`);
          if (e.ts !== want.clock)
            bad.push(`entry ${i}: ts ${e.ts} is not the tick wall ${want.clock}`);
        });
        expect(bad.slice(0, 20)).toEqual([]);
        expect(verifyChain(entries)).toEqual([true, run.audit.length]);
        expect(r.rt.auditFailures).toBe(0);
      });

      it("builds the same hello message", () => {
        expect(strictDiff(r.rt.hello(), run.hello)).toEqual([]);
      });

      it("counts its ticks, keeps the history bounded and the latest messages per person", () => {
        expect(r.rt.tickIndex).toBe(run.ticks.length);
        expect(r.rt.history.length).toBeLessThanOrEqual(run.settings.window_ticks * 8);
        const last = run.ticks[run.ticks.length - 1];
        const lastPreds = (last?.messages ?? []).filter((m) => m["type"] === "prediction");
        for (const p of lastPreds) {
          const pid = p["person_id"] as number;
          expect(strictDiff(r.rt.latest.get(pid)?.prediction, p)).toEqual([]);
        }
      });
    });
  }
});

// ---------- Extra runs (expectations from the Python Runtime) ----------

/** One extra run as Python produced it (scripts/golden_runtime/extra_runs.py). */
interface ExtraExpect {
  ticks: number;
  messages: number;
  predictions: number;
  /** sha256 of the run's canonical text: one line per tick, canon(list of that tick's messages). */
  digest: string;
  auditN: number;
  auditLast: string;
  /** The operator entries of the chain, in order (kind:action). */
  operator: string[];
  gapTicks: number;
  latest: number[];
  hello: string;
}

const HELLO_CORE = (mode: string, w: number, h: number, mirrored: boolean): string =>
  `{"type":"hello","contract_version":"1.0.0","mode":"${mode}","room_only":false,"states":[` +
  `{"id":"Focused","display":"Focused","tier":"P0"},` +
  `{"id":"Frustrated_Masked","display":"Frustrated / Masked Stress","tier":"P0"},` +
  `{"id":"Exhausted","display":"Exhausted","tier":"P0"},` +
  `{"id":"Collaborative_Happy","display":"Collaborative / Happy","tier":"P0"}],` +
  `"feature_version":"m2_lgbm@8a78efc7","frame_w":${pyRepr(w)},"frame_h":${pyRepr(h)},` +
  `"mirrored":${String(mirrored)},"affect_states":null}`;

const EXTRA: Record<string, ExtraExpect> = {
  live_capture: {
    ticks: 37,
    messages: 109,
    predictions: 36,
    digest: "b11398223936935abcf78f3001b59d3585c120ed029613c6948419bfecd9e127",
    auditN: 11,
    auditLast: "ffa87c023b0994f5ed06c6f10fac18bb9ab6d5e35092eada4f81dd25f13c1558",
    operator: ["operator:live", "operator:room_baseline", "operator:room_baseline_captured"],
    gapTicks: 27,
    latest: [1],
    hello: HELLO_CORE("live", 1280, 720, false),
  },
  recording_break_and_drop: {
    ticks: 390,
    messages: 518,
    predictions: 64,
    digest: "7fcd42e84c5bc5d7c52645ae6ccd3cc66fadfffb8b6e3ed60beabe8d1e7e96c4",
    auditN: 15,
    auditLast: "de837637f1aa6d4a4b47cd4dbcddb8a44aed8c3b590cd3d6f4678a202e00c600",
    operator: ["operator:reset", "operator:replay_recording", "operator:break"],
    gapTicks: 0,
    latest: [1],
    hello: HELLO_CORE("replay", 1280, 720, true),
  },
  overlays: {
    ticks: 40,
    messages: 120,
    predictions: 40,
    digest: "48bbd48ddfe3f3eba42e00bb290a7e400e9484712214dfd277b6da6c6bf368e2",
    auditN: 14,
    auditLast: "f4ccde8cb95342b058a05d9b2e5689261bdefae1a191de80d381fa54040a56c1",
    operator: Array<string>(5).fill("operator:sim"),
    gapTicks: 0,
    latest: [1],
    hello: HELLO_CORE("replay", 0, 0, false),
  },
  evict: {
    ticks: 312,
    messages: 792,
    predictions: 240,
    digest: "679d521bed4498cee91e41c69a6b66d55f24509926a37d26e3072896dd060da4",
    auditN: 45,
    auditLast: "3e2bacab2fdd49ecd89fefd6262ddd3e3646a25a1093b9dc95a7ce76234afda5",
    operator: [],
    gapTicks: 0,
    latest: [1],
    hello: HELLO_CORE("replay", 0, 0, false),
  },
  operator: {
    ticks: 70,
    messages: 326,
    predictions: 128,
    digest: "15d9ad514c9fc561f1845451154c19e83c72b4b0ad7262043b4df4773c90566d",
    auditN: 34,
    auditLast: "7bb68131173a76a0ba9bd6b011db0a5bdddff96ffc5ef6830b4cc353c83aa6be",
    operator: [
      "operator:demo_seed",
      "operator:mode",
      "operator:break",
      "operator:break",
      "operator:mode",
      "operator:reset",
      "operator:room_baseline_reset",
      "operator:room_baseline",
      "operator:reset",
      "operator:replay",
    ],
    gapTicks: 0,
    latest: [1, 2],
    hello: HELLO_CORE("replay", 0, 0, false),
  },
};

/** A Runtime on the generator's clocks: wall/mono settable, audit timestamps follow `now`. */
function harness(stateSet: StateSet, settings: Partial<Settings> = {}, opts: RuntimeOptions = {}) {
  const clock = { now: WALL0 - 10.0, mono: MONO0 - 10.0 };
  const rt = new Runtime(makeSettings({ state_set: stateSet, ...settings }), SERVED[stateSet], {
    sessionId: "session-golden",
    clock: () => clock.now,
    monoClock: () => clock.mono,
    tz: "UTC",
    ...opts,
  });
  const ticks: TickMessage[][] = [];
  const step = (records: TickRecord[], wall: number, mono: number) => {
    clock.now = wall;
    clock.mono = mono;
    ticks.push(rt.step(records, wall, mono));
  };
  return { rt, clock, ticks, step };
}

/** replay.ts replayInto with the runtime as the sink (recording.replay_offline); pre(i) runs 2.5 s before step i. */
function replayLive(h: ReturnType<typeof harness>, obs: Obs[], pre: (i: number) => void): void {
  replayInto(
    obs,
    {
      push: (o) => h.rt.pushObs(o),
      step: (wall, mono) => {
        const i = h.ticks.length;
        h.clock.now = wall - 2.5;
        pre(i);
        h.clock.now = wall;
        h.clock.mono = mono;
        h.ticks.push(h.rt.stepLive(wall, mono));
      },
    },
    { wall0: WALL0, mono0: MONO0, tickS: 5.0 },
  );
}

/** The fixture run's records for tick j, re-stamped as a fresh ScenarioSource would at (wall, mono). */
function scenarioRecords(run: RuntimeRun, j: number, wall: number, mono: number): TickRecord[] {
  const tick = run.ticks[j];
  if (tick === undefined) throw new Error(`${run.name} has no tick ${j}`);
  return tick.records.map((d) => ({ ...tickRecordFromDict(d), tick: j, ts: wall, t_mono: mono }));
}

/** Draws the generator patched np.random.normal with (loc + scale * draw). */
function fixedDraws(): () => number {
  let k = 0;
  return () => ((k++ * 7919) % 1000) / 250 - 2;
}

const EXTRA_RUNS: Record<string, () => ReturnType<typeof harness>> = {
  live_capture: () => {
    const h = harness("core");
    h.clock.now = WALL0 - 6.0;
    h.rt.useLive();
    h.clock.now = WALL0 - 5.0;
    h.rt.captureRoomBaseline(10.0);
    replayLive(h, obsOf("capture_fists"), (i) => {
      if (i === 10) h.rt.newSourceCounters();
      // the purge-gate stub's counters for step i (end_tick reads the aggregator's decoded count)
      const d = h.rt.aggregator.framesDecoded;
      const lag = [0, 1, 2, 3, 5, 9, -4][i % 7] as number; // -4: more purged than decoded (gap 0)
      h.rt.setPurgeCounters({
        video_purged: d - lag + (i >= 10 ? 777 : 0),
        audio_purged: 3 * i,
        transcripts_purged: Math.floor(i / 4),
        violations: i >= 20 ? 1 : 0,
        live_raw_buffers: i % 2,
      });
    });
    return h;
  },
  recording_break_and_drop: () => {
    const h = harness("core");
    h.clock.now = WALL0 - 6.0;
    h.rt.useRecording("synth_break_and_drop", 4.0);
    replayLive(h, obsOf("synth_break_and_drop"), (i) => {
      if (i === 50) h.rt.logBreak(1);
    });
    return h;
  },
  overlays: () => {
    const h = harness("core", {}, { gauss: fixedDraws() });
    h.rt.sourceLabel = "replay";
    h.rt.mode = "replay";
    const run = runNamed("frustrated_masked");
    const toggles: Record<number, ["low_light" | "face_occlusion", boolean]> = {
      5: ["low_light", true],
      12: ["face_occlusion", true],
      20: ["low_light", false],
      28: ["face_occlusion", false],
      33: ["low_light", true],
    };
    for (let i = 0; i < 40; i++) {
      const wall = WALL0 + i * 5.0;
      const mono = MONO0 + i * 5.0;
      const t = toggles[i];
      if (t !== undefined) {
        h.clock.now = wall - 2.5;
        h.rt.setOverlay(t[0], t[1]);
      }
      h.rt.setPurgeCounters({
        video_purged: 20 * i + (i % 3),
        audio_purged: i,
        transcripts_purged: Math.floor(i / 4),
        violations: i >= 20 ? 1 : 0,
        live_raw_buffers: i % 2,
      });
      h.step(scenarioRecords(run, i, wall, mono), wall, mono);
    }
    return h;
  },
  evict: () => {
    const h = harness("core", { person_drop_min: 3.0 });
    h.rt.sourceLabel = "replay";
    h.rt.mode = "replay";
    const run = runNamed("full_demo");
    run.ticks.forEach((_, i) => {
      const wall = WALL0 + i * 5.0;
      const mono = MONO0 + i * 5.0;
      h.step(scenarioRecords(run, i, wall, mono), wall, mono);
      if (i > 0 && !h.rt.window.people().includes(1)) evictedAt.push(i);
    });
    return h;
  },
  operator: () => {
    const h = harness("core");
    const rt = h.rt;
    rt.sourceLabel = "replay";
    rt.mode = "replay";
    const run = runNamed("context_blindness");
    const seedRows = runNamed("full_demo").ticks.flatMap((t) => t.records.map(tickRecordFromDict));
    h.clock.now = WALL0 - 1.0;
    h.clock.mono = MONO0 - 1.0;
    rt.seed(seedRows, "full_demo", 2.0, 3.5);
    let j = 0; // the scenario's own tick index (use_scenario restarts it)
    for (let i = 0; i < 70; i++) {
      const wall = WALL0 + i * 5.0;
      const mono = MONO0 + i * 5.0;
      h.clock.now = wall - 2.5;
      if (i === 8) rt.setRoomOnly(true);
      if (i === 14) rt.logBreak(1);
      if (i === 16) rt.logBreak(null);
      if (i === 20) rt.setRoomOnly(false);
      if (i === 25) rt.resetPeople(true);
      if (i === 30) rt.resetRoomBaseline();
      if (i === 35) rt.captureRoomBaseline(30.0);
      if (i === 40) {
        rt.useScenario("context_blindness", 4.0);
        j = 0;
      }
      h.step(scenarioRecords(run, j, wall, mono), wall, mono);
      j += 1;
    }
    return h;
  },
};
const evictedAt: number[] = [];

describe("runtime against extra Python runs (live source, overlays, eviction, operator actions)", () => {
  for (const [name, want] of Object.entries(EXTRA)) {
    describe(name, () => {
      let h: ReturnType<typeof harness>;
      beforeAll(() => {
        const t0 = performance.now();
        h = (EXTRA_RUNS[name] as () => ReturnType<typeof harness>)();
        perf.notes.push(`${name} ${(performance.now() - t0).toFixed(0)} ms`);
      }, 120_000);

      it("publishes the same messages as Python (canonical text digest)", () => {
        const msgs = h.ticks.flat();
        expect(h.ticks.length).toBe(want.ticks);
        expect(msgs.length).toBe(want.messages);
        expect(msgs.filter((m) => m.type === "prediction").length).toBe(want.predictions);
        const text = h.ticks.map((t) => canon(t)).join("\n");
        expect(sha256Hex(text)).toBe(want.digest);
      });

      it("writes the same audit chain, operator entries included", () => {
        const entries = h.rt.audit.entries();
        expect(entries.length).toBe(want.auditN);
        expect(entries[entries.length - 1]?.hash).toBe(want.auditLast);
        expect(
          entries
            .filter((e) => e.kind !== "prediction")
            .map((e) => `${e.kind}:${String(e.payload["action"] ?? "")}`),
        ).toEqual(want.operator);
        expect(verifyChain(entries)).toEqual([true, want.auditN]);
        expect(h.rt.auditFailures).toBe(0);
      });

      it("ends in the same state (hello, purge-gap ticks, people tracked)", () => {
        expect(canon(h.rt.hello())).toBe(want.hello);
        expect(h.rt.gapTicks).toBe(want.gapTicks);
        expect(h.rt.tickIndex).toBe(want.ticks);
        expect([...h.rt.latest.keys()].sort((a, b) => a - b)).toEqual(want.latest);
        expect([...h.rt.smoothers.keys()].sort((a, b) => a - b)).toEqual(want.latest);
      });
    });
  }

  it("really exercised what the extra runs are for", () => {
    // eviction: person 1 left the window during the 6-minute break (person_drop_min = 3)
    expect(evictedAt.length).toBeGreaterThan(0);
    // live: the operator capture completed and was audited with Python ints (bigint) in its payload
    const live = EXTRA_RUNS["live_capture"] as () => ReturnType<typeof harness>;
    const h = live();
    const captured = h.rt.audit
      .entries()
      .find((e) => e.payload["action"] === "room_baseline_captured");
    expect(captured).toBeDefined();
    expect(typeof captured?.payload["target_samples"]).toBe("bigint");
    expect(h.ticks.flat().some((m) => m.type === "governance" && m.purge_gap > 2)).toBe(true);
    // overlays: both SIMULATED tags reached the messages
    const ov = (EXTRA_RUNS["overlays"] as () => ReturnType<typeof harness>)();
    const sims = new Set(
      ov.ticks.flat().flatMap((m) => (m.type === "prediction" ? m.simulated_fields : [])),
    );
    expect([...sims].sort()).toEqual(["face_occlusion", "low_light"]);
  }, 120_000);
});

// ---------- Behaviour without a Python reference ----------

describe("runtime behaviour", () => {
  const run = runNamed("frustrated_masked");

  it("pauses and publishes nothing when the source is finished (records = null)", () => {
    const h = harness("core");
    expect(h.rt.step(null, WALL0, MONO0)).toEqual([]);
    expect(h.rt.paused).toBe(true);
    expect(h.rt.tickIndex).toBe(0);
  });

  it("publishes through onMessage in emission order, and returns the same messages", () => {
    const seen: string[] = [];
    const h = harness("core", {}, { onMessage: (m) => seen.push(m.type) });
    h.rt.sourceLabel = "replay";
    const msgs = h.rt.step(scenarioRecords(run, 0, WALL0, MONO0), WALL0, MONO0);
    expect(seen).toEqual(msgs.map((m) => m.type));
    expect(seen).toEqual(["governance", "state_vector", "prediction"]);
  });

  it("keeps its own copy of the records (the caller may reuse them)", () => {
    const h = harness("core");
    const recs = scenarioRecords(run, 0, WALL0, MONO0);
    h.rt.step(recs, WALL0, MONO0);
    (recs[1] as TickRecord).face_valence = 0.987;
    (recs[1] as TickRecord).simulated_fields.push("x");
    const stored = h.rt.window.snapshot(1)[0] as TickRecord;
    expect(stored.face_valence).not.toBe(0.987);
    expect(stored.simulated_fields).toEqual([]);
  });

  it("counts a refused audit payload instead of throwing (Python logs it)", () => {
    const h = harness("core");
    h.rt.recordAudit("operator", { action: "x".repeat(201) });
    expect(h.rt.auditFailures).toBe(1);
    expect(h.rt.audit.length).toBe(0);
  });

  it("rejects a baseline capture outside [10, 600] seconds (the endpoint answers 422)", () => {
    const h = harness("core");
    expect(() => h.rt.captureRoomBaseline(5)).toThrow(RangeError);
    expect(() => h.rt.captureRoomBaseline(Number.NaN)).toThrow(RangeError);
    expect(h.rt.captureRoomBaseline(600).mode).toBe("capturing");
  });

  it("publishes tracks at most every 0.1 s with only the participants' boxes", () => {
    const h = harness("core");
    const obs = obsOf("synth_background_and_two_people").filter((o) => o.type === "video");
    const first = obs[0];
    if (first === undefined || first.type !== "video") throw new Error("no video obs");
    h.clock.mono = 50.0;
    const t1 = h.rt.pushObs(first);
    expect(t1?.type).toBe("tracks");
    expect(t1?.capture_ts_ms).toBe(Math.trunc(first.wall * 1000));
    expect(h.rt.pushObs(first)).toBeNull(); // same monotonic instant: throttled
    h.clock.mono = 50.1;
    expect(h.rt.pushObs(first)).not.toBeNull();
    expect(h.rt.camera).toBe(true);
    expect(h.rt.mic).toBe(false);
  });

  it("pads event ids to six digits and names the room", () => {
    const h = harness("core");
    h.rt.sourceLabel = "replay";
    h.rt.roomOnly = true;
    const msgs = h.rt.step(scenarioRecords(run, 0, WALL0, MONO0), WALL0, MONO0);
    const pred = msgs.find((m) => m.type === "prediction");
    expect(pred?.type === "prediction" && pred.event_id).toBe("p0-000000");
    expect(pred?.type === "prediction" && pred.explanation.text).toContain("Room");
  });
});

// ---------- Edge cases against the Python runtime (stub predictor) ----------

/**
 * Inputs the golden runs never produce, run through Python's Runtime and this one with the same stub predictor
 * (outputs keyed by "tick:person_id", none = predict() returns None), a stub purge gate, the audit / aggregator /
 * engine clocks patched to `clock` and np.random.normal patched to `draws`. The specs are built below exactly as
 * the generator built them; Python's results (canonical text digest as in the extra runs, audit chain head, end
 * state) are pasted as literals. Generator: scripts/golden_runtime/edge_cases.py, which runs the specs of
 * scripts/golden_runtime/specs.py. Rerun it whenever valence/runtime/engine.py changes.
 */
type SpecRecord = Record<string, number | string | null | string[]>;
type SpecAction =
  | ["clock", number]
  | ["overlay", "low_light" | "face_occlusion", boolean]
  | ["room_only", boolean]
  | ["reset", boolean]
  | ["break", number | null]
  | ["purge", Partial<PurgeCounters>]
  | ["seed", SpecRecord[], string, number, number | null]
  | ["audit", string, Record<string, unknown>];
interface SpecOut {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  face_only_label: string | null;
  face_only_confidence: number | null;
  raw_override: boolean | null;
  ood_features: string[];
  layers: Layer[];
}
interface SpecStep {
  wall: number;
  mono: number;
  records: SpecRecord[] | null;
  outs?: Record<string, SpecOut>;
  pre?: SpecAction[];
}
interface Spec {
  state_set: StateSet;
  settings?: Partial<Settings>;
  room_only?: boolean;
  label?: "live" | "replay";
  wall_init?: number;
  mono_init?: number;
  draws?: number[];
  steps: SpecStep[];
}

const CORE = ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"];
const P4 = [0.1, 0.2, 0.3, 0.4];
const DEFAULT_LAYERS: Layer[] = [
  {
    name: "Ambient Physics",
    contribution: 0.123456,
    drivers: [
      {
        feature: "db_mean_15m",
        display: "average noise of 72 dB over 15 minutes",
        value: 72.04,
        unit: "dB",
        shap: 0.4,
        level: "high",
        effect: "supports",
        scope: "room",
      },
    ],
  },
  { name: "Facial Surface", contribution: -0.0625, drivers: [] },
];

function specOut(
  label: string,
  p: number[] | Record<string, number>,
  conf: number | null = null,
  ro: boolean | null = null,
): SpecOut {
  const probabilities = Array.isArray(p)
    ? Object.fromEntries(CORE.map((s, i) => [s, p[i] as number]))
    : p;
  const i = CORE.indexOf(label);
  return {
    label,
    confidence: conf ?? (Array.isArray(p) && i >= 0 ? (p[i] as number) : 0.5),
    probabilities,
    face_only_label: "Focused",
    face_only_confidence: 0.61234,
    raw_override: ro,
    ood_features: [],
    layers: DEFAULT_LAYERS,
  };
}

function specRoom(i: number, kw: SpecRecord = {}): SpecRecord {
  return {
    tick: i,
    ts: WALL0 + 5 * i,
    t_mono: MONO0 + 5 * i,
    person_id: 0,
    display_id: "Room",
    n_people: 1,
    ambient_db: 55.0,
    light_lux: 300.0,
    tod_h: 14.0,
    ...kw,
  };
}

function specPerson(i: number, pid = 1, kw: SpecRecord = {}): SpecRecord {
  return {
    tick: i,
    ts: WALL0 + 5 * i,
    t_mono: MONO0 + 5 * i,
    person_id: pid,
    display_id: `Person #${pid}`,
    n_people: 2,
    ambient_db: 55.0,
    light_lux: 300.0,
    face_valence: 0.2,
    face_visible: 1.0,
    tod_h: 14.0,
    posture_slump: 0.1,
    head_in_hands_frac: 0.0,
    hours_since_break: 1.5,
    ...kw,
  };
}

function specStep(
  i: number,
  records: SpecRecord[] | null,
  outs: Record<string, SpecOut> = {},
  pre?: SpecAction[],
): SpecStep {
  return { wall: WALL0 + 5 * i, mono: MONO0 + 5 * i, records, outs, ...(pre ? { pre } : {}) };
}

/** Time-of-day probes: exact and inexact half-minute ties, the hour wrap, negative and huge times. */
const TOD_PROBES = [0, 9, 13, 23]
  .flatMap((h) => [1, 3, 59, 61, 117, 119].map((k) => h + k / 120))
  .concat([
    14 + 59.5 / 60,
    5 - 1e-15,
    1e300 % 24,
    12 + 1 / 7200,
    0.5 / 60,
    11.999999999999,
    7.5 / 60,
  ]);

const SPECS: Record<string, () => Spec> = {
  // state vector, fact sheet and prediction edges: NaN / inf / -0.0 / subnormal / 1e300 fields, NaN and
  // out-of-range times, n_people 0, a label missing from the probabilities, unicode simulated fields, negative and
  // duplicate person ids, a stale row evicted on arrival, predict() -> None, 2**40 counts, the history bound
  sv: () => {
    const steps: SpecStep[] = [
      specStep(
        0,
        [
          specRoom(0),
          specPerson(0, 1, {
            tod_h: null,
            n_people: 0,
            face_visible: null,
            ambient_db: -0.0,
            light_lux: 5e-324,
            hours_since_break: "Infinity",
            posture_slump: 1e300,
            head_in_hands_frac: 0.0005,
            face_valence: -0.0005,
          }),
        ],
        { "0:1": specOut("Collaborative_Happy", P4) },
      ),
      specStep(
        1,
        [
          specRoom(1, { simulated_fields: ["room_history", "a", "é"] }),
          specPerson(1, 1, {
            tod_h: 23.999999,
            n_people: 1,
            simulated_fields: ["ß", "Z", "a", "low_light", "é", "\u{1F600}", "￿", "b", "c", "d"],
          }),
        ],
        { "1:1": specOut("Collaborative_Happy", P4) },
      ),
      specStep(2, [specRoom(2), specPerson(2, 1, { tod_h: -0.25, n_people: 1 })], {
        "2:1": specOut("Exhausted", { Focused: 0.5, Frustrated_Masked: 0.25 }, 0.33335),
      }),
      specStep(
        3,
        [
          specRoom(3),
          specPerson(3, 2, { tod_h: 25.5 }),
          specPerson(3, -3, { tod_h: 24.0 }),
          specPerson(3, 1, { tod_h: 14.5 }),
          specPerson(3, 1, { tod_h: 14.25, face_valence: null }),
        ],
        {
          "3:1": specOut("Focused", P4, 0.00005),
          "3:2": specOut("Focused", P4),
          "3:-3": specOut("Exhausted", P4),
        },
      ),
      specStep(4, [specRoom(4), specPerson(4, 1, { ts: WALL0 + 5 * 4 - 2000 }), specPerson(4, 2)], {
        "4:1": specOut("Focused", P4),
        "4:2": specOut("Focused", P4),
      }),
      specStep(5, [specRoom(5), specPerson(5, 2)], {}),
      specStep(
        6,
        [
          specRoom(6, { laughter_count: 7, compliments_count: 2 ** 40 }),
          specPerson(6, 2, { clenched_fists_count: 2 ** 41 }),
        ],
        {
          "6:2": specOut("Frustrated_Masked", P4, null, true),
        },
      ),
    ];
    TOD_PROBES.forEach((t, j) => {
      const i = 7 + j;
      steps.push(
        specStep(
          i,
          [
            specRoom(i, { tod_h: t }),
            specPerson(i, 2, {
              tod_h: t,
              light_lux: 0.0005 + j,
              ambient_db: 2.0005 + j / 1000,
              posture_slump: (j % 3) / 3,
              face_visible: j % 2,
            }),
          ],
          {
            [`${i}:2`]: specOut(
              Math.floor(j / 7) % 2 ? "Exhausted" : "Focused",
              P4,
              0.12345 + j / 100000,
            ),
          },
        ),
      );
    });
    return { state_set: "core", settings: { window_ticks: 4 }, steps };
  },
  // room-only mode, both overlays (NaN lux, noise clipped at +-1), an empty tick, the mode switch, a finished source
  room: () => {
    const steps: SpecStep[] = [];
    for (let i = 0; i < 14; i++) {
      const at = (...acts: SpecAction[]): SpecAction[] => [["clock", WALL0 + 5 * i - 2.5], ...acts];
      const pre: Record<number, SpecAction[]> = {
        2: at(["overlay", "low_light", true]),
        5: at(["overlay", "face_occlusion", true]),
        8: at(["overlay", "low_light", false]),
        10: at(["overlay", "low_light", true], ["room_only", false]),
      };
      const recs =
        i === 6
          ? []
          : [
              specRoom(i, {
                light_lux: i % 3 === 0 ? null : 120.0,
                face_valence: i % 2 ? 0.9 : -0.95,
                face_visible: 1.0,
                simulated_fields: i === 3 ? ["low_light"] : [],
              }),
              specPerson(i, 1, {
                face_valence: i % 4 === 0 ? null : 0.97 - i / 10,
                light_lux: i % 2 ? null : 80.0,
              }),
            ];
      steps.push(
        specStep(
          i,
          recs,
          {
            [`${i}:0`]: specOut(CORE[i % 4] as string, P4, null, Boolean(i % 2)),
            [`${i}:1`]: specOut(CORE[(i + 1) % 4] as string, P4),
          },
          pre[i],
        ),
      );
    }
    steps.push({ wall: WALL0 + 70, mono: MONO0 + 70, records: null });
    const draws = Array<number[]>(4)
      .fill([3.0, -3.0, 0.5, -0.25, 1e-3, 2.0, -2.0, 0.0, 4.0, -4.0])
      .flat();
    return { state_set: "core", room_only: true, draws, steps };
  },
  // operator actions on the live label: seed, breaks (None, 1 and a float id), purge counters, both resets, the
  // audit throttle (label change, 30 s), a refused (NaN) payload
  op: () => {
    const seedRows = [
      ...Array.from({ length: 30 }, (_, t) =>
        specRoom(t, { ambient_db: 60 + t, simulated_fields: [] }),
      ),
      ...Array.from({ length: 30 }, (_, t) => specPerson(t)),
    ];
    const steps: SpecStep[] = [];
    for (let i = 0; i < 30; i++) {
      const pre: Record<number, SpecAction[]> = {
        0: [
          ["clock", WALL0 - 1.0],
          ["seed", seedRows, "full_demo", 1.0, 2.5],
        ],
        3: [
          ["break", null],
          ["break", 1],
          ["break", 1.5],
          ["purge", { video_purged: 10 }],
        ],
        6: [
          ["reset", true],
          ["purge", { video_purged: 7, audio_purged: 4, violations: 2 }],
        ],
        12: [["reset", false]],
        20: [["audit", "operator", { action: "x", v: "NaN" }]],
      };
      const label = i < 9 || i > 15 ? "Focused" : "Exhausted";
      steps.push(
        specStep(
          i,
          [specRoom(i), specPerson(i, 1, { hours_since_break: null })],
          { [`${i}:1`]: specOut(label, P4) },
          pre[i],
        ),
      );
    }
    return {
      state_set: "core",
      label: "live",
      wall_init: WALL0 - 10,
      mono_init: MONO0 - 10,
      steps,
    };
  },
  ...Object.fromEntries(
    (
      [
        ["w0", { window_ticks: 0 }, false, true],
        ["drop0", { person_drop_min: 0.0, window_ticks: 3 }, false, true],
        ["noroom", { window_ticks: 5 }, true, false],
      ] as const
    ).map(([name, settings, roomOnly, withRoom]) => [
      name,
      (): Spec => ({
        state_set: "core",
        settings,
        room_only: roomOnly,
        steps: Array.from({ length: 8 }, (_, i) =>
          specStep(
            i,
            [
              ...(withRoom ? [specRoom(i)] : []),
              specPerson(i, 1, { ts: WALL0 + 5 * i - (i === 4 ? 0.5 : 0.0) }),
              specPerson(i, 2),
            ],
            {
              [`${i}:1`]: specOut("Focused", P4),
              [`${i}:2`]: specOut("Exhausted", P4),
              [`${i}:0`]: specOut("Focused", P4),
            },
          ),
        ),
      }),
    ]),
  ),
};

/** Python's results for each spec (scripts/golden_runtime/edge_cases.py). */
const SPEC_EXPECT: Record<
  string,
  { digest: string; auditN: number; auditLast: string; state: unknown }
> = {
  sv: {
    digest: "6c28dd18bcc3b2526346d0a5f95327936c5fa5484f01bf95fba0915b04acdf02",
    auditN: 9,
    auditLast: "3a699bd886c0760ffbbb6370a1eb5a0ebb89f523cc4685cb3a71a9f610ea54a1",
    state: {
      latest: [-3, 2],
      smoothers: [-3, 2],
      history: 32,
      gap_ticks: 0,
      paused: false,
      tick: 38,
      people: [-3, 2],
    },
  },
  room: {
    digest: "97d6e51dbc3802cb4b08ead54515970af73cbf54b53cd65b039858d77ad9420e",
    auditN: 8,
    auditLast: "416edc41788f217b118ea0d1aacd1c93d12d063cb9a69ecf323e7420f748a2c3",
    state: {
      latest: [1],
      smoothers: [0, 1],
      history: 28,
      gap_ticks: 0,
      paused: true,
      tick: 14,
      people: [1],
    },
  },
  op: {
    digest: "72fcb9e24f76756a88ad1ec1b7ecd3da22f826e2c322cd91e491507bda590bbf",
    auditN: 11,
    auditLast: "bcb17cdb126e825aff95e03ee89bf8b6ce5d53d65c677858482822541be19c2d",
    state: {
      latest: [1],
      smoothers: [1],
      history: 36,
      gap_ticks: 0,
      paused: false,
      tick: 30,
      people: [1],
    },
  },
  w0: {
    digest: "a848cefa316212b01bd3491455e3fc88f44917551b16992a06e8d9cc05665f17", // governance only
    auditN: 0,
    auditLast: "",
    state: {
      latest: [],
      smoothers: [],
      history: 0,
      gap_ticks: 0,
      paused: false,
      tick: 8,
      people: [1, 2],
    },
  },
  drop0: {
    digest: "5488e18092e59bec9c6150069edf5ef589f45d1c7ede8b61d8666afdf00d0894",
    auditN: 4,
    auditLast: "a2000f601f87d81c7450ae8645c6e2e3fb767f58cb8cf9bd2eb83ddb50b08b37",
    state: {
      latest: [1, 2],
      smoothers: [1, 2],
      history: 24,
      gap_ticks: 0,
      paused: false,
      tick: 8,
      people: [1, 2],
    },
  },
  noroom: {
    digest: "a848cefa316212b01bd3491455e3fc88f44917551b16992a06e8d9cc05665f17", // the same 8 governance messages
    auditN: 0,
    auditLast: "",
    state: {
      latest: [],
      smoothers: [],
      history: 0,
      gap_ticks: 0,
      paused: false,
      tick: 8,
      people: [1, 2],
    },
  },
};

/** "NaN" / "Infinity" / "-Infinity" strings -> numbers, recursively (the spec's JSON encoding). */
function decodeSpec(v: unknown): unknown {
  if (v === "NaN") return NaN;
  if (v === "Infinity") return Infinity;
  if (v === "-Infinity") return -Infinity;
  if (Array.isArray(v)) return v.map(decodeSpec);
  if (v !== null && typeof v === "object")
    return Object.fromEntries(
      Object.entries(v as Record<string, unknown>).map(([k, x]) => [k, decodeSpec(x)]),
    );
  return v;
}

function specRecord(d: SpecRecord, session: string): TickRecord {
  return tickRecordFromDict({
    session_id: session,
    display_id: `Person #${String(d["person_id"] ?? 0)}`,
    ...d,
  });
}

interface SpecRun {
  rt: Runtime;
  ticks: TickMessage[][];
  text: string[];
  /** The narrator's calls: [event_id, label_display, confidence] per request, ["clear"] per clear_cache(). */
  narrated: (string | number)[][];
}

/** Drive a Runtime through a spec the way scripts/golden_runtime/edge_cases.py drives Python's. */
function runSpec(spec: Spec): SpecRun {
  const outs = new Map<string, SpecOut>();
  for (const st of spec.steps)
    for (const [k, o] of Object.entries(st.outs ?? {})) outs.set(k, decodeSpec(o) as SpecOut);
  const stub: Predictor = {
    states: STATE_SETS[spec.state_set],
    modelId: "stub-edge",
    predict(person): PredictionOutput | null {
      const last = person[person.length - 1] as TickRecord;
      const o = outs.get(`${last.tick}:${last.person_id}`);
      if (o === undefined) return null;
      return { ...o, layers: o.layers.map((l) => ({ ...l, drivers: [...l.drivers] })), values: {} };
    },
  };
  const clock = { now: spec.wall_init ?? 0, mono: spec.mono_init ?? 0 };
  const draws = spec.draws ?? [];
  let k = 0;
  const narrated: (string | number)[][] = [];
  const rt = new Runtime(makeSettings(spec.settings ?? {}), stub, {
    sessionId: "session-edge",
    clock: () => clock.now,
    monoClock: () => clock.mono,
    tz: "UTC",
    gauss: () => draws[k++] as number,
    narrator: {
      request: (sheet) => narrated.push([sheet.event_id, sheet.label_display, sheet.confidence]),
      clearCache: () => narrated.push(["clear"]),
    },
  });
  rt.roomOnly = spec.room_only ?? false;
  rt.sourceLabel = spec.label ?? "replay";
  const ticks: TickMessage[][] = [];
  for (const st of spec.steps) {
    for (const act of st.pre ?? []) {
      if (act[0] === "clock") clock.now = act[1];
      else if (act[0] === "overlay") rt.setOverlay(act[1], act[2]);
      else if (act[0] === "room_only") rt.setRoomOnly(act[1]);
      else if (act[0] === "reset") rt.resetPeople(act[1]);
      else if (act[0] === "break") rt.logBreak(act[1]);
      else if (act[0] === "purge") rt.setPurgeCounters(act[1]);
      else if (act[0] === "seed")
        rt.seed(
          act[1].map((r) => specRecord(r, "seed")),
          act[2],
          act[3],
          act[4],
        );
      else rt.recordAudit(act[1], decodeSpec(act[2]) as AuditPayload);
    }
    clock.now = st.wall;
    clock.mono = st.mono;
    const recs = st.records === null ? null : st.records.map((r) => specRecord(r, "session-edge"));
    ticks.push(rt.step(recs, st.wall, st.mono));
  }
  return { rt, ticks, text: ticks.map((t) => canon(t)), narrated };
}

describe("runtime against Python on inputs the golden runs never produce", () => {
  const runs: Record<string, SpecRun> = {};
  beforeAll(() => {
    for (const [name, build] of Object.entries(SPECS)) runs[name] = runSpec(build());
  });

  for (const name of Object.keys(SPECS)) {
    it(`${name}: publishes the same messages and audit chain, and ends in the same state`, () => {
      const r = runs[name] as SpecRun;
      const want = SPEC_EXPECT[name];
      if (want === undefined) throw new Error(`no expectation for ${name}`);
      expect(sha256Hex(r.text.join("\n"))).toBe(want.digest);
      const entries = r.rt.audit.entries();
      expect(entries.length).toBe(want.auditN);
      if (want.auditLast) expect(entries[entries.length - 1]?.hash).toBe(want.auditLast);
      expect(verifyChain(entries)).toEqual([true, want.auditN]);
      expect({
        latest: [...r.rt.latest.keys()].sort((a, b) => a - b),
        smoothers: [...r.rt.smoothers.keys()].sort((a, b) => a - b),
        history: r.rt.history.length,
        gap_ticks: r.rt.gapTicks,
        paused: r.rt.paused,
        tick: r.rt.tickIndex,
        people: r.rt.window.people(),
      }).toEqual(want.state);
    });
  }

  /** The messages of one type, in emission order, across a run. */
  function ofType<T extends TickMessage["type"]>(
    r: SpecRun,
    type: T,
  ): Extract<TickMessage, { type: T }>[] {
    return r.ticks.flat().filter((m): m is Extract<TickMessage, { type: T }> => m.type === type);
  }

  it("prints the time of day as Python does (half-minute ties to even, 59.5 min stays in its hour)", () => {
    // int(tod_h) and int(round((tod_h - h) * 60)) % 60: 23.999999 -> "23:00", 14 + 59.5/60 -> "14:00", -0.25 ->
    // "00:45" (floor modulo), 25.5 -> "01:30", NaN -> "00:00"
    expect(ofType(runs["sv"] as SpecRun, "state_vector").map((m) => m.time_of_day)).toEqual([
      "00:00",
      "23:00",
      "00:45",
      "00:00",
      "14:15",
      "01:30",
      "14:00",
      "14:00",
      "00:00",
      "00:02",
      "00:30",
      "00:30",
      "00:58",
      "00:00",
      "09:00",
      "09:02",
      "09:30",
      "09:30",
      "09:58",
      "09:00",
      "13:00",
      "13:02",
      "13:30",
      "13:30",
      "13:58",
      "13:00",
      "23:00",
      "23:01",
      "23:30",
      "23:30",
      "23:59",
      "23:00",
      "14:00",
      "04:00",
      "00:00",
      "12:00",
      "00:00",
      "11:00",
      "00:08",
    ]);
  });

  it("rounds the state vector's fields as Python does (-0.0, subnormal, inf, 1e300, NaN, 0.0005)", () => {
    const sv = ofType(runs["sv"] as SpecRun, "state_vector")[0];
    expect(canon(sv)).toBe(
      '{"type":"state_vector","t":1792850400.0,"person_id":1.0,"ambient_db":-0.0,"light_lux":0.0,' +
        '"hours_since_break":inf,"time_of_day":"00:00","clenched_fists_5m":0.0,"compliments_15m":0.0,' +
        '"laughter_15m":0.0,"posture_slump":1e+300,"head_in_hands":0.001,"social_withdrawal":0.0,' +
        '"face_visible":false,"facial_valence":-0.001,"window_fill_min":0.08,"source":"live","simulated_fields":[]}',
    );
  });

  it("takes the predictor's confidence when the smoothed label has no probability (dict.get default)", () => {
    const pred = ofType(runs["sv"] as SpecRun, "prediction").find(
      (m) => m.event_id === "p1-000002",
    );
    expect([pred?.label, pred?.label_display, pred?.confidence]).toEqual([
      "Collaborative_Happy",
      "Upbeat",
      0.3333,
    ]);
    expect(pred?.probabilities).toEqual({ Focused: 0.5, Frustrated_Masked: 0.25 });
  });

  it("sorts simulated fields by code point, room and person tags merged", () => {
    const pred = ofType(runs["sv"] as SpecRun, "prediction").find(
      (m) => m.event_id === "p1-000001",
    );
    expect(pred?.simulated_fields).toEqual([
      "Z",
      "a",
      "b",
      "c",
      "d",
      "low_light",
      "room_history",
      "ß",
      "é",
      "￿",
      "\u{1F600}",
    ]);
    expect(pred?.explanation.text).toBe(
      "Upbeat (40%) is the most likely state for Person #1. Main evidence: average noise of 72 dB over 15 minutes.",
    );
  });

  it("asks the narrator on a new shown label or confidence band, else every 30 s, and clears it on a reset", () => {
    const C = "Collaborative / Happy";
    expect((runs["sv"] as SpecRun).narrated).toEqual([
      ["p1-000000", "Upbeat", 0.4],
      ["p-3-000003", C, 0.4],
      ["p1-000003", C, 0.4],
      ["p2-000003", C, 0.4],
      ["p2-000009", C, 0.4],
      ["p2-000015", C, 0.4],
      ["p2-000021", C, 0.4],
      ["p2-000027", C, 0.4],
      ["p2-000033", C, 0.4],
    ]);
    expect((runs["room"] as SpecRun).narrated).toEqual([
      ["p0-000000", "Upbeat", 0.4],
      ["p0-000006", "Upbeat", 0.4],
      ["p1-000010", C, 0.4],
    ]);
    expect((runs["op"] as SpecRun).narrated).toEqual([
      ["p1-000000", C, 0.4],
      ["clear"],
      ["p1-000006", C, 0.4],
      ["clear"],
      ["p1-000012", C, 0.4],
      ["p1-000018", C, 0.4],
      ["p1-000024", C, 0.4],
    ]);
  });

  it("audits a break's person id as the JSON value it was (null, int, float) instead of throwing on 1.5", () => {
    const breaks = (runs["op"] as SpecRun).rt.audit
      .entries()
      .filter((e) => e.payload["action"] === "break");
    expect(breaks.map((e) => e.payload["person_id"])).toEqual([null, 1n, 1.5]);
    expect(breaks.map((e) => auditBlob(entryBody(e)).match(/"person_id":[^,}]*/)?.[0])).toEqual([
      '"person_id":null',
      '"person_id":1',
      '"person_id":1.5',
    ]);
  });
});

describe("runtime pieces against Python (push_obs, ScenarioSource, hello)", () => {
  it("publishes tracks and measures fps exactly as push_obs does", () => {
    // scripts/golden_runtime/tracks.py: Runtime built at monotonic 10.0, one VideoObs per row
    // (frame_seq = frames_decoded = i, mirrored = i odd, wall = WALL0 + mono * 1.001), then an AudioObs;
    // the expected values are Python's Tracks and fps after each push, then its fps, mic and camera
    const clock = { mono: 10.0 };
    const rt = new Runtime(makeSettings(), SERVED.core, {
      sessionId: "session-golden",
      clock: () => WALL0,
      monoClock: () => clock.mono,
    });
    const P = (
      id: number,
      y1: number,
      y2: number,
      extra: { fist_active?: boolean; slump?: boolean } = {},
    ) => makePersonObs({ track_id: id, bbox: [0.1, y1, 0.2, y2], ...extra });
    const pushes: [number, ReturnType<typeof P>[]][] = [
      [
        10.05,
        [
          P(1, 0.1, 0.6),
          P(2, 0.5, 0.7, { fist_active: true }),
          P(3, 0.2, 0.5, { slump: true }),
          P(4, 0.0, 0.275),
        ],
      ],
      [10.1, [P(1, 0.1, 0.6)]],
      [10.15, [P(1, 0.1, 0.6)]], // 10.15 - 10.05 = 0.0999999...: still throttled
      [10.1500001, [P(1, 0.1, 0.6)]],
      [12.0, []],
      [12.05, [P(5, 0.3, 0.3), P(6, 0.3, 0.3)]],
      [12.2, [P(7, 0.0, 1.0), P(8, 0.0, 0.55), P(9, 0.0, 0.5499999)]],
      [14.2, [P(1, 0.1, 0.2)]],
    ];
    const got = pushes.map(([m, persons], i) => {
      clock.mono = m;
      const tr = rt.pushObs(
        makeVideoObs({
          t: m,
          wall: WALL0 + m * 1.001,
          frame_seq: i,
          frames_decoded: i,
          frame_w: 640,
          frame_h: 480,
          mirrored: i % 2 === 1,
          persons,
        }),
      );
      return [
        rt.fps,
        tr === null ? null : [tr.t, tr.capture_ts_ms, tr.mirrored, tr.persons.map((p) => p.id)],
      ];
    });
    expect(got).toEqual([
      [0.0, [1792850410.06005, 1792850410060, false, [1, 3, 4]]],
      [0.0, null],
      [0.0, null],
      [0.0, [1792850410.16015, 1792850410160, true, [1]]],
      [2.5, [1792850412.012, 1792850412012, false, []]],
      [2.5, null],
      [2.5, [1792850412.2122, 1792850412212, false, [7, 8]]],
      [1.363636363636364, [1792850414.2142, 1792850414214, true, [1]]],
    ]);
    clock.mono = 15.0;
    expect(
      rt.pushObs({
        type: "audio",
        t: 15.0,
        wall: WALL0 + 15,
        block_seq: 0,
        db_spl: 50.0,
        rms_power: 1e-3,
        speech_prob: 0.0,
        laughter_onsets: 0,
        compliment_hits: 0,
      }),
    ).toBeNull();
    expect([rt.fps, rt.mic, rt.camera]).toEqual([1.363636363636364, true, true]);
  });

  it("ScenarioSource re-stamps a scenario's rows into exactly the records Python's source returned", () => {
    const run = runNamed("context_blindness");
    // the fixture's records with the stamps scrambled, ticks spread out and the groups listed last tick first
    const raw = run.ticks
      .map((t, j) =>
        t.records.map((d) => ({
          ...tickRecordFromDict(d),
          tick: 10 * j + 3,
          ts: NaN,
          t_mono: -1,
          session_id: "raw",
          source: "sim",
        })),
      )
      .reverse()
      .flat();
    const src = new ScenarioSource(raw, run.scenario, { sessionId: run.session_id });
    expect(src.length).toBe(run.ticks.length);
    run.ticks.forEach((t) => {
      const recs = src.nextTick(t.wall, t.mono);
      expect(recs?.map(tickRecordToDict)).toEqual(t.records);
    });
    expect(src.done).toBe(true);
    expect(src.nextTick(WALL0, MONO0)).toBeNull();
    // loop=True starts over at the first tick; the id carries six random hex digits
    const looped = new ScenarioSource(raw, "x", { loop: true });
    expect(looped.sessionId).toMatch(/^replay-x-[0-9a-f]{6}$/);
    for (let i = 0; i < run.ticks.length; i++) looped.nextTick(WALL0, MONO0);
    expect(looped.done).toBe(false);
    expect(looped.nextTick(WALL0, MONO0)?.[0]?.tick).toBe(0);
    expect(() => new ScenarioSource([], "empty", { loop: true }).nextTick(WALL0, MONO0)).toThrow(
      RangeError,
    );
  });

  it("announces a mode outside live / replay / hybrid as replay (app.py hello)", () => {
    const h = harness("core");
    (h.rt as { mode: string }).mode = "demo";
    expect(h.rt.hello().mode).toBe("replay");
    h.rt.mode = "hybrid";
    expect(h.rt.hello().mode).toBe("hybrid");
  });

  it("keeps its history and latest state apart from the messages it hands out", () => {
    const h = harness("core");
    h.rt.sourceLabel = "replay";
    const msgs = h.rt.step(
      scenarioRecords(runNamed("frustrated_masked"), 0, WALL0, MONO0),
      WALL0,
      MONO0,
    );
    const pred = msgs.find((m) => m.type === "prediction");
    if (pred?.type !== "prediction") throw new Error("no prediction");
    const before = canon(h.rt.latest.get(1)?.prediction);
    pred.layers.reverse();
    pred.simulated_fields.push("edited");
    pred.probabilities["Focused"] = -1;
    expect(canon(h.rt.latest.get(1)?.prediction)).toBe(before);
    expect(canon(h.rt.history[1])).toBe(before);
    expect(h.rt.history[1]).toBe(h.rt.latest.get(1)?.prediction); // one model_dump() shared, as in Python
  });
});

// ---------- Perturbed golden runs (served predictor, against Python) ----------

/**
 * The golden runs' records, perturbed by a xorshift32 stream (NaN, +-0.0, 1e-300, 1e300, half-unit ties, scaled
 * and shifted values, odd counts and people numbers, extra SIMULATED tags), with person 1 leaving for 20 ticks
 * and a second person (person 1's row as id 3) coming and going, stepped through Python's Runtime with the served
 * predictor and through this one. Python's canonical-text digest and audit head are pasted below; the generator
 * is scripts/golden_runtime/perturb.py, which uses exactly the constants and draw order of perturbRecord().
 * Exploratory runs of the same kind over all 9 scenarios (6480 ticks, both state sets) matched Python on every
 * tick.
 */
const PERTURB = {
  floats: [
    "ambient_db",
    "light_lux",
    "posture_slump",
    "head_in_hands_frac",
    "motion_energy",
    "face_valence",
    "face_visible",
    "hours_since_break",
    "tod_h",
    "room_db_base",
    "smile",
    "head_yaw",
  ] as const,
  special: [
    0.0, -0.0, 1e-300, 1e6, -1e6, 1e300, 0.5, 2.5, 0.0005, 0.00049999999999999, 23.999999, -0.01,
  ],
  mult: [0.5, 1.5, -1.0, 10.0],
  add: [0.0, 0.001, -0.25],
  ints: [
    "laughter_count",
    "compliments_count",
    "clenched_fists_count",
    "social_withdrawal_flag",
    "n_people",
  ] as const,
  intChoices: [0, 1, 2, 3, 7, 40],
  tags: ["low_light", "room_history", "été"],
};

/** xorshift32 (13, 17, 5), as scripts/golden_runtime/perturb.py's XorShift32: draws x / 2**32. */
class XorShift32 {
  constructor(private x: number) {}
  next(): number {
    let x = this.x;
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    this.x = x;
    return x / 4294967296;
  }
  pick<T>(xs: readonly T[]): T {
    return xs[Math.floor(this.next() * xs.length)] as T;
  }
}

function perturbRecord(rng: XorShift32, rec: TickRecord): TickRecord {
  const d = { ...rec, simulated_fields: [...rec.simulated_fields] };
  for (const k of PERTURB.floats) {
    const u = rng.next();
    if (u < 0.04) d[k] = NaN;
    else if (u < 0.055) d[k] = rng.pick(PERTURB.special);
    else if (u < 0.1 && !Number.isNaN(d[k]))
      d[k] = d[k] * rng.pick(PERTURB.mult) + rng.pick(PERTURB.add);
  }
  for (const k of PERTURB.ints) if (rng.next() < 0.03) d[k] = rng.pick(PERTURB.intChoices);
  if (rng.next() < 0.02) d.simulated_fields.push(rng.pick(PERTURB.tags));
  return d;
}

function perturbedTicks(
  run: RuntimeRun,
  n: number,
): { wall: number; mono: number; records: TickRecord[] }[] {
  const rng = new XorShift32(0x9e3779b9);
  return run.ticks.slice(0, n).map((t, i) => {
    let recs = t.records.map((d) => perturbRecord(rng, tickRecordFromDict(d)));
    if (i % 97 >= 40 && i % 97 < 60) recs = recs.filter((r) => r.person_id !== 1); // person 1 away
    if (i % 50 > 35) {
      const twins = recs
        .filter((r) => r.person_id === 1)
        .map((r) => ({
          ...r,
          person_id: 3,
          display_id: "Person #3",
          n_people: 2,
          simulated_fields: [...r.simulated_fields],
        }));
      recs = [...recs, ...twins];
    }
    return { wall: t.wall, mono: t.mono, records: recs };
  });
}

const PERTURBED: {
  name: string;
  ticks: number;
  predictions: number;
  digest: string;
  auditN: number;
  auditLast: string;
}[] = [
  {
    name: "full_demo",
    ticks: 200,
    predictions: 147,
    digest: "a86815ed26dff92da206f2902e674c8ef7526257f0eafc73d2ac630751d5729b",
    auditN: 37,
    auditLast: "9979a27c84526d68421c848b7498e0cc96a1118d935517d4b88b3c14a1e385b3",
  },
  {
    name: "overwhelmed",
    ticks: 156,
    predictions: 136,
    digest: "a5b14a238389e518a4e403582c5cbe86fb9105c138abe963d6e1098c1db82255",
    auditN: 33,
    auditLast: "2f3a1f6da04b5984a715a85f8c5307a8694cb6f169353251e9f1832a2cfdeaf5",
  },
  {
    name: "context_blindness",
    ticks: 120,
    predictions: 164,
    digest: "c2657d671596892efbf67fc1554f6a340e4a3b55f5966c301342219a907c267d",
    auditN: 34,
    auditLast: "ade2057cc7c125af7d3669ec2ad8a770601fdaddd3d85819d64855b2840569b7",
  },
];

describe("runtime against Python on perturbed golden runs (served predictor)", () => {
  for (const want of PERTURBED) {
    it(`${want.name}: same messages and audit chain for ${want.ticks} perturbed ticks`, () => {
      const run = runNamed(want.name);
      let now = WALL0;
      const rt = new Runtime(makeSettings({ state_set: run.state_set }), SERVED[run.state_set], {
        sessionId: "session-golden",
        clock: () => now,
        tz: "UTC",
      });
      rt.sourceLabel = "replay";
      rt.mode = "replay";
      const texts: string[] = [];
      let predictions = 0;
      for (const t of perturbedTicks(run, want.ticks)) {
        now = t.wall;
        const msgs = rt.step(t.records, t.wall, t.mono);
        predictions += msgs.filter((m) => m.type === "prediction").length;
        texts.push(canon(msgs));
      }
      expect([texts.length, predictions]).toEqual([want.ticks, want.predictions]);
      expect(sha256Hex(texts.join("\n"))).toBe(want.digest);
      const entries = rt.audit.entries();
      expect([entries.length, entries[entries.length - 1]?.hash]).toEqual([
        want.auditN,
        want.auditLast,
      ]);
    }, 120_000);
  }
});

// ---------- Performance (3 people) ----------

describe("runtime performance", () => {
  it("steps three people well inside the 5 s tick", () => {
    // frustrated_masked has one person: copy them as persons 2 and 3 (three full predictions per tick)
    const h = harness("core");
    h.rt.sourceLabel = "replay";
    const ticks = run3();
    const times: number[] = [];
    ticks.forEach((recs, i) => {
      const wall = WALL0 + i * 5.0;
      const t0 = performance.now();
      h.rt.step(recs, wall, MONO0 + i * 5.0);
      times.push(performance.now() - t0);
    });
    const steady = times.slice(10).sort((a, b) => a - b);
    const mean = steady.reduce((a, b) => a + b, 0) / steady.length;
    const p50 = steady[Math.floor(steady.length / 2)] as number;
    const p95 = steady[Math.floor(steady.length * 0.95)] as number;
    perf.notes.push(
      `3 people: ${steady.length} steps, mean ${mean.toFixed(2)} ms, p50 ${p50.toFixed(2)} ms, ` +
        `p95 ${p95.toFixed(2)} ms per step`,
    );
    expect(h.rt.latest.size).toBe(3);
    expect(p50).toBeLessThan(250); // a generous bound for a loaded CI box; measured ~20 ms
  }, 120_000);

  function run3(): TickRecord[][] {
    const r = runNamed("frustrated_masked");
    return r.ticks.map((t, i) => {
      const recs = scenarioRecords(r, i, WALL0 + i * 5.0, MONO0 + i * 5.0).map((x) => ({
        ...x,
        n_people: 3,
      }));
      const p1 = recs.filter((x) => x.person_id === 1);
      const clones = [2, 3].flatMap((pid) =>
        p1.map((x) => ({
          ...x,
          person_id: pid,
          display_id: `Person #${pid}`,
          face_valence: Number.isNaN(x.face_valence)
            ? x.face_valence
            : x.face_valence * (pid === 2 ? 0.5 : -0.5),
        })),
      );
      return [...recs, ...clones];
    });
  }
});

afterAll(() => {
  if (perf.steps > 0) {
    perf.notes.unshift(
      `golden replay: ${perf.steps} steps, ${perf.messages} messages, ${(perf.ms / perf.steps).toFixed(2)} ms per step`,
    );
  }
  console.info(`[engine-runtime] ${perf.notes.join("; ")}`);
});
