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
 *   pasted below together with the audit chain head. The expected values come from gen_runtime_extra.py
 *   (not in the repo yet; suggested home scripts/export_web_golden_runtime.py), which drives the Python Runtime
 *   through exactly the steps of EXTRA_RUNS, on the WALL0/MONO0 clocks of runtime.json.gz, with a purge-gate
 *   stub and np.random.normal patched to the draws of fixedDraws(). Rerun it whenever valence/ changes.
 * - Behaviour without a Python counterpart (null records, onMessage, record copies, refused audit payloads,
 *   the tracks throttle) and a 3-person timing.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { auditBlob, entryBody, pyJsonDumps, verifyChain, type AuditEntry } from "@/engine/audit";
import {
  makeSettings,
  obsFromDict,
  tickRecordFromDict,
  type Obs,
  type Settings,
  type TickRecord,
} from "@/engine/contracts";
import { pyRepr } from "@/engine/num";
import { replayInto } from "@/engine/replay";
import { Runtime, type RuntimeOptions, type TickMessage } from "@/engine/runtime";
import { ServedPredictor, type ClipBounds, type ServedMeta } from "@/engine/served";
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

/** One extra run as Python produced it (gen_runtime_extra.py). */
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

// ---------- SCRATCH (edge-case exploration; removed before handing in) ----------
import { readFileSync } from "node:fs";
import type { PredictionOutput, Predictor } from "@/engine/served";
import { STATE_SETS, makeTickRecord } from "@/engine/contracts";

const SCRATCH = "/tmp/claude-0/-home-user-Valence/edc06eea-8c6d-582a-80d8-fd955eff8124/scratchpad/specs";
function decS(v: unknown): unknown {
  if (v === "NaN") return NaN;
  if (v === "Infinity") return Infinity;
  if (v === "-Infinity") return -Infinity;
  if (Array.isArray(v)) return v.map(decS);
  if (v !== null && typeof v === "object")
    return Object.fromEntries(Object.entries(v as Record<string, unknown>).map(([k, x]) => [k, decS(x)]));
  return v;
}
function recS(d: Record<string, unknown>, session: string): TickRecord {
  const x: Record<string, unknown> = { session_id: session, display_id: `Person #${String(d["person_id"] ?? 0)}`, ...d };
  return tickRecordFromDict(x);
}
function runSpec(spec: any): { ticks: string[]; audit: { kind: string; hash: string }[]; state: unknown } {
  const outs: Record<string, any> = {};
  for (const st of spec.steps) for (const [k, o] of Object.entries(st.outs ?? {})) outs[k] = decS(o);
  const states = (STATE_SETS as Record<string, readonly string[]>)[spec.state_set ?? "core"] as readonly string[];
  const stub: Predictor = {
    states,
    modelId: "stub-edge",
    predict(person) {
      const last = person[person.length - 1] as TickRecord;
      const o = outs[`${last.tick}:${last.person_id}`];
      if (o === undefined) return null;
      return {
        label: o.label, confidence: o.confidence, probabilities: o.probabilities,
        face_only_label: o.face_only_label ?? null, face_only_confidence: o.face_only_confidence ?? null,
        layers: (o.layers ?? []).map((l: any) => ({ name: l.name, contribution: l.contribution, drivers: l.drivers ?? [] })),
        raw_override: o.raw_override ?? null, values: {}, ood_features: o.ood_features ?? [],
      } as PredictionOutput;
    },
  };
  const clock = { now: spec.wall_init ?? 0, mono: spec.mono_init ?? 0 };
  let k = 0;
  const draws: number[] = spec.draws ?? [];
  const rt = new Runtime(makeSettings(spec.settings ?? {}), stub, {
    sessionId: "session-edge", clock: () => clock.now, monoClock: () => clock.mono, tz: "UTC",
    gauss: () => draws[k++] as number,
  });
  rt.roomOnly = spec.room_only ?? false;
  rt.sourceLabel = spec.label ?? "replay";
  const ticks: string[] = [];
  for (const st of spec.steps) {
    for (const act of st.pre ?? []) {
      const op = act[0];
      if (op === "clock") clock.now = act[1];
      else if (op === "overlay") rt.setOverlay(act[1], act[2]);
      else if (op === "room_only") rt.setRoomOnly(act[1]);
      else if (op === "reset") rt.resetPeople(act[1]);
      else if (op === "break") rt.logBreak(act[1]);
      else if (op === "purge") rt.setPurgeCounters(act[1]);
      else if (op === "seed") rt.seed(act[1].map((r: any) => recS(r, "seed")), act[2], decS(act[3]) as number, decS(act[4]) as number | null);
      else if (op === "audit") rt.recordAudit(act[1], decS(act[2]) as any);
      else throw new Error(op);
    }
    clock.now = st.wall; clock.mono = st.mono;
    const recs = st.records === null ? null : st.records.map((r: any) => recS(r, "session-edge"));
    ticks.push(canon(rt.step(recs, st.wall, st.mono)));
  }
  const entries = rt.audit.entries();
  return {
    ticks,
    audit: entries.map((e) => ({ kind: e.kind, hash: e.hash })),
    state: {
      latest: [...rt.latest.keys()].sort((a, b) => a - b), smoothers: [...rt.smoothers.keys()].sort((a, b) => a - b),
      history: rt.history.length, gap_ticks: rt.gapTicks, paused: rt.paused, tick: rt.tickIndex,
      people: rt.window.people(), verify: verifyChain(entries),
    },
  };
}
describe("SCRATCH edge specs", () => {
  for (const name of ["sv", "room", "op"]) {
    it(`matches Python on ${name}`, () => {
      const spec = JSON.parse(readFileSync(`${SCRATCH}/${name}.json`, "utf8"));
      const want = JSON.parse(readFileSync(`${SCRATCH}/${name}.py.json`, "utf8"));
      let got: ReturnType<typeof runSpec>;
      try { got = runSpec(spec); } catch (e) { throw new Error(`${name} threw: ${String(e)}`); }
      const bad: string[] = [];
      want.ticks.forEach((w: string, i: number) => {
        const g = got.ticks[i] ?? "<missing>";
        if (g !== w) {
          let j = 0;
          while (j < w.length && g[j] === w[j]) j++;
          bad.push(`tick ${i} @${j}: got ...${g.slice(Math.max(0, j - 80), j + 120)}\n   want ...${w.slice(Math.max(0, j - 80), j + 120)}`);
        }
      });
      console.log(name, bad.slice(0, 6).join("\n"));
      console.log(name, "audit", JSON.stringify(got.audit.map((a) => a.kind)), JSON.stringify(want.audit.map((a: any) => a.kind)));
      console.log(name, "state", JSON.stringify(got.state), JSON.stringify(want.state));
      expect(bad.length).toBe(0);
      expect(got.audit.map((a) => a.hash)).toEqual(want.audit.map((a: any) => a.hash));
    });
  }
});
