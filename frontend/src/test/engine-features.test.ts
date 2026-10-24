/**
 * window.ts + features.ts against the Python feature function (valence/features/compute.py) and the
 * rolling window (valence/features/window.py), via the golden replay in features.json.gz
 * (scripts/export_web_golden.py) and the feature lists in artifacts/.
 *
 * 118 of the 123 features must be bit-identical (Object.is, so NaN and -0.0 included), the ewm features
 * among them. tod_sin/tod_cos and lux_room_* pass through glibc's math.sin / math.cos / math.log2 in
 * Python and V8's Math functions here, which round differently in the last bit, so those five are held
 * to 2 ulps (and 1e-9 relative, the parity budget). The NaN pattern must match everywhere.
 */
import { describe, expect, it } from "vitest";
import { makeTickRecord, tickRecordFromDict, type TickRecord } from "@/engine/contracts";
import {
  ABS_ROOM,
  FEATURE_INDEX,
  FEATURE_NAMES,
  FEATURE_SETS,
  N_FEATURES,
  PERSON_FIELDS,
  arraysFromRecords,
  compute,
  featuresFromWindow,
  selectFeatures,
} from "@/engine/features";
import { isnan, ulpDistance } from "@/engine/num";
import { ROOM, RollingWindow } from "@/engine/window";
import { artifactJson, artifactText, loadFixture, nn } from "./engine-fixtures";

// ---------- Fixture ----------

type Num = number | null;
interface Checkpoint {
  tick: number;
  person_id: number;
  features: Num[];
}
interface Scenario {
  name: string;
  state_set: string;
  room_only: boolean;
  session_id: string;
  settings: { person_drop_min: number; tick_s: number; window_ticks: number };
  ticks: { wall: number; mono: number; records: Record<string, unknown>[] }[];
  checkpoints: Checkpoint[];
}
interface EdgeCase {
  name: string;
  person: Record<string, Num[]>;
  room: Record<string, Num[]>;
  features: Num[];
}
interface FeaturesFixture {
  version: number;
  wall0: number;
  mono0: number;
  scenarios: Scenario[];
  edge_cases: EdgeCase[];
}
interface ModelMeta {
  feature_names: string[];
  feature_sets: Record<string, string[]>;
}

const fixture = loadFixture<FeaturesFixture>("features");

// ---------- Comparison ----------

/** Features computed through glibc's math.sin / math.cos / math.log2, which V8 rounds differently. */
const TRANSCENDENTAL = new Set(
  FEATURE_NAMES.filter((n) => n === "tod_sin" || n === "tod_cos" || n.startsWith("lux_room_")),
);
const MAX_ULPS = 2;
const REL_TOL = 1e-9;

interface Stats {
  values: number;
  exact: number;
  maxAbs: number;
  maxRel: number;
  maxUlps: number;
  worst: string;
  failures: string[];
  /** Feature name -> number of values that were not bit-exact. */
  inexact: Map<string, number>;
}

const newStats = (): Stats => ({
  values: 0,
  exact: 0,
  maxAbs: 0,
  maxRel: 0,
  maxUlps: 0,
  worst: "",
  failures: [],
  inexact: new Map(),
});
const total = newStats();

/** Compares one vector, folding the result into `s` (and the file-wide total). */
function compare(s: Stats, where: string, actual: Float64Array, expectedRaw: Num[]): void {
  const expected = nn(expectedRaw);
  if (actual.length !== expected.length) {
    s.failures.push(`${where}: ${actual.length} features, Python ${expected.length}`);
    return;
  }
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] as number;
    const e = expected[i] as number;
    const name = FEATURE_NAMES[i] as string;
    for (const st of [s, total]) st.values++;
    if (Object.is(a, e)) {
      for (const st of [s, total]) st.exact++;
      continue;
    }
    if (isnan(a) !== isnan(e)) {
      s.failures.push(`${where} ${name}: ${a} vs Python ${e} (NaN pattern)`);
      continue;
    }
    for (const st of [s, total]) st.inexact.set(name, (st.inexact.get(name) ?? 0) + 1);
    const abs = Math.abs(a - e);
    const rel = abs / Math.max(1, Math.abs(e));
    const ulps = ulpDistance(a, e);
    for (const st of [s, total]) {
      if (abs > st.maxAbs) st.maxAbs = abs;
      if (rel > st.maxRel) st.maxRel = rel;
      if (ulps > st.maxUlps) {
        st.maxUlps = ulps;
        st.worst = `${where} ${name}: ${a} vs ${e}`;
      }
    }
    if (!TRANSCENDENTAL.has(name))
      s.failures.push(`${where} ${name}: ${a} vs Python ${e} (should be bit-exact)`);
    else if (rel > REL_TOL || ulps > MAX_ULPS)
      s.failures.push(`${where} ${name}: ${a} vs Python ${e} (${ulps} ulps)`);
  }
}

const summary = (s: Stats): string =>
  `${s.values} values, ${s.values - s.exact} not bit-exact, max abs ${s.maxAbs.toExponential(2)}, ` +
  `max rel ${s.maxRel.toExponential(2)}, max ${s.maxUlps} ulps${s.worst ? ` (${s.worst})` : ""}`;

// ---------- Replay ----------

/** valence/runtime/engine.py:346 Runtime._evict: drop people whose last row is older than person_drop_min. */
function evict(window: RollingWindow, wall: number, personDropMin: number): void {
  for (const pid of window.people()) {
    const rows = window.snapshot(pid);
    const lastRow = rows[rows.length - 1];
    if (lastRow !== undefined && wall - lastRow.ts > personDropMin * 60) window.drop(pid);
  }
}

/** Replays a scenario like Runtime.step and checks every checkpoint; returns the stats and checkpoint keys. */
function replay(sc: Scenario): { stats: Stats; keys: string[]; expectedKeys: string[] } {
  const stats = newStats();
  const window = new RollingWindow(sc.settings.window_ticks);
  const byTick = new Map<number, Checkpoint[]>();
  for (const c of sc.checkpoints) byTick.set(c.tick, [...(byTick.get(c.tick) ?? []), c]);
  const keys: string[] = [];
  const expectedKeys = sc.checkpoints.map((c) => `${c.tick}/${c.person_id}`);
  const lastTick = sc.ticks.length - 1;
  sc.ticks.forEach((tk, i) => {
    const records = tk.records.map(tickRecordFromDict);
    for (const r of records) window.append(r);
    evict(window, tk.wall, sc.settings.person_drop_min);
    // the exporter's checkpoint rule, re-derived here so window lengths and eviction are checked too
    const targets = sc.room_only
      ? [ROOM]
      : [...new Set(records.map((r) => r.person_id).filter((p) => p !== ROOM))].sort(
          (a, b) => a - b,
        );
    for (const pid of targets) {
      const person = window.snapshot(pid);
      if (person.length && (i % 2 === 0 || person.length <= 3 || i === lastTick))
        keys.push(`${i}/${pid}`);
    }
    for (const c of byTick.get(i) ?? []) {
      const person = window.snapshot(c.person_id);
      if (!person.length) {
        stats.failures.push(`tick ${i} person ${c.person_id}: empty window`);
        continue;
      }
      const room = window.room();
      compare(
        stats,
        `tick ${i} person ${c.person_id}`,
        featuresFromWindow(person, room.length ? room : person),
        c.features,
      );
    }
  });
  return { stats, keys, expectedKeys };
}

// ---------- Tests ----------

describe("feature names and sets", () => {
  const meta = artifactJson<ModelMeta>("model_meta.json");
  const metaAll = artifactJson<ModelMeta>("all/model_meta.json");

  it("lists the 123 unique feature names in the trained models' order", () => {
    expect(N_FEATURES).toBe(123);
    expect(new Set(FEATURE_NAMES).size).toBe(123);
    expect([...FEATURE_NAMES]).toEqual(meta.feature_names);
    expect([...FEATURE_NAMES]).toEqual(metaAll.feature_names);
    FEATURE_NAMES.forEach((n, i) => expect(FEATURE_INDEX[n]).toBe(i));
  });

  it("matches every feature set recorded in model_meta.json for both state sets", () => {
    for (const m of [meta, metaAll]) {
      expect(Object.keys(FEATURE_SETS).sort()).toEqual(Object.keys(m.feature_sets).sort());
      for (const [name, cols] of Object.entries(m.feature_sets))
        expect([...(FEATURE_SETS[name] ?? [])], name).toEqual(cols);
    }
    expect(FEATURE_SETS["m2"]).toHaveLength(101);
  });

  it("matches the feature_names header of the served m2 and m0 model files", () => {
    const header = (text: string): string[] => {
      const line = text.split("\n").find((l) => l.startsWith("feature_names="));
      return (line ?? "").slice("feature_names=".length).trim().split(" ");
    };
    for (const dir of ["", "all/"]) {
      expect(header(artifactText(`${dir}m2.txt`)), `${dir}m2.txt`).toEqual([
        ...(FEATURE_SETS["m2"] ?? []),
      ]);
      expect(header(artifactText(`${dir}m0.txt`)), `${dir}m0.txt`).toEqual([
        ...(FEATURE_SETS["m0"] ?? []),
      ]);
    }
  });

  it("keeps the absolute room levels out of the served m2 set", () => {
    expect(ABS_ROOM).toHaveLength(18);
    for (const n of ABS_ROOM) expect(FEATURE_SETS["m2"]).not.toContain(n);
  });

  it("selects model columns by name in the model's order", () => {
    const x = Float64Array.from(FEATURE_NAMES, (_n, i) => i);
    expect([...selectFeatures(x, ["face_now", "db_room_now", "tod_sin"])]).toEqual([
      FEATURE_INDEX["face_now"],
      FEATURE_INDEX["db_room_now"],
      FEATURE_INDEX["tod_sin"],
    ]);
    expect(() => selectFeatures(x, ["no_such_feature"])).toThrow();
  });
});

describe("rolling window", () => {
  const rec = (pid: number, tick: number): TickRecord =>
    makeTickRecord({
      tick,
      ts: tick * 5,
      t_mono: tick * 5,
      session_id: "s",
      person_id: pid,
      display_id: `P${pid}`,
      n_people: 1,
    });

  it("keeps the newest max_ticks rows per slot, oldest first", () => {
    const w = new RollingWindow(3);
    for (let t = 0; t < 5; t++) w.append(rec(1, t));
    expect(w.snapshot(1).map((r) => r.tick)).toEqual([2, 3, 4]);
    expect(w.fillMinutes(1)).toBe(0.25);
    expect(w.fillMinutes(1, 10)).toBe(0.5);
  });

  it("returns copies, so callers cannot change the window", () => {
    const w = new RollingWindow();
    w.append(rec(2, 0));
    w.snapshot(2).pop();
    expect(w.snapshot(2)).toHaveLength(1);
    expect(w.snapshot(9)).toEqual([]);
  });

  it("lists people in ascending id order without the room slot", () => {
    const w = new RollingWindow();
    for (const pid of [10, ROOM, 2, 33]) w.append(rec(pid, 0));
    expect(w.people()).toEqual([2, 10, 33]);
    expect(w.room().map((r) => r.person_id)).toEqual([ROOM]);
  });

  it("drops one person, every person, or everything", () => {
    const w = new RollingWindow();
    for (const pid of [ROOM, 1, 2]) w.append(rec(pid, 0));
    w.drop(1);
    w.drop(99);
    expect(w.people()).toEqual([2]);
    w.resetPeople();
    expect(w.people()).toEqual([]);
    expect(w.room()).toHaveLength(1);
    w.clear();
    expect(w.room()).toEqual([]);
    expect(w.fillMinutes(ROOM)).toBe(0);
  });
});

describe("scenario replay against Python", () => {
  it("covers all nine scenarios plus room-only windows", () => {
    expect(fixture.version).toBe(1);
    expect(new Set(fixture.scenarios.map((s) => s.name)).size).toBe(9);
    expect(fixture.scenarios.some((s) => s.room_only)).toBe(true);
  });

  for (const sc of fixture.scenarios) {
    const label = `${sc.name} (${sc.state_set}${sc.room_only ? ", room only" : ""})`;
    it(`reproduces every checkpoint of ${label}`, () => {
      const { stats, keys, expectedKeys } = replay(sc);
      expect(keys).toEqual(expectedKeys);
      expect(stats.failures.slice(0, 10)).toEqual([]);
      expect(stats.values).toBe(sc.checkpoints.length * N_FEATURES);
      console.info(`[features] ${label}: ${summary(stats)}`);
    });
  }
});

describe("compute edge cases against Python", () => {
  for (const ec of fixture.edge_cases) {
    it(`reproduces ${ec.name}`, () => {
      const stats = newStats();
      const toArrays = (d: Record<string, Num[]>): Record<string, Float64Array> =>
        Object.fromEntries(Object.entries(d).map(([k, v]) => [k, Float64Array.from(nn(v))]));
      compare(stats, ec.name, compute(toArrays(ec.person), toArrays(ec.room)), ec.features);
      expect(stats.failures).toEqual([]);
    });
  }

  it("refuses an empty person window and a missing channel, as Python raises", () => {
    const empty = Object.fromEntries(PERSON_FIELDS.map((f) => [f, new Float64Array(0)]));
    const room = {
      ambient_db: [50],
      light_lux: [300],
      laughter_count: [0],
      compliments_count: [0],
    };
    expect(() => compute(empty, room)).toThrow();
    const person = Object.fromEntries(PERSON_FIELDS.map((f) => [f, [0.5]]));
    expect(compute(person, room)).toHaveLength(N_FEATURES);
    expect(() => compute(person, { ...room, light_lux: undefined })).toThrow();
  });

  it("falls back to the person's rows for the room channels when the room slot is empty", () => {
    const rows = [0, 1, 2].map((tick) =>
      makeTickRecord({
        tick,
        ts: tick * 5,
        t_mono: tick * 5,
        session_id: "s",
        person_id: 1,
        display_id: "Person #1",
        n_people: 1,
        ambient_db: 50 + tick,
        room_db_base: 48,
        tod_h: 14,
      }),
    );
    const x = featuresFromWindow(rows, []);
    expect(x[FEATURE_INDEX["db_now"] as number]).toBe(52);
    expect(x[FEATURE_INDEX["db_room_now"] as number]).toBe(4);
    expect(x[FEATURE_INDEX["lux_room_now"] as number]).toBeNaN();
    expect(arraysFromRecords(rows, ["ambient_db"]).ambient_db).toEqual(
      Float64Array.from([50, 51, 52]),
    );
  });
});

describe("parity summary", () => {
  it("stays within the 1e-9 parity budget overall", () => {
    const inexact = [...total.inexact].map(([n, k]) => `${n} ${k}`).join(", ");
    console.info(
      `[features] total: ${summary(total)}; not bit-exact by feature: ${inexact || "none"}`,
    );
    expect(total.maxRel).toBeLessThanOrEqual(REL_TOL);
    for (const name of total.inexact.keys()) expect(TRANSCENDENTAL.has(name), name).toBe(true);
  });
});
