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
  featureIndex,
  featureSet,
  featuresFromWindow,
  selectFeatures,
} from "@/engine/features";
import { parseLightGBMText } from "@/engine/lgbm";
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

// ---------- Beyond the fixtures ----------
// Inputs the golden replay never produces, each found by an adversarial review against the Python code.
// The expected values are literals printed by .venv/bin/python running valence/features/compute.py.

/** A plain person window: n ticks, every channel constant. */
function plainPerson(n: number, tod = 14.0): Record<string, Float64Array> {
  const full = (v: number): Float64Array => new Float64Array(n).fill(v);
  return {
    motion_energy: full(0.03),
    posture_slump: full(0.2),
    head_in_hands_frac: full(0),
    clenched_fists_count: full(0),
    social_withdrawal_flag: full(0),
    n_people: full(1),
    hours_since_break: full(1.5),
    tod_h: full(tod),
    face_valence: full(0.1),
    face_visible: full(1),
  };
}

/** A room window with the given dB series, constant light and a 0 dB / luxBase baseline. */
function plainRoom(db: number[], lux?: number[], luxBase = 300.0): Record<string, Float64Array> {
  const n = db.length;
  const full = (v: number): Float64Array => new Float64Array(n).fill(v);
  return {
    ambient_db: Float64Array.from(db),
    light_lux: lux ? Float64Array.from(lux) : full(320),
    laughter_count: full(0),
    compliments_count: full(0),
    room_db_base: full(0),
    room_lux_base: full(luxBase),
  };
}

/** `v` moved k ulps along the doubles (finite v). */
function stepUlps(v: number, k: number): number {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, v);
  const bits = dv.getBigInt64(0);
  const ord = (bits < 0n ? -(bits & 0x7fffffffffffffffn) : bits) + BigInt(k);
  dv.setBigUint64(0, ord < 0n ? -ord | (1n << 63n) : ord);
  return dv.getFloat64(0);
}

describe("inputs beyond the golden fixtures", () => {
  const at = (x: Float64Array, name: string): number => x[FEATURE_INDEX[name] as number] as number;

  it("returns the zero numpy's SIMD fmax reduction returns for a zero max", () => {
    // Python: compute(person, room) with these ambient_db series; '-0' is -0.0
    const cases: [string, number[], number[]][] = [
      ["first zero wins below 8 values", [-0.0, 0.0], [-0, -0, -0, -0]],
      ["one 8-lane vector replaces the seed", [0.0, ...Array<number>(8).fill(-0.0)], [-0, -0, -0, -0]],
      [
        "150 ticks of zeros and NaNs",
        Array.from({ length: 150 }, (_v, i) => (i % 7 === 0 ? NaN : i % 3 === 0 ? 0.0 : -0.0)),
        [0, 0, 0, 0],
      ],
      [
        "75 ticks through the 64-value unrolled step",
        Array.from({ length: 75 }, (_v, i) => (i % 5 === 0 ? -1.0 : i % 4 ? -0.0 : 0.0)),
        [-0, -0, -0, -0],
      ],
    ];
    for (const [label, db, want] of cases) {
      const x = compute(plainPerson(db.length), plainRoom(db));
      const got = ["db_max_1m", "db_max_5m", "db_max_15m", "db_room_max_5m"].map((n) => at(x, n));
      got.forEach((g, i) => expect(Object.is(g, want[i]), `${label}: ${g} vs ${want[i]}`).toBe(true));
    }
  });

  it("raises where math.sin and math.log2 raise instead of returning NaN or -Infinity", () => {
    // Python: ValueError('math domain error') for each of these
    expect(() => compute(plainPerson(3, 1e308), plainRoom([50, 50, 50]))).toThrow("math domain error");
    expect(() => compute(plainPerson(3, -2.9e307), plainRoom([50, 50, 50]))).toThrow("math domain error");
    expect(() => compute(plainPerson(3, Infinity), plainRoom([50, 50, 50]))).toThrow("math domain error");
    for (const [lux, base] of [
      [320.0, Infinity], // 320 / inf == 0
      [5e-324, 2.0], // underflows to 0
      [1e-300, 1e300],
    ] as const)
      expect(() => compute(plainPerson(3), plainRoom([50, 50, 50], [lux, lux, lux], base))).toThrow(
        "math domain error",
      );
  });

  it("keeps the infinite and NaN light ratios math.log2 lets through", () => {
    // Python: tod 2.86e307 -> (0.94280732448637, -0.33333818997356524); inf/inf -> nan; inf/300 and 1e300/1e-300 -> inf
    const x = compute(plainPerson(3, 2.86e307), plainRoom([50, 50, 50]));
    expect(ulpDistance(at(x, "tod_sin"), 0.94280732448637)).toBeLessThanOrEqual(MAX_ULPS);
    expect(ulpDistance(at(x, "tod_cos"), -0.33333818997356524)).toBeLessThanOrEqual(MAX_ULPS);
    const lux = (l: number, base: number): number[] => {
      const y = compute(plainPerson(3), plainRoom([50, 50, 50], [l, l, l], base));
      return ["lux_room_now", "lux_room_mean_5m", "lux_room_mean_15m"].map((n) => at(y, n));
    };
    expect(lux(Infinity, Infinity)).toEqual([NaN, NaN, NaN]);
    expect(lux(Infinity, 300)).toEqual([Infinity, Infinity, Infinity]);
    expect(lux(1e300, 1e-300)).toEqual([Infinity, Infinity, Infinity]);
  });

  it("reads a null record field as NaN, like np.array(..., dtype=float)", () => {
    // Python: np.array([None, 1, True], dtype=float) -> [nan, 1.0, 1.0]
    const rows = [null, 1, true].map(
      (v, tick) =>
        ({
          ...makeTickRecord({
            tick,
            ts: tick * 5,
            t_mono: tick * 5,
            session_id: "s",
            person_id: 1,
            display_id: "Person #1",
            n_people: 1,
          }),
          face_valence: v,
        }) as unknown as TickRecord,
    );
    expect([...arraysFromRecords(rows, ["face_valence"]).face_valence]).toEqual([NaN, 1, 1]);
  });

  it("rejects a negative or fractional window length on the first append, as deque(maxlen=...) does", () => {
    const row = makeTickRecord({
      tick: 0,
      ts: 0,
      t_mono: 0,
      session_id: "s",
      person_id: 4,
      display_id: "Person #4",
      n_people: 1,
    });
    // Python: RollingWindow(-1) and RollingWindow(2.5) construct; append raises ValueError / TypeError
    for (const bad of [-1, 2.5]) {
      const w = new RollingWindow(bad);
      expect(w.people()).toEqual([]);
      expect(() => w.append(row)).toThrow(bad < 0 ? "maxlen must be non-negative" : "an integer is required");
    }
    // Python: RollingWindow(0) keeps an empty slot -> people() [4], snapshot(4) [], fill_minutes(4) 0.0
    const w0 = new RollingWindow(0);
    w0.append(row);
    expect(w0.people()).toEqual([4]);
    expect(w0.snapshot(4)).toEqual([]);
    expect(w0.fillMinutes(4)).toBe(0);
  });

  it("treats Object method names as unknown features and sets, like a Python KeyError", () => {
    for (const name of ["constructor", "toString", "__proto__", "hasOwnProperty"]) {
      expect(FEATURE_INDEX[name], name).toBeUndefined();
      expect(FEATURE_SETS[name], name).toBeUndefined();
      expect(() => featureIndex(name)).toThrow();
      expect(() => featureSet(name)).toThrow();
    }
  });

  it("keeps the last-bit sin/cos rounding gap from flipping any served split for every whole-second tod", () => {
    // Live tod_h is local_tod(): h + m/60 + s/3600, so all 86400 values can be checked. V8 and glibc differ by
    // 1 ulp on about 3% of them; a decision `x <= t` (or a clip bound) can only flip if t lies within the
    // 2-ulp band around the V8 value.
    // The sweep evaluates compute()'s formula directly (compute() itself is too slow for 86400 calls in
    // jsdom); every 97th second goes through compute() to tie the two together.
    const live = { tod_sin: new Float64Array(86400), tod_cos: new Float64Array(86400) };
    for (let k = 0; k < 86400; k++) {
      const h = Math.floor(k / 3600);
      const m = Math.floor((k % 3600) / 60);
      const tod = h + m / 60.0 + (k % 60) / 3600.0;
      live.tod_sin[k] = Math.sin((2 * Math.PI * tod) / 24);
      live.tod_cos[k] = Math.cos((2 * Math.PI * tod) / 24);
      if (k % 97 === 0) {
        const x = compute(plainPerson(1, tod), plainRoom([50]));
        expect(at(x, "tod_sin")).toBe(live.tod_sin[k]);
        expect(at(x, "tod_cos")).toBe(live.tod_cos[k]);
      }
    }
    for (const dir of ["", "all/"]) {
      const booster = parseLightGBMText(artifactText(`${dir}m2.txt`));
      const bounds = artifactJson<Record<string, number[]>>(`${dir}clip_bounds.json`);
      for (const feat of ["tod_sin", "tod_cos"] as const) {
        const fi = booster.featureNames.indexOf(feat);
        const cuts: number[] = [...(bounds[feat] ?? [])];
        for (const t of booster.trees)
          t.split_feature.forEach((f, i) => {
            if (f === fi) cuts.push(t.threshold[i] as number);
          });
        expect(cuts.length, `${dir}m2 ${feat}`).toBeGreaterThan(100);
        cuts.sort((a, b) => a - b);
        let near = 0;
        for (const v of live[feat]) {
          const lo = stepUlps(v, -MAX_ULPS);
          const hi = stepUlps(v, MAX_ULPS);
          // first cut >= lo
          let a = 0;
          let b = cuts.length;
          while (a < b) {
            const mid = (a + b) >> 1;
            if ((cuts[mid] as number) < lo) a = mid + 1;
            else b = mid;
          }
          if (a < cuts.length && (cuts[a] as number) < hi) near++;
        }
        expect(near, `${dir}m2 ${feat}: cuts within ${MAX_ULPS} ulps of a live value`).toBe(0);
      }
    }
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
