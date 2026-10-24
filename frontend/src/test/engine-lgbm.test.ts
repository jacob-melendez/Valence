/**
 * lgbm.ts + treeshap.ts against LightGBM 4.7.0 itself.
 *
 * models.json.gz (scripts/export_web_golden.py) holds, for the core and 6-state served models, the clipped
 * feature vector of every case with LightGBM's raw scores (`predict(raw_score=True)`) and contributions
 * (`predict(pred_contrib=True)`) for M2, and M0's raw scores. The models are read from ../artifacts.
 * m0-dump.json.gz holds `dump_model()` of artifacts/m0.txt and two small edge models (zero_as_missing splits,
 * a single-leaf tree) with LightGBM's predictions; its generator script is stored inside it ("generator").
 */
import { afterAll, describe, expect, it } from "vitest";
import {
  type Booster,
  type Tree,
  K_ZERO_THRESHOLD,
  loadDumpModel,
  parseLightGBMText,
  predictRaw,
} from "@/engine/lgbm";
import { expectedValue, treeShap } from "@/engine/treeshap";
import { artifactJson, artifactText, loadFixture, nn } from "./engine-fixtures";

// ---------- Fixtures ----------

interface ModelCase {
  name: string;
  xc: (number | null)[];
  m2_raw: number[];
  m2_contrib: number[][];
  k: number;
  m0: null | { raw: number[]; k0: number; face_push: number };
}
interface ModelSet {
  state_set: string;
  artifacts: string;
  states: string[];
  m2_features: string[];
  m0_features: string[];
  cases: ModelCase[];
}
interface EdgeModel {
  name: string;
  text: string;
  dump: unknown;
  cases: { x: (number | string | null)[]; raw: number[]; contrib: number[][] }[];
}

const models = loadFixture<{ version: number; sets: ModelSet[] }>("models");
const dumps = loadFixture<{ version: number; m0: unknown; edge: EdgeModel[] }>("m0-dump");

const FACE_LAYER = /^face_/; // the 7 face_* M2 columns (features/spec.py layer "Facial Surface")

interface Loaded {
  set: ModelSet;
  m2: Booster;
  m0: Booster;
  idx2: number[];
  idx0: number[];
}

function load(set: ModelSet): Loaded {
  const dir = set.artifacts.replace(/^artifacts\/?/, "");
  const rel = (f: string) => (dir ? `${dir}/${f}` : f);
  const names = artifactJson<{ feature_names: string[] }>(rel("model_meta.json")).feature_names;
  const index = (n: string) => {
    const i = names.indexOf(n);
    if (i < 0) throw new Error(`unknown feature ${n}`);
    return i;
  };
  return {
    set,
    m2: parseLightGBMText(artifactText(rel("m2.txt"))),
    m0: parseLightGBMText(artifactText(rel("m0.txt"))),
    idx2: set.m2_features.map(index),
    idx0: set.m0_features.map(index),
  };
}

const LOADED = models.sets.map(load);
const pick = (x: number[], idx: number[]) => Float64Array.from(idx, (i) => x[i] as number);

// Largest errors seen, printed once at the end (the numbers the port is reported with).
const maxErr: Record<string, number> = {};
function track(key: string, err: number): void {
  maxErr[key] = Math.max(maxErr[key] ?? 0, err);
}
afterAll(() => {
  console.log(`engine-lgbm max abs errors: ${JSON.stringify(maxErr)}`);
});

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    if (Number.isNaN(x) || Number.isNaN(y)) {
      if (!(Number.isNaN(x) && Number.isNaN(y))) return Infinity;
      continue;
    }
    m = Math.max(m, Math.abs(x - y));
  }
  return m;
}

function sum(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] as number;
  return s;
}

function expectSameTrees(a: Booster, b: Booster): void {
  expect(a.numClass).toBe(b.numClass);
  expect(a.numTreePerIteration).toBe(b.numTreePerIteration);
  expect(a.numFeatures).toBe(b.numFeatures);
  expect(a.featureNames).toEqual(b.featureNames);
  expect(a.trees.length).toBe(b.trees.length);
  const keys: (keyof Tree)[] = [
    "num_leaves",
    "split_feature",
    "threshold",
    "decision_type",
    "left_child",
    "right_child",
    "leaf_value",
    "leaf_count",
    "internal_count",
    "shrinkage",
    "max_depth",
  ];
  a.trees.forEach((t, i) => {
    const u = b.trees[i] as Tree;
    for (const key of keys) {
      const x = t[key];
      const y = u[key];
      if (typeof x === "number") expect([i, key, x]).toEqual([i, key, y]);
      else expect([i, key, Array.from(x)]).toEqual([i, key, Array.from(y as typeof x)]);
    }
  });
}

// ---------- Text model ----------

describe("text model parser", () => {
  it("reads every tree and the feature order of the served models", () => {
    const expected: Record<string, [number, number, number, number]> = {
      core: [4, 1044, 2, 408],
      all: [6, 1524, 2, 624],
    };
    for (const { set, m2, m0 } of LOADED) {
      const [k, n2, f0, n0] = expected[set.state_set] ?? [0, 0, 0, 0];
      expect(m2.numClass).toBe(k);
      expect(m2.numTreePerIteration).toBe(k);
      expect(m2.trees.length).toBe(n2);
      expect(m0.trees.length).toBe(n0);
      expect(m0.numFeatures).toBe(f0);
      expect(set.states.length).toBe(k);
      // served.py feeds xc[idx2] by position, so the booster's own column order must be cols2.
      expect(m2.featureNames).toEqual(set.m2_features);
      expect(m0.featureNames).toEqual(set.m0_features);
      expect(m2.numFeatures).toBe(set.m2_features.length);
    }
  });

  it("rejects models it cannot evaluate faithfully", () => {
    const text = artifactText("m0.txt");
    expect(() => parseLightGBMText(text.replace("version=v4", "version=v3"))).toThrow(/version/);
    expect(() => parseLightGBMText(text.replace("num_cat=0", "num_cat=1"))).toThrow(/categorical/);
    expect(() => parseLightGBMText(text.replace("is_linear=0", "is_linear=1"))).toThrow(/linear/);
    expect(() =>
      parseLightGBMText(text.replace(/^decision_type=(\d+)/m, "decision_type=1")),
    ).toThrow(/categorical/);
    expect(() => parseLightGBMText(text.replace(/^left_child=(-?\d+)/m, "left_child=99"))).toThrow(
      /out of range/,
    );
  });

  it("refuses a row of the wrong width, as Booster.predict does", () => {
    const { m2 } = LOADED[0] as Loaded;
    expect(() => predictRaw(m2, new Float64Array(123))).toThrow(/expects 101 features/);
    expect(() => treeShap(m2, new Float64Array(100))).toThrow(/expects 101 features/);
  });

  it("uses LightGBM's float zero threshold", () => {
    expect(K_ZERO_THRESHOLD).toBe(1.0000000180025095e-35);
  });
});

// ---------- Raw scores ----------

describe("raw scores", () => {
  for (const { set, m2, m0, idx2, idx0 } of LOADED) {
    it(`match LightGBM for every ${set.state_set} M2 case`, () => {
      for (const c of set.cases) {
        const raw = predictRaw(m2, pick(nn(c.xc), idx2));
        const err = maxAbsDiff(raw, c.m2_raw);
        track(`${set.state_set}.m2_raw`, err);
        expect(err, c.name).toBeLessThanOrEqual(1e-12);
      }
    });

    it(`match LightGBM for every ${set.state_set} M0 case`, () => {
      let n = 0;
      for (const c of set.cases) {
        if (c.m0 === null) continue;
        const raw = predictRaw(m0, pick(nn(c.xc), idx0));
        const err = maxAbsDiff(raw, c.m0.raw);
        track(`${set.state_set}.m0_raw`, err);
        expect(err, c.name).toBeLessThanOrEqual(1e-12);
        n++;
      }
      expect(n).toBeGreaterThan(20);
    });
  }
});

// ---------- TreeSHAP ----------

describe("treeshap", () => {
  for (const { set, m2, idx2 } of LOADED) {
    it(`matches pred_contrib for every class of every ${set.state_set} case`, () => {
      for (const c of set.cases) {
        const phi = treeShap(m2, pick(nn(c.xc), idx2));
        expect(phi.length).toBe(set.states.length);
        phi.forEach((row, k) => {
          const err = maxAbsDiff(row, c.m2_contrib[k] as number[]);
          track(`${set.state_set}.m2_contrib`, err);
          expect(err, `${c.name} class ${k}`).toBeLessThanOrEqual(1e-9);
        });
      }
    });

    it(`sums to the raw score for every ${set.state_set} case`, () => {
      for (const c of set.cases) {
        const x = pick(nn(c.xc), idx2);
        const raw = predictRaw(m2, x);
        treeShap(m2, x).forEach((row, k) => {
          const err = Math.abs(sum(row) - (raw[k] as number));
          track(`${set.state_set}.additivity`, err);
          expect(err, `${c.name} class ${k}`).toBeLessThanOrEqual(1e-12);
        });
      }
    });

    it(`gives the same rows when asked for a subset of ${set.state_set} classes`, () => {
      for (const c of set.cases.filter((_, i) => i % 5 === 0)) {
        const x = pick(nn(c.xc), idx2);
        const all = treeShap(m2, x);
        const k0 = c.m0?.k0 ?? (c.k + 1) % set.states.length;
        const sub = treeShap(m2, x, [c.k, k0, c.k]);
        expect(Array.from(sub[0] as Float64Array)).toEqual(Array.from(all[c.k] as Float64Array));
        expect(Array.from(sub[1] as Float64Array)).toEqual(Array.from(all[k0] as Float64Array));
        expect(Array.from(sub[2] as Float64Array)).toEqual(Array.from(all[c.k] as Float64Array));
        expect(sub[2]).not.toBe(sub[0]);
      }
    });

    it(`reproduces the ${set.state_set} face push of the M0 class`, () => {
      const face = set.m2_features.map((n) => FACE_LAYER.test(n));
      expect(face.filter(Boolean).length).toBe(7);
      for (const c of set.cases) {
        if (c.m0 === null) continue;
        const [row] = treeShap(m2, pick(nn(c.xc), idx2), [c.m0.k0]);
        let push = 0;
        face.forEach((f, j) => {
          if (f) push += (row as Float64Array)[j] as number;
        });
        const err = Math.abs(push - c.m0.face_push);
        track(`${set.state_set}.face_push`, err);
        expect(err, c.name).toBeLessThanOrEqual(1e-9);
      }
    });
  }

  it("puts each tree's expected value in the bias slot", () => {
    const { m2, idx2, set } = LOADED[0] as Loaded;
    const K = m2.numTreePerIteration;
    const bias = new Float64Array(K);
    m2.trees.forEach((t, i) => {
      bias[i % K] = (bias[i % K] as number) + expectedValue(t);
    });
    const phi = treeShap(m2, pick(nn((set.cases[0] as ModelCase).xc), idx2));
    phi.forEach((row, k) => expect(row[m2.numFeatures]).toBe(bias[k]));
  });

  it("rejects classes the model does not have", () => {
    const { m2 } = LOADED[0] as Loaded;
    const x = new Float64Array(m2.numFeatures);
    expect(() => treeShap(m2, x, [4])).toThrow(/class/);
    expect(() => treeShap(m2, x, [-1])).toThrow(/class/);
  });
});

// ---------- dump_model JSON ----------

describe("dump_model loader", () => {
  it("builds the same trees from m0's dump_model JSON as from m0.txt", () => {
    expectSameTrees(loadDumpModel(dumps.m0), parseLightGBMText(artifactText("m0.txt")));
  });

  it("gives LightGBM's M0 raw scores from the dump", () => {
    const core = LOADED.find((l) => l.set.state_set === "core") as Loaded;
    const m0 = loadDumpModel(JSON.stringify(dumps.m0));
    for (const c of core.set.cases) {
      if (c.m0 === null) continue;
      const err = maxAbsDiff(predictRaw(m0, pick(nn(c.xc), core.idx0)), c.m0.raw);
      track("core.m0_raw_from_dump", err);
      expect(err, c.name).toBeLessThanOrEqual(1e-12);
    }
  });

  for (const edge of dumps.edge) {
    it(`reads the ${edge.name} edge model identically from text and dump`, () => {
      expectSameTrees(loadDumpModel(edge.dump), parseLightGBMText(edge.text));
    });

    it(`matches LightGBM on the ${edge.name} edge model (missing types, tiny values, ties)`, () => {
      const booster = parseLightGBMText(edge.text);
      for (const c of edge.cases) {
        const x = nn(c.x) as number[];
        const raw = predictRaw(booster, x);
        const errRaw = maxAbsDiff(raw, c.raw);
        track(`edge.${edge.name}.raw`, errRaw);
        expect(errRaw, JSON.stringify(c.x)).toBeLessThanOrEqual(1e-12);
        treeShap(booster, x).forEach((row, k) => {
          const err = maxAbsDiff(row, c.contrib[k] as number[]);
          track(`edge.${edge.name}.contrib`, err);
          expect(err, `${JSON.stringify(c.x)} class ${k}`).toBeLessThanOrEqual(1e-9);
        });
      }
    });
  }
});

// ---------- Speed ----------

describe("speed", () => {
  it("scores and explains one window on core M2 in a few milliseconds", () => {
    const { m2, idx2, set } = LOADED.find((l) => l.set.state_set === "core") as Loaded;
    const xs = set.cases.map((c) => ({ x: pick(nn(c.xc), idx2), k: c.k, k0: c.m0?.k0 ?? c.k }));
    for (const { x, k, k0 } of xs.slice(0, 20)) {
      predictRaw(m2, x); // warm-up (JIT)
      treeShap(m2, x, [k, k0]);
    }
    const t0 = performance.now();
    let n = 0;
    while (n < 2 * xs.length) {
      const { x, k, k0 } = xs[n % xs.length] as (typeof xs)[number];
      predictRaw(m2, x);
      treeShap(m2, x, [k, k0]);
      n++;
    }
    const ms = (performance.now() - t0) / n;
    console.log(
      `engine-lgbm speed: predictRaw + treeShap([k, k0]) on core M2 = ${ms.toFixed(3)} ms per vector (${n} vectors)`,
    );
    expect(ms).toBeLessThan(50);
  });
});
