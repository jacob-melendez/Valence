/**
 * lgbm.ts + treeshap.ts against LightGBM 4.7.0 itself.
 *
 * models.json.gz (scripts/export_web_golden.py) holds, for the core and 6-state served models, the clipped
 * feature vector of every case with LightGBM's raw scores (`predict(raw_score=True)`) and contributions
 * (`predict(pred_contrib=True)`) for M2, and M0's raw scores. The models are read from ../artifacts.
 * m0-dump.json.gz holds `dump_model()` of artifacts/m0.txt and two small edge models (zero_as_missing splits,
 * a single-leaf tree) with LightGBM's predictions; its generator script is stored inside it ("generator").
 * The golden comparisons are exact: the port reproduces LightGBM's arithmetic operation for operation. (The
 * references come from an x86-64 build; an arm64 build may fuse multiply-adds in TreeSHAP, so contributions
 * regenerated there could move by an ulp or so. Raw scores are plain sums and stay exact everywhere.)
 * "loader edge cases" pins the inputs the fixtures do not reach, with LightGBM 4.7.0's own answers pasted in
 * as literals (each test names the Python call that produced them).
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

/** The doubles of `a`, with NaN kept as NaN, for toEqual (which treats NaN as equal to NaN and -0 as not 0). */
function doubles(a: ArrayLike<number>): number[] {
  return Array.from(a);
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
        expect(err, c.name).toBe(0);
      }
    });

    it(`match LightGBM for every ${set.state_set} M0 case`, () => {
      let n = 0;
      for (const c of set.cases) {
        if (c.m0 === null) continue;
        const raw = predictRaw(m0, pick(nn(c.xc), idx0));
        const err = maxAbsDiff(raw, c.m0.raw);
        track(`${set.state_set}.m0_raw`, err);
        expect(err, c.name).toBe(0);
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
          expect(err, `${c.name} class ${k}`).toBe(0);
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
        expect(err, c.name).toBe(0);
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
      expect(err, c.name).toBe(0);
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
        expect(errRaw, JSON.stringify(c.x)).toBe(0);
        treeShap(booster, x).forEach((row, k) => {
          const err = maxAbsDiff(row, c.contrib[k] as number[]);
          track(`edge.${edge.name}.contrib`, err);
          expect(err, `${JSON.stringify(c.x)} class ${k}`).toBe(0);
        });
      }
    });
  }
});

// ---------- Loader edge cases ----------

describe("loader edge cases", () => {
  const m0Text = artifactText("m0.txt");
  const edgeText = (dumps.edge.find((e) => e.name === "zero_as_missing") as EdgeModel).text;

  it("rejects a random-forest text model flagged by a bare average_output line", () => {
    // LightGBM writes the flag as a line without "="; Booster(model_str=...) then averages in predict():
    // m0.txt with the line added gives predict([[0.3, 0.9]])[0][0] == 0.24905614103657753 instead of
    // 0.13852170104078462 (raw scores unchanged), so a loader that skips "=-less" lines would mis-serve it.
    const rf = m0Text.replace("\nobjective=", "\naverage_output\nobjective=");
    expect(rf).not.toBe(m0Text);
    expect(() => parseLightGBMText(rf)).toThrow(/average_output/);
  });

  it("rejects linear trees in dump_model JSON, where only the leaf keys mark them", () => {
    // Second tree of lgb.train({objective: regression, linear_tree: True, num_leaves: 2}, ..., 2 rounds), the
    // lossy statistics stripped. LightGBM scores x = [0.5, -1] as leaf_const + coeff * x0 = 0.08519954288403338
    // (predict(start_iteration=1, raw_score=True)); the plain leaf_value would give 0.14202784945567448.
    const dump = {
      version: "v4",
      num_class: 1,
      num_tree_per_iteration: 1,
      max_feature_idx: 1,
      average_output: false,
      feature_names: ["Column_0", "Column_1"],
      tree_info: [
        {
          tree_index: 1,
          num_leaves: 2,
          num_cat: 0,
          shrinkage: 0.1,
          tree_structure: {
            split_index: 0,
            split_feature: 0,
            threshold: 0.03773942327185754,
            decision_type: "<=",
            default_left: true,
            missing_type: "None",
            internal_count: 400,
            left_child: {
              leaf_index: 0,
              leaf_value: -0.13110263406213982,
              leaf_count: 208,
              leaf_const: 0.007438403748495137,
              leaf_features: [0],
              leaf_coeff: [0.19013176940824442],
            },
            right_child: {
              leaf_index: 1,
              leaf_value: 0.14202784945567448,
              leaf_count: 192,
              leaf_const: -0.009783398358892904,
              leaf_features: [0],
              leaf_coeff: [0.18996588248585256],
            },
          },
        },
      ],
    };
    expect(() => loadDumpModel(dump)).toThrow(/linear/);
    expect(() => loadDumpModel(JSON.stringify(dump))).toThrow(/linear/);
  });

  it("falls back to num_class when the text model has no num_tree_per_iteration", () => {
    // LightGBM: Booster(model_str=m0.txt minus that line).predict([[0.3, 0.9]], raw_score=True)
    const b = parseLightGBMText(m0Text.replace(/\nnum_tree_per_iteration=\d+/, ""));
    expect(b.numTreePerIteration).toBe(4);
    expect(doubles(predictRaw(b, [0.3, 0.9]))).toEqual([
      -2.001377489502629, -0.769627663152717, -2.471805168970631, -1.2279741859515234,
    ]);
  });

  it("refuses feature_names that are missing or do not match max_feature_idx, as LightGBM does", () => {
    // LightGBM: "Wrong size of feature_names" / "Model file doesn't contain feature_names"
    expect(() =>
      parseLightGBMText(
        m0Text.replace("feature_names=face_now face_visible_now", "feature_names=face_now"),
      ),
    ).toThrow(/feature_names/);
    expect(() => parseLightGBMText(m0Text.replace(/\nfeature_names=[^\n]*/, ""))).toThrow(
      /feature_names/,
    );
  });

  it("reads a tree without decision_type as None-type splits that send defaults right", () => {
    // LightGBM: Booster(model_str=<zero_as_missing edge model minus its decision_type lines>)
    //   .predict(row, raw_score=True). The intact model reads [0, 0, 0] as
    //   [-0.8303622667276752, -0.5061907528226879, -3.316765646994915] (zero = missing there).
    const b = parseLightGBMText(edgeText.replace(/\ndecision_type=[^\n]*/g, ""));
    for (const t of b.trees) expect(Array.from(t.decision_type).every((d) => d === 0)).toBe(true);
    const cases: [number[], number[]][] = [
      [
        [0, 0, 0],
        [0.3136973290766354, -1.8492318323998478, -3.316765646994915],
      ],
      [
        [NaN, -0.3, 0.7],
        [0.3136973290766354, -1.8492318323998478, -3.316765646994915],
      ],
      [
        [0, 1e-36, NaN],
        [0.3136973290766354, -1.8492318323998478, -3.316765646994915],
      ],
      [
        [0.2, 0, -0.4],
        [0.3136973290766354, -1.1700453969869493, -3.316765646994915],
      ],
      [
        [-1, 2, 0],
        [0.3136973290766354, -1.8492318323998478, -3.316765646994915],
      ],
    ];
    for (const [x, raw] of cases) expect(doubles(predictRaw(b, x)), JSON.stringify(x)).toEqual(raw);
  });

  it("scores a tree without counts and gives NaN contributions, as LightGBM does", () => {
    // LightGBM: the edge model minus leaf_count / internal_count. Raw scores are unchanged; pred_contrib is NaN
    // in all 3 x (3 + 1) slots, because every cover fraction is 0 / 0.
    const b = parseLightGBMText(edgeText.replace(/\n(leaf_count|internal_count)=[^\n]*/g, ""));
    expect(doubles(predictRaw(b, [0, 0, 0]))).toEqual([
      -0.8303622667276752, -0.5061907528226879, -3.316765646994915,
    ]);
    expect(doubles(predictRaw(b, [0.2, 0, -0.4]))).toEqual([
      -0.7044660204869793, -0.5061907528226879, -3.316765646994915,
    ]);
    for (const row of treeShap(b, [0.2, 0, -0.4])) {
      expect(doubles(row)).toEqual([NaN, NaN, NaN, NaN]);
    }
  });

  it("reads a null or a hole in the input row as missing (NaN), like np.asarray(x, dtype=float)", () => {
    // LightGBM on m0.txt: predict([[nan, 0.5]]) differs from predict([[0, 0.5]]) (NaN-type split on face_now);
    // the old fall-through read null as 0.0.
    const m0 = parseLightGBMText(m0Text);
    const viaNull = predictRaw(m0, [null, 0.5] as unknown as number[]);
    expect(doubles(viaNull)).toEqual([
      -1.4753771734806413, -1.3771383609536862, -1.2496999826980713, -1.455466409123163,
    ]);
    expect(doubles(predictRaw(m0, [0, 0.5]))).toEqual([
      -0.630981019138393, -2.4027826619742387, -0.6090225304922018, -4.900583668576816,
    ]);
    const hole: number[] = [0.4];
    hole.length = 2; // [0.4, <empty>]
    expect(doubles(predictRaw(m0, hole))).toEqual([
      -3.428376758746547, -0.7896856821496412, -3.6741473310138724, -0.5009428834861246,
    ]);
    expect(doubles(treeShap(m0, [null, 0.5] as unknown as number[])[0] as Float64Array)).toEqual(
      doubles(treeShap(m0, [NaN, 0.5])[0] as Float64Array),
    );
  });

  it("follows a leaf value edited in place, like pred_contrib after Booster.set_leaf_output", () => {
    // LightGBM: g = Booster(model_file="artifacts/m0.txt"); g.predict(x, pred_contrib=True);
    //   g.set_leaf_output(0, 0, 1.5); then predict(x, pred_contrib=True) and predict(x, raw_score=True)
    //   with x = [[0.25, 0.75]]. The class-0 bias moves from -1.8880536872351714 to -1.841582280784014.
    const m0 = parseLightGBMText(m0Text);
    const x = [0.25, 0.75];
    expect((treeShap(m0, x)[0] as Float64Array)[2]).toBe(-1.8880536872351714);
    (m0.trees[0] as Tree).leaf_value[0] = 1.5;
    expect(treeShap(m0, x).map(doubles)).toEqual([
      [0.3911662078169201, -0.026139131712876854, -1.841582280784014],
      [0.41422058886612856, -0.09274118235961691, -1.4401575627441647],
      [0.06803649267521975, 0.026686479730921222, -1.8867516393482313],
      [1.2030125110599956, 0.010213744904838439, -2.6568512248785594],
    ]);
    expect(doubles(predictRaw(m0, x))).toEqual([
      -1.4765552046799704, -1.118678156237653, -1.7920286669420897, -1.4436249689137262,
    ]);
  });
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
