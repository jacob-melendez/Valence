/**
 * served.ts + smoother.ts against the Python served predictor (valence/models/served.py) and PersonSmoother
 * (valence/models/predictor.py).
 *
 * - models.json.gz (scripts/export_web_golden.py): every case of the core and 6-state served models, with the
 *   clipped vector, OOD list, M2 raw scores and SHAP rows of every class, probabilities, the M0 shadow reading,
 *   the override condition and the layers. Exact (Object.is) everywhere except p and the confidences, which go
 *   through Math.exp vs numpy's SIMD np.exp and are held to 1e-12.
 * - smoother.json.gz (generator source stored inside it under "generator"): PersonSmoother on crafted and
 *   random sequences with its full state after every step, constructor errors, the ServedPredictor tables
 *   (clip bounds, OOD bands, face columns), _clip on crafted vectors, ood_band on odd names, and
 *   counterfactual() on windows replayed from runtime.json.gz.
 * - runtime.json.gz: the tick records replayed through RollingWindow + served.predict + PersonSmoother, checked
 *   against the prediction messages Runtime published (smoothed label, display label, override, warm-up,
 *   rounded probabilities, face-only reading, layers).
 */
import { afterAll, describe, expect, it } from "vitest";
import {
  ROOM,
  displayLabel,
  tickRecordFromDict,
  type Layer,
  type TickRecord,
} from "@/engine/contracts";
import { FEATURE_NAMES } from "@/engine/features";
import { pyRound, ulpDistance } from "@/engine/num";
import {
  type ClipBounds,
  type FetchLike,
  type PredictionOutput,
  type ServedMeta,
  ServedPredictor,
  loadServed,
  npArgmax,
  oodBand,
  parsePythonJson,
  softmax,
} from "@/engine/served";
import { PersonSmoother, type PersonSmootherOptions } from "@/engine/smoother";
import { RollingWindow } from "@/engine/window";
import { artifactBytes, artifactJson, artifactText, loadFixture, nn } from "./engine-fixtures";

// ---------- Fixtures ----------

type Num = number | null;

interface ModelCase {
  name: string;
  tod_h: Num;
  x: Num[];
  xc: Num[];
  m2_raw: number[];
  m2_contrib: number[][];
  p: number[];
  ood: string[];
  k: number;
  m0: null | { raw: number[]; p: number[]; k0: number; conf: number; face_push: number };
  raw_override: boolean | null;
  label: string;
  confidence: number;
  layers: Layer[];
}
interface ModelSet {
  state_set: "core" | "all";
  model_id: string;
  states: string[];
  artifacts: string;
  temperatures: { m2: number; m0: number };
  m2_features: string[];
  m0_features: string[];
  cases: ModelCase[];
}
interface ModelsFixture {
  version: number;
  sets: ModelSet[];
}

interface SmootherStep {
  probabilities: Record<string, Num>;
  raw_override: boolean | null;
  label: string;
  window_ticks: number;
}
interface SmootherExpected {
  label: string;
  mask_override: boolean;
  warming_up: boolean;
  ema: Num[];
  candidate: string | null;
  candidate_n: number;
  override_hist: boolean[];
  override_on: boolean;
}
interface SmootherCase {
  name: string;
  states: string[];
  kwargs: PersonSmootherOptions;
  steps: SmootherStep[];
  expected: SmootherExpected[];
}
interface CounterfactualCase {
  run: string;
  tick: number;
  person_id: number;
  feature: string;
  value: number;
  out: unknown;
}
type Bound = number | "Infinity" | "-Infinity";

/** Fixture bound -> number (the generator writes +-inf as "Infinity" / "-Infinity"). */
function bound(v: Bound): number {
  return v === "Infinity" ? Infinity : v === "-Infinity" ? -Infinity : v;
}
function bounds(a: readonly Bound[]): number[] {
  return a.map(bound);
}
interface ServedTables {
  state_set: "core" | "all";
  lo: Bound[];
  hi: Bound[];
  ood_lo: Bound[];
  ood_hi: Bound[];
  ood_used: boolean[];
  face_cols: boolean[];
  face_now_index: number;
  clip_cases: { name: string; x: Bound[]; xc: Bound[]; ood: string[] }[];
}
interface SmootherFixture {
  version: number;
  smoother: SmootherCase[];
  smoother_errors: { name: string; kwargs: Record<string, number | string>; error: string }[];
  counterfactual: CounterfactualCase[];
  served: ServedTables[];
  ood_band: { name: string; lo: Bound; hi: Bound; out: [Bound, Bound] }[];
  synthetic: {
    model_id: string;
    m2_text: string;
    cases: {
      name: string;
      tod_h: Num;
      x: Num[];
      out: Record<string, unknown> & { probabilities: Record<string, number>; confidence: number };
    }[];
  };
  generator: string;
}

interface PredictionMessage {
  type: "prediction";
  person_id: number;
  label: string;
  label_display: string;
  confidence: number;
  probabilities: Record<string, number>;
  face_only: { label: string | null; confidence: number | null };
  mask_override: boolean;
  warming_up: boolean;
  layers: Layer[];
}
interface RuntimeRun {
  name: string;
  state_set: "core" | "all";
  room_only: boolean;
  model_id: string;
  settings: {
    tick_s: number;
    warmup_s: number;
    override_block_s: number;
    window_ticks: number;
    person_drop_min: number;
  };
  ticks: {
    wall: number;
    mono: number;
    records: Record<string, unknown>[];
    messages: { type: string }[];
  }[];
}

const models = loadFixture<ModelsFixture>("models");
const sfx = loadFixture<SmootherFixture>("smoother");
const runtime = loadFixture<{ version: number; runs: RuntimeRun[] }>("runtime");

const ARTIFACT_DIR = { core: "", all: "all/" } as const;

function artifactsOf(stateSet: "core" | "all") {
  const d = ARTIFACT_DIR[stateSet];
  return {
    m2: artifactBytes(`${d}m2.txt`),
    m0: artifactText(`${d}m0.txt`),
    meta: artifactJson<ServedMeta>(`${d}model_meta.json`),
    clipBounds: artifactJson<ClipBounds>(`${d}clip_bounds.json`),
  };
}

const SERVED = {
  core: new ServedPredictor(artifactsOf("core")),
  all: new ServedPredictor(artifactsOf("all")),
};

// ---------- Strict comparison ----------

/** NaN -> null (the fixtures' encoding), recursively, so TS output can be diffed against fixture JSON. */
function asJson(v: unknown): unknown {
  if (typeof v === "number") return Number.isNaN(v) ? null : v;
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return Array.from(v as ArrayLike<unknown>, asJson);
  if (v !== null && typeof v === "object")
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, asJson(x)]));
  return v;
}

/** Deep equality, Object.is on numbers (-0.0 != 0.0), key order included. Returns the differing paths. */
function diff(got: unknown, want: unknown, path = "$", out: string[] = []): string[] {
  if (typeof got === "number" && typeof want === "number") {
    if (!Object.is(got, want)) out.push(`${path}: ${got} != ${want}`);
    return out;
  }
  if (Array.isArray(got) && Array.isArray(want)) {
    if (got.length !== want.length) out.push(`${path}: length ${got.length} != ${want.length}`);
    for (let i = 0; i < Math.min(got.length, want.length); i++)
      diff(got[i], want[i], `${path}[${i}]`, out);
    return out;
  }
  if (got !== null && want !== null && typeof got === "object" && typeof want === "object") {
    const gk = Object.keys(got);
    const wk = Object.keys(want);
    if (gk.join("\u0000") !== wk.join("\u0000")) out.push(`${path}: keys [${gk}] != [${wk}]`);
    for (const k of wk) {
      if (k in got)
        diff(
          (got as Record<string, unknown>)[k],
          (want as Record<string, unknown>)[k],
          `${path}.${k}`,
          out,
        );
    }
    return out;
  }
  if (got !== want) out.push(`${path}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
  return out;
}

/** Python round(v, 4) of a value that may be an ulp off Python's: true when it matches, or straddles a tie. */
function round4Matches(got: number, want: number, stats: { tieFlips: number }): boolean {
  if (Object.is(pyRound(got, 4), want)) return true;
  // a 1-ulp softmax difference can only flip round(., 4) when the exact value sits on a rounding boundary
  const boundary = want + Math.sign(got - want) * 0.00005;
  if (Math.abs(got - boundary) < 1e-12) {
    stats.tieFlips++;
    return true;
  }
  return false;
}

// ---------- Error budget ----------

const report = {
  cases: 0,
  pMaxAbs: 0,
  pMaxUlp: 0,
  pNotExact: 0,
  pValues: 0,
  rawMaxAbs: 0,
  contribMaxAbs: 0,
  ticks: 0,
  predictions: 0,
  tieFlips: 0,
  predictMs: 0,
  cfUlp: 0,
};

function trackP(got: number, want: number): number {
  const e = Math.abs(got - want);
  report.pValues++;
  if (!Object.is(got, want)) report.pNotExact++;
  report.pMaxAbs = Math.max(report.pMaxAbs, e);
  report.pMaxUlp = Math.max(report.pMaxUlp, ulpDistance(got, want));
  return e;
}

afterAll(() => {
  console.info(
    `[engine-served] models: ${report.cases} cases; raw max abs ${report.rawMaxAbs}, contrib max abs ` +
      `${report.contribMaxAbs}; p max abs ${report.pMaxAbs.toExponential(2)} (${report.pMaxUlp} ulp), ` +
      `${report.pNotExact}/${report.pValues} p values not bit-exact. runtime replay: ${report.ticks} ticks, ` +
      `${report.predictions} predictions, ${report.tieFlips} round(., 4) tie flips; ` +
      `${(report.predictMs / Math.max(1, report.predictions)).toFixed(2)} ms per predict ` +
      `(features + M2 + SHAP + M0 + layers); counterfactual original_value max ${report.cfUlp} ulp`,
  );
});

// ---------- models.json.gz ----------

describe("served predictor on the golden model cases", () => {
  it("loads the same model ids, states, temperatures and feature sets as Python", () => {
    expect(models.version).toBe(1);
    expect(models.sets.map((s) => s.state_set)).toEqual(["core", "all"]);
    for (const set of models.sets) {
      const sp = SERVED[set.state_set];
      expect(sp.modelId).toBe(set.model_id);
      expect([...sp.states]).toEqual(set.states);
      expect(sp.t2).toBe(set.temperatures.m2);
      expect(sp.t0).toBe(set.temperatures.m0);
      expect([...sp.cols2]).toEqual(set.m2_features);
      expect([...sp.cols0]).toEqual(set.m0_features);
    }
  });

  for (const set of models.sets) {
    it(`reproduces every ${set.state_set} case: clip, OOD, raw, SHAP, p, M0 shadow, override, layers`, () => {
      const sp = SERVED[set.state_set];
      const K = set.states.length;
      const bad: string[] = [];
      for (const c of set.cases) {
        report.cases++;
        const x = nn(c.x);
        const tr = sp.trace(x, c.tod_h);
        const out = tr.output;
        const at = (what: string, d: string[]) =>
          d.forEach((s) => bad.push(`${c.name} ${what} ${s}`));

        at("xc", diff(Array.from(tr.xc), nn(c.xc)));
        at("ood", diff(tr.ood, c.ood));
        at("raw", diff(Array.from(tr.raw), c.m2_raw));
        tr.raw.forEach(
          (v, i) =>
            (report.rawMaxAbs = Math.max(report.rawMaxAbs, Math.abs(v - (c.m2_raw[i] as number)))),
        );
        const contrib = sp.m2Contrib(tr.xc);
        expect(contrib.length).toBe(K);
        contrib.forEach((row, k) => {
          at(`contrib[${k}]`, diff(Array.from(row), c.m2_contrib[k]));
          row.forEach((v, j) => {
            report.contribMaxAbs = Math.max(
              report.contribMaxAbs,
              Math.abs(v - (c.m2_contrib[k]?.[j] as number)),
            );
          });
        });
        at("contribK", diff(Array.from(tr.contribK), c.m2_contrib[c.k]));
        tr.p.forEach((v, i) => {
          if (trackP(v, c.p[i] as number) > 1e-12) bad.push(`${c.name} p[${i}] ${v} != ${c.p[i]}`);
        });
        if (tr.k !== c.k) bad.push(`${c.name} k ${tr.k} != ${c.k}`);
        if (out.label !== c.label) bad.push(`${c.name} label ${out.label} != ${c.label}`);
        if (trackP(out.confidence, c.confidence) > 1e-12) bad.push(`${c.name} confidence`);
        if (c.m0 === null) {
          if (tr.m0 !== null) bad.push(`${c.name} m0 should be null`);
          if (out.face_only_label !== null || out.face_only_confidence !== null)
            bad.push(`${c.name} face_only`);
        } else if (tr.m0 === null) {
          bad.push(`${c.name} m0 missing`);
        } else {
          at("m0.raw", diff(Array.from(tr.m0.raw), c.m0.raw));
          tr.m0.p.forEach((v, i) => {
            if (trackP(v, c.m0?.p[i] as number) > 1e-12) bad.push(`${c.name} m0.p[${i}]`);
          });
          if (tr.m0.k0 !== c.m0.k0) bad.push(`${c.name} k0 ${tr.m0.k0} != ${c.m0.k0}`);
          if (trackP(tr.m0.conf, c.m0.conf) > 1e-12) bad.push(`${c.name} m0.conf`);
          at("face_push", diff(tr.m0.facePush, c.m0.face_push));
          if (out.face_only_label !== set.states[c.m0.k0]) bad.push(`${c.name} face_only_label`);
          if (out.face_only_confidence !== tr.m0.conf) bad.push(`${c.name} face_only_confidence`);
        }
        if (out.raw_override !== c.raw_override)
          bad.push(`${c.name} raw_override ${out.raw_override} != ${c.raw_override}`);
        at("layers", diff(asJson(out.layers), c.layers));
        // the rest of PredictionOutput: probabilities in state order, unclipped M2 values, the OOD list
        expect(Object.keys(out.probabilities)).toEqual(set.states);
        set.states.forEach((s, i) => expect(out.probabilities[s]).toBe(tr.p[i]));
        expect(Object.keys(out.values)).toEqual(set.m2_features);
        at(
          "values",
          diff(
            Object.values(out.values),
            set.m2_features.map((n) => x[FEATURE_NAMES.indexOf(n)]),
          ),
        );
        expect(out.ood_features).toBe(tr.ood);
        expect(sp.predictVector(x, c.tod_h)).toEqual(out);
      }
      expect(bad.slice(0, 20)).toEqual([]);
    }, 120_000);
  }

  it("covers the override, no-face and OOD-shrink branches", () => {
    const cases = models.sets.flatMap((s) => s.cases);
    expect(cases.filter((c) => c.raw_override === true).length).toBeGreaterThan(10);
    expect(cases.filter((c) => c.m0 === null).length).toBeGreaterThan(10);
    expect(cases.filter((c) => c.ood.length > 3).length).toBeGreaterThan(5);
    expect(cases.filter((c) => c.ood.length > 0 && c.ood.length <= 3).length).toBeGreaterThan(5);
  });
});

// ---------- synthetic M2 ----------

describe("served predictor with a two-feature M2", () => {
  // Every other SHAP value is exactly 0.0 and the face push is exactly 0.0: this pins served.py's
  // `if c != 0.0` driver filter and the strict `face_push > 0` override gate, which the real models never
  // reach visibly.
  const syn = sfx.synthetic;
  const core = artifactsOf("core");
  const sp = new ServedPredictor({ ...core, m2: syn.m2_text });

  it("hashes the model text like Python", () => {
    expect(sp.modelId).toBe(syn.model_id);
  });

  it("reproduces every PredictionOutput, layers and override included", () => {
    const bad: string[] = [];
    for (const c of syn.cases) {
      const out = sp.predictVector(nn(c.x), c.tod_h);
      const { probabilities: wp, confidence: wc, ...want } = c.out;
      const { probabilities: gp, confidence: gc, ...got } = out;
      diff(asJson(got), want).forEach((d) => bad.push(`${c.name} ${d}`));
      if (trackP(gc, wc) > 1e-12) bad.push(`${c.name} confidence ${gc} != ${wc}`);
      expect(Object.keys(gp)).toEqual(Object.keys(wp));
      for (const [k, v] of Object.entries(gp))
        if (trackP(v, wp[k] as number) > 1e-12) bad.push(`${c.name} p[${k}]`);
    }
    expect(bad).toEqual([]);
    // the cases include a positive face over a confident non-positive M2 reading with a zero face push
    const gated = syn.cases.filter(
      (c) =>
        c.out["face_only_label"] === "Collaborative_Happy" &&
        c.out["label"] !== "Collaborative_Happy" &&
        (c.out["face_only_confidence"] as number) >= 0.45 &&
        c.out.confidence >= 0.7,
    );
    expect(gated.length).toBeGreaterThan(0);
    expect(gated.every((c) => c.out["raw_override"] === false)).toBe(true);
  });
});

// ---------- tables, clip and OOD bands ----------

describe("clip bounds and OOD bands", () => {
  for (const t of sfx.served) {
    it(`match ServedPredictor's tables for ${t.state_set}`, () => {
      const sp = SERVED[t.state_set];
      expect(diff(Array.from(sp.lo), bounds(t.lo))).toEqual([]);
      expect(diff(Array.from(sp.hi), bounds(t.hi))).toEqual([]);
      expect(diff(Array.from(sp.oodLo), bounds(t.ood_lo))).toEqual([]);
      expect(diff(Array.from(sp.oodHi), bounds(t.ood_hi))).toEqual([]);
      expect(Array.from(sp.oodUsed, Boolean)).toEqual(t.ood_used);
      expect(Array.from(sp.faceCols, Boolean)).toEqual(t.face_cols);
      expect(FEATURE_NAMES.indexOf("face_now")).toBe(t.face_now_index);
    });

    it(`clip crafted vectors exactly like _clip for ${t.state_set} (the bound wins a signed-zero tie)`, () => {
      const sp = SERVED[t.state_set];
      for (const c of t.clip_cases) {
        const { xc, ood } = sp.clip(bounds(c.x));
        expect(diff(Array.from(xc), bounds(c.xc)), c.name).toEqual([]);
        expect(ood, c.name).toEqual(c.ood);
      }
      // slump_now has lo = 0.0: numpy's array clip turns -0.0 into +0.0 there (its scalar path would not)
      const neg = sp.clip(new Array<number>(FEATURE_NAMES.length).fill(-0)).xc;
      expect(Object.is(neg[FEATURE_NAMES.indexOf("slump_now")], 0)).toBe(true);
    });
  }

  it("ood_band agrees with Python on odd names (trailing newline, Unicode \\w, overflow)", () => {
    const bad: string[] = [];
    for (const b of sfx.ood_band) {
      const got = oodBand(b.name, bound(b.lo), bound(b.hi));
      const d = diff(got, bounds(b.out));
      if (d.length) bad.push(`${JSON.stringify(b.name)} ${b.lo}..${b.hi}: ${d.join("; ")}`);
    }
    expect(sfx.ood_band.length).toBeGreaterThan(100);
    expect(bad).toEqual([]);
  });

  it("rejects vectors of the wrong length and reads null as missing", () => {
    const sp = SERVED.core;
    expect(() => sp.predictVector(new Float64Array(122))).toThrow(/123/);
    const c = models.sets[0]?.cases[0] as ModelCase;
    const withNull = c.x.slice();
    const viaNaN = sp.predictVector(nn(withNull), c.tod_h);
    expect(sp.predictVector(withNull, c.tod_h)).toEqual(viaNaN);
  });
});

// ---------- softmax / argmax ----------

describe("softmax and argmax", () => {
  it("softmax divides by the temperature, shifts by the max and sums like numpy", () => {
    const p = softmax([1, 2, 3, 4], 0.5);
    const e = [Math.exp(-6), Math.exp(-4), Math.exp(-2), Math.exp(0)];
    const s = 0 + (e[0] as number) + (e[1] as number) + (e[2] as number) + (e[3] as number);
    expect(Array.from(p)).toEqual(e.map((v) => v / s));
    expect(Array.from(softmax([1, NaN, 0], 1)).every(Number.isNaN)).toBe(true);
  });

  it("argmax returns the first maximum, or the first NaN, as np.argmax", () => {
    expect(npArgmax([0.25, 0.5, 0.5, 0.1])).toBe(1);
    expect(npArgmax([0.9, NaN, 1, NaN])).toBe(1);
    expect(npArgmax([-Infinity, -Infinity])).toBe(0);
    expect(() => npArgmax([])).toThrow();
  });
});

// ---------- PersonSmoother ----------

describe("person smoother", () => {
  it("covers the crafted and random sequences of the fixture", () => {
    expect(sfx.version).toBe(1);
    expect(sfx.smoother.length).toBeGreaterThanOrEqual(20);
    const all = sfx.smoother.flatMap((c) => c.expected);
    expect(all.filter((e) => e.mask_override).length).toBeGreaterThan(50);
    expect(all.filter((e) => e.warming_up).length).toBeGreaterThan(50);
    expect(all.filter((e) => e.candidate_n > 0).length).toBeGreaterThan(50);
  });

  for (const c of sfx.smoother) {
    it(`matches Python step by step, internal state included: ${c.name}`, () => {
      const sm = new PersonSmoother(c.states, c.kwargs);
      const bad: string[] = [];
      c.steps.forEach((st, i) => {
        const probs: Record<string, number> = {};
        for (const [k, v] of Object.entries(st.probabilities)) probs[k] = v === null ? NaN : v;
        const got = sm.update(
          { probabilities: probs, raw_override: st.raw_override, label: st.label },
          st.window_ticks,
        );
        const e = c.expected[i] as SmootherExpected;
        const state = {
          label: got.label,
          mask_override: got.mask_override,
          warming_up: got.warming_up,
          ema: asJson(sm.ema),
          candidate: sm.candidate,
          candidate_n: sm.candidateN,
          override_hist: [...sm.overrideHist],
          override_on: sm.overrideOn,
        };
        diff(state, e).forEach((d) => bad.push(`step ${i} ${d}`));
      });
      expect(bad).toEqual([]);
    });
  }

  it("accepts a Map of probabilities and ignores inherited keys", () => {
    const states = ["constructor", "Focused"];
    const a = new PersonSmoother(states);
    const b = new PersonSmoother(states);
    const r1 = a.update(
      { probabilities: { Focused: 0.2 }, raw_override: null, label: "Focused" },
      1,
    );
    const r2 = b.update(
      { probabilities: new Map([["Focused", 0.2]]), raw_override: null, label: "Focused" },
      1,
    );
    expect(r1.label).toBe("Focused"); // "constructor" reads 0.0, not Object.prototype.constructor
    expect(r2).toEqual(r1);
  });

  it("raises where Python's constructor raises", () => {
    for (const e of sfx.smoother_errors) {
      const kw: Record<string, number> = {};
      for (const [k, v] of Object.entries(e.kwargs)) kw[k] = typeof v === "string" ? Number(v) : v;
      expect(e.error, e.name).not.toBeNull();
      expect(() => new PersonSmoother(["Focused"], kw as PersonSmootherOptions), e.name).toThrow();
    }
  });

  it("builds from Settings like engine.py", () => {
    const sm = PersonSmoother.fromSettings(["A"], {
      tick_s: 2.5,
      warmup_s: 61,
      override_block_s: 120,
    });
    expect([sm.tickS, sm.warmupTicks, sm.overrideBlockTicks]).toEqual([2.5, 24, 48]);
  });
});

// ---------- runtime.json.gz replay + counterfactual ----------

/** valence/runtime/engine.py:346 Runtime._evict (window slot and smoother). */
function evict(
  window: RollingWindow,
  smoothers: Map<number, PersonSmoother>,
  wall: number,
  dropMin: number,
): void {
  for (const pid of window.people()) {
    const rows = window.snapshot(pid);
    const last = rows[rows.length - 1];
    if (last !== undefined && wall - last.ts > dropMin * 60) {
      window.drop(pid);
      smoothers.delete(pid);
    }
  }
}

/** Features computed through Math.sin/cos/log2, which may differ from CPython's glibc by an ulp (features.ts). */
const V8_SENSITIVE = new Set([
  "tod_sin",
  "tod_cos",
  "lux_room_now",
  "lux_room_mean_5m",
  "lux_room_mean_15m",
]);

/** The FactSheet face_only rule (valence/explain/explainer.py:98) the prediction message carries. */
function faceOnly(
  out: PredictionOutput,
  last: TickRecord,
): { label: string | null; confidence: number | null } {
  const visible = last.face_visible > 0 && !Number.isNaN(last.face_valence);
  return {
    label: visible ? out.face_only_label : null,
    confidence: visible && out.face_only_confidence !== null ? out.face_only_confidence : null,
  };
}

describe("runtime replay through served predictor and smoother", () => {
  const cfByKey = new Map<string, CounterfactualCase>();
  for (const c of sfx.counterfactual) cfByKey.set(`${c.run}/${c.tick}/${c.person_id}`, c);

  for (const run of runtime.runs) {
    it(`reproduces every prediction message of ${run.name} (label, override, warm-up, layers, rounding)`, () => {
      const sp = SERVED[run.state_set];
      expect(sp.modelId).toBe(run.model_id);
      const s = run.settings;
      const window = new RollingWindow(s.window_ticks);
      const smoothers = new Map<number, PersonSmoother>();
      const bad: string[] = [];
      let nPred = 0;
      let nCf = 0;
      let ms = 0;
      run.ticks.forEach((tick, i) => {
        const recs = tick.records.map(tickRecordFromDict);
        for (const r of recs) window.append(r);
        evict(window, smoothers, tick.wall, s.person_drop_min);
        const targets = run.room_only
          ? [ROOM]
          : [...new Set(recs.map((r) => r.person_id).filter((p) => p !== ROOM))].sort(
              (a, b) => a - b,
            );
        const msgs = tick.messages.filter((m): m is PredictionMessage => m.type === "prediction");
        const got: { pid: number; msg: Partial<PredictionMessage> }[] = [];
        for (const pid of targets) {
          const person = window.snapshot(pid);
          const room = window.room();
          const t0 = performance.now();
          const out = sp.predict(person, room);
          ms += performance.now() - t0;
          if (out === null) continue;
          let sm = smoothers.get(pid);
          if (sm === undefined)
            smoothers.set(pid, (sm = PersonSmoother.fromSettings(sp.states, s)));
          const smooth = sm.update(out, person.length);
          const last = person[person.length - 1] as TickRecord;
          const conf = Object.hasOwn(out.probabilities, smooth.label)
            ? (out.probabilities[smooth.label] as number)
            : out.confidence;
          got.push({
            pid,
            msg: {
              label: smooth.label,
              mask_override: smooth.mask_override,
              warming_up: smooth.warming_up,
            },
          });
          const want = msgs[got.length - 1];
          const where = `tick ${i} p${pid}`;
          if (want === undefined || want.person_id !== pid) {
            bad.push(`${where}: no matching message`);
            continue;
          }
          nPred++;
          if (smooth.label !== want.label)
            bad.push(`${where} label ${smooth.label} != ${want.label}`);
          if (smooth.mask_override !== want.mask_override) bad.push(`${where} mask_override`);
          if (smooth.warming_up !== want.warming_up) bad.push(`${where} warming_up`);
          const shown = displayLabel(smooth.label, {
            maskOverride: smooth.mask_override,
            nPeople: Math.max(1, last.n_people),
            states: sp.states,
          });
          if (shown !== want.label_display)
            bad.push(`${where} label_display ${shown} != ${want.label_display}`);
          if (!round4Matches(conf, want.confidence, report))
            bad.push(`${where} confidence ${conf} vs ${want.confidence}`);
          expect(Object.keys(want.probabilities)).toEqual(Object.keys(out.probabilities));
          for (const [st, v] of Object.entries(out.probabilities)) {
            if (!round4Matches(v, want.probabilities[st] as number, report))
              bad.push(`${where} p[${st}]`);
          }
          const fo = faceOnly(out, last);
          if (fo.label !== want.face_only.label) bad.push(`${where} face_only.label`);
          if ((fo.confidence === null) !== (want.face_only.confidence === null))
            bad.push(`${where} face_only.confidence null`);
          else if (
            fo.confidence !== null &&
            !round4Matches(fo.confidence, want.face_only.confidence as number, report)
          )
            bad.push(`${where} face_only.confidence`);
          diff(asJson(out.layers), want.layers).forEach((d) => bad.push(`${where} layers ${d}`));

          const cf = cfByKey.get(`${run.name}/${i}/${pid}`);
          if (cf !== undefined) {
            nCf++;
            const res = sp.counterfactual(person, room, cf.feature, cf.value);
            const want = cf.out as { original_value: number | null };
            // original_value is the raw feature: tod_sin/tod_cos/lux_room_* may be 1 ulp off (V8 vs glibc
            // Math, see features.ts); everything else, rounded confidences included, is exact
            if (res.original_value !== null && want.original_value !== null) {
              const u = ulpDistance(res.original_value, want.original_value);
              report.cfUlp = Math.max(report.cfUlp, u);
              if (u > (V8_SENSITIVE.has(cf.feature) ? 2 : 0))
                bad.push(`${where} counterfactual original_value`);
            }
            diff(asJson({ ...res, original_value: null }), {
              ...want,
              original_value: null,
            }).forEach((d) => bad.push(`${where} counterfactual ${d}`));
            if ((res.original_value === null) !== (want.original_value === null))
              bad.push(`${where} counterfactual original_value null`);
          }
        }
        if (got.length !== msgs.length)
          bad.push(`tick ${i}: ${got.length} predictions vs ${msgs.length} messages`);
      });
      report.ticks += run.ticks.length;
      report.predictions += nPred;
      report.predictMs += ms;
      const nMsgs = run.ticks.reduce(
        (n, t) => n + t.messages.filter((m) => m.type === "prediction").length,
        0,
      );
      expect(nPred).toBe(nMsgs);
      expect(nCf).toBe(sfx.counterfactual.filter((c) => c.run === run.name).length);
      expect(bad.slice(0, 20)).toEqual([]);
    }, 180_000);
  }

  it("covers the mask override and the warm-up in the replayed runs", () => {
    const preds = runtime.runs.flatMap((r) =>
      r.ticks.flatMap((t) =>
        t.messages.filter((m): m is PredictionMessage => m.type === "prediction"),
      ),
    );
    expect(preds.filter((p) => p.mask_override).length).toBeGreaterThan(50);
    expect(preds.filter((p) => p.warming_up).length).toBeGreaterThan(20);
    expect(sfx.counterfactual.length).toBeGreaterThan(30);
    expect(
      sfx.counterfactual.some((c) => (c.out as { label_changed: boolean }).label_changed),
    ).toBe(true);
  });

  it("counterfactual rejects unknown features (Python: KeyError)", () => {
    const run = runtime.runs[0] as RuntimeRun;
    const person = (run.ticks[0]?.records ?? [])
      .map(tickRecordFromDict)
      .filter((r) => r.person_id !== ROOM);
    expect(() => SERVED.core.counterfactual(person, [], "no_such_feature", 1)).toThrow(
      /no_such_feature/,
    );
  });

  it("predict returns null for an empty person window", () => {
    expect(SERVED.core.predict([], [])).toBeNull();
  });
});

// ---------- malformed model files ----------

describe("served predictor on malformed model files", () => {
  it("parses JSON like Python's json.loads, NaN / Infinity / -Infinity included", () => {
    // json.loads on each input (Python 3.12): value, or null where it raises JSONDecodeError
    const cases: [string, unknown][] = [
      ['{"a": [NaN, Infinity, -Infinity]}', { a: [NaN, Infinity, -Infinity] }],
      ['{"k":-Infinity }', { k: -Infinity }],
      ['["NaN", "x\\"NaN", "Infinity"]', ["NaN", 'x"NaN', "Infinity"]],
      ["[-NaN]", null],
      ["[+Infinity]", null],
      ["[nan]", null],
      ["[Infinityx]", null],
      ["[NaN1]", null],
      ["{NaN: 1}", null],
      ["\ufeff{}", null],
    ];
    for (const [text, want] of cases) {
      if (want === null) expect(() => parsePythonJson(text), text).toThrow();
      else expect(parsePythonJson(text), text).toEqual(want);
    }
  });

  it("raises on clip bounds Python cannot use instead of clipping to NaN", () => {
    // Python (load_served + predict_vector on a copy of artifacts/ with one bound changed):
    // face_now [null, 1] -> TypeError at the first predict; db_room_now [null, hi] -> TypeError in __init__;
    // db_room_now 5.0 -> TypeError (not subscriptable); [1.0] -> IndexError; "ab" -> TypeError
    const core = artifactsOf("core");
    const bounds = core.clipBounds as Record<string, unknown>;
    const variants: [string, unknown][] = [
      ["face_now", [null, 1.0]],
      ["db_room_now", [null, 70.0]],
      ["slump_now", [null, 1.0]],
      ["db_room_now", 5.0],
      ["db_room_now", [1.0]],
      ["db_room_now", "ab"],
      ["db_room_now", { 0: 1, 1: 2 }],
    ];
    for (const [feature, b] of variants) {
      const clipBounds = { ...bounds, [feature]: b } as unknown as ClipBounds;
      expect(
        () => new ServedPredictor({ ...core, clipBounds }),
        `${feature} ${JSON.stringify(b)}`,
      ).toThrow();
    }
    expect(
      () => new ServedPredictor({ ...core, clipBounds: [] as unknown as ClipBounds }),
    ).toThrow();
    // a bool is a number to numpy (True -> 1.0), and extra entries past [0], [1] are ignored
    const ok = new ServedPredictor({
      ...core,
      clipBounds: { ...bounds, db_room_now: [false, true, "x"] } as unknown as ClipBounds,
    });
    const i = FEATURE_NAMES.indexOf("db_room_now");
    expect([ok.lo[i], ok.hi[i]]).toEqual([0, 1]);
  });

  it("reads temperatures with float() semantics: a missing or null one raises", () => {
    // Python float(meta["temperatures"]["m2"]) for each stored value; null = raises (KeyError / TypeError)
    const core = artifactsOf("core");
    const cases: [unknown, number | null][] = [
      ["1.25", 1.25],
      [" 2 ", 2.0],
      [true, 1.0],
      ["1_0", 10.0],
      ["inf", Infinity],
      ["-Infinity", -Infinity],
      [3, 3],
      [null, null],
      [undefined, null],
      ["", null],
      ["1__0", null],
      ["0x10", null],
      [[1.5], null],
    ];
    for (const [t, want] of cases) {
      const temperatures: Record<string, unknown> = { ...core.meta.temperatures };
      if (t === undefined) delete temperatures["m2"];
      else temperatures["m2"] = t;
      const meta = { ...core.meta, temperatures } as unknown as ServedMeta;
      const build = () => new ServedPredictor({ ...core, meta });
      if (want === null) expect(build, JSON.stringify(t)).toThrow();
      else expect(build().t2, JSON.stringify(t)).toBe(want);
    }
    const noTemps = { ...core.meta, temperatures: undefined } as unknown as ServedMeta;
    expect(() => new ServedPredictor({ ...core, meta: noTemps })).toThrow(/temperatures/);
  });
});

// ---------- loadServed ----------

describe("load served", () => {
  /** A fetch stub over artifacts/: /models/core/x -> artifacts/x, /models/all/x -> artifacts/all/x. */
  function stub(overrides: Record<string, { status: number; body?: string }> = {}): FetchLike {
    return async (url: string) => {
      const o = overrides[url];
      if (o !== undefined) {
        const bytes = new TextEncoder().encode(o.body ?? "");
        return {
          ok: o.status < 400,
          status: o.status,
          arrayBuffer: async () => bytes.slice().buffer,
        };
      }
      const m = /^\/models\/(core|all)\/(.+)$/.exec(url);
      if (!m) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
      const rel = `${m[1] === "all" ? "all/" : ""}${m[2]}`;
      try {
        const bytes = artifactBytes(rel);
        return { ok: true, status: 200, arrayBuffer: async () => bytes.slice().buffer };
      } catch {
        return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
      }
    };
  }

  it("fetches the four files and predicts exactly like the directly built predictor", async () => {
    for (const set of models.sets) {
      const sp = await loadServed(`/models/${set.state_set}/`, stub(), set.state_set);
      expect(sp).not.toBeNull();
      expect(sp?.modelId).toBe(set.model_id);
      const c = set.cases[0] as ModelCase;
      expect(sp?.predictVector(nn(c.x), c.tod_h)).toEqual(
        SERVED[set.state_set].predictVector(nn(c.x), c.tod_h),
      );
    }
    expect((await loadServed("/models/all", stub()))?.modelId).toBe(models.sets[1]?.model_id);
  });

  it("resolves null like load_served when a model file is missing or the state set differs", async () => {
    expect(
      await loadServed("/models/core/", stub({ "/models/core/m0.txt": { status: 404 } })),
    ).toBeNull();
    expect(
      await loadServed("/models/core/", stub({ "/models/core/model_meta.json": { status: 410 } })),
    ).toBeNull();
    const spa = { status: 200, body: "<!doctype html><html></html>" };
    expect(await loadServed("/models/core/", stub({ "/models/core/m2.txt": spa }))).toBeNull();
    expect(await loadServed("/models/core/", stub(), "all")).toBeNull();
  });

  it("reads Python's NaN / Infinity JSON tokens in clip_bounds.json like json.loads", async () => {
    // Python: clip_bounds.json written by json.dumps with db_room_now = [-inf, inf] and laugh_cnt_15m =
    // [0.0, inf]; load_served + predict_vector on models case 0 with both features set to 1e6
    // -> Focused 0.7441824396965814, neither feature clipped nor OOD (the unmodified files flag db_room_now
    // and give 0.7430864246904249).
    const clip = artifactJson<Record<string, unknown>>("clip_bounds.json");
    clip["db_room_now"] = ["@NINF@", "@INF@"];
    clip["laugh_cnt_15m"] = [0.0, "@INF@"];
    const body = JSON.stringify(clip, null, 1)
      .replaceAll('"@NINF@"', "-Infinity")
      .replaceAll('"@INF@"', "Infinity");
    expect(() => JSON.parse(body)).toThrow(); // what a plain JSON.parse loader would do
    const sp = await loadServed(
      "/models/core/",
      stub({ "/models/core/clip_bounds.json": { status: 200, body } }),
    );
    if (sp === null) throw new Error("loadServed returned null");
    const c = models.sets[0]?.cases[0] as ModelCase;
    expect(c.name).toBe("focused/t003/p1");
    const x = nn(c.x);
    const db = FEATURE_NAMES.indexOf("db_room_now");
    const laugh = FEATURE_NAMES.indexOf("laugh_cnt_15m");
    x[db] = 1e6;
    x[laugh] = 1e6;
    expect([sp.lo[db], sp.hi[db], sp.oodLo[db], sp.oodHi[db]]).toEqual([
      -Infinity,
      Infinity,
      -Infinity,
      Infinity,
    ]);
    const tr = sp.trace(x, c.tod_h);
    expect([tr.xc[db], tr.xc[laugh]]).toEqual([1e6, 1e6]);
    expect(tr.ood).toEqual(["db_slope_5m", "db_slope_15m", "lux_slope_5m", "lux_slope_15m"]);
    expect(tr.output.label).toBe("Focused");
    expect(Math.abs(tr.output.confidence - 0.7441824396965814)).toBeLessThan(1e-12);
    expect(tr.output.raw_override).toBe(false);
    const plain = SERVED.core.predictVector(x, c.tod_h);
    expect(plain.ood_features).toContain("db_room_now");
    expect(Math.abs(plain.confidence - 0.7430864246904249)).toBeLessThan(1e-12);
  });

  it("rejects a model_meta.json with a BOM or bad UTF-8, as Path.read_text + json.loads do", async () => {
    const meta = artifactText("model_meta.json");
    await expect(
      loadServed(
        "/models/core/",
        stub({ "/models/core/model_meta.json": { status: 200, body: `\ufeff${meta}` } }),
      ),
    ).rejects.toThrow();
    const bad = (url: string) =>
      url.endsWith("clip_bounds.json")
        ? Promise.resolve({
            ok: true,
            status: 200,
            arrayBuffer: async () => new Uint8Array([0x7b, 0xff, 0x7d]).buffer,
          })
        : stub()(url);
    await expect(loadServed("/models/core/", bad)).rejects.toThrow();
  });

  it("fails loudly on a missing clip_bounds.json or a server error", async () => {
    await expect(
      loadServed("/models/core/", stub({ "/models/core/clip_bounds.json": { status: 404 } })),
    ).rejects.toThrow(/clip_bounds/);
    await expect(
      loadServed("/models/core/", stub({ "/models/core/m2.txt": { status: 500 } })),
    ).rejects.toThrow(/500/);
  });
});
