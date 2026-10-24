/**
 * describe.ts, explainer.ts and template.ts against the Python explanation code (valence/explain/describe.py,
 * explainer.py, template.py, narrator_client.py band) via explain.json.gz (scripts/export_web_golden.py).
 *
 * Everything here is held to exact equality: display strings and template texts character for character,
 * numbers with Object.is (so -0.0 vs 0.0 counts), after the same Python rounding. No value in this module
 * passes through exp/log/pow/sin/cos, so there is no ulp budget.
 */
import { describe, expect, it } from "vitest";
import type { Driver, FactSheet, Layer } from "@/engine/contracts";
import { describe as describeFeature, type Description } from "@/engine/describe";
import {
  band,
  buildFactSheet,
  buildLayers,
  channelOf,
  topDrivers,
  type FactSheetArgs,
} from "@/engine/explainer";
import { OPENERS, pick, templateText } from "@/engine/template";
import { loadFixture } from "./engine-fixtures";

// ---------- Fixture ----------

type Num = number | null;
type NumDict = Record<string, Num>;
interface DescribeCase {
  feature: string;
  value: Num;
  tod_h: Num;
  absolute: Num;
  out: Description | null;
}
interface LayersCase {
  name: string;
  contribs: NumDict;
  x: NumDict;
  absolute: NumDict | null;
  tod_h: Num;
  drivers_per_layer: number;
  out: Layer[];
}
interface TopDriversCase {
  layers: Layer[];
  k: number;
  effect: "supports" | "opposes" | null;
  out: Driver[];
}
interface ExplainFixture {
  version: number;
  describe: DescribeCase[];
  layers: LayersCase[];
  top_drivers: TopDriversCase[];
  channel_of: { feature: string; out: string }[];
  band: { confidence: number; out: string }[];
  fact_sheets: { args: FactSheetArgs; out: FactSheet }[];
  template: { sheet: FactSheet; out: string }[];
}

const fx = loadFixture<ExplainFixture>("explain");

/** Fixture dict of floats: null is NaN here (the exporter wrote NaN as null). */
function nanDict(d: NumDict): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(d)) out[k] = v === null ? NaN : v;
  return out;
}

// ---------- Strict comparison ----------

/**
 * Deep equality with Object.is on numbers (NaN == NaN, -0.0 != 0.0) and exact strings. Returns the paths that
 * differ (empty when equal) and records the largest numeric difference seen.
 */
function diff(got: unknown, want: unknown, path = "$", out: string[] = []): string[] {
  if (typeof got === "number" && typeof want === "number") {
    if (!Object.is(got, want)) {
      out.push(`${path}: ${got} != ${want}`);
      const e = Math.abs(got - want);
      if (Number.isFinite(e)) maxNumErr = Math.max(maxNumErr, e);
    }
    return out;
  }
  if (Array.isArray(got) && Array.isArray(want)) {
    if (got.length !== want.length) out.push(`${path}: length ${got.length} != ${want.length}`);
    for (let i = 0; i < Math.min(got.length, want.length); i++)
      diff(got[i], want[i], `${path}[${i}]`, out);
    return out;
  }
  if (got !== null && want !== null && typeof got === "object" && typeof want === "object") {
    const g = got as Record<string, unknown>;
    const w = want as Record<string, unknown>;
    const keys = new Set([...Object.keys(g), ...Object.keys(w)]);
    for (const k of keys) {
      if (!(k in g)) out.push(`${path}.${k}: missing`);
      else if (!(k in w)) out.push(`${path}.${k}: unexpected`);
      else diff(g[k], w[k], `${path}.${k}`, out);
    }
    return out;
  }
  if (got !== want) out.push(`${path}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
  return out;
}

let maxNumErr = 0;

// ---------- describe ----------

describe("describe", () => {
  it("matches every golden row exactly, strings included", () => {
    expect(fx.version).toBe(1);
    expect(fx.describe.length).toBeGreaterThan(12000);
    const bad: string[] = [];
    let narrated = 0;
    for (const c of fx.describe) {
      const got = describeFeature(c.feature, c.value === null ? NaN : c.value, {
        todH: c.tod_h,
        absolute: c.absolute,
      });
      if (got !== null) narrated++;
      const errs = diff(got, c.out);
      if (errs.length)
        bad.push(
          `${c.feature} value=${c.value} tod=${c.tod_h} abs=${c.absolute}: ${errs.join("; ")}`,
        );
    }
    expect(bad.slice(0, 20)).toEqual([]);
    expect(narrated).toBe(fx.describe.filter((c) => c.out !== null).length);
  });

  it("treats null, undefined and NaN values alike and never narrates fill_ratio", () => {
    expect(describeFeature("db_now", null)).toBeNull();
    expect(describeFeature("db_now", undefined)).toBeNull();
    expect(describeFeature("db_now", NaN)).toBeNull();
    expect(describeFeature("fill_ratio", 1.0)).toBeNull();
    expect(describeFeature("tod_sin", 0.5)).toBeNull();
  });

  it("prints Python integers in full and rounds ties to even", () => {
    expect(describeFeature("n_people", 1e21)?.display).toBe(
      "1000000000000000000000 people present",
    );
    expect(describeFeature("db_now", 72.5)?.display).toBe("current noise of 72 dB");
    expect(describeFeature("db_now", 73.5)?.display).toBe("current noise of 74 dB");
    expect(describeFeature("hours_since_break", 2.25)?.display).toBe(
      "2.2 hours since the last break",
    );
    expect(describeFeature("tod_cos", 0.5, { todH: 23.999999 })?.display).toBe(
      "time of day (23:00)",
    );
  });
});

// ---------- explainer ----------

describe("build layers", () => {
  it("matches every golden case exactly", () => {
    expect(fx.layers.length).toBeGreaterThan(30);
    for (const c of fx.layers) {
      const got = buildLayers(nanDict(c.contribs), nanDict(c.x), {
        todH: c.tod_h,
        driversPerLayer: c.drivers_per_layer,
        absolute: c.absolute === null ? null : nanDict(c.absolute),
      });
      expect(diff(got, c.out), c.name).toEqual([]);
    }
  });

  it("reads Maps and plain objects alike", () => {
    const c = fx.layers[0] as LayersCase;
    const asMap = (d: NumDict) => new Map(Object.entries(nanDict(d)));
    const got = buildLayers(asMap(c.contribs), asMap(c.x), {
      todH: c.tod_h,
      absolute: c.absolute === null ? null : asMap(c.absolute),
    });
    expect(diff(got, c.out)).toEqual([]);
  });

  it("keeps the first driver of a channel when |shap| ties", () => {
    const layers = buildLayers(
      { motion_now: 0.25, motion_ewm_1m: -0.25 },
      { motion_now: 0.04, motion_ewm_1m: 0.05 },
    );
    expect(layers[0]?.drivers.map((d) => d.feature)).toEqual(["motion_now"]);
  });
});

describe("top drivers", () => {
  it("matches every golden case exactly", () => {
    expect(fx.top_drivers.length).toBeGreaterThan(40);
    for (const c of fx.top_drivers)
      expect(diff(topDrivers(c.layers, c.k, c.effect), c.out)).toEqual([]);
  });
});

describe("channel of and band", () => {
  it("maps every feature name to its sensor channel", () => {
    for (const c of fx.channel_of) expect(channelOf(c.feature), c.feature).toBe(c.out);
  });

  it("bands confidences at 0.6 and 0.85", () => {
    for (const c of fx.band) expect(band(c.confidence), String(c.confidence)).toBe(c.out);
  });
});

describe("build fact sheet", () => {
  it("matches every golden case exactly", () => {
    expect(fx.fact_sheets.length).toBeGreaterThan(40);
    for (const c of fx.fact_sheets)
      expect(diff(buildFactSheet(c.args), c.out), c.args.event_id).toEqual([]);
  });
});

// ---------- template ----------

describe("template text", () => {
  it("renders every golden sheet character for character", () => {
    expect(fx.template.length).toBeGreaterThan(40);
    for (const c of fx.template) expect(templateText(c.sheet), c.sheet.event_id).toBe(c.out);
  });

  it("gives the same text from a fact sheet built here", () => {
    fx.fact_sheets.forEach((c, i) => {
      expect(templateText(buildFactSheet(c.args))).toBe(fx.template[i]?.out);
    });
  });

  it("picks the opener from sha1(event_id) mod 3", () => {
    // int(hashlib.sha1(key.encode()).hexdigest(), 16) % 3, computed in CPython
    const python: [string, number][] = [
      ["p1-000000", 0],
      ["p1-000003", 1],
      ["p2-000037", 2],
      ["Person é☃", 2],
    ];
    for (const [key, i] of python) expect(pick(OPENERS, key), key).toBe(OPENERS[i]);
    const seen = new Set(fx.template.map((c) => pick(OPENERS, c.sheet.event_id)));
    expect(seen.size).toBe(3);
  });
});

// ---------- Divergences found in review ----------
// Inputs the golden fixture does not reach. Every expected value below was computed with the Python reference
// (.venv/bin/python, valence/explain/*.py) and pasted in as a literal.

describe("parity on inputs outside the fixture", () => {
  it("parses a feature name with one trailing newline like Python's $ does", () => {
    // describe("db_mean_5m\n", 61.5, tod_h=15.5)
    expect(describeFeature("db_mean_5m\n", 61.5, { todH: 15.5 })).toEqual({
      display: "average noise of 62 dB over the last 5 minutes",
      unit: "dB",
      level: "elevated",
      scope: "room",
      layer: "Ambient Physics",
    });
    expect(describeFeature("fist_cnt_1m\n", 2.0)?.display).toBe(
      "2 clenched-fist events in the last minute",
    );
    expect(describeFeature("db_room_now\n", 4.5)?.display).toBe(
      "current noise 4 dB above this room's baseline",
    );
    // only "\n" is special: "\r" falls through to the raw-name fallback
    expect(describeFeature("db_mean_1m\r", 61.5)?.display).toBe("db_mean_1m\r = 61.5");
  });

  it("orders a NaN layer total (inf + -inf) exactly like CPython's sorted", () => {
    const nan = NaN;
    const layers = buildLayers(
      { db_now: -Infinity, hours_since_break: 0.37, motion_now: Infinity, slump_now: -Infinity, face_now: 0.39 },
      { db_now: 60.0, hours_since_break: 1.0, motion_now: 0.04, face_now: 0.2, slump_now: 0.5 },
      { todH: 12.0 },
    );
    // Python: [('Time Dynamics', 0.37), ('Ambient Physics', -inf), ('Somatic & Social', nan), ('Facial Surface', 0.39)]
    expect(layers.map((l) => [l.name, l.contribution])).toEqual([
      ["Time Dynamics", 0.37],
      ["Ambient Physics", -Infinity],
      ["Somatic & Social", nan],
      ["Facial Surface", 0.39],
    ]);
    expect(layers[2]?.drivers.map((d) => [d.feature, d.shap, d.effect])).toEqual([
      ["motion_now", Infinity, "supports"],
      ["slump_now", -Infinity, "opposes"],
    ]);
  });

  it("orders drivers with a NaN shap exactly like CPython's sorted", () => {
    const drv = (feature: string, shap: number): Driver => ({
      feature,
      display: feature,
      value: null,
      unit: "",
      shap,
      level: "normal",
      effect: "supports",
      scope: "person",
    });
    const layer = (drivers: Driver[]): Layer[] => [
      { name: "Somatic & Social", contribution: 0.0, drivers },
    ];
    const order = (ds: Driver[]) => topDrivers(layer(ds), 10, null).map((d) => d.feature);
    // a stable descending Array.sort gives a, b, d, c and a, b, c, f, e, d here
    expect(order([drv("a", 0.2), drv("b", NaN), drv("c", -0.1), drv("d", 0.3)])).toEqual([
      "b",
      "d",
      "a",
      "c",
    ]);
    expect(
      order([drv("a", 0.1), drv("b", NaN), drv("c", 0.3), drv("d", 0.05), drv("e", -0.1), drv("f", 0.3)]),
    ).toEqual(["b", "c", "f", "a", "e", "d"]);
  });

  it("raises like Python slicing on a non-integer driver count, only when a slice happens", () => {
    // build_layers(..., drivers_per_layer=2.5) -> TypeError; with no drivers -> []
    expect(() =>
      buildLayers({ motion_now: 0.1 }, { motion_now: 0.04 }, { driversPerLayer: 2.5 }),
    ).toThrow(TypeError);
    expect(buildLayers({}, {}, { driversPerLayer: 2.5 })).toEqual([]);
    // top_drivers([], k=2.5) -> TypeError
    expect(() => topDrivers([], 2.5)).toThrow(TypeError);
    // negative bounds count from the end, as in Python: [:-1] drops the last driver
    const layers = buildLayers(
      { motion_now: 0.3, slump_now: 0.2, hih_now: 0.1 },
      { motion_now: 0.04, slump_now: 0.5, hih_now: 0.2 },
      { driversPerLayer: -1 },
    );
    expect(layers[0]?.drivers.map((d) => d.feature)).toEqual(["motion_now", "slump_now"]);
  });

  const sheetArgs = (over: Partial<FactSheetArgs> = {}): FactSheetArgs => ({
    event_id: "p1-000001",
    person: "Person #1",
    label: "Focused",
    label_display: "Focused",
    confidence: 0.5,
    probabilities: { Focused: 0.87655 },
    window_fill_min: 1.0,
    n_people: 1,
    face_only_label: null,
    face_only_confidence: null,
    face_valence: null,
    face_visible: false,
    mask_override: false,
    layers: [],
    face_visible_frac: 0.5,
    lux: null,
    ood_features: [],
    ...over,
  });

  it("validates n_people like the pydantic int field", () => {
    // FactSheet(n_people=2.0).n_people == 2; -0.0 -> 0; 2.5, nan, inf -> ValidationError
    expect(buildFactSheet(sheetArgs({ n_people: 2.0 })).n_people).toBe(2);
    expect(Object.is(buildFactSheet(sheetArgs({ n_people: -0 })).n_people, 0)).toBe(true);
    for (const n of [2.5, NaN, Infinity]) expect(() => buildFactSheet(sheetArgs({ n_people: n }))).toThrow();
  });

  it("copies the layers and ood lists like pydantic, sharing their items", () => {
    // Python: sheet.layers is not layers, sheet.layers[0] is layers[0]; same for ood_features
    const layers = buildLayers({ motion_now: 0.2 }, { motion_now: 0.04 });
    const ood = ["a"];
    const sheet = buildFactSheet(sheetArgs({ layers, ood_features: ood }));
    layers.push(layers[0] as Layer);
    ood.push("b");
    expect(sheet.layers).toHaveLength(1);
    expect(sheet.layers[0]).toBe(layers[0]);
    expect(sheet.data_quality.ood_features).toEqual(["a"]);
  });

  it("keeps a __proto__ state name as a plain probability key", () => {
    // Python: {'__proto__': 0.1235, 'Focused': 0.8766}
    const probabilities = JSON.parse('{"__proto__": 0.12345, "Focused": 0.87655}') as Record<string, number>;
    const sheet = buildFactSheet(sheetArgs({ probabilities }));
    expect(Object.entries(sheet.probabilities)).toEqual([
      ["__proto__", 0.1235],
      ["Focused", 0.8766],
    ]);
    expect(Object.getPrototypeOf(sheet.probabilities)).toBe(Object.prototype);
  });

  it("refuses an event id with a lone surrogate like str.encode()", () => {
    // _pick(OPENERS, "p1-\ud800") -> UnicodeEncodeError: surrogates not allowed
    expect(() => pick(OPENERS, "p1-\ud800")).toThrow(/UnicodeEncodeError/);
    expect(() => pick(OPENERS, "\udc00x")).toThrow(/UnicodeEncodeError/);
    expect(pick(OPENERS, "Person é☃")).toBe(OPENERS[2]); // a valid pair or BMP text still hashes
  });
});

describe("numeric agreement", () => {
  it("reports no numeric difference anywhere", () => {
    expect(maxNumErr).toBe(0);
  });
});
