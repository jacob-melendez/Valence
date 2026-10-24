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

describe("numeric agreement", () => {
  it("reports no numeric difference anywhere", () => {
    expect(maxNumErr).toBe(0);
  });
});
