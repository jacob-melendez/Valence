/**
 * num.ts against numpy / CPython golden values (scripts/export_web_golden_num.py -> num.json.gz).
 * Numbers must match bit for bit (Object.is: -0.0 vs 0.0 and NaN included); strings exactly.
 */
import { describe, expect, it } from "vitest";
import * as num from "@/engine/num";
import { loadFixture, nn } from "./engine-fixtures";

// ---------- Fixture ----------

type Raises = { raises: string };
interface NumCase { fn: string; args: unknown[]; out: unknown }
interface NumFixture { version: number; numpy: string; cases: NumCase[] }

const fixture = loadFixture<NumFixture>("num");

/** Fixture arguments: arrays and scalars through nn(), option objects as they are. */
const decodeArgs = (args: unknown[]): unknown[] =>
  args.map((a) => (a !== null && typeof a === "object" && !Array.isArray(a) ? a : nn(a)));

const isRaises = (out: unknown): out is Raises =>
  out !== null && typeof out === "object" && !Array.isArray(out) && "raises" in out;

type AnyFn = (...args: never[]) => unknown;
const exportsByName = num as unknown as Record<string, AnyFn | undefined>;

const show = (v: unknown): string =>
  typeof v === "number" ? (Object.is(v, -0) ? "-0" : String(v)) : JSON.stringify(v, (_k, x) => (typeof x === "number" && Number.isNaN(x) ? "NaN" : x));

function sameValue(actual: unknown, expected: unknown): boolean {
  if (typeof expected === "number") return typeof actual === "number" && Object.is(actual, expected);
  if (Array.isArray(expected)) {
    const a = Array.from(actual as ArrayLike<unknown>);
    return a.length === expected.length && expected.every((e, i) => sameValue(a[i], e));
  }
  return actual === expected;
}

/** Every mismatch of one export, as readable lines (an empty list means bit-exact). */
function mismatches(fn: string, cases: NumCase[]): string[] {
  const impl = exportsByName[fn];
  if (!impl) return [`num.ts has no export ${fn}`];
  const bad: string[] = [];
  for (const c of cases) {
    const args = decodeArgs(c.args);
    const call = () => (impl as (...a: unknown[]) => unknown)(...args);
    if (isRaises(c.out)) {
      let threw = false;
      try { call(); } catch { threw = true; }
      if (!threw) bad.push(`${fn}(${show(args)}) should throw (${c.out.raises})`);
      continue;
    }
    const expected = nn(c.out);
    let actual: unknown;
    try { actual = call(); } catch (e) { actual = `threw ${String(e)}`; }
    if (!sameValue(actual, expected)) bad.push(`${fn}(${show(args).slice(0, 160)}) = ${show(actual)}, numpy/Python ${show(expected)}`);
  }
  return bad;
}

const byFn = new Map<string, NumCase[]>();
for (const c of fixture.cases) byFn.set(c.fn, [...(byFn.get(c.fn) ?? []), c]);

// ---------- Golden parity ----------

describe("num golden fixture", () => {
  it("has version 1 and covers every reduction and formatter", () => {
    expect(fixture.version).toBe(1);
    for (const fn of ["npSum", "nanmean", "nanstd", "nanmedian", "percentileLinear", "pyRound", "pyFixed", "pyG", "pyRepr", "pyIntRound"]) {
      expect(byFn.get(fn)?.length ?? 0, fn).toBeGreaterThan(50);
    }
    const formatting = ["pyRound", "pyFixed", "pyG", "pyRepr", "pyIntRound"].reduce((n, fn) => n + (byFn.get(fn)?.length ?? 0), 0);
    expect(formatting).toBeGreaterThanOrEqual(2000);
  });

  for (const [fn, cases] of byFn) {
    it(`matches numpy/CPython bit for bit: ${fn} (${cases.length} cases)`, () => {
      expect(mismatches(fn, cases)).toEqual([]);
    });
  }
});

// ---------- Behaviour spot checks ----------

describe("pairwise summation", () => {
  it("sums blocks of 8 partial sums rather than left to right", () => {
    // Left to right, every +1 on 1e16 is a tie that rounds back to 1e16. numpy's 8 accumulators lose only
    // the two added to the 1e16 lane: ((1e16 + 2) + 4) + 8.
    const a = [1e16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let naive = 0;
    for (const x of a) naive += x;
    expect(naive).toBe(1e16);
    expect(num.npSum(a)).toBe(1e16 + 14);
  });

  it("seeds the reduction with +0.0, so negative zeros sum to +0.0", () => {
    expect(Object.is(num.npSum([-0]), 0)).toBe(true);
    expect(Object.is(num.npSum(new Array(9).fill(-0)), 0)).toBe(true);
    expect(num.npSum([])).toBe(0);
  });

  it("zero-fills NaNs in place for nanmean and nanstd", () => {
    expect(num.nanmean([1, NaN, 3])).toBe(2);
    expect(num.nanstd([1, NaN, 3])).toBe(1);
    expect(num.nanmean([NaN, NaN])).toBeNaN();
    expect(num.nanstd([5], 1)).toBeNaN();
  });
});

describe("order statistics", () => {
  it("returns NaN from median and percentile when any value is NaN", () => {
    expect(num.median([1, NaN, 2])).toBeNaN();
    expect(num.percentileLinear([1, 2, NaN], 50)).toBeNaN();
    expect(num.nanmedian([1, NaN, 2])).toBe(1.5);
  });

  it("interpolates percentiles with numpy's lerp", () => {
    expect(num.percentileLinear([1, 2, 3, 4], 25)).toBe(1.75);
    expect(num.percentileLinear([10], 90)).toBe(10);
    expect(num.percentileLinear([3, 1, 2], 100)).toBe(3);
  });

  it("takes Python's statistics.median of the middle pair", () => {
    expect(num.statsMedian([3, 1, 4, 1])).toBe(2);
    expect(num.statsMedian([5])).toBe(5);
  });
});

describe("python float formatting", () => {
  it("rounds the exact binary value with ties to even", () => {
    expect(num.pyRound(2.675, 2)).toBe(2.67);
    expect(num.pyRound(0.125, 2)).toBe(0.12);
    expect(num.pyRound(0.375, 2)).toBe(0.38);
    expect(Object.is(num.pyRound(-0.001, 2), -0)).toBe(true);
    expect(num.pyIntRound(2.5)).toBe(2);
    expect(num.pyIntRound(-2.5)).toBe(-2);
    expect(Object.is(num.pyIntRound(-0.4), 0)).toBe(true);
  });

  it("formats fixed point like f-strings, keeping the sign of negative zeros", () => {
    expect(num.pyFixed(0.125, 2)).toBe("0.12");
    expect(num.pyFixed(-0.0, 2, { sign: true })).toBe("-0.00");
    expect(num.pyFixed(-0.001, 2)).toBe("-0.00");
    expect(num.pyFixed(0, 1, { sign: true })).toBe("+0.0");
    expect(num.pyFixed(2.5, 0)).toBe("2");
    expect(num.pyFixed(1e22, 2)).toBe("10000000000000000000000.00");
    expect(num.pyFixed(NaN, 1, { sign: true })).toBe("+nan");
  });

  it("formats general and repr forms with Python's exponent rules", () => {
    expect(num.pyG(1e16)).toBe("1e+16");
    expect(num.pyG(100000)).toBe("100000");
    expect(num.pyG(1e6)).toBe("1e+06");
    expect(num.pyG(-0.0)).toBe("-0");
    expect(num.pyG(0.0001)).toBe("0.0001");
    expect(num.pyRepr(1e-5)).toBe("1e-05");
    expect(num.pyRepr(2.0)).toBe("2.0");
    expect(num.pyRepr(1e16)).toBe("1e+16");
    expect(num.pyRepr(1e15)).toBe("1000000000000000.0");
    expect(num.pyRepr(-0.0)).toBe("-0.0");
    expect(num.pyRepr(Infinity)).toBe("inf");
  });

  it("throws where Python raises", () => {
    expect(() => num.pyIntRound(NaN)).toThrow();
    expect(() => num.pyIntRound(Infinity)).toThrow();
    expect(() => num.pyRound(1.7976931348623157e308, -308)).toThrow();
  });
});

describe("numpy rounding", () => {
  it("uses the inexact multiply-rint-divide algorithm", () => {
    expect(num.rint(2.5)).toBe(2);
    expect(Object.is(num.rint(-0.4), -0)).toBe(true);
    expect(num.npRound(2.675, 2)).toBe(2.68); // 267.49999999999997 * ... rounds up in numpy, unlike CPython
  });
});

describe("numpy clip", () => {
  it("propagates NaN from any argument and keeps the input's zero on a tie", () => {
    expect(num.clip(NaN, 0, 1)).toBeNaN();
    expect(num.clip(0.5, NaN, 1)).toBeNaN();
    expect(num.clip(0.5, 0, NaN)).toBeNaN();
    expect(Object.is(num.clip(-0, 0, 1), -0)).toBe(true);
    expect(Object.is(num.clip(2, 0, -0), -0)).toBe(true);
    expect(num.clip(-3, -1, 1)).toBe(-1);
  });
});

describe("python slices", () => {
  it("copies the last n values like a[-n:], whole array for n <= 0 or n > length", () => {
    expect(Array.from(num.tail([1, 2, 3], 2))).toEqual([2, 3]);
    expect(Array.from(num.tail([1, 2, 3], 5))).toEqual([1, 2, 3]);
    expect(Array.from(num.tail([1, 2, 3], 0))).toEqual([1, 2, 3]);
  });
});
