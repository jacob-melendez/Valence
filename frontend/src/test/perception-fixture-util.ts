/**
 * Parity fixtures written by scripts/export_perception_fixtures.py (strict JSON: non-finite floats are the strings
 * "NaN", "Infinity" and "-Infinity"), revived here into numbers.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { expect } from "vitest";

const NON_FINITE: Record<string, number> = { NaN: NaN, Infinity: Infinity, "-Infinity": -Infinity };

export function loadFixture<T = unknown>(name: string): T {
  const file = path.resolve(__dirname, "perception-fixtures", `${name}.json`);
  return JSON.parse(readFileSync(file, "utf8"), (_k, v: unknown) =>
    typeof v === "string" && v in NON_FINITE ? NON_FINITE[v] : v,
  ) as T;
}

/** Same value, NaN equal to NaN, within `tol` (absolute, plus relative to the magnitude). */
export function expectClose(actual: number, expected: number, tol = 1e-9, what = ""): void {
  if (Number.isNaN(expected) || !Number.isFinite(expected)) {
    expect(actual, what).toBe(expected);
    return;
  }
  const err = Math.abs(actual - expected);
  if (!(err <= tol * (1 + Math.abs(expected)))) {
    expect.fail(`${what}: expected ${expected}, got ${actual} (|err| ${err} > ${tol} x (1 + |expected|))`);
  }
}
