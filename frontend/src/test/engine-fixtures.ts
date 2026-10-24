/**
 * Golden-fixture helpers for the engine parity tests (node only: fs + zlib).
 *
 * Fixtures are written by scripts/export_web_golden*.py into src/test/fixtures/engine/ as gzipped JSON.
 * Floats are written with Python's repr, so JSON.parse yields the identical double; NaN is encoded as
 * null and +-inf as the strings "Infinity" / "-Infinity" (JSON has no literal for them). nn() undoes both.
 * Models are read straight from the repository's artifacts/ directory (artifactText / artifactJson).
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

// ---------- Paths ----------

/** The frontend package root (this file lives in src/test/). */
export const FRONTEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
/** Where the exporters write the engine fixtures. */
export const FIXTURE_DIR = path.join(FRONTEND_ROOT, "src/test/fixtures/engine");
/** The repository's model artifacts (m2.txt, m0.txt, model_meta.json, clip_bounds.json, all/...). */
export const ARTIFACTS_DIR = path.resolve(FRONTEND_ROOT, "../artifacts");

/**
 * Path of a fixture: a bare name ("num", "num.json.gz") resolves in FIXTURE_DIR; anything containing a
 * slash is taken relative to the frontend root.
 */
export function fixturePath(name: string): string {
  if (name.includes("/")) return path.resolve(FRONTEND_ROOT, name);
  return path.join(FIXTURE_DIR, name.endsWith(".json.gz") ? name : `${name}.json.gz`);
}

// ---------- Loading ----------

/** Parse a gzipped JSON fixture. The caller states the schema type; nothing is validated here. */
export function loadFixture<T>(name: string): T {
  return JSON.parse(gunzipSync(readFileSync(fixturePath(name))).toString("utf8")) as T;
}

/** Text of a file under artifacts/ (e.g. "m2.txt", "all/m2.txt"). */
export function artifactText(rel: string): string {
  return readFileSync(path.join(ARTIFACTS_DIR, rel), "utf8");
}

/** Raw bytes of a file under artifacts/ (for hashes that Python takes over read_bytes()). */
export function artifactBytes(rel: string): Uint8Array {
  return new Uint8Array(readFileSync(path.join(ARTIFACTS_DIR, rel)));
}

/** Parsed JSON file under artifacts/ (e.g. "model_meta.json", "clip_bounds.json"). */
export function artifactJson<T>(rel: string): T {
  return JSON.parse(artifactText(rel)) as T;
}

// ---------- Decoding ----------

/** The type nn() returns: null and number leaves become number, arrays are mapped, the rest is kept. */
export type NN<T> = T extends null | number ? number : T extends readonly (infer U)[] ? NN<U>[] : T;

/**
 * Decode fixture numbers: null -> NaN, "Infinity" / "-Infinity" / "NaN" -> the number, recursively through
 * (nested) arrays. Other values, objects included, are returned unchanged.
 */
export function nn<T>(x: T): NN<T> {
  return decode(x) as NN<T>;
}

function decode(x: unknown): unknown {
  if (x === null) return NaN;
  if (x === "Infinity") return Infinity;
  if (x === "-Infinity") return -Infinity;
  if (x === "NaN") return NaN;
  if (Array.isArray(x)) return x.map(decode);
  return x;
}
