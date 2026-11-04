/**
 * Hash-chained, features-free audit log: a port of valence/governance/audit.py.
 *
 * It records decisions, not data: labels, confidences, SHAP layer sums and operator actions. Each entry
 * carries the SHA-256 of the previous one, so an edit, deletion or reordering breaks verify():
 *   hash = sha256(prev + json.dumps({seq, ts, kind, payload}, sort_keys=True, separators=(",", ":")))
 * Hashes are bit-identical to Python's, so a chain written here verifies with AuditLog.verify() and back.
 *
 * Python floats vs ints. json.dumps prints the float 2.0 as "2.0" and the int 2 as "2", and the hash
 * covers that text, but a JS number cannot say which one it is. The rule here is:
 *   - a JS `number` is a Python float (repr: 0.8123, 2.0, -0.0, 1e-05, 1e+16);
 *   - a JS `bigint` is a Python int (2n -> "2", 2n ** 63n -> "9223372036854775808").
 * Floats are the common case. Everything a 'prediction' entry carries is a float or not a number at all:
 * confidence = float(p[k]) (valence/models/served.py:129), layer contributions = round(float, 4)
 * (valence/explain/explainer.py:56), mask_override is a bool. Operator payloads that hold Python ints
 * must pass a bigint: room_baseline_captured's `target_samples` / `db_samples` / `lux_samples`
 * (valence/runtime/aggregator.py:108), sim_event's `laughter` / `compliments` and self_report's
 * `person_id` (int() in valence/api/app.py:470, :559). Stay floats: demo_seed's `minutes` and
 * `hours_since_break` (float() in valence/api/app.py:488), replay `speed`, room_baseline `seconds`,
 * calibrate_room's values. break's `person_id` and sim_lux's `lux` echo the request's JSON number, so
 * they are whatever the client sent. The entry's own fields are typed: `seq` is an int and
 * `ts` = round(now, 3) a float, whatever JS number holds them.
 *
 * Storage. Python appends each entry to var/audit/<session>.jsonl and deletes the file at session end.
 * In the browser the chain lives in this object's memory and is gone when the tab closes, so there is
 * nothing on disk to expire (AuditLog.expire_old has no counterpart). toJsonl() gives the exact text of
 * Python's file, and verifyJsonl() checks such text the way AuditLog.verify() does.
 */
import { pyRepr, pyRound } from "./num";
import { sha256Hex } from "./sha";

// ---------- Values ----------

/** A JSON value with Python's number types: `number` is a float, `bigint` an int. */
export type PyJson = null | boolean | number | bigint | string | readonly PyJson[] | PyObject;
/** A JSON object (Python dict with str keys). */
export interface PyObject {
  readonly [key: string]: PyJson;
}
/** What an entry may carry (valence/governance/audit.py:31 _check decides what is allowed). */
export type AuditPayload = PyObject;

/** One chain entry, as AuditLog.entries() reads it back (valence/governance/audit.py:84). */
export interface AuditEntry {
  /** Python int: 0, 1, 2, ... */
  readonly seq: number;
  /** Python float: round(clock, 3). */
  readonly ts: number;
  readonly kind: string;
  readonly payload: AuditPayload;
  /** The previous entry's hash (GENESIS for the first). */
  readonly prev: string;
  readonly hash: string;
}

/** The `prev` of the first entry (valence/governance/audit.py:22). */
export const GENESIS = "0".repeat(64);
/** Longer strings are refused so free text (a transcript) can never be logged (audit.py:23). */
export const MAX_STR = 200;
/** Deepest allowed nesting below the payload (audit.py:24). */
export const MAX_DEPTH = 4;
/** Longest allowed list (audit.py:49). */
export const MAX_LIST = 64;
/**
 * CPython's default sys.get_int_max_str_digits(): json.dumps and json.loads raise ValueError for an int
 * with more decimal digits (sign not counted), so Python's append() refuses such a payload.
 */
export const PY_INT_MAX_STR_DIGITS = 4300;
/**
 * Deepest container nesting CPython 3.12's C json encoder and scanner accept before RecursionError (measured
 * with this repo's .venv, also from inside AuditLog.verify()). The budget is CPython's C recursion limit minus
 * the C frames already on the stack, so it is environment-dependent; the point is that it is ~10k, while a
 * recursive JS encoder overflows the stack near 2.5k, so both codecs here use an explicit stack instead.
 */
export const PY_JSON_MAX_DEPTH = 9997;

/** Raised for a payload the audit log refuses (valence/governance/audit.py:27). */
export class AuditPayloadError extends Error {
  override name = "AuditPayloadError";
}

// ---------- Python str helpers ----------

/** Python `len(s)`: code points, so an emoji counts once (it is two UTF-16 units in JS). */
export function pyLen(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
      const d = s.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) i++;
    }
    n++;
  }
  return n;
}

/**
 * Python's str ordering (code point order), which `sorted()` and json.dumps(sort_keys=True) use. JS's
 * default sort compares UTF-16 units instead, which puts U+E000..U+FFFF after the astral planes.
 */
export function comparePyStr(a: string, b: string): number {
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const x = a.codePointAt(i) as number;
    const y = b.codePointAt(j) as number;
    if (x !== y) return x - y;
    i += x > 0xffff ? 2 : 1;
    j += y > 0xffff ? 2 : 1;
  }
  return a.length - i - (b.length - j);
}

const isPlainObject = (v: object): boolean => {
  const proto: unknown = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
};

/** Python's type(value).__name__ stand-in for error messages: the JS typeof or constructor name. */
function typeName(v: unknown): string {
  if (typeof v !== "object" || v === null) return typeof v;
  const ctor: unknown = (v as { constructor?: unknown }).constructor;
  return typeof ctor === "function" && ctor.name ? ctor.name : "object";
}

/** Own enumerable keys with a property named "__proto__" kept as data, not as the prototype. */
function setKey(obj: Record<string, unknown>, key: string, value: unknown): void {
  if (key === "__proto__") Object.defineProperty(obj, key, { value, writable: true, enumerable: true, configurable: true });
  else obj[key] = value;
}

// ---------- Payload rules ----------

/**
 * `_check` and the snapshot in one pass: every value is read once, so a getter or Proxy cannot hand the
 * check one value and the hash another. The copy is deep and frozen because Python reads entries back
 * from its file, so later edits to the caller's object never reach the chain.
 */
function checkedCopy(value: unknown, depth: number): PyJson {
  if (depth > MAX_DEPTH) throw new AuditPayloadError("payload nested too deeply");
  if (value === null || typeof value === "boolean" || typeof value === "bigint") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new AuditPayloadError("NaN/inf not allowed; use null");
    return value;
  }
  if (typeof value === "string") {
    if (pyLen(value) > MAX_STR) throw new AuditPayloadError(`string longer than ${MAX_STR} chars refused`);
    return value;
  }
  if (Array.isArray(value)) {
    const n = value.length;
    if (n > MAX_LIST) throw new AuditPayloadError(`lists longer than ${MAX_LIST} items refused`);
    // indexed reads, so a hole is undefined (refused) like it is for json.dumps
    const out: PyJson[] = [];
    for (let i = 0; i < n; i++) out.push(checkedCopy((value as unknown[])[i], depth + 1));
    return Object.freeze(out);
  }
  // JS object keys are always strings, so Python's "keys must be strings" cannot happen here. Python checks
  // items in insertion order; Object.keys lists integer-like keys first, so with two bad items the message
  // can name a different one (both refuse the payload).
  if (typeof value === "object" && isPlainObject(value)) {
    const out: Record<string, PyJson> = {};
    for (const k of Object.keys(value)) setKey(out, k, checkedCopy((value as Record<string, unknown>)[k], depth + 1));
    return Object.freeze(out);
  }
  throw new AuditPayloadError(`type ${typeName(value)} not allowed in audit payloads`);
}

/**
 * valence/governance/audit.py:31 `_check`: throw AuditPayloadError unless `value` is JSON made of
 * null / bool / int (bigint) / finite float (number) / str of at most MAX_STR code points / plain objects /
 * arrays of at most MAX_LIST items, nested at most MAX_DEPTH levels. The depth test comes first, as in Python.
 */
export function checkAuditValue(value: unknown, depth = 0): void {
  checkedCopy(value, depth);
}

// ---------- json.dumps ----------

/** Options of Python's json.dumps that matter here, with Python's defaults. */
export interface PyJsonDumpsOptions {
  /** sort_keys (default false; then keys come in JS property order, integer-like keys first). */
  sortKeys?: boolean;
  /** separators (default [", ", ": "], Python's default when indent is None). */
  separators?: readonly [string, string];
  /** ensure_ascii (default true): every non-ASCII UTF-16 unit becomes \uXXXX (lower-case hex). */
  ensureAscii?: boolean;
  /** allow_nan (default true): NaN / Infinity / -Infinity are written bare; false throws instead. */
  allowNan?: boolean;
}

const SHORT_ESCAPES: Record<number, string> = {
  0x22: '\\"', 0x5c: "\\\\", 0x08: "\\b", 0x0c: "\\f", 0x0a: "\\n", 0x0d: "\\r", 0x09: "\\t",
};

/** Python json's encode_basestring_ascii (ensureAscii) or encode_basestring. */
function encodeString(s: string, ascii: boolean): string {
  let out = '"';
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    // printable ASCII but '"' and '\' passes through; without ensure_ascii, so does everything >= 0x7f
    if ((c >= 0x20 && c <= 0x7e && c !== 0x22 && c !== 0x5c) || (!ascii && c >= 0x7f)) continue;
    out += s.slice(start, i) + (SHORT_ESCAPES[c] ?? "\\u" + c.toString(16).padStart(4, "0"));
    start = i + 1;
  }
  return out + s.slice(start) + '"';
}

/** Python json's float encoding: repr, or the bare NaN / Infinity / -Infinity that allow_nan permits. */
function encodeFloat(x: number, allowNan: boolean): string {
  if (Number.isFinite(x)) return pyRepr(x);
  if (!allowNan) throw new RangeError(`Out of range float values are not JSON compliant: ${pyRepr(x)}`);
  return Number.isNaN(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
}

/** Python int.__repr__, with the ValueError CPython raises past PY_INT_MAX_STR_DIGITS digits. */
function encodeInt(v: bigint): string {
  const s = v.toString();
  if ((v < 0n ? s.length - 1 : s.length) > PY_INT_MAX_STR_DIGITS) {
    throw new RangeError(`Exceeds the limit (${PY_INT_MAX_STR_DIGITS} digits) for integer string conversion; `
      + "use sys.set_int_max_str_digits() to increase the limit");
  }
  return s;
}

/** An open array or object while encoding: `keys` is null for an array. */
interface EncodeFrame {
  readonly v: object;
  readonly keys: readonly string[] | null;
  readonly n: number;
  i: number;
}

/**
 * Python `json.dumps(value, ...)` with ensure_ascii / allow_nan / sort_keys / separators, byte for byte:
 * numbers print as Python floats (repr), bigints as Python ints, keys sort by code point. Errors follow
 * CPython's encoder: anything that is not JSON (including an array hole) throws TypeError ("Object of type
 * X is not JSON serializable"); Python's ValueErrors (a cycle, NaN with allowNan off, an int over
 * PY_INT_MAX_STR_DIGITS digits) and its RecursionError past PY_JSON_MAX_DEPTH levels throw RangeError.
 * Without sortKeys, keys come in JS property order, which lists integer-like keys first (a Python dict
 * keeps insertion order), so pass sortKeys whenever such keys can occur.
 */
export function pyJsonDumps(value: unknown, opts: PyJsonDumpsOptions = {}): string {
  const sortKeys = opts.sortKeys ?? false;
  const [itemSep, keySep] = opts.separators ?? [", ", ": "];
  const ascii = opts.ensureAscii ?? true;
  const allowNan = opts.allowNan ?? true;
  const parts: string[] = [];
  // an explicit stack instead of recursion, so nesting as deep as CPython's does not overflow the JS stack
  const stack: EncodeFrame[] = [];
  const active = new Set<object>();

  /** Write a scalar, or open a container (its items are written by the loop below). */
  const emit = (v: unknown): void => {
    if (v === null) parts.push("null");
    else if (v === true) parts.push("true");
    else if (v === false) parts.push("false");
    else if (typeof v === "string") parts.push(encodeString(v, ascii));
    else if (typeof v === "number") parts.push(encodeFloat(v, allowNan));
    else if (typeof v === "bigint") parts.push(encodeInt(v));
    else if (typeof v === "object" && (Array.isArray(v) || isPlainObject(v))) {
      // CPython's order: the recursion budget, the empty shortcut, then the circular-reference check
      if (stack.length >= PY_JSON_MAX_DEPTH) throw new RangeError("maximum recursion depth exceeded while encoding a JSON object");
      const keys = Array.isArray(v) ? null : Object.keys(v);
      const n = keys === null ? (v as unknown[]).length : keys.length;
      if (n === 0) {
        parts.push(keys === null ? "[]" : "{}");
        return;
      }
      if (active.has(v)) throw new RangeError("Circular reference detected");
      if (keys !== null && sortKeys) keys.sort(comparePyStr);
      active.add(v);
      parts.push(keys === null ? "[" : "{");
      stack.push({ v, keys, n, i: 0 });
    } else {
      throw new TypeError(`Object of type ${typeName(v)} is not JSON serializable`);
    }
  };

  emit(value);
  for (let top = stack[stack.length - 1]; top !== undefined; top = stack[stack.length - 1]) {
    if (top.i === top.n) {
      parts.push(top.keys === null ? "]" : "}");
      active.delete(top.v);
      stack.pop();
      continue;
    }
    if (top.i > 0) parts.push(itemSep);
    const at = top.i++;
    if (top.keys === null) {
      emit((top.v as unknown[])[at]);
    } else {
      const k = top.keys[at] as string;
      parts.push(encodeString(k, ascii) + keySep);
      emit((top.v as Record<string, unknown>)[k]);
    }
  }
  return parts.join("");
}

// ---------- json.loads ----------

/** Thrown by pyJsonLoads for text Python's json.loads would refuse (json.JSONDecodeError). */
export class PyJsonDecodeError extends SyntaxError {
  override name = "PyJsonDecodeError";
}

const NUMBER_RE = /-?(?:0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?/y;
const SIMPLE_ESCAPES = new Map([
  ['"', '"'], ["\\", "\\"], ["/", "/"], ["b", "\b"], ["f", "\f"], ["n", "\n"], ["r", "\r"], ["t", "\t"],
]);

/** An open array or object while decoding; an object frame holds the key its next value goes under. */
type DecodeFrame = { readonly arr: PyJson[] } | { readonly obj: Record<string, PyJson>; key: string };

/**
 * Python `json.loads(text)` keeping Python's number types: an integer token ("2", "-7") becomes a
 * bigint, a token with a fraction or exponent ("2.0", "1e-05") or NaN / Infinity / -Infinity becomes a
 * number, exactly how json.loads picks int or float. Strict like Python: raw control characters inside
 * strings, bad escapes and trailing data throw PyJsonDecodeError. Duplicate keys keep the last value (in
 * the first one's position, as in a dict). Python's ValueError for an int token over PY_INT_MAX_STR_DIGITS
 * digits and its RecursionError past PY_JSON_MAX_DEPTH nested containers throw RangeError.
 */
export function pyJsonLoads(text: string): PyJson {
  let i = 0;
  const fail = (msg: string): never => {
    throw new PyJsonDecodeError(`${msg}: char ${i}`);
  };
  const ws = (): void => {
    while (i < text.length) {
      const c = text.charCodeAt(i);
      if (c !== 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d) break;
      i++;
    }
  };
  const lit = (word: string): boolean => {
    if (!text.startsWith(word, i)) return false;
    i += word.length;
    return true;
  };

  const str = (): string => {
    i++; // opening quote
    let out = "";
    let start = i;
    for (;;) {
      if (i >= text.length) fail("Unterminated string starting at");
      const c = text.charCodeAt(i);
      if (c === 0x22) {
        out += text.slice(start, i);
        i++;
        return out;
      }
      if (c < 0x20) fail("Invalid control character at");
      if (c !== 0x5c) {
        i++;
        continue;
      }
      out += text.slice(start, i);
      const e = text[i + 1] ?? "";
      const simple = SIMPLE_ESCAPES.get(e);
      if (simple !== undefined) {
        out += simple;
        i += 2;
      } else if (e === "u") {
        const hex = text.slice(i + 2, i + 6);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail("Invalid \\uXXXX escape");
        out += String.fromCharCode(parseInt(hex, 16)); // an escaped surrogate pair rejoins by itself
        i += 6;
      } else {
        fail("Invalid \\escape");
      }
      start = i;
    }
  };

  /** int(token), with the ValueError CPython raises past PY_INT_MAX_STR_DIGITS digits. */
  const intToken = (tok: string): bigint => {
    const digits = tok.startsWith("-") ? tok.length - 1 : tok.length;
    if (digits > PY_INT_MAX_STR_DIGITS) {
      throw new RangeError(`Exceeds the limit (${PY_INT_MAX_STR_DIGITS} digits) for integer string conversion: `
        + `value has ${digits} digits; use sys.set_int_max_str_digits() to increase the limit`);
    }
    return BigInt(tok);
  };

  /** A string, constant or number at i; containers are opened by the loop below. */
  const scalar = (): PyJson => {
    if (text[i] === '"') return str();
    if (lit("null")) return null;
    if (lit("true")) return true;
    if (lit("false")) return false;
    if (lit("NaN")) return NaN;
    if (lit("Infinity")) return Infinity;
    if (lit("-Infinity")) return -Infinity;
    NUMBER_RE.lastIndex = i;
    const m = NUMBER_RE.exec(text);
    if (m) {
      i += m[0].length;
      // Python: float(token) if it has a fraction or an exponent, else int(token); both parse exactly
      return m[1] !== undefined || m[2] !== undefined ? Number(m[0]) : intToken(m[0]);
    }
    return fail("Expecting value");
  };

  /** After "{" or ",": the next key and its ":" (CPython's _parse_object_unicode). */
  const key = (): string => {
    ws();
    if (text[i] !== '"') fail("Expecting property name enclosed in double quotes");
    const k = str();
    ws();
    if (text[i] !== ":") fail("Expecting ':' delimiter");
    i++;
    return k;
  };

  // an explicit stack instead of recursion, so nesting as deep as CPython's does not overflow the JS stack
  const stack: DecodeFrame[] = [];
  for (;;) {
    // a value starts here
    ws();
    const c = text[i];
    let v: PyJson;
    if (c === "[" || c === "{") {
      // CPython checks its recursion budget before it reads the container's contents
      if (stack.length >= PY_JSON_MAX_DEPTH) {
        throw new RangeError(`maximum recursion depth exceeded while decoding a JSON ${c === "[" ? "array" : "object"} from a unicode string`);
      }
      i++;
      ws();
      if (c === "[") {
        const arr: PyJson[] = [];
        if (text[i] !== "]") {
          stack.push({ arr });
          continue;
        }
        v = arr;
      } else {
        const obj: Record<string, PyJson> = {};
        if (text[i] !== "}") {
          stack.push({ obj, key: key() });
          continue;
        }
        v = obj;
      }
      i++; // the empty container's closing bracket
    } else {
      v = scalar();
    }
    // store the finished value, then close every container that ends right after it
    for (;;) {
      const top = stack[stack.length - 1];
      if (top === undefined) {
        ws();
        if (i < text.length) fail("Extra data");
        return v;
      }
      if ("arr" in top) top.arr.push(v);
      else setKey(top.obj, top.key, v);
      ws();
      if (text[i] === ",") {
        i++;
        if ("obj" in top) top.key = key();
        break;
      }
      if (text[i] === ("arr" in top ? "]" : "}")) {
        i++;
        stack.pop();
        v = "arr" in top ? top.arr : top.obj;
        continue;
      }
      fail("Expecting ',' delimiter");
    }
  }
}

// ---------- Hashing ----------

/**
 * The text an entry's hash covers: `json.dumps(body, sort_keys=True, separators=(",", ":"))`
 * (valence/governance/audit.py:59). Always ASCII.
 */
export function auditBlob(body: PyJson): string {
  return pyJsonDumps(body, { sortKeys: true, separators: [",", ":"] });
}

/** valence/governance/audit.py:58 `_digest`: sha256(prev + blob), hex. */
export function auditDigest(prev: string, body: PyJson): string {
  return sha256Hex(prev + auditBlob(body));
}

/** A typed `seq` as the Python int it stands for (a non-integer, only possible after tampering, stays a float). */
const seqAsInt = (seq: number | bigint): number | bigint =>
  typeof seq === "number" && Number.isInteger(seq) ? BigInt(seq) : seq;

/** The hashed body {seq, ts, kind, payload} of a typed entry (valence/governance/audit.py:86, :104). */
export function entryBody(entry: AuditEntry): PyObject {
  return { seq: seqAsInt(entry.seq), ts: entry.ts, kind: entry.kind, payload: entry.payload };
}

/** The entry as a Python dict, with seq as an int: what json.dumps sees when Python writes the file. */
function pyEntry(entry: AuditEntry): PyObject {
  return { ...entryBody(entry), prev: entry.prev, hash: entry.hash };
}

/** One line of Python's audit file: `json.dumps(entry, sort_keys=True)` (valence/governance/audit.py:90). */
export function entryToJsonLine(entry: AuditEntry): string {
  return pyJsonDumps(pyEntry(entry), { sortKeys: true });
}

// ---------- Verification ----------

/** Python AuditLog.verify's loop (valence/governance/audit.py:100) over any row shape. */
function verifyRows<T>(rows: readonly T[], bodyOf: (row: T) => PyJson, prevOf: (row: T) => unknown,
  hashOf: (row: T) => unknown): [ok: boolean, checked: number] {
  let prev = GENESIS;
  let n = 0;
  for (const row of rows) {
    n++;
    // Python's order: the body's four keys first, then prev, then the digest, and hash only when prev
    // matched (the `or` short-circuits), so a damaged row raises the same KeyError, or none, as in Python
    const body = bodyOf(row);
    if (prevOf(row) !== prev) return [false, n];
    const digest = auditDigest(prev, body);
    const hash = hashOf(row);
    if (digest !== hash) return [false, n];
    prev = hash as string;
  }
  return [true, n];
}

/**
 * valence/governance/audit.py:100 `verify`, over typed entries: recompute the chain from GENESIS.
 * Returns [ok, entries checked]; on a break, the 1-based position of the first bad entry.
 */
export function verifyChain(entries: readonly AuditEntry[]): [ok: boolean, checked: number] {
  return verifyRows(entries, entryBody, (e) => e.prev, (e) => e.hash);
}

/** Code points of Python's str.isspace() (numbers, so no invisible line separator sits in the source). */
const PY_SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

/** A line Python's `if line.strip()` skips: empty or whitespace only. */
function isBlankLine(line: string): boolean {
  for (let i = 0; i < line.length; i++) if (!PY_SPACE.has(line.charCodeAt(i))) return false;
  return true;
}

/** `row[key]` with Python's KeyError for a missing key or a row that is not a dict. */
function field(row: PyJson, key: string): PyJson {
  if (row === null || typeof row !== "object" || Array.isArray(row)) {
    throw new TypeError(`audit row is not an object: ${pyJsonDumps(row)}`);
  }
  const v = (row as PyObject)[key];
  if (v === undefined || !Object.prototype.hasOwnProperty.call(row, key)) throw new Error(`KeyError: '${key}'`);
  return v;
}

/**
 * AuditLog.verify() over the text of an audit file (valence/governance/audit.py:93 entries + :100 verify):
 * every non-blank line is parsed with Python's number types (pyJsonLoads), rows with seq < 0 are skipped
 * like entries(since_seq=0) does, and each body is hashed exactly as parsed, so a float written where
 * Python wrote an int is caught too. A row without seq / ts / kind / payload / prev / hash throws
 * (Python: KeyError).
 */
export function verifyJsonl(text: string): [ok: boolean, checked: number] {
  // text-mode reading splits on \n, \r and \r\n (universal newlines)
  const rows = text.split(/\r\n|\r|\n/).filter((line) => !isBlankLine(line)).map(pyJsonLoads);
  const kept = rows.filter((r) => {
    const seq = field(r, "seq");
    if (typeof seq === "boolean") return true; // Python: True >= 0 and False >= 0
    if (typeof seq !== "number" && typeof seq !== "bigint") throw new TypeError("'>=' not supported between seq and int");
    return seq >= 0;
  });
  const bodyOf = (r: PyJson): PyJson => ({
    seq: field(r, "seq"), ts: field(r, "ts"), kind: field(r, "kind"), payload: field(r, "payload"),
  });
  return verifyRows(kept, bodyOf, (r) => field(r, "prev"), (r) => field(r, "hash"));
}

// ---------- Chain ----------

/**
 * The session's audit chain: valence/governance/audit.py:65 AuditLog without the file. append() checks
 * and snapshots the payload, stamps seq and ts = round(now, 3), and links the entry to the previous hash.
 * Entries are frozen; a verify() after any change made to a copy reports where the chain breaks.
 */
export class AuditChain {
  readonly sessionId: string;
  private seq = 0;
  private last = GENESIS;
  private rows: AuditEntry[] = [];

  constructor(sessionId = "browser") {
    this.sessionId = sessionId;
  }

  /** The newest hash (GENESIS before the first entry); governance messages carry its first 16 chars. */
  get lastHash(): string {
    return this.last;
  }

  /** How many entries are held. */
  get length(): number {
    return this.rows.length;
  }

  /**
   * valence/governance/audit.py:81 `append`. `now` is the clock reading in seconds (Python: time.time(); the
   * runtime passes the tick wall); the entry's ts is round(now, 3). Throws AuditPayloadError, leaving the
   * chain unchanged, for a kind or payload Python refuses. Python returns only the hash; this returns the
   * whole entry (entry.hash is that hash).
   */
  append(kind: string, payload: AuditPayload, now: number = Date.now() / 1000): AuditEntry {
    checkAuditValue(kind);
    const snapshot = checkedCopy(payload, 0) as AuditPayload;
    // time.time() is always finite; a NaN clock here would be a caller bug, not something to hash
    if (!Number.isFinite(now)) throw new RangeError(`audit clock must be finite, got ${now}`);
    const ts = pyRound(now, 3);
    // json.dumps can still refuse a checked payload (an int over PY_INT_MAX_STR_DIGITS digits): it throws
    // here, before anything changes, as Python's append raises before writing or advancing seq
    const hash = auditDigest(this.last, { seq: BigInt(this.seq), ts, kind, payload: snapshot });
    const entry: AuditEntry = Object.freeze({ seq: this.seq, ts, kind, payload: snapshot, prev: this.last, hash });
    this.rows.push(entry);
    this.seq += 1;
    this.last = hash;
    return entry;
  }

  /** valence/governance/audit.py:93 `entries`: the entries with seq >= sinceSeq, oldest first. */
  entries(sinceSeq = 0): AuditEntry[] {
    return this.rows.filter((e) => e.seq >= sinceSeq);
  }

  /** valence/governance/audit.py:100 `verify`: [ok, entries checked] over this chain or the given entries. */
  verify(entries: readonly AuditEntry[] = this.rows): [ok: boolean, checked: number] {
    return verifyChain(entries);
  }

  /** The exact text of Python's var/audit/<session>.jsonl for these entries (one json.dumps line each). */
  toJsonl(): string {
    return this.rows.map((e) => entryToJsonLine(e) + "\n").join("");
  }

  /**
   * valence/governance/audit.py:121 `close(delete=True)`: drop the entries. As in Python (which only deletes
   * the file), seq and lastHash are kept. close(false) keeps everything, like Python's close(delete=False).
   */
  close(deleteEntries = true): void {
    if (deleteEntries) this.rows = [];
  }
}
