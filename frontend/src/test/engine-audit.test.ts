/**
 * audit.ts against valence/governance/audit.py: the golden chains in audit.json.gz and the runtime's audit
 * chains in runtime.json.gz are re-hashed, verified, tampered with and rebuilt by appending the same
 * payloads at the same clock. Python's int/float split is recovered two ways and the two must agree: by
 * parsing the fixture text with pyJsonLoads, and from the exporter's `floats` pointers.
 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  AuditChain,
  AuditPayloadError,
  GENESIS,
  auditBlob,
  auditDigest,
  checkAuditValue,
  comparePyStr,
  entryToJsonLine,
  pyJsonDumps,
  pyJsonLoads,
  pyLen,
  verifyChain,
  verifyJsonl,
  type AuditEntry,
  type AuditPayload,
  type PyJson,
  type PyObject,
} from "@/engine/audit";
import { sha256Hex } from "@/engine/sha";
import { fixturePath, loadFixture } from "./engine-fixtures";

// ---------- Fixtures ----------

interface GoldenEntry {
  seq: number;
  ts: number;
  kind: string;
  payload: Record<string, unknown>;
  prev: string;
  hash: string;
  clock: number;
  blob: string;
  floats: string[];
}
type Row = Omit<GoldenEntry, "clock" | "blob" | "floats">;
interface AuditFixture {
  version: number;
  genesis: string;
  cases: { name: string; entries: GoldenEntry[]; verify: [boolean, number] }[];
  tampered: { name: string; entries: Row[]; verify: [boolean, number] }[];
  rejects: { name: string; kind: string; payload: unknown; error: string }[];
}
interface RuntimeFixture {
  runs: { name: string; audit: GoldenEntry[] }[];
}

const RAW = gunzipSync(readFileSync(fixturePath("audit"))).toString("utf8");
/** Plain JSON.parse: every number is a JS number. */
const FX = JSON.parse(RAW) as AuditFixture;
/** The same text with Python's number types (int -> bigint, float -> number). */
const PY = pyJsonLoads(RAW) as PyObject;
const RT = loadFixture<RuntimeFixture>("runtime");

const get = (v: PyJson | undefined, ...keys: (string | number)[]): PyJson => {
  let cur: PyJson | undefined = v;
  for (const k of keys) cur = (cur as Record<string | number, PyJson> | undefined)?.[k];
  if (cur === undefined) throw new Error(`missing ${keys.join("/")}`);
  return cur;
};
const pyCaseEntry = (ci: number, ei: number): PyObject => get(PY, "cases", ci, "entries", ei) as PyObject;

/** RFC 6901 pointers of every float (JS number) in a Python-typed value, like the exporter's float_paths. */
function floatPaths(v: PyJson, path = ""): string[] {
  if (typeof v === "number") return [path];
  if (v === null || typeof v !== "object") return [];
  if (Array.isArray(v)) return (v as readonly PyJson[]).flatMap((x, i) => floatPaths(x, `${path}/${i}`));
  return Object.keys(v).flatMap((k) =>
    floatPaths((v as PyObject)[k] as PyJson, `${path}/${k.replace(/~/g, "~0").replace(/\//g, "~1")}`));
}

/** JSON.parse output typed back to Python: a number whose pointer is not in `floats` is an int (bigint). */
function typeByPointers(v: unknown, floats: ReadonlySet<string>, path: string): PyJson {
  if (typeof v === "number") {
    if (floats.has(path)) return v;
    if (!Number.isInteger(v)) throw new Error(`int at ${path} is not integral: ${v}`);
    return BigInt(v);
  }
  if (v === null || typeof v !== "object") return v as PyJson;
  if (Array.isArray(v)) return v.map((x, i) => typeByPointers(x, floats, `${path}/${i}`));
  const out: Record<string, PyJson> = {};
  for (const [k, x] of Object.entries(v)) {
    out[k] = typeByPointers(x, floats, `${path}/${k.replace(/~/g, "~0").replace(/\//g, "~1")}`);
  }
  return out;
}

/** A golden entry as a typed AuditEntry whose payload carries Python's number types (from the pointers). */
function typedEntry(e: GoldenEntry | Row, floats: readonly string[]): AuditEntry {
  const payload = typeByPointers(e.payload, new Set(floats), "/payload") as AuditPayload;
  return { seq: e.seq, ts: e.ts, kind: e.kind, payload, prev: e.prev, hash: e.hash };
}

/** The exact text Python writes to var/audit/<session>.jsonl for these rows. */
const pyFileText = (rows: readonly PyJson[]): string =>
  rows.map((r) => pyJsonDumps(r, { sortKeys: true }) + "\n").join("");

const ch = (...codes: number[]): string => String.fromCharCode(...codes);
/** "\u" + hex as printed by json.dumps (spelled this way so the source holds no escape sequence). */
const esc = (hex: string): string => "\\" + "u" + hex;

// ---------- json.loads ----------

describe("python json.loads", () => {
  it("types integer tokens as ints and the rest as floats, matching the exporter's float pointers", () => {
    let n = 0;
    FX.cases.forEach((c, ci) => {
      c.entries.forEach((e, ei) => {
        const { clock: _c, blob: _b, floats: _f, ...row } = pyCaseEntry(ci, ei);
        expect(floatPaths(row).sort(), `${c.name}[${ei}]`).toEqual([...e.floats].sort());
        n++;
      });
    });
    expect(n).toBe(14);
  });

  it("keeps big ints exact and -0.0, NaN and the infinities as floats", () => {
    const v = pyJsonLoads('[-0.0, NaN, Infinity, -Infinity, 9223372036854775808, -0, 1E+2, 5e-324, "a\\/b"]');
    expect(v).toEqual([-0, NaN, Infinity, -Infinity, 9223372036854775808n, 0n, 100, 5e-324, "a/b"]);
    expect(Object.is((v as PyJson[])[0], -0)).toBe(true);
    expect(get(PY, "cases", 1, "entries", 0, "payload", "x")).toBe(2n ** 63n);
    expect(get(PY, "cases", 1, "entries", 0, "payload", "h")).toBe(10n ** 20n);
    expect(get(PY, "cases", 1, "entries", 0, "payload", "w")).toBe(1.2345678901234567e19);
  });

  it("decodes escapes and surrogate pairs into the same string python holds", () => {
    const text = get(PY, "cases", 2, "entries", 0, "payload") as PyObject;
    expect(text["emoji"]).toBe("smile " + String.fromCodePoint(0x1f600) + " ok");
    expect(text["line_sep"]).toBe(ch(0x2028, 0x2029));
    expect(text["controls"]).toBe("tab\there\nnew" + ch(0) + "nul" + ch(0x1f) + "us" + ch(0x7f) + "del");
    expect(pyJsonLoads(`"${esc("d83d")}${esc("de00")}${esc("D800")}"`)).toBe(ch(0xd83d, 0xde00, 0xd800));
  });

  it("refuses text python's json.loads refuses", () => {
    for (const bad of ['{"a":1,}', "[01]", '"a' + ch(1) + '"', '"\\x"', "1 2", "", "{'a':1}", "[1", '"\\u12"', "-", "nan"]) {
      expect(() => pyJsonLoads(bad), JSON.stringify(bad)).toThrow(SyntaxError);
    }
  });

  it("keeps a __proto__ key as data", () => {
    const v = pyJsonLoads('{"__proto__": {"x": 1}, "a": 2}') as PyObject;
    expect(Object.keys(v)).toEqual(["__proto__", "a"]);
    expect(Object.getPrototypeOf(v)).toBe(Object.prototype);
    expect(pyJsonDumps(v, { sortKeys: true })).toBe('{"__proto__": {"x": 1}, "a": 2}');
  });
});

// ---------- json.dumps ----------

describe("python json.dumps", () => {
  it("reproduces every hashed blob of the golden chains byte for byte", () => {
    let n = 0;
    FX.cases.forEach((c, ci) => {
      c.entries.forEach((e, ei) => {
        const py = pyCaseEntry(ci, ei);
        const body = { seq: py["seq"] as PyJson, ts: py["ts"] as PyJson, kind: py["kind"] as PyJson, payload: py["payload"] as PyJson };
        expect(auditBlob(body), `${c.name}[${ei}] via json.loads`).toBe(e.blob);
        const t = typedEntry(e, e.floats);
        expect(auditBlob({ seq: BigInt(t.seq), ts: t.ts, kind: t.kind, payload: t.payload }), `${c.name}[${ei}] via pointers`).toBe(e.blob);
        expect(pyJsonDumps(pyJsonLoads(e.blob), { sortKeys: true, separators: [",", ":"] })).toBe(e.blob);
        n++;
      });
    });
    expect(n).toBe(14);
  });

  it("writes numbers as python floats and bigints as python ints", () => {
    const cases: [PyJson, string][] = [
      [1, "1.0"], [1n, "1"], [-0, "-0.0"], [0, "0.0"], [0.1, "0.1"], [1e-5, "1e-05"], [1e16, "1e+16"],
      [1e15 + 0.3, "1000000000000000.2"], [123456789.123, "123456789.123"], [5e-324, "5e-324"],
      [1.7976931348623157e308, "1.7976931348623157e+308"], [2n ** 63n, "9223372036854775808"], [-7n, "-7"],
      [true, "true"], [false, "false"], [null, "null"], [[], "[]"], [{}, "{}"],
    ];
    for (const [v, out] of cases) expect(pyJsonDumps(v), out).toBe(out);
  });

  it("sorts keys by code point like python, not by UTF-16 unit", () => {
    const keys = [String.fromCodePoint(0x1f600), ch(0xffff), "a", ch(0xe000), ch(0xd800)];
    // Python: sorted(keys) puts U+D800 < U+E000 < U+FFFF < U+1F600
    expect([...keys].sort(comparePyStr)).toEqual(["a", ch(0xd800), ch(0xe000), ch(0xffff), String.fromCodePoint(0x1f600)]);
    expect([...keys].sort()).not.toEqual([...keys].sort(comparePyStr));
    expect(comparePyStr("ab", "abc")).toBeLessThan(0);
    expect(comparePyStr("abc", "abc")).toBe(0);
  });

  it("escapes strings like ensure_ascii, and only quotes and controls without it", () => {
    const v = { z: ch(0x2028, 0x7f, 0xd800) + '"\\/' + ch(8, 12, 10, 13, 9, 1), k: [1n, 2.5, {}] };
    expect(pyJsonDumps(v, { sortKeys: true })).toBe(
      `{"k": [1, 2.5, {}], "z": "${esc("2028")}${esc("007f")}${esc("d800")}\\"\\\\/\\b\\f\\n\\r\\t${esc("0001")}"}`);
    // json.dumps(v, sort_keys=True, ensure_ascii=False): U+2028, 0x7f and the lone surrogate pass through
    expect(pyJsonDumps(v, { sortKeys: true, ensureAscii: false })).toBe(
      `{"k": [1, 2.5, {}], "z": "${ch(0x2028, 0x7f, 0xd800)}\\"\\\\/\\b\\f\\n\\r\\t${esc("0001")}"}`);
  });

  it("writes bare NaN and infinities unless allow_nan is off", () => {
    expect(pyJsonDumps({ a: NaN, b: Infinity, c: -Infinity })).toBe('{"a": NaN, "b": Infinity, "c": -Infinity}');
    expect(() => pyJsonDumps([NaN], { allowNan: false })).toThrow(RangeError);
  });

  it("refuses values that are not JSON and circular references", () => {
    const loop: Record<string, unknown> = {};
    loop["self"] = loop;
    expect(() => pyJsonDumps(loop)).toThrow(/Circular reference/);
    expect(() => pyJsonDumps({ a: undefined })).toThrow(/not JSON serializable/);
    expect(() => pyJsonDumps(new Map())).toThrow(/Object of type Map/);
    // the same object twice is not a cycle
    const shared = { x: 1n };
    expect(pyJsonDumps([shared, shared], { separators: [",", ":"] })).toBe('[{"x":1},{"x":1}]');
  });
});

// ---------- Payload rules ----------

describe("audit payload rules", () => {
  it("refuses the payloads python refuses with the same message", () => {
    for (const r of FX.rejects) {
      const chain = new AuditChain();
      if (r.name === "bad_type") {
        // Python's payload was {"v": (1, object)}; the nearest JS value is a function in the list
        expect(r.error).toMatch(/^type \w+ not allowed in audit payloads$/);
        expect(() => chain.append(r.kind, { v: [1n, Object as unknown as PyJson] })).toThrow(
          new AuditPayloadError("type function not allowed in audit payloads"));
        continue;
      }
      expect(() => chain.append(r.kind, r.payload as AuditPayload), r.name).toThrow(new AuditPayloadError(r.error));
      expect(chain.length).toBe(0);
      expect(chain.lastHash).toBe(GENESIS);
    }
    expect(FX.rejects.map((r) => r.name)).toEqual(["long_string", "long_list", "too_deep", "bad_type", "long_kind"]);
  });

  it("refuses NaN and infinities", () => {
    for (const v of [NaN, Infinity, -Infinity]) {
      expect(() => checkAuditValue({ x: [v] })).toThrow(new AuditPayloadError("NaN/inf not allowed; use null"));
    }
  });

  it("counts string length in code points like python's len", () => {
    const emoji = String.fromCodePoint(0x1f600);
    expect(pyLen(emoji.repeat(200))).toBe(200);
    expect(pyLen(ch(0xd800) + "a" + ch(0xdc00))).toBe(3);
    expect(() => checkAuditValue({ s: emoji.repeat(200) })).not.toThrow();
    expect(() => checkAuditValue({ s: emoji.repeat(201) })).toThrow(AuditPayloadError);
    expect(() => checkAuditValue({ s: "x".repeat(200) })).not.toThrow();
  });

  it("accepts 64 list items and 4 levels of nesting but no more", () => {
    expect(() => checkAuditValue({ l: Array.from({ length: 64 }, (_, i) => BigInt(i)) })).not.toThrow();
    expect(() => checkAuditValue({ a: { b: { c: { d: 1 } } } })).not.toThrow();
    expect(() => checkAuditValue({ a: { b: { c: { d: [] } } } })).not.toThrow();
    expect(() => checkAuditValue({ a: { b: { c: { d: [1] } } } })).toThrow("payload nested too deeply");
    // the depth test runs before the type test, as in Python
    expect(() => checkAuditValue({ a: { b: { c: { d: { e: undefined } } } } })).toThrow("payload nested too deeply");
  });

  it("refuses JS values that have no python counterpart", () => {
    for (const [v, name] of [[undefined, "undefined"], [() => 1, "function"], [new Map(), "Map"], [new Date(0), "Date"],
      [new Float64Array(2), "Float64Array"], [Symbol("s"), "symbol"]] as const) {
      expect(() => checkAuditValue({ v }), name).toThrow(new AuditPayloadError(`type ${name} not allowed in audit payloads`));
    }
    expect(() => checkAuditValue(Object.create(null) as unknown)).not.toThrow();
  });

  it("refuses a non-finite clock", () => {
    expect(() => new AuditChain().append("tick", {}, NaN)).toThrow(RangeError);
  });
});

// ---------- Chain ----------

describe("audit chain against python", () => {
  it("recomputes every golden hash from prev and the hashed text", () => {
    let n = 0;
    expect(FX.genesis).toBe(GENESIS);
    for (const c of FX.cases) {
      let prev = GENESIS;
      for (const e of c.entries) {
        expect(e.prev).toBe(prev);
        expect(sha256Hex(e.prev + e.blob)).toBe(e.hash);
        const t = typedEntry(e, e.floats);
        expect(auditDigest(e.prev, { seq: BigInt(t.seq), ts: t.ts, kind: t.kind, payload: t.payload })).toBe(e.hash);
        prev = e.hash;
        n++;
      }
    }
    expect(n).toBe(14);
  });

  it("verifies every golden chain with python's result", () => {
    for (const c of FX.cases) {
      expect(c.verify).toEqual([true, c.entries.length]);
      expect(verifyChain(c.entries.map((e) => typedEntry(e, e.floats))), c.name).toEqual(c.verify);
    }
  });

  it("reproduces identical entries and hashes when the same payloads are appended at the same clock", () => {
    FX.cases.forEach((c, ci) => {
      const fromClock = new AuditChain("golden");
      const fromTs = new AuditChain("golden");
      c.entries.forEach((e, ei) => {
        const payload = get(pyCaseEntry(ci, ei), "payload") as AuditPayload;
        const got = fromClock.append(e.kind, payload, e.clock);
        expect(got.seq).toBe(e.seq);
        expect(Object.is(got.ts, e.ts), `${c.name}[${ei}] ts ${got.ts} vs ${e.ts}`).toBe(true);
        expect(got.kind).toBe(e.kind);
        expect(got.payload).toEqual(payload);
        expect(got.prev).toBe(e.prev);
        expect(got.hash, `${c.name}[${ei}]`).toBe(e.hash);
        // round(round(t, 3), 3) == round(t, 3): appending the logged ts gives the same chain
        expect(fromTs.append(e.kind, typedEntry(e, e.floats).payload, e.ts).hash).toBe(e.hash);
      });
      expect(fromClock.lastHash).toBe(c.entries[c.entries.length - 1]?.hash);
      expect(fromClock.verify()).toEqual(c.verify);
    });
  });

  it("detects tampering at the same entry as python", () => {
    const seqFloats = new Map((FX.cases[0]?.entries ?? []).map((e) => [e.seq, e.floats]));
    FX.tampered.forEach((t, ti) => {
      // Python's _verify_rows: the rows go through json.dumps(sort_keys=True) into a file, then AuditLog.verify()
      const rows = get(PY, "tampered", ti, "entries") as readonly PyJson[];
      expect(verifyJsonl(pyFileText(rows)), t.name).toEqual(t.verify);
      // the same rows as typed entries (seq keeps its float pointers from the untouched chain)
      const typed = t.entries.map((e) => typedEntry(e, seqFloats.get(e.seq) ?? []));
      expect(verifyChain(typed), t.name).toEqual(t.verify);
    });
    expect(FX.tampered.map((t) => t.verify[0])).toEqual([false, false, false, false, true]);
  });

  it("writes the same audit file as python and reads it back", () => {
    // sha256 of "".join(json.dumps(entry, sort_keys=True) + "\n" ...) in Python, per golden chain
    const fileSha: Record<string, string> = {
      basic: "c7e8ece414ae567e9f5d3762ea3a7e2d55cbfce570870a79988aa2cbecc60a10",
      tricky_floats: "b5b4c868a111c16d8fe0e71319c0dd26d8991ce578f085578004f051f36affad",
      unicode_and_escapes: "e4fa019a2f20a4127abafc18e8e3b01ec4b70ae9ae4139688adfe428af21bb7d",
      ts_rounding: "a505f6a30b6b587bc406b2df5b20e81a42e92a082a23353fc25cee01d698f149",
    };
    FX.cases.forEach((c, ci) => {
      const chain = new AuditChain();
      c.entries.forEach((e, ei) => chain.append(e.kind, get(pyCaseEntry(ci, ei), "payload") as AuditPayload, e.clock));
      const text = chain.toJsonl();
      expect(sha256Hex(text), c.name).toBe(fileSha[c.name]);
      expect(verifyJsonl(text)).toEqual(c.verify);
      expect(verifyJsonl(text.replace(/\n/g, "\r\n") + "\n  \n")).toEqual(c.verify);
    });
    const first = FX.cases[0]?.entries[2];
    if (!first) throw new Error("fixture has no basic[2]");
    expect(entryToJsonLine(typedEntry(first, first.floats))).toBe(
      '{"hash": "58435a5971dc80ed5a464273cbc00f724d6ecd29f8f80128f5c6388f438c0d58", "kind": "prediction", '
      + '"payload": {"confidence": 0.8123, "label": "Focused", "layers": {"Ambient Physics": 0.4211, '
      + '"Facial Surface": -0.0532}, "mask_override": false, "person": "Person #1", "simulated_fields": [], '
      + '"source": "replay"}, "prev": "24aa369a519162ba3a6b1a209c19077f97eba70f75f3a4ac82ff2b3a256b5b50", '
      + '"seq": 2, "ts": 1792850405.0}');
  });

  it("catches an int rewritten as a float in an audit file, as python does", () => {
    const chain = new AuditChain();
    FX.cases[0]?.entries.forEach((e, ei) => chain.append(e.kind, get(pyCaseEntry(0, ei), "payload") as AuditPayload, e.clock));
    const text = chain.toJsonl();
    expect(verifyJsonl(text)).toEqual([true, 4]);
    expect(text).toContain('"minutes": 10,');
    expect(verifyJsonl(text.replace('"minutes": 10,', '"minutes": 10.0,'))).toEqual([false, 4]);
    expect(verifyJsonl(text.replace('"seq": 1,', '"seq": 1.0,'))).toEqual([false, 2]);
    expect(verifyJsonl(text.replace('"speed": 1.0', '"speed": 1'))).toEqual([false, 2]);
    // a missing key raises, as Python's KeyError does
    expect(() => verifyJsonl(text.replace('"kind": "operator", ', ""))).toThrow(/KeyError/);
  });

  it("snapshots and freezes the payload", () => {
    const chain = new AuditChain();
    const payload = { person: "Person #1", layers: { "Facial Surface": 0.5 }, simulated_fields: ["lux"] };
    const entry = chain.append("prediction", payload, 1792850400);
    payload.layers["Facial Surface"] = 9;
    payload.simulated_fields.push("db");
    expect(entry.payload).toEqual({ person: "Person #1", layers: { "Facial Surface": 0.5 }, simulated_fields: ["lux"] });
    expect(chain.verify()).toEqual([true, 1]);
    expect(Object.isFrozen(entry)).toBe(true);
    expect(Object.isFrozen(entry.payload["layers"])).toBe(true);
    expect(() => {
      (entry.payload as Record<string, unknown>)["person"] = "x";
    }).toThrow(TypeError);
    // an edited copy breaks the chain at that entry
    const copy = { ...entry, payload: { ...entry.payload, person: "Person #2" } };
    expect(chain.verify([copy])).toEqual([false, 1]);
  });

  it("filters entries by seq and keeps the counters on close like python", () => {
    const chain = new AuditChain();
    for (let i = 0; i < 5; i++) chain.append("tick", { i: BigInt(i) }, 1792850400 + i);
    expect(chain.entries(3).map((e) => e.seq)).toEqual([3, 4]);
    expect(chain.entries().length).toBe(5);
    const head = chain.lastHash;
    chain.close();
    expect(chain.entries()).toEqual([]);
    expect(chain.verify()).toEqual([true, 0]);
    expect(chain.lastHash).toBe(head);
    const next = chain.append("tick", { i: 5n }, 1792850405);
    expect(next.seq).toBe(5);
    expect(next.prev).toBe(head);
    // as in Python, a chain resumed after close() no longer starts at GENESIS
    expect(chain.verify()).toEqual([false, 1]);
  });
});

// ---------- Runtime chains ----------

describe("runtime audit chains", () => {
  it("re-hashes and rebuilds the runtime's prediction entries", () => {
    let n = 0;
    for (const run of RT.runs) {
      expect(run.audit.length, run.name).toBeGreaterThan(0);
      const chain = new AuditChain(`replay-${run.name}-golden`);
      for (const e of run.audit) {
        const t = typedEntry(e, e.floats);
        expect(auditBlob({ seq: BigInt(t.seq), ts: t.ts, kind: t.kind, payload: t.payload })).toBe(e.blob);
        const got = chain.append(e.kind, t.payload, e.clock);
        expect(Object.is(got.ts, e.ts)).toBe(true);
        expect(got.hash, `${run.name} seq ${e.seq}`).toBe(e.hash);
        n++;
      }
      expect(chain.verify()).toEqual([true, run.audit.length]);
      expect(verifyJsonl(chain.toJsonl())).toEqual([true, run.audit.length]);
    }
    expect(n).toBe(177);
  });
});
