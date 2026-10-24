/**
 * sha.ts against Python hashlib vectors (computed with .venv python) and node:crypto.
 */
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { sha1Hex, sha256Hex, toHex, utf8Encode } from "@/engine/sha";
import { artifactBytes, artifactText } from "./engine-fixtures";

// ---------- Vectors ----------

// [input, hashlib.sha1(input.encode()).hexdigest(), hashlib.sha256(input.encode()).hexdigest()]
const STRING_VECTORS: [string, string, string][] = [
  ["", "da39a3ee5e6b4b0d3255bfef95601890afd80709", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
  ["abc", "a9993e364706816aba3e25717850c26c9cd0d89d", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"],
  [
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
  ],
  [
    "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
    "a49b2446a02c645bf419f995b67091253a04a259",
    "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
  ],
  // an event id, as hashed by valence/explain/template.py for the opener index
  ["p1-000042", "11c136ebd66d359ad5acf6840626c9d463b8f5ef", "4198071edb362ed0a04be8c2a33c970319bfb4ab7f7684ac869d75311d6fb79d"],
  [
    "Valence é日本語 \u{1F600}",
    "d232850a6e0b46404d79423dc248c4021b5f24f2",
    "1de8e6b35056fa7ee2c998b0a1eed4dbc523ef144b877b636227acf52f973d04",
  ],
];

// Prefixes of bytes(range(256)) * 3 around the 55/56/64-byte padding edges: [n, sha1, sha256].
const BYTE_VECTORS: [number, string, string][] = [
  [55, "8ae2d46729cfe68ff927af5eec9c7d1b66d65ac2", "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"],
  [56, "636e2ec698dac903498e648bd2f3af641d3c88cb", "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"],
  [63, "6d942da0c4392b123528f2905c713a3ce28364bd", "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"],
  [64, "c6138d514ffa2135bfce0ed0b8fac65669917ec7", "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"],
  [65, "69bd728ad6e13cd76ff19751fde427b00e395746", "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"],
  [119, "41c89d06001bab4ab78736b44efe7ce18ce6ae08", "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"],
  [120, "d3dbd653bd8597b7475321b60a36891278e6a04a", "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"],
  [700, "b2a17646998e72ebfe5861519a383be63c36a49c", "513df58dd095240caa52ac490c29836736d4ef0133b40ac7b7e249abf7ecf2f7"],
];

const byteRamp = (n: number): Uint8Array => Uint8Array.from({ length: n }, (_, i) => i % 256);
const nodeHex = (algo: "sha1" | "sha256", data: string | Uint8Array): string => createHash(algo).update(data).digest("hex");

/** Deterministic pseudo-random strings mixing ASCII, Latin-1, CJK, astral characters and lone surrogates. */
function randomStrings(count: number): string[] {
  let seed = 12345;
  const next = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const pools = [[0x20, 0x7e], [0xa0, 0x2ff], [0x4e00, 0x9fff], [0x1f300, 0x1faff], [0xd800, 0xdfff]] as const;
  return Array.from({ length: count }, () => {
    let s = "";
    const len = Math.floor(next() * 150);
    for (let i = 0; i < len; i++) {
      const pool = pools[Math.floor(next() * pools.length)] ?? pools[0];
      s += String.fromCodePoint(pool[0] + Math.floor(next() * (pool[1] - pool[0])));
    }
    return s;
  });
}

// ---------- Tests ----------

describe("sha1Hex and sha256Hex", () => {
  it("match hashlib on the reference strings", () => {
    for (const [s, sha1, sha256] of STRING_VECTORS) {
      expect(sha1Hex(s), s).toBe(sha1);
      expect(sha256Hex(s), s).toBe(sha256);
    }
  });

  it("match hashlib on byte input around the padding block edges", () => {
    for (const [n, sha1, sha256] of BYTE_VECTORS) {
      expect(sha1Hex(byteRamp(n)), String(n)).toBe(sha1);
      expect(sha256Hex(byteRamp(n)), String(n)).toBe(sha256);
    }
  });

  it("match hashlib on a million bytes", () => {
    const s = "a".repeat(1_000_000);
    expect(sha1Hex(s)).toBe("34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    expect(sha256Hex(s)).toBe("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  });

  it("match node:crypto for every length from 0 to 300 bytes", () => {
    for (let n = 0; n <= 300; n++) {
      const b = Uint8Array.from({ length: n }, (_, i) => (i * 31 + n) % 256);
      expect(sha1Hex(b)).toBe(nodeHex("sha1", b));
      expect(sha256Hex(b)).toBe(nodeHex("sha256", b));
    }
  });

  it("hash strings as their UTF-8 bytes", () => {
    for (const s of randomStrings(200)) {
      expect(sha1Hex(s)).toBe(nodeHex("sha1", Buffer.from(s, "utf8")));
      expect(sha256Hex(s)).toBe(nodeHex("sha256", Buffer.from(s, "utf8")));
    }
  });

  it("give the served model id prefix from the m2 artifact", () => {
    // valence/models/served.py: "m2_lgbm@" + sha256(m2.txt bytes)[:8]; the text and the bytes agree (ASCII)
    expect(sha256Hex(artifactBytes("m2.txt")).slice(0, 8)).toBe(nodeHex("sha256", artifactBytes("m2.txt")).slice(0, 8));
    expect(sha256Hex(artifactText("m2.txt"))).toBe(sha256Hex(artifactBytes("m2.txt")));
  });
});

describe("utf8Encode", () => {
  it("matches Buffer's UTF-8, lone surrogates becoming U+FFFD", () => {
    for (const s of randomStrings(300)) expect(toHex(utf8Encode(s))).toBe(Buffer.from(s, "utf8").toString("hex"));
    expect(Array.from(utf8Encode("\ud800"))).toEqual([0xef, 0xbf, 0xbd]);
    expect(Array.from(utf8Encode("\u{1F600}"))).toEqual([0xf0, 0x9f, 0x98, 0x80]);
  });
});
