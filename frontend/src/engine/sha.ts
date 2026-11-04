/**
 * Synchronous SHA-1 and SHA-256 over UTF-8 strings or bytes, hex encoded like Python's
 * `hashlib.sha1(...).hexdigest()` / `hashlib.sha256(...).hexdigest()`.
 *
 * WebCrypto's digest() is async (and absent on plain-http origins), but the engine needs hashes inside
 * synchronous code: the audit chain (valence/governance/audit.py), the template opener index
 * (valence/explain/template.py, sha1(event_id)) and the hello feature_version (sha256 of the model text).
 */

// ---------- UTF-8 ----------

/**
 * UTF-8 bytes of a string, as Python's `str.encode("utf-8")`. A lone surrogate (which Python would
 * refuse to encode) becomes U+FFFD, like TextEncoder. Implemented by hand so it does not depend on the
 * TextEncoder global or its realm.
 */
export function utf8Encode(s: string): Uint8Array {
  const out: number[] = [];
  for (let i = 0; i < s.length; i++) {
    let c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
      const d = s.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) {
        c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00);
        i++;
      }
    }
    if (c >= 0xd800 && c <= 0xdfff) c = 0xfffd; // unpaired surrogate
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 0x3f), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
  }
  return Uint8Array.from(out);
}

/** Lower-case hex of bytes. */
export function toHex(bytes: ArrayLike<number>): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += ((bytes[i] as number) & 0xff).toString(16).padStart(2, "0");
  return s;
}

const asBytes = (data: string | ArrayLike<number>): ArrayLike<number> =>
  typeof data === "string" ? utf8Encode(data) : data;

// ---------- Padding ----------

/** The message plus Merkle-Damgard padding (0x80, zeros, 64-bit big-endian bit length) as 32-bit words. */
function paddedWords(msg: ArrayLike<number>): Uint32Array {
  const n = msg.length;
  const total = (Math.floor((n + 8) / 64) + 1) * 64;
  const buf = new Uint8Array(total);
  for (let i = 0; i < n; i++) buf[i] = (msg[i] as number) & 0xff;
  buf[n] = 0x80;
  const bitsHi = Math.floor(n / 0x20000000); // (n * 8) >>> 32 without overflow
  const bitsLo = (n * 8) >>> 0;
  const view = new DataView(buf.buffer);
  view.setUint32(total - 8, bitsHi);
  view.setUint32(total - 4, bitsLo);
  const words = new Uint32Array(total / 4);
  for (let i = 0; i < words.length; i++) words[i] = view.getUint32(i * 4);
  return words;
}

function wordsToHex(h: ArrayLike<number>): string {
  let s = "";
  for (let i = 0; i < h.length; i++) s += ((h[i] as number) >>> 0).toString(16).padStart(8, "0");
  return s;
}

const rotl = (x: number, n: number): number => (x << n) | (x >>> (32 - n));
const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));

// ---------- SHA-1 ----------

/** Hex SHA-1 (FIPS 180-4) of a UTF-8 string or bytes: `hashlib.sha1(data).hexdigest()`. */
export function sha1Hex(data: string | ArrayLike<number>): string {
  const w = paddedWords(asBytes(data));
  const h = new Uint32Array([0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0]);
  const x = new Uint32Array(80);
  for (let off = 0; off < w.length; off += 16) {
    for (let t = 0; t < 16; t++) x[t] = w[off + t] as number;
    for (let t = 16; t < 80; t++) {
      x[t] = rotl((x[t - 3] as number) ^ (x[t - 8] as number) ^ (x[t - 14] as number) ^ (x[t - 16] as number), 1);
    }
    let a = h[0] as number, b = h[1] as number, c = h[2] as number, d = h[3] as number, e = h[4] as number;
    for (let t = 0; t < 80; t++) {
      let f: number, k: number;
      if (t < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
      else if (t < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
      else if (t < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
      else { f = b ^ c ^ d; k = 0xca62c1d6; }
      const tmp = (rotl(a, 5) + f + e + k + (x[t] as number)) | 0;
      e = d; d = c; c = rotl(b, 30); b = a; a = tmp;
    }
    h[0] = (h[0] as number) + a; h[1] = (h[1] as number) + b; h[2] = (h[2] as number) + c;
    h[3] = (h[3] as number) + d; h[4] = (h[4] as number) + e;
  }
  return wordsToHex(h);
}

// ---------- SHA-256 ----------

const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** Hex SHA-256 (FIPS 180-4) of a UTF-8 string or bytes: `hashlib.sha256(data).hexdigest()`. */
export function sha256Hex(data: string | ArrayLike<number>): string {
  const w = paddedWords(asBytes(data));
  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const x = new Uint32Array(64);
  for (let off = 0; off < w.length; off += 16) {
    for (let t = 0; t < 16; t++) x[t] = w[off + t] as number;
    for (let t = 16; t < 64; t++) {
      const v15 = x[t - 15] as number, v2 = x[t - 2] as number;
      const s0 = rotr(v15, 7) ^ rotr(v15, 18) ^ (v15 >>> 3);
      const s1 = rotr(v2, 17) ^ rotr(v2, 19) ^ (v2 >>> 10);
      x[t] = (x[t - 16] as number) + s0 + (x[t - 7] as number) + s1;
    }
    let a = h[0] as number, b = h[1] as number, c = h[2] as number, d = h[3] as number;
    let e = h[4] as number, f = h[5] as number, g = h[6] as number, hh = h[7] as number;
    for (let t = 0; t < 64; t++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + (K256[t] as number) + (x[t] as number)) | 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) | 0;
      hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    h[0] = (h[0] as number) + a; h[1] = (h[1] as number) + b; h[2] = (h[2] as number) + c;
    h[3] = (h[3] as number) + d; h[4] = (h[4] as number) + e; h[5] = (h[5] as number) + f;
    h[6] = (h[6] as number) + g; h[7] = (h[7] as number) + hh;
  }
  return wordsToHex(h);
}
