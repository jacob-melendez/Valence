import {
  LAUGH_FAMILY, LaughterTracker, POWER_FLOOR, PowerSmoother, YAMNET_ALLOWLIST, familyMax, meanPower, powerToDb,
  type YamnetScores,
} from "@/perception/audioFeatures";
import { HysteresisDebouncer } from "@/perception/debounce";
import { firwin, i0, kaiser, resample48kTo16k, resamplePoly, resampleTo16k } from "@/perception/resample";
import { expectClose, loadFixture } from "./perception-fixture-util";

// ---------------------------------------------------------------------------------------------- helpers

type Block =
  | { scale: number; ints: number[] } | { scale: number; i16: string } | { ints: number[] } | { floats: number[] }
  | { f32: string } | { floats64: number[] };
/** Exact float32 bits, base64 little-endian (resampled outputs). */
type F32 = { f32: string };

/** Base64 little-endian int16 / float32 (the fixtures' compact form of long blocks and exact outputs). */
function fromBase64(s: string, kind: "i16" | "f32"): Int16Array | Float32Array {
  const b = Buffer.from(s, "base64");
  const view = new DataView(b.buffer, b.byteOffset, b.length);
  if (kind === "i16") return Int16Array.from({ length: b.length / 2 }, (_, i) => view.getInt16(2 * i, true));
  return Float32Array.from({ length: b.length / 4 }, (_, i) => view.getFloat32(4 * i, true));
}

const INT_ARRAYS: Record<string, (v: number[]) => ArrayLike<number>> = {
  int8: (v) => Int8Array.from(v), uint8: (v) => Uint8Array.from(v), int16: (v) => Int16Array.from(v),
  uint16: (v) => Uint16Array.from(v), int32: (v) => Int32Array.from(v), uint32: (v) => Uint32Array.from(v),
};

/** A fixture block as the array the Python side saw: float32 / float64 data, or integer PCM of `dtype`. */
function block(data: Block, dtype = "float32"): ArrayLike<number> {
  if ("floats64" in data) return data.floats64; // float64 values: the float32 cast happens inside resamplePoly
  if (dtype === "float64") return "floats" in data ? data.floats : [];
  if (dtype !== "float32") {
    const make = INT_ARRAYS[dtype];
    if (make === undefined) throw new Error(`unknown dtype ${dtype}`);
    return make("ints" in data ? data.ints : []);
  }
  if ("floats" in data) return Float32Array.from(data.floats);
  if ("f32" in data) return fromBase64(data.f32, "f32");
  const scale = "scale" in data ? data.scale : 1;
  return Float32Array.from("i16" in data ? fromBase64(data.i16, "i16") : data.ints, (v) => v / scale);
}

function sine(freq = 1000.0, sr = 16000, seconds = 1.0, amp = 1.0): Float32Array {
  const n = Math.trunc(sr * seconds);
  return Float32Array.from({ length: n }, (_, i) => amp * Math.sin(2 * Math.PI * freq * (i / sr)));
}

/** np.rint: round half to even. */
function rint(v: number): number {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Python's legacy 0.30 x 2 rule: these cases cover LaughterTracker bookkeeping, not thresholds. */
const legacy = () => new HysteresisDebouncer(0.30, 0.15, 2, 0.0, 3.0);

// ---------------------------------------------------------------------------------------------- Python tests

describe("audio features (tests/perception/test_audio_features.py)", () => {
  it("full-scale sine is -3 dBFS", () => {
    const p = meanPower(sine());
    expect(Math.abs(p - 0.5)).toBeLessThanOrEqual(0.5e-4);
    expect(powerToDb(p)).toBeCloseTo(-3.01, 2);
    expect(powerToDb(p, 94.0)).toBeCloseTo(90.99, 2);
  });

  it("scales int16 PCM", () => {
    const pcm = Int16Array.from(sine(), (v) => rint(v * 32767));
    expect(Math.abs(powerToDb(meanPower(pcm)) - -3.01)).toBeLessThanOrEqual(0.02);
  });

  it("accumulates in float64 and handles empty blocks", () => {
    const x = new Float32Array(1_000_000).fill(1e-3);
    expect(Math.abs(meanPower(x) - 1e-6)).toBeLessThanOrEqual(1e-12);
    expect(meanPower(new Float32Array(0))).toBeNaN();
  });

  it("floors power_to_db and passes NaN", () => {
    expect(powerToDb(0.0)).toBeCloseTo(-120.0, 9);
    expect(powerToDb(0.0, 100.0)).toBeCloseTo(-20.0, 9);
    expect(powerToDb(NaN)).toBeNaN();
    expect(POWER_FLOOR).toBe(1e-12);
  });

  it("smooths power, not dB", () => {
    const sm = new PowerSmoother(0.3);
    sm.update(1e-2);
    sm.update(1e-6);
    const expectedPower = 0.3 * 1e-6 + 0.7 * 1e-2;
    expectClose(sm.value, expectedPower, 1e-12);
    expectClose(sm.db(), 10 * Math.log10(expectedPower), 1e-12);
    const dbAverage = 0.3 * -60.0 + 0.7 * -20.0;
    expect(Math.abs(sm.db() - dbAverage)).toBeGreaterThan(10.0);
  });

  it("takes the first sample, skips NaN and negative power, resets, rejects bad alpha", () => {
    const sm = new PowerSmoother();
    expect(sm.db()).toBeNaN();
    expect(sm.update(0.25)).toBe(0.25);
    expect(sm.update(NaN)).toBe(0.25);
    expect(sm.update(-1.0)).toBe(0.25);
    sm.reset();
    expect(sm.value).toBeNaN();
    expect(() => new PowerSmoother(0.0)).toThrow();
  });

  it("has the allowlist constants", () => {
    expect(YAMNET_ALLOWLIST[0]).toBe("Speech");
    expect([...LAUGH_FAMILY]).toEqual(["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]);
  });

  it("default tracker uses the measured laughter rule", () => {
    const tr = new LaughterTracker();
    expect(tr.update({ Speech: 0.98, Laughter: 0.0 }, 0.0)[2]).toBe(false);
    expect(tr.update({ Speech: 0.15, Laughter: 0.08, Snicker: 0.06 }, 1.0)[2]).toBe(true);
    expect(tr.takeOnsets()).toBe(1);
  });

  it("default tracker counts a laugh with a silent second once", () => {
    const tr = new LaughterTracker();
    const seq: YamnetScores[] = [{ Speech: 0.41, Laughter: 0.04 }, { Speech: 0.11, Laughter: 0.08 },
      { Speech: 0.94, Laughter: 0.0 }, { Speech: 0.15, Laughter: 0.08, Snicker: 0.06 }];
    const onsets: boolean[] = [];
    seq.forEach((s, t) => {
      onsets.push(tr.update(s, t)[2]);
      if (t >= 1) expect(tr.active, String(t)).toBe(true); // the instant-exit rule dropped out at t=2
    });
    expect(onsets).toEqual([false, true, false, false]);
    expect(tr.takeOnsets()).toBe(1);
  });

  it("default tracker counts a laugh with two silent seconds once", () => {
    const seq: YamnetScores[] = [{ Laughter: 0.08 }, { Laughter: 0.0 }, { Laughter: 0.0 }, { Laughter: 0.04 },
      { Laughter: 0.08 }];
    const tr = new LaughterTracker();
    expect(seq.map((s, t) => tr.update(s, t)[2])).toEqual([true, false, false, false, false]);
    expect(tr.takeOnsets()).toBe(1);
    const old = new LaughterTracker(new HysteresisDebouncer(0.05, 0.02, 1, 0.0, 4.0));
    expect(seq.filter((s, t) => old.update(s, t)[2]).length).toBe(2);
  });

  it("laughter onset semantics", () => {
    const tr = new LaughterTracker(legacy());
    expect(tr.update({ Laughter: 0.35, Speech: 0.2 }, 0.0)).toEqual([0.35, 0.2, false]);
    const [laugh, , onset] = tr.update({ Laughter: 0.1, Giggle: 0.4 }, 0.5);
    expect([laugh, onset]).toEqual([0.4, true]);
    expect(tr.active).toBe(true);
    expect(tr.takeOnsets()).toBe(1);
    expect(tr.takeOnsets()).toBe(0);
    tr.update({ Laughter: 0.05 }, 1.0);
    expect(tr.active).toBe(false);
    tr.update({ Snicker: 0.5 }, 1.5);
    expect(tr.update({ Snicker: 0.5 }, 2.0)[2]).toBe(false);
    expect(tr.takeOnsets()).toBe(0);
    tr.update({ Laughter: 0.0 }, 2.5);
    tr.update({ "Belly laugh": 0.9 }, 3.5);
    expect(tr.update({ "Chuckle, chortle": 0.9 }, 4.0)[2]).toBe(true);
    expect(tr.takeOnsets()).toBe(1);
  });

  it("never counts speech alone as laughter", () => {
    const tr = new LaughterTracker(legacy());
    for (let k = 0; k < 10; k++) {
      const [laugh, speech, onset] = tr.update({ Speech: 0.95 }, 0.5 * k);
      expect([laugh, speech, onset]).toEqual([0.0, 0.95, false]);
    }
    expect(tr.takeOnsets()).toBe(0);
  });

  it("accumulates onsets between takes", () => {
    const tr = new LaughterTracker(legacy());
    let t = 0.0;
    for (let i = 0; i < 3; i++) {
      tr.update({ Laughter: 0.6 }, t);
      tr.update({ Laughter: 0.6 }, t + 0.5);
      tr.update({ Laughter: 0.0 }, t + 1.0);
      t += 4.0;
    }
    expect(tr.takeOnsets()).toBe(3);
  });

  it("resamples 48 kHz to 16 kHz: length, dtype and content", () => {
    const y = resample48kTo16k(sine(1000.0, 48000));
    expect(y).toBeInstanceOf(Float32Array);
    expect(y.length).toBe(16000);
    expect(Math.abs(powerToDb(meanPower(y.subarray(100, -100))) - -3.01)).toBeLessThanOrEqual(0.1);
    expect(resample48kTo16k(new Float32Array(4801)).length).toBe(1601);
    // Python's (4800, 1) case has no counterpart: the port is 1-D only.
  });

  it("removes content above the new Nyquist", () => {
    const y = resample48kTo16k(sine(12000.0, 48000));
    expect(powerToDb(meanPower(y.subarray(100, -100)))).toBeLessThan(-40.0);
  });
});

describe("resample helpers", () => {
  it("picks up/down from the sample rate", () => {
    const x = sine(440.0, 44100);
    expect(resampleTo16k(x, 44100)).toEqual(resamplePoly(x, 160, 441));
    expect(resampleTo16k(x, 44100).length).toBe(16000);
    const x48 = sine(440.0, 48000);
    expect(resampleTo16k(x48, 48000)).toEqual(resample48kTo16k(x48));
    const x16 = sine(440.0, 16000);
    const copy = resampleTo16k(x16, 16000);
    expect(copy).toEqual(x16);
    expect(copy).not.toBe(x16);
    expect(() => resampleTo16k(x16, 44100.5)).toThrow();
    expect(() => resampleTo16k(x16, 0)).toThrow();
  });

  it("validates up and down, reduces by the gcd, and leaves the input alone", () => {
    expect(() => resamplePoly([1, 2, 3], 1.5, 3)).toThrow();
    expect(() => resamplePoly([1, 2, 3], 1, 0)).toThrow();
    const x = sine(300.0, 4800, 0.1);
    const before = x.slice();
    expect(resamplePoly(x, 2, 6)).toEqual(resamplePoly(x, 1, 3));
    expect(x).toEqual(before);
    expect(resamplePoly([0.1, 0.2], 7, 7)).toEqual(Float32Array.from([0.1, 0.2]));
  });

  it("i0 matches known values", () => {
    expect(i0(0)).toBe(1);
    expectClose(i0(1), 1.2660658777520082, 1e-15);
    expectClose(i0(-5), 27.239871823604442, 1e-15);
  });
});

// ---------------------------------------------------------------------------------------------- parity

interface MeanPowerCase { dtype: string; data: Block; p: number; db: number; db94: number }
interface SmootherStep { op: "update" | "reset"; p?: number; out?: number; value: number; db?: number; db94?: number }
interface LaughterStep {
  scores: Record<string, number>; t: number; laugh: number; speech: number; onset: boolean; active: boolean;
  take?: number;
}
interface DebouncerCfg {
  on: number; off: number; on_count: number; off_hold_s: number; refractory_s: number; on_hold_s: number;
}
interface FeaturesFixture {
  constants: { POWER_FLOOR: number; YAMNET_ALLOWLIST: string[]; LAUGH_FAMILY: string[] };
  mean_power: MeanPowerCase[];
  power_to_db: { p: number; offset: number; db: number }[];
  smoother: { init: { alpha: number; value?: number }; steps: SmootherStep[] }[];
  smoother_invalid: { alpha: number; raises: boolean }[];
  family_max: { scores: Record<string, number>; laugh: number }[];
  laughter: { debouncer: DebouncerCfg | null; steps: LaughterStep[]; final_take: number }[];
}
interface ResampleCase { up: number; down: number; x: Block; y: F32 }
interface ResampleFixture {
  i0: { x: number; y: number }[];
  kaiser: { M: number; beta: number; w: number[] }[];
  firwin: { numtaps: number; cutoff: number; h: number[] }[];
  cases: ResampleCase[];
  rates: (ResampleCase & { rate: number })[];
}

/**
 * resamplePoly emulates scipy's float32 arithmetic (Math.fround on every product and sum), so its output is
 * bit-identical; the 1e-6 tolerance is the float32 contract, the exact check guards the emulation.
 */
function expectResample(actual: Float32Array, y: F32, what: string): void {
  const expected = fromBase64(y.f32, "f32");
  expect(actual.length, `${what} length`).toBe(expected.length);
  let mismatches = 0;
  expected.forEach((v, i) => {
    expectClose(actual[i]!, v, 1e-6, `${what}[${i}]`);
    if (!Object.is(actual[i], v)) mismatches += 1;
  });
  expect(mismatches, `${what}: samples not bit-identical to scipy`).toBe(0);
}

describe("parity with valence/perception/audio_features.py", () => {
  const fx = loadFixture<FeaturesFixture>("audio-features");

  it("constants", () => {
    expect(POWER_FLOOR).toBe(fx.constants.POWER_FLOOR);
    expect([...YAMNET_ALLOWLIST]).toEqual(fx.constants.YAMNET_ALLOWLIST);
    expect([...LAUGH_FAMILY]).toEqual(fx.constants.LAUGH_FAMILY);
  });

  it(`mean_power (${fx.mean_power.length} blocks)`, () => {
    fx.mean_power.forEach((c, i) => {
      // float64 data alternates between number[] and Float64Array.
      const raw = block(c.data, c.dtype);
      const x = c.dtype === "float64" && i % 2 ? Float64Array.from(raw) : raw;
      const what = `mean_power[${i}] ${c.dtype} n=${x.length}`;
      const p = meanPower(x);
      expectClose(p, c.p, 1e-15, what);
      expect(p, `${what}: not bit-identical (numpy pairwise summation)`).toBe(c.p);
      expectClose(powerToDb(p), c.db, 1e-12, what);
      expectClose(powerToDb(p, 94.0), c.db94, 1e-12, what);
    });
  });

  it(`power_to_db (${fx.power_to_db.length} cases)`, () => {
    fx.power_to_db.forEach((c, i) => expectClose(powerToDb(c.p, c.offset), c.db, 1e-12, `power_to_db[${i}]`));
  });

  it(`PowerSmoother (${fx.smoother.length} sequences)`, () => {
    fx.smoother.forEach((c, i) => {
      const { alpha, value } = c.init;
      const sm = value === undefined ? new PowerSmoother(alpha) : new PowerSmoother(alpha, value);
      c.steps.forEach((s, k) => {
        const what = `smoother[${i}] step ${k}`;
        if (s.op === "reset") sm.reset();
        else {
          expect(sm.update(s.p!), what).toBe(s.out!); // same float64 operations: bit-identical
          // log10: V8's fdlibm and the C library can differ in the last bit
          expectClose(sm.db(), s.db!, 1e-12, what);
          expectClose(sm.db(94.0), s.db94!, 1e-12, what);
        }
        expect(sm.value, what).toBe(s.value);
      });
    });
    fx.smoother_invalid.forEach((c) => {
      if (c.raises) expect(() => new PowerSmoother(c.alpha), String(c.alpha)).toThrow();
      else expect(() => new PowerSmoother(c.alpha), String(c.alpha)).not.toThrow();
    });
  });

  it(`_family_max (${fx.family_max.length} cases)`, () => {
    fx.family_max.forEach((c, i) => {
      const laugh = familyMax(c.scores);
      expectClose(laugh, c.laugh, 0, `family_max[${i}]`);
      expect(Object.is(laugh, -0), `family_max[${i}] sign of zero`).toBe(Object.is(c.laugh, -0));
    });
  });

  it(`LaughterTracker (${fx.laughter.length} sequences)`, () => {
    fx.laughter.forEach((c, i) => {
      const d = c.debouncer;
      const tr = new LaughterTracker(
        d === null ? null : new HysteresisDebouncer(d.on, d.off, d.on_count, d.off_hold_s, d.refractory_s, d.on_hold_s),
      );
      c.steps.forEach((s, k) => {
        const what = `laughter[${i}] step ${k} t=${s.t}`;
        const [laugh, speech, onset] = tr.update(s.scores, s.t);
        expect(laugh, what).toBe(s.laugh); // Object.is: NaN and the sign of zero too
        expect(speech, what).toBe(s.speech);
        expect(onset, what).toBe(s.onset);
        expect(tr.active, what).toBe(s.active);
        if (s.take !== undefined) expect(tr.takeOnsets(), what).toBe(s.take);
      });
      expect(tr.takeOnsets(), `laughter[${i}] final take`).toBe(c.final_take);
    });
  });
});

describe("parity with scipy.signal.resample_poly", () => {
  const fx = loadFixture<ResampleFixture>("audio-resample");

  it(`i0 (${fx.i0.length} points)`, () => {
    // Cephes (scipy) is within ~1e-15 of the correctly rounded value this double-double series returns.
    fx.i0.forEach((c) => expectClose(i0(c.x), c.y, 1e-14, `i0(${c.x})`));
  });

  it(`kaiser (${fx.kaiser.length} windows)`, () => {
    fx.kaiser.forEach((c) => {
      const w = kaiser(c.M, c.beta);
      expect(w.length).toBe(c.w.length);
      c.w.forEach((v, i) => expectClose(w[i]!, v, 1e-14, `kaiser(${c.M}, ${c.beta})[${i}]`));
    });
  });

  it(`firwin (${fx.firwin.length} filters)`, () => {
    // Taps agree to a few float64 ulps (numpy's SIMD sin vs V8's fdlibm); none of them changes its float32 value.
    fx.firwin.forEach((c) => {
      const h = firwin(c.numtaps, c.cutoff);
      expect(h.length).toBe(c.h.length);
      c.h.forEach((v, i) => {
        expectClose(h[i]!, v, 1e-13, `firwin(${c.numtaps}, ${c.cutoff})[${i}]`);
        expect(Math.fround(h[i]!)).toBe(Math.fround(v));
      });
    });
  });

  it(`short blocks (${fx.cases.length} cases)`, () => {
    fx.cases.forEach((c, i) => {
      const x = block(c.x);
      expectResample(resamplePoly(x, c.up, c.down), c.y, `case ${i} (${c.up}/${c.down}, n=${x.length})`);
    });
  });

  it(`short blocks at other rates (${fx.rates.map((c) => c.rate).join(", ")} Hz)`, () => {
    fx.rates.forEach((c) => {
      const x = block(c.x);
      expectResample(resampleTo16k(x, c.rate), c.y, `${c.rate} Hz via resampleTo16k (n=${x.length})`);
      expectResample(resamplePoly(x, c.up, c.down), c.y, `${c.rate} Hz (${c.up}/${c.down})`);
    });
  });

  for (const rate of [48000, 44100, 32000, 22050, 16000]) {
    it(`1 s block at ${rate} Hz`, () => {
      const c = loadFixture<ResampleCase & { rate: number }>(`audio-resample-${rate}`);
      const x = block(c.x);
      expect(x.length).toBe(rate);
      expectResample(resampleTo16k(x, rate), c.y, `${rate} Hz via resampleTo16k`);
      expectResample(resamplePoly(x, c.up, c.down), c.y, `${rate} Hz (${c.up}/${c.down})`);
      if (rate === 48000) expectResample(resample48kTo16k(x), c.y, "resample48kTo16k");
    });
  }
});
