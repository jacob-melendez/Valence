/**
 * Light estimate, head boxes and luminance: ports of tests/perception/test_lux.py and test_regions.py (black_out and
 * LuxModel.fit are not ported), plus parity with valence/perception/lux.py, regions.py and cv2 BGR2GRAY over the
 * lux-*.json fixtures (perception-fixtures/gen_lux.py runs the real Python on the same inputs).
 *
 * Floats are compared with expectClose at 1e-9. Nothing here needs a looser tolerance: OpenCV's float32 filter2D
 * responses are small integers (exact in float32) and every float64 sum in lux.py is a sum of integers, so all
 * background statistics agree to the bit; only Math.exp (V8) vs math.exp (glibc) may differ in the last ulp. The
 * fma / norm cases are compared with toBe on purpose: they pin the bit-exact BLAS behaviour headBoxPx mirrors.
 */
import { BgStats, backgroundStats, LuxEstimator, LuxModel, MAX_LUX, MAX_WIDTH } from "@/perception/lux";
import { grayPixel, rgbaToGray } from "@/perception/luma";
import { fma, headBoxPx, normBbox } from "@/perception/regions";
import { expectClose, loadFixture } from "./perception-fixture-util";

const H = 360, W = 640;

const decode = (b64: string): Uint8Array => new Uint8Array(Buffer.from(b64, "base64"));

/** Seeded uniform [0, 1) (mulberry32). */
function uniform(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** clip(rint(base + N(0, sigma)), 0, 255) as uint8, row-major h x w (test_lux.py's `noisy`, with a JS RNG). */
function noisy(base: number | ((r: number, c: number) => number), sigma: number, seed = 0, h = H, w = W): Uint8Array {
  const u = uniform(seed + 1);
  const out = new Uint8Array(h * w);
  for (let r = 0, i = 0; r < h; r++) {
    for (let c = 0; c < w; c++, i++) {
      const z = Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
      const b = typeof base === "number" ? base : base(r, c);
      out[i] = Math.min(255, Math.max(0, Math.round(b + sigma * z)));
    }
  }
  return out;
}

/** img[r0:r1, c0:c1] = v (a constant or a row-major (r1 - r0) x (c1 - c0) patch). */
function paste(
  img: Uint8Array, w: number, r0: number, r1: number, c0: number, c1: number, v: number | Uint8Array,
): void {
  for (let r = r0; r < r1; r++) {
    for (let c = c0; c < c1; c++) img[r * w + c] = typeof v === "number" ? v : v[(r - r0) * (c1 - c0) + (c - c0)]!;
  }
}

/** pytest.approx: |actual - expected| <= max(rel x |expected|, abs). */
function approx(actual: number, expected: number, rel = 1e-6, abs = 1e-12): void {
  if (!(Math.abs(actual - expected) <= Math.max(rel * Math.abs(expected), abs))) {
    expect.fail(`expected ${actual} to be approximately ${expected} (rel ${rel}, abs ${abs})`);
  }
}

const stats = (y = 128.0, bg = 0.8, clipped = 0.1) => new BgStats(y, 3.0, bg, clipped);

describe("backgroundStats (test_lux.py)", () => {
  it.each([2.0, 5.0, 10.0, 20.0])("recovers sigma %s on a flat frame", (sigma) => {
    const s = backgroundStats(noisy(128.0, sigma), W, H, []);
    approx(s.sigma, sigma, 0.2);
    approx(s.yBg, 128.0, 0, 1.0);
    expect(s.bgFrac).toBe(1.0);
    expect(s.clippedFrac).toBeLessThan(0.01);
  });

  it.each([2.0, 8.0, 20.0])("ignores a smooth gradient (sigma %s)", (sigma) => {
    const s = backgroundStats(noisy((_r, c) => 60.0 + (130.0 * c) / (W - 1), sigma, 1), W, H, []);
    approx(s.sigma, sigma, 0.2);
  });

  it("measures small images at native resolution", () => {
    approx(backgroundStats(noisy(100.0, 6.0, 0, 120, 160), 160, 120, []).sigma, 6.0, 0.2);
  });

  it("masks person boxes", () => {
    const img = noisy(50.0, 3.0, 2);
    paste(img, W, 90, 270, 160, 480, noisy(220.0, 25.0, 3, 180, 320));
    const s = backgroundStats(img, W, H, [[160 / W, 90 / H, 480 / W, 270 / H]]);
    approx(s.yBg, 50.0, 0, 1.0);
    approx(s.sigma, 3.0, 0.2);
    approx(s.bgFrac, 0.75, 0, 0.02);
    const unmasked = backgroundStats(img, W, H, []);
    expect(unmasked.yBg).toBeGreaterThan(80.0);
    expect(unmasked.sigma).toBeGreaterThan(5.0);
  });

  it("counts clipping on the background only", () => {
    const img = new Uint8Array(H * W).fill(128);
    paste(img, W, 0, H, 0, W / 2, 0);
    approx(backgroundStats(img, W, H, []).clippedFrac, 0.5, 0, 0.01);
    paste(img, W, 0, H, W / 2, W, 255);
    approx(backgroundStats(img, W, H, []).clippedFrac, 1.0);
    const masked = new Uint8Array(H * W).fill(128);
    paste(masked, W, 0, H, 0, W / 4, 252);
    approx(backgroundStats(masked, W, H, [[0.0, 0.0, 0.25, 1.0]]).clippedFrac, 0.0);
  });

  it("handles all-person and degenerate boxes", () => {
    const img = noisy(100.0, 5.0);
    const full = backgroundStats(img, W, H, [[0.0, 0.0, 1.0, 1.0]]);
    expect(full.bgFrac).toBe(0.0);
    expect(full.yBg).toBeNaN();
    expect(full.valid).toBe(false);
    approx(backgroundStats(img, W, H, [[0.5, 0.5, 0.0, 0.0]]).bgFrac, 0.75, 0, 0.01);
    expect(backgroundStats(img, W, H, [[NaN, 0.0, 1.0, 1.0]]).bgFrac).toBe(1.0);
  });

  it("rejects a buffer that is not width x height", () => {
    expect(() => backgroundStats(new Uint8Array(99), 10, 10)).toThrow();
    expect(() => backgroundStats(new Uint8Array(300), 10, 10)).toThrow();
    expect(() => backgroundStats(new Uint8Array(100), 10.5, 10)).toThrow();
  });

  it.each([0, 255])("does not dilute sigma with a hard-clipped background (%s)", (level) => {
    const img = noisy(128.0, 5.0, 5);
    paste(img, W, 0, H, 0, W / 2, level);
    const s = backgroundStats(img, W, H, []);
    approx(s.clippedFrac, 0.5, 0, 0.01);
    approx(s.sigma, 5.0, 0.2);
  });

  it("reports sigma 0, not NaN, on a fully clipped background", () => {
    const s = backgroundStats(new Uint8Array(H * W), W, H, []);
    approx(s.sigma, 0.0);
    expect(s.clippedFrac).toBe(1.0);
  });

  it("decimates wide frames to MAX_WIDTH columns", () => {
    expect(MAX_WIDTH).toBe(320);
    const img = noisy(120.0, 6.0, 9, 720, 1280);
    approx(backgroundStats(img, 1280, 720, [[0.1, 0.2, 0.4, 0.9]]).sigma, 6.0, 0.2);
  });
});

describe("LuxModel (test_lux.py)", () => {
  it("maps the default model roughly", () => {
    const m = new LuxModel();
    expect(m.calibrated).toBe(false);
    expect(m.predict(30.0, 5.0)).toBeGreaterThanOrEqual(35);
    expect(m.predict(30.0, 5.0)).toBeLessThanOrEqual(75);
    expect(m.predict(128.0, 5.0)).toBeGreaterThanOrEqual(200);
    expect(m.predict(128.0, 5.0)).toBeLessThanOrEqual(400);
    expect(m.predict(220.0, 5.0)).toBeGreaterThanOrEqual(600);
    expect(m.predict(220.0, 5.0)).toBeLessThanOrEqual(1100);
  });

  it("handles NaN and overflow", () => {
    expect(new LuxModel().predict(NaN, 1.0)).toBeNaN();
    expect(new LuxModel(undefined, undefined, 0.1).predict(100.0, NaN)).toBeNaN();
    expect(new LuxModel(undefined, 10.0).predict(255.0, 0.0)).toBe(MAX_LUX);
    expect(new LuxModel(undefined, undefined, NaN).predict(100.0, 1.0)).toBeNaN(); // Python: NaN is truthy
  });

  it("round-trips through a dict", () => {
    const m = new LuxModel(1.0, 0.02, -0.05, true);
    expect(LuxModel.fromDict(m.toDict())).toEqual(m);
    expect(LuxModel.fromDict({})).toEqual(new LuxModel());
    expect(LuxModel.fromDict(new Map<string, unknown>([["b", "0.5"]])).b).toBe(0.5);
  });
});

describe("LuxEstimator (test_lux.py)", () => {
  it("is steady under constant light", () => {
    const model = new LuxModel(1.0, 0.03, 0.0, true);
    const est = new LuxEstimator(model);
    let out: [number, number] = [NaN, 0];
    for (let k = 0; k < 20; k++) out = est.update(stats(), 0.5 * k);
    approx(out[0], model.predict(128.0, 3.0));
    approx(out[1], 0.8 * 0.9);
  });

  it("halves quality while uncalibrated", () => {
    const [lux, q] = new LuxEstimator().update(stats(), 0.0);
    approx(lux, new LuxModel().predict(128.0, 3.0));
    approx(q, 0.5 * 0.8 * 0.9);
  });

  it("reports the window median and IQR stability", () => {
    const est = new LuxEstimator(new LuxModel(0.0, 1.0, 0.0, true), 10.0);
    let out: [number, number] = [NaN, 0];
    [5.0, 5.0, 5.0, 6.0, 7.0].forEach((y, k) => (out = est.update(stats(y, 1.0, 0.0), k)));
    const [q25, med, q75] = [Math.exp(5.0), Math.exp(5.0), Math.exp(6.0)]; // np.percentile of exp([5, 5, 5, 6, 7])
    approx(out[0], med);
    approx(out[1], 1.0 / (1.0 + (q75 - q25) / med));
    expect(out[1]).toBeLessThan(1.0);
    out = est.update(stats(7.0, 1.0, 0.0), 13.5);
    approx(out[0], Math.exp(7.0));
    approx(out[1], 1.0);
  });

  it("handles invalid stats", () => {
    const est = new LuxEstimator(new LuxModel(undefined, undefined, undefined, true));
    const [lux0, q0] = est.update(new BgStats(NaN, NaN, 0.0, 0.0), 0.0);
    expect(lux0).toBeNaN();
    expect(q0).toBe(0.0);
    est.update(stats(), 1.0);
    const [lux, q] = est.update(new BgStats(NaN, NaN, 0.0, 0.0), 2.0);
    expect(Number.isFinite(lux)).toBe(true);
    expect(q).toBe(0.0);
  });

  it("ignores a non-finite time", () => {
    const model = new LuxModel(1.0, 0.03, 0.0, true);
    const est = new LuxEstimator(model, 10.0);
    est.update(stats(100.0), NaN);
    est.update(stats(100.0), 0.0);
    for (let k = 1; k < 400; k++) est.update(stats(200.0), 0.5 * k);
    expect((est as unknown as { history: unknown[] }).history.length).toBeLessThanOrEqual(21);
    approx(est.update(stats(200.0), 200.0)[0], model.predict(200.0, 3.0));
  });

  it("forgets everything on reset", () => {
    const est = new LuxEstimator(new LuxModel(1.0, 0.03, 0.0, true));
    est.update(stats(), 0.0);
    est.reset();
    const [lux, q] = est.update(new BgStats(NaN, NaN, 0.0, 0.0), 1.0);
    expect(lux).toBeNaN();
    expect(q).toBe(0.0);
  });
});

const FW = 1280, FH = 720;
type Pt = readonly [number, number];
const FACE: Record<number, Pt> = { 0: [600, 260], 1: [620, 245], 2: [580, 245], 3: [640, 255], 4: [560, 255] };

function kpts(
  head: Record<number, Pt>, shoulders: readonly [Pt, Pt] | null = null, conf = 0.9,
): [Float64Array, Float64Array] {
  const xy = new Float64Array(34);
  const c = new Float64Array(17);
  const set = (i: number, p: Pt) => {
    xy[2 * i] = p[0];
    xy[2 * i + 1] = p[1];
    c[i] = conf;
  };
  for (const [i, p] of Object.entries(head)) set(Number(i), p);
  if (shoulders) {
    set(5, shoulders[0]);
    set(6, shoulders[1]);
  }
  return [xy, c];
}

describe("headBoxPx (test_regions.py)", () => {
  it("boxes the head spread", () => {
    const box = headBoxPx(...kpts(FACE), FW, FH);
    expect(box).not.toBeNull();
    const [x1, y1, x2, y2] = box!;
    expect(box!.every(Number.isInteger)).toBe(true);
    approx(x2 - x1, 1.8 * 80, 0, 2);
    approx(y2 - y1, 1.8 * 80, 0, 2);
    for (const [x, y] of Object.values(FACE)) expect(x1 <= x && x < x2 && y1 <= y && y < y2).toBe(true);
  });

  it("uses the shoulder width as a minimum size", () => {
    const box = headBoxPx(...kpts({ 0: [600, 260] }, [[700, 400], [500, 400]]), FW, FH)!;
    approx(box[2] - box[0], 1.8 * 0.6 * 200, 0, 2);
    approx((box[0] + box[2]) / 2, 600, 0, 1);
  });

  it("takes the larger of spread and shoulders", () => {
    const box = headBoxPx(...kpts(FACE, [[620, 400], [580, 400]]), FW, FH)!;
    approx(box[2] - box[0], 1.8 * 80, 0, 2);
  });

  it("falls back to 0.15 x frame width for a single point", () => {
    const box = headBoxPx(...kpts({ 3: [640, 300] }), FW, FH)!;
    approx(box[2] - box[0], 1.8 * 0.15 * FW, 0, 2);
  });

  it("honours scale", () => {
    const box = headBoxPx(...kpts(FACE), FW, FH, 1.0)!;
    approx(box[2] - box[0], 80, 0, 2);
  });

  it("is null when the head is not visible", () => {
    expect(headBoxPx(...kpts({}, [[700, 400], [500, 400]]), FW, FH)).toBeNull();
    expect(headBoxPx(...kpts(FACE, null, 0.29), FW, FH)).toBeNull();
    expect(headBoxPx(...kpts(FACE, null, 0.3), FW, FH)).not.toBeNull();
  });

  it("ignores low-confidence points", () => {
    const [xy, c] = kpts(FACE);
    xy[6] = 1200;
    xy[7] = 700;
    c[3] = 0.1;
    expect(headBoxPx(xy, c, FW, FH)![2]).toBeLessThan(1000);
  });

  it("clips to the frame", () => {
    const corner = headBoxPx(...kpts({ 0: [10, 10], 3: [40, 15], 4: [-20, 15] }), FW, FH)!;
    expect([corner[0], corner[1]]).toEqual([0, 0]);
    expect(corner[2] > 0 && corner[3] > 0).toBe(true);
    const edge = headBoxPx(...kpts({ 0: [1275, 715], 3: [1300, 720], 4: [1250, 720] }), FW, FH)!;
    expect([edge[2], edge[3]]).toEqual([FW, FH]);
  });

  it("is null entirely outside the frame", () => {
    expect(headBoxPx(...kpts({ 0: [-500, -500], 3: [-480, -500] }), FW, FH)).toBeNull();
  });

  it("covers ear to ear from the eyes alone", () => {
    const box = headBoxPx(...kpts({ 0: FACE[0]!, 1: FACE[1]!, 2: FACE[2]! }), FW, FH)!;
    const full = headBoxPx(...kpts(FACE), FW, FH)!;
    expect(box[2] - box[0]).toBeGreaterThanOrEqual(full[2] - full[0] - 2);
    expect(box[0] <= FACE[4]![0] && box[2] > FACE[3]![0]).toBe(true);
  });

  it("rejects keypoint arrays of the wrong shape", () => {
    expect(() => headBoxPx(new Float64Array(10), new Float64Array(5), FW, FH)).toThrow();
    expect(() => headBoxPx(new Float64Array(33), new Float64Array(17), FW, FH)).toThrow();
  });
});

describe("normBbox (test_regions.py)", () => {
  it("normalizes", () => {
    normBbox([128, 72, 640, 360], FW, FH).forEach((v, i) => approx(v, [0.1, 0.1, 0.5, 0.5][i]!));
  });

  it("clips to the unit range", () => {
    expect(normBbox([-10, -5, 1400, 800], FW, FH)).toEqual([0.0, 0.0, 1.0, 1.0]);
  });

  it("rejects a bad frame size", () => {
    expect(() => normBbox([0, 0, 1, 1], 0, 100)).toThrow();
  });
});

describe("grayPixel / rgbaToGray", () => {
  it("matches the corners of the colour cube", () => {
    expect(grayPixel(0, 0, 0)).toBe(0);
    expect(grayPixel(255, 255, 255)).toBe(255);
    expect([grayPixel(255, 0, 0), grayPixel(0, 255, 0), grayPixel(0, 0, 255)]).toEqual([76, 150, 29]);
    expect(grayPixel(0, 0, 250)).toBe(28); // the 14-bit formula says 29
  });

  it("rejects a buffer that is not width x height RGBA", () => {
    expect(() => rgbaToGray(new Uint8Array(15), 2, 2)).toThrow();
    expect(rgbaToGray(new Uint8ClampedArray([10, 200, 30, 0]), 1, 1)).toEqual(new Uint8Array([grayPixel(10, 200, 30)]));
  });
});

// ---------------------------------------------------------------------------------------------------- parity

type Expected<T> = T | { error: string };
const isError = (v: unknown): v is { error: string } => typeof v === "object" && v !== null && "error" in v;

interface StatsFixture {
  boxes: Record<string, number[][]>;
  images: {
    name: string;
    h: number;
    w: number;
    b64: string;
    grid?: number;
    runs: {
      boxes?: string;
      box_list?: number[][];
      max_width: number | null;
      expect: Expected<{ y_bg: number; sigma: number; bg_frac: number; clipped_frac: number; valid: boolean }>;
    }[];
  }[];
  n_cases: number;
}

/** The fixture image: raw bytes, or a grid of pixels at multiples of `grid` with (31 x + 17 y) % 256 elsewhere. */
function fixtureImage(im: StatsFixture["images"][number]): Uint8Array {
  const raw = decode(im.b64);
  if (im.grid === undefined) return raw;
  const g = im.grid, gw = Math.ceil(im.w / g);
  const img = new Uint8Array(im.w * im.h);
  for (let y = 0; y < im.h; y++) {
    for (let x = 0; x < im.w; x++) {
      img[y * im.w + x] = y % g === 0 && x % g === 0 ? raw[(y / g) * gw + x / g]! : (31 * x + 17 * y) % 256;
    }
  }
  return img;
}

describe("parity: background_stats (lux-stats.json)", () => {
  const fx = loadFixture<StatsFixture>("lux-stats");

  it("covers the fixture", () => {
    expect(fx.images.reduce((n, im) => n + im.runs.length, 0)).toBe(fx.n_cases);
    expect(fx.n_cases).toBeGreaterThan(400);
  });

  it.each(fx.images.map((im) => [im.name, im] as const))("%s", (_name, im) => {
    const gray = fixtureImage(im);
    expect(gray.length).toBe(im.w * im.h);
    for (const run of im.runs) {
      const boxes = run.boxes !== undefined ? fx.boxes[run.boxes]! : run.box_list!;
      const what = `${im.name} boxes=${run.boxes ?? JSON.stringify(run.box_list)} max_width=${run.max_width}`;
      const call = () => backgroundStats(gray, im.w, im.h, boxes, run.max_width ?? undefined);
      if (isError(run.expect)) {
        expect(call, what).toThrow();
        continue;
      }
      const s = call();
      expectClose(s.yBg, run.expect.y_bg, 1e-9, `${what} y_bg`);
      expectClose(s.sigma, run.expect.sigma, 1e-9, `${what} sigma`);
      expectClose(s.bgFrac, run.expect.bg_frac, 1e-9, `${what} bg_frac`);
      expectClose(s.clippedFrac, run.expect.clipped_frac, 1e-9, `${what} clipped_frac`);
      expect(s.valid, `${what} valid`).toBe(run.expect.valid);
    }
  });
});

interface ModelDict {
  a: number;
  b: number;
  c: number;
  calibrated: boolean;
}

interface EstimatorFixture {
  predict: { model: ModelDict; y: number; sigma: number; expect: number }[];
  from_dict: { d: Record<string, unknown>; expect: Expected<ModelDict> }[];
  sequences: {
    name: string;
    model: ModelDict;
    window_s: number;
    steps: ("reset" | { s: [number, number, number, number]; t: number; lux: number; q: number; n: number })[];
  }[];
}

const modelOf = (m: ModelDict) => new LuxModel(m.a, m.b, m.c, m.calibrated);

describe("parity: LuxModel / LuxEstimator (lux-estimator.json)", () => {
  const fx = loadFixture<EstimatorFixture>("lux-estimator");

  it("predict", () => {
    for (const { model, y, sigma, expect: want } of fx.predict) {
      expectClose(modelOf(model).predict(y, sigma), want, 1e-9, `predict ${JSON.stringify(model)} (${y}, ${sigma})`);
    }
  });

  it("fromDict", () => {
    for (const { d, expect: want } of fx.from_dict) {
      const what = `fromDict ${JSON.stringify(d)}`;
      if (isError(want)) {
        expect(() => LuxModel.fromDict(d), what).toThrow();
        continue;
      }
      const m = LuxModel.fromDict(d);
      expectClose(m.a, want.a, 1e-9, `${what} a`);
      expectClose(m.b, want.b, 1e-9, `${what} b`);
      expectClose(m.c, want.c, 1e-9, `${what} c`);
      expect(m.calibrated, `${what} calibrated`).toBe(want.calibrated);
    }
  });

  it.each(fx.sequences.map((s) => [s.name, s] as const))("sequence %s", (_name, seq) => {
    const est = new LuxEstimator(modelOf(seq.model), seq.window_s);
    seq.steps.forEach((step, i) => {
      if (step === "reset") {
        est.reset();
        return;
      }
      const [lux, q] = est.update(new BgStats(...step.s), step.t);
      expectClose(lux, step.lux, 1e-9, `${seq.name} step ${i} lux`);
      expectClose(q, step.q, 1e-9, `${seq.name} step ${i} quality`);
      expect((est as unknown as { history: unknown[] }).history.length, `${seq.name} step ${i} window`).toBe(step.n);
    });
  });
});

interface RegionsFixture {
  head_box: {
    name: string;
    xy: number[];
    conf: number[];
    frame: [number, number];
    scale: number;
    min_conf: number;
    expect: Expected<[number, number, number, number] | null>;
  }[];
  norm_bbox: { box: number[]; frame: [number, number]; expect: Expected<number[]> }[];
  norm2: [number, number, number][];
  fma: [number, number, number, number][];
}

describe("parity: regions (lux-regions.json)", () => {
  const fx = loadFixture<RegionsFixture>("lux-regions");

  it("head_box_px", () => {
    expect(fx.head_box.filter((c) => c.name.startsWith("fma_")).length).toBe(10);
    for (const c of fx.head_box) {
      const call = () => headBoxPx(Float64Array.from(c.xy), Float64Array.from(c.conf), ...c.frame, c.scale, c.min_conf);
      if (isError(c.expect)) expect(call, c.name).toThrow();
      else expect(call(), c.name).toEqual(c.expect);
    }
  });

  it("norm_bbox", () => {
    for (const c of fx.norm_bbox) {
      const what = `normBbox ${JSON.stringify(c.box)} ${c.frame}`;
      if (isError(c.expect)) {
        expect(() => normBbox(c.box, ...c.frame), what).toThrow();
        continue;
      }
      const out = normBbox(c.box, ...c.frame);
      c.expect.forEach((v, i) => expectClose(out[i]!, v, 1e-9, `${what} [${i}]`));
    }
  });

  it("np.linalg.norm of a 2-vector is sqrt(fma(dy, dy, dx * dx)) (bit-exact)", () => {
    for (const [dx, dy, want] of fx.norm2) expect(Math.sqrt(fma(dy, dy, dx * dx)), `norm(${dx}, ${dy})`).toBe(want);
  });

  it("fma is correctly rounded (bit-exact, signed zeros included)", () => {
    for (const [a, b, c, want] of fx.fma) expect(fma(a, b, c), `fma(${a}, ${b}, ${c})`).toBe(want);
  });
});

interface LumaFixture {
  cv2_version: string;
  n: number;
  rgb_b64: string;
  gray_b64: string;
  rgba: { h: number; w: number; rgba_b64: string; gray_b64: string };
}

describe("parity: cv2 BGR2GRAY (lux-luma.json)", () => {
  const fx = loadFixture<LumaFixture>("lux-luma");

  it("grayPixel on edge, random and 14-bit-disagreement colours", () => {
    const rgb = decode(fx.rgb_b64), gray = decode(fx.gray_b64);
    expect(gray.length).toBe(fx.n);
    let bad = 0;
    for (let i = 0; i < fx.n; i++) if (grayPixel(rgb[3 * i]!, rgb[3 * i + 1]!, rgb[3 * i + 2]!) !== gray[i]) bad++;
    expect(bad).toBe(0);
  });

  it("rgbaToGray on an RGBA image", () => {
    const { h, w, rgba_b64, gray_b64 } = fx.rgba;
    expect(rgbaToGray(decode(rgba_b64), w, h)).toEqual(decode(gray_b64));
  });
});
