/**
 * Posture tracker: port of tests/perception/test_posture.py, plus parity with valence/perception/posture.py over the
 * posture-*.json fixtures (long synthetic seated-person sequences replayed through the real Python tracker).
 */
import { readdirSync } from "node:fs";
import path from "node:path";
import { PostureTracker, type PostureState, type PostureTrackerOptions } from "@/perception/posture";
import { expectClose, loadFixture } from "./perception-fixture-util";

const SHOULDER_Y = 400.0;
const SHOULDER_W = 200.0;
const HIDDEN = 0.0;
const BODY = [5, 6, 7, 8, 9, 10, 11, 12]; // shoulders, elbows, wrists, hips: the points motion uses
const HEAD = [0, 1, 2, 3, 4];

type Pt = readonly [number, number];
type Pose = [Float64Array, Float64Array];

interface PoseOptions {
  headDrop?: number;
  shift?: Pt;
  wrists?: readonly [Pt, Pt];
  hidden?: readonly number[];
  earDrop?: number;
}

/**
 * Seated, camera-facing COCO-17 skeleton in pixels (flat x0, y0, ...); shoulders 200 px wide at y=400. The whole head
 * moves with `headDrop` (a slump) unless `earDrop` pins the ears (a head pitched down: nose low, ears in place).
 */
function pose({ headDrop = 0.7, shift = [0.0, 0.0], wrists, hidden = [], earDrop }: PoseOptions = {}): Pose {
  const noseY = SHOULDER_Y - headDrop * SHOULDER_W;
  const pts: [number, number][] = [
    [600, noseY],
    [620, noseY - 15], [580, noseY - 15],
    [640, noseY - 5], [560, noseY - 5],
    [700, SHOULDER_Y], [500, SHOULDER_Y],
    [720, 520], [480, 520],
    [700, 600], [500, 600],
    [660, 650], [540, 650],
    [660, 800], [540, 800], [660, 950], [540, 950],
  ];
  if (earDrop !== undefined) pts[3]![1] = pts[4]![1] = SHOULDER_Y - earDrop * SHOULDER_W;
  if (wrists !== undefined) {
    pts[9] = [wrists[0][0], wrists[0][1]];
    pts[10] = [wrists[1][0], wrists[1][1]];
  }
  const xy = Float64Array.from(pts.flatMap(([x, y]) => [x + shift[0], y + shift[1]]));
  const conf = new Float64Array(17).fill(0.9);
  conf.fill(HIDDEN, 13);
  for (const k of hidden) conf[k] = HIDDEN;
  return [xy, conf];
}

/** pytest.approx: |actual - expected| <= max(rel x |expected|, abs). */
function approx(actual: number | null, expected: number, rel = 1e-6, abs = 1e-12): void {
  const a = actual ?? NaN;
  if (!(Math.abs(a - expected) <= Math.max(rel * Math.abs(expected), abs))) {
    expect.fail(`expected ${a} to be approximately ${expected}`);
  }
}

/** Python round(v): nearest integer, ties to even. */
function pyRound(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Python round(v, 6) for the (never exactly tied) sample times used here. */
const round6 = (v: number): number => Number(v.toFixed(6));

/** Feed a constant pose on [t0, t1) at `hz`; returns (t, state) pairs. */
function run(tracker: PostureTracker, t0: number, t1: number, hz = 10.0, opts: PoseOptions = {}): [number, PostureState][] {
  const n = pyRound((t1 - t0) * hz);
  const out: [number, PostureState][] = [];
  for (let i = 0; i < n; i++) {
    const t = round6(t0 + i / hz);
    out.push([t, tracker.update(...pose(opts), t)]);
  }
  return out;
}

function calibrated(headDrop = 0.7): PostureTracker {
  const tracker = new PostureTracker();
  run(tracker, 0.0, 20.1, 10, { headDrop });
  approx(tracker.baseline, headDrop);
  return tracker;
}

function first(states: [number, PostureState][], attr: "slump" | "headInHands"): number | null {
  return states.find(([, s]) => s[attr])?.[0] ?? null;
}

describe("PostureTracker calibration", () => {
  it("calibrates after the baseline window", () => {
    const tracker = new PostureTracker();
    const states = run(tracker, 0.0, 20.0, 10, { headDrop: 0.1 });
    expect(states.some(([, s]) => s.calibrated || s.slump)).toBe(false);
    expect(tracker.update(...pose({ headDrop: 0.7 }), 20.0).calibrated).toBe(true);
  });

  it("uses the median of the window as the baseline", () => {
    const tracker = new PostureTracker();
    const values = [0.6, 0.7, 0.8, 0.75, 0.2];
    for (let i = 0; i < 201; i++) tracker.update(...pose({ headDrop: values[i % values.length]! }), i / 10);
    approx(tracker.baseline, 0.7);
  });

  it("ignores samples after the window", () => {
    const tracker = calibrated(0.7);
    run(tracker, 20.1, 60.0, 10, { headDrop: 0.4 });
    approx(tracker.baseline, 0.7);
  });

  it("needs ten valid samples", () => {
    const tracker = new PostureTracker();
    for (let t = 0; t < 5; t++) tracker.update(...pose(), t);
    for (let t = 5; t < 25; t++) expect(tracker.update(...pose({ hidden: [0] }), t).calibrated).toBe(false);
    const states = [25, 26, 27, 28, 29].map((t) => tracker.update(...pose(), t).calibrated);
    expect(states).toEqual([false, false, false, false, true]);
    approx(tracker.baseline, 0.7);
  });

  it("restarts calibration on a non-positive baseline", () => {
    const tracker = new PostureTracker();
    run(tracker, 0.0, 20.1, 10, { headDrop: -0.2 });
    expect(tracker.baseline).toBeNull();
    const states = run(tracker, 20.1, 40.1, 10, { headDrop: 0.7 });
    expect(states.at(-2)![1].calibrated).toBe(false);
    expect(states.at(-1)![1].calibrated).toBe(true);
    approx(tracker.baseline, 0.7);
  });
});

describe("PostureTracker slump", () => {
  it("slumps after three seconds below the ratio", () => {
    const tracker = calibrated(0.7);
    run(tracker, 20.1, 30.0, 10, { headDrop: 0.7 });
    const states = run(tracker, 30.0, 40.0, 10, { headDrop: 0.3 });
    const onset = first(states, "slump");
    expect(onset).not.toBeNull();
    expect(onset!).toBeGreaterThanOrEqual(33.0);
    expect(onset!).toBeLessThanOrEqual(33.5);
    expect(states.filter(([t]) => t >= onset!).every(([, s]) => s.slump)).toBe(true);
  });

  it("does not slump on a brief dip", () => {
    const tracker = calibrated(0.7);
    for (let k = 0; k < 5; k++) {
      run(tracker, 25.0 + 6 * k, 27.5 + 6 * k, 10, { headDrop: 0.3 });
      const states = run(tracker, 27.5 + 6 * k, 31.0 + 6 * k, 10, { headDrop: 0.7 });
      expect(states.some(([, s]) => s.slump)).toBe(false);
    }
  });

  it("does not count a mild lean above the ratio", () => {
    const tracker = calibrated(0.7);
    const states = run(tracker, 20.1, 40.0, 10, { headDrop: 0.52 });
    expect(states.some(([, s]) => s.slump)).toBe(false);
  });

  it("recovers as soon as the head rises", () => {
    const tracker = calibrated(0.7);
    expect(run(tracker, 20.1, 30.0, 10, { headDrop: 0.3 }).at(-1)![1].slump).toBe(true);
    const states = run(tracker, 30.0, 32.0, 10, { headDrop: 0.7 });
    const cleared = states.find(([, s]) => !s.slump)![0];
    expect(cleared).toBeLessThanOrEqual(30.3);
    expect(states.filter(([t]) => t >= cleared).some(([, s]) => s.slump)).toBe(false);
  });

  it("smooths head drop", () => {
    const tracker = calibrated(0.7);
    approx(tracker.update(...pose({ headDrop: 0.3 }), 20.1).headDrop, 0.3 * 0.3 + 0.7 * 0.7);
  });

  it("holds head drop while the nose is occluded", () => {
    const tracker = calibrated(0.7);
    approx(tracker.update(...pose({ headDrop: 0.1, hidden: [0] }), 20.1).headDrop, 0.7);
  });

  it("is scale invariant", () => {
    const near = new PostureTracker();
    const far = new PostureTracker();
    for (let i = 0; i < 30; i++) {
      const [xy, conf] = pose({ headDrop: 0.6 });
      near.update(xy, conf, i / 10);
      far.update(xy.map((v) => v * 0.5), conf, i / 10);
    }
    const [xy, conf] = pose({ headDrop: 0.6 });
    approx(near.update(xy, conf, 3.0).headDrop, far.update(xy.map((v) => v * 0.5), conf, 3.0).headDrop);
  });
});

const FACE_WRISTS: readonly [Pt, Pt] = [[650.0, 300.0], [550.0, 300.0]];

describe("PostureTracker head in hands", () => {
  it("detects hands held on the head for two seconds with the nose occluded", () => {
    const tracker = new PostureTracker();
    run(tracker, 0.0, 1.0);
    const states = run(tracker, 1.0, 5.0, 10, { wrists: FACE_WRISTS, hidden: [0] });
    const onset = first(states, "headInHands");
    approx(onset, 3.0);
    expect(states.filter(([t]) => t >= onset!).every(([, s]) => s.headInHands)).toBe(true);
  });

  it("works with only the ears visible", () => {
    const states = run(new PostureTracker(), 0.0, 3.0, 10, { wrists: FACE_WRISTS, hidden: [0, 1, 2] });
    expect(states.at(-1)![1].headInHands).toBe(true);
  });

  it("needs a head point", () => {
    const states = run(new PostureTracker(), 0.0, 4.0, 10, { wrists: FACE_WRISTS, hidden: [0, 1, 2, 3, 4] });
    expect(states.some(([, s]) => s.headInHands)).toBe(false);
  });

  it("breaks when the hands drop", () => {
    const tracker = new PostureTracker();
    run(tracker, 0.0, 3.0, 10, { wrists: FACE_WRISTS, hidden: [0] });
    expect(tracker.update(...pose(), 3.0).headInHands).toBe(false);
    const states = run(tracker, 3.1, 4.9, 10, { wrists: FACE_WRISTS, hidden: [0] });
    expect(states.some(([, s]) => s.headInHands)).toBe(false);
  });

  it("ignores raised hands far from the head", () => {
    const states = run(new PostureTracker(), 0.0, 4.0, 10, { wrists: [[900.0, 150.0], [300.0, 150.0]] });
    expect(states.some(([, s]) => s.headInHands)).toBe(false);
  });

  it("needs both hands", () => {
    const states = run(new PostureTracker(), 0.0, 4.0, 10, { wrists: [[650.0, 300.0], [500.0, 600.0]], hidden: [0] });
    expect(states.some(([, s]) => s.headInHands)).toBe(false);
  });

  it("ignores hands below the shoulder line", () => {
    const states = run(new PostureTracker(), 0.0, 4.0, 10, {
      headDrop: 0.05, wrists: [[650.0, 420.0], [550.0, 420.0]],
    });
    expect(states.some(([, s]) => s.headInHands)).toBe(false);
  });

  it("does not assume an occluded wrist", () => {
    const states = run(new PostureTracker(), 0.0, 4.0, 10, { wrists: FACE_WRISTS, hidden: [0, 10] });
    expect(states.some(([, s]) => s.headInHands)).toBe(false);
  });

  it("reuses the shoulder width when a shoulder is missing", () => {
    const tracker = new PostureTracker();
    run(tracker, 0.0, 1.0);
    approx(tracker.shoulderWidth, SHOULDER_W);
    const states = run(tracker, 1.0, 4.0, 10, { wrists: FACE_WRISTS, hidden: [0, 6] });
    approx(tracker.shoulderWidth, SHOULDER_W);
    expect(states.at(-1)![1].headInHands).toBe(true);
  });
});

describe("PostureTracker motion", () => {
  it("is zero when still and NaN before a previous sample", () => {
    const states = run(new PostureTracker(), 0.0, 2.0);
    expect(states[0]![1].motion).toBeNaN();
    for (const [, s] of states.slice(1)) approx(s.motion, 0.0);
  });

  it.each([5.0, 10.0, 30.0])("is in shoulder widths per second at %s Hz", (hz) => {
    const tracker = new PostureTracker();
    const speedPx = 100.0;
    let state: PostureState | null = null;
    for (let i = 0; i < Math.trunc(3 * hz); i++) {
      const t = i / hz;
      state = tracker.update(...pose({ shift: [speedPx * t, 0.0] }), t);
    }
    approx(state!.motion, speedPx / SHOULDER_W);
  });

  it("is smoothed and decays", () => {
    const tracker = new PostureTracker();
    tracker.update(...pose(), 0.0);
    const moved = tracker.update(...pose({ shift: [20.0, 0.0] }), 0.1);
    approx(moved.motion, 20.0 / SHOULDER_W / 0.1);
    const still = tracker.update(...pose({ shift: [20.0, 0.0] }), 0.2);
    approx(still.motion, 0.7 * moved.motion);
  });

  it("only counts keypoints valid in both samples", () => {
    const tracker = new PostureTracker();
    tracker.update(...pose({ hidden: [9] }), 0.0);
    approx(tracker.update(...pose({ wrists: [[900.0, 100.0], [500.0, 600.0]] }), 0.1).motion, 0.0);
  });

  it("reports NaN when undefined (0.0 means still)", () => {
    const tracker = new PostureTracker();
    const [xy, conf] = pose();
    expect(tracker.update(xy, new Float64Array(17), 0.0).motion).toBeNaN();
    expect(tracker.update(xy, conf, 0.0).motion).toBeNaN();
    expect(tracker.update(xy.map((v) => v + 50), conf, 0.0).motion).toBeNaN();
  });
});

type Frame = [Float64Array, Float64Array, number];

/** The pre-P1-1 definition (mean over every keypoint valid in both samples), to show the artifact. */
function allKeypointMotion(frames: Frame[], alpha = 0.3): number[] {
  const out: number[] = [];
  let prev: [Float64Array, number] | null = null;
  let m = NaN;
  for (const [xy0, conf, t] of frames) {
    const xy = xy0.map((v, j) => (conf[j >> 1]! >= 0.5 ? v : NaN));
    if (prev !== null) {
      const [pxy, pt] = prev;
      let sum = 0;
      let n = 0;
      for (let k = 0; k < 17; k++) {
        const pts = [xy[2 * k]!, xy[2 * k + 1]!, pxy[2 * k]!, pxy[2 * k + 1]!];
        if (!pts.every(Number.isFinite)) continue;
        sum += Math.hypot(pts[0]! - pts[2]!, pts[1]! - pts[3]!);
        n += 1;
      }
      const raw = sum / n / SHOULDER_W / (t - pt);
      m = Number.isNaN(m) ? raw : alpha * raw + (1 - alpha) * m;
    }
    prev = [xy, t];
    out.push(m);
  }
  return out;
}

/** Deterministic uniform [0, 1) generator (mulberry32) with Box-Muller normals; stands in for numpy's rng. */
function makeRng(seed: number): { uniform: (lo: number, hi: number) => number; normal: (mu: number, sigma: number) => number } {
  let a = seed >>> 0;
  const next = (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    normal: (mu, sigma) => mu + sigma * Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next()),
  };
}

/**
 * Seated person holding still: 0.3 px body jitter (same per seed); optionally the head-drop artifact, where
 * nose/eyes/ears wander up to ~0.3 shoulder widths and flicker between conf 0.3 and 0.95.
 */
function jittery(headDrop: number, n: number, hz: number, seed: number, headJitter: boolean): Frame[] {
  const bodyRng = makeRng(seed);
  const headRng = makeRng(seed + 1);
  const frames: Frame[] = [];
  for (let i = 0; i < n; i++) {
    const [xy, conf] = pose({ headDrop });
    for (const k of BODY) for (const c of [0, 1]) xy[2 * k + c] = xy[2 * k + c]! + bodyRng.normal(0.0, 0.3);
    if (headJitter) {
      for (const k of HEAD) for (const c of [0, 1]) xy[2 * k + c] = xy[2 * k + c]! + headRng.normal(0.0, 0.3 * SHOULDER_W);
      for (const k of HEAD) conf[k] = headRng.uniform(0.3, 0.95);
    }
    frames.push([xy, conf, round6(i / hz)]);
  }
  return frames;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

describe("PostureTracker motion: torso only, identity jumps rejected, held when unmeasurable", () => {
  it("reads a held slump with head jitter as still", () => {
    const upright = jittery(0.7, 300, 10.0, 7, false);
    const slumped = jittery(0.3, 300, 10.0, 7, true);
    const upT = new PostureTracker();
    const slT = new PostureTracker();
    const up = upright.map(([xy, c, t]) => upT.update(xy, c, t).motion).slice(10);
    const sl = slumped.map(([xy, c, t]) => slT.update(xy, c, t).motion).slice(10);
    sl.forEach((v, i) => approx(v, up[i]!)); // the head no longer contributes at all
    expect(median(sl)).toBeLessThan(0.033); // at or below the live upright p50
    const oldUp = median(allKeypointMotion(upright).slice(10));
    const oldSl = median(allKeypointMotion(slumped).slice(10));
    expect(oldSl).toBeGreaterThan(10 * oldUp); // the old definition reproduces the artifact
  });

  it("ignores head motion", () => {
    const tracker = new PostureTracker();
    const states = Array.from({ length: 30 }, (_, i) => tracker.update(...pose({ headDrop: 0.7 - 0.02 * i }), i / 10));
    for (const s of states.slice(1)) approx(s.motion, 0.0);
  });

  it.each([5.0, 10.0, 30.0])("counts one arm moving, in shoulder widths per second, at %s Hz", (hz) => {
    // 2 of 8 points move at 0.5 sw/s -> 0.125 sw/s; a median of displacements would read 0.0 here.
    const tracker = new PostureTracker();
    const speedPx = 100.0;
    let state: PostureState | null = null;
    for (let i = 0; i < Math.trunc(3 * hz); i++) {
      const t = i / hz;
      const [xy, conf] = pose();
      xy[2 * 7] = xy[2 * 7]! + speedPx * t; // left elbow + wrist
      xy[2 * 9] = xy[2 * 9]! + speedPx * t;
      state = tracker.update(xy, conf, t);
    }
    approx(state!.motion, (2 / 8) * (speedPx / SHOULDER_W));
  });

  it.each([5.0, 10.0, 30.0])("rejects a keypoint identity switch at %s Hz", (hz) => {
    // A wrist snapping to the face for one sample (conf 0.9, 1.8 sw away) and back.
    const tracker = new PostureTracker();
    const states: PostureState[] = [];
    for (let i = 0; i < Math.trunc(3 * hz); i++) {
      const [xy, conf] = pose();
      if (i === Math.trunc(hz)) xy.copyWithin(2 * 9, 0, 2);
      states.push(tracker.update(xy, conf, i / hz));
    }
    for (const s of states.slice(1)) approx(s.motion, 0.0);
  });

  it.each([0.4, 1.0])("does not count a left/right wrist swap %s sw apart as motion", (wristGapSw) => {
    const tracker = new PostureTracker();
    const half = (wristGapSw * SHOULDER_W) / 2;
    const states: PostureState[] = [];
    for (let i = 0; i < 30; i++) {
      const [xy, conf] = pose({ wrists: [[600.0 + half, 600.0], [600.0 - half, 600.0]] });
      if (i === 10) xy.set([xy[20]!, xy[21]!, xy[18]!, xy[19]!], 18);
      states.push(tracker.update(xy, conf, i / 10));
    }
    for (const s of states.slice(1)) approx(s.motion, 0.0);
  });

  it("does not latch the transition when the arms leave the frame", () => {
    const tracker = new PostureTracker();
    for (let i = 0; i < 10; i++) tracker.update(...pose({ shift: [0.0, 10.0 * i] }), i / 10); // slumping: 0.5 sw/s
    const states = Array.from({ length: 50 }, (_, j) =>
      tracker.update(...pose({ shift: [0.0, 100.0], hidden: [7, 8, 9, 10, 11, 12] }), (j + 10) / 10));
    approx(states.at(-1)!.motion, 0.0, 1e-6, 1e-6);
  });

  it("holds the last value through low-confidence frames", () => {
    const tracker = new PostureTracker();
    let moving = NaN;
    for (let i = 0; i < 20; i++) moving = tracker.update(...pose({ shift: [10.0 * i, 0.0] }), i / 10).motion;
    approx(moving, 0.5);
    for (let i = 20; i < 30; i++) { // body conf 0.55: fine for slump geometry, below the motion floor
      const [xy, conf] = pose({ shift: [10.0 * i + 300.0 * (i % 2), 0.0] });
      for (const k of BODY) conf[k] = 0.55;
      const state = tracker.update(xy, conf, i / 10);
      expect(state.motion).toBe(moving);
      expect(state.headDrop).not.toBeNaN();
    }
    for (let i = 30; i < 40; i++) { // only the left shoulder and elbow: too few points, and no torso
      const state = tracker.update(...pose({ shift: [50.0 * i, 0.0], hidden: [6, 8, 9, 10, 11, 12] }), i / 10);
      expect(state.motion).toBe(moving);
    }
    tracker.update(...pose({ shift: [2000.0, 0.0] }), 4.0);
    const state = tracker.update(...pose({ shift: [2000.0, 0.0] }), 4.1);
    approx(state.motion, 0.7 * moving); // measurable again: still, so it decays
  });

  it("is NaN only before the first valid pair", () => {
    const tracker = new PostureTracker();
    const sample = (hidden: number[]): Pose => {
      const [xy, conf] = pose({ hidden });
      conf[5] = conf[6] = 0.55; // shoulders give the width but sit below the motion floor
      return [xy, conf];
    };
    const states = Array.from({ length: 5 }, (_, i) => tracker.update(...sample([8, 10, 11, 12]), i / 10));
    approx(tracker.shoulderWidth, SHOULDER_W);
    for (const s of states) expect(s.motion).toBeNaN(); // left elbow + wrist only: nothing measured yet
    expect(tracker.update(...sample([8, 10, 12]), 0.5).motion).toBeNaN(); // 3 now, but only 2 in both
    approx(tracker.update(...sample([8, 10, 12]), 0.6).motion, 0.0);
    for (let i = 7; i < 20; i++) approx(tracker.update(new Float64Array(34), new Float64Array(17), i / 10).motion, 0.0);
  });

  it("never makes the motion floor looser than the tracker floor", () => {
    expect(new PostureTracker({ minConf: 0.7, motionMinConf: 0.6 }).motionMinConf).toBe(0.7);
  });
});

describe("PostureTracker state and validation", () => {
  it("ignores low-confidence keypoints", () => {
    const tracker = new PostureTracker();
    const [xy, conf] = pose();
    conf[0] = 0.49;
    expect(tracker.update(xy, conf, 0.0).headDrop).toBeNaN();
    conf[0] = 0.5;
    approx(tracker.update(xy, conf, 0.1).headDrop, 0.7);
  });

  it("reports nothing before anything is seen", () => {
    const state = new PostureTracker().update(new Float64Array(34), new Float64Array(17), 0.0);
    expect([state.slump, state.headInHands, state.calibrated]).toEqual([false, false, false]);
    expect(state.motion).toBeNaN();
    expect(state.headDrop).toBeNaN();
  });

  it("keeps only a private copy of the previous keypoints", () => {
    const tracker = new PostureTracker();
    const [xy, conf] = pose();
    tracker.update(xy, conf, 0.0);
    xy.forEach((v, i) => (xy[i] = v + 1000.0));
    approx(tracker.update(...pose(), 0.1).motion, 0.0);
    const arrays = Object.values(tracker).filter((v): v is Float64Array => ArrayBuffer.isView(v));
    expect(arrays.map((a) => a.length)).toEqual([34]);
    expect(arrays[0]).not.toBe(xy);
  });

  it("accepts plain arrays and Float32Array", () => {
    const [xy, conf] = pose();
    const a = new PostureTracker();
    const b = new PostureTracker();
    a.update(Array.from(xy), Array.from(conf), 0.0);
    b.update(Float32Array.from(xy), Float32Array.from(conf), 0.0);
    expect(a.shoulderWidth).toBe(SHOULDER_W);
    expect(b.shoulderWidth).toBe(SHOULDER_W);
  });

  it.each([[51, 17], [32, 16], [34, 16], [34, 18], [17, 17]])("rejects %i coordinates with %i confidences", (nXy, nConf) => {
    expect(() => new PostureTracker().update(new Float64Array(nXy), new Float64Array(nConf).fill(1), 0.0)).toThrow();
  });

  it("rejects an EMA alpha outside (0, 1]", () => {
    for (const emaAlpha of [0.0, -0.1, 1.5, NaN]) expect(() => new PostureTracker({ emaAlpha })).toThrow();
    expect(new PostureTracker({ emaAlpha: 1.0 }).emaAlpha).toBe(1.0);
  });
});

describe("PostureTracker timing edges", () => {
  it("uses a half-open calibration window", () => {
    const tracker = new PostureTracker();
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9].forEach((value, i) => {
      expect(tracker.update(...pose({ headDrop: value }), 2.0 * i).calibrated).toBe(false);
    });
    expect(tracker.update(...pose({ headDrop: 0.9 }), 20.0).calibrated).toBe(true);
    approx(tracker.baseline, 0.7);
  });

  it("tolerates float rounding in the head-in-hands hold", () => {
    const tracker = new PostureTracker();
    tracker.update(...pose(), 1.2);
    expect(tracker.update(...pose({ wrists: FACE_WRISTS, hidden: [0] }), 1.3).headInHands).toBe(false);
    expect(tracker.update(...pose({ wrists: FACE_WRISTS, hidden: [0] }), 3.3).headInHands).toBe(true);
  });

  it("tolerates float rounding in the slump hold", () => {
    const tracker = new PostureTracker({ emaAlpha: 1.0 });
    run(tracker, -25.0, -4.9, 10, { headDrop: 0.7 });
    approx(tracker.baseline, 0.7);
    expect(tracker.update(...pose({ headDrop: 0.1 }), 1.1).slump).toBe(false);
    expect(tracker.update(...pose({ headDrop: 0.1 }), 2.6).slump).toBe(false);
    expect(tracker.update(...pose({ headDrop: 0.1 }), 4.1).slump).toBe(true);
  });
});

describe("PostureTracker slump vs keyboard gaze", () => {
  it("does not call keyboard gaze a slump", () => {
    const states = run(calibrated(), 20.1, 30.1, 10, { headDrop: 0.45, earDrop: 0.7 }); // nose 0.64x, ears ~0.97x
    expect(first(states, "slump")).toBeNull();
  });

  it("calls the whole head sinking a slump", () => {
    const t = first(run(calibrated(), 20.1, 30.1, 10, { headDrop: 0.45 }), "slump"); // nose 0.64x, ears 0.65x
    expect(t).not.toBeNull();
    expect(t!).toBeGreaterThanOrEqual(23.0);
  });

  it("falls back to a stricter nose rule when the ears were never visible", () => {
    const tracker = new PostureTracker();
    run(tracker, 0.0, 20.1, 10, { hidden: [3, 4] });
    expect(tracker.earBaseline).toBeNull();
    expect(first(run(tracker, 20.1, 30.1, 10, { headDrop: 0.45, hidden: [3, 4] }), "slump")).toBeNull(); // 0.64x
    expect(first(run(tracker, 30.1, 40.1, 10, { headDrop: 0.35, hidden: [3, 4] }), "slump")).not.toBeNull(); // 0.5x
  });

  it("uses the nose fallback when the ears hide mid-slump", () => {
    const tracker = calibrated();
    expect(tracker.earBaseline).not.toBeNull();
    expect(first(run(tracker, 20.1, 30.1, 10, { headDrop: 0.35, hidden: [3, 4] }), "slump")).not.toBeNull();
  });
});

// ------------------------------------------------------------------------------------------------- Python parity

interface PostureCase {
  name: string;
  params: Record<string, number>;
  t: number[];
  xy: number[][];
  conf: number[][];
  slump: boolean[];
  head_in_hands: boolean[];
  motion: number[];
  head_drop: number[];
  calibrated: boolean[];
  baseline: (number | null)[];
  ear_baseline: (number | null)[];
  shoulder_width: number[];
}

interface ConstructorCase {
  params: Record<string, number>;
  error: boolean;
  motion_min_conf: number | null;
}

interface PostureFixture {
  cases: PostureCase[];
  constructor_cases?: ConstructorCase[];
}

const OPTION_NAMES = {
  baseline_s: "baselineS",
  slump_ratio: "slumpRatio",
  slump_hold_s: "slumpHoldS",
  hih_hold_s: "hihHoldS",
  hih_radius: "hihRadius",
  ema_alpha: "emaAlpha",
  min_conf: "minConf",
  ear_ratio: "earRatio",
  nose_only_ratio: "noseOnlyRatio",
  motion_min_conf: "motionMinConf",
  motion_jump: "motionJump",
  motion_min_points: "motionMinPoints",
} as const satisfies Record<string, keyof PostureTrackerOptions>;

/** Python keyword arguments -> PostureTrackerOptions (an unknown name fails loudly). */
function options(params: Record<string, number>): PostureTrackerOptions {
  const out: PostureTrackerOptions = {};
  for (const [key, value] of Object.entries(params)) {
    if (!Object.hasOwn(OPTION_NAMES, key)) throw new Error(`unknown PostureTracker parameter ${key}`);
    out[OPTION_NAMES[key as keyof typeof OPTION_NAMES]] = value;
  }
  return out;
}

const PARTS = readdirSync(path.resolve(__dirname, "perception-fixtures"))
  .filter((f) => /^posture-.+\.json(\.gz)?$/.test(f))
  .map((f) => f.replace(/\.json(\.gz)?$/, ""))
  .sort();

describe("PostureTracker parity with valence/perception/posture.py", () => {
  it("finds every fixture part", () => {
    expect(PARTS).toEqual(["posture-calibration", "posture-edges", "posture-motion", "posture-slump"]);
  });

  describe.each(PARTS)("%s", (part) => {
    const fixture = loadFixture<PostureFixture>(part);

    it.each(fixture.cases.map((c) => [c.name, c] as const))("replays %s", (_name, c) => {
      const tracker = new PostureTracker(options(c.params));
      expect(c.t.length).toBeGreaterThan(100);
      for (let i = 0; i < c.t.length; i++) {
        const at = `${c.name}[${i}] t=${c.t[i]}`;
        const s = tracker.update(c.xy[i]!, c.conf[i]!, c.t[i]!);
        expect(s.slump, `${at} slump`).toBe(c.slump[i]);
        expect(s.headInHands, `${at} head_in_hands`).toBe(c.head_in_hands[i]);
        expect(s.calibrated, `${at} calibrated`).toBe(c.calibrated[i]);
        expectClose(s.motion, c.motion[i]!, 1e-9, `${at} motion`);
        expectClose(s.headDrop, c.head_drop[i]!, 1e-9, `${at} head_drop`);
        expectClose(tracker.shoulderWidth, c.shoulder_width[i]!, 1e-9, `${at} shoulder_width`);
        for (const [actual, expected, what] of [
          [tracker.baseline, c.baseline[i] ?? null, "baseline"],
          [tracker.earBaseline, c.ear_baseline[i] ?? null, "ear_baseline"],
        ] as const) {
          if (expected === null) expect(actual, `${at} ${what}`).toBeNull();
          else expectClose(actual ?? NaN, expected, 1e-9, `${at} ${what}`);
        }
      }
    });

    if (fixture.constructor_cases) {
      it.each(fixture.constructor_cases.map((c) => [JSON.stringify(c.params), c] as const))("constructs %s", (_n, c) => {
        if (c.error) {
          expect(() => new PostureTracker(options(c.params))).toThrow();
          return;
        }
        expectClose(new PostureTracker(options(c.params)).motionMinConf, c.motion_min_conf ?? NaN);
      });
    }
  });
});
