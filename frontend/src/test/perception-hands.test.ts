/**
 * Hands, display ids and the hysteresis debouncer: ports of tests/perception/test_hands.py, test_ids.py and
 * test_debounce.py, plus parity with valence/perception/{hands,ids,debounce}.py over the hands-*.json fixtures
 * (written by perception-fixtures/gen_hands.py from the real Python code).
 */
import { fistDebouncer, HysteresisDebouncer, laughterDebouncer } from "@/perception/debounce";
import { assignHands, curledFingers, FistTracker, isCurlFist, pyHypot, type Side } from "@/perception/hands";
import { DisplayIdMapper } from "@/perception/ids";
import { expectClose, loadFixture } from "./perception-fixture-util";

const W = 1000;
const NAN = NaN;

type Wrist = readonly [number, number, number];
type People = Map<number, Record<Side, Wrist>>;

function person(lx: number, ly: number, rx: number, ry: number, lc = 0.9, rc = 0.9): Record<Side, Wrist> {
  return { L: [lx, ly, lc], R: [rx, ry, rc] };
}

const people = (...entries: [number, Record<Side, Wrist>][]): People => new Map(entries);

/** Original 0.6 x 3-sample fist entry: exercises the generic state machine and FistTracker bookkeeping. */
const legacyFist = () => new HysteresisDebouncer(0.6, 0.4, 3, 1.0, 3.0);
const legacyLaughter = () => new HysteresisDebouncer(0.3, 0.15, 2, 0.0, 3.0);

// ---------------------------------------------------------------------------------------------- assignHands

describe("assignHands", () => {
  it("matches the nearest wrist", () => {
    const p = people([1, person(200, 500, 400, 500)], [2, person(700, 500, 900, 500)]);
    expect(assignHands([[705, 510], [395, 490]], p, W)).toEqual([[2, "L"], [1, "R"]]);
  });

  it("is globally greedy, not hand-ordered", () => {
    const p = people([1, person(500, 500, 0, 0, 0.9, 0.0)], [2, person(0, 0, 612, 500, 0.0)]);
    expect(assignHands([[510, 500], [505, 500]], p, W)).toEqual([[2, "R"], [1, "L"]]);
  });

  it("uses each wrist once", () => {
    const p = people([1, person(500, 500, 0, 0, 0.9, 0.0)]);
    expect(assignHands([[510, 500], [505, 500]], p, W)).toEqual([null, [1, "L"]]);
  });

  it("ignores low-confidence wrists", () => {
    expect(assignHands([[500, 500]], people([1, person(500, 500, 520, 500, 0.49, 0.5)]), W)).toEqual([[1, "R"]]);
    expect(assignHands([[500, 500]], people([1, person(500, 500, 520, 500, 0.2, 0.3)]), W)).toEqual([null]);
  });

  it("scales the distance limit with the frame width", () => {
    const p = people([1, person(500, 500, 0, 0, 0.9, 0.0)]);
    expect(assignHands([[619, 500]], p, W)).toEqual([[1, "L"]]);
    expect(assignHands([[620, 500]], p, W)).toEqual([[1, "L"]]); // inclusive
    expect(assignHands([[621, 500]], p, W)).toEqual([null]);
    expect(assignHands([[621, 500]], p, 2 * W)).toEqual([[1, "L"]]);
    expect(assignHands([[560, 500]], p, W, 0.05)).toEqual([null]);
  });

  it("handles NaN and empty inputs", () => {
    const p = people([1, person(NAN, NAN, 500, 500)]);
    expect(assignHands([[NAN, NAN], [505, 500]], p, W)).toEqual([null, [1, "R"]]);
    expect(assignHands([], p, W)).toEqual([]);
    expect(assignHands([[500, 500]], new Map(), W)).toEqual([null]);
  });

  it("breaks exact ties by hand index, then display id, then side", () => {
    expect(assignHands([[500, 500]], people([1, person(510, 500, 490, 500)]), W)).toEqual([[1, "L"]]);
    const p = people([7, person(503, 504, 0, 0, 0.9, 0.0)], [3, person(504, 503, 0, 0, 0.9, 0.0)]);
    expect(assignHands([[500, 500]], p, W)).toEqual([[3, "L"]]);
    expect(assignHands([[497, 504], [503, 496]], people([1, person(500, 500, 0, 0, 0.9, 0.0)]), W)).toEqual([
      [1, "L"], null,
    ]);
  });
});

// ---------------------------------------------------------------------------------------------- curl

/** Synthetic upright hand (21 x 3), normalized. `curled` holds finger indices 0..3 (index..pinky). */
function makeHand(curled: ReadonlySet<number> = new Set(), wrist: readonly [number, number] = [0.5, 0.8]): number[][] {
  const lm = Array.from({ length: 21 }, () => [0, 0, 0]);
  const [wx, wy] = wrist;
  const set = (k: number, x: number, y: number) => {
    lm[k]![0] = x;
    lm[k]![1] = y;
  };
  set(0, wx, wy);
  [[-0.04, -0.03], [-0.06, -0.06], [-0.07, -0.09], [-0.08, -0.11]].forEach(([x, y], i) => set(i + 1, wx + x!, wy + y!));
  for (let f = 0; f < 4; f++) {
    const x = wx - 0.03 + 0.02 * f;
    const base = 5 + 4 * f;
    set(base, x, wy - 0.15);
    set(base + 1, x, wy - 0.22);
    if (curled.has(f)) {
      set(base + 2, x, wy - 0.18);
      set(base + 3, x, wy - 0.12);
    } else {
      set(base + 2, x, wy - 0.27);
      set(base + 3, x, wy - 0.31);
    }
  }
  return lm;
}

describe("curledFingers / isCurlFist", () => {
  it("open hand", () => {
    expect(curledFingers(makeHand())).toBe(0);
    expect(isCurlFist(makeHand())).toBe(false);
  });

  it("closed hand", () => {
    const hand = makeHand(new Set([0, 1, 2, 3]));
    expect(curledFingers(hand)).toBe(4);
    expect(isCurlFist(hand)).toBe(true);
  });

  it.each([
    [[0], false], [[1, 2], false], [[0, 1, 2], true], [[1, 2, 3], true],
  ] as const)("three of four threshold %j", (curled, fist) => {
    const hand = makeHand(new Set(curled));
    expect(curledFingers(hand)).toBe(curled.length);
    expect(isCurlFist(hand)).toBe(fist);
  });

  it("accepts 2-D landmarks and ignores z", () => {
    const hand = makeHand(new Set([0, 1, 2]));
    expect(curledFingers(hand.map((p) => p.slice(0, 2)))).toBe(3);
    hand.forEach((p, i) => (p[2] = -1 + i / 10));
    expect(curledFingers(hand)).toBe(3);
    expect(curledFingers(hand.map((p) => Float32Array.from(p)))).toBe(3);
  });

  it("aspect makes distances isotropic", () => {
    const hand = makeHand();
    const [wx, wy] = hand[0]! as [number, number];
    for (let f = 0; f < 4; f++) {
      hand[6 + 4 * f] = [wx, wy - 0.2, 0];
      hand[8 + 4 * f] = [wx + 0.15, wy, 0];
    }
    expect(curledFingers(hand, 1.0)).toBe(4);
    expect(curledFingers(hand, 9 / 16)).toBe(0);
  });

  it("NaN landmarks never count", () => {
    const hand = makeHand(new Set([0, 1, 2, 3]));
    hand[8] = [NAN, NAN, NAN];
    hand[12] = [NAN, NAN, NAN];
    expect(curledFingers(hand)).toBe(2);
  });

  it("rejects the wrong shape", () => {
    expect(() => curledFingers(Array.from({ length: 20 }, () => [0, 0]))).toThrow();
    expect(() => curledFingers(Array.from({ length: 21 }, () => [0]))).toThrow();
    expect(() => curledFingers([...Array.from({ length: 20 }, () => [0, 0]), [0, 0, 0]])).toThrow();
    expect(() => isCurlFist([])).toThrow();
    expect(curledFingers(Array.from({ length: 21 }, () => [1, 1, 1, 1]))).toBe(0); // (21, 4): extra columns ignored
  });
});

// ---------------------------------------------------------------------------------------------- FistTracker

function clench(ft: FistTracker, pid: number, side: Side, t0: number, step = 0.2): boolean[] {
  return [0, 1, 2].map((i) => ft.update(pid, side, 0.9, t0 + i * step));
}

const sortKeys = (keys: [number, Side][]) => [...keys].sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : 1));

describe("FistTracker", () => {
  it("reports onsets as a delta", () => {
    const ft = new FistTracker(legacyFist);
    expect(clench(ft, 1, "L", 0.0)).toEqual([false, false, true]);
    expect(ft.personState(1)).toEqual([1, true]);
    expect(ft.personState(1)).toEqual([0, true]);
  });

  it("counts both hands together as one onset", () => {
    const ft = new FistTracker(legacyFist);
    clench(ft, 1, "L", 0.0);
    clench(ft, 1, "R", 0.0);
    expect(ft.personState(1)).toEqual([1, true]);
    for (let i = 1; i < 8; i++) ft.update(1, "L", 0.0, 0.4 + i * 0.2);
    expect(ft.personState(1)).toEqual([0, true]);
    for (let i = 1; i < 8; i++) ft.update(1, "R", 0.0, 0.4 + i * 0.2);
    expect(ft.personState(1)).toEqual([0, false]);
  });

  it("keeps people independent", () => {
    const ft = new FistTracker(legacyFist);
    clench(ft, 1, "L", 0.0);
    expect(ft.personState(2)).toEqual([0, false]);
    expect(ft.personState(1)).toEqual([1, true]);
  });

  it("applies the refractory per person", () => {
    const ft = new FistTracker(legacyFist);
    clench(ft, 1, "L", 0.0); // onset at 0.4
    for (let i = 1; i < 8; i++) ft.update(1, "L", 0.0, 0.4 + i * 0.2);
    expect(ft.personState(1)).toEqual([1, false]);
    expect(clench(ft, 1, "R", 2.0).at(-1)).toBe(false); // other hand, but within 3 s of the person's onset
    expect(ft.personState(1)).toEqual([0, true]);
    for (let i = 1; i < 8; i++) ft.update(1, "R", 0.0, 2.4 + i * 0.2);
    expect(clench(ft, 1, "L", 5.0).at(-1)).toBe(true); // after the refractory
    expect(ft.personState(1)).toEqual([1, true]);
  });

  it("adds nothing for a second hand while the first is clenched", () => {
    const ft = new FistTracker(legacyFist);
    clench(ft, 1, "L", 0.0);
    ft.personState(1);
    expect(clench(ft, 1, "R", 4.0).at(-1)).toBe(false); // L still clenched (no low samples fed)
    expect(ft.personState(1)).toEqual([0, true]);
  });

  it("markUnobserved ends unseen fists", () => {
    const ft = new FistTracker(legacyFist);
    clench(ft, 1, "L", 0.0);
    clench(ft, 2, "R", 0.0);
    for (let i = 1; i < 8; i++) {
      const t = 0.4 + i * 0.2;
      ft.update(2, "R", 0.9, t);
      ft.markUnobserved(t, [[2, "R"]]);
    }
    expect(ft.personState(1)).toEqual([1, false]);
    expect(ft.personState(2)).toEqual([1, true]);
  });

  it("markUnobserved breaks unseen streaks only", () => {
    const ft = new FistTracker(legacyFist);
    for (const side of ["L", "R"] as const) {
      ft.update(1, side, 0.9, 0.0);
      ft.update(1, side, 0.9, 0.2);
    }
    ft.markUnobserved(0.3, new Set([[1, "R"] as const]));
    expect(ft.update(1, "L", 0.9, 0.4)).toBe(false);
    expect(ft.update(1, "R", 0.9, 0.4)).toBe(true);
  });

  it("drop forgets a person", () => {
    const ft = new FistTracker(legacyFist);
    clench(ft, 3, "L", 0.0);
    clench(ft, 4, "L", 0.0);
    ft.drop(3);
    expect(ft.personState(3)).toEqual([0, false]);
    expect(ft.known()).toEqual([[4, "L"]]);
    expect(clench(ft, 3, "L", 0.8).at(-1)).toBe(true);
    ft.drop(99);
    expect(sortKeys(ft.known())).toEqual([[3, "L"], [4, "L"]]);
  });

  it("rejects an unknown side", () => {
    expect(() => new FistTracker().update(1, "left" as Side, 0.9, 0.0)).toThrow();
  });

  it("defaults to the contract fist rule", () => {
    const ft = new FistTracker();
    expect([0.0, 0.2, 0.4, 0.6].map((t) => ft.update(1, "L", 0.75, t))).toEqual([false, false, false, true]);
  });
});

// ---------------------------------------------------------------------------------------------- DisplayIdMapper

type Box = [number, number, number, number];

function box(cx: number, cy: number, w = 0.1, h = 0.2): Box {
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

const asObj = (m: Map<number, number>) => Object.fromEntries(m);

describe("DisplayIdMapper allocation", () => {
  it("gives first appearances sequential ids", () => {
    const m = new DisplayIdMapper();
    expect(asObj(m.update([[11, box(0.7, 0.5)], [10, box(0.3, 0.5)]], 0.0))).toEqual({ 10: 1, 11: 2 });
    expect(asObj(m.update([[10, box(0.3, 0.5)], [11, box(0.7, 0.5)], [14, box(0.5, 0.5)]], 0.1))).toEqual({
      10: 1, 11: 2, 14: 3,
    });
  });

  it("keeps active ids stable when people move", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.1, 0.5)], [2, box(0.9, 0.5)]], 0.0);
    expect(asObj(m.update([[1, box(0.9, 0.5)], [2, box(0.1, 0.5)]], 0.1))).toEqual({ 1: 1, 2: 2 });
  });

  it("takes the smallest free id after expiry", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.1, 0.5)], [2, box(0.5, 0.5)], [3, box(0.9, 0.5)]], 0.0);
    m.update([[1, box(0.1, 0.5)], [3, box(0.9, 0.5)]], 1.0);
    m.update([[1, box(0.1, 0.5)], [3, box(0.9, 0.5)]], 4.0);
    expect(m.update([[1, box(0.1, 0.5)], [3, box(0.9, 0.5)], [9, box(0.5, 0.1)]], 4.1).get(9)).toBe(2);
  });

  it("skips reserved ids", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.1, 0.5)], [2, box(0.5, 0.5)], [3, box(0.9, 0.5)]], 0.0);
    m.update([[1, box(0.1, 0.5)], [3, box(0.9, 0.5)]], 1.0);
    expect(m.reserved()).toEqual(new Set([2]));
    expect(m.update([[1, box(0.1, 0.5)], [3, box(0.9, 0.5)], [9, box(0.2, 0.95, 0.1, 0.05)]], 1.5).get(9)).toBe(4);
  });

  it("ignores duplicate tracker ids", () => {
    expect(asObj(new DisplayIdMapper().update([[5, box(0.2, 0.5)], [5, box(0.8, 0.5)]], 0.0))).toEqual({ 5: 1 });
  });
});

describe("DisplayIdMapper reuse", () => {
  it("reuses inside the window and radius", () => {
    const m = new DisplayIdMapper();
    m.update([[10, box(0.3, 0.5)], [11, box(0.8, 0.5)]], 0.0);
    m.update([[11, box(0.8, 0.5)]], 1.0);
    expect(asObj(m.update([[11, box(0.8, 0.5)], [12, box(0.5, 0.5)]], 3.5))).toEqual({ 11: 2, 12: 1 });
    expect(m.reserved()).toEqual(new Set());
  });

  it("does not reuse outside the radius", () => {
    const m = new DisplayIdMapper();
    m.update([[10, box(0.3, 0.5)], [11, box(0.8, 0.5)]], 0.0);
    m.update([[11, box(0.8, 0.5)]], 1.0);
    expect(m.update([[11, box(0.8, 0.5)], [12, box(0.56, 0.5)]], 1.5).get(12)).toBe(3);
  });

  it("treats the window boundary as exclusive", () => {
    const run = (tBack: number) => {
      const m = new DisplayIdMapper();
      m.update([[10, box(0.2, 0.5)], [11, box(0.5, 0.5)]], 0.0);
      m.update([[11, box(0.5, 0.5)]], 1.0);
      return asObj(m.update([[11, box(0.5, 0.5)], [12, box(0.9, 0.5)], [13, box(0.2, 0.5)]], tBack));
    };
    expect(run(3.75)).toEqual({ 11: 2, 12: 3, 13: 1 });
    expect(run(4.0)).toEqual({ 11: 2, 12: 1, 13: 3 });
  });

  it("measures distance in frame widths", () => {
    const reuses = (dx: number, dy: number, aspect = 9 / 16) => {
      const m = new DisplayIdMapper(3.0, 0.25, aspect);
      m.update([[1, box(0.5, 0.3)], [2, box(0.05, 0.9)]], 0.0);
      m.update([[2, box(0.05, 0.9)]], 0.5);
      return m.update([[2, box(0.05, 0.9)], [3, box(0.5 + dx, 0.3 + dy)]], 1.0).get(3) === 1;
    };
    expect(reuses(0.0, 0.4)).toBe(true);
    expect(reuses(0.4, 0.0)).toBe(false);
    expect(reuses(0.0, 0.4, 1.0)).toBe(false);
    expect(reuses(0.24, 0.0)).toBe(true);
    expect(reuses(0.26, 0.0)).toBe(false);
    expect(reuses(0.15, -0.25)).toBe(true);
  });

  it("lets two people swapping reclaim by position", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.2, 0.5)], [2, box(0.8, 0.5)]], 0.0);
    m.update([], 1.0);
    expect(asObj(m.update([[20, box(0.75, 0.5)], [21, box(0.25, 0.5)]], 1.5))).toEqual({ 20: 2, 21: 1 });
  });

  it("matches globally, nearest first", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.4, 0.5)], [2, box(0.6, 0.5)]], 0.0);
    m.update([], 1.0);
    expect(asObj(m.update([[30, box(0.52, 0.5)], [31, box(0.62, 0.5)]], 1.2))).toEqual({ 30: 1, 31: 2 });
  });

  it("lets a revived tracker reclaim its own id even if far", () => {
    const m = new DisplayIdMapper();
    m.update([[7, box(0.1, 0.5)], [8, box(0.5, 0.5)]], 0.0);
    m.update([[8, box(0.5, 0.5)]], 1.0);
    expect(asObj(m.update([[8, box(0.5, 0.5)], [7, box(0.95, 0.5)]], 2.0))).toEqual({ 7: 1, 8: 2 });
  });

  it("never double-assigns a reused id", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.5, 0.5)]], 0.0);
    m.update([], 1.0);
    expect([...m.update([[2, box(0.52, 0.5)], [3, box(0.48, 0.5)]], 1.5).values()].sort()).toEqual([1, 2]);
  });
});

describe("DisplayIdMapper lifecycle", () => {
  it("reports expired ids once", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.2, 0.5)], [2, box(0.8, 0.5)]], 0.0);
    m.update([[2, box(0.8, 0.5)]], 1.0);
    m.update([[2, box(0.8, 0.5)]], 3.9);
    expect(m.popReleased()).toEqual([]);
    m.update([[2, box(0.8, 0.5)]], 4.0);
    expect(m.popReleased()).toEqual([1]);
    expect(m.popReleased()).toEqual([]);
    expect(m.reserved()).toEqual(new Set());
  });

  it("never releases a reused id", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.2, 0.5)]], 0.0);
    m.update([], 1.0);
    m.update([[2, box(0.22, 0.5)]], 2.0);
    m.update([[2, box(0.22, 0.5)]], 10.0);
    expect(m.popReleased()).toEqual([]);
  });

  it("returns copies, and reset forgets", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.2, 0.5)], [2, box(0.8, 0.5)]], 0.0);
    const snapshot = m.active();
    snapshot.set(1, 99);
    expect(asObj(m.active())).toEqual({ 1: 1, 2: 2 });
    m.reserved().add(42);
    m.update([[2, box(0.8, 0.5)]], 1.0);
    m.reset();
    expect(m.active().size).toBe(0);
    expect(m.reserved().size).toBe(0);
    expect(asObj(m.update([[5, box(0.8, 0.5)]], 2.0))).toEqual({ 5: 1 });
  });

  it.each([[-1, 0.25, 9 / 16], [3.0, -0.1, 9 / 16], [3.0, 0.25, 0]])("rejects bad config %j", (w, d, a) => {
    expect(() => new DisplayIdMapper(w, d, a)).toThrow();
  });

  it("tolerates float rounding at window expiry", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.2, 0.5)], [2, box(0.8, 0.5)]], 0.0);
    m.update([[2, box(0.8, 0.5)]], 1.1);
    expect(m.update([[2, box(0.8, 0.5)], [3, box(0.5, 0.95)]], 4.1).get(3)).toBe(1);
    expect(m.popReleased()).toEqual([1]);
  });

  it("reports every known id as released on reset", () => {
    const m = new DisplayIdMapper();
    m.update([[1, box(0.2, 0.5)], [2, box(0.8, 0.5)]], 0.0);
    m.update([[2, box(0.8, 0.5)]], 1.0);
    m.reset();
    expect(m.popReleased()).toEqual([1, 2]);
    expect(m.popReleased()).toEqual([]);
  });

  it("truncates tracker ids like int() and rejects non-finite ones without changing state", () => {
    const m = new DisplayIdMapper();
    expect([...m.update([[4.9, box(0.2, 0.5)], [-0.5, box(0.8, 0.5)]], 0.0)]).toEqual([[0, 1], [4, 2]]);
    expect(() => m.update([[NAN, box(0.5, 0.5)]], 0.5)).toThrow();
    expect(() => m.update([[Infinity, box(0.5, 0.5)]], 0.5)).toThrow();
    expect(m.reserved().size).toBe(0);
    expect([...m.active()]).toEqual([[0, 1], [4, 2]]);
  });
});

// ---------------------------------------------------------------------------------------------- debounce

type Sample = readonly [number, number]; // (t, score), as the Python tests write them

const feed = (deb: HysteresisDebouncer, samples: readonly Sample[]) => samples.map(([t, s]) => deb.update(s, t));

function enterFist(deb: HysteresisDebouncer, t0 = 0.0, step = 0.25): number {
  const times = [0, 1, 2].map((i) => t0 + i * step);
  feed(deb, times.map((t) => [t, 0.8] as const));
  expect(deb.active).toBe(true);
  return times[2]!;
}

describe("HysteresisDebouncer", () => {
  it("needs three consecutive high samples (legacy fist)", () => {
    const deb = legacyFist();
    expect(feed(deb, [[0.0, 0.7], [0.25, 0.7]])).toEqual([false, false]);
    expect(deb.active).toBe(false);
    expect(deb.update(0.7, 0.5)).toBe(true);
    expect([deb.active, deb.onsets]).toEqual([true, 1]);
  });

  it("has an inclusive threshold; band, low and NaN samples break the streak", () => {
    expect(feed(legacyFist(), [[0.0, 0.6], [0.25, 0.6], [0.5, 0.6]])).toEqual([false, false, true]);
    let deb = legacyFist();
    feed(deb, [[0.0, 0.7], [0.25, 0.7], [0.5, 0.5], [0.75, 0.7], [1.0, 0.7]]);
    expect([deb.active, deb.onsets]).toEqual([false, 0]);
    expect(deb.update(0.7, 1.25)).toBe(true);
    deb = legacyFist();
    feed(deb, [[0.0, 0.9], [0.25, 0.9], [0.5, 0.1], [0.75, 0.9], [1.0, 0.9]]);
    expect(deb.active).toBe(false);
    deb = legacyFist();
    feed(deb, [[0.0, 0.9], [0.25, 0.9], [0.5, NAN], [0.75, 0.9], [1.0, 0.9]]);
    expect([deb.active, deb.onsets]).toEqual([false, 0]);
  });

  it("holds through the band and exits after low held one second", () => {
    let deb = legacyFist();
    const t = enterFist(deb);
    for (let i = 1; i < 41; i++) deb.update(0.5, t + i * 0.25);
    deb.update(0.4, t + 11.0);
    expect(deb.active).toBe(true);
    deb = legacyFist();
    enterFist(deb);
    feed(deb, [[1.0, 0.3], [1.5, 0.3], [1.75, 0.3]]);
    expect(deb.active).toBe(true);
    deb.update(0.3, 2.0);
    expect(deb.active).toBe(false);
    deb = legacyFist();
    enterFist(deb);
    feed(deb, [[1.0, 0.3], [1.5, 0.5], [1.75, 0.3], [2.5, 0.3]]);
    expect(deb.active).toBe(true);
    deb.update(0.3, 2.75);
    expect(deb.active).toBe(false);
  });

  it("counts NaN as low for exit, sharing the hold with low samples", () => {
    let deb = legacyFist();
    enterFist(deb);
    feed(deb, [[1.0, NAN], [1.5, NAN]]);
    expect(deb.active).toBe(true);
    deb.update(NAN, 2.0);
    expect(deb.active).toBe(false);
    deb = legacyFist();
    enterFist(deb);
    feed(deb, [[1.0, 0.1], [1.5, NAN], [2.0, 0.2]]);
    expect(deb.active).toBe(false);
  });

  it("applies the refractory from the last counted onset", () => {
    let deb = legacyFist();
    expect(enterFist(deb, 0.0)).toBe(0.5);
    feed(deb, [[0.75, 0.0], [1.75, 0.0]]);
    expect(feed(deb, [[2.0, 0.9], [2.25, 0.9], [2.5, 0.9]])).toEqual([false, false, false]);
    expect([deb.active, deb.onsets]).toEqual([true, 1]);
    feed(deb, [[2.75, 0.0], [3.75, 0.0]]);
    expect(feed(deb, [[4.0, 0.9], [4.25, 0.9], [4.5, 0.9]])).toEqual([false, false, true]);
    expect(deb.onsets).toBe(2);
    deb = legacyFist();
    enterFist(deb, 0.0);
    feed(deb, [[1.0, 0.0], [2.0, 0.0]]);
    enterFist(deb, 2.25);
    feed(deb, [[3.0, 0.0], [4.0, 0.0]]);
    expect([deb.active, deb.onsets]).toEqual([false, 1]);
    feed(deb, [[4.25, 0.9], [4.5, 0.9], [4.75, 0.9]]);
    expect(deb.onsets).toBe(2);
    deb = legacyFist();
    const held = feed(deb, Array.from({ length: 200 }, (_, i) => [i * 0.1, 0.95] as const));
    expect([held.filter(Boolean).length, deb.onsets]).toEqual([1, 1]);
  });

  it("legacy laughter: two results on, first low result off, 3 s refractory", () => {
    let deb = legacyLaughter();
    expect([deb.update(0.3, 0.0), deb.update(0.3, 0.48), deb.active]).toEqual([false, true, true]);
    deb = legacyLaughter();
    feed(deb, [[0.0, 0.5], [0.48, 0.5]]);
    deb.update(0.15, 0.96);
    expect(deb.active).toBe(true);
    deb.update(0.14, 1.44);
    expect(deb.active).toBe(false);
    deb = legacyLaughter();
    feed(deb, [[0.0, 0.5], [0.48, 0.5], [0.96, NAN]]);
    expect(deb.active).toBe(false);
    deb = legacyLaughter();
    feed(deb, [[0.0, 0.5], [0.5, 0.5], [1.0, 0.0], [1.5, 0.5], [2.0, 0.5]]);
    expect([deb.active, deb.onsets]).toEqual([true, 1]);
    feed(deb, [[2.5, 0.0], [3.0, 0.5], [3.5, 0.5]]);
    expect([deb.active, deb.onsets]).toEqual([true, 2]);
  });

  it("rejects inverted thresholds; reset clears everything; defaults enter and exit at once", () => {
    expect(() => new HysteresisDebouncer(0.3, 0.5)).toThrow();
    let deb = legacyFist();
    enterFist(deb);
    deb.reset();
    expect([deb.active, deb.onsets]).toEqual([false, 0]);
    expect(feed(deb, [[1.0, 0.9], [1.25, 0.9], [1.5, 0.9]]).at(-1)).toBe(true);
    deb = new HysteresisDebouncer(0.5, 0.5);
    expect(deb.update(0.5, 0.0)).toBe(true);
    deb.update(0.49, 0.1);
    expect(deb.active).toBe(false);
    expect(deb.update(0.9, 0.2)).toBe(true);
    expect(deb.onsets).toBe(2);
  });

  it("tolerates float rounding in holds and the refractory", () => {
    let deb = legacyFist();
    enterFist(deb);
    deb.update(0.1, 1.3);
    deb.update(0.1, 2.3);
    expect(deb.active).toBe(false);
    deb = legacyFist();
    expect(feed(deb, [[0.9, 0.9], [1.0, 0.9], [1.1, 0.9]]).at(-1)).toBe(true);
    feed(deb, [[1.5, 0.0], [2.5, 0.0]]);
    expect(feed(deb, [[3.9, 0.9], [4.0, 0.9], [4.1, 0.9]]).at(-1)).toBe(true);
    expect(deb.onsets).toBe(2);
  });

  it("fist rule: >= 0.7 held 0.5 s; dips and NaN restart the hold", () => {
    expect(feed(fistDebouncer(), [[0.0, 0.7], [0.2, 0.75], [0.4, 0.8], [0.5, 0.8]])).toEqual([
      false, false, false, true,
    ]);
    expect(feed(fistDebouncer(), Array.from({ length: 20 }, (_, i) => [i * 0.2, 0.65] as const)).some(Boolean)).toBe(
      false,
    );
    expect(
      feed(fistDebouncer(), [[0.0, 0.9], [0.2, 0.9], [0.4, 0.5], [0.6, 0.9], [0.8, 0.9], [1.0, 0.9], [1.1, 0.9]]),
    ).toEqual([false, false, false, false, false, false, true]);
    expect(feed(fistDebouncer(), [[0.0, 0.9], [0.2, 0.9], [0.3, NAN], [0.4, 0.9], [0.6, 0.9]]).some(Boolean)).toBe(
      false,
    );
  });

  it("laughter rule: parameters and the probe replay", () => {
    const deb = laughterDebouncer();
    expect([deb.on, deb.off, deb.onCount, deb.offHoldS, deb.refractoryS]).toEqual([0.05, 0.02, 1, 1.5, 5.5]);
    const probe = [
      0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0,
      0.0, 0.043, 0.082, 0.0, 0.082, 0.004, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.004, 0.0, 0.043,
      0.008, 0.0, 0.0, 0.0,
    ];
    const onsets = probe.map((x, i) => deb.update(x, i));
    const phases: [number, number][] = [[0, 6], [6, 12], [12, 20], [20, 26], [26, 32], [32, 38], [38, 42]];
    expect(phases.map(([a, b]) => onsets.slice(a, b).filter(Boolean).length)).toEqual([0, 0, 0, 1, 0, 1, 0]);
  });

  it.each([0.98, 0.995, 0.999, 1.0, 1.001, 1.02])("laughter exit on the third low result, dt %s", (dt) => {
    const deb = laughterDebouncer();
    [0.08, 0.0, 0.0].forEach((x, i) => deb.update(x, 100.0 + i * dt));
    expect(deb.active).toBe(true);
    deb.update(0.0, 100.0 + 3 * dt);
    expect(deb.active).toBe(false);
    const seq = [0.08, 0.0, 0.08, 0.0, 0.0, 0.0, 0.08, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0];
    const d2 = laughterDebouncer();
    const hits = seq.map((x, i) => d2.update(x, 50.0 + i * dt));
    expect(hits.flatMap((o, i) => (o ? [i] : []))).toEqual([0, 6]);
  });
});

// ---------------------------------------------------------------------------------------------- pyHypot

describe("pyHypot", () => {
  it("is exact on integer triangles and symmetric", () => {
    expect(pyHypot(3, 4)).toBe(5);
    expect(pyHypot(-4, 3)).toBe(5);
    expect(pyHypot(0, -7)).toBe(7);
    expect(pyHypot(-0, -0)).toBe(0);
  });

  it("returns Infinity over NaN, NaN otherwise", () => {
    expect(pyHypot(Infinity, NAN)).toBe(Infinity);
    expect(pyHypot(NAN, -Infinity)).toBe(Infinity);
    expect(pyHypot(NAN, 1)).toBeNaN();
  });

  it("does not overflow or underflow", () => {
    expect(pyHypot(1e308, 1e308)).toBe(1.4142135623730951e308);
    expect(pyHypot(5e-324, 5e-324)).toBe(5e-324);
  });
});

// ---------------------------------------------------------------------------------------------- parity

interface HypotFixture {
  cases: [number, number, number][];
}

interface AssignCase {
  hands: [number, number][];
  people: [number, [Side, [number, number, number]][]][];
  frame_w: number;
  max_dist_frac?: number;
  min_conf?: number;
  expected: ([number, Side] | null)[];
}

interface CurlCase {
  lm: number[][];
  aspect?: number;
  min_curled?: number;
  curled?: number;
  fist?: boolean;
  error?: boolean;
}

type FistOp =
  | ["u", number, Side, number, number, boolean]
  | ["m", number, [number, Side][]]
  | ["p", number, number, boolean]
  | ["d", number]
  | ["k", [number, Side][]];

interface FistCase {
  name: string;
  factory: [number, number, number, number, number, number] | null;
  ops: FistOp[];
}

type IdsOp = ["u", [number, Box][], number, [number, number][] | null, number[]] | ["p", number[]] | ["r"];

interface IdsParams {
  reuse_window_s?: number;
  reuse_dist?: number;
  aspect?: number;
}

interface IdsFixture {
  cases: { name: string; params: IdsParams; ops: IdsOp[] }[];
  constructor_cases: { params: IdsParams; error: boolean }[];
}

type DebParams = [number, number, number, number, number, number];
type DebOp = ["s", number, number, boolean, boolean, number] | ["r", boolean, number];

interface DebFixture {
  cases: { name: string; preset: "fist" | "laughter" | null; params: DebParams; ops: DebOp[] }[];
  constructor_cases: { on: number; off: number; error: boolean }[];
}

/** The Python {side: wrist} dict, keys in the same insertion order. */
function sidesRecord(sides: readonly [Side, Wrist][]): Record<Side, Wrist> {
  const out = {} as Record<Side, Wrist>;
  for (const [side, wrist] of sides) out[side] = wrist;
  return out;
}

const makeDebouncer = (p: DebParams) => new HysteresisDebouncer(...p);
const debParams = (d: HysteresisDebouncer): DebParams => [d.on, d.off, d.onCount, d.offHoldS, d.refractoryS, d.onHoldS];

describe("parity with valence/perception (hands-*.json)", () => {
  it("pyHypot matches math.hypot bit for bit", () => {
    const { cases } = loadFixture<HypotFixture>("hands-hypot");
    expect(cases.length).toBeGreaterThan(1000);
    for (const [x, y, h] of cases) {
      const got = pyHypot(x, y);
      if (!Object.is(got, h)) expect.fail(`hypot(${x}, ${y}): expected ${h}, got ${got}`);
      expectClose(got, h, 1e-9, `hypot(${x}, ${y})`);
    }
  });

  it("assignHands", () => {
    const { cases } = loadFixture<{ cases: AssignCase[] }>("hands-assign");
    expect(cases.length).toBeGreaterThan(300);
    cases.forEach((c, i) => {
      const wrists: People = new Map(c.people.map(([pid, sides]) => [pid, sidesRecord(sides)]));
      expect(assignHands(c.hands, wrists, c.frame_w, c.max_dist_frac, c.min_conf), `case ${i}`).toEqual(c.expected);
    });
  });

  it("curledFingers / isCurlFist", () => {
    const { cases } = loadFixture<{ cases: CurlCase[] }>("hands-curl");
    expect(cases.length).toBeGreaterThan(200);
    cases.forEach((c, i) => {
      if (c.error) {
        expect(() => curledFingers(c.lm, c.aspect), `case ${i}`).toThrow();
        expect(() => isCurlFist(c.lm, c.aspect, c.min_curled), `case ${i}`).toThrow();
        return;
      }
      expect(curledFingers(c.lm, c.aspect), `case ${i}`).toBe(c.curled);
      expect(isCurlFist(c.lm, c.aspect, c.min_curled), `case ${i}`).toBe(c.fist);
    });
  });

  it("FistTracker op sequences", () => {
    const { cases } = loadFixture<{ cases: FistCase[] }>("hands-fist");
    expect(cases.length).toBeGreaterThan(20);
    let onsets = 0;
    for (const c of cases) {
      const factory = c.factory;
      const ft = factory === null ? new FistTracker() : new FistTracker(() => makeDebouncer(factory));
      c.ops.forEach((op, i) => {
        const at = `${c.name} op ${i}`;
        switch (op[0]) {
          case "u":
            expect(ft.update(op[1], op[2], op[3], op[4]), at).toBe(op[5]);
            if (op[5]) onsets += 1;
            break;
          case "m":
            ft.markUnobserved(op[1], op[2]);
            break;
          case "p":
            expect(ft.personState(op[1]), at).toEqual([op[2], op[3]]);
            break;
          case "d":
            ft.drop(op[1]);
            break;
          case "k":
            expect(ft.known(), at).toEqual(op[1]);
            break;
        }
      });
    }
    expect(onsets).toBeGreaterThan(50);
  });

  it("DisplayIdMapper op sequences", () => {
    const { cases, constructor_cases } = loadFixture<IdsFixture>("hands-ids");
    expect(cases.length).toBeGreaterThan(50);
    for (const { params, error } of constructor_cases) {
      const make = () => new DisplayIdMapper(params.reuse_window_s, params.reuse_dist, params.aspect);
      if (error) expect(make, JSON.stringify(params)).toThrow();
      else expect(make, JSON.stringify(params)).not.toThrow();
    }
    for (const c of cases) {
      const m = new DisplayIdMapper(c.params.reuse_window_s, c.params.reuse_dist, c.params.aspect);
      c.ops.forEach((op, i) => {
        const at = `${c.name} op ${i}`;
        if (op[0] === "u") {
          const [, tracks, t, items, reserved] = op;
          if (items === null) {
            expect(() => m.update(tracks, t), at).toThrow();
          } else {
            const mapping = m.update(tracks, t);
            expect([...mapping], at).toEqual(items); // dict order included
            expect([...m.active()], at).toEqual(items);
          }
          expect([...m.reserved()].sort((a, b) => a - b), at).toEqual(reserved);
        } else if (op[0] === "p") {
          expect(m.popReleased(), at).toEqual(op[1]);
        } else {
          m.reset();
        }
      });
    }
  });

  it("HysteresisDebouncer sample sequences", () => {
    const { cases, constructor_cases } = loadFixture<DebFixture>("hands-debounce");
    expect(cases.length).toBeGreaterThan(100);
    for (const { on, off, error } of constructor_cases) {
      if (error) expect(() => new HysteresisDebouncer(on, off), `${on}, ${off}`).toThrow();
      else expect(() => new HysteresisDebouncer(on, off), `${on}, ${off}`).not.toThrow();
    }
    for (const c of cases) {
      const preset = c.preset === "fist" ? fistDebouncer : c.preset === "laughter" ? laughterDebouncer : null;
      const deb = preset === null ? makeDebouncer(c.params) : preset();
      expect(debParams(deb), c.name).toEqual(c.params);
      c.ops.forEach((op, i) => {
        const at = `${c.name} op ${i}`;
        if (op[0] === "r") {
          deb.reset();
          expect([deb.active, deb.onsets], at).toEqual([op[1], op[2]]);
        } else {
          expect([deb.update(op[1], op[2]), deb.active, deb.onsets], at).toEqual([op[3], op[4], op[5]]);
        }
      });
    }
  });
});
