/**
 * Face perception: faceValence.ts (port of valence/perception/face_valence.py), faceCues.ts (port of
 * valence/features/face_cues.py, docs/EMOTION_FEATURES.md section 1) and assign.ts (port of scipy's
 * linear_sum_assignment). Ports of tests/perception/test_face_valence.py and the face_cues parts of
 * tests/test_emotion_features.py, then a replay of every case in perception-fixtures/face-*.json (written by
 * perception-fixtures/gen_face.py from the real Python and scipy).
 */
import { describe, expect, it } from "vitest";
import { linearSumAssignment } from "@/perception/assign";
import {
  FaceCueTracker,
  headPoseDeg,
  headRollDeg,
  rawCues,
  type FaceCues,
  type RawCues,
} from "@/perception/faceCues";
import {
  assignFaces,
  blendshapeScores,
  CALIBRATION_S,
  FaceValenceCalibrator,
  GAIN,
  noseTipPx,
  valenceRaw,
} from "@/perception/faceValence";
import { expectClose, loadFixture } from "./perception-fixture-util";

/** pytest.approx defaults: relative 1e-6, absolute 1e-12. */
function approx(actual: number, expected: number, what = ""): void {
  const tol = Math.max(1e-6 * Math.abs(expected), 1e-12);
  if (!(Math.abs(actual - expected) <= tol))
    expect.fail(`${what}: expected ~${expected}, got ${actual}`);
}

type Noses = Map<number, readonly [number, number, number]>;
const noses = (entries: [number, number, number, number][]): Noses =>
  new Map(entries.map(([pid, x, y, conf]) => [pid, [x, y, conf] as const]));

/** 4x4 nested rows -> the 16 numbers, column-major (MediaPipe's packing) or row-major. */
function pack(m: number[][], rowMajor = false): number[] {
  const out: number[] = [];
  for (let a = 0; a < 4; a++)
    for (let b = 0; b < 4; b++) out.push(rowMajor ? m[a]![b]! : m[b]![a]!);
  return out;
}

const IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -40, 1];
const RAW_KEYS = ["brow", "closure", "frown", "smile", "yaw", "pitch"] as const;
const CUE_KEYS = [
  "brow_down",
  "eye_open",
  "mouth_frown",
  "smile",
  "head_yaw",
  "head_pitch",
] as const;
const toRaw = (v: readonly number[]): RawCues => ({
  brow: v[0]!,
  closure: v[1]!,
  frown: v[2]!,
  smile: v[3]!,
  yaw: v[4]!,
  pitch: v[5]!,
});

// ---------------------------------------------------------------------------------------------------------------
// tests/perception/test_face_valence.py

const BS: Record<string, number> = {
  mouthSmileLeft: 0.8,
  mouthSmileRight: 0.6,
  mouthFrownLeft: 0.2,
  mouthFrownRight: 0.4,
  browDownLeft: 0.1,
  browDownRight: 0.3,
  noseSneerLeft: 0.5,
  noseSneerRight: 0.1,
  mouthPressLeft: 0.2,
  mouthPressRight: 0.0,
};

describe("valenceRaw (test_face_valence.py)", () => {
  it("applies the formula", () => {
    approx(valenceRaw(BS), 0.7 - 0.7 * 0.3 - 0.5 * 0.2 - 0.3 * 0.3 - 0.2 * 0.1);
  });

  it("averages left and right", () => {
    approx(valenceRaw({ mouthSmileLeft: 1.0, mouthSmileRight: 0.0 }), 0.5);
    approx(valenceRaw({ browDownLeft: 0.4, browDownRight: 0.8 }), -0.5 * 0.6);
  });

  it("uses the other side when one is missing, and a missing category is zero", () => {
    approx(valenceRaw({ mouthSmileLeft: 0.6 }), 0.6);
    approx(valenceRaw({ mouthFrownLeft: 1.0, mouthFrownRight: 1.0 }), -0.7);
  });

  it("ignores cheekSquint and other shapes", () => {
    const noisy = {
      ...BS,
      cheekSquintLeft: 1.0,
      cheekSquintRight: 1.0,
      eyeSquintLeft: 0.9,
      jawOpen: 0.7,
    };
    approx(valenceRaw(noisy), valenceRaw(BS));
    expect(valenceRaw({ cheekSquintLeft: 1.0, cheekSquintRight: 1.0 })).toBeNaN();
  });

  it("is NaN without a face", () => {
    expect(valenceRaw({})).toBeNaN();
  });

  it("blendshapeScores keeps only the needed categories", () => {
    const cats = [
      ["mouthSmileLeft", 0.5],
      ["cheekSquintLeft", 0.9],
      ["_neutral", 0.1],
      ["browDownRight", 0.2],
    ] as const;
    expect(
      blendshapeScores(cats.map(([categoryName, score]) => ({ categoryName, score }))),
    ).toEqual({ mouthSmileLeft: 0.5, browDownRight: 0.2 });
  });

  it("noseTipPx reads only index 1", () => {
    expect(
      noseTipPx(
        [
          { x: 0.0, y: 0.0 },
          { x: 0.25, y: 0.5 },
        ],
        640,
        360,
      ),
    ).toEqual([160.0, 180.0]);
  });
});

describe("FaceValenceCalibrator (test_face_valence.py)", () => {
  it("freezes the baseline after 20 s of visible face", () => {
    const cal = new FaceValenceCalibrator();
    let t = 0.0;
    while (t < 20.0) {
      approx(cal.update(0.2, t), 0.0);
      expect(cal.calibrated).toBe(false);
      t += 0.5;
    }
    const out = cal.update(0.9, 20.0);
    expect(cal.calibrated).toBe(true);
    approx(cal.baseline, 0.2);
    approx(out, Math.tanh(2.5 * 0.7));
    for (let k = 0; k < 100; k++) cal.update(0.9, 20.5 + 0.5 * k);
    approx(cal.baseline, 0.2);
  });

  it("uses the running median while calibrating", () => {
    const cal = new FaceValenceCalibrator();
    expect(cal.baseline).toBe(0.0);
    cal.update(0.1, 0.0);
    cal.update(0.5, 0.1);
    const out = cal.update(0.3, 0.2);
    approx(cal.baseline, 0.3);
    approx(out, 0.0);
    cal.update(0.9, 0.3);
    approx(cal.baseline, 0.4);
  });

  it("applies tanh and clips", () => {
    const cal = new FaceValenceCalibrator(0.0);
    approx(cal.update(5.0, 0.0), 1.0);
    expect(cal.baseline).toBe(0.0);
    approx(cal.update(-0.2, 1.0), Math.tanh(-0.5));
    const out = cal.update(-50.0, 2.0);
    expect(out).toBeGreaterThanOrEqual(-1.0);
    expect(out).toBeLessThanOrEqual(1.0);
  });

  it("returns NaN for NaN input and does not add it", () => {
    const cal = new FaceValenceCalibrator();
    cal.update(0.4, 0.0);
    expect(cal.update(NaN, 0.5)).toBeNaN();
    approx(cal.baseline, 0.4);
    expect(cal.update(Infinity, 0.6)).toBeNaN();
  });

  it("does not accrue calibration time while the face is hidden", () => {
    const cal = new FaceValenceCalibrator();
    for (let k = 0; k < 21; k++) cal.update(0.1, 0.5 * k);
    for (let k = 0; k < 60; k++) cal.update(NaN, 10.5 + 0.5 * k);
    cal.update(0.1, 40.0);
    expect(cal.calibrated).toBe(false);
    approx(cal.visibleSeconds, 10.0);
    let t = 40.5;
    while (!cal.calibrated) {
      cal.update(0.1, t);
      t += 0.5;
    }
    approx(t, 50.5);
  });

  it("caps a large gap", () => {
    const cal = new FaceValenceCalibrator(CALIBRATION_S, GAIN, 1.0);
    cal.update(0.0, 0.0);
    cal.update(0.0, 100.0);
    approx(cal.visibleSeconds, 1.0);
  });

  it("is not poisoned by a non-finite time", () => {
    const cal = new FaceValenceCalibrator();
    cal.update(0.1, 0.0);
    cal.update(0.1, NaN);
    cal.update(0.1, 0.5);
    expect(Number.isFinite(cal.visibleSeconds)).toBe(true);
    expect(cal.visibleSeconds).toBeLessThanOrEqual(1.0);
    expect(cal.calibrated).toBe(false);
  });

  it("clips a NaN tanh like Python's max(-1, min(1, x)): to 1", () => {
    // gain inf x (v - baseline) 0 = NaN; Python's min(1.0, nan) is 1.0 where Math.min gives NaN.
    expect(new FaceValenceCalibrator(CALIBRATION_S, Infinity).update(0.0, 0.0)).toBe(1.0);
    // max_gap_s NaN: Python's min(gap, nan) keeps the gap.
    const cal = new FaceValenceCalibrator(CALIBRATION_S, GAIN, NaN);
    cal.update(0.1, 0.0);
    cal.update(0.1, 0.5);
    expect(cal.visibleSeconds).toBe(0.5);
  });
});

describe("assignFaces (test_face_valence.py)", () => {
  it("is optimal, not greedy", () => {
    expect(
      assignFaces(
        [
          [130.0, 100.0],
          [100.0, 100.0],
        ],
        noses([
          [1, 120.0, 100.0, 0.9],
          [2, 160.0, 100.0, 0.9],
        ]),
        640,
      ),
    ).toEqual([2, 1]);
  });

  it("gates by distance", () => {
    const gate = 0.15 * 640;
    expect(assignFaces([[100.0, 100.0]], noses([[3, 100.0 + gate - 1, 100.0, 0.9]]), 640)).toEqual([
      3,
    ]);
    expect(assignFaces([[100.0, 100.0]], noses([[3, 100.0 + gate + 1, 100.0, 0.9]]), 640)).toEqual([
      null,
    ]);
    expect(assignFaces([[100.0, 100.0]], noses([[3, 150.0, 100.0, 0.9]]), 640, 0.05)).toEqual([
      null,
    ]);
  });

  it("gates on numpy's sqrt(dx*dx + dy*dy), not Math.hypot", () => {
    // numpy's distance is exactly 96 = the gate (Math.hypot gives 96.00000000000001), and the reverse.
    expect(
      assignFaces([[82.32989846953481, 49.37395890544011]], noses([[3, 0, 0, 0.9]]), 640),
    ).toEqual([3]);
    expect(
      assignFaces([[47.964392161577464, 83.15898679379401]], noses([[3, 0, 0, 0.9]]), 640),
    ).toEqual([null]);
  });

  it("ignores low-confidence noses", () => {
    expect(assignFaces([[100.0, 100.0]], noses([[4, 101.0, 100.0, 0.2]]), 640)).toEqual([null]);
    expect(
      assignFaces(
        [[100.0, 100.0]],
        noses([
          [4, 101.0, 100.0, 0.2],
          [5, 130.0, 100.0, 0.5],
        ]),
        640,
      ),
    ).toEqual([5]);
    expect(assignFaces([[100.0, 100.0]], noses([[4, 101.0, 100.0, 0.2]]), 640, 0.15, 0.1)).toEqual([
      4,
    ]);
  });

  it("handles more faces than people", () => {
    expect(
      assignFaces(
        [
          [0.0, 0.0],
          [50.0, 0.0],
        ],
        noses([[7, 40.0, 0.0, 0.9]]),
        400,
      ),
    ).toEqual([null, 7]);
  });

  it("maximises the number of valid matches", () => {
    expect(
      assignFaces(
        [
          [0.0, 0.0],
          [55.0, 0.0],
        ],
        noses([
          [1, 50.0, 0.0, 0.9],
          [2, 105.0, 0.0, 0.9],
        ]),
        400,
      ),
    ).toEqual([1, 2]);
  });

  it("handles empty input and NaN", () => {
    expect(assignFaces([], noses([[1, 0.0, 0.0, 1.0]]), 640)).toEqual([]);
    expect(assignFaces([[1.0, 1.0]], noses([]), 640)).toEqual([null]);
    expect(assignFaces([[NaN, 1.0]], noses([[1, 1.0, 1.0, 1.0]]), 640)).toEqual([null]);
    expect(assignFaces([[1.0, 1.0]], noses([[1, NaN, 1.0, 1.0]]), 640)).toEqual([null]);
  });
});

describe("linearSumAssignment (scipy)", () => {
  it("breaks ties like scipy", () => {
    // Found by the fixture sweep: a subset DP picks another optimum ([0, 1, 3, 4, 2]).
    const m = [
      [0, 2, 1, 1, 1, 0],
      [2, 0, 0, 1, 1, 1],
      [0, 0, 0, 0, 0, 2],
      [0, 1, 2, 0, 0, 1],
      [0, 2, 0, 2, 2, 2],
    ];
    expect(linearSumAssignment(m)).toEqual([
      [0, 1, 2, 3, 4],
      [0, 1, 4, 3, 2],
    ]);
    expect(
      linearSumAssignment([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]),
    ).toEqual([
      [0, 1, 2],
      [0, 1, 2],
    ]);
  });

  it("solves tall matrices transposed and returns rows ascending", () => {
    expect(
      linearSumAssignment([
        [5, 1],
        [1, 5],
        [0, 0],
      ]),
    ).toEqual([
      [1, 2],
      [0, 1],
    ]);
    expect(linearSumAssignment([])).toEqual([[], []]);
    expect(linearSumAssignment([[], []])).toEqual([[], []]);
  });

  it("throws like scipy on invalid or infeasible matrices", () => {
    expect(() =>
      linearSumAssignment([
        [NaN, 1],
        [1, 2],
      ]),
    ).toThrow("matrix contains invalid numeric entries");
    expect(() =>
      linearSumAssignment([
        [-Infinity, 1],
        [1, 2],
      ]),
    ).toThrow("matrix contains invalid numeric entries");
    expect(() =>
      linearSumAssignment([
        [Infinity, 1],
        [Infinity, 2],
      ]),
    ).toThrow("cost matrix is infeasible");
    expect(
      linearSumAssignment([
        [Infinity, 1],
        [2, Infinity],
      ]),
    ).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// tests/test_emotion_features.py (face_cues parts) and the roll extra

describe("headPoseDeg / headRollDeg", () => {
  function matrix(yaw: number, pitch: number, rowMajor = false): number[] {
    const y = (yaw * Math.PI) / 180;
    const p = (pitch * Math.PI) / 180;
    const f = [Math.sin(y) * Math.cos(p), Math.sin(p), Math.cos(y) * Math.cos(p)];
    return pack(
      [
        [1, 0, f[0]!, 0],
        [0, 1, f[1]!, 0],
        [0, 0, f[2]!, -50.0],
        [0, 0, 0, 1],
      ],
      rowMajor,
    );
  }

  it("follows the sign convention in both layouts", () => {
    for (const rowMajor of [false, true]) {
      const [yaw, pitch] = headPoseDeg(matrix(30, -10, rowMajor));
      approx(yaw, 30);
      approx(pitch, -10);
    }
    const [yaw, pitch] = headPoseDeg(IDENT);
    approx(yaw, 0.0);
    approx(pitch, 0.0);
  });

  it("reads zero yaw, pitch and roll for the identity", () => {
    expect(headPoseDeg(IDENT)).toEqual([0, 0]);
    expect(headRollDeg(IDENT)).toBe(0);
    expect(
      headRollDeg(
        pack(
          [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -40],
            [0, 0, 0, 1],
          ],
          true,
        ),
      ),
    ).toBe(0);
  });

  it("reads roll > 0 when the top of the head tilts toward the image's right", () => {
    // MediaPipe metric space: +X image right, +Y up, camera looking down -Z. The face's up axis (second column)
    // leans toward +X: (sin 15deg, cos 15deg, 0).
    const a = (15 * Math.PI) / 180;
    const m = [
      [Math.cos(a), Math.sin(a), 0, 2],
      [-Math.sin(a), Math.cos(a), 0, -3],
      [0, 0, 1, -45],
      [0, 0, 0, 1],
    ];
    for (const rowMajor of [false, true]) {
      const data = pack(m, rowMajor);
      const roll = headRollDeg(data);
      expect(roll).toBeGreaterThan(0);
      approx(roll, 15);
      const [yaw, pitch] = headPoseDeg(data);
      approx(yaw, 0);
      approx(pitch, 0);
    }
    expect(
      headRollDeg(
        pack([
          [Math.cos(a), -Math.sin(a), 0, 0],
          [Math.sin(a), Math.cos(a), 0, 0],
          [0, 0, 1, -45],
          [0, 0, 0, 1],
        ]),
      ),
    ).toBeLessThan(0);
    expect(headRollDeg([1, 2, 3])).toBeNaN();
    expect(headRollDeg(null)).toBeNaN();
  });
});

describe("FaceCueTracker (test_emotion_features.py)", () => {
  it("emptyFreezesNaN (the browser's setting): a crop-only calibration leaves yaw and pitch missing, not absolute", () => {
    const bs = {
      browDownLeft: 0.1, browDownRight: 0.1, eyeBlinkLeft: 0.1, eyeBlinkRight: 0.1,
      mouthFrownLeft: 0, mouthFrownRight: 0, mouthSmileLeft: 0, mouthSmileRight: 0,
    };
    const a = (30 * Math.PI) / 180; // turned 30 degrees about y, 50 cm away
    const turned = [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, -50, 1];
    const feed = (tr: FaceCueTracker) => {
      for (let i = 0; i < 300; i++) tr.update(rawCues(bs, null), i * 100); // 30 s of head-crop faces: no yaw
      let out: FaceCues | undefined;
      for (let i = 300; i < 310; i++) out = tr.update(rawCues(bs, turned), i * 100); // then full-frame faces
      return out!;
    };
    expect(feed(new FaceCueTracker()).head_yaw).toBeCloseTo(30, 6); // face_cues.py's 0 neutral (kept for parity)
    const browser = new FaceCueTracker(20, true);
    const out = feed(browser);
    expect(out.head_yaw).toBeNaN();
    expect(out.head_pitch).toBeNaN();
    expect(out.brow_down).toBeCloseTo(0, 9); // the channels that had samples froze as usual
    // ...and the missing ones calibrate on their own 20 s, counting only frames that carry them: with crop and
    // full-frame faces alternating, 30 s of face is ~15 s of yaw, still not enough.
    const turnedBy = (deg: number) => {
      const r = (deg * Math.PI) / 180;
      return [Math.cos(r), 0, -Math.sin(r), 0, 0, 1, 0, 0, Math.sin(r), 0, Math.cos(r), 0, 0, 0, -50, 1];
    };
    const yaws = [20, 30, 40, 30];
    let late: FaceCues | undefined;
    let k = 0;
    const step = (i: number) => {
      late = browser.update(i % 2 ? rawCues(bs, null) : rawCues(bs, turnedBy(yaws[k++ % 4]!)), i * 100);
    };
    for (let i = 310; i < 610; i++) step(i);
    expect(browser.baseline().yaw).toBeNaN();
    for (let i = 610; i < 740; i++) step(i);
    expect(browser.baseline().yaw).toBeCloseTo(30, 6); // the median of the late samples, not the last one
    browser.update(rawCues(bs, turnedBy(40)), 74_000);
    late = browser.update(rawCues(bs, turnedBy(40)), 74_100);
    expect(late!.head_yaw).toBeCloseTo(10, 6);
  });

  it("calibrates a baseline and counts blinks", () => {
    const tr = new FaceCueTracker();
    const neutral = {
      browDownLeft: 0.1,
      browDownRight: 0.1,
      eyeBlinkLeft: 0.1,
      eyeBlinkRight: 0.1,
      mouthFrownLeft: 0.0,
      mouthFrownRight: 0.0,
      mouthSmileLeft: 0.05,
      mouthSmileRight: 0.05,
    };
    let t = 0.0;
    let out: FaceCues | undefined;
    for (let k = 0; k < 25 * 15; k++) {
      out = tr.update(rawCues(neutral, IDENT), t);
      t += 1000 / 15;
    }
    approx(out!.brow_down, 0.0);
    approx(out!.eye_open, 0.0);
    const closed = { ...neutral, eyeBlinkLeft: 0.8, eyeBlinkRight: 0.8 };
    const blinks: number[] = [];
    for (const frame of [
      closed,
      closed,
      neutral,
      closed,
      ...Array<typeof closed>(12).fill(closed),
      neutral,
    ]) {
      blinks.push(tr.update(rawCues(frame, IDENT), t).blink_onsets);
      t += 1000 / 15;
    }
    expect(blinks.reduce((a, b) => a + b, 0)).toBe(1);
    expect(blinks[2]).toBe(1);
    const lowered = { ...neutral, browDownLeft: 0.4, browDownRight: 0.3 };
    approx(tr.update(rawCues(lowered, IDENT), t).brow_down, 0.25);
    expect(tr.update(null, t + 70).brow_down).toBeNaN();
  });

  it("wraps yaw with Python's floored modulo, rounding included", () => {
    // raw - base = -180.00000000000003: Python's (x + 180) % 360 rounds -2.8e-14 + 360 to 360.0, so +180 -> +90.
    const z = { brow: 0, closure: 0, frown: 0, smile: 0, pitch: 0 };
    const tr = new FaceCueTracker(0.5);
    tr.update({ ...z, yaw: 2.842170943040401e-14 }, 0);
    expect(tr.update({ ...z, yaw: -180 }, 1000).head_yaw).toBe(90);
    expect(tr.update({ ...z, yaw: 180 }, 1100).head_yaw).toBe(-90);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Parity with Python: replay every fixture case

interface ValenceFixture {
  valence_raw: { bs: Record<string, number>; out: number }[];
  blendshape_scores: { cats: [string, number][]; out: Record<string, number> }[];
  nose_tip_px: { lms: [number, number][]; w: number; h: number; out: [number, number] }[];
}

interface CalibratorFixture {
  cases: {
    name: string;
    params: { calibration_s?: number; gain?: number; max_gap_s?: number };
    v: number[];
    t: number[];
    out: number[];
    state: [number, boolean, number, number][];
  }[];
}

interface AssignFixture {
  cases: {
    name: string;
    faces: [number, number][];
    persons: [number, number, number, number][];
    frame_w: number;
    max_dist_frac: number;
    min_conf: number;
    out: (number | null)[];
  }[];
}

type LsaCase =
  | { m: number[][]; error: string }
  | { m: number[][]; rows: number[]; cols: number[]; total: number; unique: boolean };

type Blendshapes = Record<string, number | null>;

interface PoseFixture {
  head_pose: { data: number[]; layout: string; yaw: number; pitch: number }[];
  raw_cues: { bs: Blendshapes; matrix: number[] | null; out: number[] }[];
}

type TrackerFrame = null | number[] | { bs: Blendshapes; matrix: number[] | null };

interface TrackerFixture {
  cases: {
    name: string;
    calibration_s: number | null;
    frames: TrackerFrame[];
    t_ms: number[];
    out: number[][];
    baseline: [number, number[]][];
  }[];
}

describe("parity with face_valence.py", () => {
  const fx = loadFixture<ValenceFixture>("face-valence");

  it("valenceRaw", () => {
    expect(fx.valence_raw.length).toBeGreaterThan(300);
    fx.valence_raw.forEach(({ bs, out }, i) =>
      expectClose(valenceRaw(bs), out, 1e-9, `valence_raw[${i}]`),
    );
  });

  it("blendshapeScores", () => {
    for (const { cats, out } of fx.blendshape_scores) {
      expect(
        blendshapeScores(cats.map(([categoryName, score]) => ({ categoryName, score }))),
      ).toEqual(out);
    }
  });

  it("noseTipPx", () => {
    fx.nose_tip_px.forEach(({ lms, w, h, out }, i) => {
      const [x, y] = noseTipPx(
        lms.map(([lx, ly]) => ({ x: lx, y: ly })),
        w,
        h,
      );
      expectClose(x, out[0], 1e-9, `nose_tip_px[${i}].x`);
      expectClose(y, out[1], 1e-9, `nose_tip_px[${i}].y`);
    });
  });

  it("FaceValenceCalibrator sequences", () => {
    const fxc = loadFixture<CalibratorFixture>("face-calibrator");
    expect(fxc.cases.length).toBeGreaterThan(15);
    for (const c of fxc.cases) {
      const p = c.params;
      const cal = new FaceValenceCalibrator(
        p.calibration_s ?? CALIBRATION_S,
        p.gain ?? GAIN,
        p.max_gap_s ?? 1.0,
      );
      const states = new Map(c.state.map((s) => [s[0], s] as const));
      c.v.forEach((v, k) => {
        expectClose(cal.update(v, c.t[k]!), c.out[k]!, 1e-9, `${c.name}[${k}] out`);
        const s = states.get(k);
        if (s) {
          expect(cal.calibrated, `${c.name}[${k}] calibrated`).toBe(s[1]);
          expectClose(cal.baseline, s[2], 1e-9, `${c.name}[${k}] baseline`);
          expectClose(cal.visibleSeconds, s[3], 1e-9, `${c.name}[${k}] visible_s`);
        }
      });
    }
  });

  it("assignFaces", () => {
    const fxa = loadFixture<AssignFixture>("face-assign");
    expect(fxa.cases.length).toBeGreaterThan(400);
    for (const c of fxa.cases) {
      const got = assignFaces(c.faces, noses(c.persons), c.frame_w, c.max_dist_frac, c.min_conf);
      expect(got, c.name).toEqual(c.out);
    }
  });
});

describe("parity with scipy.optimize.linear_sum_assignment", () => {
  it("random rectangular matrices up to 6x6", () => {
    const cases = loadFixture<{ cases: LsaCase[] }>("face-lsa").cases;
    expect(cases.length).toBeGreaterThan(700);
    cases.forEach((c, i) => {
      const what = `lsa[${i}] ${JSON.stringify(c.m)}`;
      if ("error" in c) {
        expect(() => linearSumAssignment(c.m), what).toThrow(c.error);
        return;
      }
      const [rows, cols] = linearSumAssignment(c.m);
      let total = 0;
      rows.forEach((r, k) => {
        total += c.m[r]![cols[k]!]!;
      });
      expect(total, what).toBe(c.total);
      if (c.unique) expect([rows, cols], what).toEqual([c.rows, c.cols]);
      // The port follows scipy's tie-breaking too (assignFaces' output depends on it).
      expect([rows, cols], `${what} (ties)`).toEqual([c.rows, c.cols]);
    });
  });
});

describe("parity with face_cues.py", () => {
  const fx = loadFixture<PoseFixture>("face-pose");

  it("headPoseDeg (column- and row-major, edge layouts)", () => {
    expect(fx.head_pose.length).toBeGreaterThan(480);
    fx.head_pose.forEach(({ data, layout, yaw, pitch }, i) => {
      const [y, p] = headPoseDeg(data);
      expectClose(y, yaw, 1e-9, `head_pose[${i}] ${layout} yaw`);
      expectClose(p, pitch, 1e-9, `head_pose[${i}] ${layout} pitch`);
    });
  });

  it("rawCues", () => {
    fx.raw_cues.forEach(({ bs, matrix, out }, i) => {
      const raw = rawCues(bs, matrix);
      RAW_KEYS.forEach((key, j) => expectClose(raw[key], out[j]!, 1e-9, `raw_cues[${i}].${key}`));
    });
  });

  it("FaceCueTracker sequences", () => {
    const cases = loadFixture<TrackerFixture>("face-tracker").cases;
    expect(cases.length).toBeGreaterThan(10);
    for (const c of cases) {
      const tr =
        c.calibration_s === null ? new FaceCueTracker() : new FaceCueTracker(c.calibration_s);
      const checkpoints = new Map(c.baseline.map(([k, b]) => [k, b] as const));
      c.frames.forEach((frame, k) => {
        const raw =
          frame === null
            ? null
            : Array.isArray(frame)
              ? toRaw(frame)
              : rawCues(frame.bs, frame.matrix);
        const got = tr.update(raw, c.t_ms[k]!);
        const want = c.out[k]!;
        CUE_KEYS.forEach((key, j) =>
          expectClose(got[key], want[j]!, 1e-9, `${c.name}[${k}].${key}`),
        );
        expect(got.blink_onsets, `${c.name}[${k}].blink_onsets`).toBe(want[6]);
        const b = checkpoints.get(k);
        if (b) {
          const base = tr.baseline();
          RAW_KEYS.forEach((key, j) =>
            expectClose(base[key], b[j]!, 1e-9, `${c.name}[${k}] baseline.${key}`),
          );
        }
      });
    }
  });
});
