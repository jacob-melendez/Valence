import { describe, expect, it, vi } from "vitest";
import { AudioPipeline, YAMNET_WINDOW } from "@/perception/audio";
import { BLAZEPOSE_TO_COCO, blazeposeToCoco, poseBoxPx, type Landmark } from "@/perception/coco";
import { PostureTracker } from "@/perception/posture";
import { IouTracker, iou } from "@/perception/tracker";
import { BLENDSHAPE_NAMES, type VideoObs } from "@/perception/types";
import { FACE_HOLD_S, VisionPipeline, type FrameInput, type PixelSource } from "@/perception/video";

// ---------- synthetic MediaPipe outputs ----------

/** A seated person facing the camera (BlazePose's 33 landmarks, normalized); `dx` shifts them sideways. */
function seated(opts: { dx?: number; scale?: number; wristY?: number; vis?: number } = {}): Landmark[] {
  const { dx = 0, scale = 1, wristY = 0.88, vis = 0.98 } = opts;
  const at = (x: number, y: number, v = vis): Landmark => ({ x: 0.5 + (x - 0.5) * scale + dx, y: 0.1 + (y - 0.1) * scale, z: 0, visibility: v });
  const lm: Landmark[] = Array.from({ length: 33 }, () => at(0.5, 0.5, 0.1));
  lm[0] = at(0.5, 0.3); // nose
  lm[1] = at(0.51, 0.28); lm[2] = at(0.52, 0.28); lm[3] = at(0.53, 0.28); // left eye (image right)
  lm[4] = at(0.49, 0.28); lm[5] = at(0.48, 0.28); lm[6] = at(0.47, 0.28); // right eye
  lm[7] = at(0.55, 0.29); lm[8] = at(0.45, 0.29); // ears
  lm[9] = at(0.51, 0.33); lm[10] = at(0.49, 0.33); // mouth
  lm[11] = at(0.6, 0.5); lm[12] = at(0.4, 0.5); // shoulders
  lm[13] = at(0.66, 0.7); lm[14] = at(0.34, 0.7); // elbows
  lm[15] = at(0.62, wristY); lm[16] = at(0.38, wristY); // wrists
  for (const i of [17, 19, 21]) lm[i] = at(0.62, wristY + 0.02);
  for (const i of [18, 20, 22]) lm[i] = at(0.38, wristY + 0.02);
  lm[23] = at(0.58, 1.05, 0.2); lm[24] = at(0.42, 1.05, 0.2); // hips below the frame
  for (const i of [25, 27, 29, 31]) lm[i] = at(0.58, 1.4, 0.05);
  for (const i of [26, 28, 30, 32]) lm[i] = at(0.42, 1.4, 0.05);
  return lm;
}

function blendshapes(smile: number, over: Record<string, number> = {}) {
  return {
    categories: BLENDSHAPE_NAMES.map((name, index) => ({
      index, categoryName: name, displayName: "", score: over[name] ?? (name.startsWith("mouthSmile") ? smile : 0.02),
    })),
    headIndex: 0, headName: "",
  };
}

/** Column-major 4x4 with the face 50 cm in front of the camera, facing it. */
const MATRIX = { rows: 4, columns: 4, data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -50, 1] };

interface Scene {
  poses: Landmark[][];
  faces: { nose: [number, number]; smile: number; bs?: Record<string, number> }[];
  hands: { wrist: [number, number]; fist: number }[];
  cropFace?: number | null;
}

function fakeModels(scene: () => Scene) {
  const calls = { pose: [] as number[], faces: [] as number[], gestures: [] as number[], crop: 0 };
  const models = {
    pose: (_img: unknown, ts: number) => {
      calls.pose.push(ts);
      return { landmarks: scene().poses, worldLandmarks: [], close() {} };
    },
    faces: (_img: unknown, ts: number) => {
      calls.faces.push(ts);
      const s = scene();
      return {
        faceLandmarks: s.faces.map((f) => Array.from({ length: 478 }, () => ({ x: f.nose[0], y: f.nose[1], z: 0, visibility: 0 }))),
        faceBlendshapes: s.faces.map((f) => blendshapes(f.smile, f.bs)),
        facialTransformationMatrixes: s.faces.map(() => MATRIX),
      };
    },
    faceCrop: () => {
      calls.crop += 1;
      const smile = scene().cropFace;
      return smile == null
        ? { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] }
        : { faceLandmarks: [[]], faceBlendshapes: [blendshapes(smile)], facialTransformationMatrixes: [MATRIX] };
    },
    gestures: (_img: unknown, ts: number) => {
      calls.gestures.push(ts);
      const s = scene();
      return {
        landmarks: s.hands.map((hd) => Array.from({ length: 21 }, () => ({ x: hd.wrist[0], y: hd.wrist[1], z: 0, visibility: 0 }))),
        worldLandmarks: [], handedness: [], handednesses: [],
        gestures: s.hands.map((hd) => [{ index: 1, categoryName: "Closed_Fist", displayName: "", score: hd.fist }]),
      };
    },
  };
  return { models: models as unknown as ConstructorParameters<typeof VisionPipeline>[0], calls };
}

function fakePixels() {
  const calls = { luma: 0, crop: 0, wipe: 0 };
  const pixels: PixelSource = {
    lumaStrided: (_f, w, h, stride, toGray) => {
      calls.luma += 1;
      const gw = Math.ceil(w / stride), gh = Math.ceil(h / stride);
      const gray = new Uint8Array(gw * gh);
      for (let i = 0; i < gray.length; i++) gray[i] = toGray(120, 120, 120) + ((i * 7919) % 5);
      return { gray, w: gw, h: gh };
    },
    cropTo: () => {
      calls.crop += 1;
      return {} as never;
    },
    wipeCrop: () => {
      calls.wipe += 1;
    },
  };
  return { pixels, calls };
}

const FRAME: FrameInput = { image: {} as never, width: 1280, height: 720 };

function run(pipeline: VisionPipeline, frames: number, t0 = 100, dt = 0.1): VideoObs[] {
  return Array.from({ length: frames }, (_, i) => pipeline.process(FRAME, t0 + i * dt, 1.7e9 + i * dt, i, i + 1));
}

// ---------- tests ----------

describe("BlazePose -> COCO-17", () => {
  it("maps landmarks 0,2,5,7,8,11-16,23-28 to COCO 0-16, in pixels", () => {
    expect([...BLAZEPOSE_TO_COCO]).toEqual([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);
    const lm = seated();
    const { xy, conf } = blazeposeToCoco(lm, 1280, 720);
    expect(xy[0]).toBeCloseTo(0.5 * 1280);
    expect(xy[1]).toBeCloseTo(0.3 * 720 * 1 + 0, 6);
    expect(xy[2 * 5]).toBeCloseTo(lm[11]!.x * 1280); // COCO left shoulder <- BlazePose 11
    expect(xy[2 * 9 + 1]).toBeCloseTo(lm[15]!.y * 720); // COCO left wrist <- BlazePose 15
    expect(conf[0]).toBeCloseTo(0.98);
    expect(conf[10]).toBeCloseTo(0.98); // right wrist
    expect([...conf.slice(11)]).toEqual([0, 0, 0, 0, 0, 0]); // hips and legs: never confident (BlazePose extrapolates)
    const offFrame = blazeposeToCoco(lm.map((l, i) => (i === 0 ? { ...l, x: -0.1 } : l)), 1280, 720);
    expect(offFrame.conf[0]).toBe(0);
  });

  it("boxes the person to the frame edge and raises the top above the eyes", () => {
    const box = poseBoxPx(seated(), 1280, 720)!;
    expect(box[3]).toBe(720);
    expect(box[1]).toBeLessThan(0.28 * 720);
    expect(box[0]).toBeGreaterThan(0);
  });
});

describe("IouTracker", () => {
  it("keeps ids while boxes overlap, revives lost tracks, and expires them after maxLostS", () => {
    const tr = new IouTracker(0.2, 6);
    const a = [100, 100, 300, 500] as const, b = [700, 100, 900, 500] as const;
    expect(tr.update([a, b], 0)).toEqual([1, 2]);
    expect(tr.update([[110, 100, 310, 500], b], 0.1)).toEqual([1, 2]);
    expect(tr.update([b], 0.2)).toEqual([2]); // 1 lost
    expect(tr.update([b, a], 3)).toEqual([2, 1]); // revived
    expect(tr.update([b], 3.1)).toEqual([2]);
    expect(tr.update([b], 6)).toEqual([2]);
    expect(tr.update([b, a], 9.2)).toEqual([2, 3]); // 1 unseen for 6.2 s > 6 s: expired, new id
    expect(iou(a, a)).toBe(1);
    expect(iou(a, b)).toBe(0);
  });
});

describe("VisionPipeline (mocked MediaPipe)", () => {
  it("one seated person: tracked, posture, face valence after the first odd frame, extras, lux at 1 Hz", () => {
    const { models, calls } = fakeModels(() => ({ poses: [seated()], faces: [{ nose: [0.5, 0.3], smile: 0.3 }], hands: [] }));
    const { pixels, calls: px } = fakePixels();
    const obs = run(new VisionPipeline(models, pixels), 30);
    expect(obs.every((o) => o.persons.length === 1 && o.persons[0]!.track_id === 1)).toBe(true);
    expect(obs[0]!.persons[0]!.face_visible).toBe(false); // faces run on odd iterations
    const p = obs[1]!.persons[0]!;
    expect(p.face_visible).toBe(true);
    expect(p.face_valence).toBe(0); // first sample is its own baseline
    expect(Object.keys(p.raw!.blendshapes)).toHaveLength(52);
    expect(p.raw!.head_yaw_deg).toBeCloseTo(0);
    expect(Number.isFinite(p.brow_down)).toBe(true);
    expect(obs.every((o) => o.persons[0]!.face_visible === (o.frame_seq > 0))).toBe(true); // held between face passes
    expect(px.luma).toBe(3); // t = 100.0, 101.0, 102.0
    expect(Number.isFinite(obs[0]!.lux)).toBe(true);
    expect(obs[0]!.lux_quality).toBeGreaterThan(0);
    expect(calls.pose).toHaveLength(30);
    expect(calls.faces).toHaveLength(15);
    expect(calls.gestures).toHaveLength(15);
    expect(calls.crop).toBe(0);
    expect(obs.at(-1)!.frames_decoded).toBe(30);
    expect(obs[5]!.persons[0]!.bbox.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it("drops the face after FACE_HOLD_S without a face pass, and tries a head crop when unmatched", () => {
    let face = true;
    const { models, calls } = fakeModels(() => ({
      poses: [seated()], faces: face ? [{ nose: [0.5, 0.3], smile: 0.2 }] : [], hands: [], cropFace: null,
    }));
    const { pixels, calls: px } = fakePixels();
    const pipe = new VisionPipeline(models, pixels);
    run(pipe, 4);
    face = false;
    const later = run(pipe, 10, 100.4);
    expect(later.some((o) => o.persons[0]!.face_visible)).toBe(true); // held right after
    expect(later.at(-1)!.persons[0]!.face_visible).toBe(false);
    expect(later.at(-1)!.persons[0]!.face_valence).toBeNaN();
    expect(later.at(-1)!.persons[0]!.raw).toBeNull();
    expect(calls.crop).toBeGreaterThan(0);
    expect(px.wipe).toBe(px.crop); // every crop wiped
    expect(FACE_HOLD_S).toBe(0.45);
  });

  it("stretches the face hold on slow devices (4 fps) but matches Python at 10 Hz", () => {
    const { models } = fakeModels(() => ({ poses: [seated()], faces: [{ nose: [0.5, 0.3], smile: 0.2 }], hands: [] }));
    const fast = new VisionPipeline(models, fakePixels().pixels);
    run(fast, 20, 100, 0.1);
    expect(fast.faceHoldS).toBeCloseTo(0.45, 9);
    const slow = new VisionPipeline(fakeModels(() => ({ poses: [seated()], faces: [{ nose: [0.5, 0.3], smile: 0.2 }], hands: [] })).models, fakePixels().pixels);
    const obs = run(slow, 20, 100, 0.25);
    expect(slow.faceHoldS).toBeCloseTo(1.125, 6);
    expect(obs.slice(2).every((o) => o.persons[0]!.face_visible)).toBe(true); // no flicker between passes
  });

  it("a pause (tab hidden) is not taken for a slow device: the face hold stays 0.45 s after it", () => {
    let face = true;
    const { models } = fakeModels(() => ({
      poses: [seated()], faces: face ? [{ nose: [0.5, 0.3], smile: 0.2 }] : [], hands: [], cropFace: null,
    }));
    const pipe = new VisionPipeline(models, fakePixels().pixels);
    run(pipe, 20, 100); // 10 Hz
    run(pipe, 4, 160); // back after a minute in a hidden tab: face passes at 160.1 and 160.3
    expect(pipe.faceHoldS).toBeCloseTo(FACE_HOLD_S, 9);
    face = false; // the person leaves
    const after = run(pipe, 10, 160.4);
    expect(after.at(-1)!.persons[0]!.face_visible).toBe(false); // 1 s after the last face
  });

  it("uses the head-crop face when the full-frame pass misses it", () => {
    const { models } = fakeModels(() => ({ poses: [seated()], faces: [], hands: [], cropFace: 0.4 }));
    const obs = run(new VisionPipeline(models, fakePixels().pixels), 4);
    expect(obs[1]!.persons[0]!.face_visible).toBe(true);
    expect(obs[1]!.persons[0]!.raw!.blendshapes["mouthSmileLeft"]).toBe(0.4);
  });

  it("counts a debounced fist for a raised hand, ignores hands in the keyboard zone", () => {
    const scene = { wrist: 0.6, fist: 0.95 };
    const { models } = fakeModels(() => ({
      poses: [seated({ wristY: scene.wrist })], faces: [], hands: [{ wrist: [0.62, scene.wrist], fist: scene.fist }],
    }));
    const pipe = new VisionPipeline(models, fakePixels().pixels);
    const raised = run(pipe, 20);
    expect(raised.reduce((n, o) => n + o.persons[0]!.fist_onsets, 0)).toBe(1);
    expect(raised.at(-1)!.persons[0]!.fist_active).toBe(true);

    scene.wrist = 0.86; // keyboard zone
    const { models: m2 } = fakeModels(() => ({
      poses: [seated({ wristY: 0.86 })], faces: [], hands: [{ wrist: [0.62, 0.86], fist: 0.99 }],
    }));
    const typing = run(new VisionPipeline(m2, fakePixels().pixels), 20);
    expect(typing.reduce((n, o) => n + o.persons[0]!.fist_onsets, 0)).toBe(0);
  });

  it("two people: ids by position, each face goes to its own person, small people ignored", () => {
    const { models } = fakeModels(() => ({
      poses: [seated({ dx: 0.25, scale: 0.8 }), seated({ dx: -0.25, scale: 0.8 }), seated({ dx: 0, scale: 0.15 })],
      faces: [{ nose: [0.75, 0.26], smile: 0.6 }, { nose: [0.25, 0.26], smile: 0.0 }],
      hands: [],
    }));
    const obs = run(new VisionPipeline(models, fakePixels().pixels), 6);
    const last = obs.at(-1)!;
    expect(last.persons.map((p) => p.track_id)).toEqual([1, 2]);
    const right = last.persons.find((p) => p.bbox[0] > 0.5)!;
    const left = last.persons.find((p) => p.bbox[0] < 0.5)!;
    expect(right.raw!.blendshapes["mouthSmileLeft"]).toBe(0.6);
    expect(left.raw!.blendshapes["mouthSmileLeft"]).toBe(0.0);
  });

  it("faceEveryFrame runs the face pass on every frame (default: odd frames, the Python cadence)", () => {
    const scene = () => ({ poses: [seated()], faces: [{ nose: [0.5, 0.3] as [number, number], smile: 0.2 }], hands: [] });
    const a = fakeModels(scene);
    run(new VisionPipeline(a.models, fakePixels().pixels), 10);
    expect(a.calls.faces).toHaveLength(5);
    expect(a.calls.gestures).toHaveLength(5);
    const b = fakeModels(scene);
    const obs = run(new VisionPipeline(b.models, fakePixels().pixels, { faceEveryFrame: true }), 10);
    expect(b.calls.faces).toHaveLength(10);
    expect(b.calls.gestures).toHaveLength(5);
    expect(obs[0]!.persons[0]!.face_visible).toBe(true); // visible from the first frame
  });

  it("sends strictly increasing timestamps to each VIDEO-mode task even if t repeats", () => {
    const { models, calls } = fakeModels(() => ({ poses: [], faces: [], hands: [] }));
    const pipe = new VisionPipeline(models, fakePixels().pixels);
    for (let i = 0; i < 6; i++) pipe.process(FRAME, 5.0, 0, i);
    for (const ts of [calls.pose, calls.faces, calls.gestures]) {
      for (let i = 1; i < ts.length; i++) expect(ts[i]!).toBeGreaterThan(ts[i - 1]!);
    }
  });

  it("emotion cues: a face pass that misses while the face is held is not a frame without a visible face", () => {
    // Face passes run on odd frames (pass k on frame 2k + 1, every 0.2 s). null = the pass misses the face.
    // Passes 4-6: closed, miss (held: face_visible stays true), open 0.4 s later -> one blink (doc 1.3).
    // Passes 8-11: closed, two misses (frame 22 is past FACE_HOLD_S: face_visible false), open 600 ms later -> the
    // non-visible frame reset the detector, no blink.
    const passes: (number | null)[] = [0.05, 0.05, 0.05, 0.05, 0.9, null, 0.05, 0.05, 0.9, null, null, 0.05, 0.05];
    let closure: number | null = null;
    const { models } = fakeModels(() => ({
      poses: [seated()], hands: [],
      faces: closure === null ? [] : [{ nose: [0.5, 0.3], smile: 0.1, bs: { eyeBlinkLeft: closure, eyeBlinkRight: closure } }],
    }));
    const pipe = new VisionPipeline(models, fakePixels().pixels);
    const persons = Array.from({ length: 2 * passes.length }, (_, i) => {
      closure = passes[i >> 1] ?? null;
      return pipe.process(FRAME, 100 + 0.1 * i, 0, i, i + 1).persons[0]!;
    });
    expect(persons.map((p, i) => p.face_visible === (i >= 1 && i !== 22))).not.toContain(false);
    expect(persons.map((p) => p.blink_onsets)).toEqual(persons.map((_, i) => (i === 13 ? 1 : 0)));
    expect(persons[22]!.eye_open).toBeNaN();
  });

  it("emotion cues: the neutral baseline freezes after 20 s of visible face when every other face pass misses", () => {
    let hit = true;
    let brow = 0.1;
    const { models } = fakeModels(() => ({
      poses: [seated()], hands: [],
      faces: hit ? [{ nose: [0.5, 0.3], smile: 0.1, bs: { browDownLeft: brow, browDownRight: brow } }] : [],
    }));
    const pipe = new VisionPipeline(models, fakePixels().pixels);
    let last: VideoObs | null = null;
    for (let i = 0; i < 700; i++) {
      hit = (i >> 1) % 2 === 0; // face pass i >> 1 on odd frames: every other pass finds the face (every 0.4 s)
      brow = i < 300 ? 0.1 : 0.5; // 30 s at 0.1, then 40 s at 0.5
      last = pipe.process(FRAME, 100 + 0.1 * i, 0, i, i + 1);
      if (i >= 1) expect(last.persons[0]!.face_visible).toBe(true);
    }
    // Frozen at 20 s of visible face with the neutral at 0.1 (a running median would have moved to 0.5: brow_down 0).
    expect(last!.persons[0]!.brow_down).toBeCloseTo(0.4, 9);
  });
});

describe("AudioPipeline (mocked YAMNet)", () => {
  it("Leq in dB SPL, YAMNet on the last 0.975 s, debounced laughter, speech passthrough", () => {
    const seen: number[] = [];
    const laugh = [0, 0.3, 0.4, 0, 0, 0, 0, 0, 0.5];
    let i = 0;
    const classify = vi.fn((samples: Float32Array) => {
      seen.push(samples.length);
      const l = laugh[i++] ?? 0;
      return [{
        timestampMs: 0,
        classifications: [{ headIndex: 0, headName: "", categories: [
          { index: 0, categoryName: "Speech", displayName: "", score: 0.8 },
          { index: 13, categoryName: "Laughter", displayName: "", score: l },
        ] }],
      }];
    });
    const pipe = new AudioPipeline({ classify });
    const block = new Float32Array(16000).fill(0.1);
    const out = laugh.map((_, k) => pipe.process(block, 10 + k, 1.7e9 + k, k));
    expect(seen.every((n) => n === YAMNET_WINDOW)).toBe(true);
    expect(out[0]!.db_spl).toBeCloseTo(10 * Math.log10(0.01) + 94, 4);
    expect(out[0]!.rms_power).toBeCloseTo(0.01, 6);
    expect(out[0]!.speech_prob).toBeCloseTo(0.8);
    expect(out.map((o) => o.laughter_onsets)).toEqual([0, 1, 0, 0, 0, 0, 0, 0, 1]);
    expect(out.every((o) => o.compliment_hits === 0)).toBe(true);
  });
});

describe("KeypointSmoother (One Euro for BlazePose jitter)", () => {
  const frame = (dx: number, noise: () => number) =>
    Float64Array.from({ length: 34 }, (_, i) => (i % 2 ? 300 : 400 + (i === 10 ? 0 : i === 12 ? 200 : i * 5)) + dx + noise());

  it("passes the first sample through, then cuts frame-to-frame jitter on a still person", async () => {
    const { KeypointSmoother } = await import("@/perception/smoothing");
    let seed = 1;
    const noise = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 6; // +-3 px
    const sm = new KeypointSmoother();
    const first = frame(0, noise);
    expect([...sm.update(first, 0)]).toEqual([...first]);
    let raw = 0, smooth = 0;
    let prevRaw: Float64Array = first, prevOut: Float64Array = first;
    for (let i = 1; i < 100; i++) {
      const f = frame(0, noise);
      const out = sm.update(f, i * 0.1);
      raw += Math.abs(f[0]! - prevRaw[0]!);
      smooth += Math.abs(out[0]! - prevOut[0]!);
      prevRaw = f;
      prevOut = out;
    }
    expect(smooth).toBeLessThan(raw / 5);
  });

  it("follows real movement (higher cutoff when moving), holds non-finite points, ignores repeated timestamps", async () => {
    const { KeypointSmoother } = await import("@/perception/smoothing");
    const sm = new KeypointSmoother();
    sm.update(frame(0, () => 0), 0);
    let out: Float64Array = new Float64Array(34);
    for (let i = 1; i <= 20; i++) out = sm.update(frame(i * 20, () => 0), i * 0.1); // 200 px/s, ~1 sw/s
    expect(out[0]!).toBeGreaterThan(400 + 400 * 0.85 - 5); // within ~15% of the true 800 px after 2 s
    const missing = frame(400, () => 0);
    missing[0] = NaN;
    const held = sm.update(missing, 2.1);
    expect(held[0]).toBeNaN();
    const again = sm.update(frame(400, () => 0), 2.1); // same t: nothing changes
    expect(again[2]).toBe(held[2]);
  });

  it("PostureTracker measures motion on the configured points only (default: Python's COCO 5-12)", () => {
    const base = Float64Array.from({ length: 34 }, (_, i) => (i % 2 ? 300 + i : 400 + i * 10));
    const conf = new Float64Array(17).fill(0.99);
    const make = (opts = {}) => new PostureTracker(opts);
    for (const [opts, moving] of [[{}, true], [{ motionPoints: [5, 6] }, false]] as const) {
      const pt = make(opts);
      pt.update(base, conf, 0);
      const wristsMoved = Float64Array.from(base);
      wristsMoved[18] = wristsMoved[18]! + 5; // left wrist x (shoulders 20 px apart: 0.25 sw)
      wristsMoved[20] = wristsMoved[20]! + 5; // right wrist x
      const st = pt.update(wristsMoved, conf, 0.1);
      expect(st.motion > 0).toBe(moving);
    }
  });
});
