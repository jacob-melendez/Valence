/**
 * Mock perception: plausible observations without a camera, at the Python edge's cadence (video at `fps`, ~10 Hz;
 * one audio block per second). Deterministic for a given seed and clock, so tests and demos repeat.
 *
 * Each person loops through a 180 s script (offset per person): focused, upbeat (smiles, laughter in the room),
 * frustrated (brow down, fists), worn out (slump, head in hands, eyes closing). Blendshapes are drawn from that
 * script and face_valence is computed from them with the real formula (face_valence.py), so the extras and the
 * valence agree.
 */
import { FaceCueTracker, NO_CUES, rawCues } from "./faceCues";
import { valenceRaw } from "./faceValence";
import { Hub } from "./hub";
import {
  BLENDSHAPE_NAMES, emptyCounters, idleStatus, type AudioObs, type BBox, type FaceRaw, type Obs, type Perception,
  type PerceptionCounters, type PerceptionStatus, type PersonObs, type VideoObs,
} from "./types";

export interface MockClock {
  /** Monotonic seconds. */
  now(): number;
  /** Unix seconds. */
  wall(): number;
}

export interface MockPerceptionOptions {
  seed?: number;
  /** People in view, 0..3 (default 1). */
  people?: number;
  /** Video observations per second (default 10). */
  fps?: number;
  /** Seconds per audio block (default 1). */
  audioBlockS?: number;
  splOffsetDb?: number;
  frameW?: number;
  frameH?: number;
  clock?: MockClock;
}

const SCRIPT_S = 180;
type Phase = "focused" | "upbeat" | "frustrated" | "worn_out";
const phaseAt = (s: number): Phase => {
  const x = ((s % SCRIPT_S) + SCRIPT_S) % SCRIPT_S;
  return x < 60 ? "focused" : x < 100 ? "upbeat" : x < 140 ? "frustrated" : "worn_out";
};

/** mulberry32: small, fast, deterministic. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const round = (v: number, digits = 4) => Math.round(v * 10 ** digits) / 10 ** digits;

interface MockPerson {
  id: number;
  offsetS: number;
  cx: number;
  width: number;
  jitter: [number, number];
  motion: number;
  dropoutUntil: number;
  fistActiveUntil: number;
  nextFistAt: number;
  slumpSince: number | null;
  nextBlinkAt: number;
  baseline: number;
  cues: FaceCueTracker;
}

export function createMockPerception(opts: MockPerceptionOptions = {}): Perception {
  const fps = opts.fps ?? 10;
  const blockS = opts.audioBlockS ?? 1;
  const offset = opts.splOffsetDb ?? 94;
  const frameW = opts.frameW ?? 1280;
  const frameH = opts.frameH ?? 720;
  const clock: MockClock = opts.clock ?? { now: () => performance.now() / 1000, wall: () => Date.now() / 1000 };
  const nPeople = clamp(Math.round(opts.people ?? 1), 0, 3);

  const obsHub = new Hub<Obs>();
  const statusHub = new Hub<PerceptionStatus>();
  let status = idleStatus(true);
  let counters = emptyCounters();
  let timers: ReturnType<typeof setInterval>[] = [];
  let rand = rng(opts.seed ?? 7);
  let people: MockPerson[] = [];
  let t0 = 0;
  let frameSeq = 0;
  let blockSeq = 0;
  let lastLuxS = -Infinity;
  let lux = NaN;
  let luxQ = 0;
  let laughUntil = -Infinity;
  let nextLaughAt = 0;

  const gauss = () => {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };

  const setStatus = (patch: Partial<PerceptionStatus>) => {
    status = { ...status, ...patch };
    statusHub.emit(status);
  };

  const makePeople = (): MockPerson[] => {
    const centres = nPeople === 1 ? [0.5] : nPeople === 2 ? [0.3, 0.7] : [0.2, 0.5, 0.8];
    return centres.slice(0, nPeople).map((cx, i) => ({
      id: i + 1, offsetS: i * 47, cx, width: nPeople === 3 ? 0.22 : 0.3, jitter: [0, 0], motion: 0.04,
      dropoutUntil: -Infinity, fistActiveUntil: -Infinity, nextFistAt: 105 + i * 47, slumpSince: null,
      nextBlinkAt: 2 + i, baseline: 0.02, cues: new FaceCueTracker(),
    }));
  };

  const faceRaw = (p: MockPerson, phase: Phase, s: number): FaceRaw => {
    const target = { focused: 0.02, upbeat: 0.55, frustrated: -0.45, worn_out: -0.2 }[phase];
    const v = target + 0.08 * gauss();
    const blink = s >= p.nextBlinkAt && s < p.nextBlinkAt + 0.15;
    if (s >= p.nextBlinkAt + 0.15) p.nextBlinkAt = s + (phase === "worn_out" ? 1.5 : 3.5) + 2 * rand();
    const eyesHeavy = phase === "worn_out" ? 0.35 : 0.05;
    const bs: Record<string, number> = {};
    for (const name of BLENDSHAPE_NAMES) bs[name] = clamp(0.03 * rand(), 0, 1);
    const both = (stem: string, value: number) => {
      bs[`${stem}Left`] = clamp(value + 0.02 * gauss(), 0, 1);
      bs[`${stem}Right`] = clamp(value + 0.02 * gauss(), 0, 1);
    };
    both("mouthSmile", Math.max(0, v) * 0.9);
    both("cheekSquint", Math.max(0, v) * 0.4);
    both("mouthFrown", Math.max(0, -v) * 0.35);
    both("browDown", Math.max(0, -v) * 0.6);
    both("noseSneer", Math.max(0, -v) * 0.15);
    both("mouthPress", phase === "frustrated" ? 0.25 : 0.05);
    both("eyeBlink", blink ? 0.92 : eyesHeavy);
    both("eyeSquint", eyesHeavy * 0.6);
    bs["browInnerUp"] = clamp(phase === "frustrated" ? 0.2 : 0.05, 0, 1);
    bs["jawOpen"] = clamp(phase === "upbeat" ? 0.15 + 0.1 * rand() : 0.03, 0, 1);
    bs["_neutral"] = clamp(1 - Math.abs(v), 0, 1);
    for (const k of Object.keys(bs)) bs[k] = round(bs[k]!);
    return {
      blendshapes: bs,
      head_yaw_deg: round(8 * Math.sin((s + p.offsetS) / 9) + 2 * gauss(), 2),
      head_pitch_deg: round((phase === "worn_out" ? -14 : -3) + 2 * gauss(), 2),
      head_roll_deg: round(2 * gauss(), 2),
    };
  };

  const person = (p: MockPerson, s: number): PersonObs => {
    const ps = s + p.offsetS;
    const phase = phaseAt(ps);
    p.jitter = [clamp(p.jitter[0] + 0.002 * gauss(), -0.02, 0.02), clamp(p.jitter[1] + 0.002 * gauss(), -0.02, 0.02)];
    const half = p.width / 2;
    const slumpDrop = phase === "worn_out" ? 0.06 : 0;
    const bbox: BBox = [
      round(clamp(p.cx - half + p.jitter[0], 0, 1)), round(clamp(0.12 + slumpDrop + p.jitter[1], 0, 1)),
      round(clamp(p.cx + half + p.jitter[0], 0, 1)), round(clamp(0.98, 0, 1)),
    ];

    if (s >= p.dropoutUntil && rand() < 0.01) p.dropoutUntil = s + 0.5 + 1.5 * rand();
    const visible = s >= p.dropoutUntil;
    const raw = visible ? faceRaw(p, phase, s) : null;
    const faceValence = raw ? round(clamp(Math.tanh(2.5 * (valenceRaw(raw.blendshapes) - p.baseline)), -1, 1)) : NaN;
    const cues = raw
      ? p.cues.update({ ...rawCues(raw.blendshapes, null), yaw: raw.head_yaw_deg, pitch: raw.head_pitch_deg }, s * 1000)
      : (p.cues.update(null, s * 1000), NO_CUES);

    let onsets = 0;
    if (phase === "frustrated" && s >= p.nextFistAt) {
      onsets = 1;
      p.fistActiveUntil = s + 1.5;
      p.nextFistAt = s + 9 + 6 * rand();
    }
    if (phase !== "frustrated" && p.nextFistAt <= s) p.nextFistAt = s + 1;

    const slumped = phase === "worn_out";
    p.slumpSince = slumped ? (p.slumpSince ?? s) : null;
    const slump = p.slumpSince !== null && s - p.slumpSince >= 3;
    const inPhase = ((ps % SCRIPT_S) + SCRIPT_S) % SCRIPT_S;
    const headInHands = slumped && inPhase >= 150 && inPhase < 158;

    const targetMotion = { focused: 0.05, upbeat: 0.1, frustrated: 0.14, worn_out: 0.02 }[phase];
    p.motion = 0.3 * clamp(targetMotion + 0.03 * gauss(), 0, 1) + 0.7 * p.motion;

    return {
      track_id: p.id, bbox, fist_onsets: onsets, fist_active: s < p.fistActiveUntil, slump, head_in_hands: headInHands,
      motion: round(p.motion), face_visible: visible, face_valence: faceValence, ...cues, raw,
    };
  };

  const videoTick = () => {
    const t = clock.now();
    const s = t - t0;
    counters.frames_decoded += 1;
    counters.frames_purged += 1;
    if (s - lastLuxS >= 1) {
      lastLuxS = s;
      lux = round(320 + 20 * Math.sin(s / 40) + 4 * gauss(), 1);
      luxQ = round(clamp(0.42 + 0.02 * gauss(), 0, 1), 3);
      counters.derived_purged += 1;
    }
    const obs: VideoObs = {
      type: "video", t, wall: clock.wall(), frame_seq: frameSeq++, frames_decoded: counters.frames_decoded, lux,
      lux_quality: luxQ, frame_w: frameW, frame_h: frameH, mirrored: false, persons: people.map((p) => person(p, s)),
    };
    obsHub.emit(obs);
  };

  const audioTick = () => {
    const t = clock.now();
    const s = t - t0;
    const phase = nPeople ? phaseAt(s + (people[0]?.offsetS ?? 0)) : "focused";
    const talking = rand() < (phase === "upbeat" ? 0.7 : phase === "frustrated" ? 0.5 : 0.25);
    let laughter = 0;
    if (phase === "upbeat" && s >= nextLaughAt) {
      laughter = 1;
      laughUntil = s + 2;
      nextLaughAt = s + 12 + 10 * rand();
    }
    const db = round(46 + (talking ? 10 + 3 * rand() : 2 * rand()) + (s < laughUntil ? 8 : 0) + gauss(), 2);
    const parts = Math.round(blockS * 10);
    counters.audio_blocks_purged += parts;
    counters.derived_purged += 2;
    const obs: AudioObs = {
      type: "audio", t, wall: clock.wall(), block_seq: blockSeq++, db_spl: db,
      rms_power: 10 ** ((db - offset) / 10), speech_prob: round(talking ? 0.6 + 0.35 * rand() : 0.05 * rand(), 3),
      laughter_onsets: laughter, compliment_hits: 0,
    };
    obsHub.emit(obs);
  };

  return {
    async start() {
      if (status.phase === "running") return;
      rand = rng(opts.seed ?? 7);
      people = makePeople();
      counters = emptyCounters();
      t0 = clock.now();
      frameSeq = blockSeq = 0;
      lastLuxS = -Infinity;
      lux = NaN;
      luxQ = 0;
      laughUntil = -Infinity;
      nextLaughAt = 0;
      timers = [setInterval(videoTick, 1000 / fps), setInterval(audioTick, 1000 * blockS)];
      setStatus({ ...idleStatus(true), phase: "running", camera: true, mic: true, fps, load_ms: 0 });
    },
    stop() {
      for (const id of timers) clearInterval(id);
      timers = [];
      if (status.phase === "running") setStatus({ phase: "stopped", camera: false, mic: false, fps: 0 });
    },
    onObservation: (cb) => obsHub.on(cb),
    counters: (): PerceptionCounters => ({ ...counters }),
    status: () => status,
    onStatus: (cb) => statusHub.on(cb),
  };
}
