/**
 * aggregator.ts + replay.ts against valence/runtime/aggregator.py and recording.replay_offline
 * (scripts/export_web_golden.py -> aggregator.json.gz), plus TS ports of the aggregator tests in
 * tests/test_runtime.py.
 *
 * Records must match per tick: counts, flags, ids and strings exactly, the NaN pattern exactly, other
 * numbers within 1e-9 (only ambient_db goes through pow/log10, where V8 and numpy may differ in the last bit;
 * every other field is required to be bit-exact). Features at the checkpoints go through features.ts.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { Aggregator, localTod, roomRecord, type RoomFields } from "@/engine/aggregator";
import {
  makeAudioObs,
  makePersonObs,
  makeSettings,
  makeTickRecord,
  makeVideoObs,
  obsFromDict,
  obsToDict,
  ROOM,
  TICK_FIELDS,
  tickRecordToDict,
  type Obs,
  type PersonObs,
  type Settings,
  type TickRecord,
  type TickRecordDict,
  type VideoObs,
} from "@/engine/contracts";
import { compute, featureIndex, featuresFromWindow } from "@/engine/features";
import { parseJsonl, replayInto, replayOffline, restamp, type ReplayTick } from "@/engine/replay";
import { RollingWindow } from "@/engine/window";
import { pyRepr } from "@/engine/num";
import { sha256Hex } from "@/engine/sha";
import { FRONTEND_ROOT, loadFixture, nn } from "./engine-fixtures";

// ---------- Fixture ----------

interface Checkpoint {
  tick: number;
  person_id: number;
  features: (number | null)[];
}
interface AggCase {
  name: string;
  kind: "recording" | "synthetic";
  wall0: number;
  mono0: number;
  tz: string;
  tick_s: number;
  session_id: string;
  source: string;
  settings: Record<string, unknown>;
  obs: Record<string, unknown>[];
  ticks: { wall: number; mono: number; records: TickRecordDict[] }[];
  checkpoints: Checkpoint[];
}
interface AggFixture {
  version: number;
  settings: Record<string, unknown>;
  cases: AggCase[];
}

const fixture = loadFixture<AggFixture>("aggregator");

/** Fields that pass through Math.pow / Math.log10 (tick Leq): allowed to differ in the last bits. */
const TRANSCENDENTAL = new Set<string>(["ambient_db"]);
const TOL = 1e-9;

// ---------- Replay like the exporter's stand-in runtime ----------

interface CaseRun {
  ticks: ReplayTick[];
  checkpoints: { tick: number; person_id: number; features: Float64Array }[];
}

/** Runtime._evict on a bare window (engine.py:346). */
function evict(window: RollingWindow, settings: Settings, wall: number): void {
  for (const pid of window.people()) {
    const rows = window.snapshot(pid);
    const last = rows[rows.length - 1];
    if (last !== undefined && wall - last.ts > settings.person_drop_min * 60) window.drop(pid);
  }
}

function runCase(c: AggCase): CaseRun {
  const settings = makeSettings(c.settings as Partial<Settings>);
  const window = new RollingWindow(settings.window_ticks);
  const checkpoints: CaseRun["checkpoints"] = [];
  const ticks = replayOffline(
    c.obs.map((d) => obsFromDict(d)),
    {
      wall0: c.wall0,
      mono0: c.mono0,
      settings,
      sessionId: c.session_id,
      source: c.source,
      tz: c.tz,
      onTick: (tick, i) => {
        for (const r of tick.records) window.append(r);
        evict(window, settings, tick.wall);
        const targets = [
          ...new Set(tick.records.map((r) => r.person_id).filter((p) => p !== ROOM)),
        ].sort((a, b) => a - b);
        for (const pid of targets) {
          const person = window.snapshot(pid);
          if (person.length === 0) continue;
          checkpoints.push({
            tick: i,
            person_id: pid,
            features: featuresFromWindow(person, window.room()),
          });
        }
      },
    },
  );
  return { ticks, checkpoints };
}

const runs = new Map<string, CaseRun>(fixture.cases.map((c) => [c.name, runCase(c)]));
const runOf = (c: AggCase): CaseRun => runs.get(c.name) as CaseRun;

/** Max |error| per float field, and every hard mismatch, for one case. */
function compareRecords(
  c: AggCase,
  run: CaseRun,
): { bad: string[]; maxErr: Record<string, number> } {
  const bad: string[] = [];
  const maxErr: Record<string, number> = {};
  if (run.ticks.length !== c.ticks.length)
    bad.push(`${run.ticks.length} ticks, expected ${c.ticks.length}`);
  c.ticks.forEach((exp, i) => {
    const got = run.ticks[i];
    if (got === undefined) return;
    if (!Object.is(got.wall, exp.wall) || !Object.is(got.mono, exp.mono)) {
      bad.push(`tick ${i}: clock ${got.wall}/${got.mono}, expected ${exp.wall}/${exp.mono}`);
    }
    if (got.records.length !== exp.records.length) {
      bad.push(`tick ${i}: ${got.records.length} records, expected ${exp.records.length}`);
      return;
    }
    exp.records.forEach((e, j) => {
      const a = tickRecordToDict(got.records[j] as TickRecord) as unknown as Record<
        string,
        unknown
      >;
      const er = e as unknown as Record<string, unknown>;
      if (JSON.stringify(Object.keys(a)) !== JSON.stringify(Object.keys(er)))
        bad.push(`tick ${i} row ${j}: key set differs`);
      for (const k of TICK_FIELDS) {
        const ev = er[k];
        const av = a[k];
        const where = `tick ${i} pid ${String(er["person_id"])} ${k}`;
        if (typeof ev === "number" && typeof av === "number") {
          const d = Math.abs(av - ev);
          maxErr[k] = Math.max(maxErr[k] ?? 0, d);
          // bit-exact (Object.is: signed zeros too), except the pow/log10 fields
          const ok = Object.is(av, ev) || (TRANSCENDENTAL.has(k) && d <= TOL);
          if (!ok) bad.push(`${where}: ${String(av)} vs ${String(ev)}`);
        } else if (JSON.stringify(av) !== JSON.stringify(ev)) {
          bad.push(`${where}: ${JSON.stringify(av)} vs ${JSON.stringify(ev)}`); // null pattern, ints, strings, lists
        }
      }
    });
  });
  return { bad, maxErr };
}

/** Feature mismatches at the checkpoints: same (tick, person) list, NaN pattern, |err| <= 1e-9 * max(1, |x|). */
function compareCheckpoints(c: AggCase, run: CaseRun): { bad: string[]; maxErr: number } {
  const bad: string[] = [];
  let maxErr = 0;
  const key = (cp: { tick: number; person_id: number }): string => `${cp.tick}/${cp.person_id}`;
  const got = run.checkpoints.map(key).join(",");
  const exp = c.checkpoints.map(key).join(",");
  if (got !== exp) bad.push(`checkpoints ${got} vs ${exp}`);
  c.checkpoints.forEach((cp, i) => {
    const g = run.checkpoints[i];
    if (g === undefined) return;
    const want = nn(cp.features);
    want.forEach((w, f) => {
      const v = g.features[f] as number;
      if (Number.isNaN(w) || Number.isNaN(v)) {
        if (Number.isNaN(w) !== Number.isNaN(v)) bad.push(`${key(cp)} feature ${f}: ${v} vs ${w}`);
        return;
      }
      const d = Math.abs(v - w);
      maxErr = Math.max(maxErr, d);
      if (!(d <= TOL * Math.max(1, Math.abs(w)))) bad.push(`${key(cp)} feature ${f}: ${v} vs ${w}`);
    });
  });
  return { bad, maxErr };
}

// ---------- Golden parity ----------

describe("aggregator golden replay", () => {
  it("covers every capture and synthetic stream", () => {
    expect(fixture.version).toBe(1);
    expect(fixture.cases.map((c) => c.kind)).toContain("recording");
    expect(fixture.cases.map((c) => c.kind)).toContain("synthetic");
    expect(fixture.cases.length).toBe(14);
  });

  for (const c of fixture.cases) {
    it(`${c.name}: produces the reference TickRecords on every tick`, () => {
      const { bad } = compareRecords(c, runOf(c));
      expect(bad.slice(0, 10)).toEqual([]);
    });

    it(`${c.name}: produces the reference features at every checkpoint`, () => {
      const { bad } = compareCheckpoints(c, runOf(c));
      expect(bad.slice(0, 10)).toEqual([]);
    });
  }

  it("is bit-exact everywhere except the pow/log10 Leq, which stays within 1e-9", () => {
    const worst: Record<string, number> = {};
    let featErr = 0;
    for (const c of fixture.cases) {
      for (const [k, v] of Object.entries(compareRecords(c, runOf(c)).maxErr))
        worst[k] = Math.max(worst[k] ?? 0, v);
      featErr = Math.max(featErr, compareCheckpoints(c, runOf(c)).maxErr);
    }
    for (const [k, v] of Object.entries(worst))
      expect(v, k).toBeLessThanOrEqual(TRANSCENDENTAL.has(k) ? TOL : 0);
    expect(featErr).toBeLessThanOrEqual(TOL * 1e3);
  });
});

// ---------- Ports of tests/test_runtime.py ----------

const video = (t: number, persons: PersonObs[], seq: number): VideoObs =>
  makeVideoObs({
    t,
    wall: 1_000_000 + t,
    frame_seq: seq,
    frames_decoded: seq,
    lux: 300.0,
    lux_quality: 0.8,
    frame_w: 1280,
    frame_h: 720,
    persons,
  });

describe("aggregator tick", () => {
  it("aggregates one person's tick and gates far-field laughter", () => {
    const agg = new Aggregator(makeSettings(), "s");
    for (let i = 0; i < 50; i++) {
      // 10 Hz for 5 s, one person, two fist onsets, half the frames slumped
      const p = makePersonObs({
        track_id: 1,
        bbox: [0.1, 0.1, 0.5, 0.9],
        fist_onsets: i === 10 || i === 30 ? 1 : 0,
        slump: i % 2 === 0,
        motion: 0.05,
        face_visible: i < 40,
        face_valence: i < 40 ? 0.4 : NaN,
      });
      agg.push(video(i * 0.1, [p], i + 1));
    }
    for (let i = 0; i < 20; i++) {
      // quiet room at 50 dB
      agg.push(
        makeAudioObs({ t: i, wall: 1_000_000 + i, block_seq: i, db_spl: 50.0, rms_power: 1e-5 }),
      );
    }
    agg.push(
      makeAudioObs({
        t: 21,
        wall: 1_000_021,
        block_seq: 21,
        db_spl: 51.0,
        rms_power: 1e-5,
        laughter_onsets: 1,
      }),
    ); // far away
    agg.push(
      makeAudioObs({
        t: 22,
        wall: 1_000_022,
        block_seq: 22,
        db_spl: 62.0,
        rms_power: 1e-4,
        compliment_hits: 1,
      }),
    ); // at the desk
    const [room, person] = agg.flush(1_000_005, 5, 15.5) as [TickRecord, TickRecord];
    expect(person.clenched_fists_count).toBe(2);
    expect(person.posture_slump).toBeCloseTo(0.5, 12);
    expect(person.face_visible).toBeCloseTo(0.8, 12);
    expect(person.face_valence).toBeCloseTo(0.4, 12);
    expect([person.laughter_count, person.compliments_count, agg.gatedOut]).toEqual([0, 1, 1]);
    expect(room.person_id).toBe(0);
    expect(room.clenched_fists_count).toBe(2);
    expect(person.ambient_db).toBeGreaterThan(50);
    expect(person.ambient_db).toBeLessThan(53);
  });

  it("restarts the break clock after an absence and seeds it on request", () => {
    const agg = new Aggregator(makeSettings({ break_absent_min: 1 }), "s");
    const p = makePersonObs({ track_id: 1, bbox: [0.1, 0.1, 0.5, 0.9] });
    agg.push(video(0, [p], 1));
    agg.seedHours(4.2, 1_000_000);
    expect(agg.hoursSinceBreak(1, 1_000_000)).toBeCloseTo(4.2, 12);
    agg.push(
      makeVideoObs({ t: 200, wall: 1_000_200, frame_seq: 2, frames_decoded: 2, persons: [p] }),
    ); // back after 200 s
    expect(agg.hoursSinceBreak(1, 1_000_200)).toBe(0.0);
    expect(agg.hoursSinceBreak(9, 1_000_200)).toBeNaN();
  });

  it("tags a seeded break clock as simulated in the person's record", () => {
    const agg = new Aggregator(makeSettings(), "s");
    agg.push(video(0, [makePersonObs({ track_id: 1, bbox: [0.1, 0.1, 0.5, 0.9] })], 1));
    agg.seedHours(2.0, 1_000_000);
    const [room, person] = agg.flush(1_000_005, 5, 12.0) as [TickRecord, TickRecord];
    expect(person.simulated_fields).toEqual(["hours_since_break"]);
    expect(room.simulated_fields).toEqual(["hours_since_break"]);
    expect(person.hours_since_break).toBeCloseTo(2.0 + 5 / 3600, 12);
  });

  it("does not count background tracks as participants", () => {
    const agg = new Aggregator(makeSettings(), "s");
    for (let i = 0; i < 50; i++) {
      const desk = makePersonObs({
        track_id: 1,
        bbox: [0.2, 0.05, 0.7, 0.98],
        motion: 0.03,
        face_visible: true,
        face_valence: 0.0,
      });
      const background = makePersonObs({ track_id: 3, bbox: [0.8, 0.3, 0.9, 0.62] }); // a third of the height, always there
      agg.push(video(i * 0.1, [desk, background], i + 1));
    }
    const [room, ...people] = agg.flush(1_000_005, 5, 12.0) as [TickRecord, ...TickRecord[]];
    expect(people.map((p) => p.person_id)).toEqual([1]);
    expect(room.n_people).toBe(1);
    expect(people[0]?.n_people).toBe(1);
  });

  it("sums the room row over participants only (room_record)", () => {
    const base: RoomFields = {
      tick: 0,
      ts: 1,
      t_mono: 1,
      session_id: "s",
      n_people: 2,
      source: "live",
      ambient_db: NaN,
      db_p90: NaN,
      light_lux: NaN,
      lux_quality: 0,
      room_db_base: NaN,
      room_lux_base: NaN,
      laughter_count: 0,
      compliments_count: 0,
      tod_h: 1,
    };
    const a = makeTickRecord({
      ...base,
      person_id: 1,
      display_id: "Person #1",
      clenched_fists_count: 2,
      face_valence: 0.5,
      face_visible: 1.0,
      posture_slump: 0.7,
      motion_energy: 0.01,
      social_withdrawal_flag: 1,
      simulated_fields: ["z"],
    });
    const b = makeTickRecord({
      ...base,
      person_id: 2,
      display_id: "Person #2",
      clenched_fists_count: 1,
      face_visible: 0.0,
      posture_slump: 0.1,
      motion_energy: NaN,
      simulated_fields: ["a"],
    });
    const room = roomRecord(base, [a, b], ["m"]);
    expect([
      room.person_id,
      room.display_id,
      room.clenched_fists_count,
      room.social_withdrawal_flag,
    ]).toEqual([0, "Room", 3, 1]);
    expect(room.simulated_fields).toEqual(["a", "m", "z"]);
    expect(room.face_valence).toBe(0.5);
    expect(room.face_visible).toBe(0.5);
    expect(room.motion_energy).toBe(0.01);
    expect(room.hours_since_break).toBeNaN();
    expect(roomRecord(base, []).face_visible).toBe(0.0);
  });
});

describe("room baseline", () => {
  it("captures the venue baseline, carries it in records and gates on it", () => {
    const agg = new Aggregator(makeSettings({ room_baseline_s: 10 }), "t");
    expect(agg.baselineStatus().mode).toBe("auto");
    agg.startBaseline(10);
    for (let i = 0; i < 12; i++) {
      agg.push(
        makeAudioObs({
          t: i,
          wall: 1000.0 + i,
          block_seq: i,
          db_spl: 70.0 + (i % 3) - 1,
          rms_power: 1.0,
        }),
      );
      agg.push(
        makeVideoObs({ t: i, wall: 1000.0 + i, frame_seq: i, lux: 300.0, lux_quality: 1.0 }),
      );
    }
    const snap = agg.baselineStatus();
    expect([snap.mode, snap.db_median, snap.lux_median]).toEqual(["captured", 70.0, 300.0]);
    // gate: the loudest of the last 3 blocks must reach baseline + 3 dB
    agg.push(
      makeAudioObs({
        t: 13.0,
        wall: 1013.0,
        block_seq: 13,
        db_spl: 72.0,
        rms_power: 1.0,
        laughter_onsets: 1,
      }),
    ); // max 72 < 73
    agg.push(makeAudioObs({ t: 14.0, wall: 1014.0, block_seq: 14, db_spl: 78.0, rms_power: 1.0 })); // the user's laugh...
    agg.push(
      makeAudioObs({
        t: 15.0,
        wall: 1015.0,
        block_seq: 15,
        db_spl: 70.0,
        rms_power: 1.0,
        laughter_onsets: 1,
      }),
    ); // ...reported late
    const room = agg.flush(1016.0, 16.0, 12.0)[0] as TickRecord;
    expect([room.room_db_base, room.room_lux_base]).toEqual([70.0, 300.0]);
    expect([room.laughter_count, agg.gatedOut]).toEqual([1, 1]);
  });

  it("makes room-relative features follow the baseline", () => {
    const n = 30;
    const full = (v: number): Float64Array => new Float64Array(n).fill(v);
    const person: Record<string, Float64Array> = {};
    for (const f of [
      "motion_energy",
      "posture_slump",
      "head_in_hands_frac",
      "clenched_fists_count",
      "social_withdrawal_flag",
      "n_people",
      "hours_since_break",
      "tod_h",
      "face_valence",
      "face_visible",
    ])
      person[f] = full(0.0);
    const room: Record<string, Float64Array> = {
      ambient_db: full(72.0),
      light_lux: full(150.0),
      laughter_count: full(0),
      compliments_count: full(0),
      room_db_base: full(66.0),
      room_lux_base: full(300.0),
    };
    const x = compute(person, room);
    expect(x[featureIndex("db_room_now")]).toBeCloseTo(6.0, 12);
    expect(x[featureIndex("lux_room_mean_5m")]).toBeCloseTo(-1.0, 12); // half the baseline light
    room["room_db_base"] = full(NaN); // no baseline yet: relative features are missing, not zero
    expect(compute(person, room)[featureIndex("db_room_now")]).toBeNaN();
  });

  it("samples lux at 1 Hz and flags a drifted room", () => {
    const agg = new Aggregator(makeSettings({ room_baseline_s: 10 }), "t");
    for (let i = 0; i < 50; i++) {
      // 5 s of video at 10 Hz
      agg.push(
        makeVideoObs({
          t: i / 10,
          wall: 1000.0 + i / 10,
          frame_seq: i,
          lux: 300.0,
          lux_quality: 1.0,
        }),
      );
    }
    const samples = agg.baselineStatus().lux_samples;
    expect(samples).toBeGreaterThanOrEqual(4);
    expect(samples).toBeLessThanOrEqual(6);
    for (let i = 0; i < 12; i++)
      agg.push(
        makeAudioObs({ t: i, wall: 1000.0 + i, block_seq: i, db_spl: 50.0, rms_power: 1.0 }),
      );
    expect(agg.baselineStatus().complete).toBe(true);
    expect(agg.baselineStatus().drifted).toBe(false);
    for (let i = 12; i < 400; i++) {
      // the hall gets 20 dB louder for several minutes
      agg.push(
        makeAudioObs({ t: i, wall: 1000.0 + i, block_seq: i, db_spl: 70.0, rms_power: 1.0 }),
      );
    }
    const snap = agg.baselineStatus();
    expect(snap.drifted).toBe(true);
    expect(snap.drift_db).toBe(20.0);
  });

  it("completes an operator capture on light alone when there is no microphone", () => {
    const agg = new Aggregator(makeSettings({ room_baseline_s: 3 }), "t");
    agg.startBaseline();
    for (let i = 0; i < 4; i++)
      agg.push(
        makeVideoObs({ t: i, wall: 1000.0 + i, frame_seq: i, lux: 200.0 + i, lux_quality: 1.0 }),
      );
    const snap = agg.baselineStatus();
    expect([snap.mode, snap.complete, snap.lux_samples, snap.lux_median, snap.db_median]).toEqual([
      "captured",
      false,
      3,
      201.0,
      null,
    ]);
  });
});

// ---------- Replay ----------

describe("offline replay", () => {
  it("steps once per 5 s tick of a recording", () => {
    // tests/test_runtime.py test_recording_replay_offline: 60 s of video at 10 Hz plus 1 Hz audio, via JSONL text
    const lines: string[] = [];
    for (let i = 0; i < 600; i++) {
      const p = makePersonObs({
        track_id: 1,
        bbox: [0.2, 0.1, 0.6, 0.95],
        motion: 0.07,
        face_visible: true,
        face_valence: -0.1,
      });
      lines.push(JSON.stringify(obsToDict(video(i * 0.1, [p], i + 1))));
      if (i % 10 === 0) {
        lines.push(
          JSON.stringify(
            obsToDict(
              makeAudioObs({
                t: i * 0.1,
                wall: 1_000_000 + i * 0.1,
                block_seq: i,
                db_spl: 47.0,
                rms_power: 1e-5,
              }),
            ),
          ),
        );
      }
    }
    const ticks = replayOffline(parseJsonl(lines.join("\n") + "\n"), {
      wall0: 1792850400.0,
      mono0: 1000.0,
      tz: "UTC",
    });
    expect(ticks.length).toBeGreaterThanOrEqual(11);
    expect(ticks.length).toBeLessThanOrEqual(13);
    const people = ticks.flatMap((t) => t.records.filter((r) => r.person_id !== ROOM));
    expect(people.every((r) => r.person_id === 1 && r.source === "replay")).toBe(true);
    // the break clock follows the replay clock, not the capture's
    expect(Math.max(...people.map((r) => r.hours_since_break))).toBeLessThan(1.0);
    expect(ticks[0]?.records[0]?.tod_h).toBeCloseTo(14 + 5 / 3600, 12);
  });

  it("restamps observations onto the replay clock without touching the original", () => {
    const o = video(3.5, [], 1);
    const r = restamp(o, 10.0, 20.0);
    expect([r.wall, r.t, o.wall, o.t]).toEqual([10.0, 20.0, 1_000_003.5, 3.5]);
    const steps: number[][] = [];
    const pushed: number[] = [];
    replayInto(
      [video(100.0, [], 1), video(104.0, [], 2), video(111.0, [], 3)],
      {
        push: (x) => pushed.push(x.wall),
        step: (wall, mono) => steps.push([wall, mono]),
      },
      { wall0: 50.0, mono0: 7.0, tickS: 5.0 },
    );
    expect(pushed).toEqual([50.0, 54.0, 61.0]);
    expect(steps).toEqual([
      [55.0, 12.0],
      [60.0, 17.0],
      [65.0, 22.0],
    ]);
    replayInto(
      [],
      { push: () => pushed.push(-1), step: () => steps.push([-1]) },
      { wall0: 0, mono0: 0, tickS: 5 },
    );
    expect(steps.length).toBe(3);
  });
});

describe("local time of day", () => {
  it("rounds to whole microseconds like datetime.fromtimestamp before reading the seconds", () => {
    // reference values: TZ=UTC python -c "from valence.runtime.engine import local_tod; ..."
    expect(localTod(1792850404.9999995, "UTC")).toBe(14.001388888888888);
    expect(localTod(1792850400.0000005, "UTC")).toBe(14.0);
    expect(localTod(1792850410.25, "UTC")).toBe(14.002777777777778);
    expect(localTod(1792886399.9999995, "UTC")).toBe(0.0); // carries into the next day
    expect(localTod(1792886399.9999993, "UTC")).toBe(23.999722222222225);
    expect(localTod(0.0, "UTC")).toBe(0.0);
    expect(localTod(-0.5, "UTC")).toBe(23.999722222222225);
    expect(localTod(-1.0000004, "UTC")).toBe(23.999722222222225);
  });

  it("reads the hour in a named time zone", () => {
    expect(localTod(1792850400.0, "America/New_York")).toBe(10.0);
    expect(localTod(1792850404.9999995, "America/New_York")).toBe(10.001388888888888);
    expect(localTod(1700000000.0, "America/New_York")).toBe(17.22222222222222);
    expect(localTod(1792850400.0, "Asia/Kolkata")).toBe(19.5);
  });
});

// ---------- Adversarial parity probes ----------
//
// Inputs the golden fixture does not reach: operator actions mid-stream, Python truthiness on NaN and
// -0.0, int() on the baseline length, inputs where Python raises, and JSONL text quirks. Every literal below
// was produced by the .venv Python reference (valence/runtime/aggregator.py, recording.py, engine.py).

/** Park-Miller minimal standard generator: exact in doubles (products stay below 2**47), same in Python. */
class ParkMiller {
  constructor(private s: number) {}
  u(): number {
    this.s = (this.s * 48271) % 2147483647;
    return this.s / 2147483647;
  }
}

const D_WALL0 = 1792850400.0;
const D_MONO0 = 1000.0;
const D_TICK_S = 5.0;
const D_TICKS = 56;

function diffPerson(rng: ParkMiller, pid: number, t: number): PersonObs {
  const h = ({ 1: 0.8, 2: 0.6, 3: 0.25, 4: 0.7 } as Record<number, number>)[pid] as number;
  const x1 = 0.1 * (pid - 1) + 0.05 * rng.u();
  const y1 = 0.05 + 0.1 * rng.u();
  const y2 = Math.min(1.0, y1 + h * (0.9 + 0.2 * rng.u()));
  const late = pid === 1 && t > 200.0;
  const vis = rng.u() < 0.8;
  const nanval = rng.u() < 0.05;
  const cue = (scale: number): number => (rng.u() < 0.1 ? NaN : (rng.u() * 2 - 1) * scale);
  const fist_onsets = rng.u() < 0.02 ? 1 : 0;
  const fist_active = rng.u() < 0.1;
  const slump = rng.u() < (late ? 0.9 : 0.4);
  const head_in_hands = rng.u() < 0.1;
  const motion = rng.u() < 0.1 ? NaN : rng.u() * (late ? 0.01 : 0.05);
  const face_valence = vis ? (nanval ? NaN : rng.u() * 2 - 1) : NaN;
  const blink_onsets = rng.u() < 0.05 ? 1 : 0;
  const p: Partial<PersonObs> & Pick<PersonObs, "track_id" | "bbox"> = {
    track_id: pid,
    bbox: [x1, y1, x1 + 0.3, y2],
    fist_onsets,
    fist_active,
    slump,
    head_in_hands,
    motion,
    face_visible: vis,
    face_valence,
    blink_onsets,
  };
  if (vis) {
    p.brow_down = cue(0.3);
    p.eye_open = cue(0.2);
    p.mouth_frown = cue(0.3);
    p.smile = cue(0.5);
    p.head_yaw = cue(30.0);
    p.head_pitch = cue(25.0);
  }
  return makePersonObs(p);
}

function diffPresent(rng: ParkMiller, pid: number, t: number): boolean {
  if (pid === 1) return (t < 120.0 || t >= 160.0) && rng.u() >= 0.1;
  if (pid === 2) return t >= 30.0 && t < 90.0 && rng.u() < 0.5;
  if (pid === 3) return rng.u() < 0.9;
  return t >= 10.0 && t < 10.5;
}

/** 10 Hz video and 1 Hz audio (at k + 0.05) in time order, generated in Python's evaluation order. */
function diffStream(rng: ParkMiller): Obs[] {
  const out: Obs[] = [];
  let decoded = 0;
  for (let i = 0; i < D_TICKS * D_TICK_S * 10; i++) {
    const t = i * 0.1;
    decoded = i === 1500 ? 1 : decoded + 1; // the capture restarts at t = 150
    const persons: PersonObs[] = [];
    for (const pid of [1, 2, 3, 4])
      if (diffPresent(rng, pid, t)) persons.push(diffPerson(rng, pid, t));
    const lux = rng.u() < 0.3 ? NaN : t > 250.0 ? 40.0 + rng.u() : 200.0 + 100.0 * rng.u();
    out.push(
      makeVideoObs({
        t,
        wall: D_WALL0 + t,
        frame_seq: i,
        frames_decoded: decoded,
        lux,
        lux_quality: rng.u(),
        frame_w: 640,
        frame_h: 480,
        mirrored: true,
        persons,
      }),
    );
    if (i % 10 === 0) {
      const k = i / 10;
      const ta = k + 0.05;
      const loud = k >= 100 && k <= 130;
      const db = rng.u() < 0.05 ? NaN : loud ? 70.0 + rng.u() : 45.0 + 10.0 * rng.u();
      const speech_prob = rng.u();
      const laughter_onsets = rng.u() < 0.15 ? 1 : 0;
      const compliment_hits = rng.u() < 0.1 ? 1 : 0;
      out.push(
        makeAudioObs({
          t: ta,
          wall: D_WALL0 + ta,
          block_seq: k,
          db_spl: db,
          rms_power: 1e-5,
          speech_prob,
          laughter_onsets,
          compliment_hits,
        }),
      );
    }
  }
  return out.sort((a, b) => a.t - b.t); // stable, like list.sort(key=...)
}

/** Python `fmt`: None/True/False, strings as is, lists bracketed, every number as repr(float(v)). */
function pyFmt(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return `[${v.map(pyFmt).join(",")}]`;
  return pyRepr(v as number);
}

/** Operator actions applied just before the flush of tick k. */
function diffOps(agg: Aggregator, k: number, wall: number): object | null {
  switch (k) {
    case 3:
      agg.startBaseline();
      return null;
    case 6:
      agg.inject({ laughter: 2, compliments: 1, fists: { 1: 2, 9: 1 } });
      return null;
    case 8:
      return agg.calibrate(60.0, 250.0);
    case 10:
      agg.luxOverride = 123.0;
      return null;
    case 12:
      agg.luxOverride = null;
      agg.seedHours(3.5, wall);
      return null;
    case 15:
      agg.logBreak(1, wall);
      return null;
    case 20:
      return agg.startBaseline(8);
    case 24:
      agg.logBreak(null, wall);
      return null;
    case 30:
      agg.resetFrameCounter();
      return null;
    case 40:
      agg.resetPeople();
      return null;
    case 45:
      return agg.calibrate(null, 300.0);
    case 50:
      return agg.startBaseline(null, "auto");
    default:
      return null;
  }
}

describe("aggregator under operator actions", () => {
  // .venv/bin/python -B diff_ops.py: same stream and actions through valence.runtime.aggregator.Aggregator;
  // per tick sha256(text)[:12] of every record (all fields but ambient_db), baseline_status(), the action's
  // return value and the counters, numbers as repr(float(v)).
  const PY_DIGESTS = [
    "5a36b70032ff",
    "31fbe55fb92c",
    "e502e0518aa3",
    "36722cfbec60",
    "bf4a5c0b0b7b",
    "6357311e7ba7",
    "81a4e140ee40",
    "dc7fe0e8cfac",
    "acdaa746f383",
    "9cb806c2d658",
    "f824cef39d1c",
    "ed3cd1cc4bea",
    "e77c3207ad01",
    "5aa377440560",
    "3c0448b7448f",
    "efc43d7b8725",
    "b34cd4298ea1",
    "7010b395f868",
    "b831a6eb0b21",
    "3da9859c2d6e",
    "f7558f9d5ee3",
    "ed835cb7b36f",
    "b0195460ba1d",
    "eddae89cfb55",
    "9a8e51b2b4c4",
    "79088ee34669",
    "9e1861b72c40",
    "114b6acc7930",
    "4a6908eaa877",
    "a0354559940f",
    "fa0c22bfd7ff",
    "4e7c6907b57e",
    "3eba3b2e3653",
    "003ce0775dd1",
    "05de8a8f637b",
    "fd9c4acef331",
    "ddf0d6520632",
    "728e52634284",
    "2addd874d055",
    "5df232fb8c73",
    "96a895966709",
    "51d4925f4da0",
    "c3807e76d39b",
    "46aa7cd8354e",
    "d15872d0f5d1",
    "a88602cb1490",
    "9844c7dc2cb8",
    "c377aa996781",
    "b6566eac4cb9",
    "3d86b697823e",
    "2f10bd5c3ef5",
    "7bc1c6b9c272",
    "122b2e5823dd",
    "78daa5c540e6",
    "355dfc13bfa8",
    "1703cc06274c",
  ];
  const PY_AMBIENT = [
    50.68022474332462, 51.69811035754819, 50.97888736752562, 50.25582468209559, 51.14164823854025,
    47.5550110773346, 50.815704632467416, 50.08224176345446, 51.233775370048235, 60.51446408805884,
    60.141535998288546, 56.7164485822422, 57.205595661944066, 59.35058229025948, 60.7153537505894,
    58.88162269847575, 59.18566246798481, 60.71741309848485, 58.9336993518443, 61.08258088698764,
    79.2049201883319, 79.30064956564206, 79.3119667703674, 79.36206701350281, 79.38546920015023,
    79.11952822705291, 73.1582448390843, 60.463603805686915, 60.1124949175233, 56.583586755195114,
    58.51141961812218, 57.28057881118731, 61.14667337980325, 61.99247854752174, 60.712345088792325,
    59.234131874929346, 60.41341483878273, 56.161492319944216, 62.00495089052734,
    59.221932502171015, 60.72162489017754, 60.434258739322416, 61.58380956289923, 59.50054410220161,
    60.669667364954414, 61.250401663099375, 57.66776297729666, 58.72201449943228,
    60.564856546906455, 58.49246341447482, 58.53564321561479, 60.24781988559191, 57.5726427780529,
    58.16649029851666, 59.6428459945094, 60.00614928270483,
  ];
  const PY_RECENT_LEVEL = 59.64215906891952;

  const rng = new ParkMiller(20261024);
  const agg = new Aggregator(
    makeSettings({ room_baseline_s: 20, break_absent_min: 0.5, person_drop_min: 1.0 }),
    "diff",
    "live",
  );
  const obs = diffStream(rng);
  const digests: string[] = [];
  const texts: string[] = [];
  const ambient: number[] = [];
  let j = 0;
  for (let k = 0; k < D_TICKS; k++) {
    const b = (k + 1) * D_TICK_S;
    while (j < obs.length && (obs[j] as Obs).t < b) agg.push(obs[j++] as Obs);
    const wall = D_WALL0 + b;
    const op = diffOps(agg, k, wall);
    const recs = agg.flush(wall, D_MONO0 + b, 14.0 + b / 3600.0);
    const parts = recs.map((r) => {
      const d = tickRecordToDict(r) as unknown as Record<string, unknown>;
      return TICK_FIELDS.filter((f) => f !== "ambient_db")
        .map((f) => pyFmt(d[f]))
        .join(";");
    });
    parts.push(
      Object.entries(agg.baselineStatus())
        .map(([key, v]) => `${key}=${pyFmt(v)}`)
        .join(";"),
    );
    parts.push(
      op === null
        ? ""
        : Object.entries(op)
            .map(([key, v]) => `${key}=${pyFmt(v)}`)
            .join(";"),
    );
    parts.push(
      pyFmt([
        agg.gatedOut,
        agg.framesDecoded,
        agg.videoFramesTotal,
        agg.audioBlocksTotal,
        [...agg.latestBoxes.keys()].sort((a, b2) => a - b2),
        agg.hoursSinceBreak(1, wall),
        agg.hoursSinceBreak(4, wall),
      ]),
    );
    const text = parts.join("\n");
    texts.push(text);
    digests.push(sha256Hex(text).slice(0, 12));
    ambient.push((recs[0] as TickRecord).ambient_db);
  }

  it("generates the same observation stream as the Python script", () => {
    expect(obs.length).toBe(3080);
  });

  it("matches the reference records, baseline status and counters on every tick", () => {
    expect(digests.length).toBe(PY_DIGESTS.length);
    const first = digests.findIndex((d, i) => d !== PY_DIGESTS[i]);
    expect(first === -1 ? "all ticks match" : `tick ${first}:\n${texts[first] ?? ""}`).toBe(
      "all ticks match",
    );
  });

  it("keeps the pow/log10 Leq within 1e-9 of numpy", () => {
    expect(ambient.length).toBe(PY_AMBIENT.length);
    ambient.forEach((a, i) =>
      expect(Math.abs(a - (PY_AMBIENT[i] as number))).toBeLessThanOrEqual(1e-9),
    );
    expect(Math.abs(agg.recentLevel() - PY_RECENT_LEVEL)).toBeLessThanOrEqual(1e-9);
  });
});

describe("aggregator off the golden path", () => {
  const ROOM_BASE: RoomFields = {
    tick: 0,
    ts: 1.0,
    t_mono: 1.0,
    session_id: "s",
    n_people: 2,
    source: "live",
    ambient_db: NaN,
    db_p90: NaN,
    light_lux: NaN,
    lux_quality: 0.0,
    room_db_base: NaN,
    room_lux_base: NaN,
    laughter_count: 0,
    compliments_count: 0,
    tod_h: 1.0,
  };
  const rec = (pid: number, flag: number, fists: number): TickRecord =>
    makeTickRecord({
      ...ROOM_BASE,
      person_id: pid,
      display_id: `Person #${pid}`,
      social_withdrawal_flag: flag,
      clenched_fists_count: fists,
    });

  it("treats NaN as truthy and -0.0 as falsy in inject(), like Python's `if laughter:`", () => {
    // agg.inject(laughter=nan) -> laughter_count nan, simulated ['laughter_count']
    let agg = new Aggregator(makeSettings(), "s");
    agg.inject({ laughter: NaN });
    let room = agg.flush(10.0, 1.0, 12.0)[0] as TickRecord;
    expect(room.laughter_count).toBeNaN();
    expect(room.simulated_fields).toEqual(["laughter_count"]);
    // agg.inject(laughter=-0.0, compliments=nan) -> 0, nan, ['compliments_count']
    agg = new Aggregator(makeSettings(), "s");
    agg.inject({ laughter: -0.0, compliments: NaN });
    room = agg.flush(10.0, 1.0, 12.0)[0] as TickRecord;
    expect(Object.is(room.laughter_count, 0)).toBe(true);
    expect(room.compliments_count).toBeNaN();
    expect(room.simulated_fields).toEqual(["compliments_count"]);
  });

  it("raises the room withdrawal flag on a NaN flag and sums counts like Python's sum()", () => {
    // room_record(base, [a(nan, 0.1), b(0, 0.2), c(-0.0, 0.3)]) -> flag 1, fists 0.6 (3.12 compensated sum)
    const r = roomRecord(ROOM_BASE, [rec(1, NaN, 0.1), rec(2, 0, 0.2), rec(3, -0.0, 0.3)]);
    expect(r.social_withdrawal_flag).toBe(1);
    expect(r.clenched_fists_count).toBe(0.6);
    // room_record(base, [b, c]) -> 0, 0.5; room_record(base, []) -> fists 0
    const r2 = roomRecord(ROOM_BASE, [rec(2, 0, 0.2), rec(3, -0.0, 0.3)]);
    expect([r2.social_withdrawal_flag, r2.clenched_fists_count]).toEqual([0, 0.5]);
    expect(Object.is(roomRecord(ROOM_BASE, []).clenched_fists_count, 0)).toBe(true);
  });

  it("sizes the baseline with int(seconds or room_baseline_s)", () => {
    const agg = new Aggregator(makeSettings({ room_baseline_s: 10 }), "s");
    // None, 0, 0.0, -0.0 -> 10; 2.9 -> 2; -2.9 -> -2; 1e-09 -> 0
    const cases: [number | null, number][] = [
      [null, 10],
      [0, 10],
      [-0.0, 10],
      [2.9, 2],
      [-2.9, -2],
      [1e-9, 0],
    ];
    for (const [s, want] of cases) expect(agg.startBaseline(s).target_samples).toBe(want);
    // ValueError: cannot convert float NaN to integer; OverflowError: cannot convert float infinity to integer
    expect(() => agg.startBaseline(NaN)).toThrow();
    expect(() => agg.startBaseline(Infinity)).toThrow();
    expect(new Aggregator(makeSettings({ room_baseline_s: 2.5 }), "s").baseline.target).toBe(2);
    expect(new Aggregator(makeSettings({ room_baseline_s: -1.5 }), "s").baseline.target).toBe(-1);
    expect(() => new Aggregator(makeSettings({ room_baseline_s: NaN }), "s")).toThrow();
    expect(() => new Aggregator(makeSettings({ room_baseline_s: Infinity }), "s")).toThrow();
  });

  it("fails like Python's KeyError when an injected-only person passes a zero presence bar", () => {
    const agg = new Aggregator(makeSettings({ participant_min_presence: 0.0 }), "s");
    agg.push(
      makeVideoObs({
        t: 0.0,
        wall: 100.0,
        frame_seq: 1,
        persons: [makePersonObs({ track_id: 1, bbox: [0.1, 0.1, 0.5, 0.9] })],
      }),
    );
    agg.inject({ fists: { 7: 1 } });
    expect(() => agg.flush(105.0, 5.0, 12.0)).toThrow(/7/); // KeyError: 7
  });

  it("raises where Python's statistics.median and math.log10 raise", () => {
    // obs.t NaN leaves no block inside the drift window: StatisticsError no median for empty data
    let agg = new Aggregator(makeSettings({ room_baseline_s: 2 }), "s");
    agg.push(makeAudioObs({ t: 0.0, wall: 1.0, block_seq: 0, db_spl: 50.0, rms_power: 1.0 }));
    agg.push(makeAudioObs({ t: NaN, wall: 1.0, block_seq: 1, db_spl: 50.0, rms_power: 1.0 }));
    expect(() => agg.baselineStatus()).toThrow();
    // every block's power underflows to 0.0: ValueError math domain error. The failed flush keeps the
    // tick's blocks (no _reset_tick), so Python fails again on every later flush, as here.
    agg = new Aggregator(makeSettings(), "s");
    agg.dbCorrection = -1e6;
    agg.push(makeAudioObs({ t: 0.0, wall: 1.0, block_seq: 0, db_spl: 50.0, rms_power: 1.0 }));
    expect(() => agg.flush(5.0, 5.0, 1.0)).toThrow(/math domain error/);
    expect(() => agg.recentLevel()).toThrow(/math domain error/);
    agg.dbCorrection = 0.0;
    expect(() => agg.flush(10.0, 10.0, 1.0)).toThrow(/math domain error/);
    // a lux drift ratio that underflows to 0.0: math.log2 raises ValueError math domain error
    agg = new Aggregator(makeSettings({ room_baseline_s: 1 }), "s");
    agg.push(makeAudioObs({ t: 0.0, wall: 1.0, block_seq: 0, db_spl: 50.0, rms_power: 1.0 }));
    [1e300, 1e-300, 1e-300].forEach((lux, i) =>
      agg.push(makeVideoObs({ t: i, wall: 1.0 + i, frame_seq: i, lux, lux_quality: 1.0 })),
    );
    expect(() => agg.baselineStatus()).toThrow(/math domain error/);
  });

  it("keeps failing like Python when injected fists meet a frame with nobody in it", () => {
    // POST /api/sim/event {fists: {1: 2}} with the camera on and nobody in frame: max() of an empty mean_h
    // raises ValueError, and since _reset_tick never runs every later flush raises too (a reference bug).
    const agg = new Aggregator(makeSettings(), "s");
    agg.push(makeVideoObs({ t: 0.0, wall: 100.0, frame_seq: 1 }));
    agg.inject({ fists: { 1: 2 } });
    expect(() => agg.flush(105.0, 5.0, 12.0)).toThrow(/empty/);
    expect(() => agg.flush(110.0, 10.0, 12.0)).toThrow(/empty/);
  });

  it("propagates a NaN calibration the way Python does", () => {
    const agg = new Aggregator(makeSettings(), "s");
    agg.push(makeAudioObs({ t: 0.0, wall: 1.0, block_seq: 0, db_spl: 50.0, rms_power: 1.0 }));
    expect(agg.recentLevel(-1.0)).toBeNaN(); // recent_level(-1.0) -> nan (mean of nothing)
    agg.push(makeAudioObs({ t: 1.0, wall: 2.0, block_seq: 1, db_spl: 52.0, rms_power: 1.0 }));
    const out = agg.calibrate(NaN); // {'db_correction': nan}
    expect(Object.keys(out)).toEqual(["db_correction"]);
    expect(out.db_correction).toBeNaN();
    agg.push(
      makeAudioObs({
        t: 2.0,
        wall: 3.0,
        block_seq: 2,
        db_spl: 52.0,
        rms_power: 1.0,
        laughter_onsets: 1,
      }),
    );
    const r = agg.flush(5.0, 5.0, 1.0)[0] as TickRecord;
    // ambient_db nan, db_p90 nan, laughter 1, gated_out 0, room_db_base nan
    expect(r.ambient_db).toBeNaN();
    expect(r.db_p90).toBeNaN();
    expect([r.laughter_count, agg.gatedOut]).toEqual([1, 0]);
    expect(r.room_db_base).toBeNaN();
  });

  it("follows DST changes and odd offsets in named zones like localtime()", () => {
    // TZ=<zone> python -c "from valence.runtime.engine import local_tod; ...". Post-1970 only: earlier
    // offsets come from the tz database build (ICU here, system zoneinfo for Python) and can differ.
    const cases: [string, number, number][] = [
      ["America/New_York", 1699164000.0, 1.0],
      ["America/New_York", 1699167599.5, 1.9997222222222222], // fall back: last second of the first 1 am
      ["America/New_York", 1699167600.0, 2.0],
      ["America/New_York", 1710053999.5, 1.9997222222222222], // spring forward
      ["America/New_York", 1710054000.0, 3.0],
      ["Australia/Lord_Howe", 1699167599.5, 17.999722222222225], // half-hour DST
      ["Pacific/Chatham", 1699164000.0, 19.75], // +13:45
      ["Pacific/Chatham", 1e9 + 0.9999996, 14.528055555555556],
      ["Asia/Kolkata", 1e9 + 0.9999996, 7.2780555555555555],
      ["Africa/Monrovia", 64060588799.0, 23.999722222222225],
    ];
    for (const [tz, wall, want] of cases) expect(localTod(wall, tz), `${tz} ${wall}`).toBe(want);
  });

  it("rejects a non-finite timestamp like datetime.fromtimestamp", () => {
    // ValueError Invalid value NaN (not a number); OverflowError timestamp out of range for platform time_t
    expect(() => localTod(NaN, "UTC")).toThrow();
    expect(() => localTod(Infinity, "UTC")).toThrow();
    expect(() => localTod(-Infinity)).toThrow();
  });
});

describe("jsonl recordings like recording.iter_obs", () => {
  const A0 = '{"type":"audio","t":0.0,"wall":5.0,"block_seq":0,"db_spl":NaN,"rms_power":1e-5}';
  const A1 = '{"type":"audio","t":1.0,"wall":6.0,"block_seq":1,"db_spl":Infinity,"rms_power":1e-5}';
  const A2 =
    '{"type":"audio","t":2.0,"wall":7.0,"block_seq":2,"db_spl":-Infinity,"rms_power":1e-5}';

  it("splits universal newlines and reads json.loads' NaN and Infinity literals", () => {
    // a file of "\r"-separated lines plus a trailing "\r\n  \r" -> 3 obs, db_spl [nan, inf, -inf]
    const obs = parseJsonl([A0, A1, A2].join("\r") + "\r\n  \r");
    expect(obs.map((o) => (o.type === "audio" ? o.db_spl : 0))).toEqual([NaN, Infinity, -Infinity]);
  });

  it("strips the whitespace str.strip() strips and nothing else", () => {
    // "\x1c" + line + "\x85\n\u3000\n" -> 1 obs
    expect(parseJsonl("\x1c" + A0 + "\x85\n\u3000\n").length).toBe(1);
    // a UTF-8 BOM is not whitespace to Python: JSONDecodeError
    expect(() => parseJsonl("\ufeff" + A0 + "\n")).toThrow();
  });

  it("keeps NaN-looking text inside strings untouched", () => {
    // json.loads keeps "x": "NaN Infinity" as a string; obs_from_dict then raises TypeError on the unknown
    // field (AudioObs.__init__() got an unexpected keyword argument 'x'), and so must parseJsonl.
    expect(() => parseJsonl(A0.replace("}", ',"x":"NaN Infinity"}'))).toThrow(/unexpected field/);
  });
});

describe("offline replay tick length", () => {
  it("takes tick_s as its own argument, defaulting to the settings' tick", () => {
    const obs = [0.0, 1.0, 2.5, 4.0].map((t, i) =>
      makeAudioObs({ t, wall: t, block_seq: i, db_spl: 50.0, rms_power: 1.0 }),
    );
    // replay_offline(rt, path, tick_s=2.0) steps at mono 2, 4, 6
    expect(
      replayOffline(obs, { wall0: 0, mono0: 0, tickS: 2.0, tz: "UTC" }).map((t) => t.mono),
    ).toEqual([2, 4, 6]);
    expect(
      replayOffline(obs, {
        wall0: 0,
        mono0: 0,
        settings: makeSettings({ tick_s: 3.0 }),
        tz: "UTC",
      }).map((t) => t.mono),
    ).toEqual([3, 6]);
  });
});

describe("real capture files", () => {
  it("parse to the observations recording.iter_obs reads from the same files", () => {
    // The golden recording cases hold obs_to_dict(iter_obs(data/real/<name>.jsonl)), untrimmed (all < 6 min).
    const recordings = fixture.cases.filter((c) => c.kind === "recording");
    expect(recordings.length).toBe(7);
    for (const c of recordings) {
      const text = readFileSync(
        path.resolve(FRONTEND_ROOT, "../data/real", `${c.name}.jsonl`),
        "utf8",
      );
      const got = parseJsonl(text).map((o) => obsToDict(o));
      expect(got.length, c.name).toBe(c.obs.length);
      expect(got, c.name).toEqual(c.obs);
    }
  });
});
