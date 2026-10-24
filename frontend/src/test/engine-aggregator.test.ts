/**
 * aggregator.ts + replay.ts against valence/runtime/aggregator.py and recording.replay_offline
 * (scripts/export_web_golden.py -> aggregator.json.gz), plus TS ports of the aggregator tests in
 * tests/test_runtime.py.
 *
 * Records must match per tick: counts, flags, ids and strings exactly, the NaN pattern exactly, other
 * numbers within 1e-9 (only ambient_db goes through pow/log10, where V8 and numpy may differ in the last bit;
 * every other field is required to be bit-exact). Features at the checkpoints go through features.ts.
 */
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
  type PersonObs,
  type Settings,
  type TickRecord,
  type TickRecordDict,
  type VideoObs,
} from "@/engine/contracts";
import { compute, featureIndex, featuresFromWindow } from "@/engine/features";
import { parseJsonl, replayInto, replayOffline, restamp, type ReplayTick } from "@/engine/replay";
import { RollingWindow } from "@/engine/window";
import { loadFixture, nn } from "./engine-fixtures";

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
