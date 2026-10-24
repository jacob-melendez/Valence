/**
 * Deterministic offline replay of an observation stream (valence/runtime/recording.py replay_offline).
 *
 * The recording is mapped onto one synthetic clock starting at (wall0, mono0), so observations and ticks
 * agree: the break clock starts when the replayed person first appears instead of when the capture was
 * made. Tick boundaries come from the observation timestamps; no timers, no threads.
 *
 * Recordings hold features only (never frames, audio or transcripts), so nothing here touches the disk.
 */
import { Aggregator, localTod } from "./aggregator";
import {
  DEFAULT_SETTINGS,
  obsFromDict,
  type Obs,
  type Settings,
  type TickRecord,
} from "./contracts";

export { localTod };

// ---------- Restamping ----------

/** The same observation on another clock; observations are treated as immutable (recording.py:70 restamp). */
export function restamp<T extends Obs>(obs: T, wall: number, t: number): T {
  return { ...obs, wall, t };
}

/** Observations of a features-only JSONL recording, in file order (recording.py:24 iter_obs, from text). */
export function parseJsonl(text: string): Obs[] {
  const out: Obs[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (line) out.push(obsFromDict(JSON.parse(line) as Record<string, unknown>));
  }
  return out;
}

// ---------- Generic driver ----------

/** What replay needs from a runtime: Runtime.push_obs and Runtime.step(wall, mono). */
export interface ReplaySink {
  push(obs: Obs): void;
  step(wall: number, mono: number): void;
}

export interface ReplayClock {
  /** Wall clock of the first observation (Python: time.time() when the replay starts). */
  wall0: number;
  /** Monotonic clock of the first observation (Python: time.monotonic()). */
  mono0: number;
  /** Tick length in seconds (Settings.tick_s). */
  tickS: number;
}

/**
 * Feed `obs` through `sink`, stepping once per tick boundary (valence/runtime/recording.py:75 replay_offline).
 *
 * Same double arithmetic as Python, left to right: tick wall = wall0 + next_boundary - t0, restamped
 * obs wall = wall0 + obs.t - t0 and t = mono0 + obs.t - t0, where t0 is the first obs.t and boundaries start
 * at t0 + tick_s. One final step closes the last (partial) tick; an empty stream steps nothing.
 */
export function replayInto(obs: Iterable<Obs>, sink: ReplaySink, clock: ReplayClock): void {
  const { wall0, mono0, tickS } = clock;
  let t0: number | null = null;
  let nextBoundary = 0;
  for (const o of obs) {
    if (t0 === null) {
      t0 = o.t;
      nextBoundary = o.t + tickS;
    }
    while (o.t >= nextBoundary) {
      sink.step(wall0 + nextBoundary - t0, mono0 + nextBoundary - t0);
      nextBoundary += tickS;
    }
    sink.push(restamp(o, wall0 + o.t - t0, mono0 + o.t - t0));
  }
  if (t0 !== null) sink.step(wall0 + nextBoundary - t0, mono0 + nextBoundary - t0);
}

// ---------- Aggregator replay ----------

/** One tick of an offline replay: its clock and the records Aggregator.flush produced (room row first). */
export interface ReplayTick {
  wall: number;
  mono: number;
  records: TickRecord[];
}

export interface ReplayOptions {
  wall0: number;
  mono0: number;
  settings?: Settings;
  /** Aggregator session id (Python: the runtime's session id). */
  sessionId?: string;
  /** Record `source` ("replay" for recordings, as Runtime.use_recording sets it). */
  source?: string;
  /** Zone for the time of day: "UTC", an IANA name, or undefined for the local zone (see localTod). */
  tz?: string;
  /** Replay into this aggregator instead of a fresh one (e.g. to inspect its baseline afterwards). */
  aggregator?: Aggregator;
  /** Called after each tick's flush, e.g. to append the records to a RollingWindow. */
  onTick?: (tick: ReplayTick, index: number) => void;
}

/**
 * Replay an observation stream through an Aggregator, one flush per tick boundary, and return every tick's
 * records: what Runtime.step sees from its LiveSource during recording.replay_offline, without a Runtime
 * (flush(wall, mono, local_tod(wall)), engine.py:71).
 */
export function replayOffline(obs: Iterable<Obs>, opts: ReplayOptions): ReplayTick[] {
  const settings = opts.settings ?? DEFAULT_SETTINGS;
  const agg =
    opts.aggregator ??
    new Aggregator(settings, opts.sessionId ?? "replay", opts.source ?? "replay");
  const ticks: ReplayTick[] = [];
  replayInto(
    obs,
    {
      push: (o) => agg.push(o),
      step: (wall, mono) => {
        const tick: ReplayTick = {
          wall,
          mono,
          records: agg.flush(wall, mono, localTod(wall, opts.tz)),
        };
        ticks.push(tick);
        opts.onTick?.(tick, ticks.length - 1);
      },
    },
    { wall0: opts.wall0, mono0: opts.mono0, tickS: settings.tick_s },
  );
  return ticks;
}
