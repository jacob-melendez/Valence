/**
 * Cloud B's emotion tour (artifacts/emotion/tour.json.gz): a scripted, simulated 50-minute walk through the ten
 * emotion-map states (5 minutes each), as TickRecords per tick plus Python's per-tick outputs and the scripted
 * truth. The console replays it through the engine for a camera-free demo (?demo=tour); the tests use the
 * Python outputs for end-to-end parity.
 */
import { tickRecordFromDict, type TickRecord } from "./contracts";
import type { FetchLike } from "./served";

/** One tick of the tour: the room row and the person's row, the scripted truth and Python's outputs. */
export interface TourTick {
  records: TickRecord[];
  true_state: string;
  true_valence: number;
  true_arousal: number;
  /** Python's outputs for this tick (valence/models/emotion.py on the person's window). */
  emotion_features: (number | null)[];
  probabilities: number[];
  label: string;
  valence: number;
  arousal: number;
}

export interface Tour {
  model_id: string;
  feature_version: string;
  description: string;
  tick_s: number;
  states: string[];
  ticks: TourTick[];
}

/** Parse the tour JSON (null -> NaN in records, via tickRecordFromDict). */
export function parseTour(json: unknown): Tour {
  const t = json as Omit<Tour, "ticks"> & { ticks: (Omit<TourTick, "records"> & { records: Record<string, unknown>[] })[] };
  if (!t || !Array.isArray(t.ticks)) throw new Error("not a tour file");
  return { ...t, ticks: t.ticks.map((k) => ({ ...k, records: k.records.map((r) => tickRecordFromDict(r)) })) };
}

/** Fetch and parse /models/emotion/tour.json.gz (gzip inflated in the page); null when it is missing. */
export async function loadTour(url = "/models/emotion/tour.json.gz", fetchFn: FetchLike = (u) => globalThis.fetch(u)): Promise<Tour | null> {
  const res = await fetchFn(url);
  if (!res.ok) return null;
  let bytes = new Uint8Array(await res.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const text = new TextDecoder().decode(bytes);
  if (/^\s*</.test(text)) return null; // a static host's SPA fallback page
  return parseTour(JSON.parse(text));
}
