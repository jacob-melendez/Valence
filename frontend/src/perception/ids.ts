/**
 * Stable, ephemeral "Person #N" labels on top of tracker ids. Port of valence/perception/ids.py.
 *
 * The tracker hands out ever-increasing tracker ids and re-issues a fresh one after an occlusion, which would turn one
 * person into Person #7, #8, #9 over a session. This mapper keeps the label small and steady: a track that vanishes
 * keeps its display id reserved for `reuseWindowS`, and a new track that appears near where it vanished inherits it.
 * Only integer ids, normalized box centres and timestamps are kept; there is no appearance model, embedding or
 * identity, and everything is forgotten once the reservation expires.
 */
import { TIME_EPS } from "./debounce";
import { pyHypot } from "./hands";

export type BBox = readonly [number, number, number, number];
export type Track = readonly [number, BBox];
type Centre = readonly [number, number];

interface Lost {
  trackerId: number;
  centre: Centre;
  lostT: number;
}

function centreOf([x1, y1, x2, y2]: BBox): Centre {
  return [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
}

/** Python's int() on a float: truncate toward zero; NaN and infinities raise. */
function pyInt(v: number): number {
  if (!Number.isFinite(v)) throw new Error(`cannot convert ${v} to an integer tracker id`);
  return Math.trunc(v) + 0; // + 0 turns -0 into 0
}

/** Python's ordering of two numbers inside a tuple comparison (no NaN here: gated distances are never NaN). */
const cmp = (a: number, b: number): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * Map tracker ids to display ids 1..N.
 *
 * `update` takes the tracks of one frame as `[trackerId, [x1, y1, x2, y2]]` with boxes normalized to [0, 1].
 * Distances are measured in frame widths: `dx` in normalized x and `dy` in normalized y times `aspect` (frame h / w).
 * A new tracker id first reclaims its own reservation (the tracker revived it), then the closest reserved display id
 * within `reuseDist` (matched globally, nearest pairs first), else the smallest display id that is neither active nor
 * reserved.
 */
export class DisplayIdMapper {
  /** tracker id -> display id, in Python dict (insertion) order. */
  private activeMap = new Map<number, number>();
  private centres = new Map<number, Centre>();
  /** display id -> the track that last held it. */
  private lost = new Map<number, Lost>();
  private released = new Set<number>();

  constructor(
    readonly reuseWindowS = 3.0,
    readonly reuseDist = 0.25,
    readonly aspect = 9 / 16,
  ) {
    if (reuseWindowS < 0 || reuseDist < 0 || aspect <= 0) {
      throw new Error("reuseWindowS, reuseDist must be >= 0 and aspect > 0");
    }
    this.reset();
  }

  /** Forget all tracks; every display id known so far is reported by `popReleased`. */
  reset(): void {
    const released = new Set([...this.released, ...this.activeMap.values(), ...this.lost.keys()]);
    this.activeMap = new Map();
    this.centres = new Map();
    this.lost = new Map();
    this.released = released;
  }

  /** Current `trackerId -> displayId` mapping (a copy). */
  active(): Map<number, number> {
    return new Map(this.activeMap);
  }

  /** Display ids of recently lost people still held for reuse. */
  reserved(): Set<number> {
    return new Set(this.lost.keys());
  }

  /** Display ids whose reservation expired since the last call (gone for good), ascending. */
  popReleased(): number[] {
    const released = [...this.released].sort(cmp);
    this.released = new Set();
    return released;
  }

  /** Feed one frame of tracks at time `t`; returns `trackerId -> displayId`. */
  update(tracks: ReadonlyArray<Track>, t: number): Map<number, number> {
    const current = DisplayIdMapper.dedupe(tracks);
    this.expire(t);
    this.markLost(current, t);

    const newIds = [...current.keys()].filter((tid) => !this.activeMap.has(tid));
    for (const tid of newIds) this.centres.set(tid, current.get(tid)!);
    let unassigned = this.reclaimOwn(newIds);
    unassigned = this.reuseNearby(unassigned);
    for (const tid of [...unassigned].sort(cmp)) this.activeMap.set(tid, this.smallestFree());

    for (const [tid, centre] of current) this.centres.set(tid, centre);
    return new Map(this.activeMap);
  }

  /** First box per tracker id wins; ids go through int() first. */
  private static dedupe(tracks: ReadonlyArray<Track>): Map<number, Centre> {
    const current = new Map<number, Centre>();
    for (const [tid, bbox] of tracks) {
      const id = pyInt(tid);
      if (!current.has(id)) current.set(id, centreOf(bbox));
    }
    return current;
  }

  private expire(t: number): void {
    const horizon = this.reuseWindowS - TIME_EPS;
    const expired = [...this.lost].filter(([, lost]) => t - lost.lostT >= horizon).map(([did]) => did);
    for (const did of expired) {
      this.lost.delete(did);
      this.released.add(did);
    }
  }

  private markLost(current: ReadonlyMap<number, Centre>, t: number): void {
    for (const tid of [...this.activeMap.keys()].filter((id) => !current.has(id))) {
      const did = this.activeMap.get(tid)!;
      this.activeMap.delete(tid);
      const centre = this.centres.get(tid)!;
      this.centres.delete(tid);
      this.lost.set(did, { trackerId: tid, centre, lostT: t });
    }
  }

  private reclaimOwn(newIds: readonly number[]): number[] {
    const byTracker = new Map<number, number>();
    for (const [did, lost] of this.lost) byTracker.set(lost.trackerId, did);
    const rest: number[] = [];
    for (const tid of newIds) {
      const did = byTracker.get(tid);
      if (did === undefined) {
        rest.push(tid);
      } else {
        this.lost.delete(did);
        this.activeMap.set(tid, did);
      }
    }
    return rest;
  }

  private reuseNearby(newIds: readonly number[]): number[] {
    const pairs: [number, number, number][] = [];
    for (const tid of newIds) {
      for (const [did, lost] of this.lost) {
        const d = this.distance(this.centres.get(tid)!, lost.centre);
        if (d <= this.reuseDist) pairs.push([d, tid, did]);
      }
    }
    pairs.sort((a, b) => cmp(a[0], b[0]) || cmp(a[1], b[1]) || cmp(a[2], b[2]));
    const taken = new Set<number>();
    for (const [, tid, did] of pairs) {
      if (this.activeMap.has(tid) || taken.has(did)) continue;
      taken.add(did);
      this.lost.delete(did);
      this.activeMap.set(tid, did);
    }
    return newIds.filter((tid) => !this.activeMap.has(tid));
  }

  private distance(a: Centre, b: Centre): number {
    return pyHypot(a[0] - b[0], (a[1] - b[1]) * this.aspect);
  }

  private smallestFree(): number {
    const used = new Set([...this.activeMap.values(), ...this.lost.keys()]);
    let did = 1;
    while (used.has(did)) did += 1;
    return did;
  }
}
