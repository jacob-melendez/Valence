/**
 * 15-minute rolling window of TickRecords, one slot per person plus the room slot
 * (valence/features/window.py).
 *
 * The room slot (`person_id == ROOM`) is appended every tick, whoever is present, so a person who sits
 * down inherits 15 minutes of room history (noise, light, laughter) and only their own body and face
 * channels start cold. The Python class takes a lock because the API thread reads it; the browser engine
 * is single-threaded, so this port has none.
 */
import { ROOM, type TickRecord } from "./contracts";

export { ROOM };

// ---------- RollingWindow ----------

/** valence/features/window.py:18 RollingWindow: per-slot `deque(maxlen=max_ticks)`, oldest row first. */
export class RollingWindow {
  readonly maxTicks: number;
  private readonly slots = new Map<number, TickRecord[]>();

  constructor(maxTicks = 180) {
    this.maxTicks = maxTicks;
  }

  /** valence/features/window.py:24. Creates the slot on first sight; the oldest row falls off at maxTicks. */
  append(rec: TickRecord): void {
    let slot = this.slots.get(rec.person_id);
    if (slot === undefined) {
      slot = [];
      this.slots.set(rec.person_id, slot);
    }
    slot.push(rec);
    // deque(maxlen=n) semantics, including maxlen=0 (the slot exists but stays empty)
    while (slot.length > this.maxTicks) slot.shift();
  }

  /** valence/features/window.py:31. A copy of the slot (oldest first); empty for an unknown id. */
  snapshot(personId: number): TickRecord[] {
    return [...(this.slots.get(personId) ?? [])];
  }

  /** valence/features/window.py:35. The room slot's rows. */
  room(): TickRecord[] {
    return this.snapshot(ROOM);
  }

  /** valence/features/window.py:38. Person ids with a slot, ascending, room excluded. */
  people(): number[] {
    return [...this.slots.keys()].filter((pid) => pid !== ROOM).sort((a, b) => a - b);
  }

  /** valence/features/window.py:42. Forget one slot; unknown ids are ignored. */
  drop(personId: number): void {
    this.slots.delete(personId);
  }

  /** valence/features/window.py:46. Forget every person (between judge groups); keep the room history. */
  resetPeople(): void {
    for (const pid of [...this.slots.keys()]) if (pid !== ROOM) this.slots.delete(pid);
  }

  /** valence/features/window.py:52. Forget everything, room included. */
  clear(): void {
    this.slots.clear();
  }

  /** valence/features/window.py:56. Minutes of history in a slot: `len(slot) * tick_s / 60`. */
  fillMinutes(personId: number, tickS = 5.0): number {
    return ((this.slots.get(personId)?.length ?? 0) * tickS) / 60.0;
  }
}
