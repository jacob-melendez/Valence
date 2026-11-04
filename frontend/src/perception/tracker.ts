/**
 * Simple person tracker standing in for ByteTrack (the Python edge runs ultralytics ByteTrack on YOLO boxes).
 * PoseLandmarker returns unordered poses per frame; this gives each one a tracker id that persists while the person
 * stays roughly in place. Greedy IoU association, best pairs first; a track that goes unmatched is kept as "lost"
 * for `maxLostS` (ByteTrack's track_buffer: 60 frames at 10 Hz) and revived with the same id if a box overlaps it
 * again. Ids only grow, like ByteTrack's; DisplayIdMapper (ids.ts) turns them into small, steady "Person #N" labels.
 * Only boxes and timestamps are kept.
 */

export type BoxPx = readonly [number, number, number, number];

interface Track {
  id: number;
  box: BoxPx;
  lastT: number;
}

export function iou(a: BoxPx, b: BoxPx): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export class IouTracker {
  private tracks: Track[] = [];
  private nextId = 1;

  constructor(
    readonly minIou = 0.2,
    readonly maxLostS = 6.0,
  ) {}

  /** Feed one frame's boxes (pixels) at time `t`; returns a tracker id per box, in input order. */
  update(boxes: readonly BoxPx[], t: number): number[] {
    this.tracks = this.tracks.filter((tr) => t - tr.lastT <= this.maxLostS);
    const pairs: [number, number, number][] = [];
    this.tracks.forEach((tr, ti) => {
      boxes.forEach((box, bi) => {
        const v = iou(tr.box, box);
        if (v >= this.minIou) pairs.push([v, ti, bi]);
      });
    });
    pairs.sort((a, b) => b[0] - a[0] || this.tracks[a[1]]!.id - this.tracks[b[1]]!.id || a[2] - b[2]);
    const ids: (number | null)[] = boxes.map(() => null);
    const usedTracks = new Set<number>();
    for (const [, ti, bi] of pairs) {
      if (usedTracks.has(ti) || ids[bi] !== null) continue;
      usedTracks.add(ti);
      const tr = this.tracks[ti]!;
      tr.box = boxes[bi]!;
      tr.lastT = t;
      ids[bi] = tr.id;
    }
    return ids.map((id, bi) => {
      if (id !== null) return id;
      const tr: Track = { id: this.nextId++, box: boxes[bi]!, lastT: t };
      this.tracks.push(tr);
      return tr.id;
    });
  }
}
