/**
 * Audio perception: one ~1 s mono 16 kHz Float32Array block in, one AudioObs out. Port of
 * valence/perception/audio.py without the compliment recognizer (compliment_hits stays 0).
 *
 * Loudness is the block's Leq: mean power -> dB SPL with the SPL offset (94 dB uncalibrated, as on the Python edge;
 * the engine only uses levels relative to the venue baseline). YAMNet (allowlisted classes) runs on the block's last
 * 0.975 s, so its laughter and speech scores belong to exactly this block; laughter onsets are debounced. The caller
 * zero-fills the block afterwards.
 */
import { LaughterTracker, meanPower, powerToDb } from "./audioFeatures";
import type { AudioModels } from "./models";
import type { AudioObs } from "./types";

export const SR = 16000;
export const YAMNET_WINDOW = 15600; // 0.975 s at 16 kHz

export class AudioPipeline {
  private readonly laughter = new LaughterTracker();
  readonly offset: number;

  constructor(private readonly models: Pick<AudioModels, "classify">, opts: { splOffsetDb?: number } = {}) {
    this.offset = opts.splOffsetDb ?? 94.0;
  }

  process(block: Float32Array, t: number, wall: number, seq: number): AudioObs {
    const power = meanPower(block);
    const scores: Record<string, number> = {};
    const tail = block.length > YAMNET_WINDOW ? block.subarray(block.length - YAMNET_WINDOW) : block;
    for (const res of this.models.classify(tail, SR)) {
      for (const cl of res.classifications) {
        for (const c of cl.categories) scores[c.categoryName] = Math.max(scores[c.categoryName] ?? 0.0, Number(c.score));
      }
    }
    const [, speech] = this.laughter.update(scores, t);
    return {
      type: "audio", t, wall, block_seq: seq, db_spl: powerToDb(power, this.offset), rms_power: power,
      speech_prob: speech, laughter_onsets: this.laughter.takeOnsets(), compliment_hits: 0,
    };
  }
}
