/**
 * The real-data protocol (docs/PROTOCOL.md section 1): timed behaviour blocks whose boundaries the session recorder
 * stamps as markers, so valence/eval/benchmark.py can score each model per block. Keep this table and the doc's in
 * step: the expectations travel in the recorded markers, so the benchmark needs no copy of them.
 */
export interface ProtocolBlock {
  id: string; label: string; minutes: number; instruction: string;
  /** Scored by the benchmark (the settle block only fills the face and room baselines). */
  scored: boolean;
  /** A calm block: any non-calm label there counts as a false alarm. */
  calm: boolean;
  /** Labels that count as a hit: `card` for the served 4-state model (state ids), `map` for the emotion map. */
  expect: { card: readonly string[]; map: readonly string[] };
}
export interface Protocol { id: string; title: string; blocks: readonly ProtocolBlock[] }

export const PROTOCOL_V1: Protocol = {
  id: "valence-protocol/1",
  title: "Real-data protocol v1 (about 24 minutes)",
  blocks: [
    { id: "settle", label: "Settle in", minutes: 2, scored: false, calm: false, expect: { card: [], map: [] },
      instruction: "Sit as you normally work, relaxed. Keep a neutral face for the first 20 seconds." },
    { id: "focused", label: "Focused work", minutes: 6, scored: true, calm: true,
      expect: { card: ["Focused"], map: ["Focused", "Calm"] },
      instruction: "Read or type something engaging. Stay quiet and still." },
    { id: "restless", label: "Restless", minutes: 3, scored: true, calm: false,
      expect: { card: ["Frustrated_Masked"], map: ["Frustrated_Masked", "Tense", "Anxious", "Overwhelmed"] },
      instruction: "Fidget, shift in your seat, tap, clench your fists now and then, sigh." },
    { id: "frustrated", label: "Frustrating task", minutes: 4, scored: true, calm: false,
      expect: { card: ["Frustrated_Masked"], map: ["Frustrated_Masked", "Tense", "Confused", "Overwhelmed", "Anxious"] },
      instruction: "Count down from 1000 by 7s out loud, against the clock. Let the frustration show." },
    { id: "tired", label: "Tired", minutes: 3, scored: true, calm: false,
      expect: { card: ["Exhausted"], map: ["Exhausted", "Bored"] },
      instruction: "Slump, rest your head in your hands, look down, slow down." },
    { id: "upbeat", label: "Upbeat", minutes: 3, scored: true, calm: false,
      expect: { card: ["Collaborative_Happy"], map: ["Collaborative_Happy"] },
      instruction: "Watch or listen to something funny, or chat with someone. Smile and laugh." },
    { id: "calm", label: "Calm", minutes: 3, scored: true, calm: true,
      expect: { card: ["Focused"], map: ["Calm", "Focused"] },
      instruction: "Sit back, breathe slowly, relaxed face." },
  ],
};

export const protocolMinutes = (p: Protocol) => p.blocks.reduce((s, b) => s + b.minutes, 0);
