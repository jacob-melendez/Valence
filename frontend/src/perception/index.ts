/**
 * Valence Web browser perception: camera and mic in, contracts.py-shaped observations out.
 *
 *   const { stream } = await openMediaStream();          // echo cancellation / noise suppression / AGC off
 *   const perception = createPerception();               // or createMockPerception() without a camera
 *   const off = perception.onObservation((obs) => engine.push(obs));   // VideoObs ~10 Hz, AudioObs 1 Hz
 *   await perception.start(stream);
 *   perception.counters();                               // governance: frames decoded/purged, audio blocks purged
 *   perception.stop(); off();
 *
 * Frames and audio are processed in memory and dropped after each step; nothing is recorded, persisted or sent.
 * WASM and models load from the site's own origin (`assetBase`, default "/mediapipe/").
 */
export * from "./types";
export { createPerception } from "./perception";
export { createMockPerception, type MockClock, type MockPerceptionOptions } from "./mock";
export { openMediaStream, AUDIO_CONSTRAINTS, VIDEO_CONSTRAINTS, type OpenedMedia } from "./media";
