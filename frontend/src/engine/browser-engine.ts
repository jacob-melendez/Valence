/**
 * Glue between the in-browser source (frontend/src/lib/browser-source.ts) and the engine: the served predictor
 * (+ the emotion map, when its models are present) and a live Runtime, as valence/api/app.py wires them for the
 * desktop backend. Loaded lazily by the console (import.meta.glob in lib/valence.ts) only for `?source=browser`.
 *
 * Per tick it runs Runtime.stepLive (aggregator flush, governance / state_vector / prediction) and, when the
 * emotion models are loaded, attaches `prediction.affect` the way valence/models/emotion.py affect_for_window
 * builds it, keeping each person's model input for the self-report capture.
 */
import type { BrowserEngine, ObsLike, PerceptionCounters } from "../lib/browser-source";
import type { RoomBaseline, ServerMessage } from "../lib/valence";
import { ROOM, makeSettings, stateInfos, type Obs, type TickRecord } from "./contracts";
import { computeEmotion } from "./emotion-features";
import { emotionInput, loadEmotionModel, predictEmotion, toAffect, type EmotionModel } from "./emotion-model";
import { FEATURE_NAMES, featuresFromWindow } from "./features";
import { Runtime, ScenarioSource, type OverlayName } from "./runtime";
import { loadServed, type FetchLike } from "./served";
import { loadTour, type Tour } from "./tour";

export interface BrowserEngineOptions {
  /** Where the served models live (scripts/model-files.mjs): "/models/core/" (4 states) or "/models/all/". */
  modelBase?: string;
  /** The emotion map's files; null skips it. */
  emotionBase?: string | null;
  fetchFn?: FetchLike;
  /** Time zone for the time of day (undefined = the device's local time, as the desktop edge uses). */
  tz?: string;
  /**
   * Replay Cloud B's emotion tour (a simulated 50-minute walk through the 10 states) instead of live sensors:
   * the camera-free demo. Predictions then carry the scripted truth for the affect map.
   */
  tour?: boolean;
  tourSpeed?: number;
}

/** The emotion model's input for a person's window, keyed by feature name (null = missing), for self-reports. */
function inputsByName(names: readonly string[], x: Float64Array): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  names.forEach((n, i) => {
    const v = x[i] ?? NaN;
    out[n] = Number.isFinite(v) ? v : null;
  });
  return out;
}

export async function createBrowserEngine(opts: BrowserEngineOptions = {}): Promise<BrowserEngine & {
  latestInputs(personId: number): { feature_version: string; features: Record<string, number | null> } | null;
  runtime: Runtime;
  emotion: EmotionModel | null;
  tour: Tour | null;
}> {
  const fetchFn: FetchLike = opts.fetchFn ?? ((url) => globalThis.fetch(url));
  const modelBase = opts.modelBase ?? "/models/core/";
  const predictor = await loadServed(modelBase, fetchFn);
  if (!predictor) throw new Error(`no served models at ${modelBase}`);
  // The emotion map is optional: without its files (or if they fail to load) the console runs without it.
  let emotion: EmotionModel | null = null;
  if (opts.emotionBase !== null) {
    try { emotion = await loadEmotionModel(opts.emotionBase ?? "/models/emotion/", fetchFn); } catch { emotion = null; }
  }
  const runtime = new Runtime(makeSettings({ mode: opts.tour ? "replay" : "live" }), predictor, opts.tz === undefined ? {} : { tz: opts.tz });
  let tour: Tour | null = null, source: ScenarioSource | null = null;
  if (opts.tour) {
    tour = await loadTour(`${(opts.emotionBase ?? "/models/emotion/").replace(/\/?$/, "/")}tour.json.gz`, fetchFn);
    if (!tour) throw new Error("the emotion tour (tour.json.gz) is not available");
    source = new ScenarioSource(tour.ticks.flatMap((t) => t.records), "emotion_tour", { loop: true });
    runtime.useScenario("emotion_tour", opts.tourSpeed ?? 10);
  } else runtime.useLive();
  let tourTick = 0;
  const inputs = new Map<number, Float64Array>();
  /** Keep model inputs only for the people predicted this tick: a tap must never pair now with someone who left. */
  const keepInputs = (msgs: readonly { type: string; person_id?: number }[]) => {
    const predicted = new Set(msgs.flatMap((m) => (m.type === "prediction" && typeof m.person_id === "number" ? [m.person_id] : [])));
    for (const pid of [...inputs.keys()]) if (!predicted.has(pid)) inputs.delete(pid);
  };

  /** valence/models/emotion.py affect_for_window, keeping the input for the self-report capture. */
  const affectFor = (pid: number) => {
    if (!emotion) return null;
    const person: readonly TickRecord[] = runtime.window.snapshot(pid);
    if (!person.length) { inputs.delete(pid); return null; }
    const room = runtime.window.room();
    const x = emotionInput(featuresFromWindow(person, room.length ? room : person), computeEmotion(person), emotion.meta);
    inputs.set(pid, x);
    return toAffect(predictEmotion(emotion, x), emotion);
  };

  return {
    runtime,
    emotion,
    tour,
    states: stateInfos(predictor.states),
    featureVersion: predictor.modelId,
    affectStates: emotion ? emotion.meta.states.map((s) => ({ id: s.id, display: s.display, valence: s.valence, arousal: s.arousal })) : [],
    push(obs: ObsLike) {
      // Perception's observations carry the contracts.py fields (plus a `raw` extra the engine ignores).
      return runtime.pushObs(obs as unknown as Obs) as ServerMessage | null;
    },
    tick(wall: number, mono: number, c: PerceptionCounters) {
      if (source && tour) {
        // The tour: the scripted TickRecords, re-stamped to now (ScenarioSource), with the truth attached.
        const truth = tour.ticks[tourTick % tour.ticks.length];
        tourTick += 1;
        const out = runtime.step(source.nextTick(wall, mono), wall, mono);
        for (const m of out) {
          if (m.type !== "prediction") continue;
          const a = affectFor(m.person_id);
          if (a) (m as { affect?: unknown }).affect = a;
          if (truth) (m as { truth?: unknown }).truth = { state: truth.true_state, valence: truth.true_valence, arousal: truth.true_arousal };
        }
        keepInputs(out);
        return out as ServerMessage[];
      }
      runtime.setPurgeCounters({
        video_purged: c.frames_purged, audio_purged: c.audio_blocks_purged,
        transcripts_purged: c.transcripts_purged ?? 0, violations: c.violations ?? 0, live_raw_buffers: c.live_raw_buffers ?? 0,
      });
      runtime.setFramesDecoded(c.frames_decoded);
      const msgs = runtime.stepLive(wall, mono);
      for (const m of msgs) {
        if (m.type !== "prediction") continue;
        const a = affectFor(m.person_id === ROOM && runtime.roomOnly ? ROOM : m.person_id);
        if (a) (m as { affect?: unknown }).affect = a;
      }
      keepInputs(msgs);
      return msgs as ServerMessage[];
    },
    setRoomOnly: (v: boolean) => runtime.setRoomOnly(v),
    setOverlay: (which: OverlayName, enabled: boolean) => runtime.setOverlay(which, enabled),
    logBreak: () => runtime.logBreak(null),
    startBaseline: (seconds: number | null, mode: "capturing" | "auto") => {
      if (mode === "capturing") runtime.captureRoomBaseline(seconds ?? runtime.settings.room_baseline_s);
      else runtime.resetRoomBaseline();
    },
    baseline: (): RoomBaseline => runtime.aggregator.baseline.snapshot(),
    resetPeople: (keepRoom = true) => { runtime.resetPeople(keepRoom); inputs.clear(); },
    audit: (action: string) => runtime.recordAudit("operator", { action }),
    latestInputs(personId: number) {
      const x = inputs.get(personId);
      if (!x || !emotion) return null;
      return { feature_version: emotion.meta.feature_version, features: inputsByName(emotion.meta.input_features, x) };
    },
  };
}

/** Names of the 123 core features, re-exported for callers that build their own vectors. */
export { FEATURE_NAMES };
