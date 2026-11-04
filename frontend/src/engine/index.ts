/**
 * Public API of the Valence Web engine: the Python engine (valence/) ported to synchronous, DOM-free TypeScript.
 *
 * The pipeline, in the order data flows:
 * - contracts: observations, TickRecords, settings and message types (valence/contracts.py, valence/config.py)
 * - aggregator / replay: observations -> one TickRecord per person per tick (valence/runtime/aggregator.py,
 *   valence/runtime/recording.py)
 * - window / features: rolling 15-minute window -> the 123 window features (valence/features/)
 * - lgbm / treeshap / served / smoother: LightGBM models, TreeSHAP, the served M2 + M0 predictor and the
 *   per-person smoother (valence/models/)
 * - describe / explainer / template: explanation layers, fact sheets and template text (valence/explain/)
 * - audit: the hash-chained audit log (valence/governance/audit.py)
 * - runtime: the tick loop that turns records into governance, state_vector and prediction messages
 *   (valence/runtime/engine.py)
 * - emotion-features / emotion-model: the emotion-v1 map (valence/features/emotion.py, valence/models/emotion.py)
 *
 * Names exported by more than one module are taken from their owner (ROOM from contracts, localTod from
 * aggregator, the fetch types from served); the low-level numerics and hashes are grouped as `num` and `sha`.
 * describe.ts's `describe` is exported as `describeFeature` so it cannot shadow a test runner's describe.
 */

// ---------- Contracts ----------

export {
  AFFECT_ANCHORS,
  ALL_STATES,
  CONTRACT_VERSION,
  CORE_STATES,
  DEFAULT_SETTINGS,
  EMOTION_DISPLAY,
  EMOTION_STATES,
  EXTRA_STATES,
  FACE_CHANNELS,
  GOVERNANCE_LINE,
  LAYERS,
  NEVER_NARRATED,
  P1_STATES,
  PERSON_CHANNELS,
  ROOM,
  ROOM_CHANNELS,
  ROOM_PREFIXES,
  STATE_DISPLAY,
  STATE_LEXICON,
  STATE_SETS,
  STATE_TIER,
  TICK_FIELDS,
  TICK_INT_FIELDS,
  TIME_CHANNELS,
  TIME_FEATURES,
  WITHDRAW_MOTION_MAX,
  WITHDRAW_SLUMP_MIN,
  displayLabel,
  layerOf,
  makeAudioObs,
  makePersonObs,
  makeSettings,
  makeTickRecord,
  makeVideoObs,
  obsFromDict,
  obsToDict,
  scopeOf,
  stateInfos,
  tickRecordFromDict,
  tickRecordToDict,
  withdrawalFlag,
} from "./contracts";
export type {
  Affect,
  AffectStateInfo,
  AudioObs,
  DataQuality,
  Driver,
  ExplanationMsg,
  FaceOnly,
  FactSheet,
  GovernanceMsg,
  HeartbeatMsg,
  HelloMsg,
  Layer,
  LayerName,
  NarrationResult,
  NarrationSource,
  Obs,
  PersonObs,
  PredictionMsg,
  ServerMessage,
  Settings,
  Source,
  StateId,
  StateInfo,
  StateVectorMsg,
  StatusMsg,
  TickRecord,
  TickRecordDict,
  TickRecordInit,
  TrackPerson,
  TracksMsg,
  VideoObs,
} from "./contracts";

// ---------- Numerics and hashes ----------

export * as num from "./num";
export * as sha from "./sha";

// ---------- Aggregation and replay ----------

export {
  Aggregator,
  CUE_FIELDS,
  DRIFT_DB,
  DRIFT_LUX_LOG2,
  DRIFT_WINDOW_S,
  HEAD_AWAY_DEG,
  NEAR_FIELD_BLOCKS,
  NEAR_FIELD_MIN_BLOCKS,
  RoomBaseline,
  localTod,
  roomRecord,
} from "./aggregator";
export type { BaselineMode, BaselineSnapshot, BaselineStatus, RoomFields } from "./aggregator";
export { parseJsonl, replayInto, replayOffline, restamp } from "./replay";
export type { ReplayClock, ReplayOptions, ReplaySink, ReplayTick } from "./replay";

// ---------- Window and features ----------

export { RollingWindow } from "./window";
export {
  ABS_ROOM,
  CONTEXT,
  FACE_FEATURES,
  FEATURE_INDEX,
  FEATURE_NAMES,
  FEATURE_SETS,
  MAX_TICKS,
  N_FEATURES,
  NOISY_LIVE,
  ROOM_REL,
  SNAPSHOT,
  TICK_S,
  WIN,
  arraysFromRecords,
  compute,
  featureIndex,
  featureSet,
  featuresFromWindow,
  selectFeatures,
} from "./features";
export type { Channels, NumericField, WindowName } from "./features";

// ---------- Models ----------

export {
  K_ZERO_THRESHOLD,
  MISSING_NAN,
  MISSING_NONE,
  MISSING_ZERO,
  decide,
  leafIndex,
  loadDumpModel,
  parseLightGBMText,
  predictRaw,
  predictorRow,
} from "./lgbm";
export type { Booster, Tree } from "./lgbm";
export { expectedValue, treeShap } from "./treeshap";
export {
  M0_MIN_CONF,
  M2_MIN_CONF,
  OOD_MARGIN,
  OOD_SHRINK,
  OOD_SHRINK_AFTER,
  ServedPredictor,
  loadServed,
  npArgmax,
  oodBand,
  parsePythonJson,
  softmax,
} from "./served";
export type {
  ClipBounds,
  Counterfactual,
  FetchLike,
  FetchResponseLike,
  M0Reading,
  PredictionOutput,
  PredictionTrace,
  Predictor,
  ServedArtifacts,
  ServedMeta,
} from "./served";
export { PersonSmoother } from "./smoother";
export type { PersonSmootherOptions, SmoothedPrediction, SmootherInput } from "./smoother";

// ---------- Explanations ----------

export { WINDOWS, describe as describeFeature } from "./describe";
export type { DescribeOptions, Description, Level, Scope } from "./describe";
export { band, buildFactSheet, buildLayers, channelOf, topDrivers } from "./explainer";
export type { BuildLayersOptions, FactSheetArgs, FloatDict } from "./explainer";
export { OOD_NOTE_AFTER, OPENERS, pick, templateText } from "./template";

// ---------- Governance ----------

export {
  AuditChain,
  AuditPayloadError,
  GENESIS,
  PyJsonDecodeError,
  auditBlob,
  auditDigest,
  checkAuditValue,
  comparePyStr,
  entryBody,
  entryToJsonLine,
  pyJsonDumps,
  pyJsonLoads,
  verifyChain,
  verifyJsonl,
} from "./audit";
export type { AuditEntry, AuditPayload, PyJson, PyJsonDumpsOptions, PyObject } from "./audit";

// ---------- Runtime ----------

export { Runtime, ScenarioSource } from "./runtime";
export type {
  LatestPerson,
  NarratorLike,
  OverlayName,
  PurgeCounters,
  RuntimeMessage,
  RuntimeOptions,
  SourceLabel,
  TickMessage,
} from "./runtime";

// ---------- Emotion map (emotion-v1) ----------

export {
  EMOTION_FEATURE_NAMES,
  EMOTION_FEATURE_VERSION,
  EMOTION_FIELDS,
  EMOTION_INDEX,
  N_EMOTION_FEATURES,
  computeEmotion,
  emotionFromChannels,
  emotionFromWindow,
} from "./emotion-features";
export type { EmotionChannels, EmotionField } from "./emotion-features";
export {
  affectForWindow,
  emotionInput,
  loadEmotionModel,
  makeEmotionModel,
  predictEmotion,
  toAffect,
  validateEmotionMeta,
} from "./emotion-model";
export type {
  EmotionMeta,
  EmotionModel,
  EmotionModelFiles,
  EmotionModelName,
  EmotionResult,
  EmotionStateMeta,
} from "./emotion-model";
