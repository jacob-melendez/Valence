/**
 * Group per-feature attributions into the PRD's explanation layers and build narrator fact sheets
 * (valence/explain/explainer.py), plus the confidence band (valence/explain/narrator_client.py:20).
 *
 * Rounding is Python's round() on floats (num.pyRound: exact binary value, ties to even), and every sort is
 * CPython's sorted(..., reverse=True) (pySortedDesc), so drivers with equal |shap| keep their input order and
 * even NaN keys land where Python puts them.
 */
import {
  LAYERS,
  type DataQuality,
  type Driver,
  type FaceOnly,
  type FactSheet,
  type Layer,
  type LayerName,
} from "./contracts";
import { describe } from "./describe";
import { pyIntRound, pyRound } from "./num";

// ---------- Dict helpers ----------

/**
 * A Python `dict[str, float]`: a plain object or a Map. Iteration follows insertion order, as in Python; for
 * plain objects that holds because feature and state names are never array-index-like keys ("0", "1", ...),
 * which JS would move to the front.
 */
export type FloatDict = Readonly<Record<string, number>> | ReadonlyMap<string, number>;

function entriesOf(d: FloatDict): Iterable<readonly [string, number]> {
  return d instanceof Map ? d.entries() : Object.entries(d as Readonly<Record<string, number>>);
}

/** `d.get(key)`: undefined (None) when absent; own keys only, so "constructor" and friends are absent. */
function getOf(d: FloatDict, key: string): number | undefined {
  if (d instanceof Map) return d.get(key);
  const rec = d as Readonly<Record<string, number>>;
  return Object.hasOwn(rec, key) ? rec[key] : undefined;
}

/** Python `slice[:n]` bound: an int (negative counts from the end), else TypeError like CPython. */
function sliceBound(n: number, what: string): number {
  if (!Number.isInteger(n))
    throw new TypeError(`slice indices must be integers or None (${what}=${n})`);
  return n;
}

// ---------- Python sorted(..., reverse=True) ----------

/**
 * Python `sorted(xs, key=key, reverse=True)` (CPython 3.12 Objects/listobject.c list_sort_impl).
 *
 * Why not Array.prototype.sort: on totally ordered keys any stable sort gives Python's answer, but a NaN key
 * (a NaN SHAP value, or a layer total of inf + -inf) compares false both ways, and the order then depends on
 * the exact algorithm. CPython reverses the list, sorts ascending using `<` only, and reverses back. Under 64
 * items the sort is one natural run (count_run) extended by binary insertion (binarysort), ported here step for
 * step, so it agrees in every JS engine. From 64 items on CPython merges runs with galloping, which is not
 * ported: there Array.prototype.sort runs on the reversed list with an ascending `<` comparator. V8's sort is
 * itself a port of CPython's timsort and matched Python on every NaN-laden 64-100 item list tried; other
 * engines are exact there only for NaN-free keys (served SHAP values and layer totals are always finite).
 */
function pySortedDesc<T>(xs: readonly T[], key: (x: T) => number): T[] {
  const items = [...xs].reverse();
  const keys = items.map(key);
  const n = items.length;
  const k = (i: number): number => keys[i] as number;
  if (n >= 64) {
    const order = items.map((_, i) => i);
    order.sort((a, b) => (k(a) < k(b) ? -1 : k(b) < k(a) ? 1 : 0)); // stable, ascending
    return order.map((i) => items[i] as T).reverse();
  }
  if (n >= 2) {
    // count_run: the longest prefix that is strictly descending, or else non-descending
    let run = 2;
    const descending = k(1) < k(0);
    if (descending) while (run < n && k(run) < k(run - 1)) run++;
    else while (run < n && !(k(run) < k(run - 1))) run++;
    if (descending) {
      for (let i = 0, j = run - 1; i < j; i++, j--) {
        [keys[i], keys[j]] = [k(j), k(i)];
        [items[i], items[j]] = [items[j] as T, items[i] as T];
      }
    }
    // binarysort: insert each later item after every key it is not less than (stable)
    for (let start = run; start < n; start++) {
      const pivot = k(start);
      const item = items[start] as T;
      let l = 0;
      let r = start;
      do {
        const p = l + ((r - l) >> 1);
        if (pivot < k(p)) r = p;
        else l = p + 1;
      } while (l < r);
      for (let p = start; p > l; p--) {
        keys[p] = k(p - 1);
        items[p] = items[p - 1] as T;
      }
      keys[l] = pivot;
      items[l] = item;
    }
  }
  return items.reverse();
}

/** Python `sorted(xs, key=lambda d: abs(d.shap), reverse=True)`. */
function byAbsShapDesc<T extends { shap: number }>(xs: readonly T[]): T[] {
  return pySortedDesc(xs, (d) => Math.abs(d.shap));
}

// ---------- Layers ----------

/** Keyword arguments of buildLayers (valence/explain/explainer.py:19 build_layers). */
export interface BuildLayersOptions {
  /** Local time of day in hours, for the tod_sin / tod_cos drivers; null/undefined drops them. */
  todH?: number | null | undefined;
  /** Drivers kept per layer (default 3). */
  driversPerLayer?: number | undefined;
  /**
   * Feature name -> value for the absolute room levels quoted next to room-relative features
   * (`db_room_mean_5m` reads `db_mean_5m`). Defaults to `values`.
   */
  absolute?: FloatDict | null | undefined;
}

/**
 * Sum attributions per layer; keep each layer's strongest narratable drivers (valence/explain/explainer.py:19).
 *
 * Positive attribution supports the predicted label, negative opposes it. Layers are ordered by contribution,
 * strongest support first; the Facial Surface layer is kept even when it opposes, so a corrected mask stays
 * visible. `contribs` is used as given: the served predictor drops exact-zero SHAP values before calling this
 * (valence/models/served.py:125), so callers mirroring it must do the same.
 */
export function buildLayers(
  contribs: FloatDict,
  values: FloatDict,
  opts: BuildLayersOptions = {},
): Layer[] {
  const todH = opts.todH ?? null;
  const driversPerLayer = opts.driversPerLayer ?? 3;
  const ref = opts.absolute ?? values; // absolute levels behind room-relative features
  const perLayer = new Map<LayerName, Driver[]>();
  const totals = new Map<LayerName, number>();
  for (const [feature, shap] of entriesOf(contribs)) {
    const value = getOf(values, feature);
    const desc = describe(feature, value, {
      todH,
      absolute: getOf(ref, feature.replaceAll("_room", "")),
    });
    if (desc === null) continue;
    totals.set(desc.layer, (totals.get(desc.layer) ?? 0.0) + shap); // defaultdict(float): starts at +0.0
    let list = perLayer.get(desc.layer);
    if (list === undefined) perLayer.set(desc.layer, (list = []));
    list.push({
      feature,
      display: desc.display,
      value: value === undefined ? null : pyRound(value, 4),
      unit: desc.unit,
      shap: pyRound(shap, 4),
      level: desc.level,
      effect: shap >= 0 ? "supports" : "opposes", // the raw value decides: -0.0 supports, -1e-05 opposes
      scope: desc.scope,
    });
  }
  const layers: Layer[] = [];
  for (const name of LAYERS) {
    const list = perLayer.get(name);
    if (list === undefined) continue;
    // One driver per sensor channel (motion_now and motion_ewm_1m would read as the same sentence).
    // Ranked by the ROUNDED |shap|, as in Python (the key reads Driver.shap).
    const best = new Map<string, Driver>();
    for (const drv of byAbsShapDesc(list)) {
      const ch = channelOf(drv.feature);
      if (!best.has(ch)) best.set(ch, drv);
    }
    // checked here, not up front: Python only evaluates the slice when some layer has drivers
    const drivers = [...best.values()].slice(0, sliceBound(driversPerLayer, "drivers_per_layer"));
    layers.push({ name, contribution: pyRound(totals.get(name) ?? 0.0, 4), drivers });
  }
  return pySortedDesc(layers, (layer) => layer.contribution);
}

/** Sensor channel of a feature name, for the one-driver-per-channel rule (valence/explain/explainer.py:58). */
export function channelOf(feature: string): string {
  if (feature.startsWith("tod_")) return "tod";
  if (feature.startsWith("face_visible")) return "face_visible";
  return feature.split("_")[0] as string;
}

/**
 * The k strongest drivers across layers, optionally only those with one effect; stable on ties
 * (valence/explain/explainer.py:66 top_drivers).
 */
export function topDrivers(
  layers: readonly Layer[],
  k = 5,
  effect: "supports" | "opposes" | null = "supports",
): Driver[] {
  const n = sliceBound(k, "k");
  const drivers = layers.flatMap((layer) =>
    layer.drivers.filter((d) => effect === null || d.effect === effect),
  );
  return byAbsShapDesc(drivers).slice(0, n);
}

// ---------- Fact sheet ----------

/**
 * Keyword arguments of build_fact_sheet (valence/explain/explainer.py:71), snake_case like the Python call
 * and the FactSheet it fills. null means None.
 */
export interface FactSheetArgs {
  event_id: string;
  person: string;
  label: string;
  label_display: string;
  confidence: number;
  probabilities: FloatDict;
  window_fill_min: number;
  n_people: number;
  face_only_label: string | null;
  face_only_confidence: number | null;
  face_valence: number | null;
  face_visible: boolean;
  mask_override: boolean;
  layers: Layer[];
  face_visible_frac: number;
  lux: number | null;
  ood_features: string[];
}

/**
 * What the narrator and the template fallback read (valence/explain/explainer.py:71 build_fact_sheet).
 *
 * Like the pydantic model it fills: `layers` and `ood_features` are fresh lists holding the caller's items
 * (pydantic copies a list on validation, so appending to the caller's list later does not reach the sheet),
 * and a non-integral `n_people` is rejected (pydantic `int` accepts 2.0 but raises on 2.5, NaN and inf).
 */
export function buildFactSheet(a: FactSheetArgs): FactSheet {
  // fromEntries defines own keys, so even a "__proto__" state name stays a plain key, as in the Python dict
  const probabilities: Record<string, number> = Object.fromEntries(
    Array.from(entriesOf(a.probabilities), ([k, v]) => [k, pyRound(v, 4)] as const),
  );
  const faceOnly: FaceOnly = {
    label: a.face_visible ? a.face_only_label : null,
    confidence:
      a.face_only_confidence === null || !a.face_visible
        ? null
        : pyRound(a.face_only_confidence, 4),
    face_valence: a.face_valence === null || !a.face_visible ? null : pyRound(a.face_valence, 2),
  };
  const dataQuality: DataQuality = {
    face_visible_frac: pyRound(a.face_visible_frac, 3),
    lux: a.lux === null ? null : pyRound(a.lux, 1),
    ood_features: [...a.ood_features],
  };
  const confidenceDisplay = `${BigInt(pyIntRound(100 * a.confidence))}%`; // f"{round(100 * confidence)}%"
  if (!Number.isInteger(a.n_people))
    throw new TypeError(`FactSheet.n_people: input should be a valid integer, got ${a.n_people}`);
  return {
    event_id: a.event_id,
    person: a.person,
    label: a.label,
    label_display: a.label_display,
    confidence: pyRound(a.confidence, 4),
    confidence_display: confidenceDisplay,
    probabilities,
    window_fill_min: pyRound(a.window_fill_min, 1),
    n_people: a.n_people + 0, // a Python int has no -0
    face_only: faceOnly,
    face_visible: a.face_visible,
    mask_override: a.mask_override,
    layers: [...a.layers],
    data_quality: dataQuality,
  };
}

// ---------- Confidence band ----------

/** Narration cache band of a confidence (valence/explain/narrator_client.py:20 band). */
export function band(confidence: number): "high" | "medium" | "low" {
  return confidence >= 0.85 ? "high" : confidence >= 0.6 ? "medium" : "low";
}
