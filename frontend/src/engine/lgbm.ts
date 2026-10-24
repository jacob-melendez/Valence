/**
 * LightGBM inference for the browser engine: model loading and raw scores.
 *
 * The served predictor (valence/models/served.py:76-96) loads artifacts/m2.txt and m0.txt with
 * `lgb.Booster(model_file=...)` and calls `predict(row, raw_score=True)`. This module reads the same text
 * model (format v4) into flat per-tree arrays, mirroring LightGBM's own `Tree` (include/LightGBM/tree.h), and
 * walks them with LightGBM's decision rule, so raw scores match the C++ library bit for bit: leaves are
 * summed in the same order (tree i * K + k adds to class k, starting from 0.0).
 * `loadDumpModel` reads `booster.dump_model()` JSON into the same arrays (for models shipped as JSON).
 * SHAP contributions (`pred_contrib`) live in treeshap.ts.
 *
 * Supported: numerical splits with any missing type (None / Zero / NaN), default_left, single-leaf trees.
 * Rejected at load: categorical splits, linear trees and average_output (random forest) models; none of the
 * Valence artifacts use them.
 */

// ---------- Types ----------

/** LightGBM `MissingType` (decision_type bits 2-3). */
export const MISSING_NONE = 0;
export const MISSING_ZERO = 1;
export const MISSING_NAN = 2;

/** decision_type bit 0: categorical split (unsupported). */
const CATEGORICAL_MASK = 1;
/** decision_type bit 1: missing values go left. */
const DEFAULT_LEFT_MASK = 2;

/**
 * LightGBM's kZeroThreshold (include/LightGBM/meta.h): the float literal 1e-35f widened to double. Inputs with
 * |v| <= this are dropped from the sparse row the C API builds, i.e. read as 0.0 (c_api.cpp
 * RowPairFunctionFromDenseMatric), and Zero-type splits treat them as missing.
 */
export const K_ZERO_THRESHOLD = Math.fround(1e-35);

/**
 * One tree as flat arrays, named after the text model keys (LightGBM src/io/tree.cpp Tree::ToString).
 * Node i < num_leaves - 1 is internal; a child index c < 0 means leaf ~c. Counts are training row counts,
 * the weights TreeSHAP uses for its path fractions.
 */
export interface Tree {
  num_leaves: number;
  split_feature: Int32Array;
  threshold: Float64Array;
  decision_type: Uint8Array;
  left_child: Int32Array;
  right_child: Int32Array;
  leaf_value: Float64Array;
  leaf_count: Float64Array;
  internal_count: Float64Array;
  shrinkage: number;
  /** Depth of the deepest leaf (root children are depth 1; 0 for a single leaf), as Tree::RecomputeMaxDepth. */
  max_depth: number;
}

/** A loaded model. Tree t contributes to class t % numTreePerIteration. */
export interface Booster {
  version: string;
  numClass: number;
  numTreePerIteration: number;
  maxFeatureIdx: number;
  /** Model input width: max_feature_idx + 1 (pred_contrib rows have numFeatures + 1 entries per class). */
  numFeatures: number;
  featureNames: string[];
  objective: string;
  trees: Tree[];
}

// ---------- Text model (format v4) ----------

/**
 * Parse a LightGBM text model (`booster.save_model()` / artifacts/*.txt), as `lgb.Booster(model_file=...)`
 * reads it (LightGBM src/boosting/gbdt_model_text.cpp LoadModelFromString, src/io/tree.cpp Tree(const char*)).
 * Everything after "end of trees" (feature importances, parameters) is ignored.
 */
export function parseLightGBMText(text: string): Booster {
  const lines = text.split(/\r?\n/);
  let i = 0;
  const header = new Map<string, string>();
  // The header runs up to the first "Tree=" block; a bare "tree" line opens the file. A line without "=" is a
  // key with an empty value, as LoadModelFromString reads it: random-forest models flag themselves with a bare
  // "average_output" line, which a key=value-only reader would miss.
  for (; i < lines.length; i++) {
    const line = (lines[i] as string).trim();
    if (line.startsWith("Tree=") || line === "end of trees") break;
    if (line.length === 0) continue;
    const eq = line.indexOf("=");
    if (eq < 0) header.set(line, "");
    else if (eq > 0) header.set(line.slice(0, eq), line.slice(eq + 1));
  }
  const version = header.get("version") ?? "";
  if (version !== "v4")
    throw new Error(`unsupported LightGBM model version "${version}" (expected v4)`);
  if (header.has("average_output"))
    throw new Error("average_output (random forest) models are not supported");
  const numClass = requireInt(header, "num_class");
  // Older files omit num_tree_per_iteration; LightGBM then uses num_class.
  const numTreePerIteration = header.has("num_tree_per_iteration")
    ? requireInt(header, "num_tree_per_iteration")
    : numClass;
  const maxFeatureIdx = requireInt(header, "max_feature_idx");
  // LightGBM refuses a file without feature_names, or with a count other than max_feature_idx + 1.
  const names = header.get("feature_names");
  if (names === undefined) throw new Error("LightGBM model: missing feature_names");
  const featureNames = names.split(" ").filter((s) => s.length > 0);
  if (featureNames.length !== maxFeatureIdx + 1) {
    throw new Error(
      `LightGBM model: ${featureNames.length} feature_names for max_feature_idx ${maxFeatureIdx}`,
    );
  }

  const trees: Tree[] = [];
  while (i < lines.length) {
    const line = (lines[i] as string).trim();
    if (line === "end of trees") break;
    if (!line.startsWith("Tree=")) {
      i++;
      continue;
    }
    const index = Number(line.slice(5));
    if (index !== trees.length)
      throw new Error(`tree blocks out of order: Tree=${index} after ${trees.length} trees`);
    const kv = new Map<string, string>();
    for (i++; i < lines.length; i++) {
      const l = (lines[i] as string).trim();
      if (l.length === 0 || l.startsWith("Tree=") || l === "end of trees") break;
      const eq = l.indexOf("=");
      if (eq > 0) kv.set(l.slice(0, eq), l.slice(eq + 1));
    }
    trees.push(treeFromText(kv, index));
  }
  return finishBooster({
    version,
    numClass,
    numTreePerIteration,
    maxFeatureIdx,
    featureNames,
    objective: header.get("objective") ?? "",
    trees,
  });
}

function requireInt(kv: Map<string, string>, key: string): number {
  const s = kv.get(key);
  const v = s === undefined ? NaN : Number(s);
  if (!Number.isInteger(v)) throw new Error(`LightGBM model: missing or bad "${key}"`);
  return v;
}

// Numbers in the text model are printed with enough digits to round-trip (%.17g); Number() is correctly
// rounded, so every threshold and leaf value comes back as the identical double. LightGBM writes an infinite
// threshold (the open upper end of the last bin) as "inf" and reads it back with std::stod, as +inf: m2.txt
// tree 644 has one, and LightGBM routes x = +inf left there (inf <= inf), which only +inf reproduces.
const SPECIAL: Readonly<Record<string, number>> = {
  inf: Infinity,
  "+inf": Infinity,
  infinity: Infinity,
  "+infinity": Infinity,
  "-inf": -Infinity,
  "-infinity": -Infinity,
  nan: NaN,
  "-nan": NaN,
};

function parseNumber(token: string): number {
  const special = SPECIAL[token.toLowerCase()];
  if (special !== undefined) return special;
  const v = Number(token);
  if (Number.isNaN(v)) throw new Error(`LightGBM model: bad number "${token}"`);
  return v;
}

function numbers(kv: Map<string, string>, key: string, n: number, required: boolean): number[] {
  const s = kv.get(key);
  if (s === undefined || s.trim().length === 0) {
    if (n > 0 && required) throw new Error(`LightGBM tree: missing "${key}"`);
    return [];
  }
  const out = s.trim().split(/\s+/).map(parseNumber);
  if (out.length < n)
    throw new Error(`LightGBM tree: "${key}" has ${out.length} values, expected ${n}`);
  return out.slice(0, n);
}

function treeFromText(kv: Map<string, string>, index: number): Tree {
  const numLeaves = requireInt(kv, "num_leaves");
  if (numLeaves < 1) throw new Error(`Tree=${index}: num_leaves ${numLeaves}`);
  if ((kv.get("num_cat") ?? "0").trim() !== "0")
    throw new Error(`Tree=${index}: categorical splits are not supported`);
  if ((kv.get("is_linear") ?? "0").trim() !== "0")
    throw new Error(`Tree=${index}: linear trees are not supported`);
  const nInternal = numLeaves - 1;
  const leafValue = numbers(kv, "leaf_value", numLeaves, true);
  // Tree(const char*) fills absent optional arrays with zeros: decision_type 0 is a None-type split that sends
  // defaults right, and zero counts leave raw scores intact while TreeSHAP turns NaN (0 / 0), as in LightGBM.
  const decisionType = orZeros(numbers(kv, "decision_type", nInternal, false), nInternal);
  for (const d of decisionType) {
    if (d & CATEGORICAL_MASK)
      throw new Error(`Tree=${index}: categorical splits are not supported`);
  }
  const shrinkage = kv.has("shrinkage") ? Number(kv.get("shrinkage")) : 1;
  return makeTree(
    {
      numLeaves,
      splitFeature: numbers(kv, "split_feature", nInternal, true),
      threshold: numbers(kv, "threshold", nInternal, true),
      decisionType,
      leftChild: numbers(kv, "left_child", nInternal, true),
      rightChild: numbers(kv, "right_child", nInternal, true),
      leafValue,
      leafCount: orZeros(numbers(kv, "leaf_count", numLeaves, false), numLeaves),
      internalCount: orZeros(numbers(kv, "internal_count", nInternal, false), nInternal),
      shrinkage,
    },
    index,
  );
}

function orZeros(values: number[], n: number): number[] {
  return values.length === n ? values : new Array<number>(n).fill(0);
}

// ---------- dump_model() JSON ----------

/** The parts of `booster.dump_model()` this loader reads; everything else is ignored. */
interface DumpNode {
  split_index?: number;
  split_feature?: number;
  threshold?: number;
  decision_type?: string;
  default_left?: boolean;
  missing_type?: string;
  internal_count?: number;
  left_child?: DumpNode;
  right_child?: DumpNode;
  leaf_index?: number;
  leaf_value?: number;
  leaf_count?: number;
}

interface DumpModel {
  version?: string;
  num_class?: number;
  num_tree_per_iteration?: number;
  max_feature_idx?: number;
  objective?: string;
  average_output?: boolean;
  feature_names?: string[];
  tree_info?: {
    tree_index?: number;
    num_leaves?: number;
    num_cat?: number;
    shrinkage?: number;
    tree_structure?: DumpNode;
  }[];
}

const MISSING_BY_NAME: Readonly<Record<string, number>> = {
  None: MISSING_NONE,
  Zero: MISSING_ZERO,
  NaN: MISSING_NAN,
};

/**
 * Load `booster.dump_model()` output (an object, or its JSON text) into the same arrays parseLightGBMText
 * builds from the text model (LightGBM src/io/tree.cpp Tree::ToJSON / NodeToJSON). Node numbering is taken
 * from split_index / leaf_index, so the two loaders agree array for array.
 */
export function loadDumpModel(json: unknown): Booster {
  const d = (typeof json === "string" ? JSON.parse(json) : json) as DumpModel;
  if (d.version !== "v4")
    throw new Error(`unsupported dump_model version "${String(d.version)}" (expected v4)`);
  if (d.average_output) throw new Error("average_output (random forest) models are not supported");
  const numClass = d.num_class;
  const numTreePerIteration = d.num_tree_per_iteration;
  const maxFeatureIdx = d.max_feature_idx;
  if (
    !Number.isInteger(numClass) ||
    !Number.isInteger(numTreePerIteration) ||
    !Number.isInteger(maxFeatureIdx)
  ) {
    throw new Error("dump_model: missing num_class / num_tree_per_iteration / max_feature_idx");
  }
  const trees = (d.tree_info ?? []).map((info, index) => treeFromDump(info, index));
  return finishBooster({
    version: d.version,
    numClass: numClass as number,
    numTreePerIteration: numTreePerIteration as number,
    maxFeatureIdx: maxFeatureIdx as number,
    featureNames: [...(d.feature_names ?? [])],
    objective: d.objective ?? "",
    trees,
  });
}

function treeFromDump(info: NonNullable<DumpModel["tree_info"]>[number], index: number): Tree {
  const numLeaves = info.num_leaves ?? 1;
  if ((info.num_cat ?? 0) !== 0)
    throw new Error(`tree ${index}: categorical splits are not supported`);
  const root = info.tree_structure;
  if (root === undefined) throw new Error(`tree ${index}: no tree_structure`);
  const nInternal = numLeaves - 1;
  const splitFeature = new Array<number>(nInternal).fill(0);
  const threshold = new Array<number>(nInternal).fill(0);
  const decisionType = new Array<number>(nInternal).fill(0);
  const leftChild = new Array<number>(nInternal).fill(0);
  const rightChild = new Array<number>(nInternal).fill(0);
  const internalCount = new Array<number>(nInternal).fill(0);
  const leafValue = new Array<number>(numLeaves).fill(0);
  const leafCount = new Array<number>(numLeaves).fill(0);

  // Child reference as the text format stores it: an internal node's index, or ~leaf.
  const ref = (node: DumpNode): number =>
    node.split_index !== undefined ? node.split_index : ~(node.leaf_index ?? 0);
  const visit = (node: DumpNode): void => {
    if (node.split_index === undefined) {
      // A linear tree's leaf adds leaf_const + leaf_coeff . x (Tree::LinearModelToJSON); the tree-level
      // is_linear flag is not in the dump, so the leaf keys are the only sign. Reading leaf_value alone would
      // silently give the wrong score.
      if ("leaf_const" in node || "leaf_coeff" in node || "leaf_features" in node)
        throw new Error(`tree ${index}: linear trees are not supported`);
      const leaf = node.leaf_index ?? 0;
      if (leaf < 0 || leaf >= numLeaves)
        throw new Error(`tree ${index}: leaf_index ${leaf} out of range`);
      leafValue[leaf] = req(node.leaf_value, "leaf_value");
      leafCount[leaf] = node.leaf_count ?? 0;
      return;
    }
    const s = node.split_index;
    if (s < 0 || s >= nInternal) throw new Error(`tree ${index}: split_index ${s} out of range`);
    if (node.decision_type !== "<=")
      throw new Error(
        `tree ${index}: decision_type "${String(node.decision_type)}" is not supported`,
      );
    const missing = MISSING_BY_NAME[node.missing_type ?? "None"];
    if (missing === undefined)
      throw new Error(`tree ${index}: unknown missing_type "${String(node.missing_type)}"`);
    const left = node.left_child;
    const right = node.right_child;
    if (left === undefined || right === undefined)
      throw new Error(`tree ${index}: node ${s} lacks a child`);
    splitFeature[s] = req(node.split_feature, "split_feature");
    threshold[s] = dumpThreshold(req(node.threshold, "threshold"));
    decisionType[s] = (node.default_left ? DEFAULT_LEFT_MASK : 0) | (missing << 2);
    leftChild[s] = ref(left);
    rightChild[s] = ref(right);
    internalCount[s] = req(node.internal_count, "internal_count");
    visit(left);
    visit(right);
  };
  visit(root);
  return makeTree(
    {
      numLeaves,
      splitFeature,
      threshold,
      decisionType,
      leftChild,
      rightChild,
      leafValue,
      leafCount,
      internalCount,
      shrinkage: info.shrinkage ?? 1,
    },
    index,
  );
}

/**
 * dump_model() passes thresholds through Common::AvoidInf, which prints anything >= 1e300 as 1e300 (and
 * <= -1e300 as -1e300). Split thresholds are bin bounds, so in practice that is an infinite bound (see SPECIAL
 * above); restore it so the dump decides like the text model for x = +-inf.
 */
function dumpThreshold(v: number): number {
  if (v >= 1e300) return Infinity;
  if (v <= -1e300) return -Infinity;
  return v;
}

function req(v: number | undefined, key: string): number {
  if (typeof v !== "number") throw new Error(`dump_model: node lacks "${key}"`);
  return v;
}

// ---------- Construction ----------

interface TreeArrays {
  numLeaves: number;
  splitFeature: number[];
  threshold: number[];
  decisionType: number[];
  leftChild: number[];
  rightChild: number[];
  leafValue: number[];
  leafCount: number[];
  internalCount: number[];
  shrinkage: number;
}

function makeTree(a: TreeArrays, index: number): Tree {
  const n = a.numLeaves - 1;
  const tree: Tree = {
    num_leaves: a.numLeaves,
    split_feature: Int32Array.from(a.splitFeature),
    threshold: Float64Array.from(a.threshold),
    decision_type: Uint8Array.from(a.decisionType),
    left_child: Int32Array.from(a.leftChild),
    right_child: Int32Array.from(a.rightChild),
    leaf_value: Float64Array.from(a.leafValue),
    leaf_count: Float64Array.from(a.leafCount),
    internal_count: Float64Array.from(a.internalCount),
    shrinkage: a.shrinkage,
    max_depth: 0,
  };
  // Every child must point at a node or leaf that exists; walking also yields the depth TreeSHAP sizes its
  // path buffer with (a malformed, cyclic tree would otherwise hang the walk).
  let maxDepth = 0;
  let seen = 0;
  const stack: [number, number][] = n > 0 ? [[0, 0]] : [];
  while (stack.length > 0) {
    const [node, depth] = stack.pop() as [number, number];
    if (++seen > n) throw new Error(`tree ${index}: malformed (cycle)`);
    for (const child of [tree.left_child[node] as number, tree.right_child[node] as number]) {
      if (child >= 0) {
        if (child >= n) throw new Error(`tree ${index}: child ${child} out of range`);
        stack.push([child, depth + 1]);
      } else {
        if (~child >= a.numLeaves) throw new Error(`tree ${index}: leaf ${~child} out of range`);
        maxDepth = Math.max(maxDepth, depth + 1);
      }
    }
  }
  tree.max_depth = maxDepth;
  return tree;
}

function finishBooster(b: Omit<Booster, "numFeatures">): Booster {
  if (b.numClass < 1 || b.numTreePerIteration < 1)
    throw new Error("LightGBM model: bad class count");
  if (b.trees.length % b.numTreePerIteration !== 0) {
    throw new Error(
      `LightGBM model: ${b.trees.length} trees is not a multiple of ${b.numTreePerIteration}`,
    );
  }
  const numFeatures = b.maxFeatureIdx + 1;
  for (const [t, tree] of b.trees.entries()) {
    for (const f of tree.split_feature) {
      if (f < 0 || f >= numFeatures)
        throw new Error(`tree ${t}: split_feature ${f} outside 0..${b.maxFeatureIdx}`);
    }
  }
  return { ...b, numFeatures };
}

// ---------- Prediction ----------

/**
 * The dense row LightGBM's predictor actually sees: the inputs with |v| <= kZeroThreshold (signed zeros
 * included) replaced by 0.0; NaN and +-inf pass through. Like `Booster.predict`'s shape check, a row of the
 * wrong width throws (passing the full 123-feature vector instead of the model's columns is the likely slip).
 * A null or undefined entry (a hole in a plain array) is missing, NaN, as `np.asarray(x, dtype=float)` makes
 * None; without the check it would fall through to 0.0, which a NaN-type split sends the other way.
 */
export function predictorRow(booster: Booster, x: ArrayLike<number>): Float64Array {
  const row = new Float64Array(booster.numFeatures);
  if (x.length !== row.length) {
    throw new Error(`LightGBM model expects ${row.length} features, got ${x.length}`);
  }
  for (let i = 0; i < row.length; i++) {
    const v = x[i] as number | null | undefined;
    if (v === null || v === undefined) row[i] = NaN;
    else if (Math.abs(v) > K_ZERO_THRESHOLD || v !== v) row[i] = v;
  }
  return row;
}

/**
 * One split decision, LightGBM include/LightGBM/tree.h NumericalDecision: NaN on a non-NaN-type split reads as
 * 0.0; a missing value (NaN on a NaN-type split, or |v| <= kZeroThreshold on a Zero-type split) follows
 * default_left; anything else goes left iff v <= threshold.
 */
export function decide(tree: Tree, node: number, value: number): number {
  const dt = tree.decision_type[node] as number;
  const missingType = (dt >> 2) & 3;
  let v = value;
  if (v !== v && missingType !== MISSING_NAN) v = 0.0;
  if (
    (missingType === MISSING_ZERO && v >= -K_ZERO_THRESHOLD && v <= K_ZERO_THRESHOLD) ||
    (missingType === MISSING_NAN && v !== v)
  ) {
    return dt & DEFAULT_LEFT_MASK
      ? (tree.left_child[node] as number)
      : (tree.right_child[node] as number);
  }
  return v <= (tree.threshold[node] as number)
    ? (tree.left_child[node] as number)
    : (tree.right_child[node] as number);
}

/** Leaf index x lands in (Tree::GetLeaf); `row` is a predictorRow. */
export function leafIndex(tree: Tree, row: ArrayLike<number>): number {
  if (tree.num_leaves <= 1) return 0;
  let node = 0;
  while (node >= 0) node = decide(tree, node, row[tree.split_feature[node] as number] as number);
  return ~node;
}

/**
 * Raw scores, `booster.predict(x[None, :], raw_score=True)[0]` (LightGBM GBDT::PredictRaw): one value per class,
 * summing every tree's leaf in model order. `x` holds the model's features in model order (served.py selects
 * them with idx2 / idx0 first).
 */
export function predictRaw(booster: Booster, x: ArrayLike<number>): Float64Array {
  const K = booster.numTreePerIteration;
  const out = new Float64Array(K);
  const row = predictorRow(booster, x);
  const trees = booster.trees;
  for (let t = 0; t < trees.length; t++) {
    const tree = trees[t] as Tree;
    const k = t % K;
    out[k] = (out[k] as number) + (tree.leaf_value[leafIndex(tree, row)] as number);
  }
  return out;
}
