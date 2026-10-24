/**
 * Path-dependent TreeSHAP for LightGBM models, as `booster.predict(x, pred_contrib=True)` computes it.
 *
 * valence/models/served.py:91 reads the M2 contributions with LightGBM's `pred_contrib`; the face push and the
 * explanation layers are built from them. This is Lundberg et al.'s Algorithm 2 exactly as LightGBM implements
 * it (src/io/tree.cpp Tree::TreeSHAP / ExtendPath / UnwindPath / UnwoundPathSum, include/LightGBM/tree.h
 * Tree::PredictContrib, src/boosting/gbdt_prediction.cpp GBDT::PredictContrib): the cover of a node is its
 * training row count (leaf_count / internal_count, not the hessian weights), each tree first adds its expected
 * value to the bias slot, and the arithmetic follows the C++ expression for expression and in the same order.
 * Measured against LightGBM 4.7.0 (src/test/engine-lgbm.test.ts): identical doubles for every class of every
 * golden case of both served models.
 */
import { type Booster, type Tree, decide, predictorRow } from "./lgbm";

// ---------- Expected value ----------

const expectedCache = new WeakMap<Tree, number>();

/**
 * Tree::ExpectedValue: the leaf values weighted by their share of the root's training rows, summed in leaf
 * order. This is what each tree adds to the bias slot (and what makes contributions sum to the raw score).
 */
export function expectedValue(tree: Tree): number {
  const cached = expectedCache.get(tree);
  if (cached !== undefined) return cached;
  let e: number;
  if (tree.num_leaves <= 1) {
    e = tree.leaf_value[0] as number;
  } else {
    const total = tree.internal_count[0] as number;
    e = 0.0;
    for (let i = 0; i < tree.num_leaves; i++)
      e += ((tree.leaf_count[i] as number) / total) * (tree.leaf_value[i] as number);
  }
  expectedCache.set(tree, e);
  return e;
}

// ---------- Path ----------

/**
 * LightGBM's PathElement array as four parallel columns. Each recursion level works on its own slice that
 * starts `unique_depth` elements after its parent's (the C++ `parent_unique_path + unique_depth`), so a
 * sibling call still sees the parent's path untouched.
 */
interface Path {
  feature: Int32Array;
  zero: Float64Array;
  one: Float64Array;
  pweight: Float64Array;
}

function newPath(maxDepth: number): Path {
  // LightGBM allocates (d + 1)(d + 2) / 2 for max depth d; one extra level of slack costs nothing.
  const n = ((maxDepth + 2) * (maxDepth + 3)) / 2;
  return {
    feature: new Int32Array(n),
    zero: new Float64Array(n),
    one: new Float64Array(n),
    pweight: new Float64Array(n),
  };
}

function extendPath(
  p: Path,
  off: number,
  depth: number,
  zero: number,
  one: number,
  feature: number,
): void {
  const pw = p.pweight;
  p.feature[off + depth] = feature;
  p.zero[off + depth] = zero;
  p.one[off + depth] = one;
  pw[off + depth] = depth === 0 ? 1.0 : 0.0;
  for (let i = depth - 1; i >= 0; i--) {
    pw[off + i + 1] =
      (pw[off + i + 1] as number) + (one * (pw[off + i] as number) * (i + 1)) / (depth + 1);
    pw[off + i] = (zero * (pw[off + i] as number) * (depth - i)) / (depth + 1);
  }
}

function unwindPath(p: Path, off: number, depth: number, pathIndex: number): void {
  const pw = p.pweight;
  const one = p.one[off + pathIndex] as number;
  const zero = p.zero[off + pathIndex] as number;
  let nextOnePortion = pw[off + depth] as number;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = pw[off + i] as number;
      pw[off + i] = (nextOnePortion * (depth + 1)) / ((i + 1) * one);
      nextOnePortion = tmp - ((pw[off + i] as number) * zero * (depth - i)) / (depth + 1);
    } else {
      pw[off + i] = ((pw[off + i] as number) * (depth + 1)) / (zero * (depth - i));
    }
  }
  for (let i = pathIndex; i < depth; i++) {
    p.feature[off + i] = p.feature[off + i + 1] as number;
    p.zero[off + i] = p.zero[off + i + 1] as number;
    p.one[off + i] = p.one[off + i + 1] as number;
  }
}

function unwoundPathSum(p: Path, off: number, depth: number, pathIndex: number): number {
  const pw = p.pweight;
  const one = p.one[off + pathIndex] as number;
  const zero = p.zero[off + pathIndex] as number;
  let nextOnePortion = pw[off + depth] as number;
  let total = 0;
  // The C++ tests one_fraction inside the loop; it is loop-invariant, so branch once (same arithmetic).
  if (one !== 0) {
    for (let i = depth - 1; i >= 0; i--) {
      const tmp = (nextOnePortion * (depth + 1)) / ((i + 1) * one);
      total += tmp;
      nextOnePortion = (pw[off + i] as number) - tmp * zero * ((depth - i) / (depth + 1));
    }
  } else {
    for (let i = depth - 1; i >= 0; i--)
      total += (pw[off + i] as number) / zero / ((depth - i) / (depth + 1));
  }
  return total;
}

// ---------- Recursion ----------

function dataCount(tree: Tree, node: number): number {
  return node >= 0 ? (tree.internal_count[node] as number) : (tree.leaf_count[~node] as number);
}

/** Tree::TreeSHAP: `phi` is one class's output row (numFeatures + 1, bias last). */
function recurse(
  tree: Tree,
  row: Float64Array,
  phi: Float64Array,
  p: Path,
  node: number,
  uniqueDepth: number,
  parentOff: number,
  parentZero: number,
  parentOne: number,
  parentFeature: number,
): void {
  // Extend this level's copy of the parent's path with the split that led here.
  const off = parentOff + uniqueDepth;
  // Plain loops: the slices are a handful of elements, where copyWithin's call overhead dominates.
  const { feature: pf, zero: pz, one: po, pweight: pw } = p;
  for (let i = 0; i < uniqueDepth; i++) {
    pf[off + i] = pf[parentOff + i] as number;
    pz[off + i] = pz[parentOff + i] as number;
    po[off + i] = po[parentOff + i] as number;
    pw[off + i] = pw[parentOff + i] as number;
  }
  extendPath(p, off, uniqueDepth, parentZero, parentOne, parentFeature);

  if (node < 0) {
    // Leaf: every feature on the path gets its Shapley share of this leaf's value (element 0 is the root dummy).
    const leaf = tree.leaf_value[~node] as number;
    for (let i = 1; i <= uniqueDepth; i++) {
      const w = unwoundPathSum(p, off, uniqueDepth, i);
      const f = p.feature[off + i] as number;
      phi[f] =
        (phi[f] as number) + w * ((p.one[off + i] as number) - (p.zero[off + i] as number)) * leaf;
    }
    return;
  }

  const feature = tree.split_feature[node] as number;
  const hot = decide(tree, node, row[feature] as number);
  const cold =
    hot === tree.left_child[node]
      ? (tree.right_child[node] as number)
      : (tree.left_child[node] as number);
  const w = dataCount(tree, node);
  const hotZero = dataCount(tree, hot) / w;
  const coldZero = dataCount(tree, cold) / w;
  let incomingZero = 1;
  let incomingOne = 1;

  // A feature split on again further down: undo its earlier split so this node can redo it.
  let depth = uniqueDepth;
  let pathIndex = 0;
  for (; pathIndex <= depth; pathIndex++) {
    if (p.feature[off + pathIndex] === feature) break;
  }
  if (pathIndex !== depth + 1) {
    incomingZero = p.zero[off + pathIndex] as number;
    incomingOne = p.one[off + pathIndex] as number;
    unwindPath(p, off, depth, pathIndex);
    depth -= 1;
  }

  recurse(tree, row, phi, p, hot, depth + 1, off, hotZero * incomingZero, incomingOne, feature);
  recurse(tree, row, phi, p, cold, depth + 1, off, coldZero * incomingZero, 0, feature);
}

// ---------- Public ----------

/**
 * SHAP contributions, `booster.predict(x[None, :], pred_contrib=True)` reshaped to [K][numFeatures + 1]:
 * one row per requested class (default: every class, in order), entry j < numFeatures for model feature j,
 * the last entry the bias (sum of the class's tree expected values). Each row sums to that class's raw score
 * up to rounding. Only the trees of the requested classes are walked, so asking for [k, k0] costs 2/K of all.
 * `x` holds the model's features in model order, like predictRaw.
 */
export function treeShap(
  booster: Booster,
  x: ArrayLike<number>,
  classes?: readonly number[],
): Float64Array[] {
  const K = booster.numTreePerIteration;
  const wanted = classes ?? Array.from({ length: K }, (_, k) => k);
  const slot = new Int32Array(K).fill(-1);
  wanted.forEach((k, i) => {
    if (!Number.isInteger(k) || k < 0 || k >= K)
      throw new Error(`treeShap: class ${k} outside 0..${K - 1}`);
    if (slot[k] === -1) slot[k] = i;
  });
  const out = wanted.map(() => new Float64Array(booster.numFeatures + 1));
  const row = predictorRow(booster, x);
  const bias = booster.numFeatures;

  let maxDepth = 0;
  for (const tree of booster.trees) maxDepth = Math.max(maxDepth, tree.max_depth);
  const path = newPath(maxDepth);

  const trees = booster.trees;
  for (let t = 0; t < trees.length; t++) {
    const s = slot[t % K] as number;
    if (s < 0) continue;
    const tree = trees[t] as Tree;
    const phi = out[s] as Float64Array;
    phi[bias] = (phi[bias] as number) + expectedValue(tree);
    if (tree.num_leaves > 1) recurse(tree, row, phi, path, 0, 0, 0, 1, 1, -1);
  }
  // A class asked for twice gets its own copy of the same row.
  return wanted.map((k, i) =>
    slot[k] === i
      ? (out[i] as Float64Array)
      : Float64Array.from(out[slot[k] as number] as Float64Array),
  );
}
