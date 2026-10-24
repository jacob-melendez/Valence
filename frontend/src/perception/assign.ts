/**
 * Minimum-cost one-to-one assignment for the small matrices perception needs (a handful of faces x people), like
 * scipy.optimize.linear_sum_assignment: min(n_rows, n_cols) pairs, row indices ascending. Exact, by dynamic
 * programming over subsets of the larger side, so it is meant for at most ~12 columns.
 */
export function linearSumAssignment(cost: readonly (readonly number[])[]): [number[], number[]] {
  const nRows = cost.length;
  const nCols = nRows ? cost[0]!.length : 0;
  if (!nRows || !nCols) return [[], []];
  if (nRows > nCols) {
    const transposed = Array.from({ length: nCols }, (_, c) => cost.map((row) => row[c]!));
    const [cs, rs] = linearSumAssignment(transposed);
    const pairs = rs.map((r, i) => [r, cs[i]!] as const).sort((a, b) => a[0] - b[0]);
    return [pairs.map((p) => p[0]), pairs.map((p) => p[1])];
  }
  if (nCols > 20) throw new Error("linearSumAssignment: matrix too large for the exact solver");
  // best[mask] = minimum cost of giving rows 0..popcount(mask)-1 the columns in mask.
  const size = 1 << nCols;
  const best = new Float64Array(size).fill(Infinity);
  const choice = new Int32Array(size).fill(-1);
  best[0] = 0;
  for (let mask = 0; mask < size; mask++) {
    const base = best[mask]!;
    if (base === Infinity) continue;
    let row = 0;
    for (let m = mask; m; m &= m - 1) row++;
    if (row >= nRows) continue;
    for (let c = 0; c < nCols; c++) {
      if (mask & (1 << c)) continue;
      const next = mask | (1 << c);
      const v = base + cost[row]![c]!;
      if (v < best[next]!) {
        best[next] = v;
        choice[next] = c;
      }
    }
  }
  let end = -1;
  for (let mask = 0; mask < size; mask++) {
    let bits = 0;
    for (let m = mask; m; m &= m - 1) bits++;
    if (bits === nRows && (end < 0 || best[mask]! < best[end]!)) end = mask;
  }
  const cols = new Array<number>(nRows);
  for (let mask = end, row = nRows - 1; row >= 0; row--) {
    const c = choice[mask]!;
    cols[row] = c;
    mask &= ~(1 << c);
  }
  return [Array.from({ length: nRows }, (_, r) => r), cols];
}
