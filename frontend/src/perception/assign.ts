/**
 * Minimum-cost one-to-one assignment, like scipy.optimize.linear_sum_assignment: min(n_rows, n_cols) pairs, row
 * indices ascending. A port of scipy's solver (scipy/optimize/rectangular_lsap/rectangular_lsap.cpp: Crouse's
 * shortest augmenting path), step for step and in the same floating-point order, so it returns the same pairs as
 * scipy, ties included. As in scipy, +Infinity marks a forbidden pair, and NaN / -Infinity entries or a matrix with
 * no complete assignment throw.
 */
export function linearSumAssignment(cost: readonly (readonly number[])[]): [number[], number[]] {
  let nr = cost.length;
  let nc = nr ? cost[0]!.length : 0;
  if (!nr || !nc) return [[], []];
  // The solver wants at least as many columns as rows: a tall matrix is solved transposed.
  const transpose = nc < nr;
  const c = new Float64Array(nr * nc);
  for (let i = 0; i < nr; i++) {
    const row = cost[i]!;
    if (row.length !== nc) throw new Error("linearSumAssignment: ragged cost matrix");
    for (let j = 0; j < nc; j++) c[transpose ? j * nr + i : i * nc + j] = row[j]!;
  }
  if (transpose) [nr, nc] = [nc, nr];
  for (const x of c) {
    if (Number.isNaN(x) || x === -Infinity) {
      throw new Error("linearSumAssignment: matrix contains invalid numeric entries");
    }
  }

  const u = new Float64Array(nr);
  const v = new Float64Array(nc);
  const shortest = new Float64Array(nc);
  const path = new Int32Array(nc).fill(-1);
  const col4row = new Int32Array(nr).fill(-1);
  const row4col = new Int32Array(nc).fill(-1);
  const sr = new Uint8Array(nr);
  const sc = new Uint8Array(nc);
  const remaining = new Int32Array(nc);

  for (let curRow = 0; curRow < nr; curRow++) {
    // Shortest augmenting path from curRow (augmenting_path in scipy).
    let minVal = 0;
    let numRemaining = nc;
    // Filled in reverse so that a constant matrix gives the identity (scipy #11602).
    for (let it = 0; it < nc; it++) remaining[it] = nc - it - 1;
    sr.fill(0);
    sc.fill(0);
    shortest.fill(Infinity);
    let sink = -1;
    let i = curRow;
    while (sink === -1) {
      let index = -1;
      let lowest = Infinity;
      sr[i] = 1;
      for (let it = 0; it < numRemaining; it++) {
        const j = remaining[it]!;
        const r = minVal + c[i * nc + j]! - u[i]! - v[j]!;
        if (r < shortest[j]!) {
          path[j] = i;
          shortest[j] = r;
        }
        // On equal costs prefer a column that ends the path (matters for small integer costs).
        if (shortest[j]! < lowest || (shortest[j] === lowest && row4col[j] === -1)) {
          lowest = shortest[j]!;
          index = it;
        }
      }
      minVal = lowest;
      if (minVal === Infinity) throw new Error("linearSumAssignment: cost matrix is infeasible");
      const j = remaining[index]!;
      if (row4col[j] === -1) sink = j;
      else i = row4col[j]!;
      sc[j] = 1;
      remaining[index] = remaining[--numRemaining]!;
    }

    // Update the dual variables.
    u[curRow] = u[curRow]! + minVal;
    for (let r = 0; r < nr; r++) {
      if (sr[r] && r !== curRow) u[r] = u[r]! + (minVal - shortest[col4row[r]!]!);
    }
    for (let j = 0; j < nc; j++) {
      if (sc[j]) v[j] = v[j]! - (minVal - shortest[j]!);
    }

    // Augment the previous solution along the path.
    for (let j = sink; ;) {
      const r = path[j]!;
      row4col[j] = r;
      const prev = col4row[r]!;
      col4row[r] = j;
      j = prev;
      if (r === curRow) break;
    }
  }

  if (transpose) {
    // Rows of the transposed problem are the original columns: list the pairs by original row.
    const order = Array.from({ length: nr }, (_, k) => k).sort((a, b) => col4row[a]! - col4row[b]!);
    return [order.map((k) => col4row[k]!), order];
  }
  return [Array.from({ length: nr }, (_, k) => k), Array.from(col4row)];
}
