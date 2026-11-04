"""Writes frontend/src/test/fixtures/engine/m0-dump.json.gz (lgbm.ts loader parity + decision-rule edge cases).

Run from the repo root: .venv/bin/python -B <this file>. Output is byte-identical on rerun (gzip mtime=0).
Schema (version 1):
  {version, generator: this script's source,
   m0: booster.dump_model() of artifacts/m0.txt minus split_gain / internal_value / internal_weight / leaf_weight
       (statistics LightGBM prints with ~6 digits and no inference path reads),
   edge: [{name, text: model_to_string(), dump: dump_model() untouched,
           cases: [{x, raw: predict(raw_score=True), contrib: predict(pred_contrib=True) reshaped [K][F+1]}]}]}
  NaN -> null, +-inf -> "Infinity" / "-Infinity"; floats via Python repr (exact round-trip through JSON.parse).
"""
import gzip
import io
import itertools
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / "frontend/src/test/fixtures/engine/m0-dump.json.gz"
LOSSY = {"split_gain", "internal_value", "internal_weight", "leaf_weight"}


def strip(node):
    for k in LOSSY & set(node):
        del node[k]
    for c in ("left_child", "right_child"):
        if c in node:
            strip(node[c])


def enc(v):
    if isinstance(v, float):
        return None if math.isnan(v) else ("Infinity" if v == math.inf else "-Infinity" if v == -math.inf else v)
    if isinstance(v, list):
        return [enc(u) for u in v]
    return v


def edge_case(name, booster, X):
    K = booster.num_model_per_iteration()
    cases = []
    for row in X:
        r = row[None, :]
        raw = np.asarray(booster.predict(r, raw_score=True, num_threads=1)).reshape(-1)
        contrib = np.asarray(booster.predict(r, pred_contrib=True, num_threads=1)).reshape(K, -1)
        cases.append({"x": enc([float(v) for v in row]), "raw": enc(raw.tolist()), "contrib": enc(contrib.tolist())})
    return {"name": name, "text": booster.model_to_string(), "dump": booster.dump_model(), "cases": cases}


def main():
    m0 = lgb.Booster(model_file=str(ROOT / "artifacts/m0.txt")).dump_model()
    for t in m0["tree_info"]:
        strip(t["tree_structure"])

    rng = np.random.default_rng(7)
    n = 600
    X = rng.normal(size=(n, 3))
    X[rng.random((n, 3)) < 0.25] = 0.0
    X[rng.random((n, 3)) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0.2).astype(int) + (np.nan_to_num(X[:, 1], nan=1.0) == 0.0).astype(int)
    params = dict(objective="multiclass", num_class=3, num_leaves=6, min_data_in_leaf=10, zero_as_missing=True,
                  verbose=-1, seed=1, deterministic=True, num_threads=1, learning_rate=0.3)
    zero = lgb.train(params, lgb.Dataset(X, y), num_boost_round=3)
    thr = sorted({float(t) for t in zero.trees_to_dataframe()["threshold"].dropna()})
    tiny = [0.0, -0.0, 1e-36, -1e-36, 1e-35, -1e-35, float(np.float32(1e-35)), -float(np.float32(1e-35)),
            float(np.nextafter(np.float32(1e-35).astype(np.float64), 1.0)), math.nan, math.inf, -math.inf, 1e300]
    probe = [list(v) for v in itertools.product(tiny, repeat=3)]  # every zero / tiny / missing combination
    probe += [[t] * 3 for t in thr] + [[float(np.nextafter(t, math.inf))] * 3 for t in thr]  # exact split ties
    probe += X[:40].tolist()
    zero_case = edge_case("zero_as_missing", zero, np.array(probe, dtype=float))

    Xr = rng.normal(size=(50, 2))
    single = lgb.train(dict(objective="regression", verbose=-1, num_threads=1, deterministic=True),
                       lgb.Dataset(Xr, np.full(50, 2.5)), num_boost_round=3)
    single_case = edge_case("single_leaf", single, np.array([[0.5, -1.0], [math.nan, 0.0]]))

    doc = {"version": 1, "generator": Path(__file__).read_text(), "m0": m0, "edge": [zero_case, single_case]}
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    OUT.write_bytes(buf.getvalue())
    print(f"{OUT}: {len(buf.getvalue())} bytes, m0 {len(m0['tree_info'])} trees, "
          f"edge cases {[len(e['cases']) for e in doc['edge']]}", file=sys.stderr)


if __name__ == "__main__":
    main()
