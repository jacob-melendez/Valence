"""Evaluate LightGBM ``dump_model()`` JSON without LightGBM: the reference the TypeScript runtime ports.

Numerical splits only (the emotion models have no categorical features), LightGBM's missing-value rules:

* ``missing_type == "NaN"``: a NaN input goes to ``default_left ? left : right``;
* ``missing_type == "Zero"``: NaN is read as 0, and a value with ``|x| <= 1e-35`` goes the default way;
* ``missing_type == "None"``: NaN is read as 0, then compared;
* otherwise ``x <= threshold`` goes left.

Multiclass dumps hold ``num_class`` trees per round in class order (tree ``i`` adds to class ``i % num_class``); the raw
score is the plain sum of leaf values (the regressors' first tree already carries the average).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

K_ZERO = 1e-35


def load_dump(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def _leaf(node: dict[str, Any], x: np.ndarray) -> float:
    while "leaf_value" not in node:
        v = float(x[node["split_feature"]])
        missing = node.get("missing_type", "None")
        if math.isnan(v) and missing != "NaN":
            v = 0.0
        if (missing == "Zero" and abs(v) <= K_ZERO) or (missing == "NaN" and math.isnan(v)):
            go_left = bool(node["default_left"])
        else:
            if node.get("decision_type", "<=") != "<=":
                raise ValueError(f"unsupported decision_type {node['decision_type']!r}")
            go_left = v <= float(node["threshold"])
        node = node["left_child"] if go_left else node["right_child"]
    return float(node["leaf_value"])


def raw_scores(dump: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Raw scores (one per class; one value for a regressor) for one input vector."""
    k = int(dump.get("num_tree_per_iteration", dump.get("num_class", 1)))
    out = np.zeros(k)
    for i, tree in enumerate(dump["tree_info"]):
        out[i % k] += _leaf(tree["tree_structure"], x)
    return out
