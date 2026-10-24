"""Golden values for frontend/src/engine/num.ts: numpy reductions and CPython float rounding/formatting.

Writes frontend/src/test/fixtures/engine/num.json.gz:

    {"version": 1, "cases": [{"fn": <num.ts export>, "args": [...], "out": ...}, ...]}

``args`` are the TypeScript call arguments (arrays for the reductions, ``{"sign": true}`` for the "+" format
flag). Floats are written with Python's repr, so JSON.parse yields the identical double; NaN is null and
+-inf are the strings "Infinity" / "-Infinity" (frontend/src/test/engine-fixtures.ts nn() decodes both).

The arrays are adversarial on purpose: lengths 0..400 around numpy's pairwise-summation block edges
(8, 128), huge dynamic ranges and cancellations where the summation order changes the last bits, and NaNs
in random positions (np.nanmean/nanstd sum the zero-filled array, so NaN positions shape the tree).
Signed zeros are left out of the max/min/median/percentile inputs: numpy's answer there depends on its
SIMD lanes and partition order, and num.ts documents that it does not chase it.

Deterministic (fixed seed). Run from the repository root:

    .venv/bin/python -B scripts/export_web_golden_num.py
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
import struct
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "src" / "test" / "fixtures" / "engine" / "num.json.gz"
SEED = 20261024

rng = np.random.default_rng(SEED)
cases: list[dict[str, Any]] = []


# ---------- Encoding ----------

def enc(v: Any) -> Any:
    """JSON-safe value: NaN -> None, +-inf -> "Infinity"/"-Infinity", numpy scalars -> Python numbers."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [enc(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {k: enc(x) for k, x in v.items()}
    return v


def add(fn: str, args: list[Any], out: Any) -> None:
    cases.append({"fn": fn, "args": enc(args), "out": enc(out)})


def add_call(fn: str, args: list[Any], call: Callable[[], Any]) -> None:
    """Like add(), but a Python exception becomes {"raises": "<ExceptionName>"} (the TS export must throw)."""
    try:
        out = call()
    except (OverflowError, ValueError) as e:
        cases.append({"fn": fn, "args": enc(args), "out": {"raises": type(e).__name__}})
        return
    add(fn, args, out)


def quiet(f: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return f()


# ---------- numpy's summation model (checked, so a numpy change fails loudly) ----------

def _pairwise(a: list[float], lo: int, n: int) -> float:
    if n < 8:
        r = 0.0
        for i in range(lo, lo + n):
            r += a[i]
        return r
    if n <= 128:
        r = a[lo:lo + 8]
        i = 8
        while i < n - (n % 8):
            for j in range(8):
                r[j] += a[lo + i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise(a, lo, n2) + _pairwise(a, lo + n2, n - n2)


def model_sum(a: list[float]) -> float:
    """np.add.reduce: identity +0.0 plus the pairwise sum of all elements (no first-element seed)."""
    return 0.0 + _pairwise(a, 0, len(a))


def naive_sum(a: list[float]) -> float:
    r = 0.0
    for x in a:
        r += x
    return r


# ---------- Arrays ----------

BOUNDARY_LENGTHS = [0, 1, 2, 3, 5, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 60, 63, 64, 65, 127, 128, 129, 130, 135,
                    136, 137, 143, 144, 145, 180, 191, 192, 193, 255, 256, 257, 271, 272, 300, 383, 384, 385, 399, 400]


def gen_values(kind: str, n: int) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal(n) * 10.0 ** int(rng.integers(-3, 4)) + float(rng.integers(-5, 6))
    if kind == "db":  # dB-like room levels
        return 55.0 + 8.0 * rng.standard_normal(n)
    if kind == "fraction":  # posture_slump / face_visible fractions with exact 0 and 1
        v = rng.random(n)
        v[rng.random(n) < 0.15] = 0.0
        v[rng.random(n) < 0.1] = 1.0
        return v
    if kind == "valence":
        return rng.uniform(-1.0, 1.0, n)
    if kind == "counts":
        return rng.integers(0, 4, n).astype(float)
    if kind == "range":  # huge dynamic range: the order of additions decides the last bits
        return rng.choice([-1.0, 1.0], n) * 10.0 ** rng.integers(-12, 17, n) * rng.random(n)
    if kind == "cancel":  # large values cancelling around small ones
        big = 10.0 ** float(rng.integers(8, 17))
        v = rng.standard_normal(n)
        v[::3] = big * rng.choice([-1.0, 1.0], len(v[::3]))
        return v
    if kind == "tenths":  # decimals that are not exact in binary
        return rng.integers(-500, 500, n) / 10.0
    if kind == "dupes":  # many ties for the order statistics
        return rng.choice([0.1, 0.2, 0.25, 0.5, 1.0, 2.0], n)
    raise ValueError(kind)


KINDS = ["normal", "db", "fraction", "valence", "counts", "range", "cancel", "tenths", "dupes"]


def with_nans(v: np.ndarray, frac: float) -> np.ndarray:
    v = v.copy()
    if v.size:
        v[rng.random(v.size) < frac] = np.nan
    return v


def build_arrays() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for n in BOUNDARY_LENGTHS:
        kind = KINDS[len(out) % len(KINDS)]
        out.append((kind, gen_values(kind, n)))
        out.append((kind + "+nan", with_nans(gen_values(kind, n), float(rng.choice([0.05, 0.3, 0.7])))))
    for _ in range(70):
        n = int(rng.integers(0, 401))
        kind = str(rng.choice(KINDS))
        v = gen_values(kind, n)
        if rng.random() < 0.5:
            v = with_nans(v, float(rng.uniform(0.0, 0.95)))
        out.append((kind, v))
    # Hand-made edge cases.
    nan, inf = math.nan, math.inf
    out += [
        ("all-nan", np.array([nan] * 9)),
        ("all-nan-1", np.array([nan])),
        ("nan-first", np.array([nan, 1.0, 2.0, 3.0])),
        ("nan-last", np.array([1.0, 2.0, 3.0, nan])),
        ("one", np.array([7.25])),
        ("two", np.array([0.1, 0.2])),
        ("inf", np.array([1.0, inf, 2.0])),
        ("-inf", np.array([1.0, -inf, 2.0, nan])),
        ("inf-cancel", np.array([inf, -inf, 1.0])),
        ("big", np.array([1.7976931348623157e308, 1.7976931348623157e308])),
        ("subnormal", np.array([5e-324, 1e-310, -5e-324, 2.2250738585072014e-308])),
        ("tie-16", np.array([1e16, 1.0, 1.0, -1e16] * 4)),
        ("tie-9", np.array([2.0 ** 53, 1.0] * 9)),
    ]
    return out


SIGNED_ZERO_ARRAYS = [
    np.array([-0.0]),
    np.array([-0.0, -0.0]),
    np.array([-0.0] * 8),
    np.array([-0.0] * 9),
    np.array([-0.0] * 130),
    np.array([-0.0, 0.0, -0.0]),
    np.array([-1.0, 1.0]),
    np.array([1.0, -1.0, -0.0]),
    np.array([-0.0, math.nan, -0.0]),
]


def has_neg_zero(v: np.ndarray) -> bool:
    return bool(np.any((v == 0) & np.signbit(v)))


# ---------- Reductions ----------

def export_reductions() -> None:
    arrays = build_arrays()
    differs_naive = 0
    for kind, v in arrays:
        a = [float(x) for x in v]
        ref = quiet(lambda: float(np.sum(v)))
        assert struct.pack("<d", ref) == struct.pack("<d", model_sum(a)) or (math.isnan(ref) and math.isnan(model_sum(a))), (
            f"numpy's summation changed ({kind}, n={len(a)})")
        if not math.isnan(ref) and ref != naive_sum(a):
            differs_naive += 1
        add("npSum", [v], ref)
        if len(cases) % 5 == 0:
            add("pairwiseSum", [v], float(np.add.reduce(v)))
        if v.size:
            add("npMean", [v], quiet(lambda: float(np.mean(v))))
        add("nanmean", [v], quiet(lambda: float(np.nanmean(v))))
        add("nanstd", [v], quiet(lambda: float(np.nanstd(v))))
        if len(cases) % 3 == 0:
            add("nanstd", [v, 1], quiet(lambda: float(np.nanstd(v, ddof=1))))
            add("nanvar", [v], quiet(lambda: float(np.nanvar(v))))
        if v.size:
            add("nanmax", [v], quiet(lambda: float(np.nanmax(v))))
            add("nanmin", [v], quiet(lambda: float(np.nanmin(v))))
            add("median", [v], quiet(lambda: float(np.median(v))))
            for q in ([90] if len(cases) % 2 else [25, 75]) + [float(rng.choice([0, 10, 33.3, 50, 99.9, 100]))]:
                add("percentileLinear", [v, q], quiet(lambda: float(np.percentile(v, q))))
        add("nanmedian", [v], quiet(lambda: float(np.nanmedian(v))))
        finite = [x for x in a if not math.isnan(x)]
        if finite:
            add("statsMedian", [finite], statistics.median(finite))
            add("pySum", [finite], sum(finite))
        if np.isnan(v).any() or np.isinf(v).any():
            add("nanToNum", [v], np.nan_to_num(v, nan=0.0))
    for v in SIGNED_ZERO_ARRAYS:  # sums, means and Python's sum only
        add("npSum", [v], float(np.sum(v)))
        add("npMean", [v], float(np.mean(v)))
        add("nanmean", [v], quiet(lambda: float(np.nanmean(v))))
        add("nanstd", [v], quiet(lambda: float(np.nanstd(v))))
        finite = [float(x) for x in v if not math.isnan(x)]
        add("pySum", [finite], sum(finite))
    for v in ([-0.0], [-0.0, -0.0], [-1.0, 1e-300], [1e100, 1.0, -1e100], [0.1] * 10, [1e16, 1.0, -1e16],
              [math.inf, 1.0], [math.inf, -math.inf], [-0.0, -1.0, 1.0]):
        add("pySum", [v], sum(v))
    # Python slices x[-n:] (compute.py windows) are the caller's job; median of a single -0.0 is +0.0:
    add("median", [np.array([-0.0])], float(np.median(np.array([-0.0]))))
    print(f"reductions: {len(arrays)} arrays, {differs_naive} where pairwise != left-to-right")


def export_clip() -> None:
    xs = [-1.5, -0.0, 0.0, 0.3, 1.0, 2.5, math.nan, math.inf, -math.inf, 1e300]
    bounds = [(-1.0, 1.0), (0.0, 1.0), (0.2, 0.2), (-math.inf, math.inf), (math.nan, 1.0), (0.0, math.nan),
              (math.nan, math.nan), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0), (-0.0, 1.0)]
    for x in xs:
        for lo, hi in bounds:
            add("clip", [x, lo, hi], quiet(lambda: float(np.clip(np.float64(x), lo, hi))))


# ---------- Rounding and formatting ----------

def random_doubles(n: int, lo_exp: int = -12, hi_exp: int = 22) -> list[float]:
    mags = 10.0 ** rng.uniform(lo_exp, hi_exp, n)
    return [float(x) for x in mags * rng.choice([-1.0, 1.0], n)]


def exact_ties(n: int, max_j: int = 10) -> list[float]:
    """m / 2**j with m odd: an exact binary value whose decimal expansion ends in 5 at digit j."""
    out = []
    for _ in range(n):
        j = int(rng.integers(1, max_j + 1))
        m = int(rng.integers(0, 20000)) * 2 + 1
        out.append(float(rng.choice([-1.0, 1.0])) * m / 2 ** j)
    return out


def random_bits(n: int) -> list[float]:
    out = []
    while len(out) < n:
        (x,) = struct.unpack("<d", struct.pack("<Q", int(rng.integers(0, 2 ** 63)) * 2 + int(rng.integers(0, 2))))
        if math.isfinite(x):
            out.append(x)
    return out


SPECIAL = [0.0, -0.0, math.nan, math.inf, -math.inf, 5e-324, -5e-324, 2.2250738585072014e-308,
           1.7976931348623157e308, -1.7976931348623157e308, 2.0 ** 52, 2.0 ** 53, 2.0 ** 53 + 2, 2.0 ** 52 - 0.5,
           1e16, 1e15, 9999999999999998.0, 1e17, 1e21, 1e22, 1e23, 1e-4, 1e-5, 0.0001234, 0.00001234, 0.1, 0.2,
           0.3, 1 / 3, 2 / 3, 2.675, 1.005, 0.285, 1.115, 0.125, 0.375, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 9.9999995,
           999999.5, 99999.95, 123456789.0, 100000.0, 1000000.0, 0.000999999, 72.04999999999, 57.25, 3.14159,
           -0.001, -0.004, 0.0049999999999999, 0.005, 0.015, 0.025, 0.035, 0.045]


def export_formatting() -> None:
    finite_special = [x for x in SPECIAL if math.isfinite(x)]
    pool = SPECIAL + exact_ties(260) + random_doubles(260) + random_doubles(120, -8, 4) + [
        round(float(x), 3) for x in rng.uniform(-100, 100, 80)]

    # round(x, ndigits)
    for x in pool:
        nd = int(rng.integers(-3, 9))
        add_call("pyRound", [x, nd], lambda: round(x, nd))
    for x in exact_ties(150, 6):
        j = 0
        while x * 2 ** j != int(x * 2 ** j):
            j += 1
        add("pyRound", [x, max(j - 1, 0)], round(x, max(j - 1, 0)))
    for x in finite_special + random_doubles(40, -320, 308):
        for nd in (0, 2, 4, 15, 17, 20, 300, 323, 324, 400, -1, -308, -309, -400):
            add_call("pyRound", [x, nd], lambda: round(x, nd))

    # round(x) -> int
    for x in [x for x in pool if math.isfinite(x)] + [k + 0.5 for k in range(-6, 7)] + [2.0 ** 52 + 1, 4503599627370495.5]:
        add("pyIntRound", [x], round(x))
    for x in (math.nan, math.inf, -math.inf):
        add_call("pyIntRound", [x], lambda: round(x))

    # np.round / round(np.float64)
    for x in [x for x in pool if math.isfinite(x)][:300]:
        d = int(rng.integers(-3, 12))
        ref = quiet(lambda: round(np.float64(x), d))
        assert float(ref) == float(quiet(lambda: np.round(x, d))) or math.isnan(float(ref))
        add("npRound", [x, d], float(ref))
    for x in (0.5, 1.5, 2.5, -0.5, -2.5, 2.675, 0.125):
        add("npRound", [x, 0], float(np.round(np.float64(x), 0)))
        add("npRound", [x, 2], float(np.round(np.float64(x), 2)))
    for x in (0.5, 1.5, 2.5, -0.5, -0.4, 0.4, -2.5, 3.5, 1e300, -1e-300, math.nan, math.inf):
        add("rint", [x], float(np.rint(x)))

    # f"{x:.Nf}" and f"{x:+.Nf}"
    for x in pool:
        d = int(rng.choice([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 10, 17, 20]))
        if rng.random() < 0.35:
            add("pyFixed", [x, d, {"sign": True}], format(x, f"+.{d}f"))
        else:
            add("pyFixed", [x, d], format(x, f".{d}f"))
    for x in exact_ties(150, 6):
        j = 0
        while x * 2 ** j != int(x * 2 ** j):
            j += 1
        add("pyFixed", [x, max(j - 1, 0)], format(x, f".{max(j - 1, 0)}f"))
    for x in finite_special[:40]:
        add("pyFixed", [x, 1], format(x, ".1f"))
        add("pyFixed", [x, 2, {"sign": True}], format(x, "+.2f"))
        add("pyFixed", [x, 3], format(x, ".3f"))

    # f"{x:g}" and f"{x:.Pg}"
    for x in pool:
        if rng.random() < 0.75:
            add("pyG", [x], format(x, "g"))
        else:
            p = int(rng.choice([0, 1, 2, 3, 4, 10, 15, 17]))
            add("pyG", [x, p], format(x, f".{p}g"))
    for k in range(20):  # 6-significant-digit exact ties: 1234565 -> 1.23456e+06
        m = int(rng.integers(100000, 999999)) * 10 + 5
        x = float(m) * 10.0 ** int(rng.integers(-3, 3)) if rng.random() < 0.3 else float(m)
        add("pyG", [x], format(x, "g"))
        add("pyG", [m / 2 ** 4], format(m / 2 ** 4, "g"))

    # repr(x)
    for x in SPECIAL + random_bits(700) + random_doubles(300, -30, 30) + [10.0 ** k for k in range(-20, 25)] + [
            float(f"1e{k}") for k in range(-325, 309, 7)] + [float(x) for x in rng.uniform(0, 1, 100)]:
        add("pyRepr", [x], repr(x))


def main() -> None:
    export_reductions()
    export_clip()
    export_formatting()
    payload = {"version": 1, "numpy": np.__version__, "seed": SEED, "cases": cases}
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:  # mtime=0: byte-identical reruns
        f.write(text.encode("utf-8"))
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["fn"]] = counts.get(c["fn"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)}: {len(cases)} cases, {OUT.stat().st_size / 1024:.0f} KiB gzipped")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
