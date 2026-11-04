"""Parity fixtures for frontend/src/perception/hands.ts and ids.ts (ports of valence/perception/hands.py and ids.py),
plus debounce.ts (valence/perception/debounce.py), which FistTracker is built on.

Runs the real Python functions on seeded synthetic inputs and records what they return:

    hands-hypot.json     math.hypot on assorted pairs (both ports gate and order pairs on it, bit for bit)
    hands-assign.json    assign_hands: ties, gated pairs, low-confidence / NaN / inf wrists and hands, odd gates
    hands-curl.json      curled_fingers / is_curl_fist: synthetic and random hands, several aspects, NaN / inf, shapes
    hands-fist.json      FistTracker op sequences (update / mark_unobserved / person_state / drop / known)
    hands-ids.json       DisplayIdMapper op sequences (update / pop_released / reset) and constructor checks
    hands-debounce.json  HysteresisDebouncer sample sequences (fist / laughter presets, random parameters)

Strict JSON; non-finite floats are the strings "NaN", "Infinity", "-Infinity".
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from valence.perception.debounce import HysteresisDebouncer, fist_debouncer, laughter_debouncer  # noqa: E402
from valence.perception.hands import FistTracker, assign_hands, curled_fingers, is_curl_fist  # noqa: E402
from valence.perception.ids import DisplayIdMapper  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent
SEED = 20261024
NAN, INF = math.nan, math.inf
DBL_MIN, DBL_MAX, DBL_TRUE_MIN = 2.2250738585072014e-308, 1.7976931348623157e308, 5e-324

LEGACY_FIST = (0.6, 0.4, 3, 1.0, 3.0, 0.0)  # tests/perception/test_hands.py legacy_fist()
LEGACY_LAUGH = (0.30, 0.15, 2, 0.0, 3.0, 0.0)


def enc(v: Any) -> Any:
    """Strict-JSON value: non-finite floats become strings, numpy scalars become Python numbers."""
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, np.ndarray):
        return enc(v.tolist())
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    raise TypeError(f"cannot encode {type(v)}")


def rng_for(part: str) -> np.random.Generator:
    return np.random.default_rng([SEED, sum(map(ord, part))])


def debouncer_params(deb: HysteresisDebouncer) -> list[float]:
    """Constructor order of HysteresisDebouncer (and of the TS class)."""
    return [deb.on, deb.off, deb.on_count, deb.off_hold_s, deb.refractory_s, deb.on_hold_s]


def make_debouncer(params: tuple | list) -> HysteresisDebouncer:
    on, off, on_count, off_hold_s, refractory_s, on_hold_s = params
    return HysteresisDebouncer(on, off, on_count, off_hold_s, refractory_s, on_hold_s)


# --------------------------------------------------------------------------------------------------- hypot


def hypot_cases() -> list[list[float]]:
    rng = rng_for("hypot")
    pairs: list[tuple[float, float]] = []
    pairs += [tuple(map(float, p)) for p in rng.integers(-40, 41, (200, 2))]  # integer offsets: exact ties
    px = rng.uniform(0, 1280, (200, 4))
    pairs += [(a - b, c - d) for a, b, c, d in px]  # pixel differences
    nx = rng.uniform(0, 1, (200, 4))
    pairs += [(a - b, (c - d) * (9 / 16)) for a, b, c, d in nx]  # normalized differences, frame widths
    for scale in (1e-3, 1e3, 1e-160, 1e160, 1e-306, 1e306, 1e-312):
        pairs += [tuple(p) for p in rng.normal(0, 1, (25, 2)) * scale]
    wide = rng.normal(0, 1, (100, 2))
    wide[:, 1] *= 10.0 ** rng.uniform(-30, 30, 100)
    pairs += [tuple(p) for p in wide]
    pairs += [tuple(p) for p in rng.integers(0, 2**63, (100, 2), dtype=np.int64).view(np.float64)]  # any bits
    specials = [0.0, -0.0, 1.0, -1.0, 3.0, 4.0, DBL_MIN, DBL_TRUE_MIN, DBL_MAX, -DBL_MAX, 1e-310, INF, -INF, NAN]
    pairs += [(a, b) for a in specials for b in specials]
    return [[float(a), float(b), math.hypot(a, b)] for a, b in pairs]


# --------------------------------------------------------------------------------------------------- assign


def assign_case(hands: list, people: list, frame_w: float, **kw: float) -> dict[str, Any]:
    """``people``: [(pid, [(side, (x, y, conf)), ...]), ...] in dict order."""
    wrists = {pid: dict(sides) for pid, sides in people}
    expected = assign_hands([tuple(h) for h in hands], wrists, frame_w, **kw)
    return {"hands": hands, "people": [[pid, [[s, list(w)] for s, w in sides]] for pid, sides in people],
            "frame_w": frame_w, **kw, "expected": [None if e is None else list(e) for e in expected]}


def lr(lx: float, ly: float, rx: float, ry: float, lc: float = 0.9, rc: float = 0.9, rl: bool = False) -> list:
    sides = [("L", (lx, ly, lc)), ("R", (rx, ry, rc))]
    return sides[::-1] if rl else sides


def assign_cases() -> list[dict[str, Any]]:
    rng = rng_for("assign")
    out = [
        # tests/perception/test_hands.py
        assign_case([[705, 510], [395, 490]], [(1, lr(200, 500, 400, 500)), (2, lr(700, 500, 900, 500))], 1000),
        assign_case([[510, 500], [505, 500]], [(1, lr(500, 500, 0, 0, rc=0.0)), (2, lr(0, 0, 612, 500, lc=0.0))], 1000),
        assign_case([[510, 500], [505, 500]], [(1, lr(500, 500, 0, 0, rc=0.0))], 1000),
        assign_case([[500, 500]], [(1, lr(500, 500, 520, 500, lc=0.49, rc=0.5))], 1000),
        assign_case([[500, 500]], [(1, lr(500, 500, 520, 500, lc=0.2, rc=0.3))], 1000),
        assign_case([[619, 500]], [(1, lr(500, 500, 0, 0, rc=0.0))], 1000),
        assign_case([[620, 500]], [(1, lr(500, 500, 0, 0, rc=0.0))], 1000),  # d == max_dist: inclusive
        assign_case([[621, 500]], [(1, lr(500, 500, 0, 0, rc=0.0))], 1000),
        assign_case([[621, 500]], [(1, lr(500, 500, 0, 0, rc=0.0))], 2000),
        assign_case([[560, 500]], [(1, lr(500, 500, 0, 0, rc=0.0))], 1000, max_dist_frac=0.05),
        assign_case([[NAN, NAN], [505, 500]], [(1, lr(NAN, NAN, 500, 500))], 1000),
        assign_case([], [(1, lr(NAN, NAN, 500, 500))], 1000),
        assign_case([[500, 500]], [], 1000),
        # exact ties: L before R, smaller pid first, smaller hand index first
        assign_case([[500, 500]], [(1, lr(510, 500, 490, 500, rl=True))], 1000),
        assign_case([[500, 500]], [(7, lr(503, 504, 0, 0, rc=0.0)), (3, lr(504, 503, 0, 0, rc=0.0))], 1000),
        assign_case([[500, 500]], [(3, lr(0, 0, 505, 500, lc=0.0)), (7, lr(500, 505, 0, 0, rc=0.0))], 1000),
        assign_case([[497, 504], [503, 496]], [(1, lr(500, 500, 0, 0, rc=0.0))], 1000),
        assign_case([[500, 500], [500, 500]], [(2, lr(500, 500, 500, 500)), (1, lr(500, 500, 500, 500))], 1000),
        # display id breaks a tie before side
        assign_case([[500, 500]], [(2, lr(510, 500, 0, 0, rc=0.0)), (1, lr(0, 0, 490, 500, lc=0.0))], 1000),
        # 3-4-5 exactly on the gate: 0.05 * 100 == 5.0
        assign_case([[3, 4], [4, 3], [5, 0]], [(1, lr(0, 0, 9, 4)), (2, lr(0, 9, 100, 100))], 100,
                    max_dist_frac=0.05),
        # odd gates and confidences
        assign_case([[INF, 0], [0, -INF], [10, 10]], [(1, lr(0, 0, 1e300, 0)), (2, lr(5, 5, 7, 7))], 100,
                    max_dist_frac=INF),
        assign_case([[INF, NAN], [NAN, 3]], [(1, lr(0, 0, 1, 1))], 100, max_dist_frac=INF),
        assign_case([[1, 1]], [(1, lr(1, 1, 2, 2))], 100, max_dist_frac=NAN),
        assign_case([[1, 1], [2, 2]], [(1, lr(1, 1, 2, 2))], 0),
        assign_case([[1, 1]], [(1, lr(1, 1, 2, 2, lc=NAN, rc=0.6))], 100),
        assign_case([[1, 1]], [(1, lr(1, 1, 2, 2))], 100, min_conf=NAN),
        assign_case([[1, 1]], [(1, lr(1, 1, 2, 2, lc=-5.0, rc=NAN))], 100, min_conf=-INF),
        assign_case([[1, 1]], [(1, lr(INF, 1, -INF, 1)), (2, lr(1, NAN, 3, 1))], 100),
        assign_case([[-0.0, 0.0]], [(0, lr(0.0, -0.0, 1, 1)), (-4, lr(0, 0, 0, 0))], 100),
    ]
    confs = [0.0, 0.3, 0.49, 0.5, 0.51, 0.9, 1.0, NAN]
    for i in range(160):  # integer grid: many exactly equal distances
        n_people, n_hands = int(rng.integers(0, 6)), int(rng.integers(0, 7))
        pids = [int(p) for p in rng.choice([0, 1, 2, 3, 4, 5, 9, 17, 2**40, -3], n_people, replace=False)]
        people = []
        for pid in pids:
            x = rng.integers(0, 31, 4)
            cs = [confs[int(k)] if rng.random() < 0.4 else 0.9 for k in rng.integers(0, len(confs), 2)]
            people.append((pid, lr(float(x[0]), float(x[1]), float(x[2]), float(x[3]), cs[0], cs[1],
                                   rl=bool(rng.random() < 0.3))))
        hands = [[float(v) for v in rng.integers(0, 31, 2)] for _ in range(n_hands)]
        kw: dict[str, float] = {}
        if i % 3 == 1:
            kw["max_dist_frac"] = float(rng.choice([0.05, 0.08, 0.2, 1.0]))
        if i % 4 == 2:
            kw["min_conf"] = float(rng.choice([0.0, 0.3, 0.51]))
        out.append(assign_case(hands, people, 100, **kw))
    for i in range(140):  # people spread over a 1280 x 720 frame, hands near (some) wrists
        n_people, n_hands = int(rng.integers(1, 6)), int(rng.integers(0, 8))
        people, wrists = [], []
        for pid in rng.permutation(np.arange(1, 9))[:n_people]:
            cx, cy = rng.uniform(100, 1180), rng.uniform(200, 650)
            lx, ly = cx + rng.normal(60, 40), cy + rng.normal(0, 60)
            rx, ry = cx - rng.normal(60, 40), cy + rng.normal(0, 60)
            lc, rc = rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)
            if rng.random() < 0.08:
                lx = float(rng.choice([NAN, INF, -INF]))
            if rng.random() < 0.08:
                ry = float(rng.choice([NAN, INF, -INF]))
            if rng.random() < 0.05:
                lc = NAN
            people.append((int(pid), lr(lx, ly, rx, ry, lc, rc, rl=bool(rng.random() < 0.3))))
            wrists += [(lx, ly), (rx, ry)]
        hands = []
        for _ in range(n_hands):
            if wrists and rng.random() < 0.75:
                wx, wy = wrists[int(rng.integers(len(wrists)))]
                h = [wx + rng.normal(0, 70), wy + rng.normal(0, 70)]
            else:
                h = [rng.uniform(0, 1280), rng.uniform(0, 720)]
            if rng.random() < 0.06:
                h[int(rng.integers(2))] = float(rng.choice([NAN, INF]))
            hands.append([float(v) for v in h])
        kw = {}
        if i % 5 == 3:
            kw["max_dist_frac"] = float(rng.uniform(0.01, 0.4))
        if i % 7 == 4:
            kw["min_conf"] = float(rng.uniform(0.0, 0.9))
        frame_w = 1280 if i % 6 else int(rng.choice([640, 1920]))
        out.append(assign_case(hands, people, frame_w, **kw))
    return out


# --------------------------------------------------------------------------------------------------- curl


def make_hand(curled: set[int] = frozenset(), wrist: tuple[float, float] = (0.5, 0.8)) -> np.ndarray:
    """tests/perception/test_hands.py make_hand: synthetic upright hand (21, 3); ``curled`` holds fingers 0..3."""
    lm = np.zeros((21, 3))
    wx, wy = wrist
    lm[0, :2] = wrist
    for i, (x, y) in enumerate([(-0.04, -0.03), (-0.06, -0.06), (-0.07, -0.09), (-0.08, -0.11)], start=1):
        lm[i, :2] = (wx + x, wy + y)
    for f in range(4):
        x = wx - 0.03 + 0.02 * f
        base = 5 + 4 * f
        lm[base, :2] = (x, wy - 0.15)
        lm[base + 1, :2] = (x, wy - 0.22)
        if f in curled:
            lm[base + 2, :2] = (x, wy - 0.18)
            lm[base + 3, :2] = (x, wy - 0.12)
        else:
            lm[base + 2, :2] = (x, wy - 0.27)
            lm[base + 3, :2] = (x, wy - 0.31)
    return lm


ASPECTS = [9 / 16, 3 / 4, 1.0, 16 / 9, 0.0, 1e-9, 1e9, -1.0, NAN, INF]
MIN_CURLED = [0, 1, 2, 3, 4, 5, -1, 2.5, NAN]


def curl_case(lm: list, aspect: float | None = None, min_curled: float | None = None) -> dict[str, Any]:
    kw_c = {} if aspect is None else {"aspect": aspect}
    kw_f = dict(kw_c) if min_curled is None else {**kw_c, "min_curled": min_curled}
    case: dict[str, Any] = {"lm": lm, **kw_f}
    try:
        with np.errstate(invalid="ignore"):  # inf - inf in a landmark difference
            case["curled"] = curled_fingers(lm, **kw_c)
            case["fist"] = is_curl_fist(lm, **kw_f)
    except ValueError:
        case["error"] = True
    return case


def curl_cases() -> list[dict[str, Any]]:
    rng = rng_for("curl")
    out = []
    subsets = [{f for f in range(4) if m >> f & 1} for m in range(16)]
    for k, curled in enumerate(subsets):  # every subset, default aspect / min_curled, 3 then 2 columns
        out.append(curl_case(make_hand(curled).tolist()))
        out.append(curl_case(make_hand(curled)[:, :2].tolist(), min_curled=MIN_CURLED[k % len(MIN_CURLED)]))
    hand = make_hand({0, 1, 2})  # z is ignored
    hand[:, 2] = np.linspace(-1, 1, 21)
    out.append(curl_case(hand.tolist()))
    iso = make_hand()  # tests: aspect makes distances isotropic
    wx, wy = iso[0, :2]
    for f in range(4):
        iso[6 + 4 * f, :2] = (wx, wy - 0.20)
        iso[8 + 4 * f, :2] = (wx + 0.15, wy)
    for aspect in ASPECTS:
        out.append(curl_case(iso.tolist(), aspect))
    nan_hand = make_hand({0, 1, 2, 3})
    nan_hand[8] = NAN
    nan_hand[12] = NAN
    out.append(curl_case(nan_hand.tolist()))
    # exact ties (tip as far as the PIP: not curled) and infinities
    tie = np.zeros((21, 2))
    tie[[6, 10, 14, 18]] = [(0.25, 0.5), (0.5, 0.25), (-0.25, 0.5), (0.125, 0.0)]
    tie[[8, 12, 16, 20]] = [(-0.25, -0.5), (0.5, 0.25), (0.25, -0.5), (0.0, 0.125)]
    for aspect in (1.0, 9 / 16, 2.0):
        out.append(curl_case(tie.tolist(), aspect))
    inf_hand = make_hand(set())[:, :2]
    inf_hand[6] = (INF, 0.0)  # PIP at infinity: tip closer -> curled
    inf_hand[8] = (INF, 0.0)  # index tip too: inf < inf is False
    inf_hand[16] = (-INF, INF)
    inf_hand[18] = (0.5, -INF)
    out.append(curl_case(inf_hand.tolist()))
    wrist_nan = make_hand({0, 1, 2, 3})
    wrist_nan[0, 1] = NAN
    out.append(curl_case(wrist_nan.tolist()))
    wrist_inf = make_hand({0, 1, 2, 3})
    wrist_inf[0, 0] = INF
    out.append(curl_case(wrist_inf.tolist(), 1.0))
    for i in range(150):  # noisy synthetic hands and plain random landmarks
        if i % 3:
            curled = {f for f in range(4) if rng.random() < 0.5}
            lm = make_hand(curled, (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.3, 0.95))))
            lm[:, :2] += rng.normal(0, float(rng.choice([0.002, 0.01, 0.03])), (21, 2))
        else:
            lm = rng.uniform(0, 1, (21, 3))
        cols = int(rng.choice([2, 3, 3, 4]))
        lm = np.round(np.concatenate([lm, rng.uniform(-1, 1, (21, 1))], axis=1)[:, :cols], 5)
        if rng.random() < 0.15:
            idx = rng.integers(0, 21, int(rng.integers(1, 4)))
            lm[idx, int(rng.integers(0, 2))] = float(rng.choice([NAN, INF, -INF]))
        aspect = None if rng.random() < 0.3 else float(ASPECTS[int(rng.integers(len(ASPECTS)))])
        min_curled = None if rng.random() < 0.5 else MIN_CURLED[int(rng.integers(len(MIN_CURLED)))]
        out.append(curl_case(lm.tolist(), aspect, min_curled))
    # shapes: (20, 2), (22, 3), (21, 1), (21, 0), (0,), (21, 4) (fine), ragged
    for lm in (np.zeros((20, 2)).tolist(), np.zeros((22, 3)).tolist(), np.zeros((21, 1)).tolist(),
               [[] for _ in range(21)], [], np.ones((21, 4)).tolist(),
               [[0.0, 0.0, 0.0]] * 10 + [[0.0, 0.0]] * 11, [[0.0, 0.0]] * 20 + [[0.0, 0.0, 0.0]]):
        out.append(curl_case(lm))
    return out


# --------------------------------------------------------------------------------------------------- fist


class FistRecorder:
    """Runs FistTracker ops and records each with its result."""

    def __init__(self, params: tuple | None) -> None:
        self.params = params
        self.ft = FistTracker() if params is None else FistTracker(factory=lambda: make_debouncer(params))
        self.ops: list[list[Any]] = []

    def u(self, did: int, side: str, score: float, t: float) -> bool:
        ret = self.ft.update(did, side, score, t)
        self.ops.append(["u", did, side, score, t, ret])
        return ret

    def m(self, t: float, seen: list[tuple[int, str]]) -> None:
        self.ft.mark_unobserved(t, set(seen))
        self.ops.append(["m", t, [list(k) for k in seen]])

    def p(self, did: int) -> tuple[int, bool]:
        onsets, active = self.ft.person_state(did)
        self.ops.append(["p", did, onsets, active])
        return onsets, active

    def d(self, did: int) -> None:
        self.ft.drop(did)
        self.ops.append(["d", did])

    def k(self) -> None:
        known = list(self.ft._debouncers)  # dict order; equal as a set to ft.known()
        assert frozenset(known) == self.ft.known()
        self.ops.append(["k", [list(key) for key in known]])

    def case(self, name: str) -> dict[str, Any]:
        return {"name": name, "factory": None if self.params is None else list(self.params), "ops": self.ops}


def clench(r: FistRecorder, did: int, side: str, t0: float, step: float = 0.2) -> None:
    for i in range(3):
        r.u(did, side, 0.9, t0 + i * step)


def fist_unit_cases() -> list[dict[str, Any]]:
    """The scenarios of tests/perception/test_hands.py TestFistTracker, recorded op by op."""
    out = []
    r = FistRecorder(LEGACY_FIST)
    clench(r, 1, "L", 0.0)
    r.p(1), r.p(1)
    out.append(r.case("onset_reported_as_delta"))
    r = FistRecorder(LEGACY_FIST)
    clench(r, 1, "L", 0.0)
    clench(r, 1, "R", 0.0)
    r.p(1)
    for i in range(1, 8):
        r.u(1, "L", 0.0, 0.4 + i * 0.2)
    r.p(1)
    for i in range(1, 8):
        r.u(1, "R", 0.0, 0.4 + i * 0.2)
    r.p(1), r.k()
    out.append(r.case("both_hands_together_are_one_onset"))
    r = FistRecorder(LEGACY_FIST)
    clench(r, 1, "L", 0.0)
    for i in range(1, 8):
        r.u(1, "L", 0.0, 0.4 + i * 0.2)
    r.p(1)
    clench(r, 1, "R", 2.0)
    r.p(1)
    for i in range(1, 8):
        r.u(1, "R", 0.0, 2.4 + i * 0.2)
    clench(r, 1, "L", 5.0)
    r.p(1)
    out.append(r.case("refractory_is_per_person"))
    r = FistRecorder(LEGACY_FIST)
    clench(r, 1, "L", 0.0)
    clench(r, 2, "R", 0.0)
    for i in range(1, 8):
        t = 0.4 + i * 0.2
        r.u(2, "R", 0.9, t)
        r.m(t, [(2, "R")])
    r.p(1), r.p(2), r.k()
    out.append(r.case("mark_unobserved_ends_unseen_fists"))
    r = FistRecorder(LEGACY_FIST)
    clench(r, 3, "L", 0.0)
    clench(r, 4, "L", 0.0)
    r.d(3)
    r.p(3), r.k()
    clench(r, 3, "L", 0.8)
    r.d(99), r.k()
    out.append(r.case("drop_forgets_person"))
    r = FistRecorder(None)
    for t in (0.0, 0.2, 0.4, 0.6):
        r.u(1, "L", 0.75, t)
    r.p(1)
    out.append(r.case("default_contract_rule"))
    # refractory boundary with float timestamps: onsets 3 s apart on a 0.1 s grid (i * 0.1 and running sums)
    for params in (None, LEGACY_FIST):
        r = FistRecorder(params)
        t = 0.0
        for i in range(120):
            score = 0.9 if (i // 10) % 3 == 0 else 0.1
            r.u(5, "R" if i % 60 < 30 else "L", score, t)
            r.m(t, [(5, "R" if i % 60 < 30 else "L")])
            if i % 5 == 4:
                r.p(5)
            t += 0.1
        r.k()
        out.append(r.case(f"float_grid_{'default' if params is None else 'legacy'}"))
    # odd timestamps: repeated, backwards, NaN
    r = FistRecorder(LEGACY_FIST)
    for i, t in enumerate([0.0, 0.0, 0.0, 0.2, 0.1, NAN, 0.5, 0.5, 4.0, NAN, NAN, NAN, 9.0, 9.1, 9.2, -1.0]):
        r.u(2, "L" if i % 4 else "R", 0.95 if i % 7 else NAN, t)
        r.p(2)
    r.m(NAN, [])
    r.p(2), r.k()
    out.append(r.case("odd_timestamps"))
    return out


def fist_random_case(rng: np.random.Generator, name: str, params: tuple | None, n_frames: int,
                     pool: list[int]) -> dict[str, Any]:
    r = FistRecorder(params)
    clenched: dict[tuple[int, str], bool] = {}
    present = {pid: bool(rng.random() < 0.6) for pid in pool}
    t = float(rng.uniform(0, 100))
    dt = float(rng.choice([1 / 15, 0.1, 0.2, 1 / 30]))
    for _ in range(n_frames):
        t = round(t + dt * float(rng.choice([1.0, 1.0, 1.0, rng.uniform(0.5, 1.5), 4.0])), 4)
        for pid in pool:
            if rng.random() < 0.02:
                present[pid] = not present[pid]
                if not present[pid] and rng.random() < 0.5:
                    r.d(pid)
        seen = []
        for pid in pool:
            if not present[pid]:
                continue
            for side in ("L", "R"):
                if rng.random() < 0.2:
                    continue  # hand not seen this frame
                key = (pid, side)
                if rng.random() < 0.06:
                    clenched[key] = not clenched.get(key, False)
                hi = clenched.get(key, False)
                score = round(rng.uniform(0.62, 1.0) if hi else rng.uniform(0.0, 0.55), 3)
                if rng.random() < 0.05:
                    score = float(rng.choice([NAN, 0.7, 0.4, 0.6, np.nextafter(0.7, 0.0)]))
                r.u(pid, side, float(score), t)
                seen.append(key)
        if rng.random() < 0.1 and seen:
            seen.append((int(rng.choice(pool)) + 1000, "L"))  # a hand the tracker never saw
        r.m(t, seen)
        for pid in pool:
            if present[pid] or rng.random() < 0.1:
                r.p(pid)
        if rng.random() < 0.05:
            r.k()
    r.k()
    return r.case(name)


def fist_cases() -> list[dict[str, Any]]:
    rng = rng_for("fist")
    out = fist_unit_cases()
    for i in range(14):
        if i % 3 == 0:
            params = None
        elif i % 3 == 1:
            params = LEGACY_FIST
        else:
            on = float(rng.uniform(0.5, 0.9))
            params = (on, float(on - rng.uniform(0, 0.3)), int(rng.integers(1, 4)), float(rng.choice([0.0, 0.5, 1.0])),
                      float(rng.choice([0.0, 1.0, 3.0])), float(rng.choice([0.0, 0.2, 0.5])))
        pool = [int(p) for p in rng.choice([1, 2, 3, 4, 5, 7, 12], int(rng.integers(1, 5)), replace=False)]
        out.append(fist_random_case(rng, f"random_{i}", params, 170, pool))
    return out


# --------------------------------------------------------------------------------------------------- ids


def box(cx: float, cy: float, w: float = 0.1, h: float = 0.2) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def rbox(cx: float, cy: float) -> list[float]:
    """box() rounded to 5 decimals (short JSON; the centres are still computed by the mapper)."""
    return [round(v, 5) for v in box(cx, cy)]


class IdsRecorder:
    def __init__(self, params: dict[str, float]) -> None:
        self.params = params
        self.m = DisplayIdMapper(**params)
        self.ops: list[list[Any]] = []

    def u(self, tracks: list[tuple[float, list[float]]], t: float) -> dict[int, int] | None:
        try:
            mapping = self.m.update([(tid, tuple(b)) for tid, b in tracks], t)
        except (ValueError, OverflowError):
            mapping = None
        items = None if mapping is None else [list(kv) for kv in mapping.items()]
        assert mapping is None or mapping == self.m.active()
        self.ops.append(["u", [[tid, b] for tid, b in tracks], t, items, sorted(self.m.reserved())])
        return mapping

    def p(self) -> list[int]:
        released = self.m.pop_released()
        self.ops.append(["p", released])
        return released

    def r(self) -> None:
        self.m.reset()
        self.ops.append(["r"])

    def case(self, name: str) -> dict[str, Any]:
        return {"name": name, "params": self.params, "ops": self.ops}


def ids_unit_cases() -> list[dict[str, Any]]:
    """tests/perception/test_ids.py scenarios plus hand-made edges, recorded op by op."""
    out = []

    def rec(name: str, steps: list, params: dict[str, float] | None = None) -> None:
        r = IdsRecorder(params or {})
        for step in steps:
            if step == "p":
                r.p()
            elif step == "r":
                r.r()
            else:
                r.u(*step)
        out.append(r.case(name))

    rec("first_appearance", [([(11, box(0.7, 0.5)), (10, box(0.3, 0.5))], 0.0),
                             ([(10, box(0.3, 0.5)), (11, box(0.7, 0.5)), (14, box(0.5, 0.5))], 0.1)])
    rec("stable_when_moving", [([(1, box(0.1, 0.5)), (2, box(0.9, 0.5))], 0.0),
                               ([(1, box(0.9, 0.5)), (2, box(0.1, 0.5))], 0.1)])
    rec("smallest_free_after_expiry", [
        ([(1, box(0.1, 0.5)), (2, box(0.5, 0.5)), (3, box(0.9, 0.5))], 0.0),
        ([(1, box(0.1, 0.5)), (3, box(0.9, 0.5))], 1.0), ([(1, box(0.1, 0.5)), (3, box(0.9, 0.5))], 4.0),
        ([(1, box(0.1, 0.5)), (3, box(0.9, 0.5)), (9, box(0.5, 0.1))], 4.1), "p"])
    rec("reserved_skipped", [
        ([(1, box(0.1, 0.5)), (2, box(0.5, 0.5)), (3, box(0.9, 0.5))], 0.0),
        ([(1, box(0.1, 0.5)), (3, box(0.9, 0.5))], 1.0),
        ([(1, box(0.1, 0.5)), (3, box(0.9, 0.5)), (9, box(0.2, 0.95, h=0.05))], 1.5)])
    rec("duplicates", [([(5, box(0.2, 0.5)), (5, box(0.8, 0.5))], 0.0), ([(5, box(0.8, 0.5))], 1.0),
                       ([(6, box(NAN, 0.5)), (6, box(0.2, 0.5)), (5, box(0.8, 0.5)), (5, box(0.1, 0.1))], 2.0),
                       ([(5, box(0.8, 0.5))], 2.5), ([(7, box(0.2, 0.5)), (6, box(0.2, 0.5))], 3.0)])
    rec("reuse_inside_window", [([(10, box(0.3, 0.5)), (11, box(0.8, 0.5))], 0.0), ([(11, box(0.8, 0.5))], 1.0),
                                ([(11, box(0.8, 0.5)), (12, box(0.5, 0.5))], 3.5)])
    rec("no_reuse_outside_radius", [([(10, box(0.3, 0.5)), (11, box(0.8, 0.5))], 0.0), ([(11, box(0.8, 0.5))], 1.0),
                                    ([(11, box(0.8, 0.5)), (12, box(0.56, 0.5))], 1.5)])
    for t_back in (3.75, 3.999999998, 3.999999999, 3.9999999995, 4.0, 4.000000001):
        rec(f"window_boundary_{t_back}", [
            ([(10, box(0.2, 0.5)), (11, box(0.5, 0.5))], 0.0), ([(11, box(0.5, 0.5))], 1.0),
            ([(11, box(0.5, 0.5)), (12, box(0.9, 0.5)), (13, box(0.2, 0.5))], t_back), "p"])
    for aspect in (9 / 16, 1.0):
        for dx, dy in ((0.0, 0.4), (0.4, 0.0), (0.24, 0.0), (0.25, 0.0), (0.26, 0.0), (0.15, -0.25), (0.15, 0.2)):
            rec(f"distance_{dx}_{dy}_{aspect}", [
                ([(1, box(0.5, 0.3)), (2, box(0.05, 0.9))], 0.0), ([(2, box(0.05, 0.9))], 0.5),
                ([(2, box(0.05, 0.9)), (3, box(0.5 + dx, 0.3 + dy))], 1.0)], {"aspect": aspect})
    rec("swap_by_position", [([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0), ([], 1.0),
                             ([(20, box(0.75, 0.5)), (21, box(0.25, 0.5))], 1.5)])
    rec("global_nearest_first", [([(1, box(0.40, 0.5)), (2, box(0.60, 0.5))], 0.0), ([], 1.0),
                                 ([(30, box(0.52, 0.5)), (31, box(0.62, 0.5))], 1.2)])
    rec("revived_far", [([(7, box(0.1, 0.5)), (8, box(0.5, 0.5))], 0.0), ([(8, box(0.5, 0.5))], 1.0),
                        ([(8, box(0.5, 0.5)), (7, box(0.95, 0.5))], 2.0)])
    rec("revived_beats_nearby", [([(7, box(0.25, 0.5)), (8, box(0.75, 0.5))], 0.0), ([(8, box(0.75, 0.5))], 1.0),
                                 ([(9, box(0.25, 0.5)), (8, box(0.75, 0.5)), (7, box(0.5, 0.5))], 2.0)])
    rec("not_double_assigned", [([(1, box(0.5, 0.5))], 0.0), ([], 1.0),
                                ([(2, box(0.52, 0.5)), (3, box(0.48, 0.5))], 1.5)])
    # exact ties on dyadic positions: one new track between two reservations, two new tracks around one
    rec("tie_two_reservations", [([(1, box(0.25, 0.5, 0.125, 0.25)), (2, box(0.75, 0.5, 0.125, 0.25))], 0.0),
                                 ([], 0.5), ([(9, box(0.5, 0.5, 0.125, 0.25))], 1.0)])
    rec("tie_two_reservations_reversed", [([(2, box(0.75, 0.5, 0.125, 0.25)), (1, box(0.25, 0.5, 0.125, 0.25))], 0.0),
                                          ([], 0.5), ([(9, box(0.5, 0.5, 0.125, 0.25))], 1.0)],
             {"reuse_dist": 0.25})
    rec("tie_two_new_tracks", [([(1, box(0.5, 0.5, 0.125, 0.25))], 0.0), ([], 0.5),
                               ([(12, box(0.625, 0.5, 0.125, 0.25)), (11, box(0.375, 0.5, 0.125, 0.25))], 1.0)])
    rec("pop_released_once", [([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0), ([(2, box(0.8, 0.5))], 1.0),
                              ([(2, box(0.8, 0.5))], 3.9), "p", ([(2, box(0.8, 0.5))], 4.0), "p", "p"])
    rec("reused_never_released", [([(1, box(0.2, 0.5))], 0.0), ([], 1.0), ([(2, box(0.22, 0.5))], 2.0),
                                  ([(2, box(0.22, 0.5))], 10.0), "p"])
    rec("reset_forgets", [([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0), ([(2, box(0.8, 0.5))], 1.0), "r",
                          ([(5, box(0.8, 0.5))], 2.0), "p", "p"])
    rec("reset_releases_everything", [
        ([(1, box(0.2, 0.5)), (2, box(0.8, 0.5)), (3, box(0.5, 0.5))], 0.0), ([(2, box(0.8, 0.5))], 1.0),
        ([(2, box(0.8, 0.5)), (4, box(0.5, 0.95))], 4.5), ([(4, box(0.5, 0.95))], 5.0), "r", "r", "p", "p",
        ([(4, box(0.5, 0.95))], 5.1), ([], 5.2), "r", ([(9, box(0.1, 0.1))], 5.3), "p"])
    rec("float_rounding_expiry", [([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0), ([(2, box(0.8, 0.5))], 1.1),
                                  ([(2, box(0.8, 0.5)), (3, box(0.5, 0.95))], 4.1), "p"])
    rec("int_tracker_ids", [([(4.0, box(0.2, 0.5)), (2.7, box(0.8, 0.5)), (-0.5, box(0.5, 0.5))], 0.0),
                            ([(4, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.5),
                            ([(NAN, box(0.2, 0.5))], 0.7), ([(4, box(0.2, 0.5)), (INF, box(0.8, 0.5))], 0.8),
                            ([(0.9, box(0.5, 0.5)), (2**40, box(0.9, 0.9)), (-7, box(0.1, 0.1))], 1.0)])
    rec("truncation_duplicates", [
        ([(2.7, box(0.3, 0.3)), (2, box(0.9, 0.9)), (-0.0, box(0.5, 0.5)), (0.4, box(0.1, 0.1))], 0.0),
        ([(2.0, box(0.3, 0.3))], 0.5), ([(3, box(0.3, 0.3)), (2.9, box(0.9, 0.9))], 1.0)])
    rec("zero_window", [([(1, box(0.2, 0.5))], 0.0), ([(2, box(0.21, 0.5))], 0.0), ([(3, box(0.22, 0.5))], 0.0),
                        ([], 0.0), ([(1, box(0.2, 0.5))], 0.0), "p"], {"reuse_window_s": 0.0})
    rec("infinite_window_and_radius", [([(1, box(0.2, 0.5)), (2, box(0.8, 0.5))], 0.0), ([], 1.0),
                                       ([(3, box(INF, 0.5)), (4, box(NAN, 0.5)), (5, box(0.9, 0.9))], 1e9),
                                       ([], INF), ([(6, box(0.5, 0.5))], INF), "p"],
        {"reuse_window_s": INF, "reuse_dist": INF})
    rec("nan_params", [([(1, box(0.2, 0.5))], 0.0), ([], 1.0), ([(2, box(0.2, 0.5))], 50.0), ([], 51.0), "p"],
        {"reuse_window_s": NAN, "reuse_dist": NAN, "aspect": NAN})
    rec("nan_and_inf_times", [([(1, box(0.2, 0.5)), (2, box(0.6, 0.5))], NAN), ([(2, box(0.6, 0.5))], 1.0),
                              ([(2, box(0.6, 0.5))], 100.0), ([], NAN), ([(3, box(0.6, 0.5))], 200.0), "p",
                              ([(3, box(0.6, 0.5))], -INF), ([], INF), ([], INF), "p"])
    return out


def ids_random_case(rng: np.random.Generator, name: str, params: dict[str, float], n_frames: int) -> dict[str, Any]:
    r = IdsRecorder(params)
    next_tid = int(rng.integers(1, 50))
    people: list[dict[str, Any]] = []
    t = float(rng.uniform(0, 10))
    dt = float(rng.choice([1 / 15, 0.1, 0.25]))
    grid = bool(rng.random() < 0.3)  # dyadic positions: exact distance ties

    def pos() -> tuple[float, float]:
        if grid:
            return float(rng.integers(0, 9)) / 8, float(rng.integers(0, 5)) / 4
        return float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.2, 0.8))

    for _ in range(n_frames):
        if rng.random() < 0.03:
            t += float(rng.choice([2.0, 2.9, 3.0, 3.1, 4.0, 1 - 1e-9]))  # gaps around the window
        else:
            t += dt * float(rng.choice([1.0, 1.0, rng.uniform(0.5, 1.5)]))
        if len(people) < 6 and rng.random() < 0.06:
            people.append({"tid": next_tid, "pos": pos(), "hidden": 0, "size": (0.1, 0.2)})
            next_tid += int(rng.integers(1, 3))
        for p in people:
            if p["hidden"] > 0:
                p["hidden"] -= 1
                if p["hidden"] == 0:
                    roll = rng.random()
                    if roll < 0.25:
                        pass  # the tracker revived its own id
                    else:
                        p["tid"] = next_tid
                        next_tid += 1
                    if rng.random() < 0.35:
                        p["pos"] = pos()  # came back far away
            elif rng.random() < 0.05:
                p["hidden"] = int(rng.choice([1, 3, 10, 30, 45, 60]))
            if not grid:
                x, y = p["pos"]
                nx = float(np.clip(x + rng.normal(0, 0.01), 0, 1))
                p["pos"] = (nx, float(np.clip(y + rng.normal(0, 0.01), 0, 1)))
        if people and rng.random() < 0.02:
            people.pop(int(rng.integers(len(people))))
        tracks = [(p["tid"], rbox(*p["pos"])) for p in people if p["hidden"] == 0]
        if tracks and rng.random() < 0.05:
            tid, _ = tracks[int(rng.integers(len(tracks)))]
            tracks.append((tid, rbox(*pos())))  # duplicate tracker id in one frame
        if tracks and rng.random() < 0.02:
            i = int(rng.integers(len(tracks)))
            tracks[i] = (tracks[i][0], box(NAN, 0.5))
        order = rng.permutation(len(tracks))
        r.u([tracks[i] for i in order], t)
        if rng.random() < 0.2:
            r.p()
        if rng.random() < 0.004:
            r.r()
    r.p()
    return r.case(name)


def ids_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = rng_for("ids")
    out = ids_unit_cases()
    param_sets: list[dict[str, float]] = [{}, {}, {}, {"reuse_window_s": 1.0}, {"reuse_dist": 0.1},
                                          {"reuse_dist": 0.5, "aspect": 1.0}, {"aspect": 16 / 9},
                                          {"reuse_window_s": 0.0}, {"reuse_dist": 0.0}, {"reuse_window_s": 5.5}]
    for i, params in enumerate(param_sets * 2):
        out.append(ids_random_case(rng, f"random_{i}", params, 150))
    ctor = []
    for params in ({}, {"reuse_window_s": -1}, {"reuse_dist": -0.1}, {"aspect": 0}, {"aspect": -1.0},
                   {"reuse_window_s": -INF}, {"reuse_window_s": 0.0, "reuse_dist": 0.0}, {"reuse_window_s": NAN},
                   {"reuse_dist": NAN}, {"aspect": NAN}, {"aspect": INF}, {"aspect": -0.0}, {"reuse_dist": -0.0}):
        try:
            DisplayIdMapper(**params)
            ctor.append({"params": params, "error": False})
        except ValueError:
            ctor.append({"params": params, "error": True})
    return out, ctor


# --------------------------------------------------------------------------------------------------- debounce


def debounce_case(name: str, make: Callable[[], HysteresisDebouncer], preset: str | None,
                  samples: list[Any]) -> dict[str, Any]:
    deb = make()
    ops = []
    for s in samples:
        if s == "r":
            deb.reset()
            ops.append(["r", deb.active, deb.onsets])
        else:
            score, t = s
            ret = deb.update(score, t)
            ops.append(["s", score, t, ret, deb.active, deb.onsets])
    return {"name": name, "preset": preset, "params": debouncer_params(deb), "ops": ops}


def debounce_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = rng_for("debounce")
    legacy = lambda: make_debouncer(LEGACY_FIST)  # noqa: E731
    out = [
        debounce_case("fist_hold", fist_debouncer, "fist", [(0.7, 0.0), (0.75, 0.2), (0.8, 0.4), (0.8, 0.5)]),
        debounce_case("fist_0_65", fist_debouncer, "fist", [(0.65, i * 0.2) for i in range(20)]),
        debounce_case("fist_dip", fist_debouncer, "fist",
                      [(0.9, 0.0), (0.9, 0.2), (0.5, 0.4), (0.9, 0.6), (0.9, 0.8), (0.9, 1.0), (0.9, 1.1)]),
        debounce_case("fist_nan", fist_debouncer, "fist", [(0.9, 0.0), (0.9, 0.2), (NAN, 0.3), (0.9, 0.4), (0.9, 0.6)]),
        debounce_case("legacy_exit_rounding", legacy, None,
                      [(0.8, 0.0), (0.8, 0.25), (0.8, 0.5), (0.1, 1.3), (0.1, 2.3)]),
        debounce_case("legacy_refractory_rounding", legacy, None,
                      [(0.9, 0.9), (0.9, 1.0), (0.9, 1.1), (0.0, 1.5), (0.0, 2.5), (0.9, 3.9), (0.9, 4.0), (0.9, 4.1)]),
        debounce_case("legacy_reset", legacy, None, [(0.8, 0.0), (0.8, 0.25), (0.8, 0.5), "r", (0.9, 1.0), (0.9, 1.25),
                                                     (0.9, 1.5)]),
        debounce_case("defaults", lambda: HysteresisDebouncer(0.5, 0.5), None, [(0.5, 0.0), (0.49, 0.1), (0.9, 0.2)]),
        debounce_case("legacy_laughter", lambda: make_debouncer(LEGACY_LAUGH), None,
                      [(0.5, 0.0), (0.5, 0.5), (0.0, 1.0), (0.5, 1.5), (0.5, 2.0), (0.0, 2.5), (0.5, 3.0), (0.5, 3.5)]),
    ]
    probe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0,
             0.0, 0.043, 0.082, 0.0, 0.082, 0.004, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.004, 0.0, 0.043,
             0.008, 0.0, 0.0, 0.0]
    out.append(debounce_case("laughter_probe", laughter_debouncer, "laughter",
                             [(x, float(i)) for i, x in enumerate(probe)]))
    laugh = [0.08, 0.00, 0.08]
    for dt in (0.98, 0.995, 0.999, 1.0, 1.001, 1.02):
        seq = laugh + [0.0] * 3 + laugh + [0.0] * 4
        out.append(debounce_case(f"laughter_jitter_{dt}", laughter_debouncer, "laughter",
                                 [(x, 50.0 + i * dt) for i, x in enumerate(seq)]))
        seq = [0.08, 0.00, 0.00, 0.00, 0.00, 0.08]
        out.append(debounce_case(f"laughter_dip_{dt}", laughter_debouncer, "laughter",
                                 [(x, 7.0 + i * dt) for i, x in enumerate(seq)]))
    # exact boundary timestamps around each hold of the fist preset (0.5 s entry, 1 s exit, 3 s refractory)
    for eps in (-2e-9, -1e-9, -5e-10, 0.0, 1e-9):
        out.append(debounce_case(f"fist_boundaries_{eps}", fist_debouncer, "fist", [
            (0.7, 10.0), (0.7, 10.5 + eps), (0.39, 10.6), (0.39, 11.6 + eps), (0.39, 11.6), (0.9, 12.0),
            (0.9, 12.5), (0.1, 12.6), (0.1, 13.6), (0.9, 13.0 + eps), (0.9, 13.5 + eps), (0.1, 14.0), (0.1, 15.0),
            (0.9, 13.0 + 3.0 + 0.5 + eps), (0.9, 16.5 + eps)]))
    # random parameters and samples
    for i in range(90):
        if i % 10 == 0:
            make, preset = fist_debouncer, "fist"
        elif i % 10 == 1:
            make, preset = laughter_debouncer, "laughter"
        else:
            on = float(rng.choice([rng.uniform(0, 1), 0.5, 0.7]))
            off = float(rng.choice([on, on - rng.uniform(0, on), 0.0]))
            params = (on, off, int(rng.integers(1, 5)),
                      float(rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, rng.uniform(0, 2)])),
                      float(rng.choice([0.0, 1.0, 3.0, 5.5, rng.uniform(0, 6)])),
                      float(rng.choice([0.0, 0.0, 0.5, 1.0, rng.uniform(0, 1)])))
            make, preset = (lambda p=params: make_debouncer(p)), None
        deb = make()
        dt = float(rng.choice([0.1, 0.2, 0.25, 1 / 15, 0.48, 1.0]))
        t = float(rng.choice([0.0, rng.uniform(0, 1000)]))
        samples: list[Any] = []
        level = rng.random() < 0.5
        for k in range(int(rng.integers(20, 120))):
            if rng.random() < 0.12:
                level = not level
            u = rng.random()
            if u < 0.08:
                score = NAN
            elif u < 0.2:
                score = float(rng.choice([deb.on, deb.off, np.nextafter(deb.on, -1.0), np.nextafter(deb.off, -1.0)]))
            else:
                score = round(float(rng.uniform(deb.on, 1.0) if level else rng.uniform(0.0, max(deb.off, 1e-3))), 4)
            samples.append((score, t))
            if rng.random() < 0.01:
                samples.append("r")
            v = rng.random()
            if v < 0.03:
                t += 0.0
            elif v < 0.05:
                t -= dt
            elif v < 0.06:
                t = NAN if rng.random() < 0.5 else t
            else:
                t = (t if not math.isnan(t) else float(k)) + dt * float(rng.choice([1.0, 1.0, rng.uniform(0.5, 2.0)]))
                if i % 2:
                    t = round(t, 4)  # the even cases keep running sums (float rounding at the hold boundaries)
        out.append(debounce_case(f"random_{i}", make, preset, samples))
    ctor = []
    for args in ((0.3, 0.5), (0.5, 0.5), (0.5, 0.3), (NAN, 0.3), (0.3, NAN), (INF, -INF), (-INF, INF)):
        try:
            HysteresisDebouncer(*args)
            ctor.append({"on": args[0], "off": args[1], "error": False})
        except ValueError:
            ctor.append({"on": args[0], "off": args[1], "error": True})
    return out, ctor


# --------------------------------------------------------------------------------------------------- export


def write(out_dir: Path, part: str, payload: dict[str, Any]) -> None:
    text = json.dumps(enc(payload), allow_nan=False, separators=(",", ":"))
    (out_dir / f"hands-{part}.json").write_text(text + "\n", encoding="utf-8")


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir, "hypot", {"cases": hypot_cases()})
    write(out_dir, "assign", {"cases": assign_cases()})
    write(out_dir, "curl", {"cases": curl_cases()})
    write(out_dir, "fist", {"cases": fist_cases()})
    ids, ids_ctor = ids_cases()
    write(out_dir, "ids", {"cases": ids, "constructor_cases": ids_ctor})
    deb, deb_ctor = debounce_cases()
    write(out_dir, "debounce", {"cases": deb, "constructor_cases": deb_ctor})


if __name__ == "__main__":
    export(DEFAULT_OUT)
