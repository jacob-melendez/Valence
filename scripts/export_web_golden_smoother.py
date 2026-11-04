"""Golden values for the TypeScript smoother and served counterfactual -> frontend/src/test/fixtures/engine/smoother.json.gz.

Run from the repo root:  .venv/bin/python -B <this file>

Schema (version 1):
  {version,
   smoother: [{name, states, kwargs: {PersonSmoother keyword args},
               steps: [{probabilities: {state: p | null}, raw_override: true|false|null, label, window_ticks}],
               expected: [{label, mask_override, warming_up, ema: [K] (null = NaN), candidate, candidate_n,
                           override_hist: [bool], override_on}]}],
   smoother_errors: [{name, kwargs (NaN as the string "NaN"), error: Python exception type name}],
   counterfactual: [{run, tick, person_id, feature, value, out: ServedPredictor.counterfactual(...)}],
   served: [{state_set, lo, hi, ood_lo, ood_hi: [123], ood_used, face_cols: [bool], face_now_index,
             clip_cases: [{name, x: [123], xc: [123], ood: [names]}]}],     # ServedPredictor tables and _clip
   ood_band: [{name, lo, hi, out: [lo, hi]}],                                # ood_band on odd names
   synthetic: {model_id, m2_text, cases: [{name, tod_h, x: [123], out: PredictionOutput dict}]}
             # a tiny M2 (two features) served with the real core M0 / meta / clip bounds: see synthetic_model()
   (+-inf is written as the strings "Infinity" / "-Infinity")
   generator: this file's source}

Counterfactual windows come from frontend/src/test/fixtures/engine/runtime.json.gz: its tick records replayed
through RollingWindow(window_ticks) with Runtime._evict's rule, exactly as the TS test replays them.
Deterministic (seeded RNG, gzip mtime=0): reruns are byte-identical.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from valence.contracts import TickRecord  # noqa: E402
from valence.features.window import ROOM  # noqa: E402
from valence.features.window import RollingWindow  # noqa: E402
from valence.models.predictor import PersonSmoother, PredictionOutput  # noqa: E402
from valence.models.served import load_served  # noqa: E402

OUT = ROOT / "frontend" / "src" / "test" / "fixtures" / "engine" / "smoother.json.gz"
CORE = ("Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy")
ALL = CORE + ("Overwhelmed", "Tense")


def nan_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: nan_none(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [nan_none(x) for x in v]
    return v


def out_of(step) -> PredictionOutput:
    return PredictionOutput(label=step["label"], confidence=0.0, probabilities=step["probabilities"],
                            face_only_label=None, face_only_confidence=None, layers=[],
                            raw_override=step["raw_override"])


def run_case(name, states, kwargs, steps):
    sm = PersonSmoother(tuple(states), **kwargs)
    expected = []
    for st in steps:
        r = sm.update(out_of(st), st["window_ticks"])
        expected.append({
            "label": r.label, "mask_override": r.mask_override, "warming_up": r.warming_up,
            "ema": [float(v) for v in sm.ema], "candidate": sm.candidate, "candidate_n": sm.candidate_n,
            "override_hist": list(sm.override_hist), "override_on": sm.override_on,
        })
    return {"name": name, "states": list(states), "kwargs": kwargs, "steps": steps, "expected": expected}


def step(probs, raw_override, label, window_ticks):
    return {"probabilities": probs, "raw_override": raw_override, "label": label, "window_ticks": window_ticks}


def dist(states, top, w=0.7):
    rest = (1 - w) / (len(states) - 1)
    return {s: (w if s == top else rest) for s in states}


def crafted():
    cases = []
    S = CORE
    # constant distribution: label fixed, warm-up ends at int(60 / 5) = 12 ticks
    cases.append(run_case("steady_warmup", S, {}, [step(dist(S, "Focused"), None, "Focused", i + 1) for i in range(16)]))
    # hysteresis: a one-tick blip never switches; two in a row do (hold=2)
    seq = ["Focused"] * 4 + ["Exhausted"] + ["Focused"] * 2 + ["Exhausted"] * 4 + ["Collaborative_Happy", "Exhausted"] * 3
    cases.append(run_case("hysteresis", S, {}, [step(dist(S, t, 0.95), None, t, 30 + i) for i, t in enumerate(seq)]))
    # override: on at 3 hits of the last 6, off at <= 1; None ticks hold; only counts when out.label == shown label
    ro = [True, True, None, True, False, None, False, True, False, False, False, False, True, True, True, None, False]
    cases.append(run_case("override_hysteresis", S, {},
                          [step(dist(S, "Frustrated_Masked", 0.9), r, "Frustrated_Masked", 40 + i) for i, r in enumerate(ro)]))
    # override block: the condition holds from the start but the window is shorter than int(120 / 5) = 24 ticks
    cases.append(run_case("override_block", S, {},
                          [step(dist(S, "Frustrated_Masked", 0.9), True, "Frustrated_Masked", 18 + i) for i in range(10)]))
    # override raised for the raw label while the smoothed label is still something else: does not count
    seq = [("Focused", True)] * 3 + [("Frustrated_Masked", True)] * 6
    cases.append(run_case("override_other_label", S, {},
                          [step(dist(S, t, 0.9), r, t, 50 + i) for i, (t, r) in enumerate(seq)]))
    # a label switch clears the override history and state
    seq = [("Frustrated_Masked", True)] * 5 + [("Exhausted", True)] * 2 + [("Exhausted", False)] * 3
    cases.append(run_case("override_cleared_on_switch", S, {},
                          [step(dist(S, t, 0.9), r, "Frustrated_Masked", 60 + i) for i, (t, r) in enumerate(seq)]))
    # exact EMA ties: the first state wins (np.argmax)
    tie = {"Focused": 0.25, "Frustrated_Masked": 0.25, "Exhausted": 0.25, "Collaborative_Happy": 0.25}
    tie2 = {"Focused": 0.125, "Frustrated_Masked": 0.375, "Exhausted": 0.375, "Collaborative_Happy": 0.125}
    cases.append(run_case("exact_ties", S, {}, [step(tie, None, "Focused", 1), step(tie2, None, "Exhausted", 2),
                                                 step(tie2, None, "Exhausted", 3), step(tie, None, "Focused", 4),
                                                 step(tie2, False, "Frustrated_Masked", 5)]))
    # probabilities missing states (get(s, 0.0)) and carrying extra keys
    cases.append(run_case("missing_and_extra_states", S, {}, [
        step({"Exhausted": 0.6, "Bored": 0.9}, None, "Exhausted", 1),
        step({"Focused": 0.4}, None, "Focused", 2),
        step({"Focused": 0.4, "Tense": 0.99}, False, "Focused", 3),
        step({}, None, "Focused", 4),
        step({"Collaborative_Happy": 1.0}, True, "Collaborative_Happy", 5),
        step({"Collaborative_Happy": 1.0}, True, "Collaborative_Happy", 6),
    ]))
    # a NaN probability poisons the EMA; np.argmax then picks the first NaN
    cases.append(run_case("nan_probability", S, {}, [
        step(dist(S, "Focused"), None, "Focused", 1),
        step({"Focused": 0.1, "Frustrated_Masked": 0.2, "Exhausted": math.nan, "Collaborative_Happy": 0.7}, None, "Collaborative_Happy", 2),
        step(dist(S, "Focused"), None, "Focused", 3),
        step(dist(S, "Focused"), True, "Exhausted", 4),
    ]))
    # custom parameters
    cases.append(run_case("custom_alpha_hold", S, {"alpha": 0.3, "hold": 3, "override_on": 2, "override_off": 0,
                                                    "override_window": 4, "tick_s": 2.5, "warmup_s": 7.0,
                                                    "override_block_s": 13.0},
                          [step(dist(S, t, 0.8), r, t, i) for i, (t, r) in enumerate(
                              [("Focused", None)] * 2 + [("Exhausted", True)] * 5 + [("Exhausted", False)] * 2
                              + [("Exhausted", True)] * 2 + [("Focused", False)] * 4)]))
    cases.append(run_case("hold_zero", S, {"hold": 0},
                          [step(dist(S, t), None, t, 1 + i) for i, t in enumerate(["Focused", "Exhausted", "Focused", "Tense"])]))
    cases.append(run_case("hold_one", S, {"hold": 1},
                          [step(dist(S, t), None, t, 1 + i) for i, t in enumerate(["Focused", "Exhausted", "Exhausted", "Focused"])]))
    cases.append(run_case("override_window_zero", S, {"override_window": 0, "override_on": 0},
                          [step(dist(S, "Frustrated_Masked", 0.9), r, "Frustrated_Masked", 30) for r in (True, None, False, True)]))
    cases.append(run_case("int_truncation", S, {"tick_s": 3.0, "warmup_s": 10.0, "override_block_s": 8.9999},
                          [step(dist(S, "Frustrated_Masked", 0.9), True, "Frustrated_Masked", i) for i in range(1, 6)]))
    return cases


def randomized(rng: random.Random):
    cases = []
    for ci in range(10):
        states = ALL if ci % 3 == 2 else CORE
        kwargs = {}
        if ci >= 6:
            kwargs = {"alpha": rng.choice([0.2, 0.5, 0.75, 1.0]), "hold": rng.choice([1, 2, 3]),
                      "override_on": rng.choice([2, 3, 4]), "override_off": rng.choice([0, 1, 2]),
                      "override_window": rng.choice([3, 6, 8]), "warmup_s": rng.choice([30.0, 60.0, 61.0]),
                      "override_block_s": rng.choice([60.0, 120.0, 119.0])}
        top = rng.choice(states)
        steps = []
        wt = 0
        for i in range(160):
            if rng.random() < 0.12:
                top = rng.choice(states)
            raw = [rng.random() ** 2 for _ in states]
            raw[states.index(top)] += rng.random() * 3
            s = sum(raw)
            probs = {st: v / s for st, v in zip(states, raw)}
            label = max(probs, key=probs.get) if rng.random() < 0.85 else rng.choice(states)
            r = rng.random()
            raw_override = None if r < 0.25 else (r < 0.25 + 0.45 * (top == "Frustrated_Masked") + 0.15)
            wt = 1 if rng.random() < 0.02 else min(180, wt + 1)  # occasional restart of the window
            steps.append(step(probs, raw_override, label, wt))
        cases.append(run_case(f"random_{ci}_{len(states)}", states, kwargs, steps))
    return cases


def errors():
    out = []
    for name, kw in [("tick_zero", {"tick_s": 0.0}), ("warmup_nan", {"warmup_s": math.nan}),
                     ("block_inf", {"override_block_s": math.inf}), ("window_negative", {"override_window": -1}),
                     ("window_fraction", {"override_window": 2.5})]:
        try:
            PersonSmoother(CORE, **kw)
            err = None
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
        enc = {k: ("NaN" if isinstance(v, float) and math.isnan(v) else ("Infinity" if v == math.inf else v))
               for k, v in kw.items()}
        out.append({"name": name, "kwargs": enc, "error": err})
    return out


def evict(window: RollingWindow, wall: float, person_drop_min: float) -> None:
    for pid in window.people():  # valence/runtime/engine.py Runtime._evict
        rows = window.snapshot(pid)
        if rows and wall - rows[-1].ts > person_drop_min * 60:
            window.drop(pid)


def counterfactuals():
    rt = json.load(gzip.open(ROOT / "frontend/src/test/fixtures/engine/runtime.json.gz"))
    served = {s: load_served(ROOT / d, s) for s, d in (("core", "artifacts"), ("all", "artifacts/all"))}
    probes = [("face_now", 0.8), ("face_now", -0.6), ("hours_since_break", 6.0), ("db_room_mean_5m", 12.0),
              ("fist_cnt_5m", 5.0), ("slump_ewm_1m", 0.9), ("face_visible_frac_5m", 0.0), ("tod_sin", -1.0),
              ("db_now", 95.0)]
    out = []
    for run in rt["runs"]:
        sp = served[run["state_set"]]
        st = run["settings"]
        window = RollingWindow(st["window_ticks"])
        for i, tick in enumerate(run["ticks"]):
            recs = [TickRecord(**{k: (math.nan if v is None else v) for k, v in d.items()}) for d in tick["records"]]
            for r in recs:
                window.append(r)
            evict(window, tick["wall"], st["person_drop_min"])
            if i % 23 != 11:
                continue
            targets = [ROOM] if run["room_only"] else sorted({r.person_id for r in recs if r.person_id != ROOM})
            for j, pid in enumerate(targets):
                feature, value = probes[(i // 23 + j) % len(probes)]
                person, room = window.snapshot(pid), window.room()
                res = sp.counterfactual(person, room, feature, value)
                out.append({"run": run["name"], "tick": i, "person_id": pid, "feature": feature, "value": value,
                            "out": res})
    return out


def inf_str(v):
    """+-inf -> "Infinity" / "-Infinity" (JSON has no literal; nn() in engine-fixtures.ts decodes them)."""
    if isinstance(v, float) and math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if isinstance(v, (list, tuple)):
        return [inf_str(x) for x in v]
    return v


def served_tables():
    """Per state set: clip bounds, OOD bands, ood_used, face_cols, and _clip on crafted vectors."""
    from valence.features.compute import FEATURE_INDEX, FEATURE_NAMES
    from valence.models.served import ood_band

    out = []
    for s, d in (("core", "artifacts"), ("all", "artifacts/all")):
        sp = load_served(ROOT / d, s)
        n = len(FEATURE_NAMES)
        crafted = {
            "all_neg_zero": np.full(n, -0.0), "all_pos_zero": np.zeros(n), "at_lo": sp.lo.copy(),
            "at_hi": sp.hi.copy(), "all_pos_inf": np.full(n, math.inf), "all_neg_inf": np.full(n, -math.inf),
            "next_beyond_band": np.array([np.nextafter(h, math.inf) if math.isfinite(h) else 1.0 for h in sp.ood_hi]),
            "exact_band_edge": np.array([h if math.isfinite(h) else 1.0 for h in sp.ood_hi]),
            "below_band": np.array([np.nextafter(lo, -math.inf) if math.isfinite(lo) else -1.0 for lo in sp.ood_lo]),
        }
        clip_cases = []
        for name, x in crafted.items():
            xc, ood = sp._clip(x)
            clip_cases.append({"name": name, "x": inf_str([float(v) for v in x]), "xc": inf_str([float(v) for v in xc]),
                               "ood": ood})
        out.append({"state_set": s, "lo": inf_str([float(v) for v in sp.lo]), "hi": inf_str([float(v) for v in sp.hi]),
                    "ood_lo": inf_str([float(v) for v in sp.ood_lo]), "ood_hi": inf_str([float(v) for v in sp.ood_hi]),
                    "ood_used": [bool(v) for v in sp.ood_used], "face_cols": [bool(v) for v in sp.face_cols],
                    "face_now_index": FEATURE_INDEX["face_now"], "clip_cases": clip_cases})
    names = ["face_now\n", "face_now\n\n", "face_mean_", "face_mean_x", "face_mean_\u00e9", "face_visible_now",
             "withdraw_cnt_5m", "slump_std_", "slump_std_1m", "hih_max_15m", "a_cnt_b", "_cnt_5m", "x_cnt_",
             "foo_since_s", "since_s", "tod_sin", "fill_ratio", "unknown", "db_now", "face_std_5m", "Face_now",
             "slump_ewm_\u0663", "x\u00b2_cnt_y"]
    bands = []
    for nm in names:
        for lo, hi in ((0.0, 10.0), (-1.5, 2.25), (-math.inf, 3.0), (1.0, math.inf), (5e307, 1.7e308)):
            bands.append({"name": nm, "lo": inf_str(lo), "hi": inf_str(hi), "out": inf_str(list(ood_band(nm, lo, hi)))})
    return out, bands


def synthetic_model():
    """A tiny M2 that splits on two non-face features only, served next to the real M0 (core meta, clip bounds).

    Every other SHAP value is exactly 0.0, so served.py's `if c != 0.0` filter decides which drivers the layers
    show, and the face push is exactly 0.0 (raw_override needs face_push > 0, not >= 0). The real models never
    reach either branch visibly.
    """
    import shutil
    import tempfile

    import lightgbm as lgb

    from valence.models.served import ServedPredictor

    meta = json.loads((ROOT / "artifacts/model_meta.json").read_text())
    cols2 = meta["feature_sets"]["m2"]
    rng = np.random.default_rng(7)
    n = 400
    X = np.zeros((n, len(cols2)))
    j1, j2 = cols2.index("db_room_mean_5m"), cols2.index("hours_since_break")
    X[:, j1] = rng.normal(0, 3, n)
    X[:, j2] = rng.uniform(0, 6, n)
    y = np.where(X[:, j2] > 4, 2, 0)
    y[(X[:, j1] > 3) & (X[:, j2] < 2)] = 1
    y[(X[:, j1] < -3) & (X[:, j2] < 2)] = 3
    params = {"objective": "multiclass", "num_class": 4, "num_leaves": 4, "learning_rate": 0.4, "verbose": -1,
              "min_data_in_leaf": 5, "seed": 1, "deterministic": True, "num_threads": 1}
    bst = lgb.train(params, lgb.Dataset(X, y, feature_name=cols2, params={"verbose": -1}), num_boost_round=6)
    text = bst.model_to_string()
    models = json.load(gzip.open(ROOT / "frontend/src/test/fixtures/engine/models.json.gz"))
    core = next(s for s in models["sets"] if s["state_set"] == "core")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "m2.txt").write_text(text)
        shutil.copy(ROOT / "artifacts/m0.txt", tmp / "m0.txt")
        shutil.copy(ROOT / "artifacts/clip_bounds.json", tmp / "clip_bounds.json")
        sp = ServedPredictor(tmp, meta)
        from valence.features.compute import FEATURE_INDEX
        cases = []
        bases = [c for c in core["cases"] if "/" not in c["name"].split("/", 2)[-1]][:10]
        bases += [c for c in core["cases"] if c["name"].endswith(("no_face", "face_extremes", "all_nan", "huge_all",
                                                                    "ood_four_shrinks"))]
        for c in bases:
            x0 = np.array([math.nan if v is None else v for v in c["x"]], dtype=float)
            variants = [("", x0)]
            for tag, hsb, face in (("tired_happy_face", 5.5, 0.9), ("fresh_happy_face", 0.5, 0.95)):
                x = x0.copy()
                x[FEATURE_INDEX["hours_since_break"]] = hsb
                x[FEATURE_INDEX["face_now"]] = face
                x[FEATURE_INDEX["face_visible_now"]] = 1.0
                variants.append((tag, x))
            for tag, x in variants:
                out = sp.predict_vector(x, tod_h=c["tod_h"])
                d = {"label": out.label, "confidence": out.confidence, "probabilities": out.probabilities,
                     "face_only_label": out.face_only_label, "face_only_confidence": out.face_only_confidence,
                     "layers": [layer.model_dump() for layer in out.layers], "raw_override": out.raw_override,
                     "values": out.values, "ood_features": out.ood_features}
                cases.append({"name": c["name"] + (f"/{tag}" if tag else ""), "tod_h": c["tod_h"],
                              "x": [float(v) for v in x], "out": d})
        return {"model_id": sp.model_id, "m2_text": text, "cases": cases}


def main() -> None:
    rng = random.Random(20261024)
    doc = {"version": 1, "smoother": crafted() + randomized(rng), "smoother_errors": errors(),
           "counterfactual": counterfactuals(), "generator": Path(__file__).read_text()}
    doc["served"], doc["ood_band"] = served_tables()
    doc["synthetic"] = synthetic_model()
    text = json.dumps(nan_none(doc), allow_nan=False, separators=(",", ":"))
    with open(OUT, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
            gz.write(text.encode())
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.1f} KiB, {len(doc['smoother'])} smoother cases, "
          f"{len(doc['counterfactual'])} counterfactuals")


if __name__ == "__main__":
    main()
