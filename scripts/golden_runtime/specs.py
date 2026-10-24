"""Edge-case specs for edge_cases.py, built exactly as SPECS in frontend/src/test/engine-runtime.test.ts builds them.

Each spec is a JSON object: state_set, optional settings / room_only / label / wall_init / mono_init / draws, and
steps [{wall, mono, records, outs, pre}]. records are TickRecord dicts (null = NaN, "Infinity" = inf); outs maps
"tick:person_id" to the stub predictor's output; pre lists operator actions run before the step.

    .venv/bin/python -B scripts/golden_runtime/specs.py             # {name: spec} as one JSON object on stdout
    .venv/bin/python -B scripts/golden_runtime/specs.py --out DIR   # DIR/<name>.json, one spec per file
"""

from __future__ import annotations

import argparse
import json

from _common import out_dir

W0, M0 = 1792850400.0, 1000.0  # the test's WALL0 / MONO0
CORE = ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"]


def probs(*p):
    return dict(zip(CORE, p))


def out(label, p, conf=None, **kw):
    o = {"label": label, "confidence": conf if conf is not None else p[CORE.index(label)] if label in CORE else 0.5,
         "probabilities": probs(*p) if isinstance(p, (list, tuple)) else p,
         "face_only_label": kw.get("fol", "Focused"), "face_only_confidence": kw.get("foc", 0.61234),
         "raw_override": kw.get("ro", None), "ood_features": kw.get("ood", []),
         "layers": kw.get("layers", [
             {"name": "Ambient Physics", "contribution": 0.123456, "drivers": [
                 {"feature": "db_mean_15m", "display": "average noise of 72 dB over 15 minutes", "value": 72.04,
                  "unit": "dB", "shap": 0.4, "level": "high", "effect": "supports", "scope": "room"}]},
             {"name": "Facial Surface", "contribution": -0.0625, "drivers": []}])}
    return o


def room(i, **kw):
    r = {"tick": i, "ts": W0 + 5 * i, "t_mono": M0 + 5 * i, "person_id": 0, "display_id": "Room", "n_people": 1,
         "ambient_db": 55.0, "light_lux": 300.0, "tod_h": 14.0}
    r.update(kw)
    return r


def person(i, pid=1, **kw):
    r = {"tick": i, "ts": W0 + 5 * i, "t_mono": M0 + 5 * i, "person_id": pid, "display_id": f"Person #{pid}",
         "n_people": 2, "ambient_db": 55.0, "light_lux": 300.0, "face_valence": 0.2, "face_visible": 1.0,
         "tod_h": 14.0, "posture_slump": 0.1, "head_in_hands_frac": 0.0, "hours_since_break": 1.5}
    r.update(kw)
    return r


def step(i, records, outs=None, pre=None):
    s = {"wall": W0 + 5 * i, "mono": M0 + 5 * i, "records": records, "outs": outs or {}}
    if pre:
        s["pre"] = pre
    return s


def spec_sv() -> dict:
    """State vector, fact sheet and prediction edges."""
    P = [0.1, 0.2, 0.3, 0.4]
    steps = []
    steps.append(step(0, [room(0), person(0, tod_h=None, n_people=0, face_visible=None, ambient_db=-0.0,
                                           light_lux=5e-324, hours_since_break="Infinity", posture_slump=1e300,
                                           head_in_hands_frac=0.0005, face_valence=-0.0005)],
                      {"0:1": out("Collaborative_Happy", P)}))
    steps.append(step(1, [room(1, simulated_fields=["room_history", "a", "é"]),
                          person(1, tod_h=23.999999, n_people=1, simulated_fields=["ß", "Z", "a", "low_light", "é",
                                                                                  "\U0001F600", "￿", "b", "c", "d"])],
                      {"1:1": out("Collaborative_Happy", P)}))
    steps.append(step(2, [room(2), person(2, tod_h=-0.25, n_people=1)],
                      {"2:1": out("Exhausted", {"Focused": 0.5, "Frustrated_Masked": 0.25}, conf=0.33335)}))
    steps.append(step(3, [room(3), person(3, pid=2, tod_h=25.5), person(3, pid=-3, tod_h=24.0), person(3, tod_h=14.5),
                          person(3, tod_h=14.25, face_valence=None)],
                      {"3:1": out("Focused", P, conf=0.00005), "3:2": out("Focused", P), "3:-3": out("Exhausted", P)}))
    steps.append(step(4, [room(4), person(4, ts=W0 + 5 * 4 - 2000), person(4, pid=2)],
                      {"4:1": out("Focused", P), "4:2": out("Focused", P)}))
    steps.append(step(5, [room(5), person(5, pid=2)], {}))
    steps.append(step(6, [room(6, laughter_count=7, compliments_count=2**40),
                          person(6, pid=2, clenched_fists_count=2**41)],
                      {"6:2": out("Frustrated_Masked", P, ro=True)}))
    ties = [h + k / 120 for h in (0, 9, 13, 23) for k in (1, 3, 59, 61, 117, 119)]
    ties += [14 + 59.5 / 60, 5 - 1e-15, 1e300 % 24, 12 + 1 / 7200, 0.5 / 60, 11.999999999999, 7.5 / 60]
    for j, t in enumerate(ties):
        i = 7 + j
        steps.append(step(i, [room(i, tod_h=t), person(i, pid=2, tod_h=t, light_lux=0.0005 + j,
                                                       ambient_db=2.0005 + j / 1000, posture_slump=(j % 3) / 3,
                                                       face_visible=j % 2)],
                          {f"{i}:2": out(["Focused", "Exhausted"][j // 7 % 2], P, conf=0.12345 + j / 100000)}))
    return {"state_set": "core", "settings": {"window_ticks": 4}, "steps": steps}


def spec_room() -> dict:
    """Room only, overlays, empty and finished ticks."""
    P = [0.1, 0.2, 0.3, 0.4]
    steps = []
    draws = [3.0, -3.0, 0.5, -0.25, 1e-3, 2.0, -2.0, 0.0, 4.0, -4.0] * 4
    for i in range(14):
        pre = []
        if i == 2:
            pre = [["clock", W0 + 5 * i - 2.5], ["overlay", "low_light", True]]
        if i == 5:
            pre = [["clock", W0 + 5 * i - 2.5], ["overlay", "face_occlusion", True]]
        if i == 8:
            pre = [["clock", W0 + 5 * i - 2.5], ["overlay", "low_light", False]]
        if i == 10:
            pre = [["clock", W0 + 5 * i - 2.5], ["overlay", "low_light", True], ["room_only", False]]
        recs = [room(i, light_lux=None if i % 3 == 0 else 120.0, face_valence=0.9 if i % 2 else -0.95,
                     face_visible=1.0, simulated_fields=["low_light"] if i == 3 else []),
                person(i, face_valence=None if i % 4 == 0 else 0.97 - i / 10, light_lux=None if i % 2 else 80.0)]
        if i == 6:
            recs = []
        outs = {f"{i}:0": out(CORE[i % 4], P, ro=bool(i % 2)), f"{i}:1": out(CORE[(i + 1) % 4], P)}
        steps.append(step(i, recs, outs, pre))
    steps.append({"wall": W0 + 70, "mono": M0 + 70, "records": None})
    return {"state_set": "core", "room_only": True, "draws": draws, "steps": steps}


def spec_op() -> dict:
    """Operator actions: seed, reset, break, purge counters, audit throttle."""
    P = [0.1, 0.2, 0.3, 0.4]
    seed_rows = [room(t, ambient_db=60 + t, simulated_fields=[]) for t in range(30)] + [person(t) for t in range(30)]
    steps = []
    for i in range(30):
        pre = []
        if i == 0:
            pre = [["clock", W0 - 1.0], ["seed", seed_rows, "full_demo", 1.0, 2.5]]
        if i == 3:
            pre = [["break", None], ["break", 1], ["break", 1.5], ["purge", {"video_purged": 10}]]
        if i == 6:
            pre = [["reset", True], ["purge", {"video_purged": 7, "audio_purged": 4, "violations": 2}]]
        if i == 12:
            pre = [["reset", False]]
        if i == 20:
            pre = [["audit", "operator", {"action": "x", "v": "NaN"}]]
        label = "Focused" if i < 9 or i > 15 else "Exhausted"
        steps.append(step(i, [room(i), person(i, hours_since_break=None)], {f"{i}:1": out(label, P)}, pre))
    return {"state_set": "core", "label": "live", "wall_init": W0 - 10, "mono_init": M0 - 10, "steps": steps}


def spec_degenerate(settings: dict, room_only: bool, with_room: bool) -> dict:
    """Degenerate settings: empty window, zero drop time, room-only without room rows."""
    P = [0.1, 0.2, 0.3, 0.4]
    steps = []
    for i in range(8):
        recs = ([room(i)] if with_room else []) + [person(i, ts=W0 + 5 * i - (0.5 if i == 4 else 0.0)),
                                                   person(i, pid=2)]
        steps.append(step(i, recs, {f"{i}:1": out("Focused", P), f"{i}:2": out("Exhausted", P),
                                    f"{i}:0": out("Focused", P)}))
    return {"state_set": "core", "settings": settings, "room_only": room_only, "steps": steps}


def build_specs() -> dict[str, dict]:
    """All specs, in the order of the test's SPECS."""
    return {
        "sv": spec_sv(),
        "room": spec_room(),
        "op": spec_op(),
        "w0": spec_degenerate({"window_ticks": 0}, False, True),
        "drop0": spec_degenerate({"person_drop_min": 0.0, "window_ticks": 3}, False, True),
        "noroom": spec_degenerate({"window_ticks": 5}, True, False),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", metavar="DIR", help="write DIR/<name>.json instead of printing")
    out = out_dir(ap.parse_args().out)
    specs = build_specs()
    if out is None:
        print(json.dumps(specs))
        return
    for name, spec in specs.items():
        with open(out / f"{name}.json", "w") as f:
            json.dump(spec, f)
        print(out / f"{name}.json")


if __name__ == "__main__":
    main()
