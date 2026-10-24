"""Golden fixture for the TypeScript emotion map (Valence Web) -> frontend/src/test/fixtures/engine/emotion.json.gz.

    .venv/bin/python -B scripts/export_web_golden_emotion.py            # write, then re-check the written file
    .venv/bin/python -B scripts/export_web_golden_emotion.py --check-only

The Python emotion map is the reference: ``valence/features/emotion.py`` (the 31 emotion-v1 window features) and
``valence/models/emotion.py`` ``EmotionPredictor`` (Cloud B's exported models in artifacts/emotion/). The vitest
parity test (frontend/src/test/engine-emotion.test.ts) replays this file through frontend/src/engine/
emotion-features.ts and emotion-model.ts. artifacts/emotion/golden.json (Cloud B's own golden data) is tested
next to it and is not touched here.

Data
----
Simulator sessions with the ten-state set (``simulate_session(sid, default_rng(seed), state_set="emotion")``,
the generator Cloud B trains on; the face cues come from the session's own ``crc32(sid + "/face-cues")``
stream). SESSIONS picks seed 4 (three people, all ten states), seed 2 (one person, a third of the ticks
without a readable face) and seed 7 (two people, all ten states, a third of the ticks without a face). Rows become TickRecords exactly as the replay source does (``TickRecord(**row)``
over TICK_FIELDS); each person's window is their own rows, oldest first, the last <= 180 (RollingWindow), and
the room window is the room rows up to the same tick (last <= 180).

* checkpoints: every 4th row of every person (row 0, 4, 8, ... and the last row) -> ``emotion_from_window``.
* edge_windows: synthetic variants of a real window (whole window without a readable face, partial face, face on
  the last tick only, the 10 s blink-rate threshold, NaN in face_visible / blink_count, pose missing, constant
  cues, more than 180 ticks, a single tick, an empty window) -> ``compute_emotion`` on their channels.
* model_cases: every 24th row of every person plus every edge window that has a full TickRecord window:
  x123 = ``features_from_window(person, room)``, x = ``model_input(x123, emotion_from_window(person))`` (132 values,
  meta.json input_features order) and ``EmotionPredictor.predict(x)``; plus, for the first case of each state, the
  same x with the whole face block hidden (face_* and the 31 NaN, face_visible_now / face_visible_frac_5m = 0, as
  scripts/emotion_export.py builds its face-hidden vectors; x123 is null there). Each case's ``affect`` is built
  the way ``valence/models/emotion.py`` ``affect_for_window`` builds it (unrounded floats, label = argmax p); for
  every case with a TickRecord window it is checked equal to ``affect_for_window(pred, person, room)`` itself.
* affect_windows: a few whole TickRecord windows (``TickRecord.to_dict()``, person and room) with
  ``affect_for_window``'s result: a short window, the same window with no room rows (``room or person``), a window
  with no readable face, and an empty person window (None).

Schema
------
{version: 1, generator, feature_version, model_id, temperature, states: [10 ids],
 emotion_feature_names: [31], input_features: [132], emotion_fields: [9 TickRecord fields],
 sessions: [{session_id, seed, state_set, n_ticks,
             people: [{person_id, n_rows, channels: {field: [n_rows]}}]}],        # EMOTION_FIELDS columns
 checkpoints: [{session, person_id, end, hist, state, features: [31]}],
     # session = index into sessions; window = that person's rows[end - hist + 1 .. end]
 edge_windows: [{name, channels: {field: [n]}, features: [31]}],
 model_cases: [{name, state, window, x123: [123] | null, x: [132], raw: [10], p: [10], k, label, confidence,
                valence_mix, arousal_mix, valence_raw, arousal_raw, valence, arousal, affect: AffectDict}],
 affect_windows: [{name, person: [TickRecordDict], room: [TickRecordDict], affect: AffectDict | null}]}
     # window: {session, person_id, end, hist} (a checkpoint window), {edge: name} (an edge window) or null
     # (the face-hidden x variants); valence/arousal are the clipped regressors, *_mix the served map position;
     # affect = Affect(...).model_dump() as affect_for_window builds it (no rounding)

Encoding: compact JSON, floats written by json.dumps (Python repr, so JSON.parse yields the identical double),
NaN -> null, +-inf refused, gzip with mtime=0 (reruns are byte-identical). Size budget 1.5 MB.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from valence.contracts import EMOTION_DISPLAY, TICK_FIELDS, Affect, TickRecord  # noqa: E402
from valence.features.compute import FEATURE_NAMES, MAX_TICKS, features_from_window  # noqa: E402
from valence.features.emotion import (  # noqa: E402
    EMOTION_FEATURE_NAMES,
    EMOTION_FIELDS,
    FEATURE_VERSION,
    INPUT_FEATURES,
    compute_emotion,
    emotion_from_window,
    model_input,
)
from valence.models.emotion import EmotionPredictor, affect_for_window  # noqa: E402
from valence.sim.generator import simulate_session  # noqa: E402

ART = ROOT / "artifacts" / "emotion"
OUT = ROOT / "frontend" / "src" / "test" / "fixtures" / "engine" / "emotion.json.gz"
SIZE_BUDGET = 1_500_000
SESSIONS = (4, 2, 7)  # simulate_session seeds; see the module docstring
CHECKPOINT_EVERY = 4
MODEL_EVERY = 24
ROOM = 0
CUE_FIELDS = ("brow_down", "eye_open", "mouth_frown", "smile", "head_yaw", "head_pitch", "head_away")

# ---------- encoding ----------


def jsonable(v: Any) -> Any:
    """Plain JSON types: NaN -> None, numpy -> Python, tuples -> lists. Refuses inf."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            raise ValueError("inf cannot be written to a fixture")
        return f
    if v is None or isinstance(v, (int, str)):
        return v
    raise TypeError(f"cannot encode {type(v).__name__}")


def encode(obj: dict[str, Any]) -> bytes:
    text = json.dumps(jsonable(obj), allow_nan=False, separators=(",", ":"))
    return gzip.compress(text.encode("ascii"), compresslevel=9, mtime=0)


def nn(v: Any) -> Any:
    """null -> NaN, recursively."""
    if v is None:
        return math.nan
    if isinstance(v, list):
        return [nn(x) for x in v]
    return v


def same(a: Any, b: Any) -> bool:
    """Bit-identical float arrays (NaN pattern included)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))


# ---------- windows ----------


def records_of(df) -> dict[int, list[TickRecord]]:
    """person_id -> TickRecords sorted by tick (the room slot is person 0)."""
    out: dict[int, list[TickRecord]] = {}
    for row in df.sort_values(["tick", "person_id"]).to_dict("records"):
        rec = TickRecord(**{k: row[k] for k in TICK_FIELDS})
        rec.tick, rec.person_id, rec.n_people = int(rec.tick), int(rec.person_id), int(rec.n_people)
        for k in ("laughter_count", "compliments_count", "clenched_fists_count", "social_withdrawal_flag",
                  "blink_count"):
            setattr(rec, k, int(getattr(rec, k)))
        out.setdefault(rec.person_id, []).append(rec)
    return out


def states_of(df) -> dict[tuple[int, int], str]:
    p = df[df["person_id"] != ROOM]
    return {(int(pid), int(t)): str(s) for pid, t, s in zip(p["person_id"], p["tick"], p["true_state"])}


def room_window(room: list[TickRecord], tick: int) -> list[TickRecord]:
    rows = [r for r in room if r.tick <= tick]
    return rows[-MAX_TICKS:]


def channels(rows: list[TickRecord]) -> dict[str, list[float]]:
    return {f: [float(getattr(r, f)) for r in rows] for f in EMOTION_FIELDS}


def hide_face(r: TickRecord) -> TickRecord:
    """A tick without a readable face (what the aggregator writes when no face frame was counted)."""
    return dataclasses.replace(r, face_valence=math.nan, face_visible=0.0, blink_count=0,
                               **{k: math.nan for k in CUE_FIELDS})


# ---------- model ----------


def affect_of(pred: EmotionPredictor, p: np.ndarray, valence: float, arousal: float) -> dict:
    """The Prediction.affect message for one prediction, exactly as valence/models/emotion.py affect_for_window
    builds it (map position, label = first argmax of p, unrounded floats), validated by the pydantic model.
    """
    k = int(np.argmax(p))
    return Affect(valence=float(valence), arousal=float(arousal), label=pred.states[k],
                  label_display=EMOTION_DISPLAY[pred.states[k]], confidence=float(p[k]),
                  probabilities=dict(zip(pred.states, map(float, p))), model=pred.model_id).model_dump()


def model_case(pred: EmotionPredictor, name: str, state: str, window: dict | None, x123: np.ndarray | None,
               x: np.ndarray) -> dict:
    r = pred.predict(x)
    p = r["proba"][0]
    k = int(np.argmax(p))
    mix_v, mix_a = float(r["valence_mix"][0]), float(r["arousal_mix"][0])
    return {"name": name, "state": state, "window": window, "x123": None if x123 is None else x123, "x": x,
            "raw": r["raw"][0], "p": p, "k": k, "label": pred.states[k], "confidence": float(p[k]),
            "valence_mix": mix_v, "arousal_mix": mix_a,
            "valence_raw": float(r["valence_raw"][0]), "arousal_raw": float(r["arousal_raw"][0]),
            "valence": float(r["valence"][0]), "arousal": float(r["arousal"][0]),
            "affect": affect_of(pred, p, mix_v, mix_a)}


def face_hidden_x(pred: EmotionPredictor, x: np.ndarray) -> np.ndarray:
    """scripts/emotion_export.py's face-hidden variant: every face_* and emotion feature NaN, visibility 0."""
    xh = x.copy()
    emo = set(pred.meta["emotion_feature_names"])
    for i, c in enumerate(pred.cols):
        if c.startswith("face_") or c in emo:
            xh[i] = np.nan
    for c in ("face_visible_now", "face_visible_frac_5m"):
        xh[pred.cols.index(c)] = 0.0
    return xh


# ---------- edge windows ----------


def edge_windows(person: list[TickRecord], room: list[TickRecord]) -> list[tuple[str, list[TickRecord] | None,
                                                                                 list[TickRecord] | None,
                                                                                 dict[str, list[float]] | None]]:
    """(name, person window, room window, raw channels) for the synthetic cases.

    A case with a TickRecord window gets a model case too; channel-only cases (NaN in int fields, empty) feed
    compute_emotion directly.
    """
    end = min(len(person) - 1, 400)
    base = person[end - MAX_TICKS + 1:end + 1]
    rw = room_window(room, base[-1].tick)
    out: list[tuple[str, Any, Any, Any]] = []
    out.append(("no readable face in the window", [hide_face(r) for r in base], rw, None))
    out.append(("face hidden on the last 30 ticks", [hide_face(r) if i >= len(base) - 30 else r
                                                     for i, r in enumerate(base)], rw, None))
    out.append(("face hidden on the last 70 ticks", [hide_face(r) if i >= len(base) - 70 else r
                                                     for i, r in enumerate(base)], rw, None))
    out.append(("face on alternate ticks", [hide_face(r) if i % 2 else r for i, r in enumerate(base)], rw, None))
    out.append(("face on the last tick only", [r if i == len(base) - 1 else hide_face(r)
                                               for i, r in enumerate(base)], rw, None))
    two = [hide_face(r) for r in base[-12:]]
    two[-1] = dataclasses.replace(base[-1], face_visible=1.0, blink_count=1)
    two[-5] = dataclasses.replace(base[-5], face_visible=1.0, blink_count=2)
    out.append(("blink threshold: 2 visible ticks = 10 s", two, rw, None))
    short = list(two)
    short[-1] = dataclasses.replace(two[-1], face_visible=0.99)
    out.append(("blink threshold: 9.95 s visible", short, rw, None))
    out.append(("pose missing, expressions present",
                [dataclasses.replace(r, head_yaw=math.nan, head_pitch=math.nan, head_away=math.nan) for r in base],
                rw, None))
    out.append(("first tick only", base[:1], room_window(room, base[0].tick), None))
    out.append(("first tick only, no face", [hide_face(base[0])], room_window(room, base[0].tick), None))
    long = person[max(0, end - 239):end + 1]
    out.append(("240 ticks (the last 180 are used)", long, rw, None))
    # channel-only cases
    ch = channels(base[-60:])
    for j in range(0, 60, 7):
        ch["face_visible"][j] = math.nan
        ch["blink_count"][j] = math.nan
    out.append(("NaN face_visible and blink_count count as 0", None, None, ch))
    const = channels(base[-24:])
    for f in CUE_FIELDS:
        const[f] = [0.125] * 24
    out.append(("constant cues (std exactly 0)", None, None, const))
    out.append(("empty window", None, None, {f: [] for f in EMOTION_FIELDS}))
    neg = channels(base[-20:])
    neg["brow_down"] = [-abs(v) - 0.01 if not math.isnan(v) else v for v in neg["brow_down"]]
    out.append(("negative brow (max of negatives)", None, None, neg))
    return out


def affect_window_cases(pred: EmotionPredictor, person: list[TickRecord], room: list[TickRecord]) -> list[dict]:
    """Whole TickRecord windows through affect_for_window (the live path: features_from_window, emotion_from_window,
    model_input, predict, Affect), for the TypeScript affectForWindow."""
    end = min(len(person) - 1, 300)
    pw = person[end - 39:end + 1]
    rw = room_window(room, pw[-1].tick)[-40:]
    cases = [("40 ticks", pw, rw), ("40 ticks, no room rows (room or person)", pw, []),
             ("12 ticks without a readable face", [hide_face(r) for r in pw[-12:]], rw[-12:]),
             ("empty person window", [], rw[-5:])]
    out = []
    for name, p, r in cases:
        a = affect_for_window(pred, p, r)
        out.append({"name": name, "person": [x.to_dict() for x in p], "room": [x.to_dict() for x in r],
                    "affect": None if a is None else a.model_dump()})
    return out


def record_of(d: dict) -> TickRecord:
    """TickRecord.to_dict() -> TickRecord (null -> NaN, as the replay source and tickRecordFromDict read it)."""
    return TickRecord(**{k: (math.nan if v is None else v) for k, v in d.items()})


# ---------- export ----------


def export() -> dict[str, Any]:
    pred = EmotionPredictor(ART)
    assert pred.cols == INPUT_FEATURES, "artifacts/emotion/meta.json input_features differ from INPUT_FEATURES"
    sessions, checkpoints, edges, models = [], [], [], []
    hidden_done: set[str] = set()
    for s_idx, seed in enumerate(SESSIONS):
        sid = f"web-emotion-{seed}"
        df = simulate_session(sid, np.random.default_rng(seed), state_set="emotion")
        recs = records_of(df)
        state = states_of(df)
        room = recs.pop(ROOM)
        people = []
        for pid in sorted(recs):
            rows = recs[pid]
            people.append({"person_id": pid, "n_rows": len(rows), "channels": channels(rows)})
            ends = sorted(set(range(0, len(rows), CHECKPOINT_EVERY)) | {len(rows) - 1})
            for end in ends:
                win = rows[max(0, end - MAX_TICKS + 1):end + 1]
                st = state[(pid, rows[end].tick)]
                feats = emotion_from_window(win)
                arrays = {f: np.array([getattr(r, f) for r in win], dtype=float) for f in EMOTION_FIELDS}
                assert same(feats, compute_emotion(arrays))
                checkpoints.append({"session": s_idx, "person_id": pid, "end": end, "hist": len(win), "state": st,
                                    "features": feats})
                if end % MODEL_EVERY == MODEL_EVERY // 2:
                    x123 = features_from_window(win, room_window(room, rows[end].tick))
                    x = model_input(x123, feats)
                    name = f"{sid} person {pid} row {end}"
                    window = {"session": s_idx, "person_id": pid, "end": end, "hist": len(win)}
                    models.append(model_case(pred, name, st, window, x123, x))
                    live = affect_for_window(pred, win, room_window(room, rows[end].tick))
                    assert live is not None and live.model_dump() == models[-1]["affect"], name
                    if st not in hidden_done:
                        hidden_done.add(st)
                        models.append(model_case(pred, name + " (x face hidden)", st, None, None,
                                                  face_hidden_x(pred, x)))
        sessions.append({"session_id": sid, "seed": seed, "state_set": "emotion", "n_ticks": int(df["tick"].max()) + 1,
                         "people": people})
        if s_idx == 0:
            pid = sorted(recs)[0]
            for name, pw, rw, ch in edge_windows(recs[pid], room):
                if pw is not None:
                    feats = emotion_from_window(pw)
                    ch = channels(pw)
                    x123 = features_from_window(pw, rw)
                    st = state[(pid, pw[-1].tick)]
                    models.append(model_case(pred, f"edge: {name}", st, {"edge": name}, x123,
                                              model_input(x123, feats)))
                    live = affect_for_window(pred, pw, rw)
                    assert live is not None and live.model_dump() == models[-1]["affect"], name
                else:
                    feats = compute_emotion({f: np.asarray(v, dtype=float) for f, v in ch.items()})
                edges.append({"name": name, "channels": ch, "features": feats})
            affect_windows = affect_window_cases(pred, recs[pid], room)
    probes = model_probes(pred)
    return {
        "version": 1, "generator": "scripts/export_web_golden_emotion.py", "feature_version": FEATURE_VERSION,
        "model_id": pred.model_id, "temperature": pred.t, "states": pred.states,
        "emotion_feature_names": EMOTION_FEATURE_NAMES, "input_features": pred.cols,
        "emotion_fields": list(EMOTION_FIELDS), "sessions": sessions, "checkpoints": checkpoints,
        "edge_windows": edges, "model_cases": models, "affect_windows": affect_windows, "probes": probes,
    }


def model_probes(pred: EmotionPredictor) -> dict[str, Any]:
    """Model-dependent expectations for the TS tests, so a retrain only needs a re-export (no literals in tests):
    golden vector 0 with every third input missing, golden vector 1 with blink_rate_1m = +inf and 1e301 (dump
    thresholds printed as 1e300 by LightGBM's AvoidInf), and the classifier's tree count."""
    golden = json.loads((ROOT / "artifacts" / "emotion" / "golden.json").read_text())
    vec = lambda i: np.array([math.nan if v is None else v for v in golden["vectors"][i]["x"]], dtype=float)

    def run(x: np.ndarray) -> dict[str, Any]:
        o = pred.predict(x.reshape(1, -1))
        k = int(np.argmax(o["proba"][0]))
        return {"raw": [float(v) for v in o["raw"][0]], "label": pred.states[k],
                "valence_reg": float(o["valence"][0]), "arousal_reg": float(o["arousal"][0]),
                "valence_mix": float(o["valence_mix"][0]), "arousal_mix": float(o["arousal_mix"][0])}

    nulls = vec(0)
    nulls[::3] = math.nan
    f = pred.cols.index("blink_rate_1m")
    inf = {}
    for name, v in (("inf", math.inf), ("1e301", 1e301)):
        x = vec(1)
        x[f] = v
        inf[name] = run(x)
    return {"null_every_third": run(nulls), "blink_rate_1m_huge": inf,
            "classifier_trees": int(pred.models["classifier"].num_trees())}


def check(path: Path) -> None:
    """Re-derive every value from the written file (after the JSON round trip) and compare bit for bit."""
    g = json.loads(gzip.decompress(path.read_bytes()))
    pred = EmotionPredictor(ART)
    assert g["emotion_feature_names"] == EMOTION_FEATURE_NAMES and g["input_features"] == pred.cols
    assert g["model_id"] == pred.model_id and len(FEATURE_NAMES) == 123
    bad = 0

    def window_emotion(session: int, pid: int, end: int, hist: int) -> np.ndarray:
        person = next(p for p in g["sessions"][session]["people"] if p["person_id"] == pid)
        lo = end - hist + 1
        return compute_emotion({f: np.asarray(nn(person["channels"][f][lo:end + 1]), dtype=float)
                                for f in EMOTION_FIELDS})

    for c in g["checkpoints"]:
        bad += not same(window_emotion(c["session"], c["person_id"], c["end"], c["hist"]), nn(c["features"]))
    edges = {}
    for e in g["edge_windows"]:
        edges[e["name"]] = compute_emotion({f: np.asarray(nn(v), dtype=float) for f, v in e["channels"].items()})
        bad += not same(edges[e["name"]], nn(e["features"]))
    for m in g["model_cases"]:
        r = pred.predict(np.asarray(nn(m["x"]), dtype=float))
        bad += not (same(r["raw"][0], m["raw"]) and same(r["proba"][0], m["p"])
                    and float(r["valence_mix"][0]) == m["valence_mix"] and float(r["valence_raw"][0]) == m["valence_raw"])
        bad += affect_of(pred, r["proba"][0], m["valence_mix"], m["arousal_mix"]) != m["affect"]
        w = m["window"]
        if w is not None:
            xe = edges[w["edge"]] if "edge" in w else window_emotion(w["session"], w["person_id"], w["end"], w["hist"])
            bad += not same(model_input(np.asarray(nn(m["x123"]), dtype=float), xe), nn(m["x"]))
    for a in g["affect_windows"]:
        live = affect_for_window(pred, [record_of(d) for d in a["person"]], [record_of(d) for d in a["room"]])
        bad += (None if live is None else live.model_dump()) != a["affect"]
    states = sorted({c["state"] for c in g["checkpoints"]})
    labels = sorted({m["label"] for m in g["model_cases"]})
    n_nan = sum(all(v is None for v in c["features"]) for c in g["checkpoints"])
    n_part = sum(any(v is None for v in c["features"]) and not all(v is None for v in c["features"])
                 for c in g["checkpoints"])
    print(f"check: {len(g['checkpoints'])} checkpoints ({n_nan} all-NaN, {n_part} partly NaN), "
          f"{len(g['edge_windows'])} edge windows, {len(g['model_cases'])} model cases, "
          f"{len(g['affect_windows'])} affect windows; "
          f"{len(states)} true states {states}; predicted labels {labels}; mismatches {bad}")
    if bad or len(states) != len(pred.states):
        raise SystemExit("check failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.check_only:
        data = encode(export())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"{out}: {len(data) / 1e6:.3f} MB")
        if len(data) > SIZE_BUDGET:
            raise SystemExit(f"over the {SIZE_BUDGET / 1e6} MB budget")
    check(out)


if __name__ == "__main__":
    main()
