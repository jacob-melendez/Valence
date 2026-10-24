"""Parity fixtures for frontend/src/perception/audioFeatures.ts and resample.ts.

Ports valence/perception/audio_features.py (mean_power, power_to_db, PowerSmoother, _family_max, LaughterTracker,
resample_48k_to_16k) and scipy.signal.resample_poly with its defaults as resample_48k_to_16k calls it (float32 in,
float32 out). Writes audio-features.json, audio-resample.json and audio-resample-<rate>.json (one 1 s block per rate).

Encodings: a float32 block is {"scale": s, "ints": [...]} (x = ints / s, exact in float32), {"scale": s, "i16": b64}
(the same with the ints as base64 little-endian int16), {"floats": [...]} or {"f32": b64} (base64 little-endian
float32); a resampled output is {"f32": b64}, its exact float32 bits. Non-finite floats are the strings "NaN",
"Infinity", "-Infinity". The base64 forms keep the long blocks small (gzip packs float32 bits far better than their
decimal digits).
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scipy.signal import firwin, resample_poly  # noqa: E402
from scipy.signal.windows import kaiser  # noqa: E402
from scipy.special import i0  # noqa: E402

from valence.perception.audio_features import (  # noqa: E402
    LAUGH_FAMILY,
    POWER_FLOOR,
    YAMNET_ALLOWLIST,
    LaughterTracker,
    PowerSmoother,
    _family_max,
    mean_power,
    power_to_db,
    resample_48k_to_16k,
)
from valence.perception.debounce import HysteresisDebouncer  # noqa: E402

SEED = 20261024
RATES = (48000, 44100, 32000, 22050, 16000)
# Other sample rates a browser AudioContext may run at: a short block each (every tap of the filter still meets data).
EXTRA_RATES = (8000, 11025, 12000, 24000, 88200, 96000, 192000)


def enc(v: object) -> object:
    """Strict-JSON value: non-finite floats become strings, numpy scalars become Python numbers."""
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
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
    return v


def write(out_dir: Path, name: str, obj: object) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(enc(obj), allow_nan=False, separators=(",", ":"))
    (out_dir / f"{name}.json").write_text(text + "\n", encoding="utf-8")


def ints_block(ints: np.ndarray, scale: int = 32768) -> tuple[np.ndarray, dict]:
    ints = np.asarray(ints, dtype=np.int64)
    x = (ints.astype(np.float64) / scale).astype(np.float32)
    return x, {"scale": scale, "ints": ints.tolist()}


def i16_block(ints: np.ndarray, scale: int = 32768) -> tuple[np.ndarray, dict]:
    """ints_block with the ints as base64 int16 (int16 PCM range only)."""
    ints = np.asarray(ints, dtype=np.int64)
    assert ints.size == 0 or (ints.min() >= -32768 and ints.max() <= 32767)
    x = (ints.astype(np.float64) / scale).astype(np.float32)
    return x, {"scale": scale, "i16": base64.b64encode(ints.astype("<i2").tobytes()).decode("ascii")}


def f32_b64(values: np.ndarray) -> dict:
    """Exact float32 bits as base64 little-endian float32."""
    return {"f32": base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")}


def floats_block(values: np.ndarray, dtype: type = np.float32) -> tuple[np.ndarray, dict]:
    x = np.asarray(values, dtype=dtype)
    return x, {"floats": [float(v) for v in x]}


def pcm(rng: np.random.Generator, n: int, sr: int = 16000, level: float = 0.25) -> np.ndarray:
    """Random int16-range audio: noise + tones (one above 8 kHz) + a few clicks."""
    t = np.arange(n) / sr
    x = level * rng.standard_normal(n) * rng.uniform(0.2, 1.0)
    for _ in range(3):
        f = rng.uniform(50.0, min(0.49 * sr, 12000.0))
        x += rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    for i in rng.integers(0, max(n, 1), size=min(5, n)):
        x[i] += rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
    return np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int64)


# --------------------------------------------------------------------------- features


def mean_power_cases(rng: np.random.Generator) -> list[dict]:
    cases: list[dict] = []

    def add(x: np.ndarray, data: dict, dtype: str) -> None:
        p = mean_power(x)
        cases.append({"dtype": dtype, "data": data, "p": p, "db": power_to_db(p), "db94": power_to_db(p, 94.0)})

    for n in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 130, 136, 255, 256, 257, 1000, 1023, 4800, 16000):
        x, data = ints_block(pcm(rng, n))
        add(x, data, "float32")
    x, data = i16_block(pcm(rng, 48000, 48000))
    add(x, data, "float32")  # a 1 s block at 48 kHz: deep pairwise recursion
    for n in (0, 5, 100, 1000, 16000):
        ints = pcm(rng, n)
        add(ints.astype(np.int16), {"ints": ints.tolist()}, "int16")
    add(np.array([-32768] * 9 + [32767] * 3, dtype=np.int16), {"ints": [-32768] * 9 + [32767] * 3}, "int16")
    for dtype, lo, hi in (("int8", -128, 128), ("uint8", 0, 256), ("uint16", 0, 65536),
                          ("int32", -2**31, 2**31), ("uint32", 0, 2**32)):
        for n in (0, 1, 33, 300):
            ints = rng.integers(lo, hi, size=n, dtype=np.int64)
            add(ints.astype(dtype), {"ints": ints.tolist()}, dtype)
    for n in (1, 9, 200, 2000):
        x, data = floats_block(rng.standard_normal(n) * 10.0 ** rng.uniform(-6, 0), np.float32)
        add(x, data, "float32")
    for n in (0, 3, 129, 1500):
        x, data = floats_block(rng.standard_normal(n) * 10.0 ** rng.integers(-150, 150, size=n), np.float64)
        add(x, data, "float64")
    zeros = np.zeros(1000, np.float32)
    add(zeros, {"scale": 32768, "ints": [0] * 1000}, "float32")  # -> floor
    add(np.full(17, 1e-30, np.float32), {"floats": [float(np.float32(1e-30))] * 17}, "float32")
    add(np.full(4, 1e-160), {"floats": [1e-160] * 4}, "float64")  # squares underflow to 0 -> floor
    for special in ([np.nan], [np.inf], [-np.inf], [np.inf, np.nan], [np.inf, -np.inf]):
        vals = rng.standard_normal(40).astype(np.float32).astype(np.float64)
        pos = rng.choice(40, size=len(special), replace=False)
        vals[pos] = special
        for dtype in (np.float32, np.float64):
            x, data = floats_block(vals, dtype)
            add(x, data, "float32" if dtype is np.float32 else "float64")
    return cases


def power_to_db_cases(rng: np.random.Generator) -> list[dict]:
    ps = [0.0, -0.0, -1.0, 1e-13, POWER_FLOOR, 1.0000000000000002e-12, 1e-11, 0.5, 1.0, 2.0, 1e10, 5e-324,
          math.inf, -math.inf, math.nan]
    ps += [float(10.0 ** v) for v in rng.uniform(-14, 3, 25)]
    out = []
    for p in ps:
        for off in (0.0, 94.0, -3.5, math.nan, math.inf):
            out.append({"p": p, "offset": off, "db": power_to_db(p, off)})
    return out


def power_sequence(rng: np.random.Generator, n: int) -> list[float]:
    seq: list[float] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.05:
            seq.append(math.nan)
        elif r < 0.08:
            seq.append(-float(rng.uniform(0, 1)))
        elif r < 0.10:
            seq.append(float(rng.choice([math.inf, -math.inf, 0.0, -0.0])))
        else:
            seq.append(float(10.0 ** rng.uniform(-12, 0)))
    return seq


def smoother_cases(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    configs: list[dict] = [{"alpha": 0.3}, {"alpha": 1.0}, {"alpha": 0.05}, {"alpha": float(rng.uniform(0.01, 1))},
                           {"alpha": 0.3, "value": 0.01}, {"alpha": 0.5, "value": math.inf},
                           {"alpha": 0.2, "value": -1.0}, {"alpha": 1e-300}]
    cases = []
    for cfg in configs:
        sm = PowerSmoother(**cfg)
        steps = []
        n = 500 if cfg == {"alpha": 0.3} else 80
        for p in [math.nan, -1.0] + power_sequence(rng, n):
            if rng.random() < 0.03:
                sm.reset()
                steps.append({"op": "reset", "value": sm.value})
            out = sm.update(p)
            steps.append({"op": "update", "p": p, "out": out, "value": sm.value, "db": sm.db(), "db94": sm.db(94.0)})
        cases.append({"init": cfg, "steps": steps})
    invalid = []
    for a in (0.0, -0.0, -0.1, 1.0, 1.0000001, 0.3, 1e-300, math.nan, math.inf, -math.inf):
        try:
            PowerSmoother(alpha=a)
            raises = False
        except ValueError:
            raises = True
        invalid.append({"alpha": a, "raises": raises})
    return cases, invalid


def random_scores(rng: np.random.Generator, level: float) -> dict[str, float]:
    scores: dict[str, float] = {}
    keys = list(YAMNET_ALLOWLIST) + ["Music", "Silence"]
    for k in keys:
        if rng.random() < 0.6:
            r = rng.random()
            if r < 0.05:
                v = math.nan
            elif r < 0.10:
                v = float(rng.choice([0.05, 0.02, 0.0499999999, 0.0200000001, 0.0, -0.0, 0.3, 0.15]))
            elif k == "Speech":
                v = float(rng.random())
            else:
                v = float(level * rng.random())
            scores[k] = v
    return scores


def family_max_cases(rng: np.random.Generator) -> list[dict]:
    fixed: list[dict[str, float]] = [
        {}, {"Speech": 0.9}, {k: math.nan for k in LAUGH_FAMILY}, {"Laughter": math.nan},
        {"Laughter": math.nan, "Giggle": 0.2}, {"Laughter": 0.2, "Giggle": 0.2}, {"Laughter": -0.0},
        {"Laughter": 0.0, "Giggle": -0.0}, {k: -0.5 for k in LAUGH_FAMILY}, {k: -0.0 for k in LAUGH_FAMILY},
        {"Snicker": math.inf}, {"Snicker": -math.inf}, {k: -math.inf for k in LAUGH_FAMILY},
        {"Belly laugh": 0.7, "Chuckle, chortle": 0.9, "Music": 1.0},
    ]
    cases = [{"scores": s, "laugh": _family_max(s)} for s in fixed]
    for _ in range(60):
        s = random_scores(rng, float(rng.choice([0.03, 0.5])))
        cases.append({"scores": s, "laugh": _family_max(s)})
    return cases


def laughter_sequence(rng: np.random.Generator, n: int, jitter: float, step: float = 1.0,
                      loud: float = 0.5) -> list[tuple[dict, float]]:
    """Mostly quiet (laugh family below the 0.02 exit), with bursts, silent gaps, unobserved (all-NaN) results."""
    seq = []
    t = float(rng.uniform(0, 100))
    burst = 0
    for _ in range(n):
        if burst == 0 and rng.random() < 0.08:
            burst = int(rng.integers(1, 6))
        if burst > 0:
            burst -= 1
            level = 0.0 if rng.random() < 0.2 else loud  # a silent second inside a laugh
        else:
            level = float(rng.choice([0.01, 0.015, 0.03], p=[0.6, 0.3, 0.1]))
        scores = random_scores(rng, level)
        r = rng.random()
        if r < 0.04:
            scores.update({k: math.nan for k in LAUGH_FAMILY})  # not observed
        elif r < 0.06:
            scores = {}
        seq.append((scores, t))
        t += step + (float(rng.uniform(-jitter, jitter)) if jitter else 0.0)
    return seq


def laughter_cases(rng: np.random.Generator) -> list[dict]:
    configs: list[dict | None] = [
        None,
        {"on": 0.30, "off": 0.15, "on_count": 2, "off_hold_s": 0.0, "refractory_s": 3.0, "on_hold_s": 0.0},
        {"on": 0.05, "off": 0.02, "on_count": 1, "off_hold_s": 0.0, "refractory_s": 4.0, "on_hold_s": 0.0},
        {"on": 0.10, "off": 0.05, "on_count": 2, "off_hold_s": 1.0, "refractory_s": 2.0, "on_hold_s": 0.5},
    ]
    seqs: list[tuple[dict | None, list[tuple[dict, float]]]] = []
    for cfg in configs:
        seqs.append((cfg, laughter_sequence(rng, 300 if cfg is None else 120, 0.2, loud=0.6 if cfg else 0.5)))
    seqs.append((None, laughter_sequence(rng, 200, 0.0)))  # exact 1 s spacing: off-hold / refractory boundaries
    seqs.append((None, laughter_sequence(rng, 200, 0.0, 0.5)))
    seqs.append((None, laughter_sequence(rng, 200, 0.0, 0.1)))  # accumulated float error: TIME_EPS
    seqs.append((configs[1], laughter_sequence(rng, 120, 0.0, 0.5)))
    # the Python tests' own sequences
    seqs.append((None, [({"Speech": 0.41, "Laughter": 0.04}, 0.0), ({"Speech": 0.11, "Laughter": 0.08}, 1.0),
                        ({"Speech": 0.94, "Laughter": 0.0}, 2.0),
                        ({"Speech": 0.15, "Laughter": 0.08, "Snicker": 0.06}, 3.0)]))
    seqs.append((None, [({"Laughter": 0.08}, 0.0), ({"Laughter": 0.0}, 1.0), ({"Laughter": 0.0}, 2.0),
                        ({"Laughter": 0.04}, 3.0), ({"Laughter": 0.08}, 4.0)]))
    # odd input: NaN and backwards timestamps, +-inf family scores, a large clock (coarser float spacing)
    odd: list[tuple[dict, float]] = []
    t = 1e5 + float(rng.uniform(0, 1))
    for _ in range(150):
        r = rng.random()
        odd.append((random_scores(rng, float(rng.choice([0.01, 0.5]))),
                    math.nan if r < 0.05 else t - float(rng.uniform(0, 3)) if r < 0.12 else t))
        if rng.random() < 0.06:
            odd[-1][0][str(rng.choice(LAUGH_FAMILY))] = float(rng.choice([math.inf, -math.inf]))
        t += float(rng.choice([0.1, 0.5, 1.0, 1.5]))
    seqs.append((None, odd))
    seqs.append((configs[3], odd))
    cases = []
    for cfg, seq in seqs:
        tr = LaughterTracker() if cfg is None else LaughterTracker(debouncer=HysteresisDebouncer(**cfg))
        steps = []
        for scores, t in seq:
            laugh, speech, onset = tr.update(scores, t)
            step = {"scores": scores, "t": t, "laugh": laugh, "speech": speech, "onset": onset, "active": tr.active}
            if rng.random() < 0.1:
                step["take"] = tr.take_onsets()
            steps.append(step)
        cases.append({"debouncer": cfg, "steps": steps, "final_take": tr.take_onsets()})
    return cases


# --------------------------------------------------------------------------- resample


def resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """resample_48k_to_16k's expression with general up/down (the real function for 1/3)."""
    if (up, down) == (1, 3):
        return resample_48k_to_16k(x)
    return resample_poly(np.asarray(x, dtype=np.float32), up, down, axis=0).astype(np.float32, copy=False)


def resample_case(x: np.ndarray, data: dict, up: int, down: int) -> dict:
    y = resample(x, up, down)
    assert y.dtype == np.float32
    return {"up": up, "down": down, "x": data, "y": f32_b64(y)}


def short_resample_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    for n in (0, 1, 2, 3, 4, 5, 7, 10, 33, 100, 4801):
        cases.append(resample_case(*ints_block(pcm(rng, n, 48000)), 1, 3))
    cases.append(resample_case(*ints_block(np.zeros(4801, np.int64)), 1, 3))
    pairs = [(160, 441), (320, 441), (1, 2), (2, 1), (3, 1), (2, 3), (3, 2), (7, 5), (4, 6), (6, 6), (1, 1),
             (5, 1), (1, 7), (441, 160), (10, 30), (1, 20), (13, 1)]
    for up, down in pairs:
        for n in (0, 1, 2, 9, int(rng.integers(50, 700))):
            cases.append(resample_case(*ints_block(pcm(rng, n)), up, down))
    for rate in (44100, 22050, 32000):
        g = math.gcd(16000, rate)
        cases.append(resample_case(*i16_block(pcm(rng, rate // 10 + 7, rate)), 16000 // g, rate // g))
    # non-finite samples spread over the filter span (inf * a zero tap is NaN, as in scipy)
    for special in ([np.nan], [np.inf], [-np.inf, np.inf]):
        for up, down in ((1, 3), (160, 441), (3, 2)):
            vals = (rng.standard_normal(150) * 0.1).astype(np.float32).astype(np.float64)
            vals[rng.choice(np.arange(20, 130), size=len(special), replace=False)] = special
            cases.append(resample_case(*floats_block(vals, np.float32), up, down))
    # float64 input values: the float32 cast happens inside (overflow to inf, underflow to 0)
    for up, down in ((1, 3), (2, 3)):
        vals = rng.standard_normal(60) * 10.0 ** rng.integers(-50, 40, size=60)
        vals[:3] = [1e39, -1e-50, 3.4028235677973366e38]
        x = np.asarray(vals, dtype=np.float64)
        with np.errstate(over="ignore"):
            y = resample(x, up, down)
        cases.append({"up": up, "down": down, "x": {"floats64": [float(v) for v in x]}, "y": f32_b64(y)})
    return cases


def export(out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    smoother, smoother_invalid = smoother_cases(rng)
    write(out_dir, "audio-features", {
        "constants": {"POWER_FLOOR": POWER_FLOOR, "YAMNET_ALLOWLIST": list(YAMNET_ALLOWLIST),
                      "LAUGH_FAMILY": list(LAUGH_FAMILY)},
        "mean_power": mean_power_cases(rng),
        "power_to_db": power_to_db_cases(rng),
        "smoother": smoother,
        "smoother_invalid": smoother_invalid,
        "family_max": family_max_cases(rng),
        "laughter": laughter_cases(rng),
    })

    xs = [0.0, -0.0, 1e-300, 1e-8, 0.5, -3.0, 5.0, 8.0, 8.000000000000002, 20.0, 100.0, 600.0, 700.0, 709.78,
          709.782712893384, 709.7827128933841, 713.0, 714.0, 1e6, math.inf, -math.inf, math.nan]
    xs += [float(v) for v in np.linspace(0, 30, 121)] + [float(v) for v in rng.uniform(0, 5, 60)]
    kaisers = []
    for m in (0, 1, 2, 3, 4, 21, 61, 62):
        for beta in (5.0, 0.0, 8.6, 14.0):
            kaisers.append({"M": m, "beta": beta, "w": [float(v) for v in kaiser(m, beta, sym=True)]})
    firwins = []
    for numtaps, cutoff in ((61, 1 / 3), (41, 1 / 2), (81, 1 / 4), (21, 0.9), (64, 0.3), (5, 0.25), (1, 0.5),
                            (6401, 1 / 320), (8821, 1 / 441)):
        firwins.append({"numtaps": numtaps, "cutoff": cutoff,
                        "h": [float(v) for v in firwin(numtaps, cutoff, window=("kaiser", 5.0))]})
    write(out_dir, "audio-resample", {
        "i0": [{"x": x, "y": float(i0(x))} for x in xs],
        "kaiser": kaisers,
        "firwin": firwins,
        "cases": short_resample_cases(rng),
        "rates": [{"rate": rate, **resample_case(*i16_block(pcm(rng, rate // 20 + 3, rate, level=0.3)),
                                                 16000 // math.gcd(16000, rate), rate // math.gcd(16000, rate))}
                  for rate in EXTRA_RATES],
    })

    for rate in RATES:
        g = math.gcd(16000, rate)
        up, down = 16000 // g, rate // g
        x, data = i16_block(pcm(rng, rate, rate, level=0.3))
        case = resample_case(x, data, up, down)
        write(out_dir, f"audio-resample-{rate}", {"rate": rate, **case})


if __name__ == "__main__":
    export(Path(__file__).resolve().parent)
