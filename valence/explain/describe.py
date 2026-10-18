"""Turn a feature name + value into the narrator's vocabulary: display text, unit, level, scope, layer.

Display strings carry the only numbers the narrator may use, formatted once here (dB as integers,
hours to one decimal, shares as whole percents, valence with a sign and two decimals) so the
validator can match numbers in generated text against the fact sheet exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from valence.features.spec import layer_of, scope_of

WINDOWS = {"1m": "minute", "5m": "5 minutes", "15m": "15 minutes"}
_NAME = re.compile(r"^(?P<ch>[a-z]+)_(?P<stat>[a-z0-9_]+?)(?:_(?P<win>1m|5m|15m))?$")


@dataclass(frozen=True)
class Description:
    display: str
    unit: str
    level: str  # high | elevated | normal | low
    scope: str  # room | person
    layer: str


def _level(value: float, high: float, elevated: float, low: float) -> str:
    if value >= high:
        return "high"
    if value >= elevated:
        return "elevated"
    if value < low:
        return "low"
    return "normal"


def _span(win: str | None) -> str:
    return f"the last {WINDOWS[win]}" if win and win != "1m" else "the last minute"


def _pct(v: float) -> str:
    return f"{round(100 * v)}%"


def _clock(tod_h: float) -> str:
    h = int(tod_h) % 24
    return f"{h:02d}:{int(round((tod_h - int(tod_h)) * 60)) % 60:02d}"


def _room_relative(ch: str, stat: str, span: str, value: float, absolute: float | None, d) -> Description:
    """Room level relative to the venue baseline; the display leads with the absolute level when known."""
    known = absolute is not None and not math.isnan(absolute)
    if ch == "db":
        what = {"now": "current noise", "mean": "average noise", "max": "peak noise", "ewm": "recent noise"}[stat]
        tail = "" if stat == "now" else f" over {span}"
        delta = round(value)
        rel = "at this room's baseline" if delta == 0 else \
            f"{abs(delta)} dB {'above' if delta > 0 else 'below'} this room's baseline"
        text = f"{what} of {round(absolute)} dB{tail}, {rel}" if known else f"{what} {rel}{tail}"
        return d(text, "dB", _level(value, 8, 4, -3))
    rel = "brighter than this room's usual light" if value >= 0.5 else \
        "dimmer than this room's usual light" if value <= -0.5 else "about this room's usual light"
    text = f"{round(absolute)} lux, {rel}" if known else f"light {rel}"
    return d(text, "lux", _level(value, 1.0, 0.5, -0.5))


def describe(feature: str, value: float | None, *, tod_h: float | None = None,
             absolute: float | None = None) -> Description | None:
    """Return None for features that are never narrated (or have no value).

    ``absolute`` is the absolute level behind a room-relative feature (``db_room_mean_5m`` -> ``db_mean_5m``),
    quoted in its display.
    """
    if feature == "fill_ratio" or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    layer, scope = layer_of(feature), scope_of(feature)

    def d(display: str, unit: str, level: str) -> Description:
        return Description(display, unit, level, scope, layer)

    if feature == "hours_since_break":
        return d(f"{value:.1f} hours since the last break", "h", _level(value, 3.5, 2.0, 0.75))
    if feature in ("tod_sin", "tod_cos"):
        if tod_h is None:
            return None
        late = tod_h >= 22 or tod_h < 6
        return d(f"time of day ({_clock(tod_h)})", "", "high" if late else "normal")
    if feature == "n_people":
        n = int(value)
        return d(f"{n} {'person' if n == 1 else 'people'} present", "people", "elevated" if n >= 2 else "normal")
    if feature == "withdraw_now":
        return d("withdrawn right now" if value else "not withdrawn right now", "", "high" if value else "low")

    m = _NAME.match(feature)
    if not m:
        return d(f"{feature} = {value:g}", "", "normal")
    ch, stat, win = m["ch"], m["stat"], m["win"]
    span = _span(win)

    if ch in ("db", "lux") and stat.startswith("room_"):
        return _room_relative(ch, stat.removeprefix("room_"), span, value, absolute, d)

    if ch in ("fist", "laugh", "compl"):
        noun = {"fist": "clenched-fist event", "laugh": "laughter event", "compl": "compliment"}[ch]
        where = " in the room" if ch != "fist" else ""
        if stat == "cnt":
            n = int(round(value))
            per5 = n * {"1m": 5, "5m": 1, "15m": 1 / 3}.get(win or "5m", 1)
            text = f"no {noun}s{where} in {span}" if n == 0 else f"{n} {noun}{'s' if n != 1 else ''}{where} in {span}"
            return d(text, "events", _level(per5, 3 if ch == "fist" else 2, 1, 0.01))
        if stat.startswith("since"):
            s = int(round(value))
            text = f"no {noun} in the last 15 minutes" if s >= 900 else f"last {noun} {s} s ago"
            return d(text, "s", "low" if s >= 900 else ("high" if s < 60 else "elevated"))

    if ch == "db":
        if stat in ("now", "mean", "max", "ewm", "p90"):
            what = {"now": "current noise", "mean": "average noise", "max": "peak noise",
                    "ewm": "recent noise", "p90": "noise peaks"}[stat]
            tail = "" if stat == "now" else f" over {span}"
            return d(f"{what} of {round(value)} dB{tail}", "dB", _level(value, 65, 58, 45))
        if stat == "d1m15m":
            sign = "above" if value >= 0 else "below"
            return d(f"noise {abs(round(value))} dB {sign} its 15-minute average", "dB",
                     _level(value, 6, 3, -3))
        if stat == "slope":
            trend = "rising" if value >= 0 else "falling"
            return d(f"noise {trend} {abs(value):.1f} dB per minute over {span}", "dB/min", _level(value, 1.0, 0.4, -0.4))
        if stat.startswith("rel"):
            sign = "above" if value >= 0 else "below"
            return d(f"noise {abs(round(value))} dB {sign} its 15-minute median", "dB", _level(value, 6, 3, -3))
        if stat == "std":
            return d(f"noise varying by {value:.1f} dB over {span}", "dB", _level(value, 6, 3, 1))

    if ch == "lux":
        if stat in ("now", "mean", "ewm", "max"):
            lvl = _level(value, 450, 250, 200)
            word = {"low": "dim light", "normal": "normal light", "elevated": "normal light", "high": "bright light"}[lvl]
            return d(f"{word} ({round(value)} lux)", "lux", lvl)
        if stat == "d1m15m" or stat.startswith("rel"):
            sign = "brighter" if value >= 0 else "darker"
            ref = "its 15-minute median" if stat.startswith("rel") else "its 15-minute average"
            return d(f"light {abs(round(value))} lux {sign} than {ref}", "lux", _level(value, 100, 40, -40))
        if stat in ("slope", "std"):
            return d(f"light changing ({value:+.1f} lux/min)", "lux/min", _level(abs(value), 20, 8, 0.5))

    if ch in ("slump", "hih", "withdraw"):
        what = {"slump": "slumped", "hih": "head in hands", "withdraw": "withdrawn"}[ch]
        if stat in ("now", "mean", "frac", "ewm", "max"):
            v = min(max(value, 0.0), 1.0)
            text = f"{what} {_pct(v)} of {span}" if stat != "now" else f"{what} {_pct(v)} of the last 5 seconds"
            return d(text, "%", _level(v, 0.6, 0.35, 0.1))
        if stat in ("slope", "d1m15m", "std") or stat.startswith("rel"):
            return d(f"{what} share changing ({value:+.2f})", "", _level(abs(value), 0.3, 0.15, 0.02))

    if ch == "motion":
        if stat in ("now", "mean", "ewm", "max"):
            lvl = _level(value, 0.10, 0.05, 0.015)  # live: still ~0.014, typing ~0.03
            word = {"low": "nearly still", "normal": "moving normally", "elevated": "restless", "high": "very restless"}[lvl]
            return d(f"{word} (motion {value:.3f})", "sw/s", lvl)
        return d(f"motion changing ({value:+.3f})", "sw/s", _level(abs(value), 0.05, 0.02, 0.003))

    if ch == "face":
        if stat.startswith("visible"):
            return d(f"face visible {_pct(min(max(value, 0.0), 1.0))} of {span}", "%", _level(value, 0.8, 0.5, 0.4))
        if stat in ("now", "mean"):
            lvl = _level(value, 0.3, 0.1, -0.2)
            word = {"high": "smiling expression", "elevated": "slightly positive expression",
                    "normal": "neutral expression", "low": "negative expression"}[lvl]
            tail = "" if stat == "now" else f" over {span}"
            return d(f"{word}{tail} (valence {value:+.2f})", "", lvl)
        if stat == "std":
            return d(f"expression varying (spread {value:.2f})", "", _level(value, 0.35, 0.2, 0.05))

    return d(f"{feature} = {value:g}", "", "normal")
