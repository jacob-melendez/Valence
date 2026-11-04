/**
 * Turn a feature name + value into the narrator's vocabulary: display text, unit, level, scope, layer
 * (valence/explain/describe.py).
 *
 * Display strings carry the only numbers the narrator may use, formatted once here (dB as integers, hours
 * to one decimal, shares as whole percents, valence with a sign and two decimals) so the validator can match
 * numbers in generated text against the fact sheet exactly. Every string must therefore be identical to
 * Python's, which is why all formatting goes through num.ts: Python's round() and f"{x:.Nf}" round the exact
 * binary value half to even, and round()/int() print every integer digit (`round(1e21)` is 22 digits, not
 * "1e+21"), so integers are rendered through BigInt here.
 */
import { layerOf, scopeOf, type LayerName } from "./contracts";
import { isnan, pyFixed, pyG, pyIntRound } from "./num";

// ---------- Types ----------

export type Level = "high" | "elevated" | "normal" | "low";
export type Scope = "room" | "person";

/** valence/explain/describe.py:20 Description */
export interface Description {
  display: string;
  unit: string;
  level: Level;
  scope: Scope;
  layer: LayerName;
}

/** Keyword arguments of describe() (valence/explain/describe.py:70). null and undefined both mean None. */
export interface DescribeOptions {
  /** Local time of day in hours, quoted by tod_sin / tod_cos; without it they are not narrated. */
  todH?: number | null | undefined;
  /** Absolute level behind a room-relative feature (`db_room_mean_5m` -> `db_mean_5m`), quoted in its display. */
  absolute?: number | null | undefined;
}

// ---------- Constants ----------

/** valence/explain/describe.py:16 */
export const WINDOWS: Readonly<Record<string, string>> = {
  "1m": "minute",
  "5m": "5 minutes",
  "15m": "15 minutes",
};

/**
 * valence/explain/describe.py:17. Same backtracking semantics in JS: the lazy stat grows until the optional
 * window suffix and the end anchor both match (`fist_since_s` -> stat "since_s", `db_mean_15m` -> "mean"/"15m").
 * Python's `$` (no MULTILINE) also matches just before one trailing "\n", so `db_mean_5m\n` parses as
 * db/mean/5m there; JS's `$` only matches at the very end, hence the `(?=\n?$)` lookahead.
 */
const NAME = /^([a-z]+)_([a-z0-9_]+?)(?:_(1m|5m|15m))?(?=\n?$)/;

// ---------- Python integer helpers ----------

/** Python `round(x)` (no ndigits) as an exact integer; throws on NaN / inf like Python. */
function roundInt(x: number): bigint {
  return BigInt(pyIntRound(x)); // pyIntRound returns an integral double, so BigInt() is exact
}

/** Python `int(x)` for a float: truncation toward zero, exact; throws on NaN / inf like Python. */
function truncInt(x: number): bigint {
  if (!Number.isFinite(x)) throw new RangeError(`cannot convert float ${x} to integer`);
  return BigInt(Math.trunc(x));
}

function absInt(n: bigint): bigint {
  return n < 0n ? -n : n;
}

/** Python `a % b` for integral doubles: the result takes the sign of b (JS % is exact fmod). */
function pyMod(a: number, b: number): number {
  const r = a % b;
  return r !== 0 && r < 0 !== b < 0 ? r + b : r + 0;
}

/** Python `max(x, lo)` then `min(.., hi)`: builtins return the first argument unless the other is strictly larger/smaller. */
function clamp01(x: number): number {
  const a = 0.0 > x ? 0.0 : x;
  return 1.0 < a ? 1.0 : a;
}

// ---------- Helpers (describe.py:29-49) ----------

/** valence/explain/describe.py:29 */
function level(value: number, high: number, elevated: number, low: number): Level {
  if (value >= high) return "high";
  if (value >= elevated) return "elevated";
  if (value < low) return "low";
  return "normal";
}

/** valence/explain/describe.py:39 */
function span(win: string | undefined): string {
  return win && win !== "1m" ? `the last ${WINDOWS[win] ?? ""}` : "the last minute";
}

/** valence/explain/describe.py:43: `f"{round(100 * v)}%"`. */
function pct(v: number): string {
  return `${roundInt(100 * v)}%`;
}

/** valence/explain/describe.py:47: "HH:MM" from fractional hours. */
function clock(todH: number): string {
  if (!Number.isFinite(todH)) throw new RangeError(`cannot convert float ${todH} to integer`); // Python: int(nan)
  const whole = Math.trunc(todH);
  const h = pyMod(whole, 24);
  const m = pyMod(pyIntRound((todH - whole) * 60), 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

type Make = (display: string, unit: string, lvl: Level) => Description;

/**
 * Room level relative to the venue baseline; the display leads with the absolute level when known
 * (valence/explain/describe.py:52).
 */
function roomRelative(
  ch: string,
  stat: string,
  sp: string,
  value: number,
  absolute: number | null | undefined,
  d: Make,
): Description {
  const known = absolute !== null && absolute !== undefined && !isnan(absolute);
  if (ch === "db") {
    const whatMap: Readonly<Record<string, string>> = {
      now: "current noise",
      mean: "average noise",
      max: "peak noise",
      ewm: "recent noise",
    };
    const what = whatMap[stat];
    if (what === undefined) throw new Error(`KeyError: ${JSON.stringify(stat)}`); // Python indexes the dict
    const tail = stat === "now" ? "" : ` over ${sp}`;
    const delta = roundInt(value);
    const rel =
      delta === 0n
        ? "at this room's baseline"
        : `${absInt(delta)} dB ${delta > 0n ? "above" : "below"} this room's baseline`;
    const text = known
      ? `${what} of ${roundInt(absolute)} dB${tail}, ${rel}`
      : `${what} ${rel}${tail}`;
    return d(text, "dB", level(value, 8, 4, -3));
  }
  const rel =
    value >= 0.5
      ? "brighter than this room's usual light"
      : value <= -0.5
        ? "dimmer than this room's usual light"
        : "about this room's usual light";
  const text = known ? `${roundInt(absolute)} lux, ${rel}` : `light ${rel}`;
  return d(text, "lux", level(value, 1.0, 0.5, -0.5));
}

// ---------- describe ----------

/**
 * Narrator description of one feature value (valence/explain/describe.py:70 describe). Returns null for
 * features that are never narrated (fill_ratio), for a missing value (null, undefined or NaN), and for
 * tod_sin / tod_cos without a clock time.
 *
 * Throws where Python raises: a NaN `todH` with a finite tod value (int(nan)), an infinite value reaching
 * round()/int(), or a room-relative dB stat outside now/mean/max/ewm (KeyError).
 */
export function describe(
  feature: string,
  value: number | null | undefined,
  opts: DescribeOptions = {},
): Description | null {
  if (feature === "fill_ratio" || value === null || value === undefined || isnan(value))
    return null;
  const todH = opts.todH;
  const absolute = opts.absolute;
  const layer = layerOf(feature);
  const scope = scopeOf(feature);
  const d: Make = (display, unit, lvl) => ({ display, unit, level: lvl, scope, layer });

  if (feature === "hours_since_break") {
    return d(`${pyFixed(value, 1)} hours since the last break`, "h", level(value, 3.5, 2.0, 0.75));
  }
  if (feature === "tod_sin" || feature === "tod_cos") {
    if (todH === null || todH === undefined) return null;
    const late = todH >= 22 || todH < 6;
    return d(`time of day (${clock(todH)})`, "", late ? "high" : "normal");
  }
  if (feature === "n_people") {
    const n = truncInt(value);
    return d(
      `${n} ${n === 1n ? "person" : "people"} present`,
      "people",
      n >= 2n ? "elevated" : "normal",
    );
  }
  if (feature === "withdraw_now") {
    const truthy = value !== 0; // Python truthiness: 0.0 and -0.0 are falsy (NaN returned above)
    return d(
      truthy ? "withdrawn right now" : "not withdrawn right now",
      "",
      truthy ? "high" : "low",
    );
  }

  const m = NAME.exec(feature);
  if (!m) return d(`${feature} = ${pyG(value)}`, "", "normal");
  const ch = m[1] as string;
  const stat = m[2] as string;
  const win = m[3];
  const sp = span(win);

  if ((ch === "db" || ch === "lux") && stat.startsWith("room_")) {
    return roomRelative(ch, stat.slice("room_".length), sp, value, absolute, d);
  }

  if (ch === "fist" || ch === "laugh" || ch === "compl") {
    const noun = { fist: "clenched-fist event", laugh: "laughter event", compl: "compliment" }[ch];
    const where = ch !== "fist" ? " in the room" : "";
    if (stat === "cnt") {
      const n = roundInt(value);
      // Python: n * {"1m": 5, "5m": 1, "15m": 1 / 3}.get(win or "5m", 1); n came from a double, so Number(n) is exact
      const mult =
        ({ "1m": 5, "5m": 1, "15m": 1 / 3 } as Readonly<Record<string, number>>)[win || "5m"] ?? 1;
      const per5 = Number(n) * mult;
      const text =
        n === 0n
          ? `no ${noun}s${where} in ${sp}`
          : `${n} ${noun}${n !== 1n ? "s" : ""}${where} in ${sp}`;
      return d(text, "events", level(per5, ch === "fist" ? 3 : 2, 1, 0.01));
    }
    if (stat.startsWith("since")) {
      const s = roundInt(value);
      const text = s >= 900n ? `no ${noun} in the last 15 minutes` : `last ${noun} ${s} s ago`;
      return d(text, "s", s >= 900n ? "low" : s < 60n ? "high" : "elevated");
    }
  }

  if (ch === "db") {
    if (stat === "now" || stat === "mean" || stat === "max" || stat === "ewm" || stat === "p90") {
      const what = {
        now: "current noise",
        mean: "average noise",
        max: "peak noise",
        ewm: "recent noise",
        p90: "noise peaks",
      }[stat];
      const tail = stat === "now" ? "" : ` over ${sp}`;
      return d(`${what} of ${roundInt(value)} dB${tail}`, "dB", level(value, 65, 58, 45));
    }
    if (stat === "d1m15m") {
      const sign = value >= 0 ? "above" : "below";
      return d(
        `noise ${absInt(roundInt(value))} dB ${sign} its 15-minute average`,
        "dB",
        level(value, 6, 3, -3),
      );
    }
    if (stat === "slope") {
      const trend = value >= 0 ? "rising" : "falling";
      return d(
        `noise ${trend} ${pyFixed(Math.abs(value), 1)} dB per minute over ${sp}`,
        "dB/min",
        level(value, 1.0, 0.4, -0.4),
      );
    }
    if (stat.startsWith("rel")) {
      const sign = value >= 0 ? "above" : "below";
      return d(
        `noise ${absInt(roundInt(value))} dB ${sign} its 15-minute median`,
        "dB",
        level(value, 6, 3, -3),
      );
    }
    if (stat === "std") {
      return d(`noise varying by ${pyFixed(value, 1)} dB over ${sp}`, "dB", level(value, 6, 3, 1));
    }
  }

  if (ch === "lux") {
    if (stat === "now" || stat === "mean" || stat === "ewm" || stat === "max") {
      const lvl = level(value, 450, 250, 200);
      const word = {
        low: "dim light",
        normal: "normal light",
        elevated: "normal light",
        high: "bright light",
      }[lvl];
      return d(`${word} (${roundInt(value)} lux)`, "lux", lvl);
    }
    if (stat === "d1m15m" || stat.startsWith("rel")) {
      const sign = value >= 0 ? "brighter" : "darker";
      const ref = stat.startsWith("rel") ? "its 15-minute median" : "its 15-minute average";
      return d(
        `light ${absInt(roundInt(value))} lux ${sign} than ${ref}`,
        "lux",
        level(value, 100, 40, -40),
      );
    }
    if (stat === "slope" || stat === "std") {
      return d(
        `light changing (${pyFixed(value, 1, { sign: true })} lux/min)`,
        "lux/min",
        level(Math.abs(value), 20, 8, 0.5),
      );
    }
  }

  if (ch === "slump" || ch === "hih" || ch === "withdraw") {
    const what = { slump: "slumped", hih: "head in hands", withdraw: "withdrawn" }[ch];
    if (stat === "now" || stat === "mean" || stat === "frac" || stat === "ewm" || stat === "max") {
      const v = clamp01(value);
      const text =
        stat !== "now" ? `${what} ${pct(v)} of ${sp}` : `${what} ${pct(v)} of the last 5 seconds`;
      return d(text, "%", level(v, 0.6, 0.35, 0.1));
    }
    if (stat === "slope" || stat === "d1m15m" || stat === "std" || stat.startsWith("rel")) {
      return d(
        `${what} share changing (${pyFixed(value, 2, { sign: true })})`,
        "",
        level(Math.abs(value), 0.3, 0.15, 0.02),
      );
    }
  }

  if (ch === "motion") {
    if (stat === "now" || stat === "mean" || stat === "ewm" || stat === "max") {
      const lvl = level(value, 0.1, 0.05, 0.015); // live: still ~0.014, typing ~0.03
      const word = {
        low: "nearly still",
        normal: "moving normally",
        elevated: "restless",
        high: "very restless",
      }[lvl];
      return d(`${word} (motion ${pyFixed(value, 3)})`, "sw/s", lvl);
    }
    return d(
      `motion changing (${pyFixed(value, 3, { sign: true })})`,
      "sw/s",
      level(Math.abs(value), 0.05, 0.02, 0.003),
    );
  }

  if (ch === "face") {
    if (stat.startsWith("visible")) {
      return d(`face visible ${pct(clamp01(value))} of ${sp}`, "%", level(value, 0.8, 0.5, 0.4));
    }
    if (stat === "now" || stat === "mean") {
      const lvl = level(value, 0.3, 0.1, -0.2);
      const word = {
        high: "smiling expression",
        elevated: "slightly positive expression",
        normal: "neutral expression",
        low: "negative expression",
      }[lvl];
      const tail = stat === "now" ? "" : ` over ${sp}`;
      return d(`${word}${tail} (valence ${pyFixed(value, 2, { sign: true })})`, "", lvl);
    }
    if (stat === "std") {
      return d(
        `expression varying (spread ${pyFixed(value, 2)})`,
        "",
        level(value, 0.35, 0.2, 0.05),
      );
    }
  }

  return d(`${feature} = ${pyG(value)}`, "", "normal");
}
