/**
 * The one feature function: rolling window -> Rolling State Vector (valence/features/compute.py).
 *
 * Inputs are two windows of per-tick channel arrays, oldest first: the person's own window and the room
 * slot's window (room history is continuous even when the person just arrived). Missing values stay NaN.
 * The 123 features come out in FEATURE_NAMES order as a Float64Array.
 *
 * Parity: every reduction goes through num.ts (numpy's pairwise sums, two-pass nanstd, nanmedian), the
 * ewm weights come from a table of numpy's own values and the max features copy numpy's SIMD fmax reduction
 * (fmaxReduce, for the sign of a zero max), so 118 of the 123 features are bit-identical to Python.
 * tod_sin/tod_cos (math.sin/cos) and lux_room_* (math.log2) go through V8's Math functions, which round
 * differently from glibc in the last bit for a few percent of inputs; see Elementary functions. Inputs that
 * make Python raise (an empty person window, a missing channel, math.sin(inf), math.log2(0)) throw here.
 * Golden values: scripts/export_web_golden.py -> src/test/fixtures/engine/features.json.gz.
 */
import type { TickRecord } from "./contracts";
import {
  allNaN,
  isnan,
  nanToNum,
  nanmean,
  nanmedian,
  nanstd,
  npMean,
  pairwiseSum,
  tail,
} from "./num";

// ---------- Windows and channels ----------

/** Seconds per tick (valence/features/compute.py:20). */
export const TICK_S = 5.0;
/** Window lengths in ticks, in feature order (valence/features/compute.py:21). */
export const WIN = { "1m": 12, "5m": 60, "15m": 180 } as const;
export type WindowName = keyof typeof WIN;
const WIN_NAMES = Object.keys(WIN) as WindowName[];
/** Longest window; longer inputs are cut to their last MAX_TICKS ticks (valence/features/compute.py:22). */
export const MAX_TICKS = 180;

/** Numeric TickRecord fields (the ones a channel array can be built from). */
export type NumericField = {
  [K in keyof TickRecord]: TickRecord[K] extends number ? K : never;
}[keyof TickRecord];

/** Continuous room channels: feature prefix -> TickRecord field (valence/features/compute.py:24). */
export const ROOM_CONT = { db: "ambient_db", lux: "light_lux" } as const satisfies Record<
  string,
  NumericField
>;
/** Continuous person channels (valence/features/compute.py:25). */
export const PERSON_CONT = {
  motion: "motion_energy",
  slump: "posture_slump",
  hih: "head_in_hands_frac",
} as const satisfies Record<string, NumericField>;
/** Event counters: prefix -> [scope, field] (valence/features/compute.py:26). */
export const EVENTS = {
  fist: ["person", "clenched_fists_count"],
  laugh: ["room", "laughter_count"],
  compl: ["room", "compliments_count"],
} as const satisfies Record<string, readonly ["person" | "room", NumericField]>;
/** Fields read from the person's rows (valence/features/compute.py:28). */
export const PERSON_FIELDS = [
  "motion_energy",
  "posture_slump",
  "head_in_hands_frac",
  "clenched_fists_count",
  "social_withdrawal_flag",
  "n_people",
  "hours_since_break",
  "tod_h",
  "face_valence",
  "face_visible",
] as const satisfies readonly NumericField[];
/** Fields read from the room slot's rows (valence/features/compute.py:30). */
export const ROOM_FIELDS = [
  "ambient_db",
  "light_lux",
  "laughter_count",
  "compliments_count",
  "room_db_base",
  "room_lux_base",
] as const satisfies readonly NumericField[];

// ---------- Feature names ----------

/** The 17 names of one continuous channel (valence/features/compute.py:33 _cont_names). */
function contNames(ch: string): string[] {
  const names = [`${ch}_now`];
  for (const w of WIN_NAMES)
    names.push(`${ch}_mean_${w}`, `${ch}_std_${w}`, `${ch}_slope_${w}`, `${ch}_max_${w}`);
  return [...names, `${ch}_ewm_1m`, `${ch}_ewm_5m`, `${ch}_d1m15m`, `${ch}_rel_15m`];
}

/** valence/features/compute.py:40 */
export const CONTINUOUS: readonly string[] = [
  ...Object.keys(ROOM_CONT),
  ...Object.keys(PERSON_CONT),
];
/** valence/features/compute.py:46 */
export const FACE_FEATURES: readonly string[] = [
  "face_now",
  "face_visible_now",
  "face_mean_1m",
  "face_mean_5m",
  "face_mean_15m",
  "face_std_5m",
  "face_visible_frac_5m",
];
/**
 * Room level relative to the venue baseline (TickRecord.room_db_base / room_lux_base): dB above the
 * baseline, light as log2(lux / baseline lux). A loud hall is then "this room's normal", not "loud"
 * (valence/features/compute.py:51).
 */
export const ROOM_REL: readonly string[] = [
  "db_room_now",
  "db_room_mean_1m",
  "db_room_mean_5m",
  "db_room_mean_15m",
  "db_room_max_5m",
  "db_room_ewm_5m",
  "lux_room_now",
  "lux_room_mean_5m",
  "lux_room_mean_15m",
];

/** All 123 features in vector order (valence/features/compute.py:41-53). */
export const FEATURE_NAMES: readonly string[] = (() => {
  const names = CONTINUOUS.flatMap(contNames);
  for (const e of Object.keys(EVENTS))
    names.push(`${e}_cnt_now`, `${e}_cnt_1m`, `${e}_cnt_5m`, `${e}_cnt_15m`, `${e}_since_s`);
  names.push(
    "withdraw_now",
    "withdraw_frac_5m",
    "n_people",
    "hours_since_break",
    "tod_sin",
    "tod_cos",
    "fill_ratio",
  );
  return [...names, ...FACE_FEATURES, ...ROOM_REL];
})();
/** Number of features (123). */
export const N_FEATURES = FEATURE_NAMES.length;
/**
 * A dict-like lookup table: no prototype, so `table["constructor"]` or `table["toString"]` is undefined
 * (a Python KeyError) instead of an inherited Object method.
 */
function dict<V>(entries: Iterable<readonly [string, V]>): Record<string, V> {
  return Object.assign(Object.create(null) as Record<string, V>, Object.fromEntries(entries));
}

/** Feature name -> index in the vector (valence/features/compute.py:54). */
export const FEATURE_INDEX: Readonly<Record<string, number>> = dict(
  FEATURE_NAMES.map((n, i) => [n, i] as const),
);

/** Index of a feature; throws on an unknown name (Python: KeyError). */
export function featureIndex(name: string): number {
  const i = FEATURE_INDEX[name];
  if (i === undefined) throw new Error(`unknown feature ${JSON.stringify(name)}`);
  return i;
}

// ---------- Feature sets ----------

/**
 * Absolute room levels: computed and quoted in explanations, but not used by the served models (venue
 * shift) (valence/features/compute.py:57).
 */
export const ABS_ROOM: readonly string[] = FEATURE_NAMES.filter((n) => {
  const parts = n.split("_");
  return (
    ["db", "lux"].includes(parts[0] ?? "") && ["now", "mean", "max", "ewm"].includes(parts[1] ?? "")
  );
});
/** valence/features/compute.py:59 */
export const SNAPSHOT: readonly string[] = [
  "db_room_now",
  "lux_room_now",
  "motion_now",
  "slump_now",
  "hih_now",
  "fist_cnt_now",
  "laugh_cnt_now",
  "compl_cnt_now",
  "withdraw_now",
  "n_people",
  "hours_since_break",
  "tod_sin",
  "tod_cos",
];
/**
 * 1-minute dB/lux slopes and spreads are dominated by transients live (typing starting, doors, webcam
 * auto-exposure): computed and shown, but not used by the served context models
 * (valence/features/compute.py:63).
 */
export const NOISY_LIVE: readonly string[] = [
  "db_slope_1m",
  "db_std_1m",
  "lux_slope_1m",
  "lux_std_1m",
];
/** The pre-baseline design (valence/features/compute.py:64 _ABS_M2). */
const ABS_M2 = FEATURE_NAMES.filter((n) => !NOISY_LIVE.includes(n) && !ROOM_REL.includes(n));
/** valence/features/compute.py:65 _M2 */
const M2 = [...ABS_M2.filter((n) => !ABS_ROOM.includes(n)), ...ROOM_REL];
/** The served context features without the face block (valence/features/compute.py:66). */
export const CONTEXT: readonly string[] = M2.filter((n) => !FACE_FEATURES.includes(n));

/** Feature sets of the model ladder, docs/PLAN.md "Model ladder" (valence/features/compute.py:69). */
export const FEATURE_SETS: Readonly<Record<string, readonly string[]>> = dict<readonly string[]>([
  ["m0", ["face_now", "face_visible_now"]],
  ["m0t", FACE_FEATURES],
  ["m1", SNAPSHOT],
  ["m2_noface", CONTEXT],
  ["m2_abs", ABS_M2], // absolute dB/lux instead of room-relative (ablation)
  ["m2_notime", M2.filter((n) => !["hours_since_break", "tod_sin", "tod_cos"].includes(n))],
  ["m2", M2],
]);

/** A named feature set; throws on an unknown one (Python: KeyError). */
export function featureSet(name: string): readonly string[] {
  const set = FEATURE_SETS[name];
  if (set === undefined) throw new Error(`unknown feature set ${JSON.stringify(name)}`);
  return set;
}

/** `x[[FEATURE_INDEX[n] for n in names]]`: the columns a model reads, in the model's order. */
export function selectFeatures(x: ArrayLike<number>, names: readonly string[]): Float64Array {
  return Float64Array.from(names, (n) => x[featureIndex(n)] as number);
}

// ---------- Channel arrays ----------

/** Per-tick channel arrays by field name, oldest first (Python: dict[str, np.ndarray]). */
export type Channels = Readonly<Record<string, ArrayLike<number> | undefined>>;

/**
 * `{f: np.array([getattr(r, f) for r in records], dtype=float)}` (valence/features/compute.py:80
 * arrays_from_records).
 */
export function arraysFromRecords<F extends NumericField>(
  records: readonly TickRecord[],
  fields: readonly F[],
): Record<F, Float64Array> {
  const out = {} as Record<F, Float64Array>;
  // dtype=float turns None into nan; a typed array would turn null into 0
  for (const f of fields)
    out[f] = Float64Array.from(records, (r) => {
      const v = r[f] as number | null | undefined;
      return v === null || v === undefined ? NaN : Number(v);
    });
  return out;
}

/** `d[key]`, throwing like Python's KeyError when the channel is missing. */
function need(d: Readonly<Record<string, Float64Array | undefined>>, key: string): Float64Array {
  const v = d[key];
  if (v === undefined) throw new Error(`missing channel ${JSON.stringify(key)}`);
  return v;
}

/** `x[-1]`; throws on an empty array like Python's IndexError. */
function last(x: Float64Array): number {
  if (x.length === 0) throw new RangeError("index out of range: empty window"); // Python: IndexError
  return x[x.length - 1] as number;
}

// ---------- Elementary functions ----------
// Python evaluates the ewm weights with np.power and the time-of-day / light features with glibc's
// math.sin, math.cos and math.log2. V8's Math functions round differently in the last bit for a few
// percent of inputs (Math.pow(0.5, q) differs from numpy in 67 of the 360 ewm weights), so the weights
// come from a table and only sin/cos/log2 remain last-bit faithful rather than bit-exact.

/**
 * `np.power(0.5, np.arange(180) / hl)` for hl = 12 and 60, exactly as numpy 2.5.4 computes it on the
 * golden reference machine (x86-64 AVX-512, SIMD power). 19 of the 360 entries are 1 ulp away from the
 * correctly rounded value (what glibc's pow and numpy builds without AVX-512 return); the table keeps
 * numpy's so the ewm features are bit-identical to the fixtures. Regenerate with:
 * `python -c "import numpy as np; print([repr(float(v)) for v in np.power(0.5, np.arange(180.) / 12)])"`.
 */
const EWM_WEIGHTS: Readonly<Record<number, readonly number[]>> = {
  12: [
    1.0, 0.9438743126816935, 0.8908987181403393, 0.8408964152537145, 0.7937005259840998,
    0.7491535384383408, 0.7071067811865476, 0.6674199270850172, 0.6299605249474366,
    0.5946035575013605, 0.5612310241546865, 0.5297315471796477, 0.5, 0.47193715634084676,
    0.44544935907016964, 0.42044820762685725, 0.3968502629920499, 0.3745767692191703,
    0.3535533905932738, 0.3337099635425086, 0.3149802624737183, 0.29730177875068026,
    0.28061551207734325, 0.26486577358982377, 0.25, 0.23596857817042335, 0.22272467953508485,
    0.21022410381342863, 0.19842513149602492, 0.18728838460958522, 0.1767766952966369,
    0.16685498177125427, 0.15749013123685915, 0.14865088937534013, 0.1403077560386716,
    0.1324328867949119, 0.125, 0.11798428908521168, 0.11136233976754242, 0.10511205190671431,
    0.09921256574801246, 0.09364419230479261, 0.08838834764831845, 0.08342749088562713,
    0.07874506561842957, 0.07432544468767006, 0.0701538780193358, 0.06621644339745596, 0.0625,
    0.05899214454260585, 0.0556811698837712, 0.052556025953357156, 0.049606282874006244,
    0.046822096152396284, 0.04419417382415922, 0.04171374544281358, 0.03937253280921478,
    0.03716272234383503, 0.035076939009667914, 0.03310822169872797, 0.03125, 0.029496072271302926,
    0.0278405849418856, 0.026278012976678578, 0.024803141437003122, 0.023411048076198142,
    0.02209708691207961, 0.02085687272140679, 0.01968626640460739, 0.018581361171917516,
    0.017538469504833957, 0.016554110849363986, 0.015625, 0.014748036135651463, 0.0139202924709428,
    0.013139006488339289, 0.012401570718501561, 0.011705524038099071, 0.011048543456039806,
    0.010428436360703395, 0.009843133202303695, 0.009290680585958758, 0.008769234752416978,
    0.008277055424681993, 0.0078125, 0.0073740180678257316, 0.0069601462354714,
    0.0065695032441696445, 0.0062007853592507805, 0.0058527620190495355, 0.005524271728019903,
    0.005214218180351698, 0.0049215666011518475, 0.004645340292979379, 0.004384617376208489,
    0.0041385277123409964, 0.00390625, 0.0036870090339128636, 0.0034800731177357016,
    0.0032847516220848223, 0.0031003926796253885, 0.00292638100952477, 0.0027621358640099515,
    0.0026071090901758475, 0.002460783300575925, 0.0023226701464896895, 0.0021923086881042433,
    0.0020692638561704995, 0.001953125, 0.0018435045169564318, 0.0017400365588678508,
    0.0016423758110424111, 0.0015501963398126943, 0.001463190504762385, 0.0013810679320049757,
    0.0013035545450879238, 0.0012303916502879625, 0.0011613350732448448, 0.0010961543440521217,
    0.0010346319280852498, 0.0009765625, 0.0009217522584782159, 0.0008700182794339254,
    0.0008211879055212056, 0.0007750981699063471, 0.0007315952523811925, 0.0006905339660024879,
    0.0006517772725439619, 0.0006151958251439813, 0.0005806675366224224, 0.0005480771720260608,
    0.0005173159640426249, 0.00048828125, 0.00046087612923910795, 0.0004350091397169627,
    0.0004105939527606028, 0.00038754908495317356, 0.00036579762619059624, 0.00034526698300124393,
    0.00032588863627198094, 0.00030759791257199063, 0.0002903337683112112, 0.0002740385860130304,
    0.00025865798202131244, 0.000244140625, 0.00023043806461955398, 0.00021750456985848135,
    0.0002052969763803014, 0.00019377454247658678, 0.00018289881309529812, 0.00017263349150062197,
    0.00016294431813599047, 0.00015379895628599532, 0.0001451668841556056, 0.0001370192930065152,
    0.00012932899101065622, 0.0001220703125, 0.00011521903230977699, 0.00010875228492924067,
    0.0001026484881901507, 9.688727123829339e-5, 9.144940654764906e-5, 8.631674575031098e-5,
    8.147215906799523e-5, 7.689947814299766e-5, 7.25834420778028e-5, 6.85096465032576e-5,
    6.466449550532811e-5, 6.103515625e-5, 5.7609516154888494e-5, 5.437614246462034e-5,
    5.132424409507535e-5, 4.8443635619146695e-5, 4.572470327382453e-5, 4.315837287515549e-5,
    4.073607953399762e-5, 3.844973907149883e-5, 3.62917210389014e-5, 3.42548232516288e-5,
    3.2332247752664055e-5,
  ],
  60: [
    1.0, 0.9885140203528962, 0.9771599684342459, 0.9659363289248456, 0.9548416039104165,
    0.9438743126816935, 0.9330329915368074, 0.9223161935859392, 0.9117224885582168,
    0.9012504626108302, 0.8908987181403393, 0.8806658735961485, 0.8705505632961241,
    0.86055143724433, 0.8506671609508557, 0.8408964152537145, 0.8312378961427878,
    0.8216903145857902, 0.8122523963562355, 0.8029228818633765, 0.7937005259840998,
    0.7845840978967508, 0.7755723809168673, 0.7666641723348002, 0.757858283255199,
    0.7491535384383408, 0.7405487761432821, 0.7320428479728127, 0.7236346187201891,
    0.7153229662176291, 0.7071067811865476, 0.6989849670895096, 0.690956439983888,
    0.6830201283771977, 0.675174973084095, 0.6674199270850172, 0.6597539553864471,
    0.6521760348827821, 0.6446851542197896, 0.6372803136596311, 0.6299605249474366,
    0.6227248111794115, 0.6155722066724582, 0.6085017568352955, 0.6015125180410583,
    0.5946035575013605, 0.5877739531418044, 0.5810227934789198, 0.5743491774985174,
    0.5677522145354387, 0.5612310241546865, 0.5547847360339225, 0.548412489847313,
    0.5421134351507092, 0.5358867312681466, 0.5297315471796477, 0.5236470614103134,
    0.5176324619206888, 0.5116869459983875, 0.5058097201509613, 0.5, 0.4942570101764481,
    0.48857998421712295, 0.48296816446242274, 0.4774208019552083, 0.47193715634084676,
    0.46651649576840365, 0.4611580967929696, 0.4558612442791084, 0.45062523130541515,
    0.44544935907016964, 0.4403329367980742, 0.43527528164806206, 0.43027571862216507,
    0.42533358047542785, 0.42044820762685725, 0.4156189480713939, 0.4108451572928951,
    0.40612619817811774, 0.40146144093168823, 0.3968502629920499, 0.3922920489483753,
    0.38778619045843365, 0.38333208616740017, 0.37892914162759955, 0.3745767692191703,
    0.37027438807164104, 0.36602142398640636, 0.36181730936009454, 0.35766148310881457,
    0.3535533905932738, 0.3494924835447549, 0.345478219991944, 0.34151006418859886,
    0.3375874865420475, 0.3337099635425086, 0.32987697769322355, 0.32608801744139104,
    0.3223425771098948, 0.31864015682981556, 0.3149802624737183, 0.31136240558970574,
    0.3077861033362291, 0.30425087841764775, 0.30075625902052916, 0.29730177875068026,
    0.2938869765709022, 0.29051139673945986, 0.2871745887492587, 0.28387610726771934,
    0.28061551207734325, 0.2773923680169612, 0.2742062449236565, 0.2710567175753546,
    0.2679433656340733, 0.26486577358982377, 0.2618235307051567, 0.2588162309603444,
    0.25584347299919374, 0.25290486007548063, 0.25, 0.24712850508822404, 0.2442899921085615,
    0.24148408223121143, 0.2387104009776041, 0.23596857817042335, 0.23325824788420182,
    0.2305790483964848, 0.2279306221395542, 0.22531261565270758, 0.22272467953508485,
    0.22016646839903714, 0.217637640824031, 0.2151378593110825, 0.21266679023771393,
    0.21022410381342863, 0.20780947403569694, 0.20542257864644758, 0.2030630990890589,
    0.2007307204658441, 0.19842513149602492, 0.19614602447418766, 0.19389309522921683,
    0.19166604308370008, 0.18946457081379978, 0.18728838460958522, 0.18513719403582055,
    0.18301071199320315, 0.18090865468004724, 0.17883074155440729, 0.1767766952966369,
    0.17474624177237744, 0.17273910999597203, 0.17075503209429946, 0.16879374327102373,
    0.16685498177125427, 0.16493848884661177, 0.16304400872069552, 0.1611712885549474,
    0.15932007841490778, 0.15749013123685915, 0.1556812027948529, 0.1538930516681145,
    0.15212543920882385, 0.15037812951026458, 0.14865088937534013, 0.1469434882854511,
    0.14525569836972996, 0.1435872943746294, 0.14193805363385964, 0.1403077560386716,
    0.1386961840084806, 0.13710312246182824, 0.1355283587876773, 0.13397168281703664,
    0.1324328867949119, 0.13091176535257837, 0.12940811548017217, 0.12792173649959684,
    0.12645243003774032,
  ],
};

/** `np.power(0.5, age / halfLife)`: the table for the compute() windows, Math.pow otherwise. */
function ewmWeight(age: number, halfLife: number): number {
  return EWM_WEIGHTS[halfLife]?.[age] ?? Math.pow(0.5, age / halfLife);
}

/**
 * CPython's `math_1` wrapper (Modules/mathmodule.c) around a libm call, with can_overflow = 0: a NaN result
 * from a non-NaN input, or an infinite result from a finite input, raises ValueError("math domain error").
 * That is how `math.sin(inf)` and `math.log2(0.0)` fail in compute(); the port throws at the same inputs
 * instead of returning NaN / -Infinity.
 */
function math1(fn: (x: number) => number, x: number): number {
  const r = fn(x);
  if (isnan(r) ? !isnan(x) : !Number.isFinite(r) && Number.isFinite(x))
    throw new RangeError("math domain error");
  return r;
}

/** `math.sin` / `math.cos` of the time-of-day angle. */
const sin = (x: number): number => math1(Math.sin, x);
const cos = (x: number): number => math1(Math.cos, x);
/** `math.log2` for the room-relative light features. */
const log2 = (x: number): number => math1(Math.log2, x);

// ---------- Window statistics ----------

/**
 * valence/features/compute.py:84 _nanstat: NaN for an empty or all-NaN window (numpy would warn), else
 * the reduction.
 */
function nanstat(fn: (a: Float64Array) => number, x: Float64Array): number {
  if (x.length === 0 || allNaN(x)) return NaN;
  return fn(x);
}

/** x86 `maxpd` lane: `a > b ? a : b`, so the second operand wins on +0/-0 ties and on NaN. */
const vmax = (a: number, b: number): number => (a > b ? a : b);
/** numpy `npyv_maxp_f64` lane (`_mm512_mask_max_pd(a, b == b, a, b)`): keeps `a` where `b` is NaN. */
const vmaxp = (a: number, b: number): number => (isnan(b) ? a : a > b ? a : b);
/** glibc `fmax` (math/s_fmax_template.c), numpy's scalar tail: the first operand wins ties. */
const fmax = (x: number, y: number): number => (x >= y ? x : x < y ? y : isnan(y) ? x : y);
const LANES = 8;

/**
 * `np.nanmax(x)` for a non-empty float64 window, i.e. `np.fmax.reduce(x)`, including the sign of a zero
 * maximum. num.ts nanmax always prefers +0.0; numpy's answer depends on where each zero lands in its SIMD
 * reduction (numpy/_core/src/umath/loops_minmax.dispatch.c.src simd_reduce_c_fmax_f64, AVX-512, 8 lanes):
 * the accumulator starts as x[0] in every lane, takes 8 vectors per step and then single vectors, folds
 * the lanes with GCC's _mm512_reduce_max_pd order (upper half against lower half, NaN lanes as -inf), and
 * finishes the last < 8 values with fmax. Checked against np.fmax.reduce on 60k random arrays of zeros,
 * NaNs, infinities and normals, lengths 1..261. Like the ewm table, this is the AVX-512 reference
 * machine's answer: an AVX2 build (4 lanes) can return the other zero for the same window.
 */
function fmaxReduce(x: Float64Array): number {
  const n = x.length;
  if (n === 0) return NaN; // numpy raises; nanstat never calls this on an empty window
  const acc = new Float64Array(LANES).fill(x[0] as number);
  let i = 1;
  for (; n - i >= 8 * LANES; i += 8 * LANES) {
    for (let l = 0; l < LANES; l++) {
      const v = (k: number): number => x[i + k * LANES + l] as number;
      const r0123 = vmaxp(vmaxp(v(0), v(1)), vmaxp(v(2), v(3)));
      const r4567 = vmaxp(vmaxp(v(4), v(5)), vmaxp(v(6), v(7)));
      acc[l] = vmaxp(acc[l] as number, vmaxp(r0123, r4567));
    }
  }
  for (; n - i >= LANES; i += LANES)
    for (let l = 0; l < LANES; l++) acc[l] = vmaxp(acc[l] as number, x[i + l] as number);
  let r: number;
  if (allNaN(acc)) r = acc[0] as number;
  else {
    const a = Array.from(acc, (v) => (isnan(v) ? -Infinity : v));
    const h4 = [0, 1, 2, 3].map((l) => vmax(a[l + 4] as number, a[l] as number));
    const h2 = [0, 1].map((l) => vmax(h4[l + 2] as number, h4[l] as number));
    r = vmax(h2[0] as number, h2[1] as number);
  }
  for (; i < n; i++) r = fmax(r, x[i] as number);
  return r;
}

/**
 * Least-squares slope per minute over the finite ticks; NaN below 3 of them
 * (valence/features/compute.py:92 _slope_per_min). Tick i sits at minute i * 5 / 60.
 */
function slopePerMin(x: Float64Array): number {
  const tl: number[] = [];
  const yl: number[] = [];
  const step = TICK_S / 60.0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i] as number;
    if (!isnan(v)) {
      tl.push(i * step);
      yl.push(v);
    }
  }
  if (tl.length < 3) return NaN;
  const t = Float64Array.from(tl);
  const tm = npMean(t);
  for (let i = 0; i < t.length; i++) t[i] = (t[i] as number) - tm;
  const tt = t.map((v) => v * v);
  const denom = pairwiseSum(tt);
  if (denom === 0) return NaN;
  const ym = npMean(yl);
  const ty = t.map((v, i) => v * ((yl[i] as number) - ym));
  return pairwiseSum(ty) / denom;
}

/**
 * Exponentially weighted mean over the finite ticks, weight 0.5 ** (age / half_life) with age 0 for
 * the newest tick (valence/features/compute.py:103 _ewm).
 */
function ewm(x: Float64Array, halfLifeTicks: number): number {
  const w: number[] = [];
  const wx: number[] = [];
  for (let i = 0; i < x.length; i++) {
    const v = x[i] as number;
    if (isnan(v)) continue;
    const wi = ewmWeight(x.length - 1 - i, halfLifeTicks);
    w.push(wi);
    wx.push(wi * v);
  }
  if (w.length === 0) return NaN;
  return pairwiseSum(wx) / pairwiseSum(w);
}

/**
 * The 17 statistics of one continuous channel in contNames order (valence/features/compute.py:112
 * _continuous): now, then mean/std/slope/max per window, ewm 1m/5m, mean 1m - mean 15m, now - median 15m.
 */
function continuous(x: Float64Array): number[] {
  const now = x.length ? last(x) : NaN;
  const out = [now];
  for (const w of WIN_NAMES) {
    const seg = tail(x, WIN[w]);
    out.push(nanstat(nanmean, seg), nanstat(nanstd, seg), slopePerMin(seg), nanstat(fmaxReduce, seg));
  }
  const m1 = out[1] as number;
  const m15 = out[9] as number;
  const med15 = nanstat(nanmedian, tail(x, WIN["15m"]));
  out.push(ewm(x, WIN["1m"]), ewm(x, WIN["5m"]), m1 - m15, now - med15);
  return out;
}

/** Last non-NaN value; NaN for a missing channel or no finite value (valence/features/compute.py:124). */
function lastFinite(x: Float64Array | undefined): number {
  if (x === undefined) return NaN;
  for (let i = x.length - 1; i >= 0; i--) if (!isnan(x[i] as number)) return x[i] as number;
  return NaN;
}

/** The 9 ROOM_REL features from the absolute ones (valence/features/compute.py:131 _room_rel). */
function roomRel(v: ReadonlyMap<string, number>, dbBase: number, luxBase: number): number[] {
  return ROOM_REL.map((name) => {
    const absolute = v.get(name.replaceAll("_room", "")) as number;
    if (name.split("_")[0] === "db") return absolute - dbBase;
    const ok = absolute > 0 && luxBase > 0; // false for NaN
    // log2 throws where Python raises: a ratio that underflows to 0 (e.g. a baseline of inf). inf / inf is
    // NaN, which math.log2 passes through.
    return ok ? log2(absolute / luxBase) : NaN;
  });
}

/**
 * Count features of one event channel: count now, counts per window and seconds since the last event,
 * capped at 900 (valence/features/compute.py:143 _events). NaN counts read as 0.
 */
function events(raw: Float64Array): number[] {
  const x = nanToNum(raw, { nan: 0.0 });
  const now = x.length ? last(x) : 0.0;
  const counts = WIN_NAMES.map((w) => pairwiseSum(tail(x, WIN[w])));
  let hit = -1;
  for (let i = x.length - 1; i >= 0; i--) {
    if ((x[i] as number) > 0) {
      hit = i;
      break;
    }
  }
  const since = hit < 0 ? 900.0 : Math.min(900.0, (x.length - 1 - hit) * TICK_S);
  return [now, ...counts, since];
}

// ---------- Feature vector ----------

/**
 * Person/room channel arrays (oldest first, any length; only the last 180 ticks are used) -> the 123
 * features in FEATURE_NAMES order (valence/features/compute.py:152 compute).
 *
 * The person dict needs every PERSON_FIELDS channel and at least one tick (Python raises IndexError on an
 * empty person window); the room dict needs ambient_db, light_lux, laughter_count and compliments_count
 * and may omit room_db_base / room_lux_base (then the room-relative dB/lux features are NaN).
 */
export function compute(person: Channels, room: Channels): Float64Array {
  const p: Record<string, Float64Array> = {};
  for (const [k, v] of Object.entries(person)) if (v !== undefined) p[k] = tail(v, MAX_TICKS);
  const r: Record<string, Float64Array> = {};
  for (const [k, v] of Object.entries(room)) if (v !== undefined) r[k] = tail(v, MAX_TICKS);

  const feats: number[] = [];
  for (const field of Object.values(ROOM_CONT)) feats.push(...continuous(need(r, field)));
  for (const field of Object.values(PERSON_CONT)) feats.push(...continuous(need(p, field)));
  for (const [scope, field] of Object.values(EVENTS))
    feats.push(...events(need(scope === "person" ? p : r, field)));

  const withdraw = nanToNum(need(p, "social_withdrawal_flag"), { nan: 0.0 });
  const tod = last(need(p, "tod_h")); // an empty person window fails here, like Python's IndexError
  const nPeople = need(p, "n_people");
  // Pushed one by one in Python's evaluation order, so a bad window fails at the same step. The angle, not
  // tod, is what math.sin sees: |tod| > ~2.9e307 overflows it to inf, which raises there.
  feats.push(last(withdraw));
  feats.push(npMean(tail(withdraw, WIN["5m"])));
  feats.push(last(nPeople));
  feats.push(last(need(p, "hours_since_break")));
  feats.push(isnan(tod) ? NaN : sin((2 * Math.PI * tod) / 24));
  feats.push(isnan(tod) ? NaN : cos((2 * Math.PI * tod) / 24));
  feats.push(nPeople.length / MAX_TICKS);

  const face = need(p, "face_valence");
  const vis = need(p, "face_visible");
  feats.push(
    last(face),
    last(vis),
    nanstat(nanmean, tail(face, WIN["1m"])),
    nanstat(nanmean, tail(face, WIN["5m"])),
    nanstat(nanmean, tail(face, WIN["15m"])),
    nanstat(nanstd, tail(face, WIN["5m"])),
    nanstat(nanmean, tail(vis, WIN["5m"])),
  );

  // dict(zip(FEATURE_NAMES, feats)): the absolute features computed so far, by name
  const v = new Map<string, number>(feats.map((f, i) => [FEATURE_NAMES[i] as string, f]));
  feats.push(...roomRel(v, lastFinite(r["room_db_base"]), lastFinite(r["room_lux_base"])));
  return Float64Array.from(feats);
}

/**
 * Live path: RollingWindow snapshots -> feature vector (valence/features/compute.py:183
 * features_from_window). Room fields also live in person rows (copied each tick); the room slot's
 * history takes precedence and the person's rows stand in only when the room slot is empty.
 */
export function featuresFromWindow(
  person: readonly TickRecord[],
  room: readonly TickRecord[],
): Float64Array {
  const p = arraysFromRecords(person, PERSON_FIELDS);
  const r = arraysFromRecords(room.length ? room : person, ROOM_FIELDS);
  return compute(p, r);
}
