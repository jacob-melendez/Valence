"""Chart style for the evaluation figures (static PNGs for slides and the frontend's results panel).

Fixed color per model everywhere (color follows the entity, never its rank), validated categorical
slots; 2px lines, 8px ringed markers, ~10% CI washes, hairline recessive grid, legend plus direct end
labels in text color, one y-axis per chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = {  # categorical slots 1-6 in fixed order (validated: light mode, adjacent pairs)
    "m2": "#2a78d6",
    "m0": "#eb6834",
    "m1": "#1baf7a",
    "m2_noface": "#eda100",
    "m0t": "#e87ba4",
    "m3": "#008300",
}
NAMES = {
    "m2": "M2 context + face history (served)",
    "m2_noface": "M2 context history, no face",
    "m1": "M1 context snapshot",
    "m0t": "M0t face history",
    "m0": "M0 face snapshot",
    "m3": "M3 GRU sequence (benchmark)",
}
ORDER = ["m2", "m2_noface", "m3", "m1", "m0t", "m0"]


def _axes(title: str, subtitle: str, size=(8.0, 4.6)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=9.5, color=TEXT_2, va="top")
    return fig, ax


def _finish(fig, ax, path: Path, legend: bool = True) -> None:
    if legend:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3, frameon=False, fontsize=8.5,
                        labelcolor=TEXT_2, handlelength=1.6)
        for line in leg.get_lines():
            line.set_linewidth(2)
    fig.subplots_adjust(top=0.82, bottom=0.27, left=0.08, right=0.80)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _line(ax, xs, ys, key, label_end: str | None = None, lo=None, hi=None):
    color = SERIES[key]
    if lo is not None:
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=NAMES[key],
            marker="o", markersize=8, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2)
    if label_end:
        ax.annotate(label_end, (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=TEXT, annotation_clip=False)


def _spread_labels(ends: dict[str, float], min_gap: float) -> dict[str, float]:
    """Nudge end-label y positions apart so they never collide."""
    items = sorted(ends.items(), key=lambda kv: kv[1])
    placed: dict[str, float] = {}
    last = -np.inf
    for k, v in items:
        y = max(v, last + min_gap)
        placed[k] = y
        last = y
    return placed


def history_chart(curves: dict[str, list], path: Path) -> None:
    fig, ax = _axes("Context history beats a face snapshot",
                    "Macro-F1 on held-out simulated sessions vs. minutes of history available (95% session-bootstrap CI)")
    ends = {}
    for key in [k for k in ORDER if k in curves]:
        pts = curves[key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        _line(ax, xs, ys, key, lo=[p[2] for p in pts], hi=[p[3] for p in pts])
        ends[key] = ys[-1]
    for key, y in _spread_labels(ends, 0.035).items():
        ax.annotate(f"{NAMES[key].split(' ')[0]}  {ends[key]:.2f}", (15, y), xytext=(12, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT, annotation_clip=False)
    ax.set_xticks(list(sorted({p[0] for pts in curves.values() for p in pts})))
    ax.set_xlabel("minutes of history in the rolling window", color=TEXT_2, fontsize=9)
    ax.set_ylabel("macro-F1", color=TEXT_2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(1 / 4 if len(curves) else 0, color=TEXT_2, linewidth=1, alpha=0.35)
    _finish(fig, ax, path)


def robustness_chart(series: dict[str, list], kind: str, path: Path) -> None:
    if kind == "face_dropout":
        title, sub, xlabel = ("Decisions stay stable when the face disappears",
                              "Macro-F1 as a growing share of ticks has no readable face (face channel only)",
                              "share of ticks with the face hidden")
    else:
        title, sub, xlabel = ("Decisions stay stable when the face is poorly lit",
                              "Macro-F1 as lighting degrades face readings (face noise + misses; lux feature unchanged)",
                              "relative light level (1 = normal)")
    fig, ax = _axes(title, sub)
    ends = {}
    for key in [k for k in ORDER if k in series]:
        xs = [p[0] for p in series[key]]
        ys = [p[1] for p in series[key]]
        _line(ax, xs, ys, key)
        ends[key] = (xs[-1], ys[-1])
    for key, y in _spread_labels({k: v[1] for k, v in ends.items()}, 0.035).items():
        ax.annotate(f"{NAMES[key].split(' ')[0]}  {ends[key][1]:.2f}", (ends[key][0], y), xytext=(12, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT, annotation_clip=False)
    if kind == "low_light":
        ax.invert_xaxis()
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel("macro-F1", color=TEXT_2, fontsize=9)
    ax.set_ylim(0, 1)
    _finish(fig, ax, path)


def episode_age_chart(rows: list[dict], path: Path) -> None:
    fig, ax = _axes("Context needs a minute to catch up",
                    "Accuracy by time since the current emotional episode began (full 15-minute history)")
    bins = [r["bin"] for r in rows]
    x = np.arange(len(bins))
    keys = [k for k in ("m2", "m1", "m0") if k in rows[0]]
    width = min(0.24, 0.8 / len(keys))
    for i, key in enumerate(keys):
        vals = [r[key] for r in rows]
        ax.bar(x + (i - (len(keys) - 1) / 2) * (width + 0.02), vals, width=width, color=SERIES[key],
               label=NAMES[key], linewidth=0)
        if key == "m2":
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}", (xi + (i - (len(keys) - 1) / 2) * (width + 0.02), v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x, bins)
    ax.set_ylabel("accuracy", color=TEXT_2, fontsize=9)
    ax.set_ylim(0, 1)
    _finish(fig, ax, path)


def reliability_chart(table: list[dict], ece_before: float, ece_after: float, path: Path) -> None:
    fig, ax = _axes("Confidence means what it says",
                    f"M2 top-label reliability after temperature scaling (ECE {ece_before:.3f} before, {ece_after:.3f} after)",
                    size=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, alpha=0.5)
    xs = [r["confidence"] for r in table]
    ys = [r["accuracy"] for r in table]
    _line(ax, xs, ys, "m2")
    ax.set_xlabel("predicted confidence", color=TEXT_2, fontsize=9)
    ax.set_ylabel("observed accuracy", color=TEXT_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finish(fig, ax, path, legend=False)


def confusion_pair(cms: dict[str, np.ndarray], states: list[str], path: Path) -> None:
    short = [s.replace("_", " ").replace("Collaborative Happy", "Collab. Happy").replace("Frustrated Masked", "Frustr. Masked")
             for s in states]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.02, 0.965, "Where the face fails, context recovers", fontsize=13, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, 0.905, "Row-normalized confusion matrices on held-out simulated sessions (rows = true state)",
             fontsize=9.5, color=TEXT_2, va="top")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", [SURFACE, "#2a78d6"])
    for ax, (key, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(NAMES[key], fontsize=10, color=TEXT, loc="left")
        ax.set_xticks(range(len(states)), short, rotation=30, ha="right", fontsize=8, color=TEXT_2)
        ax.set_yticks(range(len(states)), short, fontsize=8, color=TEXT_2)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        for i in range(len(states)):
            for j in range(len(states)):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if v > 0.55 else TEXT)
    fig.subplots_adjust(top=0.80, bottom=0.2, wspace=0.55, left=0.12, right=0.98)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def attention_chart(att_by_class: dict[str, list[float]], path: Path) -> None:
    """Mean M3 attention weight by minutes-ago, one line per true state (full 15-minute windows)."""
    fig, ax = _axes("Which minutes mattered",
                    "M3 (GRU + attention) mean attention weight over the 15-minute window, by true state")
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for color, (state, weights) in zip(palette, att_by_class.items()):
        w = np.asarray(weights)
        minutes_ago = (len(w) - 1 - np.arange(len(w))) * 5 / 60
        ax.plot(minutes_ago, w, color=color, linewidth=2, label=state.replace("_", " "))
    ax.invert_xaxis()
    ax.set_xlabel("minutes ago", color=TEXT_2, fontsize=9)
    ax.set_ylabel("attention weight", color=TEXT_2, fontsize=9)
    _finish(fig, ax, path)
