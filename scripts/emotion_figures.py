"""Figure for the model card: where each state lands on the valence-arousal plane (held-out simulated sessions).

    python scripts/emotion_figures.py   # -> artifacts/emotion/figs/affect_map.png

Hollow gray circles are the assumed anchors; filled dots are the mean served map position (probability-weighted
anchors) over the test windows of each true state, with a 1-SD ellipse; a thin line joins each pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from valence.contracts import EMOTION_DISPLAY  # noqa: E402
from valence.models.emotion import EmotionPredictor, labels, load_corpus  # noqa: E402
from valence.models.train_ladder import split_sessions  # noqa: E402

SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
# Label offset from the anchor (dx, dy, alignment): the negative, high-arousal corner is crowded.
LABEL = {"Overwhelmed": (-0.05, 0.0, "right"), "Frustrated_Masked": (-0.05, 0.0, "right"),
         "Anxious": (0.0, 0.08, "center"), "Tense": (0.06, 0.07, "left"), "Exhausted": (-0.05, 0.0, "right"),
         "Collaborative_Happy": (0.0, 0.08, "center")}
SHORT = {"Frustrated_Masked": "Frustrated / Masked", "Collaborative_Happy": "Collaborative / Happy",
         "Overwhelmed": "Overwhelmed", "Tense": "Tense"}


def main() -> None:
    art = ROOT / "artifacts" / "emotion"
    pred = EmotionPredictor(art)
    meta = pred.meta["training"]
    X, m = load_corpus(meta["sessions"], meta["seed"], meta["noise_scale"], Path.home() / ".cache" / "valence-emotion")
    y = labels(m)
    split = split_sessions(m, meta["seed"])
    r = pred.predict(X.loc[split.test, pred.cols].to_numpy())
    yt = y[split.test]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    for k, s in enumerate(pred.states):
        av, aa = pred.anchors[k]
        sel = yt == k
        v, a = r["valence_mix"][sel], r["arousal_mix"][sel]
        mv, ma = v.mean(), a.mean()
        ax.add_patch(Ellipse((mv, ma), 2 * v.std(), 2 * a.std(), facecolor=SERIES, alpha=0.08, edgecolor="none",
                             zorder=1))
        ax.plot([av, mv], [aa, ma], color=MUTED, lw=1, zorder=2)
        ax.scatter([av], [aa], s=70, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=3)
        ax.scatter([mv], [ma], s=40, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=4)
        dx, dy, ha = LABEL.get(s, (0.05, 0.0, "left"))
        ax.text(av + dx, aa + dy, SHORT.get(s, EMOTION_DISPLAY[s]), fontsize=8.5, color=INK, ha=ha, va="center",
                zorder=5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence (unpleasant → pleasant)", color=MUTED, fontsize=9)
    ax.set_ylabel("Arousal (calm / tired → activated)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Emotion map on held-out simulated sessions", loc="left", color=INK, fontsize=11, pad=22)
    ax.text(0, 1.02, f"○ assumed anchor   ● mean served position (Σ p·anchor), shaded ±1 SD   ·   "
            f"{int(split.test.sum()):,} windows, {pred.model_id}", transform=ax.transAxes, color=MUTED, fontsize=7.5)
    out = art / "figs"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "affect_map.png", facecolor=SURFACE)
    print(out / "affect_map.png")


if __name__ == "__main__":
    main()
