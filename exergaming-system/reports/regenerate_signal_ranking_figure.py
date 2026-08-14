"""
Signal-ranking figure for the push-up demonstration (Chapter 4).

Replaces generate_evaluation_figures.py:fig3_signal_importance_pushup, which
re-ran the MediaPipe pipeline over the demonstration video to produce the plot.
Two defects in that version are corrected here:

  1. No legend. The bars were coloured red / blue / grey to mark the
     foreshortening artefact, the true driver, and the remaining candidates,
     but nothing on the figure said so.
  2. Dimensional error. The two spread-ratio signals were annotated in degrees
     ("1.9 deg", "1.4 deg"). Spread ratios are dimensionless -- a distance
     divided by a body-scale distance -- so a degree symbol is wrong.

The ranked ranges below are those produced by the learner for
Videos/Pushups/pushups-man-gym.mp4 and reported in Section 4.4.3. They are
stated as constants because regenerating them requires MediaPipe and the
demonstration video, neither of which ships with this repository.

Usage:  python3 regenerate_signal_ranking_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
_THESIS = HERE.parent.parent.parent / "DissertationWriting" / "thesis" / "figures"
OUT = _THESIS if _THESIS.parent.exists() else HERE / "figures"

# Validated palette: CVD dE 19.7 deutan / 20.0 normal, all checks pass
RED, BLUE, GREY = "#c0392b", "#1f77b4", "#9aa0a6"
INK, MUTED, GRID = "#222222", "#555555", "#d9d9d9"

# name, range, unit ("deg" | "ratio")
SIGNALS = [
    ("arm_spread_ratio",  1.4,   "ratio"),
    ("leg_spread_ratio",  1.9,   "ratio"),
    ("ankle_angle",      55.2,   "deg"),
    ("arm_angle",        62.8,   "deg"),
    ("hip_angle",        89.1,   "deg"),
    ("elbow_angle",     122.7,   "deg"),
    ("knee_angle",      124.3,   "deg"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
})


def main():
    names = [s[0] for s in SIGNALS]
    vals = [s[1] for s in SIGNALS]
    units = [s[2] for s in SIGNALS]
    colours = [RED if n == "knee_angle" else BLUE if n == "elbow_angle" else GREY
               for n in names]

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    bars = ax.barh(names, vals, color=colours, height=0.68, zorder=3)
    for bar, v, u in zip(bars, vals, units):
        label = f"{v:.1f}°" if u == "deg" else f"{v:.1f}  (ratio, dimensionless)"
        ax.text(v + 2, bar.get_y() + bar.get_height()/2, label,
                va="center", fontsize=10, color=INK)

    ax.set_xlabel("Range of motion over the demonstration\n"
                  "(degrees for angle signals; dimensionless for spread ratios)")
    ax.set_xlim(0, 168)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Auto-ranked candidate signals on a push-up demonstration",
                 fontsize=13, loc="left", pad=14)

    ax.legend(handles=[
        Patch(facecolor=RED,  label="Ranked first, but a 2D foreshortening artefact"),
        Patch(facecolor=BLUE, label="The true driver of the movement"),
        Patch(facecolor=GREY, label="Other candidate signals"),
    ], frameon=False, loc="lower right", fontsize=10)

    ax.text(0.0, -0.30,
            "The knees stay nearly straight in a plank, so a 124° knee range is "
            "physically implausible: it is\nprojection, not motion. The human "
            "reviewer deselects knee_angle, promoting elbow_angle to primary.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top",
            linespacing=1.5)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_signal_importance_pushup.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("signals:", dict(zip(names, vals)))
    print("written:", OUT / "fig3_signal_importance_pushup.{png,pdf}")


if __name__ == "__main__":
    main()
