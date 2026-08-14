"""
Generates two new dissertation-presentation figures from the real result
CSVs produced this evaluation phase, matching the exact visual style of
generate_evaluation_figures.py (same palette, same rcParams, same
_no_top_right helper) so they look native to the existing deck.

Fig6: at-scale evaluation (Track A detection rate on 753 videos, Track B
      rep-counting accuracy on 185 videos) -- replaces the old "100% on 7
      videos" slide with the real, much larger, in-the-wild evaluation.
Fig7: camera angle/distance sensitivity (Track C, 24 controlled videos)
      and the before/after of the 400ms -> 200ms timing-floor fix.

Usage: python reports/generate_presentation_figures.py
Output: reports/figures/fig6_at_scale_evaluation.png
        reports/figures/fig7_angle_distance_sensitivity.png
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"
GOOD = "#2f9e4f"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _no_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def fig6_at_scale_evaluation():
    # Track A: detection rate, 753 videos
    det_rows = list(csv.DictReader(open(REPORTS / "dataset_detection_rate.csv")))
    det_rates = [float(r["detection_rate"]) for r in det_rows if r["detection_rate"]]

    # Track B: rep-counting accuracy, 185 videos (squat/push-up/bicep curl)
    acc_rows = list(csv.DictReader(open(REPORTS / "evaluation_results.csv")))
    by_ex = defaultdict(list)
    for r in acc_rows:
        by_ex[r["category"]].append(float(r["accuracy_pct"]))
    ex_labels = {"squat": "Squat", "push-up": "Push-up", "barbell_biceps_curl": "Bicep curl"}
    exercises = ["squat", "push-up", "barbell_biceps_curl"]
    means = [st.mean(by_ex[e]) for e in exercises]
    ns = [len(by_ex[e]) for e in exercises]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: detection-rate distribution (histogram)
    ax1.hist(det_rates, bins=20, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
    mean_det = st.mean(det_rates)
    ax1.axvline(mean_det, color=CRITICAL, linewidth=1.6, zorder=4, linestyle="--")
    ax1.text(mean_det - 3, ax1.get_ylim()[1] if ax1.get_ylim()[1] else 1,
              f"mean {mean_det:.1f}%", ha="right", va="top", fontsize=9.5, color=CRITICAL, weight="bold")
    ax1.set_xlabel("Per-video pose detection rate (%)")
    ax1.set_ylabel("Number of videos")
    ax1.set_title(f"Track A — Detection rate\n{len(det_rates)} videos, every frame, no sampling",
                   fontsize=11, weight="bold", color=INK, loc="left")
    _no_top_right(ax1)

    # Right: rep-counting accuracy by exercise (bar)
    x = range(len(exercises))
    bars = ax2.bar(x, means, color=ORANGE, zorder=3, width=0.55)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax2.text(i, m + 2, f"{m:.1f}%\n(n={n})", ha="center", fontsize=9.5, color=INK)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([ex_labels[e] for e in exercises])
    ax2.set_ylabel("Mean rep-counting accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Track B — Rep-counting accuracy\n{len(acc_rows)} videos, real manually-counted ground truth",
                   fontsize=11, weight="bold", color=INK, loc="left")
    _no_top_right(ax2)

    fig.tight_layout()
    out = FIG_DIR / "fig6_at_scale_evaluation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def fig7_angle_distance_sensitivity():
    final_rows = list(csv.DictReader(open(REPORTS / "angle_distance_eval_results.csv")))
    sweep_rows = list(csv.DictReader(open(REPORTS / "timing_floor_sweep_results.csv")))

    # Angle sensitivity, pooled across all 4 exercises, at the shipped 200ms floor
    by_angle = defaultdict(list)
    for r in final_rows:
        by_angle[int(r["angle"])].append(float(r["accuracy_pct"]))
    angles = sorted(by_angle)
    angle_means = [st.mean(by_angle[a]) for a in angles]

    # Timing-floor before/after, per exercise
    sweep_agg = defaultdict(list)
    for r in sweep_rows:
        sweep_agg[(r["exercise"], r["threshold_ms"])].append(float(r["accuracy_pct"]))
    ex_labels = {"squat": "Squat", "pushup": "Push-up", "bicepcurl": "Bicep curl", "jumpingjacks": "Jumping jacks"}
    exercises = ["squat", "pushup", "bicepcurl", "jumpingjacks"]
    before = [st.mean(sweep_agg[(e, "400")]) for e in exercises]
    after = [st.mean(sweep_agg[(e, "200")]) for e in exercises]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: angle sensitivity line/bar
    bars = ax1.bar([str(a) + "°" for a in angles], angle_means, color=BLUE, zorder=3, width=0.5)
    for i, m in enumerate(angle_means):
        ax1.text(i, m + 2, f"{m:.1f}%", ha="center", fontsize=9.5, color=INK)
    ax1.set_ylabel("Mean accuracy (%)")
    ax1.set_ylim(0, 110)
    ax1.set_title("Camera angle sensitivity\n24 controlled videos, 4 exercises, 5 known reps each",
                   fontsize=11, weight="bold", color=INK, loc="left")
    _no_top_right(ax1)

    # Right: timing floor before/after grouped bars
    x = range(len(exercises))
    w = 0.36
    b1 = ax2.bar([i - w/2 for i in x], before, width=w, color=MUTED, label="400ms (old)", zorder=3)
    b2 = ax2.bar([i + w/2 for i in x], after, width=w, color=GOOD, label="200ms (shipped)", zorder=3)
    for i, (bv, av) in enumerate(zip(before, after)):
        ax2.text(i - w/2, bv + 2, f"{bv:.0f}", ha="center", fontsize=8.5, color=SECONDARY_INK)
        ax2.text(i + w/2, av + 2, f"{av:.0f}", ha="center", fontsize=8.5, color=INK, weight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([ex_labels[e] for e in exercises], fontsize=9.5)
    ax2.set_ylabel("Mean accuracy (%)")
    ax2.set_ylim(0, 110)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.set_title("Rep-timing floor: 400ms → 200ms\nsame 24 clips, before vs after",
                   fontsize=11, weight="bold", color=INK, loc="left")
    _no_top_right(ax2)

    fig.tight_layout()
    out = FIG_DIR / "fig7_angle_distance_sensitivity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig6_at_scale_evaluation()
    fig7_angle_distance_sensitivity()
