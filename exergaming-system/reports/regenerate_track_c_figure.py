"""
Track C figure (camera angle sensitivity + repetition-timing floor).

Replaces the earlier generate_presentation_figures.py:fig7_* which read
angle_distance_eval_results.csv -- the July evaluation run. That file predates
the freeze and gives 0 deg = 90.0% and 90 deg = 70.0%, which contradicts the
figures reported in the Evaluation chapter.

This version reads angle_distance_eval_both_variants.csv and filters to
variant == "original" (the shipped definitions), which is the frozen source of
truth: 0 deg = 87.5%, 45 deg = 97.5%, 90 deg = 65.0%.

Also adds the x-axis label the earlier left panel was missing.

Usage:  python3 regenerate_track_c_figure.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_THESIS = HERE.parent.parent.parent / "DissertationWriting" / "thesis" / "figures"
OUT = _THESIS if _THESIS.parent.exists() else HERE / "figures"

# Validated palette (CVD dE 24.6 protan / 35.7 normal, all checks pass)
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
INK, MUTED, GRID = "#222222", "#555555", "#d9d9d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    rows = [r for r in csv.DictReader(
        (HERE / "angle_distance_eval_both_variants.csv").open(encoding="utf-8"))
        if r["variant"] == "original"]
    sweep = list(csv.DictReader(
        (HERE / "timing_floor_sweep_results.csv").open(encoding="utf-8")))

    by_angle = defaultdict(list)
    for r in rows:
        by_angle[int(r["angle"])].append(float(r["accuracy_pct"]))
    angles = sorted(by_angle)
    means = [st.mean(by_angle[a]) for a in angles]
    sds = [st.stdev(by_angle[a]) for a in angles]

    agg = defaultdict(list)
    for r in sweep:
        agg[(r["exercise"], r["threshold_ms"])].append(float(r["accuracy_pct"]))
    order = ["squat", "pushup", "bicepcurl", "jumpingjacks"]
    labels = ["Squat", "Push-up", "Bicep curl", "Jumping jacks"]
    before = [st.mean(agg[(e, "400")]) for e in order]
    after = [st.mean(agg[(e, "200")]) for e in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # ---- left: angle sensitivity ------------------------------------------
    xs = [f"{a}°" for a in angles]
    bars = ax1.bar(xs, means, color=BLUE, width=0.55, zorder=3,
                   yerr=sds, capsize=5, error_kw={"ecolor": MUTED, "lw": 1.2})
    for x, m, s in zip(range(len(xs)), means, sds):
        ax1.text(x, m + s + 3, f"{m:.1f}%", ha="center", fontsize=10.5, color=INK)
    ax1.set_ylabel("Mean repetition-counting accuracy (%)")
    ax1.set_xlabel("Camera angle to the performer")
    ax1.set_ylim(0, 125)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("Camera angle sensitivity\n"
                  "24 controlled clips, 4 exercises, 5 known repetitions each\n"
                  "bars show mean, whiskers ±1 SD",
                  fontsize=11.5, loc="left", pad=12)

    # ---- right: timing floor before / after --------------------------------
    w = 0.36
    idx = range(len(order))
    b1 = ax2.bar([i - w/2 for i in idx], before, w, color=ORANGE, zorder=3,
                 label="400 ms floor (original)")
    b2 = ax2.bar([i + w/2 for i in idx], after, w, color=BLUE, zorder=3,
                 label="200 ms floor (shipped)")
    for bars_, vals in ((b1, before), (b2, after)):
        for bar, v in zip(bars_, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 2.5, f"{v:.0f}",
                     ha="center", fontsize=10, color=INK)
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Mean repetition-counting accuracy (%)")
    ax2.set_xlabel("Exercise")
    ax2.set_ylim(0, 118)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper left", fontsize=10, ncol=1)
    ax2.set_title("Repetition-timing floor, same 24 clips\n"
                  "lowering the floor recovers jumping jacks without\n"
                  "over-counting any slower exercise",
                  fontsize=11.5, loc="left", pad=12)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig7_angle_distance_sensitivity.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("angle means (frozen, variant=original):",
          {a: round(m, 1) for a, m in zip(angles, means)})
    print("SDs:", {a: round(s, 1) for a, s in zip(angles, sds)})
    print("timing floor before:", [round(v) for v in before],
          "after:", [round(v) for v in after])
    print("written:", OUT / "fig7_angle_distance_sensitivity.{png,pdf}")


if __name__ == "__main__":
    main()
