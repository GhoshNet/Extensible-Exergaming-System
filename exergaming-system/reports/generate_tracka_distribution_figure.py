"""
Track A detection-rate distribution figure.

Reads the frozen reports/dataset_detection_rate.csv (640 videos, 3 Aug 2026
freeze) and renders the two-panel figure used in Chapter 5.

Left  : where the 96.3% mean comes from -- per-video detection rate binned into
        five bands, so the "median 100%, small tail" claim is visible.
Right : mean detection rate per exercise category, which shows the tail is not
        random -- it concentrates in bench and machine movements where the
        subject is supine or seated behind equipment.

Every number is computed from the CSV; nothing is hard-coded.

Usage:  python3 generate_tracka_distribution_figure.py
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "dataset_detection_rate.csv"
# Write into the dissertation figure directory when this lives inside the
# dissertation tree; otherwise fall back to reports/figures/ so the script also
# runs standalone from the code submission.
_THESIS = HERE.parent.parent.parent / "DissertationWriting" / "thesis" / "figures"
OUT = _THESIS if _THESIS.parent.exists() else HERE / "figures"

# Palette: validated (CVD separation dE 19.7 deutan / 20.0 normal, all checks pass)
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#c0392b"
INK, MUTED, GRID = "#222222", "#555555", "#d9d9d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    for r in rows:
        r["dr"] = float(r["detection_rate"])
    d = [r["dr"] for r in rows]
    n = len(d)
    mean, median, sd = st.mean(d), st.median(d), st.pstdev(d)
    fw = 100 * sum(int(r["frames_with_pose"]) for r in rows) / sum(
        int(r["frames_processed"]) for r in rows)

    bands = [
        ("100%", lambda x: x == 100, BLUE),
        ("95–<100%", lambda x: 95 <= x < 100, BLUE),
        ("80–<95%", lambda x: 80 <= x < 95, BLUE),
        ("50–<80%", lambda x: 50 <= x < 80, ORANGE),
        ("<50%", lambda x: x < 50, RED),
    ]
    counts = [sum(1 for x in d if f(x)) for _, f, _ in bands]
    tail = counts[3] + counts[4]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.suptitle("Track A: pose detection across the 640-video corpus",
                 x=0.012, ha="left", fontsize=15, y=0.98)

    # ---- left: banded distribution -----------------------------------------
    labels = [b[0] for b in bands]
    colours = [b[2] for b in bands]
    bars = ax1.bar(labels, counts, color=colours, width=0.68, zorder=3)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, c + n * 0.018,
                 f"{c}\n({100*c/n:.1f}%)", ha="center", va="bottom",
                 fontsize=10, color=INK, linespacing=1.25)
    ax1.set_title(
        f"What the {mean:.1f}% mean is made of  (n={n})\n"
        f"mean {mean:.1f}%   ·   median {median:.0f}%   ·   SD {sd:.1f} pp"
        f"   ·   frame-weighted {fw:.1f}%",
        fontsize=12, pad=12, loc="left", color=INK)
    ax1.set_ylabel("Videos")
    ax1.set_xlabel("Per-video pose detection rate")
    ax1.set_ylim(0, max(counts) * 1.42)
    ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.text(0.40, 0.90,
             f"{counts[0]+counts[1]} of {n} videos ({100*(counts[0]+counts[1])/n:.0f}%)\n"
             f"detect on at least 95% of frames.\n"
             f"The mean is pulled down by a tail\nof {tail} videos, not by broad weakness.",
             transform=ax1.transAxes, fontsize=9.5, color=INK, va="top",
             linespacing=1.6)

    # ---- right: mean detection per category ---------------------------------
    bycat = {}
    for r in rows:
        bycat.setdefault(r["category"], []).append(r["dr"])
    cats = sorted(bycat.items(), key=lambda kv: st.mean(kv[1]))
    names = [c.replace("_", " ") for c, _ in cats]
    means = [st.mean(v) for _, v in cats]
    cols = [ORANGE if m < 95 else BLUE for m in means]
    ypos = range(len(names))
    ax2.barh(list(ypos), means, color=cols, height=0.72, zorder=3)
    for y, m, (c, v) in zip(ypos, means, cats):
        ax2.text(101.0, y, f"{m:.1f}  (n={len(v)})", va="center",
                 fontsize=8.5, color=MUTED)
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlim(78, 109)
    ax2.set_xticks([80, 85, 90, 95, 100])
    ax2.set_xlabel("Mean per-video detection rate (%)")
    ax2.set_title("The tail is not random: it concentrates in supine and\n"
                  "machine-based movements", fontsize=12, pad=10, loc="left")
    ax2.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axvline(95, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
    ax2.text(94.4, len(names) - 0.2, "95%", fontsize=9, color=MUTED, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig17_tracka_distribution.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"n={n} mean={mean:.2f} median={median} sd={sd:.2f} frame-weighted={fw:.2f}")
    print("bands:", dict(zip(labels, counts)), "| tail(<80%):", tail)
    print("worst 3 categories:", [(c, round(st.mean(v), 1)) for c, v in cats[:3]])
    print("written:", OUT / "fig17_tracka_distribution.{png,pdf}")


if __name__ == "__main__":
    main()
