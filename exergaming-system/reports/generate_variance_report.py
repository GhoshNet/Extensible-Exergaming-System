"""
Variance and distribution analysis across all four evaluation tracks.

Written in response to supervisor direction (Meeting 11, 4 Aug 2026): report
dispersion, not only means -- "you don't just say the average is blah, you say
the average over 800, but also the deviation, so we know are there outliers or
is there something hidden by the mean".

Pure analysis over the existing result CSVs. It re-runs nothing and changes no
system behaviour, so the codebase freeze holds.

Outputs:
  reports/variance_summary.csv                     every reported mean, with SD,
                                                   SE, range and n
  reports/figures/fig12_distribution_and_variance.png
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path(__file__).parent
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _no_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


def stats(values):
    n = len(values)
    if n == 0:
        return None
    mean = st.mean(values)
    sd = st.stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 2),
        "median": round(st.median(values), 2),
        "sd": round(sd, 2),
        "se": round(sd / (n ** 0.5), 2) if n else 0.0,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def collect():
    rows = []

    def add(track, subset, unit, values):
        s = stats(values)
        if s:
            rows.append({"track": track, "subset": subset, "unit": unit, **s})

    # Track A
    a = [float(r["detection_rate"]) for r in
         csv.DictReader(open(REPORTS / "dataset_detection_rate.csv"))
         if r["detection_rate"]]
    add("A", "all videos", "% frames with pose", a)

    # Track B
    tb = [r for r in csv.DictReader(open(REPORTS / "evaluation_results_both_variants.csv"))
          if r["variant"] == "original" and r["accuracy_pct"]]
    add("B", "all videos", "% rep accuracy", [float(r["accuracy_pct"]) for r in tb])
    for c in ("squat", "push-up", "barbell_biceps_curl", "hammer_curl", "shoulder_press"):
        add("B", c, "% rep accuracy",
            [float(r["accuracy_pct"]) for r in tb if r["category"] == c])
    uni = [r for r in tb if r["category"] == "hammer_curl"
           and "one hand at a time" in (r["notes"] or "").lower()]
    add("B", "hammer_curl unilateral", "% rep accuracy",
        [float(r["accuracy_pct"]) for r in uni])
    add("B", "hammer_curl bilateral", "% rep accuracy",
        [float(r["accuracy_pct"]) for r in tb
         if r["category"] == "hammer_curl" and r not in uni])

    # Track C
    tc = [r for r in csv.DictReader(open(REPORTS / "angle_distance_eval_both_variants.csv"))
          if r["variant"] == "original"]
    add("C", "all cells", "% rep accuracy", [float(r["accuracy_pct"]) for r in tc])
    for ang in ("0", "45", "90"):
        add("C", f"angle {ang} deg", "% rep accuracy",
            [float(r["accuracy_pct"]) for r in tc if r["angle"] == ang])
    for d in ("2m", "5m"):
        add("C", f"distance {d}", "% rep accuracy",
            [float(r["accuracy_pct"]) for r in tc if r["distance"] == d])

    # Track D
    td = list(csv.DictReader(open(REPORTS / "latency_eval_results.csv")))
    for cond in ("detect_only", "baseline_render", "full_c2"):
        add("D", cond, "ms per frame",
            [float(r["mean_ms"]) for r in td if r["condition"] == cond])
    base = {r["video"]: float(r["mean_ms"]) for r in td if r["condition"] == "baseline_render"}
    full = {r["video"]: float(r["mean_ms"]) for r in td if r["condition"] == "full_c2"}
    add("D", "C2 marginal cost", "ms per frame", [full[k] - base[k] for k in base])
    return rows


def figure(rows):
    tb = [r for r in csv.DictReader(open(REPORTS / "evaluation_results_both_variants.csv"))
          if r["variant"] == "original" and r["accuracy_pct"]]
    acc = [float(r["accuracy_pct"]) for r in tb]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # --- Panel A: the distribution behind the Track B mean ---
    bands = [
        ("Total miss\n(0%)", sum(1 for v in acc if v == 0), CRITICAL),
        ("Partial\n(0-50%)", sum(1 for v in acc if 0 < v < 50), ORANGE),
        ("Close\n(50-<100%)", sum(1 for v in acc if 50 <= v < 100), MUTED),
        ("Exact match\n(100%)", sum(1 for v in acc if v == 100), BLUE),
    ]
    labels = [b[0] for b in bands]
    counts = [b[1] for b in bands]
    bars = ax1.bar(range(4), counts, width=0.6, color=[b[2] for b in bands], zorder=3)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, c + 3,
                 f"{c}\n({100*c/len(acc):.1f}%)", ha="center", va="bottom",
                 fontsize=9.5, color=SECONDARY_INK)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Videos")
    ax1.set_ylim(0, max(counts) * 1.28)
    ax1.grid(axis="x", visible=False)
    _no_top_right(ax1)
    ax1.set_title("What the 75.9% mean is made of (Track B, n=239)",
                  fontsize=12, color=INK, pad=10, loc="left")
    mid = sum(1 for v in acc if 0 < v < 100)
    ax1.text(0.02, 0.97,
             f"mean {st.mean(acc):.1f}%    median {st.median(acc):.0f}%    "
             f"SD {st.stdev(acc):.1f} pp\n\n"
             f"Only {mid} of {len(acc)} videos ({100*mid/len(acc):.0f}%) fall between\n"
             f"the two extremes. The mean sits where\n"
             f"almost no individual video actually is.",
             transform=ax1.transAxes, ha="left", va="top",
             fontsize=9.5, color=SECONDARY_INK)

    # --- Panel B: dispersion on every Track B/C mean ---
    sel = [r for r in rows if (r["track"], r["subset"]) in {
        ("B", "squat"), ("B", "push-up"), ("B", "barbell_biceps_curl"),
        ("B", "hammer_curl"), ("B", "shoulder_press"),
        ("C", "angle 0 deg"), ("C", "angle 45 deg"), ("C", "angle 90 deg")}]
    order = ["squat", "push-up", "barbell_biceps_curl", "hammer_curl",
             "shoulder_press", "angle 0 deg", "angle 45 deg", "angle 90 deg"]
    sel.sort(key=lambda r: order.index(r["subset"]))
    # n goes into the tick label so it cannot collide with the error bars,
    # which routinely run past 100 given how wide the dispersion is.
    names = [r["subset"].replace("barbell_biceps_curl", "bicep curl")
             .replace("shoulder_press", "shoulder press")
             .replace("hammer_curl", "hammer curl").replace(" deg", "°")
             + f"  (n={r['n']})" for r in sel]
    means = [r["mean"] for r in sel]
    sds = [r["sd"] for r in sel]
    cols = [BLUE if r["track"] == "B" else ORANGE for r in sel]

    y = list(range(len(sel)))[::-1]
    ax2.errorbar(means, y, xerr=sds, fmt="none", ecolor=SECONDARY_INK,
                 elinewidth=1.4, capsize=4, zorder=4)
    ax2.scatter(means, y, s=70, c=cols, zorder=5)
    # Accuracy is bounded at 100%; the guide marks where the scale actually ends
    # so a bar running past it is not read as achievable performance.
    ax2.axvline(100, color=GRID, linewidth=1.2, zorder=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.set_xlabel("Mean rep-count accuracy (%) $\\pm$ 1 SD")
    ax2.set_xlim(0, 122)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylim(-1.4, len(sel) - 0.4)
    ax2.grid(axis="y", visible=False)
    _no_top_right(ax2)
    ax2.set_title("Dispersion around every reported mean",
                  fontsize=12, color=INK, pad=10, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Track B (in the wild)"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="Track C (controlled)")]
    ax2.legend(handles=handles, frameon=False, ncol=2, fontsize=9,
               loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("Dispersion behind the reported means",
                 fontsize=13.5, color=INK, y=0.99, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "fig12_distribution_and_variance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out.name}")


def main():
    rows = collect()
    out = REPORTS / "variance_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out.name} ({len(rows)} rows)\n")
    print(f"{'trk':4}{'subset':26}{'mean':>8}{'SD':>8}{'SE':>7}{'range':>18}{'n':>6}")
    for r in rows:
        rng = "[{}, {}]".format(r["min"], r["max"])
        print(f"{r['track']:4}{r['subset']:26}{r['mean']:>8}{r['sd']:>8}"
              f"{r['se']:>7}{rng:>18}{r['n']:>6}")
    figure(rows)


if __name__ == "__main__":
    main()
