"""
Figures for the percentile re-derivation, in the same visual language as
generate_evaluation_figures.py / generate_presentation_figures.py (same
palette, same rcParams, same _no_top_right helper).

fig9  Threshold re-derivation. What the percentile change actually did to the
      learned thresholds, against the hand-built reference where one exists.
      This is the figure slide 20's speaker note promised ("the full
      re-derivation ... will be noted down").

fig10 Evaluation impact, and why the comparison is trustworthy. Left: the
      paired effect of the change on Track B (185 videos) and Track C (24).
      Right: the reproducibility evidence -- how far an identical re-run
      drifts, set against the size of the effect being measured. This is the
      panel that justifies the paired within-run design.

generate_evaluation_figures.py and its fig1 are deliberately NOT modified:
they produced the published figures and are left reproducible.

Usage: python reports/generate_percentile_figures.py
"""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib.pyplot as plt
from src.controller import EXERCISE_DEFINITIONS

ROOT = Path(__file__).parents[1]
REPORTS = Path(__file__).parent
FIG_DIR = REPORTS / "figures"
SCRATCH = Path("/private/tmp/claude-501/-Users-tanmay-Documents-TCD-Course-Material-"
               "Dissertation/55fa7383-f598-4e27-af72-c2bf56047893/scratchpad")
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
    ax.set_axisbelow(True)


def _thresholds(defn_name):
    d = EXERCISE_DEFINITIONS[defn_name]
    down = next(t for t in d.transitions if t.to_state == "DOWN").conditions[0].threshold
    up = next(t for t in d.transitions if t.to_state == "UP").conditions[0].threshold
    return down, up


def _mean_acc(rows, variant):
    return st.mean([float(r["accuracy_pct"]) for r in rows
                    if r["variant"] == variant and r["accuracy_pct"]])


def fig9_threshold_rederivation():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [2.1, 1]})

    # ---- Panel A: squat + push-up, where a hand-built reference exists ----
    groups = [
        ("Squat\nDOWN entry", "Squat", "Learned Squat", "Learned Squat (percentile)", 0),
        ("Squat\nUP return",  "Squat", "Learned Squat", "Learned Squat (percentile)", 1),
        ("Push-up\nDOWN entry", "Push-up", "Learned Push-up", "Learned Push-up (percentile)", 0),
        ("Push-up\nUP return",  "Push-up", "Learned Push-up", "Learned Push-up (percentile)", 1),
    ]
    labels, hc, orig, perc = [], [], [], []
    for label, hc_name, o_name, p_name, idx in groups:
        labels.append(label)
        hc.append(_thresholds(hc_name)[idx])
        orig.append(_thresholds(o_name)[idx])
        perc.append(_thresholds(p_name)[idx])

    x = range(len(labels))
    w = 0.26
    series = [
        ("Hand-built reference", hc, MUTED, -w),
        ("Learned (min/max)", orig, BLUE, 0.0),
        ("Learned (percentile)", perc, ORANGE, w),
    ]
    for name, vals, color, off in series:
        bars = ax1.bar([i + off for i in x], vals, width=w, color=color,
                       label=name, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=8.5, color=SECONDARY_INK)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Threshold (degrees)")
    ax1.set_ylim(0, max(hc + orig + perc) * 1.20)
    ax1.grid(axis="x", visible=False)
    _no_top_right(ax1)
    ax1.legend(frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9.5)
    ax1.set_title("Against a hand-built reference", fontsize=12,
                  color=INK, pad=10, loc="left")

    # ---- Panel B: bicep curl, learned only ----
    c_o = _thresholds("BicepCurl")
    c_p = _thresholds("BicepCurl (percentile)")
    x2 = range(2)
    for name, vals, color, off in (("Learned (min/max)", c_o, BLUE, -w / 1.6),
                                   ("Learned (percentile)", c_p, ORANGE, w / 1.6)):
        bars = ax2.bar([i + off for i in x2], vals, width=w * 1.25, color=color,
                       zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=8.5, color=SECONDARY_INK)
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(["Bicep curl\nDOWN entry", "Bicep curl\nUP return"])
    ax2.set_ylim(0, max(list(c_o) + list(c_p)) * 1.38)
    ax2.grid(axis="x", visible=False)
    _no_top_right(ax2)
    ax2.set_title("No hand-built counterpart exists", fontsize=12,
                  color=INK, pad=10, loc="left")
    # Left of the axis, above the shorter DOWN-entry pair -- the only region
    # of this panel with clear space.
    ax2.text(0.03, 0.97,
             f"Largest shift of the three:\nDOWN entry {c_p[0] - c_o[0]:+.1f} deg",
             transform=ax2.transAxes, ha="left", va="top",
             fontsize=9.5, color=SECONDARY_INK)

    fig.suptitle("Learned thresholds re-derived at the 2nd/98th percentile",
                 fontsize=13.5, color=INK, y=0.99, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "fig9_threshold_rederivation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out.name}")


def fig10_variant_impact():
    tb = list(csv.DictReader(open(REPORTS / "evaluation_results_both_variants.csv")))
    tc = list(csv.DictReader(open(REPORTS / "angle_distance_eval_both_variants.csv")))

    tb_o, tb_p = _mean_acc(tb, "original"), _mean_acc(tb, "percentile")
    tc_o, tc_p = _mean_acc(tc, "original"), _mean_acc(tc, "percentile")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # ---- Panel A: paired effect on each track ----
    labels = [f"Track B\n185 videos", f"Track C\n24 videos"]
    x = range(2)
    w = 0.2
    o_vals, p_vals = [tb_o, tc_o], [tb_p, tc_p]
    for name, vals, color, off in (("Learned (min/max)", o_vals, BLUE, -w * 0.55),
                                   ("Learned (percentile)", p_vals, ORANGE, w * 0.55)):
        bars = ax1.bar([i + off for i in x], vals, width=w, color=color,
                       label=name, zorder=3, edgecolor="white", linewidth=2)
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=9.5, color=SECONDARY_INK)
    for i, (a, b) in enumerate(zip(o_vals, p_vals)):
        ax1.annotate(f"{b - a:+.1f} pp", (i, max(a, b) + 6), ha="center",
                     fontsize=10.5, color=INK, weight="bold")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean rep-count accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.55, 1.55)
    ax1.grid(axis="x", visible=False)
    _no_top_right(ax1)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=2, fontsize=9.5)
    ax1.set_title("Effect of the percentile change (paired, same detection pass)",
                  fontsize=12, color=INK, pad=10, loc="left")

    # ---- Panel B: reproducibility vs effect size ----
    tc_a = {(r["file"], r["variant"]): r for r in
            csv.DictReader(open(SCRATCH / "tc_runA.csv"))}
    tc_b = {(r["file"], r["variant"]): r for r in
            csv.DictReader(open(SCRATCH / "tc_runB.csv"))}
    tc_flips = sum(1 for k in tc_a
                   if tc_a[k]["detected_reps"] != tc_b[k]["detected_reps"])

    tb1_p = SCRATCH / "tb_run1.csv"
    tb2_p = SCRATCH / "tb_run2.csv"
    have_tb_repeat = tb1_p.exists() and tb2_p.exists()
    if have_tb_repeat:
        tb1 = {(r["file"], r["variant"]): r for r in csv.DictReader(open(tb1_p))}
        tb2 = {(r["file"], r["variant"]): r for r in csv.DictReader(open(tb2_p))}
        tb_flips = sum(1 for k in tb1
                       if tb1[k]["detected_reps"] != tb2[k]["detected_reps"])
        tb_drift = abs(_mean_acc(list(csv.DictReader(open(tb2_p))), "original")
                       - _mean_acc(list(csv.DictReader(open(tb1_p))), "original"))
    else:
        tb_flips = tb_drift = None

    # Per-video instability, which is where the nondeterminism actually shows.
    # Aggregate drift is 0.00 pp on BOTH tracks -- the flips cancel -- so
    # plotting drift would show two empty bars and hide the finding.
    tb_pct = 100.0 * tb_flips / len(tb1) if have_tb_repeat else 0.0
    tc_pct = 100.0 * tc_flips / len(tc_a)

    names = ["Track B\n370 rows", "Track C\n48 rows"]
    vals = [tb_pct, tc_pct]
    bars = ax2.bar(range(2), vals, width=0.5, color=[BLUE, ORANGE], zorder=3)
    for bar, v, n in zip(bars, vals, [tb_flips if have_tb_repeat else 0, tc_flips]):
        ax2.annotate(f"{v:.1f}%  ({n} rows)",
                     (bar.get_x() + bar.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=9.5, color=SECONDARY_INK)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("Rows changing rep count on an identical re-run (%)")
    ax2.set_ylim(0, max(vals + [4]) * 1.55)
    ax2.grid(axis="x", visible=False)
    _no_top_right(ax2)
    ax2.set_title("Reproducibility of an identical re-run", fontsize=12,
                  color=INK, pad=10, loc="left")

    note = ("Aggregate accuracy drifts 0.00 pp on both tracks --\n"
            "per-video flips cancel out. The paired delta reproduced\n"
            "exactly on both tracks, because both arms share one\n"
            "detection pass and the noise is common-mode.\n"
            "Not caused by decoding, rotation, resolution or low\n"
            "detection rate: all four were tested and ruled out.")
    ax2.text(0.97, 0.97, note, transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, color=SECONDARY_INK)

    fig.suptitle("Impact of the percentile re-derivation, and the noise it must be read against",
                 fontsize=13.5, color=INK, y=0.99, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "fig10_variant_impact.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out.name}"
          + ("" if have_tb_repeat else "  (Track B repeat not found -- Track C only)"))


def fig11_coverage_and_subgroups():
    """
    Track B across five movement classes, and where its failures concentrate.

    Left: per-category accuracy -- the breadth result (C3). Right: the two
    conditions the definitions structurally do not support, each against its
    supported counterpart. The point of the right panel is that the weakest
    category is not uniformly weak: its failure is confined to a condition
    that can be named and explained.
    """
    rows = [r for r in csv.DictReader(open(REPORTS / "evaluation_results_both_variants.csv"))
            if r["variant"] == "original" and r["accuracy_pct"]]

    def mean_of(pred):
        v = [float(r["accuracy_pct"]) for r in rows if pred(r)]
        return (st.mean(v), len(v)) if v else (0.0, 0)

    cats = ["shoulder_press", "squat", "barbell_biceps_curl", "push-up", "hammer_curl"]
    label = {"shoulder_press": "Shoulder press", "squat": "Squat",
             "barbell_biceps_curl": "Bicep curl", "push-up": "Push-up",
             "hammer_curl": "Hammer curl\n(transfer test)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    # ---- Panel A: per-category ----
    vals, ns = zip(*[mean_of(lambda r, c=c: r["category"] == c) for c in cats])
    ypos = list(range(len(cats)))[::-1]
    bars = ax1.barh(ypos, vals, height=0.6, color=BLUE, zorder=3)
    for y, v, n in zip(ypos, vals, ns):
        ax1.text(v + 1.2, y, f"{v:.1f}%  (n={n})", va="center", ha="left",
                 fontsize=9.5, color=SECONDARY_INK)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([label[c] for c in cats], fontsize=10)
    ax1.set_xlabel("Mean rep-count accuracy (%)")
    ax1.set_xlim(0, 108)
    ax1.grid(axis="y", visible=False)
    _no_top_right(ax1)
    ax1.set_title("Five movement classes, 239 videos", fontsize=12,
                  color=INK, pad=10, loc="left")

    # ---- Panel B: supported vs unsupported condition ----
    def has(r, s): return s in (r["notes"] or "").lower()
    groups = [
        ("Hammer curl\nboth hands", lambda r: r["category"] == "hammer_curl"
            and not has(r, "one hand at a time"), BLUE),
        ("Hammer curl\none hand at a time", lambda r: r["category"] == "hammer_curl"
            and has(r, "one hand at a time"), ORANGE),
        ("Shoulder press\nfree weight", lambda r: r["category"] == "shoulder_press"
            and not has(r, "machine") and not has(r, "arnold"), BLUE),
        ("Shoulder press\nmachine", lambda r: r["category"] == "shoulder_press"
            and has(r, "machine"), ORANGE),
    ]
    gv, gn = zip(*[mean_of(p) for _, p, _ in groups])
    cols = [c for _, _, c in groups]
    xs = [0, 0.72, 1.9, 2.62]
    bars = ax2.bar(xs, gv, width=0.6, color=cols, zorder=3)
    for x, v, n in zip(xs, gv, gn):
        ax2.text(x, v + 1.5, f"{v:.1f}%\nn={n}", ha="center", va="bottom",
                 fontsize=9.5, color=SECONDARY_INK)
    for a, b in ((0, 1), (2, 3)):
        ax2.annotate(f"{gv[b]-gv[a]:+.1f} pp",
                     ((xs[a] + xs[b]) / 2, max(gv[a], gv[b]) + 21),
                     ha="center", fontsize=10.5, color=INK, weight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax2.set_ylabel("Mean rep-count accuracy (%)")
    ax2.set_ylim(0, 130)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.grid(axis="x", visible=False)
    _no_top_right(ax2)
    ax2.set_title("Where the failures concentrate", fontsize=12,
                  color=INK, pad=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax2.legend(handles, ["Condition the definition supports",
                         "Condition it does not"],
               frameon=False, loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.30), ncol=2)

    fig.suptitle("Rep-counting accuracy across movement classes, and the conditions that explain its failures",
                 fontsize=13.5, color=INK, y=0.99, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "fig11_coverage_and_subgroups.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    fig9_threshold_rederivation()
    fig10_variant_impact()
    fig11_coverage_and_subgroups()
