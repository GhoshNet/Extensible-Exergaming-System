"""
Generates the Track D latency figure from the real timing CSVs produced by
run_latency_eval.py, matching the exact visual style of
generate_evaluation_figures.py / generate_presentation_figures.py (same
palette, same rcParams, same _no_top_right helper).

Fig8, panel A: where the per-frame compute budget actually goes. Pose
      detection dominates at ~92%; the three C2 stages (form analysis,
      coloured skeleton, error text) are a sliver. The visual point IS
      that the C2 bars are barely visible.
Fig8, panel B: reconstruction of the live frame budget. Measured compute
      plus the fixed 10 ms capture-thread sleep account for only part of
      the observed ~66 ms live frame; the remainder is camera capture and
      display, which this harness deliberately does not measure (it is
      I/O and cannot be held constant across a repeatable run). Shown as
      a labelled residual rather than an attributed measurement.

Usage: python reports/generate_latency_figure.py
Output: reports/figures/fig8_latency_breakdown.png
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

# The three stages that exist only because of C2 (per-body-part feedback).
C2_STAGES = {"form_analyze", "draw_skeleton", "draw_errors"}

STAGE_LABELS = {
    "detect":        "Pose detection",
    "draw_overlay":  "Text overlay",
    "draw_errors":   "Form error text",
    "analyze_pose":  "Rep state machine",
    "draw_skeleton": "Coloured skeleton",
    "landmarks":     "Landmark extraction",
    "form_analyze":  "Form analysis",
}

# Fixed sleep in main.py's capture thread (time.sleep(0.01)).
CAPTURE_SLEEP_MS = 10.0


def _no_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


def _load_stage_means():
    rows = list(csv.DictReader(open(REPORTS / "latency_eval_stages.csv")))
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(float(r["mean_ms"]))
    return {k: st.mean(v) for k, v in by_stage.items()}


def _load_condition_means():
    rows = list(csv.DictReader(open(REPORTS / "latency_eval_results.csv")))
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(float(r["mean_ms"]))
    return {k: st.mean(v) for k, v in by_cond.items()}


def _observed_live_frame_ms():
    """Median live frame time from the real logged sessions."""
    rows = list(csv.DictReader(open(ROOT / "sessions" / "sessions_summary.csv")))
    # Drop the trivially short / zero-detection sessions -- they are start-up
    # artefacts, not workouts, and their FPS is not a steady-state reading.
    fps = [float(r["fps_median"]) for r in rows
           if int(r["total_frames"]) >= 100 and float(r["pose_detection_rate"]) > 0]
    return 1000.0 / st.median(fps), st.median(fps)


def fig8_latency_breakdown():
    stages = _load_stage_means()
    conds = _load_condition_means()
    frame_ms, live_fps = _observed_live_frame_ms()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- Panel A: per-stage compute cost -------------------------------
    ordered = sorted(stages.items(), key=lambda kv: kv[1])
    names = [STAGE_LABELS.get(k, k) for k, _ in ordered]
    values = [v for _, v in ordered]
    colors = [ORANGE if k in C2_STAGES else BLUE for k, _ in ordered]

    bars = ax1.barh(names, values, color=colors, height=0.62, zorder=3)
    total = sum(values)
    for bar, v in zip(bars, values):
        ax1.text(v + total * 0.012, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f} ms" if v >= 0.01 else "<0.01 ms",
                 va="center", ha="left", fontsize=9.5, color=SECONDARY_INK)

    ax1.set_xlabel("Mean cost per frame (ms)")
    ax1.set_xlim(0, max(values) * 1.22)
    ax1.grid(axis="x", zorder=0)
    ax1.grid(axis="y", visible=False)
    _no_top_right(ax1)
    ax1.set_title("Where the per-frame compute goes",
                  fontsize=12.5, color=INK, pad=12, loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax1.legend(handles, ["C2: per-body-part feedback", "Everything else"],
               loc="lower right", frameon=False, fontsize=9.5)

    c2_total = sum(v for k, v in stages.items() if k in C2_STAGES)
    ax1.text(0.99, 0.42,
             f"All three C2 stages together: {c2_total:.2f} ms\n"
             f"Marginal cost vs. the pre-C2 renderer: "
             f"{conds['full_c2'] - conds['baseline_render']:+.2f} ms",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=9.5, color=SECONDARY_INK)

    # ---- Panel B: live frame budget reconstruction ---------------------
    compute = conds["full_c2"]
    residual = max(0.0, frame_ms - compute - CAPTURE_SLEEP_MS)

    segments = [
        ("Compute (measured)", compute, BLUE),
        ("Capture-thread sleep", CAPTURE_SLEEP_MS, MUTED),
        ("Camera capture + display\n(residual, not measured)", residual, GRID),
    ]

    left = 0.0
    for label, width, color in segments:
        ax2.barh([0], [width], left=[left], color=color, height=0.34,
                 zorder=3, edgecolor="white", linewidth=2)
        if width > 14:
            # Wide enough to hold the label inside the segment.
            ax2.text(left + width / 2, 0, f"{width:.1f} ms",
                     ha="center", va="center", fontsize=10,
                     color=INK if color is GRID else "white", zorder=4)
        else:
            # Narrow segment: label above it rather than overflowing it.
            ax2.text(left + width / 2, 0.23, f"{width:.1f} ms",
                     ha="center", va="bottom", fontsize=9.5,
                     color=SECONDARY_INK, zorder=4)
        left += width

    ax2.axvline(100, color=CRITICAL, linestyle="--", linewidth=1.6, zorder=5)
    ax2.text(100 - 2.5, -0.30, "100 ms real-time target", rotation=90,
             ha="right", va="bottom", fontsize=9.5, color=CRITICAL)

    ax2.set_yticks([])
    ax2.set_ylim(-0.62, 0.62)
    ax2.set_xlim(0, 112)
    ax2.set_xlabel("Time per frame (ms)")
    ax2.grid(axis="x", zorder=0)
    ax2.grid(axis="y", visible=False)
    _no_top_right(ax2)
    ax2.set_title(f"Live frame budget at the observed {live_fps:.0f} fps",
                  fontsize=12.5, color=INK, pad=12, loc="left")

    handles2 = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segments]
    ax2.legend(handles2, [s[0].replace("\n", " ") for s in segments],
               loc="lower left", frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.0, -0.02))

    ax2.text(0.01, 0.94,
             f"Compute alone would allow {1000 / compute:.0f} fps.\n"
             f"The live rate is bound by capture and the fixed sleep,\n"
             f"not by the algorithm or by C2.",
             transform=ax2.transAxes, ha="left", va="top",
             fontsize=9.5, color=SECONDARY_INK)

    fig.suptitle(
        "Per-frame cost of the feedback pipeline (Track D, 2,618 frames x 3 conditions, 5 exercises)",
        fontsize=13.5, color=INK, y=0.99, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "fig8_latency_breakdown.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig8_latency_breakdown()
