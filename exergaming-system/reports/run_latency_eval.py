"""
Latency / per-stage cost evaluation (Track D).

Answers three questions the existing evaluation does not:

  1. What does C2 (per-body-part feedback) actually COST per frame?
     The deck claims "no measurable speed cost", but the only isolated
     number the system records -- SessionLogger's processing_time_ms --
     is captured in controller.py BEFORE FormAnalyzer.analyze() and
     draw_form_skeleton() run, so it excludes the entire C2 pipeline by
     construction. It cannot support (or refute) that claim. This script
     measures the marginal cost directly by running the same frames with
     C2 on and off.

  2. Where does the per-frame budget actually go? At the observed live
     rate of ~15 fps (66.7 ms/frame), roughly 20 ms is the recorded
     processing time and 10 ms is the fixed sleep in main.py's capture
     thread -- leaving ~36 ms unattributed. The deck attributes that gap
     to "camera input overhead + a 10ms background sleep"; this measures
     each stage instead of assuming.

  3. Is end-to-end per-frame latency under the 100 ms target set in
     Evaluation_Strategy.md? Never measured until now.

METHOD

Three conditions, each a separate full pass over the same clip so that
one condition's rendering cannot warm caches for another:

  detect_only     detect + landmark extraction + analyze_pose. This is
                  exactly the span controller.py currently reports as
                  processing_time_ms.
  baseline_render detect_only + plain white skeleton + text overlay.
                  This is the pre-C2 system: visual.py's own comment
                  records that the coloured form skeleton "replaces
                  plain white skeleton", so this is the honest
                  counterfactual, not a no-rendering strawman.
  full_c2         baseline_render with the coloured skeleton, plus
                  FormAnalyzer.analyze() and draw_form_errors(). This is
                  shipped behaviour.

  C2 marginal cost   = full_c2 - baseline_render
  total render cost  = full_c2 - detect_only

Camera capture is deliberately excluded: it is I/O, it varies with the
webcam, and it cannot be held constant across a repeatable run. What is
measured is the compute the system does per frame, which is the part the
C2 claim is about. The camera's contribution is what the frame-budget
arithmetic in the dissertation attributes to the residual.

Timestamps passed to analyze_pose() are on the VIDEO's own timeline
(frame_index / fps * 1000), not wall-clock -- same reason as
run_angle_distance_eval.py: this loop runs faster than real-time and
GenericExercise rejects reps faster than MIN_REP_DURATION_MS.

No PositionGate, matching Track B/C, so timings describe the same
pipeline those accuracy numbers came from.

OUTPUT
  latency_eval_results.csv   one row per (video, condition): frame counts,
                             mean/median/p95 total ms, derived fps ceiling
  latency_eval_stages.csv    one row per (video, stage): per-stage mean/
                             median/p95 ms, measured during the full_c2 pass
"""
import sys
import csv
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
from src.pose.detector import PoseDetector
from src.exercises.generic_exercise import GenericExercise
from src.feedback.visual import VisualFeedback
from src.feedback.form_errors import FormAnalyzer
from src.controller import EXERCISE_DEFINITIONS

VIDEOS_ROOT = Path(__file__).parents[1] / "Videos"
OUT_CSV = Path(__file__).parent / "latency_eval_results.csv"
OUT_STAGES_CSV = Path(__file__).parent / "latency_eval_stages.csv"

CONDITIONS = ("detect_only", "baseline_render", "full_c2")

# One clip per exercise from the hand-curated set, so the timing run covers
# every movement class the system supports (lower body, horizontal push,
# vertical push, pull, full body) rather than over-weighting one.
CLIPS = [
    ("Squats/man-home-squats.mp4",            "Learned Squat"),
    ("Pushups/Pushups-man-outside.mp4",       "Learned Push-up"),
    ("JumpingJacks/JumpingJacks-Man.mp4",     "Jumping Jack"),
    ("BicepCurl/BicepCurlIndividualHands.mp4", "BicepCurl"),
    ("ShoulderPress/ShoulderPressTraining.mov", "ShoulderPress"),
]


def _percentile(values, pct):
    """p-th percentile by nearest-rank. Avoids a numpy dependency here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def _summarise(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean":   round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95":    round(_percentile(values, 95), 2),
    }


def run_pass(video_path: Path, definition, condition: str):
    """
    Run one full pass over a clip under one condition.

    Returns (per_frame_total_ms, stage_times, frames, frames_with_pose, reps).
    stage_times is only populated for the full_c2 condition.
    """
    detector = PoseDetector()
    exercise = GenericExercise(definition)
    visual = VisualFeedback()
    form_analyzer = FormAnalyzer(definition)
    exercise.start()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    totals = []
    stages = {k: [] for k in
              ("detect", "landmarks", "analyze_pose", "form_analyze",
               "draw_skeleton", "draw_overlay", "draw_errors")}
    frames = frames_with_pose = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        # Timing starts here, after decode: decode is the file-mode stand-in
        # for camera capture and is excluded for the same reason.
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        pose_detected = detector.detect(frame)
        t_detect = time.perf_counter() - t0

        raw_lm = None
        landmarks = {}
        t_landmarks = 0.0
        if pose_detected:
            frames_with_pose += 1
            t0 = time.perf_counter()
            landmarks = detector.get_all_landmarks_dict(frame.shape)
            raw_lm = detector.get_raw_landmarks()
            t_landmarks = time.perf_counter() - t0

        t_analyze = 0.0
        if pose_detected:
            t0 = time.perf_counter()
            exercise.analyze_pose(landmarks, (frames / src_fps) * 1000)
            t_analyze = time.perf_counter() - t0

        info = exercise.get_info()

        t_form = t_skeleton = t_overlay = t_errors = 0.0

        if condition != "detect_only":
            form_errors = []
            is_tracking = False

            if condition == "full_c2" and pose_detected:
                current_state = info.get("state", "idle").upper()
                is_tracking = current_state == form_analyzer.definition.form.during_state
                t0 = time.perf_counter()
                form_errors = form_analyzer.analyze(
                    current_state,
                    exercise._form_tracked,
                    exercise._signal_values,
                )
                t_form = time.perf_counter() - t0

            t0 = time.perf_counter()
            if condition == "full_c2":
                visual.draw_form_skeleton(
                    frame, raw_lm, form_errors,
                    relevant_parts=form_analyzer.relevant_body_parts,
                    is_tracking=is_tracking,
                )
            else:
                # Pre-C2 behaviour: plain white MediaPipe skeleton.
                if pose_detected:
                    detector.draw_landmarks(frame)
            t_skeleton = time.perf_counter() - t0

            t0 = time.perf_counter()
            visual.draw_complete_overlay(
                frame, info, show_debug=True, definition=definition
            )
            t_overlay = time.perf_counter() - t0

            if condition == "full_c2" and form_errors:
                t0 = time.perf_counter()
                visual.draw_form_errors(frame, form_errors)
                t_errors = time.perf_counter() - t0

        totals.append((time.perf_counter() - t_start) * 1000)

        if condition == "full_c2":
            stages["detect"].append(t_detect * 1000)
            stages["landmarks"].append(t_landmarks * 1000)
            stages["analyze_pose"].append(t_analyze * 1000)
            stages["form_analyze"].append(t_form * 1000)
            stages["draw_skeleton"].append(t_skeleton * 1000)
            stages["draw_overlay"].append(t_overlay * 1000)
            stages["draw_errors"].append(t_errors * 1000)

    cap.release()
    return totals, stages, frames, frames_with_pose, info.get("rep_count", 0)


def main():
    rows = []
    stage_rows = []

    for rel_path, defn_name in CLIPS:
        video_path = VIDEOS_ROOT / rel_path
        if not video_path.exists():
            print(f"SKIP (missing): {rel_path}")
            continue
        definition = EXERCISE_DEFINITIONS.get(defn_name)
        if definition is None:
            print(f"SKIP (no definition '{defn_name}'): {rel_path}")
            continue

        per_condition = {}
        for condition in CONDITIONS:
            totals, stages, frames, with_pose, reps = run_pass(
                video_path, definition, condition
            )
            summary = _summarise(totals)
            per_condition[condition] = summary["mean"]

            rows.append({
                "video":            video_path.name,
                "exercise":         defn_name,
                "condition":        condition,
                "frames":           frames,
                "frames_with_pose": with_pose,
                "detection_rate":   round(100.0 * with_pose / frames, 1) if frames else 0.0,
                "reps":             reps,
                "mean_ms":          summary["mean"],
                "median_ms":        summary["median"],
                "p95_ms":           summary["p95"],
                "fps_ceiling":      round(1000.0 / summary["mean"], 1) if summary["mean"] else 0.0,
            })

            if condition == "full_c2":
                for stage, values in stages.items():
                    s = _summarise(values)
                    stage_rows.append({
                        "video":     video_path.name,
                        "exercise":  defn_name,
                        "stage":     stage,
                        "mean_ms":   s["mean"],
                        "median_ms": s["median"],
                        "p95_ms":    s["p95"],
                    })

            print(f"  {video_path.name:38} {condition:16} "
                  f"{summary['mean']:6.2f} ms  ({frames} frames)")

        c2_cost = per_condition["full_c2"] - per_condition["baseline_render"]
        render_cost = per_condition["full_c2"] - per_condition["detect_only"]
        print(f"  {'':38} {'-> C2 marginal':16} {c2_cost:6.2f} ms   "
              f"(all rendering: {render_cost:.2f} ms)\n")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_STAGES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stage_rows)

    print(f"Wrote {OUT_CSV.name} ({len(rows)} rows)")
    print(f"Wrote {OUT_STAGES_CSV.name} ({len(stage_rows)} rows)")


if __name__ == "__main__":
    main()
