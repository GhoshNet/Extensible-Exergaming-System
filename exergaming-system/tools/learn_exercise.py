"""
learn_exercise.py — CLI tool to learn an ExerciseDefinition from a demo video.

Usage:
    python tools/learn_exercise.py --video <path> --name "Learned Squat" \\
        [--output src/exercises/definitions/learned_squat.json] \\
        [--compare Squat] [--test-video <path> --ground-truth N] \\
        [--no-confirm]

The tool:
  1. Runs pose detection frame-by-frame on the video
  2. Captures all candidate joint angles via DemonstrationCapture
  3. Shows the detected signals and lets the user confirm / deselect any
     that moved incidentally (e.g. arms swinging during squats)   [NEW]
  4. Runs ExerciseLearner with the user-approved signal list
  5. Saves the definition to the output JSON path
  6. Optionally compares learned vs. an existing hardcoded definition
  7. Optionally runs the learned definition on a test video
"""
import argparse
import sys
import os
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2

from src.pose.detector import PoseDetector
from src.learning.demonstration_capture import DemonstrationCapture, CANDIDATE_RATIO_SIGNALS
from src.learning.exercise_learner import ExerciseLearner
from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise

# Ratio signal names — displayed without a degree symbol
_RATIO_SIGNAL_NAMES = {r[0] for r in CANDIDATE_RATIO_SIGNALS}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — video analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(video_path: str, exercise_name: str):
    """
    Run pose detection on video_path and return (learner, report).
    Does NOT yet build the ExerciseDefinition — that happens after confirmation.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_source   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\nVideo : {video_path}")
    print(f"Frames: {total_frames}  |  FPS: {fps_source:.1f}  |  Exercise: '{exercise_name}'")
    print()

    detector = PoseDetector()
    capture  = DemonstrationCapture()
    capture.start_recording()

    frame_num = 0
    detected  = 0
    print("Running pose detection... ", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        pose_ok = detector.detect(frame)
        landmarks = {}
        if pose_ok:
            detected += 1
            landmarks = detector.get_all_landmarks_dict(frame.shape)
        capture.process_frame(frame_num, landmarks, pose_ok)

    cap.release()
    detector.cleanup()

    pct = detected / frame_num * 100 if frame_num > 0 else 0
    print(f"done. {detected}/{frame_num} frames detected ({pct:.1f}%)")

    frames  = capture.stop_recording()
    learner = ExerciseLearner(frames, exercise_name)
    report  = learner.get_analysis_report()
    return learner, report


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — interactive signal confirmation  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def interactive_confirm(report: dict) -> List[str]:
    """
    Show the algorithm-detected signals and let the user review them.

    The user can deselect signals that moved incidentally (e.g. arms swinging
    during squats) before the definition is built. This is the human-in-the-loop
    correction step that addresses the known limitation of variance-based detection.

    Returns the user-approved signal list (subset of report['important_signals']).
    """
    important  = report["important_signals"]
    ranges_map = report["signal_ranges"]
    peaks_map  = report["signal_peaks"]
    troughs_map= report["signal_troughs"]

    if not important:
        return []

    selected = set(important)   # start with everything selected

    def _get_primary(sig_list: List[str]) -> Optional[str]:
        """Primary = highest-range non-ratio signal in the list."""
        for s in sig_list:
            if s not in _RATIO_SIGNAL_NAMES:
                return s
        return sig_list[0] if sig_list else None

    def _render():
        """Print the current signal table."""
        current_selected = [s for s in important if s in selected]
        primary = _get_primary(current_selected)

        print()
        print("  " + "─" * 68)
        print("  SIGNAL REVIEW")
        print("  The algorithm detected the signals below by measuring how much")
        print("  each joint moved. Signals that moved a lot but aren't relevant")
        print("  to the exercise (e.g. arms swinging during squats) should be")
        print("  deselected before the definition is built.")
        print("  " + "─" * 68)
        print(
            f"\n  {'#':>2}  {'Signal':<22}  {'Range':>8}  "
            f"{'Peak':>7}  {'Trough':>7}  Status"
        )
        print(
            f"  {'──':>2}  {'──────────────────────':<22}  {'────────':>8}  "
            f"{'───────':>7}  {'───────':>7}  ──────"
        )

        for i, name in enumerate(important, 1):
            rng    = ranges_map.get(name, 0)
            peak   = peaks_map.get(name, 0)
            trough = troughs_map.get(name, 0)
            unit   = "" if name in _RATIO_SIGNAL_NAMES else "°"

            tick   = "✓" if name in selected else "✗ (deselected)"
            tags   = []
            if name == primary and name in selected:
                tags.append("PRIMARY")
            # Flag signals that look like incidental motion: high range but not the top-2
            if i > 2 and name in selected and name not in _RATIO_SIGNAL_NAMES:
                tags.append("incidental?")
            tag_str = "  [" + ", ".join(tags) + "]" if tags else ""

            print(
                f"  {i:>2}  {name:<22}  {rng:>7.1f}{unit}  "
                f"{peak:>7.1f}  {trough:>7.1f}  {tick}{tag_str}"
            )

        n_sel = len(current_selected)
        print(f"\n  {n_sel} of {len(important)} signals selected.")
        if primary:
            print(f"  Primary signal (drives state machine): {primary}")

    while True:
        _render()
        print()
        print("  Enter a number to toggle a signal, or press Enter to confirm: ", end="", flush=True)

        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            # On interrupt, return whatever is currently selected
            return [s for s in important if s in selected]

        if raw == "":
            # Confirm
            approved = [s for s in important if s in selected]
            if not approved:
                print("\n  ✗  At least one signal must remain. Please select at least one.")
                continue
            print(f"\n  Confirmed {len(approved)} signal(s): {', '.join(approved)}")
            return approved

        # Toggle by number
        if not raw.isdigit():
            print(f"  ✗  Enter a number between 1 and {len(important)}, or press Enter.")
            continue

        idx = int(raw) - 1
        if idx < 0 or idx >= len(important):
            print(f"  ✗  Number out of range. Enter 1–{len(important)}.")
            continue

        name = important[idx]

        if name in selected:
            # Deselect — check we won't empty the list
            remaining = [s for s in important if s in selected and s != name]
            if not remaining:
                print(f"\n  ✗  Cannot deselect '{name}' — it is the only remaining signal.")
                continue
            selected.discard(name)
            new_primary = _get_primary(remaining)
            old_primary = _get_primary([s for s in important if s in selected | {name}])
            if name == old_primary and new_primary:
                print(f"\n  Note: primary signal changed to '{new_primary}'.")
        else:
            selected.add(name)
            print(f"\n  Added '{name}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — build definition and save
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save(
    learner: ExerciseLearner,
    approved_signals: List[str],
    output_path: str,
) -> ExerciseDefinition:
    """Build the ExerciseDefinition from approved signals, display it, and save."""
    print("\nBuilding exercise definition from approved signals...")
    definition = learner.learn(approved_signals=approved_signals)

    print("\nLearned state transitions:")
    for t in definition.transitions:
        conds = " AND ".join(
            f"{c.signal} {'>' if c.direction == 'above' else '<'} {c.threshold}"
            for c in t.conditions
        )
        print(f"  {','.join(t.from_states):<14} → {t.to_state:<6}  when  {conds}")

    print(
        f"\nForm tracking: {definition.form.tracking} of "
        f"{', '.join(definition.form.tracked_signals)} during {definition.form.during_state}"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    definition.to_json(output_path)
    print(f"\nSaved to: {output_path}")
    return definition


# ─────────────────────────────────────────────────────────────────────────────
# Optional steps — comparison and rep-count validation
# ─────────────────────────────────────────────────────────────────────────────

def compare_to_hardcoded(learned: ExerciseDefinition, hardcoded_name: str) -> None:
    from pathlib import Path
    definitions_dir = Path(_ROOT) / "src" / "exercises" / "definitions"
    hardcoded = None
    for path in sorted(definitions_dir.glob("*.json")):
        if "learned_" not in path.name:
            try:
                d = ExerciseDefinition.from_json(str(path))
                if d.name.lower() == hardcoded_name.lower():
                    hardcoded = d
                    break
            except Exception:
                pass

    if hardcoded is None:
        print(f"\nCould not find hardcoded definition for '{hardcoded_name}'.")
        return

    print(f"\n{'='*62}")
    print(f"  Comparison: '{hardcoded.name}' (hardcoded)  vs.  '{learned.name}' (learned)")
    print(f"{'='*62}")

    def _summary(defn):
        out = {}
        for t in defn.transitions:
            key = f"→{t.to_state}"
            out[key] = " AND ".join(
                f"{c.signal} {'>' if c.direction == 'above' else '<'} {c.threshold}"
                for c in t.conditions
            )
        return out

    h = _summary(hardcoded)
    l = _summary(learned)
    for key in sorted(set(h) | set(l)):
        match = "✓" if h.get(key) == l.get(key) else "≠"
        print(f"  {key:8}  hardcoded: {h.get(key,'—'):<35}  learned: {l.get(key,'—')}  {match}")

    print(f"\n  Form (hardcoded): {hardcoded.form.tracking} of "
          f"{', '.join(hardcoded.form.tracked_signals)} during {hardcoded.form.during_state}")
    print(f"  Form (learned):   {learned.form.tracking} of "
          f"{', '.join(learned.form.tracked_signals)} during {learned.form.during_state}")


def run_rep_test(definition: ExerciseDefinition, video_path: str, ground_truth: int) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open test video: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\nRep-count test: {video_path}  ({total} frames)")

    detector = PoseDetector()
    exercise = GenericExercise(definition)
    exercise.start()

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if detector.detect(frame):
            # Video's own timeline, not wall-clock -- this loop runs much
            # faster than real-time.
            video_timestamp_ms = (frame_index / src_fps) * 1000
            exercise.analyze_pose(detector.get_all_landmarks_dict(frame.shape), video_timestamp_ms)

    cap.release()
    detector.cleanup()

    reps = exercise.get_info()["rep_count"]
    err  = abs(reps - ground_truth)
    pct  = (1 - err / max(ground_truth, 1)) * 100
    print(f"  Detected: {reps}  |  Ground truth: {ground_truth}  |  "
          f"Accuracy: {pct:.1f}%  |  Error: {err} reps")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Learn an ExerciseDefinition from a demonstration video."
    )
    parser.add_argument("--video",       required=True,
                        help="Path to demonstration video")
    parser.add_argument("--name",        required=True,
                        help="Name for the learned exercise (e.g. 'Learned Squat')")
    parser.add_argument("--output",      default=None,
                        help="Output JSON path (default: definitions/learned_<slug>.json)")
    parser.add_argument("--compare",     default=None,
                        help="Hardcoded exercise name to compare against (e.g. 'Squat')")
    parser.add_argument("--test-video",  default=None,
                        help="Video to validate rep counting on after learning")
    parser.add_argument("--ground-truth",type=int, default=None,
                        help="Known rep count for the test video")
    parser.add_argument("--no-confirm",  action="store_true",
                        help="Skip the interactive signal review (use auto-detected signals)")
    args = parser.parse_args()

    # Default output path
    output = args.output
    if output is None:
        slug = args.name.lower().replace(" ", "_").replace("-", "_").replace("learned_", "")
        output = os.path.join(_ROOT, "src", "exercises", "definitions", f"learned_{slug}.json")

    # --- Analysis ---
    learner, report = run_analysis(args.video, args.name)

    print(f"\nAnalysis complete.")
    print(f"  Detected frames  : {report['total_frames']}")
    print(f"  All signals      : {', '.join(report['all_signals'])}")
    print(f"  Important signals: {', '.join(report['important_signals'])}")

    # --- Signal confirmation ---
    if args.no_confirm:
        approved = report["important_signals"]
        print(f"\n--no-confirm: using all {len(approved)} auto-detected signals.")
    else:
        approved = interactive_confirm(report)

    # --- Build and save ---
    learned = build_and_save(learner, approved, output)

    # --- Optional steps ---
    if args.compare:
        compare_to_hardcoded(learned, args.compare)

    if args.test_video:
        run_rep_test(learned, args.test_video, args.ground_truth or 0)

    print("\nDone.")


if __name__ == "__main__":
    main()
