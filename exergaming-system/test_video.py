"""
Video Test Script — Validate exercise detection on pre-recorded videos.

Bypasses the camera and live controller entirely. Feeds video frames directly
into pose detection and exercise logic, then reports rep accuracy and form stats.

Usage:
    python test_video.py <video> --exercise <name> [--ground-truth N] [--preview] [--speed N]

Examples:
    python test_video.py squats.mp4 --exercise Squat --ground-truth 20 --preview
    python test_video.py pushups.mp4 --exercise Push-up --ground-truth 15
    python test_video.py jacks.mp4 --exercise "Jumping Jack" --preview
"""

import argparse
import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pose.detector import PoseDetector
from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise
from src.feedback.visual import VisualFeedback
from src.controller import EXERCISE_DEFINITIONS

# Build exercise map from loaded definitions (includes learned definitions)
EXERCISE_MAP = EXERCISE_DEFINITIONS


def accuracy_str(detected: int, ground_truth: int) -> str:
    pct = max(0.0, (1 - abs(detected - ground_truth) / ground_truth) * 100)
    return f"{pct:.1f}%"


def run_test(video_path: str, exercise_name: str, ground_truth: int,
             preview: bool, speed: int) -> None:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / src_fps

    print(f"\n{'='*60}")
    print(f"  Video Test: {exercise_name}")
    print(f"{'='*60}")
    print(f"  File      : {video_path}")
    print(f"  Video     : {width}x{height} @ {src_fps:.1f} fps  ({duration:.1f}s, {total_frames} frames)")
    print(f"  Sampling  : every {speed} frame(s)")
    if ground_truth > 0:
        print(f"  Expected  : {ground_truth} reps")
    print(f"{'='*60}\n")

    exercise       = GenericExercise(EXERCISE_MAP[exercise_name])
    pose_detector  = PoseDetector()
    visual         = VisualFeedback()

    exercise.start()

    frame_index       = 0
    frames_processed  = 0
    frames_with_pose  = 0
    prev_reps         = 0
    rep_form_log: list[str] = []
    start_time        = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % speed != 0:
                continue

            frames_processed += 1
            pose_detected = pose_detector.detect(frame)

            if pose_detected:
                frames_with_pose += 1
                frame = pose_detector.draw_landmarks(frame)
                landmarks = pose_detector.get_all_landmarks_dict(frame.shape)
                # Video's own timeline, not wall-clock -- this loop runs much
                # faster than real-time, so time.perf_counter() would make
                # every rep look impossibly fast (see MIN_REP_DURATION_MS).
                video_timestamp_ms = (frame_index / src_fps) * 1000
                exercise.analyze_pose(landmarks, video_timestamp_ms)

            current_reps = exercise.get_rep_count()
            if current_reps > prev_reps:
                form = exercise.get_form_feedback().value
                rep_form_log.append(form)
                prev_reps = current_reps
                print(f"  Rep {current_reps:2d}  —  form: {form.upper()}")

            if preview:
                display = frame.copy()
                visual.draw_complete_overlay(display, exercise.get_info(), show_debug=True,
                                             definition=exercise.definition)
                progress = frame_index / max(total_frames, 1)
                bar_w    = width - 20
                cv2.rectangle(display, (10, height - 20), (10 + bar_w, height - 8), (60, 60, 60), -1)
                cv2.rectangle(display, (10, height - 20),
                              (10 + int(bar_w * progress), height - 8), (0, 200, 0), -1)
                cv2.imshow(f"Test: {exercise_name}", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n  [quit early]")
                    break

    except KeyboardInterrupt:
        print("\n  [interrupted]")
    finally:
        cap.release()
        pose_detector.cleanup()
        if preview:
            cv2.destroyAllWindows()

    elapsed      = time.time() - start_time
    detected     = exercise.get_rep_count()
    detect_rate  = frames_with_pose / frames_processed * 100 if frames_processed else 0
    proc_fps     = frames_processed / elapsed if elapsed else 0

    form_counts  = {r: rep_form_log.count(r) for r in ('excellent', 'good', 'fair', 'poor')}

    print(f"\n{'='*60}")
    print(f"  RESULTS: {exercise_name}")
    print(f"{'='*60}")
    print(f"  Reps detected   : {detected}")
    if ground_truth > 0:
        print(f"  Reps expected   : {ground_truth}")
        print(f"  Accuracy        : {accuracy_str(detected, ground_truth)}")
        diff = detected - ground_truth
        if diff == 0:
            print(f"  Count error     : 0  ✓ perfect")
        elif diff > 0:
            print(f"  Count error     : +{diff} (over-counted by {diff})")
        else:
            print(f"  Count error     : {diff} (under-counted by {abs(diff)})")
    print()
    print(f"  Pose detection  : {frames_with_pose}/{frames_processed} frames ({detect_rate:.1f}%)")
    print(f"  Processing speed: {proc_fps:.1f} fps")
    print(f"  Wall time       : {elapsed:.1f}s")
    print()
    if rep_form_log:
        print(f"  Form breakdown ({detected} reps):")
        for rating in ('excellent', 'good', 'fair', 'poor'):
            n = form_counts[rating]
            if n:
                bar = '█' * n
                pct = n / detected * 100
                print(f"    {rating.capitalize():<10} {bar}  {n}  ({pct:.0f}%)")
    else:
        print("  Form: no reps detected")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test exercise detection accuracy on a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video",
                        help="Path to video file (.mp4, .mov, .avi, …)")
    parser.add_argument("--exercise", "-e", required=True,
                        choices=list(EXERCISE_DEFINITIONS.keys()),
                        help="Exercise type to test")
    parser.add_argument("--ground-truth", "-g", type=int, default=0, metavar="N",
                        help="Your manually-counted rep total (enables accuracy output)")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="Show video with pose overlay while processing (press Q to quit early)")
    parser.add_argument("--speed", "-s", type=int, default=1, metavar="N",
                        help="Process every Nth frame — use 2 or 3 to speed up long videos (default: 1)")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: file not found: {args.video}")
        sys.exit(1)
    if args.speed < 1:
        print("Error: --speed must be >= 1")
        sys.exit(1)

    run_test(args.video, args.exercise, args.ground_truth, args.preview, args.speed)


if __name__ == "__main__":
    main()
