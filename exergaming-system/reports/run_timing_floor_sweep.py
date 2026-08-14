"""
Sweeps MIN_REP_DURATION_MS across 100/200/400ms and reruns all 24
angle/distance videos at each threshold, to check whether lowering the
floor fixes jumping jacks (see angle_distance_eval_results.csv finding:
its rejected reps cluster at 233-393ms, just under 400ms) WITHOUT
introducing over-counting (spurious fast "reps" from noise) on the other
three exercises.

Monkey-patches the module-level constant directly rather than adding a
constructor parameter -- this is a one-off sweep for a decision, not a
permanent API change (that's a separate decision: global constant vs.
per-exercise config, still open).
"""
import sys
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import src.exercises.generic_exercise as ge_module
from src.pose.detector import PoseDetector
from src.exercises.generic_exercise import GenericExercise
from src.controller import EXERCISE_DEFINITIONS
from run_angle_distance_eval import discover_videos, EXERCISE_MAP, GROUND_TRUTH_REPS

OUT_CSV = Path(__file__).parent / "timing_floor_sweep_results.csv"
THRESHOLDS_MS = [100.0, 200.0, 400.0]


def run_video(path: Path, exercise_name: str, detector: PoseDetector) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot open", "frames_processed": 0, "frames_with_pose": 0,
                "detected_reps": None, "rejected_fast_reps": None}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    exercise = GenericExercise(EXERCISE_DEFINITIONS[exercise_name])
    exercise.start()

    frames_processed = 0
    frames_with_pose = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
        if detector.detect(frame):
            frames_with_pose += 1
            landmarks = detector.get_all_landmarks_dict(frame.shape)
            video_timestamp_ms = (frames_processed / fps) * 1000
            exercise.analyze_pose(landmarks, video_timestamp_ms)

    cap.release()
    return {
        "error": "",
        "frames_processed": frames_processed,
        "frames_with_pose": frames_with_pose,
        "detected_reps": exercise.get_rep_count(),
        "rejected_fast_reps": exercise.rejected_fast_reps,
    }


def main():
    videos = discover_videos()
    print(f"Discovered {len(videos)} videos, sweeping thresholds: {THRESHOLDS_MS}")

    _original_threshold = ge_module.MIN_REP_DURATION_MS
    detector = PoseDetector()
    results = []
    t0 = time.time()

    for threshold in THRESHOLDS_MS:
        ge_module.MIN_REP_DURATION_MS = threshold
        print(f"\n=== threshold = {threshold:.0f}ms ===")
        for i, v in enumerate(videos):
            exercise_name = EXERCISE_MAP[v["exercise_key"]]
            r = run_video(v["path"], exercise_name, detector)

            detected = r["detected_reps"]
            if detected is not None:
                count_error = detected - GROUND_TRUTH_REPS
                accuracy = max(0.0, (1 - abs(count_error) / GROUND_TRUTH_REPS) * 100)
            else:
                count_error = None
                accuracy = None

            row = {
                "threshold_ms": int(threshold),
                "exercise": v["exercise_key"],
                "angle": v["angle"],
                "distance": v["distance"],
                "file": v["path"].name,
                "ground_truth_reps": GROUND_TRUTH_REPS,
                "detected_reps": detected,
                "count_error": count_error,
                "accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
                "rejected_fast_reps": r["rejected_fast_reps"],
                "error": r["error"],
            }
            results.append(row)
            print(f"  [{i+1}/{len(videos)}] {v['exercise_key']:<13} angle={v['angle']:>2} dist={v['distance']:<3} "
                  f"-> detected={detected} (gt={GROUND_TRUTH_REPS}) acc={row['accuracy_pct']}% rejected={r['rejected_fast_reps']}")

    detector.cleanup()
    ge_module.MIN_REP_DURATION_MS = _original_threshold  # restore, in case anything else imports this process

    fieldnames = ["threshold_ms", "exercise", "angle", "distance", "file", "ground_truth_reps",
                  "detected_reps", "count_error", "accuracy_pct", "rejected_fast_reps", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")


if __name__ == "__main__":
    main()
