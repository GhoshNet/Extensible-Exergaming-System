"""
Angle/distance sensitivity evaluation (Track C) -- runs the 24 controlled
household recordings (4 exercises x 3 angles x 2 distances, 5 known reps
each) through the same core pipeline used for the Track A/B evaluations
(PoseDetector + GenericExercise, every frame, no sampling), WITHOUT
PositionGate -- this measures the raw algorithm's sensitivity to camera
angle/distance directly, not the gating layer's ability to suppress bad
input. Ground truth is not measured after the fact: each clip is a known,
controlled 5 reps (recorder-controlled), so no manual counting step here.

Timestamps passed to analyze_pose() are on the VIDEO's own timeline
(frame_index / fps * 1000), not wall-clock -- this loop runs much faster
than real-time, and GenericExercise now rejects reps completing faster
than MIN_REP_DURATION_MS (see generic_exercise.py), so a wall-clock
default would silently reject every real rep.
"""
import sys
import csv
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
from src.pose.detector import PoseDetector
from src.exercises.generic_exercise import GenericExercise
from src.controller import EXERCISE_DEFINITIONS

VIDEOS_ROOT = Path(__file__).parents[1] / "Videos" / "Evaluation-videos"
OUT_CSV = Path(__file__).parent / "angle_distance_eval_results.csv"

DISTANCE_FOLDERS = {
    "2Meters": "2m",
    "5Meters": "5m",
}

EXERCISE_MAP = {
    "squat": "Learned Squat",
    "pushup": "Learned Push-up",
    "bicepcurl": "BicepCurl",
    "jumpingjacks": "Jumping Jack",
}

GROUND_TRUTH_REPS = 5  # every clip: 5 controlled reps, recorder-known

FNAME_RE = re.compile(
    r"^(?:Dist_)?(?P<exercise>squat|pushup|bicepcurl|jumpingjacks)_angle_(?P<angle>\d+)\.(?:MOV|mov)$"
)


def discover_videos():
    videos = []
    for folder, distance in DISTANCE_FOLDERS.items():
        folder_path = VIDEOS_ROOT / folder
        for f in sorted(folder_path.glob("*.MOV")) + sorted(folder_path.glob("*.mov")):
            m = FNAME_RE.match(f.name)
            if not m:
                print(f"WARNING: filename doesn't match expected pattern, skipping: {f}")
                continue
            videos.append({
                "path": f,
                "exercise_key": m.group("exercise"),
                "angle": int(m.group("angle")),
                "distance": distance,
            })
    return videos


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
    print(f"Discovered {len(videos)} videos")
    by_exercise = {}
    for v in videos:
        by_exercise.setdefault(v["exercise_key"], 0)
        by_exercise[v["exercise_key"]] += 1
    for k, n in by_exercise.items():
        print(f"  {k}: {n}")

    detector = PoseDetector()
    results = []
    t0 = time.time()

    for i, v in enumerate(videos):
        exercise_name = EXERCISE_MAP[v["exercise_key"]]
        r = run_video(v["path"], exercise_name, detector)

        detected = r["detected_reps"]
        if detected is not None:
            count_error = detected - GROUND_TRUTH_REPS
            accuracy = max(0.0, (1 - abs(count_error) / GROUND_TRUTH_REPS) * 100)
            detection_rate = (r["frames_with_pose"] / r["frames_processed"] * 100) if r["frames_processed"] else 0
        else:
            count_error = None
            accuracy = None
            detection_rate = None

        row = {
            "exercise": v["exercise_key"],
            "angle": v["angle"],
            "distance": v["distance"],
            "file": v["path"].name,
            "ground_truth_reps": GROUND_TRUTH_REPS,
            "detected_reps": detected,
            "count_error": count_error,
            "accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
            "rejected_fast_reps": r["rejected_fast_reps"],
            "frames_processed": r["frames_processed"],
            "frames_with_pose": r["frames_with_pose"],
            "detection_rate": round(detection_rate, 2) if detection_rate is not None else None,
            "error": r["error"],
        }
        results.append(row)
        print(f"  [{i+1}/{len(videos)}] {v['exercise_key']:<13} angle={v['angle']:>2} dist={v['distance']:<3} "
              f"-> detected={detected} (gt={GROUND_TRUTH_REPS}) acc={row['accuracy_pct']}%")

    detector.cleanup()

    fieldnames = ["exercise", "angle", "distance", "file", "ground_truth_reps", "detected_reps",
                  "count_error", "accuracy_pct", "rejected_fast_reps", "frames_processed",
                  "frames_with_pose", "detection_rate", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")


if __name__ == "__main__":
    main()
