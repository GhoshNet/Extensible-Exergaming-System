"""
Track C, both threshold variants -- angle/distance sensitivity re-run.

Companion to run_angle_distance_eval.py, which is left untouched because it
produced the published angle_distance_eval_results.csv. This version scores
the same 24 controlled recordings (4 exercises x 3 angles x 2 distances, 5
known reps each) under BOTH learned-definition variants:

  original    the shipped definitions (min/max extremes)
  percentile  the same definitions re-derived at EXTREME_PERCENTILE=2.0

As in the Track B variants run, detection runs ONCE per frame and both
scorers are fed from it: MediaPipe output does not depend on thresholds, so
this halves runtime and guarantees both arms see identical landmarks, making
any difference attributable to the thresholds alone.

JUMPING JACKS HAVE NO PERCENTILE COUNTERPART. The jumping jack definition is
hardcoded (jumping_jack.json), not learned, so there is nothing to re-derive
and its two arms use the same definition. Its rows are therefore identical by
construction and are excluded from the variant delta -- they are kept in the
output so the factorial stays complete.

Same conditions as the published run: no PositionGate (this measures the raw
algorithm's sensitivity to camera geometry, not the gating layer), every
frame processed, and video-timeline timestamps rather than wall-clock.
"""
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
from src.pose.detector import PoseDetector
from src.exercises.generic_exercise import GenericExercise
from src.controller import EXERCISE_DEFINITIONS

VIDEOS_ROOT = Path(__file__).parents[1] / "Videos" / "Evaluation-videos"
OUT_CSV = Path(__file__).parent / "angle_distance_eval_both_variants.csv"

DISTANCE_FOLDERS = {"2Meters": "2m", "5Meters": "5m"}

# exercise key -> (original definition, percentile definition)
VARIANT_MAP = {
    "squat":        ("Learned Squat",   "Learned Squat (percentile)"),
    "pushup":       ("Learned Push-up", "Learned Push-up (percentile)"),
    "bicepcurl":    ("BicepCurl",       "BicepCurl (percentile)"),
    # Hardcoded, not learned -- same definition on both arms, see module docstring.
    "jumpingjacks": ("Jumping Jack",    "Jumping Jack"),
}
NO_PERCENTILE_COUNTERPART = {"jumpingjacks"}

GROUND_TRUTH_REPS = 5

FNAME_RE = re.compile(
    r"^(?:Dist_)?(?P<exercise>squat|pushup|bicepcurl|jumpingjacks)_angle_(?P<angle>\d+)\.(?:MOV|mov)$"
)


def discover_videos():
    videos = []
    for folder, distance in DISTANCE_FOLDERS.items():
        for f in sorted((VIDEOS_ROOT / folder).glob("*.MOV")) + \
                 sorted((VIDEOS_ROOT / folder).glob("*.mov")):
            m = FNAME_RE.match(f.name)
            if not m:
                print(f"WARNING: unexpected filename, skipping: {f.name}")
                continue
            videos.append({
                "path": f,
                "exercise_key": m.group("exercise"),
                "angle": int(m.group("angle")),
                "distance": distance,
            })
    return videos


def run_video_both(path: Path, exercise_key: str):
    """
    A FRESH PoseDetector is created per video, deliberately.

    PoseDetector leaves mp_pose.Pose's static_image_mode at its default of
    False, i.e. tracking mode, which carries temporal state across frames.
    Reusing one detector across the whole batch therefore leaks tracking
    state from the end of one clip into the start of the next, making
    results depend on processing order. The published Track C script
    constructs the detector once in main() and reuses it; that is the most
    likely source of the run-to-run variance documented in the diary entry
    for this session. Constructing per video costs a little startup time
    and removes the dependency entirely.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "cannot open"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = PoseDetector()

    exercises = {}
    for variant, defn_name in zip(("original", "percentile"), VARIANT_MAP[exercise_key]):
        ex = GenericExercise(EXERCISE_DEFINITIONS[defn_name])
        ex.start()
        exercises[variant] = ex

    frames = with_pose = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if detector.detect(frame):
            with_pose += 1
            ts = (frames / fps) * 1000
            for ex in exercises.values():
                ex.analyze_pose(detector.get_all_landmarks_dict(frame.shape), ts)

    cap.release()
    detector.cleanup()
    shared = {
        "frames_processed": frames,
        "frames_with_pose": with_pose,
        "detection_rate": round(100.0 * with_pose / frames, 2) if frames else 0.0,
    }
    return {v: {"detected_reps": e.get_rep_count(),
                "rejected_fast_reps": e.rejected_fast_reps,
                **shared}
            for v, e in exercises.items()}, ""


def main():
    videos = discover_videos()
    print(f"Discovered {len(videos)} videos\n")

    rows = []
    t0 = time.time()

    for i, v in enumerate(videos, 1):
        res, err = run_video_both(v["path"], v["exercise_key"])
        if err:
            print(f"  [{i}/{len(videos)}] ERROR {v['path'].name}: {err}")
            continue

        parts = []
        for variant in ("original", "percentile"):
            d = res[variant]
            count_error = d["detected_reps"] - GROUND_TRUTH_REPS
            accuracy = max(0.0, (1 - abs(count_error) / GROUND_TRUTH_REPS) * 100)
            rows.append({
                "variant": variant,
                "exercise": v["exercise_key"],
                "angle": v["angle"],
                "distance": v["distance"],
                "file": v["path"].name,
                "ground_truth_reps": GROUND_TRUTH_REPS,
                "detected_reps": d["detected_reps"],
                "count_error": count_error,
                "accuracy_pct": round(accuracy, 2),
                "rejected_fast_reps": d["rejected_fast_reps"],
                "frames_processed": d["frames_processed"],
                "frames_with_pose": d["frames_with_pose"],
                "detection_rate": d["detection_rate"],
                "has_percentile_counterpart":
                    v["exercise_key"] not in NO_PERCENTILE_COUNTERPART,
                "error": "",
            })
            parts.append(f"{variant[:4]}={d['detected_reps']}({accuracy:.0f}%)")

        flag = ""
        if v["exercise_key"] not in NO_PERCENTILE_COUNTERPART and \
                res["original"]["detected_reps"] != res["percentile"]["detected_reps"]:
            flag = " *"
        print(f"  [{i}/{len(videos)}] {v['path'].name:34} "
              f"{v['distance']:>3} {v['angle']:>3}deg  {'  '.join(parts)}{flag}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV.name} ({len(rows)} rows) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
