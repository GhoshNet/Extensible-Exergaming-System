"""
Derives the left-side-only variant of synthetic_squat_frames.json (all
RIGHT_* landmarks stripped, to exercise the bilateral single-side fallback
path) and replays it through the real GenericExercise, same as
run_python_trace.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "exergaming-system"))

from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise

HERE = Path(__file__).parent
frames = json.loads((HERE / "synthetic_squat_frames.json").read_text())
left_only = [{k: v for k, v in f.items() if not k.startswith("RIGHT_")} for f in frames]
(HERE / "synthetic_squat_frames_leftonly.json").write_text(json.dumps(left_only))
print(f"Wrote {len(left_only)} left-only frames")

DEFN = ExerciseDefinition.from_json(
    str(Path(__file__).parents[2] / "exergaming-system/src/exercises/definitions/learned_squat.json")
)
exercise = GenericExercise(DEFN)
exercise.start()

FPS = 30.0
trace = []
for i, frame in enumerate(left_only):
    landmarks = {k: tuple(v) for k, v in frame.items()}
    timestamp_ms = i * (1000.0 / FPS)
    exercise.analyze_pose(landmarks, timestamp_ms)
    trace.append({
        "frame": i,
        "state": exercise.state.value.upper(),
        "rep_count": exercise.get_rep_count(),
        "form": exercise.form_feedback.value,
        "knee_angle": exercise._signal_values.get("knee_angle"),
        "side_used": exercise._last_side_used,
        "rejected_fast_reps": exercise.rejected_fast_reps,
    })

out_path = HERE / "python_trace_leftonly.json"
out_path.write_text(json.dumps(trace, indent=2))
print(f"Wrote {len(trace)}-frame trace -> {out_path}")
print(f"Final rep count: {exercise.get_rep_count()}")
