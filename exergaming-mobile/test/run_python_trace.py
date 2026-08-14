"""
Replays synthetic_squat_frames.json through the REAL desktop-app
GenericExercise (learned_squat.json) and dumps a frame-by-frame trace for
the JS parity test to compare against.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "exergaming-system"))

from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise

FRAMES = json.loads((Path(__file__).parent / "synthetic_squat_frames.json").read_text())
DEFN = ExerciseDefinition.from_json(
    str(Path(__file__).parents[2] / "exergaming-system/src/exercises/definitions/learned_squat.json")
)

exercise = GenericExercise(DEFN)
exercise.start()

# 30fps synthetic timescale -- frames aren't free-running anymore now that
# GenericExercise rejects reps completing faster than MIN_REP_DURATION_MS
# (see analyze_pose()), so a real elapsed-time basis matters for the trace
# to mean anything.
FPS = 30.0

trace = []
for i, frame in enumerate(FRAMES):
    landmarks = {k: tuple(v) for k, v in frame.items()}
    timestamp_ms = i * (1000.0 / FPS)
    exercise.analyze_pose(landmarks, timestamp_ms)
    trace.append({
        "frame": i,
        "state": exercise.state.value.upper(),
        "rep_count": exercise.get_rep_count(),
        "form": exercise.form_feedback.value,
        "knee_angle": exercise._signal_values.get("knee_angle"),
        "rejected_fast_reps": exercise.rejected_fast_reps,
    })

out_path = Path(__file__).parent / "python_trace.json"
out_path.write_text(json.dumps(trace, indent=2))
print(f"Wrote {len(trace)}-frame trace -> {out_path}")
print(f"Final rep count: {exercise.get_rep_count()}")
