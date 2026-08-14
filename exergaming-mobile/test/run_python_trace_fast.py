"""Same as run_python_trace.py but for the too-fast synthetic sequence."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "exergaming-system"))

from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise

FRAMES = json.loads((Path(__file__).parent / "synthetic_squat_frames_fast.json").read_text())
DEFN = ExerciseDefinition.from_json(
    str(Path(__file__).parents[2] / "exergaming-system/src/exercises/definitions/learned_squat.json")
)

exercise = GenericExercise(DEFN)
exercise.start()

# Assumed 60fps (not the normal-speed test's 30fps) -- compresses the same
# 5-sample-per-rep motion into 83ms/rep, comfortably under the 200ms
# MIN_REP_DURATION_MS floor. See generate_fast_frames.py for why frame
# COUNT couldn't just be reduced instead.
FPS = 60.0
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

out_path = Path(__file__).parent / "python_trace_fast.json"
out_path.write_text(json.dumps(trace, indent=2))
print(f"Wrote {len(trace)}-frame trace -> {out_path}")
print(f"Final rep count: {exercise.get_rep_count()} | rejected_fast_reps: {exercise.rejected_fast_reps}")
