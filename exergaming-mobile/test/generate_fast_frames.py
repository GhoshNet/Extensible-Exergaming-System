"""
Same synthetic-squat generator as generate_synthetic_frames.py, but with a
period short enough that each 'rep' completes well under
MIN_REP_DURATION_MS (200ms, as of 2026-07-21) once timestamped, to verify
the too-fast-reps-aren't-counted behavior matches between Python and JS.

Frame COUNT per rep (5) is unchanged from the original design -- that's
the motion resolution needed to actually dip below the knee_angle DOWN
threshold (verified: 3 frames/rep undersamples the cosine curve and never
reaches the threshold at all, producing zero reps AND zero rejections --
a silently broken test, not a passing one). Instead, the consuming
scripts (run_python_trace_fast.py, parity-test-fast.mjs) timestamp these
frames at an assumed 60fps instead of the normal-speed test's 30fps, which
compresses the same 5-sample motion into 83ms/rep -- comfortably under
the 200ms floor -- without changing the geometry.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).parent / "synthetic_squat_frames_fast.json"

N_REPS = 3
FRAMES_PER_REP = 5
MAX_THETA = 110

def theta_at(t_in_rep):
    return MAX_THETA * (1 - math.cos(2 * math.pi * t_in_rep)) / 2

frames = []
total_frames = N_REPS * FRAMES_PER_REP
for i in range(total_frames):
    t_in_rep = (i % FRAMES_PER_REP) / FRAMES_PER_REP
    theta = math.radians(theta_at(t_in_rep))

    hip = [0.0, 0.0]
    knee = [0.0, 150.0]
    ankle = [150 * math.sin(theta), 150.0 + 150 * math.cos(theta)]
    shoulder = [0.0, -150.0]
    elbow = [30.0, -80.0]
    wrist = [30.0, -20.0]
    foot_index = [ankle[0] + 40.0, ankle[1] + 10.0]

    frame = {}
    for side in ("LEFT", "RIGHT"):
        frame[f"{side}_HIP"] = hip
        frame[f"{side}_KNEE"] = knee
        frame[f"{side}_ANKLE"] = ankle
        frame[f"{side}_SHOULDER"] = shoulder
        frame[f"{side}_ELBOW"] = elbow
        frame[f"{side}_WRIST"] = wrist
        frame[f"{side}_FOOT_INDEX"] = foot_index
    frames.append(frame)

OUT.write_text(json.dumps(frames))
print(f"Wrote {len(frames)} fast synthetic frames ({N_REPS} reps @ {FRAMES_PER_REP} frames/rep) -> {OUT}")
