"""
Generates a synthetic squat motion as raw landmark-name -> [x, y] frames,
shared by both the Python parity trace and the JS parity trace so they're
running the exact same input.

Only knee_angle actually drives learned_squat.json's transitions and form
tracking, so that's the signal that's precisely parametrized (hip/ankle/
elbow/ratio landmarks are present but geometrically simple placeholders --
their exact values don't affect this exercise's rep counting).

Geometry: HIP fixed at origin, KNEE fixed directly below HIP (vertex),
ANKLE swings out at angle theta from vertical below the KNEE. This makes
knee_angle = 180 - theta exactly (via the law-of-cosines angle formula),
so driving theta from 0 -> 110 -> 0 sweeps knee_angle from 180 (standing)
down to 70 (deep squat) and back -- one clean rep per period.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).parent / "synthetic_squat_frames.json"

N_REPS = 3
FRAMES_PER_REP = 40
MAX_THETA = 110  # degrees -> knee_angle bottoms out at 70

def theta_at(t_in_rep):
    # smooth 0 -> MAX_THETA -> 0 across one rep, t_in_rep in [0, 1)
    return MAX_THETA * (1 - math.cos(2 * math.pi * t_in_rep)) / 2

frames = []
total_frames = N_REPS * FRAMES_PER_REP
for i in range(total_frames):
    t_in_rep = (i % FRAMES_PER_REP) / FRAMES_PER_REP
    theta_deg = theta_at(t_in_rep)
    theta = math.radians(theta_deg)

    hip = [0.0, 0.0]
    knee = [0.0, 150.0]
    ankle = [150 * math.sin(theta), 150.0 + 150 * math.cos(theta)]

    # simple fixed placeholders for the other joints (identical L/R, don't
    # affect squat's rep counting since only knee_angle drives it)
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
print(f"Wrote {len(frames)} synthetic frames ({N_REPS} reps) -> {OUT}")
