"""
Direct unit tests of PositionGate, mirroring
files/exergaming-mobile/test/position-gate-test.mjs so both engines agree
on the same scenarios. Not part of an automated pytest suite (none exists
in this repo yet) -- run directly: python tests_position_gate.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")

from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise
from src.pose.position_gate import PositionGate

DEFDIR = "src/exercises/definitions"
FAKE_FRAME = np.full((1280, 720, 3), 128, dtype=np.uint8)  # mid-gray, passes lighting check


def load(name):
    return ExerciseDefinition.from_json(f"{DEFDIR}/{name}")


def full_squat_landmarks():
    return {
        "LEFT_HIP": (300, 300), "RIGHT_HIP": (340, 300),
        "LEFT_KNEE": (300, 450), "RIGHT_KNEE": (340, 450),
        "LEFT_ANKLE": (300, 600), "RIGHT_ANKLE": (340, 600),
        "LEFT_SHOULDER": (300, 150), "RIGHT_SHOULDER": (340, 150),
        "LEFT_ELBOW": (280, 250), "RIGHT_ELBOW": (360, 250),
        "LEFT_WRIST": (270, 350), "RIGHT_WRIST": (370, 350),
        "LEFT_FOOT_INDEX": (300, 630), "RIGHT_FOOT_INDEX": (340, 630),
    }


def bicep_curl_landmarks(include_lower=True):
    lm = {
        "LEFT_HIP": (300, 300), "RIGHT_HIP": (340, 300),
        "LEFT_SHOULDER": (300, 150), "RIGHT_SHOULDER": (340, 150),
        "LEFT_ELBOW": (280, 250), "RIGHT_ELBOW": (360, 250),
        "LEFT_WRIST": (270, 350), "RIGHT_WRIST": (370, 350),
    }
    if include_lower:
        lm.update({
            "LEFT_KNEE": (300, 450), "RIGHT_KNEE": (340, 450),
            "LEFT_ANKLE": (300, 600), "RIGHT_ANKLE": (340, 600),
        })
    return lm


failures = 0


def check(name, cond):
    global failures
    if cond:
        print(f"  ok - {name}")
    else:
        failures += 1
        print(f"  FAIL - {name}")


print("1. Debounce: a single bad frame should NOT gate")
ex = GenericExercise(load("learned_squat.json"))
ex.start()
gate = PositionGate(ex)
gate.calibrate(full_squat_landmarks())
r1 = gate.check(FAKE_FRAME, full_squat_landmarks(), True, 0)
check("frame 0 (good) not gated", r1.gated is False)
r2 = gate.check(FAKE_FRAME, {}, False, 50)
check("frame 1 (single blip) not gated", r2.gated is False)
check("frame 1 still reports a warning", len(r2.warnings) > 0)
r3 = gate.check(FAKE_FRAME, full_squat_landmarks(), True, 100)
check("frame 2 (recovered) not gated, warnings clear", r3.gated is False and len(r3.warnings) == 0)

print("2. Sustained failure (>400ms) SHOULD gate")
ex = GenericExercise(load("learned_squat.json"))
ex.start()
gate = PositionGate(ex)
gate.calibrate(full_squat_landmarks())
gate.check(FAKE_FRAME, {}, False, 0)
gate.check(FAKE_FRAME, {}, False, 200)
r = gate.check(FAKE_FRAME, {}, False, 450)
check("sustained no-person gates after grace period", r.gated is True)
check("warning mentions no person detected", any("No person detected" in w for w in r.warnings))

print("3. Bicep curl should NOT require lower body")
ex = GenericExercise(load("learned_bicepcurl.json"))
ex.start()
gate = PositionGate(ex)
gate.calibrate(bicep_curl_landmarks(include_lower=True))
gate.check(FAKE_FRAME, bicep_curl_landmarks(include_lower=False), True, 0)
r = gate.check(FAKE_FRAME, bicep_curl_landmarks(include_lower=False), True, 500)
check("bicep curl with no lower body visible is NOT gated", r.gated is False)

print("4. Bicep curl SHOULD require elbow/wrist")
ex = GenericExercise(load("learned_bicepcurl.json"))
ex.start()
gate = PositionGate(ex)
full = bicep_curl_landmarks(include_lower=True)
gate.calibrate(full)
no_arms = {"LEFT_HIP": full["LEFT_HIP"], "RIGHT_HIP": full["RIGHT_HIP"],
           "LEFT_SHOULDER": full["LEFT_SHOULDER"], "RIGHT_SHOULDER": full["RIGHT_SHOULDER"]}
gate.check(FAKE_FRAME, no_arms, True, 0)
r = gate.check(FAKE_FRAME, no_arms, True, 500)
check("bicep curl with elbows/wrists missing IS gated", r.gated is True)
check("warning mentions elbows/wrists", any("elbow" in w.lower() or "wrist" in w.lower() for w in r.warnings))

print("5. Squat SHOULD require knees/ankles")
ex = GenericExercise(load("learned_squat.json"))
ex.start()
gate = PositionGate(ex)
full = full_squat_landmarks()
gate.calibrate(full)
upper_only = {"LEFT_HIP": full["LEFT_HIP"], "RIGHT_HIP": full["RIGHT_HIP"],
              "LEFT_SHOULDER": full["LEFT_SHOULDER"], "RIGHT_SHOULDER": full["RIGHT_SHOULDER"],
              "LEFT_ELBOW": full["LEFT_ELBOW"], "RIGHT_ELBOW": full["RIGHT_ELBOW"],
              "LEFT_WRIST": full["LEFT_WRIST"], "RIGHT_WRIST": full["RIGHT_WRIST"]}
gate.check(FAKE_FRAME, upper_only, True, 0)
r = gate.check(FAKE_FRAME, upper_only, True, 500)
check("squat with knees/ankles missing IS gated", r.gated is True)
check("warning mentions knees/ankles", any("knee" in w.lower() or "ankle" in w.lower() for w in r.warnings))

print("6. Position drift: moving far from the calibrated anchor gates")
ex = GenericExercise(load("learned_squat.json"))
ex.start()
gate = PositionGate(ex)
gate.calibrate(full_squat_landmarks())
drifted = {k: (x + 500, y) for k, (x, y) in full_squat_landmarks().items()}
gate.check(FAKE_FRAME, drifted, True, 0)
r = gate.check(FAKE_FRAME, drifted, True, 500)
check("drifted position IS gated", r.gated is True)
check("warning mentions returning to position", any("Return to your starting position" in w for w in r.warnings))

print("7. Position drift: small natural movement does NOT gate")
ex = GenericExercise(load("learned_squat.json"))
ex.start()
gate = PositionGate(ex)
gate.calibrate(full_squat_landmarks())
slight = {k: (x + 20, y) for k, (x, y) in full_squat_landmarks().items()}
gate.check(FAKE_FRAME, slight, True, 0)
r = gate.check(FAKE_FRAME, slight, True, 500)
check("small natural movement is NOT gated", r.gated is False)

print(f"\n{'ALL POSITION-GATE TESTS PASSED' if failures == 0 else str(failures) + ' TEST(S) FAILED'}")
sys.exit(0 if failures == 0 else 1)
