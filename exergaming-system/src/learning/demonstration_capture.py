"""
Demonstration Capture — records per-frame angle data from a pose detection session.

Used by the learning algorithm to build an exercise definition from a video.
Every major bilateral joint angle is computed for each frame so the learner
can decide which ones actually move during the exercise.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from ..pose.analyzer import PoseAnalyzer

# All bilateral joint angle signals to compute on every frame.
# Format: (signal_name, [base_landmark_p1, base_landmark_vertex, base_landmark_p3])
# GenericExercise / learning code adds LEFT_ and RIGHT_ prefixes automatically.
CANDIDATE_ANGLE_SIGNALS = [
    ("knee_angle",   ["HIP",      "KNEE",     "ANKLE"]),
    ("elbow_angle",  ["SHOULDER", "ELBOW",    "WRIST"]),
    ("arm_angle",    ["HIP",      "SHOULDER", "ELBOW"]),     # shoulder abduction
    ("hip_angle",    ["SHOULDER", "HIP",      "KNEE"]),
    ("ankle_angle",  ["KNEE",     "ANKLE",    "FOOT_INDEX"]),
]

# Ratio signals: (name, [numerator_p1, numerator_p2], [denom_p1, denom_p2])
CANDIDATE_RATIO_SIGNALS = [
    ("leg_spread_ratio",
     ["LEFT_ANKLE",  "RIGHT_ANKLE"],
     ["LEFT_HIP",    "RIGHT_HIP"]),
    ("arm_spread_ratio",
     ["LEFT_WRIST",  "RIGHT_WRIST"],
     ["LEFT_SHOULDER", "RIGHT_SHOULDER"]),
]


class DemonstrationCapture:
    """
    Captures per-frame pose data (all candidate signal values) from a video or
    live camera session. The captured frames are passed to ExerciseLearner.
    """

    def __init__(self):
        self._frames: List[Dict] = []
        self._start_time: Optional[float] = None
        self.is_recording = False

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        self._frames = []
        self._start_time = time.perf_counter()
        self.is_recording = True

    def stop_recording(self) -> List[Dict]:
        """Stop recording and return the captured frame list."""
        self.is_recording = False
        return list(self._frames)

    def process_frame(
        self,
        frame_num: int,
        landmarks: Dict[str, Tuple[int, int]],
        pose_detected: bool,
    ) -> None:
        """
        Compute all candidate signals for this frame and store the record.

        Call this for every frame while is_recording is True, even if pose was
        not detected (pose_detected=False frames are stored with empty signals so
        the learner can see detection gaps).
        """
        if not self.is_recording:
            return

        elapsed_ms = int((time.perf_counter() - self._start_time) * 1000) if self._start_time else 0
        record: Dict = {
            "frame_num":    frame_num,
            "timestamp_ms": elapsed_ms,
            "pose_detected": pose_detected,
            "signals":      {},
        }

        if pose_detected and landmarks:
            record["signals"] = self._compute_all_signals(landmarks)

        self._frames.append(record)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, path: str) -> None:
        """Save captured frames to a JSON file for offline analysis."""
        data = {
            "total_frames":  len(self._frames),
            "detected_frames": sum(1 for f in self._frames if f["pose_detected"]),
            "frames": self._frames,
        }
        with open(path, "w") as fp:
            json.dump(data, fp, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> List[Dict]:
        """Load previously saved frame data."""
        with open(path, "r") as fp:
            data = json.load(fp)
        return data["frames"]

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_all_signals(
        self, landmarks: Dict[str, Tuple[int, int]]
    ) -> Dict[str, float]:
        """Compute every candidate angle and ratio signal for one frame."""
        signals: Dict[str, float] = {}

        for name, (p1, p2, p3) in CANDIDATE_ANGLE_SIGNALS:
            val = _bilateral_angle(landmarks, p1, p2, p3)
            if val is not None:
                signals[name] = round(val, 2)

        for name, num_lms, den_lms in CANDIDATE_RATIO_SIGNALS:
            val = _ratio_signal(landmarks, num_lms, den_lms)
            if val is not None:
                signals[name] = round(val, 3)

        return signals


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _bilateral_angle(
    landmarks: Dict[str, Tuple[int, int]],
    base_p1: str,
    base_p2: str,
    base_p3: str,
) -> Optional[float]:
    """
    Compute the average bilateral angle. Falls back to one side if the other
    is unavailable. Returns None if neither side is visible.
    """
    left_lms  = [f"LEFT_{base_p1}",  f"LEFT_{base_p2}",  f"LEFT_{base_p3}"]
    right_lms = [f"RIGHT_{base_p1}", f"RIGHT_{base_p2}", f"RIGHT_{base_p3}"]

    left_ok  = all(lm in landmarks for lm in left_lms)
    right_ok = all(lm in landmarks for lm in right_lms)

    if not left_ok and not right_ok:
        return None

    left_val  = PoseAnalyzer.calculate_angle(*[landmarks[lm] for lm in left_lms])  if left_ok  else None
    right_val = PoseAnalyzer.calculate_angle(*[landmarks[lm] for lm in right_lms]) if right_ok else None

    if left_val is not None and right_val is not None:
        return (left_val + right_val) / 2
    return left_val if left_val is not None else right_val


def _ratio_signal(
    landmarks: Dict[str, Tuple[int, int]],
    numerator: List[str],
    denominator: List[str],
) -> Optional[float]:
    """Compute a distance ratio. Returns None if any landmark is missing."""
    all_lms = numerator + denominator
    if not all(lm in landmarks for lm in all_lms):
        return None
    num_dist = PoseAnalyzer.calculate_distance(landmarks[numerator[0]], landmarks[numerator[1]])
    den_dist = PoseAnalyzer.calculate_distance(landmarks[denominator[0]], landmarks[denominator[1]])
    if den_dist <= 0:
        return None
    return num_dist / den_dist
