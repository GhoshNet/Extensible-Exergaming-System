"""
Frame Quality Checker — detects bad input conditions and returns human-readable
warning strings that the UI can display as overlays.

Checks run on every frame while an exercise is active:
  - No person detected
  - Lighting too dark (mean brightness < 45/255)

These are the only checks that are genuinely exercise-agnostic. Framing
(too close/too far) and which specific body parts need to be visible are
both exercise-specific -- e.g. a tightly-framed bicep curl legitimately
never shows the legs, so a bounding box over ALL visible landmarks would
wrongly flag it as "too far" the moment the legs happen to be out of
frame. Both live in PositionGate, scoped to
GenericExercise.get_critical_landmarks() instead.

All checks are cheap: brightness requires one grayscale conversion (~1 ms).
No additional ML calls.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple


# Mean pixel brightness (0–255 grayscale) below which we warn about lighting
_DARK_THRESHOLD = 45


class FrameQualityChecker:
    """
    Stateless quality checker — call check() on every frame.
    Returns a list of warning strings (empty = all good).
    """

    def check(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Tuple[int, int]],
        pose_detected: bool,
    ) -> List[str]:
        """
        Inspect frame and landmarks for quality issues.

        Args:
            frame:         BGR frame as captured (before overlays)
            landmarks:     visible landmark dict from PoseDetector
                           (empty dict if pose_detected is False)
            pose_detected: True if MediaPipe returned any landmarks

        Returns:
            List of warning strings, ordered by severity.
            Empty list means no issues detected.
        """
        warnings: List[str] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < _DARK_THRESHOLD:
            warnings.append("Too dark - improve lighting")

        if not pose_detected or not landmarks:
            warnings.append("No person detected - move into camera view")

        return warnings
