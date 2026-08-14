"""
Push-up Exercise Implementation
Detects and counts push-ups with form checking
"""
from .base_exercise import BaseExercise, ExerciseState, FormFeedback
from ..pose.analyzer import PoseAnalyzer
from ..utils.config import PUSHUP_ELBOW_ANGLE_MIN, PUSHUP_ELBOW_ANGLE_MAX
from typing import Dict, Tuple


class PushUpExercise(BaseExercise):
    """Push-up exercise detection and counting"""

    def __init__(self):
        super().__init__("Push-up")
        self.elbow_angle_down = PUSHUP_ELBOW_ANGLE_MAX   # ~110° — elbow bent (down)
        self.elbow_angle_up = 160                         # Arms extended (up)
        self.current_elbow_angle = 180
        self.min_elbow_angle_in_rep = 180  # Track deepest point

    def get_required_landmarks(self) -> list:
        """Required landmarks for push-up detection"""
        return [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
        ]

    def analyze_pose(self, landmarks: Dict[str, Tuple[int, int]]) -> None:
        """
        Analyze push-up movement and count reps

        Args:
            landmarks: Dictionary of landmark positions
        """
        if not self.is_active:
            return

        # Determine which arms are reliably visible.
        # In front-facing push-ups the far arm is often occluded by the torso.
        left_ok  = all(lm in landmarks for lm in ('LEFT_SHOULDER',  'LEFT_ELBOW',  'LEFT_WRIST'))
        right_ok = all(lm in landmarks for lm in ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'))

        if not left_ok and not right_ok:
            self._last_side_used = "none"
            return

        if left_ok:
            left_elbow_angle = PoseAnalyzer.calculate_elbow_angle(
                landmarks['LEFT_SHOULDER'], landmarks['LEFT_ELBOW'], landmarks['LEFT_WRIST'])
        if right_ok:
            right_elbow_angle = PoseAnalyzer.calculate_elbow_angle(
                landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_ELBOW'], landmarks['RIGHT_WRIST'])

        if left_ok and right_ok:
            self.current_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            self._last_side_used = "both"
        elif left_ok:
            self.current_elbow_angle = left_elbow_angle
            self._last_side_used = "left"
        else:
            self.current_elbow_angle = right_elbow_angle
            self._last_side_used = "right"

        # Track minimum (most bent) angle in current rep
        if self.state == ExerciseState.DOWN:
            self.min_elbow_angle_in_rep = min(
                self.min_elbow_angle_in_rep, self.current_elbow_angle
            )

        # State machine
        if self.state == ExerciseState.IDLE or self.state == ExerciseState.UP:
            # Waiting for person to go down
            if self.current_elbow_angle < self.elbow_angle_down:
                self.state = ExerciseState.DOWN
                self.min_elbow_angle_in_rep = self.current_elbow_angle

        elif self.state == ExerciseState.DOWN:
            # Person is down, waiting to push back up
            if self.current_elbow_angle > self.elbow_angle_up:
                self.increment_rep()
                self.form_feedback = self._evaluate_form()
                self.state = ExerciseState.UP
                self.min_elbow_angle_in_rep = 180

    def check_form(self, landmarks: Dict[str, Tuple[int, int]]) -> FormFeedback:
        """Check push-up form quality"""
        return self._evaluate_form()

    def _evaluate_form(self) -> FormFeedback:
        """Evaluate form based on elbow depth achieved"""
        if self.min_elbow_angle_in_rep < PUSHUP_ELBOW_ANGLE_MIN:
            return FormFeedback.EXCELLENT
        elif self.min_elbow_angle_in_rep < self.elbow_angle_down:
            return FormFeedback.GOOD
        elif self.min_elbow_angle_in_rep < self.elbow_angle_down + 20:
            return FormFeedback.FAIR
        else:
            return FormFeedback.POOR

    def reset(self):
        """Reset exercise state"""
        super().reset()
        self.current_elbow_angle = 180
        self.min_elbow_angle_in_rep = 180

    def get_info(self) -> Dict:
        """Get extended info including elbow angle"""
        info = super().get_info()
        info['elbow_angle'] = round(self.current_elbow_angle, 1)
        return info
