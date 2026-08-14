"""
Jumping Jacks Exercise Implementation
Detects and counts jumping jacks with form checking
"""
from .base_exercise import BaseExercise, ExerciseState, FormFeedback
from ..pose.analyzer import PoseAnalyzer
from ..utils.config import JUMPING_JACK_ARM_ANGLE_MIN
from typing import Dict, Tuple


class JumpingJackExercise(BaseExercise):
    """Jumping jacks exercise detection and counting"""

    def __init__(self):
        super().__init__("Jumping Jack")
        self.arms_up_threshold = JUMPING_JACK_ARM_ANGLE_MIN
        self.leg_spread_ratio_threshold = 1.8  # ankle distance / hip distance

        self.current_arm_angle = 0.0
        self.current_leg_spread = 0.0
        # Peak values tracked during the UP (out) phase — used for form evaluation
        self.peak_arm_angle_in_rep = 0.0
        self.peak_leg_spread_in_rep = 0.0

        # ExerciseState.IDLE / DOWN = feet together (in), UP = feet apart (out)

    def get_required_landmarks(self) -> list:
        """Required landmarks for jumping jack detection"""
        return [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
        ]

    def analyze_pose(self, landmarks: Dict[str, Tuple[int, int]]) -> None:
        """
        Analyze jumping jack movement and count reps

        Args:
            landmarks: Dictionary of landmark positions
        """
        if not self.is_active:
            return

        required = self.get_required_landmarks()
        if not all(lm in landmarks for lm in required):
            self._last_side_used = "none"
            return
        self._last_side_used = "both"

        # Calculate shoulder angle for both arms (hip-shoulder-elbow)
        left_arm_angle = PoseAnalyzer.calculate_shoulder_angle(
            landmarks['LEFT_HIP'],
            landmarks['LEFT_SHOULDER'],
            landmarks['LEFT_ELBOW']
        )
        right_arm_angle = PoseAnalyzer.calculate_shoulder_angle(
            landmarks['RIGHT_HIP'],
            landmarks['RIGHT_SHOULDER'],
            landmarks['RIGHT_ELBOW']
        )
        self.current_arm_angle = (left_arm_angle + right_arm_angle) / 2

        # Calculate leg spread ratio (ankle distance / hip distance)
        ankle_dist = PoseAnalyzer.calculate_distance(
            landmarks['LEFT_ANKLE'], landmarks['RIGHT_ANKLE']
        )
        hip_dist = PoseAnalyzer.calculate_distance(
            landmarks['LEFT_HIP'], landmarks['RIGHT_HIP']
        )

        if hip_dist > 0:
            self.current_leg_spread = ankle_dist / hip_dist
        else:
            return

        arms_up = self.current_arm_angle > self.arms_up_threshold
        legs_apart = self.current_leg_spread > self.leg_spread_ratio_threshold

        # State machine: IDLE/DOWN = feet together, UP = feet apart
        if self.state == ExerciseState.IDLE or self.state == ExerciseState.DOWN:
            if arms_up and legs_apart:
                self.state = ExerciseState.UP
                self.peak_arm_angle_in_rep = self.current_arm_angle
                self.peak_leg_spread_in_rep = self.current_leg_spread

        elif self.state == ExerciseState.UP:
            # Track peak reach during the out phase for accurate form scoring
            self.peak_arm_angle_in_rep = max(self.peak_arm_angle_in_rep, self.current_arm_angle)
            self.peak_leg_spread_in_rep = max(self.peak_leg_spread_in_rep, self.current_leg_spread)

            if not arms_up and not legs_apart:
                self.increment_rep()
                self.form_feedback = self._evaluate_form()
                self.state = ExerciseState.DOWN
                self.peak_arm_angle_in_rep = 0.0
                self.peak_leg_spread_in_rep = 0.0

    def check_form(self, landmarks: Dict[str, Tuple[int, int]]) -> FormFeedback:
        """Check jumping jack form quality"""
        return self._evaluate_form()

    def _evaluate_form(self) -> FormFeedback:
        """Evaluate form based on peak arm height and leg spread reached during the rep"""
        arms_high_enough = self.peak_arm_angle_in_rep > self.arms_up_threshold
        legs_spread_enough = self.peak_leg_spread_in_rep > self.leg_spread_ratio_threshold

        if arms_high_enough and legs_spread_enough:
            if (self.peak_arm_angle_in_rep > self.arms_up_threshold + 20 and
                    self.peak_leg_spread_in_rep > self.leg_spread_ratio_threshold + 0.5):
                return FormFeedback.EXCELLENT
            return FormFeedback.GOOD
        elif arms_high_enough or legs_spread_enough:
            return FormFeedback.FAIR
        else:
            return FormFeedback.POOR

    def reset(self):
        """Reset exercise state"""
        super().reset()
        self.current_arm_angle = 0.0
        self.current_leg_spread = 0.0
        self.peak_arm_angle_in_rep = 0.0
        self.peak_leg_spread_in_rep = 0.0

    def get_info(self) -> Dict:
        """Get extended info including arm angle and leg spread"""
        info = super().get_info()
        info['arm_angle'] = round(self.current_arm_angle, 1)
        info['leg_spread'] = round(self.current_leg_spread, 2)
        return info
