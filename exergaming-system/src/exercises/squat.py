"""
Squat Exercise Module
Implements squat detection and rep counting
"""
from .base_exercise import BaseExercise, ExerciseState, FormFeedback
from ..pose.analyzer import PoseAnalyzer
from typing import Dict, Tuple


class SquatExercise(BaseExercise):
    """Squat exercise implementation"""
    
    def __init__(self):
        super().__init__("Squat")
        
        # Angle thresholds for squat
        self.knee_angle_down = 90   # Knees bent (down position)
        self.knee_angle_up = 160    # Knees straight (up position)
        
        # Form quality thresholds
        self.good_depth_angle = 100  # Good squat depth
        self.excellent_depth_angle = 85  # Excellent squat depth
        
        # Tracking
        self.current_knee_angle = 180
        self.min_knee_angle_in_rep = 180  # Track deepest point
        
    def get_required_landmarks(self) -> list:
        """Required landmarks for squat detection"""
        return [
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER'
        ]
        
    def analyze_pose(self, landmarks: Dict[str, Tuple[int, int]]) -> None:
        """
        Analyze squat movement and count reps
        
        Args:
            landmarks: Dictionary of landmark positions
        """
        if not self.is_active:
            return
            
        # Determine which sides have all required landmarks visible.
        # When one leg is behind the other, MediaPipe's confidence drops below
        # the visibility threshold and that side is absent from the dict.
        left_ok  = all(lm in landmarks for lm in ('LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE'))
        right_ok = all(lm in landmarks for lm in ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'))

        # Also require at least one shoulder to be visible (used for form context)
        if not left_ok and not right_ok:
            self._last_side_used = "none"
            return

        if left_ok:
            left_knee_angle = PoseAnalyzer.calculate_knee_angle(
                landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE'])
        if right_ok:
            right_knee_angle = PoseAnalyzer.calculate_knee_angle(
                landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'], landmarks['RIGHT_ANKLE'])

        if left_ok and right_ok:
            self.current_knee_angle = (left_knee_angle + right_knee_angle) / 2
            self._last_side_used = "both"
        elif left_ok:
            self.current_knee_angle = left_knee_angle
            self._last_side_used = "left"
        else:
            self.current_knee_angle = right_knee_angle
            self._last_side_used = "right"
        
        # Track minimum angle in current rep (deepest squat)
        if self.state == ExerciseState.DOWN or self.state == ExerciseState.TRANSITION_DOWN:
            self.min_knee_angle_in_rep = min(self.min_knee_angle_in_rep, self.current_knee_angle)
        
        # State machine for rep counting
        if self.state == ExerciseState.IDLE or self.state == ExerciseState.UP:
            # Waiting for person to go down
            if self.current_knee_angle < self.knee_angle_down:
                self.state = ExerciseState.DOWN
                
        elif self.state == ExerciseState.DOWN:
            # Person is in down position, waiting to come up
            if self.current_knee_angle > self.knee_angle_up:
                # Complete rep!
                self.increment_rep()
                self.state = ExerciseState.UP
                
                # Evaluate form based on depth achieved
                self.form_feedback = self._evaluate_form()
                
                # Reset min angle for next rep
                self.min_knee_angle_in_rep = 180
                
    def check_form(self, landmarks: Dict[str, Tuple[int, int]]) -> FormFeedback:
        """
        Check squat form quality
        
        Args:
            landmarks: Dictionary of landmark positions
            
        Returns:
            Form quality feedback
        """
        # Form checks:
        # 1. Squat depth (knee angle)
        # 2. Back straightness (could add torso angle check)
        # 3. Knee tracking (knees don't cave in - advanced)
        
        # For now, base form on squat depth
        return self._evaluate_form()
        
    def _evaluate_form(self) -> FormFeedback:
        """
        Evaluate form based on minimum knee angle achieved
        
        Returns:
            Form quality rating
        """
        if self.min_knee_angle_in_rep < self.excellent_depth_angle:
            return FormFeedback.EXCELLENT
        elif self.min_knee_angle_in_rep < self.good_depth_angle:
            return FormFeedback.GOOD
        elif self.min_knee_angle_in_rep < self.knee_angle_down + 20:
            return FormFeedback.FAIR
        else:
            return FormFeedback.POOR
            
    def reset(self):
        """Reset exercise state including squat-specific tracking"""
        super().reset()
        self.current_knee_angle = 180
        self.min_knee_angle_in_rep = 180

    def get_current_angle(self) -> float:
        """Get current knee angle for display"""
        return self.current_knee_angle
        
    def get_info(self) -> Dict:
        """Get extended exercise info including angles"""
        info = super().get_info()
        info['knee_angle'] = round(self.current_knee_angle, 1)
        info['min_angle_in_rep'] = round(self.min_knee_angle_in_rep, 1)
        return info
