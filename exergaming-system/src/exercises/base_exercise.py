"""
Base Exercise Module
Abstract base class for all exercises
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from enum import Enum


class ExerciseState(Enum):
    """States of exercise movement"""
    IDLE = "idle"
    UP = "up"
    DOWN = "down"
    TRANSITION_UP = "transition_up"
    TRANSITION_DOWN = "transition_down"


class FormFeedback(Enum):
    """Form quality feedback"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class BaseExercise(ABC):
    """Abstract base class for exercises"""
    
    def __init__(self, name: str):
        """
        Initialize exercise

        Args:
            name: Exercise name
        """
        self.name = name
        self.rep_count = 0
        self.state = ExerciseState.IDLE
        self.form_feedback = FormFeedback.GOOD
        self.is_active = False
        self._last_side_used: str = "none"  # 'both' | 'left' | 'right' | 'none'

        # Thresholds (to be set by child classes)
        self.angle_threshold_down = 90  # Example
        self.angle_threshold_up = 160   # Example
        
    @abstractmethod
    def analyze_pose(self, landmarks: Dict[str, Tuple[int, int]]) -> None:
        """
        Analyze current pose and update state
        
        Args:
            landmarks: Dictionary of landmark_name: (x, y)
        """
        pass
        
    @abstractmethod
    def check_form(self, landmarks: Dict[str, Tuple[int, int]]) -> FormFeedback:
        """
        Check exercise form quality
        
        Args:
            landmarks: Dictionary of landmark_name: (x, y)
            
        Returns:
            Form quality feedback
        """
        pass
        
    @abstractmethod
    def get_required_landmarks(self) -> list:
        """
        Get list of required landmark names for this exercise
        
        Returns:
            List of landmark names (strings)
        """
        pass
        
    def reset(self):
        """Reset exercise counters and state"""
        self.rep_count = 0
        self.state = ExerciseState.IDLE
        self.is_active = False
        
    def start(self):
        """Start the exercise"""
        self.is_active = True
        
    def stop(self):
        """Stop the exercise"""
        self.is_active = False
        
    def get_rep_count(self) -> int:
        """Get current repetition count"""
        return self.rep_count
        
    def get_state(self) -> ExerciseState:
        """Get current exercise state"""
        return self.state
        
    def get_form_feedback(self) -> FormFeedback:
        """Get current form feedback"""
        return self.form_feedback
        
    def increment_rep(self):
        """Increment repetition counter"""
        self.rep_count += 1
        
    def get_info(self) -> Dict:
        """
        Get exercise information

        Returns:
            Dictionary with exercise stats
        """
        return {
            'name':      self.name,
            'reps':      self.rep_count,
            'rep_count': self.rep_count,   # alias used by session logger
            'state':     self.state.value,
            'form':      self.form_feedback.value,
            'active':    self.is_active,
            'side_used': self._last_side_used,
        }
