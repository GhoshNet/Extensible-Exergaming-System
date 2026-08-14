"""
Pose Analyzer Module
Calculates angles and analyzes body positions
"""
import numpy as np
from typing import Tuple, Optional


class PoseAnalyzer:
    """Analyzes pose data and calculates joint angles"""
    
    @staticmethod
    def calculate_angle(point1: Tuple[int, int], 
                       point2: Tuple[int, int], 
                       point3: Tuple[int, int]) -> float:
        """
        Calculate angle between three points
        
        Args:
            point1: First point (x, y)
            point2: Middle point (vertex) (x, y)
            point3: Third point (x, y)
            
        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clamp to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        
        # Convert to degrees
        return np.degrees(angle)
        
    @staticmethod
    def calculate_knee_angle(hip: Tuple[int, int],
                            knee: Tuple[int, int],
                            ankle: Tuple[int, int]) -> float:
        """
        Calculate knee angle (hip-knee-ankle)
        
        Returns:
            Angle in degrees
        """
        return PoseAnalyzer.calculate_angle(hip, knee, ankle)
        
    @staticmethod
    def calculate_hip_angle(shoulder: Tuple[int, int],
                           hip: Tuple[int, int],
                           knee: Tuple[int, int]) -> float:
        """
        Calculate hip angle (shoulder-hip-knee)
        
        Returns:
            Angle in degrees
        """
        return PoseAnalyzer.calculate_angle(shoulder, hip, knee)
        
    @staticmethod
    def calculate_elbow_angle(shoulder: Tuple[int, int],
                             elbow: Tuple[int, int],
                             wrist: Tuple[int, int]) -> float:
        """
        Calculate elbow angle (shoulder-elbow-wrist)
        
        Returns:
            Angle in degrees
        """
        return PoseAnalyzer.calculate_angle(shoulder, elbow, wrist)
        
    @staticmethod
    def calculate_shoulder_angle(hip: Tuple[int, int],
                                shoulder: Tuple[int, int],
                                elbow: Tuple[int, int]) -> float:
        """
        Calculate shoulder angle (hip-shoulder-elbow)
        
        Returns:
            Angle in degrees
        """
        return PoseAnalyzer.calculate_angle(hip, shoulder, elbow)
        
    @staticmethod
    def calculate_torso_angle(shoulder: Tuple[int, int],
                             hip: Tuple[int, int]) -> float:
        """
        Calculate torso angle from vertical
        
        Returns:
            Angle from vertical in degrees (0 = upright, 90 = horizontal)
        """
        # Create vertical reference point
        vertical_point = (shoulder[0], shoulder[1] - 100)
        
        return PoseAnalyzer.calculate_angle(vertical_point, shoulder, hip)
        
    @staticmethod
    def calculate_distance(point1: Tuple[int, int], 
                          point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points
        
        Returns:
            Distance in pixels
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    @staticmethod
    def is_point_above(point1: Tuple[int, int], 
                       point2: Tuple[int, int]) -> bool:
        """Check if point1 is above point2 (lower y value)"""
        return point1[1] < point2[1]
        
    @staticmethod
    def get_body_center(left_hip: Tuple[int, int],
                       right_hip: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate center point between hips
        
        Returns:
            (x, y) center coordinates
        """
        center_x = (left_hip[0] + right_hip[0]) // 2
        center_y = (left_hip[1] + right_hip[1]) // 2
        return (center_x, center_y)
