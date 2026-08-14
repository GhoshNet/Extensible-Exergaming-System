"""
Pose Detection Module
Uses MediaPipe to detect human pose keypoints
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict


class PoseDetector:
    """Detects human pose using MediaPipe Pose"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
        )
        
        self.landmarks = None
        self.is_detected = False
        
    def detect(self, frame: np.ndarray) -> bool:
        """
        Detect pose in frame
        
        Args:
            frame: BGR image from camera
            
        Returns:
            True if pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(image_rgb)
        
        # Store landmarks
        self.landmarks = results.pose_landmarks
        self.is_detected = results.pose_landmarks is not None
        
        return self.is_detected
        
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw pose landmarks on frame
        
        Args:
            frame: BGR image
            
        Returns:
            Frame with landmarks drawn
        """
        if self.landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                self.landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
        
    def get_landmark_coords(self, landmark_id: int, frame_shape: tuple) -> Optional[tuple]:
        """
        Get pixel coordinates of a landmark
        
        Args:
            landmark_id: MediaPipe landmark ID
            frame_shape: (height, width, channels)
            
        Returns:
            (x, y) pixel coordinates or None
        """
        if not self.landmarks:
            return None
            
        h, w = frame_shape[:2]
        landmark = self.landmarks.landmark[landmark_id]
        
        # Convert normalized coordinates to pixels
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        
        return (x, y)
        
    def get_all_landmarks_dict(self, frame_shape: tuple,
                               visibility_threshold: float = 0.25) -> Dict[str, tuple]:
        """
        Get all landmarks as a dictionary with names.

        Only landmarks whose MediaPipe visibility score exceeds
        visibility_threshold are included. Landmarks below this are omitted
        so that exercise classes which check 'if lm in landmarks' will
        naturally fall back to the visible side rather than using MediaPipe's
        hallucinated coordinates for occluded joints.

        Args:
            frame_shape: (height, width, channels)
            visibility_threshold: 0.0–1.0. Default 0.25 filters genuinely
                invisible joints (e.g. completely behind the body) while
                keeping partially occluded but estimable landmarks.

        Returns:
            Dictionary of landmark_name: (x, y) for visible landmarks only.
        """
        if not self.landmarks:
            return {}

        h, w = frame_shape[:2]
        landmarks_dict = {}

        for landmark_name in self.mp_pose.PoseLandmark:
            landmark = self.landmarks.landmark[landmark_name]
            if landmark.visibility < visibility_threshold:
                continue   # occluded — skip rather than use guessed coords
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_dict[landmark_name.name] = (x, y)

        return landmarks_dict
        
    def get_raw_landmarks(self):
        """Return the raw MediaPipe NormalizedLandmarkList (or None if not detected).
        Used by the skeleton overlay which needs direct access to landmark visibility
        and normalised coordinates to draw coloured body segments."""
        return self.landmarks

    def cleanup(self):
        """Release MediaPipe resources"""
        self.pose.close()


# MediaPipe Pose Landmark IDs (for reference)
class PoseLandmark:
    """MediaPipe Pose Landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
