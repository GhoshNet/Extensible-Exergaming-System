"""
Configuration file for the Exergaming System
"""

# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# Pose Detection Settings
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1  # 0, 1, or 2 (higher = more accurate but slower)

# Exercise Settings
REP_COUNT_THRESHOLD = 0.8  # Confidence threshold for counting a rep
POSE_HOLD_TIME = 0.5  # Seconds to hold a position
RESET_TIME = 3.0  # Seconds of no detection before resetting

# Feedback Settings
GOOD_FORM_COLOR = (0, 255, 0)  # Green
BAD_FORM_COLOR = (0, 0, 255)  # Red
NEUTRAL_COLOR = (255, 255, 0)  # Yellow
SKELETON_THICKNESS = 2

# Audio Settings
AUDIO_ENABLED = True
VOLUME = 0.7  # 0.0 to 1.0

# UI Settings
WINDOW_TITLE = "Exergaming System"
FONT_SIZE = 12
BACKGROUND_COLOR = "#2C3E50"
TEXT_COLOR = "#ECF0F1"

# Exercise Angles (in degrees) - Standard thresholds for correct form
SQUAT_KNEE_ANGLE_MIN = 70
SQUAT_KNEE_ANGLE_MAX = 110
SQUAT_HIP_ANGLE_MIN = 70

PUSHUP_ELBOW_ANGLE_MIN = 70
PUSHUP_ELBOW_ANGLE_MAX = 110

JUMPING_JACK_ARM_ANGLE_MIN = 140  # Arms up (loosened from 160 — some videos peak at ~153°)
JUMPING_JACK_LEG_ANGLE_MIN = 20   # Legs apart

# Data Storage
DATABASE_PATH = "exergaming_data.db"
SESSION_AUTO_SAVE = True

# Diagnostics
DIAGNOSTICS_ENABLED = True   # Set False to disable session logging
SESSION_LOG_DIR = "sessions"
