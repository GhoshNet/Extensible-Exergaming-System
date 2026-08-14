"""
Main Controller Module
Coordinates camera, pose detection, exercises, and feedback
"""
import collections
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from src.utils.camera import Camera
from src.pose.detector import PoseDetector
from src.exercises.exercise_definition import ExerciseDefinition
from src.exercises.generic_exercise import GenericExercise
from src.feedback.visual import VisualFeedback, CAMERA_GUIDE_DURATION
from src.feedback.diagnostic_overlay import DiagnosticOverlay
from src.feedback.form_errors import FormAnalyzer
from src.diagnostics.session_logger import SessionLogger
from src.pose.position_gate import PositionGate
from src.utils.config import DIAGNOSTICS_ENABLED, SESSION_LOG_DIR

# Directory containing exercise definition JSON files
_DEFINITIONS_DIR = Path(__file__).parent / "exercises" / "definitions"


def _load_exercise_definitions() -> Dict[str, ExerciseDefinition]:
    """
    Scan the definitions directory and load all valid ExerciseDefinition JSON files.
    Files prefixed with 'learned_' are appended after the hardcoded ones so the
    dropdown shows standard exercises first.
    """
    definitions: Dict[str, ExerciseDefinition] = {}

    if not _DEFINITIONS_DIR.exists():
        print(f"Warning: definitions directory not found: {_DEFINITIONS_DIR}")
        return definitions

    # Load hardcoded first, then learned (so learned versions appear after originals)
    files = sorted(_DEFINITIONS_DIR.glob("*.json"), key=lambda p: p.name.startswith("learned_"))
    for path in files:
        try:
            defn = ExerciseDefinition.from_json(str(path))
            definitions[defn.name] = defn
        except Exception as exc:
            print(f"Warning: could not load exercise definition {path.name}: {exc}")

    return definitions


# Load once at import time
EXERCISE_DEFINITIONS: Dict[str, ExerciseDefinition] = _load_exercise_definitions()


class ExergamingController:
    """Main controller for the exergaming system"""

    def __init__(self):
        self.camera = Camera(camera_id=0)   # resolution auto-detected on start()
        self.pose_detector = PoseDetector()
        self.visual_feedback = VisualFeedback()

        default_defn = next(iter(EXERCISE_DEFINITIONS.values()))
        self.current_exercise = GenericExercise(default_defn)

        self.is_running = False
        self.current_frame = None
        self.processed_frame = None

        self._logger: Optional[SessionLogger] = (
            SessionLogger(SESSION_LOG_DIR) if DIAGNOSTICS_ENABLED else None
        )
        self._frame_count: int = 0

        self._diagnostic_overlay = DiagnosticOverlay()
        self._position_gate = PositionGate(self.current_exercise)
        self._form_analyzer: FormAnalyzer = FormAnalyzer(default_defn)
        self.show_diagnostics: bool = False
        # Set True while VideoWorker / AnalysisWorker is running so the
        # background thread skips MediaPipe inference and avoids a crash
        # caused by two TFLite instances sharing the same Metal GPU context.
        self.processing_paused: bool = False
        self._frame_timestamps: collections.deque = collections.deque(maxlen=60)
        self._last_proc_ms: float = 0.0
        self._last_pose_detected: bool = False
        self._last_quality_warnings: list = []
        self._last_form_errors: list = []
        self._last_raw_landmarks = None     # MediaPipe NormalizedLandmarkList for skeleton overlay
        self._exercise_start_time: float = 0.0

        # Pre-exercise countdown: Start begins a countdown instead of
        # activating rep-counting immediately, so a user still getting into
        # position isn't counted for a false-positive rep. See start_exercise().
        self.START_COUNTDOWN_SECONDS: float = 5.0
        self._countdown_active: bool = False
        self._countdown_start: float = 0.0

    def initialize(self) -> bool:
        """Initialize the system and start camera"""
        if not self.camera.start():
            print("Error: Failed to start camera")
            return False
        self.is_running = True
        print("System initialized successfully")
        return True

    def process_frame(self) -> bool:
        """Process one frame: detect pose, analyze exercise, draw feedback"""
        if not self.is_running:
            return False

        success, frame = self.camera.read_frame()
        if not success or frame is None:
            return False

        self.current_frame = frame.copy()
        frame_start = time.perf_counter()
        self._frame_timestamps.append(frame_start)

        pose_detected = self.pose_detector.detect(frame)
        landmarks: Dict = {}

        if pose_detected:
            landmarks = self.pose_detector.get_all_landmarks_dict(frame.shape)
            self._last_raw_landmarks = self.pose_detector.get_raw_landmarks()
        else:
            self._last_raw_landmarks = None

        # Pre-exercise countdown: camera + pose detection keep running (so the
        # user gets live feedback that they're in frame) but analyze_pose() is
        # not called yet, so getting into position can't be counted as a rep.
        if self._countdown_active:
            remaining = self.START_COUNTDOWN_SECONDS - (time.perf_counter() - self._countdown_start)
            if remaining <= 0:
                self._activate_exercise(landmarks)
            else:
                if pose_detected:
                    self.pose_detector.draw_landmarks(frame)
                self.visual_feedback.draw_countdown(frame, remaining, self.current_exercise.name)
                self.processed_frame = frame
                return True

        timestamp_ms = time.perf_counter() * 1000
        if self.current_exercise.is_active:
            gate_result = self._position_gate.check(
                self.current_frame, landmarks, pose_detected, timestamp_ms
            )
            self._last_quality_warnings = gate_result.warnings
            # A single dropped-detection frame is NOT gated (see PositionGate's
            # grace period) -- analyze_pose() still runs and its own None-guards
            # handle a momentarily-missing signal gracefully. Only a SUSTAINED
            # bad frame (person genuinely out of position, not a blip) actually
            # skips it, so a workout doesn't visibly stall over one bad frame.
            if pose_detected and not gate_result.gated:
                self.current_exercise.analyze_pose(landmarks, timestamp_ms)
        else:
            self._last_quality_warnings = []

        exercise_info = self.current_exercise.get_info()
        proc_ms = (time.perf_counter() - frame_start) * 1000
        self._last_proc_ms = proc_ms
        self._last_pose_detected = pose_detected

        # Form error analysis (only while exercise is active and pose is detected)
        is_tracking = False
        if self.current_exercise.is_active and pose_detected and isinstance(
            self.current_exercise, GenericExercise
        ):
            current_state_str = exercise_info.get("state", "idle").upper()
            is_tracking = current_state_str == self._form_analyzer.definition.form.during_state
            self._last_form_errors = self._form_analyzer.analyze(
                current_state=current_state_str,
                tracked_values=self.current_exercise._form_tracked,
                signal_values=self.current_exercise._signal_values,
            )
        else:
            self._last_form_errors = []

        # Coloured skeleton overlay (contribution 1) — drawn BEFORE the text overlay
        self.visual_feedback.draw_form_skeleton(
            frame, self._last_raw_landmarks, self._last_form_errors,
            relevant_parts=self._form_analyzer.relevant_body_parts,
            is_tracking=is_tracking,
        )

        self.visual_feedback.draw_complete_overlay(
            frame, exercise_info, show_debug=True,
            definition=self.current_exercise.definition,
        )

        # Camera placement guide (first N seconds after Start)
        if self.current_exercise.is_active:
            elapsed = time.perf_counter() - self._exercise_start_time
            self.visual_feedback.draw_camera_guide(
                frame, self.current_exercise.name, elapsed
            )

        # Form error text list (bottom-left)
        if self._last_form_errors:
            self.visual_feedback.draw_form_errors(frame, self._last_form_errors)

        # Safety/over-limit warning (e.g. shoulder press hand dropping too
        # low on the way back up) -- takes the creator's over_limit prompt
        # if set, else the plain safety_message.
        if exercise_info.get('safety_violated'):
            defn = self.current_exercise.definition
            safety_text = defn.prompts.over_limit or defn.safety_message
            self.visual_feedback.draw_safety_warning(frame, safety_text)

        # Quality warning banners
        if self._last_quality_warnings:
            self.visual_feedback.draw_quality_warnings(frame, self._last_quality_warnings)

        if self.show_diagnostics:
            self._diagnostic_overlay.draw(frame, self._build_runtime_stats(), exercise_info)

        self.processed_frame = frame

        # Diagnostic logging
        if self._logger and self._logger.is_active:
            self._frame_count += 1
            log_entry: Dict = {
                "frame_num":          self._frame_count,
                "pose_detected":      pose_detected,
                "processing_time_ms": round(proc_ms, 2),
                "quality_warnings":   self._last_quality_warnings or None,
            }
            for key in ("state", "rep_count", "form", "side_used",
                        "knee_angle", "elbow_angle", "arm_angle", "leg_spread"):
                if key in exercise_info:
                    log_entry[key] = exercise_info[key]
            self._logger.log_frame(log_entry)

        return True

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current processed frame for display"""
        return self.processed_frame

    def get_exercise_info(self) -> Dict:
        """Get current exercise information"""
        return self.current_exercise.get_info()

    def get_available_exercises(self) -> list:
        """Return list of available exercise names"""
        return list(EXERCISE_DEFINITIONS.keys())

    def set_exercise(self, name: str):
        """Switch to a different exercise by name"""
        if name not in EXERCISE_DEFINITIONS:
            print(f"Unknown exercise: {name}")
            return
        was_active = self.current_exercise.is_active
        self._countdown_active = False   # cancel any pending countdown for the old exercise
        self.current_exercise.stop()
        defn = EXERCISE_DEFINITIONS[name]
        self.current_exercise = GenericExercise(defn)
        self._position_gate = PositionGate(self.current_exercise)
        self._form_analyzer = FormAnalyzer(defn)
        self._last_form_errors = []
        if was_active:
            self.current_exercise.start()
        print(f"Switched to exercise: {name}")

    def start_exercise(self):
        """
        Begin the pre-exercise countdown (START_COUNTDOWN_SECONDS). Rep-
        counting does not activate until the countdown elapses -- see
        process_frame() and _activate_exercise(). This avoids counting a
        false-positive rep while the user is still getting into position.
        """
        self.current_exercise.reset()
        self._countdown_active = True
        self._countdown_start = time.perf_counter()
        print(f"Countdown started: '{self.current_exercise.name}' begins "
              f"in {self.START_COUNTDOWN_SECONDS:.0f}s")

    def _activate_exercise(self, landmarks: Optional[Dict] = None):
        """Called once the pre-exercise countdown elapses."""
        self._countdown_active = False
        self.current_exercise.start()
        self._position_gate.reset()
        if landmarks:
            # "Home" anchor for the position-drift check -- wherever the user
            # is standing right as counting begins, not a fixed frame position.
            self._position_gate.calibrate(landmarks)
        self._frame_count = 0
        self._exercise_start_time = time.perf_counter()
        self._last_quality_warnings = []
        if self._logger:
            resolution = f"{self.camera.width}x{self.camera.height}"
            self._logger.start_session(self.current_exercise.name, resolution)
        print(f"Started exercise: {self.current_exercise.name}")

    def stop_exercise(self):
        """Stop the current exercise (also cancels a pending countdown)."""
        self._countdown_active = False
        self.current_exercise.stop()
        self._position_gate.reset()
        if self._logger:
            self._logger.end_session()
        print(f"Stopped exercise: {self.current_exercise.name}")

    def reset_exercise(self):
        """Reset exercise counters"""
        self.current_exercise.reset()
        print(f"Reset exercise: {self.current_exercise.name}")

    def is_exercise_active(self) -> bool:
        """True while actively counting reps OR while the pre-start countdown is running."""
        return self.current_exercise.is_active or self._countdown_active

    def toggle_diagnostics(self) -> bool:
        """Toggle diagnostic overlay on/off. Returns new state."""
        self.show_diagnostics = not self.show_diagnostics
        return self.show_diagnostics

    def _build_runtime_stats(self) -> Dict:
        """Compute rolling FPS and assemble runtime stats dict for the overlay."""
        fps = 0.0
        ts = self._frame_timestamps
        if len(ts) >= 2:
            elapsed = ts[-1] - ts[0]
            if elapsed > 0:
                fps = (len(ts) - 1) / elapsed

        return {
            "fps":                round(fps, 1),
            "processing_time_ms": round(self._last_proc_ms, 1),
            "frame_num":          self._frame_count,
            "pose_detected":      self._last_pose_detected,
            "logger_active":      bool(self._logger and self._logger.is_active),
            "logger_frames":      self._logger.frame_count if self._logger else 0,
            "skipped_frames":     self._logger.skipped_frames if self._logger else 0,
            "warned_frames":      self._logger.warned_frames if self._logger else 0,
            "quality_warnings":   self._last_quality_warnings,
        }

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self._logger and self._logger.is_active:
            self._logger.end_session()
        self.camera.stop()
        self.pose_detector.cleanup()
        print("System cleaned up")

    def run_headless(self, duration_seconds: int = 60):
        """Run without GUI for testing (displays in OpenCV window)"""
        if not self.initialize():
            return

        import time
        start_time = time.time()
        print("Running in headless mode. Press 'q' to quit, 's' to start/stop, 'r' to reset")

        try:
            while time.time() - start_time < duration_seconds:
                if not self.process_frame():
                    break
                if self.processed_frame is not None:
                    cv2.imshow('Exergaming System', self.processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if self.is_exercise_active():
                        self.stop_exercise()
                    else:
                        self.start_exercise()
                elif key == ord('r'):
                    self.reset_exercise()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
            self.cleanup()
