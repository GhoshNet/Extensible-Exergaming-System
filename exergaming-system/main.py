"""
Exergaming System — Unified Application Entry Point

Launches a single window with two modes:
  • Live Camera  — real-time pose detection and rep counting via webcam
  • Video Analysis — load a pre-recorded video, replay with overlays, inspect results

Run:
    python main.py
"""

import sys
import time
import json
import threading
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QComboBox, QCheckBox, QFileDialog, QProgressBar,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy,
    QRadioButton, QButtonGroup, QLineEdit, QScrollArea,
    QSpacerItem, QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QFontDatabase

sys.path.insert(0, str(Path(__file__).parent))

from src.controller import ExergamingController, EXERCISE_DEFINITIONS, _load_exercise_definitions
from src.exercises.generic_exercise import GenericExercise
from src.exercises.exercise_definition import ExerciseDefinition, ExercisePrompts, SignalCondition
from src.pose.detector import PoseDetector
from src.feedback.visual import (
    VisualFeedback, _generate_default_instructions, _generate_default_almost_there,
    _definition_states, generate_default_safety_message,
)
from src.feedback.form_errors import FormAnalyzer
from src.feedback.diagnostic_overlay import DiagnosticOverlay
from src.learning.demonstration_capture import DemonstrationCapture, CANDIDATE_RATIO_SIGNALS
from src.learning.exercise_learner import ExerciseLearner

_RATIO_NAMES = {r[0] for r in CANDIDATE_RATIO_SIGNALS}


# ─────────────────────────────────────────────────────────────────────────────
#  Design tokens
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "bg":       "#0d0d1a",
    "surface":  "#141428",
    "panel":    "#1a1a2e",
    "card":     "#16213e",
    "card2":    "#1c2a4d",   # one step lighter than card -- hover/pressed states
    "border":   "#2b2b52",
    "border_soft": "#252545",
    "accent":   "#6c63ff",
    "accent2":  "#4f46e5",
    "accent_press": "#3f37c9",
    "success":  "#10b981",
    "success2": "#059669",
    "warning":  "#f59e0b",
    "orange":   "#f97316",
    "error":    "#ef4444",
    "error2":   "#dc2626",
    "text":     "#f0f0fa",
    "sub":      "#8a8aac",   # lifted from #6b6b8a for better legibility
    "muted":    "#2a2a45",
}

FORM_COLORS = {
    "excellent": _C["success"],
    "good":      _C["warning"],
    "fair":      _C["orange"],
    "poor":      _C["error"],
}

SPEED_OPTIONS = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0, "Max": 0.0}

APP_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {_C['bg']};
    color: {_C['text']};
    font-family: -apple-system, "Segoe UI", sans-serif;
}}
QFrame {{
    background-color: {_C['panel']};
    border: 1px solid {_C['border_soft']};
    border-radius: 10px;
}}
QPushButton {{
    background-color: {_C['card']};
    color: {_C['text']};
    border: 1px solid {_C['border']};
    border-radius: 7px;
    padding: 9px 20px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover  {{ background-color: {_C['card2']}; border-color: {_C['accent']}; }}
QPushButton:pressed {{ background-color: {_C['accent2']}; border-color: {_C['accent2']}; }}
QPushButton:disabled {{ background-color: {_C['muted']}; color: {_C['sub']}; border-color: {_C['muted']}; }}
QComboBox {{
    background-color: {_C['card']};
    color: {_C['text']};
    border: 1px solid {_C['border']};
    border-radius: 7px;
    padding: 7px 12px;
    font-size: 13px;
    selection-background-color: {_C['accent']};
}}
QComboBox:hover  {{ border-color: {_C['accent']}; }}
QComboBox:focus  {{ border-color: {_C['accent']}; }}
QComboBox:disabled {{ color: {_C['sub']}; background-color: {_C['muted']}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background-color: {_C['card']};
    color: {_C['text']};
    border: 1px solid {_C['accent']};
    border-radius: 6px;
    selection-background-color: {_C['accent']};
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}}
QLineEdit {{
    background-color: {_C['card']};
    color: {_C['text']};
    border: 1px solid {_C['border']};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {_C['accent']};
}}
QLineEdit:hover  {{ border-color: {_C['sub']}; }}
QLineEdit:focus  {{ border-color: {_C['accent']}; }}
QLineEdit:disabled {{ color: {_C['sub']}; background-color: {_C['muted']}; }}
QListWidget {{
    background-color: {_C['surface']};
    color: {_C['text']};
    border: 1px solid {_C['border_soft']};
    border-radius: 7px;
    font-size: 12px;
    outline: none;
}}
QListWidget::item {{ padding: 3px 6px; border-radius: 4px; }}
QListWidget::item:hover {{ background-color: {_C['card']}; }}
QProgressBar {{
    background-color: {_C['surface']};
    border: none;
    border-radius: 3px;
    height: 6px;
}}
QProgressBar::chunk {{ background-color: {_C['success']}; border-radius: 3px; }}
QCheckBox {{ color: {_C['text']}; font-size: 13px; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {_C['border']}; background-color: {_C['card']};
}}
QCheckBox::indicator:hover {{ border-color: {_C['accent']}; }}
QCheckBox::indicator:checked {{
    background-color: {_C['accent']}; border-color: {_C['accent']};
}}
QRadioButton {{ color: {_C['text']}; font-size: 13px; spacing: 8px; }}
QRadioButton::indicator {{
    width: 16px; height: 16px; border-radius: 9px;
    border: 1px solid {_C['border']}; background-color: {_C['card']};
}}
QRadioButton::indicator:hover {{ border-color: {_C['accent']}; }}
QRadioButton::indicator:checked {{
    border: 5px solid {_C['accent']}; background-color: {_C['panel']};
}}
QLabel {{ background: transparent; border: none; }}
QScrollBar:vertical {{
    background: {_C['panel']}; width: 6px; border-radius: 3px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {_C['border']}; border-radius: 3px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {_C['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QToolTip {{
    background-color: {_C['card2']}; color: {_C['text']};
    border: 1px solid {_C['accent']}; border-radius: 5px;
    padding: 5px 8px; font-size: 12px;
}}
"""

# ── Semantic button styles ──────────────────────────────────────────────────
# Every button in the app should use one of these instead of the bare
# QPushButton default, so intent is visible at a glance: primary = the one
# action to take on this screen; secondary = supporting/reversible action;
# success/danger = state-changing verbs (Start/Stop).

_BTN_PRIMARY = (
    f"QPushButton {{ background-color: {_C['accent']}; color: #ffffff;"
    f"  border: none; border-radius: 7px; padding: 10px 20px;"
    f"  font-size: 13px; font-weight: 600; }}"
    f"QPushButton:hover {{ background-color: {_C['accent2']}; }}"
    f"QPushButton:pressed {{ background-color: {_C['accent_press']}; }}"
    f"QPushButton:disabled {{ background-color: {_C['muted']}; color: {_C['sub']}; }}"
)
_BTN_SECONDARY = (
    f"QPushButton {{ background-color: transparent; color: {_C['text']};"
    f"  border: 1px solid {_C['border']}; border-radius: 7px; padding: 9px 18px;"
    f"  font-size: 13px; font-weight: 600; }}"
    f"QPushButton:hover {{ background-color: {_C['card']}; border-color: {_C['accent']}; }}"
    f"QPushButton:pressed {{ background-color: {_C['muted']}; }}"
    f"QPushButton:disabled {{ color: {_C['sub']}; border-color: {_C['muted']}; background: transparent; }}"
)
_START_BTN_ACTIVE = (
    f"QPushButton {{ background-color: {_C['error']}; color: #fff;"
    f"  border: none; border-radius: 7px; padding: 12px 20px;"
    f"  font-size: 14px; font-weight: 600; }}"
    f"QPushButton:hover {{ background-color: {_C['error2']}; }}"
    f"QPushButton:disabled {{ background-color: {_C['muted']}; color: {_C['sub']}; }}"
)
_START_BTN_IDLE = (
    f"QPushButton {{ background-color: {_C['success']}; color: #fff;"
    f"  border: none; border-radius: 7px; padding: 12px 20px;"
    f"  font-size: 14px; font-weight: 600; }}"
    f"QPushButton:hover {{ background-color: {_C['success2']}; }}"
    f"QPushButton:disabled {{ background-color: {_C['muted']}; color: {_C['sub']}; }}"
)
_BTN_SUCCESS = _START_BTN_IDLE   # semantic alias for non-toggle "go" actions

_NAV_BTN_ACTIVE = (
    f"QPushButton {{"
    f"  background-color: {_C['accent']}; color: #fff;"
    f"  border: none; border-radius: 7px;"
    f"  padding: 8px 24px; font-size: 13px; font-weight: 600;"
    f"}}"
    f"QPushButton:hover {{ background-color: {_C['accent2']}; }}"
)
_NAV_BTN_IDLE = (
    f"QPushButton {{"
    f"  background-color: transparent; color: {_C['sub']};"
    f"  border: none; border-radius: 7px;"
    f"  padding: 8px 24px; font-size: 13px; font-weight: 500;"
    f"}}"
    f"QPushButton:hover {{ background-color: {_C['muted']}; color: {_C['text']}; }}"
)


def _combo_style(min_width: int = 0, margin_left: int = 0) -> str:
    """
    Shared QComboBox style -- box + hover/focus border + popup list.
    Always use this (never a raw ad-hoc stylesheet) so every dropdown's
    popup gets proper dark-theme contrast: an instance-level stylesheet that
    forgets the "QComboBox QAbstractItemView" rule silently falls back to
    the OS default popup (light background, hard-to-read text on this
    theme), which is what made several dropdowns look broken.
    """
    extra = f"min-width: {min_width}px;" if min_width else ""
    margin = f"margin-left: {margin_left}px;" if margin_left else ""
    return (
        f"QComboBox {{ background-color: {_C['card']}; color: {_C['text']}; "
        f"  border: 1px solid {_C['border']}; border-radius: 7px; "
        f"  padding: 7px 10px; font-size: 13px; {extra} {margin} }}"
        f"QComboBox:hover {{ border-color: {_C['accent']}; }}"
        f"QComboBox:focus {{ border-color: {_C['accent']}; }}"
        f"QComboBox:disabled {{ color: {_C['sub']}; background-color: {_C['muted']}; }}"
        f"QComboBox::drop-down {{ border: none; width: 22px; }}"
        f"QComboBox QAbstractItemView {{ background-color: {_C['card']}; color: {_C['text']}; "
        f"  border: 1px solid {_C['accent']}; border-radius: 6px; "
        f"  selection-background-color: {_C['accent']}; selection-color: #ffffff; "
        "  outline: none; padding: 4px; }"
    )


def _input_style(margin_left: int = 0) -> str:
    """Shared QLineEdit style with a visible focus ring."""
    margin = f"margin-left: {margin_left}px;" if margin_left else ""
    return (
        f"QLineEdit {{ background-color: {_C['card']}; color: {_C['text']}; "
        f"  border: 1px solid {_C['border']}; border-radius: 6px; "
        f"  padding: 8px 12px; font-size: 13px; {margin} }}"
        f"QLineEdit:hover {{ border-color: {_C['sub']}; }}"
        f"QLineEdit:focus {{ border-color: {_C['accent']}; }}"
        f"QLineEdit:disabled {{ color: {_C['sub']}; background-color: {_C['muted']}; }}"
    )


def _divider() -> QFrame:
    """Thin horizontal divider line."""
    d = QFrame()
    d.setFrameShape(QFrame.Shape.HLine)
    d.setStyleSheet(
        f"border: none; border-top: 1px solid {_C['border']}; "
        "background: transparent; max-height: 1px; margin: 4px 0;"
    )
    return d


def _section_label(text: str) -> QLabel:
    """Small uppercase section heading."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"font-size: 10px; font-weight: 700; color: {_C['sub']}; "
        "letter-spacing: 1px; background: transparent;"
    )
    return lbl


def _lock_hint(text: str) -> QLabel:
    """
    Small pill shown at the top of a step card while it's locked (see
    TrainingPanel._update_step_gating). The card itself also dims via a
    QGraphicsOpacityEffect, but the dim alone is ambiguous ("locked" vs.
    "just nothing to do yet") -- this makes the reason explicit.
    """
    lbl = QLabel(f"🔒  {text}")
    lbl.setStyleSheet(
        f"font-size: 11px; font-weight: 600; color: {_C['warning']}; "
        f"background-color: rgba(245, 158, 11, 0.12); "
        "border-radius: 5px; padding: 5px 9px;"
    )
    lbl.setWordWrap(True)
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
#  VideoWorker — runs video file analysis in a background QThread
# ─────────────────────────────────────────────────────────────────────────────

class VideoWorker(QThread):
    frame_ready   = pyqtSignal(object)    # np.ndarray BGR with overlays
    stats_updated = pyqtSignal(dict)
    rep_completed = pyqtSignal(int, str)  # (rep_number, form_value)
    finished      = pyqtSignal(dict)
    error         = pyqtSignal(str)

    def __init__(self, video_path: str, exercise_name: str, speed: float,
                 show_diagnostics: bool = False):
        super().__init__()
        self.video_path    = video_path
        self.exercise_name = exercise_name
        self.speed         = speed
        self._stop_flag    = False
        self._paused_flag  = False
        # Read every frame by the worker, written from the GUI thread when the
        # checkbox or [D] is used -- same plain-bool pattern as the flags above,
        # so the panel can be toggled mid-playback.
        self.show_diagnostics = show_diagnostics

    def stop(self):   self._stop_flag   = True
    def pause(self):  self._paused_flag = True
    def resume(self): self._paused_flag = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open: {self.video_path}")
            return

        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration     = total_frames / src_fps

        if self.exercise_name not in EXERCISE_DEFINITIONS:
            self.error.emit(f"Unknown exercise: {self.exercise_name}")
            cap.release()
            return

        defn     = EXERCISE_DEFINITIONS[self.exercise_name]
        exercise = GenericExercise(defn)
        detector = PoseDetector()
        visual   = VisualFeedback()
        form_analyzer = FormAnalyzer(defn)
        diagnostic_overlay = DiagnosticOverlay()
        exercise.start()

        frame_index      = 0
        frames_with_pose = 0
        prev_reps        = 0
        rep_log          = []
        frame_interval   = 1.0 / src_fps
        # Rolling window for the diagnostics FPS read-out, mirroring the live
        # controller's. Here it reflects the playback rate (the loop sleeps to
        # match the source fps x speed); "Proc" is the honest cost-per-frame
        # number, since it excludes that sleep.
        frame_timestamps = deque(maxlen=30)
        last_proc_ms     = 0.0

        while not self._stop_flag:
            while self._paused_flag and not self._stop_flag:
                time.sleep(0.05)
            if self._stop_flag:
                break

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            proc_start = time.perf_counter()
            frame_timestamps.append(proc_start)
            pose_detected = detector.detect(frame)
            raw_lm = None
            form_errors = []

            is_tracking = False
            if pose_detected:
                frames_with_pose += 1
                raw_lm    = detector.get_raw_landmarks()
                landmarks = detector.get_all_landmarks_dict(frame.shape)
                # Video's own timeline, not wall-clock -- see the same fix in
                # video_analyzer_ui.py for why (matters whenever playback
                # speed != 1x).
                video_timestamp_ms = (frame_index / src_fps) * 1000
                exercise.analyze_pose(landmarks, video_timestamp_ms)

                info = exercise.get_info()
                current_state = info.get("state", "idle").upper()
                is_tracking = current_state == form_analyzer.definition.form.during_state
                form_errors = form_analyzer.analyze(
                    current_state,
                    exercise._form_tracked,
                    exercise._signal_values,
                )

            info = exercise.get_info()
            last_proc_ms = (time.perf_counter() - proc_start) * 1000

            # Coloured form skeleton (Contribution 2) — replaces plain white skeleton
            visual.draw_form_skeleton(
                frame, raw_lm, form_errors,
                relevant_parts=form_analyzer.relevant_body_parts,
                is_tracking=is_tracking,
            )
            visual.draw_complete_overlay(frame, info, show_debug=True, definition=defn)
            if form_errors:
                visual.draw_form_errors(frame, form_errors)
            if info.get('safety_violated'):
                visual.draw_safety_warning(frame, defn.prompts.over_limit or defn.safety_message)

            # Diagnostics panel — same overlay the live panel uses, so the two
            # modes read identically side by side. Drawn before the progress
            # strip so the strip stays on top of it.
            if self.show_diagnostics:
                fps = 0.0
                if len(frame_timestamps) >= 2:
                    elapsed = frame_timestamps[-1] - frame_timestamps[0]
                    if elapsed > 0:
                        fps = (len(frame_timestamps) - 1) / elapsed
                diagnostic_overlay.draw(
                    frame,
                    {
                        "fps":                round(fps, 1),
                        "processing_time_ms": round(last_proc_ms, 1),
                        "frame_num":          frame_index,
                        "pose_detected":      pose_detected,
                        # No SessionLogger and no PositionGate in video analysis
                        # -- gating is deliberately not applied here, so counts
                        # stay comparable with the evaluation runs.
                        "logger_active":      False,
                        "logger_frames":      0,
                        "skipped_frames":     0,
                        "warned_frames":      0,
                        "quality_warnings":   [],
                    },
                    info,
                )

            # Progress strip along the bottom of the video frame
            progress = frame_index / max(total_frames, 1)
            bar_h = max(5, height // 100)
            cv2.rectangle(frame, (0, height - bar_h), (width, height), (20, 20, 30), -1)
            cv2.rectangle(
                frame, (0, height - bar_h),
                (int(width * progress), height), (16, 185, 129), -1
            )

            self.frame_ready.emit(frame.copy())

            current_reps = info.get("rep_count", 0)
            if current_reps > prev_reps:
                form_val = info.get("form", "good")
                rep_log.append(form_val)
                self.rep_completed.emit(current_reps, form_val)
                prev_reps = current_reps

            detect_rate = frames_with_pose / frame_index * 100
            extra = {k: info[k] for k in
                     ("knee_angle", "elbow_angle", "arm_angle", "leg_spread_ratio")
                     if k in info}
            self.stats_updated.emit({
                "reps":          current_reps,
                "form":          info.get("form", "good"),
                "state":         info.get("state", "idle"),
                "frame":         frame_index,
                "total_frames":  total_frames,
                "elapsed_video": frame_index / src_fps,
                "duration":      duration,
                "detect_rate":   detect_rate,
                **extra,
            })

            if self.speed > 0:
                sleep = (frame_interval / self.speed) - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        cap.release()
        detector.cleanup()

        fc = {r: rep_log.count(r) for r in ("excellent", "good", "fair", "poor")}
        self.finished.emit({
            "reps":         exercise.get_info().get("rep_count", 0),
            "frames":       frame_index,
            "total_frames": total_frames,
            "detect_rate":  frames_with_pose / max(frame_index, 1) * 100,
            "form_counts":  fc,
            "rep_log":      rep_log,
            "completed":    not self._stop_flag,
        })


# ─────────────────────────────────────────────────────────────────────────────
#  LivePanel — live webcam exercise detection
# ─────────────────────────────────────────────────────────────────────────────

class LivePanel(QWidget):

    def __init__(self, controller: ExergamingController):
        super().__init__()
        self.controller = controller
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_frame)
        self._build_ui()

    # ── construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self._video_panel(), stretch=3)
        root.addWidget(self._control_panel(), stretch=1)

    def _video_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {_C['surface']}; border: 1px solid {_C['border']}; border-radius: 10px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("Camera loading…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"color: {_C['sub']}; font-size: 16px; background: transparent; border: none;"
        )
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.video_label)
        return frame

    def _control_panel(self) -> QFrame:
        panel = QFrame()
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # ── Exercise selector ──────────────────────────────────────────
        layout.addWidget(_section_label("Exercise"))
        self.exercise_combo = QComboBox()
        self.exercise_combo.setStyleSheet(_combo_style())
        for name in self.controller.get_available_exercises():
            self.exercise_combo.addItem(name)
        self.exercise_combo.currentTextChanged.connect(self._on_exercise_changed)
        layout.addWidget(self.exercise_combo)

        layout.addWidget(_divider())

        # ── Rep counter ────────────────────────────────────────────────
        self.reps_label = QLabel("0")
        self.reps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reps_label.setStyleSheet(
            f"font-size: 72px; font-weight: 700; color: {_C['success']}; "
            "letter-spacing: -2px;"
        )
        layout.addWidget(self.reps_label)

        reps_sub = QLabel("REPS")
        reps_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reps_sub.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {_C['sub']}; "
            "letter-spacing: 2px;"
        )
        layout.addWidget(reps_sub)

        # ── Form badge ─────────────────────────────────────────────────
        self.form_badge = QLabel("—")
        self.form_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.form_badge.setFixedHeight(38)
        self.form_badge.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C['sub']}; "
            f"border: 1px solid {_C['border']}; border-radius: 8px; padding: 4px;"
        )
        layout.addWidget(self.form_badge)

        # ── Live signals ───────────────────────────────────────────────
        self.state_label = QLabel("State: IDLE")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        layout.addWidget(self.state_label)

        self.angle_label = QLabel("")
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angle_label.setStyleSheet(
            f"font-size: 12px; color: {_C['sub']}; font-family: monospace;"
        )
        layout.addWidget(self.angle_label)

        layout.addWidget(_divider())

        # ── Diagnostics toggle ─────────────────────────────────────────
        self.diag_checkbox = QCheckBox("Diagnostics overlay  [D]")
        self.diag_checkbox.stateChanged.connect(
            lambda s: setattr(self.controller, "show_diagnostics", bool(s))
        )
        layout.addWidget(self.diag_checkbox)

        layout.addStretch()

        # ── Action buttons ─────────────────────────────────────────────
        self.start_btn = QPushButton("Start Exercise")
        self.start_btn.setStyleSheet(_START_BTN_IDLE)
        self.start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_btn)

        self.reset_btn = QPushButton("Reset Count")
        self.reset_btn.setStyleSheet(_BTN_SECONDARY)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self.reset_btn)

        layout.addWidget(_divider())

        # ── Status dot ─────────────────────────────────────────────────
        self.status_label = QLabel("● Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            f"font-size: 12px; color: {_C['success']}; font-weight: 500;"
        )
        layout.addWidget(self.status_label)

        return panel

    # ── Timer control ──────────────────────────────────────────────────────

    def start_display(self):
        """Start the 30ms display timer (called when this panel becomes visible)."""
        self._timer.start(30)

    def stop_display(self):
        """Pause display updates (called when switching to another mode)."""
        self._timer.stop()

    # ── Frame + stats update ───────────────────────────────────────────────

    def _update_frame(self):
        frame = self.controller.get_current_frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(img)
            lw = self.video_label.width()
            if lw > 0:
                self.video_label.setPixmap(
                    pix.scaledToWidth(lw, Qt.TransformationMode.SmoothTransformation)
                )

        info = self.controller.get_exercise_info()
        if not info:
            return

        self.reps_label.setText(str(info.get("reps", 0)))

        form  = info.get("form", "good").lower()
        color = FORM_COLORS.get(form, _C["sub"])
        self.form_badge.setText(form.upper())
        self.form_badge.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {color}; "
            f"border: 1px solid {color}44; border-radius: 8px; padding: 4px; "
            "background: transparent;"
        )

        self.state_label.setText(f"State: {info.get('state', 'idle').upper()}")

        if "knee_angle" in info:
            self.angle_label.setText(f"Knee: {info['knee_angle']:.1f}°")
        elif "elbow_angle" in info:
            self.angle_label.setText(f"Elbow: {info['elbow_angle']:.1f}°")
        elif "arm_angle" in info:
            self.angle_label.setText(
                f"Arm: {info['arm_angle']:.1f}°   Spread: {info.get('leg_spread_ratio', 0):.2f}×"
            )

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_exercise_changed(self, name: str):
        if self.controller.is_exercise_active():
            self.controller.stop_exercise()
            self._set_idle_state()
        self.controller.set_exercise(name)
        self.angle_label.setText("")

    def _on_start_clicked(self):
        if self.controller.is_exercise_active():
            self.controller.stop_exercise()
            self._set_idle_state()
        else:
            self.controller.start_exercise()
            self.start_btn.setText("Stop Exercise")
            self.start_btn.setStyleSheet(_START_BTN_ACTIVE)
            self.status_label.setText("● Active")
            self.status_label.setStyleSheet(
                f"font-size: 12px; color: {_C['success']}; font-weight: 500;"
            )

    def _on_reset_clicked(self):
        self.controller.reset_exercise()
        self.reps_label.setText("0")
        self.form_badge.setText("—")
        self.form_badge.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C['sub']}; "
            f"border: 1px solid {_C['border']}; border-radius: 8px; padding: 4px;"
        )
        self.status_label.setText("● Reset")
        self.status_label.setStyleSheet(
            f"font-size: 12px; color: {_C['warning']}; font-weight: 500;"
        )

    def _set_idle_state(self):
        self.start_btn.setText("Start Exercise")
        self.start_btn.setStyleSheet(_START_BTN_IDLE)
        self.status_label.setText("● Ready")
        self.status_label.setStyleSheet(
            f"font-size: 12px; color: {_C['success']}; font-weight: 500;"
        )

    def toggle_diagnostics(self):
        """Called by the parent window's D-key handler."""
        new_state = self.controller.toggle_diagnostics()
        self.diag_checkbox.blockSignals(True)
        self.diag_checkbox.setChecked(new_state)
        self.diag_checkbox.blockSignals(False)

    def stop_exercise_if_active(self):
        """Called before the panel goes off-screen."""
        if self.controller.is_exercise_active():
            self.controller.stop_exercise()
            self._set_idle_state()


# ─────────────────────────────────────────────────────────────────────────────
#  VideoPanel — pre-recorded video analysis
# ─────────────────────────────────────────────────────────────────────────────

class VideoPanel(QWidget):

    def __init__(self, controller: "ExergamingController"):
        super().__init__()
        self._controller = controller
        self.worker      = None
        self.video_path  = None
        self._paused     = False
        self._fps_frames = 0
        self._fps_timer  = time.time()
        self._show_diagnostics = False
        self._build_ui()

    # ── construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(8)

        root.addWidget(self._top_bar())

        mid = QHBoxLayout()
        mid.setSpacing(12)
        mid.addWidget(self._video_frame(), stretch=3)
        mid.addWidget(self._stats_panel(), stretch=1)
        root.addLayout(mid)

        root.addWidget(self._bottom_bar())

    def _top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background: transparent;")
        bar.setFixedHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.browse_btn = QPushButton("Browse Video…")
        self.browse_btn.setFixedWidth(150)
        self.browse_btn.setStyleSheet(_BTN_SECONDARY)
        self.browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self.browse_btn)

        self.path_label = QLabel("No video selected")
        self.path_label.setStyleSheet(
            f"color: {_C['sub']}; font-size: 12px; font-style: italic;"
        )
        layout.addWidget(self.path_label, stretch=1)

        layout.addWidget(_section_label("Exercise"))
        self.exercise_combo = QComboBox()
        self.exercise_combo.setStyleSheet(_combo_style())
        for name in EXERCISE_DEFINITIONS:
            self.exercise_combo.addItem(name)
        self.exercise_combo.setFixedWidth(150)
        layout.addWidget(self.exercise_combo)

        layout.addWidget(_section_label("Speed"))
        self.speed_combo = QComboBox()
        self.speed_combo.setStyleSheet(_combo_style())
        for label in SPEED_OPTIONS:
            self.speed_combo.addItem(label)
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.setFixedWidth(80)
        layout.addWidget(self.speed_combo)

        return bar

    def _video_frame(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {_C['surface']}; border: 1px solid {_C['border']}; border-radius: 10px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("Browse a video and press  ▶  Start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"color: {_C['sub']}; font-size: 16px; background: transparent; border: none;"
        )
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.video_label)
        return frame

    def _stats_panel(self) -> QFrame:
        panel = QFrame()
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        layout.addWidget(_section_label("Current rep"))

        self.reps_label = QLabel("0")
        self.reps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reps_label.setStyleSheet(
            f"font-size: 72px; font-weight: 700; color: {_C['success']}; "
            "letter-spacing: -2px;"
        )
        layout.addWidget(self.reps_label)

        reps_sub = QLabel("REPS")
        reps_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reps_sub.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {_C['sub']}; "
            "letter-spacing: 2px;"
        )
        layout.addWidget(reps_sub)

        self.form_badge = QLabel("—")
        self.form_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.form_badge.setFixedHeight(38)
        self.form_badge.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C['sub']}; "
            f"border: 1px solid {_C['border']}; border-radius: 8px; padding: 4px;"
        )
        layout.addWidget(self.form_badge)

        self.state_label = QLabel("State: IDLE")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        layout.addWidget(self.state_label)

        self.angle_label = QLabel("")
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angle_label.setStyleSheet(
            f"font-size: 12px; color: {_C['sub']}; font-family: monospace;"
        )
        layout.addWidget(self.angle_label)

        layout.addWidget(_divider())
        layout.addWidget(_section_label("Rep log"))

        self.rep_list = QListWidget()
        self.rep_list.setMaximumHeight(200)
        layout.addWidget(self.rep_list)

        layout.addStretch()

        self.detect_label = QLabel("Detection: —")
        self.detect_label.setStyleSheet(f"font-size: 11px; color: {_C['sub']};")
        layout.addWidget(self.detect_label)

        return panel

    def _bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self.start_btn = QPushButton("▶   Start")
        self.start_btn.setFixedWidth(110)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(_BTN_SUCCESS)
        self.start_btn.clicked.connect(self._on_start)
        ctrl.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setFixedWidth(100)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(_BTN_SECONDARY)
        self.pause_btn.clicked.connect(self._on_pause)
        ctrl.addWidget(self.pause_btn)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setFixedWidth(90)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setStyleSheet(_BTN_SECONDARY)
        self.reset_btn.clicked.connect(self._on_reset)
        ctrl.addWidget(self.reset_btn)

        # Same toggle as the live panel, so the diagnostics can be shown over a
        # recorded video exactly as they are over the camera.
        self.diag_checkbox = QCheckBox("Diagnostics overlay  [D]")
        self.diag_checkbox.stateChanged.connect(self._on_diag_toggled)
        ctrl.addWidget(self.diag_checkbox)

        ctrl.addStretch()

        self.time_label = QLabel("0.0s / 0.0s")
        self.time_label.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        ctrl.addWidget(self.time_label)

        ctrl.addWidget(self._sep())

        self.fps_label = QLabel("— fps")
        self.fps_label.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        ctrl.addWidget(self.fps_label)

        ctrl.addWidget(self._sep())

        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_label.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        ctrl.addWidget(self.frame_label)

        layout.addLayout(ctrl)
        return bar

    def _sep(self) -> QLabel:
        lbl = QLabel("|")
        lbl.setStyleSheet(f"color: {_C['muted']}; background: transparent;")
        return lbl

    def _on_diag_toggled(self, state):
        """Apply the toggle to a running worker, and remember it for the next."""
        self._show_diagnostics = bool(state)
        if self.worker:
            self.worker.show_diagnostics = self._show_diagnostics

    def toggle_diagnostics(self):
        """Flip the overlay from the [D] shortcut (checkbox stays in sync)."""
        self.diag_checkbox.setChecked(not self.diag_checkbox.isChecked())

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_browse(self):
        default = str(Path(__file__).parent / "Videos")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Exercise Video", default,
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        self.video_path = path
        name = Path(path).name
        self.path_label.setText(name)
        self.path_label.setStyleSheet(
            f"color: {_C['text']}; font-size: 12px; font-style: normal;"
        )
        self.start_btn.setEnabled(True)

        # Auto-detect exercise from folder name
        folder = Path(path).parent.name.lower()
        for ex_name in EXERCISE_DEFINITIONS:
            slug = ex_name.lower().replace("-", "").replace(" ", "")
            if slug in folder.replace("-", "").replace(" ", ""):
                self.exercise_combo.setCurrentText(ex_name)
                return
        if "squat"   in folder: self.exercise_combo.setCurrentText("Squat")
        elif "push"  in folder: self.exercise_combo.setCurrentText("Push-up")
        elif "jack"  in folder or "jumping" in folder:
            self.exercise_combo.setCurrentText("Jumping Jack")

    def _on_start(self):
        if self.worker and self.worker.isRunning():
            return
        if not self.video_path:
            return

        self._reset_stats()
        exercise_name = self.exercise_combo.currentText()
        speed         = SPEED_OPTIONS[self.speed_combo.currentText()]

        # Pause the live-camera MediaPipe inference before starting a worker
        # that creates its own PoseDetector — two simultaneous TFLite instances
        # on macOS Metal cause a hard crash.
        self._controller.processing_paused = True

        self.worker = VideoWorker(self.video_path, exercise_name, speed,
                                  show_diagnostics=self._show_diagnostics)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.rep_completed.connect(self._on_rep)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.exercise_combo.setEnabled(False)
        self.speed_combo.setEnabled(False)
        self._paused      = False
        self._fps_frames  = 0
        self._fps_timer   = time.time()

    def _on_pause(self):
        if not self.worker:
            return
        if not self._paused:
            self.worker.pause()
            self.pause_btn.setText("▶  Resume")
            self._paused = True
        else:
            self.worker.resume()
            self.pause_btn.setText("⏸  Pause")
            self._paused = False

    def _on_reset(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self._controller.processing_paused = False   # resume live inference
        self._reset_stats()
        self.start_btn.setEnabled(bool(self.video_path))
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self.reset_btn.setEnabled(False)
        self.exercise_combo.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.video_label.setText("Browse a video and press  ▶  Start")

    def _reset_stats(self):
        self.reps_label.setText("0")
        self.form_badge.setText("—")
        self.form_badge.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {_C['sub']}; "
            f"border: 1px solid {_C['border']}; border-radius: 8px; padding: 4px;"
        )
        self.state_label.setText("State: IDLE")
        self.angle_label.setText("")
        self.detect_label.setText("Detection: —")
        self.rep_list.clear()
        self.progress_bar.setValue(0)
        self.time_label.setText("0.0s / 0.0s")
        self.frame_label.setText("Frame 0 / 0")
        self.fps_label.setText("— fps")
        self._paused = False
        self._fps_frames = 0
        self._fps_timer  = time.time()

    # ── Worker signal handlers ─────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.fps_label.setText(f"{self._fps_frames / elapsed:.1f} fps")
            self._fps_frames = 0
            self._fps_timer  = now

        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_stats(self, stats: dict):
        self.reps_label.setText(str(stats["reps"]))

        form  = stats.get("form", "good").lower()
        color = FORM_COLORS.get(form, _C["sub"])
        self.form_badge.setText(form.upper())
        self.form_badge.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {color}; "
            f"border: 1px solid {color}44; border-radius: 8px; padding: 4px; "
            "background: transparent;"
        )
        self.state_label.setText(f"State: {stats.get('state', 'idle').upper()}")

        if "knee_angle"  in stats: self.angle_label.setText(f"Knee: {stats['knee_angle']:.1f}°")
        elif "elbow_angle" in stats: self.angle_label.setText(f"Elbow: {stats['elbow_angle']:.1f}°")
        elif "arm_angle"   in stats:
            self.angle_label.setText(
                f"Arm: {stats['arm_angle']:.1f}°   Spread: {stats.get('leg_spread_ratio', 0):.2f}×"
            )

        total = max(stats["total_frames"], 1)
        self.progress_bar.setValue(int(stats["frame"] / total * 1000))
        self.time_label.setText(
            f"{stats['elapsed_video']:.1f}s / {stats['duration']:.1f}s"
        )
        self.frame_label.setText(f"Frame {stats['frame']} / {stats['total_frames']}")
        self.detect_label.setText(f"Detection: {stats['detect_rate']:.1f}%")

    def _on_rep(self, rep_num: int, form_val: str):
        color = FORM_COLORS.get(form_val.lower(), _C["sub"])
        item  = QListWidgetItem(f"  Rep {rep_num:2d}  —  {form_val.upper()}")
        item.setForeground(QColor(color))
        self.rep_list.addItem(item)
        self.rep_list.scrollToBottom()

    def _on_finished(self, summary: dict):
        self._controller.processing_paused = False   # resume live inference
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self.reset_btn.setEnabled(True)
        self.exercise_combo.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.progress_bar.setValue(1000)

        if not summary.get("completed", True):
            return

        reps    = summary["reps"]
        detect  = summary["detect_rate"]
        fc      = summary["form_counts"]

        lines = [
            f"Total reps:        {reps}",
            f"Pose detection:    {detect:.1f}%",
            "",
            "Form breakdown:",
        ]
        for rating in ("excellent", "good", "fair", "poor"):
            n = fc.get(rating, 0)
            if n and reps > 0:
                lines.append(f"    {rating.capitalize():<12} {n}  ({n/reps*100:.0f}%)")

        if reps == 0:
            lines.append(
                "\nNo reps detected. Check that the person is clearly visible "
                "and the correct exercise is selected."
            )

        msg = QMessageBox(self)
        msg.setWindowTitle("Analysis Complete")
        msg.setText("\n".join(lines))
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStyleSheet(
            f"QMessageBox {{ background-color: {_C['panel']}; color: {_C['text']}; }}"
            f"QPushButton {{ background-color: {_C['card']}; color: {_C['text']}; "
            f"border: 1px solid {_C['border']}; border-radius: 6px; padding: 6px 20px; }}"
        )
        msg.exec()

    def _on_error(self, message: str):
        self._controller.processing_paused = False   # resume live inference
        QMessageBox.critical(self, "Error", message)
        self._reset_stats()
        self.start_btn.setEnabled(True)
        self.exercise_combo.setEnabled(True)
        self.speed_combo.setEnabled(True)

    def stop_worker_if_running(self):
        """Called when switching away from this panel."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self._controller.processing_paused = False   # always resume


# ─────────────────────────────────────────────────────────────────────────────
#  AnalysisWorker — runs pose detection + DemonstrationCapture in background
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    """
    Processes a demo video frame-by-frame and builds an ExerciseLearner.

    The controller's background thread is paused BEFORE this worker starts
    (via controller.processing_paused = True) so that only one TFLite/MediaPipe
    instance is active on the Metal GPU at any time.  The TrainingPanel resumes
    processing in the done/error/stop callbacks.
    """

    progress = pyqtSignal(int, int)   # (current_frame, total_frames)
    complete = pyqtSignal()           # learner and report available on self
    error    = pyqtSignal(str)

    def __init__(self, video_path: str, exercise_name: str):
        super().__init__()
        self.video_path    = video_path
        self.exercise_name = exercise_name
        self.learner       = None   # ExerciseLearner — set after complete
        self.report        = None   # dict — set after complete
        self._stop_flag    = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {self.video_path}")
            return

        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        detector = PoseDetector()
        capture  = DemonstrationCapture()
        capture.start_recording()

        frame_num = 0
        while not self._stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            pose_ok = detector.detect(frame)
            lms = detector.get_all_landmarks_dict(frame.shape) if pose_ok else {}
            capture.process_frame(frame_num, lms, pose_ok)
            if frame_num % 15 == 0:
                self.progress.emit(frame_num, total)

        self.progress.emit(frame_num, total)
        cap.release()
        detector.cleanup()

        if self._stop_flag:
            return   # cancelled — don't emit complete or error

        frames = capture.stop_recording()
        try:
            self.learner = ExerciseLearner(frames, self.exercise_name)
            self.report  = self.learner.get_analysis_report()
            self.complete.emit()
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  SignalReviewWidget — checkbox table for selecting which signals matter
# ─────────────────────────────────────────────────────────────────────────────

class SignalReviewWidget(QFrame):
    """
    Shows one row per detected signal.  The user checks or unchecks signals
    to tell the algorithm which movements were intentional vs incidental.
    The [PRIMARY] badge updates live as the selection changes.
    """

    changed = pyqtSignal()   # emitted whenever any checkbox is toggled

    def __init__(self):
        super().__init__()
        self._report      = None
        self._checkboxes  = {}   # signal_name → QCheckBox
        self._row_frames  = {}   # signal_name → QFrame (for badge updates)
        self._badges      = {}   # signal_name → QLabel (PRIMARY / incidental)
        self._layout      = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._primary_lbl = None  # updated label at bottom
        self._placeholder = QLabel(
            "Analyze a video in Step 2 to see the detected signals here."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {_C['sub']}; font-size: 13px; padding: 40px 20px;"
        )
        self._layout.addWidget(self._placeholder)

    # ── Public API ────────────────────────────────────────────────────────

    def populate(self, report: dict):
        """Build rows from the analysis report dict."""
        self._report = report
        self._checkboxes.clear()
        self._row_frames.clear()
        self._badges.clear()

        # Remove everything except placeholder (which we hide, not destroy —
        # it's a persistent widget reused by clear() later).
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget and widget is not self._placeholder:
                widget.deleteLater()
        self._placeholder.hide()

        important = report.get("important_signals", [])
        if not important:
            lbl = QLabel("No significant joint movements found in this video.")
            lbl.setStyleSheet(f"color: {_C['error']}; padding: 16px;")
            self._layout.addWidget(lbl)
            return

        # ── Column header ──────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background: transparent;")
        hrow = QHBoxLayout(hdr)
        hrow.setContentsMargins(14, 6, 14, 6)
        hrow.setSpacing(0)
        for text, stretch in [("Signal", 3), ("Range", 1), ("Peak", 1), ("Trough", 1), ("", 2)]:
            lbl = QLabel(text.upper())
            lbl.setStyleSheet(
                f"font-size: 10px; font-weight: 700; color: {_C['sub']}; "
                "letter-spacing: 1px;"
            )
            hrow.addWidget(lbl, stretch)
        self._layout.addWidget(hdr)
        self._layout.addWidget(_divider())

        # ── Signal rows ────────────────────────────────────────────────────
        ranges  = report.get("signal_ranges",  {})
        peaks   = report.get("signal_peaks",   {})
        troughs = report.get("signal_troughs", {})

        for i, name in enumerate(important):
            row_frame = QFrame()
            row_frame.setStyleSheet(
                f"background-color: {'#12122a' if i % 2 == 0 else _C['surface']}; "
                "border: none; border-radius: 0;"
            )
            row = QHBoxLayout(row_frame)
            row.setContentsMargins(14, 8, 14, 8)
            row.setSpacing(0)

            # Checkbox + signal name
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(
                f"font-size: 13px; color: {_C['text']}; font-weight: 500; spacing: 8px;"
            )
            cb.toggled.connect(self._on_toggle)
            self._checkboxes[name] = cb
            row.addWidget(cb, 3)

            # Range
            rng  = ranges.get(name, 0)
            unit = "" if name in _RATIO_NAMES else "°"
            lbl_rng = QLabel(f"{rng:.1f}{unit}")
            lbl_rng.setStyleSheet(f"font-size: 12px; color: {_C['text']}; font-family: monospace;")
            row.addWidget(lbl_rng, 1)

            # Peak
            lbl_pk = QLabel(f"{peaks.get(name,0):.1f}")
            lbl_pk.setStyleSheet(f"font-size: 12px; color: {_C['sub']}; font-family: monospace;")
            row.addWidget(lbl_pk, 1)

            # Trough
            lbl_tr = QLabel(f"{troughs.get(name,0):.1f}")
            lbl_tr.setStyleSheet(f"font-size: 12px; color: {_C['sub']}; font-family: monospace;")
            row.addWidget(lbl_tr, 1)

            # Badge (PRIMARY / incidental?)
            badge = QLabel("")
            badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(badge, 2)

            self._row_frames[name] = row_frame
            self._badges[name]     = badge
            self._layout.addWidget(row_frame)

        self._layout.addWidget(_divider())

        # ── Primary indicator label ────────────────────────────────────────
        self._primary_lbl = QLabel("")
        self._primary_lbl.setStyleSheet(
            f"font-size: 12px; color: {_C['sub']}; padding: 10px 14px 4px 14px;"
        )
        self._layout.addWidget(self._primary_lbl)
        self._layout.addStretch()

        self._update_badges()

    def get_approved(self) -> list:
        """Return names of all currently checked signals."""
        return [n for n, cb in self._checkboxes.items() if cb.isChecked()]

    def get_primary(self) -> str:
        """Return the highest-range non-ratio checked signal (the state-machine driver)."""
        approved = self.get_approved()
        for n in approved:
            if n not in _RATIO_NAMES:
                return n
        return approved[0] if approved else ""

    def clear(self):
        """Reset to placeholder state."""
        self._report = None
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget and widget is not self._placeholder:
                widget.deleteLater()
        self._checkboxes.clear()
        self._row_frames.clear()
        self._badges.clear()
        self._layout.addWidget(self._placeholder)
        self._placeholder.show()

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_toggle(self):
        approved = self.get_approved()
        if not approved:
            # At least one signal must remain — restore the one just unticked
            sender = self.sender()
            if sender is not None:
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
        self._update_badges()
        self.changed.emit()

    def _update_badges(self):
        if not self._badges:
            return
        primary   = self.get_primary()
        approved  = set(self.get_approved())
        important = list(self._checkboxes.keys())

        for i, name in enumerate(important):
            badge = self._badges.get(name)
            if badge is None:
                continue
            if name not in approved:
                badge.setText("")
                continue
            if name == primary:
                badge.setText("● Primary signal")
                badge.setStyleSheet(
                    f"font-size: 11px; font-weight: 700; color: {_C['accent']}; "
                    "letter-spacing: 0.3px;"
                )
            elif i >= 2 and name not in _RATIO_NAMES:
                badge.setText("May be incidental")
                badge.setStyleSheet(f"font-size: 11px; color: {_C['orange']};")
            else:
                badge.setText("")

        if self._primary_lbl:
            rng = (self._report or {}).get("signal_ranges", {}).get(primary, 0)
            unit = "" if primary in _RATIO_NAMES else "°"
            self._primary_lbl.setText(
                f"Primary signal: {primary} ({rng:.1f}{unit} range) — drives the state machine"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  TrainingPanel — Step-by-step exercise training wizard
# ─────────────────────────────────────────────────────────────────────────────

class TrainingPanel(QWidget):
    """
    Four-step wizard:
      Step 1 — Choose exercise (retrain existing, create new, or just edit
               an existing exercise's prompts without retraining)
      Step 2 — Load demo video and run analysis (skipped in edit-prompts mode)
      Step 3 — Review detected signals (skipped in edit-prompts mode)
      Step 4 — Prompts: customize what the user sees during a rep, with
               auto-generated defaults pre-filled and editable; save applies
               to both new/retrained exercises and prompt-only edits.
    """

    DEFS_DIR = Path(__file__).parent / "src" / "exercises" / "definitions"

    def __init__(self, controller: "ExergamingController", reload_callback):
        super().__init__()
        self._controller      = controller
        self._reload_callback = reload_callback
        self._worker          = None
        self._video_path      = None
        self._edit_definition: Optional[ExerciseDefinition] = None  # loaded in edit mode
        self._prompt_dirty = {"resting": False, "tracked": False, "almost_there": False}
        self._build_ui()

        # Steps 3 and 4 start locked (dimmed + non-interactive) until their
        # prerequisite step completes -- see _update_step_gating(). The
        # opacity effect goes on the inner card, not the wrapper, so the
        # lock-hint pill above it stays fully legible while dimmed.
        self._step3_opacity = QGraphicsOpacityEffect(self._step3_inner)
        self._step3_inner.setGraphicsEffect(self._step3_opacity)
        self._step4_opacity = QGraphicsOpacityEffect(self._step4_inner)
        self._step4_inner.setGraphicsEffect(self._step4_opacity)
        self._update_step_gating()

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        outer.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        scroll.setWidget(content)

        root = QHBoxLayout(content)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(20)

        # Left column — Steps 1 + 2
        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._step1_card())
        self._step2_frame = self._step2_card()
        left.addWidget(self._step2_frame)
        left.addStretch()
        left_w = QWidget()
        left_w.setStyleSheet("background: transparent;")
        left_w.setLayout(left)
        left_w.setFixedWidth(460)
        root.addWidget(left_w)

        # Right column — Step 3 (signals) + Step 4 (prompts + save)
        right = QVBoxLayout()
        right.setSpacing(16)
        self._step3_frame = self._step3_card()
        right.addWidget(self._step3_frame)
        self._step4_frame = self._step4_card()
        right.addWidget(self._step4_frame)
        right_w = QWidget()
        right_w.setStyleSheet("background: transparent;")
        right_w.setLayout(right)
        root.addWidget(right_w, stretch=1)

    # ── Step 1 card ───────────────────────────────────────────────────────

    def _step1_card(self) -> QFrame:
        card = QFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(_section_label("Step 1 — Exercise"))

        title = QLabel("What would you like to do?")
        title.setStyleSheet(f"font-size: 14px; color: {_C['text']}; font-weight: 600;")
        layout.addWidget(title)

        # Radio: retrain existing
        self.retrain_radio = QRadioButton("Retrain an existing exercise with my own video")
        self.retrain_radio.setChecked(True)
        layout.addWidget(self.retrain_radio)

        self.existing_combo = QComboBox()
        self.existing_combo.setStyleSheet(_combo_style(margin_left=24))
        for name in EXERCISE_DEFINITIONS:
            self.existing_combo.addItem(name)
        self.existing_combo.currentTextChanged.connect(self._on_existing_combo_changed)
        layout.addWidget(self.existing_combo)

        # Radio: create new
        self.create_radio = QRadioButton("Define a brand new exercise")
        layout.addWidget(self.create_radio)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g.  Bicep Curl,  Lunge,  Shoulder Press")
        self.name_input.setStyleSheet(_input_style(margin_left=24))
        self.name_input.setEnabled(False)
        layout.addWidget(self.name_input)

        # Radio: edit prompts only (no retraining)
        self.edit_radio = QRadioButton("Edit prompts for an existing exercise (no retraining)")
        layout.addWidget(self.edit_radio)

        # Wire radios
        self.retrain_radio.toggled.connect(self._on_mode_toggle)
        self.create_radio.toggled.connect(self._on_mode_toggle)
        self.edit_radio.toggled.connect(self._on_mode_toggle)

        return card

    # ── Step 2 card ───────────────────────────────────────────────────────

    def _step2_card(self) -> QFrame:
        card = QFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(_section_label("Step 2 — Demo Video"))

        desc = QLabel(
            "Record yourself doing the exercise (5–15 reps). "
            "Then browse to that video file."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: 13px; color: {_C['sub']}; line-height: 1.4;")
        layout.addWidget(desc)

        # Browse row
        browse_row = QHBoxLayout()
        browse_row.setSpacing(10)
        self.browse_btn = QPushButton("Browse Video…")
        self.browse_btn.setFixedWidth(140)
        self.browse_btn.setStyleSheet(_BTN_SECONDARY)
        self.browse_btn.clicked.connect(self._on_browse)
        browse_row.addWidget(self.browse_btn)

        self.video_path_lbl = QLabel("No video selected")
        self.video_path_lbl.setStyleSheet(
            f"font-size: 12px; color: {_C['sub']}; font-style: italic;"
        )
        self.video_path_lbl.setWordWrap(True)
        browse_row.addWidget(self.video_path_lbl, stretch=1)
        layout.addLayout(browse_row)

        # Analyze button
        self.analyze_btn = QPushButton("Analyze Video")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(_BTN_PRIMARY)
        self.analyze_btn.clicked.connect(self._on_analyze)
        layout.addWidget(self.analyze_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        layout.addWidget(self.status_lbl)

        return card

    # ── Step 3 card ───────────────────────────────────────────────────────

    def _step3_card(self) -> QWidget:
        # Outer wrapper holds the lock-hint pill (always full opacity) plus
        # the inner card (which is what actually dims + disables when
        # locked -- see _update_step_gating). Keeping the hint outside the
        # dimmed area is what makes it legible while everything below it
        # fades out, instead of the "why is the reason also dimmed" problem
        # you'd get from putting everything under one opacity effect.
        wrapper = QWidget()
        wrapper.setStyleSheet("background: transparent;")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(6)

        self._step3_lock_hint = _lock_hint("Complete Step 2 first")
        wrapper_layout.addWidget(self._step3_lock_hint)

        card = QFrame()
        self._step3_inner = card
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(_section_label("Step 3 — Review Signals"))

        self.review_desc = QLabel(
            "The algorithm detects which joints moved the most during your recording. "
            "Uncheck any that moved incidentally (e.g. arms swinging while squatting)."
        )
        self.review_desc.setWordWrap(True)
        self.review_desc.setStyleSheet(f"font-size: 13px; color: {_C['sub']}; line-height: 1.4;")
        layout.addWidget(self.review_desc)

        # Signal review widget
        self.signal_review = SignalReviewWidget()
        self.signal_review.changed.connect(self._on_signals_changed)
        layout.addWidget(self.signal_review, stretch=1)

        wrapper_layout.addWidget(card)
        return wrapper

    # ── Step 4 card — Prompts ────────────────────────────────────────────

    def _step4_card(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setStyleSheet("background: transparent;")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(6)

        self._step4_lock_hint = _lock_hint("Approve at least one signal in Step 3 first")
        wrapper_layout.addWidget(self._step4_lock_hint)

        card = QFrame()
        self._step4_inner = card
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        layout.addWidget(_section_label("Step 4 — Prompts"))

        desc = QLabel(
            "What the user sees on screen during a rep. Every field is "
            "pre-filled with a sensible default derived from the exercise's "
            "own signals — edit any of them, or leave them as-is."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: 13px; color: {_C['sub']}; line-height: 1.4;")
        layout.addWidget(desc)

        def _field_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"font-size: 12px; color: {_C['text']}; font-weight: 600; margin-top: 6px;")
            return lbl

        def _line_edit():
            le = QLineEdit()
            le.setStyleSheet(_input_style())
            return le

        layout.addWidget(_field_label("Resting — shown before the rep starts"))
        self.resting_prompt_input = _line_edit()
        self.resting_prompt_input.setPlaceholderText("e.g. Bend your elbow to begin the rep")
        self.resting_prompt_input.textEdited.connect(lambda: self._mark_dirty("resting"))
        layout.addWidget(self.resting_prompt_input)

        layout.addWidget(_field_label("Complete — shown to finish the rep"))
        self.tracked_prompt_input = _line_edit()
        self.tracked_prompt_input.setPlaceholderText("e.g. Extend your arm to complete the rep")
        self.tracked_prompt_input.textEdited.connect(lambda: self._mark_dirty("tracked"))
        layout.addWidget(self.tracked_prompt_input)

        layout.addWidget(_field_label("Almost there — close but not quite (yellow)"))
        self.almost_there_input = _line_edit()
        self.almost_there_input.setPlaceholderText("e.g. Bend your elbow a little further")
        self.almost_there_input.textEdited.connect(lambda: self._mark_dirty("almost_there"))
        layout.addWidget(self.almost_there_input)

        layout.addWidget(_divider())

        self.safety_checkbox = QCheckBox("Add a safety / over-limit warning")
        self.safety_checkbox.setStyleSheet(f"font-size: 13px; color: {_C['text']};")
        self.safety_checkbox.toggled.connect(self._on_safety_toggle)
        layout.addWidget(self.safety_checkbox)

        self.safety_row_widget = QWidget()
        safety_row = QHBoxLayout(self.safety_row_widget)
        safety_row.setContentsMargins(0, 4, 0, 0)
        safety_row.setSpacing(6)

        self.safety_signal_combo = QComboBox()
        self.safety_signal_combo.setStyleSheet(_combo_style())
        self.safety_signal_combo.currentTextChanged.connect(self._on_safety_signal_changed)
        safety_row.addWidget(self.safety_signal_combo, stretch=2)

        self.safety_direction_combo = QComboBox()
        self.safety_direction_combo.addItems(["below", "above"])
        self.safety_direction_combo.setStyleSheet(_combo_style())
        self.safety_direction_combo.currentTextChanged.connect(self._on_safety_signal_changed)
        safety_row.addWidget(self.safety_direction_combo, stretch=1)

        self.safety_threshold_input = QLineEdit()
        self.safety_threshold_input.setPlaceholderText("threshold")
        self.safety_threshold_input.setFixedWidth(80)
        self.safety_threshold_input.setStyleSheet(_input_style())
        safety_row.addWidget(self.safety_threshold_input, stretch=1)
        layout.addWidget(self.safety_row_widget)

        self.safety_message_input = _line_edit()
        self.safety_message_input.setPlaceholderText(
            "Warning shown when the safety limit is crossed"
        )
        layout.addWidget(self.safety_message_input)
        self.safety_row_widget.setVisible(False)
        self.safety_message_input.setVisible(False)

        layout.addWidget(_divider())

        # Save button (applies to all 3 Step 1 modes)
        self.save_btn = QPushButton("Save Exercise Definition")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(_BTN_SUCCESS)
        self.save_btn.clicked.connect(self._on_save)
        layout.addWidget(self.save_btn)

        self.save_status_lbl = QLabel("")
        self.save_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.save_status_lbl.setWordWrap(True)
        self.save_status_lbl.setStyleSheet(
            f"font-size: 13px; color: {_C['success']}; font-weight: 500;"
        )
        layout.addWidget(self.save_status_lbl)

        wrapper_layout.addWidget(card)
        return wrapper

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_mode_toggle(self):
        retrain = self.retrain_radio.isChecked()
        create  = self.create_radio.isChecked()
        edit    = self.edit_radio.isChecked()

        self.existing_combo.setEnabled(retrain or edit)
        self.name_input.setEnabled(create)
        if create:
            self.name_input.setFocus()

        # Video analysis (Step 2) and signal review (Step 3) are meaningless
        # in edit-prompts mode -- only Step 4 (prompts) applies.
        self._step2_frame.setVisible(not edit)
        self._step3_frame.setVisible(not edit)

        if edit:
            self._load_for_edit(self.existing_combo.currentText())
        else:
            # Leaving edit mode: save button reflects the train/retrain flow's
            # own state again (whether a video has been analyzed + approved).
            self._edit_definition = None
            self.save_btn.setEnabled(bool(self.signal_review.get_approved()))
        self._update_step_gating()

    def _on_existing_combo_changed(self, name: str):
        if self.edit_radio.isChecked() and name:
            self._load_for_edit(name)

    def _on_browse(self):
        default = str(Path(__file__).parent / "Videos")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Demo Video", default,
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if not path:
            return
        self._video_path = path
        self.video_path_lbl.setText(Path(path).name)
        self.video_path_lbl.setStyleSheet(
            f"font-size: 12px; color: {_C['text']}; font-style: normal;"
        )
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.signal_review.clear()
        self.status_lbl.setText("")
        self.save_status_lbl.setText("")
        # A new video means any previous analysis no longer applies -- re-lock
        # Step 3/4 until the new video has actually been analyzed.
        self._worker = None
        self._update_step_gating()

    def _on_analyze(self):
        if not self._video_path:
            return

        exercise_name = self._get_exercise_name()
        if not exercise_name:
            QMessageBox.warning(self, "No Exercise Name",
                                "Please enter a name for the new exercise.")
            self.name_input.setFocus()
            return

        self.analyze_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Preparing analysis…")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        self.save_btn.setEnabled(False)
        self.save_status_lbl.setText("")
        self.signal_review.clear()

        # Pause the background MediaPipe thread so AnalysisWorker can safely
        # create its own PoseDetector without two TFLite instances competing.
        self._controller.processing_paused = True

        # Give the background thread ~150 ms to finish its current frame
        # before the worker starts, then kick off the analysis.
        self._pending_exercise_name = exercise_name
        QTimer.singleShot(150, self._start_analysis_worker)

    def _start_analysis_worker(self):
        """Called 150 ms after _on_analyze to safely start the worker."""
        self.status_lbl.setText("Analyzing…")
        self._worker = AnalysisWorker(self._video_path, self._pending_exercise_name)
        self._worker.progress.connect(self._on_progress)
        self._worker.complete.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_progress(self, cur: int, total: int):
        self.progress_bar.setValue(int(cur / max(total, 1) * 1000))
        self.status_lbl.setText(f"Analyzing… {cur}/{total} frames")

    def _on_analysis_done(self):
        self._controller.processing_paused = False   # resume live inference
        self.progress_bar.setValue(1000)
        report  = self._worker.report
        n_sigs  = len(report.get("important_signals", []))
        n_total = len(report.get("all_signals", []))
        self.status_lbl.setText(
            f"Done — {report['total_frames']} frames analyzed.  "
            f"Found {n_sigs} important signals out of {n_total}."
        )
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {_C['success']};")
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.signal_review.populate(report)
        self.save_btn.setEnabled(True)
        self._refresh_prompt_defaults(force=True)
        self._update_step_gating()

    def _on_analysis_error(self, msg: str):
        self._controller.processing_paused = False   # resume live inference
        self.status_lbl.setText(f"Error: {msg}")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {_C['error']};")
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self._update_step_gating()

    def _on_signals_changed(self):
        self.save_btn.setEnabled(bool(self.signal_review.get_approved()))
        self._refresh_prompt_defaults(force=False)
        self._update_step_gating()

    def _build_preview_definition(self) -> Optional[ExerciseDefinition]:
        """Build (but don't save) a definition from the current signal
        selection, purely to derive Step 4's prompt defaults and the safety
        signal choices as the user reviews signals in Step 3."""
        if self._worker is None or self._worker.learner is None:
            return None
        approved = self.signal_review.get_approved()
        if not approved:
            return None
        try:
            name = self._get_exercise_name() or self._worker.learner._exercise_name
            self._worker.learner._exercise_name = name
            return self._worker.learner.learn(approved_signals=approved)
        except Exception:
            return None

    def _refresh_prompt_defaults(self, force: bool):
        definition = self._build_preview_definition()
        if definition is not None:
            self._populate_prompt_fields(definition, force=force)
            signal_names = [s.name for s in definition.angle_signals] + \
                           [s.name for s in definition.ratio_signals]
            self._set_safety_signal_choices(signal_names)

    def _populate_prompt_fields(self, definition: ExerciseDefinition, force: bool):
        """
        Fill Step 4's text fields from a definition. If force=True (fresh
        analysis, or a different exercise loaded for editing), every field
        is overwritten and dirty flags reset. If force=False (signal
        selection changed), only fields the user hasn't manually edited are
        refreshed, so in-progress edits are never silently clobbered.
        """
        if force:
            self._prompt_dirty = {"resting": False, "tracked": False, "almost_there": False}

        prompts = definition.prompts
        generated = _generate_default_instructions(definition)
        resting_states, tracked_state, _, _ = _definition_states(definition)
        resting_default = generated.get(resting_states[0].lower(), "") if resting_states else ""
        tracked_default = generated.get(tracked_state.lower(), "") if tracked_state else ""
        almost_default  = _generate_default_almost_there(definition)

        if not self._prompt_dirty["resting"]:
            self.resting_prompt_input.setText(prompts.resting or resting_default)
        if not self._prompt_dirty["tracked"]:
            self.tracked_prompt_input.setText(prompts.tracked or tracked_default)
        if not self._prompt_dirty["almost_there"]:
            self.almost_there_input.setText(prompts.almost_there or almost_default)

        if force:
            has_safety = definition.safety_condition is not None
            self.safety_checkbox.setChecked(has_safety)
            if has_safety:
                self.safety_direction_combo.setCurrentText(definition.safety_condition.direction)
                self.safety_threshold_input.setText(str(definition.safety_condition.threshold))
                self.safety_message_input.setText(
                    prompts.over_limit or definition.safety_message
                )
            else:
                self.safety_threshold_input.clear()
                self.safety_message_input.clear()

    def _mark_dirty(self, field: str):
        self._prompt_dirty[field] = True

    # ── Stepwise gating ──────────────────────────────────────────────────
    # Step 3 (signal review) and Step 4 (prompts) are only meaningful once
    # their prerequisite step has actually produced something -- lock them
    # (dimmed + non-interactive) until then, rather than leaving every field
    # editable from the moment the wizard opens.

    def _set_locked(self, frame: QFrame, effect: QGraphicsOpacityEffect,
                     hint: QLabel, locked: bool):
        frame.setEnabled(not locked)
        effect.setOpacity(0.4 if locked else 1.0)
        hint.setVisible(locked)
        # QGraphicsOpacityEffect's internal pixmap cache can go stale across
        # an opacity change, leaving a ghosted composite of the old + new
        # render until something forces a full repaint.
        frame.update()
        effect.update()

    def _update_step_gating(self):
        if self.edit_radio.isChecked():
            # Step 2/3 don't apply in edit mode (already hidden); Step 4
            # unlocks as soon as an exercise has been loaded to edit.
            self._set_locked(self._step4_inner, self._step4_opacity, self._step4_lock_hint,
                              self._edit_definition is None)
            return

        has_report = self._worker is not None and self._worker.report is not None
        self._set_locked(self._step3_inner, self._step3_opacity, self._step3_lock_hint,
                          not has_report)

        has_approved = has_report and bool(self.signal_review.get_approved())
        self._set_locked(self._step4_inner, self._step4_opacity, self._step4_lock_hint,
                          not has_approved)

    def _set_safety_signal_choices(self, signal_names: list):
        current = self.safety_signal_combo.currentText()
        self.safety_signal_combo.blockSignals(True)
        self.safety_signal_combo.clear()
        self.safety_signal_combo.addItems(signal_names)
        if current in signal_names:
            self.safety_signal_combo.setCurrentText(current)
        self.safety_signal_combo.blockSignals(False)

    def _on_safety_toggle(self, checked: bool):
        self.safety_row_widget.setVisible(checked)
        self.safety_message_input.setVisible(checked)
        if checked and not self.safety_message_input.text().strip():
            self._on_safety_signal_changed()

    def _on_safety_signal_changed(self):
        if not self.safety_checkbox.isChecked():
            return
        signal = self.safety_signal_combo.currentText()
        direction = self.safety_direction_combo.currentText()
        if signal:
            self.safety_message_input.setText(generate_default_safety_message(signal, direction))

    def _load_for_edit(self, exercise_name: str):
        """Load an existing exercise's saved definition straight into Step 4,
        with no video/analysis involved -- pure prompt editing."""
        if not exercise_name or exercise_name not in EXERCISE_DEFINITIONS:
            return
        self._edit_definition = EXERCISE_DEFINITIONS[exercise_name]
        self._populate_prompt_fields(self._edit_definition, force=True)
        signal_names = [s.name for s in self._edit_definition.angle_signals] + \
                       [s.name for s in self._edit_definition.ratio_signals]
        self._set_safety_signal_choices(signal_names)
        self.save_btn.setEnabled(True)
        self.save_status_lbl.setText("")
        self._update_step_gating()

    def _apply_prompts_to_definition(self, definition: ExerciseDefinition) -> None:
        """Read Step 4's current field values onto a definition in-place."""
        definition.prompts = ExercisePrompts(
            resting=self.resting_prompt_input.text().strip(),
            tracked=self.tracked_prompt_input.text().strip(),
            almost_there=self.almost_there_input.text().strip(),
            over_limit=self.safety_message_input.text().strip() if self.safety_checkbox.isChecked() else "",
        )
        if self.safety_checkbox.isChecked():
            signal = self.safety_signal_combo.currentText()
            try:
                threshold = float(self.safety_threshold_input.text())
            except ValueError:
                raise ValueError(
                    "Safety limit threshold must be a number "
                    f"(got {self.safety_threshold_input.text()!r})."
                )
            if not signal:
                raise ValueError("Pick a signal for the safety limit, or uncheck it.")
            definition.safety_condition = SignalCondition(
                signal=signal,
                direction=self.safety_direction_combo.currentText(),
                threshold=threshold,
            )
            definition.safety_message = self.safety_message_input.text().strip()
        else:
            definition.safety_condition = None
            definition.safety_message = ""

    def _find_definition_path(self, exercise_name: str) -> Optional[Path]:
        """Locate the JSON file backing an already-saved exercise by its
        'name' field (filenames don't map deterministically from the name,
        e.g. "Push-up" -> pushup.json), so editing an exercise overwrites
        the same file it was loaded from."""
        for path in self.DEFS_DIR.glob("*.json"):
            try:
                with open(path) as f:
                    if json.load(f).get("name") == exercise_name:
                        return path
            except (OSError, json.JSONDecodeError):
                continue
        return None

    def _on_save(self):
        if self.edit_radio.isChecked():
            self._save_edit_mode()
        else:
            self._save_trained_mode()

    def _save_edit_mode(self):
        if self._edit_definition is None:
            return
        path = self._find_definition_path(self._edit_definition.name)
        if path is None:
            QMessageBox.critical(self, "Save Failed",
                                 f"Could not find the definition file for "
                                 f"'{self._edit_definition.name}'.")
            return
        try:
            self._apply_prompts_to_definition(self._edit_definition)
            self._edit_definition.to_json(str(path))
            self.save_status_lbl.setText(
                f"Saved!  Prompts updated for '{self._edit_definition.name}'."
            )
            self.save_status_lbl.setStyleSheet(
                f"font-size: 13px; color: {_C['success']}; font-weight: 500;"
            )
            self._reload_callback(self._edit_definition.name)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Safety Limit", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _save_trained_mode(self):
        if self._worker is None or self._worker.learner is None:
            return

        approved = self.signal_review.get_approved()
        if not approved:
            QMessageBox.warning(self, "No Signals Selected",
                                "Select at least one signal before saving.")
            return

        exercise_name = self._get_exercise_name()
        if not exercise_name:
            QMessageBox.warning(self, "No Exercise Name",
                                "Please provide an exercise name.")
            return

        try:
            # Build definition with user-approved signals
            self._worker.learner._exercise_name = exercise_name
            definition = self._worker.learner.learn(approved_signals=approved)
            self._apply_prompts_to_definition(definition)

            # Determine output path. When retraining an exercise that is
            # already a learned one (e.g. "Learned Squat"), the slug already
            # contains "learned_" -- strip it so the filename doesn't double
            # up (learned_learned_squat.json) and instead overwrites the
            # same learned_squat.json each time it's retrained.
            slug = exercise_name.lower().replace(" ", "_").replace("-", "_")
            if slug.startswith("learned_"):
                slug = slug[len("learned_"):]
            out_dir  = Path(__file__).parent / "src" / "exercises" / "definitions"
            out_path = out_dir / f"learned_{slug}.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            definition.to_json(str(out_path))

            self.save_status_lbl.setText(
                f"Saved!  '{exercise_name}' is now available in Live Camera and Video Analysis."
            )
            self.save_status_lbl.setStyleSheet(
                f"font-size: 13px; color: {_C['success']}; font-weight: 500;"
            )

            # Reload definitions and refresh all dropdowns
            self._reload_callback(exercise_name)

        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Safety Limit", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_exercise_name(self) -> str:
        if self.retrain_radio.isChecked() or self.edit_radio.isChecked():
            return self.existing_combo.currentText()
        return self.name_input.text().strip()

    def refresh_exercise_list(self):
        """Rebuild the existing-exercise combo from current EXERCISE_DEFINITIONS."""
        self.existing_combo.blockSignals(True)
        self.existing_combo.clear()
        for name in EXERCISE_DEFINITIONS:
            self.existing_combo.addItem(name)
        self.existing_combo.blockSignals(False)

    def stop_worker_if_running(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._controller.processing_paused = False   # always resume


# ─────────────────────────────────────────────────────────────────────────────
#  AppWindow — top-level window with navigation bar
# ─────────────────────────────────────────────────────────────────────────────

class AppWindow(QMainWindow):

    def __init__(self, controller: ExergamingController, camera_available: bool = True):
        super().__init__()
        self.controller       = controller
        self.camera_available = camera_available
        self._build_ui()
        # Default to live camera mode
        self._switch_mode(0)

    def _build_ui(self):
        self.setWindowTitle("Exergaming System")
        self.setMinimumSize(1200, 720)
        self.resize(1440, 860)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_nav())

        # Thin accent separator under nav
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {_C['border']}; border: none; border-radius: 0;")
        layout.addWidget(sep)

        # Stacked panels
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background: transparent; border: none;")

        self.live_panel   = LivePanel(self.controller)
        self.video_panel  = VideoPanel(self.controller)
        self.train_panel  = TrainingPanel(self.controller, reload_callback=self._reload_exercises)

        self.stack.addWidget(self.live_panel)    # index 0
        self.stack.addWidget(self.video_panel)   # index 1
        self.stack.addWidget(self.train_panel)   # index 2

        layout.addWidget(self.stack)

    def _build_nav(self) -> QWidget:
        nav = QWidget()
        nav.setFixedHeight(56)
        nav.setStyleSheet(
            f"background-color: {_C['surface']}; border: none; border-radius: 0;"
        )
        layout = QHBoxLayout(nav)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(0)

        # App name
        logo = QLabel("ExerGaming")
        logo.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {_C['text']}; "
            "letter-spacing: -0.5px;"
        )
        layout.addWidget(logo)

        # Version / subtitle
        sub = QLabel("  Pose-based Exercise Detection")
        sub.setStyleSheet(f"font-size: 12px; color: {_C['sub']};")
        layout.addWidget(sub)

        layout.addStretch()

        # Mode toggle buttons
        self.live_nav_btn  = QPushButton("LIVE CAMERA")
        self.video_nav_btn = QPushButton("VIDEO ANALYSIS")
        self.train_nav_btn = QPushButton("TRAINING")
        for btn in (self.live_nav_btn, self.video_nav_btn, self.train_nav_btn):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.live_nav_btn.clicked.connect(lambda: self._switch_mode(0))
        self.video_nav_btn.clicked.connect(lambda: self._switch_mode(1))
        self.train_nav_btn.clicked.connect(lambda: self._switch_mode(2))

        layout.addWidget(self.live_nav_btn)
        layout.addSpacing(6)
        layout.addWidget(self.video_nav_btn)
        layout.addSpacing(6)
        layout.addWidget(self.train_nav_btn)

        layout.addStretch()

        # Camera status indicator
        cam_color = _C["success"] if self.camera_available else _C["error"]
        cam_text  = "● Camera ready" if self.camera_available else "● No camera"
        self.cam_status = QLabel(cam_text)
        self.cam_status.setStyleSheet(
            f"font-size: 12px; color: {cam_color}; font-weight: 500;"
        )
        layout.addWidget(self.cam_status)

        return nav

    def _switch_mode(self, index: int):
        """Switch between Live Camera (0), Video Analysis (1), Training (2)."""
        # Always stop live display when leaving it
        if index != 0:
            self.live_panel.stop_display()
            self.live_panel.stop_exercise_if_active()
        # Stop video worker when leaving video panel (also resumes processing)
        if index != 1:
            self.video_panel.stop_worker_if_running()
        # Stop training worker when leaving training panel (also resumes processing)
        if index != 2:
            self.train_panel.stop_worker_if_running()

        if index == 0:
            # Ensure processing is definitely unpaused on return to live mode
            self.controller.processing_paused = False
            self.live_panel.start_display()

        # Update nav button styles
        styles = [_NAV_BTN_IDLE, _NAV_BTN_IDLE, _NAV_BTN_IDLE]
        styles[index] = _NAV_BTN_ACTIVE
        self.live_nav_btn.setStyleSheet(styles[0])
        self.video_nav_btn.setStyleSheet(styles[1])
        self.train_nav_btn.setStyleSheet(styles[2])

        self.stack.setCurrentIndex(index)

    def _reload_exercises(self, new_name: str = ""):
        """Reload exercise definitions from disk and refresh all dropdowns."""
        EXERCISE_DEFINITIONS.clear()
        EXERCISE_DEFINITIONS.update(_load_exercise_definitions())

        # Refresh live panel combo
        current_live = self.live_panel.exercise_combo.currentText()
        self.live_panel.exercise_combo.blockSignals(True)
        self.live_panel.exercise_combo.clear()
        for name in EXERCISE_DEFINITIONS:
            self.live_panel.exercise_combo.addItem(name)
        idx = self.live_panel.exercise_combo.findText(current_live)
        if idx >= 0:
            self.live_panel.exercise_combo.setCurrentIndex(idx)
        self.live_panel.exercise_combo.blockSignals(False)

        # Refresh video panel combo
        current_vid = self.video_panel.exercise_combo.currentText()
        self.video_panel.exercise_combo.blockSignals(True)
        self.video_panel.exercise_combo.clear()
        for name in EXERCISE_DEFINITIONS:
            self.video_panel.exercise_combo.addItem(name)
        idx = self.video_panel.exercise_combo.findText(current_vid)
        if idx >= 0:
            self.video_panel.exercise_combo.setCurrentIndex(idx)
        self.video_panel.exercise_combo.blockSignals(False)

        # Refresh training panel existing-exercise combo
        self.train_panel.refresh_exercise_list()

        # Select the newly saved exercise in the video panel for immediate testing
        if new_name and new_name in EXERCISE_DEFINITIONS:
            idx = self.video_panel.exercise_combo.findText(new_name)
            if idx >= 0:
                self.video_panel.exercise_combo.setCurrentIndex(idx)

    # ── Keyboard shortcuts ─────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_D:
            # Route to whichever panel is on screen — both support the overlay.
            idx = self.stack.currentIndex()
            if idx == 0:
                self.live_panel.toggle_diagnostics()
            elif idx == 1:
                self.video_panel.toggle_diagnostics()
        super().keyPressEvent(event)

    # ── Window close ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.live_panel.stop_display()
        self.video_panel.stop_worker_if_running()
        self.train_panel.stop_worker_if_running()
        self.controller.cleanup()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    controller    = ExergamingController()
    camera_ok     = controller.initialize()

    if not camera_ok:
        print("Warning: camera not available — live mode disabled.")

    window = AppWindow(controller, camera_available=camera_ok)
    window.show()

    # Background thread: processes camera frames continuously.
    # Skips inference when processing_paused is set so VideoWorker /
    # AnalysisWorker can safely run their own PoseDetector on macOS Metal.
    def _process_frames():
        while controller.is_running:
            if camera_ok and not controller.processing_paused:
                controller.process_frame()
            time.sleep(0.01)

    thread = threading.Thread(target=_process_frames, daemon=True)
    thread.start()

    exit_code = app.exec()
    controller.cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
