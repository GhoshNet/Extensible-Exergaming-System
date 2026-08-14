"""
Video Analyzer UI — Test exercise detection on pre-recorded videos with live overlay.

Usage:
    python video_analyzer_ui.py
    python video_analyzer_ui.py Videos/Squats/man-home-squats.mp4
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QComboBox, QFileDialog, QProgressBar,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont

sys.path.insert(0, str(Path(__file__).parent))

from src.pose.detector import PoseDetector
from src.exercises.squat import SquatExercise
from src.exercises.pushup import PushUpExercise
from src.exercises.jumping_jack import JumpingJackExercise
from src.feedback.visual import VisualFeedback

EXERCISE_MAP = {
    'Squat': SquatExercise,
    'Push-up': PushUpExercise,
    'Jumping Jack': JumpingJackExercise,
}

SPEED_OPTIONS = {
    '0.5x': 0.5,
    '1x':   1.0,
    '2x':   2.0,
    '4x':   4.0,
    'Max':  0.0,
}

FORM_COLORS = {
    'excellent': '#2ecc71',
    'good':      '#f1c40f',
    'fair':      '#e67e22',
    'poor':      '#e74c3c',
}

APP_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
QFrame {
    border: 1px solid #2a2a4a;
    border-radius: 6px;
}
QPushButton {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #3a3a5a;
    border-radius: 5px;
    padding: 8px 18px;
    font-size: 13px;
}
QPushButton:hover  { background-color: #0f3460; }
QPushButton:pressed { background-color: #533483; }
QPushButton:disabled { color: #444; border-color: #2a2a3a; }
QComboBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #3a3a5a;
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 13px;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0e0;
    selection-background-color: #0f3460;
}
QListWidget {
    background-color: #12122a;
    color: #e0e0e0;
    border: 1px solid #2a2a4a;
    border-radius: 5px;
    font-size: 12px;
}
QProgressBar {
    background-color: #12122a;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    height: 10px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2ecc71;
    border-radius: 3px;
}
"""


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class VideoWorker(QThread):
    """Reads frames, runs pose detection and exercise logic, emits results to UI."""

    frame_ready   = pyqtSignal(object)   # np.ndarray (BGR, with overlays drawn)
    stats_updated = pyqtSignal(dict)
    rep_completed = pyqtSignal(int, str) # (rep_number, form_value)
    finished      = pyqtSignal(dict)
    error         = pyqtSignal(str)

    def __init__(self, video_path: str, exercise_name: str, speed: float):
        super().__init__()
        self.video_path    = video_path
        self.exercise_name = exercise_name
        self.speed         = speed   # 0.0 = process as fast as possible
        self._stop_flag    = False
        self._paused_flag  = False

    def stop(self):
        self._stop_flag = True

    def pause(self):
        self._paused_flag = True

    def resume(self):
        self._paused_flag = False

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

        exercise = EXERCISE_MAP[self.exercise_name]()
        detector = PoseDetector()
        visual   = VisualFeedback()

        exercise.start()

        frame_index      = 0
        frames_with_pose = 0
        prev_reps        = 0
        rep_log          = []
        frame_interval   = 1.0 / src_fps   # real-time spacing at 1×

        while not self._stop_flag:
            # Pause support
            while self._paused_flag and not self._stop_flag:
                time.sleep(0.05)
            if self._stop_flag:
                break

            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # --- Pose detection & exercise analysis ---
            pose_detected = detector.detect(frame)
            if pose_detected:
                frames_with_pose += 1
                frame     = detector.draw_landmarks(frame)
                landmarks = detector.get_all_landmarks_dict(frame.shape)
                # Video's own timeline, not wall-clock -- at speed > 1x this
                # loop runs faster than real-time, so time.perf_counter()
                # would make every rep look impossibly fast regardless of
                # how quickly the video is actually being played back.
                video_timestamp_ms = (frame_index / src_fps) * 1000
                exercise.analyze_pose(landmarks, video_timestamp_ms)

            # --- Visual overlay ---
            info = exercise.get_info()
            visual.draw_complete_overlay(frame, info, show_debug=True)

            # --- Thin progress strip at the bottom of the video frame ---
            progress = frame_index / max(total_frames, 1)
            bar_h    = max(6, height // 80)
            cv2.rectangle(frame, (0, height - bar_h), (width, height), (20, 20, 20), -1)
            cv2.rectangle(frame, (0, height - bar_h),
                          (int(width * progress), height), (46, 204, 113), -1)

            self.frame_ready.emit(frame.copy())

            # --- Rep detection ---
            current_reps = exercise.get_rep_count()
            if current_reps > prev_reps:
                form_val = exercise.get_form_feedback().value
                rep_log.append(form_val)
                self.rep_completed.emit(current_reps, form_val)
                prev_reps = current_reps

            # --- Stats ---
            detect_rate = frames_with_pose / frame_index * 100
            extra = {k: info[k] for k in
                     ('knee_angle', 'elbow_angle', 'arm_angle', 'leg_spread')
                     if k in info}
            self.stats_updated.emit({
                'reps':          current_reps,
                'form':          info.get('form', 'good'),
                'state':         info.get('state', 'idle'),
                'frame':         frame_index,
                'total_frames':  total_frames,
                'elapsed_video': frame_index / src_fps,
                'duration':      duration,
                'detect_rate':   detect_rate,
                **extra,
            })

            # --- Pace to source FPS (at chosen speed) ---
            if self.speed > 0:
                target = frame_interval / self.speed
                sleep  = target - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        cap.release()
        detector.cleanup()

        # Final summary
        fc = {}
        for r in ('excellent', 'good', 'fair', 'poor'):
            fc[r] = rep_log.count(r)

        self.finished.emit({
            'reps':         exercise.get_rep_count(),
            'frames':       frame_index,
            'total_frames': total_frames,
            'detect_rate':  frames_with_pose / max(frame_index, 1) * 100,
            'form_counts':  fc,
            'rep_log':      rep_log,
            'completed':    not self._stop_flag,
        })


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class VideoAnalyzerWindow(QMainWindow):

    def __init__(self, initial_video: str = None):
        super().__init__()
        self.worker      = None
        self.video_path  = None
        self._paused     = False
        self._fps_frames = 0
        self._fps_timer  = time.time()

        self._build_ui()

        if initial_video and Path(initial_video).exists():
            self._load_video(initial_video)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Exercise Video Analyzer")
        self.setGeometry(80, 80, 1440, 900)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._top_bar())

        mid = QHBoxLayout()
        mid.setSpacing(10)
        mid.addWidget(self._video_panel(), stretch=3)
        mid.addWidget(self._stats_panel(), stretch=1)
        layout.addLayout(mid)

        layout.addWidget(self._bottom_bar())

    def _top_bar(self) -> QWidget:
        bar    = QWidget()
        bar.setStyleSheet("border: none;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.browse_btn = QPushButton("Browse Video…")
        self.browse_btn.setFixedWidth(140)
        self.browse_btn.clicked.connect(self.on_browse)
        layout.addWidget(self.browse_btn)

        self.path_label = QLabel("No video selected")
        self.path_label.setStyleSheet("color: #555; font-size: 12px; border: none;")
        layout.addWidget(self.path_label, stretch=1)

        layout.addWidget(QLabel("Exercise:"))
        self.exercise_combo = QComboBox()
        for name in EXERCISE_MAP:
            self.exercise_combo.addItem(name)
        self.exercise_combo.setFixedWidth(140)
        layout.addWidget(self.exercise_combo)

        layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for label in SPEED_OPTIONS:
            self.speed_combo.addItem(label)
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.setFixedWidth(80)
        layout.addWidget(self.speed_combo)

        return bar

    def _video_panel(self) -> QFrame:
        panel  = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel("Select a video and press  Start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #0d0d1a; color: #333; "
            "font-size: 18px; border: none; border-radius: 4px;"
        )
        self.video_label.setMinimumSize(860, 520)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.video_label)
        return panel

    def _stats_panel(self) -> QFrame:
        panel  = QFrame()
        panel.setMinimumWidth(270)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)

        # Title
        title = QLabel("Live Stats")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #888; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Big rep counter
        self.reps_label = QLabel("0")
        self.reps_label.setStyleSheet(
            "font-size: 80px; font-weight: bold; color: #2ecc71; border: none;"
        )
        self.reps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.reps_label)

        sub = QLabel("REPS")
        sub.setStyleSheet("font-size: 12px; color: #555; border: none; margin-top: -14px;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        # Form badge
        self.form_label = QLabel("—")
        self.form_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.form_label.setFixedHeight(40)
        self.form_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #555;"
            "border: 1px solid #2a2a4a; border-radius: 5px; padding: 4px;"
        )
        layout.addWidget(self.form_label)

        # State
        self.state_label = QLabel("State: IDLE")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("font-size: 12px; color: #666; border: none;")
        layout.addWidget(self.state_label)

        # Angle / spread
        self.angle_label = QLabel("")
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angle_label.setStyleSheet("font-size: 12px; color: #666; border: none;")
        layout.addWidget(self.angle_label)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("border: none; border-top: 1px solid #2a2a4a; margin: 4px 0;")
        layout.addWidget(div)

        # Rep log
        rep_log_title = QLabel("Rep Log")
        rep_log_title.setStyleSheet("font-size: 12px; color: #888; border: none;")
        layout.addWidget(rep_log_title)

        self.rep_list = QListWidget()
        self.rep_list.setMaximumHeight(220)
        self.rep_list.setStyleSheet(
            "QListWidget { background-color: #12122a; border: 1px solid #2a2a4a; "
            "border-radius: 4px; font-size: 12px; }"
        )
        layout.addWidget(self.rep_list)

        layout.addStretch()

        # Pose detection rate
        self.detect_label = QLabel("Pose detection: —")
        self.detect_label.setStyleSheet("font-size: 11px; color: #555; border: none;")
        layout.addWidget(self.detect_label)

        return panel

    def _bottom_bar(self) -> QWidget:
        bar    = QWidget()
        bar.setStyleSheet("border: none;")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setFixedWidth(110)
        self.start_btn.clicked.connect(self.on_start)
        self.start_btn.setEnabled(False)
        ctrl.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setFixedWidth(100)
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)
        ctrl.addWidget(self.pause_btn)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setFixedWidth(90)
        self.reset_btn.clicked.connect(self.on_reset)
        self.reset_btn.setEnabled(False)
        ctrl.addWidget(self.reset_btn)

        ctrl.addStretch()

        self.time_label = QLabel("0.0s / 0.0s")
        self.time_label.setStyleSheet("font-size: 12px; color: #666;")
        ctrl.addWidget(self.time_label)

        sep = QLabel("|")
        sep.setStyleSheet("color: #333;")
        ctrl.addWidget(sep)

        self.fps_label = QLabel("— fps")
        self.fps_label.setStyleSheet("font-size: 12px; color: #666;")
        ctrl.addWidget(self.fps_label)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #333;")
        ctrl.addWidget(sep2)

        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_label.setStyleSheet("font-size: 12px; color: #666;")
        ctrl.addWidget(self.frame_label)

        layout.addLayout(ctrl)
        return bar

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_browse(self):
        default_dir = str(Path(__file__).parent / "Videos")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Exercise Video", default_dir,
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        self.video_path = path
        name = Path(path).name
        self.path_label.setText(name)
        self.path_label.setStyleSheet("color: #ccc; font-size: 12px; border: none;")
        self.start_btn.setEnabled(True)

        # Auto-select exercise from folder name if possible
        folder = Path(path).parent.name.lower()
        for ex_name in EXERCISE_MAP:
            if ex_name.lower().replace('-', '').replace(' ', '') in folder.replace(' ', '').replace('-', ''):
                self.exercise_combo.setCurrentText(ex_name)
                break
            if 'squat' in folder:
                self.exercise_combo.setCurrentText('Squat')
                break
            if 'push' in folder:
                self.exercise_combo.setCurrentText('Push-up')
                break
            if 'jack' in folder or 'jumping' in folder:
                self.exercise_combo.setCurrentText('Jumping Jack')
                break

    def on_start(self):
        if self.worker and self.worker.isRunning():
            return
        if not self.video_path:
            return

        self._reset_stats()

        exercise_name = self.exercise_combo.currentText()
        speed         = SPEED_OPTIONS[self.speed_combo.currentText()]

        self.worker = VideoWorker(self.video_path, exercise_name, speed)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.stats_updated.connect(self.on_stats)
        self.worker.rep_completed.connect(self.on_rep)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.exercise_combo.setEnabled(False)
        self.speed_combo.setEnabled(False)
        self._paused      = False
        self._fps_frames  = 0
        self._fps_timer   = time.time()

    def on_pause(self):
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

    def on_reset(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self._reset_stats()
        self.start_btn.setEnabled(bool(self.video_path))
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self.reset_btn.setEnabled(False)
        self.exercise_combo.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.video_label.setText("Select a video and press  Start")
        self.video_label.setStyleSheet(
            "background-color: #0d0d1a; color: #333; "
            "font-size: 18px; border: none; border-radius: 4px;"
        )

    def _reset_stats(self):
        self.reps_label.setText("0")
        self.form_label.setText("—")
        self.form_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #555;"
            "border: 1px solid #2a2a4a; border-radius: 5px; padding: 4px;"
        )
        self.state_label.setText("State: IDLE")
        self.angle_label.setText("")
        self.detect_label.setText("Pose detection: —")
        self.rep_list.clear()
        self.progress_bar.setValue(0)
        self.time_label.setText("0.0s / 0.0s")
        self.frame_label.setText("Frame 0 / 0")
        self.fps_label.setText("— fps")
        self._paused     = False
        self._fps_frames = 0
        self._fps_timer  = time.time()

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def on_frame(self, frame: np.ndarray):
        # FPS counter
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.fps_label.setText(f"{self._fps_frames / elapsed:.1f} fps")
            self._fps_frames = 0
            self._fps_timer  = now

        # Convert and display
        h, w, ch = frame.shape
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(img)
        scaled = pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def on_stats(self, stats: dict):
        self.reps_label.setText(str(stats['reps']))

        form  = stats.get('form', 'good').lower()
        color = FORM_COLORS.get(form, '#888')
        self.form_label.setText(form.upper())
        self.form_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
            "border: 1px solid #2a2a4a; border-radius: 5px; padding: 4px;"
        )

        self.state_label.setText(f"State: {stats.get('state', 'idle').upper()}")

        if 'knee_angle' in stats:
            self.angle_label.setText(f"Knee angle: {stats['knee_angle']}°")
        elif 'elbow_angle' in stats:
            self.angle_label.setText(f"Elbow angle: {stats['elbow_angle']}°")
        elif 'arm_angle' in stats:
            self.angle_label.setText(
                f"Arm: {stats['arm_angle']}°   Spread: {stats.get('leg_spread', 0):.2f}×"
            )

        total = max(stats['total_frames'], 1)
        self.progress_bar.setValue(int(stats['frame'] / total * 1000))
        self.time_label.setText(
            f"{stats['elapsed_video']:.1f}s / {stats['duration']:.1f}s"
        )
        self.frame_label.setText(
            f"Frame {stats['frame']} / {stats['total_frames']}"
        )
        self.detect_label.setText(f"Pose detection: {stats['detect_rate']:.1f}%")

    def on_rep(self, rep_num: int, form_val: str):
        color = FORM_COLORS.get(form_val.lower(), '#888')
        item  = QListWidgetItem(f"  Rep {rep_num:2d}  —  {form_val.upper()}")
        item.setForeground(QColor(color))
        self.rep_list.addItem(item)
        self.rep_list.scrollToBottom()

    def on_finished(self, summary: dict):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self.reset_btn.setEnabled(True)
        self.exercise_combo.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.progress_bar.setValue(1000)

        if not summary.get('completed', True):
            return   # user clicked Reset — don't show dialog

        reps    = summary['reps']
        detect  = summary['detect_rate']
        fc      = summary['form_counts']
        rep_log = summary['rep_log']

        lines = [
            f"Total reps detected:  {reps}",
            f"Pose detection rate:  {detect:.1f}%",
            "",
            "Form breakdown:",
        ]
        for rating in ('excellent', 'good', 'fair', 'poor'):
            n = fc.get(rating, 0)
            if n and reps > 0:
                pct = n / reps * 100
                lines.append(f"    {rating.capitalize():<12} {n}  ({pct:.0f}%)")

        if reps == 0:
            lines.append(
                "\nNo reps detected. Try adjusting the exercise type or check"
                "\nthat the person is clearly visible in the video."
            )

        msg = QMessageBox(self)
        msg.setWindowTitle("Analysis Complete")
        msg.setText("\n".join(lines))
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStyleSheet(
            "QMessageBox { background-color: #1a1a2e; color: #e0e0e0; }"
            "QPushButton { background-color: #16213e; color: #e0e0e0; "
            "border: 1px solid #3a3a5a; border-radius: 4px; padding: 6px 20px; }"
        )
        msg.exec()

    def on_error(self, message: str):
        QMessageBox.critical(self, "Error", message)
        self._reset_stats()
        self.start_btn.setEnabled(True)
        self.exercise_combo.setEnabled(True)
        self.speed_combo.setEnabled(True)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    initial = sys.argv[1] if len(sys.argv) > 1 else None
    window  = VideoAnalyzerWindow(initial_video=initial)
    window.show()
    sys.exit(app.exec())
