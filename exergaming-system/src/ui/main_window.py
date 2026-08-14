"""
Main Window UI Module
PyQt6 GUI for the exergaming application
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFrame, QComboBox, QCheckBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Exergaming System")
        # 1400 wide: video panel ≈1050px → 16:9 height ≈590px + chrome ≈640 total
        self.setGeometry(100, 100, 1400, 640)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        central_widget.setLayout(main_layout)

        left_panel = self.create_video_panel()
        main_layout.addWidget(left_panel, stretch=3)

        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def create_video_panel(self) -> QFrame:
        """Create video display panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 360)

        layout.addWidget(self.video_label)
        layout.setContentsMargins(0, 0, 0, 0)
        return panel

    def create_control_panel(self) -> QFrame:
        """Create control panel with buttons and stats"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title
        title = QLabel("Exergaming System")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Exercise selector
        selector_label = QLabel("Select Exercise:")
        selector_label.setStyleSheet("font-size: 14px; margin-top: 15px;")
        layout.addWidget(selector_label)

        self.exercise_combo = QComboBox()
        self.exercise_combo.setStyleSheet("font-size: 14px; padding: 5px;")
        for name in self.controller.get_available_exercises():
            self.exercise_combo.addItem(name)
        self.exercise_combo.currentTextChanged.connect(self.on_exercise_changed)
        layout.addWidget(self.exercise_combo)

        # Stats
        self.reps_label = QLabel("Reps: 0")
        self.reps_label.setStyleSheet("font-size: 28px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(self.reps_label)

        self.form_label = QLabel("Form: GOOD")
        self.form_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.form_label)

        self.state_label = QLabel("State: IDLE")
        self.state_label.setStyleSheet("font-size: 14px; color: gray;")
        layout.addWidget(self.state_label)

        self.debug_label = QLabel("")
        self.debug_label.setStyleSheet("font-size: 12px; color: gray;")
        layout.addWidget(self.debug_label)

        self.diag_checkbox = QCheckBox("Diagnostics overlay  [D]")
        self.diag_checkbox.setStyleSheet("font-size: 12px; color: gray; margin-top: 8px;")
        self.diag_checkbox.stateChanged.connect(self.on_diagnostics_toggled)
        layout.addWidget(self.diag_checkbox)

        layout.addStretch()

        # Buttons
        self.start_button = QPushButton("Start Exercise")
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.start_button.clicked.connect(self.on_start_clicked)
        layout.addWidget(self.start_button)

        self.reset_button = QPushButton("Reset Count")
        self.reset_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.reset_button.clicked.connect(self.on_reset_clicked)
        layout.addWidget(self.reset_button)

        self.quit_button = QPushButton("Quit")
        self.quit_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: green; margin-top: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        return panel

    def start_video(self):
        """Start the video feed update timer"""
        self.timer.start(30)  # ~33 FPS

    def stop_video(self):
        """Stop the video feed update timer"""
        self.timer.stop()

    def update_frame(self):
        """Update video frame from controller"""
        frame = self.controller.get_current_frame()

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.data, fw, fh, ch * fw, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Scale to the label's full width — height follows aspect ratio.
            # This eliminates black bars: the video fills edge-to-edge horizontally
            # and any remaining vertical space is just the black background.
            label_w = self.video_label.width()
            if label_w > 0:
                scaled = pixmap.scaledToWidth(
                    label_w, Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(scaled)

        self.update_stats()

    def update_stats(self):
        """Update statistics display"""
        info = self.controller.get_exercise_info()
        if not info:
            return

        self.reps_label.setText(f"Reps: {info.get('reps', 0)}")
        self.form_label.setText(f"Form: {info.get('form', 'N/A').upper()}")
        self.state_label.setText(f"State: {info.get('state', 'N/A').upper()}")

        # Show exercise-specific debug value
        if 'knee_angle' in info:
            self.debug_label.setText(f"Knee angle: {info['knee_angle']}°")
        elif 'elbow_angle' in info:
            self.debug_label.setText(f"Elbow angle: {info['elbow_angle']}°")
        elif 'arm_angle' in info:
            self.debug_label.setText(
                f"Arm: {info['arm_angle']}°  Spread: {info.get('leg_spread', 0):.2f}x"
            )

        form_colors = {
            'excellent': 'green', 'good': 'yellow',
            'fair': 'orange', 'poor': 'red'
        }
        color = form_colors.get(info.get('form', 'good'), 'white')
        self.form_label.setStyleSheet(f"font-size: 16px; color: {color};")

    def on_exercise_changed(self, name: str):
        """Handle exercise selection change"""
        # Stop current exercise before switching
        if self.controller.is_exercise_active():
            self.controller.stop_exercise()
            self.start_button.setText("Start Exercise")
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("font-size: 12px; color: green;")
        self.controller.set_exercise(name)
        self.debug_label.setText("")

    def on_start_clicked(self):
        """Handle start/stop button click"""
        if self.controller.is_exercise_active():
            self.controller.stop_exercise()
            self.start_button.setText("Start Exercise")
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("font-size: 12px; color: red;")
        else:
            self.controller.start_exercise()
            self.start_button.setText("Stop Exercise")
            self.status_label.setText("Active")
            self.status_label.setStyleSheet("font-size: 12px; color: green;")

    def on_reset_clicked(self):
        """Handle reset button click"""
        self.controller.reset_exercise()
        self.status_label.setText("Reset")

    def on_diagnostics_toggled(self, state: int):
        """Sync checkbox state to controller"""
        self.controller.show_diagnostics = bool(state)

    def keyPressEvent(self, event):
        """'D' key toggles diagnostic overlay"""
        if event.key() == Qt.Key.Key_D:
            new_state = self.controller.toggle_diagnostics()
            self.diag_checkbox.blockSignals(True)
            self.diag_checkbox.setChecked(new_state)
            self.diag_checkbox.blockSignals(False)
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_video()
        self.controller.cleanup()
        event.accept()
