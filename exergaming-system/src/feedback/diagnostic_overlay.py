"""
Diagnostic Overlay — draws a live debug panel on the video frame.

Rendered in the bottom-right corner, on top of the standard feedback overlay.
Toggle with the 'D' key or the "Diagnostics" checkbox in the UI.
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


# BGR colours used in the panel
_GREEN  = (100, 255, 100)
_YELLOW = (80,  220, 255)
_RED    = (80,  80,  255)
_WHITE  = (220, 220, 220)
_GRAY   = (130, 130, 130)
_CYAN   = (255, 210, 80)
_ORANGE = (60,  165, 255)

_ORANGE = (60,  140, 255)

_FORM_COLORS = {
    'excellent': _GREEN,
    'good':      _YELLOW,
    'fair':      _ORANGE,
    'poor':      _RED,
}


class DiagnosticOverlay:
    """Renders a semi-transparent diagnostic panel onto a BGR frame."""

    PANEL_W    = 272
    LINE_H     = 19
    PAD        = 9
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.46
    THICKNESS  = 1

    def draw(
        self,
        frame: np.ndarray,
        runtime: Dict[str, Any],
        exercise_info: Dict[str, Any],
    ) -> None:
        """
        Overlay the diagnostic panel on *frame* in-place.

        Args:
            frame:         BGR image (modified in-place)
            runtime:       dict with fps, processing_time_ms, frame_num,
                           pose_detected, logger_active, logger_frames
            exercise_info: dict from exercise.get_info()
        """
        lines = self._build_lines(runtime, exercise_info)

        panel_h = self.PAD * 2 + len(lines) * self.LINE_H + max(0, len(lines) - 1) * 2
        h, w = frame.shape[:2]

        # Widen to fit the longest row rather than trusting a fixed width --
        # threshold strings vary per exercise, and a long one used to run off
        # the right-hand edge of the frame.
        widest = max(
            cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.THICKNESS)[0][0]
            for text, _ in lines
        )
        panel_w = min(max(self.PANEL_W, widest + self.PAD * 2), w - 20)

        x1 = w - panel_w - 10
        y1 = h - panel_h - 10
        x2 = w - 10
        y2 = h - 10

        # Semi-transparent dark background
        bg = frame.copy()
        cv2.rectangle(bg, (x1, y1), (x2, y2), (12, 12, 12), -1)
        cv2.addWeighted(bg, 0.78, frame, 0.22, 0, frame)

        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

        # Text rows
        ty = y1 + self.PAD + self.LINE_H - 5
        for text, color in lines:
            cv2.putText(
                frame, text,
                (x1 + self.PAD, ty),
                self.FONT, self.FONT_SCALE, color, self.THICKNESS,
                cv2.LINE_AA,
            )
            ty += self.LINE_H + 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lines(
        self,
        runtime: Dict[str, Any],
        info: Dict[str, Any],
    ) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Return (text, colour) pairs for every row of the panel."""
        lines: List[Tuple[str, Tuple[int, int, int]]] = []

        # ── Header ───────────────────────────────────────────────────
        lines.append(("  DIAGNOSTICS", _GRAY))

        # ── Runtime ──────────────────────────────────────────────────
        fps  = runtime.get("fps", 0.0)
        proc = runtime.get("processing_time_ms", 0.0)
        fps_color = _GREEN if fps >= 24 else (_YELLOW if fps >= 15 else _RED)
        lines.append((f"FPS: {fps:.1f}    Proc: {proc:.1f} ms", fps_color))
        lines.append((f"Frame: {runtime.get('frame_num', 0)}", _GRAY))

        # ── Pose detection ───────────────────────────────────────────
        detected = runtime.get("pose_detected", False)
        lines.append((
            f"Pose: {'DETECTED' if detected else 'NOT DETECTED'}",
            _GREEN if detected else _RED,
        ))

        # ── Bilateral side ───────────────────────────────────────────
        side = info.get("side_used", "none").upper()
        side_color = _GREEN if side == "BOTH" else (_YELLOW if side in ("LEFT", "RIGHT") else _RED)
        lines.append((f"Side: {side}", side_color))

        # ── Divider ──────────────────────────────────────────────────
        lines.append(("  " + "-" * 30, _GRAY))

        # ── Exercise state ───────────────────────────────────────────
        active = info.get("active", False)
        state  = info.get("state", "idle").upper()
        lines.append((f"State: {state}", _CYAN if active else _GRAY))

        # ── Angles + thresholds ──────────────────────────────────────
        if "knee_angle" in info:
            a = info["knee_angle"]
            lines.append((f"Knee:  {a:.1f} deg  [down<90  up>160]", _WHITE))
        elif "elbow_angle" in info:
            a = info["elbow_angle"]
            lines.append((f"Elbow: {a:.1f} deg  [down<110 up>160]", _WHITE))
        elif "arm_angle" in info:
            arm    = info["arm_angle"]
            # GenericExercise (every JSON-driven exercise) publishes this as
            # "leg_spread_ratio"; only the legacy hard-coded JumpingJack class
            # used the short name, so read that as a fallback. Without this the
            # panel always showed 0.00 while the side panel showed the real value.
            spread = info.get("leg_spread_ratio", info.get("leg_spread", 0.0))
            lines.append((f"Arm:   {arm:.1f} deg  [up>140]", _WHITE))
            lines.append((f"Spread:{spread:.2f}x   [out>1.8]", _WHITE))

        # ── Form ─────────────────────────────────────────────────────
        form = info.get("form", "good")
        lines.append((
            f"Form:  {form.upper()}",
            _FORM_COLORS.get(form, _WHITE),
        ))

        lines.append((f"Reps:  {info.get('reps', 0)}", _WHITE))

        # ── Frame quality ────────────────────────────────────────────
        skipped = runtime.get("skipped_frames", 0)
        warned  = runtime.get("warned_frames", 0)
        total   = max(runtime.get("logger_frames", 1), 1)
        lines.append(("  " + "-" * 30, _GRAY))
        skip_color = _RED if skipped / total > 0.15 else (_YELLOW if skipped > 0 else _GREEN)
        lines.append((f"Skipped: {skipped}  Warned: {warned}", skip_color))

        # Active quality warnings (compact — just first one if multiple)
        active_warnings = runtime.get("quality_warnings") or []
        if active_warnings:
            short = active_warnings[0][:32]  # truncate long strings
            lines.append((f"! {short}", _ORANGE))

        # ── Logger status ────────────────────────────────────────────
        if runtime.get("logger_active", False):
            n = runtime.get("logger_frames", 0)
            lines.append((f"Log: ACTIVE  ({n} frames)", _GREEN))
        else:
            lines.append(("Log: idle", _GRAY))

        return lines
