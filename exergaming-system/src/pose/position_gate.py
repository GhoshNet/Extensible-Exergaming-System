"""
Position Gate — decides, frame by frame, whether an exercise's rep-counting
should actually run.

Wraps four signals into one gate decision:
  1. Truly generic checks (FrameQualityChecker: lighting, no person at all)
     -- exercise-independent.
  2. Exercise-specific critical-signal visibility -- are the body parts THIS
     exercise's transitions actually depend on visible right now? (e.g.
     knees for squat, elbows for bicep curl -- not a fixed upper/lower
     split; see GenericExercise.is_critical_signal_visible().)
  3. Framing (too close/too far), scoped to ONLY the critical landmarks --
     not a bounding box over everything visible, so an exercise that never
     shows the legs (e.g. a tightly-framed bicep curl) isn't judged
     against a full-body box.
  4. Position drift -- has the user moved away from where they started the
     exercise? A "home anchor" (hip/shoulder-center, scaled by torso size)
     is captured once when the exercise activates.

Any failure is debounced: a bad frame only actually gates (tells the
caller to skip analyze_pose()) once it's persisted for grace_period_ms.
A single dropped-detection frame is common and harmless -- GenericExercise
already handles a missing signal gracefully on its own -- so the gate's
only job is to stop SUSTAINED bad input (noisy-but-present landmarks from
someone who has genuinely wandered out of position) from feeding the
counter, without visibly interrupting a workout over a one-frame blip.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .quality_checker import FrameQualityChecker
from ..exercises.generic_exercise import GenericExercise

_FRIENDLY_NAMES = {
    "HIP": "hips", "KNEE": "knees", "ANKLE": "ankles", "FOOT_INDEX": "feet",
    "SHOULDER": "shoulders", "ELBOW": "elbows", "WRIST": "wrists",
}

# How far (in multiples of torso length) the anchor point may drift before
# it counts as "left position".
_DRIFT_TOLERANCE_SCALE = 1.6

# How long a bad condition must persist before it actually gates counting.
_DEFAULT_GRACE_PERIOD_MS = 400.0

# Fraction of frame width/height that counts as too close/too far, applied
# to the bounding box of just the critical landmarks (see
# PositionGate._check_framing). NOT the same value the old full-body check
# used (0.28) -- a critical-signal span (e.g. squat's hip-to-ankle, or
# bicep curl's shoulder-to-wrist) is naturally smaller than a full-body
# bounding box, so reusing 0.28 here false-positived on completely normal
# framing during testing. Loosened accordingly; still a rough heuristic,
# worth tuning further against real footage.
_TOO_CLOSE_RATIO = 0.85
_TOO_FAR_RATIO = 0.10


def _strip_side(name: str) -> str:
    return name.replace("LEFT_", "").replace("RIGHT_", "")


def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class PositionGateResult:
    ok: bool                  # True if this exact frame passed all checks
    gated: bool                # True if the caller should skip analyze_pose()
    warnings: List[str] = field(default_factory=list)


class PositionGate:
    """Stateful -- one instance per active exercise session."""

    def __init__(
        self,
        exercise: GenericExercise,
        grace_period_ms: float = _DEFAULT_GRACE_PERIOD_MS,
        drift_tolerance_scale: float = _DRIFT_TOLERANCE_SCALE,
    ):
        self.exercise = exercise
        self.quality_checker = FrameQualityChecker()
        self.grace_period_ms = grace_period_ms
        self.drift_tolerance_scale = drift_tolerance_scale

        self._anchor: Optional[Tuple[float, float]] = None
        self._anchor_scale: Optional[float] = None
        self._bad_since: Optional[float] = None

    def calibrate(self, landmarks: Dict[str, Tuple[int, int]]) -> None:
        """Capture the 'home' position. Call once when the exercise activates."""
        center, scale = self._compute_anchor(landmarks)
        self._anchor = center
        self._anchor_scale = scale

    def reset(self) -> None:
        self._anchor = None
        self._anchor_scale = None
        self._bad_since = None

    def check(
        self,
        frame,
        landmarks: Dict[str, Tuple[int, int]],
        pose_detected: bool,
        timestamp_ms: float,
    ) -> PositionGateResult:
        warnings: List[str] = list(self.quality_checker.check(frame, landmarks, pose_detected))

        if pose_detected and landmarks:
            visible, missing = self.exercise.is_critical_signal_visible(landmarks)
            if not visible:
                warnings.append(self._missing_signal_message(missing))
            else:
                # Only check framing (too close/far) once we know the
                # critical parts ARE visible -- scoped to just those
                # landmarks, so an exercise that never shows the legs
                # (e.g. a tightly-framed bicep curl) isn't judged against
                # a full-body bounding box.
                framing_warning = self._check_framing(frame.shape, landmarks)
                if framing_warning:
                    warnings.append(framing_warning)

            if self._anchor is not None:
                center, scale = self._compute_anchor(landmarks)
                ref_scale = scale or self._anchor_scale
                if ref_scale:
                    drift = _distance(center, self._anchor)
                    if drift > ref_scale * self.drift_tolerance_scale:
                        warnings.append("Return to your starting position")

        ok = len(warnings) == 0
        if ok:
            self._bad_since = None
            gated = False
        else:
            if self._bad_since is None:
                self._bad_since = timestamp_ms
            gated = (timestamp_ms - self._bad_since) >= self.grace_period_ms

        return PositionGateResult(ok=ok, gated=gated, warnings=warnings)

    # ------------------------------------------------------------------

    def _check_framing(
        self, frame_shape: Tuple[int, int, int], landmarks: Dict[str, Tuple[int, int]]
    ) -> Optional[str]:
        """Too close/too far, using only the critical landmarks' bounding box."""
        critical_names = set(self.exercise.get_critical_landmarks())
        pts = [pt for name, pt in landmarks.items() if name in critical_names]
        if len(pts) < 2:
            return None  # not enough points for a meaningful box
        h, w = frame_shape[:2]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        if bbox_w > w * _TOO_CLOSE_RATIO:
            return "Too close - step back"
        if bbox_h < h * _TOO_FAR_RATIO:
            return "Too far - step closer"
        return None

    def _missing_signal_message(self, missing_signal_names: List[str]) -> str:
        parts: List[str] = []
        seen = set()
        for sig in self.exercise.definition.angle_signals:
            if sig.name not in missing_signal_names:
                continue
            for lm in sig.landmarks:
                friendly = _FRIENDLY_NAMES.get(_strip_side(lm))
                if friendly and friendly not in seen:
                    seen.add(friendly)
                    parts.append(friendly)
        for sig in self.exercise.definition.ratio_signals:
            if sig.name not in missing_signal_names:
                continue
            for lm in sig.numerator + sig.denominator:
                friendly = _FRIENDLY_NAMES.get(_strip_side(lm))
                if friendly and friendly not in seen:
                    seen.add(friendly)
                    parts.append(friendly)
        if not parts:
            return "Position yourself so I can see you clearly"
        return f"Show your {', '.join(parts)}"

    def _compute_anchor(
        self, landmarks: Dict[str, Tuple[int, int]]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """
        Reference point + scale for drift detection. Prefers hip-center
        (best full-body reference), falls back to shoulder-center for
        exercises framed tight enough that hips aren't visible (e.g. a
        close-up bicep curl), falls back further to the centroid of
        whatever landmarks are visible. Scale prefers shoulder-hip
        distance; without both, drift checking is skipped for this frame
        (no reliable way to judge "how far" without a reference length).
        """
        if "LEFT_HIP" in landmarks and "RIGHT_HIP" in landmarks:
            center = _midpoint(landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"])
        elif "LEFT_HIP" in landmarks:
            center = landmarks["LEFT_HIP"]
        elif "RIGHT_HIP" in landmarks:
            center = landmarks["RIGHT_HIP"]
        elif "LEFT_SHOULDER" in landmarks and "RIGHT_SHOULDER" in landmarks:
            center = _midpoint(landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"])
        elif landmarks:
            xs = [p[0] for p in landmarks.values()]
            ys = [p[1] for p in landmarks.values()]
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            return None, None

        scale = None
        shoulder = None
        hip = None
        if "LEFT_SHOULDER" in landmarks and "RIGHT_SHOULDER" in landmarks:
            shoulder = _midpoint(landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"])
        elif "LEFT_SHOULDER" in landmarks:
            shoulder = landmarks["LEFT_SHOULDER"]
        elif "RIGHT_SHOULDER" in landmarks:
            shoulder = landmarks["RIGHT_SHOULDER"]
        if "LEFT_HIP" in landmarks and "RIGHT_HIP" in landmarks:
            hip = _midpoint(landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"])
        elif "LEFT_HIP" in landmarks:
            hip = landmarks["LEFT_HIP"]
        elif "RIGHT_HIP" in landmarks:
            hip = landmarks["RIGHT_HIP"]
        if shoulder and hip:
            scale = _distance(shoulder, hip)

        return center, scale
