"""
Generic Exercise — data-driven exercise runner.

Consumes an ExerciseDefinition and implements the full exercise lifecycle
(signal computation, state machine, rep counting, form evaluation) without
any hardcoded exercise-specific logic.
"""
import time
from typing import Dict, List, Optional, Tuple

from .base_exercise import BaseExercise, ExerciseState, FormFeedback
from .exercise_definition import ExerciseDefinition, AngleSignal, RatioSignal
from ..pose.analyzer import PoseAnalyzer

# A rep completing faster than this is treated as noise/an erratic movement
# rather than a genuine controlled repetition, and isn't counted (state
# still transitions normally, so nothing gets stuck). Measured against the
# timestamp passed into analyze_pose(), NOT wall-clock time, so this stays
# correct whether frames arrive live (real-time) or from offline batch
# processing of a recorded video (which can run much faster than real-time)
# -- callers processing recorded video must pass the video's own frame
# timestamp, not time.perf_counter(), or every rep will look impossibly fast.
#
# Was 400ms; lowered to 200ms 2026-07-21 after a controlled 24-video
# angle/distance evaluation showed 400ms was exercise-tempo-naive --
# jumping jacks' genuine rep cadence (233-393ms, this project's fastest
# exercise) was landing just under it and getting silently rejected
# (0/6 clips exact-matched ground truth at 400ms). Swept 100/200/400ms:
# 200ms fixed jumping jacks as well as 100ms did (5/6 exact matches) with
# NO over-counting risk on squat/push-up/bicep-curl (100ms introduced one
# false-positive rep; 200ms introduced none), and even slightly improved
# those three exercises' own exact-match rate (9/18 -> 11/18 vs. 400ms).
# See reports/timing_floor_sweep_results.csv for the full sweep.
MIN_REP_DURATION_MS = 200.0


# Map ExerciseDefinition state strings to ExerciseState enum values
_STATE_MAP = {
    "IDLE": ExerciseState.IDLE,
    "UP":   ExerciseState.UP,
    "DOWN": ExerciseState.DOWN,
}
# Reverse map for state-string lookup
_STATE_STR = {v: k for k, v in _STATE_MAP.items()}


class GenericExercise(BaseExercise):
    """
    An exercise driven entirely by an ExerciseDefinition.

    Replaces hand-written exercise classes (SquatExercise, PushUpExercise,
    JumpingJackExercise). Any exercise that can be described by the
    ExerciseDefinition format is handled here without code changes.
    """

    def __init__(self, definition: ExerciseDefinition):
        super().__init__(definition.name)
        self.definition = definition
        self.state = _STATE_MAP.get(definition.initial_state, ExerciseState.IDLE)

        # Latest computed signal values (updated every frame)
        self._signal_values: Dict[str, float] = {}

        # Peak/trough tracking for form evaluation (reset at rep completion)
        # Keys are signal names; values are the tracked extreme so far in this rep phase.
        self._form_tracked: Dict[str, float] = self._init_form_tracked()

        # True while definition.safety_condition is set AND currently violated
        # (e.g. a shoulder-press hand dropping too low on the way back up).
        # Independent of the state machine -- checked every active frame.
        self._safety_violated: bool = False

        # Timestamp (caller's timescale, ms) the current state was entered.
        # None until the first analyze_pose() call, which seeds it rather
        # than assuming t=0 (see analyze_pose()).
        self._state_entered_at: Optional[float] = None
        # Reps that completed faster than MIN_REP_DURATION_MS and were
        # therefore not counted. Exposed for debugging/tuning, not used
        # internally.
        self.rejected_fast_reps: int = 0
        # Actual phase durations (ms) of each rejected rep, in rejection
        # order -- lets a caller see HOW fast the rejected reps actually
        # were (e.g. distinguishing "60ms, clearly noise" from "350ms,
        # genuinely fast but real" when tuning MIN_REP_DURATION_MS for a
        # specific exercise).
        self.rejected_fast_rep_durations: List[float] = []

    # ------------------------------------------------------------------
    # BaseExercise interface
    # ------------------------------------------------------------------

    def get_critical_signal_names(self) -> List[str]:
        """
        Signal names that actually drive this exercise's rep-counting
        transitions (as opposed to signals only used for form feedback).
        Used by PositionGate to decide which body parts genuinely need to
        be visible to keep counting -- e.g. bicep curl's secondary
        arm_angle signal references HIP, but HIP isn't required to count a
        curl since only elbow_angle drives its transitions.
        """
        names = set()
        for t in self.definition.transitions:
            for c in t.conditions:
                names.add(c.signal)
        return sorted(names)

    def get_critical_landmarks(self) -> List[str]:
        """
        Flat landmark names (bilateral-expanded) referenced by rep-counting
        signals only -- e.g. squat's knee_angle -> HIP/KNEE/ANKLE both
        sides, NOT the SHOULDER/WRIST/etc. from its secondary form-only
        signals. Used to scope PositionGate's framing check (too close/too
        far) to the body parts that actually matter for this exercise, so
        a tightly-framed bicep curl (legs never in shot) isn't judged
        against a full-body bounding box.
        """
        critical = set(self.get_critical_signal_names())
        landmarks = set()
        for sig in self.definition.angle_signals:
            if sig.name not in critical:
                continue
            if sig.bilateral:
                for base in sig.landmarks:
                    landmarks.add(f"LEFT_{base}")
                    landmarks.add(f"RIGHT_{base}")
            else:
                landmarks.update(sig.landmarks)
        for sig in self.definition.ratio_signals:
            if sig.name not in critical:
                continue
            landmarks.update(sig.numerator)
            landmarks.update(sig.denominator)
        return sorted(landmarks)

    def is_critical_signal_visible(
        self, landmarks: Dict[str, Tuple[int, int]]
    ) -> Tuple[bool, List[str]]:
        """
        Whether every signal driving rep-counting is currently computable
        from landmarks. Reuses the same bilateral-fallback logic the state
        machine itself uses (_compute_angle_signal/_compute_ratio_signal),
        so this always agrees with what analyze_pose() can actually see --
        e.g. only one leg needs to be visible for a bilateral signal, same
        as normal signal computation.

        Returns (all_visible, missing_signal_names).
        """
        critical = set(self.get_critical_signal_names())
        missing: List[str] = []
        for sig in self.definition.angle_signals:
            if sig.name not in critical:
                continue
            val, _ = self._compute_angle_signal(sig, landmarks)
            if val is None:
                missing.append(sig.name)
        for sig in self.definition.ratio_signals:
            if sig.name not in critical:
                continue
            if self._compute_ratio_signal(sig, landmarks) is None:
                missing.append(sig.name)
        return (len(missing) == 0, missing)

    def get_required_landmarks(self) -> List[str]:
        """
        Derive the required landmark list from the definition.
        Used by the controller to query which joints this exercise needs.
        """
        landmarks = set()
        for sig in self.definition.angle_signals:
            if sig.bilateral:
                for base in sig.landmarks:
                    landmarks.add(f"LEFT_{base}")
                    landmarks.add(f"RIGHT_{base}")
            else:
                landmarks.update(sig.landmarks)
        for sig in self.definition.ratio_signals:
            landmarks.update(sig.numerator)
            landmarks.update(sig.denominator)
        return sorted(landmarks)

    def analyze_pose(
        self, landmarks: Dict[str, Tuple[int, int]], timestamp_ms: Optional[float] = None
    ) -> None:
        """
        Process one frame: compute signals, advance state machine, track form.

        timestamp_ms should be on the caller's own consistent timescale --
        wall-clock (time.perf_counter() * 1000) for a live camera, but the
        video's own frame/playback position for recorded video, since that
        may be processed much faster than real-time. Defaults to wall-clock
        if omitted, which is only correct for live callers.
        """
        if not self.is_active:
            return

        if timestamp_ms is None:
            timestamp_ms = time.perf_counter() * 1000
        if self._state_entered_at is None:
            self._state_entered_at = timestamp_ms

        # Compute all signals for this frame
        signal_values, side_used = self._compute_signals(landmarks)

        # If fallback_mode="require_all", skip frame when ANY signal is missing
        if self.definition.fallback_mode == "require_all":
            if len(signal_values) < (
                len(self.definition.angle_signals) + len(self.definition.ratio_signals)
            ):
                self._last_side_used = "none"
                return

        self._last_side_used = side_used
        self._signal_values = signal_values

        # Safety/over-limit check -- independent of the state machine, so it
        # catches an overshoot regardless of which phase of the rep it
        # happens in (e.g. going too low on the way back up).
        self._safety_violated = (
            self.definition.safety_condition is not None
            and self.definition.safety_condition.evaluate(signal_values)
        )

        # Update form tracking while in the tracked phase
        current_state_str = _STATE_STR.get(self.state, "IDLE")
        if current_state_str == self.definition.form.during_state:
            self._update_form_tracking()

        # Evaluate transitions in definition order; fire the first match
        for transition in self.definition.transitions:
            if transition.matches(current_state_str, signal_values):
                new_state_str = transition.to_state
                new_state = _STATE_MAP.get(new_state_str, ExerciseState.IDLE)

                # Check for rep completion BEFORE changing state
                rep_from, rep_to = self.definition.rep_complete_transition
                if current_state_str == rep_from and new_state_str == rep_to:
                    phase_duration = timestamp_ms - self._state_entered_at
                    if phase_duration >= MIN_REP_DURATION_MS:
                        self.increment_rep()
                        self.form_feedback = self._evaluate_form()
                    else:
                        self.rejected_fast_reps += 1
                        self.rejected_fast_rep_durations.append(phase_duration)
                    self._form_tracked = self._init_form_tracked()

                self.state = new_state
                self._state_entered_at = timestamp_ms

                # Reset form tracking when entering the form-tracking phase
                if new_state_str == self.definition.form.during_state:
                    self._form_tracked = self._init_form_tracked()
                    # Seed with first value so tracking starts immediately
                    self._update_form_tracking()

                break  # only fire one transition per frame

    def check_form(self, landmarks: Dict[str, Tuple[int, int]]) -> FormFeedback:
        return self._evaluate_form()

    def reset(self):
        super().reset()
        self.state = _STATE_MAP.get(self.definition.initial_state, ExerciseState.IDLE)
        self._signal_values = {}
        self._form_tracked = self._init_form_tracked()
        self._safety_violated = False
        self._state_entered_at = None
        self.rejected_fast_reps = 0
        self.rejected_fast_rep_durations = []

    def get_info(self) -> Dict:
        """Extended info including all current signal values."""
        info = super().get_info()
        info.update(self._signal_values)
        info['safety_violated'] = self._safety_violated
        return info

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signals(
        self, landmarks: Dict[str, Tuple[int, int]]
    ) -> Tuple[Dict[str, float], str]:
        """
        Compute all angle and ratio signals for the current frame.

        Returns:
            (signal_values, side_used)
            signal_values: dict of signal_name → float (missing = couldn't compute)
            side_used: "both" | "left" | "right" | "none"
        """
        signal_values: Dict[str, float] = {}
        side_used = "none"

        for sig in self.definition.angle_signals:
            val, su = self._compute_angle_signal(sig, landmarks)
            if val is not None:
                signal_values[sig.name] = val
                side_used = _merge_side(side_used, su)

        for sig in self.definition.ratio_signals:
            val = self._compute_ratio_signal(sig, landmarks)
            if val is not None:
                signal_values[sig.name] = val

        return signal_values, side_used

    def _compute_angle_signal(
        self, sig: AngleSignal, landmarks: Dict[str, Tuple[int, int]]
    ) -> Tuple[Optional[float], str]:
        """
        Compute a single angle signal, applying bilateral fallback if needed.

        Returns (angle_value_or_None, side_used).
        """
        if not sig.bilateral:
            # Non-bilateral: all landmarks must be present
            if all(lm in landmarks for lm in sig.landmarks):
                pts = [landmarks[lm] for lm in sig.landmarks]
                return PoseAnalyzer.calculate_angle(*pts), "both"
            return None, "none"

        # Bilateral: try left and right independently
        p1, p2, p3 = sig.landmarks          # base names e.g. HIP, KNEE, ANKLE
        left_lms  = [f"LEFT_{p1}",  f"LEFT_{p2}",  f"LEFT_{p3}"]
        right_lms = [f"RIGHT_{p1}", f"RIGHT_{p2}", f"RIGHT_{p3}"]

        left_ok  = all(lm in landmarks for lm in left_lms)
        right_ok = all(lm in landmarks for lm in right_lms)

        if not left_ok and not right_ok:
            return None, "none"

        left_val  = PoseAnalyzer.calculate_angle(*[landmarks[lm] for lm in left_lms])  if left_ok  else None
        right_val = PoseAnalyzer.calculate_angle(*[landmarks[lm] for lm in right_lms]) if right_ok else None

        if left_ok and right_ok:
            return (left_val + right_val) / 2, "both"
        elif left_ok:
            return left_val, "left"
        else:
            return right_val, "right"

    def _compute_ratio_signal(
        self, sig: RatioSignal, landmarks: Dict[str, Tuple[int, int]]
    ) -> Optional[float]:
        """Compute a distance ratio signal. Returns None if any landmark is missing."""
        all_lms = sig.numerator + sig.denominator
        if not all(lm in landmarks for lm in all_lms):
            return None
        num_dist = PoseAnalyzer.calculate_distance(
            landmarks[sig.numerator[0]], landmarks[sig.numerator[1]]
        )
        den_dist = PoseAnalyzer.calculate_distance(
            landmarks[sig.denominator[0]], landmarks[sig.denominator[1]]
        )
        if den_dist <= 0:
            return None
        return num_dist / den_dist

    # ------------------------------------------------------------------
    # Form tracking and evaluation
    # ------------------------------------------------------------------

    def _init_form_tracked(self) -> Dict[str, float]:
        """Initialize form tracking values to neutral extremes."""
        init_val = float("inf") if self.definition.form.tracking == "min" else float("-inf")
        return {sig: init_val for sig in self.definition.form.tracked_signals}

    def _update_form_tracking(self) -> None:
        """Update tracked extremes with the current signal values."""
        for sig_name in self.definition.form.tracked_signals:
            val = self._signal_values.get(sig_name)
            if val is None:
                continue
            if self.definition.form.tracking == "min":
                self._form_tracked[sig_name] = min(self._form_tracked[sig_name], val)
            else:
                self._form_tracked[sig_name] = max(self._form_tracked[sig_name], val)

    def _evaluate_form(self) -> FormFeedback:
        """
        Evaluate form quality from tracked extremes.

        Evaluates form levels in order (excellent → good → fair).
        Returns the first level whose conditions all/any pass; else POOR.
        The tracked values dict is passed directly into each FormLevel so
        SignalCondition.evaluate() can compare the tracked extreme against thresholds.
        """
        # Replace inf/-inf with 0 so evaluations don't fire spuriously
        safe_values = {
            k: (v if abs(v) != float("inf") else 0.0)
            for k, v in self._form_tracked.items()
        }

        for level in self.definition.form.levels:
            if level.evaluate(safe_values):
                return FormFeedback[level.rating.upper()]

        return FormFeedback.POOR


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _merge_side(current: str, new: str) -> str:
    """
    Merge two side-used labels into the most informative one.
    Priority: both > left > right > none.
    """
    if current == "both" or new == "both":
        return "both"
    if current != "none" and new != "none" and current != new:
        return "both"
    if current != "none":
        return current
    return new
