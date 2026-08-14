"""
Exercise Learner — extracts an ExerciseDefinition from a demonstration recording.

Algorithm (3 steps):
  1. Identify important signals: those whose range (peak−trough) exceeds a threshold.
  2. Find the peak and trough of each important signal across all frames, using
     percentiles rather than the single extreme frame (see EXTREME_PERCENTILE).
  3. Derive state-machine thresholds with a guard buffer so transitions don't
     fire at the very extremes of a movement.

The output is an ExerciseDefinition with source="learned" that is compatible
with GenericExercise and can be saved as a JSON file in definitions/.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..exercises.exercise_definition import (
    AngleSignal, RatioSignal, ExerciseDefinition,
    FormEvaluation, FormLevel, SignalCondition, StateTransition,
)
from .demonstration_capture import CANDIDATE_ANGLE_SIGNALS, CANDIDATE_RATIO_SIGNALS


# Signals whose range exceeds this threshold are considered "important"
MIN_IMPORTANT_RANGE_DEG = 15.0      # degrees (for angle signals)
MIN_IMPORTANT_RANGE_RATIO = 0.3     # dimensionless (for ratio signals)

# Buffer fractions for threshold placement.
#
# For a LOW-PHASE exercise (squat / push-up):
#   DOWN entry threshold  = trough + DOWN_BUFFER * range
#     → 40% up from the lowest point, so almost anyone doing the exercise clears it.
#   UP return threshold   = peak − UP_BUFFER * range
#     → 15% down from the highest point, so "nearly standing" is required.
#
# For a HIGH-PHASE exercise (jumping jack):
#   Both directions use UP_BUFFER (15%) — symmetric is fine because the arms/spread
#   must reach a clear peak and return to a clear rest.
#
# The asymmetry for low-phase exercises is intentional: a training video that shows
# very deep squats (trough ≈ 37°) would otherwise produce a DOWN threshold that is
# too strict for someone who squats to a more typical depth (≈ 85-100°).
DOWN_BUFFER_FRACTION = 0.40   # 40 % from trough  →  DOWN entry
UP_BUFFER_FRACTION   = 0.15   # 15 % from peak    →  UP return / both for high-phase

# Minimum buffer (degrees / ratio units) even when range is small
MIN_BUFFER_DEG = 5.0
MIN_BUFFER_RATIO = 0.15

# Percentile used to define the movement extremes (peak / trough).
#
# Taking the single highest / lowest frame makes the whole definition hostage to
# one frame: a stumble, a stray landmark reading, or bending down to adjust the
# camera before starting becomes the extreme, and every derived threshold moves
# with it. Measured on the demonstration videos, one injected stray frame shifted
# the learned push-up DOWN threshold by 21.5 deg and the squat one by 6.9 deg.
#
# Taking the 2nd / 98th percentile instead discards the sparsest ~2 % of frames at
# each end. A genuine movement extreme is reached on every rep and is therefore
# dense in frames, so it survives the trim; a one-off incident spans only a
# handful of frames and does not. On clean demonstrations both methods agree to
# within ~2.5 deg and produce identical rep counts, at the same compute cost.
#
# Set to 0.0 to fall back to exact min/max (the pre-refinement behaviour).
EXTREME_PERCENTILE = 2.0


class ExerciseLearner:
    """
    Learns exercise thresholds from a list of demonstration frame records
    (as produced by DemonstrationCapture).
    """

    def __init__(self, frames: List[Dict], exercise_name: str = "Learned Exercise"):
        # Only use frames where pose was actually detected
        self._frames = [f for f in frames if f.get("pose_detected") and f.get("signals")]
        self._exercise_name = exercise_name

        if len(self._frames) < 10:
            raise ValueError(
                f"Too few detected frames ({len(self._frames)}) to learn from. "
                "Ensure the demonstration video has clear body visibility."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(self, approved_signals: Optional[List[str]] = None) -> ExerciseDefinition:
        """
        Run the full learning pipeline and return an ExerciseDefinition.

        Args:
            approved_signals: If provided, use this list instead of running the
                              automatic importance detection. Signals must be a
                              subset of what the demonstration captured. Pass this
                              after the user has reviewed and corrected the
                              auto-detected list via interactive_confirm().

        Steps:
          1. Collect per-signal time-series from frame records
          2. Identify important signals (auto or user-approved)
          3. Find percentile peak/trough for each important signal
          4. Derive UP/DOWN transition thresholds
          5. Identify whether primary angle goes low (min-tracking, squat style)
             or high (max-tracking, jumping-jack style)
          6. Build and return ExerciseDefinition
        """
        series = self._build_series()

        if approved_signals is not None:
            important = [s for s in approved_signals if s in series]
            if not important:
                raise ValueError(
                    "None of the approved signals were found in the demonstration data."
                )
        else:
            important = self._identify_important(series)

        if not important:
            raise RuntimeError(
                "No important signals found. The demonstration may not show "
                "enough movement variation."
            )

        extremes = self._compute_extremes(series, important)
        primary = self._select_primary(important, series)
        learned_angle_signals, learned_ratio_signals = self._build_signal_objects(important)
        transitions, rep_complete, form = self._derive_transitions_and_form(
            primary, extremes, series
        )

        return ExerciseDefinition(
            name=self._exercise_name,
            description=(
                f"Learned from demonstration. "
                f"Primary signal: {primary}. "
                f"Important signals: {', '.join(important)}."
            ),
            source="learned",
            camera_view="either",
            fallback_mode="bilateral",
            initial_state="IDLE",
            angle_signals=learned_angle_signals,
            ratio_signals=learned_ratio_signals,
            transitions=transitions,
            rep_complete_transition=rep_complete,
            form=form,
        )

    def get_analysis_report(self) -> Dict:
        """
        Return a dict with intermediate analysis data for reporting / comparison.
        """
        series = self._build_series()
        important = self._identify_important(series)
        extremes = self._compute_extremes(series, important)
        return {
            "total_frames":    len(self._frames),
            "all_signals":     sorted(series.keys()),
            "important_signals": important,
            # Range is reported as peak−trough of the same (percentile) extremes
            # shown alongside it, so the signal review table stays self-consistent.
            "signal_ranges":   {n: round(extremes[n]["peak"] - extremes[n]["trough"], 2) for n in important},
            "signal_peaks":    {n: round(extremes[n]["peak"], 2) for n in important},
            "signal_troughs":  {n: round(extremes[n]["trough"], 2) for n in important},
        }

    # ------------------------------------------------------------------
    # Step 1 — build per-signal time-series
    # ------------------------------------------------------------------

    def _build_series(self) -> Dict[str, List[float]]:
        """Collect frame-by-frame values for every signal across all detected frames."""
        series: Dict[str, List[float]] = {}
        for frame in self._frames:
            for name, val in frame["signals"].items():
                series.setdefault(name, []).append(val)
        return series

    # ------------------------------------------------------------------
    # Step 2 — identify important signals
    # ------------------------------------------------------------------

    def _identify_important(self, series: Dict[str, List[float]]) -> List[str]:
        """
        A signal is "important" if its range (peak−trough) across all frames
        exceeds the exercise-type-specific threshold.
        Returns signal names sorted by range descending (most important first).
        """
        important = []
        for name, vals in series.items():
            if not vals:
                continue
            peak, trough = self._extremes_of(vals)
            rng = peak - trough
            # Distinguish angle signals (degrees) from ratio signals
            threshold = (
                MIN_IMPORTANT_RANGE_RATIO
                if name in {r[0] for r in CANDIDATE_RATIO_SIGNALS}
                else MIN_IMPORTANT_RANGE_DEG
            )
            if rng >= threshold:
                important.append((name, rng))

        # Sort by range descending; return names only
        return [name for name, _ in sorted(important, key=lambda x: x[1], reverse=True)]

    # ------------------------------------------------------------------
    # Step 3 — compute percentile peak and trough
    # ------------------------------------------------------------------

    @staticmethod
    def _extremes_of(vals: List[float]) -> Tuple[float, float]:
        """
        Return (peak, trough) for one signal's frame values.

        Percentile-based so that a single stray frame cannot define the movement
        range — see EXTREME_PERCENTILE for the rationale.
        """
        if EXTREME_PERCENTILE <= 0:
            return max(vals), min(vals)
        return (
            float(np.percentile(vals, 100.0 - EXTREME_PERCENTILE)),
            float(np.percentile(vals, EXTREME_PERCENTILE)),
        )

    def _compute_extremes(
        self, series: Dict[str, List[float]], signals: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Return {signal_name: {peak, trough}} for each important signal."""
        extremes = {}
        for name in signals:
            vals = series.get(name, [])
            if vals:
                peak, trough = self._extremes_of(vals)
                extremes[name] = {"peak": peak, "trough": trough}
        return extremes

    # ------------------------------------------------------------------
    # Step 4 — select primary signal
    # ------------------------------------------------------------------

    def _select_primary(
        self, important: List[str], series: Dict[str, List[float]]
    ) -> str:
        """
        The primary signal is the one with the highest range among angle signals.
        If only ratio signals are important, the first one is chosen.
        """
        angle_signals = {r[0] for r in CANDIDATE_RATIO_SIGNALS}
        angle_candidates = [n for n in important if n not in angle_signals]
        return angle_candidates[0] if angle_candidates else important[0]

    # ------------------------------------------------------------------
    # Step 5 — derive transitions and form evaluation
    # ------------------------------------------------------------------

    def _derive_transitions_and_form(
        self,
        primary: str,
        extremes: Dict[str, Dict[str, float]],
        series: Dict[str, List[float]],
    ) -> Tuple[List[StateTransition], List[str], FormEvaluation]:
        """
        Determine whether the primary signal decreases (squat/pushup style) or
        increases (jumping jack style) during the exercise, then derive thresholds.

        Returns (transitions, rep_complete_transition, form_evaluation).
        """
        ext = extremes[primary]
        peak   = ext["peak"]
        trough = ext["trough"]
        rng    = peak - trough

        # Determine movement direction FIRST by looking at the median of the series.
        # If median is above the midpoint → exercise spends more time at the HIGH end
        # → it's a "low-phase" exercise (squat, pushup: person is upright most of the time).
        median_val = statistics.median(series[primary])
        low_phase  = median_val > (trough + rng * 0.5)

        if low_phase:
            # Asymmetric buffers for squat/push-up style:
            # DOWN entry is set well into the range (40 % from trough) so
            # any reasonable execution of the exercise clears the threshold.
            # UP return is set near the peak (15 % below) so "nearly standing"
            # is required before the rep completes.
            raw_down = rng * DOWN_BUFFER_FRACTION
            raw_up   = rng * UP_BUFFER_FRACTION
        else:
            # High-phase (jumping jack): symmetric 15 % buffers are fine.
            raw_down = rng * UP_BUFFER_FRACTION
            raw_up   = rng * UP_BUFFER_FRACTION

        threshold_down = trough + max(raw_down, MIN_BUFFER_DEG)
        threshold_up   = peak   - max(raw_up,   MIN_BUFFER_DEG)

        if low_phase:
            # Squat / push-up pattern:
            # UP (resting) → DOWN (exertion) → back to UP (rep counted on return)
            transitions = [
                StateTransition(
                    from_states=["IDLE", "UP"],
                    to_state="DOWN",
                    conditions=[SignalCondition(primary, "below", round(threshold_down, 1))],
                ),
                StateTransition(
                    from_states=["DOWN"],
                    to_state="UP",
                    conditions=[SignalCondition(primary, "above", round(threshold_up, 1))],
                ),
            ]
            rep_complete = ["DOWN", "UP"]
            tracking = "min"
            tracking_state = "DOWN"
            form_levels = self._form_levels_low(primary, trough, rng)

        else:
            # Jumping jack pattern:
            # DOWN (resting) → UP (exertion) → back to DOWN (rep counted on return)
            transitions = [
                StateTransition(
                    from_states=["IDLE", "DOWN"],
                    to_state="UP",
                    conditions=[SignalCondition(primary, "above", round(threshold_up, 1))],
                ),
                StateTransition(
                    from_states=["UP"],
                    to_state="DOWN",
                    conditions=[SignalCondition(primary, "below", round(threshold_down, 1))],
                ),
            ]
            rep_complete = ["UP", "DOWN"]
            tracking = "max"
            tracking_state = "UP"
            form_levels = self._form_levels_high(primary, peak, rng)

        form = FormEvaluation(
            tracking=tracking,
            during_state=tracking_state,
            tracked_signals=[primary],
            levels=form_levels,
        )
        return transitions, rep_complete, form

    # ------------------------------------------------------------------
    # Step 6 — build signal objects for the definition
    # ------------------------------------------------------------------

    def _build_signal_objects(
        self, important: List[str]
    ) -> Tuple[List[AngleSignal], List[RatioSignal]]:
        """Create AngleSignal and RatioSignal objects for all important signals."""
        ratio_names = {r[0] for r in CANDIDATE_RATIO_SIGNALS}
        angle_map   = {r[0]: r[1] for r in CANDIDATE_ANGLE_SIGNALS}
        ratio_map   = {r[0]: (r[1], r[2]) for r in CANDIDATE_RATIO_SIGNALS}

        angle_signals: List[AngleSignal] = []
        ratio_signals: List[RatioSignal] = []

        for i, name in enumerate(important):
            importance = round(1.0 / (i + 1), 3)   # rank-based importance
            if name in ratio_names:
                num, den = ratio_map[name]
                ratio_signals.append(RatioSignal(name, num, den, importance))
            elif name in angle_map:
                lms = angle_map[name]
                angle_signals.append(AngleSignal(name, lms, bilateral=True, importance=importance))

        return angle_signals, ratio_signals

    # ------------------------------------------------------------------
    # Form level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _form_levels_low(signal: str, trough: float, rng: float) -> List[FormLevel]:
        """
        Form levels for a low-phase exercise (squat/pushup).
        Based on how deeply the signal drops below the trough + 25% of range.
        """
        # Excellent: reached within 10% of absolute trough
        excellent_th = round(trough + rng * 0.10, 1)
        # Good: reached within 30% of range above trough
        good_th      = round(trough + rng * 0.30, 1)
        # Fair: reached within 50% of range above trough
        fair_th      = round(trough + rng * 0.50, 1)

        return [
            FormLevel("excellent", [SignalCondition(signal, "below", excellent_th)]),
            FormLevel("good",      [SignalCondition(signal, "below", good_th)]),
            FormLevel("fair",      [SignalCondition(signal, "below", fair_th)]),
        ]

    @staticmethod
    def _form_levels_high(signal: str, peak: float, rng: float) -> List[FormLevel]:
        """
        Form levels for a high-phase exercise (jumping jack).
        Based on how high the signal rises above peak − 25% of range.
        """
        excellent_th = round(peak - rng * 0.10, 1)
        good_th      = round(peak - rng * 0.30, 1)
        fair_th      = round(peak - rng * 0.50, 1)

        return [
            FormLevel("excellent", [SignalCondition(signal, "above", excellent_th)]),
            FormLevel("good",      [SignalCondition(signal, "above", good_th)]),
            FormLevel("fair",      [SignalCondition(signal, "above", fair_th)]),
        ]
