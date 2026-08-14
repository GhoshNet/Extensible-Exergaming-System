"""
Form Errors — data structures and analyzer for body-part-specific form feedback.

This is Dissertation Contribution 1: granular, real-time visual feedback.
Unlike existing literature (binary correct/wrong), this system identifies WHICH
body part has an issue and how severe it is, enabling simultaneous multi-error
display on the skeleton overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from ..exercises.exercise_definition import ExerciseDefinition


# Body parts that can carry a FormError.
# These map onto skeleton segments drawn by the skeleton overlay.
BodyPart = Literal[
    "left_thigh", "right_thigh",
    "left_shin",  "right_shin",
    "left_upper_arm", "right_upper_arm",
    "left_forearm",   "right_forearm",
    "torso",
    "left_hip", "right_hip",
]

Severity = Literal["minor", "major"]


@dataclass
class FormError:
    """A single form issue detected during exercise."""
    body_part: str          # see BodyPart literal above
    severity: Severity      # "minor" or "major" — controls colour intensity
    message: str            # short user-facing text, e.g. "Squat deeper"
    signal_name: str        # which tracked angle/signal triggered this
    deviation: float        # how far off from the ideal threshold (degrees or ratio)


class FormAnalyzer:
    """
    Produces a list of current FormErrors from the latest exercise state.

    Errors are derived from the exercise definition's form thresholds and the
    current tracked values. Each error names a body part so the skeleton overlay
    can highlight it in red/yellow.

    Currently implements squat-specific rules.
    Future exercises can add their own rule sets here.
    """

    def __init__(self, definition: ExerciseDefinition):
        self.definition = definition
        # Map signal names to the body parts they implicate
        self._signal_to_parts: Dict[str, List[str]] = _build_signal_part_map(definition)
        # Body parts actually tracked by THIS exercise (e.g. bicep curl only
        # implicates the arms) -- used by the skeleton overlay to avoid
        # drawing/coloring joints the exercise doesn't evaluate.
        self.relevant_body_parts: set = self._compute_relevant_parts()

    def _compute_relevant_parts(self) -> set:
        signal_names = {s.name for s in self.definition.angle_signals}
        signal_names |= {s.name for s in self.definition.ratio_signals}
        parts: set = set()
        for name in signal_names:
            parts.update(self._signal_to_parts.get(name, []))
        return parts

    def analyze(
        self,
        current_state: str,
        tracked_values: Dict[str, float],
        signal_values: Dict[str, float],
    ) -> List[FormError]:
        """
        Return all current form errors, one per affected body part. Empty list = perfect form.

        The returned list is used for two purposes:
          • draw_form_skeleton() — colors every body part that has an error
          • draw_form_errors()   — shows unique text messages (dedup + cap happens there)

        Args:
            current_state:  state-machine state string ("DOWN", "UP", etc.)
            tracked_values: peak/trough values accumulated so far in this rep phase
            signal_values:  current frame signal values (for live comparison)
        """
        errors: List[FormError] = []

        # Only produce errors while in the form-tracking state
        if current_state != self.definition.form.during_state:
            return errors

        # Use current signal values for live feedback (not tracked extremes)
        # so the errors update every frame and the user can see real-time changes.
        values = signal_values

        # Check each form level to find what rating the current pose would get
        # and generate errors describing how far off from better ratings the user is.
        for level in self.definition.form.levels:
            for condition in level.conditions:
                val = values.get(condition.signal)
                if val is None:
                    continue

                # Check if the condition fails (i.e., the user isn't meeting this level)
                passes = (
                    val > condition.threshold
                    if condition.direction == "above"
                    else val < condition.threshold
                )
                if not passes:
                    deviation = abs(val - condition.threshold)
                    severity: Severity = "major" if deviation > 15 else "minor"
                    # Creator-written "almost there" prompt (Training panel
                    # Step 4) overrides the generic message, but only for the
                    # exercise's primary tracked signal -- incidental signals
                    # still use the generic per-signal phrasing.
                    override = (
                        self.definition.prompts.almost_there
                        if condition.signal in self.definition.form.tracked_signals
                        else ""
                    )
                    message = _make_message(
                        condition.signal,
                        condition.direction,
                        condition.threshold,
                        val,
                        level.rating,
                        override=override,
                    )
                    parts = self._signal_to_parts.get(condition.signal, ["torso"])
                    for part in parts:
                        errors.append(FormError(
                            body_part=part,
                            severity=severity,
                            message=message,
                            signal_name=condition.signal,
                            deviation=round(deviation, 1),
                        ))

        # Deduplicate by body part — keep the worst error per segment so the
        # skeleton overlay colours every affected segment exactly once.
        part_worst: Dict[str, FormError] = {}
        for err in errors:
            existing = part_worst.get(err.body_part)
            if existing is None:
                part_worst[err.body_part] = err
            elif err.severity == "major" and existing.severity == "minor":
                part_worst[err.body_part] = err
            elif err.severity == existing.severity and err.deviation > existing.deviation:
                part_worst[err.body_part] = err

        result = list(part_worst.values())
        # Sort: major errors first, then by deviation descending
        result.sort(key=lambda e: (0 if e.severity == "major" else 1, -e.deviation))
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_signal_part_map(definition: ExerciseDefinition) -> Dict[str, List[str]]:
    """Map signal names to the body parts they most directly implicate."""
    mapping: Dict[str, List[str]] = {
        "knee_angle":      ["left_thigh", "right_thigh", "left_shin", "right_shin"],
        "elbow_angle":     ["left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm"],
        "arm_angle":       ["left_upper_arm", "right_upper_arm"],
        "hip_angle":       ["torso", "left_hip", "right_hip"],
        "leg_spread_ratio":["left_shin", "right_shin"],
        "arm_spread_ratio":["left_forearm", "right_forearm"],
        "ankle_angle":     ["left_shin", "right_shin"],
    }
    return mapping


def _make_message(
    signal: str,
    direction: str,
    threshold: float,
    current: float,
    target_rating: str,
    override: str = "",
) -> str:
    """Generate a concise user-facing message for a form error.

    Keyed on (signal, direction) only — exercise name is intentionally excluded
    so messages work correctly for both hardcoded and learned exercise definitions.

    Args:
        override: a creator-written prompt (ExercisePrompts.almost_there,
            Training panel Step 4). Takes priority over both the hardcoded
            table below and the generic fallback when non-empty.
    """
    if override:
        return override

    messages = {
        ("knee_angle",        "below"): "Squat deeper",
        ("elbow_angle",       "below"): "Lower your chest further",
        ("arm_angle",         "above"): "Raise arms higher",
        ("leg_spread_ratio",  "above"): "Spread legs wider",
        ("arm_spread_ratio",  "above"): "Raise arms wider",
        ("hip_angle",         "below"): "Hinge hips more",
    }
    key = (signal, direction)
    if key in messages:
        return messages[key]

    # Generic fallback for any user-trained signal not listed above
    verb = "increase" if direction == "above" else "decrease"
    return f"{signal.replace('_', ' ').title()}: {verb} to {threshold:.0f}"
