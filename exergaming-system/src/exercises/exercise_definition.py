"""
Exercise Definition — data-driven format for describing any exercise.

Each exercise is fully specified by a JSON file in src/exercises/definitions/.
GenericExercise consumes an ExerciseDefinition to drive its state machine,
angle calculations, and form evaluation without any hand-written Python logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


@dataclass
class AngleSignal:
    """
    A joint angle computed from three consecutive landmarks.

    If bilateral=True the landmarks list contains BASE names (e.g. ["HIP", "KNEE", "ANKLE"]).
    GenericExercise will compute the angle for LEFT_ and RIGHT_ prefixed variants
    and apply the bilateral fallback (average / single side / skip).

    If bilateral=False the landmarks list contains FULL names
    (e.g. ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]) and no fallback is applied.
    """
    name: str                   # e.g. "knee_angle"
    landmarks: List[str]        # 3 landmark names (BASE or FULL depending on bilateral)
    bilateral: bool = True
    importance: float = 1.0     # used by the learning algorithm to rank signals

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "landmarks": self.landmarks,
            "bilateral": self.bilateral,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AngleSignal:
        return cls(
            name=d["name"],
            landmarks=d["landmarks"],
            bilateral=d.get("bilateral", True),
            importance=d.get("importance", 1.0),
        )


@dataclass
class RatioSignal:
    """
    A ratio of two Euclidean distances: dist(numerator) / dist(denominator).

    Both landmark lists must contain exactly 2 FULL landmark names.
    Used by jumping jacks (ankle distance / hip distance).
    """
    name: str                       # e.g. "leg_spread_ratio"
    numerator: List[str]            # [p1, p2] full names, e.g. ["LEFT_ANKLE", "RIGHT_ANKLE"]
    denominator: List[str]          # [p1, p2] full names, e.g. ["LEFT_HIP", "RIGHT_HIP"]
    importance: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RatioSignal:
        return cls(
            name=d["name"],
            numerator=d["numerator"],
            denominator=d["denominator"],
            importance=d.get("importance", 1.0),
        )


@dataclass
class SignalCondition:
    """
    A single threshold check on a named signal.

    direction="above"  →  condition passes when signal_value > threshold
    direction="below"  →  condition passes when signal_value < threshold
    """
    signal: str
    direction: Literal["above", "below"]
    threshold: float

    def evaluate(self, signal_values: Dict[str, float]) -> bool:
        val = signal_values.get(self.signal)
        if val is None:
            return False
        return val > self.threshold if self.direction == "above" else val < self.threshold

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "direction": self.direction,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SignalCondition:
        return cls(
            signal=d["signal"],
            direction=d["direction"],
            threshold=float(d["threshold"]),
        )


@dataclass
class StateTransition:
    """
    A state-machine transition that fires when ALL conditions are satisfied.

    from_states: list of states from which this transition can fire.
    Typically ["IDLE", "UP"] or ["DOWN"] etc.
    """
    from_states: List[str]
    to_state: str
    conditions: List[SignalCondition]   # ALL must be true (AND logic)

    def matches(self, current_state: str, signal_values: Dict[str, float]) -> bool:
        if current_state not in self.from_states:
            return False
        return all(c.evaluate(signal_values) for c in self.conditions)

    def to_dict(self) -> dict:
        return {
            "from_states": self.from_states,
            "to_state": self.to_state,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateTransition:
        return cls(
            from_states=d["from_states"],
            to_state=d["to_state"],
            conditions=[SignalCondition.from_dict(c) for c in d["conditions"]],
        )


@dataclass
class FormLevel:
    """
    One quality level in the form evaluation hierarchy.

    operator="all"  →  ALL conditions must pass (AND)
    operator="any"  →  at least ONE condition must pass (OR)

    Levels are evaluated in order (excellent → good → fair).
    The first level whose conditions pass is the rating; else POOR.
    """
    rating: str                             # "excellent" | "good" | "fair"
    conditions: List[SignalCondition]
    operator: Literal["all", "any"] = "all"

    def evaluate(self, tracked_values: Dict[str, float]) -> bool:
        results = [c.evaluate(tracked_values) for c in self.conditions]
        return all(results) if self.operator == "all" else any(results)

    def to_dict(self) -> dict:
        return {
            "rating": self.rating,
            "operator": self.operator,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FormLevel:
        return cls(
            rating=d["rating"],
            operator=d.get("operator", "all"),
            conditions=[SignalCondition.from_dict(c) for c in d["conditions"]],
        )


@dataclass
class ExercisePrompts:
    """
    Creator-editable text shown to the user at specific points in a rep.

    Any field left "" falls back to an auto-generated default: resting/
    tracked fall back to visual.py's _generate_default_instructions()
    (derived from the definition's own transitions); almost_there falls
    back to form_errors.py's _make_message() generic phrasing. There is no
    generated fallback for over_limit -- it only ever shows if the creator
    has both defined a safety_condition AND written a message for it.

    resting:      shown while resting, telling the user how to begin the rep
                  (e.g. "Bend your elbow to begin the rep").
    tracked:      shown once in the tracked/scored phase, telling the user
                  how to complete the rep (e.g. "Extend your arm to finish").
    almost_there: shown during the tracked phase when form is close but not
                  yet at the best level (the "yellow" / near-target case).
    over_limit:   shown if safety_condition (on ExerciseDefinition) is
                  violated -- e.g. "Careful — don't drop your arm that low
                  on the way back up."
    """
    resting: str = ""
    tracked: str = ""
    almost_there: str = ""
    over_limit: str = ""

    def to_dict(self) -> dict:
        return {
            "resting": self.resting,
            "tracked": self.tracked,
            "almost_there": self.almost_there,
            "over_limit": self.over_limit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExercisePrompts:
        return cls(
            resting=d.get("resting", ""),
            tracked=d.get("tracked", ""),
            almost_there=d.get("almost_there", ""),
            over_limit=d.get("over_limit", ""),
        )

    def is_empty(self) -> bool:
        return not (self.resting or self.tracked or self.almost_there or self.over_limit)


@dataclass
class FormEvaluation:
    """
    Specifies how to evaluate form quality at rep completion.

    tracking:        "min" or "max" — whether to track the minimum or maximum of each
                     signal in tracked_signals while in during_state.
    during_state:    the state during which tracking is active (e.g. "DOWN" for squats).
    tracked_signals: signal names to track (usually the primary angle).
    levels:          ordered list of FormLevel (excellent → good → fair); first match wins.
                     If none match, rating is POOR.
    """
    tracking: Literal["min", "max"]
    during_state: str
    tracked_signals: List[str]
    levels: List[FormLevel]

    def to_dict(self) -> dict:
        return {
            "tracking": self.tracking,
            "during_state": self.during_state,
            "tracked_signals": self.tracked_signals,
            "levels": [lv.to_dict() for lv in self.levels],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FormEvaluation:
        return cls(
            tracking=d["tracking"],
            during_state=d["during_state"],
            tracked_signals=d["tracked_signals"],
            levels=[FormLevel.from_dict(lv) for lv in d["levels"]],
        )


@dataclass
class ExerciseDefinition:
    """
    Complete, data-driven specification of an exercise.

    angle_signals:         joint angles to compute each frame.
    ratio_signals:         distance ratios to compute each frame.
    initial_state:         starting state when exercise begins ("IDLE").
    transitions:           state-machine transitions (evaluated in order).
    rep_complete_transition: [from_state, to_state] pair that signals rep completion.
    form:                  form evaluation configuration.
    camera_view:           recommended camera placement.
    fallback_mode:
        "bilateral"    — angle signals with bilateral=True use L/R fallback;
                         frame is processed as long as at least one side is visible.
        "require_all"  — ALL angle/ratio signals must be computable; frame is
                         skipped if any signal is unavailable (jumping jack style).
    source:                "hardcoded" (from JSON) or "learned" (from learning algorithm).
    prompts:               creator-editable rep-phase text (see ExercisePrompts).
                            Defaults to all-empty, which means "use the
                            auto-generated defaults" everywhere they're read.
    safety_condition:       optional extra SignalCondition, checked every
                            active frame independent of the state machine
                            (e.g. "shoulder_angle below 40" to catch a
                            shoulder-press hand dropping too low on the way
                            back up). None = no safety check configured.
    safety_message:         warning shown when safety_condition is violated
                            AND prompts.over_limit is empty (prompts.over_limit
                            takes priority if the creator wrote one).
    """
    name: str
    angle_signals: List[AngleSignal]
    ratio_signals: List[RatioSignal]
    initial_state: str
    transitions: List[StateTransition]
    rep_complete_transition: List[str]      # [from_state, to_state]
    form: FormEvaluation
    camera_view: Literal["side", "front", "either"]
    fallback_mode: Literal["bilateral", "require_all"] = "bilateral"
    description: str = ""
    source: Literal["hardcoded", "learned"] = "hardcoded"
    prompts: ExercisePrompts = field(default_factory=ExercisePrompts)
    safety_condition: Optional[SignalCondition] = None
    safety_message: str = ""

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "camera_view": self.camera_view,
            "fallback_mode": self.fallback_mode,
            "initial_state": self.initial_state,
            "angle_signals": [s.to_dict() for s in self.angle_signals],
            "ratio_signals": [s.to_dict() for s in self.ratio_signals],
            "transitions": [t.to_dict() for t in self.transitions],
            "rep_complete_transition": self.rep_complete_transition,
            "form": self.form.to_dict(),
        }
        if not self.prompts.is_empty():
            d["prompts"] = self.prompts.to_dict()
        if self.safety_condition is not None:
            d["safety_condition"] = self.safety_condition.to_dict()
            d["safety_message"] = self.safety_message
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExerciseDefinition:
        safety_raw = d.get("safety_condition")
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            source=d.get("source", "hardcoded"),
            camera_view=d["camera_view"],
            fallback_mode=d.get("fallback_mode", "bilateral"),
            initial_state=d["initial_state"],
            angle_signals=[AngleSignal.from_dict(s) for s in d.get("angle_signals", [])],
            ratio_signals=[RatioSignal.from_dict(s) for s in d.get("ratio_signals", [])],
            transitions=[StateTransition.from_dict(t) for t in d["transitions"]],
            rep_complete_transition=d["rep_complete_transition"],
            form=FormEvaluation.from_dict(d["form"]),
            prompts=ExercisePrompts.from_dict(d["prompts"]) if "prompts" in d else ExercisePrompts(),
            safety_condition=SignalCondition.from_dict(safety_raw) if safety_raw else None,
            safety_message=d.get("safety_message", ""),
        )

    @classmethod
    def from_json(cls, path: str) -> ExerciseDefinition:
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
