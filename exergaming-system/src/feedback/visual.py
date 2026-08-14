"""
Visual Feedback Module
Provides visual cues and overlays on video feed
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..exercises.base_exercise import FormFeedback, ExerciseState


# ------------------------------------------------------------------
# Skeleton overlay — per-body-part colour coding (Contribution 1)
# ------------------------------------------------------------------

# MediaPipe landmark index numbers for each named body segment.
# Each entry is (landmark_index_A, landmark_index_B).
# Indices from: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
_SEGMENT_LANDMARKS = {
    # Legs
    "left_thigh":     (23, 25),   # left hip → left knee
    "right_thigh":    (24, 26),   # right hip → right knee
    "left_shin":      (25, 27),   # left knee → left ankle
    "right_shin":     (26, 28),   # right knee → right ankle
    # Arms
    "left_upper_arm": (11, 13),   # left shoulder → left elbow
    "right_upper_arm":(12, 14),   # right shoulder → right elbow
    "left_forearm":   (13, 15),   # left elbow → left wrist
    "right_forearm":  (14, 16),   # right elbow → right wrist
    # Torso
    "torso":          (11, 23),   # left shoulder → left hip (representative)
    "left_hip":       (23, 24),   # left hip → right hip
    "right_hip":      (24, 26),   # right hip → right knee
}

# Default colour (BGR) when no error and actively in the tracked phase — green
_SEGMENT_DEFAULT_COLOR = (0, 230, 0)
# Colour used outside the tracked phase (e.g. standing between reps) — the
# segment isn't being evaluated right now, so it shouldn't read as "correct".
_SEGMENT_NEUTRAL_COLOR = (200, 200, 200)
# Colours for form errors
_SEVERITY_COLOR = {
    "major": (0, 0, 255),    # Red
    "minor": (0, 165, 255),  # Orange
}
_SEGMENT_THICKNESS = 4
_JOINT_RADIUS = 5


def _resolve_base_name(name: str) -> str:
    """
    Resolve an exercise name to its base key for INSTRUCTIONS / CAMERA lookups.

    Handles learned exercise names such as "Learned Squat" → "Squat" so that
    any user-trained exercise variant inherits the correct instructions and
    camera guidance from its parent exercise type.
    """
    # Exact match — hardcoded exercises pass straight through
    if name in INSTRUCTIONS:
        return name
    # Strip "Learned " prefix (e.g. "Learned Squat" → "Squat")
    if name.startswith("Learned "):
        base = name[len("Learned "):]
        if base in INSTRUCTIONS:
            return base
    # Substring scan — catches "My Squat Trainer", "Custom Push-up", etc.
    for key in INSTRUCTIONS:
        if key.lower() in name.lower():
            return key
    # No match — return original; callers handle missing keys gracefully
    return name


# Per-exercise instructions keyed by (exercise_name, state)
INSTRUCTIONS = {
    'Squat': {
        'inactive': "Click Start Exercise to begin",
        'idle':     "Squat DOWN - bend your knees",
        'up':       "Squat DOWN - bend your knees",
        'down':     "Stand UP to complete rep",
        'default':  "",
    },
    'Push-up': {
        'inactive': "Click Start Exercise to begin",
        'idle':     "Lower DOWN - bend your elbows",
        'up':       "Lower DOWN - bend your elbows",
        'down':     "Push UP to complete rep",
        'default':  "",
    },
    'Jumping Jack': {
        'inactive': "Click Start Exercise to begin",
        'idle':     "Jump OUT - raise arms & spread legs",
        'down':     "Jump OUT - raise arms & spread legs",
        'up':       "Jump IN - return to start position",
        'default':  "",
    },
}

# ------------------------------------------------------------------
# Auto-generated instruction fallback for exercises with no hand-written
# INSTRUCTIONS entry -- i.e. every newly trained exercise (Contribution 1:
# an exercise definition alone is enough to run and explain the exercise,
# with no per-exercise authoring required). Built purely from the
# definition's own state transitions (signal, direction, threshold), the
# same data already used to drive rep counting -- nothing new to learn or
# store. This is the "small version" default; a future Training-panel step
# can let the creator override any of these phrases explicitly.
# ------------------------------------------------------------------

# Natural-language action phrase per (signal, direction). Falls back to a
# generic "increase/decrease <signal>" phrasing for any signal not listed
# here (e.g. a future candidate signal not yet given a hand-written verb).
_ACTION_PHRASES = {
    ("knee_angle",       "below"): "Bend your knees",
    ("knee_angle",       "above"): "Straighten your legs",
    ("elbow_angle",      "below"): "Bend your elbow",
    ("elbow_angle",      "above"): "Extend your arm",
    ("hip_angle",        "below"): "Hinge forward at the hips",
    ("hip_angle",        "above"): "Stand tall",
    ("arm_angle",        "above"): "Raise your arms",
    ("arm_angle",        "below"): "Lower your arms",
    ("ankle_angle",      "below"): "Flex your ankle",
    ("ankle_angle",      "above"): "Point your ankle",
    ("leg_spread_ratio", "above"): "Jump your legs apart",
    ("leg_spread_ratio", "below"): "Bring your legs together",
    ("arm_spread_ratio", "above"): "Spread your arms wide",
    ("arm_spread_ratio", "below"): "Bring your arms together",
}


def _humanize_signal(signal: str) -> str:
    return signal.replace("_angle", "").replace("_ratio", "").replace("_", " ").strip()


def _action_phrase(signal: str, direction: str) -> str:
    if (signal, direction) in _ACTION_PHRASES:
        return _ACTION_PHRASES[(signal, direction)]
    verb = "Increase" if direction == "above" else "Decrease"
    return f"{verb} {_humanize_signal(signal)}"


def _definition_states(definition):
    """
    Identify the tracked (scored) state and the resting state(s) for any
    ExerciseDefinition, directly from its transitions -- the same lookup
    _generate_default_instructions() and the prompts resolver both need.

    Returns (resting_states: List[str], tracked_state: str, enter_t, exit_t)
    or (None, None, None, None) if the definition doesn't have the expected
    two-transition shape.
    """
    try:
        tracked_state = definition.form.during_state
        enter_t = next(t for t in definition.transitions if t.to_state == tracked_state)
        exit_t  = next(t for t in definition.transitions if t.to_state != tracked_state)
        return enter_t.from_states, tracked_state, enter_t, exit_t
    except (StopIteration, AttributeError):
        return None, None, None, None


def _generate_default_instructions(definition) -> Dict[str, str]:
    """
    Auto-derive an {'inactive', <state>, ..., 'default'} instruction dict
    (same shape as a hand-written INSTRUCTIONS entry) directly from an
    ExerciseDefinition, for exercises that have no hand-written entry.

    Uses gym-style action language ("Bend your elbow to begin the rep")
    rather than exposing the internal state-machine labels (UP/DOWN/IDLE)
    to the user.

    Returns {} if the definition doesn't have the expected two-transition
    shape (defensive -- callers fall back to the existing generic default).
    """
    resting_states, tracked_state, enter_t, exit_t = _definition_states(definition)
    if tracked_state is None or not enter_t.conditions or not exit_t.conditions:
        return {}

    enter_cond = enter_t.conditions[0]
    exit_cond  = exit_t.conditions[0]

    resting_instruction = f"{_action_phrase(enter_cond.signal, enter_cond.direction)} to begin the rep"
    tracked_instruction = f"{_action_phrase(exit_cond.signal, exit_cond.direction)} to complete the rep"

    instructions = {'inactive': "Click Start to begin", 'default': tracked_instruction}
    for state in resting_states:
        instructions[state.lower()] = resting_instruction
    instructions[tracked_state.lower()] = tracked_instruction
    return instructions


def generate_default_safety_message(signal: str, direction: str) -> str:
    """
    Suggest a default safety/over-limit warning for a (signal, direction)
    pair -- used by the Training panel's Step 4 to pre-fill the warning text
    as soon as the creator picks a safety signal and direction, e.g.
    ("elbow_angle", "below") -> "Careful — don't bend your elbow further
    than this, ease back."
    """
    verb = "go above" if direction == "above" else "go below"
    return f"Careful - don't let {_humanize_signal(signal)} {verb} this, ease back"


def _generate_default_almost_there(definition) -> str:
    """
    Auto-derive a default "almost there" phrase for the Training panel's
    Step 4 prefill -- reuses the same exit-transition (signal, direction)
    the tracked-phase instruction is built from, with a "further" qualifier.
    Returns "" if the definition doesn't resolve (caller falls back to the
    generic per-frame message form_errors._make_message() already produces).
    """
    _, tracked_state, _, exit_t = _definition_states(definition)
    if tracked_state is None or not exit_t.conditions:
        return ""
    cond = exit_t.conditions[0]
    return f"{_action_phrase(cond.signal, cond.direction)} a little further"


def resolve_instructions(name: str, definition=None) -> Dict[str, str]:
    """
    Build the final instruction-banner dict for one exercise, in priority
    order (highest first):
      1. Creator-written prompts   -- definition.prompts.resting / .tracked
      2. Hand-written INSTRUCTIONS -- the 3 original hardcoded exercises
      3. Auto-generated fallback   -- _generate_default_instructions()

    Creator prompts win even for a hardcoded exercise, since the Training
    panel's "Edit prompts" mode (see main.py TrainingPanel) allows editing
    any exercise's prompts, not just newly trained ones.
    """
    base_name = _resolve_base_name(name)
    instructions = dict(INSTRUCTIONS.get(base_name, {}))

    if definition is None:
        return instructions

    generated = _generate_default_instructions(definition)
    for key, val in generated.items():
        instructions.setdefault(key, val)

    prompts = definition.prompts
    if prompts.resting or prompts.tracked:
        resting_states, tracked_state, _, _ = _definition_states(definition)
        if tracked_state is not None:
            if prompts.tracked:
                instructions[tracked_state.lower()] = prompts.tracked
                instructions['default'] = prompts.tracked
            if prompts.resting:
                for state in resting_states:
                    instructions[state.lower()] = prompts.resting

    return instructions


# Camera orientation hint per exercise
CAMERA_HINT = {
    'Squat':        "Side view",
    'Push-up':      "Side view",
    'Jumping Jack': "Front view",
}

# Camera placement guide shown for the first few seconds after exercise starts
CAMERA_GUIDE = {
    'Squat':        ("Side view", "Hip height", "6-8 feet away"),
    'Push-up':      ("Side view", "Hip height", "6-8 feet away"),
    'Jumping Jack': ("Front view", "Chest height", "6-8 feet away"),
}

# How long (seconds) the camera guide is shown after Start is pressed
CAMERA_GUIDE_DURATION = 4.0


class VisualFeedback:
    """Handles visual feedback rendering"""

    COLORS = {
        'excellent': (0, 255, 0),    # Green
        'good':      (0, 200, 255),  # Yellow
        'fair':      (0, 165, 255),  # Orange
        'poor':      (0, 0, 255),    # Red
        'idle':      (200, 200, 200),
        'text':      (255, 255, 255),
        'background':(0, 0, 0),
        'hint':      (180, 180, 0),  # Cyan-ish for camera hint
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.thickness = 2

    def draw_rep_counter(self, frame: np.ndarray, reps: int, position: tuple = (50, 100)):
        """Draw repetition counter on frame"""
        text = f"Reps: {reps}"
        text_size = cv2.getTextSize(text, self.font, self.font_scale * 1.5, self.thickness + 2)[0]
        cv2.rectangle(frame,
                      (position[0] - 10, position[1] - text_size[1] - 20),
                      (position[0] + text_size[0] + 10, position[1] + 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, position, self.font,
                    self.font_scale * 1.5, self.COLORS['text'], self.thickness + 2)

    def draw_form_feedback(self, frame: np.ndarray, form: FormFeedback,
                           position: tuple = (50, 180)):
        """Draw form quality feedback"""
        text = f"Form: {form.value.upper()}"
        color = self.COLORS.get(form.value, self.COLORS['text'])
        cv2.putText(frame, text, position, self.font,
                    self.font_scale, color, self.thickness)

    def draw_complete_overlay(self, frame: np.ndarray, exercise_info: dict,
                              show_debug: bool = True, definition=None):
        """
        Draw the complete feedback overlay.

        All exercise stats (name, reps, form, angle) are grouped into a single
        compact semi-transparent panel in the top-left corner so they take up
        less space and don't visually compete with the skeleton or the subject.
        The instruction banner is placed at the bottom-centre with a
        semi-transparent background and raised high enough to avoid colliding
        with the form-error pills drawn by draw_form_errors().

        Args:
            definition: the active ExerciseDefinition (optional). Used only
                when the exercise has no hand-written INSTRUCTIONS entry --
                lets _generate_default_instructions() build a fallback
                banner from the definition's own transitions instead of
                showing nothing (see visual.py module docstring).
        """
        name   = exercise_info.get('name', 'Exercise')
        state  = exercise_info.get('state', 'idle')
        active = exercise_info.get('active', False)
        reps   = exercise_info.get('reps', 0)
        form   = exercise_info.get('form', 'good')
        if isinstance(form, str):
            form = FormFeedback(form)

        h, w = frame.shape[:2]
        form_color = self.COLORS.get(form.value, self.COLORS['text'])

        # ── Compact info panel — top-left, semi-transparent ──────────────
        # Consolidates: exercise name, rep count, form quality, live angle.
        # (text, BGR-color, font-scale, thickness)
        panel_lines: List[Tuple] = [
            (name,                          (180, 180, 180), 0.50, 1),
            (f"Reps: {reps}",               self.COLORS['text'], 0.80, 2),
            (f"Form: {form.value.upper()}", form_color,          0.58, 1),
        ]
        if show_debug:
            angle_text = None
            if 'knee_angle' in exercise_info:
                angle_text = f"Knee: {exercise_info['knee_angle']:.1f} deg"
            elif 'elbow_angle' in exercise_info:
                angle_text = f"Elbow: {exercise_info['elbow_angle']:.1f} deg"
            elif 'arm_angle' in exercise_info:
                spread = exercise_info.get('leg_spread_ratio', 0)
                angle_text = (f"Arm: {exercise_info['arm_angle']:.1f} deg"
                              f"  Spread: {spread:.2f}x")
            if angle_text:
                panel_lines.append((angle_text, (150, 150, 150), 0.45, 1))

        line_h  = 26
        pad     = 10
        panel_w = max(
            cv2.getTextSize(t, self.font, sc, tk)[0][0]
            for t, _, sc, tk in panel_lines
        ) + pad * 2
        panel_h = len(panel_lines) * line_h + pad * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h), (10, 10, 18), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        ty = 6 + pad + 16
        for text, color, scale, thick in panel_lines:
            cv2.putText(frame, text, (6 + pad, ty),
                        self.font, scale, color, thick, cv2.LINE_AA)
            ty += line_h

        # ── Camera hint — top-right, unobtrusive ─────────────────────────
        hint = CAMERA_HINT.get(_resolve_base_name(name), '')
        if hint:
            hint_text  = f"Camera: {hint}"
            hint_tw    = cv2.getTextSize(hint_text, self.font, 0.46, 1)[0][0]
            cv2.putText(frame, hint_text,
                        (w - hint_tw - 10, 22),
                        self.font, 0.46, (100, 150, 100), 1, cv2.LINE_AA)

        # ── Instruction banner — bottom-centre, semi-transparent ──────────
        # Raised to h-90 so the form-error pills (which stack up from the
        # very bottom) have room below without overlapping.
        instructions = resolve_instructions(name, definition)
        if not active:
            instruction = instructions.get('inactive', "Click Start to begin")
        else:
            instruction = instructions.get(state, instructions.get('default', ''))

        if instruction:
            fs  = 0.62
            ts  = cv2.getTextSize(instruction, self.font, fs, 2)[0]
            ix  = (w - ts[0]) // 2
            iy  = h - 90

            ov2 = frame.copy()
            cv2.rectangle(ov2,
                          (ix - 12, iy - 26), (ix + ts[0] + 12, iy + 8),
                          (10, 10, 18), -1)
            cv2.addWeighted(ov2, 0.68, frame, 0.32, 0, frame)
            cv2.putText(frame, instruction, (ix, iy),
                        self.font, fs, self.COLORS['text'], 2, cv2.LINE_AA)

    def draw_camera_guide(self, frame: np.ndarray, exercise_name: str,
                          elapsed: float) -> None:
        """
        Draw a camera placement guide banner for the first CAMERA_GUIDE_DURATION
        seconds after an exercise starts. Fades out in the final second.

        Args:
            frame:         BGR frame (modified in-place)
            exercise_name: current exercise name
            elapsed:       seconds since exercise was started
        """
        guide = CAMERA_GUIDE.get(_resolve_base_name(exercise_name))
        if not guide or elapsed >= CAMERA_GUIDE_DURATION:
            return

        h, w = frame.shape[:2]

        # Alpha fade: full opacity for first 3s, fade out in the last 1s
        fade_start = CAMERA_GUIDE_DURATION - 1.0
        alpha = 1.0 if elapsed < fade_start else (CAMERA_GUIDE_DURATION - elapsed)
        alpha = max(0.0, min(1.0, alpha))

        view, height_hint, distance = guide
        lines = [
            "CAMERA PLACEMENT",
            f"  View:      {view}",
            f"  Height:    {height_hint}",
            f"  Distance:  {distance}",
        ]

        line_h = 32
        pad = 16
        panel_h = pad * 2 + len(lines) * line_h
        panel_w = 320
        x1 = (w - panel_w) // 2
        y1 = (h - panel_h) // 2 - 20   # slightly above centre
        x2 = x1 + panel_w
        y2 = y1 + panel_h

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 200, 80), 2)
        cv2.addWeighted(overlay, alpha * 0.82, frame, 1 - alpha * 0.82, 0, frame)

        # Text — drawn directly (OpenCV has no per-pixel alpha for text,
        # so blend the whole panel then draw text at adjusted opacity)
        # Re-draw text on blended frame with reduced brightness when fading
        brightness = int(255 * alpha)
        color_title = (0, brightness, 0)
        color_body  = (brightness, brightness, brightness)

        ty = y1 + pad + line_h - 8
        for i, line in enumerate(lines):
            color = color_title if i == 0 else color_body
            scale = 0.55 if i == 0 else 0.50
            cv2.putText(frame, line, (x1 + pad, ty),
                        self.font, scale, color, 1, cv2.LINE_AA)
            ty += line_h

    def draw_countdown(self, frame: np.ndarray, seconds_remaining: float,
                       exercise_name: str) -> None:
        """
        Draw a large "get ready" countdown before rep-counting activates.

        Fixes a false-positive bug: clicking Start while still getting into
        position could count a rep immediately. During this countdown the
        camera and pose detection keep running (so the skeleton is visible
        and the user can confirm they're in frame) but analyze_pose() is not
        called yet, so no rep can be counted.

        Args:
            frame:             BGR frame (modified in-place)
            seconds_remaining: time left before the exercise activates
            exercise_name:     name shown in the "get ready" label
        """
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        label = f"Get ready: {exercise_name}"
        lbl_size = cv2.getTextSize(label, self.font, 0.85, 2)[0]
        cv2.putText(frame, label, ((w - lbl_size[0]) // 2, h // 2 - 90),
                    self.font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        count = max(1, int(seconds_remaining) + 1)
        num_text = str(count)
        num_size = cv2.getTextSize(num_text, self.font, 4.5, 8)[0]
        cv2.putText(frame, num_text,
                    ((w - num_size[0]) // 2, h // 2 + num_size[1] // 2),
                    self.font, 4.5, (0, 220, 255), 8, cv2.LINE_AA)

        hint = "Step into frame and get into position"
        hint_size = cv2.getTextSize(hint, self.font, 0.6, 1)[0]
        cv2.putText(frame, hint,
                    ((w - hint_size[0]) // 2, h // 2 + num_size[1] + 50),
                    self.font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_form_skeleton(
        self,
        frame: np.ndarray,
        raw_landmarks,          # MediaPipe NormalizedLandmarkList (or None)
        form_errors: list,      # list of FormError objects
        relevant_parts: Optional[set] = None,   # body parts this exercise tracks
        is_tracking: bool = True,               # currently in form.during_state?
    ) -> None:
        """
        Draw a coloured skeleton overlay where body segments are:
          • Green  — good form, actively being evaluated right now
          • Grey   — not currently being evaluated (outside the tracked phase,
                     e.g. standing at rest between reps)
          • Orange — minor form issue
          • Red    — major form issue

        Only segments relevant to the active exercise are drawn at all — e.g.
        a bicep curl (elbow_angle only) does not render leg/torso segments,
        since the system isn't evaluating them and drawing them in green would
        imply an accuracy claim it isn't making.

        This is Dissertation Contribution 1: visual body-part-specific feedback.
        Multiple errors are shown simultaneously by colouring each segment
        independently. This replaces the single binary correct/wrong overlay
        used in the existing literature.

        Args:
            frame:          BGR frame (modified in-place)
            raw_landmarks:  mediapipe pose.process().pose_landmarks (can be None)
            form_errors:    list of FormError from FormAnalyzer.analyze()
            relevant_parts: body-part names tracked by the active exercise
                             (FormAnalyzer.relevant_body_parts). None = draw all
                             (back-compat / callers that don't have a definition).
            is_tracking:    True while current_state == form.during_state, i.e.
                             the phase where form is actually being scored.
        """
        if raw_landmarks is None:
            return

        h, w = frame.shape[:2]
        lm = raw_landmarks.landmark

        # Build a map: body_part_name → worst severity colour
        part_colors: Dict[str, Tuple[int, int, int]] = {}
        for err in form_errors:
            color = _SEVERITY_COLOR[err.severity]
            existing = part_colors.get(err.body_part)
            # Red beats orange — keep the worse colour
            if existing is None or err.severity == "major":
                part_colors[err.body_part] = color

        # Draw each skeleton segment with its colour
        for seg_name, (idx_a, idx_b) in _SEGMENT_LANDMARKS.items():
            if relevant_parts is not None and seg_name not in relevant_parts:
                continue
            if idx_a >= len(lm) or idx_b >= len(lm):
                continue
            la, lb = lm[idx_a], lm[idx_b]
            if la.visibility < 0.25 or lb.visibility < 0.25:
                continue

            pt_a = (int(la.x * w), int(la.y * h))
            pt_b = (int(lb.x * w), int(lb.y * h))
            if seg_name in part_colors:
                color = part_colors[seg_name]
            elif is_tracking:
                color = _SEGMENT_DEFAULT_COLOR
            else:
                color = _SEGMENT_NEUTRAL_COLOR

            cv2.line(frame, pt_a, pt_b, color, _SEGMENT_THICKNESS, cv2.LINE_AA)
            cv2.circle(frame, pt_a, _JOINT_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt_b, _JOINT_RADIUS, color, -1, cv2.LINE_AA)

    def draw_form_errors(
        self,
        frame: np.ndarray,
        form_errors: list,      # list of FormError objects
    ) -> None:
        """
        Draw form error messages as a stacked list in the bottom-left corner.

        Up to 3 errors are shown simultaneously (most severe first).
        Each error has a coloured bullet matching its severity.

        Args:
            frame:       BGR frame (modified in-place)
            form_errors: list of FormError from FormAnalyzer.analyze()
        """
        if not form_errors:
            return

        # Deduplicate by message so the same cue isn't shown multiple times
        # (e.g. "Squat deeper" appears once even if 4 body segments are affected).
        # Cap at 3 unique messages to avoid cluttering the screen.
        seen_messages: set = set()
        display_errors = []
        for err in form_errors:
            if err.message not in seen_messages:
                seen_messages.add(err.message)
                display_errors.append(err)
            if len(display_errors) == 3:
                break

        if not display_errors:
            return

        h, w  = frame.shape[:2]
        x_base = 14
        # Stack pills upward from the bottom, leaving room above the instruction
        # banner (which sits at h-90).  28 px per pill keeps them compact.
        y_base = h - 20 - len(display_errors) * 28

        for err in display_errors:
            bullet_color = _SEVERITY_COLOR.get(err.severity, (0, 165, 255))
            text      = f"  {err.message}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]

            # Semi-transparent pill background
            pill_x1 = x_base - 4
            pill_y1 = y_base - text_size[1] - 5
            pill_x2 = x_base + text_size[0] + 8
            pill_y2 = y_base + 5
            overlay = frame.copy()
            cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2),
                          (10, 10, 18), -1)
            cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

            # Bullet dot
            cy = (pill_y1 + pill_y2) // 2
            cv2.circle(frame, (x_base + 6, cy), 4, bullet_color, -1, cv2.LINE_AA)

            # Message text
            cv2.putText(frame, text, (x_base, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (230, 230, 230), 1, cv2.LINE_AA)
            y_base += 28

    def draw_safety_warning(self, frame: np.ndarray, message: str) -> None:
        """
        Draw a prominent red safety warning banner across the top of the
        frame, e.g. "Careful — don't drop your arm that low on the way
        back up." Shown when ExerciseDefinition.safety_condition is
        violated (GenericExercise._safety_violated). Deliberately styled
        differently (top, red, bordered) from the ordinary form-error pills
        (bottom-left, orange/red dot) so a genuine safety cue reads as more
        urgent than routine form feedback.

        Args:
            frame:   BGR frame (modified in-place)
            message: the warning text (definition.prompts.over_limit or
                     definition.safety_message; callers only call this when
                     one of those is non-empty)
        """
        if not message:
            return

        h, w = frame.shape[:2]
        text = f"! {message}"
        ts = cv2.getTextSize(text, self.font, 0.62, 2)[0]
        x = (w - ts[0]) // 2
        y = 110

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 16, y - ts[1] - 12), (x + ts[0] + 16, y + 12),
                      (0, 0, 90), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x - 16, y - ts[1] - 12), (x + ts[0] + 16, y + 12),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (x, y), self.font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_quality_warnings(self, frame: np.ndarray,
                              warnings: list) -> None:
        """
        Draw quality warning banners on the frame.
        Warnings stack downward from the upper-centre of the frame.

        Args:
            frame:    BGR frame (modified in-place)
            warnings: list of warning strings from FrameQualityChecker
        """
        if not warnings:
            return

        h, w = frame.shape[:2]
        y = 148         # start below the top-left info panel (≈4 lines × 26px + 6px margin)
        pad_x = 12
        pad_y = 6

        for warning in warnings:
            text_size = cv2.getTextSize(warning, self.font, 0.62, 1)[0]
            tw, th = text_size
            # Centre each banner horizontally
            x = (w - tw) // 2
            # Background
            cv2.rectangle(frame,
                          (x - pad_x, y - th - pad_y),
                          (x + tw + pad_x, y + pad_y),
                          (0, 0, 0), -1)
            cv2.rectangle(frame,
                          (x - pad_x, y - th - pad_y),
                          (x + tw + pad_x, y + pad_y),
                          (0, 100, 255), 1)   # orange border
            cv2.putText(frame, warning, (x, y),
                        self.font, 0.62, (0, 140, 255), 1, cv2.LINE_AA)
            y += th + pad_y * 2 + 8
