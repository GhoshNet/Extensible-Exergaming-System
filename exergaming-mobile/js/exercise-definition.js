// Ported from exergaming-system/src/exercises/exercise_definition.py
// Same schema, same field names, so the bundled JSON files (exported
// straight from the desktop app's src/exercises/definitions/) load
// without any conversion step.

// Standard MediaPipe Pose topology (33 landmarks, indices 0-32).
// Identical between the Python `mediapipe.solutions.pose` used by the
// desktop app and MediaPipe Tasks Vision (Web) used here — same model
// family, same landmark order — so this index/name map is authoritative
// for both.
export const POSE_LANDMARK_NAMES = [
  'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
  'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
  'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
  'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
  'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
  'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
  'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
  'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
  'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
];

// Mirrors PoseDetector.get_all_landmarks_dict(): visibility-filtered,
// pixel-space (x, y) keyed by landmark name. `visibility` on MediaPipe
// Tasks Vision landmarks is the same [0,1] confidence score as the
// Python solution's `.visibility`.
export function landmarksToDict(poseLandmarks, worldWidth, worldHeight, visibilityThreshold = 0.25) {
  const dict = {};
  if (!poseLandmarks) return dict;
  for (let i = 0; i < poseLandmarks.length; i++) {
    const lm = poseLandmarks[i];
    const vis = lm.visibility ?? 1.0;
    if (vis < visibilityThreshold) continue;
    dict[POSE_LANDMARK_NAMES[i]] = [lm.x * worldWidth, lm.y * worldHeight];
  }
  return dict;
}

export class SignalCondition {
  constructor({ signal, direction, threshold }) {
    this.signal = signal;
    this.direction = direction; // "above" | "below"
    this.threshold = threshold;
  }
  evaluate(signalValues) {
    const val = signalValues[this.signal];
    if (val === undefined) return false;
    return this.direction === 'above' ? val > this.threshold : val < this.threshold;
  }
  static fromDict(d) {
    return new SignalCondition({ signal: d.signal, direction: d.direction, threshold: Number(d.threshold) });
  }
}

export class StateTransition {
  constructor({ from_states, to_state, conditions }) {
    this.fromStates = from_states;
    this.toState = to_state;
    this.conditions = conditions;
  }
  matches(currentState, signalValues) {
    if (!this.fromStates.includes(currentState)) return false;
    return this.conditions.every((c) => c.evaluate(signalValues));
  }
  static fromDict(d) {
    return new StateTransition({
      from_states: d.from_states,
      to_state: d.to_state,
      conditions: d.conditions.map(SignalCondition.fromDict),
    });
  }
}

export class FormLevel {
  constructor({ rating, operator, conditions }) {
    this.rating = rating;
    this.operator = operator || 'all';
    this.conditions = conditions;
  }
  evaluate(trackedValues) {
    const results = this.conditions.map((c) => c.evaluate(trackedValues));
    return this.operator === 'all' ? results.every(Boolean) : results.some(Boolean);
  }
  static fromDict(d) {
    return new FormLevel({
      rating: d.rating,
      operator: d.operator || 'all',
      conditions: d.conditions.map(SignalCondition.fromDict),
    });
  }
}

export class FormEvaluation {
  constructor({ tracking, during_state, tracked_signals, levels }) {
    this.tracking = tracking; // "min" | "max"
    this.duringState = during_state;
    this.trackedSignals = tracked_signals;
    this.levels = levels;
  }
  static fromDict(d) {
    return new FormEvaluation({
      tracking: d.tracking,
      during_state: d.during_state,
      tracked_signals: d.tracked_signals,
      levels: d.levels.map(FormLevel.fromDict),
    });
  }
}

export class ExerciseDefinition {
  constructor(opts) {
    Object.assign(this, opts);
  }
  static fromDict(d) {
    return new ExerciseDefinition({
      name: d.name,
      description: d.description || '',
      source: d.source || 'hardcoded',
      cameraView: d.camera_view,
      fallbackMode: d.fallback_mode || 'bilateral',
      initialState: d.initial_state,
      angleSignals: d.angle_signals || [],
      ratioSignals: d.ratio_signals || [],
      transitions: d.transitions.map(StateTransition.fromDict),
      repCompleteTransition: d.rep_complete_transition,
      form: FormEvaluation.fromDict(d.form),
      safetyCondition: d.safety_condition ? SignalCondition.fromDict(d.safety_condition) : null,
      safetyMessage: d.safety_message || '',
    });
  }
  static async fromUrl(url) {
    const res = await fetch(url);
    const d = await res.json();
    return ExerciseDefinition.fromDict(d);
  }
}
