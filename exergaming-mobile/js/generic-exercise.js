// Ported from exergaming-system/src/exercises/generic_exercise.py + base_exercise.py.
// Same state machine, same signal computation, same form-tracking logic,
// evaluated in the same order every frame. Internal state is kept as the
// plain uppercase strings ("IDLE"/"UP"/"DOWN") the Python version converts
// to/from its enum, since JS has no equivalent indirection worth keeping.

import { calculateAngle, calculateDistance } from './pose-analyzer.js';

// A rep completing faster than this is treated as noise/an erratic movement
// rather than a genuine controlled repetition, and isn't counted (state
// still transitions normally, so nothing gets stuck). Measured against the
// timestamp passed into analyzePose(), which for this app is always
// performance.now() (both live camera and file playback run in real
// wall-clock time here -- unlike the desktop's offline batch tools, mobile
// has no fast-forward processing mode). Kept identical to the Python
// engine's MIN_REP_DURATION_MS for parity.
//
// Was 400ms; lowered to 200ms 2026-07-21 after a controlled 24-video
// angle/distance evaluation showed 400ms was exercise-tempo-naive --
// jumping jacks' genuine rep cadence (233-393ms) was landing just under
// it and getting silently rejected. 200ms fixed that with no
// over-counting risk on the other three exercises. See
// exergaming-system/reports/timing_floor_sweep_results.csv.
export const MIN_REP_DURATION_MS = 200.0;

function mergeSide(current, next) {
  if (current === 'both' || next === 'both') return 'both';
  if (current !== 'none' && next !== 'none' && current !== next) return 'both';
  if (current !== 'none') return current;
  return next;
}

export class GenericExercise {
  constructor(definition) {
    this.definition = definition;
    this.name = definition.name;
    this.repCount = 0;
    this.state = definition.initialState || 'IDLE';
    this.formFeedback = 'good'; // matches BaseExercise default (FormFeedback.GOOD)
    this.isActive = false;
    this.lastSideUsed = 'none';

    this.signalValues = {};
    this.formTracked = this._initFormTracked();
    this.safetyViolated = false;

    // Timestamp (ms) the current state was entered. null until the first
    // analyzePose() call, which seeds it rather than assuming t=0.
    this._stateEnteredAt = null;
    // Reps that completed faster than MIN_REP_DURATION_MS and were
    // therefore not counted. Exposed for debugging/tuning.
    this.rejectedFastReps = 0;
    // Actual phase durations (ms) of each rejected rep, in rejection order.
    this.rejectedFastRepDurations = [];
  }

  start() { this.isActive = true; }
  stop() { this.isActive = false; }
  getRepCount() { return this.repCount; }
  getState() { return this.state; }
  getFormFeedback() { return this.formFeedback; }
  incrementRep() { this.repCount += 1; }

  reset() {
    this.repCount = 0;
    this.state = this.definition.initialState || 'IDLE';
    this.isActive = false;
    this.signalValues = {};
    this.formTracked = this._initFormTracked();
    this.safetyViolated = false;
    this._stateEnteredAt = null;
    this.rejectedFastReps = 0;
    this.rejectedFastRepDurations = [];
  }

  getInfo() {
    return {
      name: this.name,
      reps: this.repCount,
      rep_count: this.repCount,
      state: this.state.toLowerCase(),
      form: this.formFeedback,
      active: this.isActive,
      side_used: this.lastSideUsed,
      ...this.signalValues,
      safety_violated: this.safetyViolated,
    };
  }

  getCriticalSignalNames() {
    const names = new Set();
    for (const t of this.definition.transitions) {
      for (const c of t.conditions) names.add(c.signal);
    }
    return [...names].sort();
  }

  getCriticalLandmarks() {
    const critical = new Set(this.getCriticalSignalNames());
    const landmarks = new Set();
    for (const sig of this.definition.angleSignals) {
      if (!critical.has(sig.name)) continue;
      if (sig.bilateral) {
        for (const base of sig.landmarks) {
          landmarks.add(`LEFT_${base}`);
          landmarks.add(`RIGHT_${base}`);
        }
      } else {
        sig.landmarks.forEach((l) => landmarks.add(l));
      }
    }
    for (const sig of this.definition.ratioSignals) {
      if (!critical.has(sig.name)) continue;
      sig.numerator.forEach((l) => landmarks.add(l));
      sig.denominator.forEach((l) => landmarks.add(l));
    }
    return [...landmarks].sort();
  }

  isCriticalSignalVisible(landmarks) {
    const critical = new Set(this.getCriticalSignalNames());
    const missing = [];
    for (const sig of this.definition.angleSignals) {
      if (!critical.has(sig.name)) continue;
      const { value } = this._computeAngleSignal(sig, landmarks);
      if (value === null) missing.push(sig.name);
    }
    for (const sig of this.definition.ratioSignals) {
      if (!critical.has(sig.name)) continue;
      if (this._computeRatioSignal(sig, landmarks) === null) missing.push(sig.name);
    }
    return { visible: missing.length === 0, missing };
  }

  getRequiredLandmarks() {
    const landmarks = new Set();
    for (const sig of this.definition.angleSignals) {
      if (sig.bilateral) {
        for (const base of sig.landmarks) {
          landmarks.add(`LEFT_${base}`);
          landmarks.add(`RIGHT_${base}`);
        }
      } else {
        sig.landmarks.forEach((l) => landmarks.add(l));
      }
    }
    for (const sig of this.definition.ratioSignals) {
      sig.numerator.forEach((l) => landmarks.add(l));
      sig.denominator.forEach((l) => landmarks.add(l));
    }
    return [...landmarks].sort();
  }

  analyzePose(landmarks, timestampMs) {
    if (!this.isActive) return;

    if (timestampMs === undefined || timestampMs === null) {
      timestampMs = performance.now();
    }
    if (this._stateEnteredAt === null) {
      this._stateEnteredAt = timestampMs;
    }

    const { signalValues, sideUsed } = this._computeSignals(landmarks);

    if (this.definition.fallbackMode === 'require_all') {
      const totalExpected = this.definition.angleSignals.length + this.definition.ratioSignals.length;
      if (Object.keys(signalValues).length < totalExpected) {
        this.lastSideUsed = 'none';
        return;
      }
    }

    this.lastSideUsed = sideUsed;
    this.signalValues = signalValues;

    this.safetyViolated =
      this.definition.safetyCondition !== null &&
      this.definition.safetyCondition !== undefined &&
      this.definition.safetyCondition.evaluate(signalValues);

    const currentStateStr = this.state;
    if (currentStateStr === this.definition.form.duringState) {
      this._updateFormTracking();
    }

    for (const transition of this.definition.transitions) {
      if (transition.matches(currentStateStr, signalValues)) {
        const newStateStr = transition.toState;

        const [repFrom, repTo] = this.definition.repCompleteTransition;
        if (currentStateStr === repFrom && newStateStr === repTo) {
          const phaseDuration = timestampMs - this._stateEnteredAt;
          if (phaseDuration >= MIN_REP_DURATION_MS) {
            this.incrementRep();
            this.formFeedback = this._evaluateForm();
          } else {
            this.rejectedFastReps += 1;
            this.rejectedFastRepDurations.push(phaseDuration);
          }
          this.formTracked = this._initFormTracked();
        }

        this.state = newStateStr;
        this._stateEnteredAt = timestampMs;

        if (newStateStr === this.definition.form.duringState) {
          this.formTracked = this._initFormTracked();
          this._updateFormTracking();
        }

        break; // only fire one transition per frame
      }
    }
  }

  _computeSignals(landmarks) {
    const signalValues = {};
    let sideUsed = 'none';

    for (const sig of this.definition.angleSignals) {
      const { value, side } = this._computeAngleSignal(sig, landmarks);
      if (value !== null) {
        signalValues[sig.name] = value;
        sideUsed = mergeSide(sideUsed, side);
      }
    }
    for (const sig of this.definition.ratioSignals) {
      const value = this._computeRatioSignal(sig, landmarks);
      if (value !== null) signalValues[sig.name] = value;
    }
    return { signalValues, sideUsed };
  }

  _computeAngleSignal(sig, landmarks) {
    if (!sig.bilateral) {
      if (sig.landmarks.every((lm) => lm in landmarks)) {
        const pts = sig.landmarks.map((lm) => landmarks[lm]);
        return { value: calculateAngle(pts[0], pts[1], pts[2]), side: 'both' };
      }
      return { value: null, side: 'none' };
    }

    const [p1, p2, p3] = sig.landmarks;
    const leftLms = [`LEFT_${p1}`, `LEFT_${p2}`, `LEFT_${p3}`];
    const rightLms = [`RIGHT_${p1}`, `RIGHT_${p2}`, `RIGHT_${p3}`];

    const leftOk = leftLms.every((lm) => lm in landmarks);
    const rightOk = rightLms.every((lm) => lm in landmarks);

    if (!leftOk && !rightOk) return { value: null, side: 'none' };

    const leftVal = leftOk ? calculateAngle(...leftLms.map((lm) => landmarks[lm])) : null;
    const rightVal = rightOk ? calculateAngle(...rightLms.map((lm) => landmarks[lm])) : null;

    if (leftOk && rightOk) return { value: (leftVal + rightVal) / 2, side: 'both' };
    if (leftOk) return { value: leftVal, side: 'left' };
    return { value: rightVal, side: 'right' };
  }

  _computeRatioSignal(sig, landmarks) {
    const allLms = [...sig.numerator, ...sig.denominator];
    if (!allLms.every((lm) => lm in landmarks)) return null;
    const numDist = calculateDistance(landmarks[sig.numerator[0]], landmarks[sig.numerator[1]]);
    const denDist = calculateDistance(landmarks[sig.denominator[0]], landmarks[sig.denominator[1]]);
    if (denDist <= 0) return null;
    return numDist / denDist;
  }

  _initFormTracked() {
    const initVal = this.definition.form.tracking === 'min' ? Infinity : -Infinity;
    const tracked = {};
    for (const sig of this.definition.form.trackedSignals) tracked[sig] = initVal;
    return tracked;
  }

  _updateFormTracking() {
    for (const sigName of this.definition.form.trackedSignals) {
      const val = this.signalValues[sigName];
      if (val === undefined) continue;
      if (this.definition.form.tracking === 'min') {
        this.formTracked[sigName] = Math.min(this.formTracked[sigName], val);
      } else {
        this.formTracked[sigName] = Math.max(this.formTracked[sigName], val);
      }
    }
  }

  _evaluateForm() {
    const safeValues = {};
    for (const [k, v] of Object.entries(this.formTracked)) {
      safeValues[k] = Math.abs(v) === Infinity ? 0.0 : v;
    }
    for (const level of this.definition.form.levels) {
      if (level.evaluate(safeValues)) return level.rating;
    }
    return 'poor';
  }
}
