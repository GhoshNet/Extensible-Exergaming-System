// Ported from exergaming-system/src/pose/position_gate.py (+ the framing
// checks from quality_checker.py, folded in here rather than kept as a
// separate class -- mobile doesn't have the desktop's raw-frame lighting
// check, so there was no separate "generic checker" worth splitting out).
//
// Decides, frame by frame, whether analyzePose() should actually run.
// Debounced: a bad frame only gates once it's persisted for
// gracePeriodMs, so a single dropped-detection frame doesn't visibly
// interrupt a workout (GenericExercise already handles a momentarily-
// missing signal gracefully on its own).

const FRIENDLY_NAMES = {
  HIP: 'hips', KNEE: 'knees', ANKLE: 'ankles', FOOT_INDEX: 'feet',
  SHOULDER: 'shoulders', ELBOW: 'elbows', WRIST: 'wrists',
};

const DRIFT_TOLERANCE_SCALE = 1.6;
const DEFAULT_GRACE_PERIOD_MS = 400.0;
const TOO_CLOSE_RATIO = 0.85;
// Not the old full-body-bbox value (0.28) -- a critical-signal span (e.g.
// squat's hip-to-ankle, bicep curl's shoulder-to-wrist) is naturally
// smaller than a full-body box, so 0.28 false-positived on normal framing
// during testing. Loosened; still a rough heuristic, worth tuning further
// against real footage.
const TOO_FAR_RATIO = 0.10;

function stripSide(name) {
  return name.replace('LEFT_', '').replace('RIGHT_', '');
}

function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export class PositionGate {
  constructor(exercise, { gracePeriodMs = DEFAULT_GRACE_PERIOD_MS, driftToleranceScale = DRIFT_TOLERANCE_SCALE } = {}) {
    this.exercise = exercise;
    this.gracePeriodMs = gracePeriodMs;
    this.driftToleranceScale = driftToleranceScale;
    this._anchor = null;
    this._anchorScale = null;
    this._badSince = null;
  }

  calibrate(landmarks) {
    const { center, scale } = this._computeAnchor(landmarks);
    this._anchor = center;
    this._anchorScale = scale;
  }

  reset() {
    this._anchor = null;
    this._anchorScale = null;
    this._badSince = null;
  }

  check(canvasWidth, canvasHeight, landmarks, poseDetected, timestampMs) {
    const warnings = [];

    if (!poseDetected || !landmarks || Object.keys(landmarks).length === 0) {
      warnings.push('No person detected — move into camera view');
    } else {
      const { visible, missing } = this.exercise.isCriticalSignalVisible(landmarks);
      if (!visible) {
        warnings.push(this._missingSignalMessage(missing));
      } else {
        // Only check framing once we know the critical parts ARE visible --
        // scoped to just those landmarks, so an exercise that never shows
        // the legs (e.g. a tightly-framed bicep curl) isn't judged against
        // a full-body bounding box.
        const framingWarning = this._checkFraming(canvasWidth, canvasHeight, landmarks);
        if (framingWarning) warnings.push(framingWarning);
      }

      if (this._anchor !== null) {
        const { center, scale } = this._computeAnchor(landmarks);
        const refScale = scale || this._anchorScale;
        if (refScale && center) {
          const drift = distance(center, this._anchor);
          if (drift > refScale * this.driftToleranceScale) {
            warnings.push('Return to your starting position');
          }
        }
      }
    }

    const ok = warnings.length === 0;
    let gated;
    if (ok) {
      this._badSince = null;
      gated = false;
    } else {
      if (this._badSince === null) this._badSince = timestampMs;
      gated = timestampMs - this._badSince >= this.gracePeriodMs;
    }

    return { ok, gated, warnings };
  }

  _checkFraming(canvasWidth, canvasHeight, landmarks) {
    const criticalNames = new Set(this.exercise.getCriticalLandmarks());
    const pts = Object.entries(landmarks)
      .filter(([name]) => criticalNames.has(name))
      .map(([, pt]) => pt);
    if (pts.length < 2) return null; // not enough points for a meaningful box
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const bboxW = Math.max(...xs) - Math.min(...xs);
    const bboxH = Math.max(...ys) - Math.min(...ys);
    if (bboxW > canvasWidth * TOO_CLOSE_RATIO) return 'Too close — step back';
    if (bboxH < canvasHeight * TOO_FAR_RATIO) return 'Too far — step closer';
    return null;
  }

  _missingSignalMessage(missingSignalNames) {
    const parts = [];
    const seen = new Set();
    const missingSet = new Set(missingSignalNames);
    for (const sig of this.exercise.definition.angleSignals) {
      if (!missingSet.has(sig.name)) continue;
      for (const lm of sig.landmarks) {
        const friendly = FRIENDLY_NAMES[stripSide(lm)];
        if (friendly && !seen.has(friendly)) { seen.add(friendly); parts.push(friendly); }
      }
    }
    for (const sig of this.exercise.definition.ratioSignals) {
      if (!missingSet.has(sig.name)) continue;
      for (const lm of [...sig.numerator, ...sig.denominator]) {
        const friendly = FRIENDLY_NAMES[stripSide(lm)];
        if (friendly && !seen.has(friendly)) { seen.add(friendly); parts.push(friendly); }
      }
    }
    if (parts.length === 0) return 'Position yourself so I can see you clearly';
    return `Show your ${parts.join(', ')}`;
  }

  _computeAnchor(landmarks) {
    let center = null;
    if (landmarks.LEFT_HIP && landmarks.RIGHT_HIP) {
      center = midpoint(landmarks.LEFT_HIP, landmarks.RIGHT_HIP);
    } else if (landmarks.LEFT_HIP) {
      center = landmarks.LEFT_HIP;
    } else if (landmarks.RIGHT_HIP) {
      center = landmarks.RIGHT_HIP;
    } else if (landmarks.LEFT_SHOULDER && landmarks.RIGHT_SHOULDER) {
      center = midpoint(landmarks.LEFT_SHOULDER, landmarks.RIGHT_SHOULDER);
    } else {
      const pts = Object.values(landmarks);
      if (pts.length > 0) {
        const sx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
        const sy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
        center = [sx, sy];
      }
    }

    let scale = null;
    let shoulder = null;
    let hip = null;
    if (landmarks.LEFT_SHOULDER && landmarks.RIGHT_SHOULDER) {
      shoulder = midpoint(landmarks.LEFT_SHOULDER, landmarks.RIGHT_SHOULDER);
    } else if (landmarks.LEFT_SHOULDER) {
      shoulder = landmarks.LEFT_SHOULDER;
    } else if (landmarks.RIGHT_SHOULDER) {
      shoulder = landmarks.RIGHT_SHOULDER;
    }
    if (landmarks.LEFT_HIP && landmarks.RIGHT_HIP) {
      hip = midpoint(landmarks.LEFT_HIP, landmarks.RIGHT_HIP);
    } else if (landmarks.LEFT_HIP) {
      hip = landmarks.LEFT_HIP;
    } else if (landmarks.RIGHT_HIP) {
      hip = landmarks.RIGHT_HIP;
    }
    if (shoulder && hip) scale = distance(shoulder, hip);

    return { center, scale };
  }
}
