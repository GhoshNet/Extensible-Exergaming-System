// Replays the same synthetic_squat_frames.json through the JS port and
// diffs it frame-by-frame against python_trace.json (produced by the
// real desktop-app GenericExercise). Exits non-zero on any mismatch.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { ExerciseDefinition } from '../js/exercise-definition.js';
import { GenericExercise } from '../js/generic-exercise.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const frames = JSON.parse(readFileSync(join(__dirname, 'synthetic_squat_frames.json'), 'utf8'));
const pythonTrace = JSON.parse(readFileSync(join(__dirname, 'python_trace.json'), 'utf8'));
const defnDict = JSON.parse(
  readFileSync(join(__dirname, '..', 'definitions', 'learned_squat.json'), 'utf8')
);
const definition = ExerciseDefinition.fromDict(defnDict);

const exercise = new GenericExercise(definition);
exercise.start();

const FPS = 30.0; // must match run_python_trace.py's timescale

let mismatches = 0;
for (let i = 0; i < frames.length; i++) {
  const timestampMs = i * (1000.0 / FPS);
  exercise.analyzePose(frames[i], timestampMs);

  const expected = pythonTrace[i];
  const jsState = exercise.state;
  const jsReps = exercise.getRepCount();
  const jsForm = exercise.formFeedback;
  const jsKnee = exercise.signalValues.knee_angle;
  const jsRejected = exercise.rejectedFastReps;

  const stateOk = jsState === expected.state;
  const repsOk = jsReps === expected.rep_count;
  const formOk = jsForm === expected.form;
  const kneeOk =
    (jsKnee === undefined && expected.knee_angle === null) ||
    Math.abs(jsKnee - expected.knee_angle) < 1e-6;
  const rejectedOk = jsRejected === expected.rejected_fast_reps;

  if (!stateOk || !repsOk || !formOk || !kneeOk || !rejectedOk) {
    mismatches++;
    console.error(
      `MISMATCH frame ${i}: ` +
        `state js=${jsState} py=${expected.state} | ` +
        `reps js=${jsReps} py=${expected.rep_count} | ` +
        `form js=${jsForm} py=${expected.form} | ` +
        `knee js=${jsKnee} py=${expected.knee_angle} | ` +
        `rejected js=${jsRejected} py=${expected.rejected_fast_reps}`
    );
  }
}

if (mismatches === 0) {
  console.log(`PARITY OK — ${frames.length} frames, 0 mismatches, final rep count = ${exercise.getRepCount()}`);
  process.exit(0);
} else {
  console.error(`PARITY FAILED — ${mismatches} mismatched frames out of ${frames.length}`);
  process.exit(1);
}
