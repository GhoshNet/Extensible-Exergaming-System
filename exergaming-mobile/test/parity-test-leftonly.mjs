// Same as parity-test.mjs but with all RIGHT_* landmarks stripped, to
// exercise the bilateral single-side fallback path (real camera footage
// frequently only sees one side clearly).
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { ExerciseDefinition } from '../js/exercise-definition.js';
import { GenericExercise } from '../js/generic-exercise.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const frames = JSON.parse(readFileSync(join(__dirname, 'synthetic_squat_frames_leftonly.json'), 'utf8'));
const pythonTrace = JSON.parse(readFileSync(join(__dirname, 'python_trace_leftonly.json'), 'utf8'));
const defnDict = JSON.parse(
  readFileSync(join(__dirname, '..', 'definitions', 'learned_squat.json'), 'utf8')
);
const definition = ExerciseDefinition.fromDict(defnDict);

const exercise = new GenericExercise(definition);
exercise.start();

const FPS = 30.0; // must match run_python_trace_leftonly.py's timescale

let mismatches = 0;
for (let i = 0; i < frames.length; i++) {
  const timestampMs = i * (1000.0 / FPS);
  exercise.analyzePose(frames[i], timestampMs);
  const expected = pythonTrace[i];
  const ok =
    exercise.state === expected.state &&
    exercise.getRepCount() === expected.rep_count &&
    exercise.formFeedback === expected.form &&
    exercise.lastSideUsed === expected.side_used &&
    exercise.rejectedFastReps === expected.rejected_fast_reps;
  if (!ok) {
    mismatches++;
    console.error(
      `MISMATCH frame ${i}: state js=${exercise.state} py=${expected.state} | ` +
        `reps js=${exercise.getRepCount()} py=${expected.rep_count} | ` +
        `side js=${exercise.lastSideUsed} py=${expected.side_used}`
    );
  }
}

if (mismatches === 0) {
  console.log(`LEFT-ONLY PARITY OK — ${frames.length} frames, final rep count = ${exercise.getRepCount()}`);
  process.exit(0);
} else {
  console.error(`LEFT-ONLY PARITY FAILED — ${mismatches} mismatches`);
  process.exit(1);
}
