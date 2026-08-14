// Verifies the too-fast-reps-aren't-counted behavior matches between
// Python and JS: synthetic_squat_frames_fast.json, timestamped at 60fps
// (see run_python_trace_fast.py), works out to 83ms/rep, well under
// MIN_REP_DURATION_MS (200ms as of 2026-07-21), so every rep should be
// rejected in both engines identically.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { ExerciseDefinition } from '../js/exercise-definition.js';
import { GenericExercise } from '../js/generic-exercise.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const frames = JSON.parse(readFileSync(join(__dirname, 'synthetic_squat_frames_fast.json'), 'utf8'));
const pythonTrace = JSON.parse(readFileSync(join(__dirname, 'python_trace_fast.json'), 'utf8'));
const defnDict = JSON.parse(
  readFileSync(join(__dirname, '..', 'definitions', 'learned_squat.json'), 'utf8')
);
const definition = ExerciseDefinition.fromDict(defnDict);

const exercise = new GenericExercise(definition);
exercise.start();

const FPS = 60.0; // must match run_python_trace_fast.py's timescale

let mismatches = 0;
for (let i = 0; i < frames.length; i++) {
  const timestampMs = i * (1000.0 / FPS);
  exercise.analyzePose(frames[i], timestampMs);
  const expected = pythonTrace[i];
  const ok =
    exercise.state === expected.state &&
    exercise.getRepCount() === expected.rep_count &&
    exercise.rejectedFastReps === expected.rejected_fast_reps;
  if (!ok) {
    mismatches++;
    console.error(
      `MISMATCH frame ${i}: state js=${exercise.state} py=${expected.state} | ` +
        `reps js=${exercise.getRepCount()} py=${expected.rep_count} | ` +
        `rejected js=${exercise.rejectedFastReps} py=${expected.rejected_fast_reps}`
    );
  }
}

const finalReps = exercise.getRepCount();
const finalRejected = exercise.rejectedFastReps;

if (mismatches === 0 && finalReps === 0 && finalRejected > 0) {
  console.log(`FAST-REP PARITY OK — ${frames.length} frames, 0 counted (as expected), ${finalRejected} rejected as too-fast`);
  process.exit(0);
} else {
  console.error(`FAST-REP PARITY FAILED — ${mismatches} mismatches, finalReps=${finalReps} (expected 0), finalRejected=${finalRejected} (expected >0)`);
  process.exit(1);
}
