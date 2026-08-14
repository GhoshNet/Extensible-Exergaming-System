// Direct unit tests of PositionGate against the exact scenarios discussed:
// debounce (a brief blip shouldn't gate), exercise-specific critical-signal
// visibility (bicep curl shouldn't require lower-body landmarks, squat
// should require knee/ankle), and position drift.
import { ExerciseDefinition } from '../js/exercise-definition.js';
import { GenericExercise } from '../js/generic-exercise.js';
import { PositionGate } from '../js/position-gate.js';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function loadDefinition(name) {
  const d = JSON.parse(readFileSync(join(__dirname, '..', 'definitions', name), 'utf8'));
  return ExerciseDefinition.fromDict(d);
}

// Full, well-in-frame landmark set (squat standing position)
function fullSquatLandmarks() {
  return {
    LEFT_HIP: [300, 300], RIGHT_HIP: [340, 300],
    LEFT_KNEE: [300, 450], RIGHT_KNEE: [340, 450],
    LEFT_ANKLE: [300, 600], RIGHT_ANKLE: [340, 600],
    LEFT_SHOULDER: [300, 150], RIGHT_SHOULDER: [340, 150],
    LEFT_ELBOW: [280, 250], RIGHT_ELBOW: [360, 250],
    LEFT_WRIST: [270, 350], RIGHT_WRIST: [370, 350],
    LEFT_FOOT_INDEX: [300, 630], RIGHT_FOOT_INDEX: [340, 630],
  };
}

function bicepCurlLandmarks({ includeLower = true } = {}) {
  const lm = {
    LEFT_HIP: [300, 300], RIGHT_HIP: [340, 300],
    LEFT_SHOULDER: [300, 150], RIGHT_SHOULDER: [340, 150],
    LEFT_ELBOW: [280, 250], RIGHT_ELBOW: [360, 250],
    LEFT_WRIST: [270, 350], RIGHT_WRIST: [370, 350],
  };
  if (includeLower) {
    lm.LEFT_KNEE = [300, 450]; lm.RIGHT_KNEE = [340, 450];
    lm.LEFT_ANKLE = [300, 600]; lm.RIGHT_ANKLE = [340, 600];
  }
  return lm;
}

let failures = 0;
function check(name, cond) {
  if (cond) {
    console.log(`  ok - ${name}`);
  } else {
    failures++;
    console.error(`  FAIL - ${name}`);
  }
}

// ---------------------------------------------------------------
console.log('1. Debounce: a single bad frame should NOT gate');
{
  const exercise = new GenericExercise(loadDefinition('learned_squat.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  gate.calibrate(fullSquatLandmarks());

  const r1 = gate.check(720, 1280, fullSquatLandmarks(), true, 0);
  check('frame 0 (good) not gated', r1.gated === false);

  // one dropped-detection frame, 50ms later -- well within the 400ms grace period
  const r2 = gate.check(720, 1280, {}, false, 50);
  check('frame 1 (single blip) not gated', r2.gated === false);
  check('frame 1 still reports a warning (for UI) even though not gated', r2.warnings.length > 0);

  // back to good immediately after
  const r3 = gate.check(720, 1280, fullSquatLandmarks(), true, 100);
  check('frame 2 (recovered) not gated, warnings clear', r3.gated === false && r3.warnings.length === 0);
}

console.log('2. Sustained failure (>400ms) SHOULD gate');
{
  const exercise = new GenericExercise(loadDefinition('learned_squat.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  gate.calibrate(fullSquatLandmarks());

  gate.check(720, 1280, {}, false, 0);
  gate.check(720, 1280, {}, false, 200);
  const r = gate.check(720, 1280, {}, false, 450); // 450ms of sustained "no person"
  check('sustained no-person gates after grace period', r.gated === true);
  check('warning mentions no person detected', r.warnings.some((w) => w.includes('No person detected')));
}

console.log('3. Exercise-specific critical landmarks: bicep curl should NOT require lower body');
{
  const exercise = new GenericExercise(loadDefinition('learned_bicepcurl.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  gate.calibrate(bicepCurlLandmarks({ includeLower: true }));

  // lower body missing, but arms fully visible -- should NOT gate even after 500ms
  gate.check(720, 1280, bicepCurlLandmarks({ includeLower: false }), true, 0);
  const r = gate.check(720, 1280, bicepCurlLandmarks({ includeLower: false }), true, 500);
  check('bicep curl with no lower body visible is NOT gated', r.gated === false);
}

console.log('4. Exercise-specific critical landmarks: bicep curl SHOULD require elbow/wrist');
{
  const exercise = new GenericExercise(loadDefinition('learned_bicepcurl.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  const full = bicepCurlLandmarks({ includeLower: true });
  gate.calibrate(full);

  const noArms = { LEFT_HIP: full.LEFT_HIP, RIGHT_HIP: full.RIGHT_HIP,
                    LEFT_SHOULDER: full.LEFT_SHOULDER, RIGHT_SHOULDER: full.RIGHT_SHOULDER };
  gate.check(720, 1280, noArms, true, 0);
  const r = gate.check(720, 1280, noArms, true, 500);
  check('bicep curl with elbows/wrists missing IS gated', r.gated === true);
  check('warning mentions elbows/wrists', r.warnings.some((w) => /elbow|wrist/i.test(w)));
}

console.log('5. Squat SHOULD require knees/ankles (not just upper body)');
{
  const exercise = new GenericExercise(loadDefinition('learned_squat.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  const full = fullSquatLandmarks();
  gate.calibrate(full);

  const upperOnly = { LEFT_HIP: full.LEFT_HIP, RIGHT_HIP: full.RIGHT_HIP,
                       LEFT_SHOULDER: full.LEFT_SHOULDER, RIGHT_SHOULDER: full.RIGHT_SHOULDER,
                       LEFT_ELBOW: full.LEFT_ELBOW, RIGHT_ELBOW: full.RIGHT_ELBOW,
                       LEFT_WRIST: full.LEFT_WRIST, RIGHT_WRIST: full.RIGHT_WRIST };
  gate.check(720, 1280, upperOnly, true, 0);
  const r = gate.check(720, 1280, upperOnly, true, 500);
  check('squat with knees/ankles missing IS gated', r.gated === true);
  check('warning mentions knees/ankles', r.warnings.some((w) => /knee|ankle/i.test(w)));
}

console.log('6. Position drift: moving far from the calibrated anchor gates');
{
  const exercise = new GenericExercise(loadDefinition('learned_squat.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  gate.calibrate(fullSquatLandmarks());

  // shift everything 500px sideways -- torso scale here is ~150px, so this
  // is well beyond the 1.6x tolerance
  const shift = (lm) => Object.fromEntries(Object.entries(lm).map(([k, [x, y]]) => [k, [x + 500, y]]));
  const drifted = shift(fullSquatLandmarks());

  gate.check(720, 1280, drifted, true, 0);
  const r = gate.check(720, 1280, drifted, true, 500);
  check('drifted position IS gated', r.gated === true);
  check('warning mentions returning to position', r.warnings.some((w) => w.includes('Return to your starting position')));
}

console.log('7. Position drift: small natural movement within tolerance does NOT gate');
{
  const exercise = new GenericExercise(loadDefinition('learned_squat.json'));
  exercise.start();
  const gate = new PositionGate(exercise);
  gate.calibrate(fullSquatLandmarks());

  // shift by 20px -- small natural sway, well within tolerance
  const shift = (lm) => Object.fromEntries(Object.entries(lm).map(([k, [x, y]]) => [k, [x + 20, y]]));
  const slight = shift(fullSquatLandmarks());

  gate.check(720, 1280, slight, true, 0);
  const r = gate.check(720, 1280, slight, true, 500);
  check('small natural movement is NOT gated', r.gated === false);
}

console.log(`\n${failures === 0 ? 'ALL POSITION-GATE TESTS PASSED' : failures + ' TEST(S) FAILED'}`);
process.exit(failures === 0 ? 0 : 1);
