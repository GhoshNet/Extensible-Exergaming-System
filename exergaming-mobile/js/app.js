import { FilesetResolver, PoseLandmarker, DrawingUtils } from '../vendor/vision_bundle.mjs';
import { ExerciseDefinition, landmarksToDict } from './exercise-definition.js';
import { GenericExercise } from './generic-exercise.js';
import { PositionGate } from './position-gate.js';

// Matches the desktop app's START_COUNTDOWN_SECONDS -- rep-counting
// doesn't start the instant Start is tapped, so getting into position
// isn't counted as a false-positive rep.
const COUNTDOWN_SECONDS = 5.0;

const EXERCISES = [
  { id: 'learned_squat', label: 'Squat', file: 'definitions/learned_squat.json' },
  { id: 'learned_pushup', label: 'Push-up', file: 'definitions/learned_pushup.json' },
  { id: 'learned_bicepcurl', label: 'Bicep Curl', file: 'definitions/learned_bicepcurl.json' },
];

const $ = (id) => document.getElementById(id);
const els = {
  videoWrap: $('videoWrap'),
  video: $('video'),
  canvas: $('overlay'),
  exerciseSelect: $('exerciseSelect'),
  startBtn: $('startBtn'),
  stopBtn: $('stopBtn'),
  resetBtn: $('resetBtn'),
  flipBtn: $('flipBtn'),
  fileInput: $('fileInput'),
  repCount: $('repCount'),
  stateBadge: $('stateBadge'),
  formBadge: $('formBadge'),
  fpsBadge: $('fpsBadge'),
  loadingOverlay: $('loadingOverlay'),
  loadingText: $('loadingText'),
  sourceLabel: $('sourceLabel'),
  countdownOverlay: $('countdownOverlay'),
  countdownNumber: $('countdownNumber'),
  warningBanner: $('warningBanner'),
};

const ctx = els.canvas.getContext('2d');
const drawingUtils = new DrawingUtils(ctx);

let poseLandmarker = null;
let currentDefinition = null;
let currentExercise = null;
let positionGate = null;
let currentStream = null;
let facingMode = 'user';
let running = false;
let videoFrameCallbackId = null;
let lastFrameTimestamps = [];
let countdownActive = false;
let countdownStart = 0;
// 'camera' = live stream, 'file' = a video picked from the phone. The two
// sources have deliberately different start behaviour -- see startBtn.
let sourceMode = 'camera';
let objectUrl = null;
let currentFileName = '';
// Set when a file run starts: the position gate is calibrated on the first
// frame with a detected pose (for the camera this happens at countdown end).
let pendingCalibration = false;

function setLoading(text) {
  if (text) {
    els.loadingOverlay.style.display = 'flex';
    els.loadingText.textContent = text;
  } else {
    els.loadingOverlay.style.display = 'none';
  }
}

async function initPoseLandmarker() {
  setLoading('Loading pose model…');
  // Default: full model + CPU delegate. Verified end-to-end (real video ->
  // fake camera -> full pipeline) against ground truth on two exercises:
  // squat 7/7 reps, push-up 9/9 reps, at 22-30fps -- comfortably smooth for
  // rep counting (reps run 1-3s each). The lighter "lite" model was faster
  // (~28-31fps) but measurably LESS accurate specifically on push-up
  // (6/9, likely from more self-occlusion in its horizontal whole-body
  // pose vs squat's more upright one) -- accuracy matters more than the
  // fps headroom here, so "full" is the safer default despite the
  // pushup-specific evidence. The GPU delegate was measurably worse in the
  // same test (unstable WebGL context, undercounted reps from skipped
  // frames) -- worth re-testing on the real phone's GPU via ?delegate=gpu
  // since mobile GPUs are typically stronger than mobile CPUs for this
  // workload, unlike the sandboxed desktop environment this was verified
  // in. ?model=lite is kept available if fps becomes a real problem on
  // the actual device.
  const params = new URLSearchParams(location.search);
  const model = params.get('model') === 'lite' ? 'lite' : 'full';
  const delegate = params.get('delegate') === 'gpu' ? 'GPU' : 'CPU';

  const vision = await FilesetResolver.forVisionTasks('vendor/wasm');
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `vendor/pose_landmarker_${model}.task`,
      delegate,
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
  console.log(`[app] pose landmarker ready: model=${model} delegate=${delegate}`);
  setLoading(null);
}

async function loadExercise(entry) {
  currentDefinition = await ExerciseDefinition.fromUrl(entry.file);
  currentExercise = new GenericExercise(currentDefinition);
  positionGate = new PositionGate(currentExercise);
  countdownActive = false;
  updateHud();
  updateWarnings([]);
  updateCountdown(null);
}

function populateExerciseSelect() {
  els.exerciseSelect.innerHTML = EXERCISES.map((e) => `<option value="${e.id}">${e.label}</option>`).join('');
  els.exerciseSelect.addEventListener('change', () => {
    const entry = EXERCISES.find((e) => e.id === els.exerciseSelect.value);
    loadExercise(entry);
  });
}

function updateHud() {
  if (!currentExercise) return;
  els.repCount.textContent = currentExercise.getRepCount();
  els.stateBadge.textContent = currentExercise.state;
  els.formBadge.textContent = currentExercise.formFeedback;
  els.formBadge.dataset.form = currentExercise.formFeedback;
}

function updateCountdown(secondsRemaining) {
  if (secondsRemaining === null) {
    els.countdownOverlay.style.display = 'none';
    return;
  }
  els.countdownOverlay.style.display = 'flex';
  els.countdownNumber.textContent = Math.ceil(secondsRemaining);
}

function updateWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    els.warningBanner.style.display = 'none';
    els.warningBanner.textContent = '';
    return;
  }
  els.warningBanner.style.display = 'block';
  els.warningBanner.textContent = warnings[0]; // most severe first
}

function resizeCanvasToVideo() {
  const w = els.video.videoWidth || 640;
  const h = els.video.videoHeight || 480;
  els.canvas.width = w;
  els.canvas.height = h;
}

function drawSkeleton(landmarks) {
  ctx.save();
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (landmarks && landmarks.length) {
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: 'rgba(80, 220, 140, 0.9)',
      lineWidth: 3,
    });
    drawingUtils.drawLandmarks(landmarks, {
      color: 'rgba(255, 255, 255, 0.95)',
      fillColor: 'rgba(80, 220, 140, 0.9)',
      radius: 4,
    });
  }
  ctx.restore();
}

function trackFps() {
  const now = performance.now();
  lastFrameTimestamps.push(now);
  lastFrameTimestamps = lastFrameTimestamps.filter((t) => now - t < 1000);
  els.fpsBadge.textContent = `${lastFrameTimestamps.length} fps`;
}

function processFrame(timestampMs) {
  if (!poseLandmarker || !currentExercise) return;
  if (els.video.videoWidth === 0) return;

  const result = poseLandmarker.detectForVideo(els.video, timestampMs);
  const poseLandmarks = result.landmarks && result.landmarks[0] ? result.landmarks[0] : null;
  const poseDetected = !!poseLandmarks;

  drawSkeleton(poseLandmarks);

  const dict = poseDetected
    ? landmarksToDict(poseLandmarks, els.canvas.width, els.canvas.height)
    : {};

  if (countdownActive) {
    const remaining = COUNTDOWN_SECONDS - (timestampMs - countdownStart) / 1000;
    if (remaining <= 0) {
      countdownActive = false;
      updateCountdown(null);
      currentExercise.start();
      positionGate.reset();
      if (poseDetected) positionGate.calibrate(dict);
    } else {
      // Camera + pose detection keep running (skeleton visible, live
      // positioning feedback) but analyzePose() isn't called yet, so
      // getting into position can't be counted as a rep -- same idea as
      // the desktop app's pre-exercise countdown.
      updateCountdown(remaining);
      const gateResult = positionGate.check(els.canvas.width, els.canvas.height, dict, poseDetected, timestampMs);
      updateWarnings(gateResult.warnings);
      updateHud();
      trackFps();
      return;
    }
  }

  // File runs have no countdown, so the gate's home position is anchored on
  // the first frame that actually has a pose (the camera path does this at
  // countdown end instead).
  if (pendingCalibration && poseDetected) {
    positionGate.calibrate(dict);
    pendingCalibration = false;
  }

  if (currentExercise.isActive) {
    const gateResult = positionGate.check(els.canvas.width, els.canvas.height, dict, poseDetected, timestampMs);
    updateWarnings(gateResult.warnings);
    // A single dropped-detection frame is NOT gated (see PositionGate's
    // grace period) -- analyzePose() still runs and its own None-guards
    // handle a momentarily-missing signal gracefully. Only a SUSTAINED bad
    // frame (person genuinely out of position, not a blip) actually skips
    // it, so a workout doesn't visibly stall over one bad frame.
    if (poseDetected && !gateResult.gated) {
      currentExercise.analyzePose(dict, timestampMs);
    }
  } else {
    updateWarnings([]);
  }

  updateHud();
  trackFps();
}

function frameLoop() {
  if (!running) return;
  const now = performance.now();
  processFrame(now);
  if (els.video.requestVideoFrameCallback) {
    videoFrameCallbackId = els.video.requestVideoFrameCallback(() => frameLoop());
  } else {
    videoFrameCallbackId = requestAnimationFrame(frameLoop);
  }
}

function stopFrameLoop() {
  running = false;
  if (videoFrameCallbackId != null) {
    if (els.video.cancelVideoFrameCallback) els.video.cancelVideoFrameCallback(videoFrameCallbackId);
    else cancelAnimationFrame(videoFrameCallbackId);
  }
}

async function startCamera() {
  stopCamera();
  sourceMode = 'camera';
  els.sourceLabel.textContent = 'Live camera';
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  currentStream = stream;
  els.video.srcObject = stream;
  els.video.src = '';
  els.videoWrap.classList.toggle('mirrored', facingMode === 'user');
  await els.video.play();
  resizeCanvasToVideo();
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
  }
  els.video.srcObject = null;
}

async function loadVideoFile(file) {
  stopFrameLoop();
  stopCamera();
  sourceMode = 'file';

  // Selecting a file must not begin anything: clear any run in progress and
  // leave the clip parked on its first frame until Start is tapped.
  countdownActive = false;
  pendingCalibration = false;
  updateCountdown(null);
  updateWarnings([]);
  if (currentExercise) {
    currentExercise.stop();
    currentExercise.reset();
    updateHud();
  }
  if (positionGate) positionGate.reset();

  currentFileName = file.name;
  els.sourceLabel.textContent = currentFileName;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  els.video.srcObject = null;
  els.video.src = objectUrl;
  // Not looped: a looping clip would silently re-count the same reps forever.
  // The 'ended' handler stops the run and leaves the final count on screen.
  els.video.loop = false;
  els.videoWrap.classList.remove('mirrored');

  // Decode the first frame for preview, but stay paused.
  await new Promise((resolve, reject) => {
    els.video.onloadeddata = resolve;
    els.video.onerror = () => reject(new Error('could not open that video'));
  });
  els.video.pause();
  els.video.currentTime = 0;
  resizeCanvasToVideo();
  drawSkeleton(null);
}

els.startBtn.addEventListener('click', async () => {
  if (!currentExercise) return;
  currentExercise.reset();
  positionGate.reset();
  updateWarnings([]);
  updateHud();

  if (sourceMode === 'file') {
    // A pre-recorded clip needs no countdown: there is nobody getting into
    // position in front of the phone, and counting down would silently skip
    // the first seconds of the video. This matches the desktop Video Analysis
    // panel, which analyses from the first frame. Rewind so Start always
    // analyses the whole clip, however much of it was previewed.
    countdownActive = false;
    updateCountdown(null);
    els.sourceLabel.textContent = currentFileName;   // clear any "— finished"
    els.video.currentTime = 0;
    try {
      await els.video.play();
    } catch (err) {
      els.sourceLabel.textContent = `Could not play video: ${err.message}`;
      return;
    }
    currentExercise.start();
    pendingCalibration = true;
  } else {
    countdownActive = true;
    countdownStart = performance.now();
  }

  running = true;
  frameLoop();
});

els.stopBtn.addEventListener('click', () => {
  stopFrameLoop();
  countdownActive = false;
  pendingCalibration = false;
  updateCountdown(null);
  updateWarnings([]);
  if (sourceMode === 'file') els.video.pause();
  if (currentExercise) currentExercise.stop();
  if (positionGate) positionGate.reset();
});

els.resetBtn.addEventListener('click', () => {
  if (currentExercise) {
    currentExercise.reset();
    updateHud();
  }
  if (positionGate) positionGate.reset();
  // Rewind a finished/stopped clip so Reset really means "start over".
  if (sourceMode === 'file' && !running) els.video.currentTime = 0;
});

els.flipBtn.addEventListener('click', async () => {
  // While a video file is loaded this button is the way back to the live
  // camera -- otherwise picking a clip would strand you there until reload.
  if (sourceMode === 'file') {
    stopFrameLoop();
    countdownActive = false;
    pendingCalibration = false;
    updateCountdown(null);
    if (currentExercise) currentExercise.stop();
    await startCamera();
    return;
  }
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  if (currentStream) await startCamera();
});

els.fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    loadVideoFile(file).catch((err) => {
      els.sourceLabel.textContent = `Error: ${err.message}`;
    });
  }
  // Clear it so picking the SAME file again still fires a change event.
  e.target.value = '';
});

// A file run ends when the clip does -- stop cleanly and leave the count up.
els.video.addEventListener('ended', () => {
  if (sourceMode !== 'file' || !running) return;
  stopFrameLoop();
  pendingCalibration = false;
  updateWarnings([]);
  if (currentExercise) currentExercise.stop();
  updateHud();
  els.sourceLabel.textContent = `${els.sourceLabel.textContent} — finished`;
});

async function boot() {
  populateExerciseSelect();
  await loadExercise(EXERCISES[0]);
  try {
    await initPoseLandmarker();
    await startCamera();
  } catch (err) {
    setLoading(null);
    console.error(err);
    els.sourceLabel.textContent = `Error: ${err.message}`;
  }

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('service-worker.js').catch((e) => console.warn('SW registration failed', e));
  }
}

boot();
