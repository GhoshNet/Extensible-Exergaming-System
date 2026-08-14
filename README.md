# An Extensible Pose-Based Exergaming System

Reference implementation accompanying the M.Sc. dissertation *"An Extensible Pose-Based
Exergaming System: Demonstration-Based Exercise Definition with Real-Time
Per-Body-Part Feedback"* (Tanmay Ghosh, School of Computer Science and
Statistics, Trinity College Dublin).

The system assesses home exercise from a single RGB camera. Its distinguishing
property is **extensibility**: a non-programmer can add a new exercise by
performing it once on video, with no programming, no machine-learning training
step, and no clinician reference recording. There is no per-exercise code
anywhere in the runtime — every exercise, whether hand-written or learned, is a
JSON document interpreted by one generic state machine.

---

## 1. What is in this repository

```
exergaming/
├── README.md                  this file
├── requirements.txt           Python dependencies (desktop application)
│
├── exergaming-system/         DESKTOP APPLICATION (Python 3.11 + PyQt6)
│   ├── main.py                    entry point — three-panel GUI
│   ├── video_analyzer_ui.py       standalone video-file analysis window
│   ├── tests_position_gate.py     unit tests for the position gate
│   ├── test_video.py              smoke test over a video file
│   │
│   ├── src/
│   │   ├── controller.py          per-frame pipeline; loads all definitions
│   │   ├── pose/
│   │   │   ├── detector.py            MediaPipe Pose wrapper (33 landmarks)
│   │   │   ├── analyzer.py            joint angles and distance ratios
│   │   │   ├── position_gate.py       critical-signal visibility + drift check
│   │   │   └── quality_checker.py     frame-quality warnings
│   │   ├── exercises/
│   │   │   ├── exercise_definition.py the JSON definition schema
│   │   │   ├── generic_exercise.py    THE state machine — exercise-agnostic
│   │   │   ├── base_exercise.py       shared behaviour
│   │   │   ├── squat.py / pushup.py / jumping_jack.py   hard-coded originals
│   │   │   └── definitions/           ← the 10 exercise definitions (JSON)
│   │   ├── learning/
│   │   │   ├── demonstration_capture.py  candidate signals per frame
│   │   │   └── exercise_learner.py       C1 — the learning algorithm
│   │   ├── feedback/
│   │   │   ├── form_errors.py         per-body-part error model (C2)
│   │   │   ├── visual.py              coloured skeleton overlay
│   │   │   └── diagnostic_overlay.py  live debug overlay
│   │   ├── diagnostics/session_logger.py
│   │   ├── ui/main_window.py
│   │   └── utils/  camera.py, config.py
│   │
│   ├── tools/learn_exercise.py    command-line exercise trainer
│   └── reports/                   evaluation harnesses, figure generators,
│                                  and the per-video result data behind Ch. 5
│
└── exergaming-mobile/         MOBILE PWA (in-browser, on-device)
    ├── index.html, manifest.json, service-worker.js
    ├── js/                        runtime ported line-for-line from Python
    ├── definitions/               3 definitions, byte-identical to desktop
    ├── css/, icons/
    ├── vendor/wasm/               MediaPipe Tasks Vision (WebAssembly)
    └── test/                      parity + end-to-end test suite
```

**Included as evidence:** every per-video result file behind the numbers in the
dissertation's Evaluation chapter — the triage manifest for all 858 source
videos, the 239 hand-counted ground truths, Track A detection rates, Track B/C
results for both definition variants, Track D latency, the dispersion summary,
and the pre-percentile snapshots. These are the artefacts listed in Appendix A2.

**Not included:** any video files. The 4.5 GB source corpus, the demonstration
clips, session logs and project notes are all left out — see Section 4 for how
to use your own footage instead. The corpus is two public Kaggle archives, both freely available:

- Hasyim Abdillah, *Workout/Exercises Video* (2023) —
  <https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video>
- Riccardo Riccio, *Real-Time Exercise Recognition Dataset* (2024) —
  <https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset>
  (released with arXiv:2411.11548; only its real video content was used, not
  its synthetic portion)

`reports/dataset_triage_manifest.csv` records the keep/reject decision and
reason for every one of the 858 source videos, so the corpus used here can be
reconstructed exactly from the two archives above. The evaluation harnesses in `reports/` are
included so the method can be read, but re-running them end to end needs the
corpus; the result CSVs let you check the numbers without it.

---

## 2. Requirements

- **Python 3.11** recommended. MediaPipe 0.10.14 does not publish wheels for
  every newer Python release; 3.12+ may fail to install.
- A **webcam** for live mode (a video file is enough for analysis mode).
- **Node.js 18+** only if you want to run the mobile test suite.
- macOS, Windows or Linux. Developed and evaluated on macOS 15.6.1 (Apple M1).

---

## 3. Running the desktop application

```bash
cd exergaming-system

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r ../requirements.txt

python main.py
```

On macOS the first launch will ask for camera permission. If the camera feed
stays black, grant access under *System Settings → Privacy & Security → Camera*
and restart the application.

### The three panels

| Panel | What it does |
|---|---|
| **Live Camera** | Real-time tracking from the webcam: skeleton overlay coloured by per-body-part form error, repetition counter, positioning guidance. |
| **Video Analysis** | The same pipeline over a recorded file, with playback controls and a diagnostics overlay. |
| **Training** | The authoring workflow: capture a demonstration, review the ranked signals, save a new exercise definition. |

### Camera placement

Accuracy is highest at roughly **45° to the performer at about 2 m**, which was
also the most consistent condition measured. A pure side-on (90°) view degrades
counting noticeably, and worse still when combined with distance. Make sure the
joints the exercise depends on are in frame — cropping the knees out of a squat
silences the counter entirely, by design.

---

## 4. Using your own video clips

**No video files are distributed with this repository.** The evaluation corpus
runs to several gigabytes and the demonstration clips are large, so both are
left out. Everything below works with any clip you record yourself, and the
system is designed to work on ordinary phone or webcam footage — nothing
special is required.

### Recording a clip that works

The system needs to see the joints the exercise depends on. Beyond that it is
tolerant: the evaluation corpus is ordinary internet footage, and camera motion
was measured to have no effect on counting.

| | |
|---|---|
| **Angle** | About **45° to the performer** is best — it was both the most accurate and the most consistent condition measured. Head-on also works well. Avoid a pure side-on (90°) view, which degrades counting noticeably. |
| **Distance** | About **2 m**. 5 m still works but is measurably weaker, and the worst case is side-on *and* far away together. |
| **Height** | Roughly 0.5 m from the floor, where a propped-up phone naturally sits. |
| **Framing** | Keep the *critical* joints in frame for the whole clip. Cropping the knees out of a squat silences the counter entirely — that is by design, not a bug. |
| **Lighting** | Ordinary room or window light is fine. |
| **Length** | A few repetitions is enough. Any common format works (`.mp4`, `.mov`, `.avi`). |

### Analysing a clip

```bash
cd exergaming-system
python main.py
```

Open the **Video Analysis** panel → load your file → choose the exercise
definition that matches the movement → play. The repetition counter and the
per-body-part skeleton colouring update as the video runs.

Ten definitions ship in `src/exercises/definitions/`. Pick by movement:

| Movement in your clip | Definition to select |
|---|---|
| Squat | `Learned Squat` (or `Squat` for the hand-coded original) |
| Push-up | `Learned Push-up` (or `Push-up`) |
| Jumping jack | `Jumping Jack` |
| Bicep curl, both arms | `BicepCurl` |
| Shoulder press | `ShoulderPress` |

The `(percentile)` variants are re-derived versions kept in parallel for the
controlled experiment in the dissertation; either works.

### If your exercise is not in the list

Train it from your own clip — this is the point of the system, and it takes one
command. See Section 5.

### What to expect, honestly

Across 239 hand-counted real-world videos the system averages **75.9%**, and the
distribution is bimodal: it either counts a clip exactly right or fails on it
outright, rarely in between. If a clip counts wrongly, the cause is usually one
of three named conditions — the critical joints leaving the frame, a tempo or
depth far outside what the definition was trained on, or a movement performed
one limb at a time. Section 9 covers all three.

## 5. Adding a new exercise (contribution C1)

This is the central claim of the dissertation, and it needs no code.

**Through the GUI:** open the **Training** panel and follow five steps —
select a demonstration video, run capture, review the ranked signals (untick any
that are artefacts), optionally edit the prompts and a safety limit, then save.

**From the command line:**

```bash
cd exergaming-system
python tools/learn_exercise.py --video path/to/demo.mp4 --name "My Exercise"
```

Useful flags: `--output` (destination path), `--compare` (diff against an
existing definition), `--test-video` and `--ground-truth N` (score the new
definition immediately), `--no-confirm` (skip the interactive review).

The result is a JSON file in `src/exercises/definitions/`. Drop it there and it
appears in both GUI panels immediately, and runs unchanged on the mobile app.

> **Why the review step exists.** Signal ranking can be defeated by
> two-dimensional foreshortening. In a push-up filmed from a low oblique angle
> the projected knee angle sweeps ~124° even though the knees stay straight, so
> the ranking puts `knee_angle` first — ahead of the elbow angle that actually
> defines the movement. No robust statistic recovers this, because the error is
> smooth and present in every frame. Detecting it needs exercise knowledge the
> system does not have, so a one-tick human confirmation is retained by design.

---

## 6. The exercise definition format

Ten definitions ship in `src/exercises/definitions/`: three hard-coded (squat,
push-up, jumping jack), four learned from demonstration (squat, push-up, bicep
curl, shoulder press), and three percentile-refined variants retained for the
controlled re-derivation experiment reported in the dissertation.

Each definition has five parts:

| Part | Meaning |
|---|---|
| **Signals** | Named joint angles (three landmarks each) and distance ratios. Angle signals are `bilateral`: computed on both sides and averaged, falling back to whichever side is visible. |
| **Transitions** | A state machine over IDLE / UP / DOWN. A repetition is counted on completing the designated sequence. |
| **Form evaluation** | The tracked signal's extreme during the tracking state, graded into excellent / good / fair / poor. |
| **Prompts** | Creator-editable coaching text. |
| **Safety condition** | Optional bound beyond which the system warns instead of counting. |

`learned_squat.json` is reproduced and annotated in Appendix A1 of the
dissertation. Every number in it was derived by the learner; none was
hand-edited.

---

## 7. Running the mobile PWA

A static site with no build step and no runtime dependencies. Serve the folder
over HTTPS or localhost — browsers will not grant camera access over plain HTTP.

```bash
cd exergaming-mobile
python3 -m http.server 8000
```

Then open `http://localhost:8000` on the machine, or serve it over HTTPS (or a
tunnel) to reach it from a phone. It installs as a PWA and runs fully offline
once cached; the camera, inference and all analysis stay on the device, and
nothing is uploaded.

The port is **inference-only**: definitions are trained on the desktop and
loaded here as static JSON. Verified live on a OnePlus Nord CE 5G (Snapdragon
750G, 8 GB) in Chrome at 21–22 fps.

---

## 8. Tests

```bash
# Desktop — position gate unit tests
cd exergaming-system
python tests_position_gate.py

# Mobile — desktop/mobile parity (no npm install needed)
cd exergaming-mobile
node test/parity-test.mjs            # 120-frame bilateral sequence
node test/parity-test-leftonly.mjs   # single-side visibility fallback
node test/parity-test-fast.mjs       # cool-down rejection path
node test/position-gate-test.mjs
```

The parity tests run synthetic landmark sequences through both the Python engine
and the JavaScript port and compare state, repetition count and form rating on
every frame. `test/e2e-fake-camera.mjs` drives the real app in Chromium with
video injected as a fake camera; it needs Playwright and is not required.

---

## 9. Known limitations

Stated plainly, and discussed at length in the dissertation:

- **Unilateral movement fails.** Averaging left and right encodes a bilateral
  symmetry assumption, so one-arm-at-a-time exercises undercount badly (30.0%
  on the affected clips). The definition format already permits explicit
  `left_` / `right_` signals; the learner does not yet emit them.
- **Secondary signals are inert.** The learner writes only the primary signal
  into the form-tracking list, so approving extra signals in the review step
  records them but does not change counting or feedback.
- **Thresholds inherit one demonstrator's range.** Performances far outside the
  demonstrated tempo or depth fall outside the learned envelope.
- **Demonstration hygiene is the author's responsibility.** A weight put-down at
  the end of a clip can become the learned extreme. Trim demonstrations to the
  movement itself.
- **No usability claim is made.** No user study was conducted.

> **Intended use.** This is a research prototype. It shows that, given an
> exercise definition captured from a demonstration, the system can track how
> closely a performance matches that definition. It makes no promise that any
> defined exercise is safe or appropriate for any particular person. It has not
> been clinically validated, is not a medical device, and is not a substitute
> for professional supervision.

---

## 10. Evaluation environment

The environment the reported results were produced on:

| | |
|---|---|
| Machine | Apple MacBook Air (M1, 2020), 8 GB |
| OS | macOS 15.6.1 |
| Python | 3.11.15 |
| MediaPipe | 0.10.14 (`model_complexity=1`, confidence 0.5 / 0.5) |
| OpenCV | 4.11.0 |
| NumPy | 1.26.4 |
| Webcam | built-in FaceTime HD (720p), captured at 1280×720 |
| Mobile | OnePlus Nord CE 5G, Android 13, Chrome 150 |

One caveat worth knowing if you re-run anything: repeated runs over identical
inputs are **not** always bit-identical. The large in-the-wild track reproduces
exactly, but the small controlled track can change individual repetition counts
between runs while leaving aggregates unmoved. The cause was narrowed by
elimination to the pose estimator's accelerated inference. The dissertation's
comparisons are paired so that this noise cancels.
