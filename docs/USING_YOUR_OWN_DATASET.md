# Using your own data

This document covers four ways to run Chromatic against material you bring
yourself: a single image, a video file, a live camera, or a folder of
images for batch evaluation. It also explains the system's input
assumptions so you can tell in advance whether your data will work.

## What the pipeline expects

The detector was designed around webcam-style face capture. It will degrade
gracefully outside that envelope, but it will degrade.

| Property | Expected range | What happens outside it |
|---|---|---|
| Face size in frame | 80% of the frame (selfie) down to ~15% | Below 15%, MediaPipe may fail to land 468 stable landmarks. The detector will raise `FaceNotFoundError`. |
| Head pose | Roughly frontal (yaw and pitch within ±20°) | Geometry layer score drops, fusion may reject even a real human. This is intentional. |
| Resolution | Minimum 224×224 face crop after detection | Texture and PRNU layers lose discriminative power on tiny crops. |
| Frame rate | 30 fps assumed for rPPG band-pass and EAR history | At 15 fps the rPPG layer still works but takes twice as long to fill the 150-frame window. |
| Lighting | Diffuse front-light, no hard backlight | Hard backlight blows out the forehead ROI, breaking rPPG. |
| Duration (video / camera) | At least 5 seconds | The rPPG window is 150 frames. PRNU calibration is 30 frames. Before both are full, the verdict is conservative. |

A single still image is a degenerate case for this system. The detector
will run, but four of the five layers (rPPG, motion, the temporal half of
PRNU, blink detection) depend on time and will return zero. This is why
the diagnostic panel for a still image typically returns REJECT - and
that is the correct answer, because a still image is not a live human.

## 1. Single image

```bash
python demo/tech_demo.py --image path/to/face.jpg --output diag.png
```

Produces a publication-quality diagnostic panel. Useful for showing what
the pipeline measures, less useful for actually classifying liveness
because the temporal layers have no signal to work with.

## 2. Video file

```bash
python demo/live_dashboard.py --replay path/to/clip.mp4
```

The dashboard treats the file as a camera stream. It will run through the
entire file in real time. Add `--record output.mp4` to also save the
dashboard view as a video.

## 3. Live camera

```bash
python demo/live_dashboard.py                  # default camera (index 0)
python demo/live_dashboard.py --camera 1       # alternate device
```

Keyboard:

- `q` quit
- `s` save the current dashboard frame
- `r` reset detector state (after camera angle or lighting changes)
- `SPACE` pause / resume

If the window does not open, on Linux check that your user is in the
`video` group; on macOS grant Terminal camera permission in System
Settings. On Windows make sure no other application is holding the
camera.

## 4. Batch evaluation on a folder of clips

The repository ships with a small evaluation harness so you can point the
detector at a labelled directory and get per-clip verdicts back as CSV.

```bash
python scripts/eval_folder.py \
    --input /path/to/clips \
    --output results.csv \
    --label-from parent     # use the parent directory name as the label
```

Expected directory layout when using `--label-from parent`:

```
clips/
├── live/
│   ├── alice_01.mp4
│   ├── bob_05.mp4
│   └── ...
└── spoof/
    ├── photo_attack_01.mp4
    ├── replay_07.mp4
    └── ...
```

The script reports per-clip verdict, sustained confidence, the
contributing layer scores, and aggregate accuracy / FPR / FNR at the end.
It does not compute ROC curves yet - that is on the roadmap.

## Using a public deepfake benchmark

We do not redistribute deepfake datasets, because their licences forbid it.
If you have your own copy of one of the following, the layout below is
what `eval_folder.py` expects.

**FaceForensics++** (Rössler et al., 2019). Download via the form at
<https://github.com/ondyari/FaceForensics>. The "Deepfakes" subset is a
reasonable starting point.

```
faceforensics/
├── live/      <- contents of `original_sequences/youtube/c23/videos/`
└── spoof/     <- contents of `manipulated_sequences/Deepfakes/c23/videos/`
```

**Celeb-DF v2** (Li et al., 2020). Request form at
<https://github.com/yuezunli/celeb-deepfakeforensics>.

```
celeb-df/
├── live/      <- Celeb-real-v2
└── spoof/     <- Celeb-synthesis-v2
```

**DFDC** (Dolhansky et al., 2020). Released on Kaggle under a
research-only licence.

```
dfdc/
├── live/      <- frames labelled REAL in metadata.json
└── spoof/     <- frames labelled FAKE
```

For DFDC you will need a short pre-processing step to split clips by
their label - the dataset ships everything in one tree with a JSON
manifest. See `scripts/dfdc_split.py` (not yet written - tracked in
ROADMAP).

## Tuning the decision threshold

Every layer score is in `[0, 1]`. The fused score is a weighted sum,
also in `[0, 1]`. The decision threshold is environment-variable
controlled:

```bash
export CHROMATIC_DECISION_THRESHOLD=0.65     # default
export CHROMATIC_SUSTAINED_FRAMES=30          # how many frames must agree
```

- **Lower threshold** → fewer false rejections of real humans, more
  successful spoofs (lower FNR, higher FPR).
- **Higher threshold** → tighter security, more friction for real users
  (higher FNR, lower FPR).

For an account-opening flow at a bank, you almost certainly want a
threshold above 0.7. For an unlock flow you might run as low as 0.55.
There is no universally correct value; it depends on your tolerance for
each error type, which depends on what the downstream action is.

## When the detector says REJECT and you disagree

The `reasons` field in the verdict explains which layers vetoed.
Common patterns:

- `low geometry score` - the face is not frontal enough. Move closer
  to the camera, look straight at it.
- `low rppg score` - either lighting is uneven across the forehead, or
  the camera frame rate is too low. Check `diag.rppg_progress` - if
  it's not at 1.0 the window simply isn't full yet.
- `low motion score` - no head movement. A real human is never
  perfectly still; if your subject was, ask them to look around briefly.
- `low texture score` - blur or a moiré pattern was detected. If you're
  testing through a phone screen pointed at the camera, this is the
  layer doing its job.

The `attack_scenarios.png` panel in this repository is the most useful
reference: it shows what the layer scores look like for each of the
three canonical attack classes.
