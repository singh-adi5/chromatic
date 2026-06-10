# Roadmap

This is a proof of concept. The list below is what would need to be true
for it to be considered a production system. Items are ordered roughly by
the value they would add to a security-engineering review, not by
implementation difficulty.

## 1. Public-dataset benchmarks

**What's there now.** The pipeline is verified to be functionally correct
on synthesised inputs (the three scenarios in `demo/attack_scenarios.py`)
and on a single still image. There are no equal-error-rate, AUROC, or
ROC-curve numbers against any public dataset.

**What's needed.**

- Evaluation harness in `scripts/eval_dataset.py` that walks a labelled
  directory of clips, runs each through the detector, and emits per-clip
  verdicts plus aggregate metrics (TPR, FPR, EER, AUROC) as JSON and a
  matplotlib PDF.
- Reproducible numbers on at least one of:
  - **FaceForensics++** (Rössler et al., 2019) - the most-cited deepfake
    benchmark, covers four manipulation families.
  - **Celeb-DF v2** (Li et al., 2020) - higher-quality deepfakes, lower
    detection accuracy across the literature.
  - **DFDC** (Dolhansky et al., 2020) - the largest, most diverse set.

Each of those datasets requires accepting an EULA on the dataset owner's
site. This repository does not redistribute them and will not.

**Why it matters.** Any reviewer past the first 90 seconds will ask "what's
your EER on Celeb-DF v2?". Today the honest answer is "I haven't run that
yet". That answer is fine for a POC. It's not fine for v1.

## 2. Trained ML classifier layer

**What's there now.** The fusion engine combines five hand-engineered
signals through a linear weighted sum. The architecture document describes
a planned sixth layer - a learned anomaly detector based on a Vision
Transformer plus a PatchCore-style memory bank - but that layer does not
exist in code.

**What's needed.**

- Training pipeline using DINOv2 ViT-B/14 features over forehead/cheek
  patches, with a PatchCore coreset memory bank built from a clean
  reference set.
- Integration point in `src/chromatic/core/fusion.py` to accept an
  anomaly score as a sixth layer.
- Threshold calibration against a held-out set so the ML output is
  comparable in scale to the existing layers.

The signal-processing layers will remain. They are explainable in a way
that a learned model on its own is not, and that matters for the kind of
regulated-industry deployment this system is designed for.

## 3. Real-camera validation

**What's there now.** The live dashboard (`demo/live_dashboard.py`) has
been verified end-to-end against a synthesised 90-frame video clip. The
code path that opens a real webcam (`cv2.VideoCapture(0)`) has not been
exercised in the development environment.

**What's needed.**

- Validation against at least three distinct hardware configurations
  (e.g. a laptop integrated camera, a USB webcam, a phone screen
  mirrored through OBS) to catch driver-specific edge cases.
- A short benchmarks document recording observed throughput, false
  positives over a 10-minute clean run, and behaviour under degenerate
  lighting.

## 4. Operational hardening (after the above)

These are smaller items that would matter for a v1, not a POC.

- mypy strict mode (currently `continue-on-error` in CI).
- Distributed rate-limiter backend (current implementation is per-process).
- Audit log shipper (the audit emitter writes structured JSON, but
  there's no documented Fluent Bit / Vector pipeline).
- Helm chart for Kubernetes deployment (the Docker image is hardened,
  but there's no manifest set yet).
- Multi-architecture image builds.

## Out of scope

The following are intentional non-goals for this codebase. They are listed
to make the boundary explicit.

- **Web/mobile SDKs.** Chromatic is a server-side detector. Client-side
  capture is the caller's responsibility.
- **Identity matching / face recognition.** This system answers "is this
  a real, live human?", not "is this Aditya Singh?". Combining liveness
  with identity matching is a separate component.
- **Generative-model fingerprinting.** Methods that try to identify which
  specific generator produced a deepfake (e.g. StyleGAN3 vs. Stable
  Diffusion) are a different research line.
