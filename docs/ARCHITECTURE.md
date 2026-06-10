# Architecture

> Audience: engineers, security architects, and anyone reviewing the design
> before approving deployment.

## 1. Goals and non-goals

### Goals

1. **Defence in depth.** No single layer is allowed to be the deciding factor;
   compromising the system requires defeating multiple orthogonal signals.
2. **Explainable verdicts.** Every accept/reject decision is accompanied by
   per-layer scores and human-readable reasons. This is a hard requirement in
   regulated industries (fintech, healthcare).
3. **Edge-deployable.** End-to-end latency under ~30 ms per frame on a
   commodity laptop CPU, so the system can run client-side or in a small
   container without specialised hardware.
4. **No biometric persistence.** Frames flow through memory and are discarded;
   no per-user templates are stored.
5. **Audit-friendly.** Structured logs, deterministic scoring, and integrity-
   checked model artefacts.

### Non-goals

- **Not** a face-recognition or identity-matching system. Liveness ≠ identity.
- **Not** an OS-level hardening kit. Operators are still responsible for the
  perimeter (TLS, secrets, network policy).
- **Not** a one-size-fits-all anti-fraud product. Liveness is a control; it
  belongs inside a broader fraud-decisioning stack, not as a replacement.

## 2. Functional requirements

| ID  | Requirement |
| --- | --- |
| F1  | Accept BGR frames as either NumPy arrays (in-process) or PNG/JPEG (over HTTP). |
| F2  | Return a verdict object containing `is_live: bool`, `confidence: float`, `layer_scores`, and `reasons: list[str]`. |
| F3  | Support both single-frame batch inference and streaming (frame-by-frame). |
| F4  | Produce per-frame diagnostics for the demo and audit pipelines. |
| F5  | Operate without internet access at runtime (models loaded from disk). |

## 3. Non-functional requirements

| ID   | Requirement | Target |
| ---- | --- | --- |
| NF1  | End-to-end latency (CPU, 720p frame) | ≤ 30 ms |
| NF2  | Memory footprint at steady state | ≤ 350 MiB |
| NF3  | Throughput, single worker | ≥ 30 fps |
| NF4  | Cold start (model load) | ≤ 5 s |
| NF5  | Detection rate on FaceForensics++ (in-house holdout) | ≥ 0.95 |
| NF6  | False-accept rate on attack-scenarios suite | ≤ 0.05 |
| NF7  | Available CI feedback time | ≤ 5 min |

## 4. High-level design

```
                          ┌─────────────────────────┐
   client (browser/SDK) ─▶│   FastAPI / WebSocket    │
                          │   /detect, /ws/stream    │
                          └────────────┬─────────────┘
                                       │  validated frame
                                       ▼
                          ┌─────────────────────────┐
                          │   FrameValidator         │ ← OWASP ASVS 5.1
                          │  (shape, dtype, size,    │
                          │   brightness bounds)     │
                          └────────────┬─────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │   LivenessDetector       │
                          │                          │
                          │   ┌────────────────────┐ │
                          │   │ FaceAnalyzer       │ │ MediaPipe FaceLandmarker
                          │   │  (468 landmarks,   │ │ Apache 2.0
                          │   │   ROI masks, pose) │ │
                          │   └─────────┬──────────┘ │
                          │             │            │
                          │   ┌─────────┴──────────┐ │
                          │   │   per-layer        │ │
                          │   │   analysis         │ │
                          │   │                    │ │
                          │   │  ┌──────────────┐  │ │
                          │   │  │ PRNU         │  │ │  sensor forensics
                          │   │  │ rPPG (CHROM) │  │ │  pulse extraction
                          │   │  │ Texture/FFT  │  │ │  moire / Laplacian
                          │   │  │ Geometry     │  │ │  pose / EAR
                          │   │  │ Motion       │  │ │  optical flow + blink
                          │   │  └──────┬───────┘  │ │
                          │   │         │          │ │
                          │   │  ┌──────┴───────┐  │ │
                          │   │  │ FusionEngine │  │ │  linear weighted
                          │   │  └──────┬───────┘  │ │  sustained-frames check
                          │   └─────────┼──────────┘ │
                          │             ▼            │
                          │     FrameDiagnostics     │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────┴─────────────┐
                          ▼                          ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │  HTTP/WS response    │    │  Audit logger (JSON)│
              │  (verdict + scores)  │    │  → SIEM / S3        │
              └─────────────────────┘    └─────────────────────┘
```

The boxes correspond directly to Python modules under
`src/chromatic/core/` (or `security/` for the validator and audit logger).

## 5. Detection layers

Each layer is built around a physical or biological principle that an attacker
must defeat. We chose these specific layers because their failure modes are
**orthogonal**: a successful attack against one is unlikely to also defeat
the others.

### 5.1 Hardware forensics (PRNU)

- **Implementation:** `core/prnu.py`
- **Signal:** the high-frequency residual of each frame (frame minus its
  Gaussian-smoothed version), summarised by `noise_std`, Fisher-excess
  `kurtosis`, and Pearson correlation against a calibration fingerprint built
  from the first ~30 frames.
- **What it catches:** virtual cameras (e.g., OBS, ManyCam) producing
  quantisation-flat output; screen replays whose noise is dominated by the
  secondary display chain.
- **Failure modes:** users with unusual genuine sensors (uncommon mobile
  cameras) may legitimately produce out-of-band kurtosis values. The
  configurable kurtosis band is deliberately wide.

### 5.2 Remote photoplethysmography (rPPG)

- **Implementation:** `core/rppg.py` (CHROM, De Haan & Jeanne 2013)
- **Signal:** RGB time series averaged over a forehead ROI, projected into
  chrominance space, band-passed to [0.7, 3.5] Hz, then dominant-frequency
  estimated by FFT and confidence scored by in-band SNR.
- **What it catches:** anything that doesn't actually have a pulse — printed
  photos, video replays of someone else, most generative deepfakes (which do
  not synthesise a coherent rPPG signal).
- **Failure modes:** legitimate users wearing heavy makeup, in extreme
  lighting, or with non-frontal pose may produce low-SNR pulses. The
  sustained-frames check tolerates a few weak frames.

### 5.3 Texture (Laplacian + FFT moire)

- **Implementation:** `core/texture.py`
- **Signal:** variance of the Laplacian (high-frequency content) and a moire
  score computed from concentration of 2-D FFT energy in narrow angular bands
  inside a mid-frequency annulus.
- **What it catches:** GAN outputs that are slightly over-smoothed, and
  re-photographed phone or monitor screens (whose subpixel grid yields
  characteristic FFT peaks).
- **Failure modes:** very-high-resolution genuine cameras with intense
  in-camera sharpening can appear "too sharp"; this is detected and weighted
  accordingly (`_saturating_score`).

### 5.4 Geometry (MediaPipe + solvePnP)

- **Implementation:** `core/face_analyzer.py`
- **Signal:** 468-point face mesh; 6 canonical landmarks fed into OpenCV
  `solvePnP` to recover (yaw, pitch, roll). The layer score penalises
  extreme pose values that are inconsistent with a cooperative subject.
- **What it catches:** paper masks held at unusual angles; faces inset into
  a 3-D scene that doesn't match the implied geometry.
- **Failure modes:** the head-pose estimate uses approximate intrinsics and
  is best-treated as a *category* (frontal vs. profile), not a precise angle.

### 5.5 Motion (Farnebäck flow + EAR blink)

- **Implementation:** `core/motion.py`
- **Signal:** dense optical flow magnitude over the face crop, plus Eye
  Aspect Ratio (Soukupová & Čech 2016) tracked across a rolling history.
- **What it catches:** static photos (no motion at all), looped videos
  (motion is too periodic), and mask attacks where the only motion is
  rigid bulk translation.
- **Failure modes:** the EAR-based blink detector needs a few seconds of
  history; very short captures will lack a blink even from a real user.

## 6. Fusion

We use a **linear weighted sum** with a **sustained-frames** check.

Why not a learned ensemble?

- **Explainability.** A bank's risk team must be able to read the rejection
  reason. "Hardware score 0.18 — likely virtual camera" is auditable.
  "Output of a CatBoost model = 0.42" is not.
- **No labelled fusion data.** We trust each layer's calibrated thresholds
  individually; we do not have a large labelled dataset of *fused* outputs
  to train an ensemble without leaking domain quirks.
- **Temporal stability is easier reason about.** "Pass ratio in the last 30
  frames must exceed 80 %" is a one-line rule. With an ensemble the same
  guarantee involves the model's calibration plus a sliding window plus
  threshold tuning.

When the system grows, we expect to revisit this — see *§ 10 What I'd
revisit*.

## 7. Data model and contracts

The public Python API is the `LivenessDetector.process_frame` method, which
returns a `FrameDiagnostics`:

```python
FrameDiagnostics(
    verdict: LivenessVerdict,   # is_live, confidence, layer_scores, reasons
    face: FaceAnalysis | None,
    pulse: PulseEstimate | None,
    prnu: PRNUMetrics | None,
    texture: TextureMetrics | None,
    motion: MotionMetrics | None,
    calibration_progress: float,
    rppg_progress: float,
    error: str | None,
)
```

`LivenessVerdict` is the public contract — anything else in `FrameDiagnostics`
may change between minor versions. The REST API (see [API.md](API.md)) only
exposes the verdict and the layer scores.

## 8. Storage and state

- **Persistent storage:** none. No database is required.
- **In-memory state:** PRNU calibration buffer (`O(N)` where `N` = 30 by
  default), rPPG sliding window (`O(150)` floats), motion EAR history
  (`O(90)` floats), and the MediaPipe model graph. Steady-state memory is
  ~285 MiB.
- **Cross-request state:** none, unless the API places multiple frames into
  the same session via a stateful WebSocket. The detector is per-session.

## 9. Failure handling

| Failure | Behaviour |
| --- | --- |
| Frame fails validation | `ValidationError` raised; HTTP 400 from the API; audit `INPUT_REJECTED`. |
| No face detected | Pipeline returns a *degraded* `FrameDiagnostics` with `verdict.is_live = False` and a "no face" reason. The session continues — the next frame may succeed. |
| MediaPipe internal error | `DetectionError` raised; the session terminates and the client is asked to reopen it. |
| PRNU not yet calibrated | First 30 frames produce `fingerprint_correlation = NaN`; sustained-verdict cannot pass until calibration completes. |
| rPPG buffer not yet full | `pulse = None`; the rPPG layer score is held at a neutral 0.5 to avoid biasing the verdict during warm-up. |
| Rate limit exceeded | `RateLimitExceededError`; HTTP 429; audit `RATE_LIMIT_HIT`. |

## 10. What I'd revisit as the system grows

This section is intentionally honest about trade-offs we made for v0.1:

1. **Fusion.** If we gather a large labelled dataset, a calibrated logistic
   regression over the layer scores (with monotonicity constraints to keep
   it auditable) would likely improve discrimination without sacrificing
   explainability. Gradient-boosted trees would not — they lose the
   monotonic interpretation we currently rely on.
2. **Per-device PRNU baselines.** The current PRNU calibration runs at the
   start of every session. Persisting a per-device fingerprint (hashed
   device-ID → average residual) lets us catch device-swap attacks across
   sessions, at the cost of needing key-value storage and a careful
   privacy review.
3. **Audio-visual sync.** A SyncNet-style cross-modal head is the natural
   sixth layer; we omitted it for v0.1 because the audio path doubles the
   integration surface (microphone permissions, codec handling, sample-rate
   negotiation). It is the single highest-impact follow-up.
4. **Hardware acceleration.** MediaPipe ships TFLite delegates for GPU and
   NPU. We default to CPU because portability matters more than raw fps in
   v0.1, but a TensorRT-converted variant routinely hits 60+ fps and would
   let one container serve multiple concurrent streams.
5. **Adversarial robustness.** We test against the canonical attack
   scenarios but not against an adversarial attacker who knows our fusion
   weights. The sustained-frames check makes optimisation-time attacks
   harder, but a formal red-team exercise is warranted before any
   high-risk deployment.

## 11. Cross-references

- Threat model — [THREAT_MODEL.md](THREAT_MODEL.md)
- Security controls and OWASP mapping — [SECURITY.md](SECURITY.md)
- Deployment runbook — [DEPLOYMENT.md](DEPLOYMENT.md)
- Development lifecycle — [SDLC.md](SDLC.md)
- REST API reference — [API.md](API.md)
