"""
Top-level liveness detector — orchestrates all detection layers.

This module is the single entry point for the application. It owns the
sub-detectors, the frame validator, and the fusion engine, and exposes one
method (`process_frame`) that the API and demo code call.

Threading model:
    Detector instances are NOT thread-safe. Create one per worker. The
    expensive MediaPipe graph is held by the FaceAnalyzer instance and
    benefits from process-local affinity anyway.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from chromatic.config import Settings
from chromatic.core.face_analyzer import FaceAnalysis, FaceAnalyzer
from chromatic.core.fusion import FusionEngine, LayerScores, LivenessVerdict
from chromatic.core.motion import MotionAnalyzer, MotionMetrics
from chromatic.core.prnu import PRNUAnalyzer, PRNUMetrics
from chromatic.core.rppg import (
    CHROMPulseEstimator,
    PulseEstimate,
    extract_roi_mean_rgb,
)
from chromatic.core.texture import TextureAnalyzer, TextureMetrics
from chromatic.exceptions import (
    FaceNotFoundError,
    InsufficientDataError,
)
from chromatic.security import AuditEventType, FrameValidator, audit

logger = logging.getLogger(__name__)


@dataclass
class FrameDiagnostics:
    """Detailed per-frame diagnostics for visualisation and audit.

    This is a richer object than the public `LivenessVerdict`; it carries
    every intermediate metric needed by the demo dashboard.
    """

    verdict: LivenessVerdict
    face: FaceAnalysis | None
    pulse: PulseEstimate | None
    prnu: PRNUMetrics | None
    texture: TextureMetrics | None
    motion: MotionMetrics | None
    calibration_progress: float = 0.0
    rppg_progress: float = 0.0
    error: str | None = None


class LivenessDetector:
    """End-to-end multi-modal liveness detector."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._validator = FrameValidator(settings)
        self._face = FaceAnalyzer()
        self._prnu = PRNUAnalyzer(
            calibration_frames=settings.prnu_calibration_frames
        )
        self._rppg = CHROMPulseEstimator(
            fps=settings.target_fps,
            window_frames=settings.rppg_window_frames,
            freq_min_hz=settings.rppg_freq_min_hz,
            freq_max_hz=settings.rppg_freq_max_hz,
        )
        self._texture = TextureAnalyzer()
        self._motion = MotionAnalyzer()
        self._fusion = FusionEngine(settings)
        self._frame_count = 0

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Release underlying resources. Safe to call multiple times."""
        self._face.close()

    def __enter__(self) -> LivenessDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def reset(self) -> None:
        """Reset all temporal state. Useful after a session ends."""
        self._motion.reset()
        self._fusion.reset()
        self._frame_count = 0

    # --- main entry point -------------------------------------------------

    def process_frame(self, frame_bgr: npt.NDArray) -> FrameDiagnostics:
        """Process a single frame and return detailed diagnostics.

        Validation errors are raised; detection errors are captured into
        the returned diagnostics object (so the caller can render a
        warm-up state without exception handling on the hot path).
        """
        # Validation can raise — let it propagate; the caller decides how
        # to respond (e.g., HTTP 400 from the API layer).
        frame = self._validator.validate(frame_bgr)
        self._frame_count += 1
        request_id = f"frame-{self._frame_count}"

        # Always contribute to PRNU calibration first (cheap, idempotent
        # after calibration completes).
        self._prnu.observe(frame)

        # Try to analyse the face. If no face, return a degraded verdict
        # rather than raising — the demo wants to keep showing UI.
        try:
            face = self._face.analyze(frame)
        except FaceNotFoundError as exc:
            return self._degraded_diagnostics(reason=str(exc))

        # --- per-layer analysis ---
        prnu_metrics = self._prnu.analyze(frame)

        rgb_mean = extract_roi_mean_rgb(frame, face.forehead_mask)
        self._rppg.push_rgb_mean(rgb_mean)
        pulse: PulseEstimate | None
        if self._rppg.is_ready:
            try:
                pulse = self._rppg.estimate()
            except InsufficientDataError:
                pulse = None
        else:
            pulse = None

        # Texture: use the union of the two cheek masks.
        cheek_mask = cv2.bitwise_or(face.left_cheek_mask, face.right_cheek_mask)
        cheek_pixels = cv2.bitwise_and(frame, frame, mask=cheek_mask)
        cheek_gray = cv2.cvtColor(cheek_pixels, cv2.COLOR_BGR2GRAY)
        # Crop to the bounding box of the cheek mask to keep texture FFT square-friendly.
        ys, xs = np.nonzero(cheek_mask)
        if ys.size > 0:
            texture_roi = cheek_gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
            texture = self._texture.analyze(texture_roi)
        else:
            texture = None

        # Motion / blink.
        motion = self._motion.analyze(frame, face.bbox, face.mean_eye_aspect_ratio)

        # --- normalise to [0,1] layer scores ---
        scores = self._compute_scores(
            face=face, prnu=prnu_metrics, pulse=pulse,
            texture=texture, motion=motion,
        )

        verdict = self._fusion.fuse(scores)

        audit(
            AuditEventType.DETECTION_COMPLETED,
            request_id=request_id,
            outcome="success" if verdict.is_live else "failure",
            frame_index=self._frame_count,
            confidence=round(verdict.sustained_confidence, 3),
        )

        return FrameDiagnostics(
            verdict=verdict,
            face=face,
            pulse=pulse,
            prnu=prnu_metrics,
            texture=texture,
            motion=motion,
            calibration_progress=self._prnu.calibration_progress(),
            rppg_progress=self._rppg.buffer_progress(),
        )

    # --- scoring ----------------------------------------------------------

    def _compute_scores(
        self,
        *,
        face: FaceAnalysis,
        prnu: PRNUMetrics,
        pulse: PulseEstimate | None,
        texture: TextureMetrics | None,
        motion: MotionMetrics,
    ) -> LayerScores:
        """Map raw metrics into per-layer scores in [0, 1].

        Scoring uses smooth sigmoid-style transforms so that small noise
        does not flip the verdict near a threshold.
        """
        s = self._settings

        # Hardware: noise_std should be > threshold; kurtosis should be in range.
        hardware_std = _saturating_score(
            prnu.noise_std, lower=s.prnu_noise_std_min, soft_upper=4.0
        )
        if math.isnan(prnu.kurtosis):
            hardware_kurt = 0.5
        elif s.prnu_kurtosis_min <= prnu.kurtosis <= s.prnu_kurtosis_max:
            hardware_kurt = 1.0
        else:
            # Sharper penalty outside the band
            dist = min(
                abs(prnu.kurtosis - s.prnu_kurtosis_min),
                abs(prnu.kurtosis - s.prnu_kurtosis_max),
            )
            hardware_kurt = max(0.0, 1.0 - dist / 4.0)
        hardware = 0.6 * hardware_std + 0.4 * hardware_kurt

        # rPPG: pass if SNR > threshold and BPM in physiological range.
        if pulse is None:
            rppg_score = 0.5  # neutral during warm-up
        elif not (40.0 <= pulse.bpm <= 180.0):
            rppg_score = 0.1
        else:
            # Map SNR to [0,1] with a soft transition near the threshold.
            rppg_score = _saturating_score(
                pulse.snr_db, lower=s.rppg_min_snr_db, soft_upper=15.0
            )

        # Texture: high Laplacian variance is "live"; high moire score is "fake".
        if texture is None:
            texture_score = 0.5
        else:
            lap_norm = _saturating_score(
                texture.laplacian_variance,
                lower=s.blur_variance_threshold,
                soft_upper=400.0,
            )
            moire_penalty = 1.0 - min(1.0, texture.moire_score)
            texture_score = 0.6 * lap_norm + 0.4 * moire_penalty

        # Geometry: penalise extreme yaw/pitch (face must be reasonably frontal).
        # We use ±40° as the boundary at which the score is zero. Within
        # ±15° the score saturates near 1.0 to avoid penalising natural motion.
        yaw_pen = min(1.0, max(0.0, abs(face.yaw_deg) - 15.0) / 25.0)
        pitch_pen = min(1.0, max(0.0, abs(face.pitch_deg) - 15.0) / 25.0)
        geometry = 1.0 - 0.5 * (yaw_pen + pitch_pen)

        # Motion: non-trivial motion magnitude AND a blink in recent history.
        mag_score = _saturating_score(motion.motion_magnitude, lower=0.1, soft_upper=2.0)
        blink_score = 1.0 if motion.blink_detected_recently else 0.5
        motion_score = 0.5 * mag_score + 0.5 * blink_score

        return LayerScores(
            hardware=float(hardware),
            rppg=float(rppg_score),
            texture=float(texture_score),
            geometry=float(geometry),
            motion=float(motion_score),
        )

    def _degraded_diagnostics(self, *, reason: str) -> FrameDiagnostics:
        """Return a neutral diagnostics object when face detection fails."""
        zero_scores = LayerScores(0.0, 0.0, 0.0, 0.0, 0.0)
        verdict = LivenessVerdict(
            is_live=False,
            confidence=0.0,
            sustained_confidence=0.0,
            layer_scores=zero_scores,
            reasons=[reason],
        )
        return FrameDiagnostics(
            verdict=verdict, face=None, pulse=None, prnu=None,
            texture=None, motion=None,
            calibration_progress=self._prnu.calibration_progress(),
            rppg_progress=self._rppg.buffer_progress(),
            error=reason,
        )


def _saturating_score(
    value: float, *, lower: float, soft_upper: float
) -> float:
    """Map a raw metric to [0, 1] with a soft saturation curve.

    Below `lower` -> 0; at `soft_upper` -> ~0.9; asymptotic to 1.
    """
    if value <= lower:
        return 0.0
    # Quadratic ease-in then logistic top.
    x = (value - lower) / max(1e-6, soft_upper - lower)
    return float(1.0 - math.exp(-2.5 * x))
