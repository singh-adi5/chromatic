"""
Hardware forensics — sensor noise (PRNU) analysis.

Defends against:
    - Virtual camera injection (OBS, ManyCam): the synthetic frame has
      quantisation-like noise rather than Poisson-Gaussian sensor noise.
    - Re-displayed video (phone or monitor in front of the camera): the
      noise statistics shift due to the second display chain.
    - Pre-recorded video files served back through the camera pipeline.

Approach:
    PRNU (Photo-Response Non-Uniformity) is the per-pixel multiplicative
    noise pattern unique to each sensor. We approximate it via a high-pass
    residual (frame minus its smoothed version) and look at the residual's
    statistical fingerprint.

Metrics:
    - noise_std:    expected to be > ~1.5 for a real sensor.
    - kurtosis:     real sensor noise is near-Gaussian (kurtosis ≈ 0-3 excess);
                    deepfakes and screen replays exhibit higher kurtosis due
                    to quantisation peaks.

Reference:
    J. Lukáš, J. Fridrich, M. Goljan, "Digital Camera Identification from
    Sensor Pattern Noise", IEEE Trans. Information Forensics and Security,
    vol. 1, no. 2, 2006.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PRNUMetrics:
    """Per-frame PRNU forensic metrics."""

    noise_std: float
    kurtosis: float                 # Fisher's definition (subtracts 3 from raw)
    fingerprint_correlation: float  # NaN if no reference fingerprint yet


class PRNUAnalyzer:
    """Sensor-noise based forensics.

    Lifecycle:
        1.  Call `observe(frame)` for the first N frames to build a reference
            sensor fingerprint.
        2.  After calibration, `analyze(frame)` returns per-frame metrics.
    """

    def __init__(self, *, calibration_frames: int = 30) -> None:
        if calibration_frames < 5:
            raise ValueError("calibration_frames must be >= 5")
        self._calibration_target = calibration_frames
        self._calibration_buffer: deque[npt.NDArray[np.float32]] = deque(
            maxlen=calibration_frames
        )
        self._fingerprint: npt.NDArray[np.float32] | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._fingerprint is not None

    def calibration_progress(self) -> float:
        return len(self._calibration_buffer) / self._calibration_target

    def observe(self, frame_bgr: npt.NDArray[np.uint8]) -> None:
        """Contribute a frame towards the reference fingerprint.

        Idempotent once calibration completes.
        """
        if self._fingerprint is not None:
            return
        residual = self._noise_residual(frame_bgr)
        self._calibration_buffer.append(residual)
        if len(self._calibration_buffer) >= self._calibration_target:
            # Average residuals to produce a low-variance sensor fingerprint.
            stack = np.stack(self._calibration_buffer, axis=0)
            self._fingerprint = stack.mean(axis=0).astype(np.float32)
            logger.info("PRNU calibration complete (%d frames)",
                        self._calibration_target)

    def analyze(self, frame_bgr: npt.NDArray[np.uint8]) -> PRNUMetrics:
        """Compute PRNU metrics for the current frame.

        Returns NaN for `fingerprint_correlation` until calibration is complete.
        """
        residual = self._noise_residual(frame_bgr)
        noise_std = float(residual.std())
        # Excess kurtosis (Fisher = True): 0 for Gaussian, > 0 for heavy tails.
        kurt = float(stats.kurtosis(residual.ravel(), fisher=True, bias=False))

        if self._fingerprint is None:
            corr = float("nan")
        else:
            # Pearson correlation between current residual and fingerprint.
            a = residual.ravel() - residual.mean()
            b = self._fingerprint.ravel() - self._fingerprint.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0

        return PRNUMetrics(
            noise_std=noise_std,
            kurtosis=kurt,
            fingerprint_correlation=corr,
        )

    @staticmethod
    def _noise_residual(frame_bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Extract the high-frequency residual (approximate PRNU proxy).

        We use a Gaussian denoiser and subtract. A wavelet denoiser would be
        slightly more accurate but is much slower; the Gaussian approximation
        is well-attested in the rPPG/PAD literature.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        smoothed = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=2.0)
        return (gray - smoothed).astype(np.float32)
