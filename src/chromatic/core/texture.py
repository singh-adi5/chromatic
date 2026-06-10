"""
Texture-based liveness signals.

Defends against:
    - Synthetic/blurred faces (deepfakes, low-quality GAN outputs): low
      Laplacian variance (insufficient high-frequency detail).
    - Re-displayed faces (phone or monitor in front of the camera): strong
      periodic peaks in the 2-D FFT due to the second screen's pixel grid
      ("moire" pattern).

Metrics:
    - laplacian_variance: variance of the Laplacian filter response. Real
      skin has many micro-textures; over-smoothed deepfakes have low values.
    - moire_score: ratio of off-DC spectral energy concentrated in narrow
      ring-shaped bands. Higher values indicate a periodic structure consistent
      with re-photographed content.

References:
    R. Bartolomey, et al. "Face Liveness Detection from a Single Image with
    Sparse Low Rank Bilinear Discriminative Model", ECCV 2010.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextureMetrics:
    laplacian_variance: float
    moire_score: float


class TextureAnalyzer:
    """Texture-based liveness signals from a single ROI."""

    def analyze(self, roi_gray: npt.NDArray[np.uint8]) -> TextureMetrics:
        """Compute texture metrics on a grayscale region.

        Args:
            roi_gray: (H, W) uint8 grayscale region.
        """
        if roi_gray.ndim != 2:
            raise ValueError("roi_gray must be 2-D")

        # --- Laplacian variance (blurriness proxy) ---
        lap = cv2.Laplacian(roi_gray, ddepth=cv2.CV_64F)
        lap_var = float(lap.var())

        # --- Moire score from 2-D FFT ---
        moire = self._moire_score(roi_gray)

        return TextureMetrics(laplacian_variance=lap_var, moire_score=moire)

    @staticmethod
    def _moire_score(roi_gray: npt.NDArray[np.uint8]) -> float:
        """Compute a moire score from the 2-D Fourier spectrum.

        We measure how much of the off-DC spectral energy is concentrated in
        narrow annular bands (a signature of periodic interference). Returns
        a value in [0, 1], higher = more moire-like.
        """
        # Crop to a centred square to avoid asymmetry artefacts.
        h, w = roi_gray.shape
        side = min(h, w)
        if side < 32:
            return 0.0
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        patch = roi_gray[y0 : y0 + side, x0 : x0 + side].astype(np.float32)

        # Window to suppress spectral leakage at the boundary.
        window = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
        patch = (patch - patch.mean()) * window

        spec = np.fft.fftshift(np.abs(np.fft.fft2(patch)))
        # Suppress the DC bin and its immediate neighbourhood (low frequencies).
        cy, cx = side // 2, side // 2
        y, x = np.ogrid[:side, :side]
        radial = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        dc_mask = radial < (side * 0.05)
        spec[dc_mask] = 0.0

        # Mid/high-frequency ring (where moire patterns appear).
        ring = (radial >= side * 0.1) & (radial <= side * 0.4)
        ring_energy = float(spec[ring].sum())
        total_energy = float(spec.sum()) + 1e-9
        ring_fraction = ring_energy / total_energy

        # Concentration: are ring frequencies clustered in a few angular bands?
        # Compute angular histogram of the top-1% bins inside the ring.
        ring_spec = np.where(ring, spec, 0.0)
        if ring_spec.max() <= 0:
            return 0.0
        threshold = np.quantile(ring_spec[ring], 0.99)
        peaks = np.argwhere(ring_spec >= threshold)
        if len(peaks) < 4:
            return 0.0
        angles = np.degrees(
            np.arctan2(peaks[:, 0] - cy, peaks[:, 1] - cx)
        )
        # Histogram into 18 angular bins (10° each); concentration = max/total.
        hist, _ = np.histogram(angles, bins=18, range=(-180, 180))
        concentration = float(hist.max()) / float(hist.sum() + 1e-9)

        # Combine ring fraction and concentration; both saturate to ~1 for replays.
        return float(min(1.0, ring_fraction * concentration * 4.0))
