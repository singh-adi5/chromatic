"""
Remote photoplethysmography (rPPG) — extract a pulse waveform from facial video.

We implement the CHROM algorithm of De Haan & Jeanne (2013): "Robust pulse rate
from chrominance-based rPPG". CHROM is preferred over green-channel averaging
because it is robust to specular reflections and lighting drift.

Pipeline:
    1.  For each frame, take the spatial mean (R, G, B) over the forehead ROI.
    2.  Detrend and normalise each channel by its temporal mean.
    3.  Compute orthogonal chrominance signals X and Y.
    4.  Combine: S = X - alpha * Y, where alpha = std(X) / std(Y).
    5.  Band-pass filter S to the physiological range [0.7, 3.5] Hz (42-210 bpm).
    6.  Estimate BPM from the dominant frequency.
    7.  Compute an SNR-in-band metric for confidence.

Reference:
    G. de Haan and V. Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG",
    IEEE Transactions on Biomedical Engineering, vol. 60, no. 10, 2013.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal

from chromatic.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PulseEstimate:
    """A single pulse-rate estimate with confidence metrics."""

    bpm: float                         # estimated heart rate, beats per minute
    snr_db: float                      # signal-to-noise ratio in the band, dB
    pulse_waveform: npt.NDArray[np.float32]   # filtered pulse signal (window length)
    power_spectrum: npt.NDArray[np.float32]   # |FFT(pulse)|^2
    spectrum_freqs_hz: npt.NDArray[np.float32]


class CHROMPulseEstimator:
    """rPPG pulse estimation using De Haan & Jeanne's CHROM algorithm.

    The estimator buffers per-frame (R, G, B) means until `window_frames`
    samples are available, then estimates the pulse on every subsequent frame.
    """

    def __init__(
        self,
        *,
        fps: int = 30,
        window_frames: int = 150,
        freq_min_hz: float = 0.7,
        freq_max_hz: float = 3.5,
    ) -> None:
        if window_frames < 30:
            raise ValueError("window_frames must be >= 30")
        if not 0 < freq_min_hz < freq_max_hz < fps / 2:
            raise ValueError("invalid frequency band")
        self._fps = fps
        self._window = window_frames
        self._freq_min = freq_min_hz
        self._freq_max = freq_max_hz
        self._buffer: deque[npt.NDArray[np.float64]] = deque(maxlen=window_frames)

        # Pre-compute Butterworth band-pass filter coefficients.
        nyq = 0.5 * fps
        low = freq_min_hz / nyq
        high = freq_max_hz / nyq
        self._sos = signal.butter(4, [low, high], btype="bandpass", output="sos")

    @property
    def is_ready(self) -> bool:
        """True once enough samples have been buffered for a stable estimate."""
        return len(self._buffer) >= self._window

    def buffer_progress(self) -> float:
        """Fraction of the analysis window currently filled (0..1)."""
        return len(self._buffer) / self._window

    def push_rgb_mean(self, rgb_mean: npt.NDArray[np.float64]) -> None:
        """Append a single per-frame (R, G, B) mean to the buffer."""
        if rgb_mean.shape != (3,):
            raise ValueError(f"expected shape (3,), got {rgb_mean.shape}")
        self._buffer.append(rgb_mean.astype(np.float64, copy=False))

    def estimate(self) -> PulseEstimate:
        """Estimate pulse rate from the buffered samples.

        Raises:
            InsufficientDataError: Buffer not yet full.
        """
        if not self.is_ready:
            raise InsufficientDataError(
                f"buffer has {len(self._buffer)} of {self._window} frames"
            )

        rgb = np.asarray(self._buffer, dtype=np.float64)  # (N, 3) in B,G,R or R,G,B
        # OpenCV reads frames as BGR; downstream code converts means accordingly.
        # By convention here, callers pass RGB-ordered means.
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]

        # Temporal normalisation: divide each channel by its mean to remove DC.
        r_norm = r / (np.mean(r) + 1e-8)
        g_norm = g / (np.mean(g) + 1e-8)
        b_norm = b / (np.mean(b) + 1e-8)

        # CHROM chrominance signals (De Haan & Jeanne, 2013).
        x = 3.0 * r_norm - 2.0 * g_norm
        y = 1.5 * r_norm + g_norm - 1.5 * b_norm
        alpha = float(np.std(x) / (np.std(y) + 1e-8))
        s = x - alpha * y

        # Band-pass to physiological range.
        try:
            pulse = signal.sosfiltfilt(self._sos, s).astype(np.float32)
        except ValueError as exc:
            # sosfiltfilt requires at least 3 * (filter order * 2) samples.
            raise InsufficientDataError(str(exc)) from exc

        # FFT and SNR.
        n = pulse.shape[0]
        freqs = np.fft.rfftfreq(n, d=1.0 / self._fps).astype(np.float32)
        spectrum = (np.abs(np.fft.rfft(pulse)) ** 2).astype(np.float32)

        in_band = (freqs >= self._freq_min) & (freqs <= self._freq_max)
        if not in_band.any():
            raise InsufficientDataError("no FFT bin in physiological band")

        # Pulse-rate estimate: dominant frequency in the physiological band.
        peak_idx = int(np.argmax(spectrum * in_band))
        bpm = float(freqs[peak_idx] * 60.0)

        # SNR-in-band: power at the peak (and its harmonics) vs. out-of-band power.
        # Take a ±0.1 Hz window around the peak and its second harmonic.
        peak_freq = freqs[peak_idx]
        signal_mask = self._signal_mask(freqs, peak_freq, half_bw=0.1)
        noise_mask = in_band & ~signal_mask
        signal_power = float(spectrum[signal_mask].sum())
        noise_power = float(spectrum[noise_mask].sum())
        snr_db = float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))

        return PulseEstimate(
            bpm=bpm,
            snr_db=snr_db,
            pulse_waveform=pulse,
            power_spectrum=spectrum,
            spectrum_freqs_hz=freqs,
        )

    @staticmethod
    def _signal_mask(
        freqs: npt.NDArray[np.float32], peak_freq: float, *, half_bw: float = 0.1
    ) -> npt.NDArray[np.bool_]:
        """Mask covering the peak and its second harmonic."""
        m1 = np.abs(freqs - peak_freq) <= half_bw
        m2 = np.abs(freqs - 2.0 * peak_freq) <= half_bw
        return m1 | m2


def extract_roi_mean_rgb(
    frame_bgr: npt.NDArray[np.uint8], roi_mask: npt.NDArray[np.uint8]
) -> npt.NDArray[np.float64]:
    """Compute the mean (R, G, B) over the masked region.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image.
        roi_mask: (H, W) uint8 mask, non-zero pixels are included.

    Returns:
        A 1-D array of length 3 with the mean (R, G, B) of the masked region.
    """
    if frame_bgr.shape[:2] != roi_mask.shape:
        raise ValueError("frame and mask shapes do not match")
    if not roi_mask.any():
        # Empty mask — return zeros; the caller is responsible for handling this.
        return np.zeros(3, dtype=np.float64)
    # cv2.mean returns (B, G, R, alpha) in OpenCV order — reorder to RGB.
    bgr_mean = cv2.mean(frame_bgr, mask=roi_mask)[:3]
    return np.array([bgr_mean[2], bgr_mean[1], bgr_mean[0]], dtype=np.float64)


# Local import to avoid a circular dependency at module load time.
import cv2  # noqa: E402
