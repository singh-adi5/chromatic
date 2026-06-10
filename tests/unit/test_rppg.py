"""Tests for the CHROM pulse estimator."""

from __future__ import annotations

import numpy as np
import pytest

from chromatic.core.rppg import CHROMPulseEstimator
from chromatic.exceptions import InsufficientDataError


def test_estimator_requires_window_fill() -> None:
    est = CHROMPulseEstimator(fps=30, window_frames=150)
    with pytest.raises(InsufficientDataError):
        est.estimate()
    assert est.is_ready is False
    assert 0.0 <= est.buffer_progress() < 1.0


def test_estimator_recovers_known_bpm() -> None:
    """Inject a ~72 BPM pulse → expect detection within tolerance.

    We model the pulse on all three channels with the realistic ratio that
    CHROM is calibrated for: stronger green-channel modulation, weaker on
    red and blue. This is closer to what an actual photoplethysmogram
    looks like and avoids harmonic confusion from a single-channel signal.
    """
    est = CHROMPulseEstimator(fps=30, window_frames=300)
    bpm_target = 72.0
    freq_hz = bpm_target / 60.0
    rng = np.random.default_rng(1)
    for i in range(300):
        t = i / 30.0
        pulse = np.sin(2 * np.pi * freq_hz * t)
        # Realistic per-channel amplitudes (G strongest, R medium, B weakest).
        r = 130.0 + 3.0 * pulse + rng.normal(0, 0.15)
        g = 110.0 + 5.0 * pulse + rng.normal(0, 0.15)
        b = 100.0 + 1.0 * pulse + rng.normal(0, 0.15)
        est.push_rgb_mean(np.array([r, g, b]))
    result = est.estimate()
    assert result.bpm == pytest.approx(bpm_target, abs=8.0), (
        f"expected ~{bpm_target} BPM, got {result.bpm:.1f}"
    )
    assert result.snr_db > 0.0


def test_estimator_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError):
        CHROMPulseEstimator(fps=30, window_frames=10)


def test_estimator_rejects_bad_frequency_band() -> None:
    with pytest.raises(ValueError):
        CHROMPulseEstimator(fps=30, window_frames=150, freq_min_hz=2.0, freq_max_hz=1.0)
    with pytest.raises(ValueError):
        CHROMPulseEstimator(fps=30, window_frames=150, freq_min_hz=0.5, freq_max_hz=20.0)


def test_estimator_rejects_wrong_shape() -> None:
    est = CHROMPulseEstimator(fps=30, window_frames=150)
    with pytest.raises(ValueError):
        est.push_rgb_mean(np.array([1.0, 2.0]))
