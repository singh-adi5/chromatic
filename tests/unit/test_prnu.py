"""Unit tests for the PRNU (sensor-noise) analyser."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chromatic.core.prnu import PRNUAnalyzer


def _camera_like_frame(seed: int = 0, size: int = 128) -> np.ndarray:
    """Synthesise a noisy frame approximating sensor capture (mid-grey + AWGN)."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), 128, dtype=np.float32)
    noise = rng.normal(0, 3.0, size=base.shape).astype(np.float32)
    out = np.clip(base + noise, 0, 255).astype(np.uint8)
    return out


def test_constructor_rejects_too_few_calibration_frames() -> None:
    with pytest.raises(ValueError):
        PRNUAnalyzer(calibration_frames=2)


def test_is_not_calibrated_initially() -> None:
    analyzer = PRNUAnalyzer(calibration_frames=5)
    assert analyzer.is_calibrated is False
    assert analyzer.calibration_progress() == 0.0


def test_calibration_completes_after_target_frames() -> None:
    analyzer = PRNUAnalyzer(calibration_frames=5)
    for i in range(5):
        analyzer.observe(_camera_like_frame(seed=i))
    assert analyzer.is_calibrated is True
    assert analyzer.calibration_progress() == pytest.approx(1.0)


def test_observe_is_idempotent_after_calibration() -> None:
    analyzer = PRNUAnalyzer(calibration_frames=5)
    for i in range(5):
        analyzer.observe(_camera_like_frame(seed=i))
    # Should not crash, should not change state.
    analyzer.observe(_camera_like_frame(seed=99))
    assert analyzer.is_calibrated is True


def test_analyze_before_calibration_yields_nan_correlation() -> None:
    analyzer = PRNUAnalyzer(calibration_frames=5)
    metrics = analyzer.analyze(_camera_like_frame(seed=0))
    assert math.isnan(metrics.fingerprint_correlation)
    assert metrics.noise_std > 0.0


def test_analyze_after_calibration_yields_finite_correlation() -> None:
    analyzer = PRNUAnalyzer(calibration_frames=5)
    for i in range(5):
        analyzer.observe(_camera_like_frame(seed=i))
    metrics = analyzer.analyze(_camera_like_frame(seed=10))
    assert not math.isnan(metrics.fingerprint_correlation)
    assert -1.0 <= metrics.fingerprint_correlation <= 1.0


def test_synthetic_quantized_frame_changes_kurtosis() -> None:
    """Quantised (replay-like) noise has a different kurtosis profile from Gaussian.

    Discrete (e.g. three-level) quantisation is platykurtic — its excess kurtosis
    drops well below 0, while raw Gaussian sensor noise sits near 0. The contract
    we care about for forensics is that the *difference* is measurable, not the
    sign.
    """
    analyzer = PRNUAnalyzer(calibration_frames=5)
    gaussian_frame = _camera_like_frame(seed=42)

    rng = np.random.default_rng(7)
    quantised = np.full((128, 128, 3), 128, dtype=np.float32)
    quant_noise = rng.choice([-10, 0, 10], size=quantised.shape).astype(np.float32)
    quantised = np.clip(quantised + quant_noise, 0, 255).astype(np.uint8)

    gauss_metrics = analyzer.analyze(gaussian_frame)
    quant_metrics = analyzer.analyze(quantised)

    # Different noise distribution → measurably different kurtosis.
    assert abs(quant_metrics.kurtosis - gauss_metrics.kurtosis) > 0.3
