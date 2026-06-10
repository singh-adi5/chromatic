"""Unit tests for the texture/moire analyser."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from chromatic.core.texture import TextureAnalyzer


@pytest.fixture
def analyzer() -> TextureAnalyzer:
    return TextureAnalyzer()


def _sharp_image(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.integers(40, 220, size=(size, size), dtype=np.uint8)
    base[size // 4 : size // 4 + 4, :] = 255
    base[:, size // 2 : size // 2 + 4] = 0
    return base


def test_blur_reduces_laplacian_variance(analyzer: TextureAnalyzer) -> None:
    """Blurred copies must show a much lower Laplacian variance than the original."""
    sharp = _sharp_image()
    blurred = cv2.GaussianBlur(sharp, (15, 15), sigmaX=5)

    sharp_metrics = analyzer.analyze(sharp)
    blurred_metrics = analyzer.analyze(blurred)

    assert sharp_metrics.laplacian_variance > 10 * blurred_metrics.laplacian_variance


def test_moire_score_in_unit_interval(analyzer: TextureAnalyzer) -> None:
    """Moire score must always be in [0, 1] regardless of input."""
    size = 256
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    pattern = 127 + 100 * np.sin(2 * np.pi * xs / 4) * np.sin(2 * np.pi * ys / 4)
    moire_image = pattern.astype(np.uint8)
    clean = _sharp_image()

    for image in (moire_image, clean):
        metrics = analyzer.analyze(image)
        assert 0.0 <= metrics.moire_score <= 1.0


def test_analyze_rejects_non_2d_input(analyzer: TextureAnalyzer) -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        analyzer.analyze(rgb)


def test_analyze_handles_uniform_image(analyzer: TextureAnalyzer) -> None:
    """A flat patch must not raise — it has near-zero variance and zero moire."""
    flat = np.full((128, 128), 128, dtype=np.uint8)
    metrics = analyzer.analyze(flat)
    assert metrics.laplacian_variance == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= metrics.moire_score <= 1.0


def test_small_image_returns_zero_moire(analyzer: TextureAnalyzer) -> None:
    """The moire computation requires >= 32 px on the smaller side."""
    tiny = np.full((16, 16), 128, dtype=np.uint8)
    metrics = analyzer.analyze(tiny)
    assert metrics.moire_score == pytest.approx(0.0)
