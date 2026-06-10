"""Unit tests for the motion analyser."""

from __future__ import annotations

import numpy as np
import pytest

from chromatic.core.motion import MotionAnalyzer


@pytest.fixture
def analyzer() -> MotionAnalyzer:
    return MotionAnalyzer(ear_blink_threshold=0.20, ear_history=30)


def _frame(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_first_frame_returns_zero_motion(analyzer: MotionAnalyzer) -> None:
    frame = _frame(seed=1)
    metrics = analyzer.analyze(frame, face_bbox=(10, 10, 100, 100), ear=0.30)
    assert metrics.motion_magnitude == pytest.approx(0.0)
    assert metrics.motion_variability == pytest.approx(0.0)
    assert metrics.blink_detected_recently is False


def test_motion_detected_between_different_frames(analyzer: MotionAnalyzer) -> None:
    frame_a = _frame(seed=1)
    frame_b = _frame(seed=2)

    analyzer.analyze(frame_a, face_bbox=(0, 0, 128, 128), ear=0.30)
    metrics = analyzer.analyze(frame_b, face_bbox=(0, 0, 128, 128), ear=0.30)
    assert metrics.motion_magnitude > 0.0


def test_blink_detected_after_ear_dip(analyzer: MotionAnalyzer) -> None:
    """Feed EAR values modelling open → closed → open eyes."""
    frame = _frame(seed=3)
    # Open eyes baseline
    for _ in range(10):
        analyzer.analyze(frame, face_bbox=(0, 0, 128, 128), ear=0.32)
    # Blink dip
    for ear in [0.18, 0.12, 0.10, 0.15]:
        analyzer.analyze(frame, face_bbox=(0, 0, 128, 128), ear=ear)
    # Recover
    metrics = analyzer.analyze(frame, face_bbox=(0, 0, 128, 128), ear=0.32)
    assert metrics.blink_detected_recently is True


def test_reset_clears_state(analyzer: MotionAnalyzer) -> None:
    frame = _frame(seed=4)
    analyzer.analyze(frame, face_bbox=(0, 0, 128, 128), ear=0.30)
    analyzer.reset()
    metrics = analyzer.analyze(_frame(seed=5), face_bbox=(0, 0, 128, 128), ear=0.30)
    assert metrics.motion_magnitude == pytest.approx(0.0)


def test_empty_bbox_returns_zero_metrics(analyzer: MotionAnalyzer) -> None:
    frame = _frame(seed=6)
    # Bounding box outside the frame — face_crop will be empty.
    metrics = analyzer.analyze(frame, face_bbox=(500, 500, 10, 10), ear=0.30)
    assert metrics.motion_magnitude == pytest.approx(0.0)
    assert metrics.motion_variability == pytest.approx(0.0)


def test_bbox_clipped_to_frame(analyzer: MotionAnalyzer) -> None:
    """Negative origin and oversize bbox must be clipped, not crash."""
    frame = _frame(seed=7)
    analyzer.analyze(frame, face_bbox=(-20, -20, 80, 80), ear=0.30)
    metrics = analyzer.analyze(_frame(seed=8), face_bbox=(-20, -20, 80, 80), ear=0.30)
    assert metrics.motion_magnitude >= 0.0
