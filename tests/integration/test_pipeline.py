"""End-to-end integration tests for the LivenessDetector."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from chromatic import LivenessDetector
from chromatic.config import Settings, load_settings
from chromatic.exceptions import InvalidFrameError

pytestmark = pytest.mark.integration

# These tests need the MediaPipe model file. Skip if it isn't present.
MODEL = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
needs_model = pytest.mark.skipif(
    not MODEL.is_file(),
    reason="MediaPipe model not present — run scripts/download_models.sh",
)


@needs_model
def test_detector_rejects_invalid_input() -> None:
    """The validator must reject before any detection runs."""
    settings = load_settings()
    with LivenessDetector(settings) as detector, pytest.raises(InvalidFrameError):
        detector.process_frame("not a frame")  # type: ignore[arg-type]


@needs_model
def test_detector_returns_degraded_diagnostics_on_no_face() -> None:
    """No face → no exception; degraded diagnostics with is_live=False."""
    settings = load_settings()
    # A uniform grey frame: no face for MediaPipe to detect.
    rng = np.random.default_rng(0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame += rng.integers(-15, 15, size=frame.shape, dtype=np.int16).astype(np.uint8)
    with LivenessDetector(settings) as detector:
        diag = detector.process_frame(frame)
    assert diag.verdict.is_live is False
    assert diag.face is None
    assert diag.error is not None


@needs_model
def test_warm_up_progresses(valid_frame: np.ndarray) -> None:
    """Calibration progress must increase monotonically across frames."""
    settings = load_settings()
    progresses: list[float] = []
    with LivenessDetector(settings) as detector:
        for _ in range(5):
            diag = detector.process_frame(valid_frame)
            progresses.append(diag.calibration_progress)
    from_pairs = list(pairwise(progresses))
    assert all(b >= a for a, b in from_pairs)


@needs_model
def test_settings_threshold_respected() -> None:
    """A very high threshold should be impossible to clear under warm-up."""
    settings = Settings(decision_threshold=0.99, sustained_frames_required=3)
    rng = np.random.default_rng(0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame += rng.integers(-15, 15, size=frame.shape, dtype=np.int16).astype(np.uint8)
    with LivenessDetector(settings) as detector:
        for _ in range(5):
            diag = detector.process_frame(frame)
    assert diag.verdict.is_live is False
