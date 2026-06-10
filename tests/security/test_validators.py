"""
Security regression tests for FrameValidator.

These tests are the executable form of the OWASP ASVS 5.1 claims in
docs/SECURITY.md. They MUST be green for the system to ship.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from chromatic.config import Settings
from chromatic.exceptions import (
    FrameTooLargeError,
    InvalidFrameError,
)
from chromatic.security import FrameValidator

pytestmark = pytest.mark.security


# --- Type checks ---------------------------------------------------------

@pytest.mark.parametrize("bad_input", ["a string", 42, 3.14, None, b"bytes", [[1, 2, 3]]])
def test_rejects_non_ndarray(settings: Settings, bad_input: object) -> None:
    """ASVS 5.1.1 — input must match the expected schema (numpy.ndarray)."""
    validator = FrameValidator(settings)
    with pytest.raises(InvalidFrameError):
        validator.validate(bad_input)  # type: ignore[arg-type]


# --- Size / shape checks -------------------------------------------------

def test_rejects_oversized_frame(settings: Settings) -> None:
    """ASVS 11.1.4 / D-01 — must reject before allocation pressure builds."""
    validator = FrameValidator(settings)
    # Construct a frame whose nbytes exceeds the configured cap.
    # We do this by setting CHROMATIC_MAX_FRAME_BYTES via a Settings replacement
    # so we don't actually allocate a giant array.
    small_cap = Settings(max_frame_bytes=1024)
    v = FrameValidator(small_cap)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)  # 12288 bytes — > 1024
    with pytest.raises(FrameTooLargeError):
        v.validate(frame)


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),           # missing channel axis
        (64, 64, 1),        # grayscale
        (64, 64, 4),        # RGBA
        (3, 64, 64),        # channels-first
    ],
)
def test_rejects_wrong_shape(settings: Settings, shape: tuple[int, ...]) -> None:
    validator = FrameValidator(settings)
    frame = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


def test_rejects_dimensions_too_small(settings: Settings) -> None:
    validator = FrameValidator(settings)
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


def test_rejects_dimensions_too_large(settings: Settings) -> None:
    validator = FrameValidator(settings)
    big = max(settings.max_frame_width, settings.max_frame_height) + 1
    frame = np.full((720, big, 3), 128, dtype=np.uint8)
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


# --- Dtype / numeric checks ---------------------------------------------

def test_rejects_non_numeric_dtype(settings: Settings) -> None:
    """ASVS 5.1.4 — numeric bounds enforced; object dtype is hostile."""
    validator = FrameValidator(settings)
    frame = np.empty((64, 64, 3), dtype=object)
    frame[:] = 128
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


def test_rejects_nan(settings: Settings) -> None:
    """ASVS 5.1.5 — NaN / Inf must be rejected."""
    validator = FrameValidator(settings)
    frame = np.full((64, 64, 3), 128.0, dtype=np.float32)
    frame[0, 0, 0] = np.nan
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


def test_rejects_inf(settings: Settings) -> None:
    validator = FrameValidator(settings)
    frame = np.full((64, 64, 3), 128.0, dtype=np.float32)
    frame[0, 0, 0] = np.inf
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


def test_coerces_float_in_range(settings: Settings) -> None:
    """Finite floats in the valid range are coerced to uint8."""
    validator = FrameValidator(settings)
    frame = np.full((64, 64, 3), 128.7, dtype=np.float32)
    out = validator.validate(frame)
    assert out.dtype == np.uint8
    assert out.shape == frame.shape


# --- Brightness bounds --------------------------------------------------

def test_rejects_all_black(settings: Settings) -> None:
    validator = FrameValidator(settings)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


def test_rejects_all_white(settings: Settings) -> None:
    validator = FrameValidator(settings)
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    with pytest.raises(InvalidFrameError):
        validator.validate(frame)


# --- Happy path ---------------------------------------------------------

def test_accepts_valid_frame(settings: Settings, valid_frame: np.ndarray) -> None:
    validator = FrameValidator(settings)
    out = validator.validate(valid_frame)
    assert out is valid_frame or out.shape == valid_frame.shape
    assert out.dtype == np.uint8


# --- Property-based fuzz ------------------------------------------------

@hyp_settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    h=st.integers(min_value=64, max_value=128),
    w=st.integers(min_value=64, max_value=128),
    fill=st.integers(min_value=20, max_value=235),
)
def test_property_valid_random_frame_accepted(h: int, w: int, fill: int) -> None:
    """Any well-formed frame within bounds must validate."""
    settings = Settings()
    validator = FrameValidator(settings)
    frame = np.full((h, w, 3), fill, dtype=np.uint8)
    out = validator.validate(frame)
    assert out.shape == frame.shape
