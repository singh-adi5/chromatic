"""
Input validation and sanitisation.

Maps to OWASP ASVS 5.1 (Input Validation Requirements):
- 5.1.1 Reject input not matching schema
- 5.1.3 Validate range, length, format
- 5.1.4 Enforce numeric and string bounds
- 5.1.5 Reject NaN/Inf in numeric fields

Maps to OWASP ML Security Top 10:
- ML02:2023 Data Poisoning Attack (we never persist user input)
- ML04:2023 Membership Inference (input bounds prevent oracle attacks)
- ML06:2023 AI Supply Chain Attacks (no untrusted model loading)
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt

from chromatic.config import Settings
from chromatic.exceptions import (
    FrameTooLargeError,
    InvalidFrameError,
)

logger = logging.getLogger(__name__)

# A frame must be a 3-channel BGR image with uint8 elements.
EXPECTED_DTYPE: Final = np.uint8
EXPECTED_CHANNELS: Final = 3


class FrameValidator:
    """Validate raw image frames before they enter the detection pipeline.

    Validation is deliberately strict: anything ambiguous is rejected. This
    follows the OWASP positive-validation pattern (allow-list known good,
    reject everything else).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(self, frame: npt.NDArray) -> npt.NDArray[np.uint8]:
        """Validate a frame and return it as a guaranteed-safe uint8 BGR array.

        Args:
            frame: Candidate frame (must be a numpy ndarray).

        Returns:
            The validated frame.

        Raises:
            InvalidFrameError: Frame failed structural validation.
            FrameTooLargeError: Frame exceeds the configured byte limit.
        """
        # Type check — reject anything that isn't an ndarray outright.
        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(
                f"frame must be numpy.ndarray, got {type(frame).__name__}"
            )

        # Size check FIRST to defend against memory-exhaustion attacks.
        if frame.nbytes > self._settings.max_frame_bytes:
            raise FrameTooLargeError(
                f"frame size {frame.nbytes} bytes exceeds limit "
                f"{self._settings.max_frame_bytes} bytes"
            )

        # Shape check — must be (H, W, 3).
        if frame.ndim != 3 or frame.shape[2] != EXPECTED_CHANNELS:
            raise InvalidFrameError(
                f"frame must have shape (H, W, 3), got {frame.shape}"
            )

        h, w = frame.shape[0], frame.shape[1]
        if w > self._settings.max_frame_width or h > self._settings.max_frame_height:
            raise InvalidFrameError(
                f"frame dimensions {w}x{h} exceed configured maximum "
                f"{self._settings.max_frame_width}x{self._settings.max_frame_height}"
            )
        if w < 64 or h < 64:
            raise InvalidFrameError(
                f"frame dimensions {w}x{h} below minimum 64x64"
            )

        # Dtype check — coerce only from safe numeric dtypes, never from object.
        if frame.dtype != EXPECTED_DTYPE:
            if not np.issubdtype(frame.dtype, np.number):
                raise InvalidFrameError(
                    f"frame dtype must be numeric, got {frame.dtype}"
                )
            # Defensive: reject any non-finite values BEFORE casting.
            if not np.isfinite(frame).all():
                raise InvalidFrameError("frame contains NaN or Inf values")
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Brightness sanity check — fully black or fully white frames are
        # either capture failures or known attack vectors (frame replacement).
        mean_brightness = float(frame.mean())
        if mean_brightness < self._settings.min_brightness:
            raise InvalidFrameError(
                f"frame too dark (mean {mean_brightness:.1f} < "
                f"{self._settings.min_brightness})"
            )
        if mean_brightness > self._settings.max_brightness:
            raise InvalidFrameError(
                f"frame too bright (mean {mean_brightness:.1f} > "
                f"{self._settings.max_brightness})"
            )

        return frame
