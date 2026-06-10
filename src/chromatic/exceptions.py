"""
Exception hierarchy for Chromatic.

All exceptions are derived from NexusLivenessError to allow callers to catch
all package errors with a single handler. Specific subclasses enable
fine-grained error handling and security-appropriate responses (e.g., do not
echo internal state back to the caller on validation failures).
"""

from __future__ import annotations


class NexusLivenessError(Exception):
    """Base exception for all Chromatic errors."""


# --- Input validation -----------------------------------------------------

class ValidationError(NexusLivenessError):
    """Raised when input data fails validation or sanitisation."""


class InvalidFrameError(ValidationError):
    """Raised when an input frame is malformed or out of expected bounds."""


class FrameTooLargeError(ValidationError):
    """Raised when an input frame exceeds the configured size limit."""


# --- Detection pipeline ---------------------------------------------------

class DetectionError(NexusLivenessError):
    """Raised when the detection pipeline encounters an unrecoverable error."""


class FaceNotFoundError(DetectionError):
    """Raised when no face is detected in the input frame."""


class CalibrationError(DetectionError):
    """Raised when sensor calibration fails or has not yet completed."""


class InsufficientDataError(DetectionError):
    """Raised when temporal analysis is requested before enough frames buffered."""


# --- Configuration --------------------------------------------------------

class ConfigurationError(NexusLivenessError):
    """Raised when configuration is invalid or inconsistent."""


# --- Security -------------------------------------------------------------

class SecurityError(NexusLivenessError):
    """Raised when a security control is violated."""


class RateLimitExceededError(SecurityError):
    """Raised when the rate limit for an operation has been exceeded."""


class AuthenticationError(SecurityError):
    """Raised on authentication failure. Generic by design — do not leak detail."""
