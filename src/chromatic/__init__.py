"""
Chromatic - Multi-modal deepfake & presentation attack detection.

A defence-in-depth liveness verification system combining hardware forensics,
physiological signals, and texture analysis. Designed for high-assurance
identity verification flows (KYC, remote onboarding, biometric login).
"""

__version__ = "0.1.0"
__author__ = "Chromatic Contributors"
__license__ = "MIT"

from chromatic.core.detector import LivenessDetector
from chromatic.core.fusion import LivenessVerdict

__all__ = ["LivenessDetector", "LivenessVerdict", "__version__"]
