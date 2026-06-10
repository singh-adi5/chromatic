"""
Test suite for Chromatic.

This package contains comprehensive tests for the Chromatic liveness detection system:

- **unit/**: Unit tests for individual components (fusion, motion, PRNU, rPPG, texture analysis)
- **integration/**: Integration tests for the complete detection pipeline
- **security/**: Security-focused tests for audit logging, rate limiting, and input validation

Shared test fixtures (settings, RNG, sample frames, fixtures directory) are defined in conftest.py
and can be used throughout the test suite via pytest's fixture mechanism.

See CONTRIBUTING.md for test development guidelines.
"""

from __future__ import annotations

import logging

import pytest

# Core detection components (tested in unit/)
from chromatic.core import (
    CHROMPulseEstimator,
    FaceAnalysis,
    FaceAnalyzer,
    FrameDiagnostics,
    FusionEngine,
    LayerScores,
    LivenessDetector,
    LivenessVerdict,
    MotionAnalyzer,
    MotionMetrics,
    PRNUAnalyzer,
    PRNUMetrics,
    PulseEstimate,
    TextureAnalyzer,
    TextureMetrics,
)

# Configuration and settings
from chromatic.config import Settings, load_settings

# Security components (tested in security/)
from chromatic.security import AuditLog, RateLimiter, Validators

# Shared fixtures via conftest
pytest_plugins = ["tests.conftest"]

__all__ = [
    # Core
    "CHROMPulseEstimator",
    "FaceAnalysis",
    "FaceAnalyzer",
    "FrameDiagnostics",
    "FusionEngine",
    "LayerScores",
    "LivenessDetector",
    "LivenessVerdict",
    "MotionAnalyzer",
    "MotionMetrics",
    "PRNUAnalyzer",
    "PRNUMetrics",
    "PulseEstimate",
    "TextureAnalyzer",
    "TextureMetrics",
    # Config
    "Settings",
    "load_settings",
    # Security
    "AuditLog",
    "RateLimiter",
    "Validators",
    # Testing utilities
    "pytest",
    "logging",
]
