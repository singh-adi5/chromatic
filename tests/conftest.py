"""Shared fixtures for the test suite."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from chromatic.config import Settings, load_settings

# Keep test output focused: silence the audit logger by default.
logging.getLogger("chromatic.audit").setLevel(logging.WARNING)


@pytest.fixture
def settings() -> Settings:
    """A settings instance with defaults from the environment."""
    return load_settings()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xC0FFEE)


@pytest.fixture
def valid_frame(rng: np.random.Generator) -> np.ndarray:
    """A 720x1280x3 uint8 BGR frame that passes the validator."""
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    noise = rng.integers(-15, 15, size=frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the on-disk test fixtures directory."""
    return Path(__file__).resolve().parent / "fixtures"
