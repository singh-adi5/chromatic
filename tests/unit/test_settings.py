"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from chromatic.config import Settings
from chromatic.exceptions import ConfigurationError


def test_default_settings_valid() -> None:
    s = Settings()
    assert s.decision_threshold == pytest.approx(0.65)
    assert sum(s.fusion_weights.values()) == pytest.approx(1.0)


def test_threshold_must_be_in_range() -> None:
    with pytest.raises(ConfigurationError):
        Settings(decision_threshold=0.0)
    with pytest.raises(ConfigurationError):
        Settings(decision_threshold=1.0)
    with pytest.raises(ConfigurationError):
        Settings(decision_threshold=-0.5)


def test_settings_frozen() -> None:
    """Settings must not be mutable at runtime — security control T-04."""
    s = Settings()
    with pytest.raises(Exception):  # FrozenInstanceError is dataclass-specific
        s.decision_threshold = 0.1  # type: ignore[misc]


def test_fusion_weights_must_sum_to_one() -> None:
    with pytest.raises(ConfigurationError):
        Settings(fusion_weights={
            "hardware": 0.10, "rppg": 0.10, "texture": 0.10,
            "geometry": 0.10, "motion": 0.10,
        })
