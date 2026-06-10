"""Unit tests for the linear fusion engine."""

from __future__ import annotations

import pytest

from chromatic.config import Settings
from chromatic.core.fusion import FusionEngine, LayerScores
from chromatic.exceptions import ConfigurationError


def test_fusion_weights_must_sum_to_one() -> None:
    with pytest.raises(ConfigurationError):
        Settings(fusion_weights={
            "hardware": 0.5, "rppg": 0.5, "texture": 0.5,
            "geometry": 0.5, "motion": 0.5,
        })


def test_weighted_sum_uses_settings_weights() -> None:
    settings = Settings(
        fusion_weights={
            "hardware": 0.20, "rppg": 0.20, "texture": 0.20,
            "geometry": 0.20, "motion": 0.20,
        }
    )
    engine = FusionEngine(settings)
    scores = LayerScores(0.5, 0.5, 0.5, 0.5, 0.5)
    verdict = engine.fuse(scores)
    assert verdict.confidence == pytest.approx(0.5)


def test_verdict_requires_full_history(settings: Settings) -> None:
    """The sustained-frames check must hold even if every frame passes."""
    engine = FusionEngine(settings)
    perfect = LayerScores(1.0, 1.0, 1.0, 1.0, 1.0)
    # Push fewer than the required number of frames.
    for _ in range(settings.sustained_frames_required - 1):
        verdict = engine.fuse(perfect)
        assert verdict.is_live is False, "must not pass during warm-up"


def test_verdict_passes_when_sustained(settings: Settings) -> None:
    engine = FusionEngine(settings)
    perfect = LayerScores(1.0, 1.0, 1.0, 1.0, 1.0)
    for _ in range(settings.sustained_frames_required):
        verdict = engine.fuse(perfect)
    assert verdict.is_live is True
    assert verdict.sustained_confidence > settings.decision_threshold


def test_verdict_fails_when_below_threshold(settings: Settings) -> None:
    engine = FusionEngine(settings)
    weak = LayerScores(0.10, 0.10, 0.10, 0.10, 0.10)
    for _ in range(settings.sustained_frames_required):
        verdict = engine.fuse(weak)
    assert verdict.is_live is False


def test_reasons_mention_weak_layers(settings: Settings) -> None:
    engine = FusionEngine(settings)
    mixed = LayerScores(0.9, 0.1, 0.9, 0.9, 0.9)
    verdict = engine.fuse(mixed)
    assert any("rppg" in r for r in verdict.reasons)


def test_reset_clears_history(settings: Settings) -> None:
    engine = FusionEngine(settings)
    perfect = LayerScores(1.0, 1.0, 1.0, 1.0, 1.0)
    for _ in range(settings.sustained_frames_required):
        engine.fuse(perfect)
    engine.reset()
    verdict = engine.fuse(perfect)
    assert verdict.is_live is False  # history empty again
