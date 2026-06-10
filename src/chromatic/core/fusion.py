"""
Decision fusion — combine per-layer signals into a single liveness verdict.

We use a simple, explainable linear fusion (weights configured in Settings).
This was a deliberate choice over a learned ensemble for three reasons:

    1.  Explainability: regulated industries (finance, healthcare) need to
        justify every "decline" decision. A weighted sum is auditable; a
        gradient-boosted ensemble is not.
    2.  Calibration: each layer is independently calibrated against published
        thresholds. We do not have a large labelled dataset to train a fusion
        model without overfitting.
    3.  Temporal stability: we require a *sustained* pass over multiple
        consecutive frames before issuing a positive verdict. This is much
        easier to reason about with a transparent fusion rule.

Each layer score is scaled into [0, 1] inside this module from raw metrics.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

from chromatic.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerScores:
    """Normalised per-layer scores in [0, 1]. Higher = more "live"."""

    hardware: float
    rppg: float
    texture: float
    geometry: float
    motion: float

    def as_dict(self) -> dict[str, float]:
        return {
            "hardware": self.hardware,
            "rppg": self.rppg,
            "texture": self.texture,
            "geometry": self.geometry,
            "motion": self.motion,
        }


@dataclass(frozen=True)
class LivenessVerdict:
    """The user-facing result of liveness verification."""

    is_live: bool
    confidence: float                # weighted-sum confidence in [0, 1]
    sustained_confidence: float      # rolling-window confidence
    layer_scores: LayerScores
    reasons: list[str]               # human-readable explanations
    timestamp: float = field(default_factory=time.time)


class FusionEngine:
    """Per-frame fusion plus temporal smoothing for sustained-liveness checks.

    Call `fuse(scores)` on every frame; the engine maintains a sliding window
    of recent verdicts and emits a sustained verdict when the rolling pass
    ratio crosses a configured threshold.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._history: deque[float] = deque(maxlen=settings.sustained_frames_required)

    def reset(self) -> None:
        self._history.clear()

    def fuse(self, scores: LayerScores) -> LivenessVerdict:
        """Combine per-layer scores into a verdict.

        Returns a verdict reflecting both the instantaneous confidence and
        the rolling-window sustained confidence.
        """
        weights = self._settings.fusion_weights
        weighted_sum = (
            weights["hardware"] * scores.hardware
            + weights["rppg"] * scores.rppg
            + weights["texture"] * scores.texture
            + weights["geometry"] * scores.geometry
            + weights["motion"] * scores.motion
        )
        weighted_sum = max(0.0, min(1.0, weighted_sum))
        self._history.append(weighted_sum)

        sustained = sum(self._history) / len(self._history)

        # Sustained verdict requires (a) full history, and (b) enough
        # frames passing the threshold.
        is_live = False
        if len(self._history) == self._settings.sustained_frames_required:
            pass_count = sum(
                1 for s in self._history if s >= self._settings.decision_threshold
            )
            ratio = pass_count / len(self._history)
            is_live = ratio >= self._settings.sustained_pass_ratio

        reasons = self._explain(scores, sustained, is_live)
        return LivenessVerdict(
            is_live=is_live,
            confidence=float(weighted_sum),
            sustained_confidence=float(sustained),
            layer_scores=scores,
            reasons=reasons,
        )

    def _explain(
        self, scores: LayerScores, sustained: float, is_live: bool
    ) -> list[str]:
        """Produce human-readable reasons for the verdict."""
        reasons: list[str] = []
        if not is_live and len(self._history) < self._settings.sustained_frames_required:
            need = self._settings.sustained_frames_required - len(self._history)
            reasons.append(f"warming up — {need} more frames needed")
        weak = {k: v for k, v in scores.as_dict().items() if v < 0.40}
        for name, value in sorted(weak.items(), key=lambda kv: kv[1]):
            reasons.append(f"low {name} score ({value:.2f})")
        if not reasons:
            if is_live:
                reasons.append(f"all signals consistent (confidence {sustained:.2f})")
            else:
                reasons.append(f"insufficient sustained confidence ({sustained:.2f})")
        return reasons
