"""
Centralised configuration for Chromatic.

Settings are loaded from environment variables with safe defaults. Sensitive
values (API keys, secrets) MUST come from environment variables and are never
checked into source control.

Environment variables:
    CHROMATIC_LOG_LEVEL              Logging level (default: INFO)
    CHROMATIC_MAX_FRAME_WIDTH        Max input width in pixels (default: 1920)
    CHROMATIC_MAX_FRAME_HEIGHT       Max input height in pixels (default: 1080)
    CHROMATIC_MAX_FRAME_BYTES        Max frame bytes (default: 16 MiB)
    CHROMATIC_RPPG_WINDOW            rPPG sliding window length (default: 150 frames)
    CHROMATIC_DECISION_THRESHOLD     Liveness threshold 0..1 (default: 0.65)
    CHROMATIC_SUSTAINED_FRAMES       Frames required for sustained liveness (default: 30)
    CHROMATIC_RATE_LIMIT_PER_MIN     Max API requests per minute per principal (default: 60)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from chromatic.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def _get_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Read an int from env with optional bounds checking."""
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}, got {value}")
    return value


def _get_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Read a float from env with optional bounds checking."""
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be a float, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Settings are intentionally immutable (frozen=True) to prevent
    runtime mutation of security-relevant configuration.
    """

    # --- Capture / input ---
    target_fps: int = 30
    max_frame_width: int = 1920
    max_frame_height: int = 1080
    max_frame_bytes: int = 16 * 1024 * 1024
    min_brightness: int = 15
    max_brightness: int = 240

    # --- rPPG (CHROM) ---
    rppg_window_frames: int = 150          # 5 seconds at 30 fps
    rppg_freq_min_hz: float = 0.7          # 42 bpm
    rppg_freq_max_hz: float = 3.5          # 210 bpm
    rppg_min_snr_db: float = 3.0

    # --- PRNU / hardware forensics ---
    prnu_calibration_frames: int = 30
    prnu_noise_std_min: float = 1.5
    prnu_kurtosis_min: float = 0.0
    prnu_kurtosis_max: float = 12.0

    # --- Texture / Laplacian ---
    blur_variance_threshold: float = 60.0

    # --- Fusion ---
    decision_threshold: float = 0.65
    sustained_frames_required: int = 30
    sustained_pass_ratio: float = 0.80

    fusion_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "hardware": 0.25,
            "rppg": 0.25,
            "texture": 0.15,
            "geometry": 0.15,
            "motion": 0.20,
        }
    )

    # --- Security ---
    rate_limit_per_minute: int = 60
    audit_log_path: str | None = None      # None = stdout only

    # --- Observability ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # Validate fusion weights sum to 1.0 (within tolerance)
        total = sum(self.fusion_weights.values())
        if not 0.99 <= total <= 1.01:
            raise ConfigurationError(
                f"Fusion weights must sum to 1.0, got {total:.4f}"
            )
        if not 0.0 < self.decision_threshold < 1.0:
            raise ConfigurationError(
                f"decision_threshold must be in (0,1), got {self.decision_threshold}"
            )


def load_settings() -> Settings:
    """Load settings from environment variables with safe defaults."""
    return Settings(
        target_fps=_get_int("CHROMATIC_FPS", 30, minimum=1, maximum=120),
        max_frame_width=_get_int("CHROMATIC_MAX_FRAME_WIDTH", 1920, minimum=64, maximum=7680),
        max_frame_height=_get_int("CHROMATIC_MAX_FRAME_HEIGHT", 1080, minimum=64, maximum=4320),
        max_frame_bytes=_get_int("CHROMATIC_MAX_FRAME_BYTES", 16 * 1024 * 1024, minimum=1024),
        rppg_window_frames=_get_int("CHROMATIC_RPPG_WINDOW", 150, minimum=30, maximum=900),
        decision_threshold=_get_float("CHROMATIC_DECISION_THRESHOLD", 0.65, minimum=0.01, maximum=0.99),
        sustained_frames_required=_get_int("CHROMATIC_SUSTAINED_FRAMES", 30, minimum=1, maximum=900),
        rate_limit_per_minute=_get_int("CHROMATIC_RATE_LIMIT_PER_MIN", 60, minimum=1, maximum=10_000),
        log_level=os.environ.get("CHROMATIC_LOG_LEVEL", "INFO").upper(),
        audit_log_path=os.environ.get("CHROMATIC_AUDIT_LOG_PATH"),
    )


def configure_logging(settings: Settings) -> None:
    """Configure structured logging for the application.

    We use a stable format that is easy to parse with `jq` or ship to a SIEM.
    PII (frames, identifiers) is never logged here — only operational signals.
    """
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
