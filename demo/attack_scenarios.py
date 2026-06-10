"""
attack_scenarios.py — Side-by-side comparison of attack types.

Runs the same pipeline against three input streams synthesised from one base
face image:

    1. LIVE        — natural sensor noise + faint pulse modulation on the face.
    2. STATIC      — no per-frame variation, no pulse (printed-photo attack).
    3. REPLAY      — synthetic pixel grid / moire overlay (phone-screen attack).

The output is a single comparison panel showing each scenario's verdict and
the underlying layer scores. The goal is to demonstrate to a technical
audience exactly which layer flags each attack class.

USAGE:
    python demo/attack_scenarios.py --image /path/to/face.jpg \\
        --output scenarios.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromatic import LivenessDetector
from chromatic.config import Settings, configure_logging, load_settings
from chromatic.core.detector import FrameDiagnostics

logging.getLogger("chromatic.audit").setLevel(logging.WARNING)

# --- Palette ----------------------------------------------------------------
COLOR_BG = "#0d1117"
COLOR_PANEL = "#161b22"
COLOR_ACCENT = "#58a6ff"
COLOR_OK = "#3fb950"
COLOR_WARN = "#d29922"
COLOR_FAIL = "#f85149"
COLOR_TEXT = "#c9d1d9"
COLOR_MUTED = "#8b949e"


# --- Synthesised attack streams --------------------------------------------

@dataclass
class Scenario:
    name: str
    subtitle: str
    transform: Callable[[npt.NDArray[np.uint8], int, np.random.Generator], npt.NDArray[np.uint8]]


def live_transform(frame: npt.NDArray[np.uint8], i: int, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Realistic webcam stream simulating a live person.

    Adds (a) per-frame Gaussian sensor noise, (b) a faint ~72 BPM pulse on
    the face region, (c) sub-pixel head translation (natural micro-motion),
    and (d) a simulated blink every ~3 seconds (darkening the eye region).
    """
    noise = rng.normal(0, 2.5, frame.shape).astype(np.int16)
    work = frame.astype(np.int16) + noise

    # Pulse modulation on the upper half of the frame.
    pulse = 1.8 * np.sin(2 * np.pi * 1.2 * (i / 30.0))
    upper = frame.shape[0] // 2
    work[:upper, :, 2] += int(pulse * 2)

    # Simulated blink: every 90 frames (~3 s @ 30 fps), darken eye band for 3 frames.
    if (i % 90) < 3:
        eye_top = int(frame.shape[0] * 0.30)
        eye_bottom = int(frame.shape[0] * 0.50)
        work[eye_top:eye_bottom, :, :] = (work[eye_top:eye_bottom, :, :] * 0.55).astype(np.int16)

    out = np.clip(work, 0, 255).astype(np.uint8)

    # Sub-pixel head motion via tiny affine translation (±1 px sinusoidal).
    dx = float(np.sin(2 * np.pi * 0.4 * (i / 30.0))) * 1.5
    dy = float(np.cos(2 * np.pi * 0.3 * (i / 30.0))) * 1.0
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]),
                         borderMode=cv2.BORDER_REPLICATE)
    return out


def static_transform(frame: npt.NDArray[np.uint8], i: int, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Static photo attack: identical frame, no temporal variation, no pulse.

    We add an almost-imperceptible quantisation-like noise pattern to mimic
    a high-quality printed photo held in front of the camera. There is no
    Poisson-Gaussian sensor noise and no pulse signal.
    """
    # A small fixed-pattern noise overlay (NOT per-frame random).
    if not hasattr(static_transform, "_pattern"):
        rng_fixed = np.random.default_rng(7)
        static_transform._pattern = rng_fixed.integers(-1, 2, size=frame.shape, dtype=np.int16)  # type: ignore[attr-defined]
    out = frame.astype(np.int16) + static_transform._pattern  # type: ignore[attr-defined]
    return np.clip(out, 0, 255).astype(np.uint8)


def replay_transform(frame: npt.NDArray[np.uint8], i: int, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Phone-screen replay attack: introduce moire interference from the secondary display.

    We darken every 3rd row and column to simulate the sub-pixel grid of an
    LCD/OLED screen, and add slight chromatic aberration. This is the
    canonical low-budget replay attack.
    """
    out = frame.copy()
    # Periodic darkening lines (the secondary screen's row/column grid).
    out[::3, :, :] = (out[::3, :, :].astype(np.float32) * 0.88).astype(np.uint8)
    out[:, ::3, :] = (out[:, ::3, :].astype(np.float32) * 0.92).astype(np.uint8)
    # Mild chromatic offset to mimic the secondary screen's colour balance.
    out[:, :, 0] = np.roll(out[:, :, 0], 1, axis=1)
    out[:, :, 2] = np.roll(out[:, :, 2], -1, axis=1)
    # Light per-frame noise so MediaPipe still tracks frame-to-frame.
    noise = rng.normal(0, 0.5, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


SCENARIOS = [
    Scenario("LIVE", "natural webcam stream + faint 72 BPM pulse", live_transform),
    Scenario("STATIC PHOTO", "printed photo — no temporal variation, no pulse", static_transform),
    Scenario("SCREEN REPLAY", "phone screen — pixel-grid moire + chromatic offset", replay_transform),
]


# --- Pipeline ---------------------------------------------------------------

def run_scenario(
    base_frame: npt.NDArray[np.uint8],
    scenario: Scenario,
    settings: Settings,
) -> tuple[npt.NDArray[np.uint8], FrameDiagnostics]:
    """Run the detector against `frames_needed` synthesised frames of `scenario`."""
    rng = np.random.default_rng(0xC0FFEE)
    last_frame: npt.NDArray[np.uint8] = base_frame
    diag: FrameDiagnostics | None = None
    frames_needed = max(
        settings.prnu_calibration_frames,
        settings.rppg_window_frames,
        settings.sustained_frames_required,
    ) + 5

    with LivenessDetector(settings) as detector:
        for i in range(frames_needed):
            last_frame = scenario.transform(base_frame, i, rng)
            diag = detector.process_frame(last_frame)
    assert diag is not None
    return last_frame, diag


# --- Rendering --------------------------------------------------------------

def _scenario_panel(
    fig: plt.Figure,
    row: int,
    scenario: Scenario,
    last_frame: npt.NDArray[np.uint8],
    diag: FrameDiagnostics,
) -> None:
    """Render one row of the comparison panel: thumbnail | scores | metrics."""
    ax_img = fig.add_subplot(3, 3, row * 3 + 1)
    ax_scores = fig.add_subplot(3, 3, row * 3 + 2)
    ax_meta = fig.add_subplot(3, 3, row * 3 + 3)

    for ax in (ax_img, ax_scores, ax_meta):
        ax.set_facecolor(COLOR_PANEL)
        for spine in ax.spines.values():
            spine.set_color(COLOR_MUTED)

    # --- Image with verdict badge ---
    rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    ax_img.imshow(rgb)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(
        f"{scenario.name}",
        color=COLOR_ACCENT, fontsize=12, weight="bold", loc="left",
    )
    ax_img.text(
        0.0, -0.08, scenario.subtitle,
        transform=ax_img.transAxes,
        color=COLOR_MUTED, fontsize=8, style="italic",
    )
    verdict = diag.verdict
    badge_color = COLOR_OK if verdict.is_live else COLOR_FAIL
    badge_text = "LIVE" if verdict.is_live else "REJECTED"
    ax_img.text(
        0.98, 0.02, badge_text,
        transform=ax_img.transAxes,
        ha="right", va="bottom",
        color="white", fontsize=11, weight="bold",
        bbox=dict(facecolor=badge_color, edgecolor="none", pad=4, boxstyle="round,pad=0.4"),
    )

    # --- Layer scores bar chart ---
    layer_labels = ["hw", "rppg", "tex", "geom", "mot"]
    scores = verdict.layer_scores
    values = [scores.hardware, scores.rppg, scores.texture, scores.geometry, scores.motion]
    colors = [
        COLOR_OK if v >= 0.65 else (COLOR_WARN if v >= 0.4 else COLOR_FAIL)
        for v in values
    ]
    bars = ax_scores.bar(layer_labels, values, color=colors)
    ax_scores.axhline(0.65, color=COLOR_WARN, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_scores.set_ylim(0, 1.05)
    ax_scores.tick_params(colors=COLOR_TEXT, labelsize=8)
    ax_scores.grid(True, color="#21262d", linewidth=0.5)
    ax_scores.set_title(
        f"sustained confidence {verdict.sustained_confidence:.3f}",
        color=COLOR_ACCENT, fontsize=10, loc="left",
    )
    for bar, value in zip(bars, values):
        ax_scores.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center", va="bottom",
            color=COLOR_TEXT, fontsize=7, weight="bold",
        )

    # --- Metrics + reasons ---
    ax_meta.axis("off")
    lines: list[tuple[str, str]] = []
    if diag.prnu is not None:
        lines.append(("noise std", f"{diag.prnu.noise_std:.2f}"))
        lines.append(("kurtosis", f"{diag.prnu.kurtosis:+.2f}"))
    if diag.texture is not None:
        lines.append(("Laplacian var", f"{diag.texture.laplacian_variance:.0f}"))
        lines.append(("moire score", f"{diag.texture.moire_score:.3f}"))
    if diag.motion is not None:
        lines.append(("flow magnitude", f"{diag.motion.motion_magnitude:.3f}"))
    if diag.pulse is not None:
        lines.append(("pulse BPM", f"{diag.pulse.bpm:.1f}"))
        lines.append(("pulse SNR", f"{diag.pulse.snr_db:+.1f} dB"))

    y = 0.97
    ax_meta.text(
        0.0, y, "Diagnostics",
        color=COLOR_ACCENT, fontsize=10, weight="bold",
        transform=ax_meta.transAxes,
    )
    y -= 0.10
    for label, value in lines:
        ax_meta.text(0.0, y, label, color=COLOR_MUTED, fontsize=8.5,
                     transform=ax_meta.transAxes)
        ax_meta.text(0.55, y, value, color=COLOR_TEXT, fontsize=8.5,
                     family="monospace", transform=ax_meta.transAxes)
        y -= 0.09

    y -= 0.03
    ax_meta.text(
        0.0, y, "Reasons",
        color=COLOR_ACCENT, fontsize=10, weight="bold",
        transform=ax_meta.transAxes,
    )
    y -= 0.09
    for reason in verdict.reasons[:3]:
        ax_meta.text(0.0, y, f"• {reason}", color=COLOR_TEXT, fontsize=8.5,
                     transform=ax_meta.transAxes, wrap=True)
        y -= 0.08


def render_comparison(
    base_frame: npt.NDArray[np.uint8],
    output_path: Path,
) -> None:
    settings = load_settings()
    configure_logging(settings)

    fig = plt.figure(figsize=(16, 11), dpi=140)
    fig.patch.set_facecolor(COLOR_BG)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.05,
                        hspace=0.55, wspace=0.25)

    fig.text(
        0.04, 0.965, "Chromatic — Attack-Scenario Comparison",
        color=COLOR_ACCENT, fontsize=18, weight="bold",
    )
    fig.text(
        0.04, 0.935,
        "Same detector. Three input streams synthesised from one base face image. "
        "Different attack classes are caught by different layers.",
        color=COLOR_MUTED, fontsize=10, style="italic",
    )

    for row, scenario in enumerate(SCENARIOS):
        last_frame, diag = run_scenario(base_frame, scenario, settings)
        _scenario_panel(fig, row, scenario, last_frame, diag)
        print(
            f"{scenario.name:14s}  "
            f"verdict={'LIVE' if diag.verdict.is_live else 'REJECTED':8s}  "
            f"sustained={diag.verdict.sustained_confidence:.3f}  "
            f"reasons={diag.verdict.reasons[:2]}"
        )

    fig.text(
        0.04, 0.015,
        "All three runs use the same configuration. Differences in verdict come "
        "from the input signal, not from per-scenario tuning.",
        color=COLOR_MUTED, fontsize=8, style="italic",
    )

    fig.savefig(output_path, facecolor=COLOR_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="base face image (used as input to all scenarios)")
    parser.add_argument("--output", type=Path, default=Path("scenarios.png"))
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"error: cannot read {args.image}", file=sys.stderr)
        return 2
    render_comparison(frame, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
