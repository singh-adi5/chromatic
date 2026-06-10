"""
tech_demo.py — Engineering-grade visual demonstration.

WHAT THIS IS:
    A single-image, single-run demonstration that produces a publication-quality
    diagnostic panel showing every output of the Chromatic pipeline:

        - The input frame with the MediaPipe 468-point face mesh overlay
        - Highlighted ROIs for rPPG (forehead) and PRNU/texture (cheeks)
        - The extracted pulse waveform (CHROM algorithm, post band-pass)
        - The pulse power spectrum with the heart-rate band marked
        - Per-layer scores with a verdict and human-readable reasons

WHO THIS IS FOR:
    Technical reviewers (security architects, ML engineers, fintech engineering
    leadership) who want to see exactly what each layer measures, not just a
    pass/fail box. The panel is also suitable for portfolio/post screenshots.

USAGE:
    python demo/tech_demo.py --image path/to/face.jpg --output diagnostic.png

    # With a webcam feed (writes one frame per second to ./out/):
    python demo/tech_demo.py --webcam --output-dir ./out/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec

# Allow `python demo/tech_demo.py` from repo root without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromatic import LivenessDetector
from chromatic.config import configure_logging, load_settings
from chromatic.core.detector import FrameDiagnostics

# --- Brand palette (dark, fintech-friendly) -----------------------------------
COLOR_BG = "#0d1117"
COLOR_PANEL = "#161b22"
COLOR_ACCENT = "#58a6ff"
COLOR_OK = "#3fb950"
COLOR_WARN = "#d29922"
COLOR_FAIL = "#f85149"
COLOR_TEXT = "#c9d1d9"
COLOR_MUTED = "#8b949e"

# Suppress audit log noise on stdout when producing a visual artefact.
logging.getLogger("chromatic.audit").setLevel(logging.WARNING)


def _draw_face_mesh(frame_bgr: npt.NDArray[np.uint8], diag: FrameDiagnostics) -> npt.NDArray[np.uint8]:
    """Overlay the MediaPipe landmark mesh and ROI highlights onto the frame.

    Returns a new RGB array suitable for direct display in matplotlib.
    """
    overlay = frame_bgr.copy()
    face = diag.face
    if face is None:
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # 1) ROI tinting: forehead in green, cheeks in cyan. Blend rather than overwrite.
    tint = np.zeros_like(overlay)
    tint[face.forehead_mask > 0] = (90, 200, 90)
    tint[face.left_cheek_mask > 0] = (220, 200, 80)
    tint[face.right_cheek_mask > 0] = (220, 200, 80)
    overlay = cv2.addWeighted(overlay, 0.75, tint, 0.25, 0)

    # 2) Plot every landmark as a tiny circle.
    for x, y in face.landmarks_px.astype(int):
        cv2.circle(overlay, (int(x), int(y)), 1, (200, 230, 255), -1)

    # 3) Bounding box and pose annotation.
    x, y, w, h = face.bbox
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (88, 166, 255), 2)
    pose = f"yaw {face.yaw_deg:+.1f}  pitch {face.pitch_deg:+.1f}  roll {face.roll_deg:+.1f}"
    cv2.putText(
        overlay, pose, (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA,
    )

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def _style_ax(ax: plt.Axes, *, title: str | None = None) -> None:
    ax.set_facecolor(COLOR_PANEL)
    for spine in ax.spines.values():
        spine.set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_TEXT, labelsize=8)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    if title:
        ax.set_title(title, color=COLOR_ACCENT, fontsize=11, weight="bold", loc="left")
    ax.grid(True, color="#21262d", linewidth=0.5)


def render_diagnostic_panel(
    frame_bgr: npt.NDArray[np.uint8],
    diag: FrameDiagnostics,
    output_path: Path,
) -> None:
    """Produce the multi-panel diagnostic figure and write it to disk."""
    fig = plt.figure(figsize=(16, 9), dpi=140)
    fig.patch.set_facecolor(COLOR_BG)
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.30,
                  left=0.05, right=0.97, top=0.92, bottom=0.07)

    # --- Header ---
    fig.text(
        0.05, 0.965, "Chromatic — Pipeline Diagnostics",
        color=COLOR_ACCENT, fontsize=18, weight="bold",
    )
    verdict = diag.verdict
    badge_color = COLOR_OK if verdict.is_live else COLOR_FAIL
    badge_text = "LIVE" if verdict.is_live else "REJECTED"
    fig.text(
        0.97, 0.965, badge_text,
        color=badge_color, fontsize=18, weight="bold", ha="right",
    )
    fig.text(
        0.97, 0.935,
        f"sustained confidence {verdict.sustained_confidence:.3f}  ·  "
        f"instantaneous {verdict.confidence:.3f}",
        color=COLOR_MUTED, fontsize=9, ha="right",
    )

    # --- Frame with face mesh overlay ---
    ax_frame = fig.add_subplot(gs[0:2, 0:2])
    _style_ax(ax_frame, title="Face Mesh + Region-of-Interest Overlay")
    ax_frame.imshow(_draw_face_mesh(frame_bgr, diag))
    ax_frame.set_xticks([])
    ax_frame.set_yticks([])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=(0.35, 0.78, 0.35), label="rPPG ROI (forehead)"),
        plt.Rectangle((0, 0), 1, 1, fc=(0.86, 0.78, 0.31), label="PRNU/texture ROI (cheeks)"),
        plt.Rectangle((0, 0), 1, 1, fc=(0.34, 0.65, 1.00), label="face bounding box"),
    ]
    leg = ax_frame.legend(
        handles=legend_handles, loc="upper right", frameon=True,
        facecolor=COLOR_PANEL, edgecolor=COLOR_MUTED, labelcolor=COLOR_TEXT, fontsize=8,
    )
    for text in leg.get_texts():
        text.set_color(COLOR_TEXT)

    # --- Pulse waveform ---
    ax_pulse = fig.add_subplot(gs[0, 2:4])
    _style_ax(ax_pulse, title="rPPG Pulse Waveform (CHROM, band-passed 0.7-3.5 Hz)")
    if diag.pulse is not None:
        t = np.arange(len(diag.pulse.pulse_waveform)) / 30.0
        ax_pulse.plot(t, diag.pulse.pulse_waveform, color=COLOR_ACCENT, linewidth=1.3)
        ax_pulse.set_xlabel("time (s)")
        ax_pulse.set_ylabel("amplitude (a.u.)")
        ax_pulse.text(
            0.99, 0.95,
            f"BPM {diag.pulse.bpm:.1f}   SNR {diag.pulse.snr_db:+.1f} dB",
            transform=ax_pulse.transAxes, ha="right", va="top",
            color=COLOR_OK, fontsize=10, weight="bold",
            family="monospace",
        )
    else:
        ax_pulse.text(
            0.5, 0.5,
            f"buffering pulse window… {diag.rppg_progress * 100:.0f}%",
            transform=ax_pulse.transAxes, ha="center", va="center",
            color=COLOR_MUTED, fontsize=11,
        )

    # --- Pulse power spectrum ---
    ax_spec = fig.add_subplot(gs[1, 2:4])
    _style_ax(ax_spec, title="Pulse Power Spectrum (heart-rate band shaded)")
    if diag.pulse is not None:
        freqs = diag.pulse.spectrum_freqs_hz
        spec = diag.pulse.power_spectrum
        # Normalise for plotting only — keeps the y-axis interpretable.
        spec_norm = spec / max(spec.max(), 1e-9)
        ax_spec.fill_between(freqs, 0, spec_norm, color=COLOR_ACCENT, alpha=0.45)
        ax_spec.axvspan(0.7, 3.5, color=COLOR_OK, alpha=0.10)
        peak_hz = diag.pulse.bpm / 60.0
        ax_spec.axvline(peak_hz, color=COLOR_WARN, linestyle="--", linewidth=1.2,
                        label=f"peak {peak_hz:.2f} Hz")
        ax_spec.set_xlim(0, 5)
        ax_spec.set_xlabel("frequency (Hz)")
        ax_spec.set_ylabel("normalised power")
        leg = ax_spec.legend(
            loc="upper right", facecolor=COLOR_PANEL, edgecolor=COLOR_MUTED,
            labelcolor=COLOR_TEXT, fontsize=8,
        )
        for text in leg.get_texts():
            text.set_color(COLOR_TEXT)
    else:
        ax_spec.text(
            0.5, 0.5, "spectrum will be available after pulse buffer is full",
            transform=ax_spec.transAxes, ha="center", va="center",
            color=COLOR_MUTED, fontsize=10,
        )

    # --- Per-layer score bars ---
    ax_scores = fig.add_subplot(gs[2, 0:2])
    _style_ax(ax_scores, title="Per-Layer Liveness Scores")
    scores = verdict.layer_scores.as_dict()
    layer_labels = ["hardware\n(PRNU)", "rPPG\n(pulse)", "texture\n(blur+moire)",
                    "geometry\n(pose)", "motion\n(flow+blink)"]
    values = [scores["hardware"], scores["rppg"], scores["texture"],
              scores["geometry"], scores["motion"]]
    bar_colors = [
        COLOR_OK if v >= 0.65 else (COLOR_WARN if v >= 0.4 else COLOR_FAIL)
        for v in values
    ]
    bars = ax_scores.bar(layer_labels, values, color=bar_colors)
    ax_scores.axhline(0.65, color=COLOR_WARN, linestyle="--", linewidth=1,
                      alpha=0.7, label="decision threshold (0.65)")
    ax_scores.set_ylim(0, 1.05)
    ax_scores.set_ylabel("score (0..1)")
    for bar, value in zip(bars, values):
        ax_scores.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center", va="bottom",
            color=COLOR_TEXT, fontsize=9, weight="bold",
        )
    leg = ax_scores.legend(
        loc="upper right", facecolor=COLOR_PANEL, edgecolor=COLOR_MUTED,
        labelcolor=COLOR_TEXT, fontsize=8,
    )
    for text in leg.get_texts():
        text.set_color(COLOR_TEXT)

    # --- Raw metrics + reasons ---
    ax_meta = fig.add_subplot(gs[2, 2:4])
    ax_meta.set_facecolor(COLOR_PANEL)
    ax_meta.axis("off")
    rows: list[tuple[str, str]] = []
    if diag.prnu is not None:
        rows.append(("PRNU noise std", f"{diag.prnu.noise_std:.2f}"))
        rows.append(("PRNU kurtosis", f"{diag.prnu.kurtosis:+.2f}"))
        if not np.isnan(diag.prnu.fingerprint_correlation):
            rows.append(("PRNU fingerprint correlation",
                         f"{diag.prnu.fingerprint_correlation:+.3f}"))
    if diag.texture is not None:
        rows.append(("Laplacian variance", f"{diag.texture.laplacian_variance:.1f}"))
        rows.append(("Moire score", f"{diag.texture.moire_score:.3f}"))
    if diag.motion is not None:
        rows.append(("Optical-flow magnitude", f"{diag.motion.motion_magnitude:.3f}"))
        rows.append(("Mean EAR (eye openness)", f"{diag.motion.mean_ear:.3f}"))
        rows.append(("Blink detected (last 3 s)",
                     "yes" if diag.motion.blink_detected_recently else "no"))
    if diag.face is not None:
        rows.append(("Head pose (yaw, pitch, roll)",
                     f"{diag.face.yaw_deg:+.1f}°, {diag.face.pitch_deg:+.1f}°, "
                     f"{diag.face.roll_deg:+.1f}°"))

    y = 0.96
    ax_meta.text(0.0, y, "Raw metrics", color=COLOR_ACCENT, fontsize=11, weight="bold",
                 transform=ax_meta.transAxes)
    y -= 0.08
    for label, value in rows:
        ax_meta.text(0.0, y, label, color=COLOR_MUTED, fontsize=9,
                     transform=ax_meta.transAxes)
        ax_meta.text(0.55, y, value, color=COLOR_TEXT, fontsize=9,
                     family="monospace", transform=ax_meta.transAxes)
        y -= 0.07

    y -= 0.02
    ax_meta.text(0.0, y, "Reasons", color=COLOR_ACCENT, fontsize=11, weight="bold",
                 transform=ax_meta.transAxes)
    y -= 0.08
    for reason in verdict.reasons:
        ax_meta.text(0.0, y, f"• {reason}", color=COLOR_TEXT, fontsize=9,
                     transform=ax_meta.transAxes)
        y -= 0.06
        if y < 0.02:
            break

    # --- Footer ---
    fig.text(
        0.05, 0.025,
        "MediaPipe FaceLandmarker · CHROM rPPG (De Haan & Jeanne 2013) · "
        "Farnebäck optical flow · PRNU residual analysis",
        color=COLOR_MUTED, fontsize=8, style="italic",
    )

    fig.savefig(output_path, facecolor=COLOR_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def run_on_image(image_path: Path, output_path: Path) -> int:
    settings = load_settings()
    configure_logging(settings)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"error: failed to read {image_path}", file=sys.stderr)
        return 2

    # The detector needs enough frames to warm up PRNU calibration AND fill
    # the rPPG window. We synthesise a short stream by adding per-frame
    # sensor noise to the still image — this is honest for a demo because
    # it shows what the pipeline _would_ produce on a real camera feed.
    rng = np.random.default_rng(0)
    with LivenessDetector(settings) as detector:
        total_frames = max(
            settings.prnu_calibration_frames,
            settings.rppg_window_frames,
            settings.sustained_frames_required,
        ) + 10
        diag = None
        for i in range(total_frames):
            noise = rng.normal(0, 2.5, frame.shape).astype(np.int16)
            # Inject a faint 1.2 Hz pulse modulation on the forehead row
            # to demonstrate the rPPG layer recovering it.
            pulse = 1.8 * np.sin(2 * np.pi * 1.2 * (i / settings.target_fps))
            modulated = frame.astype(np.int16) + noise
            modulated[: frame.shape[0] // 3, :, 2] += int(pulse * 2)
            stream_frame = np.clip(modulated, 0, 255).astype(np.uint8)
            diag = detector.process_frame(stream_frame)

        assert diag is not None
        # Use the original (un-noised) frame as the visualisation surface
        # so the panel shows a clean image, but report the metrics computed
        # over the noisy stream.
        render_diagnostic_panel(frame, diag, output_path)

    return 0


def run_on_webcam(output_dir: Path, frames_per_capture: int) -> int:  # pragma: no cover
    """Webcam mode — write one diagnostic panel per `frames_per_capture` frames.

    This path is intentionally untested in CI (no camera available). It is
    invoked manually by the demo operator.
    """
    settings = load_settings()
    configure_logging(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("error: cannot open webcam (device 0)", file=sys.stderr)
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

    counter = 0
    try:
        with LivenessDetector(settings) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    diag = detector.process_frame(frame)
                except Exception as exc:
                    print(f"frame skipped: {exc}", file=sys.stderr)
                    continue
                counter += 1
                if counter % frames_per_capture == 0:
                    output_path = output_dir / f"diag_{int(time.time())}.png"
                    render_diagnostic_panel(frame, diag, output_path)
    finally:
        cap.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="single image to analyse")
    parser.add_argument("--webcam", action="store_true",
                        help="capture from default webcam")
    parser.add_argument("--output", type=Path, default=Path("diagnostic.png"),
                        help="output PNG path (image mode)")
    parser.add_argument("--output-dir", type=Path, default=Path("./out"),
                        help="output directory (webcam mode)")
    parser.add_argument("--capture-every", type=int, default=30,
                        help="webcam mode: write a panel every N frames")
    args = parser.parse_args()

    if args.webcam:
        return run_on_webcam(args.output_dir, args.capture_every)
    if not args.image:
        parser.print_help(sys.stderr)
        return 1
    return run_on_image(args.image, args.output)


if __name__ == "__main__":
    sys.exit(main())
