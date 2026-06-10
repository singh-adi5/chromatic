"""
live_dashboard.py — Live webcam dashboard for the Chromatic pipeline.

WHAT THIS IS:
    A real-time OpenCV window that runs the full multi-modal detector against
    a webcam feed (or a recorded video, via --replay) and overlays every signal
    the system uses:

        - Camera feed with the MediaPipe 468-point face mesh + ROI outlines
        - Verdict band (LIVE / REJECT) with live and sustained confidence
        - Per-layer score bars (hardware, rPPG, texture, geometry, motion)
        - rPPG pulse waveform (CHROM, post band-pass)
        - Pulse power spectrum with the heart-rate band shaded
        - EAR (eye-aspect ratio) and confidence time-series
        - Pipeline status: PRNU calibration progress, rPPG buffer fill,
          head pose, throughput, top reasons

WHO THIS IS FOR:
    Engineering reviewers (security architects, ML engineers, hiring managers)
    who want to see exactly what the pipeline measures on a live face — not a
    static colour box on a still image.

USAGE:
    python demo/live_dashboard.py                          # default webcam
    python demo/live_dashboard.py --camera 1               # alternate camera
    python demo/live_dashboard.py --replay test.mp4        # replay a file
    python demo/live_dashboard.py --record out.mp4         # also dump dashboard

KEYBOARD:
    q       quit
    s       save the current dashboard frame to ./dashboard_<ts>.png
    r       reset detector state (after camera move, lighting change, etc.)
    SPACE   pause / resume

The demo deliberately uses OpenCV's window primitives rather than matplotlib's
FuncAnimation because that gives deterministic real-time rendering at 30 fps
without blocking I/O. It is also deliberately separate from the public API
surface — nothing in `src/` depends on this file.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Allow `python demo/live_dashboard.py` from repo root without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromatic import LivenessDetector
from chromatic.config import configure_logging, load_settings
from chromatic.core.detector import FrameDiagnostics

# ─── Visual palette (matches the rest of the project) ────────────────────────
# OpenCV uses BGR. The hex comments are the RGB equivalents.
COLOR_BG     = (13, 17, 23)        # #0d1117
COLOR_PANEL  = (22, 27, 34)        # #161b22
COLOR_GRID   = (48, 54, 61)        # #30363d
COLOR_TEXT   = (217, 209, 201)     # #c9d1d9
COLOR_MUTED  = (158, 148, 139)     # #8b949e
COLOR_ACCENT = (87, 166, 255)      # #ffa657 (BGR of #58a6ff blue accent)
COLOR_OK     = (80, 185, 63)       # #3fb950
COLOR_FAIL   = (73, 81, 248)       # #f85149
COLOR_WARN   = (34, 153, 210)      # #d29922

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_TITLE = "Chromatic — Live Diagnostic Dashboard"

# Suppress chatty audit lines on the live console.
logging.getLogger("chromatic.audit").setLevel(logging.WARNING)
logger = logging.getLogger("chromatic.demo")


# ─── Drawing primitives ───────────────────────────────────────────────────────

def _put(img: np.ndarray, text: str, org: tuple[int, int],
         color: tuple[int, int, int] = COLOR_TEXT, scale: float = 0.5,
         thickness: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(img: np.ndarray, top_left: tuple[int, int],
           size: tuple[int, int], title: str | None = None
           ) -> tuple[int, int, int, int]:
    """Draw a panel; return the inner (x, y, w, h) below any title."""
    x, y = top_left
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_PANEL, thickness=-1)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_GRID, thickness=1)
    inner_y = y + 8
    if title:
        _put(img, title, (x + 12, inner_y + 14),
             color=COLOR_ACCENT, scale=0.5, thickness=1)
        inner_y += 24
    return x + 12, inner_y, w - 24, h - (inner_y - y) - 8


def _bar(img: np.ndarray, top_left: tuple[int, int], size: tuple[int, int],
         value: float, label: str, *, threshold: float = 0.55) -> None:
    """Labelled horizontal progress bar with value displayed inline."""
    x, y = top_left
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_GRID, thickness=-1)
    fill_w = int(max(0.0, min(1.0, value)) * w)
    fill_color = COLOR_OK if value >= threshold else COLOR_FAIL
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, thickness=-1)
    _put(img, label, (x, y - 4), color=COLOR_MUTED, scale=0.42)
    _put(img, f"{value:.2f}", (x + w - 32, y - 4), color=COLOR_TEXT, scale=0.42)


def _waveform(img: np.ndarray, rect: tuple[int, int, int, int],
              data: np.ndarray, color: tuple[int, int, int],
              y_range: tuple[float, float] | None = None) -> None:
    """Plot a 1-D signal inside the given rect."""
    x, y, w, h = rect
    if data.size < 2 or w < 2 or h < 2:
        return
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_BG, thickness=-1)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_GRID, thickness=1)

    if y_range is None:
        lo = float(np.min(data))
        hi = float(np.max(data))
    else:
        lo, hi = y_range
    if hi - lo < 1e-9:
        hi = lo + 1.0

    n = data.size
    xs = np.linspace(x + 1, x + w - 1, n).astype(np.int32)
    ys = (y + h - 1 -
          ((np.asarray(data, dtype=np.float32) - lo) /
           (hi - lo) * (h - 2)).astype(np.int32))
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color,
                  thickness=1, lineType=cv2.LINE_AA)


def _draw_face_overlay(img: np.ndarray, face) -> None:
    """Overlay sparse face mesh and ROI outlines on the camera frame."""
    if face is None:
        return
    # Mesh — subsample to 1/3 of the 468 points for performance.
    if face.landmarks_px is not None:
        for (px, py) in face.landmarks_px[::3]:
            cv2.circle(img, (int(px), int(py)), 1, (170, 170, 170),
                       -1, cv2.LINE_AA)
    # ROI outlines.
    for poly, color in (
        (face.forehead_polygon_px, COLOR_OK),
        (face.left_cheek_polygon_px, COLOR_ACCENT),
        (face.right_cheek_polygon_px, COLOR_ACCENT),
    ):
        if poly is not None and len(poly) >= 3:
            cv2.polylines(img, [np.asarray(poly, np.int32)],
                          isClosed=True, color=color,
                          thickness=2, lineType=cv2.LINE_AA)


# ─── Dashboard history buffers ───────────────────────────────────────────────

@dataclass
class History:
    """Rolling buffers for the dashboard time-series."""

    ear: deque[float]
    confidence: deque[float]
    fps: deque[float]
    pulse_bpm: deque[float]

    @classmethod
    def new(cls, n: int = 300) -> History:
        return cls(
            ear=deque(maxlen=n),
            confidence=deque(maxlen=n),
            fps=deque(maxlen=60),
            pulse_bpm=deque(maxlen=n),
        )


# ─── Dashboard composition ───────────────────────────────────────────────────

def render_dashboard(frame_bgr: np.ndarray, diag: FrameDiagnostics,
                     history: History) -> np.ndarray:
    """Compose the dashboard frame from the current camera frame + diagnostics."""
    H, W = 720, 1280
    canvas = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)

    # ── Left half: live camera feed with overlay ──
    cam_w, cam_h = 760, 560
    cam_x, cam_y = 16, 16
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h),
                  COLOR_PANEL, thickness=-1)
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h),
                  COLOR_GRID, thickness=1)

    # Letterbox the frame into the panel.
    src_h, src_w = frame_bgr.shape[:2]
    inner_w = cam_w - 16
    inner_h = cam_h - 16
    scale = min(inner_w / src_w, inner_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)

    overlay = frame_bgr.copy()
    _draw_face_overlay(overlay, diag.face)
    overlay_resized = cv2.resize(overlay, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)
    inset_x = cam_x + 8 + (inner_w - new_w) // 2
    inset_y = cam_y + 8 + (inner_h - new_h) // 2
    canvas[inset_y:inset_y + new_h, inset_x:inset_x + new_w] = overlay_resized

    # Verdict band over the feed.
    v = diag.verdict
    band_color = COLOR_OK if v.is_live else COLOR_FAIL
    band_text = "LIVE" if v.is_live else "REJECT"
    band_y = cam_y + 8
    cv2.rectangle(canvas, (inset_x, band_y),
                  (inset_x + new_w, band_y + 34),
                  band_color, thickness=-1)
    _put(canvas,
         f"{band_text}    conf {v.confidence:.2f}    "
         f"sustained {v.sustained_confidence:.2f}",
         (inset_x + 12, band_y + 23),
         color=(255, 255, 255), scale=0.7, thickness=2)

    # ── Right column ──
    rx = cam_x + cam_w + 16
    rw = W - rx - 16

    # 1. Layer scores
    x, y, w, _h = _panel(canvas, (rx, 16), (rw, 200),
                         title="Per-layer scores")
    layer = v.layer_scores
    bar_h = 14
    row_gap = 24
    for i, (name, val) in enumerate(layer.as_dict().items()):
        _bar(canvas, (x, y + 16 + i * row_gap), (w, bar_h),
             val, name.upper(), threshold=0.55)

    # 2. Pulse spectrum (more useful than the raw filtered signal —
    #    the dataclass exposes power_spectrum + spectrum_freqs_hz)
    x, y, w, h = _panel(canvas, (rx, 224), (rw, 110),
                        title="rPPG power spectrum (heart-rate band shaded)")
    if (diag.pulse is not None
            and diag.pulse.power_spectrum is not None
            and diag.pulse.power_spectrum.size > 4
            and diag.pulse.spectrum_freqs_hz is not None):
        freqs = diag.pulse.spectrum_freqs_hz
        amps = diag.pulse.power_spectrum
        band_mask = (freqs >= 0.5) & (freqs <= 4.0)
        if band_mask.any():
            band_freqs = freqs[band_mask]
            band_amps = amps[band_mask]
            _waveform(canvas, (x, y, w, h - 4), band_amps,
                      color=COLOR_ACCENT)
            # Shade the physiological band (0.7-3.5 Hz, ~42-210 BPM).
            n = band_freqs.size
            lo_idx = int(np.searchsorted(band_freqs, 0.7) / max(1, n) * w)
            hi_idx = int(np.searchsorted(band_freqs, 3.5) / max(1, n) * w)
            tmp = canvas.copy()
            cv2.rectangle(tmp, (x + lo_idx, y),
                          (x + hi_idx, y + h - 4),
                          COLOR_OK, thickness=-1)
            cv2.addWeighted(tmp, 0.15, canvas, 0.85, 0, canvas)
        _put(canvas,
             f"BPM {diag.pulse.bpm:5.1f}    SNR {diag.pulse.snr_db:+5.1f} dB",
             (x, y + h - 2), color=COLOR_TEXT, scale=0.42)
    else:
        _put(canvas, f"buffering...  {diag.rppg_progress * 100:5.1f}%",
             (x, y + 24), color=COLOR_MUTED, scale=0.45)

    # 3. EAR history (blink detection signal)
    x, y, w, h = _panel(canvas, (rx, 342), (rw, 110),
                        title="Eye-aspect ratio history (blink detection)")
    if history.ear:
        _waveform(canvas, (x, y, w, h - 4),
                  np.asarray(history.ear, dtype=np.float32),
                  color=(108, 184, 255), y_range=(0.0, 0.45))
        latest = history.ear[-1]
        _put(canvas, f"EAR {latest:.3f}", (x, y + h - 2),
             color=COLOR_TEXT, scale=0.42)
    else:
        _put(canvas, "waiting for face...", (x, y + 24),
             color=COLOR_MUTED, scale=0.45)

    # 4. Rolling confidence
    x, y, w, h = _panel(canvas, (rx, 460), (rw, 110),
                        title="Sustained confidence (rolling window)")
    if history.confidence:
        _waveform(canvas, (x, y, w, h - 4),
                  np.asarray(history.confidence, dtype=np.float32),
                  color=COLOR_ACCENT, y_range=(0.0, 1.0))
    else:
        _put(canvas, "no decisions yet", (x, y + 24),
             color=COLOR_MUTED, scale=0.45)

    # 5. Status footer
    x, y, w, _h = _panel(canvas, (rx, 578), (rw, 126),
                         title="Pipeline status")
    _put(canvas, f"PRNU calibration   {diag.calibration_progress * 100:5.1f}%",
         (x, y + 14), color=COLOR_TEXT, scale=0.45)
    _put(canvas, f"rPPG window fill   {diag.rppg_progress * 100:5.1f}%",
         (x, y + 34), color=COLOR_TEXT, scale=0.45)
    if diag.face is not None:
        _put(canvas,
             f"head pose          yaw {diag.face.yaw_deg:+5.1f} deg  "
             f"pitch {diag.face.pitch_deg:+5.1f} deg",
             (x, y + 54), color=COLOR_TEXT, scale=0.45)
    avg_fps = float(np.mean(history.fps)) if history.fps else 0.0
    _put(canvas, f"throughput         {avg_fps:5.1f} fps",
         (x, y + 74), color=COLOR_TEXT, scale=0.45)
    if v.reasons:
        _put(canvas, "reasons: " + ", ".join(v.reasons[:3]),
             (x, y + 94), color=COLOR_MUTED, scale=0.4)

    # Footer
    _put(canvas,
         "Chromatic  |  multi-modal deepfake-resistant verification",
         (16, H - 12), color=COLOR_MUTED, scale=0.42)
    _put(canvas, "q quit   s save   r reset   SPACE pause",
         (W - 340, H - 12), color=COLOR_MUTED, scale=0.42)

    return canvas


# ─── Capture loop ────────────────────────────────────────────────────────────

def _open_source(camera_index: int | None, replay: str | None) -> cv2.VideoCapture:
    if replay:
        cap = cv2.VideoCapture(replay)
    else:
        cap = cv2.VideoCapture(int(camera_index or 0))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source "
            f"({'replay=' + replay if replay else 'camera=' + str(camera_index)})"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(settings)
    if args.log_level:
        logging.getLogger().setLevel(args.log_level)

    cap = _open_source(args.camera, args.replay)
    writer: cv2.VideoWriter | None = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 30.0, (1280, 720))
        if not writer.isOpened():
            logger.warning("Could not open recorder; continuing without recording.")
            writer = None

    history = History.new(n=300)
    headless = args.headless or args.frames is not None
    if not headless:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    paused = False
    t_prev = time.perf_counter()
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_idx = 0
    exit_code = 0

    with LivenessDetector(settings) as detector:
        try:
            while True:
                if not paused:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        logger.info("End of stream at frame %d.", frame_idx)
                        break

                    diag = detector.process_frame(frame)

                    if diag.face is not None:
                        history.ear.append(float(diag.face.mean_eye_aspect_ratio))
                    if diag.pulse is not None:
                        history.pulse_bpm.append(float(diag.pulse.bpm))
                    history.confidence.append(
                        float(diag.verdict.sustained_confidence))

                    t_now = time.perf_counter()
                    dt = t_now - t_prev
                    if dt > 0:
                        history.fps.append(1.0 / dt)
                    t_prev = t_now

                    canvas = render_dashboard(frame, diag, history)
                    frame_idx += 1

                    if args.frames is not None and frame_idx >= args.frames:
                        if args.snapshot_out:
                            cv2.imwrite(args.snapshot_out, canvas)
                            logger.info("Saved snapshot to %s", args.snapshot_out)
                        break

                if writer is not None and not paused:
                    writer.write(canvas)
                if headless:
                    continue

                cv2.imshow(WINDOW_TITLE, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    fname = f"dashboard_{int(time.time())}.png"
                    cv2.imwrite(fname, canvas)
                    logger.info("Saved %s", fname)
                elif key == ord("r"):
                    detector.reset()
                    history = History.new(n=300)
                    logger.info("Detector state reset.")
                elif key == 32:  # SPACE
                    paused = not paused

        except KeyboardInterrupt:
            logger.info("Interrupted.")
            exit_code = 130
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()

    return exit_code


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chromatic — live dashboard")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0). Ignored if --replay is given.")
    p.add_argument("--replay", type=str, default=None,
                   help="Replay a video file instead of using a camera.")
    p.add_argument("--record", type=str, default=None,
                   help="Optional path to record the dashboard as an .mp4.")
    p.add_argument("--headless", action="store_true",
                   help="Don't open a window (useful for CI and screenshots).")
    p.add_argument("--frames", type=int, default=None,
                   help="Process only N frames then exit. Pairs with "
                        "--headless and --snapshot-out for smoke tests.")
    p.add_argument("--snapshot-out", type=str, default=None,
                   help="Path to save the final dashboard frame as a PNG.")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
