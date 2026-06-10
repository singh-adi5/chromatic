"""
Motion-based liveness signals.

Defends against:
    - Static photo presentation attacks: no inter-frame motion at all.
    - Looped video attacks: motion is too uniform / too periodic.
    - Mask attacks: motion has the wrong spatial distribution (rigid bulk
      motion of a mask vs. expected non-rigid face motion).

Metrics:
    - motion_magnitude:    mean of dense optical flow magnitude over the face.
    - motion_variability:  spatial std of the flow magnitude. Real faces have
      non-uniform motion (mouth, eyes, micro-expressions); rigid masks and
      static photos have very uniform (or zero) motion.

References:
    G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion",
    Scandinavian Conference on Image Analysis, 2003.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionMetrics:
    motion_magnitude: float
    motion_variability: float
    # Mean EAR across a short history — useful for blink detection.
    mean_ear: float
    blink_detected_recently: bool


class MotionAnalyzer:
    """Frame-to-frame motion analyser.

    Maintains a small history of grayscale frames and EAR values to compute
    optical flow and blink detection across short windows.
    """

    def __init__(self, *, ear_blink_threshold: float = 0.20, ear_history: int = 90) -> None:
        self._prev_gray: npt.NDArray[np.uint8] | None = None
        self._ear_history: deque[float] = deque(maxlen=ear_history)
        self._blink_threshold = ear_blink_threshold

    def reset(self) -> None:
        self._prev_gray = None
        self._ear_history.clear()

    def analyze(
        self,
        frame_bgr: npt.NDArray[np.uint8],
        face_bbox: tuple[int, int, int, int],
        ear: float,
    ) -> MotionMetrics:
        """Compute motion and blink metrics.

        Args:
            frame_bgr: Full BGR frame.
            face_bbox: Bounding box (x, y, w, h) of the face.
            ear: Eye Aspect Ratio for the current frame.
        """
        self._ear_history.append(ear)
        mean_ear = float(np.mean(self._ear_history)) if self._ear_history else 0.0
        blink = self._detect_blink_in_history()

        x, y, w, h = face_bbox
        # Tight crop to face to make optical flow tractable.
        h_img, w_img = frame_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        face_crop = frame_bgr[y:y2, x:x2]
        if face_crop.size == 0:
            return MotionMetrics(0.0, 0.0, mean_ear, blink)
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != face_gray.shape:
            self._prev_gray = face_gray
            return MotionMetrics(0.0, 0.0, mean_ear, blink)

        # Farnebäck dense optical flow.
        flow = cv2.calcOpticalFlowFarneback(
            prev=self._prev_gray,
            next=face_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(magnitude.mean())
        std_mag = float(magnitude.std())

        self._prev_gray = face_gray
        return MotionMetrics(
            motion_magnitude=mean_mag,
            motion_variability=std_mag,
            mean_ear=mean_ear,
            blink_detected_recently=blink,
        )

    def _detect_blink_in_history(self) -> bool:
        """A blink is a brief dip in EAR below threshold within the history."""
        if len(self._ear_history) < 5:
            return False
        ears = np.asarray(self._ear_history, dtype=np.float32)
        # At least one sample below the threshold AND a high baseline above it.
        return bool(
            (ears.min() < self._blink_threshold) and (ears.max() > self._blink_threshold + 0.05)
        )
