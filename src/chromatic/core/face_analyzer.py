"""
Face analysis using MediaPipe Face Mesh.

This module is the geometric foundation for the rest of the pipeline. We use
MediaPipe's 468-landmark face mesh (Apache 2.0 licence) for:

- High-fidelity face detection
- Forehead region of interest (for rPPG)
- Cheek region of interest (for PRNU/texture analysis)
- Head pose estimation
- Eye Aspect Ratio (EAR) for blink detection

Landmark indices were selected from the canonical MediaPipe topology and are
documented inline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from chromatic.exceptions import FaceNotFoundError

logger = logging.getLogger(__name__)

# Default location of the MediaPipe face_landmarker.task model file.
# Operators can override via CHROMATIC_FACE_MODEL_PATH.
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"

# MediaPipe Face Mesh landmark indices (from the canonical 468-point topology).
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

# Forehead — top of face, generally hairline-clear, well perfused.
_FOREHEAD_LANDMARKS: Final = [10, 67, 69, 109, 108, 151, 337, 338, 297, 299]

# Left and right cheek — flat regions, good for noise analysis.
_LEFT_CHEEK_LANDMARKS: Final = [50, 101, 36, 205, 187, 207]
_RIGHT_CHEEK_LANDMARKS: Final = [280, 330, 266, 425, 411, 427]

# Eyes — used for Eye Aspect Ratio (blink detection).
# Order: outer, top1, top2, inner, bottom2, bottom1
_LEFT_EYE_LANDMARKS: Final = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_LANDMARKS: Final = [362, 385, 387, 263, 373, 380]


@dataclass(frozen=True)
class FaceAnalysis:
    """Result of face analysis on a single frame."""

    landmarks_px: npt.NDArray[np.float32]      # (468, 2) in image pixel coords
    bbox: tuple[int, int, int, int]            # (x, y, w, h)
    forehead_mask: npt.NDArray[np.uint8]       # boolean mask of forehead pixels
    left_cheek_mask: npt.NDArray[np.uint8]
    right_cheek_mask: npt.NDArray[np.uint8]
    # Polygon vertices used to build the masks. Useful for overlay rendering
    # (drawing the ROI outlines on top of the camera feed).
    forehead_polygon_px: npt.NDArray[np.int32]
    left_cheek_polygon_px: npt.NDArray[np.int32]
    right_cheek_polygon_px: npt.NDArray[np.int32]
    left_eye_aspect_ratio: float
    right_eye_aspect_ratio: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    @property
    def mean_eye_aspect_ratio(self) -> float:
        return (self.left_eye_aspect_ratio + self.right_eye_aspect_ratio) / 2.0

    @property
    def is_frontal(self) -> bool:
        """True if the face is roughly frontal (|yaw|, |pitch| < 20°)."""
        return abs(self.yaw_deg) < 20.0 and abs(self.pitch_deg) < 20.0


class FaceAnalyzer:
    """Wrapper over MediaPipe Face Mesh.

    The MediaPipe graph is created once per analyzer instance to amortise the
    setup cost. Instances are NOT thread-safe; create one per worker.
    """

    def __init__(
        self,
        *,
        model_path: str | os.PathLike[str] | None = None,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved = (
            Path(model_path)
            if model_path is not None
            else Path(os.environ.get("CHROMATIC_FACE_MODEL_PATH", _DEFAULT_MODEL_PATH))
        )
        if not resolved.is_file():
            raise FileNotFoundError(
                f"MediaPipe face_landmarker model not found at {resolved}. "
                "Run scripts/download_models.sh or set CHROMATIC_FACE_MODEL_PATH."
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release the underlying MediaPipe graph. Safe to call multiple times."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore[assignment]

    def __enter__(self) -> FaceAnalyzer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def analyze(self, frame_bgr: npt.NDArray[np.uint8]) -> FaceAnalysis:
        """Run face analysis on a single BGR frame.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR image (OpenCV convention).

        Returns:
            FaceAnalysis with landmarks, ROI masks, EAR, and head pose.

        Raises:
            FaceNotFoundError: No face detected with sufficient confidence.
        """
        # MediaPipe Tasks expects an mp.Image wrapping RGB data.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise FaceNotFoundError("no face detected")

        h, w = frame_bgr.shape[:2]
        landmarks_norm = result.face_landmarks[0]
        # Convert from normalised [0,1] to pixel coordinates.
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in landmarks_norm], dtype=np.float32
        )

        bbox = self._compute_bbox(landmarks_px, w, h)
        forehead_mask, forehead_poly = self._region_mask_and_polygon(
            landmarks_px, _FOREHEAD_LANDMARKS, (h, w))
        left_cheek_mask, left_cheek_poly = self._region_mask_and_polygon(
            landmarks_px, _LEFT_CHEEK_LANDMARKS, (h, w))
        right_cheek_mask, right_cheek_poly = self._region_mask_and_polygon(
            landmarks_px, _RIGHT_CHEEK_LANDMARKS, (h, w))

        l_ear = self._eye_aspect_ratio(landmarks_px[_LEFT_EYE_LANDMARKS])
        r_ear = self._eye_aspect_ratio(landmarks_px[_RIGHT_EYE_LANDMARKS])

        yaw, pitch, roll = self._head_pose(landmarks_px, (h, w))

        return FaceAnalysis(
            landmarks_px=landmarks_px,
            bbox=bbox,
            forehead_mask=forehead_mask,
            left_cheek_mask=left_cheek_mask,
            right_cheek_mask=right_cheek_mask,
            forehead_polygon_px=forehead_poly,
            left_cheek_polygon_px=left_cheek_poly,
            right_cheek_polygon_px=right_cheek_poly,
            left_eye_aspect_ratio=l_ear,
            right_eye_aspect_ratio=r_ear,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    # --- internals --------------------------------------------------------

    @staticmethod
    def _compute_bbox(
        landmarks_px: npt.NDArray[np.float32], w: int, h: int
    ) -> tuple[int, int, int, int]:
        x_min = int(max(0, landmarks_px[:, 0].min()))
        y_min = int(max(0, landmarks_px[:, 1].min()))
        x_max = int(min(w - 1, landmarks_px[:, 0].max()))
        y_max = int(min(h - 1, landmarks_px[:, 1].max()))
        return x_min, y_min, x_max - x_min, y_max - y_min

    @staticmethod
    def _region_mask_and_polygon(
        landmarks_px: npt.NDArray[np.float32],
        indices: list[int],
        shape_hw: tuple[int, int],
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32]]:
        """Build a filled mask and return the polygon used to fill it.

        Returning the polygon avoids re-computing it for rendering layers (the
        live dashboard draws the polygon outline as the ROI marker).
        """
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = landmarks_px[indices].astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask, polygon

    @staticmethod
    def _eye_aspect_ratio(eye_landmarks: npt.NDArray[np.float32]) -> float:
        """Soukupová & Čech 2016 — Eye Aspect Ratio.

        Lower values indicate a closed eye. Typical open-eye EAR ~ 0.3,
        closed-eye EAR < 0.2.
        """
        # Landmark order: 0=outer, 1=top1, 2=top2, 3=inner, 4=bottom2, 5=bottom1
        v1 = float(np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]))
        v2 = float(np.linalg.norm(eye_landmarks[2] - eye_landmarks[4]))
        h = float(np.linalg.norm(eye_landmarks[0] - eye_landmarks[3]))
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    @staticmethod
    def _head_pose(
        landmarks_px: npt.NDArray[np.float32], shape_hw: tuple[int, int]
    ) -> tuple[float, float, float]:
        """Estimate (yaw, pitch, roll) in degrees using solvePnP.

        We use a 6-point subset of canonical 3D face landmarks:
        nose tip, chin, left/right eye outer corner, left/right mouth corner.
        """
        h, w = shape_hw
        # 6 image points
        image_points = np.array(
            [
                landmarks_px[1],     # nose tip
                landmarks_px[152],   # chin
                landmarks_px[33],    # left eye outer corner
                landmarks_px[263],   # right eye outer corner
                landmarks_px[61],    # left mouth corner
                landmarks_px[291],   # right mouth corner
            ],
            dtype=np.float64,
        )
        # Corresponding 3D model points (millimetres, approximate adult skull)
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),         # nose tip
                (0.0, -63.6, -12.5),     # chin
                (-43.3, 32.7, -26.0),    # left eye outer corner
                (43.3, 32.7, -26.0),     # right eye outer corner
                (-28.9, -28.9, -24.1),   # left mouth corner
                (28.9, -28.9, -24.1),    # right mouth corner
            ],
            dtype=np.float64,
        )
        # Approximate intrinsics — fine for pose categorisation.
        focal = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal, 0.0, center[0]],
                [0.0, focal, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1))
        ok, rvec, _tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose into Euler angles
        sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
        if sy > 1e-6:
            pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
            yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
            yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
            roll = 0.0
        return yaw, pitch, roll
