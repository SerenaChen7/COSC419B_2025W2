"""
Torso localization and cropping module.

Uses MediaPipe Pose Landmarker (Tasks API, mediapipe >= 0.10) to detect
shoulder and hip keypoints, then crops the image to the upper-body / jersey
region. Falls back to a fixed-ratio crop when pose detection fails (common on
very small images).

The MediaPipe model file (~3 MB) is downloaded automatically to
~/.cache/mediapipe/ on first use.

Usage:
    from torso_crop import TorsoCropper

    cropper = TorsoCropper()
    crop, pose_detected = cropper.crop(pil_image)
    cropper.close()

    # or as a context manager:
    with TorsoCropper() as cropper:
        crop, pose_detected = cropper.crop(pil_image)
"""

import os
import urllib.request

import numpy as np
from PIL import Image

import mediapipe as mp

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "mediapipe", "pose_landmarker_lite.task"
)


def _ensure_model() -> str:
    """Download the pose landmarker model if not already cached. Returns path."""
    if not os.path.exists(_MODEL_CACHE):
        os.makedirs(os.path.dirname(_MODEL_CACHE), exist_ok=True)
        print(f"Downloading MediaPipe pose model to {_MODEL_CACHE} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
        print("Download complete.")
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Upscale images shorter than this before running MediaPipe (improves detection
# on the very small player crops typical in this dataset: avg 53x98 px).
UPSCALE_MIN_HEIGHT = 256

# Padding applied around the shoulder-to-hip torso box, expressed as a
# fraction of the torso height (shoulder_y to hip_y distance).
TORSO_TOP_PAD = 0.10   # above the shoulders
TORSO_BOT_PAD = 0.15   # below the hips

# Fixed-ratio fallback crop (fraction of image height) used when pose fails.
FALLBACK_TOP = 0.15
FALLBACK_BOT = 0.65

# Minimum MediaPipe landmark visibility score to accept a keypoint.
MIN_KP_VISIBILITY = 0.4

# MediaPipe PoseLandmarker landmark indices (same as BlazePose / COCO).
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_TORSO_KPS      = [_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]


class TorsoCropper:
    """
    Crops a player image to the torso / jersey region.

    Parameters
    ----------
    min_detection_confidence : float
        Minimum confidence for pose detection. Lower values detect more poses
        but with more false positives. 0.3 works well for small, blurry crops.
    """

    def __init__(self, min_detection_confidence: float = 0.3):
        model_path = _ensure_model()

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop(self, img: Image.Image) -> tuple[Image.Image, bool]:
        """
        Return a torso crop of *img* and whether pose was successfully detected.

        Parameters
        ----------
        img : PIL.Image
            Original player-bounding-box thumbnail (any size / mode).

        Returns
        -------
        cropped : PIL.Image
            The torso-region crop (RGB).
        pose_detected : bool
            True  -> crop was derived from MediaPipe shoulder/hip keypoints.
            False -> fallback fixed-ratio crop was used.
        """
        img = img.convert('RGB')
        W, H = img.size

        # --- Upscale for better MediaPipe detection on tiny images ---
        if H < UPSCALE_MIN_HEIGHT:
            scale = UPSCALE_MIN_HEIGHT / H
            upscaled = img.resize(
                (max(1, round(W * scale)), UPSCALE_MIN_HEIGHT),
                Image.BICUBIC,
            )
        else:
            upscaled = img

        # --- Run MediaPipe ---
        rgb_array = np.array(upscaled, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result = self._landmarker.detect(mp_image)

        if result.pose_landmarks:
            pose_crop = self._pose_crop(img, result.pose_landmarks[0])
            if pose_crop is not None:
                return pose_crop, True

        return self._fallback_crop(img), False

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pose_crop(
        self,
        img: Image.Image,
        landmarks: list,
    ) -> Image.Image | None:
        """
        Derive a torso bounding box from shoulder + hip landmarks.

        Landmark (x, y) values are normalised to [0, 1] relative to the
        image passed to the landmarker (the upscaled one). Since normalised
        coords map to [0,1] regardless of resolution, they are directly
        applied to the *original* image dimensions.

        Returns None if keypoints are low-confidence or the box is degenerate.
        """
        W, H = img.size

        # Check that all four torso keypoints are visible enough.
        for idx in _TORSO_KPS:
            if landmarks[idx].visibility < MIN_KP_VISIBILITY:
                return None

        shoulder_ys = [landmarks[_LEFT_SHOULDER].y * H,
                       landmarks[_RIGHT_SHOULDER].y * H]
        hip_ys      = [landmarks[_LEFT_HIP].y * H,
                       landmarks[_RIGHT_HIP].y * H]

        top_y    = min(shoulder_ys)
        bottom_y = max(hip_ys)
        torso_h  = max(bottom_y - top_y, 1.0)

        y1 = top_y    - TORSO_TOP_PAD * torso_h
        y2 = bottom_y + TORSO_BOT_PAD * torso_h

        # Keep full width — horizontal crops are already tight (avg 53 px).
        x1, x2 = 0, W

        # Clamp to image bounds.
        y1 = max(0, round(y1))
        y2 = min(H, round(y2))

        # Reject degenerate boxes.
        if (y2 - y1) < 10 or (x2 - x1) < 10:
            return None

        return img.crop((x1, y1, x2, y2))

    def _fallback_crop(self, img: Image.Image) -> Image.Image:
        """
        Fixed-ratio crop: keep rows [FALLBACK_TOP, FALLBACK_BOT] of image height,
        full width. Removes head and legs when pose detection fails.
        """
        W, H = img.size
        y1 = round(FALLBACK_TOP * H)
        y2 = round(FALLBACK_BOT * H)
        y2 = max(y2, y1 + 1)  # ensure at least 1 px tall
        return img.crop((0, y1, W, y2))
