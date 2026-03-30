"""
Torso localisation and cropping module  —  improved edition.

Key improvements over v1:
  • Relaxed visibility threshold (0.25 → was 0.4) so low-confidence
    keypoints on blurry / small crops are not rejected outright.
  • Partial-keypoint fallback: if only shoulders (but not hips) are
    visible, the crop height is estimated from shoulder width, which
    works well for the common case of a partially-occluded player.
  • Adjusted fallback band (0.10-0.75) keeps more of the jersey area,
    especially the number printed on the lower chest.
  • TORSO_BOT_PAD reduced (0.05) so we don't pull in irrelevant leg area.
  • Upscale threshold raised to 320 px; higher resolution → better
    landmark detection on tiny player crops.

Usage (unchanged):
    from torso_crop import TorsoCropper

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
    if not os.path.exists(_MODEL_CACHE):
        os.makedirs(os.path.dirname(_MODEL_CACHE), exist_ok=True)
        print(f"Downloading MediaPipe pose model to {_MODEL_CACHE} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
        print("Download complete.")
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Upscale images shorter than this before running MediaPipe.
# Higher → better detection on the tiny player crops in this dataset.
UPSCALE_MIN_HEIGHT = 320

# Padding around the shoulder-to-hip box as a fraction of torso height.
TORSO_TOP_PAD = 0.10   # above shoulders (keep head/collar area)
TORSO_BOT_PAD = 0.05   # below hips  (reduced – avoids pulling in legs)


# Minimum MediaPipe visibility score to accept a keypoint.
# Relaxed from 0.4 → 0.25 for blurry / small / partly-occluded players.
MIN_KP_VISIBILITY = 0.25

# Shoulder-only mode: if hips are not visible but shoulders are, estimate
# torso height as a multiple of shoulder width.
SHOULDER_ONLY_HEIGHT_MULT = 1.8   # torso_h ≈ shoulder_width × 1.8

# MediaPipe BlazePose landmark indices
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_SHOULDER_KPS   = [_LEFT_SHOULDER, _RIGHT_SHOULDER]
_HIP_KPS        = [_LEFT_HIP, _RIGHT_HIP]
_TORSO_KPS      = _SHOULDER_KPS + _HIP_KPS


class TorsoCropper:
    """
    Crops a player image to the torso / jersey region.

    Parameters
    ----------
    min_detection_confidence : float
        Minimum confidence for pose detection.
    """

    def __init__(self, min_detection_confidence: float = 0.25):
        model_path = _ensure_model()

        PoseLandmarker      = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        BaseOptions         = mp.tasks.BaseOptions
        VisionRunningMode   = mp.tasks.vision.RunningMode

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
        Return (torso_crop, pose_detected).

        pose_detected=True  → crop from MediaPipe shoulder/hip keypoints.
        pose_detected=False → fallback fixed-ratio crop.
        """
        img = img.convert('RGB')
        W, H = img.size

        # Upscale for better landmark detection on tiny images
        if H < UPSCALE_MIN_HEIGHT:
            scale    = UPSCALE_MIN_HEIGHT / H
            upscaled = img.resize(
                (max(1, round(W * scale)), UPSCALE_MIN_HEIGHT),
                Image.BICUBIC,
            )
        else:
            upscaled = img

        rgb_array = np.array(upscaled, dtype=np.uint8)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result    = self._landmarker.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            pose_crop = self._pose_crop(img, lm)
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

    def _visible(self, landmarks, indices: list[int]) -> bool:
        return all(landmarks[i].visibility >= MIN_KP_VISIBILITY for i in indices)

    def _pose_crop(self, img: Image.Image, landmarks: list) -> Image.Image | None:
        """
        Derive a torso bounding box from shoulder + hip landmarks.

        Falls back to shoulder-only mode when hips are not confidently detected.
        Returns None when keypoints are too low-confidence or the box is degenerate.
        """
        W, H = img.size

        shoulders_ok = self._visible(landmarks, _SHOULDER_KPS)
        hips_ok      = self._visible(landmarks, _HIP_KPS)

        if not shoulders_ok:
            return None  # can't do anything useful without shoulders

        # ── Shoulder y-coordinates ────────────────────────────────────────────
        left_sh_y  = landmarks[_LEFT_SHOULDER].y  * H
        right_sh_y = landmarks[_RIGHT_SHOULDER].y * H
        left_sh_x  = landmarks[_LEFT_SHOULDER].x  * W
        right_sh_x = landmarks[_RIGHT_SHOULDER].x * W
        top_y      = min(left_sh_y, right_sh_y)
        shoulder_width = abs(left_sh_x - right_sh_x)

        if hips_ok:
            # Full torso box
            bottom_y = max(
                landmarks[_LEFT_HIP].y  * H,
                landmarks[_RIGHT_HIP].y * H,
            )
        else:
            # Estimate torso height from shoulder width
            est_torso_h = max(shoulder_width * SHOULDER_ONLY_HEIGHT_MULT, 20.0)
            bottom_y    = top_y + est_torso_h

        torso_h = max(bottom_y - top_y, 1.0)
        y1 = top_y    - TORSO_TOP_PAD * torso_h
        y2 = bottom_y + TORSO_BOT_PAD * torso_h

        # Full width (horizontal crops are already tight)
        x1, x2 = 0, W

        y1 = max(0,  round(y1))
        y2 = min(H,  round(y2))

        if (y2 - y1) < 10 or (x2 - x1) < 10:
            return None

        return img.crop((x1, y1, x2, y2))

    def _fallback_crop(self, img: Image.Image) -> Image.Image:
        """Return the full image when pose detection fails.
        The jersey number is somewhere in the full frame; a wrong fixed-ratio
        crop is more harmful than keeping the full context."""
        return img