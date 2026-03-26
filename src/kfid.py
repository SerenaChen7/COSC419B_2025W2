"""
Keyframe Identification (KfId) module.

Implements the four-stage pipeline from Balaji et al. (arXiv:2309.06285),
with the JNL stage replaced by the pre-trained legibility classifier from
Koshkina & Elder (already available in models/).

Stages:
  1. JNL  — legibility classifier (ResNet-34 + sigmoid) scores each frame;
             frames below the threshold are discarded.
             Box = preset torso RoI (w/4, h/5) -> (3w/4, h/2) for kept frames.
  2. RoI  — baked into stage 1 (preset RoI is the detection box).
  3. LHC  — merges nearby detections with similar hue within a frame.
             No-op when each frame has exactly one detection box.
  4. GHC  — K-means on hue histograms of RoI crops across the tracklet;
             keeps frames belonging to the dominant cluster (target player).

Output of filter_tracklet():
    list of {'path': str, 'box': (x1,y1,x2,y2)}
    Ready to pass to TorsoCropper, then to PARSeq.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-7

_LEGIBILITY_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _roi_for_image(w: int, h: int):
    """Preset torso RoI from paper §4.2: top-left (w/4, h/5), bottom-right (3w/4, h/2)."""
    return (w // 4, h // 5, 3 * w // 4, h // 2)


def _box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _center_dist(b1, b2) -> float:
    c1, c2 = _box_center(b1), _box_center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _merge_boxes(boxes):
    return (
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    )


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _hue_histogram(crop_bgr: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """Normalized hue histogram from a BGR crop (hue 0-180 in OpenCV)."""
    if crop_bgr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    hist, _ = np.histogram(hue, bins=n_bins, range=(0, 180))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _hist_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


# ---------------------------------------------------------------------------
# Stage 1: JNL via Legibility Classifier
# ---------------------------------------------------------------------------

class _LegibilityJNL:
    """
    Scores each frame with the pre-trained ResNet-34 legibility classifier.
    Architecture matches LegibilityClassifier34 from the original codebase:
        ResNet-34 -> Linear(512, 1) -> Sigmoid
    Frames with score >= threshold are kept; their box = preset torso RoI.
    """

    def __init__(self, model_path: str, threshold: float = 0.5,
                 device: str = 'cpu'):
        self.threshold = threshold
        self.device = torch.device(device)
        self._model = self._load(model_path)

    def _load(self, model_path: str) -> nn.Module:
        model = tv_models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)

        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def filter_frames(self, image_paths: list, batch_size: int = 16) -> list:
        """
        Score all frames in a tracklet and return kept (path, box) pairs.
        Box is the preset torso RoI for each kept image.
        """
        kept = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            pil_imgs, valid_paths = [], []
            for p in batch_paths:
                try:
                    img = Image.open(str(p)).convert('RGB')
                    pil_imgs.append(img)
                    valid_paths.append(p)
                except (OSError, FileNotFoundError):
                    continue

            if not pil_imgs:
                continue

            tensors = torch.stack([_LEGIBILITY_TRANSFORM(img) for img in pil_imgs])
            tensors = tensors.to(self.device)
            scores = torch.sigmoid(self._model(tensors)).squeeze(1).cpu().tolist()

            for path, img, score in zip(valid_paths, pil_imgs, scores):
                if score >= self.threshold:
                    w, h = img.size
                    kept.append((str(path), _roi_for_image(w, h)))

        return kept


# ---------------------------------------------------------------------------
# Stage 3: Local Histogram Correlation (LHC)
# ---------------------------------------------------------------------------

def _lhc_merge(boxes: list, image_bgr: np.ndarray,
               corr_thresh: float = 0.7,
               dist_thresh: float = 0.35) -> list:
    """
    Merge nearby detections with similar hue into one holistic box.
    With the legibility JNL (one box per frame) this is a no-op.
    """
    if len(boxes) <= 1:
        return boxes

    h, w = image_bgr.shape[:2]
    pixel_dist = dist_thresh * w
    histograms = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        histograms.append(_hue_histogram(crop))

    n = len(boxes)
    merged = [False] * n
    groups = []
    for i in range(n):
        if merged[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if merged[j]:
                continue
            if (_center_dist(boxes[i], boxes[j]) < pixel_dist and
                    _hist_correlation(histograms[i], histograms[j]) >= corr_thresh):
                group.append(j)
                merged[j] = True
        merged[i] = True
        groups.append(group)

    return [_merge_boxes([boxes[idx] for idx in g]) for g in groups]


# ---------------------------------------------------------------------------
# Stage 4: Global Histogram Correlation (GHC)
# ---------------------------------------------------------------------------

def _ghc_filter(frame_detections: list, n_bins: int = 36,
                n_clusters: int = 2) -> list:
    """
    K-means on hue histograms across all frames.
    Keeps frames from the dominant cluster (target player's jersey colour).

    frame_detections: list of (image_bgr, [box])
    Returns: list of (frame_idx, box)
    """
    records = []
    for fi, (image_bgr, boxes) in enumerate(frame_detections):
        for bi, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            records.append((fi, bi, _hue_histogram(crop, n_bins=n_bins)))

    if not records:
        return []

    if len(records) < n_clusters:
        return [(fi, frame_detections[fi][1][0])
                for fi, (_, boxes) in enumerate(frame_detections) if boxes]

    X = np.stack([r[2] for r in records])
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)
    dominant = int(np.argmax(np.bincount(labels)))

    frame_to_box = {}
    for (fi, bi, _), label in zip(records, labels):
        if label == dominant and fi not in frame_to_box:
            frame_to_box[fi] = frame_detections[fi][1][bi]

    return sorted(frame_to_box.items())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class KfId:
    """
    Keyframe Identification module (Balaji et al., arXiv:2309.06285).

    Parameters
    ----------
    legibility_model : str
        Path to legibility_resnet34_soccer_*.pth checkpoint.
    legibility_threshold : float
        Min sigmoid score to keep a frame (default 0.5).
    lhc_corr_thresh : float
        Hue histogram correlation for LHC merging (default 0.7).
    lhc_dist_thresh : float
        Max center distance for LHC as fraction of image width (default 0.35).
    ghc_n_clusters : int
        K-means clusters for GHC (default 2, one per team).
    n_bins : int
        Hue histogram bins (default 36).
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        legibility_model: str = 'models/legibility_resnet34_soccer_20240215.pth',
        legibility_threshold: float = 0.5,
        lhc_corr_thresh: float = 0.7,
        lhc_dist_thresh: float = 0.35,
        ghc_n_clusters: int = 2,
        n_bins: int = 36,
        device: str = 'cpu',
    ):
        self._jnl = _LegibilityJNL(
            model_path=legibility_model,
            threshold=legibility_threshold,
            device=device,
        )
        self.lhc_corr_thresh = lhc_corr_thresh
        self.lhc_dist_thresh = lhc_dist_thresh
        self.ghc_n_clusters = ghc_n_clusters
        self.n_bins = n_bins

    def filter_tracklet(self, image_paths: list) -> list:
        """
        Run the full KfId pipeline on one player tracklet.

        Parameters
        ----------
        image_paths : list of str or Path

        Returns
        -------
        list of dict:
            'path' : str            — image file path
            'box'  : (x1,y1,x2,y2) — torso RoI box on the original image
        Empty list = no keyframes survived all stages.
        """
        # Stage 1: legibility filter
        kept = self._jnl.filter_frames(image_paths)
        if not kept:
            return []

        # Stages 3 + 4: LHC then GHC on surviving frames
        frame_detections = []
        for path_str, box in kept:
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                frame_detections.append((None, []))
                continue
            lhc_boxes = _lhc_merge([box], img_bgr,
                                   corr_thresh=self.lhc_corr_thresh,
                                   dist_thresh=self.lhc_dist_thresh)
            frame_detections.append((img_bgr, lhc_boxes))

        surviving = _ghc_filter(frame_detections,
                                n_bins=self.n_bins,
                                n_clusters=self.ghc_n_clusters)

        return [{'path': kept[fi][0], 'box': box} for fi, box in surviving]
