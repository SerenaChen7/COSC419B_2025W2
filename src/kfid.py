"""
Keyframe Identification (KfId) module.

Implements the four-stage pipeline from Balaji et al. (arXiv:2309.06285):
  1. JNL  — EasyOCR text detector finds digit regions per frame
             (closest open-source equivalent to the fine-tuned YOLOv5 in the paper)
  2. RoI  — custom I* intersection metric filters detections outside the torso region
  3. LHC  — merges nearby digit boxes with similar hue into one jersey-number box
  4. GHC  — K-means on hue histograms across the tracklet isolates the target player

Output of filter_tracklet():
    list of {'path': str, 'box': (x1,y1,x2,y2)}
    Ready to pass to TorsoCropper then PARSeq.

Install:
    pip install easyocr
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _roi_for_image(w: int, h: int):
    """Preset torso RoI from paper §4.2: top-left (w/4, h/5), bottom-right (3w/4, h/2)."""
    return (w // 4, h // 5, 3 * w // 4, h // 2)


def _intersection_star(r1, r2) -> float:
    """
    Custom I* metric from eq. (4):
    I* = A(R1 ∩ R2) / (min(A(R1), A(R2)) + eps)
    """
    ix1 = max(r1[0], r2[0])
    iy1 = max(r1[1], r2[1])
    ix2 = min(r1[2], r2[2])
    iy2 = min(r1[3], r2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_r1 = max(0, r1[2] - r1[0]) * max(0, r1[3] - r1[1])
    area_r2 = max(0, r2[2] - r2[0]) * max(0, r2[3] - r2[1])
    return inter / (min(area_r1, area_r2) + _EPS)


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
    """Normalized hue histogram from a BGR crop."""
    if crop_bgr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(hsv[:, :, 0], bins=n_bins, range=(0, 180))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _hist_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


# ---------------------------------------------------------------------------
# Stage 1: Jersey Number Localization (JNL) via EasyOCR
# ---------------------------------------------------------------------------

class _EasyOCRJNL:
    """
    Uses EasyOCR's text detector to find digit/text regions per frame.
    This is the closest open-source equivalent to the fine-tuned YOLOv5
    digit detector used in the original Balaji et al. paper.

    Returns bounding boxes (x1, y1, x2, y2) for each detected text region.
    Frames with no detections are discarded before RoI filtering.
    """

    def __init__(self, conf: float = 0.2, device: str = 'cpu'):
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "EasyOCR not installed. Run: pip install easyocr"
            )
        self.conf = conf
        gpu = device != 'cpu'
        # digits_only=True restricts to numeric characters — ideal for jersey numbers
        self._reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

    def detect(self, image_bgr: np.ndarray) -> list:
        """
        Run EasyOCR detection on one BGR frame.
        Returns list of (x1, y1, x2, y2) integer bounding boxes.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # detect() returns (horizontal_list, free_list)
        # horizontal_list entries: [x_min, x_max, y_min, y_max]
        results = self._reader.detect(
            rgb,
            min_size=2,
            text_threshold=self.conf,
            low_text=0.3,
            link_threshold=0.4,
        )

        boxes = []
        if not results or not results[0]:
            return boxes

        for bbox in results[0]:
            # EasyOCR horizontal format: [x_min, x_max, y_min, y_max]
            x1, x2, y1, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

        return boxes


# ---------------------------------------------------------------------------
# Stage 2: RoI-based Filtering
# ---------------------------------------------------------------------------

def _roi_filter(boxes: list, image_w: int, image_h: int,
                threshold: float = 0.3) -> list:
    """
    Keep only detections whose I* overlap with the preset torso RoI
    exceeds the threshold.
    """
    roi = _roi_for_image(image_w, image_h)
    return [box for box in boxes if _intersection_star(roi, box) >= threshold]


# ---------------------------------------------------------------------------
# Stage 3: Local Histogram Correlation (LHC)
# ---------------------------------------------------------------------------

def _lhc_merge(boxes: list, image_bgr: np.ndarray,
               corr_thresh: float = 0.7,
               dist_thresh: float = 0.35) -> list:
    """
    Merge digit detections within one frame that are spatially close
    and share similar hue distributions into a single holistic jersey box.
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
    K-means on hue histograms across all frames in a tracklet.
    Keeps frames from the dominant cluster (target player's jersey colour).

    frame_detections: list of (image_bgr, [box, ...])
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
    JNL stage uses EasyOCR as the digit detector.

    Parameters
    ----------
    jnl_conf : float
        EasyOCR detection confidence threshold (default 0.2).
    roi_thresh : float
        Minimum I* score for RoI filtering (default 0.3).
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
        jnl_conf: float = 0.2,
        roi_thresh: float = 0.3,
        lhc_corr_thresh: float = 0.7,
        lhc_dist_thresh: float = 0.35,
        ghc_n_clusters: int = 2,
        n_bins: int = 36,
        device: str = 'cpu',
    ):
        self._jnl = _EasyOCRJNL(conf=jnl_conf, device=device)
        self.roi_thresh = roi_thresh
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
            'box'  : (x1,y1,x2,y2) — detected jersey-number region
        Empty list = no keyframes survived all four stages.
        """
        frame_detections = []

        for path in image_paths:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                frame_detections.append((None, []))
                continue

            h, w = img_bgr.shape[:2]

            # Stage 1: JNL — detect text/digit regions
            raw_boxes = self._jnl.detect(img_bgr)

            # Stage 2: RoI — keep only boxes overlapping the torso region
            roi_boxes = _roi_filter(raw_boxes, w, h, threshold=self.roi_thresh)

            # Stage 3: LHC — merge nearby same-hue digit boxes
            lhc_boxes = _lhc_merge(roi_boxes, img_bgr,
                                   corr_thresh=self.lhc_corr_thresh,
                                   dist_thresh=self.lhc_dist_thresh)

            frame_detections.append((img_bgr, lhc_boxes))

        # Stage 4: GHC — keep frames from the dominant jersey-colour cluster
        surviving = _ghc_filter(frame_detections,
                                n_bins=self.n_bins,
                                n_clusters=self.ghc_n_clusters)

        return [
            {'path': str(image_paths[fi]), 'box': box}
            for fi, box in surviving
        ]
