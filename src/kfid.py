"""
Keyframe Identification (KfId) module.

Implements the four-stage pipeline from Balaji et al. (arXiv:2309.06285):
  1. Jersey Number Localization (JNL)  — YOLOv5 digit detector
  2. RoI-based Filtering               — custom I* intersection metric (eq. 4)
  3. Local Histogram Correlation (LHC) — merge nearby same-hue digit detections
  4. Global Histogram Correlation (GHC)— K-means to isolate the target player

Usage:
    from kfid import KfId

    kfid = KfId(jnl_weights='path/to/finetuned.pt')  # or 'yolov5s' for base
    keyframes = kfid.filter_tracklet(image_paths)
    # keyframes: list of {'path': str, 'box': (x1,y1,x2,y2)} dicts
    # empty list means no usable keyframes found for this tracklet

    # Pass just paths if you only want the filtered paths:
    filtered_paths = [kf['path'] for kf in keyframes]
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _roi_for_image(w: int, h: int):
    """Preset torso RoI as described in §4.2: top-left (w/4, h/5), bottom-right (3w/4, h/2)."""
    return (w // 4, h // 5, 3 * w // 4, h // 2)


def _intersection_star(r1, r2) -> float:
    """
    Custom intersection metric I* from equation (4).
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
    """Bounding box that contains all given boxes."""
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
    """Normalized hue histogram from a BGR crop (hue range 0-180 in OpenCV)."""
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
    """OpenCV histogram correlation in [-1, 1]; 1 = identical."""
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


# ---------------------------------------------------------------------------
# Stage 1: Jersey Number Localization (JNL)
# ---------------------------------------------------------------------------

class _JNL:
    """
    Wraps a YOLOv5 model to detect digit/jersey-number regions.

    weights: path to a fine-tuned YOLOv5 checkpoint OR a model name string
             (e.g. 'yolov5s') to load the base COCO model via torch.hub.
             Note: the base model is not trained on digits — fine-tuned weights
             are required for meaningful detections on the SoccerNet dataset.
    """

    def __init__(self, weights: str = 'yolov5s', conf: float = 0.25,
                 device: str = 'cpu'):
        try:
            import torch
            weights_path = Path(weights)
            if weights_path.exists():
                self._model = torch.hub.load(
                    'ultralytics/yolov5', 'custom',
                    path=str(weights_path), device=device, verbose=False
                )
            else:
                self._model = torch.hub.load(
                    'ultralytics/yolov5', weights,
                    device=device, verbose=False
                )
            self._model.conf = conf
        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLOv5 model '{weights}'. "
                "Install with: pip install ultralytics\n"
                f"Original error: {exc}"
            )

    def detect(self, image_bgr: np.ndarray) -> list:
        """
        Run inference on a BGR image.
        Returns list of (x1, y1, x2, y2) integer bounding boxes.
        """
        import torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._model(rgb)
        boxes = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            boxes.append((x1, y1, x2, y2))
        return boxes


# ---------------------------------------------------------------------------
# Stage 2: RoI-based Filtering
# ---------------------------------------------------------------------------

def _roi_filter(boxes: list, image_w: int, image_h: int,
                threshold: float = 0.3) -> list:
    """
    Keep only detections whose I* overlap with the preset torso RoI exceeds
    the threshold. Returns the surviving subset of boxes.
    """
    roi = _roi_for_image(image_w, image_h)
    kept = []
    for box in boxes:
        score = _intersection_star(roi, box)
        if score >= threshold:
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# Stage 3: Local Histogram Correlation (LHC)
# ---------------------------------------------------------------------------

def _lhc_merge(boxes: list, image_bgr: np.ndarray,
               corr_thresh: float = 0.7,
               dist_thresh: float = 0.35) -> list:
    """
    Merge digit detections within the same frame that are spatially close
    and share similar hue distributions.

    dist_thresh is expressed as a fraction of image width so it scales with
    image size (paper uses fixed-size 150×120 crops).

    Returns a list of merged bounding boxes (one per holistic jersey number).
    """
    if not boxes:
        return []

    h, w = image_bgr.shape[:2]
    pixel_dist_thresh = dist_thresh * w

    # Compute hue histogram for each detection crop
    histograms = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        histograms.append(_hue_histogram(crop))

    n = len(boxes)
    merged = [False] * n
    groups = []  # each group = list of box indices to merge

    for i in range(n):
        if merged[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if merged[j]:
                continue
            close = _center_dist(boxes[i], boxes[j]) < pixel_dist_thresh
            similar_hue = _hist_correlation(histograms[i], histograms[j]) >= corr_thresh
            if close and similar_hue:
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
    Cluster hue histograms across all frames in a tracklet (K-means).
    Keep only the frames that have at least one detection belonging to
    the dominant cluster (the one with the most detections).

    frame_detections: list of (image_bgr, boxes) — one entry per frame,
                      where boxes is the output of the LHC stage.

    Returns a list of (frame_index, box) pairs for the surviving keyframes,
    using the first box from the dominant cluster for each frame.
    """
    # Collect (frame_idx, box_idx, histogram) for every detection
    records = []  # (frame_idx, box_local_idx, hist)
    for fi, (image_bgr, boxes) in enumerate(frame_detections):
        for bi, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            hist = _hue_histogram(crop, n_bins=n_bins)
            records.append((fi, bi, hist))

    if not records:
        return []

    if len(records) < n_clusters:
        # Too few detections to cluster — keep everything
        result = []
        for fi, (_, boxes) in enumerate(frame_detections):
            if boxes:
                result.append((fi, boxes[0]))
        return result

    X = np.stack([r[2] for r in records])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Dominant cluster = the one with the most detections
    counts = np.bincount(labels)
    dominant = int(np.argmax(counts))

    # Collect the frame indices that have at least one detection in the
    # dominant cluster; use the first such box as the representative crop
    frame_to_box = {}
    for (fi, bi, _), label in zip(records, labels):
        if label == dominant and fi not in frame_to_box:
            frame_to_box[fi] = frame_detections[fi][1][bi]

    return sorted(frame_to_box.items())  # list of (frame_idx, box)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class KfId:
    """
    Keyframe Identification module (Balaji et al., arXiv:2309.06285).

    Parameters
    ----------
    jnl_weights : str
        Path to fine-tuned YOLOv5 weights (.pt) or a model name string
        ('yolov5s', 'yolov5m', …) for the base COCO model.
    jnl_conf : float
        YOLOv5 confidence threshold (default 0.25).
    roi_thresh : float
        Minimum I* score for a detection to pass the RoI filter (default 0.3).
    lhc_corr_thresh : float
        Minimum hue-histogram correlation for LHC merging (default 0.7).
    lhc_dist_thresh : float
        Maximum center distance for LHC merging, as a fraction of image
        width (default 0.35).
    ghc_n_clusters : int
        Number of K-means clusters in the GHC stage (default 2 — one per team).
    n_bins : int
        Number of histogram bins for hue (default 36; covers 0-180 in 5° steps).
    device : str
        Torch device string for JNL ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        jnl_weights: str = 'yolov5s',
        jnl_conf: float = 0.25,
        roi_thresh: float = 0.3,
        lhc_corr_thresh: float = 0.7,
        lhc_dist_thresh: float = 0.35,
        ghc_n_clusters: int = 2,
        n_bins: int = 36,
        device: str = 'cpu',
    ):
        self._jnl = _JNL(weights=jnl_weights, conf=jnl_conf, device=device)
        self.roi_thresh = roi_thresh
        self.lhc_corr_thresh = lhc_corr_thresh
        self.lhc_dist_thresh = lhc_dist_thresh
        self.ghc_n_clusters = ghc_n_clusters
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Per-frame stages (exposed for unit testing / debugging)
    # ------------------------------------------------------------------

    def jnl(self, image_bgr: np.ndarray) -> list:
        """Stage 1: detect digit bounding boxes in one frame."""
        return self._jnl.detect(image_bgr)

    def roi_filter(self, boxes: list, image_bgr: np.ndarray) -> list:
        """Stage 2: discard detections outside the torso RoI."""
        h, w = image_bgr.shape[:2]
        return _roi_filter(boxes, w, h, threshold=self.roi_thresh)

    def lhc(self, boxes: list, image_bgr: np.ndarray) -> list:
        """Stage 3: merge nearby same-hue digit detections."""
        return _lhc_merge(
            boxes, image_bgr,
            corr_thresh=self.lhc_corr_thresh,
            dist_thresh=self.lhc_dist_thresh,
        )

    # ------------------------------------------------------------------
    # Full tracklet pipeline
    # ------------------------------------------------------------------

    def filter_tracklet(self, image_paths: list) -> list:
        """
        Run the full KfId pipeline on a player tracklet.

        Parameters
        ----------
        image_paths : list of str or Path
            Ordered list of frame image paths belonging to one tracklet.

        Returns
        -------
        list of dict, each with keys:
            'path'  : str  — original image path
            'box'   : (x1, y1, x2, y2) — representative jersey-number crop box
        An empty list means no keyframes passed all four stages.
        """
        # Load images (BGR) and run JNL + RoI + LHC per frame
        frame_detections = []  # (image_bgr, post-lhc boxes)
        loaded_images = []

        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                frame_detections.append((None, []))
                loaded_images.append(None)
                continue

            raw_boxes = self.jnl(img)
            roi_boxes = self.roi_filter(raw_boxes, img)
            lhc_boxes = self.lhc(roi_boxes, img)
            frame_detections.append((img, lhc_boxes))
            loaded_images.append(img)

        # Stage 4: GHC — filter across the tracklet
        surviving = _ghc_filter(
            frame_detections,
            n_bins=self.n_bins,
            n_clusters=self.ghc_n_clusters,
        )

        keyframes = []
        for frame_idx, box in surviving:
            keyframes.append({
                'path': str(image_paths[frame_idx]),
                'box': box,
            })

        return keyframes
