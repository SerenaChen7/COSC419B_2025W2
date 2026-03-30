"""
Keyframe selection for jersey number recognition.

Scores frames by sharpness, brightness, and contrast, then deduplicates
near-identical frames so the selected pool covers distinct moments.

Usage:
    from keyframe_selection import select_keyframes
    selected, scored = select_keyframes(tracklet_dir, stride=3, top_k=10)
"""
from pathlib import Path
import cv2
import numpy as np


def _get_frame_paths(tracklet_dir):
    exts = {'.jpg', '.jpeg', '.png'}
    return sorted(p for p in Path(tracklet_dir).iterdir() if p.suffix.lower() in exts)


def _extract_metrics(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        'path':       image_path,
        'sharpness':  float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'brightness': float(gray.mean()),
        'contrast':   float(gray.std()),
    }


def _minmax(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _is_similar(p1, p2, threshold=10):
    i1 = cv2.imread(str(p1))
    i2 = cv2.imread(str(p2))
    if i1 is None or i2 is None:
        return False
    i1 = cv2.resize(i1, (64, 64))
    i2 = cv2.resize(i2, (64, 64))
    return np.mean(np.abs(i1.astype(float) - i2.astype(float))) < threshold


def select_keyframes(tracklet_dir, stride=3, top_k=5):
    """
    Return the top_k highest-quality, visually distinct frames from a tracklet.

    Parameters
    ----------
    tracklet_dir : str or Path
    stride       : sample every Nth frame before scoring (reduces cost)
    top_k        : maximum number of frames to return

    Returns
    -------
    selected : list[Path]   – up to top_k frames in time order
    scored   : list[tuple]  – all (path, score) pairs, sorted by score desc
    """
    frame_paths = _get_frame_paths(tracklet_dir)
    sampled = frame_paths[::stride]

    metrics = [m for m in (_extract_metrics(p) for p in sampled) if m is not None]
    if not metrics:
        return [], []

    sharpness_scores  = _minmax([m['sharpness']  for m in metrics])
    contrast_scores   = _minmax([m['contrast']   for m in metrics])
    brightness_vals   = [m['brightness'] for m in metrics]
    center            = float(np.median(brightness_vals))
    max_dist          = max(abs(v - center) for v in brightness_vals) or 1.0
    brightness_scores = [1.0 - abs(v - center) / max_dist for v in brightness_vals]

    scored = sorted(
        [
            (m['path'], 0.6 * sharpness_scores[i]
                       + 0.2 * brightness_scores[i]
                       + 0.2 * contrast_scores[i])
            for i, m in enumerate(metrics)
        ],
        key=lambda x: x[1], reverse=True,
    )

    selected = []
    for path, score in scored:
        if len(selected) >= top_k:
            break
        if all(not _is_similar(path, s) for s in selected):
            selected.append(path)

    return sorted(selected), scored  # time order
