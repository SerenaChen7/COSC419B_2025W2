# V1 uses hardcoded constants
# v2 normalizes sharpness and contrast based on the sampled frames in that tracklet
from pathlib import Path
import shutil
import cv2
import numpy as np


def get_frame_paths(tracklet_dir):
    tracklet_dir = Path(tracklet_dir)
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in tracklet_dir.iterdir() if p.suffix.lower() in exts])


def sample_frames(frame_paths, stride=3):
    return frame_paths[::stride]


def extract_frame_metrics(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast = gray.std()

    return {
        "path": image_path,
        "sharpness": float(sharpness),
        "brightness": float(brightness),
        "contrast": float(contrast),
    }


def minmax_normalize(values):
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def select_keyframes(tracklet_dir, stride=3, top_k=5):
    frame_paths = get_frame_paths(tracklet_dir)
    sampled_paths = sample_frames(frame_paths, stride=stride)

    metrics = []
    for path in sampled_paths:
        m = extract_frame_metrics(path)
        if m is not None:
            metrics.append(m)

    if not metrics:
        return [], []

    sharpness_vals = [m["sharpness"] for m in metrics]
    contrast_vals = [m["contrast"] for m in metrics]
    brightness_vals = [m["brightness"] for m in metrics]

    sharpness_scores = minmax_normalize(sharpness_vals)
    contrast_scores = minmax_normalize(contrast_vals)

    # brightness: prefer values near the tracklet median
    brightness_center = np.median(brightness_vals)
    brightness_distances = [abs(v - brightness_center) for v in brightness_vals]

    if max(brightness_distances) == 0:
        brightness_scores = [1.0 for _ in brightness_distances]
    else:
        brightness_scores = [1.0 - (d / max(brightness_distances)) for d in brightness_distances]

    scored = []
    for i, m in enumerate(metrics):
        final_score = (
            0.6 * sharpness_scores[i] +
            0.2 * brightness_scores[i] +
            0.2 * contrast_scores[i]
        )
        scored.append((m["path"], float(final_score)))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected = [path for path, _ in scored[:top_k]]
    selected.sort()  # preserve time order

    return selected, scored


def save_selected_keyframes(selected_paths, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in selected_paths:
        shutil.copy(path, output_dir / path.name)


if __name__ == "__main__":
    tracklet_dir = input("Enter tracklet folder path: ").strip()
    selected, scored = select_keyframes(tracklet_dir, stride=3, top_k=5)

    print("\nTop selected frames:")
    for path in selected:
        print(path)

    print("\nTop 10 scored frames:")
    for path, score in scored[:10]:
        print(f"{path.name}: {score:.4f}")

    tracklet_name = Path(tracklet_dir).name
    # output_dir = Path("outputs") / "selected_keyframes" / tracklet_name
    output_dir = Path("outputs") / "selected_keyframes_v2" / tracklet_name
    save_selected_keyframes(selected, output_dir)

    print(f"\nSaved selected keyframes to: {output_dir}")