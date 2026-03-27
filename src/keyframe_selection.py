from pathlib import Path
import shutil
import cv2


def get_frame_paths(tracklet_dir):
    tracklet_dir = Path(tracklet_dir)
    exts = {".jpg", ".jpeg", ".png"}
    frame_paths = sorted([p for p in tracklet_dir.iterdir() if p.suffix.lower() in exts])
    return frame_paths


def sample_frames(frame_paths, stride=3):
    return frame_paths[::stride]


def score_frame(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return -1.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast = gray.std()

    sharpness_score = min(sharpness / 500.0, 1.0)
    brightness_score = 1.0 - abs(brightness - 140.0) / 140.0
    brightness_score = max(0.0, brightness_score)
    contrast_score = min(contrast / 80.0, 1.0)

    final_score = 0.6 * sharpness_score + 0.2 * brightness_score + 0.2 * contrast_score
    return float(final_score)


def select_keyframes(tracklet_dir, stride=3, top_k=5):
    frame_paths = get_frame_paths(tracklet_dir)
    sampled_paths = sample_frames(frame_paths, stride=stride)

    scored = []
    for path in sampled_paths:
        s = score_frame(path)
        if s >= 0:
            scored.append((path, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [path for path, _ in scored[:top_k]]
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
    output_dir = Path("outputs") / "selected_keyframes" / tracklet_name
    save_selected_keyframes(selected, output_dir)

    print(f"\nSaved selected keyframes to: {output_dir}")