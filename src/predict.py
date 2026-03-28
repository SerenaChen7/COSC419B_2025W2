"""
Inference script: runs the trained model on test tracklets and outputs predictions.json.

Usage:
    python src/predict.py [--data-dir DATA_DIR] [--checkpoint PATH]
                          [--output predictions.json] [--batch-size N] [--img-size N]
                          [--crops-dir DIR] [--no-keyframes]
                          [--keyframe-stride N] [--keyframe-top-k N]
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import JerseyTestDataset, get_val_transforms, class_to_jersey
from model import load_checkpoint
from keyframe_selection import sample_frames, extract_frame_metrics, minmax_normalize, is_similar
from consolidate import consolidate_tracklet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/jersey-2023', help='Root dataset directory')
    p.add_argument('--checkpoint', default='outputs/best_model.pth')
    p.add_argument('--output', default='predictions.json')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Images per batch within a tracklet')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--crops-dir', default=None,
                   help='Path to pre-computed torso crops (output of preprocess_crops.py). '
                        'If set, images are loaded from here instead of images/.')
    p.add_argument('--no-keyframes', action='store_true',
                   help='Disable keyframe selection; run inference on all frames.')
    p.add_argument('--keyframe-stride', type=int, default=3,
                   help='Frame sampling stride for keyframe selection (default: 3).')
    p.add_argument('--keyframe-top-k', type=int, default=5,
                   help='Maximum keyframes to select per tracklet (default: 5).')
    p.add_argument('--tta', action='store_true',
                   help='Test-time augmentation: average predictions over original and '
                        'horizontally flipped image for each frame.')
    p.add_argument('--conf-per-frame', type=float, default=0.15,
                   help='Per-frame confidence contribution required for a valid prediction. '
                        'Threshold = num_frames * conf_per_frame. '
                        'Lower values predict fewer illegible tracklets (default: 0.15).')
    return p.parse_args()


def filter_keyframes(image_paths, stride=3, top_k=5):
    """
    Apply keyframe selection to an already-resolved list of image paths.

    Mirrors the v3 logic from keyframe_selection.select_keyframes() but
    accepts paths directly instead of a directory, so it works correctly
    when crops_dir fallback produces a mix of paths from different locations.

    Returns the selected subset (up to top_k, in time order), or the
    original list unchanged if selection yields no results.
    """
    sampled = sample_frames(image_paths, stride=stride)

    metrics = []
    for path in sampled:
        m = extract_frame_metrics(path)
        if m is not None:
            metrics.append(m)

    if not metrics:
        return image_paths

    sharpness_vals = [m["sharpness"] for m in metrics]
    contrast_vals = [m["contrast"] for m in metrics]
    brightness_vals = [m["brightness"] for m in metrics]

    sharpness_scores = minmax_normalize(sharpness_vals)
    contrast_scores = minmax_normalize(contrast_vals)

    brightness_center = np.median(brightness_vals)
    brightness_distances = [abs(v - brightness_center) for v in brightness_vals]
    if max(brightness_distances) == 0:
        brightness_scores = [1.0] * len(brightness_distances)
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

    selected = []
    for path, _ in scored:
        if len(selected) >= top_k:
            break
        if all(not is_similar(path, existing) for existing in selected):
            selected.append(path)

    if not selected:
        return image_paths

    selected.sort()  # restore time order
    return selected


@torch.no_grad()
def predict_frames(model, image_paths, transform, device, batch_size, use_tta=False):
    """
    Run the model on a list of images and return per-frame (jersey_str, confidence) pairs.

    Each pair contains the predicted jersey number as a string (e.g. "7", "14", or "-1"
    for illegible) and the top softmax probability as the confidence score.
    Illegible frames ("-1") are automatically filtered by consolidate_tracklet().

    If use_tta=True, each frame is also run through the model horizontally flipped and
    the two softmax distributions are averaged before taking the argmax. The model is
    trained with RandomHorizontalFlip so it is broadly flip-invariant, meaning the same
    jersey number should win under both orientations; averaging reduces prediction noise.
    """
    from PIL import Image

    frame_predictions = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)
        probs = F.softmax(model(batch), dim=1)

        if use_tta:
            batch_flip = torch.flip(batch, dims=[3])  # horizontal flip
            probs_flip = F.softmax(model(batch_flip), dim=1)
            probs = (probs + probs_flip) / 2

        top_probs, top_classes = probs.max(dim=1)
        for cls, conf in zip(top_classes.cpu().tolist(), top_probs.cpu().tolist()):
            jersey_num = class_to_jersey(cls)
            frame_predictions.append((str(jersey_num), conf))

    return frame_predictions


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_checkpoint(args.checkpoint, device)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    test_images_dir = os.path.join(args.data_dir, 'test', 'images')
    transform = get_val_transforms(args.img_size)
    test_dataset = JerseyTestDataset(test_images_dir, transform=transform,
                                     crops_dir=args.crops_dir)

    use_keyframes = not args.no_keyframes
    print(f'Running inference on {len(test_dataset)} tracklets | '
          f'keyframes: {"on" if use_keyframes else "off"} | '
          f'TTA: {"on" if args.tta else "off"} | '
          f'conf-per-frame: {args.conf_per_frame}')

    predictions = {}
    for tracklet_id in tqdm(test_dataset.tracklet_ids):
        image_paths = test_dataset.get_image_paths(tracklet_id)

        if use_keyframes:
            image_paths = filter_keyframes(
                image_paths,
                stride=args.keyframe_stride,
                top_k=args.keyframe_top_k,
            )

        frame_preds = predict_frames(model, image_paths, transform, device, args.batch_size,
                                     use_tta=args.tta)
        adaptive_threshold = max(1, len(frame_preds)) * args.conf_per_frame
        jersey_num = consolidate_tracklet(frame_preds, confidence_threshold=adaptive_threshold)
        predictions[tracklet_id] = jersey_num

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f'Saved {len(predictions)} predictions to {args.output}')

    neg1_count = sum(1 for v in predictions.values() if v == -1)
    print(f'Predicted -1 (illegible): {neg1_count} / {len(predictions)}')
    print(f'Predicted valid number:   {len(predictions) - neg1_count} / {len(predictions)}')


if __name__ == '__main__':
    main()
