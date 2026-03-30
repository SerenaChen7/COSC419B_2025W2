"""
Inference script: runs the trained classifier on test tracklets and writes predictions.json.

Usage:
    python src/predict.py [--data-dir DIR] [--checkpoint PATH]
                          [--output PATH] [--img-size N] [--batch-size N]
                          [--keyframes] [--gt PATH]

Pipeline:
    1. (Optional) Keyframe selection — pick sharp, high-contrast, distinct frames
    2. Classifier inference          — per-frame softmax probabilities
    3. Consolidation                 — confidence-weighted vote across frames,
                                       with 1-digit down-weighting when 2-digit
                                       predictions exist in the same tracklet
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_val_transforms, class_to_jersey
from model import load_checkpoint
from keyframe_selection import select_keyframes
from consolidate import consolidate_tracklet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',   default='data/jersey-2023')
    p.add_argument('--checkpoint', default='outputs/best_model.pth')
    p.add_argument('--output',     default='outputs/predictions.json')
    p.add_argument('--img-size',   type=int, default=128)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--keyframes',  action='store_true',
                   help='Use keyframe selection to pick best frames per tracklet.')
    p.add_argument('--kf-stride',  type=int, default=3)
    p.add_argument('--kf-top-k',   type=int, default=15)
    p.add_argument('--gt',         default=None,
                   help='Ground truth JSON — prints accuracy after inference.')
    return p.parse_args()


@torch.no_grad()
def predict_tracklet(model, image_paths, transform, device, batch_size):
    """
    Run the classifier on all frames and return per-frame (jersey_str, confidence)
    pairs for legible predictions, ready for consolidation.
    """
    frame_predictions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack([
            transform(Image.open(p).convert('RGB')) for p in batch_paths
        ]).to(device, non_blocking=True)

        probs = F.softmax(model(imgs).float(), dim=1).cpu()

        for frame_probs in probs:
            pred_class = frame_probs.argmax().item()
            jersey_num = class_to_jersey(pred_class)
            if jersey_num != -1:
                frame_predictions.append((str(jersey_num), frame_probs[pred_class].item()))

    return frame_predictions


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_checkpoint(args.checkpoint, device)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    transform = get_val_transforms(args.img_size)
    test_images_dir = os.path.join(args.data_dir, 'test', 'images')

    tracklet_ids = sorted(
        tid for tid in os.listdir(test_images_dir)
        if os.path.isdir(os.path.join(test_images_dir, tid))
    )

    mode = 'keyframes' if args.keyframes else 'all frames'
    print(f'Running inference on {len(tracklet_ids)} tracklets ({mode})...')

    predictions = {}
    for tid in tqdm(tracklet_ids):
        tracklet_dir = os.path.join(test_images_dir, tid)

        if args.keyframes:
            selected, _ = select_keyframes(tracklet_dir,
                                           stride=args.kf_stride,
                                           top_k=args.kf_top_k)
            image_paths = [str(p) for p in selected]
            if not image_paths:
                image_paths = [
                    os.path.join(tracklet_dir, f)
                    for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')
                ]
        else:
            image_paths = [
                os.path.join(tracklet_dir, f)
                for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')
            ]

        if not image_paths:
            predictions[tid] = -1
            continue

        frame_preds = predict_tracklet(model, image_paths, transform, device, args.batch_size)
        predictions[tid] = consolidate_tracklet(frame_preds)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    neg1 = sum(1 for v in predictions.values() if v == -1)
    print(f'Saved {len(predictions)} predictions → {args.output}')
    print(f'Illegible (-1): {neg1} | Valid: {len(predictions) - neg1}')

    if args.gt:
        with open(args.gt) as f:
            gt = json.load(f)
        total = correct = 0
        for tid, gt_num in gt.items():
            if tid not in predictions:
                continue
            total   += 1
            correct += predictions[tid] == gt_num
        print(f'Accuracy: {correct}/{total} = {100 * correct / total:.1f}%')


if __name__ == '__main__':
    main()
