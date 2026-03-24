"""
Inference script: runs the trained model on test tracklets and outputs predictions.json.

Usage:
    python src/predict.py [--data-dir DATA_DIR] [--checkpoint PATH]
                          [--output predictions.json] [--batch-size N] [--img-size N]
                          [--crops-dir DIR]
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import JerseyTestDataset, get_val_transforms, class_to_jersey
from model import load_checkpoint


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
    return p.parse_args()


@torch.no_grad()
def predict_tracklet(model, image_paths, transform, device, batch_size):
    """
    Run the model on all images in a tracklet and return aggregated class probabilities.
    Images are processed in mini-batches to avoid OOM with large tracklets.
    """
    all_probs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        from PIL import Image
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    # Average probabilities across all frames -> majority vote via max
    avg_probs = torch.cat(all_probs, dim=0).mean(dim=0)
    predicted_class = avg_probs.argmax().item()
    return predicted_class


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

    print(f'Running inference on {len(test_dataset)} tracklets...')

    predictions = {}
    for tracklet_id in tqdm(test_dataset.tracklet_ids):
        image_paths = test_dataset.get_image_paths(tracklet_id)
        pred_class = predict_tracklet(model, image_paths, transform, device, args.batch_size)
        jersey_num = class_to_jersey(pred_class)
        predictions[tracklet_id] = jersey_num

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f'Saved {len(predictions)} predictions to {args.output}')

    # Quick summary
    neg1_count = sum(1 for v in predictions.values() if v == -1)
    print(f'Predicted -1 (illegible): {neg1_count} / {len(predictions)}')
    print(f'Predicted valid number:   {len(predictions) - neg1_count} / {len(predictions)}')


if __name__ == '__main__':
    main()
