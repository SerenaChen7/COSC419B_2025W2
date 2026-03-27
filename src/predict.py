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
    p.add_argument('--tta', action='store_true',
                   help='Test-time augmentation: average predictions over multiple augmented views')
    p.add_argument('--max-images', type=int, default=None,
                   help='Max frames to sample per tracklet. Evenly spaced to cover the full clip.')
    p.add_argument('--gt', default=None,
                   help='Path to ground truth JSON. If provided, prints accuracy after inference.')
    return p.parse_args()


def get_tta_transforms(img_size):
    """
    Three additional views per image: brighter, darker, higher contrast.
    Geometric flips are excluded — they mirror digits and hurt number recognition.
    """
    import torchvision.transforms as T
    norm = [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    resize = T.Resize((img_size, img_size))
    # Single extra pass with contrast boost — sharpens digit edges, most useful
    # augmentation for jersey number reading without doubling inference cost.
    return [
        T.Compose([resize, T.ColorJitter(contrast=(1.3, 1.6)), *norm]),
    ]


@torch.no_grad()
def predict_tracklet(model, image_paths, transform, device, batch_size, tta_transforms=None):
    """
    Run the model on all images in a tracklet and return aggregated class probabilities.
    Images are processed in mini-batches to avoid OOM with large tracklets.
    If tta_transforms is provided, predictions are averaged over all transforms.
    """
    from PIL import Image

    all_transforms = [transform] + (tta_transforms or [])
    all_probs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        # Load each image once, then apply every transform to the same PIL image
        pil_images = [Image.open(p).convert('RGB') for p in batch_paths]

        for t in all_transforms:
            batch = torch.stack([t(img) for img in pil_images]).to(device)
            logits = model(batch)
            all_probs.append(F.softmax(logits, dim=1).cpu())

    # Average over all frames and TTA passes, then pick best class
    predicted_class = torch.cat(all_probs, dim=0).mean(dim=0).argmax().item()
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
    tta_transforms = get_tta_transforms(args.img_size) if args.tta else None
    test_dataset = JerseyTestDataset(test_images_dir, transform=transform,
                                     crops_dir=args.crops_dir)

    print(f'Running inference on {len(test_dataset)} tracklets'
          f'{" with TTA" if args.tta else ""}...')

    predictions = {}
    for tracklet_id in tqdm(test_dataset.tracklet_ids):
        image_paths = test_dataset.get_image_paths(tracklet_id)
        if args.max_images and len(image_paths) > args.max_images:
            # Evenly spaced sample so we cover the full clip, not just the start
            indices = [int(i * len(image_paths) / args.max_images) for i in range(args.max_images)]
            image_paths = [image_paths[i] for i in indices]
        pred_class = predict_tracklet(model, image_paths, transform, device,
                                      args.batch_size, tta_transforms)
        jersey_num = class_to_jersey(pred_class)
        predictions[tracklet_id] = jersey_num

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f'Saved {len(predictions)} predictions to {args.output}')

    # Quick summary
    neg1_count = sum(1 for v in predictions.values() if v == -1)
    print(f'Predicted -1 (illegible): {neg1_count} / {len(predictions)}')
    print(f'Predicted valid number:   {len(predictions) - neg1_count} / {len(predictions)}')

    if args.gt:
        with open(args.gt) as f:
            gt = json.load(f)

        total = correct = 0
        legible_total = legible_correct = 0
        illegible_total = illegible_correct = 0

        for tid, gt_num in gt.items():
            if tid not in predictions:
                continue
            pred_num = predictions[tid]
            total += 1
            correct += pred_num == gt_num

            if gt_num == -1:
                illegible_total += 1
                illegible_correct += pred_num == -1
            else:
                legible_total += 1
                legible_correct += pred_num == gt_num

        print()
        print('=== Evaluation ===')
        print(f'Overall accuracy:    {correct}/{total} = {100*correct/total:.1f}%')
        if legible_total:
            print(f'Legible accuracy:    {legible_correct}/{legible_total} = {100*legible_correct/legible_total:.1f}%')
        if illegible_total:
            print(f'Illegible accuracy:  {illegible_correct}/{illegible_total} = {100*illegible_correct/illegible_total:.1f}%')


if __name__ == '__main__':
    main()
