"""
Inference script: runs the trained model on test tracklets and outputs predictions.json.

Usage:
    python src/predict.py [--data-dir DATA_DIR] [--checkpoint PATH]
                          [--output predictions.json] [--batch-size N] [--img-size N]
                          [--crops-dir DIR] [--use-keyframes] [--keyframe-top-k N]
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
    p.add_argument('--tta', action='store_true',
                   help='Test-time augmentation: average predictions over multiple augmented views')
    p.add_argument('--max-images', type=int, default=None,
                   help='Max frames to sample per tracklet (evenly spaced). '
                        'Ignored when --use-keyframes is set.')
    p.add_argument('--use-keyframes', action='store_true',
                   help='Use quality-based keyframe selection (sharpness/brightness/diversity) '
                        'instead of evenly-spaced frame sampling.')
    p.add_argument('--keyframe-top-k', type=int, default=5,
                   help='Number of keyframes to select per tracklet (used with --use-keyframes).')
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
    Run the model on all images in a tracklet.

    Returns a (N, num_classes) tensor of per-frame averaged probabilities,
    where N is the number of input frames. TTA passes are averaged per-frame
    so each row represents the mean prediction for that specific frame.
    """
    from PIL import Image

    all_transforms = [transform] + (tta_transforms or [])
    pil_images = [Image.open(p).convert('RGB') for p in image_paths]
    n_frames = len(pil_images)
    accumulated_probs = None

    for t in all_transforms:
        t_probs = []
        for i in range(0, n_frames, batch_size):
            batch = torch.stack([t(img) for img in pil_images[i:i + batch_size]]).to(device)
            logits = model(batch)
            t_probs.append(F.softmax(logits, dim=1).cpu())
        t_probs_cat = torch.cat(t_probs, dim=0)  # (N, num_classes)
        accumulated_probs = t_probs_cat if accumulated_probs is None else accumulated_probs + t_probs_cat

    return accumulated_probs / len(all_transforms)  # (N, num_classes)


def select_image_paths(tracklet_id, test_images_dir, crops_dir, args):
    """
    Return the list of image paths to run inference on for one tracklet.

    Priority:
      1. --use-keyframes: quality-based selection on original images, then
         remap to crop paths if --crops-dir is set.
      2. --max-images: evenly-spaced sampling from the full path list.
      3. Default: all frames (from crops_dir if set, else images_dir).
    """
    orig_tracklet_dir = Path(test_images_dir) / tracklet_id

    if args.use_keyframes:
        from keyframe_selection import select_keyframes
        # Always score quality on the original (unprocessed) images so that
        # sharpness/brightness metrics aren't affected by crop-padding artefacts.
        selected_orig, _ = select_keyframes(
            str(orig_tracklet_dir), stride=3, top_k=args.keyframe_top_k
        )
        if not selected_orig:
            # Fallback: use all frames if keyframe selection yields nothing
            selected_orig = list(orig_tracklet_dir.iterdir())

        if crops_dir:
            # Remap selected filenames to crop paths, falling back to originals.
            image_paths = []
            for orig_path in selected_orig:
                crop_path = Path(crops_dir) / tracklet_id / orig_path.name
                image_paths.append(str(crop_path) if crop_path.exists() else str(orig_path))
        else:
            image_paths = [str(p) for p in selected_orig]
        return image_paths

    # Build full path list (crops if available, else originals)
    exts = {'.jpg', '.jpeg', '.png'}
    all_orig = sorted(p for p in orig_tracklet_dir.iterdir() if p.suffix.lower() in exts)
    if crops_dir:
        image_paths = []
        for p in all_orig:
            crop_path = Path(crops_dir) / tracklet_id / p.name
            image_paths.append(str(crop_path) if crop_path.exists() else str(p))
    else:
        image_paths = [str(p) for p in all_orig]

    if args.max_images and len(image_paths) > args.max_images:
        # Evenly spaced so we cover the full clip, not just the start
        indices = [int(i * len(image_paths) / args.max_images) for i in range(args.max_images)]
        image_paths = [image_paths[i] for i in indices]

    return image_paths


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

    # We only need tracklet_ids from the dataset; path resolution is handled by
    # select_image_paths so that keyframe selection and crops_dir interact correctly.
    test_dataset = JerseyTestDataset(test_images_dir, transform=transform,
                                     crops_dir=args.crops_dir)

    if args.use_keyframes:
        print(f'Keyframe selection enabled (top_k={args.keyframe_top_k})')
    print(f'Running inference on {len(test_dataset)} tracklets'
          f'{" with TTA" if args.tta else ""}...')

    predictions = {}
    for tracklet_id in tqdm(test_dataset.tracklet_ids):
        image_paths = select_image_paths(
            tracklet_id, test_images_dir, args.crops_dir, args
        )

        # Per-frame averaged probabilities: shape (N, num_classes)
        frame_probs = predict_tracklet(
            model, image_paths, transform, device, args.batch_size, tta_transforms
        )

        # Convert to (jersey_str, confidence) pairs for consolidation.
        # Illegible frames (class 0) are excluded — consolidate handles
        # the overall illegibility decision via its confidence threshold.
        frame_predictions = []
        for prob in frame_probs:
            pred_class = prob.argmax().item()
            confidence = prob.max().item()
            jersey_num = class_to_jersey(pred_class)
            if jersey_num != -1:
                frame_predictions.append((str(jersey_num), float(confidence)))

        # Confidence-weighted majority vote (Koshkina & Elder, CVPRW 2024)
        jersey_num = consolidate_tracklet(frame_predictions)
        predictions[tracklet_id] = jersey_num

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f'Saved {len(predictions)} predictions to {args.output}')

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
