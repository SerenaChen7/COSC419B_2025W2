"""
KfId + Torso Crop batch preprocessing script.

For every tracklet in a dataset split:
  1. KfId filtering  — legibility classifier keeps only frames where the
                       jersey number is likely visible
  2. Torso Crop      — MediaPipe crops each keyframe to the jersey region

Saves crops to:
    data/SoccerNet/<split>/kfid_crops/<tracklet_id>/<filename>.jpg

Also writes a JSON summary:
    data/SoccerNet/<split>/kfid_crops/summary.json
    {tracklet_id: {"total": N, "keyframes": M, "files": [...]}}

These crops are ready to pass directly to PARSeq.

Usage:
    python src/preprocess_kfid_crops.py --split test
    python src/preprocess_kfid_crops.py --split challenge --device cuda
    python src/preprocess_kfid_crops.py --split test --legibility-threshold 0.4
"""

import os
import sys
import json
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from kfid import KfId
from torso_crop import TorsoCropper


def parse_args():
    p = argparse.ArgumentParser(
        description='Run KfId filtering + torso cropping on a dataset split.'
    )
    p.add_argument('--data-dir', default='data/SoccerNet',
                   help='Root dataset directory (default: data/SoccerNet)')
    p.add_argument('--split', default='test',
                   choices=['train', 'test', 'challenge'],
                   help='Split to process (default: test)')
    p.add_argument('--jnl-conf', type=float, default=0.2,
                   help='EasyOCR detection confidence threshold (default: 0.2)')
    p.add_argument('--roi-thresh', type=float, default=0.3,
                   help='RoI I* threshold (default: 0.3)')
    p.add_argument('--ghc-clusters', type=int, default=2,
                   help='K-means clusters for GHC stage (default: 2)')
    p.add_argument('--device', default='cpu',
                   help='Device for EasyOCR: cpu / cuda / mps')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-process tracklets that already have crops')
    return p.parse_args()


def get_device(requested: str) -> str:
    import torch
    if requested == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if requested == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f'Device: {device}')

    images_dir = Path(args.data_dir) / args.split / 'images'
    crops_dir  = Path(args.data_dir) / args.split / 'kfid_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise FileNotFoundError(f'Images directory not found: {images_dir}')

    # Initialise KfId (EasyOCR loads once, shared across all tracklets)
    print('Loading EasyOCR detector...')
    kfid = KfId(
        jnl_conf=args.jnl_conf,
        roi_thresh=args.roi_thresh,
        ghc_n_clusters=args.ghc_clusters,
        device=device,
    )

    print('Loading TorsoCropper...')
    cropper = TorsoCropper()

    tracklet_ids = sorted(
        d for d in os.listdir(images_dir)
        if (images_dir / d).is_dir()
    )
    print(f'Found {len(tracklet_ids)} tracklets in {args.split} split.\n')

    summary = {}
    total_frames = 0
    total_keyframes = 0

    for tid in tqdm(tracklet_ids, desc='Tracklets'):
        out_dir = crops_dir / tid

        # Skip if already processed and not overwriting
        if not args.overwrite and out_dir.is_dir() and any(out_dir.iterdir()):
            # Load existing summary entry if present
            existing_summary = crops_dir / 'summary.json'
            if existing_summary.exists():
                with open(existing_summary) as f:
                    saved = json.load(f)
                if tid in saved:
                    summary[tid] = saved[tid]
                    total_frames    += saved[tid]['total']
                    total_keyframes += saved[tid]['keyframes']
                    continue

        tracklet_dir = images_dir / tid
        image_paths = sorted(
            tracklet_dir / f
            for f in os.listdir(tracklet_dir)
            if f.lower().endswith('.jpg')
        )
        n_total = len(image_paths)
        total_frames += n_total

        if n_total == 0:
            summary[tid] = {'total': 0, 'keyframes': 0, 'files': []}
            continue

        # Stage 1+2+3+4: KfId filtering
        keyframes = kfid.filter_tracklet(image_paths)

        if not keyframes:
            summary[tid] = {'total': n_total, 'keyframes': 0, 'files': []}
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for kf in keyframes:
            src_path = Path(kf['path'])
            dst_path = out_dir / src_path.name

            try:
                pil_img = Image.open(src_path).convert('RGB')
            except (OSError, FileNotFoundError):
                continue

            # Stage 2: Torso Crop
            # Use KfId box as a pre-crop hint, then let TorsoCropper refine
            box = kf['box']
            if box is not None:
                w, h = pil_img.size
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    pil_img = pil_img.crop((x1, y1, x2, y2))

            crop, _ = cropper.crop(pil_img)
            crop.save(str(dst_path), quality=95)
            saved_files.append(src_path.name)

        n_kf = len(saved_files)
        total_keyframes += n_kf
        summary[tid] = {'total': n_total, 'keyframes': n_kf, 'files': saved_files}

    cropper.close()

    # Write summary JSON
    summary_path = crops_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final stats
    kept_pct = (total_keyframes / max(total_frames, 1)) * 100
    empty = sum(1 for v in summary.values() if v['keyframes'] == 0)
    print(f'\nDone.')
    print(f'  Crops saved to : {crops_dir}')
    print(f'  Summary        : {summary_path}')
    print(f'  Frames kept    : {total_keyframes} / {total_frames} ({kept_pct:.1f}%)')
    print(f'  Empty tracklets: {empty} / {len(tracklet_ids)}')


if __name__ == '__main__':
    main()
