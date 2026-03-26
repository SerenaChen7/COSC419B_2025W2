"""
KfId Evaluation Script.

Compares recognition accuracy WITH vs WITHOUT KfId filtering, using the
pre-computed kfid_crops/ directory and the ground-truth JSON.

Metrics reported:
  - Frame retention rate (how many frames KfId kept)
  - Top-1 tracklet accuracy: no-KfId vs KfId
  - Per-tracklet breakdown (optional --verbose)

This mirrors the ablation in Balaji et al. Table 3 and gives concrete
evidence that KfId improves performance.

Requires:
  - data/SoccerNet/<split>/kfid_crops/summary.json  (from preprocess_kfid_crops.py)
  - data/SoccerNet/<split>/test_gt.json or train_gt.json
  - A trained model checkpoint (--checkpoint)

Usage:
    python src/evaluate_kfid.py --split test --checkpoint outputs/best_model.pth
    python src/evaluate_kfid.py --split test --checkpoint outputs/best_model.pth --verbose
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/SoccerNet')
    p.add_argument('--split', default='test', choices=['train', 'test', 'challenge'])
    p.add_argument('--checkpoint', default='outputs/best_model.pth')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--verbose', action='store_true',
                   help='Print per-tracklet results')
    return p.parse_args()


@torch.no_grad()
def predict_from_paths(model, paths, transform, device, batch_size):
    """Average softmax probabilities over a list of image paths → jersey number."""
    if not paths:
        return -1
    all_probs = []
    for i in range(0, len(paths), batch_size):
        tensors = []
        for p in paths[i:i + batch_size]:
            try:
                img = Image.open(str(p)).convert('RGB')
                tensors.append(transform(img))
            except (OSError, FileNotFoundError):
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        probs = F.softmax(model(batch), dim=1).cpu()
        all_probs.append(probs)
    if not all_probs:
        return -1
    avg = torch.cat(all_probs).mean(dim=0)
    return class_to_jersey(avg.argmax().item())


def main():
    args = parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    # Load model
    model = load_checkpoint(args.checkpoint, device)
    model.to(device).eval()
    transform = get_val_transforms(args.img_size)

    # Paths
    data_dir   = Path(args.data_dir)
    images_dir = data_dir / args.split / 'images'
    crops_dir  = data_dir / args.split / 'kfid_crops'
    gt_path    = data_dir / args.split / f'{args.split}_gt.json'

    # Ground truth
    with open(gt_path) as f:
        ground_truth = json.load(f)  # {tracklet_id: jersey_number}

    # KfId summary
    summary_path = crops_dir / 'summary.json'
    if not summary_path.exists():
        print(f'ERROR: {summary_path} not found.')
        print('Run preprocess_kfid_crops.py first.')
        sys.exit(1)
    with open(summary_path) as f:
        kfid_summary = json.load(f)

    tracklet_ids = sorted(ground_truth.keys())
    print(f'Evaluating {len(tracklet_ids)} tracklets...\n')

    # -----------------------------------------------------------------------
    # Run inference: without KfId (all frames) vs with KfId (kfid_crops only)
    # -----------------------------------------------------------------------
    results = []

    for tid in tqdm(tracklet_ids):
        gt_jersey = ground_truth[tid]

        # --- Without KfId: all original frames ---
        all_paths = sorted(
            (images_dir / tid / f)
            for f in os.listdir(images_dir / tid)
            if f.lower().endswith('.jpg')
        ) if (images_dir / tid).is_dir() else []

        pred_no_kfid = predict_from_paths(
            model, all_paths, transform, device, args.batch_size
        )

        # --- With KfId: only the pre-cropped keyframes ---
        kfid_files = kfid_summary.get(tid, {}).get('files', [])
        kfid_paths = [crops_dir / tid / f for f in kfid_files]
        kfid_paths = [p for p in kfid_paths if p.exists()]

        pred_kfid = predict_from_paths(
            model, kfid_paths, transform, device, args.batch_size
        )

        n_total   = len(all_paths)
        n_keyframes = len(kfid_paths)

        results.append({
            'id':           tid,
            'gt':           gt_jersey,
            'pred_no_kfid': pred_no_kfid,
            'pred_kfid':    pred_kfid,
            'n_total':      n_total,
            'n_keyframes':  n_keyframes,
        })

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    n = len(results)
    correct_no_kfid = sum(1 for r in results if r['pred_no_kfid'] == r['gt'])
    correct_kfid    = sum(1 for r in results if r['pred_kfid']    == r['gt'])
    total_frames    = sum(r['n_total']     for r in results)
    total_keyframes = sum(r['n_keyframes'] for r in results)
    empty_tracklets = sum(1 for r in results if r['n_keyframes'] == 0)

    acc_no_kfid = 100 * correct_no_kfid / n
    acc_kfid    = 100 * correct_kfid    / n
    retention   = 100 * total_keyframes / max(total_frames, 1)
    improvement = acc_kfid - acc_no_kfid

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print('=' * 55)
    print('  KfId Evaluation Report')
    print('=' * 55)
    print(f'  Split           : {args.split}')
    print(f'  Tracklets       : {n}')
    print(f'  Total frames    : {total_frames}')
    print(f'  Keyframes kept  : {total_keyframes} ({retention:.1f}%)')
    print(f'  Frames dropped  : {total_frames - total_keyframes} ({100-retention:.1f}%)')
    print(f'  Empty tracklets : {empty_tracklets}')
    print('-' * 55)
    print(f'  Accuracy WITHOUT KfId : {acc_no_kfid:.2f}%  ({correct_no_kfid}/{n})')
    print(f'  Accuracy WITH    KfId : {acc_kfid:.2f}%  ({correct_kfid}/{n})')
    print(f'  Improvement           : {improvement:+.2f}%')
    print('=' * 55)

    if args.verbose:
        print('\nPer-tracklet breakdown (wrong predictions):')
        print(f"{'ID':>6}  {'GT':>4}  {'No-KfId':>8}  {'KfId':>6}  {'Frames':>10}")
        print('-' * 45)
        for r in results:
            no_ok = r['pred_no_kfid'] == r['gt']
            kf_ok = r['pred_kfid']    == r['gt']
            if not no_ok or not kf_ok:
                flag = '  <- KfId fixed' if (not no_ok and kf_ok) else \
                       '  <- KfId broke' if (no_ok and not kf_ok) else ''
                print(f"{r['id']:>6}  {r['gt']:>4}  "
                      f"{r['pred_no_kfid']:>8}  {r['pred_kfid']:>6}  "
                      f"{r['n_keyframes']:>4}/{r['n_total']:<5}{flag}")

    # Save results JSON
    out_path = Path(f'kfid_eval_{args.split}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'summary': {
                'split':           args.split,
                'n_tracklets':     n,
                'total_frames':    total_frames,
                'keyframes_kept':  total_keyframes,
                'retention_pct':   round(retention, 2),
                'acc_no_kfid':     round(acc_no_kfid, 2),
                'acc_kfid':        round(acc_kfid, 2),
                'improvement':     round(improvement, 2),
            },
            'per_tracklet': results,
        }, f, indent=2)
    print(f'\nFull results saved to {out_path}')


if __name__ == '__main__':
    main()
