"""
Inference script for the spatio-temporal (Bi-LSTM) jersey number model.

Outputs the same predictions.json format as predict.py so downstream
evaluation (evaluate.py) works unchanged.

Usage:
    python src/predict_temporal.py [--data-dir DATA_DIR] [--crops-dir DIR]
                                   [--checkpoint outputs/best_temporal.pth]
                                   [--seq-len N] [--img-size N]
                                   [--output predictions_temporal.json]
                                   [--gt PATH]
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset_temporal import JerseySequenceTestDataset, get_seq_val_transforms, SEQ_LEN
from model_temporal import load_checkpoint, digits_to_jersey


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',   default='data/jersey-2023')
    p.add_argument('--crops-dir',  default=None,
                   help='Pre-computed torso crops directory')
    p.add_argument('--checkpoint', default='outputs/best_temporal.pth')
    p.add_argument('--seq-len',    type=int, default=SEQ_LEN)
    p.add_argument('--img-size',   type=int, default=224)
    p.add_argument('--batch-size', type=int, default=8,
                   help='Tracklets per batch (each contains seq_len frames)')
    p.add_argument('--output',     default='predictions_temporal.json')
    p.add_argument('--gt',         default=None,
                   help='Ground-truth JSON for on-the-fly evaluation')
    p.add_argument('--tta-passes', type=int, default=1,
                   help='MC Dropout TTA: run model N times with dropout enabled and average '
                        'logits (1 = no TTA; 5 recommended after training with dropout >= 0.4)')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    print(f'Using device: {device}')

    model = load_checkpoint(args.checkpoint, device)
    model.to(device)
    print(f'Loaded checkpoint: {args.checkpoint}')
    if args.tta_passes > 1:
        print(f'MC Dropout TTA enabled: {args.tta_passes} passes')

    test_images_dir = os.path.join(args.data_dir, 'test', 'images')
    transform = get_seq_val_transforms(args.img_size)

    test_ds = JerseySequenceTestDataset(
        test_images_dir,
        transform=transform,
        seq_len=args.seq_len,
        crops_dir=args.crops_dir,
    )

    # Use batch_size > 1 to parallelize sequence inference
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0,   # JIT-opened PIL images don't multiprocess safely here
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],
        ),
    )

    print(f'Running inference on {len(test_ds)} tracklets...')

    predictions = {}

    use_tta = args.tta_passes > 1
    if use_tta:
        model.train()   # enable dropout stochasticity for MC Dropout
    else:
        model.eval()

    with torch.no_grad():
        for frames_batch, tracklet_ids in tqdm(loader):
            frames_batch = frames_batch.to(device)   # (B, T, 3, H, W)
            if use_tta:
                logits_d1_list, logits_d2_list = [], []
                for _ in range(args.tta_passes):
                    ld1, ld2 = model(frames_batch)
                    logits_d1_list.append(ld1)
                    logits_d2_list.append(ld2)
                logits_d1 = torch.stack(logits_d1_list).mean(0)
                logits_d2 = torch.stack(logits_d2_list).mean(0)
            else:
                logits_d1, logits_d2 = model(frames_batch)
            pred_d1 = logits_d1.argmax(dim=1).cpu().tolist()
            pred_d2 = logits_d2.argmax(dim=1).cpu().tolist()
            for tid, d1, d2 in zip(tracklet_ids, pred_d1, pred_d2):
                predictions[tid] = digits_to_jersey(d1, d2)

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
            total   += 1
            correct += pred_num == gt_num

            if gt_num == -1:
                illegible_total   += 1
                illegible_correct += pred_num == -1
            else:
                legible_total   += 1
                legible_correct += pred_num == gt_num

        print()
        print('=== Evaluation ===')
        print(f'Overall accuracy:   {correct}/{total} = {100*correct/total:.1f}%')
        if legible_total:
            print(f'Legible accuracy:   {legible_correct}/{legible_total} = '
                  f'{100*legible_correct/legible_total:.1f}%')
        if illegible_total:
            print(f'Illegible accuracy: {illegible_correct}/{illegible_total} = '
                  f'{100*illegible_correct/illegible_total:.1f}%')


if __name__ == '__main__':
    main()
