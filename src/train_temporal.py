"""
Training script for the spatio-temporal (Bi-LSTM) jersey number model.

Usage:
    python src/train_temporal.py [--data-dir DATA_DIR] [--crops-dir DIR]
                                 [--seq-len N] [--img-size N]
                                 [--epochs N] [--batch-size N] [--lr LR]
                                 [--freeze-epochs N] [--patience N]
                                 [--val-split FLOAT] [--workers N]
                                 [--output-dir DIR]

Smoke test (fast sanity check):
    python src/train_temporal.py --max-tracklets 100 --epochs 3
"""
import os
import sys
import json
import argparse
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset_temporal import (
    JerseySequenceDataset, get_seq_transforms, get_seq_val_transforms, SEQ_LEN
)
from model_temporal import (
    SpatioTemporalNetwork, save_checkpoint, NUM_DIGIT_CLASSES, LSTM_LAYERS
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',  default='data/jersey-2023')
    p.add_argument('--crops-dir', default=None,
                   help='Pre-computed torso crops (output of preprocess_crops.py)')
    p.add_argument('--seq-len',   type=int, default=SEQ_LEN)
    p.add_argument('--img-size',  type=int, default=224)
    p.add_argument('--epochs',    type=int, default=30)
    p.add_argument('--batch-size',type=int, default=16)
    p.add_argument('--lr',        type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=5e-4,
                   help='AdamW weight decay (L2 regularisation)')
    p.add_argument('--freeze-epochs', type=int, default=5,
                   help='Epochs to train only the Bi-LSTM heads before unfreezing the encoder')
    p.add_argument('--patience',  type=int, default=10,
                   help='Early-stopping patience (epochs with no val improvement)')
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--max-tracklets', type=int, default=None,
                   help='Cap total tracklets for quick sanity checks')
    p.add_argument('--workers',   type=int, default=2)
    p.add_argument('--output-dir',default='outputs')
    p.add_argument('--seed',      type=int, default=None)
    p.add_argument('--dropout',   type=float, default=0.4,
                   help='Dropout probability applied before LSTM, between LSTM layers, and before heads')
    p.add_argument('--keyframes', action='store_true',
                   help='Use quality-based keyframe pre-selection during training')
    return p.parse_args()


def accuracy(logits_d1, logits_d2, targets_d1, targets_d2) -> float:
    """Both digits must be correct for the prediction to count as correct."""
    pred_d1 = logits_d1.argmax(dim=1)
    pred_d2 = logits_d2.argmax(dim=1)
    correct = (pred_d1 == targets_d1) & (pred_d2 == targets_d2)
    return correct.float().mean().item()


def train_epoch(model, loader, criterion_d1, criterion_d2, optimizer, device, scaler=None):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for frames, d1, d2 in tqdm(loader, desc='Train', leave=False):
        frames = frames.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits_d1, logits_d2 = model(frames)
                loss = criterion_d1(logits_d1, d1) + criterion_d2(logits_d2, d2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_d1, logits_d2 = model(frames)
            loss = criterion_d1(logits_d1, d1) + criterion_d2(logits_d2, d2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        total_acc  += accuracy(logits_d1.detach(), logits_d2.detach(), d1, d2)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def val_epoch(model, loader, criterion_d1, criterion_d2, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for frames, d1, d2 in tqdm(loader, desc='Val  ', leave=False):
        frames = frames.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        logits_d1, logits_d2 = model(frames)
        loss = criterion_d1(logits_d1, d1) + criterion_d2(logits_d2, d2)
        total_loss += loss.item()
        total_acc  += accuracy(logits_d1, logits_d2, d1, d2)
    n = len(loader)
    return total_loss / n, total_acc / n


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    print(f'Using device: {device}')
    print(f'seq_len={args.seq_len} | img_size={args.img_size} | '
          f'epochs={args.epochs} | batch_size={args.batch_size}')

    train_images = os.path.join(args.data_dir, 'train', 'images')
    train_gt     = os.path.join(args.data_dir, 'train', 'train_gt.json')

    # Build full dataset to get tracklet list for splitting.
    # use_keyframes=True here so the filtered paths survive the sample assignment below.
    # (Val split also gets the filtered pool; _evenly_sample draws uniformly from it.)
    full_ds = JerseySequenceDataset(
        train_images, train_gt,
        transform=None,   # transforms applied per-split below
        seq_len=args.seq_len,
        crops_dir=args.crops_dir,
        use_keyframes=args.keyframes,
    )

    n_tracklets = len(full_ds)
    indices = list(range(n_tracklets))
    random.seed(args.seed)
    random.shuffle(indices)

    if args.max_tracklets and args.max_tracklets < n_tracklets:
        indices = indices[:args.max_tracklets]
        print(f'  [--max-tracklets] Using {args.max_tracklets} / {n_tracklets} tracklets')

    n_val = max(1, int(len(indices) * args.val_split))
    val_idx   = set(indices[:n_val])
    train_idx = set(indices[n_val:])

    # Splits only differ by transform; samples are overwritten from full_ds below.
    train_ds = JerseySequenceDataset(
        train_images, train_gt,
        transform=get_seq_transforms(args.img_size),
        seq_len=args.seq_len,
        crops_dir=args.crops_dir,
    )
    val_ds = JerseySequenceDataset(
        train_images, train_gt,
        transform=get_seq_val_transforms(args.img_size),
        seq_len=args.seq_len,
        crops_dir=args.crops_dir,
    )

    train_ds.samples = [full_ds.samples[i] for i in sorted(train_idx)]
    val_ds.samples   = [full_ds.samples[i] for i in sorted(val_idx)]

    print(f'Train: {len(train_ds)} tracklets | Val: {len(val_ds)} tracklets')
    if args.keyframes:
        print('  -> Keyframe pre-selection enabled for training split')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        persistent_workers=args.workers > 0,
    )

    # --- Class-weighted loss (mirrors train.py; handles severe d1 class imbalance) ---
    d1_counts = Counter(d1 for _, d1, d2 in train_ds.samples)
    d2_counts = Counter(d2 for _, d1, d2 in train_ds.samples)
    total_d1 = sum(d1_counts.values())
    total_d2 = sum(d2_counts.values())

    weights_d1 = torch.ones(NUM_DIGIT_CLASSES)
    for cls, cnt in d1_counts.items():
        weights_d1[cls] = (total_d1 / (NUM_DIGIT_CLASSES * cnt)) ** 0.5
    weights_d2 = torch.ones(NUM_DIGIT_CLASSES)
    for cls, cnt in d2_counts.items():
        weights_d2[cls] = (total_d2 / (NUM_DIGIT_CLASSES * cnt)) ** 0.5
    # Cap weights to prevent extreme gradients from classes with very few samples
    weights_d1.clamp_(max=3.0)
    weights_d2.clamp_(max=3.0)

    criterion_d1 = nn.CrossEntropyLoss(weight=weights_d1.to(device), label_smoothing=0.1)
    criterion_d2 = nn.CrossEntropyLoss(weight=weights_d2.to(device), label_smoothing=0.1)

    # --- Model ---
    model = SpatioTemporalNetwork(
        pretrained=True, dropout=args.dropout, lstm_layers=LSTM_LAYERS,
    ).to(device)

    # Phase 1: freeze the spatial encoder, train only LSTM + heads
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    scaler = torch.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):

        # Phase 2: unfreeze encoder with lower LR after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            encoder_params = list(model.encoder.parameters())
            head_params = (
                list(model.bilstm.parameters()) +
                list(model.feat_dropout.parameters()) +
                list(model.attn_pool.parameters()) +
                list(model.head_d1.parameters()) +
                list(model.head_d2.parameters())
            )
            optimizer = optim.AdamW([
                {'params': head_params,    'lr': args.lr},
                {'params': encoder_params, 'lr': args.lr * 0.1},
            ], weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.freeze_epochs,
                eta_min=1e-5,
            )
            print(f'  -> Unfreezing encoder at epoch {epoch} '
                  f'(encoder LR={args.lr * 0.1:.2e})')

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_d1, criterion_d2, optimizer, device, scaler
        )
        val_loss, val_acc = val_epoch(model, val_loader, criterion_d1, criterion_d2, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'Val loss={val_loss:.4f} acc={val_acc:.4f} | '
              f'LR={lr_now:.6f}')

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss':   val_loss,   'val_acc':   val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, 'best_temporal.pth')
            # Store lstm_layers in args so load_checkpoint can reconstruct the model
            args.lstm_layers = LSTM_LAYERS
            save_checkpoint(ckpt_path, model, optimizer, epoch, val_acc, args)
            print(f'  -> Saved best temporal model (val_acc={val_acc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  -> Early stopping (no improvement for {args.patience} epochs)')
                break

    with open(os.path.join(args.output_dir, 'history_temporal.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete. Best val acc: {best_val_acc:.4f}')
    print(f'Checkpoint saved to: {args.output_dir}/best_temporal.pth')


if __name__ == '__main__':
    main()
