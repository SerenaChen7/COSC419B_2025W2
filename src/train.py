"""
Training script for jersey number recognition.

Usage:
    python src/train.py [--data-dir DATA_DIR] [--arch ARCH]
                        [--epochs N] [--batch-size N] [--lr LR]
                        [--img-size N] [--max-per-tracklet N]
                        [--max-tracklets N] [--val-split FLOAT]
                        [--label-smoothing FLOAT] [--patience N]
                        [--output-dir DIR] [--crops-dir DIR]

Quick subset test (fast sanity check before a full run):
    python src/train.py --max-tracklets 500 --epochs 5

Architecture choices (--arch):
    mobilenet_v3_small  (default) – fastest on CPU, good accuracy
    mobilenet_v3_large            – better accuracy, moderately slower
    resnet18                      – original baseline
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

from dataset import (
    JerseyTrainDataset, NUM_CLASSES, get_train_transforms, get_val_transforms, class_to_jersey
)
from model import build_model, freeze_backbone, unfreeze_backbone, _is_head_param


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/jersey-2023')
    p.add_argument('--arch', default='mobilenet_v3_large',
                   choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18'],
                   help='Backbone architecture')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    # 128×128: minimum reliable resolution for reading jersey digits
    # ~3× faster than 224×224, still enough for digit features
    p.add_argument('--img-size', type=int, default=128)
    # 25 images/tracklet: covers ~55% of each tracklet over 15 epochs via resampling
    p.add_argument('--max-per-tracklet', type=int, default=25)
    # Limit total tracklets — useful for quick accuracy checks before full training
    p.add_argument('--max-tracklets', type=int, default=None,
                   help='Cap total tracklets (train+val). Use ~500 for a quick sanity check.')
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--label-smoothing', type=float, default=0.1,
                   help='Label smoothing for CrossEntropyLoss (reduces overconfidence)')
    p.add_argument('--patience', type=int, default=5,
                   help='Early-stopping patience (epochs with no val improvement)')
    p.add_argument('--output-dir', default='outputs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--freeze-epochs', type=int, default=2,
                   help='Epochs to train only the head before unfreezing the backbone')
    p.add_argument('--crops-dir', default=None,
                   help='Path to pre-computed torso crops (output of preprocess_crops.py)')
    p.add_argument('--use-keyframes', action='store_true',
                   help='Select top-k keyframes per tracklet during training (sharpness/contrast/'
                        'diversity). Aligns training distribution with keyframe-based inference.')
    p.add_argument('--keyframe-top-k', type=int, default=5,
                   help='Number of keyframes to select per tracklet when --use-keyframes is set.')
    return p.parse_args()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in tqdm(loader, desc='Val  ', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


def make_loader(dataset, batch_size, shuffle, workers, device):
    """Create a DataLoader with settings tuned for the active device."""
    use_persistent = workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == 'cuda',   # speeds up CPU->GPU transfers
        persistent_workers=use_persistent,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Arch: {args.arch} | img_size: {args.img_size} | '
          f'max_per_tracklet: {args.max_per_tracklet} | epochs: {args.epochs}')

    # --- Dataset ---
    train_images = os.path.join(args.data_dir, 'train', 'images')
    train_gt = os.path.join(args.data_dir, 'train', 'train_gt.json')

    full_dataset = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_train_transforms(args.img_size),
        max_per_tracklet=args.max_per_tracklet,
        crops_dir=args.crops_dir,
        use_keyframes=args.use_keyframes,
        keyframe_top_k=args.keyframe_top_k,
    )

    # Split by tracklets to avoid data leakage
    n_tracklets = len(full_dataset.tracklets)
    tracklet_indices = list(range(n_tracklets))
    random.shuffle(tracklet_indices)

    # Optional: cap total tracklets for quick testing
    if args.max_tracklets and args.max_tracklets < n_tracklets:
        tracklet_indices = tracklet_indices[:args.max_tracklets]
        print(f'  [--max-tracklets] Using {args.max_tracklets} / {n_tracklets} tracklets')

    n_val = max(1, int(len(tracklet_indices) * args.val_split))
    val_tracklet_idx = set(tracklet_indices[:n_val])
    train_tracklet_idx = set(tracklet_indices[n_val:])

    train_ds = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_train_transforms(args.img_size),
        max_per_tracklet=args.max_per_tracklet,
        crops_dir=args.crops_dir,
        use_keyframes=args.use_keyframes,
        keyframe_top_k=args.keyframe_top_k,
    )
    val_ds = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_val_transforms(args.img_size),
        max_per_tracklet=None,
        crops_dir=args.crops_dir,
        # Val uses all frames for a stable accuracy signal, not keyframe subset
    )

    train_ds.tracklets = [train_ds.tracklets[i] for i in sorted(train_tracklet_idx)]
    val_ds.tracklets = [val_ds.tracklets[i] for i in sorted(val_tracklet_idx)]
    train_ds._build_samples()
    val_ds._build_samples()

    if args.use_keyframes:
        print(f'Keyframe selection enabled for training (top_k={args.keyframe_top_k})')
    print(f'Train: {len(train_ds.tracklets)} tracklets, {len(train_ds)} images')
    print(f'Val:   {len(val_ds.tracklets)} tracklets, {len(val_ds)} images')

    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, workers=args.workers, device=device)

    # --- Model ---
    model = build_model(pretrained=True, arch=args.arch).to(device)
    freeze_backbone(model)

    # Class-weighted loss with label smoothing
    label_counts = Counter(label for _, label in train_ds.samples)
    total_samples = sum(label_counts.values())
    class_weights = torch.ones(NUM_CLASSES)
    for cls, count in label_counts.items():
        class_weights[cls] = (total_samples / (NUM_CLASSES * count)) ** 0.5
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Phase 2: unfreeze backbone with a lower LR after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            head_params = [p for n, p in model.named_parameters() if _is_head_param(n)]
            backbone_params = [p for n, p in model.named_parameters() if not _is_head_param(n)]
            optimizer = optim.AdamW([
                {'params': head_params, 'lr': args.lr},
                {'params': backbone_params, 'lr': args.lr * 0.1},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-5,
            )
            print(f'  -> Unfreezing backbone at epoch {epoch} (backbone LR={args.lr * 0.1:.2e})')

        # Resample training images from tracklets each epoch
        train_ds.resample()
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True, workers=args.workers, device=device)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'Val loss={val_loss:.4f} acc={val_acc:.4f} | '
              f'LR={lr_now:.6f}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, ckpt_path)
            print(f'  -> Saved best model (val_acc={val_acc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  -> Early stopping (no improvement for {args.patience} epochs)')
                break

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete. Best val acc: {best_val_acc:.4f}')
    print(f'Checkpoint saved to: {args.output_dir}/best_model.pth')


if __name__ == '__main__':
    main()
