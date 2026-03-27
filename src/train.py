"""
Training script for jersey number recognition.

Usage:
    python src/train.py [--data-dir DATA_DIR] [--epochs N] [--batch-size N]
                        [--lr LR] [--img-size N] [--max-per-tracklet N]
                        [--val-split FLOAT] [--output-dir DIR]
                        [--crops-dir DIR]
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    JerseyTrainDataset, NUM_CLASSES, get_train_transforms, get_val_transforms, class_to_jersey
)
from model import build_model, freeze_backbone, unfreeze_backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/jersey-2023', help='Root dataset directory')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--max-per-tracklet', type=int, default=20,
                   help='Max images sampled per tracklet per epoch (None = all)')
    p.add_argument('--val-split', type=float, default=0.1,
                   help='Fraction of tracklets reserved for validation')
    p.add_argument('--output-dir', default='outputs', help='Where to save checkpoints')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--freeze-epochs', type=int, default=5,
                   help='Epochs to train only the head before unfreezing the backbone')
    p.add_argument('--crops-dir', default=None,
                   help='Path to pre-computed torso crops (output of preprocess_crops.py). '
                        'If set, images are loaded from here instead of images/.')
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Dataset ---
    train_images = os.path.join(args.data_dir, 'train', 'images')
    train_gt = os.path.join(args.data_dir, 'train', 'train_gt.json')

    full_dataset = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_train_transforms(args.img_size),
        max_per_tracklet=args.max_per_tracklet,
        crops_dir=args.crops_dir,
    )

    # Split by tracklets (not by images) to avoid data leakage
    n_tracklets = len(full_dataset.tracklets)
    tracklet_indices = list(range(n_tracklets))
    random.shuffle(tracklet_indices)
    n_val = max(1, int(n_tracklets * args.val_split))
    val_tracklet_idx = set(tracklet_indices[:n_val])
    train_tracklet_idx = set(tracklet_indices[n_val:])

    # Rebuild datasets with the split
    train_ds = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_train_transforms(args.img_size),
        max_per_tracklet=args.max_per_tracklet,
        crops_dir=args.crops_dir,
    )
    val_ds = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_val_transforms(args.img_size),
        max_per_tracklet=None,  # Use all images for validation
        crops_dir=args.crops_dir,
    )

    # Filter tracklets by split
    train_ds.tracklets = [train_ds.tracklets[i] for i in sorted(train_tracklet_idx)]
    val_ds.tracklets = [val_ds.tracklets[i] for i in sorted(val_tracklet_idx)]
    train_ds._build_samples()
    val_ds._build_samples()

    print(f'Train: {len(train_ds.tracklets)} tracklets, {len(train_ds)} images')
    print(f'Val:   {len(val_ds.tracklets)} tracklets, {len(val_ds)} images')

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    # --- Model ---
    model = build_model(pretrained=True).to(device)

    # Freeze backbone for the first `freeze_epochs` epochs so the randomly
    # initialised head stabilises before disturbing pretrained weights.
    freeze_backbone(model)

    # Class-weighted loss: use sqrt inverse frequency so rare jersey numbers get
    # more signal without aggressively down-weighting illegible (class 0),
    # which is common and must not be suppressed.
    label_counts = Counter(label for _, label in train_ds.samples)
    total_samples = sum(label_counts.values())
    class_weights = torch.ones(NUM_CLASSES)
    for cls, count in label_counts.items():
        class_weights[cls] = (total_samples / (NUM_CLASSES * count)) ** 0.5
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Mixed precision for CUDA
    scaler = torch.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Phase 2: unfreeze backbone with a lower LR after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            optimizer = optim.AdamW([
                {'params': model.fc.parameters(), 'lr': args.lr},
                {'params': [p for n, p in model.named_parameters()
                            if not n.startswith('fc.')], 'lr': args.lr * 0.1},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-5,
            )
            print(f'  -> Unfreezing backbone at epoch {epoch} (backbone LR={args.lr * 0.1:.2e})')

        # Resample training images from tracklets each epoch
        train_ds.resample()
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=pin)

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
            ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, ckpt_path)
            print(f'  -> Saved best model (val_acc={val_acc:.4f})')

    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete. Best val acc: {best_val_acc:.4f}')
    print(f'Checkpoint saved to: {args.output_dir}/best_model.pth')


if __name__ == '__main__':
    main()
