"""
Training script for jersey number recognition.

Usage:
    python src/train.py [options]

Quick sanity check:
    python src/train.py --max-tracklets 500 --epochs 5

Architecture choices (--arch):
    resnet18             default
    mobilenet_v3_small   fastest on CPU
    mobilenet_v3_large   better accuracy, moderately slower
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from dataset import JerseyTrainDataset, NUM_CLASSES, get_train_transforms, get_val_transforms
from model import build_model, freeze_backbone, unfreeze_backbone, _is_head_param
from consolidate import consolidate_tracklet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',         default='data/jersey-2023')
    p.add_argument('--arch',             default='resnet34',
                   choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34'])
    p.add_argument('--epochs',           type=int,   default=20)
    p.add_argument('--batch-size',       type=int,   default=32)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--img-size',         type=int,   default=224)
    p.add_argument('--max-per-tracklet', type=int,   default=20)
    p.add_argument('--max-tracklets',    type=int,   default=None)
    p.add_argument('--val-split',        type=float, default=0.1)
    p.add_argument('--label-smoothing',  type=float, default=0.1)
    p.add_argument('--patience',         type=int,   default=5)
    p.add_argument('--freeze-epochs',    type=int,   default=2)
    p.add_argument('--output-dir',       default='outputs')
    p.add_argument('--workers',          type=int,   default=2)
    p.add_argument('--keyframes',        action='store_true')
    return p.parse_args()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_acc  += accuracy(logits.detach().float(), labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    """Per-frame val loss — monitors training stability."""
    model.eval()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_tracklet_accuracy(model, val_ds, transform, device, batch_size):
    """
    Per-tracklet consolidation accuracy — identical logic to predict.py.
    Uses Bayesian log-prob aggregation (full softmax vectors, not just argmax).
    """
    model.eval()
    correct = 0

    for imgs_paths, jersey_num in tqdm(val_ds.tracklets, desc='Val  ', leave=False):
        frame_probs_list = []
        for i in range(0, len(imgs_paths), batch_size):
            batch = imgs_paths[i:i + batch_size]
            tensors = torch.stack([
                transform(Image.open(p).convert('RGB')) for p in batch
            ]).to(device, non_blocking=True)

            probs = F.softmax(model(tensors).float(), dim=1).cpu().numpy()
            for frame_probs in probs:
                frame_probs_list.append(frame_probs)

        predicted = consolidate_tracklet(frame_probs_list)
        if predicted == jersey_num:
            correct += 1

    return correct / len(val_ds.tracklets) if val_ds.tracklets else 0.0


def make_loader(dataset, batch_size, shuffle, workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device     = torch.device('cuda' if torch.cuda.is_available() else
                              'mps'  if torch.backends.mps.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    scaler     = torch.GradScaler(device=device.type)
    print(f'Device: {device} | AMP: {device.type == "cuda"}')
    print(f'Arch: {args.arch} | img_size: {args.img_size} | '
          f'max_per_tracklet: {args.max_per_tracklet} | epochs: {args.epochs}')

    train_images = os.path.join(args.data_dir, 'train', 'images')
    train_gt     = os.path.join(args.data_dir, 'train', 'train_gt.json')

    if args.keyframes:
        print('  Keyframe filtering enabled')

    train_ds = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_train_transforms(args.img_size),
        max_per_tracklet=args.max_per_tracklet,
        use_keyframes=args.keyframes,
    )

    n_tracklets = len(train_ds.tracklets)
    indices     = list(range(n_tracklets))
    random.shuffle(indices)

    if args.max_tracklets and args.max_tracklets < n_tracklets:
        indices = indices[:args.max_tracklets]
        print(f'  [--max-tracklets] Using {len(indices)} / {n_tracklets} tracklets')

    n_val     = max(1, int(len(indices) * args.val_split))
    val_idx   = set(indices[:n_val])
    train_idx = set(indices[n_val:])

    val_ds = JerseyTrainDataset(
        train_images, train_gt,
        transform=get_val_transforms(args.img_size),
        max_per_tracklet=None,
        use_keyframes=args.keyframes,
    )

    train_ds.tracklets = [train_ds.tracklets[i] for i in sorted(train_idx)]
    val_ds.tracklets   = [val_ds.tracklets[i]   for i in sorted(val_idx)]
    train_ds._build_samples()
    val_ds._build_samples()

    print(f'Train: {len(train_ds.tracklets)} tracklets, {len(train_ds)} images')
    print(f'Val:   {len(val_ds.tracklets)} tracklets, {len(val_ds)} images')

    val_loader = make_loader(val_ds, args.batch_size, shuffle=False,
                             workers=args.workers, pin_memory=pin_memory)

    # Class-weighted loss to handle imbalance across 100 jersey numbers
    label_counts  = Counter(label for _, label in train_ds.samples)
    total         = sum(label_counts.values())
    class_weights = torch.ones(NUM_CLASSES)
    for cls, count in label_counts.items():
        class_weights[cls] = (total / (NUM_CLASSES * count)) ** 0.5
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    model     = build_model(pretrained=True, arch=args.arch).to(device)
    freeze_backbone(model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc     = 0.0
    patience_counter = 0
    history          = []

    for epoch in range(1, args.epochs + 1):

        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            head_params     = [p for n, p in model.named_parameters() if _is_head_param(n)]
            backbone_params = [p for n, p in model.named_parameters() if not _is_head_param(n)]
            optimizer = optim.AdamW([
                {'params': head_params,     'lr': args.lr},
                {'params': backbone_params, 'lr': args.lr * 0.1},
            ], weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-5,
            )
            print(f'  -> Backbone unfrozen at epoch {epoch} (LR={args.lr * 0.1:.2e})')

        train_ds.resample()
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True,
                                   workers=args.workers, pin_memory=pin_memory)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss  = val_epoch(model, val_loader, criterion, device)
        val_acc   = val_tracklet_accuracy(model, val_ds, get_val_transforms(args.img_size),
                                          device, args.batch_size)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'val loss={val_loss:.4f} tracklet_acc={val_acc:.4f} | '
              f'lr={lr_now:.2e}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            ckpt = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          val_acc,
                'args':             vars(args),
            }, ckpt)
            print(f'  -> Saved best model (val_acc={val_acc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  -> Early stopping after {args.patience} epochs with no improvement')
                break

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nDone. Best val acc: {best_val_acc:.4f}')
    print(f'Checkpoint: {args.output_dir}/best_model.pth')


if __name__ == '__main__':
    main()
