"""
Sequence dataset for the Bi-LSTM temporal model.

Each sample is one tracklet represented as a fixed-length sequence of T frames.
Returns: (frames_tensor, d1_label, d2_label)
    frames_tensor : (T, 3, H, W)  float32 tensor
    d1_label      : int  (tens digit class, 0-9 or 10=blank)
    d2_label      : int  (units digit class, 0-9 or 10=blank)
"""
import os
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from model_temporal import jersey_to_digits

# Default sequence length — balances temporal coverage vs. memory/speed.
# Swap the _evenly_sample call in __getitem__ for keyframe_selection.select()
# once that module is merged.
SEQ_LEN = 16


def get_seq_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.88, 1.12)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_seq_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _evenly_sample(paths: list, n: int) -> list:
    """
    Return n paths evenly spaced across the full tracklet.
    If the tracklet has fewer than n frames, the last frame is repeated to pad.

    NOTE: Replace this call in __getitem__ with keyframe_selection.select()
    once that module is integrated — it is the only integration point needed.
    """
    if len(paths) >= n:
        indices = [int(i * len(paths) / n) for i in range(n)]
        return [paths[i] for i in indices]
    return paths + [paths[-1]] * (n - len(paths))


class JerseySequenceDataset(Dataset):
    """
    Per-tracklet sequence dataset for training and validation.

    Parameters
    ----------
    images_dir  : directory containing <tracklet_id>/<frame>.jpg sub-folders
    gt_json     : path to ground-truth JSON  {tracklet_id: jersey_number}
    transform   : per-frame transform (use get_seq_transforms / get_seq_val_transforms)
    seq_len     : number of frames per sequence
    crops_dir   : pre-computed torso crops directory; must exist and be complete
                  if provided (FileNotFoundError raised on missing crops)
    """

    def __init__(self, images_dir: str, gt_json: str,
                 transform=None, seq_len: int = SEQ_LEN,
                 crops_dir: str = None):
        self.transform = transform
        self.seq_len = seq_len
        self.crops_dir = crops_dir
        self.samples = []   # list of (sorted_frame_paths, d1, d2)
        self._load(images_dir, gt_json)

    def _load(self, images_dir: str, gt_json: str):
        with open(gt_json) as f:
            gt = json.load(f)
        for tracklet_id, jersey_num in gt.items():
            tracklet_dir = os.path.join(images_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            paths = sorted([
                os.path.join(tracklet_dir, fname)
                for fname in os.listdir(tracklet_dir)
                if fname.lower().endswith('.jpg')
            ])
            if not paths:
                continue
            d1, d2 = jersey_to_digits(int(jersey_num))
            self.samples.append((paths, d1, d2))

    def _resolve(self, img_path: str) -> str:
        """Return the crop path when crops_dir is set, otherwise the original."""
        if not self.crops_dir:
            return img_path
        tracklet_id = Path(img_path).parent.name
        fname = Path(img_path).name
        crop_path = os.path.join(self.crops_dir, tracklet_id, fname)
        if not os.path.exists(crop_path):
            raise FileNotFoundError(
                f"Crop missing: {crop_path}\n"
                "Run preprocess_crops.py before training with --crops-dir."
            )
        return crop_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, d1, d2 = self.samples[idx]

        # --- keyframe integration point ---
        # To use quality-based keyframe selection instead of uniform sampling,
        # replace the line below with: selected = keyframe_selection.select(paths, self.seq_len)
        selected = _evenly_sample(paths, self.seq_len)

        frames = []
        for p in selected:
            img = Image.open(self._resolve(p)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), d1, d2   # (T, 3, H, W), int, int


class JerseySequenceTestDataset(Dataset):
    """
    Per-tracklet sequence dataset for inference (no labels).

    Returns (frames_tensor, tracklet_id) for each tracklet.
    """

    def __init__(self, images_dir: str, transform=None,
                 seq_len: int = SEQ_LEN, crops_dir: str = None):
        self.transform = transform
        self.seq_len = seq_len
        self.crops_dir = crops_dir
        self.tracklet_ids = []
        self.tracklet_paths = {}

        for tracklet_id in sorted(os.listdir(images_dir)):
            tracklet_dir = os.path.join(images_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            paths = sorted([
                os.path.join(tracklet_dir, fname)
                for fname in os.listdir(tracklet_dir)
                if fname.lower().endswith('.jpg')
            ])
            if paths:
                self.tracklet_ids.append(tracklet_id)
                self.tracklet_paths[tracklet_id] = paths

    def _resolve(self, img_path: str) -> str:
        if not self.crops_dir:
            return img_path
        tracklet_id = Path(img_path).parent.name
        fname = Path(img_path).name
        crop_path = os.path.join(self.crops_dir, tracklet_id, fname)
        if not os.path.exists(crop_path):
            raise FileNotFoundError(
                f"Crop missing: {crop_path}\n"
                "Run preprocess_crops.py --split test before inference."
            )
        return crop_path

    def __len__(self):
        return len(self.tracklet_ids)

    def __getitem__(self, idx):
        tracklet_id = self.tracklet_ids[idx]
        paths = self.tracklet_paths[tracklet_id]

        # --- keyframe integration point ---
        # Replace with: selected = keyframe_selection.select(paths, self.seq_len)
        selected = _evenly_sample(paths, self.seq_len)

        frames = []
        for p in selected:
            img = Image.open(self._resolve(p)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), tracklet_id   # (T, 3, H, W), str
