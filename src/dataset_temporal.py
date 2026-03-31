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
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        T.RandomPerspective(distortion_scale=0.05, p=0.15),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
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
                 crops_dir: str = None, use_keyframes: bool = False):
        self.transform = transform
        self.seq_len = seq_len
        self.crops_dir = crops_dir
        self.use_keyframes = use_keyframes
        self.samples = []   # list of (frame_paths, d1, d2)
        self._load(images_dir, gt_json)

    def _load(self, images_dir: str, gt_json: str):
        if self.use_keyframes:
            from keyframe_selection import select_keyframes
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
            if self.use_keyframes:
                # Pre-cache quality-filtered frame paths at load time (not per __getitem__)
                # top_k=seq_len*2 retains enough diversity for _evenly_sample to work from
                kf_paths, _ = select_keyframes(
                    Path(tracklet_dir), stride=2, top_k=self.seq_len * 2
                )
                if kf_paths:
                    paths = [str(p) for p in kf_paths]
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

        # When use_keyframes=True, paths is already a quality-filtered candidate pool
        # (pre-cached in _load). _evenly_sample draws seq_len frames from that pool.
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
