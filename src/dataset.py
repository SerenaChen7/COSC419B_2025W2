"""
Dataset classes for jersey number recognition.

Label convention:
    class 0      -> illegible / not visible  (ground truth: -1)
    class 1-99   -> jersey number 1-99       (ground truth == class index)
"""
import os
import json
import random
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

NUM_CLASSES = 100  # 0 = illegible, 1-99 = jersey number


def jersey_to_class(jersey_num: int) -> int:
    return 0 if jersey_num == -1 else jersey_num


def class_to_jersey(class_idx: int) -> int:
    return -1 if class_idx == 0 else class_idx


def get_train_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomGrayscale(p=0.2),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class JerseyTrainDataset(Dataset):
    """
    Per-image dataset for training.
    Tracklets store (image_paths, jersey_num) where jersey_num is the raw
    ground truth (-1 or 1-99).
    """

    def __init__(self, images_dir: str, gt_json: str,
                 transform=None, max_per_tracklet: int = None,
                 use_keyframes: bool = False):
        self.transform = transform
        self.max_per_tracklet = max_per_tracklet
        self.use_keyframes = use_keyframes
        self._load(images_dir, gt_json)

    def _load(self, images_dir: str, gt_json: str):
        with open(gt_json) as f:
            gt = json.load(f)

        if self.use_keyframes:
            from keyframe_selection import select_keyframes

        self.tracklets = []  # list of (image_paths, jersey_num)
        for tracklet_id, jersey_num in gt.items():
            tracklet_dir = os.path.join(images_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            if self.use_keyframes:
                pool_k = (self.max_per_tracklet or 25) * 2
                selected, _ = select_keyframes(tracklet_dir, stride=1, top_k=pool_k)
                imgs = [str(p) for p in selected]
                if not imgs:
                    imgs = [os.path.join(tracklet_dir, f)
                            for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')]
            else:
                imgs = [os.path.join(tracklet_dir, f)
                        for f in os.listdir(tracklet_dir)
                        if f.lower().endswith('.jpg')]
            if not imgs:
                continue
            self.tracklets.append((imgs, jersey_num))

        self._build_samples()

    def _build_samples(self):
        self.samples = []  # list of (image_path, class_idx)
        for imgs, jersey_num in self.tracklets:
            pool = (random.sample(imgs, self.max_per_tracklet)
                    if self.max_per_tracklet and len(imgs) > self.max_per_tracklet
                    else imgs)
            for p in pool:
                self.samples.append((p, jersey_to_class(jersey_num)))

    def resample(self):
        self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, class_idx
