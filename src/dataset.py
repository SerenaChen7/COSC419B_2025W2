# """
# Dataset classes for jersey number recognition.

# Label convention:
#     class 0  -> illegible / not visible  (ground truth value: -1)
#     class 1-99 -> jersey number 1-99     (ground truth value == class index)
# """
# import os
# import json
# import random
# from PIL import Image

# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as T

# NUM_CLASSES = 100  # 0 = illegible, 1-99 = jersey number


# def jersey_to_class(jersey_num: int) -> int:
#     """Map ground-truth jersey number to class index."""
#     return 0 if jersey_num == -1 else jersey_num


# def class_to_jersey(class_idx: int) -> int:
#     """Map class index back to jersey number."""
#     return -1 if class_idx == 0 else class_idx


# def get_train_transforms(img_size: int = 128):
#     return T.Compose([
#         T.Resize((img_size, img_size)),
#         T.RandomHorizontalFlip(),
#         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # removed hue because of overflow error
#         T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])


# def get_val_transforms(img_size: int = 128):
#     return T.Compose([
#         T.Resize((img_size, img_size)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])


# class JerseyTrainDataset(Dataset):
#     """
#     Per-image dataset for training.
#     Optionally subsamples up to `max_per_tracklet` images per tracklet per epoch
#     to keep training time manageable.

#     If `crops_dir` is provided, images are loaded from there instead of
#     `images_dir` (falling back to the original if a crop file is missing).
#     `crops_dir` is expected to mirror the structure of `images_dir`:
#         crops_dir/<tracklet_id>/<filename>.jpg
#     """

#     def __init__(self, images_dir: str, gt_json: str,
#                  transform=None, max_per_tracklet: int = None,
#                  crops_dir: str = None):
#         self.transform = transform
#         self.max_per_tracklet = max_per_tracklet
#         self.crops_dir = crops_dir
#         self._load(images_dir, gt_json)

#     def _load(self, images_dir: str, gt_json: str):
#         with open(gt_json) as f:
#             gt = json.load(f)

#         self.tracklets = []  # list of (list_of_image_paths, class_idx)
#         for tracklet_id, jersey_num in gt.items():
#             tracklet_dir = os.path.join(images_dir, tracklet_id)
#             if not os.path.isdir(tracklet_dir):
#                 continue
#             imgs = [os.path.join(tracklet_dir, f)
#                     for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')]
#             if not imgs:
#                 continue
#             class_idx = jersey_to_class(jersey_num)
#             self.tracklets.append((imgs, class_idx))

#         self._build_samples()

#     def _build_samples(self):
#         """Build flat list of (image_path, class_idx) with optional subsampling."""
#         self.samples = []
#         for imgs, class_idx in self.tracklets:
#             if self.max_per_tracklet and len(imgs) > self.max_per_tracklet:
#                 selected = random.sample(imgs, self.max_per_tracklet)
#             else:
#                 selected = imgs
#             for p in selected:
#                 self.samples.append((p, class_idx))

#     def resample(self):
#         """Call between epochs to get a fresh subsample."""
#         self._build_samples()

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, class_idx = self.samples[idx]
#         # Use pre-computed torso crop if available, fall back to original.
#         if self.crops_dir:
#             tracklet_id = os.path.basename(os.path.dirname(img_path))
#             fname = os.path.basename(img_path)
#             crop_path = os.path.join(self.crops_dir, tracklet_id, fname)
#             load_path = crop_path if os.path.exists(crop_path) else img_path
#         else:
#             load_path = img_path
#         img = Image.open(load_path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, class_idx


# class JerseyTestDataset(Dataset):
#     """
#     Per-tracklet dataset for inference.
#     Each item is all images from one tracklet, returned as a stacked tensor.
#     Because tracklets can have hundreds of images, images are loaded lazily.

#     If `crops_dir` is provided, images are loaded from there instead of
#     `images_dir` (falling back to the original if a crop file is missing).
#     `crops_dir` is expected to mirror the structure of `images_dir`:
#         crops_dir/<tracklet_id>/<filename>.jpg
#     """

#     def __init__(self, images_dir: str, transform=None, batch_size: int = 64,
#                  crops_dir: str = None):
#         self.transform = transform
#         self.batch_size = batch_size
#         self.crops_dir = crops_dir
#         self.tracklet_ids = []
#         self.tracklet_paths = {}

#         for tracklet_id in sorted(os.listdir(images_dir)):
#             tracklet_dir = os.path.join(images_dir, tracklet_id)
#             if not os.path.isdir(tracklet_dir):
#                 continue
#             imgs = [os.path.join(tracklet_dir, f)
#                     for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')]
#             if imgs:
#                 self.tracklet_ids.append(tracklet_id)
#                 self.tracklet_paths[tracklet_id] = imgs

#     def __len__(self):
#         return len(self.tracklet_ids)

#     def get_image_paths(self, tracklet_id: str):
#         """
#         Return the list of paths to load for this tracklet.
#         If crops_dir is set, returns crop paths (with fallback to originals).
#         """
#         orig_paths = self.tracklet_paths[tracklet_id]
#         if not self.crops_dir:
#             return orig_paths
#         resolved = []
#         for p in orig_paths:
#             fname = os.path.basename(p)
#             crop_path = os.path.join(self.crops_dir, tracklet_id, fname)
#             resolved.append(crop_path if os.path.exists(crop_path) else p)
#         return resolved

#     def load_images(self, paths):
#         """Load and transform a list of image paths into a batch tensor."""
#         tensors = []
#         for p in paths:
#             img = Image.open(p).convert('RGB')
#             if self.transform:
#                 img = self.transform(img)
#             tensors.append(img)
#         return torch.stack(tensors)

"""
Dataset classes for jersey number recognition.

Label convention:
    class 0  -> illegible / not visible  (ground truth value: -1)
    class 1-99 -> jersey number 1-99     (ground truth value == class index)
"""
import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

NUM_CLASSES = 100  # 0 = illegible, 1-99 = jersey number


def jersey_to_class(jersey_num: int) -> int:
    return 0 if jersey_num == -1 else jersey_num


def class_to_jersey(class_idx: int) -> int:
    return -1 if class_idx == 0 else class_idx


def get_train_transforms(img_size: int = 128):
    """
    Training augmentations tuned for digit readability.

    Design rationale:
    - GaussianBlur removed: at ≤128px, even a 3-pixel kernel smears digits
      enough to destroy the label signal — the model cannot read the number.
    - RandomPerspective kept but reduced: mild distortion helps generalise
      to tilted/partial jerseys without warping digits beyond recognition.
    - RandomErasing kept but weakened: large erasure at low resolution
      can cover the entire jersey number (≈300px at 128×128).
    - ColorJitter and RandomAffine are safe — they don't destroy digit shape.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        T.RandomPerspective(distortion_scale=0.05, p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 128):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class JerseyTrainDataset(Dataset):
    """
    Per-image dataset for training.
    Optionally subsamples up to `max_per_tracklet` images per tracklet per epoch.
    When `use_keyframes=True`, each tracklet's image pool is pre-filtered to the
    highest-quality frames (sharpness + contrast), keeping 2× max_per_tracklet
    candidates so resample() still provides epoch-to-epoch variation.
    """

    def __init__(self, images_dir: str, gt_json: str,
                 transform=None, max_per_tracklet: int = None,
                 crops_dir: str = None, use_keyframes: bool = False):
        self.transform = transform
        self.max_per_tracklet = max_per_tracklet
        self.crops_dir = crops_dir
        self.use_keyframes = use_keyframes
        self._load(images_dir, gt_json)

    def _load(self, images_dir: str, gt_json: str):
        with open(gt_json) as f:
            gt = json.load(f)

        if self.use_keyframes:
            from keyframe_selection import select_keyframes

        self.tracklets = []
        for tracklet_id, jersey_num in gt.items():
            tracklet_dir = os.path.join(images_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            if self.use_keyframes:
                pool_k = (self.max_per_tracklet or 25) * 2
                selected, _ = select_keyframes(tracklet_dir, stride=1, top_k=pool_k)
                imgs = [str(p) for p in selected]
                if not imgs:  # fallback if keyframe selection fails
                    imgs = [os.path.join(tracklet_dir, f)
                            for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')]
            else:
                imgs = [os.path.join(tracklet_dir, f)
                        for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')]
            if not imgs:
                continue
            class_idx = jersey_to_class(jersey_num)
            self.tracklets.append((imgs, class_idx))

        self._build_samples()

    def _build_samples(self):
        self.samples = []
        for imgs, class_idx in self.tracklets:
            if self.max_per_tracklet and len(imgs) > self.max_per_tracklet:
                selected = random.sample(imgs, self.max_per_tracklet)
            else:
                selected = imgs
            for p in selected:
                self.samples.append((p, class_idx))

    def resample(self):
        self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        if self.crops_dir:
            tracklet_id = os.path.basename(os.path.dirname(img_path))
            fname = os.path.basename(img_path)
            crop_path = os.path.join(self.crops_dir, tracklet_id, fname)
            if not os.path.exists(crop_path):
                raise FileNotFoundError(
                    f"Crop missing: {crop_path}\n"
                    "Run preprocess_crops.py before training with --crops-dir."
                )
            load_path = crop_path
        else:
            load_path = img_path
        img = Image.open(load_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, class_idx


class JerseyTestDataset(Dataset):
    """Per-tracklet dataset for inference."""

    def __init__(self, images_dir: str, transform=None, batch_size: int = 64,
                 crops_dir: str = None):
        self.transform = transform
        self.batch_size = batch_size
        self.crops_dir = crops_dir
        self.tracklet_ids = []
        self.tracklet_paths = {}

        for tracklet_id in sorted(os.listdir(images_dir)):
            tracklet_dir = os.path.join(images_dir, tracklet_id)
            if not os.path.isdir(tracklet_dir):
                continue
            imgs = [os.path.join(tracklet_dir, f)
                    for f in os.listdir(tracklet_dir) if f.lower().endswith('.jpg')]
            if imgs:
                self.tracklet_ids.append(tracklet_id)
                self.tracklet_paths[tracklet_id] = imgs

    def __len__(self):
        return len(self.tracklet_ids)

    def get_image_paths(self, tracklet_id: str):
        orig_paths = self.tracklet_paths[tracklet_id]
        if not self.crops_dir:
            return orig_paths
        resolved = []
        for p in orig_paths:
            fname = os.path.basename(p)
            crop_path = os.path.join(self.crops_dir, tracklet_id, fname)
            if not os.path.exists(crop_path):
                raise FileNotFoundError(
                    f"Crop missing: {crop_path}\n"
                    "Run preprocess_crops.py --split test before inference."
                )
            resolved.append(crop_path)
        return resolved

    def load_images(self, paths):
        tensors = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            tensors.append(img)
        return torch.stack(tensors)