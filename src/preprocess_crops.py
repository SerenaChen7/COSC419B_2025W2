# """
# Batch torso-crop preprocessing script.

# For every image in a dataset split, runs TorsoCropper and saves the resulting
# crop to a parallel `crops/` directory alongside `images/`.

# Output structure:
#     data/jersey-2023/
#         train/
#             images/   <- original (untouched)
#             crops/    <- torso crops written here
#                 0/
#                     0_1.jpg
#                     0_2.jpg
#                     ...
#         test/
#             images/
#             crops/

# Usage:
#     python src/preprocess_crops.py [--data-dir DATA_DIR] [--split {train,test,all}]
#                                    [--workers N] [--overwrite]

# The script is idempotent by default: images that already exist in crops/ are
# skipped unless --overwrite is passed.
# """

# import os
# import sys
# import argparse
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from PIL import Image
# from tqdm import tqdm

# sys.path.insert(0, str(Path(__file__).parent))
# from torso_crop import TorsoCropper


# def parse_args():
#     p = argparse.ArgumentParser(description='Pre-compute torso crops for all tracklets.')
#     p.add_argument('--data-dir', default='data/jersey-2023',
#                    help='Root dataset directory (default: data/jersey-2023)')
#     p.add_argument('--split', default='all', choices=['train', 'test', 'all'],
#                    help='Which split(s) to process (default: all)')
#     p.add_argument('--workers', type=int, default=1,
#                    help='Number of parallel worker threads (default: 1). '
#                         'NOTE: MediaPipe is not thread-safe; use 1 worker or '
#                         'set >1 only if you understand the implications.')
#     p.add_argument('--overwrite', action='store_true',
#                    help='Re-process images that already exist in crops/')
#     return p.parse_args()


# def process_image(
#     src_path: str,
#     dst_path: str,
#     cropper: TorsoCropper,
#     overwrite: bool,
# ) -> tuple[bool, bool]:
#     """
#     Crop one image and save it.

#     Returns
#     -------
#     (processed, pose_detected)
#         processed     - True if the image was actually written (False if skipped).
#         pose_detected - True if MediaPipe found the pose (False = fallback used).
#     """
#     if not overwrite and os.path.exists(dst_path):
#         return False, False

#     try:
#         img = Image.open(src_path).convert('RGB')
#     except Exception:
#         return False, False

#     crop, pose_detected = cropper.crop(img)

#     os.makedirs(os.path.dirname(dst_path), exist_ok=True)
#     crop.save(dst_path, quality=95)
#     return True, pose_detected


# def process_split(split: str, data_dir: str, overwrite: bool, n_workers: int):
#     images_root = os.path.join(data_dir, split, 'images')
#     crops_root  = os.path.join(data_dir, split, 'crops')

#     if not os.path.isdir(images_root):
#         print(f'[{split}] images dir not found: {images_root} — skipping.')
#         return

#     # Collect all (src, dst) pairs
#     tasks = []
#     for tracklet_id in sorted(os.listdir(images_root)):
#         tracklet_dir = os.path.join(images_root, tracklet_id)
#         if not os.path.isdir(tracklet_dir):
#             continue
#         for fname in os.listdir(tracklet_dir):
#             if not fname.lower().endswith('.jpg'):
#                 continue
#             src = os.path.join(tracklet_dir, fname)
#             dst = os.path.join(crops_root, tracklet_id, fname)
#             tasks.append((src, dst))

#     total = len(tasks)
#     print(f'[{split}] {total} images across {len(os.listdir(images_root))} tracklets')
#     print(f'[{split}] Saving crops to: {crops_root}')

#     processed = 0
#     pose_hits  = 0

#     if n_workers > 1:
#         # Each worker needs its own TorsoCropper instance (MediaPipe is not
#         # thread-safe when sharing a single Pose object).
#         # We chunk the task list and give each thread its own cropper.
#         chunk_size = max(1, total // n_workers)
#         chunks = [tasks[i:i + chunk_size] for i in range(0, total, chunk_size)]

#         def worker(chunk):
#             cropper = TorsoCropper()
#             results = []
#             for src, dst in chunk:
#                 r = process_image(src, dst, cropper, overwrite)
#                 results.append(r)
#             cropper.close()
#             return results

#         with tqdm(total=total, desc=f'{split:5s}') as pbar:
#             with ThreadPoolExecutor(max_workers=n_workers) as pool:
#                 futures = [pool.submit(worker, chunk) for chunk in chunks]
#                 for fut in as_completed(futures):
#                     for did_process, pose_ok in fut.result():
#                         if did_process:
#                             processed += 1
#                             if pose_ok:
#                                 pose_hits += 1
#                         pbar.update(1)
#     else:
#         # Single-threaded path — one shared TorsoCropper.
#         with TorsoCropper() as cropper:
#             for src, dst in tqdm(tasks, desc=f'{split:5s}'):
#                 did_process, pose_ok = process_image(src, dst, cropper, overwrite)
#                 if did_process:
#                     processed += 1
#                     if pose_ok:
#                         pose_hits += 1

#     skipped = total - processed
#     pose_rate = (pose_hits / processed * 100) if processed > 0 else 0.0

#     print(f'[{split}] Done.')
#     print(f'  Written  : {processed}')
#     print(f'  Skipped  : {skipped} (already existed)')
#     print(f'  Pose rate: {pose_hits}/{processed} ({pose_rate:.1f}%)')
#     print(f'  Fallback : {processed - pose_hits}/{processed} ({100 - pose_rate:.1f}%)')


# def main():
#     args = parse_args()

#     splits = ['train', 'test'] if args.split == 'all' else [args.split]

#     for split in splits:
#         process_split(split, args.data_dir, args.overwrite, args.workers)

#     print('\nPreprocessing complete.')
#     print(f'Use --crops-dir data/jersey-2023/<split>/crops when running train.py / predict.py.')


# if __name__ == '__main__': 
#     main()

"""
Batch torso-crop preprocessing script -- optimised edition.

Speed strategy:
  - Uses 'fork' multiprocessing on Linux/Docker. Forked workers inherit
    the parent's already-loaded MediaPipe shared libraries -- no re-loading,
    no libGLESv2/libEGL crashes.
  - Output directories are pre-created in the parent before workers start,
    eliminating makedirs races between workers.
  - chunksize=1 so tqdm updates after every single image (previously a huge
    chunksize meant workers silently processed thousands before returning
    anything, making the progress bar appear frozen for minutes).
  - --dry-run flag lets you count work without writing anything.

Output structure:
    data/jersey-2023/
        train/
            images/   <- original (untouched)
            crops/    <- torso crops written here
        test/
            images/
            crops/

Usage:
    python src/preprocess_crops.py [--data-dir DATA_DIR]
                                   [--split {train,test,all}]
                                   [--workers N]
                                   [--overwrite]
                                   [--dry-run]
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from torso_crop import TorsoCropper


# ---------------------------------------------------------------------------
# Global cropper -- inherited by forked workers, never re-initialised
# ---------------------------------------------------------------------------

_cropper = None


def _init_worker():
    """Called once per worker after fork. Creates a fresh Landmarker session."""
    global _cropper
    _cropper = TorsoCropper()


def _process_one(task):
    """
    Crop one image and save it.
    task = (src_path, dst_path, overwrite)
    Returns (processed: bool, pose_detected: bool)
    """
    src, dst, overwrite = task

    if not overwrite and os.path.exists(dst):
        return False, False

    try:
        img = Image.open(src).convert('RGB')
    except Exception:
        return False, False

    crop, pose_detected = _cropper.crop(img)
    crop.save(dst, quality=95)
    return True, pose_detected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_tasks(images_root, crops_root):
    tasks = []
    if not os.path.isdir(images_root):
        return tasks
    for tracklet_id in sorted(os.listdir(images_root)):
        tracklet_dir = os.path.join(images_root, tracklet_id)
        if not os.path.isdir(tracklet_dir):
            continue
        for fname in os.listdir(tracklet_dir):
            if fname.lower().endswith('.jpg'):
                src = os.path.join(tracklet_dir, fname)
                dst = os.path.join(crops_root, tracklet_id, fname)
                tasks.append((src, dst))
    return tasks


def process_split(split, data_dir, overwrite, n_workers, dry_run):
    images_root = os.path.join(data_dir, split, 'images')
    crops_root  = os.path.join(data_dir, split, 'crops')

    if not os.path.isdir(images_root):
        print(f'[{split}] images dir not found: {images_root} -- skipping.')
        return

    tasks_raw = _collect_tasks(images_root, crops_root)
    tasks     = [(s, d, overwrite) for s, d in tasks_raw]

    n_tracklets = len({os.path.dirname(s) for s, d in tasks_raw})
    total       = len(tasks)
    print(f'[{split}] {total} images across {n_tracklets} tracklets')
    print(f'[{split}] Saving crops to: {crops_root}')
    print(f'[{split}] Workers: {n_workers}')

    if dry_run:
        print(f'[{split}] --dry-run: no files written.')
        return

    # Pre-create all output directories in the parent before any worker starts.
    # This eliminates races where multiple workers try to mkdir the same path.
    print(f'[{split}] Creating output directories...')
    unique_dirs = {os.path.dirname(d) for s, d in tasks_raw}
    for d in unique_dirs:
        os.makedirs(d, exist_ok=True)
    print(f'[{split}] Starting workers...')

    processed = 0
    pose_hits  = 0

    if n_workers <= 1:
        # Single-process path
        _init_worker()
        for task in tqdm(tasks, desc=f'{split:5s}'):
            did_process, pose_ok = _process_one(task)
            if did_process:
                processed += 1
                pose_hits += int(pose_ok)
    else:
        # Multi-process with fork.
        # chunksize=1: each worker returns one result at a time.
        # MediaPipe inference (~50ms/image) massively dominates IPC overhead,
        # so chunksize=1 costs nothing in speed but makes tqdm update live.
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
            with tqdm(total=total, desc=f'{split:5s}') as pbar:
                for did_process, pose_ok in pool.imap_unordered(
                    _process_one, tasks, chunksize=1
                ):
                    if did_process:
                        processed += 1
                        pose_hits += int(pose_ok)
                    pbar.update(1)

    skipped   = total - processed
    pose_rate = (pose_hits / processed * 100) if processed else 0.0

    print(f'[{split}] Done.')
    print(f'  Written  : {processed}')
    print(f'  Skipped  : {skipped} (already existed)')
    print(f'  Pose rate: {pose_hits}/{processed} ({pose_rate:.1f}%)')
    print(f'  Fallback : {processed - pose_hits}/{processed} ({100 - pose_rate:.1f}%)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Pre-compute torso crops (fork-based parallel edition).'
    )
    p.add_argument('--data-dir',  default='data/jersey-2023')
    p.add_argument('--split',     default='all', choices=['train', 'test', 'all'])
    p.add_argument('--workers',   type=int, default=os.cpu_count() or 4,
                   help='Parallel worker processes (default: all CPU cores).')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-process images that already exist in crops/')
    p.add_argument('--dry-run',   action='store_true',
                   help='Count work without writing any files.')
    return p.parse_args()


def main():
    args   = parse_args()
    splits = ['train', 'test'] if args.split == 'all' else [args.split]

    for split in splits:
        process_split(split, args.data_dir, args.overwrite, args.workers, args.dry_run)

    print('\nPreprocessing complete.')
    print('Use --crops-dir data/jersey-2023/<split>/crops when running train.py / predict.py.')


if __name__ == '__main__':
    main()