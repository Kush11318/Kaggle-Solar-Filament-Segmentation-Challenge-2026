"""
PyTorch Dataset for 3-Channel Box-Prompted CropUNet Training.
Features Francisco Mendez's instance-conditioning prompt:
- Channel 0: Raw H-alpha normalized intensity [0, 1]
- Channel 1: Optical Eddington-Barbier flattened intensity [0, 1]
- Channel 2: Binary Box Prompt [0, 1] (target proposal bounding box)

Target Ground Truth:
- Strictly the TARGET filament's own polygon (ignoring neighboring filaments).
- Teaches the U-Net to segment ONLY the prompted instance and suppress neighboring flux ropes!
"""

import os
import time
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from src.crop_utils import compute_dynamic_square_crop
from src.disk import detect_disk, make_two_channel_input, DiskCache


class FilamentCropPromptDataset(Dataset):
    """
    Dataset of filament crops conditioned on a 3rd Box-Prompt channel.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        img_dir: str,
        target_size: int = 256,
        context: float = 1.8,
        augment: bool = True,
        disk_cache: Optional[DiskCache] = None,
        precache: bool = True,
    ):
        self.samples = samples
        self.img_dir = img_dir
        self.target_size = target_size
        self.context = context
        self.augment = augment
        self.disk_cache = disk_cache
        self.precache = precache
        self.cached_crops: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []

        if self.precache and len(self.samples) > 0:
            self._precache_all_crops()

    def _precache_all_crops(self):
        t0 = time.time()
        file_to_items = defaultdict(list)
        for idx, s in enumerate(self.samples):
            file_to_items[s["image_file"]].append((idx, s))

        total_files = len(file_to_items)
        print(f"Pre-caching {len(self.samples)} box-prompted crops across {total_files} images into RAM...", flush=True)
        self.cached_crops = [None] * len(self.samples)

        for completed, (fn, items) in enumerate(file_to_items.items(), 1):
            img_path = os.path.join(self.img_dir, fn)
            raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                continue

            h_img, w_img = raw_img.shape
            disk = self.disk_cache.get(fn) if self.disk_cache else None
            if disk is None:
                disk = detect_disk(raw_img)
                if self.disk_cache:
                    self.disk_cache.set(fn, disk)

            two_ch = make_two_channel_input(raw_img, disk)

            for orig_idx, sample in items:
                bbox = sample["bbox"]
                bx, by, bw, bh = bbox

                # Jitter box slightly during augmentation to simulate detector noise
                if self.augment:
                    j_bx = bx + random.uniform(-0.08, 0.08) * bw
                    j_by = by + random.uniform(-0.08, 0.08) * bh
                    j_bw = max(6.0, bw * random.uniform(0.92, 1.10))
                    j_bh = max(6.0, bh * random.uniform(0.92, 1.10))
                    crop_bbox = [j_bx, j_by, j_bw, j_bh]
                else:
                    crop_bbox = [bx, by, bw, bh]

                x0, y0, x1, y1, _ = compute_dynamic_square_crop(
                    bbox=crop_bbox,
                    context=self.context,
                    min_side=96,
                    max_side=2048,
                    img_h=h_img,
                    img_w=w_img,
                )

                crop_h, crop_w = y1 - y0, x1 - x0
                if crop_h < 8 or crop_w < 8:
                    continue

                # Channel 0: Raw
                ch0 = cv2.resize(two_ch[0, y0:y1, x0:x1], (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                # Channel 1: Flattened
                ch1 = cv2.resize(two_ch[1, y0:y1, x0:x1], (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

                # Channel 2: Box Prompt in crop coordinate space
                rx0 = max(0, min(crop_w - 1, int(crop_bbox[0] - x0)))
                ry0 = max(0, min(crop_h - 1, int(crop_bbox[1] - y0)))
                rx1 = max(rx0 + 1, min(crop_w, int(crop_bbox[0] + crop_bbox[2] - x0)))
                ry1 = max(ry0 + 1, min(crop_h, int(crop_bbox[1] + crop_bbox[3] - y0)))

                box_prompt_native = np.zeros((crop_h, crop_w), dtype=np.float32)
                box_prompt_native[ry0:ry1, rx0:rx1] = 1.0
                ch2 = cv2.resize(box_prompt_native, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

                # Ground Truth: STRICTLY the target filament instance only!
                inst_mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
                for poly in sample.get("segmentation", []):
                    if len(poly) >= 6:
                        pts = np.array(poly, dtype=np.float32).reshape((-1, 2))
                        # Offset polygon to crop coordinates
                        pts[:, 0] -= x0
                        pts[:, 1] -= y0
                        pts = np.round(pts).astype(np.int32)
                        cv2.fillPoly(inst_mask_crop, [pts], 1)

                crop_mask = cv2.resize(inst_mask_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

                crop_three_ch = np.stack([ch0, ch1, ch2], axis=-1)
                crop_three_ch_u8 = np.clip(crop_three_ch * 255.0, 0, 255).astype(np.uint8)
                crop_mask_u8 = (crop_mask > 0).astype(np.uint8)

                self.cached_crops[orig_idx] = (crop_three_ch_u8, crop_mask_u8)

            if completed % 50 == 0 or completed == total_files:
                pct = (completed / total_files) * 100
                print(f"  Pre-cached {completed}/{total_files} images ({pct:.1f}%) in {time.time()-t0:.1f}s...", flush=True)

        valid_count = sum(1 for c in self.cached_crops if c is not None)
        print(f"Pre-caching complete in {time.time()-t0:.1f}s! ({valid_count}/{len(self.samples)} box-prompted crops in RAM)", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_tuple = self.cached_crops[idx]
        if c_tuple is not None:
            c_three_ch, c_mask = c_tuple
            crop_three_ch = c_three_ch.astype(np.float32) / 255.0
            crop_mask = c_mask.astype(np.float32)
        else:
            # Fallback for rare un-cached sample
            crop_three_ch = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
            crop_mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)

        # Spatial data augmentations
        if self.augment:
            if random.random() > 0.5:
                crop_three_ch = np.fliplr(crop_three_ch).copy()
                crop_mask = np.fliplr(crop_mask).copy()

            if random.random() > 0.5:
                crop_three_ch = np.flipud(crop_three_ch).copy()
                crop_mask = np.flipud(crop_mask).copy()

            k = random.choice([0, 1, 2, 3])
            if k > 0:
                crop_three_ch = np.rot90(crop_three_ch, k=k).copy()
                crop_mask = np.rot90(crop_mask, k=k).copy()

        # Transpose to PyTorch (C, H, W)
        tensor_img = torch.from_numpy(crop_three_ch.transpose(2, 0, 1))
        tensor_mask = torch.from_numpy(crop_mask).unsqueeze(0)

        return tensor_img, tensor_mask

