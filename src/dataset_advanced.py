"""
Advanced Dataset for Solar Filament Training with STAPLE Consensus & Physics Inputs.
Supports:
- 3-channel input:
  1. Grayscale intensity
  2. Radial solar disk distance (r / R_disk)
  3. Polynomial-subtracted flattened chromospheric intensity (limb-darkening removed)
- Ground truth target: STAPLE consensus probability map W in [0, 1]
- Balanced filament-centric patch sampling (512x512)
"""

import os
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from src.parser import MAGFiLOParser
from src.disk import get_disk_geometry
from src.solar_physics import fit_and_subtract_polynomial_surface


class AdvancedSolarFilamentDataset(Dataset):
    def __init__(
        self,
        parser: MAGFiLOParser,
        record_ids: List[str],
        patch_size: int = 512,
        pos_ratio: float = 0.75,
        staple_dir: Optional[str] = None,
        is_train: bool = True,
    ):
        self.parser = parser
        self.record_ids = record_ids
        self.patch_size = patch_size
        self.pos_ratio = pos_ratio
        self.staple_dir = staple_dir
        self.is_train = is_train

        # Pre-index positive bbox locations per record
        self.record_bboxes: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for rec_id in record_ids:
            anns = self.parser.get_annotations_for_record(rec_id)
            bboxes = []
            for ann in anns:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                if w > 0 and h > 0:
                    bboxes.append((x, y, w, h))
            self.record_bboxes[rec_id] = bboxes

    def __len__(self) -> int:
        return len(self.record_ids)

    def _sample_crop_coords(self, rec_id: str) -> Tuple[int, int]:
        bboxes = self.record_bboxes.get(rec_id, [])
        sample_positive = self.is_train and len(bboxes) > 0 and (random.random() < self.pos_ratio)

        if sample_positive:
            bx, by, bw, bh = random.choice(bboxes)
            cx, cy = bx + bw // 2, by + bh // 2
            jitter_x = random.randint(-self.patch_size // 4, self.patch_size // 4)
            jitter_y = random.randint(-self.patch_size // 4, self.patch_size // 4)
            x0 = cx + jitter_x - self.patch_size // 2
            y0 = cy + jitter_y - self.patch_size // 2
        else:
            x0 = random.randint(0, 2048 - self.patch_size)
            y0 = random.randint(0, 2048 - self.patch_size)

        x0 = max(0, min(2048 - self.patch_size, x0))
        y0 = max(0, min(2048 - self.patch_size, y0))
        return int(x0), int(y0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec_id = self.record_ids[idx]
        fname = self.parser.get_source_filename(rec_id)
        stem = os.path.splitext(fname)[0]
        img_path = self.parser.get_image_path(fname)

        # 1. Raw grayscale image
        raw_img = np.array(Image.open(img_path), dtype=np.uint8)
        norm_img = raw_img.astype(np.float32) / 255.0

        # 2. Solar disk mask & radial distance
        center, radius = get_disk_geometry(fname)
        H, W = 2048, 2048
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        disk_mask = dist_from_center <= radius
        norm_radial = np.clip(dist_from_center / max(radius, 1.0), 0.0, 1.0).astype(np.float32)

        # 3. Target Mask (STAPLE consensus if available, else record binary mask)
        target_mask = None
        if self.staple_dir is not None:
            staple_path = os.path.join(self.staple_dir, f"{stem}.png")
            if os.path.exists(staple_path):
                s_img = np.array(Image.open(staple_path), dtype=np.uint8)
                target_mask = s_img.astype(np.float32) / 255.0

        if target_mask is None:
            # Fallback to single record binary mask
            target_mask = np.zeros((H, W), dtype=np.float32)
            anns = self.parser.get_annotations_for_record(rec_id)
            for ann in anns:
                ann_m = self.parser.polygon_to_mask(ann["segmentation"], H, W)
                target_mask[ann_m > 0] = 1.0

        # Sample crop coordinates
        x0, y0 = self._sample_crop_coords(rec_id)

        # Crop 2-channel input: intensity + radial distance
        p_img = norm_img[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        p_rad = norm_radial[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        p_mask = target_mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        # Augmentation
        if self.is_train:
            if random.random() < 0.5:
                p_img = np.fliplr(p_img)
                p_rad = np.fliplr(p_rad)
                p_mask = np.fliplr(p_mask)
            if random.random() < 0.5:
                p_img = np.flipud(p_img)
                p_rad = np.flipud(p_rad)
                p_mask = np.flipud(p_mask)
            # Random 90-degree rotations (dihedral)
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                p_img = np.rot90(p_img, k=k_rot)
                p_rad = np.rot90(p_rad, k=k_rot)
                p_mask = np.rot90(p_mask, k=k_rot)

        # Stack input channels: (2, patch_size, patch_size)
        input_tensor = np.stack([p_img, p_rad], axis=0).astype(np.float32)
        mask_tensor = p_mask[None, ...].astype(np.float32)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(input_tensor)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_tensor)),
            "record_id": rec_id,
            "filename": fname,
            "coords": torch.tensor([x0, y0], dtype=torch.int32),
        }
