"""
PyTorch Dataset and Tiling Pipeline for Solar Filament Segmentation.
Supports full-resolution 2048x2048 solar images, filament-centric patch sampling,
overlapping grid tiling, and seamless full-resolution windowed reconstruction.
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


class SolarFilamentDataset(Dataset):
    """
    Filament-aware patch-based PyTorch Dataset for training.
    Extracts crops of size (patch_size, patch_size) with balanced foreground/background sampling.
    """

    def __init__(
        self,
        parser: MAGFiLOParser,
        record_ids: List[str],
        patch_size: int = 512,
        pos_ratio: float = 0.75,
        transform: Optional[Any] = None,
        is_train: bool = True,
    ):
        self.parser = parser
        self.record_ids = record_ids
        self.patch_size = patch_size
        self.pos_ratio = pos_ratio
        self.transform = transform
        self.is_train = is_train

        # Pre-index positive bbox locations per record for fast filament-centric sampling
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
            # Pick a random filament bounding box
            bx, by, bw, bh = random.choice(bboxes)
            # Center of filament box
            cx, cy = bx + bw // 2, by + bh // 2
            # Add random jitter within [-patch_size//4, patch_size//4]
            jitter_x = random.randint(-self.patch_size // 4, self.patch_size // 4)
            jitter_y = random.randint(-self.patch_size // 4, self.patch_size // 4)
            x0 = cx + jitter_x - self.patch_size // 2
            y0 = cy + jitter_y - self.patch_size // 2
        else:
            # Uniform random crop across the 2048x2048 solar disk
            x0 = random.randint(0, 2048 - self.patch_size)
            y0 = random.randint(0, 2048 - self.patch_size)

        # Clamp inside image bounds [0, 2048 - patch_size]
        x0 = max(0, min(2048 - self.patch_size, x0))
        y0 = max(0, min(2048 - self.patch_size, y0))
        return int(x0), int(y0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec_id = self.record_ids[idx]
        fname = self.parser.get_source_filename(rec_id)
        img_path = self.parser.get_image_path(fname)

        # Load raw grayscale image
        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Reconstruct full 2048x2048 binary mask for this record
        mask = np.zeros((2048, 2048), dtype=np.float32)
        anns = self.parser.get_annotations_for_record(rec_id)
        for ann in anns:
            ann_m = self.parser.polygon_to_mask(ann["segmentation"])
            mask[ann_m > 0] = 1.0

        # Sample patch coordinates
        x0, y0 = self._sample_crop_coords(rec_id)
        img_patch = img[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        mask_patch = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        # Spatial augmentations if training
        if self.is_train:
            # Horizontal flip
            if random.random() < 0.5:
                img_patch = np.fliplr(img_patch)
                mask_patch = np.fliplr(mask_patch)
            # Vertical flip
            if random.random() < 0.5:
                img_patch = np.flipud(img_patch)
                mask_patch = np.flipud(mask_patch)

        # Convert to torch tensor: (1, H, W)
        img_t = torch.from_numpy(np.ascontiguousarray(img_patch)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_patch)).unsqueeze(0).float()

        return {
            "image": img_t,
            "mask": mask_t,
            "record_id": rec_id,
            "filename": fname,
            "coords": torch.tensor([x0, y0], dtype=torch.int32),
        }


# =========================================================================
# Tiler & Full-Resolution Reconstructor for Validation & Test Inference
# =========================================================================

class GridTiler:
    """
    Decomposes a 2048x2048 image into overlapping tiles and seamlessly
    reconstructs the full 2048x2048 prediction using a 2D Gaussian/spline blend window.
    """

    def __init__(self, img_size: int = 2048, patch_size: int = 512, overlap: int = 128):
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

        # Compute grid step positions
        self.coords: List[Tuple[int, int]] = []
        y_steps = list(range(0, img_size - patch_size, self.stride)) + [img_size - patch_size]
        x_steps = list(range(0, img_size - patch_size, self.stride)) + [img_size - patch_size]
        y_steps = sorted(list(set(y_steps)))
        x_steps = sorted(list(set(x_steps)))

        for y in y_steps:
            for x in x_steps:
                self.coords.append((x, y))

        # Build 2D Hann / Cosine blending window to eliminate seam artifacts at tile boundaries
        w1d = np.hanning(patch_size)
        w2d = np.outer(w1d, w1d).astype(np.float32)
        # Avoid zeros at edges
        w2d = np.clip(w2d, 1e-4, 1.0)
        self.blend_window = w2d

    @property
    def num_tiles(self) -> int:
        return len(self.coords)

    def extract_tiles(self, img_np: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Extracts all tiles and their (x, y) coordinates from a (2048, 2048) image."""
        tiles = []
        for x, y in self.coords:
            tile = img_np[y : y + self.patch_size, x : x + self.patch_size]
            tiles.append((tile, (x, y)))
        return tiles

    def reconstruct(self, pred_tiles: List[Tuple[np.ndarray, Tuple[int, int]]]) -> np.ndarray:
        """
        Reconstructs full (2048, 2048) probability map with weighted window blending.
        pred_tiles: list of (pred_tile_2d, (x, y))
        """
        full_prob = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        weight_sum = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        for pred_tile, (x, y) in pred_tiles:
            weighted_tile = pred_tile * self.blend_window
            full_prob[y : y + self.patch_size, x : x + self.patch_size] += weighted_tile
            weight_sum[y : y + self.patch_size, x : x + self.patch_size] += self.blend_window

        full_prob /= np.maximum(weight_sum, 1e-6)
        return full_prob
