"""
Automated unit tests for Dataset loading, Tiling, and Reconstruction (Priority 4).
"""

import os
import sys
import unittest
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.parser import MAGFiLOParser
from src.dataset import SolarFilamentDataset, GridTiler
from src.paths import get_dataset_paths, get_split_path

JSON_PATH, TRAIN_IMG_DIR, _ = get_dataset_paths()
SPLIT_PATH = get_split_path()


class TestDatasetAndTiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = MAGFiLOParser(JSON_PATH, TRAIN_IMG_DIR)
        with open(SPLIT_PATH, "r") as f:
            cls.split = json.load(f)
        cls.train_recs = cls.split["train_record_ids"]

    def test_dataset_item_generation(self):
        ds = SolarFilamentDataset(
            parser=self.parser,
            record_ids=self.train_recs[:10],
            patch_size=512,
            pos_ratio=1.0,  # Force filament-centric crop
            is_train=True,
        )
        self.assertEqual(len(ds), 10)
        item = ds[0]

        # Verify shapes and types
        self.assertEqual(item["image"].shape, (1, 512, 512))
        self.assertEqual(item["mask"].shape, (1, 512, 512))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertEqual(item["mask"].dtype, torch.float32)

        # Value bounds
        self.assertTrue((item["image"] >= 0.0).all() and (item["image"] <= 1.0).all())
        self.assertTrue(set(item["mask"].unique().tolist()).issubset({0.0, 1.0}))
        # With pos_ratio=1.0 on annotated image, foreground should be present
        self.assertGreater(item["mask"].sum().item(), 0)

    def test_grid_tiler_512_reconstruction(self):
        tiler = GridTiler(img_size=2048, patch_size=512, overlap=128)
        self.assertGreater(tiler.num_tiles, 0)

        # Synthetic smooth test pattern: 2D linear gradient
        y = np.linspace(0, 1, 2048, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, 2048, dtype=np.float32)[None, :]
        test_pattern = (y + x) / 2.0

        tiles = tiler.extract_tiles(test_pattern)
        self.assertEqual(len(tiles), tiler.num_tiles)
        for tile, coords in tiles:
            self.assertEqual(tile.shape, (512, 512))

        # Reconstruct full image
        reconstructed = tiler.reconstruct(tiles)
        self.assertEqual(reconstructed.shape, (2048, 2048))

        # Max absolute error between original pattern and reconstructed
        max_err = np.max(np.abs(test_pattern - reconstructed))
        self.assertLess(max_err, 1e-4, f"Reconstruction error too high: {max_err}")

    def test_grid_tiler_1024_reconstruction(self):
        tiler_1024 = GridTiler(img_size=2048, patch_size=1024, overlap=256)
        self.assertGreater(tiler_1024.num_tiles, 0)

        # Constant pattern
        test_pattern = np.ones((2048, 2048), dtype=np.float32) * 0.75
        tiles = tiler_1024.extract_tiles(test_pattern)
        reconstructed = tiler_1024.reconstruct(tiles)

        max_err = np.max(np.abs(test_pattern - reconstructed))
        self.assertLess(max_err, 1e-4, f"1024x1024 reconstruction error too high: {max_err}")


if __name__ == "__main__":
    unittest.main()
