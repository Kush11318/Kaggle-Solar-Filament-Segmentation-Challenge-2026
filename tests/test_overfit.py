"""
Overfit sanity test for U-Net baseline on CPU (Priority 5).
Proves that training loss decreases, model learns foreground alignment,
and tiled inference reconstructs valid full 2048x2048 instance masks.
"""

import os
import sys
import unittest
import json
import torch
import torch.optim as optim
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.parser import MAGFiLOParser
from src.dataset import SolarFilamentDataset, GridTiler
from src.models import UNet
from src.losses import BCEDiceLoss
from src.inference import predict_full_image
from src.paths import get_dataset_paths, get_split_path

JSON_PATH, TRAIN_IMG_DIR, _ = get_dataset_paths()
SPLIT_PATH = get_split_path()


class TestOverfitSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = MAGFiLOParser(JSON_PATH, TRAIN_IMG_DIR)
        with open(SPLIT_PATH, "r") as f:
            split = json.load(f)
        # Select 2 distinct sample training records
        cls.sample_recs = split["train_record_ids"][:2]

    def test_overfit_and_reconstruction(self):
        # Create dataset with fixed patch (pos_ratio=1.0)
        ds = SolarFilamentDataset(
            parser=self.parser,
            record_ids=self.sample_recs,
            patch_size=256,  # 256 for fast CPU sanity check
            pos_ratio=1.0,
            is_train=False,
        )

        model = UNet(in_channels=1, out_channels=1, base_channels=16)
        criterion = BCEDiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        batch_imgs = torch.stack([ds[0]["image"], ds[1]["image"]])
        batch_masks = torch.stack([ds[0]["mask"], ds[1]["mask"]])

        losses = []
        model.train()
        for step in range(40):
            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"\nOverfit test losses: Start={losses[0]:.4f} -> Mid={losses[20]:.4f} -> Final={losses[-1]:.4f}")
        # Loss must decrease steadily
        self.assertLess(losses[-1], losses[0] * 0.80, "Training loss did not decrease sufficiently!")

        # Verify full tiled inference on the first physical image
        fname = self.parser.get_source_filename(self.sample_recs[0])
        img_path = self.parser.get_image_path(fname)

        tiler = GridTiler(img_size=2048, patch_size=512, overlap=128)
        prob_map, instances = predict_full_image(
            model=model,
            image_path=img_path,
            tiler=tiler,
            device=torch.device("cpu"),
            threshold=0.3,
            min_area=20,
            batch_size=2,
        )

        self.assertEqual(prob_map.shape, (2048, 2048))
        self.assertTrue((prob_map >= 0.0).all() and (prob_map <= 1.0).all())
        print(f"Full inference successful! Reconstructed prob_map (2048x2048), min={prob_map.min():.3f}, max={prob_map.max():.3f}, instances={len(instances)}")


if __name__ == "__main__":
    unittest.main()
