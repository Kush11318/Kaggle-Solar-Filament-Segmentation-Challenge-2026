"""
Full-Resolution Tiled Inference and Instance Segmentation Module.
Takes full 2048x2048 images, evaluates overlapping tiles, reconstructs
seamless probability maps, and extracts discrete filament instances.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
from pycocotools import mask as mask_util

from src.dataset import GridTiler


def predict_full_image(
    model: torch.nn.Module,
    image_path: str,
    tiler: Optional[GridTiler] = None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
    min_area: int = 40,
    batch_size: int = 4,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Runs sliding-window tiled inference on a single 2048x2048 image.
    Returns:
        prob_map: (2048, 2048) float32 probability map
        instance_masks: list of (2048, 2048) uint8 binary masks, one per discrete instance
    """
    if tiler is None:
        tiler = GridTiler(img_size=2048, patch_size=512, overlap=128)

    model.eval()
    img_raw = np.array(Image.open(image_path), dtype=np.float32) / 255.0

    tiles = tiler.extract_tiles(img_raw)
    tile_preds: List[Tuple[np.ndarray, Tuple[int, int]]] = []

    # Batch evaluation of tiles
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i : i + batch_size]
            batch_tensors = torch.stack([
                torch.from_numpy(t[0]).unsqueeze(0).float() for t in batch
            ]).to(device)

            logits = model(batch_tensors)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for j, (tile_np, coords) in enumerate(batch):
                tile_preds.append((probs[j], coords))

    # Reconstruct full 2048x2048 probability map
    prob_map = tiler.reconstruct(tile_preds)

    # Threshold binary foreground
    binary_mask = (prob_map >= threshold).astype(np.uint8)

    # Extract connected components (baseline instance separation)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    instance_masks = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            inst_m = (labels == lbl).astype(np.uint8)
            instance_masks.append(inst_m)

    return prob_map, instance_masks


def generate_submission_dataframe(
    model: torch.nn.Module,
    image_paths: List[str],
    image_ids: List[str],
    tiler: Optional[GridTiler] = None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
    min_area: int = 40,
) -> pd.DataFrame:
    """
    Runs full inference over a list of images and builds the official submission DataFrame.
    Columns: ['filament_id', 'segmentation_rle']
    """
    rows = []
    for img_p, img_id in zip(image_paths, image_ids):
        _, inst_masks = predict_full_image(
            model=model,
            image_path=img_p,
            tiler=tiler,
            device=device,
            threshold=threshold,
            min_area=min_area,
        )

        for idx, inst_m in enumerate(inst_masks, start=1):
            f_mask = np.asfortranarray(inst_m)
            rle_dict = mask_util.encode(f_mask)
            rle_str = rle_dict["counts"].decode("utf-8")
            fil_id = f"{img_id}_{idx}"
            rows.append({"filament_id": fil_id, "segmentation_rle": rle_str})

    return pd.DataFrame(rows)
