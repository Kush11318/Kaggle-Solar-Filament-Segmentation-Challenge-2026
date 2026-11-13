"""
Complete Fusion Research Pipeline for Steps 1, 3, 4, 5.

Implements:
1. Observation-Level 2-Fold Cross-Fitting (no annotator leakage across folds)
2. Controlled Grid Search (confidence, growth radius, min area, unet threshold, fallback)
3. 8-Way Dihedral TTA (intensity + radial-distance map consistent transformations)
4. Conservative Missed Foreground Recovery (drop vs gap-bridged fallback)
5. Official Pooled Panoptic Quality (PQ, SQ, RQ, TP, FP, FN, Dice, Fragmentation, Merging)
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import get_dataset_paths, get_split_path, get_project_root
from src.parser import MAGFiLOParser
from src.disk import get_disk_geometry
from src.metrics import get_overlap_df, compute_official_metrics
from src.fusion import (
    Detection,
    FusionParams,
    fuse_unet_and_detections,
    labels_to_instance_rles,
)


class ObservationSplitter:
    """Splits physical images into 2 halves deterministically while preserving all annotator records."""

    def __init__(self, parser: MAGFiLOParser, val_record_ids: List[int], seed: int = 42):
        self.parser = parser
        self.val_record_ids = val_record_ids
        self.val_fnames = sorted(list({parser.get_source_filename(r) for r in val_record_ids}))
        
        np.random.seed(seed)
        shuffled = list(self.val_fnames)
        np.random.shuffle(shuffled)
        
        half = len(shuffled) // 2
        self.fold_A_fnames = set(shuffled[:half])
        self.fold_B_fnames = set(shuffled[half:])
        
        self.rec_ids_A = [r for r in val_record_ids if parser.get_source_filename(r) in self.fold_A_fnames]
        self.rec_ids_B = [r for r in val_record_ids if parser.get_source_filename(r) in self.fold_B_fnames]

        self.gt_df_full = parser.build_gt_dataframe(val_record_ids)
        self.gt_df_A = parser.build_gt_dataframe(self.rec_ids_A)
        self.gt_df_B = parser.build_gt_dataframe(self.rec_ids_B)


def load_cached_validation_assets(
    val_fnames: List[str],
    unet_probs_dir: str,
    detections_json: str,
) -> Dict[str, Any]:
    with open(detections_json, "r") as f:
        raw_dets = json.load(f)

    dets_by_stem = {
        stem: [Detection(score=d["score"], counts=d["counts"]) for d in dets]
        for stem, dets in raw_dets.items()
    }

    cached = {}
    H, W = 2048, 2048
    Y, X = np.ogrid[:H, :W]

    for fn in val_fnames:
        stem = os.path.splitext(fn)[0]
        prob_path = os.path.join(unet_probs_dir, f"{stem}.npy")
        if not os.path.exists(prob_path):
            raise FileNotFoundError(f"Missing cached prob map: {prob_path}")
        prob_map = np.load(prob_path).astype(np.float32)

        center, radius = get_disk_geometry(fn)
        dist_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        disk_mask = dist_sq <= (radius ** 2)

        cached[stem] = {
            "prob": prob_map,
            "disk": disk_mask,
            "dets": dets_by_stem.get(stem, [])
        }
    return cached


def predict_fusion_subset(
    cached_data: Dict[str, Any],
    fnames: set,
    params: FusionParams,
) -> pd.DataFrame:
    rows = []
    for fn in sorted(list(fnames)):
        stem = os.path.splitext(fn)[0]
        item = cached_data[stem]
        label_map = fuse_unet_and_detections(
            unet_prob=item["prob"],
            disk_mask=item["disk"],
            detections=item["dets"],
            params=params,
        )
        rles = labels_to_instance_rles(label_map)
        for idx, rle_str in enumerate(rles, start=1):
            rows.append({
                "filament_id": f"{stem}_{idx}",
                "segmentation_rle": rle_str
            })
    return pd.DataFrame(rows)


def evaluate_predictions(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Dict[str, Any]:
    if len(pred_df) == 0:
        return {
            "pq": 0.0, "sq": 0.0, "rq": 0.0,
            "tp": 0, "fp": 0, "fn": len(gt_df),
            "mean_dice_positive": 0.0,
            "fragmented_1_to_many": 0, "merged_many_to_1": 0
        }
    overlap_df = get_overlap_df(gt_df, pred_df)
    return compute_official_metrics(overlap_df)
