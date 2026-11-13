"""
Foundation Model Solar Filament Boundary Refiner using Segment Anything (SAM).
Combines:
1. Stage 1 YOLO11s-seg candidate box proposals.
2. Chromospheric darkest medial spine point prompting (absorption valley detection).
3. Negative corner quiet sun point prompting.
4. SAM Vision Transformer boundary segmentation.
5. Physics-informed rejection filters (Bernasconi et al. 2005 / Fuller et al. 2008).
6. Collinear Medial Axis Spine Reconnection.
7. Disjoint merge for official competition compliance.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

from segment_anything import sam_model_registry, SamPredictor

from src.crop_utils import (
    compute_dynamic_square_crop,
    get_four_quadrants,
    positive_overlap_grouping,
    disjoint_merge_instances,
    is_sunspot_or_compact_blob,
    is_outside_solar_disk_margin,
    is_insufficient_contrast,
)
from src.sam_prompting import extract_darkest_spine_points
from src.spine_reconnection import reconnect_collinear_spines
from src.disk import Disk, detect_disk, DiskCache


class SAMFilamentRefiner:
    """
    Prompt-guided Foundation Model Refiner for Solar Filaments.
    """

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda:0",
        min_area: int = 180,
        enable_physics_filters: bool = True,
        enable_spine_reconnection: bool = True,
    ):
        self.device = device
        self.min_area = min_area
        self.enable_physics_filters = enable_physics_filters
        self.enable_spine_reconnection = enable_spine_reconnection

        print(f"Loading SAM ({model_type}) from {sam_checkpoint} on {device}...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        self.predictor = SamPredictor(sam)
        print("SAM Predictor initialized successfully.")

    def refine_single_candidate(
        self,
        raw_rgb: np.ndarray,
        raw_gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
        stage1_score: float,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Runs SAM on a single candidate bounding box using prompt engineering.
        """
        bx0, by0, bx1, by1 = bbox
        if (bx1 - bx0) < 8 or (by1 - by0) < 8:
            return None

        # 1. Extract darkest chromospheric spine point prompts
        coords, labels = extract_darkest_spine_points(raw_gray, bbox, num_points=3)
        if len(coords) == 0:
            return None

        # 2. Dynamic 1.8x native square crop around candidate
        cx0, cy0, cx1, cy1, side = compute_dynamic_square_crop(
            bbox=bbox,
            context=1.8,
            min_side=96,
            max_side=2048,
            img_h=2048,
            img_w=2048,
        )

        crop_rgb = raw_rgb[cy0:cy1, cx0:cx1]
        if crop_rgb.shape[0] < 16 or crop_rgb.shape[1] < 16:
            return None

        # 3. Transform global coordinates to local crop coordinates
        local_pts = coords - np.array([cx0, cy0], dtype=np.float32)
        local_box = np.array([bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0], dtype=np.float32)

        # 4. Predict with SAM
        self.predictor.set_image(crop_rgb)
        masks, scores, logits = self.predictor.predict(
            point_coords=local_pts,
            point_labels=labels,
            box=local_box[None, :],
            multimask_output=False,
        )

        sam_mask_local = masks[0]
        sam_score = float(scores[0])

        if sam_mask_local.sum() < self.min_area:
            return None

        # 5. Project back onto full 2048x2048 global canvas
        global_mask = np.zeros((2048, 2048), dtype=bool)
        global_mask[cy0:cy1, cx0:cx1] = sam_mask_local

        # Combined confidence score
        blended_score = 0.5 * stage1_score + 0.5 * sam_score
        return global_mask, blended_score

    def process_image(
        self,
        raw_img: np.ndarray,
        disk_mask: np.ndarray,
        candidate_tasks: List[Dict[str, Any]],
        disk: Optional[Disk] = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Processes all detected candidate boxes for one 2048x2048 image with SAM.
        """
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
        refined_masks = []
        refined_scores = []

        for task in candidate_tasks:
            bbox = task["bbox"]
            score = task["score"]

            res = self.refine_single_candidate(raw_rgb, raw_img, bbox, score)
            if res is None:
                continue

            canvas_mask, final_score = res
            canvas_mask &= (disk_mask > 0)
            mask_area = int(canvas_mask.sum())
            if mask_area < self.min_area:
                continue

            # Physics Rejection Filters
            if self.enable_physics_filters:
                if is_sunspot_or_compact_blob(canvas_mask, area=mask_area, min_eccentricity=0.65):
                    continue
                if disk is not None and is_outside_solar_disk_margin(
                    canvas_mask, disk.cx, disk.cy, disk.radius, max_r_ratio=0.965
                ):
                    continue
                if is_insufficient_contrast(canvas_mask, raw_img, max_contrast_ratio=0.92):
                    continue

            refined_masks.append(canvas_mask)
            refined_scores.append(final_score)

        if len(refined_masks) == 0:
            return []

        # Collinear Spine Reconnection
        if self.enable_spine_reconnection and len(refined_masks) > 1:
            candidate_pairs = list(zip(refined_masks, refined_scores))
            candidate_pairs = reconnect_collinear_spines(
                candidate_pairs,
                raw_img=raw_img,
                max_gap=40.0,
                max_angle_deg=35.0,
                bridge_thickness=5,
                min_area=self.min_area,
            )
            refined_masks = [m for m, s in candidate_pairs]
            refined_scores = [s for m, s in candidate_pairs]

        if len(refined_masks) == 0:
            return []

        # Disjoint Merge (Zero Overlap Guarantee)
        final_instances = disjoint_merge_instances(refined_masks, refined_scores, min_area=self.min_area)
        return final_instances
