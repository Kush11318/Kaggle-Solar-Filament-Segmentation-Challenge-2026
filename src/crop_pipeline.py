"""
Complete Decoupled Two-Stage Inference Pipeline for Solar Filament Segmentation.
Resolves the tension between memory limitations and fine boundary preservation:
- Stage 1: 4-Quadrant Tiled Detection (1100x1100, 152 px overlap) via YOLO11-seg.
- Decoupling Module: Positive Overlap Grouping (S > 0) to eliminate boundary seam tears.
- Stage 2: Dynamic Native Square Crops (1.8x context) refined by CropUNet at 256x256.
- Multi-Model Ensemble Support (ResNet-18 + ResNet-34).
- Topology-Safe Flip-4 TTA with guardrails (area ratio, CC limit, internal Dice >= 0.62).
- Bernasconi & Fuller Chromospheric Physics Rejection Engine:
    1. Sunspot & compact pore rejection (eccentricity < 0.65, area < 1800 px).
    2. Solar limb fringe exclusion (r / R_sun > 0.965).
    3. Chromospheric absorption contrast verification (I_filament / I_sun <= 0.92).
- Continuous re-projection, high thresholding (tau = 0.70) with detector fallback.
- Disjoint merge for official zero-overlap guarantee.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union
from pycocotools import mask as mask_util
from ultralytics import YOLO

from src.crop_utils import (
    compute_dynamic_square_crop,
    get_four_quadrants,
    positive_overlap_grouping,
    disjoint_merge_instances,
    split_narrow_necks,
    is_sunspot_or_compact_blob,
    is_outside_solar_disk_margin,
    is_insufficient_contrast,
)
from src.spine_reconnection import reconnect_collinear_spines
from src.topological_cleaner import topological_clean_and_extend_tail
from src.disk import Disk, detect_disk, make_two_channel_input, DiskCache


def snap_to_absorption_edges(
    mask: np.ndarray,
    raw_img: np.ndarray,
    fringe_width: int = 2,
    min_core_area: int = 40,
) -> np.ndarray:
    """
    Feature 4 (Halo Dilation Cure): Snaps soft CNN probability masks directly
    to the 2K physical optical absorption gradient (nabla I).
    Evaluates raw H-alpha absorption intensities in the outer boundary fringe.
    Trims boundary pixels that are brighter than the local dark absorption trench.
    """
    if mask.sum() < min_core_area or raw_img is None:
        return mask

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask
    y0 = max(0, int(ys.min()) - 3)
    y1 = min(mask.shape[0], int(ys.max()) + 4)
    x0 = max(0, int(xs.min()) - 3)
    x1 = min(mask.shape[1], int(xs.max()) + 4)

    sub_mask = mask[y0:y1, x0:x1]
    sub_raw = raw_img[y0:y1, x0:x1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_width * 2 + 1, fringe_width * 2 + 1))
    eroded = cv2.erode(sub_mask.astype(np.uint8), kernel)
    fringe = (sub_mask > 0) & (eroded == 0)
    core = (eroded > 0)

    if core.sum() < 20 or fringe.sum() == 0:
        return mask

    core_vals = sub_raw[core]
    core_mean = float(np.mean(core_vals))
    core_std = float(np.std(core_vals))

    # Boundary cutoff: any fringe pixel brighter than local absorption trench belongs to background granulation
    cutoff = core_mean + 1.25 * max(core_std, 12.0)
    fringe_vals = sub_raw[fringe]
    keep_fringe = fringe_vals <= cutoff

    snapped = mask.copy()
    fy, fx = np.where(fringe)
    remove_y = y0 + fy[~keep_fringe]
    remove_x = x0 + fx[~keep_fringe]
    snapped[remove_y, remove_x] = False
    return snapped


class TwoStageCropPipeline:
    """
    Decoupled Two-Stage Pipeline: Global Tiled Detection + Local Native Crop Refinement.
    """

    def __init__(
        self,
        crop_unet_model: Union[torch.nn.Module, List[torch.nn.Module]],
        yolo_model: YOLO,
        device: torch.device,
        det_conf: float = 0.20,
        refine_thresh: float = 0.75,
        min_area: int = 180,
        enable_flip4_tta: bool = True,
        enable_physics_filters: bool = True,
        sunspot_ecc_thresh: float = 0.65,
        max_limb_r_ratio: float = 0.965,
        enable_spine_reconnection: bool = True,
        max_spine_gap: float = 40.0,
        max_spine_angle: float = 35.0,
        use_full_disk: bool = True,
        context_factor: float = 1.8,
        enable_edge_snapping: bool = False,
        enable_topological_cleaner: bool = False,
        tau_tail: float = 0.40,
        max_tail_grow: int = 15,
        yolo_imgsz: int = 1280,
        quad_imgsz: int = 1120,
        fallback_to_stage1: bool = False,
        model_weights: Optional[List[float]] = None,
        crop_size: int = 256,
    ):
        if crop_unet_model is None:
            self.unets = []
        elif isinstance(crop_unet_model, (list, tuple)):
            self.unets = [m.to(device).eval() for m in crop_unet_model if m is not None]
        else:
            self.unets = [crop_unet_model.to(device).eval()]
        self.model_weights = None
        if model_weights is not None and len(model_weights) == len(self.unets):
            w_arr = np.array(model_weights, dtype=np.float32)
            self.model_weights = (w_arr / w_arr.sum()).tolist()
        self.yolo = yolo_model
        self.device = device
        self.crop_size = crop_size
        self.det_conf = det_conf
        self.refine_thresh = refine_thresh
        self.min_area = min_area
        self.enable_flip4_tta = enable_flip4_tta
        self.enable_physics_filters = enable_physics_filters
        self.sunspot_ecc_thresh = sunspot_ecc_thresh
        self.max_limb_r_ratio = max_limb_r_ratio
        self.enable_spine_reconnection = enable_spine_reconnection
        self.max_spine_gap = max_spine_gap
        self.max_spine_angle = max_spine_angle
        self.use_full_disk = use_full_disk
        self.context_factor = context_factor
        self.enable_edge_snapping = enable_edge_snapping
        self.enable_topological_cleaner = enable_topological_cleaner
        self.tau_tail = tau_tail
        self.max_tail_grow = max_tail_grow
        self.yolo_imgsz = yolo_imgsz
        self.quad_imgsz = quad_imgsz
        self.fallback_to_stage1 = fallback_to_stage1

    @torch.no_grad()
    def _predict_crop_with_tta(self, crop_two_ch: np.ndarray) -> np.ndarray:
        """
        Runs CropUNet (or ensemble of CropUNets) with Topology-Safe Flip-4 TTA.
        Views: identity, hflip, vflip, hvflip.
        Guardrails: Area ratio [0.60, 1.70], connected components limit, internal Dice >= 0.62.
        """
        model_predictions = []

        views = [
            ("hflip", lambda img: np.fliplr(img).copy(), lambda p: np.fliplr(p)),
            ("vflip", lambda img: np.flipud(img).copy(), lambda p: np.flipud(p)),
            ("hvflip", lambda img: np.flipud(np.fliplr(img)).copy(), lambda p: np.flipud(np.fliplr(p))),
        ]

        for model in self.unets:
            tensor = torch.from_numpy(crop_two_ch).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            prob_id = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

            if not self.enable_flip4_tta:
                model_predictions.append(prob_id)
                continue

            id_bin = (prob_id >= 0.50).astype(np.uint8)
            id_area = int(id_bin.sum())
            num_cc_id, _ = cv2.connectedComponents(id_bin)

            valid_probs = [prob_id]

            for name, flip_img, unflip_prob in views:
                img_flipped = flip_img(crop_two_ch)
                t_flip = torch.from_numpy(img_flipped).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                p_flip = torch.sigmoid(model(t_flip)).squeeze().cpu().numpy()
                p_unflipped = unflip_prob(p_flip)

                flip_bin = (p_unflipped >= 0.50).astype(np.uint8)
                flip_area = int(flip_bin.sum())

                # Guardrail 1: Area ratio [0.60, 1.70]
                if id_area > 0:
                    ratio = flip_area / max(1, id_area)
                    if ratio < 0.60 or ratio > 1.70:
                        continue

                # Guardrail 2: Connected components limit (prevent artificial fragmentation)
                num_cc_flip, _ = cv2.connectedComponents(flip_bin)
                if num_cc_flip > num_cc_id + 1:
                    continue

                # Guardrail 3: Internal Dice >= 0.62
                inter = np.logical_and(id_bin, flip_bin).sum()
                total = id_bin.sum() + flip_bin.sum()
                internal_dice = (2.0 * inter) / max(1, total)
                if internal_dice < 0.62:
                    continue

                valid_probs.append(p_unflipped)

            model_predictions.append(np.mean(valid_probs, axis=0))

        if self.model_weights is not None and len(self.model_weights) == len(model_predictions):
            weighted_arr = sum(w * p for w, p in zip(self.model_weights, model_predictions))
            return weighted_arr
        return np.mean(model_predictions, axis=0)

    def detect_global_quadrants(self, raw_img: np.ndarray, disk_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Stage 1: Evaluates 4 overlapping quadrants (1100x1100, 152 px overlap).
        Projects detections to global 2048x2048 coordinate canvas.
        Applies Positive Overlap Grouping (S > 0) to unify cross-quadrant candidates.
        """
        quads = get_four_quadrants(2048, 2048, tile_size=1100, overlap=152)
        raw_masks = []
        raw_scores = []

        for q in quads:
            crop = raw_img[q["y0"]:q["y1"], q["x0"]:q["x1"]]
            res = self.yolo.predict(
                source=crop,
                conf=self.det_conf,
                imgsz=self.quad_imgsz,
                device=str(self.device).replace("cuda:", ""),
                verbose=False,
                retina_masks=True,
            )[0]

            if res.masks is None or len(res.masks) == 0:
                continue

            masks_data = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for m, conf in zip(masks_data, confs):
                # Resize to quadrant size 1100x1100
                m_quad = cv2.resize((m > 0.5).astype(np.uint8), (q["x1"] - q["x0"], q["y1"] - q["y0"]), interpolation=cv2.INTER_NEAREST)
                if m_quad.sum() == 0:
                    continue

                # Place onto global 2048x2048 canvas
                m_global = np.zeros((2048, 2048), dtype=bool)
                m_global[q["y0"]:q["y1"], q["x0"]:q["x1"]] = (m_quad > 0)
                m_global &= (disk_mask > 0)

                if m_global.sum() < 60:
                    continue

                raw_masks.append(m_global)
                raw_scores.append(float(conf))

        if len(raw_masks) == 0:
            return []

        # Decoupling Module: Positive Overlap Grouping (S > 0)
        tasks = positive_overlap_grouping(raw_masks, raw_scores)
        return tasks

    def detect_full_disk(self, raw_img: np.ndarray, disk_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Stage 1: Evaluates the full 2048x2048 solar disk directly at imgsz=1280.
        Eliminates all quadrant boundary seam tears and center-crossing amputations.
        """
        res = self.yolo.predict(
            source=raw_img,
            conf=self.det_conf,
            imgsz=self.yolo_imgsz,
            device=str(self.device).replace("cuda:", ""),
            verbose=False,
            retina_masks=True,
        )[0]

        if res.masks is None or len(res.masks) == 0:
            return []

        masks_data = res.masks.data.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        raw_masks = []
        raw_scores = []

        for m, conf in zip(masks_data, confs):
            m_2048 = cv2.resize((m > 0.5).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST)
            m_global = (m_2048 > 0) & (disk_mask > 0)
            if m_global.sum() < 60:
                continue

            raw_masks.append(m_global)
            raw_scores.append(float(conf))

        if len(raw_masks) == 0:
            return []

        return positive_overlap_grouping(raw_masks, raw_scores, iou_thresh=0.40)

    def process_image(
        self,
        raw_img: np.ndarray,
        disk_mask: np.ndarray,
        two_ch: np.ndarray,
        disk: Optional[Disk] = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Full 2-Stage processing for one 2048x2048 image:
        1. Stage 1: Full-disk global detection (or 4-Quadrant detection) + positive overlap grouping.
        2. Stage 2: Dynamic square crop extraction (1.4x context) + CropUNet Flip-4 TTA.
        3. Bilinear un-resize to native crop dimensions.
        4. High thresholding with fallback to Stage 1 detector mask if empty.
        5. Bernasconi & Fuller Chromospheric Physics Rejection Engine:
           - Sunspot / circular active region pore rejection
           - Outer solar limb fringe exclusion
           - Chromospheric absorption contrast verification
        6. Disjoint merge on global canvas.
        """
        tasks = self.detect_full_disk(raw_img, disk_mask) if self.use_full_disk else self.detect_global_quadrants(raw_img, disk_mask)
        if len(tasks) == 0:
            return []

        refined_masks = []
        refined_scores = []

        for task in tasks:
            bbox = task["bbox"]
            score = task["score"]
            stage1_mask = task["merged_mask"]

            if len(self.unets) == 0:
                canvas_mask = stage1_mask.copy() & (disk_mask > 0)
                mask_area = int(canvas_mask.sum())
                if mask_area >= self.min_area:
                    refined_masks.append(canvas_mask)
                    refined_scores.append(score)
                continue

            # Dynamic Native Square Crop with configured context factor
            x0, y0, x1, y1, side = compute_dynamic_square_crop(
                bbox=bbox,
                context=self.context_factor,
                min_side=96,
                max_side=2048,
                img_h=2048,
                img_w=2048,
            )

            crop_h, crop_w = y1 - y0, x1 - x0
            if crop_h < 16 or crop_w < 16:
                continue

            # Resize each channel to self.crop_size x self.crop_size for Stage 2 CropUNet
            ch0 = cv2.resize(two_ch[0, y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            ch1 = cv2.resize(two_ch[1, y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            crop_input = np.stack([ch0, ch1], axis=-1)

            # Predict with Topology-Safe Flip-4 TTA (supports ensembling)
            crop_prob = self._predict_crop_with_tta(crop_input)

            # Un-resize continuous probability map back to native crop dimensions
            native_prob = cv2.resize(crop_prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            # Threshold at tau = 0.70
            refined_bin_crop = (native_prob >= self.refine_thresh).astype(bool)

            crop_stage1 = stage1_mask[y0:y1, x0:x1] if stage1_mask is not None else None

            candidate_subcrops = []
            if np.any(refined_bin_crop):
                num_cc, cc_labels = cv2.connectedComponents(refined_bin_crop.astype(np.uint8))
                for cc_idx in range(1, num_cc):
                    comp = (cc_labels == cc_idx)
                    if int(comp.sum()) >= self.min_area:
                        if crop_stage1 is None or np.logical_and(comp, crop_stage1).any():
                            candidate_subcrops.append(comp)

            # Robust Fallback Check:
            if len(candidate_subcrops) == 0:
                if self.fallback_to_stage1 and stage1_mask is not None and stage1_mask.sum() >= self.min_area:
                    canvas_mask = stage1_mask.copy() & (disk_mask > 0)
                    mask_area = int(canvas_mask.sum())
                    if mask_area >= self.min_area:
                        refined_masks.append(canvas_mask)
                        refined_scores.append(score)
                continue

            for sub_crop in candidate_subcrops:
                if self.enable_topological_cleaner:
                    sub_crop = topological_clean_and_extend_tail(
                        sub_crop,
                        prob_map=native_prob,
                        max_tail_grow=self.max_tail_grow,
                        tau_tail=self.tau_tail,
                    )

                if int(sub_crop.sum()) < self.min_area:
                    continue

                # Project onto full 2048x2048 canvas
                canvas_mask = np.zeros((2048, 2048), dtype=bool)
                canvas_mask[y0:y1, x0:x1] = sub_crop
                canvas_mask &= (disk_mask > 0)

                mask_area = int(canvas_mask.sum())
                if mask_area < self.min_area:
                    continue

                # Bernasconi & Fuller Physics Rejection Filters
                if self.enable_physics_filters:
                    if is_sunspot_or_compact_blob(canvas_mask, area=mask_area, min_eccentricity=self.sunspot_ecc_thresh):
                        continue
                    if disk is not None and is_outside_solar_disk_margin(
                        canvas_mask, disk.cx, disk.cy, disk.radius, max_r_ratio=self.max_limb_r_ratio
                    ):
                        continue
                    if is_insufficient_contrast(canvas_mask, raw_img, max_contrast_ratio=0.92):
                        continue

                # 2K Optical Absorption Edge Snapping (Halo Dilation Cure)
                if self.enable_edge_snapping and raw_img is not None:
                    canvas_mask = snap_to_absorption_edges(canvas_mask, raw_img)
                    if int(canvas_mask.sum()) < self.min_area:
                        continue

                refined_masks.append(canvas_mask)
                refined_scores.append(score)

        if len(refined_masks) == 0:
            return []

        # Step 5b: Collinear Medial Axis Spine Reconnection
        if self.enable_spine_reconnection and len(refined_masks) > 1:
            candidate_pairs = list(zip(refined_masks, refined_scores))
            candidate_pairs = reconnect_collinear_spines(
                candidate_pairs,
                raw_img=raw_img,
                max_gap=self.max_spine_gap,
                max_angle_deg=self.max_spine_angle,
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

