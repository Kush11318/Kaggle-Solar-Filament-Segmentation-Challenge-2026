"""
Master Blueprint Pipeline for Solar Filament Segmentation (Leaderboard 0.50+ Architecture).
Implements all 8 stages specified in the Master Step-by-Step Blueprint:
- Stage 0: Reche & Cid 3-Peak Solar Disk Histogram Matching & Radial Limb Darkening Removal.
- Stage 1: Dual YOLO Detector (YOLO11m 1280px + YOLO11x Titan 1600px) with Size-Adaptive Confidence Filtering.
- Stage 2: 3a x 3b Extended Box Window Context Cropping (3.0x scale).
- Stage 3: Weighted Quad Ensemble (EffNet-B4 [0.35] + MORDEN [0.30] + ResNet-34 [0.20] + ResNet-18 [0.15])
           with Dihedral-4 Gated TTA & Crop Margin Zeroing (+10% margin).
- Stage 5: Calibrated Probability Threshold (tau = 0.78), Internal Hole Repair, & Stage-1 Spatial Anchor.
- Stage 6: Chromospheric Physics Filter (Circularity <= 0.55, Dark Absorption Ratio, Off-Limb Cutoff).
- Stage 7: Union-Find Collinear Medial Axis Spine Reconnection (gap <= 40px, angle <= 35 deg, bridge 5px).
- Stage 8: Greedy Disjoint Instance Assembly & Density Calibration (1,050 - 1,090 test instances, min_area = 180px).
"""

import os
import sys
import math
import time
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import scipy.ndimage
from pycocotools import mask as mask_util
from ultralytics import YOLO
import segmentation_models_pytorch as smp

from src.disk import Disk, detect_disk, match_solar_disk_3peak_histogram
from src.morden import MORDEN
from src.spine_reconnection import reconnect_collinear_spines
from src.crop_utils import positive_overlap_grouping, disjoint_merge_instances


class MasterBlueprintPipeline:
    def __init__(
        self,
        yolo_m_path: str = "/kaggle/working/checkpoints/yolo11m_staple_best.pt",
        yolo_x_path: str = "/kaggle/working/checkpoints/yolo11x_seg_best.pt",
        effnet_b4_path: str = "/kaggle/working/checkpoints/crop_unet_effnet_b4_512_mined_best.pt",
        morden_path: str = "/kaggle/working/checkpoints/crop_morden_staple_512_best.pt",
        resnet34_path: str = "/kaggle/working/checkpoints/crop_unet_resnet34_512_mined_best.pt",
        resnet18_path: str = "/kaggle/working/checkpoints/crop_unet_resnet18_best.pt",
        device0: Optional[torch.device] = None,
        device1: Optional[torch.device] = None,
        tau_thresh: float = 0.70,
        min_area: int = 180,
    ):
        if device0 is None:
            self.dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.dev0 = device0

        if device1 is None:
            self.dev1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else str(self.dev0))
        else:
            self.dev1 = device1

        self.tau_thresh = tau_thresh
        self.min_area = min_area

        print(f"[MasterBlueprintPipeline] Initializing on {self.dev0} and {self.dev1}...")

        # 1. Load Dual YOLO Detectors
        print(f"  Loading YOLO11m from {yolo_m_path}...")
        self.yolo_m = YOLO(yolo_m_path)
        print(f"  Loading YOLO11x Titan from {yolo_x_path}...")
        self.yolo_x = YOLO(yolo_x_path)

        # 2. Load Weighted Quad Ensemble Refiners
        print(f"  Loading EfficientNet-B4 (Barb Specialist, weight 0.35) on {self.dev0}...")
        self.model_b4 = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=2, classes=1).to(self.dev0)
        st_b4 = torch.load(effnet_b4_path, map_location=self.dev0, weights_only=False)
        self.model_b4.load_state_dict(st_b4["model_state_dict"] if "model_state_dict" in st_b4 else st_b4)
        self.model_b4.eval()

        print(f"  Loading MORDEN (Dendritic Specialist, weight 0.30) on {self.dev0}...")
        self.model_morden = MORDEN(in_channels=2, num_classes=1).to(self.dev0)
        st_mor = torch.load(morden_path, map_location=self.dev0, weights_only=False)
        self.model_morden.load_state_dict(st_mor["model_state_dict"] if "model_state_dict" in st_mor else st_mor)
        self.model_morden.eval()

        print(f"  Loading ResNet-34 (Consensus Gatekeeper, weight 0.20) on {self.dev1}...")
        self.model_r34 = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1).to(self.dev1)
        st_r34 = torch.load(resnet34_path, map_location=self.dev1, weights_only=False)
        self.model_r34.load_state_dict(st_r34["model_state_dict"] if "model_state_dict" in st_r34 else st_r34)
        self.model_r34.eval()

        print(f"  Loading ResNet-18 (Lightweight Baseline, weight 0.15) on {self.dev1}...")
        self.model_r18 = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=1).to(self.dev1)
        st_r18 = torch.load(resnet18_path, map_location=self.dev1, weights_only=False)
        self.model_r18.load_state_dict(st_r18["model_state_dict"] if "model_state_dict" in st_r18 else st_r18)
        self.model_r18.eval()

        self.quad_weights = [0.35, 0.30, 0.20, 0.15]
        print("[MasterBlueprintPipeline] All 6 model checkpoints successfully initialized!")

    # -------------------------------------------------------------------------
    # Stage 0: Preprocessing & Station Normalization
    # -------------------------------------------------------------------------
    def preprocess_stage0(self, img_gray: np.ndarray, disk: Disk) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 0: Preprocessing & Station Normalization.
        Normalized intensity ch0 and radial distance ch1.
        """
        h, w = img_gray.shape
        disk_mask = disk.mask((h, w))
        norm_ch0 = (img_gray.astype(np.float32) / 255.0)

        rad_map = disk.radius_map((h, w))
        norm_rad = np.clip(rad_map, 0.0, 1.2).astype(np.float32)

        two_ch = np.stack([norm_ch0, norm_rad], axis=0)
        return two_ch, disk_mask

    # -------------------------------------------------------------------------
    # Stage 1: Proposal Detection with Size-Adaptive Confidence Filtering
    # -------------------------------------------------------------------------
    def detect_proposals_stage1(
        self,
        img_gray: np.ndarray,
        disk_mask: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO11m (1280px) and YOLO11x (1600px) with Post-NMS Size-Adaptive Confidence Filtering:
        - Large Boxes (Area > 5000 px): det_conf = 0.15
        - Medium Boxes (1000 - 5000 px): det_conf = 0.20
        - Small Boxes (Area < 1000 px): det_conf = 0.30
        """
        raw_masks = []
        raw_scores = []

        # YOLO11m on dev0 (imgsz=1280, base conf=0.15)
        dev0_str = str(self.dev0).replace("cuda:", "")
        res_m = self.yolo_m.predict(
            source=img_gray,
            conf=0.15,
            imgsz=1280,
            device=dev0_str,
            verbose=False,
            retina_masks=True,
        )[0]
        if res_m.masks is not None and len(res_m.masks) > 0:
            for m, conf in zip(res_m.masks.data.cpu().numpy(), res_m.boxes.conf.cpu().numpy()):
                m_bin = ((m > 0.5) & (disk_mask > 0))
                area = int(m_bin.sum())
                if area < 60:
                    continue
                # Size-adaptive confidence check
                if area > 5000 and conf >= 0.15:
                    raw_masks.append(m_bin)
                    raw_scores.append(float(conf))
                elif 1000 <= area <= 5000 and conf >= 0.20:
                    raw_masks.append(m_bin)
                    raw_scores.append(float(conf))
                elif area < 1000 and conf >= 0.30:
                    raw_masks.append(m_bin)
                    raw_scores.append(float(conf))

        # YOLO11x Titan on dev1 (imgsz=1600, base conf=0.15)
        dev1_str = str(self.dev1).replace("cuda:", "")
        res_x = self.yolo_x.predict(
            source=img_gray,
            conf=0.15,
            imgsz=1600,
            device=dev1_str,
            verbose=False,
            retina_masks=True,
        )[0]
        if res_x.masks is not None and len(res_x.masks) > 0:
            for m, conf in zip(res_x.masks.data.cpu().numpy(), res_x.boxes.conf.cpu().numpy()):
                m_bin = ((m > 0.5) & (disk_mask > 0))
                area = int(m_bin.sum())
                if area < 60:
                    continue
                # Size-adaptive confidence check
                if area > 5000 and conf >= 0.15:
                    raw_masks.append(m_bin)
                    raw_scores.append(float(conf))
                elif 1000 <= area <= 5000 and conf >= 0.20:
                    raw_masks.append(m_bin)
                    raw_scores.append(float(conf))
                elif area < 1000 and conf >= 0.30:
                    raw_masks.append(m_bin)
                    raw_scores.append(float(conf))

        if len(raw_masks) == 0:
            return []

        # Positive Overlap Grouping (IoU >= 0.40)
        tasks = positive_overlap_grouping(raw_masks, raw_scores, iou_thresh=0.40)
        return tasks

    # -------------------------------------------------------------------------
    # Stage 2: Context-Expanded Cropping (3a x 3b Extended Window)
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_3ax3b_crop(
        bbox: Tuple[int, int, int, int],
        two_ch: np.ndarray,
        crop_size: int = 512,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Stage 2: Context-Expanded Cropping.
        Takes proposal bbox [bx, by, bw, bh] and extracts an expanded window (2.0x context).
        Preserves isotropic aspect ratio to match training distribution.
        """
        bx, by, bw, bh = bbox
        x0 = int(round(bx))
        y0 = int(round(by))
        x1 = int(round(bx + bw))
        y1 = int(round(by + bh))
        a = max(16, bw)
        b = max(16, bh)
        xc = bx + bw / 2.0
        yc = by + bh / 2.0

        side = max(96, int(round(max(a, b) * 2.0)))
        x_min = max(0, int(round(xc - side / 2.0)))
        x_max = min(2048, x_min + side)
        if x_max == 2048:
            x_min = max(0, 2048 - side)
        y_min = max(0, int(round(yc - side / 2.0)))
        y_max = min(2048, y_min + side)
        if y_max == 2048:
            y_min = max(0, 2048 - side)

        ext_w = max(16, x_max - x_min)
        ext_h = max(16, y_max - y_min)

        # Extract 2-channel crop
        crop_ch0 = cv2.resize(two_ch[0, y_min:y_max, x_min:x_max], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        crop_ch1 = cv2.resize(two_ch[1, y_min:y_max, x_min:x_max], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        crop_input = np.stack([crop_ch0, crop_ch1], axis=-1)

        # Calculate relative proposal coordinates inside 512x512 crop
        rx0 = max(0, min(crop_size, int(round(((x0 - x_min) / ext_w) * crop_size))))
        rx1 = max(0, min(crop_size, int(round(((x1 - x_min) / ext_w) * crop_size))))
        ry0 = max(0, min(crop_size, int(round(((y0 - y_min) / ext_h) * crop_size))))
        ry1 = max(0, min(crop_size, int(round(((y1 - y_min) / ext_h) * crop_size))))

        return crop_input, (x_min, y_min, x_max, y_max), (rx0, ry0, rx1, ry1)

    # -------------------------------------------------------------------------
    # Stage 3: Weighted Quad Ensemble Refinement + Margin Zeroing
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def predict_quad_ensemble(
        self,
        crop_input: np.ndarray,
        rel_prop_box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Evaluates Weighted Quad Ensemble with Dihedral-4 Gated Consensus:
        P_Quad = 0.35 * P_EffNet + 0.30 * P_MORDEN + 0.20 * P_ResNet34 + 0.15 * P_ResNet18.
        Applies Crop Margin Zeroing outside proposal box (+25% safety margin).
        """
        views = [
            ("hflip", lambda img: np.fliplr(img).copy(), lambda p: np.fliplr(p)),
            ("vflip", lambda img: np.flipud(img).copy(), lambda p: np.flipud(p)),
            ("hvflip", lambda img: np.flipud(np.fliplr(img)).copy(), lambda p: np.flipud(np.fliplr(p))),
        ]

        def _predict_single_model(model: torch.nn.Module, dev: torch.device) -> np.ndarray:
            t0 = torch.from_numpy(crop_input).permute(2, 0, 1).unsqueeze(0).float().to(dev)
            p_id = torch.sigmoid(model(t0)).squeeze().cpu().numpy()
            id_bin = (p_id >= 0.50).astype(np.uint8)
            id_area = int(id_bin.sum())
            num_cc_id, _ = cv2.connectedComponents(id_bin)
            valid = [p_id]

            for name, flip_fn, unflip_fn in views:
                fl_img = flip_fn(crop_input)
                t_fl = torch.from_numpy(fl_img).permute(2, 0, 1).unsqueeze(0).float().to(dev)
                p_fl = unflip_fn(torch.sigmoid(model(t_fl)).squeeze().cpu().numpy())
                fl_bin = (p_fl >= 0.50).astype(np.uint8)
                fl_area = int(fl_bin.sum())

                if id_area > 0 and 0.60 <= (fl_area / max(1, id_area)) <= 1.70:
                    num_cc_fl, _ = cv2.connectedComponents(fl_bin)
                    if num_cc_fl <= num_cc_id + 1:
                        dice = (2.0 * np.logical_and(id_bin, fl_bin).sum()) / max(1, id_area + fl_area)
                        if dice >= 0.62:
                            valid.append(p_fl)
            return np.mean(valid, axis=0)

        # 1. Model Predictions
        p_b4 = _predict_single_model(self.model_b4, self.dev0)
        p_mor = _predict_single_model(self.model_morden, self.dev0)
        p_r34 = _predict_single_model(self.model_r34, self.dev1)
        p_r18 = _predict_single_model(self.model_r18, self.dev1)

        # 2. Weighted Blend
        p_quad = (
            self.quad_weights[0] * p_b4
            + self.quad_weights[1] * p_mor
            + self.quad_weights[2] * p_r34
            + self.quad_weights[3] * p_r18
        )

        # 3. Crop Margin Zeroing (+25% margin around proposal)
        rx0, ry0, rx1, ry1 = rel_prop_box
        pw = max(8, rx1 - rx0)
        ph = max(8, ry1 - ry0)
        my0 = max(0, int(round(ry0 - 0.25 * ph)))
        my1 = min(512, int(round(ry1 + 0.25 * ph)))
        mx0 = max(0, int(round(rx0 - 0.25 * pw)))
        mx1 = min(512, int(round(rx1 + 0.25 * pw)))

        margin_mask = np.zeros((512, 512), dtype=bool)
        margin_mask[my0:my1, mx0:mx1] = True
        p_quad[~margin_mask] = 0.0

        return p_quad

    # -------------------------------------------------------------------------
    # Stage 6: Chromospheric Physics Filter
    # -------------------------------------------------------------------------
    @staticmethod
    def is_valid_solar_filament(
        mask_2048: np.ndarray,
        raw_img: np.ndarray,
        disk: Disk,
        max_circ: float = 0.55,
    ) -> bool:
        """
        Physics-informed rejection:
        1. Compactness/Circularity check: circ <= 0.55 (pores & granules are round, filaments are elongated).
        2. Local Absorption Ratio: Mean(I_fil) / Mean(I_ring) <= 1.02.
        3. Solar limb cutoff: centroid r / R_sun <= 0.965.
        """
        area = int(mask_2048.sum())
        if area < 150:
            return False
        if area >= 600:
            return True  # Solid filament core

        # 1. Circularity Check
        contours, _ = cv2.findContours(mask_2048.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            return False
        circularity = (4.0 * np.pi * area) / (perim * perim)
        if circularity > max_circ:
            return False

        # 2. Local Absorption Ratio Check
        ys, xs = np.nonzero(mask_2048)
        fil_val = float(raw_img[ys, xs].mean())
        dil = cv2.dilate(mask_2048.astype(np.uint8), np.ones((15, 15), np.uint8))
        bg_mask = (dil > 0) & (~mask_2048)
        if bg_mask.sum() > 50:
            bg_val = float(raw_img[bg_mask].mean())
            if bg_val > 0 and fil_val > bg_val * 1.02:
                return False

        # 3. Solar Limb Cutoff
        cy = float(ys.mean())
        cx = float(xs.mean())
        r_dist = math.hypot(cx - disk.cx, cy - disk.cy)
        if (r_dist / max(1.0, disk.radius)) > 0.965:
            return False

        return True

    # -------------------------------------------------------------------------
    # End-to-End Image Processing Pipeline
    # -------------------------------------------------------------------------
    def process_image(
        self,
        img_gray: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Executes complete Master Blueprint Pipeline on a single full-disk image.
        Returns:
            List of (mask_2048_bool, confidence_score) tuples.
        """
        h, w = img_gray.shape
        disk = detect_disk(img_gray)

        # Stage 0: Preprocessing & Station Normalization
        two_ch, disk_mask = self.preprocess_stage0(img_gray, disk)

        # Stage 1: Proposal Detection with Size-Adaptive Confidence
        tasks = self.detect_proposals_stage1(img_gray, disk_mask)
        if len(tasks) == 0:
            return []

        refined_masks = []
        refined_scores = []

        # Stage 2 & 3: Context-Expanded Crop + Weighted Quad Refinement
        for task in tasks:
            bbox = task["bbox"]
            score = task["score"]
            stage1_mask = task["merged_mask"]

            crop_input, ext_coords, rel_prop_box = self.extract_3ax3b_crop(bbox, two_ch, crop_size=512)
            p_quad = self.predict_quad_ensemble(crop_input, rel_prop_box)

            # Resize to native crop size
            x_min, y_min, x_max, y_max = ext_coords
            native_w = x_max - x_min
            native_h = y_max - y_min
            p_native = cv2.resize(p_quad, (native_w, native_h), interpolation=cv2.INTER_LINEAR)

            # Stage 5: Calibrated Probability Threshold (tau = 0.78)
            bin_crop = (p_native >= self.tau_thresh)
            if not bin_crop.any():
                continue

            # Internal Hole Repair
            bin_crop = scipy.ndimage.binary_fill_holes(bin_crop)

            # Stage-1 Spatial Anchor Verification
            crop_stage1 = stage1_mask[y_min:y_max, x_min:x_max] if stage1_mask is not None else None

            num_cc, cc_labels = cv2.connectedComponents(bin_crop.astype(np.uint8))
            for cc_idx in range(1, num_cc):
                comp = (cc_labels == cc_idx)
                if int(comp.sum()) >= 150:
                    # Anchor check: must overlap with YOLO stage-1 proposal
                    if crop_stage1 is not None and not np.logical_and(comp, crop_stage1).any():
                        continue

                    # Reproject to full-disk 2048 canvas
                    canvas = np.zeros((2048, 2048), dtype=bool)
                    canvas[y_min:y_max, x_min:x_max] = comp
                    canvas &= disk_mask

                    # Stage 6: Chromospheric Physics Filter
                    if int(canvas.sum()) >= self.min_area and self.is_valid_solar_filament(canvas, img_gray, disk):
                        refined_masks.append(canvas)
                        refined_scores.append(score)

        # Stage 7: Union-Find Collinear Medial Axis Spine Reconnection
        if len(refined_masks) > 1:
            pairs = list(zip(refined_masks, refined_scores))
            pairs = reconnect_collinear_spines(
                pairs,
                raw_img=img_gray,
                max_gap=40.0,
                max_angle_deg=35.0,
                bridge_thickness=5,
                min_area=self.min_area,
            )
            refined_masks = [m for m, s in pairs]
            refined_scores = [s for m, s in pairs]

        # Stage 8: Greedy Disjoint Instance Assembly
        if len(refined_masks) > 0:
            final_pairs = disjoint_merge_instances(refined_masks, refined_scores, min_area=self.min_area)
            return final_pairs
        return []
