"""
MAGFiLO 2.0: High-Recall Native Quadrant + Core-Mantle + Vision Transformer Pipeline
===================================================================================
Developed for the IEEE BigData / Kaggle Solar Filament Segmentation Challenge 2026.

Architectural Upgrades:
1. 4-Quadrant Slicing at 1:1 Native Scale (1120x1120):
   - Overlapping quadrant tiling (192 px overlap) at native sensor resolution.
   - Captures ultra-thin spines (<8 px) without downsampling blur (FN reduction).
   - Combines with global context pass to ensure large filaments (>500 px) are preserved.
2. Core-and-Mantle Geodesic Reconstruction (tau_core >= 0.68, tau_support = 0.35):
   - Strict core absorption gate (>= 25 px with P >= 0.68) vetoes quiet-sun granulation FP.
   - Geodesic morphological propagation from validated core into faint support boundaries.
3. Hybrid CNN + Vision Transformer Ensemble:
   - Pairs multi-scale CNN refiners (EffNet-B4 + ResNet-34) with hierarchical Vision Transformer
     (CSWin Stripe Attention Multi-Task U-Net with LePE positional bias).
   - Blends in log-odds space to preserve continuous medial spines and sharp edge gradients.
4. Physical Absorption Fringe Snapping (I_raw <= 138) & Disjoint Instance Claiming.
   - Strictly 0 pixel collisions guaranteed.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import scipy.ndimage as ndi
from skimage.filters import frangi

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.disk import Disk, detect_disk, make_two_channel_input
from src.crop_utils import (
    compute_dynamic_square_crop,
    positive_overlap_grouping,
    disjoint_merge_instances,
    is_sunspot_or_compact_blob,
    is_outside_solar_disk_margin,
    is_insufficient_contrast,
)
from src.spine_reconnection import reconnect_collinear_spines
from src.cswin_unet import CSWinDenseMultiTaskUNet


class QuadrantCoreMantleTransformerPipeline:
    def __init__(
        self,
        yolo_path: str = "checkpoints/yolo11m_staple_best.pt",
        b4_path: str = "checkpoints/crop_unet_effnet_b4_512_mined_best.pt",
        r34_path: str = "checkpoints/crop_unet_resnet34_512_mined_best.pt",
        cswin_path: str = "checkpoints/cswin_dense_watershed_best.pt",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        det_conf_quad: float = 0.18,
        det_conf_global: float = 0.20,
        tau_core: float = 0.68,
        tau_support: float = 0.35,
        min_core_pixels: int = 25,
        min_inst_prob: float = 0.45,
        min_area: int = 150,
        enable_barb_dilation: bool = True,
        enable_shift: bool = True,
        shift_dy: float = -0.23,
        enable_absorption_snapping: bool = True,
        absorption_threshold: int = 138,
    ):
        self.device = torch.device(device)
        self.det_conf_quad = det_conf_quad
        self.det_conf_global = det_conf_global
        self.tau_core = tau_core
        self.tau_support = tau_support
        self.min_core_pixels = min_core_pixels
        self.min_inst_prob = min_inst_prob
        self.min_area = min_area
        self.enable_barb_dilation = enable_barb_dilation
        self.enable_shift = enable_shift
        self.shift_dy = shift_dy
        self.enable_absorption_snapping = enable_absorption_snapping
        self.absorption_threshold = absorption_threshold

        print("=" * 80)
        print("MAGFiLO 2.0: HIGH-RECALL QUADRANT + CORE-MANTLE + TRANSFORMER PIPELINE")
        print(f"Device: {self.device} | tau_core={self.tau_core} | min_core_px={self.min_core_pixels}")
        print("=" * 80)

        # 1. Load YOLO Detector
        yolo_p = REPO_ROOT / yolo_path
        print(f"[Init] Loading YOLO11m Detector from {yolo_p}...")
        self.yolo = YOLO(str(yolo_p))

        # 2. Load EffNet-B4 U-Net
        b4_p = REPO_ROOT / b4_path
        print(f"[Init] Loading EfficientNet-B4 Refiner from {b4_p}...")
        self.m_b4 = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=2, classes=1, activation=None).to(self.device)
        ck_b4 = torch.load(str(b4_p), map_location=self.device, weights_only=False)
        self.m_b4.load_state_dict(ck_b4.get("model_state_dict", ck_b4))
        self.m_b4.eval()

        # 3. Load ResNet-34 U-Net
        r34_p = REPO_ROOT / r34_path
        print(f"[Init] Loading ResNet-34 Refiner from {r34_p}...")
        self.m_r34 = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=1, activation=None).to(self.device)
        ck_r34 = torch.load(str(r34_p), map_location=self.device, weights_only=False)
        self.m_r34.load_state_dict(ck_r34.get("model_state_dict", ck_r34))
        self.m_r34.eval()

        # 4. Load CSWin Transformer U-Net
        cswin_p = REPO_ROOT / cswin_path
        print(f"[Init] Loading CSWin Stripe-Attention Vision Transformer from {cswin_p}...")
        self.m_cswin = CSWinDenseMultiTaskUNet(encoder_name="efficientnet-b4", in_channels=3).to(self.device)
        ck_cs = torch.load(str(cswin_p), map_location=self.device, weights_only=False)
        self.m_cswin.load_state_dict(ck_cs.get("model_state_dict", ck_cs))
        self.m_cswin.eval()

        print("[Init] All 4 deep learning models successfully initialized.")

        # Fixed Quadrant Windows for 2048x2048 native scale
        self.quad_coords = [
            (0, 0, 1120, 1120),       # Top-Left
            (0, 928, 1120, 2048),     # Top-Right
            (928, 0, 2048, 1120),     # Bottom-Left
            (928, 928, 2048, 2048)    # Bottom-Right
        ]
        self.k_spine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process_image(self, img_path: str | Path) -> List[Tuple[np.ndarray, float]]:
        """
        Processes a single full-disk 2048x2048 image with:
        1. 4-Quadrant Slicing at 1:1 Native Scale + Global Context Pass
        2. Positive Overlap Grouping (IoU=0.35)
        3. Dynamic Square Crop Refinement with 3-Model Hybrid Ensemble
        4. Core-and-Mantle Geodesic Reconstruction (tau_core >= 0.68)
        5. Barb Expansion, Solar Physics Guardrails, Bernasconi Spine Reconnection
        6. Sub-pixel shift (-0.23 px), Physical Snapping, and Disjoint Claiming
        """
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            return []
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        h, w = raw_gray.shape[:2]

        # Solar Disk Geometry
        disk = detect_disk(raw_gray)
        disk_mask = disk.mask((h, w))
        two_ch = make_two_channel_input(raw_gray, disk)

        # Stage 1: 4-Quadrant Slicing at 1:1 Native Scale (1120x1120)
        quad_imgs = [raw_bgr[y1:y2, x1:x2] for y1, x1, y2, x2 in self.quad_coords]
        res_batch = self.yolo.predict(
            quad_imgs,
            imgsz=1120,
            conf=self.det_conf_quad,
            batch=4,
            verbose=False,
            retina_masks=True
        )

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []

        for res, (y1, x1, y2, x2) in zip(res_batch, self.quad_coords):
            if res.masks is None or len(res.masks) == 0:
                continue
            masks_data = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for m, c in zip(masks_data, confs):
                global_canvas = np.zeros((2048, 2048), dtype=bool)
                gh, gw = y2 - y1, x2 - x1
                m_b = (m > 0.5)
                if m_b.shape[0] != gh or m_b.shape[1] != gw:
                    m_b = cv2.resize(m_b.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
                global_canvas[y1:y2, x1:x2] = m_b
                global_canvas &= (disk_mask > 0)
                if global_canvas.sum() >= 40:
                    all_masks.append(global_canvas)
                    all_scores.append(float(c))

        # Global Full-Disk Pass (preserves massive >500 px flux ropes)
        res_global = self.yolo.predict(
            raw_bgr,
            imgsz=1280,
            conf=self.det_conf_global,
            verbose=False,
            retina_masks=True
        )[0]
        if res_global.masks is not None and len(res_global.masks) > 0:
            for m, c in zip(res_global.masks.data.cpu().numpy(), res_global.boxes.conf.cpu().numpy()):
                m_2048 = cv2.resize((m > 0.5).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST) > 0
                m_2048 &= (disk_mask > 0)
                if m_2048.sum() >= 40:
                    all_masks.append(m_2048)
                    all_scores.append(float(c))

        if len(all_masks) == 0:
            return []

        # Positive Overlap Grouping (IoU=0.35)
        tasks = positive_overlap_grouping(all_masks, all_scores, iou_thresh=0.35)

        candidate_masks: List[np.ndarray] = []
        candidate_scores: List[float] = []

        # Stage 2: Dynamic Crop Refinement & Core-Mantle Reconstruction
        for task in tasks:
            bbox = task["bbox"]
            score = task["score"]
            x0, y0, x1, y1, _ = compute_dynamic_square_crop(
                bbox=bbox,
                context=1.5,
                min_side=96,
                max_side=2048,
                img_h=2048,
                img_w=2048
            )
            crop_h, crop_w = y1 - y0, x1 - x0
            if crop_h < 16 or crop_w < 16:
                continue

            # Normalized 2-channel input
            ch0 = cv2.resize(two_ch[0, y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)
            ch1 = cv2.resize(two_ch[1, y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)

            # High-speed single-scale Frangi Hessian ridge response
            fr_ch = frangi(ch0, sigmas=(2.0,), black_ridges=True).astype(np.float32)
            if fr_ch.max() > 0:
                fr_ch /= fr_ch.max()

            t2 = torch.from_numpy(np.stack([ch0, ch1], axis=0)).unsqueeze(0).to(self.device)
            t3 = torch.from_numpy(np.stack([ch0, ch1, fr_ch], axis=0)).unsqueeze(0).to(self.device)

            # Hybrid CNN + Vision Transformer Ensemble
            with torch.no_grad():
                l_b4 = self.m_b4(t2)
                l_r34 = self.m_r34(t2)
                l_cs, _ = self.m_cswin(t3)
                l_ens = 0.45 * l_b4 + 0.35 * l_r34 + 0.20 * l_cs
                p_ens = torch.sigmoid(l_ens).squeeze().cpu().numpy()

            native_prob = cv2.resize(p_ens, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            # Core-and-Mantle Geodesic Reconstruction
            core_mask = (native_prob >= self.tau_core)
            support_mask = (native_prob >= self.tau_support)

            # Strict Core Gate: Discard candidates without a robust high-absorption core
            if core_mask.sum() < self.min_core_pixels:
                continue

            # Geodesic morphological propagation: Flood from core into support
            recon_crop = ndi.binary_propagation(core_mask, mask=support_mask)
            recon_closed = cv2.morphologyEx(recon_crop.astype(np.uint8), cv2.MORPH_CLOSE, self.k_spine) > 0
            recon_crop = recon_closed & support_mask

            # Barb dilation: Restore true tapering barbs
            if self.enable_barb_dilation:
                recon_crop = cv2.dilate(recon_crop.astype(np.uint8), self.k_dil) > 0

            num_cc, cc_labels = cv2.connectedComponents(recon_crop.astype(np.uint8))
            for cc_idx in range(1, num_cc):
                comp = (cc_labels == cc_idx)
                # Ensure component has sufficient core support
                if (comp & core_mask).sum() < self.min_core_pixels or comp.sum() < self.min_area:
                    continue
                inst_mean = float(native_prob[comp].mean())
                if inst_mean < self.min_inst_prob:
                    continue

                canvas = np.zeros((2048, 2048), dtype=bool)
                canvas[y0:y1, x0:x1] = comp
                canvas &= (disk_mask > 0)
                area = int(canvas.sum())
                if area < self.min_area:
                    continue

                # Solar Physics Guardrails
                if is_sunspot_or_compact_blob(canvas, area=area, min_eccentricity=0.65):
                    continue
                if is_outside_solar_disk_margin(canvas, disk.cx, disk.cy, disk.radius, max_r_ratio=0.975):
                    continue
                if is_insufficient_contrast(canvas, raw_gray, max_contrast_ratio=0.92):
                    continue

                candidate_masks.append(canvas)
                candidate_scores.append(score * inst_mean)

        if len(candidate_masks) == 0:
            return []

        # Stage 3: Bernasconi PIL Collinear Spine Reconnection
        if len(candidate_masks) > 1:
            pairs = reconnect_collinear_spines(
                list(zip(candidate_masks, candidate_scores)),
                raw_img=raw_gray,
                max_gap=35.0,
                max_angle_deg=30.0,
                bridge_thickness=5,
                min_area=self.min_area,
            )
            candidate_masks = [m for m, s in pairs]
            candidate_scores = [s for m, s in pairs]

        # Stage 4: Physical Optical Absorption Fringe Snapping (I_raw <= 138)
        if self.enable_absorption_snapping:
            snapped_masks: List[np.ndarray] = []
            snapped_scores: List[float] = []
            for m, s in zip(candidate_masks, candidate_scores):
                snapped = m & (raw_gray <= self.absorption_threshold)
                if snapped.sum() >= 0.85 * m.sum() and snapped.sum() >= self.min_area:
                    snapped_masks.append(snapped)
                else:
                    snapped_masks.append(m)
                snapped_scores.append(s)
            candidate_masks = snapped_masks
            candidate_scores = snapped_scores

        # Stage 5: Annotator Spatial Calibration (-0.23 px shift)
        if self.enable_shift and abs(self.shift_dy) > 1e-4:
            shifted_masks: List[np.ndarray] = []
            shifted_scores: List[float] = []
            for m, s in zip(candidate_masks, candidate_scores):
                sh = ndi.shift(m.astype(np.float32), (self.shift_dy, 0.0), order=1, mode="nearest")
                final_m = (sh >= 0.5) & (disk_mask > 0)
                if final_m.sum() >= self.min_area:
                    shifted_masks.append(final_m)
                    shifted_scores.append(s)
            candidate_masks = shifted_masks
            candidate_scores = shifted_scores

        # Stage 6: Greedy Disjoint Instance Claiming (Strictly 0 Pixel Collisions)
        final_instances = disjoint_merge_instances(
            candidate_masks,
            candidate_scores,
            min_area=self.min_area,
        )

        return final_instances

    def run_directory(self, image_dir: str | Path, output_csv: str | Path) -> None:
        """
        Runs batch inference across all images in image_dir and writes/updates output_csv.
        Features automatic resume: skips images already present in output_csv.
        """
        img_dir = Path(image_dir)
        out_p = Path(output_csv)
        image_files = sorted(
            [f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]
        )

        print(f"\n[Run] Found {len(image_files)} test images in {img_dir}")
        print(f"[Run] Output will be saved to: {out_p}")

        processed_stems = set()
        existing_rows = []
        if out_p.exists():
            try:
                existing_df = pd.read_csv(out_p)
                if "filament_id" in existing_df.columns:
                    for fid in existing_df["filament_id"]:
                        stem = str(fid).split("_")[0]
                        processed_stems.add(stem)
                    existing_rows = existing_df.to_dict("records")
                    print(f"[Run] Resuming: {len(processed_stems)} images already completed ({len(existing_rows)} instances)")
            except Exception as e:
                print(f"[Run] Notice: Could not read existing CSV ({e}), starting fresh.")

        rows = existing_rows
        t_start = time.time()

        for idx, img_path in enumerate(image_files, 1):
            stem = img_path.stem
            if stem in processed_stems:
                continue

            t0 = time.time()
            try:
                instances = self.process_image(img_path)
            except Exception as e:
                print(f"  [ERROR] Failed processing {stem}: {e}")
                instances = []

            dt = time.time() - t0
            img_rows = []
            for inst_idx, (m, score) in enumerate(instances, 1):
                rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("utf-8")
                img_rows.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": rle
                })

            rows.extend(img_rows)
            processed_stems.add(stem)

            # Persist checkpoint after every image
            df_cur = pd.DataFrame(rows)
            df_cur.to_csv(out_p, index=False)

            elapsed = time.time() - t_start
            rate = len(processed_stems) / max(elapsed, 1.0)
            remaining = (len(image_files) - len(processed_stems)) / max(rate, 1e-4)

            print(
                f"[{len(processed_stems)}/{len(image_files)}] {stem}: "
                f"{len(instances)} filaments ({dt:.1f}s) | "
                f"Total inst: {len(rows)} | ETA: {remaining/60:.1f}m",
                flush=True
            )

        print("\n" + "=" * 80)
        print(f"[Complete] Finished processing {len(image_files)} images in {(time.time()-t_start)/60:.1f} minutes.")
        print(f"[Complete] Total valid disjoint filaments: {len(rows)}")
        print(f"[Complete] Submission saved to: {out_p}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run High-Recall Quadrant + Core-Mantle + Transformer Pipeline")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        help="Directory containing full-disk solar test images",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="submission_quadrant_coremantle_champion.csv",
        help="Output CSV filepath",
    )
    parser.add_argument("--tau_core", type=float, default=0.68, help="Core probability threshold")
    parser.add_argument("--tau_support", type=float, default=0.35, help="Support probability threshold")
    parser.add_argument("--min_core_pixels", type=int, default=25, help="Minimum core pixels required")
    parser.add_argument("--det_conf_quad", type=float, default=0.18, help="YOLO quadrant confidence threshold")
    args = parser.parse_args()

    pipeline = QuadrantCoreMantleTransformerPipeline(
        tau_core=args.tau_core,
        tau_support=args.tau_support,
        min_core_pixels=args.min_core_pixels,
        det_conf_quad=args.det_conf_quad,
    )
    pipeline.run_directory(args.image_dir, args.output_csv)


if __name__ == "__main__":
    main()
