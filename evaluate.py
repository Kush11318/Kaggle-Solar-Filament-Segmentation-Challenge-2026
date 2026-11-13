"""
Official Reproduction Script: MAGFiLO 1.0 Champion Validation Evaluation
Evaluates the TwoStageCropPipeline on the official 80/20 validation split (137 images, 227 annotator records).
Reproduces the verified Panoptic Quality (PQ = 0.4618, SQ = 0.6838).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from pycocotools import mask as mask_util
from ultralytics import YOLO
import segmentation_models_pytorch as smp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.paths import get_dataset_paths, get_split_path
from src.parser import MAGFiLOParser
from src.disk import detect_disk, make_two_channel_input
from src.crop_pipeline import TwoStageCropPipeline
from src.metrics import get_overlap_df, compute_official_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAGFiLO 1.0 Champion Pipeline")
    parser.add_argument("--yolo_path", type=str, default="checkpoints/yolo11m_staple_best.pt", help="Path to YOLO11m weights")
    parser.add_argument("--b4_path", type=str, default="checkpoints/crop_unet_effnet_b4_512_mined_best.pt", help="Path to EffNet-B4 weights")
    parser.add_argument("--r34_path", type=str, default="checkpoints/crop_unet_resnet34_512_mined_best.pt", help="Path to ResNet-34 weights")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--det_conf", type=float, default=0.24, help="Stage 1 detection confidence threshold")
    parser.add_argument("--refine_thresh", type=float, default=0.44, help="Stage 2 probability refinement threshold")
    parser.add_argument("--min_area", type=int, default=180, help="Minimum pixel area for valid filament instances")
    parser.add_argument("--max_images", type=int, default=None, help="Optional subset of validation images for quick evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("=" * 80)
    print("MAGFiLO 1.0 - OFFICIAL VALIDATION BENCHMARK EVALUATION")
    print(f"Device: {device} | det_conf: {args.det_conf} | refine_thresh: {args.refine_thresh} | min_area: {args.min_area}")
    print("=" * 80)

    # 1. Dataset & Split Paths
    train_json, train_img_dir, _ = get_dataset_paths()
    split_file = get_split_path()
    with open(split_file, "r") as f:
        split_data = json.load(f)

    val_filenames = sorted(split_data.get("val_filenames", []))
    val_record_ids = sorted(split_data.get("val_record_ids", []))
    if args.max_images:
        val_filenames = val_filenames[:args.max_images]

    parser = MAGFiLOParser(train_json, train_img_dir)
    gt_df = parser.build_gt_dataframe(val_record_ids)
    print(f"Ground Truth Loaded: {len(val_filenames)} images, {len(val_record_ids)} annotator records.")

    # 2. Load Models
    print("Loading Neural Ensembles...")
    yolo_model = YOLO(args.yolo_path)

    # Stage 2 Models
    m_b4 = smp.Unet("efficientnet-b4", in_channels=2, classes=1, activation=None).to(device)
    if os.path.exists(args.b4_path):
        c_b4 = torch.load(args.b4_path, map_location=device, weights_only=False)
        m_b4.load_state_dict(c_b4.get("model_state_dict", c_b4))
    m_b4.eval()

    m_r34 = smp.Unet("resnet34", in_channels=2, classes=1, activation=None).to(device)
    if os.path.exists(args.r34_path):
        c_r34 = torch.load(args.r34_path, map_location=device, weights_only=False)
        m_r34.load_state_dict(c_r34.get("model_state_dict", c_r34))
    m_r34.eval()

    # 3. Assemble TwoStageCropPipeline
    pipeline = TwoStageCropPipeline(
        crop_unet_model=[m_b4, m_r34],
        yolo_model=yolo_model,
        device=device,
        det_conf=args.det_conf,
        refine_thresh=args.refine_thresh,
        min_area=args.min_area,
        enable_flip4_tta=True,
        enable_physics_filters=True,
        sunspot_ecc_thresh=0.65,
        max_limb_r_ratio=0.965,
        enable_spine_reconnection=True,
        max_spine_gap=50.0,
        max_spine_angle=30.0,
        use_full_disk=True,
        yolo_imgsz=1280,
        context_factor=1.8,
        crop_size=512,
    )

    # 4. Run Evaluation Loop
    val_pred_rows = []
    t0 = time.time()
    print("Starting Validation Inference...")
    for idx, fn in enumerate(val_filenames, 1):
        stem = fn.rsplit(".", 1)[0]
        img_path = os.path.join(train_img_dir, fn)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape
        disk = detect_disk(img_gray)
        disk_mask = disk.mask((h, w))
        two_ch = make_two_channel_input(img_gray, disk)

        preds = pipeline.process_image(img_gray, disk_mask, two_ch, disk=disk)
        for p_idx, (p_mask, p_score) in enumerate(preds, 1):
            rle_obj = mask_util.encode(np.asfortranarray(p_mask.astype(np.uint8)))
            val_pred_rows.append({
                "filament_id": f"{stem}_{p_idx}",
                "segmentation_rle": rle_obj["counts"].decode("utf-8")
            })

        if idx % 10 == 0 or idx == len(val_filenames):
            elapsed = time.time() - t0
            print(f"  Processed {idx}/{len(val_filenames)} images ({elapsed:.1f}s | {idx/elapsed:.2f} img/s)...", flush=True)

    pred_df = pd.DataFrame(val_pred_rows)
    overlap_df = get_overlap_df(gt_df, pred_df)
    m = compute_official_metrics(overlap_df)

    print("\n" + "=" * 80)
    print("OFFICIAL VALIDATION BENCHMARK RESULTS:")
    print(f"  Panoptic Quality (PQ):     {m['pq']:.4f}  (Benchmark: 0.4618)")
    print(f"  Segmentation Quality (SQ): {m['sq']:.4f}  (Benchmark: 0.6838)")
    print(f"  Recognition Quality (RQ):  {m['rq']:.4f}")
    print(f"  True Positives (TP):       {m['tp']}")
    print(f"  False Positives (FP):      {m['fp']}")
    print(f"  False Negatives (FN):      {m['fn']}")
    print(f"  Total Predictions:         {len(pred_df)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
