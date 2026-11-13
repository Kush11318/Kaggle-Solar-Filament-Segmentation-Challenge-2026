"""
Official Reproduction Script: MAGFiLO 1.0 Champion Test Inference
Runs full-disk inference across all 180 competition test observations.
Generates compliant submission.csv (1,221 instances, verified 0.39 Public Leaderboard score).
"""

import os
import sys
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
from src.paths import get_dataset_paths
from src.disk import detect_disk, make_two_channel_input
from src.crop_pipeline import TwoStageCropPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MAGFiLO 1.0 Champion Submission")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory containing test images")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to write output submission CSV")
    parser.add_argument("--yolo_path", type=str, default="checkpoints/yolo11m_staple_best.pt", help="Path to YOLO11m weights")
    parser.add_argument("--b4_path", type=str, default="checkpoints/crop_unet_effnet_b4_512_mined_best.pt", help="Path to EffNet-B4 weights")
    parser.add_argument("--r34_path", type=str, default="checkpoints/crop_unet_resnet34_512_mined_best.pt", help="Path to ResNet-34 weights")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--det_conf", type=float, default=0.24, help="Stage 1 detection confidence threshold")
    parser.add_argument("--refine_thresh", type=float, default=0.44, help="Stage 2 probability refinement threshold")
    parser.add_argument("--min_area", type=int, default=180, help="Minimum pixel area for valid filament instances")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("=" * 80)
    print("MAGFiLO 1.0 - OFFICIAL COMPETITION TEST INFERENCE")
    print(f"Device: {device} | Output CSV: {args.output_csv}")
    print("=" * 80)

    # 1. Resolve test images directory
    if args.test_dir:
        test_dir = args.test_dir
    else:
        _, _, default_test_dir = get_dataset_paths()
        test_dir = default_test_dir

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"Discovered {len(test_files)} test images in {test_dir}.")

    # 2. Load Models
    print("Loading Neural Ensembles...")
    yolo_model = YOLO(args.yolo_path)

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

    # 4. Generate Predictions
    out_rows = []
    t0 = time.time()
    print("Executing End-to-End Inference Loop...")
    for idx, fn in enumerate(test_files, 1):
        stem = fn.rsplit(".", 1)[0]
        fpath = os.path.join(test_dir, fn)
        img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape
        disk = detect_disk(img_gray)
        disk_mask = disk.mask((h, w))
        two_ch = make_two_channel_input(img_gray, disk)

        preds = pipeline.process_image(img_gray, disk_mask, two_ch, disk=disk)
        for p_idx, (p_mask, p_score) in enumerate(preds, 1):
            rle_obj = mask_util.encode(np.asfortranarray(p_mask.astype(np.uint8)))
            out_rows.append({
                "filament_id": f"{stem}_{p_idx}",
                "segmentation_rle": rle_obj["counts"].decode("utf-8")
            })

        if idx % 20 == 0 or idx == len(test_files):
            elapsed = time.time() - t0
            print(f"  Processed {idx}/{len(test_files)} images ({elapsed:.1f}s | {idx/elapsed:.2f} img/s) -> Instances: {len(out_rows)}", flush=True)

    sub_df = pd.DataFrame(out_rows)
    sub_df.to_csv(args.output_csv, index=False)
    print("\n" + "=" * 80)
    print("TEST INFERENCE FINISHED SUCCESSFULLY!")
    print(f"  Target File:     {args.output_csv}")
    print(f"  Total Instances: {len(sub_df)}  (Target: 1,221 instances)")
    print(f"  Unique Images:   {sub_df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} / {len(test_files)}")
    print(f"  File Size:       {os.path.getsize(args.output_csv):,} bytes")
    print("=" * 80)


if __name__ == "__main__":
    main()
