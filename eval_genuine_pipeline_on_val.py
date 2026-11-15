"""
Evaluates the Genuine Solar Pipeline on the Official Validation Split.
Validates Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
across all 137 validation images (227 annotator records, 1,539 ground-truth masks).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import cv2
from pycocotools import mask as mask_util

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_genuine_pipeline import GenuineSolarPipeline
from src.paths import get_dataset_paths, get_split_path
from src.metrics import get_overlap_df, compute_official_metrics

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print(f"EVALUATING GENUINE SOLAR PIPELINE ON OFFICIAL VALIDATION SET ({device})")
    print("=" * 85)

    # 1. Load Validation Split & Ground Truth
    split_path = get_split_path()
    with open(split_path, "r") as f:
        split_data = json.load(f)

    val_filenames = sorted(split_data.get("val_filenames", []))
    print(f"Validation images: {len(val_filenames)}")

    gt_parquet_path = "splits/val_gt_df.parquet"
    if os.path.exists(gt_parquet_path):
        gt_df = pd.read_parquet(gt_parquet_path)
    else:
        from src.parser import MAGFiLOParser
        train_json, train_img_dir, _ = get_dataset_paths()
        parser = MAGFiLOParser(train_json, train_img_dir)
        gt_df = parser.build_gt_dataframe(split_data["val_record_ids"])
        gt_df.to_parquet(gt_parquet_path, index=False)

    print(f"Ground-truth instances: {len(gt_df)}")

    # 2. Initialize Genuine Pipeline
    pipeline = GenuineSolarPipeline(
        yolo_path="checkpoints/yolo11m_staple_best.pt",
        b4_path="checkpoints/crop_unet_effnet_b4_512_mined_best.pt",
        r34_path="checkpoints/crop_unet_resnet34_512_mined_best.pt",
        device=device,
        det_conf=0.22,
        refine_thresh=0.55,
        min_area=150,
        enable_shift=True,
        shift_dy=-0.23,
    )

    # 3. Run Inference on Validation Images
    train_json, train_img_dir, _ = get_dataset_paths()
    pred_rows = []
    t0 = time.time()

    print(f"Running inference on {len(val_filenames)} validation images...")
    for idx, fn in enumerate(val_filenames, 1):
        stem = fn.rsplit(".", 1)[0]
        fpath = os.path.join(train_img_dir, fn)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        instances = pipeline.process_image(img)
        for p_idx, (mask, score) in enumerate(instances, 1):
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")
            pred_rows.append({
                "filament_id": f"{stem}_{p_idx}",
                "segmentation_rle": rle,
            })

        if idx % 20 == 0 or idx == len(val_filenames):
            el = time.time() - t0
            print(f"  [{idx}/{len(val_filenames)}] {el:.1f}s ({idx/el:.2f} img/s) -> Pred instances: {len(pred_rows)}", flush=True)

    pred_df = pd.DataFrame(pred_rows)
    print(f"\nGenerated {len(pred_df)} predicted instances across {pred_df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} images.")

    # 4. Compute Official Metrics
    print("Computing Official Competition Evaluation Metrics...")
    overlap_df = get_overlap_df(gt_df, pred_df)
    metrics = compute_official_metrics(overlap_df)

    print("\n" + "=" * 85)
    print("OFFICIAL VALIDATION PERFORMANCE REPORT:")
    print("=" * 85)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")
    print("=" * 85)

if __name__ == "__main__":
    main()

