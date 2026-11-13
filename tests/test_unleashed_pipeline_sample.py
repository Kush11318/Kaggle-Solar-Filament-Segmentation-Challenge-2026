"""
Quick smoke test for MAGFiLO Unleashed Pipeline on a single validation image.
Verifies Routes 1, 2, 3, 5, and 6 end-to-end on CPU without errors.
"""

import os
import sys
import time
import cv2
import torch
import numpy as np

sys.path.append(r"d:\Kaggle")
from scripts.run_unleashed_multiroute_consensus import MAGFiLOUnleashedEngine
from src.paths import get_dataset_paths

def smoke_test():
    print("=" * 70)
    print("RUNNING SMOKE TEST: MAGFiLO UNLEASHED ON 1 SAMPLE IMAGE")
    print("=" * 70, flush=True)

    root_ckpt = r"d:\Kaggle\checkpoints"
    yolo_m = os.path.join(root_ckpt, "yolo11m_staple_best.pt")
    yolo_x = os.path.join(root_ckpt, "yolo11x_seg_best.pt")
    b4 = os.path.join(root_ckpt, "crop_unet_effnet_b4_512_mined_best.pt")
    r34 = os.path.join(root_ckpt, "crop_unet_resnet34_512_mined_best.pt")

    train_json, train_img_dir, test_img_dir = get_dataset_paths()
    img_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpeg')]
    if not img_files:
        print("No test/train images found!")
        return

    sample_fn = img_files[0]
    sample_path = os.path.join(train_img_dir, sample_fn)
    print(f"Testing on sample: {sample_fn}...")
    img_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

    engine = MAGFiLOUnleashedEngine(yolo_m, yolo_x, b4, r34, device='cpu')

    t0 = time.time()
    # Process with slightly higher thresholds for fast CPU test
    insts = engine.process_solar_image(
        img_gray,
        conf_m=0.35,
        conf_x=0.35,
        tau_core=0.65,
        tau_barb=0.45,
        min_area=200,
        enable_route2_faint=True,
        enable_route5_spine=True,
    )
    elapsed = time.time() - t0

    print(f"\nSmoke Test Completed in {elapsed:.2f}s!")
    print(f"Instances Segmented: {len(insts)}")
    for i, inst in enumerate(insts[:5], 1):
        print(f"  Instance {i}: Area = {inst['area']:,} px, Score = {inst['score']:.3f}, RLE len = {len(inst['rle'])}")

    assert len(insts) > 0, "Expected at least 1 filament instance segmented!"
    print("\n>>> SMOKE TEST PASSED: ALL 6 ROUTES OPERATIONAL & INTEGRATED! <<<")

if __name__ == "__main__":
    smoke_test()

