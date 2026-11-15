"""
MAGFiLO 2.0: Champion V4 Pure Precision-Contraction Builder
Anchors strictly to the 0.410 Champion (submission_champion_v1_precision_snapped.csv, 1,199 rows).
Applies high-precision absorption contour contraction (I > 134) with 1-pixel skeleton core protection.
Enforces >= 92% area retention guardrail and strictly preserves the exact 1,199-instance invariant.
Guarantees 0 pixel collisions.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from pycocotools import mask as mask_util

REPO_ROOT = Path("d:/Kaggle")
sys.path.append(str(REPO_ROOT))

def extract_skeleton_core(mask: np.ndarray) -> np.ndarray:
    """Extracts 1-pixel skeleton spine using bbox-cropped morphological thinning."""
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint8)
    
    y_pts, x_pts = np.where(mask)
    pad = 3
    ymin, ymax = max(0, int(y_pts.min()) - pad), min(mask.shape[0], int(y_pts.max()) + pad + 1)
    xmin, xmax = max(0, int(x_pts.min()) - pad), min(mask.shape[1], int(x_pts.max()) + pad + 1)
    
    sub = (mask[ymin:ymax, xmin:xmax] > 0).astype(np.uint8) * 255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel_sub = np.zeros(sub.shape, dtype=np.uint8)
    temp = sub.copy()
    
    while True:
        eroded = cv2.erode(temp, element)
        temp_open = cv2.dilate(eroded, element)
        temp_open = cv2.subtract(temp, temp_open)
        skel_sub = cv2.bitwise_or(skel_sub, temp_open)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
            
    full_skel = np.zeros(mask.shape, dtype=np.uint8)
    full_skel[ymin:ymax, xmin:xmax] = skel_sub
    return full_skel

def apply_precision_contraction(mask: np.ndarray, raw_gray: np.ndarray, skel: np.ndarray, threshold: int = 134, min_retention: float = 0.92):
    """
    Trims outer boundary contour pixels that exceed absorption threshold (bright granulation),
    while strictly protecting the 1-pixel skeleton spine and enforcing min_retention.
    """
    orig_area = int(mask.sum())
    if orig_area < 120 or raw_gray is None:
        return mask
        
    # Boundary fringe: pixels within 1-2 pixels of contour
    k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask.astype(np.uint8), k_erode) > 0
    fringe = mask & (~eroded)
    
    # Identify bright non-absorbing pixels in the fringe
    bright_fringe = fringe & (raw_gray > threshold)
    n_bright = int(bright_fringe.sum())
    if n_bright == 0:
        return mask
        
    # Contract mask by removing bright fringe
    contracted = mask & (~bright_fringe)
    
    # Always protect the skeleton core
    contracted |= (skel > 0)
    
    # Safety guardrail: enforce minimum area retention
    contracted_area = int(contracted.sum())
    if contracted_area >= min_retention * orig_area and contracted_area >= 120:
        return contracted
    return mask

def main():
    print("=" * 80)
    print("BUILDING MAGFiLO 2.0 CHAMPION V4: PRECISION-CONTRACTED BASELINE")
    print("Target: Elevate Segmentation Quality (SQ) past 0.410 into 0.42 - 0.44+")
    print("=" * 80)

    in_csv = REPO_ROOT / "submissions/submission_champion_v1_precision_snapped.csv"
    out_csv = REPO_ROOT / "submission_champion_v4_precision_contracted.csv"
    sub_dir_out = REPO_ROOT / "submissions/submission_champion_v4_precision_contracted.csv"
    test_img_dir = REPO_ROOT / "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"

    df = pd.read_csv(in_csv)
    df['stem'] = df['filament_id'].apply(lambda s: s.rsplit('_', 1)[0])
    stems = sorted(df['stem'].unique().tolist())
    print(f"Loaded {len(df)} champion instances across {len(stems)} unique test images.")

    final_rows = []
    total_contracted = 0
    total_pixels_trimmed = 0
    t0 = time.time()

    for idx, stem in enumerate(stems, 1):
        grp = df[df['stem'] == stem].reset_index(drop=True)
        img_path = None
        for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
            cand = test_img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = str(cand)
                break

        raw_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if img_path else None

        canvas = np.zeros((2048, 2048), dtype=bool)
        for _, row in grp.iterrows():
            rle_dict = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('utf-8')}
            m = mask_util.decode(rle_dict) > 0
            orig_area = int(m.sum())

            if raw_gray is not None:
                skel = extract_skeleton_core(m)
                c_mask = apply_precision_contraction(m, raw_gray, skel, threshold=134, min_retention=0.92)
            else:
                c_mask = m

            # Ensure strict disjoint claiming against canvas
            c_mask = c_mask & (~canvas)
            if int(c_mask.sum()) < 100:
                c_mask = m & (~canvas)  # Fallback to avoid dropping instance

            contracted_area = int(c_mask.sum())
            if contracted_area < orig_area:
                total_contracted += 1
                total_pixels_trimmed += (orig_area - contracted_area)

            canvas |= c_mask

            fortran_m = np.asfortranarray(c_mask.astype(np.uint8))
            rle_enc = mask_util.encode(fortran_m)["counts"].decode("utf-8")
            final_rows.append({
                "filament_id": row["filament_id"],
                "segmentation_rle": rle_enc
            })

        if idx % 50 == 0 or idx == len(stems):
            print(f"Processed [{idx}/{len(stems)}] images ({time.time()-t0:.1f}s)...", flush=True)

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(out_csv, index=False)
    out_df.to_csv(sub_dir_out, index=False)

    print("\n" + "=" * 80)
    print("SUCCESSFULLY GENERATED CHAMPION V4 SUBMISSION")
    print("=" * 80)
    print(f"Total Instances: {len(out_df)} (Exact Champion Invariant Preserved: 1,199)")
    print(f"Unique Images: {out_df['filament_id'].apply(lambda s: s.rsplit('_', 1)[0]).nunique()} / 178")
    print(f"Contour-Contracted Instances: {total_contracted} / {len(out_df)} ({total_contracted/len(out_df)*100:.1f}%)")
    print(f"Total Granulation Fringe Pixels Trimmed: {total_pixels_trimmed:,} px (Avg {total_pixels_trimmed/max(1, total_contracted):.1f} px/mod)")
    print(f"File Saved to: {out_csv}")
    print(f"Mirror Saved to: {sub_dir_out}")

if __name__ == "__main__":
    main()
