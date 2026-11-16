import sys, os, pandas as pd, numpy as np
from pycocotools import mask as mu

def fast_audit(csv_path):
    print("=" * 70, flush=True)
    print(f"FAST AUDIT: {csv_path}", flush=True)
    print("=" * 70, flush=True)
    if not os.path.exists(csv_path):
        print("ERROR: File does not exist!", flush=True)
        return False

    df = pd.read_csv(csv_path)
    print(f"Row count: {len(df)}", flush=True)
    print(f"Columns: {list(df.columns)}", flush=True)

    expected_cols = ["filament_id", "segmentation_rle"]
    if list(df.columns) != expected_cols:
        print(f"FAILED: Expected columns {expected_cols}, got {list(df.columns)}", flush=True)
        return False

    null_counts = df.isnull().sum().to_dict()
    print(f"Null counts: {null_counts}", flush=True)
    if any(v > 0 for v in null_counts.values()):
        print("FAILED: Found null values!", flush=True)
        return False

    stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    num_unique_stems = stems.nunique()
    print(f"Unique test images: {num_unique_stems} / 178", flush=True)

    total_px = 0
    total_merged_px = 0
    collision_images = 0
    empty_masks = 0
    min_area = 10**9
    max_area = 0

    for stem, grp in df.groupby(stems, sort=False):
        rles = []
        img_sum_area = 0
        for _, row in grp.iterrows():
            rle_str = str(row["segmentation_rle"]).strip()
            if not rle_str or rle_str.lower() == 'nan':
                empty_masks += 1
                continue
            rle_obj = {'counts': rle_str.encode('utf-8'), 'size': [2048, 2048]}
            a = int(mu.area([rle_obj])[0])
            if a == 0:
                empty_masks += 1
                continue
            rles.append(rle_obj)
            img_sum_area += a
            min_area = min(min_area, a)
            max_area = max(max_area, a)

        total_px += img_sum_area
        if len(rles) > 1:
            merged = mu.merge(rles)
            merged_a = int(mu.area([merged])[0])
            total_merged_px += merged_a
            if merged_a < img_sum_area:
                collision_images += 1
                print(f"WARNING: Collision detected in {stem}! Sum={img_sum_area}, Merged={merged_a}, Diff={img_sum_area - merged_a}", flush=True)
        elif len(rles) == 1:
            total_merged_px += img_sum_area

    print(f"Empty RLE masks:        {empty_masks}", flush=True)
    print(f"Collision images:       {collision_images}", flush=True)
    print(f"Total segmentation px:  {total_px:,}", flush=True)
    print(f"Total merged px:        {total_merged_px:,}", flush=True)
    print(f"Overlap discrepancy:    {total_px - total_merged_px:,} px", flush=True)
    print(f"Min instance area:      {min_area:,} px", flush=True)
    print(f"Max instance area:      {max_area:,} px", flush=True)
    print(f"Mean instance area:     {total_px / max(1, len(df)):.1f} px", flush=True)
    print(f"File size:              {os.path.getsize(csv_path):,} bytes", flush=True)

    passed = (empty_masks == 0) and (collision_images == 0) and (num_unique_stems == 178) and (len(df) == 1199)
    print("=" * 70, flush=True)
    print(f"VERDICT: {'PASSED PERFECTLY (100% VALID)' if passed else 'FAILED CHECKS'}", flush=True)
    print("=" * 70, flush=True)
    return passed

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'd:/Kaggle/submissions/submission_champion_cldice_guardrailed_043plus.csv'
    fast_audit(p)
