import os, hashlib
import pandas as pd
import numpy as np
from pycocotools import mask as mu

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def audit_submission(csv_path, expected_count):
    print("=" * 80)
    print(f"AUDITING: {os.path.basename(csv_path)}")
    print("=" * 80)
    
    assert os.path.exists(csv_path), f"File missing: {csv_path}"
    df = pd.read_csv(csv_path)
    
    # 1. Structure
    assert list(df.columns) == ['filament_id', 'segmentation_rle'], f"Invalid columns: {df.columns}"
    assert len(df) == expected_count, f"Row count mismatch: {len(df)} vs {expected_count}"
    
    df['image_stem'] = df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])
    n_images = df['image_stem'].nunique()
    assert n_images == 178, f"Expected 178 images, found {n_images}"
    
    # 2. Collision & Validity Audit
    collisions = 0
    areas = []
    
    for stem, grp in df.groupby('image_stem'):
        canvas = np.zeros((2048, 2048), dtype=np.uint8)
        for idx, (_, row) in enumerate(grp.iterrows(), 1):
            rle = row['segmentation_rle']
            assert isinstance(rle, str) and len(rle) > 0, f"Empty RLE in {row['filament_id']}"
            m = mu.decode({"counts": rle.encode('utf-8'), "size": [2048, 2048]})
            area = int(m.sum())
            assert area > 0, f"Zero area mask in {row['filament_id']}"
            areas.append(area)
            
            # Collision check
            overlap = (canvas > 0) & (m > 0)
            if np.any(overlap):
                collisions += int(overlap.sum())
            canvas[m > 0] = idx
            
    assert collisions == 0, f"Found {collisions} overlapping pixels!"
    
    sha256 = compute_sha256(csv_path)
    print(f"  Rows:             {len(df)}")
    print(f"  Images:           {n_images}")
    print(f"  Pixel Collisions: {collisions} (100% Mutually Disjoint)")
    print(f"  Median Area:      {np.median(areas):.1f} px")
    print(f"  Mean Area:        {np.mean(areas):.1f} px")
    print(f"  Min/Max Area:     {np.min(areas)} / {np.max(areas)} px")
    print(f"  SHA-256:          {sha256}")
    print("  AUDIT RESULT:     >>> PASSED (CHAMPION GRADE) <<<\n")
    return sha256

if __name__ == "__main__":
    sub_dir = r"d:\Kaggle\submissions"
    s1 = audit_submission(os.path.join(sub_dir, "submission_champion_v1_precision_snapped.csv"), 1199)
    s2 = audit_submission(os.path.join(sub_dir, "submission_rebuilt_041_precision_snapped.csv"), 1206)
