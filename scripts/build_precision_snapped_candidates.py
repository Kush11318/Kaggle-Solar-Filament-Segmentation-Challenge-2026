import os, sys, cv2, time
import pandas as pd
import numpy as np
from pycocotools import mask as mu

def build_snapped_submission(input_csv, output_csv, img_dir, intensity_threshold=138, min_retention=0.95):
    print("=" * 80)
    print(f"Building: {os.path.basename(output_csv)}")
    print(f"Source:   {os.path.basename(input_csv)}")
    print("=" * 80)
    
    df = pd.read_csv(input_csv)
    df['image_stem'] = df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])
    
    stems = sorted(df['image_stem'].unique())
    print(f"Loaded {len(df)} instances across {len(stems)} unique test images.")
    
    out_rows = []
    total_insts = 0
    modified_insts = 0
    total_removed_pixels = 0
    t0 = time.time()
    
    for idx, stem in enumerate(stems, 1):
        img_path = os.path.join(img_dir, stem + ".jpeg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, stem + ".jpg")
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_path) else None
        
        grp = df[df['image_stem'] == stem]
        canvas = np.zeros((2048, 2048), dtype=np.uint8)
        
        for inst_i, (_, row) in enumerate(grp.iterrows(), 1):
            total_insts += 1
            rle = row['segmentation_rle']
            m = mu.decode({"counts": rle.encode('utf-8') if isinstance(rle, str) else rle, "size": [2048, 2048]})
            orig_area = int(m.sum())
            
            chosen = m
            if raw_img is not None and orig_area >= 100:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                eroded = cv2.erode(m.astype(np.uint8), kernel)
                fringe = (m > 0) & (eroded == 0)
                
                # Granulation fringe pixels (intensity > threshold)
                gran_fringe = fringe & (raw_img > intensity_threshold)
                n_gran = int(gran_fringe.sum())
                
                if n_gran > 0 and (orig_area - n_gran) >= min_retention * orig_area:
                    m_snapped = m.copy()
                    m_snapped[gran_fringe] = 0
                    
                    # Ensure no collision with canvas
                    if not np.any((canvas > 0) & (m_snapped > 0)):
                        chosen = m_snapped
                        modified_insts += 1
                        total_removed_pixels += n_gran
            
            # Allocation on canvas to prevent any collision
            canvas[chosen > 0] = inst_i
            
            f_mask = np.asfortranarray(chosen.astype(np.uint8))
            enc = mu.encode(f_mask)['counts'].decode('utf-8')
            out_rows.append({
                'filament_id': row['filament_id'],
                'segmentation_rle': enc
            })
            
        if idx % 50 == 0 or idx == len(stems):
            print(f"  Processed {idx}/{len(stems)} images ({time.time()-t0:.1f}s)...")
            
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Successfully wrote {len(out_df)} instances to {output_csv}")
    print(f"Modified instances: {modified_insts}/{total_insts} ({modified_insts/total_insts*100:.1f}%)")
    print(f"Total granulation pixels removed: {total_removed_pixels} (avg {total_removed_pixels/max(1, modified_insts):.1f} px/inst)")
    
    # Audit
    print("--- Running Verification Audit ---")
    assert len(out_df) == len(df), f"Instance count mismatch: {len(out_df)} vs {len(df)}"
    assert out_df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique() == len(stems), "Missing images!"
    print("Verification Audit: 100% PASSED!\n")

if __name__ == "__main__":
    img_dir = r"d:\Kaggle\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026\test\test_images"
    sub_dir = r"d:\Kaggle\submissions"
    
    v1_in = os.path.join(sub_dir, "submission_champion_hybrid_v1.csv")
    v1_out = os.path.join(sub_dir, "submission_champion_v1_precision_snapped.csv")
    
    reb_in = os.path.join(sub_dir, "submission_rebuilt_champion_041.csv")
    reb_out = os.path.join(sub_dir, "submission_rebuilt_041_precision_snapped.csv")
    
    build_snapped_submission(v1_in, v1_out, img_dir, intensity_threshold=138, min_retention=0.95)
    build_snapped_submission(reb_in, reb_out, img_dir, intensity_threshold=138, min_retention=0.95)
