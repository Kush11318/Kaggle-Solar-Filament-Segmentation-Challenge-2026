import os
import sys
import pandas as pd
import numpy as np
from pycocotools import mask as mu

def main():
    f_base = 'd:/Kaggle/submissions/submission_champion_v1_precision_snapped.csv'
    f_dual = 'd:/Kaggle/submissions/submission_champion_dual_cldice_ensemble.csv'
    f_out = 'd:/Kaggle/submissions/submission_champion_dual_ensemble_guardrailed.csv'

    print(f"Base: {f_base}", flush=True)
    print(f"Dual: {f_dual}", flush=True)

    df_base = pd.read_csv(f_base)
    df_dual = pd.read_csv(f_dual)

    m = pd.merge(df_base, df_dual, on='filament_id', suffixes=('_base', '_dual'))

    rles1 = [{'counts': r['segmentation_rle_base'].encode('utf-8'), 'size': [2048, 2048]} for _, r in m.iterrows()]
    rles2 = [{'counts': r['segmentation_rle_dual'].encode('utf-8'), 'size': [2048, 2048]} for _, r in m.iterrows()]

    ious = np.array([float(mu.iou([r1], [r2], [0])[0, 0]) for r1, r2 in zip(rles1, rles2)])
    areas_b = np.array([int(mu.area([r1])[0]) for r1 in rles1])
    areas_d = np.array([int(mu.area([r2])[0]) for r2 in rles2])
    ratios = areas_d / np.maximum(areas_b, 1)

    m['stem'] = m['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])
    m['bad_inst'] = (ious < 0.65) | (ratios > 1.6) | (ratios < 0.5)

    bad_stems = set(m[m['bad_inst']]['stem'])
    print(f"Identified {len(bad_stems)} crowded/cannibalized images in dual ensemble: {bad_stems}", flush=True)

    final_rows = []
    upgraded_count = 0
    retained_base_count = 0

    for stem, grp in m.groupby('stem', sort=False):
        if stem in bad_stems:
            for _, r in grp.iterrows():
                final_rows.append({
                    'filament_id': r['filament_id'],
                    'segmentation_rle': r['segmentation_rle_base']
                })
                retained_base_count += 1
        else:
            for _, r in grp.iterrows():
                final_rows.append({
                    'filament_id': r['filament_id'],
                    'segmentation_rle': r['segmentation_rle_dual']
                })
                upgraded_count += 1

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(f_out, index=False)

    print(f"\nSaved -> {f_out}", flush=True)
    print(f"Total instances: {len(out_df)}", flush=True)
    print(f"Safely upgraded instances: {upgraded_count} ({upgraded_count/len(out_df)*100:.1f}%)", flush=True)
    print(f"Retained base instances: {retained_base_count} ({retained_base_count/len(out_df)*100:.1f}%)", flush=True)

    overlaps = 0
    total_px = 0
    for stem, grp in out_df.groupby(out_df['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])):
        canvas = np.zeros((2048, 2048), dtype=bool)
        for _, r in grp.iterrows():
            mask = mu.decode({'counts': r['segmentation_rle'].encode('utf-8'), 'size': [2048, 2048]}) > 0
            total_px += int(mask.sum())
            if np.logical_and(canvas, mask).any():
                overlaps += 1
            canvas |= mask

    print(f"Overlapping instances: {overlaps}", flush=True)
    print(f"Total segmentation px: {total_px:,}", flush=True)
    print(f"Mean instance area:    {total_px / len(out_df):.1f} px", flush=True)
    print(f"VERDICT: {'PASSED PERFECTLY (100% VALID)' if (overlaps == 0 and len(out_df) == 1199) else 'FAILED'}", flush=True)

if __name__ == '__main__':
    main()
