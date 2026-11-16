import os, sys, pandas as pd, numpy as np
from pycocotools import mask as mu

# Base 0.410 champion
f_base = 'd:/Kaggle/submissions/submission_champion_v1_precision_snapped.csv'
if not os.path.exists(f_base):
    f_base = 'd:/Kaggle/submissions/submission_rebuilt_champion_041.csv'

# clDice candidate
f_cldice = 'd:/Kaggle/submissions/submission_champion_cldice_tta_tau050.csv'
f_out = 'd:/Kaggle/submissions/submission_champion_cldice_guardrailed_043plus.csv'

print(f"Base: {f_base}")
print(f"clDice: {f_cldice}")

df_base = pd.read_csv(f_base)
df_cldice = pd.read_csv(f_cldice)

m = pd.merge(df_base, df_cldice, on='filament_id', suffixes=('_base', '_cldice'))

rles1 = [{'counts': r['segmentation_rle_base'].encode('utf-8'), 'size': [2048, 2048]} for _, r in m.iterrows()]
rles2 = [{'counts': r['segmentation_rle_cldice'].encode('utf-8'), 'size': [2048, 2048]} for _, r in m.iterrows()]

ious = [float(mu.iou([r1], [r2], [0])[0, 0]) for r1, r2 in zip(rles1, rles2)]
ious = np.array(ious)
areas_b = np.array([int(mu.area([r1])[0]) for r1 in rles1])
areas_c = np.array([int(mu.area([r2])[0]) for r2 in rles2])
ratios = areas_c / np.maximum(areas_b, 1)

m['stem'] = m['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])
# An image is flagged if any instance has severe anomaly (IoU < 0.65 or ratio > 1.6 or ratio < 0.5)
m['bad_inst'] = (ious < 0.65) | (ratios > 1.6) | (ratios < 0.5)

bad_stems = set(m[m['bad_inst']]['stem'])
print(f"Identified {len(bad_stems)} crowded/cannibalized images: {bad_stems}")

final_rows = []
upgraded_count = 0
retained_base_count = 0

for stem, grp in m.groupby('stem', sort=False):
    if stem in bad_stems:
        # Revert entire image to 0.410 baseline to ensure 100% zero collisions and zero TP loss
        for _, r in grp.iterrows():
            final_rows.append({
                'filament_id': r['filament_id'],
                'segmentation_rle': r['segmentation_rle_base']
            })
            retained_base_count += 1
    else:
        # Safe image: keep clDice TTA upgraded masks
        for _, r in grp.iterrows():
            final_rows.append({
                'filament_id': r['filament_id'],
                'segmentation_rle': r['segmentation_rle_cldice']
            })
            upgraded_count += 1

out_df = pd.DataFrame(final_rows)
out_df.to_csv(f_out, index=False)

print(f"\nSaved Guardrailed Champion -> {f_out}")
print(f"Total instances: {len(out_df)}")
print(f"Safely upgraded instances: {upgraded_count} ({upgraded_count/len(out_df)*100:.1f}%)")
print(f"Retained 0.410 baseline instances: {retained_base_count} ({retained_base_count/len(out_df)*100:.1f}%)")

# Audit overlaps
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

print(f"Overlapping instances: {overlaps}")
print(f"Total segmentation px: {total_px:,}")
print(f"Mean instance area:    {total_px / len(out_df):.1f} px")
print(f"VERDICT: {'PASSED PERFECTLY (100% VALID)' if (overlaps == 0 and len(out_df) == 1199) else 'FAILED'}")
