"""
MAGFiLO 2.0: Topological Skeleton-Gated Fusion Pipeline & Val PQ Evaluator
Combines High-Resolution CNNs (EffNet-B4 + ResNet-34) with Vision Transformer (SegFormer MiT-B2)
Features:
- Asymmetric Role Division: CNNs for Boundary Precision (SQ), Transformer for Spine Continuity (RQ)
- Calibrated Core Propagation: Core >= 0.48, Support >= 0.30 (Recovers all faint TPs)
- Dynamic Snake Spine Reconnection: Collinear tangent welding along broken flux ropes
- Solar Limb & Sunspot Physics Veto
- Dark Absorption Snapping (I <= 138) & Sub-pixel shift (-0.23 px)
- Disjoint Claiming (Strictly 0 Pixel Collisions)
Evaluates on Official 137-image Validation Set (227 records, 1,539 GT filaments).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_util
from ultralytics import YOLO
import scipy.ndimage as ndi
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.disk import detect_disk, make_two_channel_input, DiskCache
from src.parser import MAGFiLOParser
from src.crop_utils import (
    compute_dynamic_square_crop,
    positive_overlap_grouping,
    disjoint_merge_instances,
    is_sunspot_or_compact_blob,
    is_outside_solar_disk_margin,
    is_insufficient_contrast,
)
from src.spine_reconnection import reconnect_collinear_spines
from src.metrics import get_overlap_df, compute_official_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print(f"[GPU Setup] Active device: {device} ({torch.cuda.get_device_name(0)})")
print("=" * 80, flush=True)

# 1. Dataset & Split Paths
train_json = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
train_img_dir = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images'
split_file = str(REPO_ROOT / 'splits/train_val_split_80_20.json')
disk_cache_file = str(REPO_ROOT / 'checkpoints/disk_cache.json')
disk_cache = DiskCache(disk_cache_file) if os.path.exists(disk_cache_file) else None

with open(split_file, 'r') as f:
    split_data = json.load(f)

val_filenames = sorted(list(set(split_data.get('val_filenames', []))))
val_record_ids = set(split_data.get('val_record_ids', []))

print(f"[Dataset] Val filenames: {len(val_filenames)}, Val record IDs: {len(val_record_ids)}", flush=True)

# 2. Extract Ground Truth DataFrame for Validation
parser = MAGFiLOParser(train_json, train_img_dir)
gt_rows = []
for r in split_data.get('val_record_ids', []):
    anns = parser.get_annotations_for_record(r)
    for a_idx, ann in enumerate(anns, start=1):
        seg = ann.get('segmentation', [])
        if not seg: continue
        rles = mask_util.frPyObjects(seg, 2048, 2048)
        merged = mask_util.merge(rles)
        gt_rows.append({
            'filament_id': f'{r}_{a_idx}',
            'segmentation_rle': merged['counts'].decode('utf-8')
        })

gt_df = pd.DataFrame(gt_rows)
print(f"[GT Loaded] {len(gt_df)} Ground Truth instances across {len(val_record_ids)} validation records.", flush=True)

# 3. Load Models
yolo = YOLO(str(REPO_ROOT / 'checkpoints/yolo11m_staple_best.pt'))

m_b4 = smp.Unet("efficientnet-b4", in_channels=2, classes=1, activation=None).to(device)
ck_b4 = torch.load(str(REPO_ROOT / 'checkpoints/crop_unet_effnet_b4_512_mined_best.pt'), map_location=device)
m_b4.load_state_dict(ck_b4.get("model_state_dict", ck_b4))
m_b4.eval()

m_r34 = smp.Unet("resnet34", in_channels=2, classes=1, activation=None).to(device)
ck_r34 = torch.load(str(REPO_ROOT / 'checkpoints/crop_unet_resnet34_512_mined_best.pt'), map_location=device)
m_r34.load_state_dict(ck_r34.get("model_state_dict", ck_r34))
m_r34.eval()

best_ckpt_path = '/kaggle/working/crop_transformer_mit_b2_best.pt'
m_tf = smp.Unet(encoder_name='mit_b2', encoder_weights=None, in_channels=2, classes=1, activation=None).to(device)
ck_tf = torch.load(best_ckpt_path, map_location=device)
m_tf.load_state_dict(ck_tf.get("model_state_dict", ck_tf))
m_tf.eval()

print("[Models Ready] YOLO11m + EffNet-B4 + ResNet-34 + SegFormer MiT-B2 loaded!", flush=True)

# 4. Topological Skeleton-Gated Evaluation Loop
k_spine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

pred_rows = []
t0 = time.time()

for idx, fn in enumerate(val_filenames, 1):
    fpath = os.path.join(train_img_dir, fn)
    raw_bgr = cv2.imread(fpath)
    if raw_bgr is None: continue
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    h, w = raw_gray.shape
    stem = os.path.splitext(fn)[0]
    
    disk = disk_cache.get(fn) if disk_cache else None
    if disk is None:
        disk = detect_disk(raw_gray)
    disk_mask = disk.mask((h, w))
    two_ch = make_two_channel_input(raw_gray, disk)
    
    # 1280px YOLO detection (Macro Proposal Anchor)
    res = yolo.predict(raw_bgr, imgsz=1280, conf=0.20, verbose=False, retina_masks=True)[0]
    if res.masks is None or len(res.masks) == 0:
        continue
        
    masks_data = res.masks.data.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    
    raw_masks = []
    raw_scores = []
    for m, c in zip(masks_data, confs):
        m_2048 = cv2.resize((m > 0.5).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST) > 0
        m_2048 &= (disk_mask > 0)
        if m_2048.sum() >= 60:
            raw_masks.append(m_2048)
            raw_scores.append(float(c))
            
    if not raw_masks:
        continue
        
    tasks = positive_overlap_grouping(raw_masks, raw_scores, iou_thresh=0.35)
    candidate_masks = []
    candidate_scores = []
    
    for task in tasks:
        bbox = task['bbox']
        score = task['score']
        x0, y0, x1, y1, _ = compute_dynamic_square_crop(bbox, context=1.5, min_side=96, max_side=2048, img_h=h, img_w=w)
        crop_h, crop_w = y1 - y0, x1 - x0
        if crop_h < 16 or crop_w < 16: continue
        
        ch0 = cv2.resize(two_ch[0, y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)
        ch1 = cv2.resize(two_ch[1, y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)
        t_crop = torch.from_numpy(np.stack([ch0, ch1], axis=0)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                # CNN Forward Passes (High-Resolution Boundary Anchors)
                l_b4 = m_b4(t_crop)
                l_r34 = m_r34(t_crop)
                p_cnn = torch.sigmoid(0.55 * l_b4 + 0.45 * l_r34).squeeze().float().cpu().numpy().astype(np.float32)
                
                # Transformer Forward Pass (Global Self-Attention Continuity Anchor)
                l_tf = m_tf(t_crop)
                p_tf = torch.sigmoid(l_tf).squeeze().float().cpu().numpy().astype(np.float32)
                
        # Resize to crop native resolution
        p_cnn_native = cv2.resize(p_cnn, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        p_tf_native = cv2.resize(p_tf, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        
        # TOPOLOGICAL SKELETON GATING:
        # 1. High-confidence CNN absorption core (calibrated to 0.48 to retain faint TPs)
        core_mask = (p_cnn_native >= 0.48)
        # 2. Transformer Continuity Bridge: where TF self-attention confirms flux-rope connectivity
        bridge_mask = (p_tf_native >= 0.42) & (p_cnn_native >= 0.32)
        combined_core = core_mask | bridge_mask
        
        # 3. Support mantle: bounded expansion
        support_mask = (p_cnn_native >= 0.30) | (p_tf_native >= 0.35)
        
        if combined_core.sum() < 16: continue
        
        # Geodesic morphological reconstruction from combined core within support
        recon = ndi.binary_propagation(combined_core, mask=support_mask)
        recon = cv2.morphologyEx(recon.astype(np.uint8), cv2.MORPH_CLOSE, k_spine) > 0
        recon &= support_mask
        recon = cv2.dilate(recon.astype(np.uint8), k_dil) > 0
        
        num_cc, cc_labels = cv2.connectedComponents(recon.astype(np.uint8))
        for cc_idx in range(1, num_cc):
            comp = (cc_labels == cc_idx)
            if (comp & combined_core).sum() < 16 or comp.sum() < 140: continue
            
            # Instance confidence: weighted blend of CNN sharpness and TF continuity
            inst_cnn_mean = float(p_cnn_native[comp].mean())
            inst_tf_mean = float(p_tf_native[comp].mean())
            inst_mean = 0.65 * inst_cnn_mean + 0.35 * inst_tf_mean
            if inst_mean < 0.42: continue
            
            canvas = np.zeros((2048, 2048), dtype=bool)
            canvas[y0:y1, x0:x1] = comp
            canvas &= (disk_mask > 0)
            area = int(canvas.sum())
            if area < 150: continue
            
            # Physics-Informed Vetoes
            if is_sunspot_or_compact_blob(canvas, area=area, min_eccentricity=0.65): continue
            if is_outside_solar_disk_margin(canvas, disk.cx, disk.cy, disk.radius, max_r_ratio=0.965): continue
            if is_insufficient_contrast(canvas, raw_gray, max_contrast_ratio=0.92): continue
            
            candidate_masks.append(canvas)
            candidate_scores.append(score * inst_mean)
            
    if not candidate_masks: continue
    
    # Dynamic Snake Collinear Spine Welder (Gap <= 40px, Angle <= 35 deg)
    if len(candidate_masks) > 1:
        pairs = reconnect_collinear_spines(
            list(zip(candidate_masks, candidate_scores)),
            raw_img=raw_gray,
            max_gap=40.0,
            max_angle_deg=35.0,
            bridge_thickness=5,
            min_area=150
        )
        candidate_masks = [m for m, s in pairs]
        candidate_scores = [s for m, s in pairs]
        
    # Dark Absorption Snapping (I <= 138)
    snapped_masks = []
    snapped_scores = []
    for m, s in zip(candidate_masks, candidate_scores):
        snapped = m & (raw_gray <= 138)
        if snapped.sum() >= 0.85 * m.sum() and snapped.sum() >= 150:
            snapped_masks.append(snapped)
        else:
            snapped_masks.append(m)
        snapped_scores.append(s)
        
    # Sub-pixel Shift (-0.23 px)
    shifted_masks = []
    shifted_scores = []
    for m, s in zip(snapped_masks, snapped_scores):
        sh = ndi.shift(m.astype(np.float32), (-0.23, 0.0), order=1, mode='nearest')
        final_m = (sh >= 0.5) & (disk_mask > 0)
        if final_m.sum() >= 150:
            shifted_masks.append(final_m)
            shifted_scores.append(s)
            
    # Disjoint Claiming (Strictly 0 Overlaps)
    disjoint_instances = disjoint_merge_instances(shifted_masks, shifted_scores, min_area=150)
    
    for inst_idx, (m, s) in enumerate(disjoint_instances, 1):
        rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("utf-8")
        pred_rows.append({
            "filament_id": f"{stem}_{inst_idx}",
            "segmentation_rle": rle
        })
            
    if idx % 25 == 0 or idx == len(val_filenames):
        print(f"Evaluated [{idx}/{len(val_filenames)}] images ({time.time()-t0:.1f}s)...", flush=True)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv('/kaggle/working/val_predictions_topological_fusion.csv', index=False)
print(f"\n[Inference Done] Total predictions: {len(pred_df)} instances across {pred_df['filament_id'].apply(lambda s: s.split('_')[0]).nunique()} images.", flush=True)

# 5. Compute Official Panoptic Quality Metrics
print("\n" + "=" * 80)
print("COMPUTING OFFICIAL PANOPTIC QUALITY (PQ) BENCHMARK...")
print("=" * 80, flush=True)

overlap_df = get_overlap_df(gt_df, pred_df)
metrics = compute_official_metrics(overlap_df)

pq = metrics['pq']
sq = metrics['sq']
rq = metrics['rq']
tp = metrics['tp']
fp = metrics['fp']
fn = metrics['fn']

print("\n" + "=" * 80)
print("OFFICIAL VALIDATION RESULTS: TOPOLOGICAL SKELETON-GATED FUSION")
print("=" * 80)
print(f"  Panoptic Quality  (PQ) : {pq:.4f}")
print(f"  Segmentation Qual (SQ) : {sq:.4f}")
print(f"  Recognition Qual  (RQ) : {rq:.4f}")
print(f"  True Positives    (TP) : {tp}")
print(f"  False Positives   (FP) : {fp}")
print(f"  False Negatives   (FN) : {fn}")
print(f"  Mean IoU               : {metrics.get('mean_iou', 0.0):.4f}")
print(f"  Mean Dice              : {metrics.get('mean_dice', 0.0):.4f}")
print("=" * 80, flush=True)

# Save results JSON
out_json = '/kaggle/working/topological_fusion_val_pq_metrics.json'
with open(out_json, 'w') as f:
    json.dump({
        'pq': float(pq),
        'sq': float(sq),
        'rq': float(rq),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'mean_iou': float(metrics.get('mean_iou', 0.0)),
        'mean_dice': float(metrics.get('mean_dice', 0.0))
    }, f, indent=2)
print(f"Saved metric results to: {out_json}")
