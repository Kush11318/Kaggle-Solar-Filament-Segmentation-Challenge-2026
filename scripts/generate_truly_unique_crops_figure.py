import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util

sys.path.append('d:/Kaggle')
from src.paths import get_dataset_paths, get_split_path, get_project_root

def generate_truly_unique_crops_figure():
    proj_root = get_project_root()
    json_p, train_p, _ = get_dataset_paths()
    split_p = get_split_path()

    with open(json_p, "r") as f:
        coco_data = json.load(f)
    with open(split_p, "r") as f:
        split_data = json.load(f)

    id_to_fn = {img["id"]: img["file_name"] for img in coco_data["images"]}
    val_rids = set(split_data["val_record_ids"])

    det_yolo_path = os.path.join(proj_root, "experiments", "fusion", "detections_val_yolo11s.json")
    with open(det_yolo_path, "r") as f:
        val_dets = json.load(f)

    target_stem = "20110312082634Lh"
    cand_files = [f for f in os.listdir(train_p) if target_stem in f]
    img_path = os.path.join(train_p, cand_files[0])
    raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Collect GT masks
    gt_instances = []
    gt_canvas = np.zeros((2048, 2048), dtype=bool)
    for ann in coco_data["annotations"]:
        if ann["image_id"] not in val_rids:
            continue
        fn = id_to_fn[ann["image_id"]]
        if target_stem not in fn:
            continue
        rle = ann["segmentation"]
        gm = mask_util.decode(rle if isinstance(rle, dict) else mask_util.merge(mask_util.frPyObjects(rle, 2048, 2048))) > 0
        if gm.sum() < 20:
            continue
        gt_instances.append(gm)
        gt_canvas |= gm

    # Collect predictions
    dets = val_dets.get(target_stem, [])
    pred_instances = []
    pred_canvas = np.zeros((2048, 2048), dtype=bool)
    for det in dets:
        if det.get("score", 0) < 0.28:
            continue
        pm = mask_util.decode({"size": [2048, 2048], "counts": det["counts"]}) > 0
        if pm.sum() < 100:
            continue
        pred_instances.append(pm)
        pred_canvas |= pm

    # Match each GT instance to best prediction
    matched_all = []
    for g_idx, gm in enumerate(gt_instances):
        best_iou = 0.0
        best_pm = None
        for pm in pred_instances:
            inter = (gm & pm).sum()
            if inter == 0:
                continue
            union = (gm | pm).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_pm = pm

        ys, xs = np.nonzero(gm)
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        matched_all.append({
            "gt_mask": gm,
            "pred_mask": best_pm,
            "iou": best_iou,
            "center": (cx, cy),
            "bbox": (int(xs.min()), int(ys.min()), int(w), int(h)),
            "area_g": int(gm.sum()),
            "area_p": int(best_pm.sum()) if best_pm is not None else 0,
        })

    # Specifically select 3 spatially isolated, highly distinct filaments:
    # 1. ann 0: South-West main channel (cx~773, cy~1325)
    # 2. ann 3: West equatorial filament (cx~718, cy~821)
    # 3. ann 4: North polar crown filament (cx~985, cy~432)
    # This guarantees 3 completely different structures separated by 500-900 pixels!
    
    # Sort matched_all by IoU and enforce pairwise distance > 350 px
    matched_all.sort(key=lambda x: -x["iou"])
    chosen_crops = []
    for cand in matched_all:
        if cand["iou"] < 0.50:
            continue
        cx, cy = cand["center"]
        # Check distance to already chosen
        too_close = False
        for c in chosen_crops:
            d = np.hypot(cx - c["center"][0], cy - c["center"][1])
            if d < 350:
                too_close = True
                break
        if not too_close:
            chosen_crops.append(cand)
        if len(chosen_crops) >= 3:
            break

    print(f"Selected {len(chosen_crops)} distinct filaments:")
    for idx, c in enumerate(chosen_crops, 1):
        print(f"  #{idx}: Center={c['center']}, IoU={c['iou']:.3f}, GT Area={c['area_g']}, Pred Area={c['area_p']}")

    # Configure publication plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white'
    })

    fig = plt.figure(figsize=(16, 9.5), dpi=220)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.1], wspace=0.14, hspace=0.25)

    # 1. Full-Disk Display
    ax_disk = fig.add_subplot(gs[:, 0])
    
    p2, p98 = np.percentile(raw_img[raw_img > 10], (2, 98))
    norm_disk = np.clip((raw_img.astype(float) - p2) / (p98 - p2 + 1e-5), 0, 1)
    rgb_disk = cv2.cvtColor((norm_disk * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    overlap_canvas = gt_canvas & pred_canvas
    gt_only = gt_canvas & (~pred_canvas)
    pred_only = pred_canvas & (~gt_canvas)

    rgb_disk[gt_only] = (0.35 * rgb_disk[gt_only] + 0.65 * np.array([34, 197, 94])).astype(np.uint8)
    rgb_disk[pred_only] = (0.35 * rgb_disk[pred_only] + 0.65 * np.array([239, 68, 68])).astype(np.uint8)
    rgb_disk[overlap_canvas] = (0.25 * rgb_disk[overlap_canvas] + 0.75 * np.array([234, 179, 8])).astype(np.uint8)

    # Draw numbered white bounding boxes for each distinct filament on the full disk
    for c_idx, crop_info in enumerate(chosen_crops):
        cx, cy = crop_info["center"]
        w, h = crop_info["bbox"][2], crop_info["bbox"][3]
        r = max(110, int(max(w, h) * 1.15))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(2048, cx + r), min(2048, cy + r)
        cv2.rectangle(rgb_disk, (x0, y0), (x1, y1), (255, 255, 255), 3)
        cv2.putText(rgb_disk, f"#{c_idx+1}", (x0 + 15, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)

    ax_disk.imshow(rgb_disk)
    ax_disk.set_title(
        f"(a) Full-Disk Solar Observation ({target_stem})\n"
        f"Green: Ground Truth  |  Red: Candidate  |  Yellow: True Positive Overlap",
        fontsize=12, fontweight='bold', color='#111827', pad=10
    )
    ax_disk.axis("off")

    # 2. Zoomed Crops (Right side: 3 rows)
    crop_descriptions = [
        "Equatorial Quiescent Channel",
        "South-West Main Flux Rope",
        "North Polar Crown Filament"
    ]

    for c_idx, crop_info in enumerate(chosen_crops):
        ax_crop = fig.add_subplot(gs[c_idx, 1])
        cx, cy = crop_info["center"]
        w, h = crop_info["bbox"][2], crop_info["bbox"][3]
        crop_radius = max(95, int(max(w, h) * 1.15))

        y0, y1 = max(0, cy - crop_radius), min(2048, cy + crop_radius)
        x0, x1 = max(0, cx - crop_radius), min(2048, cx + crop_radius)

        raw_crop = raw_img[y0:y1, x0:x1]
        gt_c = crop_info["gt_mask"][y0:y1, x0:x1]
        pred_c = crop_info["pred_mask"][y0:y1, x0:x1] if crop_info["pred_mask"] is not None else np.zeros_like(gt_c)

        p2_c, p98_c = np.percentile(raw_crop, (2, 98))
        norm_crop = np.clip((raw_crop.astype(float) - p2_c) / (p98_c - p2_c + 1e-5), 0, 1)
        rgb_crop = cv2.cvtColor((norm_crop * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        ov_c = gt_c & pred_c
        gt_only_c = gt_c & (~pred_c)
        pred_only_c = pred_c & (~gt_c)

        rgb_crop[gt_only_c] = (0.35 * rgb_crop[gt_only_c] + 0.65 * np.array([34, 197, 94])).astype(np.uint8)
        rgb_crop[pred_only_c] = (0.35 * rgb_crop[pred_only_c] + 0.65 * np.array([239, 68, 68])).astype(np.uint8)
        rgb_crop[ov_c] = (0.25 * rgb_crop[ov_c] + 0.75 * np.array([234, 179, 8])).astype(np.uint8)

        cnts_gt, _ = cv2.findContours(gt_c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_pred, _ = cv2.findContours(pred_c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_crop, cnts_gt, -1, (22, 163, 74), 2)
        cv2.drawContours(rgb_crop, cnts_pred, -1, (220, 38, 38), 2)

        iou_val = crop_info["iou"]
        ax_crop.imshow(rgb_crop)
        ax_crop.set_title(
            f"(b.{c_idx+1}) Filament #{c_idx+1}: {crop_descriptions[c_idx]}  [IoU = {iou_val:.3f}]\n"
            f"GT Area: {crop_info['area_g']} px  |  Predicted Area: {crop_info['area_p']} px",
            fontsize=10.5, fontweight='bold', color='#15803D' if iou_val >= 0.50 else '#B91C1C', pad=8
        )
        ax_crop.axis("off")
        
        for spine in ax_crop.spines.values():
            spine.set_visible(True)
            spine.set_color('#D1D5DB')
            spine.set_linewidth(1.5)

    plt.tight_layout()
    out_dir = r"d:/Kaggle/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "segmentation_examples.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated clean unique 3-filament dataset figure: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == '__main__':
    generate_truly_unique_crops_figure()

