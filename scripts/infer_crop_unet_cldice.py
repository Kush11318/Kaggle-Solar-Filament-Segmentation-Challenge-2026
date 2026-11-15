"""
Spartan Phase 3: High-Resolution Test Inference with clDice-Trained CropUNet
===========================================================================
Applies the 3-channel box-prompted clDice CropUNet to refine all 1,199 instances
from the defended 0.410 champion baseline while strictly maintaining:
- Strict Instance Volume: exactly 1,199 instances across 178 test images
- Zero True Positive Drop: fallback to baseline if predicted mask < 40 px
- Zero Inter-Instance Overlap: pixel-level conflict resolution
- Independent GPU execution on cuda:1 (parallel to ongoing training on cuda:0)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import get_dataset_paths, get_project_root
from src.disk import detect_disk, make_two_channel_input, DiskCache
from src.crop_utils import compute_dynamic_square_crop


def rle_decode(rle_str, shape=(2048, 2048)):
    return mask_util.decode({"counts": rle_str.encode("utf-8"), "size": list(shape)})


def rle_encode(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def resolve_overlaps_greedy(masks, areas):
    """
    Ensures strictly zero overlapping pixels across masks in a single image.
    Higher area / earlier priority retains contentious pixels. Guaranteed 100% disjoint.
    """
    if len(masks) <= 1:
        return masks

    occupied = np.zeros((2048, 2048), dtype=bool)
    resolved = []
    # Sort descending by area to give larger features priority
    order = np.argsort(-np.array(areas))
    inv_order = np.zeros(len(masks), dtype=int)
    for rank, idx in enumerate(order):
        inv_order[idx] = rank

    sorted_masks = [masks[i] for i in order]
    for m in sorted_masks:
        clean = (m > 0) & (~occupied)
        if clean.sum() > 0:
            occupied |= clean
            resolved.append(clean.astype(np.uint8))
        else:
            # Completely submerged by a larger instance.
            # Carve an unoccupied pixel nearest to centroid to guarantee non-empty and non-colliding.
            ys, xs = np.where(m > 0)
            cy, cx = (int(np.mean(ys)), int(np.mean(xs))) if len(ys) > 0 else (1024, 1024)
            assigned = False
            for rad in range(1, 40):
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < 2048 and 0 <= nx < 2048 and not occupied[ny, nx] and not assigned:
                            dot = np.zeros((2048, 2048), dtype=np.uint8)
                            dot[ny, nx] = 1
                            occupied[ny, nx] = True
                            resolved.append(dot)
                            assigned = True
                            break
                    if assigned:
                        break
                if assigned:
                    break
            if not assigned:
                resolved.append(clean.astype(np.uint8))

    # Restore original order
    final_masks = [None] * len(masks)
    for orig_idx, rank in enumerate(inv_order):
        final_masks[orig_idx] = resolved[rank]

    return final_masks


def run_inference(
    baseline_csv: str,
    checkpoint_path: str,
    output_csv: str,
    device_id: str = "1",
    threshold: float = 0.50,
    blend_weight: float = 0.0,
    context: float = 1.6,
    crop_size: int = 512,
    use_tta: bool = False,
):
    print("=" * 80, flush=True)
    print("SPARTAN INFERENCE: clDICE-REFINED HIGH-RESOLUTION PREDICTOR", flush=True)
    print(f"Baseline CSV:     {baseline_csv}", flush=True)
    print(f"Checkpoint:       {checkpoint_path}", flush=True)
    print(f"Output CSV:       {output_csv}", flush=True)
    print(f"Device: cuda:{device_id} | Threshold: {threshold} | Blend: {blend_weight} | Size: {crop_size}x{crop_size} | TTA: {use_tta}", flush=True)
    print("=" * 80, flush=True)

    if not os.path.exists(baseline_csv):
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, _, test_img_dir = get_dataset_paths()
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image dir not found: {test_img_dir}")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Load Model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    encoder_name = cfg.get("encoder", "efficientnet-b4")
    print(f"Loaded checkpoint Epoch {ckpt.get('epoch')} | Val Dice: {ckpt.get('val_dice', 0):.4f} | Val IoU: {ckpt.get('val_iou', 0):.4f}", flush=True)

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load Baseline
    df_base = pd.read_csv(baseline_csv)
    print(f"Loaded baseline with {len(df_base):,} instances across {df_base['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} images.", flush=True)

    df_base["stem"] = df_base["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    grouped = df_base.groupby("stem", sort=False)

    proj_root = get_project_root()
    cache_path = os.path.join(proj_root, "checkpoints", "disk_cache.json")
    disk_cache = DiskCache(cache_path) if os.path.exists(cache_path) else None

    refined_rows = []
    t0 = time.time()
    total_images = len(grouped)

    total_upgraded = 0
    total_retained_fallback = 0

    with torch.no_grad():
        for img_idx, (stem, grp) in enumerate(grouped, 1):
            # Locate image file
            cand_files = [f"{stem}.jpeg", f"{stem}.png", f"{stem}.jpg"]
            img_path = None
            for cf in cand_files:
                p = os.path.join(test_img_dir, cf)
                if os.path.exists(p):
                    img_path = p
                    break

            if img_path is None:
                # Retain original rows untouched if image file missing
                for _, r in grp.iterrows():
                    refined_rows.append({"filament_id": r["filament_id"], "segmentation_rle": r["segmentation_rle"]})
                continue

            raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                for _, r in grp.iterrows():
                    refined_rows.append({"filament_id": r["filament_id"], "segmentation_rle": r["segmentation_rle"]})
                continue

            disk = disk_cache.get(os.path.basename(img_path)) if disk_cache else None
            if disk is None:
                disk = detect_disk(raw_img)
            two_ch = make_two_channel_input(raw_img, disk)

            img_masks = []
            img_fids = []
            img_areas = []

            for _, row in grp.iterrows():
                fid = row["filament_id"]
                rle = str(row["segmentation_rle"])
                orig_m = rle_decode(rle)
                img_fids.append(fid)

                ys, xs = np.where(orig_m > 0)
                if len(ys) == 0:
                    img_masks.append(orig_m)
                    img_areas.append(0)
                    total_retained_fallback += 1
                    continue

                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bw = x_max - x_min + 1
                bh = y_max - y_min + 1
                bbox = [float(x_min), float(y_min), float(bw), float(bh)]

                x0, y0, x1, y1, _ = compute_dynamic_square_crop(
                    bbox=bbox,
                    context=context,
                    min_side=96,
                    max_side=2048,
                    img_h=2048,
                    img_w=2048,
                )
                crop_h, crop_w = y1 - y0, x1 - x0
                if crop_h < 8 or crop_w < 8:
                    img_masks.append(orig_m)
                    img_areas.append(int(orig_m.sum()))
                    total_retained_fallback += 1
                    continue

                ch0 = cv2.resize(two_ch[0, y0:y1, x0:x1], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                ch1 = cv2.resize(two_ch[1, y0:y1, x0:x1], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                rx0 = max(0, min(crop_w - 1, int(bbox[0] - x0)))
                ry0 = max(0, min(crop_h - 1, int(bbox[1] - y0)))
                rx1 = max(rx0 + 1, min(crop_w, int(bbox[0] + bbox[2] - x0)))
                ry1 = max(ry0 + 1, min(crop_h, int(bbox[1] + bbox[3] - y0)))

                box_native = np.zeros((crop_h, crop_w), dtype=np.float32)
                box_native[ry0:ry1, rx0:rx1] = 1.0
                ch2 = cv2.resize(box_native, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                inp_np = np.stack([ch0, ch1, ch2], axis=0)[None, ...].astype(np.float32)

                if use_tta:
                    inp_h = np.ascontiguousarray(np.flip(inp_np, axis=3))
                    inp_v = np.ascontiguousarray(np.flip(inp_np, axis=2))
                    inp_hv = np.ascontiguousarray(np.flip(inp_np, axis=(2, 3)))
                    batch_inp = np.concatenate([inp_np, inp_h, inp_v, inp_hv], axis=0)
                    inp_t = torch.from_numpy(batch_inp).to(device)

                    with torch.cuda.amp.autocast():
                        logits = model(inp_t)
                        probs = torch.sigmoid(logits)[:, 0].float().cpu().numpy()

                    p0 = probs[0]
                    p1 = np.ascontiguousarray(np.flip(probs[1], axis=1))
                    p2 = np.ascontiguousarray(np.flip(probs[2], axis=0))
                    p3 = np.ascontiguousarray(np.flip(probs[3], axis=(0, 1)))

                    pred_prob = (p0 + p1 + p2 + p3) * 0.25
                else:
                    inp_t = torch.from_numpy(inp_np).to(device)
                    with torch.cuda.amp.autocast():
                        logits = model(inp_t)
                        pred_prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

                pred_prob_native = cv2.resize(pred_prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

                if blend_weight > 0.0:
                    orig_crop = orig_m[y0:y1, x0:x1].astype(np.float32)
                    blended_prob = (1.0 - blend_weight) * pred_prob_native + blend_weight * orig_crop
                    refined_crop_bin = (blended_prob >= threshold).astype(np.uint8)
                else:
                    refined_crop_bin = (pred_prob_native >= threshold).astype(np.uint8)

                cand_full = np.zeros((2048, 2048), dtype=np.uint8)
                cand_full[y0:y1, x0:x1] = refined_crop_bin

                # Quality checks: must have reasonable area and not drift far from box
                cand_area = int(cand_full.sum())
                orig_area = int(orig_m.sum())

                # If prediction is degenerate (< 40 px or < 15% original area), fall back to original
                if cand_area < 40 or cand_area < (orig_area * 0.15):
                    img_masks.append(orig_m)
                    img_areas.append(orig_area)
                    total_retained_fallback += 1
                else:
                    img_masks.append(cand_full)
                    img_areas.append(cand_area)
                    total_upgraded += 1

            # Resolve overlaps within image to guarantee 0 collisions
            resolved_masks = resolve_overlaps_greedy(img_masks, img_areas)

            for fid, m in zip(img_fids, resolved_masks):
                refined_rows.append({
                    "filament_id": fid,
                    "segmentation_rle": rle_encode(m),
                })

            if img_idx % 25 == 0 or img_idx == total_images:
                dt = time.time() - t0
                pct = img_idx / total_images * 100
                speed = img_idx / dt
                print(f"Processed {img_idx}/{total_images} images ({pct:.1f}%) in {dt:.1f}s ({speed:.1f} img/s)... [Upgraded: {total_upgraded}, Fallback: {total_retained_fallback}]", flush=True)

    out_df = pd.DataFrame(refined_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("=" * 80, flush=True)
    print(f"INFERENCE COMPLETE -> {output_csv}", flush=True)
    print(f"Total Instances: {len(out_df):,} | Upgraded: {total_upgraded:,} ({total_upgraded/len(out_df)*100:.1f}%) | Fallback: {total_retained_fallback:,}", flush=True)
    print(f"Elapsed Time:    {time.time()-t0:.1f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/checkpoints/crop_unet_efficientnet_b4_512_cldice_best.pt")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/submissions/submission_champion_cldice_refined_045.csv")
    parser.add_argument("--device_id", type=str, default="1")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--context", type=float, default=1.6)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--tta", action="store_true", default=False)
    args = parser.parse_args()

    run_inference(
        baseline_csv=args.baseline_csv,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        device_id=args.device_id,
        threshold=args.threshold,
        blend_weight=args.blend,
        context=args.context,
        crop_size=args.crop_size,
        use_tta=args.tta,
    )
