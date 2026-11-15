"""
Spartan Phase 2: Native High-Resolution Box-Prompted CropUNet Training with Soft-clDice
========================================================================================
Architecture:
- In-channels: 3 (Raw H-alpha, Limb-Flattened, Target Box-Prompt Gaussian)
- Target: Strictly target instance polygon (suppresses neighboring active plage)
- Loss: CombinedDiceBCEclDiceLoss (topological skeleton loss directly backpropagated)
- Resolution: Native 512x512 / 1024x1024 crops
- Optimization: AdamW + CosineAnnealingLR + Mixed Precision (AMP)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import get_dataset_paths, get_split_path, get_project_root
from src.parser import MAGFiLOParser
from src.disk import DiskCache
from src.dataset_crops_prompt import FilamentCropPromptDataset
from src.losses import CombinedDiceBCEclDiceLoss


def build_crop_samples(parser: MAGFiLOParser, record_ids: list, min_area: int = 40):
    samples = []
    for r in record_ids:
        anns = parser.get_annotations_for_record(r)
        fname = parser.get_source_filename(r)
        for a in anns:
            if a.get("area", 0) < min_area:
                continue
            x, y, w, h = a["bbox"]
            if w <= 4 or h <= 4:
                continue
            samples.append({
                "image_file": fname,
                "bbox": a["bbox"],
                "segmentation": a["segmentation"],
                "area": a.get("area", w * h),
                "source_id": r,
            })
    return samples


def compute_metrics(pred_probs: torch.Tensor, target_masks: torch.Tensor, thresh: float = 0.5):
    pred_bin = (pred_probs >= thresh).float()
    inter = (pred_bin * target_masks).sum(dim=(1, 2, 3))
    total = pred_bin.sum(dim=(1, 2, 3)) + target_masks.sum(dim=(1, 2, 3))
    union = total - inter
    dice = ((2.0 * inter + 1e-5) / (total + 1e-5)).mean().item()
    iou = ((inter + 1e-5) / (union + 1e-5)).mean().item()
    return dice, iou


def train(
    epochs: int = 20,
    batch_size: int = 16,
    crop_size: int = 512,
    lr: float = 8e-4,
    encoder_name: str = "efficientnet-b4",
    cldice_alpha: float = 0.20,
    device_id: str = "0",
    output_path: str = None,
):
    proj_root = get_project_root()
    if output_path is None:
        enc_slug = encoder_name.replace("-", "_")
        output_path = os.path.join(proj_root, "checkpoints", f"crop_unet_{enc_slug}_{crop_size}_cldice_best.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("SPARTAN RETRAINING: 3-CHANNEL BOX-PROMPTED CROPUNET WITH clDICE LOSS", flush=True)
    print(f"Device: {device} | Encoder: {encoder_name} | Size: {crop_size}x{crop_size} | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f"clDice Alpha: {cldice_alpha} | Target Checkpoint: {output_path}", flush=True)
    print("=" * 80, flush=True)

    json_path, train_img_dir, _ = get_dataset_paths()
    split_path = get_split_path()
    parser = MAGFiLOParser(json_path, train_img_dir)

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_recs = split_data["train_record_ids"]
    val_recs = split_data["val_record_ids"]

    val_filenames = set(parser.get_source_filename(r) for r in val_recs)
    clean_train_recs = [r for r in train_recs if parser.get_source_filename(r) not in val_filenames]

    train_samples = build_crop_samples(parser, clean_train_recs)
    val_samples = build_crop_samples(parser, val_recs)
    print(f"Loaded {len(train_samples):,} train crops | {len(val_samples):,} val crops.", flush=True)

    cache_path = os.path.join(proj_root, "checkpoints", "disk_cache.json")
    disk_cache = DiskCache(cache_path) if os.path.exists(cache_path) else None

    train_ds = FilamentCropPromptDataset(train_samples, train_img_dir, target_size=crop_size, context=1.6, augment=True, disk_cache=disk_cache)
    val_ds = FilamentCropPromptDataset(val_samples, train_img_dir, target_size=crop_size, context=1.6, augment=False, disk_cache=disk_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion = CombinedDiceBCEclDiceLoss(alpha=cldice_alpha, k_iterations=10)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_dice = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_dices = []
        val_ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)
                d, i = compute_metrics(probs, masks)
                val_dices.append(d)
                val_ious.append(i)

        mean_dice = float(np.mean(val_dices))
        mean_iou = float(np.mean(val_ious))
        dt = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}", flush=True)

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": mean_iou,
                "config": {"encoder": encoder_name, "crop_size": crop_size, "cldice_alpha": cldice_alpha},
            }, output_path)
            print(f"  ★ Model Saved -> {output_path} (Val Dice: {best_val_dice:.4f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--encoder", type=str, default="efficientnet-b4")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--device_id", type=str, default="0")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        encoder_name=args.encoder,
        cldice_alpha=args.alpha,
        device_id=args.device_id,
    )
