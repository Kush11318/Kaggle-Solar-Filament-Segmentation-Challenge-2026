"""
Stage 2 Training: CropUNet on Dynamic Native Square Crops.
Trains ResNet-18 U-Net on 256x256 filament-centered crops using CombinedDiceBCEclDiceLoss.
Decouples fine morphological segmentation from global memory constraints.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.paths import get_dataset_paths, get_split_path, get_project_root
from src.parser import MAGFiLOParser
from src.disk import DiskCache
from src.dataset_crops import FilamentCropDataset
from src.losses import CombinedDiceBCEclDiceLoss


def build_crop_samples(parser: MAGFiLOParser, record_ids: list):
    """Extracts instance-level crop metadata from COCO annotations."""
    samples = []
    for r in record_ids:
        anns = parser.get_annotations_for_record(r)
        fname = parser.get_source_filename(r)
        for a in anns:
            if a.get("area", 0) < 60:
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


def compute_crop_dice_iou(pred_probs: torch.Tensor, target_masks: torch.Tensor, thresh: float = 0.5):
    """Computes Dice and IoU on thresholded batch."""
    pred_bin = (pred_probs >= thresh).float()
    inter = (pred_bin * target_masks).sum(dim=(1, 2, 3))
    total = pred_bin.sum(dim=(1, 2, 3)) + target_masks.sum(dim=(1, 2, 3))
    union = total - inter

    dice = ((2.0 * inter + 1e-5) / (total + 1e-5)).mean().item()
    iou = ((inter + 1e-5) / (union + 1e-5)).mean().item()
    return dice, iou


def train_crop_unet(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    encoder_name: str = "resnet18",
    device_id: str = "0",
    output_path: str = None,
):
    proj_root = get_project_root()
    if output_path is None:
        output_path = os.path.join(proj_root, "checkpoints", "crop_unet_resnet18_best.pt")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING CROPUNET (STAGE 2 REFINER) ===")
    print(f"Device: {device}")
    print(f"Encoder: {encoder_name} | Epochs: {epochs} | Batch Size: {batch_size} | Initial LR: {lr}")
    print(f"Target Checkpoint: {output_path}")

    # Load dataset split and parser
    json_path, train_img_dir, _ = get_dataset_paths()
    split_path = get_split_path()
    parser = MAGFiLOParser(json_path, train_img_dir)

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_recs = split_data["train_record_ids"]
    val_recs = split_data["val_record_ids"]

    # Filter physical filenames to ensure leak-free validation
    val_filenames = set(parser.get_source_filename(r) for r in val_recs)
    clean_train_recs = [r for r in train_recs if parser.get_source_filename(r) not in val_filenames]

    train_samples = build_crop_samples(parser, clean_train_recs)
    val_samples = build_crop_samples(parser, val_recs)
    print(f"Loaded {len(train_samples):,} training crops and {len(val_samples):,} validation crops.")

    # Load disk cache
    cache_path = os.path.join(proj_root, "checkpoints", "disk_cache.json")
    disk_cache = DiskCache(cache_path) if os.path.exists(cache_path) else None

    # Create Datasets & DataLoaders
    train_ds = FilamentCropDataset(train_samples, train_img_dir, target_size=256, context=1.8, augment=True, disk_cache=disk_cache)
    val_ds = FilamentCropDataset(val_samples, train_img_dir, target_size=256, context=1.8, augment=False, disk_cache=disk_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=2,
        classes=1,
        activation=None,
    ).to(device)

    criterion = CombinedDiceBCEclDiceLoss(alpha=0.15, k_iterations=8)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_dice = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)

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

        train_loss /= max(1, n_train_batches)
        scheduler.step()

        # Validation Loop
        model.eval()
        val_dices, val_ious = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)

                d, i = compute_crop_dice_iou(probs, masks, thresh=0.50)
                val_dices.append(d)
                val_ious.append(i)

        mean_val_dice = float(np.mean(val_dices))
        mean_val_iou = float(np.mean(val_ious))
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Dice: {mean_val_dice:.4f} | "
            f"Val IoU: {mean_val_iou:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": mean_val_dice,
            "val_iou": mean_val_iou,
        })

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": mean_val_dice,
                "val_iou": mean_val_iou,
                "encoder_name": encoder_name,
            }, output_path)
            print(f"  >>> Best Checkpoint Saved! New Best Val Dice: {best_val_dice:.4f}")

    print(f"\nTraining Complete! Peak Val Dice: {best_val_dice:.4f}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 CropUNet Refiner")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    train_crop_unet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_name=args.encoder,
        device_id=args.device,
        output_path=args.out,
    )
