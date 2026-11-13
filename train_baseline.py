"""
Baseline U-Net Training and Official PQ Validation Pipeline.
Supports both Kaggle GPU execution and local environments.
Evaluates validation partition using the official Panoptic Quality (PQ) metric.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.parser import MAGFiLOParser
from src.dataset import SolarFilamentDataset, GridTiler
from src.models import UNet
from src.losses import BCEDiceLoss
from src.inference import predict_full_image
from src.metrics import get_overlap_df, compute_official_metrics
from src.paths import get_dataset_paths, get_split_path, get_project_root
from pycocotools import mask as mask_util


def run_validation(
    model: torch.nn.Module,
    parser: MAGFiLOParser,
    val_record_ids: list[str],
    tiler: GridTiler,
    device: torch.device,
    threshold: float = 0.5,
    min_area: int = 40,
    max_eval_images: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates the model on validation partition and computes official PQ, SQ, RQ, Dice, IoU.
    """
    model.eval()
    eval_recs = val_record_ids if max_eval_images is None else val_record_ids[:max_eval_images]

    # Map records to unique physical filenames
    records_by_file = {}
    for r in eval_recs:
        fn = parser.get_source_filename(r)
        if fn not in records_by_file:
            records_by_file[fn] = []
        records_by_file[fn].append(r)

    # 1. Build Ground Truth DataFrame for evaluation records
    gt_df = parser.build_gt_dataframe(eval_recs)

    # 2. Run inference on each unique physical image
    pred_rows = []
    inference_times = []

    for fn in records_by_file:
        img_path = parser.get_image_path(fn)
        # Extract base image id (strip extension)
        base_img_id = os.path.splitext(fn)[0]

        t0 = time.time()
        _, inst_masks = predict_full_image(
            model=model,
            image_path=img_path,
            tiler=tiler,
            device=device,
            threshold=threshold,
            min_area=min_area,
            batch_size=8 if device.type == "cuda" else 2,
        )
        inference_times.append(time.time() - t0)

        for idx, inst_m in enumerate(inst_masks, start=1):
            f_mask = np.asfortranarray(inst_m)
            rle_dict = mask_util.encode(f_mask)
            rle_str = rle_dict["counts"].decode("utf-8")
            fil_id = f"{base_img_id}_{idx}"
            pred_rows.append({"filament_id": fil_id, "segmentation_rle": rle_str})

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])

    # 3. Compute official competition metrics
    overlap_df = get_overlap_df(gt_df, pred_df)
    metrics = compute_official_metrics(overlap_df)
    metrics["mean_inference_time_per_image_sec"] = float(np.mean(inference_times)) if inference_times else 0.0

    return metrics


def train(config_path: Optional[str] = None, quick_mode: bool = False):
    proj_root = get_project_root()
    json_path, train_img_dir, test_img_dir = get_dataset_paths()
    split_manifest_path = get_split_path()

    cfg = {
        "experiment_name": "baseline_unet_512",
        "dataset": {
            "json_path": json_path,
            "train_img_dir": train_img_dir,
            "test_img_dir": test_img_dir,
            "split_manifest_path": split_manifest_path
        },
        "model": {"in_channels": 1, "out_channels": 1, "base_channels": 32},
        "training": {
            "patch_size": 512, "pos_ratio": 0.75, "batch_size": 8, "lr": 0.001,
            "weight_decay": 0.0001, "epochs": 15, "seed": 42
        },
        "inference": {"tile_patch_size": 512, "tile_overlap": 128, "threshold": 0.5, "min_area": 40},
        "checkpoints": {"dir": os.path.join(proj_root, "checkpoints"), "save_best_metric": "val_pq"}
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg.update(json.load(f))
        cfg["dataset"]["json_path"] = json_path
        cfg["dataset"]["train_img_dir"] = train_img_dir
        cfg["dataset"]["test_img_dir"] = test_img_dir
        cfg["dataset"]["split_manifest_path"] = split_manifest_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    parser = MAGFiLOParser(cfg["dataset"]["json_path"], cfg["dataset"]["train_img_dir"])
    with open(cfg["dataset"]["split_manifest_path"], "r") as f:
        split = json.load(f)

    train_recs = split["train_record_ids"]
    val_recs = split["val_record_ids"]

    if quick_mode:
        train_recs = train_recs[:16]
        val_recs = val_recs[:4]
        cfg["training"]["epochs"] = 2
        print(f"*** QUICK MODE ENABLED: {len(train_recs)} train recs, {len(val_recs)} val recs, {cfg['training']['epochs']} epochs ***")

    # Dataset & Loader
    train_ds = SolarFilamentDataset(
        parser=parser,
        record_ids=train_recs,
        patch_size=cfg["training"]["patch_size"],
        pos_ratio=cfg["training"]["pos_ratio"],
        is_train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"] if device.type == "cuda" else 2,
        shuffle=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model & Optimization
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    tiler = GridTiler(
        img_size=2048,
        patch_size=cfg["inference"]["tile_patch_size"],
        overlap=cfg["inference"]["tile_overlap"],
    )

    ckpt_dir = cfg["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_pq = -1.0
    history = []

    print("\nStarting training loop...")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        scheduler.step()
        mean_train_loss = float(np.mean(train_losses))
        train_duration = time.time() - epoch_start

        # Periodic validation
        val_start = time.time()
        val_metrics = run_validation(
            model=model,
            parser=parser,
            val_record_ids=val_recs,
            tiler=tiler,
            device=device,
            threshold=cfg["inference"]["threshold"],
            min_area=cfg["inference"]["min_area"],
            max_eval_images=4 if quick_mode else 20,  # Evaluate subset on CPU or full on GPU
        )
        val_duration = time.time() - val_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_pq": val_metrics["pq"],
            "val_sq": val_metrics["sq"],
            "val_rq": val_metrics["rq"],
            "val_dice": val_metrics["mean_dice_positive"],
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_frag_1_to_many": val_metrics["fragmented_1_to_many"],
            "val_merge_many_to_1": val_metrics["merged_many_to_1"],
            "train_time_sec": train_duration,
            "val_time_sec": val_duration,
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{cfg['training']['epochs']:02d}] "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Val PQ: {val_metrics['pq']:.4f} (SQ: {val_metrics['sq']:.4f}, RQ: {val_metrics['rq']:.4f}) | "
            f"Dice: {val_metrics['mean_dice_positive']:.4f} | "
            f"TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, FN: {val_metrics['fn']} | "
            f"Train: {train_duration:.1f}s, Val: {val_duration:.1f}s"
        )

        # Checkpoint if best PQ
        if val_metrics["pq"] > best_val_pq:
            best_val_pq = val_metrics["pq"]
            best_ckpt_path = os.path.join(ckpt_dir, f"{cfg['experiment_name']}_best_pq.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "val_metrics": val_metrics,
            }, best_ckpt_path)
            print(f"  -> Saved new best checkpoint to: {best_ckpt_path} (Val PQ: {best_val_pq:.4f})")

    # Save complete experiment log
    log_path = os.path.join(ckpt_dir, f"{cfg['experiment_name']}_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! History logged to: {log_path}")

    # Generate competition test set submission.csv using best model checkpoint
    best_ckpt_path = os.path.join(ckpt_dir, f"{cfg['experiment_name']}_best_pq.pt")
    if os.path.exists(best_ckpt_path) and os.path.exists(test_img_dir):
        print("\nLoading best model checkpoint to generate test submission.csv...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_files = sorted(os.listdir(test_img_dir))
        print(f"Predicting on {len(test_files)} test images...")
        sub_rows = []
        for idx, tf in enumerate(test_files, start=1):
            t_path = os.path.join(test_img_dir, tf)
            base_id = os.path.splitext(tf)[0]
            _, inst_masks = predict_full_image(
                model=model,
                image_path=t_path,
                tiler=tiler,
                device=device,
                threshold=cfg["inference"]["threshold"],
                min_area=cfg["inference"]["min_area"],
                batch_size=8 if device.type == "cuda" else 2,
            )
            for m_idx, inst_m in enumerate(inst_masks, start=1):
                f_mask = np.asfortranarray(inst_m)
                rle_dict = mask_util.encode(f_mask)
                rle_str = rle_dict["counts"].decode("utf-8")
                sub_rows.append({"filament_id": f"{base_id}_{m_idx}", "segmentation_rle": rle_str})

            if idx % 30 == 0 or idx == len(test_files):
                print(f"  Processed {idx}/{len(test_files)} test images ({len(sub_rows)} instances detected so far)")

        sub_df = pd.DataFrame(sub_rows)
        sub_csv_path = os.path.join(proj_root, "submission.csv")
        sub_df.to_csv(sub_csv_path, index=False)
        print(f"Successfully generated {sub_csv_path} with {len(sub_df)} rows across {len(test_files)} images!")

    return history


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--config", type=str, default=r"d:/Kaggle/configs/baseline_config.json")
    parser_arg.add_argument("--quick", action="store_true", help="Run quick sanity check on a small subset")
    args = parser_arg.parse_args()

    train(args.config, quick_mode=args.quick)
