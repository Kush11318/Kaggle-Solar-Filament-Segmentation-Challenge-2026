"""
Official evaluation metrics module for Solar Filament Segmentation Challenge 2026.
Reproduces the exact official competition implementation from:
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
Computes Panoptic Quality (PQ), Dice, IoU, fragmentation (1:n), and merging (n:1).
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from collections import Counter


def rles_to_layers(rles: List[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Decode a list of compressed COCO RLE strings into an (n_masks, H, W) binary mask stack.
    Matches official implementation.
    """
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    if masks.ndim == 2:
        masks = masks[:, :, np.newaxis]
    return np.ascontiguousarray(masks.transpose(2, 0, 1).astype(np.float32))


def get_overlap_matrices(
    gt_layers: torch.Tensor,
    pred_layers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise IoU and Dice scores between GT and predicted mask layers for one image.
    Args:
        gt_layers: (n_gt, H, W)
        pred_layers: (n_pred, H, W)
    Returns:
        iou_matrix: (n_gt, n_pred)
        dice_matrix: (n_gt, n_pred)
    """
    n_gt = gt_layers.shape[0]
    n_pred = pred_layers.shape[0]

    if n_gt == 0 or n_pred == 0:
        device = gt_layers.device if n_gt > 0 else (pred_layers.device if n_pred > 0 else torch.device("cpu"))
        return torch.zeros((n_gt, n_pred), device=device), torch.zeros((n_gt, n_pred), device=device)

    # Flatten spatial dims: (n_gt, H*W) and (n_pred, H*W)
    gt_flat = gt_layers.reshape(n_gt, -1)
    pred_flat = pred_layers.reshape(n_pred, -1)

    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1, keepdim=True)
    pred_areas = pred_flat.sum(dim=1, keepdim=True).t()

    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(
        union == 0,
        torch.tensor(0.0, device=gt_layers.device),
        intersection / union,
    )

    dice_matrix = torch.where(
        (gt_areas + pred_areas) == 0,
        torch.tensor(0.0, device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )

    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """A prediction is a false positive if its column sum in hit_matrix is zero."""
    if hit_matrix.shape[1] == 0:
        return 0
    gt_matches_per_pred = hit_matrix.sum(dim=0)
    return int((gt_matches_per_pred == 0).sum().item())


def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """A GT filament is a false negative if its row sum in hit_matrix is zero."""
    if hit_matrix.shape[0] == 0:
        return 0
    pred_matches_per_gt = hit_matrix.sum(dim=1)
    return int((pred_matches_per_gt == 0).sum().item())


def process_entry(
    annotator_image: str,
    gt_annotator_image_ids: pd.Series,
    pred_image_ids: pd.Series,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = 2048,
    width: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Matches one annotator-image entry between prediction and GT and computes overlap matrices.
    """
    image_id = annotator_image.split("-", maxsplit=1)[1] if "-" in annotator_image else annotator_image

    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles, height, width))

    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles, height, width))

    n_gt = len(gt_rles)
    n_pred = len(pred_rles)

    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, n_gt, n_pred


def get_overlap_df(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    height: int = 2048,
    width: int = 2048,
) -> pd.DataFrame:
    """Builds a per-annotator-image table of GT vs prediction overlap matrices."""
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0] if not pred_df.empty else pd.Series([], dtype=str)

    annotator_images = gt_annotator_image_ids.unique()
    overlap_df = pd.DataFrame({"annotator_image": annotator_images})

    results = []
    for ann_img in annotator_images:
        res = process_entry(ann_img, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df, height, width)
        results.append(res)

    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df["n_pred"] = zip(*results)
    return overlap_df


def compute_official_metrics(
    overlap_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ),
    total TP, FP, FN, mean Dice (non-zero pairs), fragmentation (1:n) and merging (n:1) counts.
    """
    tp_iou_scores: List[float] = []
    fp_count = 0
    fn_count = 0

    all_dice_scores: List[float] = []
    all_iou_scores: List[float] = []

    # Fragmentation and merging tracking (IoU > 0)
    frag_1_to_many = 0
    merge_many_to_1 = 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        dice_matrix = row.dice_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

        # Collect positive scores for distribution
        pos_iou = iou_matrix[iou_matrix > 0].tolist()
        pos_dice = dice_matrix[dice_matrix > 0].tolist()
        all_iou_scores.extend(pos_iou)
        all_dice_scores.extend(pos_dice)

        # 1-to-many (GT filament hits multiple predictions)
        frag_hit = (iou_matrix > 0).sum(dim=1)
        frag_1_to_many += int((frag_hit > 1).sum().item())

        # many-to-1 (predicted filament hits multiple GTs)
        merge_hit = (iou_matrix > 0).sum(dim=0)
        merge_many_to_1 += int((merge_hit > 1).sum().item())

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq_score = sum(tp_iou_scores) / denominator if denominator > 0 else 0.0
    sq_score = sum(tp_iou_scores) / tp_count if tp_count > 0 else 0.0
    rq_score = tp_count / denominator if denominator > 0 else 0.0
    mean_dice = float(np.mean(all_dice_scores)) if all_dice_scores else 0.0
    mean_iou = float(np.mean(all_iou_scores)) if all_iou_scores else 0.0

    return {
        "pq": pq_score,
        "sq": sq_score,
        "rq": rq_score,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "mean_dice_positive": mean_dice,
        "mean_iou_positive": mean_iou,
        "fragmented_1_to_many": frag_1_to_many,
        "merged_many_to_1": merge_many_to_1,
        "total_gt": sum(overlap_df["n_gt"]),
        "total_pred": sum(overlap_df["n_pred"]),
    }
