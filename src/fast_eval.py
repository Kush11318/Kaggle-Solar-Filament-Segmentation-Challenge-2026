import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from typing import Dict, Any, List

def fast_compute_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_thresh: float = 0.5) -> Dict[str, Any]:
    """
    Lightning-fast, exact official Panoptic Quality computation using C-accelerated pycocotools.
    Matches the official self-evaluation notebook logic identically.
    """
    gt_ann_imgs = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_imgs = pred_df["filament_id"].str.split("_", n=1).str[0] if not pred_df.empty else pd.Series([], dtype=str)

    unique_ann_imgs = gt_ann_imgs.unique()
    tp_ious = []
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    for ann_img in unique_ann_imgs:
        stem = ann_img.split("-", 1)[1] if "-" in ann_img else ann_img
        gt_rles = gt_df.loc[gt_ann_imgs == ann_img, "segmentation_rle"].tolist()
        pred_rles = pred_df.loc[pred_imgs == stem, "segmentation_rle"].tolist() if not pred_df.empty else []

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)
        total_gt += n_gt
        total_pred += n_pred

        if n_gt == 0:
            total_fp += n_pred
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue

        rles_g = [{"size": [2048, 2048], "counts": r} for r in gt_rles]
        rles_p = [{"size": [2048, 2048], "counts": r} for r in pred_rles]

        iou_mat = mask_util.iou(rles_g, rles_p, [0] * n_pred)
        hit_matrix = (iou_mat >= iou_thresh)

        tp_ious.extend(iou_mat[hit_matrix].tolist())
        total_fp += int((hit_matrix.sum(axis=0) == 0).sum())
        total_fn += int((hit_matrix.sum(axis=1) == 0).sum())

    tp_count = len(tp_ious)
    denom = tp_count + 0.5 * total_fp + 0.5 * total_fn
    pq = sum(tp_ious) / denom if denom > 0 else 0.0
    sq = sum(tp_ious) / tp_count if tp_count > 0 else 0.0
    rq = tp_count / denom if denom > 0 else 0.0

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": tp_count,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }

