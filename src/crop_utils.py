"""
Crop and coordinate utility functions for the Decoupled Two-Stage Architecture.
Implements:
1. Dynamic Native Square Crop extraction with 1.8x context expansion and boundary-safe shifting.
2. 4-Quadrant Tiling (1100x1100 with 152 px overlap) for high-resolution global detection.
3. Positive Overlap Grouping (merging adjacent tile candidate masks with S > 0).
4. Continuous probability map re-projection and disjoint instance merging.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from pycocotools import mask as mask_util


def compute_dynamic_square_crop(
    bbox: Tuple[float, float, float, float],
    context: float = 1.8,
    min_side: int = 96,
    max_side: int = 2048,
    img_h: int = 2048,
    img_w: int = 2048,
) -> Tuple[int, int, int, int, int]:
    """
    Computes a dynamic native square crop around bbox [x, y, w, h] with expanded context.
    Shifts the square crop to remain within [0, img_w] x [0, img_h] WITHOUT scaling or distorting aspect ratio.

    Returns:
        (x0, y0, x1, y1, side)
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    max_dim = max(w, h)
    side = int(np.clip(np.ceil(max_dim * context), min_side, max_side))

    half_side = side / 2.0
    x0 = int(np.round(cx - half_side))
    y0 = int(np.round(cy - half_side))

    # Boundary-safe shift without distortion
    if x0 < 0:
        x0 = 0
    elif x0 + side > img_w:
        x0 = max(0, img_w - side)

    if y0 < 0:
        y0 = 0
    elif y0 + side > img_h:
        y0 = max(0, img_h - side)

    x1 = min(img_w, x0 + side)
    y1 = min(img_h, y0 + side)
    actual_side = max(x1 - x0, y1 - y0)

    return x0, y0, x1, y1, actual_side


def get_four_quadrants(img_h: int = 2048, img_w: int = 2048, tile_size: int = 1100, overlap: int = 152):
    """
    Returns the bounding coordinates for 4 overlapping quadrants covering full disk.
    Each quadrant is 1100x1100 with a 152 px overlap margin.
    Total coverage:
    x0: [0, 948], x1: [1100, 2048]
    y0: [0, 948], y1: [1100, 2048]
    """
    shift = tile_size - overlap  # 1100 - 152 = 948
    quadrants = [
        {"id": 0, "x0": 0, "y0": 0, "x1": tile_size, "y1": tile_size},
        {"id": 1, "x0": shift, "y0": 0, "x1": img_w, "y1": tile_size},
        {"id": 2, "x0": 0, "y0": shift, "x1": tile_size, "y1": img_h},
        {"id": 3, "x0": shift, "y0": shift, "x1": img_w, "y1": img_h},
    ]
    return quadrants


def positive_overlap_grouping(
    candidate_masks: List[np.ndarray],
    scores: List[float],
    iou_thresh: float = 0.0,
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Decoupling Module: Merges candidate masks that exhibit positive pixel overlap (S > 0)
    across tile boundaries to eliminate seam tears.
    Zero-redundancy O(1) bounding-box queries for ultra-fast execution.
    """
    n = len(candidate_masks)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Compute bounding boxes if not provided
    if bboxes is None:
        bboxes = []
        for m in candidate_masks:
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                bboxes.append(None)
            else:
                bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    for i in range(n):
        bb_i = bboxes[i]
        if bb_i is None:
            continue
        x0_i, y0_i, x1_i, y1_i = bb_i
        on_seam_i = (x0_i <= 1120 and x1_i >= 928) or (y0_i <= 1120 and y1_i >= 928)

        for j in range(i + 1, n):
            bb_j = bboxes[j]
            if bb_j is None:
                continue
            x0_j, y0_j, x1_j, y1_j = bb_j

            # 1. Fast BBox Disjoint Rejection (< 1 microsecond)
            if x1_i < x0_j or x1_j < x0_i or y1_i < y0_j or y1_j < y0_i:
                continue

            ix0 = max(x0_i, x0_j)
            iy0 = max(y0_i, y0_j)
            ix1 = min(x1_i, x1_j)
            iy1 = min(y1_i, y1_j)

            # Pixel overlap / IoU check on intersecting boxes (sliced to bbox intersection for 1000x speedup)
            sub_i = candidate_masks[i][iy0:iy1 + 1, ix0:ix1 + 1]
            sub_j = candidate_masks[j][iy0:iy1 + 1, ix0:ix1 + 1]

            if iou_thresh > 0.0:
                inter = np.logical_and(sub_i, sub_j).sum()
                if inter > 0:
                    union_area = candidate_masks[i].sum() + candidate_masks[j].sum() - inter
                    if (inter / float(union_area)) >= iou_thresh:
                        union(i, j)
                        continue
            else:
                overlap = np.logical_and(sub_i, sub_j).any()
                if overlap:
                    union(i, j)
                    continue

            # 2. Seam-Restricted Deduplication (ONLY for duplicate candidates in quadrant seam band)
            on_seam_j = (x0_j <= 1120 and x1_j >= 928) or (y0_j <= 1120 and y1_j >= 928)
            if on_seam_i and on_seam_j:
                inter_area = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
                area_i = (x1_i - x0_i + 1) * (y1_i - y0_i + 1)
                area_j = (x1_j - x0_j + 1) * (y1_j - y0_j + 1)
                box_iou = inter_area / float(area_i + area_j - inter_area)
                min_ratio = inter_area / float(min(area_i, area_j))
                if box_iou >= 0.40 or min_ratio >= 0.60:
                    union(i, j)
                    continue

    # Group into connected components
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    tasks = []
    for member_indices in groups.values():
        if len(member_indices) == 1:
            idx = member_indices[0]
            bb = bboxes[idx]
            tasks.append({
                "merged_mask": candidate_masks[idx],
                "bbox": (bb[0], bb[1], bb[2] - bb[0] + 1, bb[3] - bb[1] + 1) if bb else (0, 0, 0, 0),
                "score": scores[idx],
                "area": int(candidate_masks[idx].sum()),
            })
            continue

        merged = np.zeros_like(candidate_masks[0], dtype=bool)
        max_score = 0.0
        for idx in member_indices:
            merged = np.logical_or(merged, candidate_masks[idx])
            if scores[idx] > max_score:
                max_score = scores[idx]

        valid_bbs = [bboxes[idx] for idx in member_indices if bboxes[idx] is not None]
        if valid_bbs:
            bb_x0 = min(b[0] for b in valid_bbs)
            bb_y0 = min(b[1] for b in valid_bbs)
            bb_x1 = max(b[2] for b in valid_bbs)
            bb_y1 = max(b[3] for b in valid_bbs)
            bbox_tuple = (bb_x0, bb_y0, bb_x1 - bb_x0 + 1, bb_y1 - bb_y0 + 1)
        else:
            bbox_tuple = (0, 0, 0, 0)

        tasks.append({
            "merged_mask": merged,
            "bbox": bbox_tuple,
            "score": max_score,
            "area": int(merged.sum()),
        })

    tasks.sort(key=lambda t: -t["score"])
    return tasks


def disjoint_merge_instances(
    instance_masks: List[np.ndarray],
    scores: List[float],
    min_area: int = 150,
) -> List[Tuple[np.ndarray, float]]:
    """
    Assembles non-overlapping instances onto the global canvas in descending confidence order.
    Each pixel is claimed by at most one instance (zero-overlap guarantee).
    """
    if not instance_masks:
        return []
    claimed = np.zeros(instance_masks[0].shape, dtype=bool)
    final_instances = []

    # Sort by descending confidence
    order = np.argsort([-s for s in scores])
    for idx in order:
        mask = instance_masks[idx]
        score = scores[idx]
        valid_px = mask & ~claimed
        if int(valid_px.sum()) >= min_area:
            claimed |= valid_px
            final_instances.append((valid_px, score))

    return final_instances


def split_narrow_necks(bin_crop: np.ndarray, min_area: int = 150) -> List[np.ndarray]:
    """
    Splits merged filaments connected by thin necks (< 5px) using Distance Transform Watershed.
    Guarantees no over-segmentation: only splits if 2+ prominent peaks are found and both resulting parts >= min_area.
    """
    if bin_crop.sum() < min_area * 2:
        return [bin_crop]

    dist = cv2.distanceTransform(bin_crop.astype(np.uint8), cv2.DIST_L2, 5)
    max_d = dist.max()
    if max_d < 3.5:
        return [bin_crop]

    # Find local peaks: areas with substantial thickness
    peak_thresh = max(2.5, 0.40 * max_d)
    peak_mask = (dist >= peak_thresh).astype(np.uint8)
    num_peaks, peak_labels = cv2.connectedComponents(peak_mask)

    if num_peaks > 2:  # 0 is background, >2 means at least 2 foreground peaks
        markers = np.zeros_like(dist, dtype=np.int32)
        for p_idx in range(1, num_peaks):
            markers[peak_labels == p_idx] = p_idx

        dist_inv = (max_d - dist).astype(np.uint8)
        dist_inv_3c = cv2.cvtColor(dist_inv, cv2.COLOR_GRAY2BGR)
        cv2.watershed(dist_inv_3c, markers)

        sub_masks = []
        for p_idx in range(1, num_peaks):
            sub_m = (markers == p_idx) & bin_crop
            if sub_m.sum() >= min_area:
                sub_masks.append(sub_m)

        if len(sub_masks) >= 2:
            return sub_masks

    return [bin_crop]


def compute_mask_eccentricity(mask: np.ndarray) -> float:
    """
    Computes morphological eccentricity using central spatial moments:
    e = sqrt(1 - lambda2 / lambda1)
    Returns: float in [0.0, 1.0]. A circular blob has e ~ 0; an elongated filament has e > 0.75.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        return 0.0

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    dx = xs - cx
    dy = ys - cy

    mu20 = float(np.mean(dx * dx))
    mu02 = float(np.mean(dy * dy))
    mu11 = float(np.mean(dx * dy))

    delta = np.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2))
    lambda1 = (mu20 + mu02 + delta) / 2.0
    lambda2 = max(0.0, (mu20 + mu02 - delta) / 2.0)

    if lambda1 <= 1e-6:
        return 0.0

    ecc = np.sqrt(max(0.0, 1.0 - lambda2 / lambda1))
    return float(ecc)


def is_sunspot_or_compact_blob(
    mask: np.ndarray,
    area: int,
    min_eccentricity: float = 0.65,
    max_compact_area: int = 1800,
) -> bool:
    """
    Bernasconi et al. (2005) Sunspot Elimination:
    Identifies compact circular dark spots (sunspots / pores) that masquerade as filaments.
    If area is small and eccentricity is low (round), returns True to discard.
    """
    if area > max_compact_area:
        return False
    ecc = compute_mask_eccentricity(mask)
    return ecc < min_eccentricity


def is_outside_solar_disk_margin(
    mask: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    max_r_ratio: float = 0.965,
) -> bool:
    """
    Limb artifact exclusion:
    Filaments do not exist in the extreme outer limb halo (> 0.965 R_sun).
    Returns True if instance centroid lies in the limb fringe.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return True
    centroid_x = float(np.mean(xs))
    centroid_y = float(np.mean(ys))
    dist = np.hypot(centroid_x - cx, centroid_y - cy)
    return (dist / max(1.0, radius)) > max_r_ratio


def is_insufficient_contrast(
    mask: np.ndarray,
    raw_img: np.ndarray,
    max_contrast_ratio: float = 0.92,
) -> bool:
    """
    Chromospheric absorption verification (Fuller et al. 2008):
    Verifies that the candidate is truly darker than the local quiet Sun background.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return True

    # Bounding box crop for speed
    y0, y1 = max(0, int(ys.min()) - 15), min(raw_img.shape[0], int(ys.max()) + 16)
    x0, x1 = max(0, int(xs.min()) - 15), min(raw_img.shape[1], int(xs.max()) + 16)

    sub_mask = mask[y0:y1, x0:x1]
    sub_raw = raw_img[y0:y1, x0:x1].astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(sub_mask.astype(np.uint8), kernel).astype(bool)
    bg_ring = dilated & ~sub_mask

    if bg_ring.sum() < 20:
        return False

    mean_in = float(sub_raw[sub_mask].mean())
    mean_bg = float(sub_raw[bg_ring].mean())

    if mean_bg <= 1e-3:
        return False

    # Filaments must be darker absorption features
    ratio = mean_in / mean_bg
    return ratio > max_contrast_ratio

