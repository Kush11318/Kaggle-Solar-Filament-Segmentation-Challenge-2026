"""
Dense Centerline-Watershed Instance Extraction Module.
Eliminates rectangular bounding boxes, anchors, and NMS entirely.
Extracts distinct curvilinear solar filament instances directly from continuous
Dense Multi-Task predictions (Filament Mask + Medial Axis Skeleton).

Algorithm:
1. Threshold medial axis centerline probability (Head B).
2. Morphological thinning / skeletonization to ensure continuous 1-pixel spine seeds.
3. Connected components on spines assigns unique integer IDs to each magnetic neutral line.
4. Marker-Controlled Watershed expands spine seeds outward into the full filament body (Head A).
5. Adjacent or intertwined filaments are cleanly split at their boundary saddle with zero box collisions.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from skimage.morphology import skeletonize
from pycocotools import mask as mask_util


def extract_instances_via_watershed(
    prob_mask: np.ndarray,
    prob_spine: np.ndarray,
    mask_thresh: float = 0.50,
    spine_thresh: float = 0.45,
    min_spine_len: int = 12,
    min_area: int = 150,
    disk_mask: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    Extracts individual filament instance masks using Centerline Marker-Controlled Watershed.
    Args:
        prob_mask: (H, W) float32 probability of filament body
        prob_spine: (H, W) float32 probability of medial axis spine
        mask_thresh: threshold for full filament body
        spine_thresh: threshold for spine seeds
        min_spine_len: minimum pixel length for a valid magnetic spine seed
        min_area: minimum area for a valid filament instance
        disk_mask: optional (H, W) boolean mask of solar disk
    Returns:
        List of (binary_mask, confidence_score) tuples
    """
    h, w = prob_mask.shape

    # 1. Binary segmentation of filament body
    bin_mask = (prob_mask >= mask_thresh)
    if disk_mask is not None:
        bin_mask &= (disk_mask > 0)

    if bin_mask.sum() < min_area:
        return []

    # 2. Binary segmentation and skeletonization of central spine
    bin_spine = (prob_spine >= spine_thresh) & bin_mask
    if bin_spine.sum() > 0:
        skeleton = skeletonize(bin_spine)
    else:
        skeleton = np.zeros_like(bin_mask, dtype=bool)

    # 3. Connected components on spine skeleton to find distinct filament seeds
    num_spines, spine_labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)

    # Filter out tiny spurious spine fragments
    valid_seed_count = 0
    clean_markers = np.zeros((h, w), dtype=np.int32)

    for s_idx in range(1, num_spines):
        seed_mask = (spine_labels == s_idx)
        if seed_mask.sum() >= min_spine_len:
            valid_seed_count += 1
            clean_markers[seed_mask] = valid_seed_count

    # Fallback: if no valid spine seeds were detected, use connected components on full mask
    if valid_seed_count == 0:
        num_cc, cc_labels = cv2.connectedComponents(bin_mask.astype(np.uint8), connectivity=8)
        instances = []
        for c_idx in range(1, num_cc):
            comp = (cc_labels == c_idx)
            area = comp.sum()
            if area >= min_area:
                score = float(prob_mask[comp].mean())
                instances.append((comp, score))
        return instances

    # 4. Marker-Controlled Watershed
    # Distance transform creates an energy valley along the center of the filament
    dist_transform = cv2.distanceTransform(bin_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Prepare pseudo-RGB gradient image for cv2.watershed
    dist_norm = dist_transform / max(1e-3, float(dist_transform.max()))
    surface = np.clip(255.0 * (1.0 - dist_norm), 0, 255).astype(np.uint8)
    surface_3ch = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

    # Markers: background is 0, seeds are 1..K
    markers = clean_markers.copy()

    # OpenCV watershed requires background marked or unmarked
    # Set pixels outside bin_mask to -1 (boundary) or keep unsegmented
    cv2.watershed(surface_3ch, markers)

    # 5. Extract separated instances
    instances = []
    for m_idx in range(1, valid_seed_count + 1):
        inst_mask = (markers == m_idx) & bin_mask
        area = inst_mask.sum()
        if area >= min_area:
            # Score is the combination of mask probability and spine confidence
            mask_conf = float(prob_mask[inst_mask].mean())
            spine_conf = float(prob_spine[inst_mask].max())
            fused_score = 0.60 * mask_conf + 0.40 * spine_conf
            instances.append((inst_mask, fused_score))

    return instances


def encode_instances_to_rle(
    instances: List[Tuple[np.ndarray, float]],
    stem: str,
) -> List[Dict[str, str]]:
    """
    Encodes binary instance masks into official Kaggle submission format rows.
    """
    rows = []
    for idx, (mask, score) in enumerate(instances, start=1):
        rle_dict = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle_str = rle_dict["counts"].decode("utf-8")
        rows.append({
            "filament_id": f"{stem}_{idx}",
            "segmentation_rle": rle_str,
        })
    return rows
