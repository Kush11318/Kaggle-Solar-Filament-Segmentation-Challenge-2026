"""
Detector-Guided Instance Fusion for Solar Filament Segmentation.

Combines high-resolution pixel-level predictions from ResNet34 2-channel U-Net
with instance-level grouping boundaries from YOLO11-seg detector.

Key principles:
1. Detections sorted in descending confidence order claim thresholded U-Net foreground pixels.
2. Detector masks are grown (dilated) by radius `grow` to encompass filament barbs and edges.
3. Each foreground pixel is claimed at most once (zero overlap guarantee).
4. Minimum claimed area filter discards weak or spurious detections.
5. Optional fallback preserves large unclaimed U-Net foreground components.
6. All instances are relabeled consecutively from 1 to N and encoded as COCO RLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class Detection:
    """One detector instance: confidence score and COCO RLE counts."""
    score: float
    counts: str


@dataclass(frozen=True)
class GrownRegion:
    """A detection's grown mask, cropped to its bounding box for fast assignment."""
    score: float
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray


@dataclass
class FusionParams:
    """Configurable hyperparameters for detector-guided instance fusion."""
    threshold: float = 0.55          # U-Net probability threshold (0.55 optimal precision)
    close_radius: int = 3            # Morphological closing radius for U-Net mask
    grow: int = 8                    # Dilate detector mask by this radius (pixels)
    min_score: float = 0.25          # Minimum detector confidence to claim pixels
    min_area: int = 300              # Minimum claimed foreground area (pixels)
    keep_unclaimed: int = 0          # Fallback: keep unclaimed U-Net components >= this area (0 to drop)
    unclaimed_gap: int = 24          # Morphological bridging gap for unclaimed fallback
    suppress_overlap_iou: float = 0.65  # NMS IoU threshold for duplicate detections
    soft_rescue_conf: float = 0.0     # Disabled: Zero unverified detector fallback (prevents FPs)
    soft_rescue_min_area: int = 0


def create_ellipse_kernel(radius: int) -> np.ndarray:
    """Creates an elliptical structuring element of given radius."""
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def prepare_unet_foreground(
    prob_map: np.ndarray,
    disk_mask: np.ndarray,
    threshold: float = 0.50,
    close_radius: int = 3,
) -> np.ndarray:
    """
    Thresholds and cleans the U-Net probability map:
    - Binarizes at probability `threshold`
    - Constrains to solar disk
    - Applies morphological closing
    - Fills internal holes
    """
    binary = ((prob_map >= threshold) & disk_mask).astype(np.uint8)
    if close_radius > 0:
        kernel = create_ellipse_kernel(close_radius)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = binary_fill_holes(binary).astype(np.uint8)
    return (binary.astype(bool) & disk_mask)


def decode_rle_mask(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decodes COCO RLE string into a binary 2D numpy array."""
    rle_dict = {"size": [height, width], "counts": counts}
    mask = mask_utils.decode(rle_dict)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def encode_binary_to_rle(mask: np.ndarray) -> str:
    """Encodes a 2D boolean mask into a compressed COCO RLE string."""
    f_mask = np.asfortranarray(mask.astype(np.uint8))
    rle_dict = mask_utils.encode(f_mask)
    return rle_dict["counts"].decode("utf-8")


def suppress_duplicate_detections(
    detections: List[Detection],
    iou_threshold: float = 0.65,
    height: int = 2048,
    width: int = 2048,
) -> List[Detection]:
    """
    Resolves duplicate detections using greedy NMS on mask IoU.
    Detections are sorted by descending confidence.
    """
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: -d.score)
    rles = [{"size": [height, width], "counts": d.counts} for d in sorted_dets]
    ious = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))

    keep = []
    suppressed = np.zeros(len(sorted_dets), dtype=bool)

    for i in range(len(sorted_dets)):
        if suppressed[i]:
            continue
        keep.append(sorted_dets[i])
        for j in range(i + 1, len(sorted_dets)):
            if not suppressed[j] and ious[i, j] > iou_threshold:
                suppressed[j] = True

    return keep


def grow_detection_regions(
    detections: List[Detection],
    grow_radius: int,
    shape: Tuple[int, int],
    min_score: float = 0.0,
) -> List[GrownRegion]:
    """
    Decodes each detection's mask, dilates it by `grow_radius`, and crops to bounding box.
    Ordered by descending confidence.
    """
    height, width = shape
    regions = []
    kernel = create_ellipse_kernel(grow_radius) if grow_radius > 0 else None

    for det in sorted(detections, key=lambda d: -d.score):
        if det.score < min_score:
            continue
        mask = decode_rle_mask(det.counts, height, width)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue

        y0 = max(int(ys.min()) - grow_radius, 0)
        y1 = min(int(ys.max()) + grow_radius + 1, height)
        x0 = max(int(xs.min()) - grow_radius, 0)
        x1 = min(int(xs.max()) + grow_radius + 1, width)

        crop = mask[y0:y1, x0:x1].astype(np.uint8)
        if kernel is not None:
            crop = cv2.dilate(crop, kernel)

        regions.append(GrownRegion(
            score=det.score,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            mask=crop.astype(bool),
        ))

    return regions


def bridge_and_label_components(binary: np.ndarray, gap: int) -> np.ndarray:
    """Bridges spatial gaps in binary mask and assigns connected component labels."""
    if gap > 0:
        kernel = create_ellipse_kernel(gap // 2)
        bridged = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    else:
        bridged = binary.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bridged, connectivity=8)
    # Mask out pixels that were not originally foreground
    labels = np.where(binary, labels, 0)
    return labels


def assign_instances(
    unet_binary: np.ndarray,
    regions: List[GrownRegion],
    min_score: float,
    min_area: int,
    keep_unclaimed: int = 0,
    unclaimed_gap: int = 24,
    soft_rescue_conf: float = 0.35,
    soft_rescue_min_area: int = 150,
) -> np.ndarray:
    """
    Assigns foreground pixels to detector instances in confidence order.
    Guarantees:
    - Zero overlap (each pixel claimed at most once)
    - Minimum area filtering
    - Soft rescue for high-confidence detector masks with small U-Net overlap
    - Optional unclaimed pixel fallback
    - Consecutive 1-indexed instance IDs
    """
    labels = np.zeros(unet_binary.shape, dtype=np.int32)
    claimed = np.zeros(unet_binary.shape, dtype=bool)
    next_label = 1

    for reg in regions:
        if reg.score < min_score:
            break
        window = (slice(reg.y0, reg.y1), slice(reg.x0, reg.x1))
        # Unclaimed U-Net foreground pixels falling inside grown detection mask
        pixels = unet_binary[window] & reg.mask & ~claimed[window]
        if int(pixels.sum()) < min_area:
            # Soft-rescue: if detector has high confidence (e.g. >= 0.35), rescue native detection mask
            if soft_rescue_conf > 0 and reg.score >= soft_rescue_conf:
                rescue_px = reg.mask & ~claimed[window]
                if int(rescue_px.sum()) >= soft_rescue_min_area:
                    labels[window][rescue_px] = next_label
                    claimed[window] |= rescue_px
                    next_label += 1
            continue

        labels[window][pixels] = next_label
        claimed[window] |= pixels
        next_label += 1

    # Optional fallback for unclaimed U-Net foreground
    if keep_unclaimed > 0:
        unclaimed_fg = (unet_binary & ~claimed).astype(np.uint8)
        if unclaimed_fg.sum() >= keep_unclaimed:
            comp_labels = bridge_and_label_components(unclaimed_fg, unclaimed_gap)
            counts = np.bincount(comp_labels.ravel())
            for c_id in range(1, len(counts)):
                if counts[c_id] >= keep_unclaimed:
                    mask_c = (comp_labels == c_id)
                    labels[mask_c] = next_label
                    next_label += 1

    return labels


def fuse_unet_and_detections(
    unet_prob: np.ndarray,
    disk_mask: np.ndarray,
    detections: List[Detection],
    params: Optional[FusionParams] = None,
) -> np.ndarray:
    """
    Complete fusion pipeline:
    1. Filter duplicate detections
    2. Prepare clean U-Net foreground
    3. Grow detector instance regions
    4. Greedily assign pixels with zero overlap
    Returns (H, W) int32 label map where 0 is background, 1..N are instances.
    """
    params = params or FusionParams()

    # 1. Resolve duplicates
    clean_dets = suppress_duplicate_detections(
        detections,
        iou_threshold=params.suppress_overlap_iou,
        height=unet_prob.shape[0],
        width=unet_prob.shape[1],
    )

    # 2. U-Net foreground
    binary = prepare_unet_foreground(
        unet_prob,
        disk_mask,
        threshold=params.threshold,
        close_radius=params.close_radius,
    )

    # 3. Grow detector masks
    regions = grow_detection_regions(
        clean_dets,
        grow_radius=params.grow,
        shape=unet_prob.shape,
        min_score=params.min_score,
    )

    # 4. Assign
    return assign_instances(
        binary,
        regions,
        min_score=params.min_score,
        min_area=params.min_area,
        keep_unclaimed=params.keep_unclaimed,
        unclaimed_gap=params.unclaimed_gap,
        soft_rescue_conf=params.soft_rescue_conf,
        soft_rescue_min_area=params.soft_rescue_min_area,
    )


def labels_to_instance_rles(label_map: np.ndarray) -> List[str]:
    """Converts an integer label map (0=bg, 1..N=instances) into a list of COCO RLE strings."""
    rles = []
    num_instances = int(label_map.max())
    for inst_id in range(1, num_instances + 1):
        mask = (label_map == inst_id)
        if mask.sum() > 0:
            rles.append(encode_binary_to_rle(mask))
    return rles
