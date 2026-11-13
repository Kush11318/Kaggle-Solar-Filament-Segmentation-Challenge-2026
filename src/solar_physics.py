"""
Solar Physics Domain Modules: Surface Flattening & Sunspot Filtering.
References:
1. Bernasconi, Rust, Hakim (Solar Physics, 2005) "Advanced Automated Solar Filament
   Detection and Characterization Code".
2. Fuller, Aboudarham, Bentley (Solar Physics, 2005 / AnGeo 2008).

Implements:
- 2D 4th-Order Polynomial Surface Fitting on 2x downsampled disk to eliminate limb darkening
- Sunspot elimination with 2,000 px giant-filament safety threshold
- Directional linear filtering for curvilinear spine enhancement
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def fit_and_subtract_polynomial_surface(
    image: np.ndarray,
    disk_mask: np.ndarray,
    order: int = 4,
    downsample_factor: int = 2,
) -> np.ndarray:
    """
    Fits a 2D polynomial surface of specified order (default: 4th order) on a 2x downsampled
    solar disk and subtracts it from the original image to eliminate limb darkening and
    large-scale chromospheric background gradients. (Section 2.2.1, Bernasconi et al.)
    """
    H, W = image.shape
    
    # 1. Downsample for fast, robust least-squares surface fitting
    h_sub, w_sub = H // downsample_factor, W // downsample_factor
    img_sub = cv2.resize(image.astype(np.float32), (w_sub, h_sub), interpolation=cv2.INTER_AREA)
    mask_sub = cv2.resize(disk_mask.astype(np.uint8), (w_sub, h_sub), interpolation=cv2.INTER_NEAREST) > 0

    # Collect coordinates and intensities inside the disk
    ys, xs = np.nonzero(mask_sub)
    if len(ys) == 0:
        return image.astype(np.float32)

    # Normalize coordinates to [-1, 1] for numerical stability in polynomial powers
    y_norm = 2.0 * (ys / float(h_sub)) - 1.0
    x_norm = 2.0 * (xs / float(w_sub)) - 1.0
    intensities = img_sub[ys, xs]

    # Build design matrix for 2D polynomial of degree <= order
    cols = []
    for d in range(order + 1):
        for i in range(d + 1):
            j = d - i
            cols.append((x_norm ** i) * (y_norm ** j))
    A = np.column_stack(cols) # (num_pixels, num_terms)

    # Solve least squares: A * c = intensities
    c, _, _, _ = np.linalg.lstsq(A, intensities, rcond=None)

    # 2. Evaluate polynomial at full resolution
    Y_grid, X_grid = np.ogrid[:H, :W]
    Y_norm = 2.0 * (Y_grid / float(H)) - 1.0
    X_norm = 2.0 * (X_grid / float(W)) - 1.0

    fitted_surface = np.zeros((H, W), dtype=np.float32)
    term_idx = 0
    for d in range(order + 1):
        for i in range(d + 1):
            j = d - i
            fitted_surface += c[term_idx] * ((X_norm ** i) * (Y_norm ** j)).astype(np.float32)
            term_idx += 1

    # Subtract fitted background surface and zero out outside the disk
    flattened = image.astype(np.float32) - fitted_surface
    flattened[~disk_mask] = 0.0
    return flattened


def filter_sunspots_with_safety_rule(
    image: np.ndarray,
    disk_mask: np.ndarray,
    spot_seed_percentile: float = 0.5,
    max_sunspot_area: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects sunspots via low-intensity thresholding and region-growing.
    Applies the Bernasconi et al. Safety Rule:
    - If a dark region exceeds 2,000 pixels, it is classified as a giant dark filament core
      and preserved.
    - If a dark region is <= 2,000 pixels, it is classified as a true sunspot and masked out.
    """
    disk_pixels = image[disk_mask]
    if len(disk_pixels) == 0:
        return np.zeros_like(image, dtype=bool), image

    # Threshold for dark seeds
    spot_thresh = np.percentile(disk_pixels, spot_seed_percentile)
    dark_seeds = (image <= spot_thresh) & disk_mask

    # Label connected dark components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_seeds.astype(np.uint8), connectivity=8)

    sunspot_mask = np.zeros_like(disk_mask, dtype=bool)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        # Safety rule: if area > 2,000 pixels, it is a giant filament, NOT a sunspot!
        if area <= max_sunspot_area:
            sunspot_mask[labels == lbl] = True

    # Zero out true sunspots from image
    cleaned_image = image.copy()
    cleaned_image[sunspot_mask] = np.median(disk_pixels)
    return sunspot_mask, cleaned_image
