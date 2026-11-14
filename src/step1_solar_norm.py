"""
Step 1: Solar Disk Fitting, Photometric Normalization & Limb Darkening Correction (LDC).
Synthesis of BarrelAged photometric statistics, Bernasconi et al. astrophysical flattening,
and Kasa robust circle fitting.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SolarGeometry:
    cx: float
    cy: float
    radius: float
    disk_mean: float
    disk_std: float
    disk_thr: float


def compute_disk_stats_barrel(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes (mean, std + 1e-6, threshold) of solar-disk pixels above 0.3 x 99th percentile.
    Exact implementation from BarrelAged (filseg/data.py) to guarantee 100% weight compatibility.
    """
    thr = float(0.3 * np.percentile(img, 99))
    mask = img > thr
    vals = img[mask].astype(np.float32)
    mean_val = float(vals.mean())
    std_val = float(vals.std() + 1e-6)
    return mean_val, std_val, thr


def fit_solar_disk_circle(img: np.ndarray, thr: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Fits exact solar disk circle (cx, cy, radius) using Canny edge detection around the solar limb
    and Kasa algebraic least-squares circle fitting with outlier rejection.
    """
    H, W = img.shape
    if thr is None:
        thr = float(0.3 * np.percentile(img, 99))

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)
    edges = cv2.Canny(blurred, 30, 80)

    ys, xs = np.nonzero(img > thr)
    if len(ys) < 100:
        return W / 2.0, H / 2.0, min(H, W) * 0.44

    dist_approx = np.hypot(xs - (W / 2.0), ys - (H / 2.0))
    r_est = float(np.percentile(dist_approx, 98.0))

    ey, ex = np.nonzero(edges)
    e_dist = np.hypot(ex - (W / 2.0), ey - (H / 2.0))
    valid_e = (e_dist > r_est - 60.0) & (e_dist < r_est + 60.0)

    ex_v = ex[valid_e].astype(np.float64)
    ey_v = ey[valid_e].astype(np.float64)

    if len(ex_v) < 20:
        # Fallback to minimum enclosing circle of thresholded mask
        contours, _ = cv2.findContours((img > thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            return float(cx), float(cy), float(radius)
        return W / 2.0, H / 2.0, min(H, W) * 0.44

    # Kasa algebraic circle fit
    design = np.column_stack([ex_v, ey_v, np.ones(len(ex_v), dtype=np.float64)])
    target = -(ex_v ** 2 + ey_v ** 2)
    sol, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    d, e, f = sol
    cx = -d / 2.0
    cy = -e / 2.0
    radius_sq = cx ** 2 + cy ** 2 - f
    radius = math.sqrt(max(radius_sq, 100.0))

    # Reject outliers if residual exceeds 3 px
    residuals = np.abs(np.hypot(ex_v - cx, ey_v - cy) - radius)
    inliers = residuals < 4.0
    if inliers.sum() > 20:
        design_in = np.column_stack([ex_v[inliers], ey_v[inliers], np.ones(int(inliers.sum()), dtype=np.float64)])
        target_in = -(ex_v[inliers] ** 2 + ey_v[inliers] ** 2)
        sol, _, _, _ = np.linalg.lstsq(design_in, target_in, rcond=None)
        d, e, f = sol
        cx = -d / 2.0
        cy = -e / 2.0
        radius = math.sqrt(max(cx ** 2 + cy ** 2 - f, 100.0))

    return float(cx), float(cy), float(radius)


def compute_radial_ldc_profile(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes median radial background intensity profile I_bg(r) and absorption depth map:
    A(y, x) = 1.0 - (I(y, x) / I_bg(r(y, x)))
    """
    H, W = img.shape
    Y, X = np.ogrid[:H, :W]
    r_map = np.hypot(X - cx, Y - cy) / max(radius, 1.0)
    inside = r_map <= 0.995

    bins = np.clip((r_map[inside] * n_bins).astype(np.int32), 0, n_bins - 1)
    vals = img[inside].astype(np.float32)

    profile = np.ones(n_bins, dtype=np.float32) * 128.0
    order = np.argsort(bins, kind="stable")
    bins_sorted, vals_sorted = bins[order], vals[order]
    edges = np.searchsorted(bins_sorted, np.arange(n_bins + 1))

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            profile[b] = np.median(vals_sorted[lo:hi])

    # Interpolate missing bins and smooth
    valid = profile > 0
    if valid.any():
        profile = np.interp(np.arange(n_bins), np.flatnonzero(valid), profile[valid])
    kernel = np.ones(5, dtype=np.float32) / 5.0
    profile = np.convolve(profile, kernel, mode="same")
    profile[profile <= 0] = 1.0

    # Build full resolution background intensity map
    bin_grid = np.clip((r_map * n_bins).astype(np.int32), 0, n_bins - 1)
    expected_bg = profile[bin_grid]

    # Normalized absorption depth: 0 at quiet Sun background, > 0 for dark absorption filaments
    absorption = np.clip(1.0 - (img.astype(np.float32) / np.maximum(expected_bg, 1.0)), 0.0, 1.0)
    absorption[r_map > 1.0] = 0.0

    return profile, absorption


def process_image_step1(
    img: np.ndarray,
) -> Tuple[SolarGeometry, np.ndarray, np.ndarray]:
    """
    Master Step 1 processing:
    Returns:
    1. SolarGeometry: cx, cy, radius, disk_mean, disk_std, disk_thr
    2. dense_norm_img: float32 array clipped to [-6.0, 4.0] (exact for dense U-Nets)
    3. absorption_map: float32 array in [0.0, 1.0] representing intrinsic chromospheric absorption
    """
    mean_val, std_val, thr = compute_disk_stats_barrel(img)
    cx, cy, radius = fit_solar_disk_circle(img, thr=thr)

    # Standard dense normalization
    dense_norm_img = np.clip((img.astype(np.float32) - mean_val) / std_val, -6.0, 4.0)

    # Solar physics LDC absorption map
    _, absorption_map = compute_radial_ldc_profile(img, cx, cy, radius, n_bins=100)

    geom = SolarGeometry(
        cx=cx,
        cy=cy,
        radius=radius,
        disk_mean=mean_val,
        disk_std=std_val,
        disk_thr=thr,
    )
    return geom, dense_norm_img, absorption_map

