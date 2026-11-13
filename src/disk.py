"""
Solar disk geometry detection, limb-darkening correction, and coordinate mapping.
Phase 2 implementation for Solar Filament Segmentation Challenge 2026.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2


@dataclass(frozen=True)
class Disk:
    cx: float
    cy: float
    radius: float

    def mask(self, shape: Tuple[int, int], shrink: float = 1.00) -> np.ndarray:
        """
        Boolean mask of the disk.
        Default shrink=1.00 ensures peripheral filaments touching the limb are preserved.
        """
        height, width = shape
        ys, xs = np.ogrid[:height, :width]
        r2 = (xs - self.cx) ** 2 + (ys - self.cy) ** 2
        return r2 <= (self.radius * shrink) ** 2

    def radius_map(self, shape: Tuple[int, int]) -> np.ndarray:
        """Normalized radial distance from disk center (1.0 at true limb)."""
        height, width = shape
        ys, xs = np.ogrid[:height, :width]
        return np.sqrt((xs - self.cx) ** 2 + (ys - self.cy) ** 2) / max(self.radius, 1.0)


# GONG standard geometry references
_STANDARD_RADIUS_FRACTION = 0.439  # ~899 px on 2048x2048
_N_RAYS = 720
_INWARD_OFFSETS = (0.0, 50.0, 100.0, 150.0)
_AGREEMENT_TOLERANCE = 3.0
_SUPPORT_RATIO = 0.8


def _fit_circle(xs: np.ndarray, ys: np.ndarray, iterations: int = 3) -> Tuple[Disk, np.ndarray]:
    """Kasa least-squares circle fit with iterative outlier rejection."""
    keep = np.ones(xs.size, dtype=bool)
    disk = Disk(0.0, 0.0, 1.0)
    for _ in range(iterations):
        if keep.sum() < 3:
            break
        design = np.column_stack([xs[keep], ys[keep], np.ones(int(keep.sum()))])
        target = -(xs[keep] ** 2 + ys[keep] ** 2)
        d, e, f = np.linalg.lstsq(design, target, rcond=None)[0]
        cx, cy = -d / 2.0, -e / 2.0
        disk = Disk(float(cx), float(cy), float(np.sqrt(max(cx * cx + cy * cy - f, 1.0))))
        residual = np.hypot(xs - disk.cx, ys - disk.cy) - disk.radius
        spread = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        keep = np.abs(residual) <= max(3.0 * spread, 2.0)
    return disk, keep


def _refine_limb(blurred: np.ndarray, start: Disk, inside: float, outside: float) -> Tuple[Disk, float]:
    """Snaps start circle to sharpest radial edge within search band."""
    height, width = blurred.shape
    scale = min(height, width) / 2048.0
    theta = np.linspace(0.0, 2.0 * np.pi, _N_RAYS, endpoint=False)[:, None]
    radii = np.arange(start.radius - inside * scale, start.radius + outside * scale + 1.0, 1.0)
    map_x = (start.cx + radii[None, :] * np.cos(theta)).astype(np.float32)
    map_y = (start.cy + radii[None, :] * np.sin(theta)).astype(np.float32)

    profiles = cv2.remap(blurred, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    in_frame = (map_x >= 1) & (map_x <= width - 2) & (map_y >= 1) & (map_y <= height - 2)
    gradient = np.abs(np.gradient(profiles, axis=1))
    gradient[~in_frame] = -1.0
    edge = np.argmax(gradient, axis=1)
    valid = gradient[np.arange(_N_RAYS), edge] > 0
    valid &= (edge > 1) & (edge < radii.size - 2)
    if valid.sum() < 3:
        return start, 0.0

    rows = np.flatnonzero(valid)
    peak = edge[valid]
    before, at, after = gradient[rows, peak - 1], gradient[rows, peak], gradient[rows, peak + 1]
    curvature = before - 2.0 * at + after
    shift = np.divide(0.5 * (before - after), curvature, out=np.zeros_like(at), where=curvature < 0)
    r = radii[peak] + np.clip(shift, -0.5, 0.5)
    xs = start.cx + r * np.cos(theta[valid, 0])
    ys = start.cy + r * np.sin(theta[valid, 0])
    disk, _ = _fit_circle(xs, ys)
    residual = np.hypot(xs - disk.cx, ys - disk.cy) - disk.radius
    agreeing = np.abs(residual) <= max(_AGREEMENT_TOLERANCE * scale, 1.0)
    return disk, float(agreeing.sum()) / _N_RAYS


def _standard_disk(image: np.ndarray) -> Disk:
    height, width = image.shape
    return Disk((width - 1) / 2.0, height / 2.0, min(height, width) * _STANDARD_RADIUS_FRACTION)


def _texture_disk(image: np.ndarray) -> Optional[Disk]:
    scale = min(image.shape) / 2048.0
    img = image.astype(np.float32)
    detail = np.abs(img - cv2.GaussianBlur(img, (0, 0), max(3.0 * scale, 1.0)))
    texture = cv2.GaussianBlur(detail, (0, 0), max(15.0 * scale, 2.0))
    texture_u8 = np.clip(texture * (255.0 / max(float(texture.max()), 1e-6)), 0, 255)
    _, binary = cv2.threshold(texture_u8.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)
    flood = component.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 1)
    component = component | (1 - flood)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(points) < 3:
        return None
    disk, _ = _fit_circle(points[:, 0], points[:, 1])
    return disk


def _threshold_disk(image: np.ndarray) -> Disk:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        height, width = image.shape
        return Disk(width / 2.0, height / 2.0, min(height, width) * 0.45)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = image.shape
        return Disk(width / 2.0, height / 2.0, min(height, width) * 0.45)
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return Disk(float(cx), float(cy), float(radius))


def detect_disk(image: np.ndarray) -> Disk:
    """
    Locates the solar disk by detecting the true limb rather than the outer halo.
    Combines intensity threshold, texture segmentation, and standard GONG geometry seeds,
    refining with radial gradient profiles and agreement scoring.
    """
    scale = min(image.shape) / 2048.0
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), max(1.5 * scale, 0.8))
    seeds = [_threshold_disk(image), _texture_disk(image), _standard_disk(image)]

    fits = []
    for seed in seeds:
        if seed is None:
            continue
        for offset in _INWARD_OFFSETS:
            start = Disk(seed.cx, seed.cy, seed.radius - offset * scale)
            if start.radius <= 0:
                continue
            disk, _ = _refine_limb(blurred, start, inside=60.0, outside=20.0)
            fits.append(_refine_limb(blurred, disk, inside=10.0, outside=10.0))

    best_support = max((support for _, support in fits), default=0.0)
    if best_support < 0.5:
        return seeds[0]
    supported = [disk for disk, support in fits if support >= _SUPPORT_RATIO * best_support]
    return min(supported, key=lambda disk: disk.radius)


def limb_darkening_profile(image: np.ndarray, disk: Disk, n_bins: int = 128, smooth: int = 9) -> np.ndarray:
    """Computes median intensity as a function of normalized radius r/R_sun."""
    radius = disk.radius_map(image.shape)
    inside = radius <= 1.0
    bins = np.clip((radius[inside] * n_bins).astype(np.int32), 0, n_bins - 1)
    values = image[inside].astype(np.float32)

    profile = np.ones(n_bins, dtype=np.float32)
    order = np.argsort(bins, kind="stable")
    bins_sorted, values_sorted = bins[order], values[order]
    edges = np.searchsorted(bins_sorted, np.arange(n_bins + 1))
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            profile[b] = np.median(values_sorted[lo:hi])

    valid = profile > 0
    if valid.any():
        profile = np.interp(np.arange(n_bins), np.flatnonzero(valid), profile[valid])
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        profile = np.convolve(profile, kernel, mode="same")
    profile[profile <= 0] = 1.0
    return profile


def flatten_limb_darkening(image: np.ndarray, disk: Disk, n_bins: int = 128) -> np.ndarray:
    """
    Normalizes solar disk by dividing out radial limb darkening.
    Returns float32 image with quiet Sun normalized to ~1.0, and filaments sitting < 1.0.
    """
    profile = limb_darkening_profile(image, disk, n_bins=n_bins)
    radius = disk.radius_map(image.shape)
    bin_index = np.clip((radius * n_bins).astype(np.int32), 0, n_bins - 1)
    expected = profile[bin_index]
    flat = image.astype(np.float32) / np.maximum(expected, 1e-3)
    flat[radius > 1.0] = 1.0
    return flat


# -------------------------------------------------------------------------
# Reche & Cid (2026) 3-Peak Solar Disk Histogram Matching (Appendix A)
# -------------------------------------------------------------------------
_RECHE_CID_X_REF: Optional[np.ndarray] = None
_RECHE_CID_CDF_REF: Optional[np.ndarray] = None


def get_reche_cid_reference_cdf(n_points: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the 3-Peak reference CDF according to Reche & Cid (2026) Eq. (A.2):
    D_ref(x) = 0.01 * N(x; 0.35, 0.012) + 1.0 * N(x; 0.60, 0.12) + 0.01 * N(x; 0.85, 0.012)
    where N(x; mu, sigma^2) has mean mu and variance sigma^2.
    """
    global _RECHE_CID_X_REF, _RECHE_CID_CDF_REF
    if _RECHE_CID_X_REF is not None and len(_RECHE_CID_X_REF) == n_points:
        return _RECHE_CID_X_REF, _RECHE_CID_CDF_REF

    import scipy.stats as stats
    x_ref = np.linspace(0.0, 1.0, n_points)
    s1, s2, s3 = np.sqrt(0.012), np.sqrt(0.12), np.sqrt(0.012)
    pdf = (
        0.01 * stats.norm.pdf(x_ref, 0.35, s1)
        + 1.00 * stats.norm.pdf(x_ref, 0.60, s2)
        + 0.01 * stats.norm.pdf(x_ref, 0.85, s3)
    )
    cdf = np.cumsum(pdf)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    _RECHE_CID_X_REF = x_ref
    _RECHE_CID_CDF_REF = cdf
    return x_ref, cdf


def match_solar_disk_3peak_histogram(
    image: np.ndarray,
    disk: Disk,
    n_bins: int = 2048,
    normalize_std: bool = False,
) -> np.ndarray:
    """
    Applies Reche & Cid (2026) Solar Disk 3-Peak Histogram Matching:
    1. Removes limb-darkening via flatten_limb_darkening.
    2. Clips the disk pixel values to [1st, 99th] percentiles and normalizes to [0, 1].
    3. Matches empirical CDF inside the disk to the 3-Peak reference CDF (Eq. A.2).
    4. Zeros out all off-limb pixels (r > R_sun).
    5. Optionally normalizes the matched disk with (val - 0.60) / 0.10 (Eq. A.1).

    Returns:
        matched_image: (H, W) float32 array
    """
    disk_mask = disk.mask(image.shape)
    flat = flatten_limb_darkening(image, disk)
    disk_vals = flat[disk_mask].astype(np.float32)
    if len(disk_vals) == 0:
        return flat

    # 1. Standardize 1st-99th percentiles inside solar disk
    p1, p99 = np.percentile(disk_vals, 1), np.percentile(disk_vals, 99)
    spread = max(p99 - p1, 1e-5)
    disk_norm = np.clip((disk_vals - p1) / spread, 0.0, 1.0)

    # 2. Fast empirical CDF
    x_ref, cdf_ref = get_reche_cid_reference_cdf(n_points=n_bins)
    hist_emp, bin_edges = np.histogram(disk_norm, bins=n_bins, range=(0.0, 1.0))
    cdf_emp = np.cumsum(hist_emp).astype(np.float64)
    cdf_emp = cdf_emp / max(cdf_emp[-1], 1e-8)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 3. Map disk pixels to reference distribution
    emp_quantiles = np.interp(disk_norm, bin_centers, cdf_emp)
    matched_disk = np.interp(emp_quantiles, cdf_ref, x_ref).astype(np.float32)

    # 4. Construct output
    out = np.zeros_like(flat, dtype=np.float32)
    if normalize_std:
        out[disk_mask] = (matched_disk - 0.60) / 0.10
    else:
        out[disk_mask] = matched_disk

    return out


def make_two_channel_input(
    image: np.ndarray,
    disk: Disk,
    use_3peak_matching: bool = False,
) -> np.ndarray:
    """
    Builds the 2-channel preprocessed representation:
    Channel 0: Limb-darkening corrected intensity, clipped [0, 1.5] and rescaled to [0, 1]
               (or Reche & Cid 3-Peak Histogram Matched intensity if use_3peak_matching=True)
    Channel 1: Normalized radial distance r / R_sun, clipped [0, 1.2]
    Returns: (2, H, W) float32 array
    """
    if use_3peak_matching:
        matched = match_solar_disk_3peak_histogram(image, disk, normalize_std=False)
        norm_ch0 = np.clip(matched, 0.0, 1.0)
    else:
        flat = flatten_limb_darkening(image, disk)
        norm_ch0 = np.clip(flat / 1.5, 0.0, 1.0)

    rad_map = disk.radius_map(image.shape)
    norm_rad = np.clip(rad_map, 0.0, 1.2).astype(np.float32)
    return np.stack([norm_ch0, norm_rad], axis=0).astype(np.float32)


def compute_frangi_vesselness(
    image: np.ndarray,
    disk: Optional[Disk] = None,
    sigmas: Tuple[float, ...] = (1.5, 3.0, 6.0),
    beta: float = 0.5,
    c_factor: float = 0.5,
    black_ridges: bool = True,
) -> np.ndarray:
    """
    Computes multi-scale Frangi vesselness / Hessian ridge filter response.
    Specifically isolates elongated, dark curvilinear absorption trenches (filaments)
    while rejecting isotropic chromospheric granulation and bright active regions.
    Returns: (H, W) float32 array in [0.0, 1.0].
    """
    try:
        from skimage.filters import frangi
        img_norm = image.astype(np.float32)
        if img_norm.max() > 1.0:
            img_norm /= max(1.0, float(img_norm.max()))
        vesselness = frangi(img_norm, sigmas=sigmas, black_ridges=black_ridges, mode='reflect')
        vesselness = np.nan_to_num(vesselness, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        # High-speed fallback using pure SciPy/NumPy separable Gaussian derivatives
        from scipy import ndimage
        input_img = -image.astype(np.float32) if black_ridges else image.astype(np.float32)
        vesselness = np.zeros_like(input_img, dtype=np.float32)
        two_beta_sq = 2.0 * (beta ** 2)

        for sigma in sigmas:
            scale = sigma ** 2
            hxx = scale * ndimage.gaussian_filter(input_img, sigma=sigma, order=[2, 0], mode='reflect')
            hyy = scale * ndimage.gaussian_filter(input_img, sigma=sigma, order=[0, 2], mode='reflect')
            hxy = scale * ndimage.gaussian_filter(input_img, sigma=sigma, order=[1, 1], mode='reflect')

            trace = hxx + hyy
            det = hxx * hyy - (hxy ** 2)
            disc = np.maximum(0.0, (trace ** 2) - 4.0 * det)
            sqrt_disc = np.sqrt(disc)

            eig1 = 0.5 * (trace - sqrt_disc)
            eig2 = 0.5 * (trace + sqrt_disc)

            cond = np.abs(eig1) > np.abs(eig2)
            lambda1 = np.where(cond, eig2, eig1)
            lambda2 = np.where(cond, eig1, eig2)

            s = np.sqrt(lambda1 ** 2 + lambda2 ** 2)
            c = c_factor * s.max() if s.max() > 0 else 1.0
            two_c_sq = 2.0 * (c ** 2)
            rb = np.abs(lambda1) / np.maximum(np.abs(lambda2), 1e-6)

            v_scale = np.exp(-(rb ** 2) / two_beta_sq) * (1.0 - np.exp(-(s ** 2) / two_c_sq))
            v_scale[lambda2 >= 0] = 0.0
            vesselness = np.maximum(vesselness, v_scale)

    if disk is not None:
        mask = disk.mask(image.shape)
        vesselness[~mask] = 0.0
        v_max = vesselness[mask].max() if mask.any() else 1.0
        if v_max > 0:
            vesselness /= v_max
    else:
        v_max = vesselness.max()
        if v_max > 0:
            vesselness /= v_max

    return vesselness.astype(np.float32)


def make_three_channel_solar_input(
    image: np.ndarray,
    disk: Disk,
    use_3peak_matching: bool = False,
    sigmas: Tuple[float, ...] = (1.5, 3.0, 6.0),
) -> np.ndarray:
    """
    Builds the 3-channel domain-tailored solar representation:
    Channel 0: Limb-darkening corrected intensity, clipped [0, 1] (or 3-Peak Matched)
    Channel 1: Normalized radial distance r / R_sun, clipped [0, 1.2]
    Channel 2: Multi-Scale Frangi Hessian Curvilinear Ridge Prior, clipped [0, 1]
    Returns: (3, H, W) float32 array
    """
    two_ch = make_two_channel_input(image, disk, use_3peak_matching=use_3peak_matching)
    norm_ch0 = two_ch[0]
    frangi_ch = compute_frangi_vesselness(norm_ch0, disk=disk, sigmas=sigmas, black_ridges=True)
    return np.stack([two_ch[0], two_ch[1], frangi_ch], axis=0).astype(np.float32)


class DiskCache:
    """Persistent cache for solar disk geometries across the dataset."""
    def __init__(self, cache_file_path: str):
        self.cache_file_path = cache_file_path
        self.cache: Dict[str, Dict[str, float]] = {}
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as f:
                self.cache = json.load(f)

    def get(self, filename: str) -> Optional[Disk]:
        if filename in self.cache:
            d = self.cache[filename]
            return Disk(cx=d["cx"], cy=d["cy"], radius=d["radius"])
        return None

    def set(self, filename: str, disk: Disk):
        self.cache[filename] = {"cx": disk.cx, "cy": disk.cy, "radius": disk.radius}

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        with open(self.cache_file_path, "w") as f:
            json.dump(self.cache, f, indent=2)


_GLOBAL_DISK_CACHE: Optional[DiskCache] = None


def get_disk_geometry(filename: str) -> Tuple[Tuple[float, float], float]:
    """Returns ((cx, cy), radius) for an image filename, loading from disk_cache.json."""
    global _GLOBAL_DISK_CACHE
    if _GLOBAL_DISK_CACHE is None:
        # Search candidate locations for disk_cache.json
        candidate_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "disk_cache.json"),
            "/kaggle/working/checkpoints/disk_cache.json",
            "d:/Kaggle/checkpoints/disk_cache.json",
        ]
        cache_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                cache_path = p
                break
        if cache_path is not None:
            _GLOBAL_DISK_CACHE = DiskCache(cache_path)

    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    if _GLOBAL_DISK_CACHE is not None:
        disk = _GLOBAL_DISK_CACHE.get(f"{stem}.jpeg") or _GLOBAL_DISK_CACHE.get(base)
        if disk is not None:
            return (disk.cx, disk.cy), disk.radius

    # Standard GONG default geometry
    return (1023.5, 1023.5), 899.0
