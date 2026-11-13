"""
Foundation Model Prompt Engineering for Solar Filament Segmentation.
Leverages Chromospheric Physics:
Filaments are dark absorption features on H-alpha filtergrams.
Given a candidate bounding box from the Stage 1 detector:
1. Finds the darkest medial spine coordinates (chromospheric absorption centers).
2. Synthesizes positive foreground point prompts along the darkest valley.
3. Synthesizes negative background point prompts at the outer bounding corners (quiet sun).
4. Generates guided prompts for SAM / SAM 2 / MobileSAM foundation models.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


def extract_darkest_spine_points(
    raw_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    num_points: int = 3,
    min_distance: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the darkest chromospheric absorption points inside a bounding box
    to serve as high-confidence foreground point prompts.

    Args:
        raw_img: 2048x2048 uint8 image.
        bbox: (x0, y0, x1, y1) in image coordinates.
        num_points: Number of positive spine points to extract.
        min_distance: Minimum pixel separation between extracted points.

    Returns:
        point_coords: (N, 2) array of (x, y) coordinates.
        point_labels: (N,) array of labels (1 for foreground, 0 for background).
    """
    x0, y0, x1, y1 = bbox
    crop = raw_img[y0:y1, x0:x1]
    if crop.size == 0:
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        return np.array([[center_x, center_y]]), np.array([1])

    # Blur to suppress instrumental speckle noise
    blurred = cv2.GaussianBlur(crop, (5, 5), 1.5)

    # Invert so darkest pixels become local maxima
    inverted = 255 - blurred

    points = []
    labels = []

    # Find the top darkest points separated by min_distance
    temp_inv = inverted.copy()
    for _ in range(num_points):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_inv)
        if max_val < 30:  # contrast is too low
            break

        lx, ly = max_loc
        gx = x0 + lx
        gy = y0 + ly
        points.append([gx, gy])
        labels.append(1)  # positive foreground prompt

        # Zero out neighborhood to enforce spatial diversity along the spine
        cv2.circle(temp_inv, (lx, ly), min_distance, 0, -1)

    if len(points) == 0:
        # Fallback to bbox center
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        points.append([center_x, center_y])
        labels.append(1)

    # Optional: Add negative background prompts at the corners of the bbox
    # Filaments are elongated; the 4 corners of the tight bbox are almost always quiet sun
    corner_margin = 4
    corners = [
        [max(0, x0 + corner_margin), max(0, y0 + corner_margin)],
        [min(2047, x1 - corner_margin), max(0, y0 + corner_margin)],
        [max(0, x0 + corner_margin), min(2047, y1 - corner_margin)],
        [min(2047, x1 - corner_margin), min(2047, y1 - corner_margin)],
    ]
    for cx, cy in corners:
        # Only add if corner is not too close to any positive spine point
        dists = [np.hypot(cx - px, cy - py) for px, py in points]
        if min(dists) > 20:
            points.append([cx, cy])
            labels.append(0)  # negative background prompt

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
