"""
Solar Filament Collinear Medial Axis Spine Reconnection Module.

Based on Solar Chromospheric Physics (Bernasconi et al. 2005, Fuller et al. 2008, Hao et al. 2015):
Solar filaments often fracture into disconnected fragments across local fibril bridges,
instrumental noise, or quadrant boundaries.
In Panoptic Quality (PQ), fragmented instances suffer double penalties:
- Each fragment fails IoU >= 0.5 with ground truth (0 TP, 1 FN, N FP).
- Reconnecting fragments along the magnetic flux rope spine converts them into 1 TP.

Algorithmic Pipeline:
1. Fast morphological skeletonization via OpenCV erosion differences.
2. Endpoint extraction via 8-neighborhood kernel convolution.
3. Outward tangent vector estimation at skeleton termination points.
4. Mutual collinearity and distance verification:
   - Euclidean gap d <= max_gap (default 40 px)
   - Angle deviation <= max_angle_deg (default 35 deg)
   - Chromospheric absorption verification across the bridge path
5. Union-Find graph clustering + morphological bridge synthesis.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


def extract_morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Extracts 1-pixel wide medial skeleton from a binary mask (uint8: 0 or 1/255).
    Uses OpenCV morphological erosion-difference loop (runs in <2ms).
    """
    skel = np.zeros(mask.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = (mask > 0).astype(np.uint8) * 255

    while True:
        eroded = cv2.erode(temp, element)
        temp_open = cv2.dilate(eroded, element)
        temp_open = cv2.subtract(temp, temp_open)
        skel = cv2.bitwise_or(skel, temp_open)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return (skel > 0).astype(np.uint8)


def extract_skeleton_endpoints_and_tangents(
    skel: np.ndarray,
    tangent_window: int = 8,
) -> List[Dict[str, Any]]:
    """
    Finds endpoints of a 1-pixel skeleton and computes their outward unit tangent vectors.
    Returns list of dicts: {'point': (x, y), 'tangent': (dx, dy)}
    """
    if cv2.countNonZero(skel) == 0:
        return []

    # 8-neighbor convolution: endpoints have exactly 1 neighbor
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel, -1, kernel)
    endpoint_mask = (skel == 1) & (neighbors == 1)

    y_pts, x_pts = np.where(endpoint_mask)
    if len(x_pts) == 0:
        return []

    skel_pts = np.argwhere(skel == 1)  # list of [y, x]
    endpoints = []

    for ex, ey in zip(x_pts, y_pts):
        # Find nearest skeleton points to estimate local direction
        # Distances from this endpoint to all skeleton points
        dists = np.hypot(skel_pts[:, 1] - ex, skel_pts[:, 0] - ey)
        nearby_indices = np.argsort(dists)
        
        # Take the closest few points (up to tangent_window)
        k = min(len(nearby_indices), tangent_window + 1)
        local_pts = skel_pts[nearby_indices[:k]]  # [y, x]

        # Vector from interior towards endpoint (outward direction)
        # mean of interior points
        if len(local_pts) > 1:
            interior_mean_x = np.mean(local_pts[1:, 1])
            interior_mean_y = np.mean(local_pts[1:, 0])
            dx = float(ex - interior_mean_x)
            dy = float(ey - interior_mean_y)
            mag = math.hypot(dx, dy)
            if mag > 1e-4:
                dx /= mag
                dy /= mag
            else:
                dx, dy = 0.0, 0.0
        else:
            dx, dy = 0.0, 0.0

        endpoints.append({
            "point": (int(ex), int(ey)),
            "tangent": (dx, dy),
        })

    return endpoints


def are_endpoints_collinear(
    ep_a: Dict[str, Any],
    ep_b: Dict[str, Any],
    max_gap: float = 40.0,
    max_angle_deg: float = 35.0,
) -> Tuple[bool, float]:
    """
    Checks if endpoint A of mask 1 and endpoint B of mask 2 face each other
    along a collinear trajectory within max_gap and max_angle_deg.
    Strict mutual alignment required: both tangents must point towards each other along the connecting vector.
    Zero unconditional proximity merging to prevent merging distinct parallel filaments.
    """
    xa, ya = ep_a["point"]
    xb, yb = ep_b["point"]
    dist = math.hypot(xb - xa, yb - ya)

    if dist < 1.0 or dist > max_gap:
        return False, dist

    # Gap unit vectors
    vx_ab = (xb - xa) / dist
    vy_ab = (yb - ya) / dist
    vx_ba = -vx_ab
    vy_ba = -vy_ab

    tx_a, ty_a = ep_a["tangent"]
    tx_b, ty_b = ep_b["tangent"]

    # If tangents were degenerate, allow based on distance alone if very close
    if math.hypot(tx_a, ty_a) < 0.5 or math.hypot(tx_b, ty_b) < 0.5:
        return dist <= (max_gap * 0.5), dist

    cos_thresh = math.cos(math.radians(max_angle_deg))

    # Dot product: tangent at A must point towards B
    dot_a = tx_a * vx_ab + ty_a * vy_ab
    # Dot product: tangent at B must point towards A
    dot_b = tx_b * vx_ba + ty_b * vy_ba

    # Strict mutual collinear alignment check
    if dot_a >= cos_thresh and dot_b >= cos_thresh:
        return True, dist

    return False, dist


def reconnect_collinear_spines(
    instances: List[Tuple[np.ndarray, float]],
    raw_img: Optional[np.ndarray] = None,
    max_gap: float = 40.0,
    max_angle_deg: float = 35.0,
    bridge_thickness: int = 5,
    min_area: int = 180,
) -> List[Tuple[np.ndarray, float]]:
    """
    Iterates over detected candidate masks and merges fragmented segments
    that share collinear skeleton endpoints and continuous dark chromospheric paths.

    Args:
        instances: List of (mask_bool_2048, score) tuples.
        raw_img: Optional 2048x2048 uint8 image for intensity verification along bridges.
        max_gap: Maximum pixel distance to reconnect across (default 40px).
        max_angle_deg: Angular tolerance in degrees (default 35 deg).
        bridge_thickness: Width in pixels of synthesized bridge.
        min_area: Minimum area for retained instances (default 180px).

    Returns:
        Reconnected, unified list of (mask_bool_2048, score) instances.
    """
    if len(instances) <= 1:
        return instances

    n = len(instances)
    # Precompute skeletons and endpoints for each instance
    instance_meta = []
    for mask, score in instances:
        # Bounding box for fast distance rejection
        y_idx, x_idx = np.where(mask)
        if len(y_idx) == 0:
            instance_meta.append(None)
            continue

        ymin, ymax = int(y_idx.min()), int(y_idx.max())
        xmin, xmax = int(x_idx.min()), int(x_idx.max())

        # Crop to bbox for fast morphological processing
        crop_mask = mask[ymin:ymax + 1, xmin:xmax + 1]
        skel_crop = extract_morphological_skeleton(crop_mask)
        endpoints_crop = extract_skeleton_endpoints_and_tangents(skel_crop)

        # Shift endpoints back to global canvas
        global_endpoints = []
        for ep in endpoints_crop:
            gx = ep["point"][0] + xmin
            gy = ep["point"][1] + ymin
            global_endpoints.append({
                "point": (gx, gy),
                "tangent": ep["tangent"],
            })

        instance_meta.append({
            "mask": mask,
            "score": score,
            "bbox": (xmin, ymin, xmax, ymax),
            "endpoints": global_endpoints,
        })

    # Disjoint-Set / Union-Find initialization
    parent = list(range(n))
    bridges = {i: [] for i in range(n)}  # bridges connecting components

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int, bridge_line: Tuple[Tuple[int, int], Tuple[int, int]]):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
        bridges[root_i].append(bridge_line)

    # Check pairwise candidates
    for i in range(n):
        if instance_meta[i] is None or len(instance_meta[i]["endpoints"]) == 0:
            continue
        bbox_i = instance_meta[i]["bbox"]

        for j in range(i + 1, n):
            if instance_meta[j] is None or len(instance_meta[j]["endpoints"]) == 0:
                continue
            bbox_j = instance_meta[j]["bbox"]

            # Fast Bounding Box Gap Check
            x_gap = max(0, max(bbox_i[0] - bbox_j[2], bbox_j[0] - bbox_i[2]))
            y_gap = max(0, max(bbox_i[1] - bbox_j[3], bbox_j[1] - bbox_i[3]))
            if x_gap > max_gap or y_gap > max_gap:
                continue

            # Check mutual endpoint collinearity
            best_match = False
            best_bridge = None
            min_dist = 1e9

            for ep_a in instance_meta[i]["endpoints"]:
                for ep_b in instance_meta[j]["endpoints"]:
                    collinear, d = are_endpoints_collinear(
                        ep_a, ep_b, max_gap=max_gap, max_angle_deg=max_angle_deg
                    )
                    if collinear and d < min_dist:
                        min_dist = d
                        best_match = True
                        best_bridge = (ep_a["point"], ep_b["point"])

            if best_match and best_bridge is not None:
                # Optional chromospheric absorption verification along bridge
                if raw_img is not None:
                    pt_a, pt_b = best_bridge
                    # Sample 5 points along bridge line
                    xs = np.linspace(pt_a[0], pt_b[0], 5).astype(int)
                    ys = np.linspace(pt_a[1], pt_b[1], 5).astype(int)
                    xs = np.clip(xs, 0, 2047)
                    ys = np.clip(ys, 0, 2047)
                    path_vals = raw_img[ys, xs]
                    # Solar flare / plage protection: filaments are dark absorption features
                    # Reject if bridge passes through bright plage (> 200/255)
                    if np.mean(path_vals) > 200:
                        continue

                union(i, j, best_bridge)

    # Group instances by connected component
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        if instance_meta[i] is None:
            continue
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Reconstruct final merged instances
    reconnected_instances = []
    for root, members in clusters.items():
        if len(members) == 1:
            m_idx = members[0]
            reconnected_instances.append((instance_meta[m_idx]["mask"], instance_meta[m_idx]["score"]))
            continue

        # Merge multiple members into a unified mask
        merged_mask = np.zeros((2048, 2048), dtype=bool)
        max_score = 0.0
        for m_idx in members:
            merged_mask |= instance_meta[m_idx]["mask"]
            max_score = max(max_score, instance_meta[m_idx]["score"])

        # Synthesize morphological bridges
        for bridge_line in bridges[root]:
            pt_a, pt_b = bridge_line
            bridge_canvas = np.zeros((2048, 2048), dtype=np.uint8)
            cv2.line(bridge_canvas, pt_a, pt_b, 255, bridge_thickness)
            merged_mask |= (bridge_canvas > 0)

        # Smooth boundary of the reconnected mask via morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged_uint8 = cv2.morphologyEx(merged_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        final_mask = (merged_uint8 > 0)

        if final_mask.sum() >= min_area:
            reconnected_instances.append((final_mask, max_score))

    # Sort instances by score descending
    reconnected_instances.sort(key=lambda x: x[1], reverse=True)
    return reconnected_instances
