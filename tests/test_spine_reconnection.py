"""
Unit Tests for Solar Chromospheric Collinear Spine Reconnection Module.
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.spine_reconnection import (
    extract_morphological_skeleton,
    extract_skeleton_endpoints_and_tangents,
    are_endpoints_collinear,
    reconnect_collinear_spines,
)


def test_extract_morphological_skeleton():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 45), (80, 55), 255, -1)  # 10 px thick rectangle
    skel = extract_morphological_skeleton(mask)

    assert skel.max() == 1
    assert np.count_nonzero(skel) > 0
    # Skeleton should be substantially thinner than original mask
    assert np.count_nonzero(skel) < np.count_nonzero(mask) * 0.25


def test_extract_endpoints_and_tangents():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Diagonal line from (20, 20) to (80, 80)
    cv2.line(mask, (20, 20), (80, 80), 255, 3)
    skel = extract_morphological_skeleton(mask)
    endpoints = extract_skeleton_endpoints_and_tangents(skel)

    assert len(endpoints) >= 2
    # Verify that endpoints are near (20, 20) and (80, 80)
    pts = [ep["point"] for ep in endpoints]
    has_start = any(np.hypot(p[0] - 20, p[1] - 20) < 5 for p in pts)
    has_end = any(np.hypot(p[0] - 80, p[1] - 80) < 5 for p in pts)
    assert has_start
    assert has_end


def test_are_endpoints_collinear():
    # Endpoints facing each other along x-axis
    ep_a = {"point": (40, 50), "tangent": (1.0, 0.0)}  # pointing right
    ep_b = {"point": (60, 50), "tangent": (-1.0, 0.0)}  # pointing left

    collinear, dist = are_endpoints_collinear(ep_a, ep_b, max_gap=30.0, max_angle_deg=30.0)
    assert collinear is True
    assert dist == pytest.approx(20.0)

    # Opposing endpoints (pointing away from each other)
    ep_c = {"point": (40, 50), "tangent": (-1.0, 0.0)}  # pointing left away from B
    collinear_c, _ = are_endpoints_collinear(ep_c, ep_b, max_gap=30.0, max_angle_deg=30.0)
    assert collinear_c is False

    # Perpendicular endpoints
    ep_d = {"point": (60, 50), "tangent": (0.0, 1.0)}  # pointing down
    collinear_d, _ = are_endpoints_collinear(ep_a, ep_d, max_gap=30.0, max_angle_deg=30.0)
    assert collinear_d is False


def test_reconnect_collinear_spines_merges_broken_fragments():
    # Create two collinear broken fragments on a 2048x2048 canvas
    canvas_a = np.zeros((2048, 2048), dtype=bool)
    canvas_b = np.zeros((2048, 2048), dtype=bool)

    # Fragment A: (500, 500) to (580, 500), thickness 11
    # Fragment B: (600, 500) to (680, 500), thickness 11
    # Gap: 20 pixels between x=580 and x=600 along y=500
    tmp_a = np.zeros((2048, 2048), dtype=np.uint8)
    tmp_b = np.zeros((2048, 2048), dtype=np.uint8)
    cv2.line(tmp_a, (500, 500), (580, 500), 255, 11)
    cv2.line(tmp_b, (600, 500), (680, 500), 255, 11)

    canvas_a = (tmp_a > 0)
    canvas_b = (tmp_b > 0)

    instances = [(canvas_a, 0.85), (canvas_b, 0.75)]
    merged = reconnect_collinear_spines(instances, max_gap=35.0, min_area=100)

    # Should be merged into 1 single instance!
    assert len(merged) == 1
    merged_mask, merged_score = merged[0]
    assert merged_score == 0.85
    # The gap at (590, 500) should now be filled
    assert merged_mask[500, 590] is True or merged_mask[500, 590] == 1


def test_reconnect_collinear_spines_preserves_separated_parallel_filaments():
    # Two distinct parallel filaments 100 pixels apart in y
    canvas_1 = np.zeros((2048, 2048), dtype=bool)
    canvas_2 = np.zeros((2048, 2048), dtype=bool)

    tmp_1 = np.zeros((2048, 2048), dtype=np.uint8)
    tmp_2 = np.zeros((2048, 2048), dtype=np.uint8)
    cv2.line(tmp_1, (500, 400), (650, 400), 255, 9)
    cv2.line(tmp_2, (500, 550), (650, 550), 255, 9)

    canvas_1 = (tmp_1 > 0)
    canvas_2 = (tmp_2 > 0)

    instances = [(canvas_1, 0.90), (canvas_2, 0.88)]
    result = reconnect_collinear_spines(instances, max_gap=40.0, min_area=100)

    # Should remain 2 separate instances!
    assert len(result) == 2


def test_snap_to_absorption_edges():
    from src.crop_pipeline import snap_to_absorption_edges

    raw = np.full((100, 100), 160, dtype=np.uint8)  # bright quiet-Sun granulation
    raw[40:60, 40:60] = 40  # dark absorption trench
    # Create mask dilated by 3px into the bright granulation
    mask = np.zeros((100, 100), dtype=bool)
    mask[37:63, 37:63] = True

    snapped = snap_to_absorption_edges(mask, raw, fringe_width=2)
    # Fringe pixels in bright granulation must be trimmed
    assert snapped.sum() < mask.sum()
    # Core absorption pixels must be preserved
    assert snapped[50, 50] is True or snapped[50, 50] == 1
