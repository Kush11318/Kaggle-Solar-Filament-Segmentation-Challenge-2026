import os
import sys
import pytest
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.topological_cleaner import topological_clean_and_extend_tail


def test_topological_cleaner_pruning_and_extension():
    canvas = np.zeros((150, 150), dtype=bool)

    # Elongated filament line
    pts = np.array([[20, 75], [50, 75], [80, 75], [110, 75]], np.int32)
    for i in range(len(pts) - 1):
        cv2.line(canvas.view(np.uint8), tuple(pts[i]), tuple(pts[i+1]), 1, 4)

    # Artificial circular bleed
    cv2.circle(canvas.view(np.uint8), (50, 75), 14, 1, -1)
    orig_area = canvas.sum()

    # Continuous prob map
    prob = np.zeros((150, 150), dtype=np.float32)
    prob[canvas] = 0.8
    # Tail extension beyond x=110 to x=125
    prob[75, 110:125] = 0.55

    cleaned = topological_clean_and_extend_tail(
        canvas, prob_map=prob, max_tail_grow=15, tau_tail=0.40, max_width_ratio=1.8
    )

    assert isinstance(cleaned, np.ndarray)
    assert cleaned.dtype == bool
    # Tail was extended
    assert cleaned[75, 120] == True
