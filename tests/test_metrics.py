"""
Synthetic unit test suite for official Panoptic Quality & evaluation metrics (Priority 3).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import (
    get_overlap_df,
    compute_official_metrics,
    rles_to_layers,
    get_overlap_matrices,
)


def create_mask(coords, h=256, w=256):
    m = np.zeros((h, w), dtype=np.uint8, order="F")
    y0, y1, x0, x1 = coords
    m[y0:y1, x0:x1] = 1
    return m


def mask_to_rle(m):
    return mask_util.encode(np.asfortranarray(m))["counts"].decode("utf-8")


class TestOfficialMetrics(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 256, 256

    def test_perfect_prediction(self):
        # 2 distinct GT filaments, 2 perfectly matching predictions
        m1 = create_mask((10, 50, 10, 50), self.h, self.w)
        m2 = create_mask((100, 160, 100, 160), self.h, self.w)
        rle1 = mask_to_rle(m1)
        rle2 = mask_to_rle(m2)

        gt_df = pd.DataFrame([
            {"filament_id": "010101-20150101000000_1", "segmentation_rle": rle1},
            {"filament_id": "010101-20150101000000_2", "segmentation_rle": rle2},
        ])
        pred_df = pd.DataFrame([
            {"filament_id": "20150101000000_1", "segmentation_rle": rle1},
            {"filament_id": "20150101000000_2", "segmentation_rle": rle2},
        ])

        overlap_df = get_overlap_df(gt_df, pred_df, height=self.h, width=self.w)
        res = compute_official_metrics(overlap_df)

        self.assertAlmostEqual(res["pq"], 1.0, places=4)
        self.assertAlmostEqual(res["sq"], 1.0, places=4)
        self.assertAlmostEqual(res["rq"], 1.0, places=4)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)

    def test_missed_instance_and_false_positive(self):
        m1 = create_mask((10, 50, 10, 50), self.h, self.w)
        m2 = create_mask((60, 100, 60, 100), self.h, self.w)
        m_fp = create_mask((180, 220, 180, 220), self.h, self.w)

        gt_df = pd.DataFrame([
            {"filament_id": "010101-imgA_1", "segmentation_rle": mask_to_rle(m1)},
            {"filament_id": "010101-imgA_2", "segmentation_rle": mask_to_rle(m2)},
        ])
        # Predict m1 (TP) and m_fp (FP), but miss m2 (FN)
        pred_df = pd.DataFrame([
            {"filament_id": "imgA_1", "segmentation_rle": mask_to_rle(m1)},
            {"filament_id": "imgA_2", "segmentation_rle": mask_to_rle(m_fp)},
        ])

        overlap_df = get_overlap_df(gt_df, pred_df, height=self.h, width=self.w)
        res = compute_official_metrics(overlap_df)

        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        # Denominator = 1 + 0.5*1 + 0.5*1 = 2.0; Sum IoU = 1.0 -> PQ = 0.5
        self.assertAlmostEqual(res["pq"], 0.5, places=4)
        self.assertAlmostEqual(res["sq"], 1.0, places=4)
        self.assertAlmostEqual(res["rq"], 0.5, places=4)

    def test_merged_instances(self):
        # 2 distinct GT filaments adjacent: (10..30) and (30..50)
        g1 = create_mask((10, 30, 10, 50), self.h, self.w)
        g2 = create_mask((30, 50, 10, 50), self.h, self.w)
        # 1 merged prediction covering both
        p_merged = create_mask((10, 50, 10, 50), self.h, self.w)

        gt_df = pd.DataFrame([
            {"filament_id": "010101-imgMerge_1", "segmentation_rle": mask_to_rle(g1)},
            {"filament_id": "010101-imgMerge_2", "segmentation_rle": mask_to_rle(g2)},
        ])
        pred_df = pd.DataFrame([
            {"filament_id": "imgMerge_1", "segmentation_rle": mask_to_rle(p_merged)},
        ])

        overlap_df = get_overlap_df(gt_df, pred_df, height=self.h, width=self.w)
        res = compute_official_metrics(overlap_df)

        # Both GTs overlap with the single prediction -> merged_many_to_1 flagged
        self.assertEqual(res["merged_many_to_1"], 1)
        # IoU for each is (20*40) / (40*40) = 0.5 -> Neither strictly > 0.5
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fn"], 2)
        self.assertEqual(res["fp"], 1)

    def test_fragmented_instances(self):
        # 1 large GT filament (10..50)
        g = create_mask((10, 50, 10, 50), self.h, self.w)
        # 2 separate fragmented predictions
        p1 = create_mask((10, 30, 10, 50), self.h, self.w)
        p2 = create_mask((30, 50, 10, 50), self.h, self.w)

        gt_df = pd.DataFrame([
            {"filament_id": "010101-imgFrag_1", "segmentation_rle": mask_to_rle(g)},
        ])
        pred_df = pd.DataFrame([
            {"filament_id": "imgFrag_1", "segmentation_rle": mask_to_rle(p1)},
            {"filament_id": "imgFrag_2", "segmentation_rle": mask_to_rle(p2)},
        ])

        overlap_df = get_overlap_df(gt_df, pred_df, height=self.h, width=self.w)
        res = compute_official_metrics(overlap_df)

        self.assertEqual(res["fragmented_1_to_many"], 1)

    def test_empty_predictions_and_empty_gt(self):
        g = create_mask((10, 50, 10, 50), self.h, self.w)
        gt_df = pd.DataFrame([{"filament_id": "010101-empty_1", "segmentation_rle": mask_to_rle(g)}])
        empty_pred_df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])

        overlap_df = get_overlap_df(gt_df, empty_pred_df, height=self.h, width=self.w)
        res = compute_official_metrics(overlap_df)
        self.assertEqual(res["pq"], 0.0)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["fp"], 0)


if __name__ == "__main__":
    unittest.main()
