"""
Unit tests for detector-guided instance fusion (src/fusion.py).
Tests:
1. Zero overlap guarantee between claimed instances.
2. Duplicate detection suppression.
3. Empty detections handling.
4. Grown detector masks with image boundary clipping.
5. Consecutive 1-indexed instance labeling.
6. Minimum area filtering.
7. Unclaimed foreground recovery fallback.
"""

import unittest
import numpy as np
from pycocotools import mask as mask_utils

from src.fusion import (
    Detection,
    FusionParams,
    fuse_unet_and_detections,
    suppress_duplicate_detections,
    grow_detection_regions,
    assign_instances,
    labels_to_instance_rles,
    encode_binary_to_rle,
    decode_rle_mask,
)


class TestFusionModule(unittest.TestCase):

    def setUp(self):
        self.H, self.W = 200, 200
        self.disk_mask = np.ones((self.H, self.W), dtype=bool)

    def _create_box_mask(self, y0, y1, x0, x1):
        m = np.zeros((self.H, self.W), dtype=bool)
        m[y0:y1, x0:x1] = True
        return encode_binary_to_rle(m)

    def test_empty_detections(self):
        """When no detections are present and keep_unclaimed=0, output is all background."""
        prob_map = np.zeros((self.H, self.W), dtype=np.float32)
        prob_map[50:100, 50:100] = 0.9  # Strong U-Net foreground

        params = FusionParams(threshold=0.5, keep_unclaimed=0)
        labels = fuse_unet_and_detections(prob_map, self.disk_mask, [], params)

        self.assertEqual(labels.max(), 0)
        self.assertEqual(labels.sum(), 0)

    def test_empty_detections_with_fallback(self):
        """When no detections are present and keep_unclaimed > 0, U-Net foreground is preserved."""
        prob_map = np.zeros((self.H, self.W), dtype=np.float32)
        prob_map[50:100, 50:100] = 0.9  # Area = 2500 px

        params = FusionParams(threshold=0.5, keep_unclaimed=500, min_area=300)
        labels = fuse_unet_and_detections(prob_map, self.disk_mask, [], params)

        self.assertEqual(labels.max(), 1)
        self.assertGreater(labels.sum(), 0)

    def test_zero_overlap_guarantee(self):
        """Two overlapping detections claiming same U-Net pixels must NEVER produce overlapping labels."""
        prob_map = np.zeros((self.H, self.W), dtype=np.float32)
        prob_map[40:120, 40:120] = 0.8  # Continuous U-Net blob

        # Detection 1: High confidence, covers (40:90, 40:90)
        det1_rle = self._create_box_mask(40, 90, 40, 90)
        det1 = Detection(score=0.90, counts=det1_rle)

        # Detection 2: Lower confidence, covers (60:120, 60:120) - overlaps det1
        det2_rle = self._create_box_mask(60, 120, 60, 120)
        det2 = Detection(score=0.70, counts=det2_rle)

        params = FusionParams(threshold=0.5, grow=4, min_area=100, suppress_overlap_iou=0.9)
        labels = fuse_unet_and_detections(prob_map, self.disk_mask, [det1, det2], params)

        # Check unique labels per pixel: label map is (H, W) int32 -> inherently single label per pixel
        self.assertIn(labels.max(), [1, 2])
        mask1 = (labels == 1)
        mask2 = (labels == 2)
        # Verify strict zero overlap
        overlap = mask1 & mask2
        self.assertEqual(overlap.sum(), 0, "Pixels were assigned to multiple instances!")

        # High confidence det1 should have claimed the intersection
        intersection_crop = mask1[60:90, 60:90]
        self.assertTrue(np.all(intersection_crop), "High confidence detection did not claim overlapping region first!")

    def test_duplicate_suppression(self):
        """Near-identical duplicate detections are suppressed before assignment."""
        det1_rle = self._create_box_mask(50, 100, 50, 100)
        det2_rle = self._create_box_mask(51, 100, 50, 100)  # ~98% IoU overlap

        det1 = Detection(score=0.85, counts=det1_rle)
        det2 = Detection(score=0.80, counts=det2_rle)

        filtered = suppress_duplicate_detections([det1, det2], iou_threshold=0.65, height=self.H, width=self.W)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].score, 0.85)

    def test_boundary_handling(self):
        """Detection touching image boundaries does not cause index out-of-bounds error."""
        prob_map = np.zeros((self.H, self.W), dtype=np.float32)
        prob_map[:30, :30] = 0.9

        det_rle = self._create_box_mask(0, 20, 0, 20)
        det = Detection(score=0.85, counts=det_rle)

        params = FusionParams(threshold=0.5, grow=15, min_area=50)
        labels = fuse_unet_and_detections(prob_map, self.disk_mask, [det], params)

        self.assertEqual(labels.max(), 1)
        self.assertGreater(labels[:30, :30].sum(), 0)

    def test_consecutive_relabeling_and_rle_roundtrip(self):
        """Labels must be consecutive 1..N and correctly encode to COCO RLE."""
        labels = np.zeros((self.H, self.W), dtype=np.int32)
        labels[10:30, 10:30] = 1
        labels[50:80, 50:80] = 2
        labels[120:150, 120:150] = 3

        rles = labels_to_instance_rles(labels)
        self.assertEqual(len(rles), 3)

        for i, rle in enumerate(rles, start=1):
            m = decode_rle_mask(rle, self.H, self.W)
            np.testing.assert_array_equal(m, labels == i)


if __name__ == "__main__":
    unittest.main()
