"""
Automated validation and integrity test suite for MAGFiLO Parser (Priority 1).
"""

import sys
import os
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.parser import MAGFiLOParser
from src.paths import get_dataset_paths

JSON_PATH, IMG_DIR, _ = get_dataset_paths()


class TestMAGFiLOParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = MAGFiLOParser(JSON_PATH, IMG_DIR)

    def test_counts_and_mapping(self):
        # 707 physical images, 1154 records, 8199 annotations
        self.assertEqual(self.parser.num_unique_filenames, 707)
        self.assertEqual(self.parser.num_image_records, 1154)
        self.assertEqual(self.parser.total_annotations, 8199)

        # Every record maps to a valid source filename
        for rec_id in self.parser.images_by_id:
            fname = self.parser.get_source_filename(rec_id)
            self.assertTrue(fname.endswith(".jpeg"))
            self.assertTrue(os.path.isfile(self.parser.get_image_path(fname)))

    def test_multi_annotator_preservation(self):
        # Verify multi-annotator images have independent records preserved
        counts = [len(recs) for recs in self.parser.records_by_filename.values()]
        self.assertIn(1, counts)
        self.assertIn(2, counts)
        self.assertIn(3, counts)

        # Pick an image with 3 annotators
        multi_file = [f for f, r in self.parser.records_by_filename.items() if len(r) == 3][0]
        records = self.parser.get_records_for_file(multi_file)
        self.assertEqual(len(records), 3)
        # Ensure distinct record IDs
        rec_ids = {r["id"] for r in records}
        self.assertEqual(len(rec_ids), 3)

    def test_polygon_mask_and_rle(self):
        # Pick first annotation
        first_rec = list(self.parser.images_by_id.keys())[0]
        anns = self.parser.get_annotations_for_record(first_rec)
        self.assertGreater(len(anns), 0)
        ann = anns[0]

        mask = self.parser.polygon_to_mask(ann["segmentation"])
        self.assertEqual(mask.shape, (2048, 2048))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(mask.sum(), 0)

        # Test RLE round trip
        rle = self.parser.mask_to_rle(mask)
        from pycocotools import mask as mask_util
        decoded = mask_util.decode({"size": [2048, 2048], "counts": rle})
        self.assertTrue(np.array_equal(mask, decoded))

    def test_full_dataset_integrity(self):
        issues = self.parser.validate_dataset()
        print("\nDataset Integrity Validation Results:")
        for k, v in issues.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} occurrences")
            else:
                print(f"  {k}: {v}")

        self.assertEqual(len(issues["missing_images"]), 0)
        self.assertEqual(len(issues["malformed_polygons"]), 0)
        self.assertEqual(len(issues["empty_masks"]), 0)
        self.assertEqual(len(issues["zero_area_annotations"]), 0)


if __name__ == "__main__":
    unittest.main()
