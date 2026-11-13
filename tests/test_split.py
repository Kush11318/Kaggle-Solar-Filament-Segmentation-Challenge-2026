"""
Automated unit test proving zero-leakage and validity of train/val split (Priority 2).
"""

import sys
import os
import json
import unittest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paths import get_dataset_paths, get_split_path

_, TRAIN_IMG_DIR, TEST_IMG_DIR = get_dataset_paths()
SPLIT_PATH = get_split_path()


class TestTrainValSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SPLIT_PATH, "r", encoding="utf-8") as f:
            cls.split = json.load(f)
        cls.train_files = set(cls.split["train_filenames"])
        cls.val_files = set(cls.split["val_filenames"])
        cls.train_recs = set(cls.split["train_record_ids"])
        cls.val_recs = set(cls.split["val_record_ids"])
        cls.test_files = set(os.listdir(TEST_IMG_DIR))
        cls.all_train_disk = set(os.listdir(TRAIN_IMG_DIR))

    def test_zero_leakage_between_train_and_val(self):
        # 1. No physical file in both train and val
        overlap = self.train_files.intersection(self.val_files)
        self.assertEqual(len(overlap), 0, f"Leakage! Overlapping files: {overlap}")

        # 2. No record ID in both train and val
        rec_overlap = self.train_recs.intersection(self.val_recs)
        self.assertEqual(len(rec_overlap), 0, f"Leakage! Overlapping record IDs: {rec_overlap}")

    def test_completeness_and_coverage(self):
        # Full coverage of the 707 training images on disk
        self.assertEqual(len(self.train_files) + len(self.val_files), 707)
        self.assertEqual(self.train_files.union(self.val_files), self.all_train_disk)

        # Full coverage of the 1154 JSON image records
        self.assertEqual(len(self.train_recs) + len(self.val_recs), 1154)

    def test_strict_test_set_isolation(self):
        # The 180 competition test files must have 0 overlap with train OR val
        overlap_train = self.train_files.intersection(self.test_files)
        overlap_val = self.val_files.intersection(self.test_files)
        self.assertEqual(len(overlap_train), 0, "Test set leaked into train!")
        self.assertEqual(len(overlap_val), 0, "Test set leaked into val!")
        self.assertEqual(len(self.test_files), 180)

    def test_stratification_balance(self):
        # Verify station distribution in train and val
        def get_stations(fnames):
            return pd.Series([f[14:16] for f in fnames]).value_counts(normalize=True)

        train_st = get_stations(self.train_files)
        val_st = get_stations(self.val_files)

        # All 6 stations present in both partitions
        self.assertEqual(len(train_st), 6)
        self.assertEqual(len(val_st), 6)
        # Check that proportion difference is small (< 8%)
        for st in train_st.index:
            diff = abs(train_st[st] - val_st[st])
            self.assertLess(diff, 0.08, f"Station {st} proportion differs by {diff:.3f}")


if __name__ == "__main__":
    unittest.main()
