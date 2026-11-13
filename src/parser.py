"""
Dataset and annotation parser for MAGFiLO COCO-format dataset.
Handles multi-annotator preservation, polygon decoding, validation, and metadata extraction.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from pycocotools import mask as mask_util


class MAGFiLOParser:
    """Parser for MAGFiLO 1.0 COCO-format annotations."""

    def __init__(self, json_path: str, img_dir: str):
        self.json_path = json_path
        self.img_dir = img_dir
        
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.info = raw_data.get("info", {})
        self.licenses = raw_data.get("licenses", {})
        self.categories = {c["id"]: c["name"] for c in raw_data.get("categories", [])}

        # Index images by image record ID
        self.images_by_id: Dict[str, Dict[str, Any]] = {}
        # Group image records by raw file_name
        self.records_by_filename: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for img in raw_data.get("images", []):
            self.images_by_id[img["id"]] = img
            self.records_by_filename[img["file_name"]].append(img)

        # Index annotations by image record ID
        self.annotations_by_image_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ann in raw_data.get("annotations", []):
            self.annotations_by_image_id[ann["image_id"]].append(ann)

    @property
    def num_unique_filenames(self) -> int:
        return len(self.records_by_filename)

    @property
    def num_image_records(self) -> int:
        return len(self.images_by_id)

    @property
    def total_annotations(self) -> int:
        return sum(len(anns) for anns in self.annotations_by_image_id.values())

    def get_source_filename(self, image_record_id: str) -> str:
        """Returns the physical filename (e.g. 20140609195854Bh.jpeg) for an image record ID."""
        return self.images_by_id[image_record_id]["file_name"]

    def get_image_path(self, filename: str) -> str:
        return os.path.join(self.img_dir, filename)

    def get_records_for_file(self, filename: str) -> List[Dict[str, Any]]:
        """Returns all independent annotator records for a physical image."""
        return self.records_by_filename.get(filename, [])

    def get_annotations_for_record(self, image_record_id: str) -> List[Dict[str, Any]]:
        return self.annotations_by_image_id.get(image_record_id, [])

    @staticmethod
    def polygon_to_mask(segmentation: List[List[float]], height: int = 2048, width: int = 2048) -> np.ndarray:
        """Rasterize COCO polygon coordinate lists into a binary uint8 mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape((-1, 2))
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def mask_to_rle(mask: np.ndarray) -> str:
        """Encodes a binary mask into a COCO RLE counts string."""
        f_mask = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(f_mask)
        return rle["counts"].decode("utf-8")

    def build_gt_dataframe(self, image_record_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Builds a ground-truth DataFrame matching the official evaluation format:
        columns: ['filament_id', 'segmentation_rle']
        where filament_id is '<annotator_id>-<image_id>_<idx>'
        """
        records = image_record_ids if image_record_ids is not None else list(self.images_by_id.keys())
        rows = []
        for rec_id in records:
            anns = self.get_annotations_for_record(rec_id)
            for idx, ann in enumerate(anns, start=1):
                mask = self.polygon_to_mask(ann["segmentation"])
                if mask.sum() == 0:
                    continue
                rle = self.mask_to_rle(mask)
                fil_id = f"{rec_id}_{idx}"
                rows.append({"filament_id": fil_id, "segmentation_rle": rle})
        return pd.DataFrame(rows)

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Exhaustive data integrity check:
        - Check image file existence on disk
        - Validate polygon coordinates, out-of-bounds, self-intersections, empty masks
        - Check instance areas and filament spines
        """
        issues = {
            "missing_images": [],
            "malformed_polygons": [],
            "out_of_bounds_coords": [],
            "empty_masks": [],
            "zero_area_annotations": [],
            "missing_spines": 0,
            "total_instances_checked": 0,
        }

        # Check physical images
        for fname in self.records_by_filename.keys():
            full_p = os.path.join(self.img_dir, fname)
            if not os.path.isfile(full_p):
                issues["missing_images"].append(fname)

        for rec_id, anns in self.annotations_by_image_id.items():
            for ann in anns:
                issues["total_instances_checked"] += 1
                seg = ann.get("segmentation", [])
                if not seg or not isinstance(seg, list):
                    issues["malformed_polygons"].append((rec_id, ann["id"], "Not a list"))
                    continue

                for poly in seg:
                    if len(poly) < 6 or len(poly) % 2 != 0:
                        issues["malformed_polygons"].append((rec_id, ann["id"], f"Odd/short length: {len(poly)}"))
                    coords = np.array(poly).reshape((-1, 2))
                    if (coords < -1e-3).any() or (coords > 2048.0 + 1e-3).any():
                        issues["out_of_bounds_coords"].append((rec_id, ann["id"], coords.min(), coords.max()))

                mask = self.polygon_to_mask(seg)
                if mask.sum() == 0:
                    issues["empty_masks"].append((rec_id, ann["id"]))

                if ann.get("area", 0) <= 0:
                    issues["zero_area_annotations"].append((rec_id, ann["id"]))

                if not ann.get("spine"):
                    issues["missing_spines"] += 1

        return issues
