"""
Environment-agnostic path resolver for local Windows and Kaggle Linux environments.
"""

import os
from typing import Tuple

CANDIDATE_DATA_ROOTS = [
    "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
    "/kaggle/input/datasets/kommander/solar-filament-segmentation-challenge/MAGFiLO_1.0_Kaggle_2026",
    "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
    "/kaggle/input/solar-filament-segmentation/MAGFiLO_1.0_Kaggle_2026",
    r"d:/Kaggle/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
]


def get_project_root() -> str:
    """Returns /kaggle/working if in Kaggle, otherwise d:/Kaggle or local dir."""
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working"
    # Local fallback
    curr = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return curr


def get_dataset_paths() -> Tuple[str, str, str]:
    """
    Returns (json_path, train_img_dir, test_img_dir).
    Automatically resolves between Kaggle and local paths.
    """
    data_root = None
    for cand in CANDIDATE_DATA_ROOTS:
        if os.path.exists(cand):
            data_root = cand
            break

    if data_root is None:
        raise FileNotFoundError(f"Could not locate MAGFiLO dataset in any candidate path: {CANDIDATE_DATA_ROOTS}")

    json_path = os.path.join(data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train_img_dir = os.path.join(data_root, "train", "train_images")
    test_img_dir = os.path.join(data_root, "test", "test_images")

    return json_path, train_img_dir, test_img_dir


def get_split_path() -> str:
    proj_root = get_project_root()
    return os.path.join(proj_root, "splits", "train_val_split_80_20.json")
