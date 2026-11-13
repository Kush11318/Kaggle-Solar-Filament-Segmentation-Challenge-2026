# MAGFiLO 1.0: Solar Filament Segmentation Challenge 2026
### Official Champion Submission & Technical Documentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Leaderboard](https://img.shields.io/badge/Kaggle%20LB-0.39%20Top%20Tier-success.svg)](https://www.kaggle.com/competitions/filament-segmentation-2026)

**Competition**: [Filament Segmentation Challenge 2026 (MAGFiLO 1.0)](https://www.kaggle.com/competitions/filament-segmentation-2026)  
**Host Organization**: Dr. Azim Ahmadzadeh & IEEE BigData Review Committee  
**Authors**: Team MAGFiLO / Kush11318  
**Repository**: [https://github.com/Kush11318/Kaggle-Solar-Filament-Segmentation-Challenge-2026](https://github.com/Kush11318/Kaggle-Solar-Filament-Segmentation-Challenge-2026)  
**Integrity Guarantee**: 100% Genuine Neural Forward Passes. Zero Hardcoded Labels. Pure PyTorch & OpenCV.

---

## 1. Executive Summary & Official Benchmarks

Solar filaments are cool, dense chromospheric plasma structures suspended in the solar corona by magnetic flux ropes along Polarity Inversion Lines (PILs). Full-disk H-$\alpha$ ($656.28\text{ nm}$) observations ($2048 \times 2048$ pixels) present extreme challenges: large scale variation (from thin $5\text{ px}$ barbs to $1000\text{ px}$ macro filaments), severe limb darkening, and low-contrast absorption against quiet-sun granulation.

Our framework—**MAGFiLO 1.0**—solves this via a **Decoupled Two-Stage Architecture with Collinear Medial-Axis Spine Reconnection**:
1. **Stage 1 (Global Macro Scout)**: YOLO11m evaluated at $1280 \times 1280\text{ px}$ proposes dynamic square bounding regions around candidate magnetic flux ropes.
2. **Stage 2 (Local Native-Scale Morphological Refiner)**: Extracts $1.8\times$ context square crops at 1:1 pixel scale, refined by a Dual U-Net ensemble (**EfficientNet-B4 + ResNet-34**) with topology-safe Flip-4 test-time augmentation (TTA).
3. **Stage 3 (Collinear Medial-Axis Spine Reconnection)**: Mathematically identifies broken filament segments and bridges collinear gaps ($\le 50\text{ px}$, $\le 30^\circ$) along dark chromospheric absorption valleys.
4. **Stage 4 (Disjoint Greedy Allocation)**: Assembles instances onto the global $2048 \times 2048$ canvas with zero-overlap guarantee, encoded into standard COCO Run-Length Encoding (RLE).

### Official Benchmark Performance

| Evaluation Level | Metric | Score | Key Statistics |
| :--- | :---: | :---: | :--- |
| **Kaggle Public Leaderboard** | **Official LB Score** | **`0.39`** | **1,221 instances** across 177 test images, 2.34M px area |
| **Local Validation Benchmark** | **Panoptic Quality (PQ)** | **`0.4618`** | Full 80/20 split (137 images, 227 annotator records) |
| Local Validation | Segmentation Quality (SQ) | `0.6838` | Average IoU on matched true positives |
| Local Validation | Recognition Quality (RQ) | `0.6754` | F1-score of detection matches |

---

## 2. System Architecture

```
                                  [Raw H-alpha Image 2048x2048]
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Phase 1: Solar Disk Detection │
                               │  & Optical Limb Flattening (2ch)│
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Phase 2: Stage 1 Macro Scout  │
                               │        (YOLO11m @ 1280px)       │
                               └────────────────┬────────────────┘
                                                │ Dynamic 1.8x Native Crops
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Phase 3: Stage 2 Crop Refinement│
                               │   (Dual EffNet-B4 + ResNet-34)  │
                               │   + Topology-Safe Flip-4 TTA    │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Phase 4: Bernasconi Collinear   │
                               │     Spine Reconnection Loop     │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Phase 5: Disjoint Greedy Merge  │
                               │     & Compliant COCO RLE Export │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                                      [submission.csv (0.39)]
```

### Core Algorithmic Components:
1. **Limb-Darkening Flattening (`src/disk.py`)**:
   Fits a robust circle $(c_x, c_y, R)$ to the solar limb and flattens optical limb darkening via:
   $$I_{\text{flat}}(r) = \frac{I(r)}{u_0 + u_1 \cos \theta}, \quad \cos \theta = \sqrt{1 - (r/R)^2}$$
   Input tensor is constructed as 2 channels: Normalized raw disk $[0, 1]$ and limb-flattened contrast.
2. **Dynamic Square Native Cropping (`src/crop_utils.py`)**:
   Expands each bounding box by $1.8\times$ context, shifts the square crop to respect solar boundary without geometric distortion or aspect ratio warping, and rescales to $512 \times 512$ for high-resolution segmentation.
3. **Dual Crop U-Net Ensemble (`src/crop_pipeline.py`)**:
   - Model A: EfficientNet-B4 U-Net trained with Combined Centerline Dice + BCE loss (`losses.py`).
   - Model B: ResNet-34 U-Net mined on hard negative active regions.
   - Inference uses 4-fold dihedral TTA with topology safety guardrails (Dice $\ge 0.62$).
4. **Bernasconi Collinear Spine Reconnection (`src/spine_reconnection.py`)**:
   Extracts medial axes of adjacent candidate masks, fits tangent vectors to endpoints, and creates geodesic bridges between collinear segments satisfying:
   $$\Delta \theta \le 30^\circ, \quad \text{distance} \le 50\text{ px}$$

---

## 3. Repository Structure

```
.
├── configs/
│   └── baseline_config.json          # Pipeline hyperparameter configurations
├── splits/
│   └── train_val_split_80_20.json    # Official 80/20 train/val split (137 val images)
├── src/
│   ├── crop_pipeline.py              # TwoStageCropPipeline implementation
│   ├── crop_utils.py                 # Dynamic square cropping & disjoint merge logic
│   ├── dataset.py                    # PyTorch solar dataset loader
│   ├── dataset_crops.py              # Instance-level crop dataset extractor
│   ├── disk.py                       # Solar disk geometry & limb correction
│   ├── losses.py                     # Soft Centerline Dice (clDice) & BCE losses
│   ├── metrics.py                    # Official competition Panoptic Quality evaluation
│   ├── parser.py                     # MAGFiLO COCO JSON and RLE parser
│   ├── paths.py                      # Dataset path resolution
│   └── spine_reconnection.py         # Bernasconi medial-axis reconnection engine
├── tests/
│   ├── test_disk.py                  # Unit tests for disk geometry detection
│   ├── test_metrics.py               # Unit tests for Panoptic Quality calculations
│   └── test_spine_reconnection.py    # Unit tests for collinear spine bridging
├── evaluate.py                       # Official script reproducing Val PQ = 0.4618
├── inference.py                      # Official script generating test submission (0.39)
├── train_crop_unet.py                # Training script for Stage 2 Crop U-Nets
├── requirements.txt                  # Python dependencies
├── submission.csv                    # Verified 0.39 LB Champion submission file
├── MAGFiLO_1_0_JURY_SUBMISSION_NOTEBOOK.ipynb  # Interactive jury demonstration notebook
└── README.md                         # Technical paper & jury instructions
```

---

## 4. Quickstart & Reproduction Guide

### Step 1: Environment Setup
Ensure Python 3.10+ and a CUDA-capable GPU (tested on NVIDIA Tesla T4 / RTX 3090 / RTX 4090) are available:
```bash
git clone https://github.com/Kush11318/Kaggle-Solar-Filament-Segmentation-Challenge-2026.git
cd Kaggle-Solar-Filament-Segmentation-Challenge-2026
pip install -r requirements.txt
```

### Step 2: Model Checkpoints
The champion ensemble uses three trained checkpoints:
- `yolo11m_staple_best.pt`: Stage 1 detector (45.3 MB)
- `crop_unet_effnet_b4_512_mined_best.pt`: Stage 2 primary refiner (81.7 MB)
- `crop_unet_resnet34_512_mined_best.pt`: Stage 2 complementary refiner (97.9 MB)

Place the checkpoint files into the `checkpoints/` directory.

### Step 3: Run Validation Evaluation (Reproduce Val PQ = 0.4618)
To evaluate the pipeline on the official 137 validation images (227 annotator records) using the competition metric:
```bash
python evaluate.py --device cuda:0
```
**Expected Output**:
```
OFFICIAL VALIDATION BENCHMARK RESULTS:
  Panoptic Quality (PQ):     0.4618
  Segmentation Quality (SQ): 0.6838
  Recognition Quality (RQ):  0.6754
  True Positives (TP):       1032
  False Positives (FP):      491
  False Negatives (FN):      502
```

### Step 4: Run Competition Test Inference (Reproduce 0.39 LB)
To run full-disk inference across all 180 competition test images and output `submission.csv`:
```bash
python inference.py --output_csv submission.csv --device cuda:0
```
**Expected Output**:
```
TEST INFERENCE FINISHED SUCCESSFULLY!
  Target File:     submission.csv
  Total Instances: 1221
  Unique Images:   177 / 180
  File Size:       327,689 bytes
```

### Step 5: Run Unit Tests
To verify mathematical and morphological components:
```bash
pytest tests/
```

---

## 5. Submission File Verification

The active `submission.csv` included in this repository is mathematically identical to the file submitted to the Kaggle competition leaderboard:
- **File Name**: `submission.csv`
- **MD5 Checksum**: `42e499166009dda2f5e9fb2dcff9bfc5`
- **Instances**: `1,221`
- **Unique Test Images**: `177 / 180`
- **Verified Public Leaderboard Score**: **`0.39`**

---

## 6. License & Citations
This repository is licensed under the [MIT License](LICENSE).

```bibtex
@inproceedings{magfilo2026,
  title={MAGFiLO 1.0: Multi-Scale Decoupled Two-Stage Architecture with Collinear Medial-Axis Spine Reconnection for Full-Disk Solar Filament Segmentation},
  author={Sharma, Kushagra and Team MAGFiLO},
  booktitle={Filament Segmentation Challenge 2026, IEEE BigData},
  year={2026}
}
```

