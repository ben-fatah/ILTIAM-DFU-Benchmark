# ILTIAM — Cross-Dataset DFU Segmentation Benchmark

**Do Diabetic Foot Ulcer Segmentation Models Generalize?**  
*A Cross-Dataset Benchmark of CNN and Transformer Architectures*

**Abderrahmane Zine Benfatah** — King Saud University, Riyadh, Saudi Arabia  
📧 444106928@student.ksu.edu.sa

---

## Overview

This repository contains all code for the ILTIAM research paper benchmarking
three segmentation architectures (U-Net, DeepLabV3+, SegFormer-B2) for
cross-dataset generalization on Diabetic Foot Ulcer (DFU) segmentation.

**Key finding:** SegFormer-B2 (Transformer) generalizes significantly better
across hospitals than CNN-based models, achieving:
- **DFUC2022 Dice: 0.557 ± 0.002** (vs. U-Net 0.501, DeepLabV3+ 0.489)
- **Medetec Dice: 0.786** (vs. U-Net 0.737, DeepLabV3+ 0.730)
- **Fewest failures on DFUC2022: 31.1%** (vs. U-Net 38.5%, DeepLabV3+ 43.0%)
- All rankings confirmed by Wilcoxon signed-rank tests (p < 0.001)

---

## Repository Structure

```
ILTIAM-DFU-Benchmark/
│
├── train_unet.py               # Train U-Net (ResNet34) — single run
├── train_deeplabv3plus.py      # Train DeepLabV3+ (ResNet34) — single run
├── train_segformer.py          # Train SegFormer-B2 (mit_b2) — single run
├── run_multiseed.py            # Run all 3 models × 3 seeds → mean±std table
│
├── check_leakage.py            # Initial leakage check (aHash)
├── verify_leakage.py           # Strict leakage verification (dHash + visual pairs)
├── overlap_matrix.py           # Full pairwise overlap matrix (all 7 datasets)
│
├── failure_analysis_v2.py      # Per-image Dice, failure table, qualitative gallery
├── wilcoxon_test.py            # Wilcoxon signed-rank tests for all comparisons
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Datasets

| Dataset | Images | Role |
|---|---|---|
| FUSeg (Foot Ulcer Seg. Challenge) | 1,210 | TRAIN |
| AZH Wound Care Center | 1,109 | TRAIN |
| DFUC2022 (train release) | 2,000 | EXTERNAL TEST #1 |
| Medetec (224px) | 160 | EXTERNAL TEST #2 |
| combination_5 | 4,000 | Excluded (contains DFUC2022 — leakage) |
| data_wound_seg | 2,760 | Excluded (same family as FUSeg/AZH) |
| CO2Wounds-V2 | 2,135 | Excluded (wrong disease + overlap) |

**Important:** Run `overlap_matrix.py` before any training to verify leakage-free splits.

---

## Results

All results: mean ± std over 3 random seeds (42, 123, 2024).

| Model | Family | FUSeg Dice | DFUC2022 Dice | Medetec Dice | Gap |
|---|---|---|---|---|---|
| U-Net (ResNet34) | CNN | 0.817±0.006 | 0.501±0.024 | 0.737 | 0.316 |
| DeepLabV3+ (ResNet34) | CNN | 0.802±0.010 | 0.489±0.016 | 0.730 | 0.313 |
| **SegFormer-B2** | **Transformer** | **0.834±0.003** | **0.557±0.002** | **0.786** | **0.277** |

---

## Installation

```bash
git clone https://github.com/[your-username]/ILTIAM-DFU-Benchmark
cd ILTIAM-DFU-Benchmark
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Verify dataset integrity (run first, always):**
```bash
python overlap_matrix.py
```

**Step 2 — Train all models with multiple seeds:**
```bash
python run_multiseed.py
```

**Step 3 — Run single model:**
```bash
python train_unet.py
python train_deeplabv3plus.py
python train_segformer.py
```

**Step 4 — Failure analysis + qualitative figures:**
```bash
python failure_analysis_v2.py
```

**Step 5 — Statistical tests:**
```bash
python wilcoxon_test.py
```

> **Note:** Update the `BASE` path at the top of each script to point to your local dataset directory.

---

## Hardware

- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM)
- CUDA: 12.1
- Training time: ~20 min per run

---

## Citation

If you use this code or find our findings useful, please cite:

```bibtex
@inproceedings{benfatah2026dfu,
  title     = {Do Diabetic Foot Ulcer Segmentation Models Generalize?
               A Cross-Dataset Benchmark of CNN and Transformer Architectures},
  author    = {Benfatah, Abderrahmane Zine},
  booktitle = {[Target venue — MICCAI Workshop / arXiv]},
  year      = {2026}
}
```

---

## License

MIT License — see `LICENSE` for details.