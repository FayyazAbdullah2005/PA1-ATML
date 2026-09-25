# EE-5102 / CS-6304: Advanced Topics in Machine Learning
## Programming Assignment 1: Beyond IID and Closed-Set Assumptions

**Author:** Abdullah Fayyaz  
**Email:** 28100069@lums.edu.pk  
**Institution:** Lahore University of Management Sciences (LUMS)  
**Repository:** https://github.com/FayyazAbdullah2005/PA1-ATML  
**Report Document:** report_skeleton.tex (NeurIPS 2026 format)

---

## Overview

This repository contains the code, configuration files, evaluation scripts, results, and LaTeX report for Programming Assignment 1.

The project covers four experimental tasks:
1. **Task 1: Inductive Biases and Feature Representations** - Evaluates ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 on STL-10 under color interventions (grayscale, hue rotation), AdaIN cue conflicts, spatial translations, and patch shuffling.
2. **Task 2: Unsupervised Domain Adaptation (UDA)** - Evaluates Source-only ERM, DAN (MMD), DANN (Adversarial), and CDAN (Conditional Adversarial) on PACS with Sketch as the unlabeled target domain.
3. **Task 3: Domain Generalization (DG)** - Evaluates unaligned ERM, Pairwise Source Alignment (DAN-DG), and Parameter-Space Stability (SAM) on PACS with Sketch strictly held-out as an unseen test domain.
4. **Task 4: Open-Set Recognition (OSR)** - Evaluates post-hoc novelty scoring functions (MSP, MLS, Energy, Mahalanobis) and trained-model methods (Vanilla ResNet-18, GCSC, PROSER) on CIFAR-10 vs CIFAR-100 Near and Far unknowns.

---

## Repository Structure

```
PA1-ATML/
├── README.md                          # Repository documentation and attribution
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes raw data and checkpoints
├── report_skeleton.tex                # NeurIPS report source
├── neurips_2026.sty                   # NeurIPS LaTeX stylesheet
│
├── common/                            # Shared utilities
│   ├── seed.py                        # Seed definition (SEED = 6304)
│   ├── logging.py                     # Logger
│   └── plotting.py                    # Plotting utilities
│
├── shared/                            # Shared PACS protocols for Tasks 2 & 3
│   ├── pacs.py                        # Class names and domain mappings
│   └── pacs_protocol.py               # Deterministic 80/20 train/val splits
│
├── task1/                             # Task 1 code
│   ├── run_task1.py                   # Master script to run all Task 1 experiments
│   ├── models.py                      # ResNet-50, ViT-B/16, OpenCLIP models
│   ├── data_utils.py                  # STL-10 500-sample test loader
│   ├── transforms.py                  # Grayscale, hue rotation, translation, patch shuffle
│   ├── cue_conflicts.py               # AdaIN style transfer and rejection filters
│   └── evaluate.py                    # Shape bias, consistency, and cosine stability
│
├── task2/                             # Task 2 code
│   ├── train.py                       # Training pipeline for UDA models
│   ├── evaluate_final.py              # Evaluation script on Sketch
│   ├── configs/                       # Configuration files (source_only, dan, dann, cdan)
│   ├── methods/                       # UDA loss implementations
│   ├── models/                        # ResNet-18 backbone and domain discriminator
│   └── evaluation/                    # Domain separability and class analysis
│
├── task3/                             # Task 3 code
│   ├── train.py                       # Training pipeline for DG models
│   ├── evaluate_sketch.py             # Evaluation script on unseen Sketch
│   ├── configs/                       # Configuration files (erm, dan_dg, sam)
│   ├── methods/                       # DG methods (ERM, DAN-DG, SAM)
│   ├── models/                        # ResNet-18 backbone
│   └── evaluation/                    # Source domain separability and sharpness proxy
│
├── task4/                             # Task 4 code
│   ├── train.py                       # Training script for Vanilla, GCSC, PROSER
│   ├── evaluate_osr.py                # Evaluation on Near, Far, All unknowns
│   ├── configs/                       # Configuration files (vanilla, gcsc, proser)
│   ├── data/                          # CIFAR-10 and CIFAR-100 loaders
│   ├── models/                        # CIFAR ResNet-18 model
│   ├── scores/                        # MSP, MLS, Energy, Mahalanobis scorers
│   └── evaluation/                    # AUROC, FPR@95TPR, failure analysis
│
├── results/                           # Saved JSON result files
│   ├── task1_results.json
│   ├── task2_results.json
│   ├── task3_main_results.json
│   └── task4_results.json
│
├── figures/                           # Generated figures embedded in report
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
│
└── Task[1-4]_*.ipynb                  # Standalone Jupyter Notebooks
```

---

## Environment Setup

Requirements: Python 3.10+ and PyTorch 2.0+ with CUDA support.

```bash
# Clone the repository
git clone https://github.com/FayyazAbdullah2005/PA1-ATML.git
cd PA1-ATML

# Create virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\activate      # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

- **STL-10 (Task 1):** Downloaded automatically through `torchvision.datasets.STL10` to `data/`.
- **PACS (Tasks 2 & 3):** Download and extract PACS into `data/PACS/` with domain folders:
  ```
  data/PACS/
  ├── art_painting/
  ├── cartoon/
  ├── photo/
  └── sketch/
  ```
- **CIFAR-10 & CIFAR-100 (Task 4):** Downloaded automatically through `torchvision.datasets` to `data/`.

---

## Reproduction Instructions

Each task can be executed directly via Python scripts or through the corresponding Jupyter Notebook.

### Task 1: Inductive Biases
```bash
python task1/run_task1.py
# Or run Task1_Inductive_Biases.ipynb
```
Outputs are saved to `results/task1_results.json` and `figures/task1/`.

### Task 2: Domain Adaptation
```bash
# Train models
python task2/train.py

# Evaluate on Sketch
python task2/evaluate_final.py
# Or run Task2_Domain_Adaptation.ipynb
```
Outputs are saved to `results/task2_results.json` and `figures/task2/`.

### Task 3: Domain Generalization
```bash
# Train models on source domains only
python task3/train.py

# Evaluate on unseen Sketch
python task3/evaluate_sketch.py
# Or run Task3_Domain_Generalization.ipynb
```
Outputs are saved to `results/task3_main_results.json` and `figures/task3/`.

### Task 4: Open-Set Recognition
```bash
# Train models on CIFAR-10
python task4/train.py

# Evaluate novelty scores and models on unknowns
python task4/evaluate_osr.py

# Generate failure plots and score distributions
python generate_task4_artifacts.py
# Or run Task4_Open_Set_Recognition.ipynb
```
Outputs are saved to `results/task4_results.json` and `figures/task4/`.

---

## Experimental Protocol and Integrity

- **Task 2 (Domain Adaptation):** Model checkpoint selection was performed exclusively using mean source validation Macro-F1 across Photo, Art, and Cartoon. Target Sketch labels were never used during training or checkpoint selection.
- **Task 3 (Domain Generalization):** Sketch images were strictly withheld during training, validation, early stopping, and hyperparameter selection. Diagnostics were computed solely on source validation sets.
- **Task 4 (Open-Set Recognition):** CIFAR-100 near and far unknown images were strictly evaluation-only. Unknown examples were never used in training or threshold calibration. Decision thresholds were calibrated on CIFAR-10 validation unknownness scores at the 95th percentile.

---

## External Code and Attribution

This project uses the following external libraries and reference implementations:

1. **PyTorch & Torchvision:** Deep learning framework, datasets, and pretrained ImageNet models (ResNet-50, ViT-B/16, ResNet-18).
2. **OpenCLIP:** Open-source implementation of CLIP (Radford et al., 2021; Cherti et al., 2023) for ViT-B/32 zero-shot and linear probing.
3. **AdaIN Style Transfer:** Implementation based on Huang & Belongie (ICCV 2017) and Geirhos et al. (ICLR 2019) for cue-conflict image generation.
4. **DAN, DANN, CDAN:** Domain adaptation discrepancy and adversarial loss implementations based on Long et al. (ICML 2015), Ganin et al. (JMLR 2016), and Long et al. (NeurIPS 2018).
5. **SAM:** Sharpness-Aware Minimization optimizer based on Foret et al. (ICLR 2021).
6. **GCSC & PROSER:** Open-set recognition methods based on Vaze et al. (ICLR 2022) and Zhou et al. (CVPR 2021).
7. **Post-Hoc Novelty Scoring:** Scoring formulations based on Hendrycks & Gimpel (ICLR 2017) for MSP, Liu et al. (NeurIPS 2020) for Energy, and Lee et al. (NeurIPS 2018) for Mahalanobis distance.
