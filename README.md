# EE-5102 / CS-6304: Advanced Topics in Machine Learning
## Programming Assignment 1: Beyond IID and Closed-Set Assumptions

**Author:** Abdullah Fayyaz  
**Email:** 28100069@lums.edu.pk  
**Institution:** Lahore University of Management Sciences (LUMS)  
**Repository:** https://github.com/FayyazAbdullah2005/PA1-ATML  
**Report Document:** report/report.tex (NeurIPS 2026 format)

---

## Overview

This repository contains the code, configuration files, evaluation scripts, machine-readable results, and LaTeX report for Programming Assignment 1.

The project investigates representation learning beyond standard IID and closed-set assumptions across four experimental settings:
1. **Task 1: Inductive Biases and Feature Representations** - Evaluates ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 on STL-10 under color interventions (grayscale, hue rotation), AdaIN cue conflicts, spatial translations, and patch shuffling.
2. **Task 2: Unsupervised Domain Adaptation (UDA)** - Evaluates Source-only ERM, DAN (MMD), DANN (Adversarial), and CDAN (Conditional Adversarial) on PACS with Sketch as the unlabeled target domain.
3. **Task 3: Domain Generalization (DG)** - Evaluates unaligned ERM, Pairwise Source Alignment (DAN-DG), and Parameter-Space Stability (SAM) on PACS with Sketch held out as an unseen test domain.
4. **Task 4: Open-Set Recognition (OSR)** - Evaluates post-hoc novelty scoring functions (MSP, MLS, Energy, Mahalanobis) and trained-model methods (Vanilla ResNet-18, GCSC, PROSER) on CIFAR-10 vs CIFAR-100 Near and Far semantic unknowns.

---

## Repository Structure

```
pa1-beyond-iid/
├── README.md                          # Repository documentation and attribution
├── requirements.txt                   # Pinned Python dependencies
├── .gitignore                         # Excludes raw data, caches, and checkpoints
│
├── common/                            # Shared utilities across tasks
│   ├── seed.py                        # Deterministic seed setting (SEED = 6304)
│   ├── logging.py                     # Logger initialization
│   ├── plotting.py                    # Standardized plotting utilities
│   ├── metrics.py                     # Accuracy and Macro-F1 computation
│   ├── pacs.py                        # PACS class names and dataset transforms
│   ├── pacs_protocol.py               # Shared PACS 80/20 stratified split loader
│   └── splits/                        # Split index records
│       └── pacs_sketch_seed6304.json
│
├── task1/                             # Task 1: Inductive Biases
│   ├── adain/                         # AdaIN style transfer network and functions
│   ├── cue_conflicts.py               # Cue-conflict generation and SSIM edge filtering
│   ├── data_utils.py                  # STL-10 500-sample test loader
│   ├── evaluate.py                    # Shape bias, consistency, and plotting
│   ├── models.py                      # ResNet-50, ViT-B/16, OpenCLIP models
│   ├── run_task1.py                   # End-to-end execution pipeline
│   ├── transforms.py                  # Color, translation, patch shuffle transforms
│   ├── results/                       # Machine-readable output
│   │   └── task1_results.json
│   └── README.md
│
├── task2/                             # Task 2: Unsupervised Domain Adaptation
│   ├── configs/                       # Hyperparameter configurations (base, source_only, dan, dann, cdan)
│   ├── evaluation/                    # Domain separability and class analysis
│   ├── methods/                       # UDA loss implementations (ERM, DAN, DANN, CDAN)
│   ├── models/                        # Pretrained ResNet-18 and domain discriminator
│   ├── train.py                       # Training pipeline
│   ├── evaluate_final.py              # Evaluation script on Sketch target
│   ├── run_task2.py                   # Master runner script
│   ├── results/                       # Machine-readable output
│   │   └── task2_results.json
│   └── README.md
│
├── task3/                             # Task 3: Domain Generalization
│   ├── configs/                       # Configuration files (base, erm, dan_dg, sam_rho_*)
│   ├── evaluation/                    # Source domain separability and sharpness proxy
│   ├── methods/                       # DG methods (ERM, DAN-DG, SAM)
│   ├── models/                        # ResNet-18 backbone and classifier head
│   ├── train.py                       # DG training script
│   ├── evaluate_sketch.py             # Evaluation script on unseen Sketch
│   ├── results/                       # Machine-readable output
│   │   └── task3_main_results.json
│   └── README.md
│
├── task4/                             # Task 4: Open-Set Recognition
│   ├── configs/                       # Configuration files (vanilla, gcsc, proser)
│   ├── data/                          # CIFAR-10 and CIFAR-100 unknown loaders
│   ├── evaluation/                    # AUROC and FPR-at-95%-TPR evaluation routines
│   ├── methods/                       # Vanilla, GCSC, and PROSER implementations
│   ├── models/                        # CIFAR-adapted ResNet-18
│   ├── scores/                        # MSP, MLS, Energy, Mahalanobis, and PROSER scorers
│   ├── train.py                       # OSR training script
│   ├── evaluate_osr.py                # Evaluation on Near, Far, and All unknowns
│   ├── results/                       # Machine-readable output
│   │   └── task4_results.json
│   └── README.md
│
├── report/                            # LaTeX Report
│   ├── report.tex                     # NeurIPS report source
│   ├── report_skeleton.tex            # Alternative report entrypoint
│   └── neurips_2026.sty               # NeurIPS LaTeX stylesheet
│
└── figures/                           # Generated figures embedded in report
    ├── task1/
    ├── task2/
    ├── task3/
    └── task4/
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

Each task can be executed directly via command-line Python scripts:

### Task 1: Inductive Biases
```bash
python -m task1.run_task1
```
Outputs are saved to `task1/results/task1_results.json` and `figures/task1/`.

### Task 2: Domain Adaptation
```bash
# Train individual models
python -m task2.train --method source_only --checkpoint checkpoints/pacs_erm_baseline.pt
python -m task2.train --method dan --lambda_mmd 1.0 --checkpoint checkpoints/pacs_dan.pt
python -m task2.train --method dann --checkpoint checkpoints/pacs_dann.pt
python -m task2.train --method cdan --checkpoint checkpoints/pacs_cdan.pt

# Evaluate on Sketch target
python -m task2.evaluate_final

# Or run the complete automated pipeline:
python -m task2.run_task2
```
Outputs are saved to `task2/results/task2_results.json` and `figures/task2/`.

### Task 3: Domain Generalization
```bash
# Train models on source domains only
python -m task3.train --config task3/configs/erm.yaml --checkpoint checkpoints/pacs_erm_baseline.pt
python -m task3.train --config task3/configs/dan_dg.yaml --checkpoint checkpoints/task3_DAN-DG.pt
python -m task3.train --config task3/configs/base.yaml --checkpoint checkpoints/task3_SAM.pt

# SAM perturbation study
python -m task3.train --config task3/configs/sam_rho_0.01.yaml --checkpoint checkpoints/task3_SAM_rho0.01.pt
python -m task3.train --config task3/configs/sam_rho_0.1.yaml --checkpoint checkpoints/task3_SAM_rho0.1.pt

# Evaluate on unseen Sketch
python -m task3.evaluate_sketch
```
Outputs are saved to `task3/results/task3_main_results.json` and `figures/task3/`.

### Task 4: Open-Set Recognition
```bash
# Train models on CIFAR-10
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml

# Evaluate novelty scores and models on unknowns
python -m task4.evaluate_osr
```
Outputs are saved to `task4/results/task4_results.json` and `figures/task4/`.

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
