# EE-5102 / CS-6304: Advanced Topics in Machine Learning
## Programming Assignment 1: Learning Beyond IID and Closed-Set Assumptions

**Author:** Abdullah Fayyaz  
**Email:** 28100069@lums.edu.pk  
**Institution:** Lahore University of Management Sciences (LUMS)  
**Repository:** [https://github.com/FayyazAbdullah2005/PA1-ATML](https://github.com/FayyazAbdullah2005/PA1-ATML)  
**Report Document:** `report_skeleton.tex` (8-page NeurIPS 2026 format)

---

## 📌 Overview

This repository contains the complete experimental code, training protocols, evaluation pipelines, machine-readable results, and LaTeX report for **Programming Assignment 1: Beyond IID and Closed-Set Assumptions**.

The project investigates four core dimensions of visual representation learning beyond standard assumptions:
1. **Task 1: Inductive Biases & Feature Representations** — Evaluating ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 on STL-10 under color shifts, AdaIN cue conflicts, spatial translations, and patch scrambling.
2. **Task 2: Unsupervised Domain Adaptation (UDA)** — Evaluating Source-only ERM, DAN (MMD), DANN (Adversarial), and CDAN (Conditional Adversarial) on PACS with Sketch as the unlabeled target domain.
3. **Task 3: Domain Generalization (DG)** — Evaluating unaligned ERM, Pairwise Source Alignment (DAN-DG), and Parameter-Space Stability (SAM) on PACS with Sketch strictly held-out as an unseen domain.
4. **Task 4: Open-Set Recognition (OSR)** — Evaluating post-hoc novelty scoring (MSP, MLS, Energy, Mahalanobis) and trained-model paradigms (Vanilla, GCSC, PROSER) on CIFAR-10 knowns vs. CIFAR-100 Near and Far semantic unknowns.

---

## 📁 Repository Structure

```
PA1-ATML/
├── README.md                          # Top-level reproduction guide and attribution
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Excludes raw data, caches, and large checkpoints
├── report_skeleton.tex                # Complete 8-page NeurIPS report
├── neurips_2026.sty                   # NeurIPS LaTeX stylesheet
│
├── common/                            # Shared utilities across all tasks
│   ├── seed.py                        # Deterministic seed setting (SEED = 6304)
│   ├── logging.py                     # Standardized experiment logger
│   └── plotting.py                    # Shared visualization utilities
│
├── shared/                            # Shared protocols across Tasks 2 & 3
│   ├── pacs.py                        # PACS class labels and domain definitions
│   └── pacs_protocol.py               # Deterministic 80/20 PACS splits and batch samplers
│
├── task1/                             # Task 1: Inductive Biases
│   ├── run_task1.py                   # Master reproduction script for Task 1
│   ├── models.py                      # ResNet-50, ViT-B/16, OpenCLIP wrappers
│   ├── data_utils.py                  # STL-10 balanced 500-sample test subset
│   ├── transforms.py                  # Grayscale, Hue rotation, translation, patch scrambling
│   ├── cue_conflicts.py               # AdaIN style transfer and visual rejection filters
│   └── evaluate.py                    # Bias, coverage, consistency, and cosine stability
│
├── task2/                             # Task 2: Unsupervised Domain Adaptation
│   ├── train.py                       # UDA training pipeline with source-val early stopping
│   ├── evaluate_final.py              # Benchmark evaluation on target Sketch
│   ├── configs/                       # Hyperparameter configs (source_only, dan, dann, cdan)
│   ├── methods/                       # Method implementations (DAN, DANN, CDAN, SourceOnly)
│   ├── models/                        # PACS ResNet-18 backbone and domain discriminators
│   └── evaluation/                    # Domain separability, metrics, class analysis
│
├── task3/                             # Task 3: Domain Generalization
│   ├── train.py                       # DG training pipeline (source-only validation)
│   ├── evaluate_sketch.py             # Evaluation on strictly unseen Sketch domain
│   ├── configs/                       # Hyperparameter configs (erm, dan_dg, sam ablations)
│   ├── methods/                       # DG methods (ERM, DAN-DG, SAM)
│   ├── models/                        # PACS ResNet-18 backbone
│   └── evaluation/                    # Source separability and standardized sharpness proxy
│
├── task4/                             # Task 4: Open-Set Recognition
│   ├── train.py                       # Training pipeline for Vanilla, GCSC, and PROSER
│   ├── evaluate_osr.py                # Benchmark evaluation on Near/Far/All unknowns
│   ├── configs/                       # Hyperparameter configs (vanilla, gcsc, proser)
│   ├── data/                          # CIFAR-10 splits and CIFAR-100 near/far unknowns
│   ├── models/                        # CIFAR-adapted ResNet-18 backbone
│   ├── scores/                        # Post-hoc novelty scorers (MSP, MLS, Energy, Mahalanobis)
│   └── evaluation/                    # AUROC, FPR@95TPR calibration, failure analysis
│
├── results/                           # Machine-readable JSON results
│   ├── task1_results.json             # Task 1 clean, color, bias, spatial, and cosine metrics
│   ├── task2_results.json             # Task 2 UDA metrics, per-class recalls, controlled study
│   ├── task3_main_results.json        # Task 3 DG metrics, per-class recalls, SAM ablation
│   └── task4_results.json             # Task 4 OSR metrics (AUROC, FPR@95TPR, CSA)
│
├── figures/                           # Generated figures embedded in LaTeX report
│   ├── task1/                         # Cue conflicts, translations, patch shuffles, UMAPs
│   ├── task2/                         # Loss curves, alignment dynamics, confusion failures
│   ├── task3/                         # DG loss trajectories, sharpness vs. accuracy scatter
│   └── task4/                         # Score distributions, ROC curves, failure cases
│
└── Task[1-4]_*.ipynb                  # Standalone interactive Jupyter Notebooks for each task
```

---

## 🛠️ Environment Setup

Python 3.10+ and PyTorch 2.0+ with CUDA support are recommended.

```bash
# 1. Clone the repository
git clone https://github.com/FayyazAbdullah2005/PA1-ATML.git
cd PA1-ATML

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# or: .\venv\Scripts\Activate.ps1 # On Windows PowerShell

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

The scripts automatically download and structure datasets where applicable. Prepare directories as follows:

- **STL-10 (Task 1):** Downloaded automatically via `torchvision.datasets.STL10(root='data', download=True)`.
- **PACS (Tasks 2 & 3):** Place the uncompressed PACS dataset in `data/PACS/` with domain subfolders:
  ```
  data/PACS/
  ├── art_painting/
  ├── cartoon/
  ├── photo/
  └── sketch/
  ```
- **CIFAR-10 & CIFAR-100 (Task 4):** Downloaded automatically via `torchvision.datasets.CIFAR10` and `CIFAR100`.

---

## 🚀 Reproduction Instructions

All reported results, tables, and figures can be reproduced using either the self-contained command-line Python scripts or the interactive Jupyter Notebooks.

### Task 1: Inductive Biases & Feature Representations
```bash
# Run complete Task 1 experiment pipeline (clean probe, color, AdaIN cue-conflict, translation, patch shuffle, UMAPs)
python task1/run_task1.py

# Alternatively, run the interactive notebook:
jupyter notebook Task1_Inductive_Biases.ipynb
```
*Outputs saved to `results/task1_results.json` and `figures/task1/`.*

---

### Task 2: Unsupervised Domain Adaptation (UDA)
```bash
# 1. Train all UDA models (Source-only, DAN, DANN, CDAN, and controlled lambda study)
python task2/train.py

# 2. Run target Sketch evaluation, confusion matrices, and domain separability diagnostics
python task2/evaluate_final.py

# Alternatively, run the interactive notebook:
jupyter notebook Task2_Domain_Adaptation.ipynb
```
*Outputs saved to `results/task2_results.json` and `figures/task2/`.*

---

### Task 3: Domain Generalization (DG)
```bash
# 1. Train DG models without target access (ERM, DAN-DG, SAM rho=0.05, 0.01, 0.10)
python task3/train.py

# 2. Run post-hoc evaluation on unseen Sketch and compute sharpness diagnostics
python task3/evaluate_sketch.py

# Alternatively, run the interactive notebook:
jupyter notebook Task3_Domain_Generalization.ipynb
```
*Outputs saved to `results/task3_main_results.json` and `figures/task3/`.*

---

### Task 4: Open-Set Recognition (OSR)
```bash
# 1. Train Vanilla ResNet-18, GCSC (RandAugment), and PROSER on CIFAR-10
python task4/train.py

# 2. Evaluate post-hoc novelty scores (MSP, MLS, Energy, Mahalanobis) and trained models
python task4/evaluate_osr.py

# 3. Generate ROC curves, score distribution plots, and qualitative failure artifacts
python generate_task4_artifacts.py

# Alternatively, run the interactive notebook:
jupyter notebook Task4_Open_Set_Recognition.ipynb
```
*Outputs saved to `results/task4_results.json` and `figures/task4/`.*

---

## 🔬 Scientific & Protocol Integrity

In strict adherence to assignment requirements:
- **No Target Leakage in Task 2:** Checkpoint selection and early stopping were computed exclusively on mean source validation Macro-F1 (`Photo`, `Art Painting`, `Cartoon`). Target Sketch labels were never accessed during training or model selection.
- **Zero Target Access in Task 3:** Sketch images were strictly withheld during training, validation, early stopping, and hyperparameter tuning. Diagnostics (source separability and sharpness proxy) were calculated exclusively on source domains.
- **Independent Task 3 Settings:** Hyperparameters for Task 3 were established using canonical defaults and prescribed ablations ($\rho \in \{0.01, 0.05, 0.10\}$) without conditioning on Task 2 Sketch test results.
- **Strict OSR Evaluation Protocol in Task 4:** Real unknown classes (CIFAR-100 Near and Far subsets) were never seen during training or threshold selection. Decision thresholds $\tau$ were calibrated strictly at the $95^{\text{th}}$ percentile of known CIFAR-10 validation unknownness scores.

---

## 📜 Attribution of Materially Reused External Code

This codebase makes use of the following open-source libraries, architectures, and foundational implementations:

1. **PyTorch & Torchvision:**
   - Deep learning framework, data loaders, and pretrained ImageNet backbones (`ResNet50_Weights.IMAGENET1K_V2`, `ViT_B_16_Weights.IMAGENET1K_V1`, `ResNet18_Weights.IMAGENET1K_V1`).
   - Official repository: [https://github.com/pytorch/vision](https://github.com/pytorch/vision)
2. **OpenCLIP:**
   - Pretrained multimodal contrastive Vision Transformer (`ViT-B-32`, OpenAI weights) used for linear probing and zero-shot evaluation (Radford et al., 2021; Cherti et al., 2023).
   - Official repository: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
3. **AdaIN Style Transfer (Cue-Conflict Synthesis):**
   - Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization based on Huang & Belongie (ICCV 2017) and Geirhos et al. (ICLR 2019).
4. **Domain Adaptation Baselines (DAN, DANN, CDAN):**
   - MMD discrepancy implementation based on Deep Adaptation Networks (Long et al., ICML 2015).
   - Domain Adversarial Neural Networks (Ganin et al., JMLR 2016) and Conditional Adversarial Domain Adaptation (Long et al., NeurIPS 2018).
5. **Sharpness-Aware Minimization (SAM):**
   - Standard non-adaptive SAM optimizer implementation based on Foret et al. (ICLR 2021).
   - Reference: [https://github.com/davda54/sam](https://github.com/davda54/sam)
6. **Open-Set Recognition (GCSC & PROSER):**
   - Good Closed-Set Classifier (GCSC) RandAugment training recipe based on Vaze et al. (ICLR 2022).
   - PROSER placeholder learning via manifold mixup between second-most-likely classes based on Zhou et al. (CVPR 2021).
7. **Post-Hoc OOD / Novelty Scorers:**
   - Maximum Softmax Probability (MSP): Hendrycks & Gimpel (ICLR 2017).
   - Energy-based OOD Detection: Liu et al. (NeurIPS 2020).
   - Mahalanobis Distance Scorer: Lee et al. (NeurIPS 2018).
