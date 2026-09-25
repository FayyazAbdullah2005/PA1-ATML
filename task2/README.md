# Task 2: Unsupervised Domain Adaptation (UDA) on PACS

This directory implements the transductive Unsupervised Domain Adaptation (UDA) benchmarks for the PACS dataset (Photo, Art Painting, Cartoon $\to$ Sketch) as specified in `ATML-PA1.pdf`.

## Directory Structure
```
task2/
├── configs/               # Hyperparameter configurations
│   ├── base.yaml          # Shared dataset, optimizer, and protocol settings
│   ├── source_only.yaml   # Source-only ERM baseline configuration
│   ├── dan.yaml           # Deep Adaptation Network (MMD) configuration
│   ├── dann.yaml          # Domain-Adversarial Neural Network configuration
│   └── cdan.yaml          # Conditional Domain Adversarial Network configuration
├── models/                # Network architectures
│   ├── backbone.py        # Pretrained ResNet-18 with frozen ImageNet BatchNorm stats
│   ├── classifier_head.py # 7-class linear classifier head
│   └── domain_discriminator.py # Binary discriminator and Gradient Reversal Layer (GRL)
├── methods/               # Adaptation algorithms
│   ├── source_only.py     # ERM baseline training logic
│   ├── dan.py             # Multi-kernel RBF Maximum Mean Discrepancy (MMD)
│   ├── dann.py            # Adversarial domain adaptation with gradient clipping
│   └── cdan.py            # Multilinear conditioning g(x) = vec(f (x) p) / sqrt(d)
├── evaluation/            # Evaluation metrics and diagnostics
│   ├── metrics.py         # Top-1 accuracy, macro-F1, and per-class accuracy
│   ├── domain_separability.py # Balanced Logistic Regression domain classifier (C=1)
│   └── class_analysis.py  # Positive/negative transfer per class & confusion analysis
├── train.py               # Unified CLI training script
├── evaluate_final.py      # Transductive evaluation and results generator
└── README.md
```

## Reproducing Experiments

### 1. Training Individual Models
```bash
# Source-only ERM baseline (also used as Task 3 ERM baseline)
python -m task2.train --method source_only --checkpoint checkpoints/pacs_erm_baseline.pt

# DAN (MMD alignment, lambda=1.0)
python -m task2.train --method dan --lambda_mmd 1.0 --checkpoint checkpoints/pacs_dan.pt

# DANN (Adversarial alignment with GRL)
python -m task2.train --method dann --checkpoint checkpoints/pacs_dann.pt

# CDAN (Class-conditional adversarial alignment)
python -m task2.train --method cdan --checkpoint checkpoints/pacs_cdan.pt
```

### 2. Final Transductive Evaluation
```bash
python -m task2.evaluate_final
```
Results will be exported to `results/task2_results.json` and figures to `figures/task2/`.
