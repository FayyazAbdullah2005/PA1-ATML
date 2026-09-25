# Task 4: Open-Set Recognition (OSR) on CIFAR-10 / CIFAR-100

This directory implements the Open-Set Recognition benchmarks (CIFAR-10 knowns, CIFAR-100 Near and Far semantic unknowns).

## Directory Structure
```
task4/
├── configs/          # YAML configs for Vanilla, GCSC, and PROSER
├── data/             # CIFAR-10 and CIFAR-100 semantic unknown loaders
├── evaluation/       # AUROC and FPR-at-95%-TPR evaluation routines
├── methods/          # Vanilla, GCSC (RandAugment), and PROSER (placeholders)
├── models/           # ResNet-18 adapted for 32x32 CIFAR inputs
├── results/          # task4_results.json
├── scores/           # MSP, MLS, Energy, Mahalanobis, and PROSER scorers
├── evaluate_osr.py   # Full post-hoc and trained-model OSR evaluation
├── train.py          # OSR training script
└── README.md
```

## Reproducing Experiments

### 1. Training Models
```bash
# Vanilla ResNet-18
python -m task4.train --config task4/configs/vanilla.yaml

# GCSC (RandAugment)
python -m task4.train --config task4/configs/gcsc.yaml

# PROSER (Placeholders)
python -m task4.train --config task4/configs/proser.yaml
```

### 2. Evaluation
```bash
python -m task4.evaluate_osr
```
Results will be exported to `task4/results/task4_results.json` and figures to `figures/task4/`.
