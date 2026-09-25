# Task 3: Domain Generalization (DG) on PACS

This directory implements the Domain Generalization benchmarks on PACS (train on Photo, Art Painting, Cartoon; evaluate on unseen Sketch).

## Directory Structure
```
task3/
├── configs/          # YAML configs for ERM, DAN-DG, and SAM (rho=0.01, 0.05, 0.1)
├── evaluation/       # Domain metrics, source domain separability, and sharpness proxy
├── methods/          # ERM, pairwise DAN-DG, and Sharpness-Aware Minimization (SAM)
├── models/           # ResNet-18 backbone and classifier head
├── results/          # task3_main_results.json
├── evaluate_sketch.py # Evaluation script on unseen Sketch domain
├── train.py          # DG training script
└── README.md
```

## Reproducing Experiments

### 1. Training Models
```bash
# ERM baseline
python -m task3.train --config task3/configs/erm.yaml --checkpoint checkpoints/pacs_erm_baseline.pt

# DAN-DG (Pairwise MMD alignment)
python -m task3.train --config task3/configs/dan_dg.yaml --checkpoint checkpoints/task3_DAN-DG.pt

# SAM (rho=0.05 default)
python -m task3.train --config task3/configs/base.yaml --checkpoint checkpoints/task3_SAM.pt

# SAM perturbation study
python -m task3.train --config task3/configs/sam_rho_0.01.yaml --checkpoint checkpoints/task3_SAM_rho0.01.pt
python -m task3.train --config task3/configs/sam_rho_0.1.yaml --checkpoint checkpoints/task3_SAM_rho0.1.pt
```

### 2. Evaluation on Unseen Sketch
```bash
python -m task3.evaluate_sketch
```
Results will be exported to `task3/results/task3_main_results.json` and figures to `figures/task3/`.
