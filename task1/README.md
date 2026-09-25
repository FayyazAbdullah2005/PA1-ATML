# Task 1: Inductive Biases and Feature Representations

This directory implements the experiments evaluating inductive biases and feature representations across ResNet-50, ViT-B/16, and OpenCLIP ViT-B-32 on STL-10.

## Directory Structure
```
task1/
├── adain/            # AdaIN style transfer network and functions
├── cue_conflicts.py  # Cue-conflict generation and SSIM edge filtering
├── data_utils.py     # STL-10 dataset utilities and normalizations
├── evaluate.py       # Metrics, shape bias, consistency, and plotting routines
├── models.py         # Backbone extractors and linear probe head
├── results/          # task1_results.json
├── run_task1.py      # End-to-end experiment pipeline
├── transforms.py     # Color, translation, and patch shuffling transforms
└── README.md
```

## Reproducing Experiments

Run the complete Task 1 pipeline:
```bash
python -m task1.run_task1
```
Results will be exported to `task1/results/task1_results.json` and figures to `figures/task1/`.
