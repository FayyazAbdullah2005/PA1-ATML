import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

cells = [
    md_cell("""# Task 2: Unsupervised Domain Adaptation (UDA)
**Course:** EE-5102 / CS-6304: Advanced Topics in Machine Learning  
**Assignment:** Programming Assignment 1: Beyond IID and Closed-Set Assumptions  

In this task, we address distribution shift on the **PACS** dataset:
- **Source Domains (Labeled):** Photo (P), Art Painting (A), Cartoon (C) with stratified 80/20 train/val splits (seed `6304`).
- **Target Domain (Unlabeled during adaptation):** Sketch (S).
- **Architecture:** ResNet-18 (`IMAGENET1K_V1`) fine-tuned end-to-end with a 7-class head.
- **BatchNorm Policy:** All BatchNorm running means and variances are frozen at ImageNet values (`bn.eval()`) throughout training; affine parameters ($\gamma, \beta$) remain trainable.
- **Methods Compared:**
  1. **Source-only ERM** (reused as the Task 3 ERM baseline)
  2. **DAN** (Multi-kernel RBF MMD discrepancy minimization)
  3. **DANN** (Adversarial domain discriminator with GRL schedule)
  4. **CDAN** (Class-conditional adversarial alignment $g(x) = \text{vec}(f \otimes p)$)
- **Protocol:** Strict **Two-Phase Execution**:
  - **Phase A:** Train models and select checkpoints strictly using mean source-validation macro-F1.
  - **Phase B:** Lock checkpoints and evaluate on the Sketch target, computing domain separability and negative transfer."""),

    md_cell("""---
## Step 1: Environment Setup, Global Seed & Hardware Detection"""),

    code_cell("""import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Global reproducibility seed
from common.seed import set_seed, SEED
set_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using hardware device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Ensure required workspace directories exist
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('figures/task2', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('cache', exist_ok=True)"""),

    md_cell("""---
## Step 2: PACS Dataset Loading & Balanced Batch Sampler
- Sources: Photo, Art Painting, Cartoon (80/20 stratified split, seed 6304).
- Target: Sketch (unlabeled during adaptation).
- Each adaptation update draws 8 Photo + 8 Art + 8 Cartoon (24 source total) and 24 Sketch (24 target total)."""),

    code_cell("""from shared.pacs import PACS_CLASSES, PACS_DOMAINS
from shared.pacs_protocol import get_pacs_datasets, BalancedDomainBatchSampler

pacs_data = get_pacs_datasets(pacs_root='data/PACS', seed=SEED)

# Dataloaders for training
batch_size_per_source = 8
batch_size_target = 24

loader_p = DataLoader(pacs_data['source_train']['photo'], batch_size=batch_size_per_source, shuffle=True, drop_last=True)
loader_a = DataLoader(pacs_data['source_train']['art_painting'], batch_size=batch_size_per_source, shuffle=True, drop_last=True)
loader_c = DataLoader(pacs_data['source_train']['cartoon'], batch_size=batch_size_per_source, shuffle=True, drop_last=True)
loader_t = DataLoader(pacs_data['target_adapt'], batch_size=batch_size_target, shuffle=True, drop_last=True)

# Validation dataloaders (source domains)
val_loaders = {
    'Photo': DataLoader(pacs_data['source_val']['photo'], batch_size=32, shuffle=False),
    'Art': DataLoader(pacs_data['source_val']['art_painting'], batch_size=32, shuffle=False),
    'Cartoon': DataLoader(pacs_data['source_val']['cartoon'], batch_size=32, shuffle=False)
}

# Target evaluation loader (Sketch) - Strictly for Phase B!
target_eval_loader = DataLoader(pacs_data['target_eval'], batch_size=32, shuffle=False)

print(f"PACS Loaded successfully!")
for d in ['photo', 'art_painting', 'cartoon']:
    print(f"  Source {d:12s}: Train={len(pacs_data['source_train'][d])}, Val={len(pacs_data['source_val'][d])}")
print(f"  Target Sketch       : Total={len(pacs_data['target_eval'])}")
print(f"  Class Names ({len(PACS_CLASSES)}): {PACS_CLASSES}")"""),

    md_cell("""---
## Step 3: Architecture & BatchNorm Freezing Policy
- ResNet-18 with ImageNet pretrained weights and 7-class linear head.
- **BatchNorm Policy:** All BatchNorm running means and variances are frozen at pretrained ImageNet values (`bn.eval()`). Trainable scale ($\gamma$) and bias ($\beta$) remain active."""),

    code_cell("""from task2.models.backbone import freeze_bn_running_stats
from task2.models.classifier_head import PACSResNet18

# Instantiate model
test_model = PACSResNet18(num_classes=7).to(DEVICE)

# Sanity check: verify BatchNorm modules stay in eval mode during train()
test_model.train()
all_bn_eval = all(not m.training for m in test_model.modules() if isinstance(m, nn.BatchNorm2d))
print(f"BatchNorm Freezing Policy Check: All BN modules in eval mode? {all_bn_eval}")
assert all_bn_eval, "BatchNorm modules must remain in eval mode to prevent implicit adaptation!\""""),

    md_cell("""---
## Step 4: Training Pipeline (Phase A - Source Validation Checkpoint Selection)
Trains for at most 30 source epochs using AdamW ($\text{lr}=10^{-4}, \text{wd}=10^{-4}$).  
Early stopping triggers after 5 epochs without improvement in mean source-validation macro-F1.  
Target labels are strictly masked during training!"""),

    code_cell("""from task2.train import train_adaptation"""),

    md_cell("""---
## Step 5: Train Source-Only ERM Baseline
Trains ordinary empirical risk minimization over the 3 source domains.  
**Crucial:** We save this checkpoint to `checkpoints/pacs_erm_baseline.pt` so it can be reused unchanged as the Task 3 ERM baseline!"""),

    code_cell("""print("=== Training Source-Only ERM Baseline ===")
model_source_only, hist_source_only = train_adaptation(
    method='source_only',
    checkpoint_path='checkpoints/pacs_erm_baseline.pt',
    device=DEVICE
)
print("Source-only model saved to checkpoints/pacs_erm_baseline.pt")"""),

    md_cell("""---
## Step 6: Train DAN (MMD Alignment)
Adds the Maximum Mean Discrepancy penalty ($\lambda_{\text{MMD}} = 1$) between source and target representations."""),

    code_cell("""print("=== Training DAN (MMD Alignment) ===")
model_dan, hist_dan = train_adaptation(
    method='dan',
    lambda_mmd=1.0,
    checkpoint_path='checkpoints/pacs_dan.pt',
    device=DEVICE
)"""),

    md_cell("""---
## Step 7: Train DANN (Adversarial Alignment)
Attaches a binary domain discriminator to the 512-d feature with a gradient reversal layer."""),

    code_cell("""print("=== Training DANN (Adversarial Alignment) ===")
model_dann, hist_dann = train_adaptation(
    method='dann',
    checkpoint_path='checkpoints/pacs_dann.pt',
    device=DEVICE
)"""),

    md_cell("""---
## Step 8: Train CDAN (Class-Conditional Alignment)
Conditions the domain discriminator on the multilinear feature $g(x) = \text{vec}(f \otimes p)$."""),

    code_cell("""print("=== Training CDAN (Class-Conditional Alignment) ===")
model_cdan, hist_cdan = train_adaptation(
    method='cdan',
    checkpoint_path='checkpoints/pacs_cdan.pt',
    device=DEVICE
)"""),

    md_cell("""---
## Step 9: Controlled Design Study: DAN Alignment Weight Sweep
Evaluates the effect of varying $\lambda_{\text{MMD}} \in \{0.1, 10\}$ (nominal $\lambda=1.0$ trained in Step 6)."""),

    code_cell("""print("=== Controlled Study: Training DAN with lambda=0.1 ===")
model_dan_low, hist_dan_low = train_adaptation(
    method='dan',
    lambda_mmd=0.1,
    checkpoint_path='checkpoints/pacs_dan_lambda0.1.pt',
    device=DEVICE
)

print("\\n=== Controlled Study: Training DAN with lambda=10.0 ===")
model_dan_high, hist_dan_high = train_adaptation(
    method='dan',
    lambda_mmd=10.0,
    checkpoint_path='checkpoints/pacs_dan_lambda10.0.pt',
    device=DEVICE
)"""),

    md_cell("""---
## Step 10: Plot Training & Alignment Loss Dynamics (Figure 1)
Exports classification loss and alignment/discriminator loss curves to `figures/task2/`."""),

    code_cell("""from common.plotting import plot_training_curves

training_histories = {
    'Source-only': hist_source_only,
    'DAN': hist_dan,
    'DANN': hist_dann,
    'CDAN': hist_cdan
}

# 1. Classification Loss Curves
cls_curves = {m: h['cls_loss'] for m, h in training_histories.items() if 'cls_loss' in h}
plot_training_curves(
    cls_curves,
    title="Source Classification Loss Dynamics",
    xlabel="Epoch",
    ylabel="Classification Loss (Cross-Entropy)",
    save_path="figures/task2/training_loss_curves.png"
)

# 2. Alignment Loss Curves
align_curves = {
    'DAN (MMD)': hist_dan['align_loss'],
    'DANN (Adversarial)': hist_dann['align_loss'],
    'CDAN (Conditional)': hist_cdan['align_loss']
}
plot_training_curves(
    align_curves,
    title="Domain Alignment Objective Dynamics",
    xlabel="Epoch",
    ylabel="Alignment / Discriminator Loss",
    save_path="figures/task2/alignment_loss_curves.png"
)
print("Loss curves saved: figures/task2/training_loss_curves.png and alignment_loss_curves.png")"""),

    md_cell("""---
# PHASE B: Transductive Target Evaluation & Alignment Diagnostics
All models, hyperparameter configurations, and checkpoints are now locked.  
Target (Sketch) labels are accessed solely in the cells below for final evaluation."""),

    md_cell("""---
## Step 11: Benchmark Evaluation on PACS & Domain Separability Diagnostic
Runs transductive evaluation on Sketch, computes domain separability scores, and analyzes per-class negative transfer."""),

    code_cell("""from task2.evaluate_final import evaluate_all_adaptation_models

task2_results = evaluate_all_adaptation_models(
    checkpoints={
        'Source-only (ERM)': 'checkpoints/pacs_erm_baseline.pt',
        'DAN': 'checkpoints/pacs_dan.pt',
        'DANN': 'checkpoints/pacs_dann.pt',
        'CDAN': 'checkpoints/pacs_cdan.pt'
    },
    results_path='results/task2_results.json',
    figures_dir='figures/task2',
    device=DEVICE
)

print("\\n=== Main Benchmark Results (PACS -> Sketch) ===")
print(f"{'Method':20s} | {'Mean Src Acc':12s} | {'Mean Src F1':11s} | {'Sketch Acc':10s} | {'Sketch F1':9s} | {'Gain':7s} | {'Dom Sep':7s}")
print("-" * 90)
for m_name, res in task2_results['main_results'].items():
    print(f"{m_name:20s} | {res['mean_src_acc']:10.2f}% | {res['mean_src_f1']:11.4f} | {res['target_acc']:8.2f}% | {res['target_f1']:9.4f} | {res['target_gain']:+6.2f}% | {res['domain_separability']:5.2f}%")"""),

    md_cell("""---
## Step 12: Controlled Study Table
Evaluates $\lambda_{\text{MMD}} \in \{0.1, 1.0, 10.0\}$."""),

    code_cell("""from task2.evaluation.metrics import evaluate_model
from task2.evaluation.domain_separability import compute_domain_separability

study_ckpts = {
    'DAN (lambda=0.1)': 'checkpoints/pacs_dan_lambda0.1.pt',
    'DAN (lambda=1.0)': 'checkpoints/pacs_dan.pt',
    'DAN (lambda=10.0)': 'checkpoints/pacs_dan_lambda10.0.pt'
}

print(f"{'Configuration':20s} | {'Mean Src Acc':12s} | {'Mean Src F1':11s} | {'Dom Sep':8s} | {'Sketch Acc':10s} | {'Sketch F1':9s}")
print("-" * 80)

for s_name, s_ckpt in study_ckpts.items():
    if os.path.exists(s_ckpt):
        model = PACSResNet18(num_classes=7).to(DEVICE)
        model.load_state_dict(torch.load(s_ckpt, map_location=DEVICE))
        model.eval()
        
        # Source val
        src_accs, src_f1s, src_feats_list = [], [], []
        for _, v_loader in val_loaders.items():
            _, f1, acc, _, _, _, feats = evaluate_model(model, v_loader, DEVICE)
            src_accs.append(acc)
            src_f1s.append(f1)
            src_feats_list.append(feats)
        src_feats = np.concatenate(src_feats_list, axis=0)
        
        # Target Sketch
        _, tgt_f1, tgt_acc, _, _, _, tgt_feats = evaluate_model(model, target_eval_loader, DEVICE)
        dom_sep = compute_domain_separability(src_feats, tgt_feats, seed=SEED)
        
        print(f"{s_name:20s} | {np.mean(src_accs):10.2f}% | {np.mean(src_f1s):11.4f} | {dom_sep:6.2f}% | {tgt_acc:8.2f}% | {tgt_f1:9.4f}")""")
]

nb = make_notebook(cells)
with open("Task2_Domain_Adaptation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Regenerated Task2_Domain_Adaptation.ipynb successfully!")
