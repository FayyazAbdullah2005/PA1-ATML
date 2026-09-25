import json

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
    md_cell("""# Task 1: Inductive Biases and Feature Representations
**Course:** EE-5102 / CS-6304: Advanced Topics in Machine Learning  
**Assignment:** Programming Assignment 1: Beyond IID and Closed-Set Assumptions  

In this notebook, you will evaluate:
1. **Backbones:** ResNet-50 (`IMAGENET1K_V2`), ViT-B/16 (`IMAGENET1K_V1`), and OpenCLIP ViT-B-32 (`pretrained='openai'`).
2. **Dataset:** STL-10 (stratified 80/20 train/val split and a class-balanced 500-sample test evaluation subset with seed `6304`).
3. **Controlled Interventions:**
   - **Clean Baseline:** Linear probe heads + Zero-shot CLIP (`"a photo of a {class}"`).
   - **Color Bias:** Grayscale and Fixed 180° Hue Rotation.
   - **Shape vs. Texture:** Bidirectional AdaIN cue conflicts across 5 class pairs ($\alpha=0.8$) with an automated pre-evaluation visual rejection rule.
   - **Translation Equivariance:** Cardinal shifts of 0, 8, 16, 32 pixels.
   - **Spatial Permutation:** $4 \times 4$ non-identity patch shuffling.
   - **Representation Analysis:** Cosine stability ($I_T$) and 2D UMAP projections."""),

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
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Set style
sns.set_theme(style='whitegrid', font_scale=1.1)

# Ensure reproducibility
SEED = 6304
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create output directories
os.makedirs('figures/task1', exist_ok=True)
os.makedirs('cache/task1', exist_ok=True)
os.makedirs('results', exist_ok=True)"""),

    md_cell("""---
## Step 2: Dataset Loading & Stratified Splitting (Seed 6304)
Loads the local STL-10 dataset, performs the stratified 80/20 train/val split, and selects the balanced 500-image evaluation subset (50 images per class)."""),

    code_cell("""from task1.data_utils import get_stl10_datasets, STL10_CLASSES

# Load splits (uses data/stl10_fast/ downloaded parquets)
train_sub, val_sub, test_500_sub, test_dataset = get_stl10_datasets()

train_loader = DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=0)
val_loader = DataLoader(val_sub, batch_size=128, shuffle=False, num_workers=0)
test_500_loader = DataLoader(test_500_sub, batch_size=128, shuffle=False, num_workers=0)

# Pre-load clean test evaluation subset into memory
clean_test_imgs = []
clean_test_targets = []
for imgs, tgts in test_500_loader:
    clean_test_imgs.append(imgs)
    clean_test_targets.append(tgts)
clean_test_imgs = torch.cat(clean_test_imgs, dim=0) # [500, 3, 224, 224]
clean_test_targets = torch.cat(clean_test_targets, dim=0) # [500]

print(f"Loaded {len(clean_test_imgs)} clean test evaluation images across classes:")
print(STL10_CLASSES)"""),

    md_cell("""---
## Step 3: Initialize Backbones & Extract Features
Initializes the three frozen backbones:
- **ResNet-50:** Global average pooled feature (2048-d)
- **ViT-B/16:** Final CLS token (768-d)
- **OpenCLIP ViT-B-32:** Normalized image embedding (512-d)"""),

    code_cell("""from task1.models import ResNet50Extractor, ViTB16Extractor, OpenCLIPExtractor, extract_features

print("Loading pretrained backbones onto device...")
rn_ext = ResNet50Extractor().to(DEVICE)
vit_ext = ViTB16Extractor().to(DEVICE)
clip_ext = OpenCLIPExtractor(device=DEVICE)

backbones = {
    'ResNet-50': rn_ext,
    'ViT-B/16': vit_ext,
    'OpenCLIP ViT-B-32': clip_ext
}

# Cache features to disk so you only extract them once
cache_path = 'cache/task1/clean_features.pt'
if os.path.exists(cache_path):
    print(f"Loading cached features from {cache_path}...")
    features_cache = torch.load(cache_path, map_location='cpu')
else:
    print("Extracting train, val, and test clean features...")
    features_cache = {}
    for name, ext in backbones.items():
        print(f"Extracting for {name}...")
        tr_f, tr_y = extract_features(ext, train_loader, DEVICE)
        va_f, va_y = extract_features(ext, val_loader, DEVICE)
        te_f, te_y = extract_features(ext, test_500_loader, DEVICE)
        features_cache[name] = {
            'train': (tr_f, tr_y),
            'val': (va_f, va_y),
            'test': (te_f, te_y)
        }
    torch.save(features_cache, cache_path)
    print("Clean features successfully extracted and cached.")"""),

    md_cell("""---
## Step 4: Clean Baseline Evaluation (Table 1)
Trains linear classifier heads for each backbone using AdamW ($\text{lr}=10^{-3}, \text{wd}=10^{-4}$, early stopping patience 5) and evaluates OpenCLIP zero-shot with `"a photo of a {class}"`."""),

    code_cell("""from task1.models import train_linear_head
from task1.evaluate import compute_metrics

trained_heads = {}
clean_metrics = {}
clean_predictions = {}

print("Training Linear Classification Heads...")
for name in ['ResNet-50', 'ViT-B/16', 'OpenCLIP ViT-B-32']:
    tr_f, tr_y = features_cache[name]['train']
    va_f, va_y = features_cache[name]['val']
    te_f, te_y = features_cache[name]['test']
    
    head, best_val = train_linear_head(
        tr_f, tr_y, va_f, va_y, num_classes=10,
        max_epochs=50, patience=5, lr=1e-3, wd=1e-4, seed=SEED, device=DEVICE
    )
    trained_heads[name] = head
    
    with torch.no_grad():
        logits = head(te_f.to(DEVICE)).cpu()
        acc, f1, conf, preds = compute_metrics(logits, te_y)
        clean_metrics[name] = {'acc': acc, 'f1': f1, 'conf': conf}
        clean_predictions[name] = preds
        print(f"  {name:20s} | Top-1: {acc:.2f}% | Macro-F1: {f1:.4f} | Conf: {conf:.4f}")

# Zero-Shot OpenCLIP
print("\\nEvaluating Zero-Shot OpenCLIP...")
with torch.no_grad():
    clip_te_f, clip_te_y = features_cache['OpenCLIP ViT-B-32']['test']
    text_feats = clip_ext.get_text_classifier().cpu() # [10, 512]
    scale = clip_ext.model.logit_scale.exp().item()
    sims = scale * (clip_te_f @ text_feats.T)
    zs_probs = torch.softmax(sims, dim=-1)
    
    acc, f1, conf, preds = compute_metrics(zs_probs, clip_te_y, is_probs=True)
    clean_metrics['OpenCLIP Zero-Shot'] = {'acc': acc, 'f1': f1, 'conf': conf}
    clean_predictions['OpenCLIP Zero-Shot'] = preds
    print(f"  OpenCLIP Zero-Shot   | Top-1: {acc:.2f}% | Macro-F1: {f1:.4f} | Conf: {conf:.4f}")"""),

    md_cell("""---
## Step 5: Color Bias Interventions (Table 2)
Tests sensitivity to removing color (**Grayscale**) vs. changing color without geometry disruption (**180° Hue Rotation** in HSV)."""),

    code_cell("""from task1.transforms import apply_grayscale, apply_hue_rotation_180
from task1.evaluate import compute_prediction_consistency
from sklearn.metrics import accuracy_score

print("Generating Grayscale and 180° Hue Rotated images...")
gray_imgs = apply_grayscale(clean_test_imgs)
hue_imgs = apply_hue_rotation_180(clean_test_imgs)

# Helper function to classify transformed batches
def evaluate_batch(trans_imgs):
    preds = {}
    feats = {}
    with torch.no_grad():
        for name, ext in backbones.items():
            f_list = []
            for b in range(0, len(trans_imgs), 64):
                f_list.append(ext(trans_imgs[b : b + 64].to(DEVICE)).cpu())
            f_cat = torch.cat(f_list, dim=0)
            feats[name] = f_cat
            logits = trained_heads[name](f_cat.to(DEVICE)).cpu()
            preds[name] = torch.softmax(logits, dim=-1).argmax(dim=-1).numpy()
            
        # CLIP Zero-Shot
        clip_f = feats['OpenCLIP ViT-B-32']
        sims = clip_ext.model.logit_scale.exp().item() * (clip_f @ text_feats.T)
        preds['OpenCLIP Zero-Shot'] = torch.softmax(sims, dim=-1).argmax(dim=-1).numpy()
    return preds, feats

preds_gray, feats_gray = evaluate_batch(gray_imgs)
preds_hue, feats_hue = evaluate_batch(hue_imgs)

model_list = ['ResNet-50', 'ViT-B/16', 'OpenCLIP ViT-B-32', 'OpenCLIP Zero-Shot']
color_results = {}

print("\\n--- Color Intervention Results ---")
for m in model_list:
    cl_acc = clean_metrics[m]['acc']
    
    # Grayscale
    gr_acc = accuracy_score(clean_test_targets.numpy(), preds_gray[m]) * 100.0
    gr_cons = compute_prediction_consistency(clean_predictions[m], preds_gray[m])
    gr_delta = gr_acc - cl_acc
    
    # Hue
    hu_acc = accuracy_score(clean_test_targets.numpy(), preds_hue[m]) * 100.0
    hu_cons = compute_prediction_consistency(clean_predictions[m], preds_hue[m])
    hu_delta = hu_acc - cl_acc
    
    color_results[m] = {
        'gray_delta': gr_delta, 'gray_cons': gr_cons,
        'hue_delta': hu_delta, 'hue_cons': hu_cons
    }
    print(f"{m:20s} | Gray: dAcc={gr_delta:+.2f}%, Cons={gr_cons:.1f}% | Hue: dAcc={hu_delta:+.2f}%, Cons={hu_cons:.1f}%")"""),

    md_cell("""---
## Step 6: Shape vs. Texture Bias via AdaIN Cue Conflicts (Table 3 & Figures)
Generates bidirectional cue conflicts across 5 class pairs using AdaIN ($\alpha=0.8$).  
Applies the objective pre-evaluation filter: $\text{SSIM} \ge 0.35$ and $\text{Sobel Edge Correlation} \ge 0.40$."""),

    code_cell("""from task1.cue_conflicts import generate_cue_conflicts
from task1.evaluate import compute_shape_bias, plot_cue_conflict_figures

cue_cache = 'cache/task1/cue_conflicts.pt'
if os.path.exists(cue_cache):
    print(f"Loading cached cue conflicts from {cue_cache}...")
    cue_data = torch.load(cue_cache, map_location='cpu')
    conflicts = cue_data['conflicts']
    acc_count = cue_data['acc_count']
    rej_count = cue_data['rej_count']
else:
    print("Generating AdaIN cue conflicts...")
    conflicts, acc_count, rej_count = generate_cue_conflicts(
        test_dataset, alpha=0.8, target_per_dir=22, ssim_thresh=0.35, edge_thresh=0.40, device=DEVICE
    )
    torch.save({'conflicts': conflicts, 'acc_count': acc_count, 'rej_count': rej_count}, cue_cache)

valid_conflicts = [c for c in conflicts if c['status'] == 'accepted']
valid_imgs = torch.stack([c['image'] for c in valid_conflicts], dim=0)
content_classes = [c['content_class'] for c in valid_conflicts]
style_classes = [c['style_class'] for c in valid_conflicts]

preds_cue, feats_cue = evaluate_batch(valid_imgs)

print(f"\\n--- Shape vs. Texture Results ({len(valid_conflicts)} valid conflicts) ---")
shape_bias_results = {}
for m in model_list:
    n_shape, n_tex, n_other, s_bias, cov = compute_shape_bias(
        preds_cue[m], content_classes, style_classes
    )
    shape_bias_results[m] = {
        'n_shape': n_shape, 'n_texture': n_tex, 'n_other': n_other,
        'shape_bias': s_bias, 'coverage': cov
    }
    print(f"{m:20s} | Shape: {n_shape:3d}, Texture: {n_tex:3d}, Other: {n_other:3d} | Shape Bias: {s_bias:5.1f}% | Coverage: {cov:5.1f}%")

# Save figures
plot_cue_conflict_figures(conflicts, preds_cue, out_dir='figures/task1')
print("Figures saved: cue_conflict_samples.png, rejection_rule_examples.png, cue_conflict_failures.png")"""),

    md_cell("""---
## Step 7: Translation Sensitivity Curves (Figure 2)
Displaces images by $\delta \in \{0, 8, 16, 32\}$ pixels in North, South, East, West cardinal directions using reflection padding and shifted crops."""),

    code_cell("""from task1.transforms import apply_translation
from task1.evaluate import plot_translation_curves

shifts = [0, 8, 16, 32]
directions = ['north', 'south', 'east', 'west']

trans_acc = {m: [] for m in model_list}
trans_cons = {m: [] for m in model_list}
feats_shift32 = None

print("Evaluating cardinal translations...")
for shift in shifts:
    if shift == 0:
        for m in model_list:
            trans_acc[m].append(clean_metrics[m]['acc'])
            trans_cons[m].append(100.0)
        continue
        
    dir_accs = {m: [] for m in model_list}
    dir_conss = {m: [] for m in model_list}
    last_feats = None
    
    for direction in directions:
        sh_imgs = apply_translation(clean_test_imgs, shift=shift, direction=direction)
        p_sh, f_sh = evaluate_batch(sh_imgs)
        last_feats = f_sh
        for m in model_list:
            dir_accs[m].append(accuracy_score(clean_test_targets.numpy(), p_sh[m]) * 100.0)
            dir_conss[m].append(compute_prediction_consistency(clean_predictions[m], p_sh[m]))
            
    if shift == 32:
        feats_shift32 = last_feats
        
    for m in model_list:
        mean_a = float(np.mean(dir_accs[m]))
        mean_c = float(np.mean(dir_conss[m]))
        trans_acc[m].append(mean_a)
        trans_cons[m].append(mean_c)
        print(f"Shift = {shift:2d} px | {m:20s} Acc: {mean_a:5.2f}% | Consistency: {mean_c:5.2f}%")

plot_translation_curves(shifts, trans_acc, trans_cons, out_dir='figures/task1')
print("Curves saved: figures/task1/translation_accuracy.png and translation_consistency.png")"""),

    md_cell("""---
## Step 8: Patch Structure Permutation ($4 \times 4$ Shuffling)
Disrupts global spatial organization while preserving local patch evidence via a fixed random non-identity permutation (seed 6304)."""),

    code_cell("""from task1.transforms import apply_patch_shuffle_4x4
from task1.evaluate import plot_patch_shuffle_samples

print("Applying 4x4 patch permutations...")
shuffled_imgs = apply_patch_shuffle_4x4(clean_test_imgs, seed=SEED)
preds_shuf, feats_shuf = evaluate_batch(shuffled_imgs)

patch_results = {}
print("\\n--- Patch Permutation Results ---")
for m in model_list:
    cl_acc = clean_metrics[m]['acc']
    sh_acc = accuracy_score(clean_test_targets.numpy(), preds_shuf[m]) * 100.0
    sh_cons = compute_prediction_consistency(clean_predictions[m], preds_shuf[m])
    d_acc = sh_acc - cl_acc
    patch_results[m] = {'shuf_acc': sh_acc, 'delta_acc': d_acc, 'consistency': sh_cons}
    print(f"{m:20s} | Shuffled Acc: {sh_acc:5.2f}% | Delta: {d_acc:+5.2f}% | Consistency: {sh_cons:5.2f}%")

plot_patch_shuffle_samples(clean_test_imgs, shuffled_imgs, clean_predictions, preds_shuf, out_dir='figures/task1')
print("Figure saved: figures/task1/patch_shuffle_examples.png")"""),

    md_cell("""---
## Step 9: Representation Cosine Stability ($I_T$) & 2D UMAP Projections
Measures representation cosine stability:
$$I_T = \\frac{1}{N} \\sum_{i=1}^N \\frac{f(x_i)^T f(T(x_i))}{\\|f(x_i)\\|_2 \\|f(T(x_i))\\|_2}$$
Fits a 2D UMAP projection with Cosine metric combining clean and transformed features."""),

    code_cell("""from task1.evaluate import compute_cosine_stability, plot_umap_embeddings

cosine_stability = {}
print("--- Representation Cosine Stability (IT) ---")
for name in ['ResNet-50', 'ViT-B/16', 'OpenCLIP ViT-B-32']:
    cl_f, _ = features_cache[name]['test']
    
    i_gray = compute_cosine_stability(cl_f, feats_gray[name])
    
    # Cue conflict: compare clean content features with stylized features
    cue_content_imgs = torch.stack([c['content_image'] for c in valid_conflicts], dim=0)
    with torch.no_grad():
        f_c_list = []
        for b in range(0, len(cue_content_imgs), 64):
            f_c_list.append(backbones[name](cue_content_imgs[b:b+64].to(DEVICE)).cpu())
        f_content = torch.cat(f_c_list, dim=0)
    i_cue = compute_cosine_stability(f_content, feats_cue[name])
    
    i_trans = compute_cosine_stability(cl_f, feats_shift32[name])
    i_shuf = compute_cosine_stability(cl_f, feats_shuf[name])
    
    cosine_stability[name] = {
        'grayscale': i_gray, 'cue_conflict': i_cue,
        'translation_32': i_trans, 'patch_shuffle': i_shuf
    }
    print(f"{name:20s} | Gray: {i_gray:.4f} | Cue: {i_cue:.4f} | Trans(32): {i_trans:.4f} | Shuf: {i_shuf:.4f}")

print("\\nFitting 2D UMAP Projections (Cosine metric)...")
umap_files = {
    'ResNet-50': 'umap_resnet50.png',
    'ViT-B/16': 'umap_vitb16.png',
    'OpenCLIP ViT-B-32': 'umap_clip.png'
}
for name, fname in umap_files.items():
    cl_f, targets_t = features_cache[name]['test']
    plot_umap_embeddings(
        cl_f.numpy(), feats_hue[name].numpy(), targets_t.numpy(),
        model_name=name, filename=fname, out_dir='figures/task1', seed=SEED
    )
    print(f"  Saved figures/task1/{fname}")"""),

    md_cell("""---
## Step 10: Export All Results for the LaTeX Report
Serializes all metrics, decisions, and counts into `results/task1_results.json` so you can directly fill in the LaTeX report skeleton tables."""),

    code_cell("""class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

results_all = {
    'clean_metrics': clean_metrics,
    'color_bias': color_results,
    'shape_vs_texture': shape_bias_results,
    'cue_conflict_counts': {'accepted': acc_count, 'rejected': rej_count, 'valid': len(valid_conflicts)},
    'translation': {'shifts': shifts, 'accuracy': trans_acc, 'consistency': trans_cons},
    'patch_shuffling': patch_results,
    'cosine_stability': cosine_stability
}

with open('results/task1_results.json', 'w') as f:
    json.dump(results_all, f, indent=2, cls=NumpyEncoder)

print("Task 1 results successfully exported to results/task1_results.json!")
print("All figures ready in figures/task1/")""")
]

nb = make_notebook(cells)
with open('Task1_Inductive_Biases.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Successfully written Task1_Inductive_Biases.ipynb")
