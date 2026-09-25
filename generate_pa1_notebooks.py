import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

# =============================================================================
# TASK 1 NOTEBOOK
# =============================================================================
t1_cells = [
    md_cell("""# Task 1: Inductive Biases and Feature Representations
**Course:** EE-5102 / CS-6304 - Advanced Topics in Machine Learning (Fall 2026)  
**Objective:** Compare convolutional (ResNet-50), transformer (ViT-B/16), and multimodal vision-language (OpenCLIP ViT-B-32) backbones under controlled image interventions (color, cue-conflict shape/texture, translation, patch shuffling)."""),
    
    code_cell("""import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Fixed global random seed for reproducibility
SEED = 6304

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")"""),

    md_cell("""## 1.1 Dataset Preparation (STL-10)
- Stratified 80/20 training/validation split from official training partition using seed 6304.
- Class-balanced subset of 500 official test images using seed 6304.
- Interventions constructed on common $224 \\times 224$ RGB images prior to backbone normalization."""),

    code_cell("""# Base transform: Resize STL-10 (96x96) to 224x224 RGB without normalization
base_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

data_dir = './data'
train_dataset = torchvision.datasets.STL10(root=data_dir, split='train', download=True, transform=base_transform)
test_dataset = torchvision.datasets.STL10(root=data_dir, split='test', download=True, transform=base_transform)

# Stratified 80/20 split on official train partition
from sklearn.model_selection import train_test_split
train_targets = train_dataset.labels
train_idx, val_idx = train_test_split(
    np.arange(len(train_targets)),
    test_size=0.2,
    stratify=train_targets,
    random_state=SEED
)

# Select balanced 500 test images (50 per class for 10 classes)
test_targets = test_dataset.labels
test_500_idx, _ = train_test_split(
    np.arange(len(test_targets)),
    train_size=500,
    stratify=test_targets,
    random_state=SEED
)

train_sub = Subset(train_dataset, train_idx)
val_sub = Subset(train_dataset, val_idx)
eval_500_sub = Subset(test_dataset, test_500_idx)

print(f"Train samples: {len(train_sub)}, Val samples: {len(val_sub)}, Test Eval subset: {len(eval_500_sub)}")"""),

    md_cell("""## 1.2 Backbones & Feature Extraction Wrappers
- ResNet-50 (`ResNet50_Weights.IMAGENET1K_V2`): Global average pooled feature (2048-d)
- ViT-B/16 (`ViT_B_16_Weights.IMAGENET1K_V1`): Final CLS token representation (768-d)
- OpenCLIP ViT-B-32 (`pretrained='openai'`): Normalized image embedding (512-d)"""),

    code_cell("""# Model wrappers for frozen feature extraction
class ResNet50Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.norm = weights.transforms()
        base = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        feat = self.backbone(x)
        return torch.flatten(feat, 1) # 2048-d

class ViTB16Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        # Extract CLS token from ViT encoder
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        return x[:, 0] # 768-d

class OpenCLIPExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True) # 512-d normalized"""),

    md_cell("""## 1.3 Clean Baseline: Linear Probe Training & Zero-Shot CLIP
- Train linear head for each backbone (AdamW, lr=1e-3, weight decay=1e-4, max 50 epochs, early stopping patience 5).
- Evaluate Zero-Shot CLIP with fixed prompt `"a photo of a {class}"`.
- Report Top-1 accuracy, Macro-F1, and mean maximum confidence on the 500 test subset."""),

    code_cell("""# TODO: Train linear classification heads on extracted train features
# Evaluate on the 500-sample clean evaluation subset
# Fill in Table 1: tab:task1_clean_baseline"""),

    md_cell("""## 1.4 Color Bias: Grayscale and Secondary Transformation
- Common intervention: Grayscale.
- Secondary intervention: Fixed hue rotation or class-swapped color statistics.
- Measure accuracy drop ($\Delta$) and prediction consistency relative to clean images."""),

    code_cell("""def to_grayscale(x):
    # x: [B, 3, H, W] in [0, 1]
    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    return gray.repeat(1, 3, 1, 1)

# TODO: Implement chosen secondary color transformation
# Evaluate consistency and accuracy drop; fill in Table 2: tab:task1_color"""),

    md_cell("""## 1.5 Shape vs. Texture Bias (AdaIN Style Transfer Cue-Conflicts)
- Generate $\ge 200$ valid cue conflicts across $\ge 5$ class pairs using AdaIN.
- Establish visual rejection rule before evaluation (record accepted & rejected counts).
- Compute Shape Bias (%) = $\\frac{N_{\\text{shape}}}{N_{\\text{shape}} + N_{\\text{texture}}} \\times 100$ and Coverage (%) = $\\frac{N_{\\text{shape}} + N_{\\text{texture}}}{N_{\\text{total}}} \\times 100$."""),

    code_cell("""# TODO: Generate and inspect AdaIN cue-conflict images
# Save figures to figures/task1/cue_conflict_samples.png and rejection_rule_examples.png
# Evaluate model predictions on accepted conflicts; fill in Table 3: tab:task1_shape_bias"""),

    md_cell("""## 1.6 Translation Sensitivity
- Displace images by $\\delta \\in \\{0, 8, 16, 32\\}$ pixels in four cardinal directions (reflection padding + shifted crop).
- Plot Top-1 Accuracy and Prediction Consistency $\\text{Consistency}(\\delta)$ against displacement."""),

    code_cell("""# TODO: Implement cardinal translation with reflection padding
# Average results over the four directions
# Plot curves and save to figures/task1/translation_accuracy.png and translation_consistency.png"""),

    md_cell("""## 1.7 Patch Structure Permutation ($4 \\times 4$ Shuffling)
- Divide each image into $4 \\times 4$ grid and apply fixed non-identity permutation (seed 6304).
- Report accuracy drop and prediction consistency; save figure to `figures/task1/patch_shuffle_examples.png`."""),

    code_cell("""# TODO: Implement 4x4 patch permutation
# Record accuracy drop and consistency; fill in Table 4: tab:task1_patch_shuffle"""),

    md_cell("""## 1.8 Representation Analysis & 2D Manifold Visualizations
- Measure cosine feature stability $I_T = \\frac{1}{N} \\sum_{i=1}^N \\frac{f(x_i)^T f(T(x_i))}{\\|f(x_i)\\|_2 \\|f(T(x_i))\\|_2}$.
- Fit 2D t-SNE / UMAP projections on combined clean + transformed features for each backbone.
- Save visualizations to `figures/task1/umap_resnet50.png`, `umap_vitb16.png`, `umap_clip.png`."""),

    code_cell("""# TODO: Extract representations across interventions and compute cosine stability IT
# Fit 2D UMAP projections and save plots for report""")]

# =============================================================================
# TASK 2 NOTEBOOK
# =============================================================================
t2_cells = [
    md_cell("""# Task 2: Unsupervised Domain Adaptation (UDA)
**Course:** EE-5102 / CS-6304 - Advanced Topics in Machine Learning (Fall 2026)  
**Objective:** Evaluate domain alignment strategies (Source-only ERM, DAN, DANN, CDAN) on PACS with Photo, Art Painting, and Cartoon as labeled sources, and Sketch as unlabeled target."""),

    code_cell("""import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 6304

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")"""),

    md_cell("""## 2.1 PACS Dataset & Batch-Balancing Loader
- Sources: Photo (P), Art Painting (A), Cartoon (C) with 80/20 stratified train/val split (seed 6304).
- Target: Sketch (S) unlabeled during adaptation.
- Adaptation batch: 8 Photo + 8 Art + 8 Cartoon (24 source total) and 24 Sketch (24 target total)."""),

    code_cell("""# Image preprocessing
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TODO: Define PACS Dataset loader and create stratified 80/20 splits per source domain"""),

    md_cell("""## 2.2 ResNet-18 Architecture & BatchNorm Policy
- Torchvision ResNet-18 (`IMAGENET1K_V1`) fine-tuned end-to-end with 7-class linear head.
- **BatchNorm Policy:** Freeze all running mean and variance at pretrained ImageNet values (modules in `eval()` mode); keep scale $\\gamma$ and bias $\\beta$ trainable."""),

    code_cell("""def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval() # Freeze running mean & var, affine params remain trainable

class PACSResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1]) # 512-d feature
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = torch.flatten(self.backbone(x), 1)
        logits = self.head(feat)
        return feat, logits

    def train(self, mode=True):
        super().train(mode)
        if mode:
            freeze_bn_stats(self)"""),

    md_cell("""## 2.3 Adaptation Alignment Modules
- **DAN:** Multi-scale RBF kernel MMD on 512-d features (bandwidths 0.5, 1, 2 $\\times$ median distance).
- **DANN:** Binary domain discriminator (512 $\\to$ 256 $\\to$ ReLU $\\to$ Dropout(0.5) $\\to$ 2) with Gradient Reversal Layer schedule $\\alpha(p) = \\frac{2}{1 + \\exp(-10p)} - 1$.
- **CDAN:** Multilinear conditioning $g(x) = \\text{vec}(f \\otimes p)$ with identical discriminator."""),

    code_cell("""from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

# TODO: Implement MMD multi-kernel function and Domain Discriminator module"""),

    md_cell("""## 2.4 Training Pipeline & Alignment Diagnostics
- Train for max 30 epochs (AdamW lr=1e-4, wd=1e-4, early stopping patience 5 on mean source val macro-F1).
- Save best checkpoints; compute domain separability score via 70/30 logistic regression on frozen features.
- Fill in Table 1 (`tab:task2_main_results`) and Table 2 (`tab:task2_per_class`)."""),

    code_cell("""# TODO: Run training for Source-only, DAN, DANN, CDAN
# Save curves to figures/task2/training_loss_curves.png and alignment_loss_curves.png
# Compute domain separability and per-class target breakdown""")]

# =============================================================================
# TASK 3 NOTEBOOK
# =============================================================================
t3_cells = [
    md_cell("""# Task 3: Domain Generalization (DG)
**Course:** EE-5102 / CS-6304 - Advanced Topics in Machine Learning (Fall 2026)  
**Objective:** Investigate whether domain invariance (DAN-DG) or local parameter stability (SAM) improves generalization to an unseen domain (Sketch strictly withheld until final evaluation)."""),

    code_cell("""import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 6304

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")"""),

    md_cell("""## 3.1 Setup & ERM Baseline Reuse
- Sources: Photo, Art Painting, Cartoon (same splits as Task 2).
- **Target Rule:** No Sketch image may be loaded during training, validation, model selection, or hyperparameter choices.
- Load the exact Source-only checkpoint saved from Task 2 as the ERM baseline."""),

    code_cell("""# TODO: Load Task 2 source-only checkpoint as Task 3 ERM baseline
# Evaluate validation metrics on Photo, Art, Cartoon; report worst-source and mean-source values"""),

    md_cell("""## 3.2 Methods: DAN-DG and SAM
- **DAN-DG:** Pairwise MMD alignment across observed sources: $L_{\\text{DAN-DG}} = L_{\\text{ERM}} + \\frac{\\lambda_{\\text{DG}}}{3} \\sum_{e < e'} \\text{MMD}^2(F(X_e), F(X_{e'}))$ with $\\lambda_{\\text{DG}} = 1$.
- **SAM:** Sharpness-Aware Minimization with perturbation radius $\\rho = 0.05$ and frozen BatchNorm statistics."""),

    code_cell("""class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w) # ascent step
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"] # restore
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        stack = [
            ((torch.abs(p.grad) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]
        return torch.norm(torch.stack(stack), p=2) if stack else torch.tensor(0.0)"""),

    md_cell("""## 3.3 Training DAN-DG & SAM and Evaluating Diagnostics
- Diagnostic 1: Source-domain separability (multinomial logistic regression on source features, chance = 33.3%).
- Diagnostic 2: Local sharpness proxy $\\Delta_{\\text{sharp}} = L_{\\text{val}}(\\theta + \\epsilon) - L_{\\text{val}}(\\theta)$ with $\\epsilon = 0.05 \\frac{\\nabla_\\theta L_{\\text{val}}}{\\|\\nabla_\\theta L_{\\text{val}}\\|_2}$ on a fixed batch of 32 examples per source."""),

    code_cell("""# TODO: Train DAN-DG and SAM models
# Measure source-domain separability and compute local sharpness proxy Delta_sharp
# Fill in Table 1: tab:task3_main_results"""),

    md_cell("""## 3.4 Final Evaluation on Unseen Sketch
- Evaluate ERM, DAN-DG, and SAM on the Sketch domain **only after all training decisions are frozen**.
- Fill in per-class breakdown Table 2 (`tab:task3_per_class`) and save plots to `figures/task3/`."""),

    code_cell("""# TODO: Load Sketch dataset and evaluate final zero-shot transfer performance""")]

# =============================================================================
# TASK 4 NOTEBOOK
# =============================================================================
t4_cells = [
    md_cell("""# Task 4: Open-Set Recognition (OSR)
**Course:** EE-5102 / CS-6304 - Advanced Topics in Machine Learning (Fall 2026)  
**Objective:** Compare post-hoc novelty scores (MSP, MLS, Energy, Mahalanobis), data augmentation (GCSC), and placeholder learning (PROSER) for rejecting near and far CIFAR-100 unknowns while preserving CIFAR-10 closed-set accuracy."""),

    code_cell("""import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 6304

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")"""),

    md_cell("""## 4.1 Datasets & Unknown Splits
- **Knowns:** CIFAR-10 (stratified 90/10 train/val split, seed 6304). Complete test set (10,000 images) for known evaluation.
- **Near Unknowns (800 images):** `bus`, `pickup_truck`, `motorcycle`, `tractor`, `wolf`, `fox`, `leopard`, `camel` (100 per class).
- **Far Unknowns (800 images):** `bottle`, `bowl`, `chair`, `clock`, `keyboard`, `mushroom`, `sunflower`, `wardrobe` (100 per class)."""),

    code_cell("""# Preprocessing for 32x32 images
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# TODO: Load CIFAR-10 and extract stratified 90/10 train/val split
# Load CIFAR-100 test set and isolate 800 near-unknown and 800 far-unknown images"""),

    md_cell("""## 4.2 CIFAR-Adapted ResNet-18
- Replace ImageNet $7 \\times 7$ conv with $3 \\times 3$ conv (stride 1, padding 1).
- Remove initial max-pooling layer."""),

    code_cell("""from torchvision.models import resnet18

class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        # Remove maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = torch.flatten(self.avgpool(x), 1)
        logits = self.fc(feat)
        return feat, logits"""),

    md_cell("""## 4.3 Step 1: Vanilla Model Training & Step 2: Post-Hoc Novelty Scoring
- Train Vanilla ResNet-18 (SGD lr=0.1, momentum 0.9, wd=5e-4, cosine decay, 100 epochs, batch size 128, seed 6304).
- Implement novelty scoring functions ($u(x)$ where higher = more unknown):
  - $u_{\\text{MSP}}(x) = 1 - \\max_k p_k(x)$
  - $u_{\\text{MLS}}(x) = -\\max_k z_k(x)$
  - $u_{\\text{Energy}}(x) = -\\log \\sum_k \\exp(z_k(x))$
  - $u_{\\text{Mah}}(x) = \\min_c (f(x) - \\mu_c)^T \\Sigma^{-1} (f(x) - \\mu_c)$"""),

    code_cell("""# TODO: Train Vanilla model and extract logits/features
# Compute MSP, MLS, Energy, and Mahalanobis unknownness scores
# Fill in Table 1: tab:task4_posthoc_scores and save curves to figures/task4/"""),

    md_cell("""## 4.4 Step 3: GCSC (Strong Classifier with RandAugment)
- Same recipe as vanilla, inserting `T.RandAugment(num_ops=2, magnitude=9)` after crop & flip.
- Evaluated with MLS score."""),

    code_cell("""# TODO: Train GCSC model with RandAugment
# Evaluate on CIFAR-10 test set and CIFAR-100 unknowns using MLS"""),

    md_cell("""## 4.5 Step 4: PROSER (Placeholders via Manifold Mixup)
- Append 5 randomly initialized dummy classifiers (total 15 output units).
- Classifier placeholder loss ($\\beta = 1$) on first half of batch.
- Manifold mixup data placeholders ($\\lambda \\sim \\text{Beta}(2, 2)$ between layer2 and layer3, $\\gamma = 0.1$) on second half of batch.
- Fine-tune for 50 epochs from Vanilla checkpoint (lr=1e-3)."""),

    code_cell("""# TODO: Implement PROSER training with manifold mixup
# Fill in Table 2: tab:task4_trained_models"""),

    md_cell("""## 4.6 Step 5: Common Evaluation, Calibration & Failure Inspection
- Calibrate threshold $\\tau$ at 95th percentile of CIFAR-10 validation unknownness (accept when $u(x) \\le \\tau$).
- Inspect 3 incorrectly accepted near-unknowns and 3 incorrectly accepted far-unknowns.
- Fill in Table 3: `tab:task4_failure_cases` and save failure inspection figures to `figures/task4/`."""),

    code_cell("""# TODO: Compute AUROC, calibrated acceptance/rejection rates, and inspect qualitative failure cases""")]

# Write notebooks
notebooks = {
    'Task1_Inductive_Biases.ipynb': t1_cells,
    'Task2_Domain_Adaptation.ipynb': t2_cells,
    'Task3_Domain_Generalization.ipynb': t3_cells,
    'Task4_Open_Set_Recognition.ipynb': t4_cells
}

for fname, cells in notebooks.items():
    nb = make_notebook(cells)
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Created {fname}")
