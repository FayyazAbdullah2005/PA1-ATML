import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision import datasets

from task4.models.resnet_cifar import CIFARResNet18
from task4.data.cifar10 import get_cifar10_datasets
from task4.data.cifar100_unknowns import get_cifar100_unknowns
from task4.scores.msp import compute_msp_score
from task4.scores.mls import compute_mls_score
from task4.scores.energy import compute_energy_score
from task4.scores.mahalanobis import MahalanobisScorer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

os.makedirs('figures/task4', exist_ok=True)

# Datasets
cifar10_data = get_cifar10_datasets(use_randaug=False)
cifar100_data = get_cifar100_unknowns()

loader_val = DataLoader(cifar10_data['val'], batch_size=256, shuffle=False)
loader_test = DataLoader(cifar10_data['test'], batch_size=256, shuffle=False)
loader_near = DataLoader(cifar100_data['near'], batch_size=256, shuffle=False)
loader_far = DataLoader(cifar100_data['far'], batch_size=256, shuffle=False)
loader_train_unaug = DataLoader(cifar10_data['train'], batch_size=256, shuffle=False)

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar100_test_raw = datasets.CIFAR100(root='data', train=False, download=True)
idx_to_cifar100 = {idx: name for name, idx in cifar100_test_raw.class_to_idx.items()}

# Load Vanilla Model
model = CIFARResNet18(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load('checkpoints/task4_vanilla.pt', map_location=DEVICE))
model.eval()

def extract_outputs(loader):
    all_feats = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            feat, logits = model(x, return_feature=True)
            all_feats.append(feat.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_feats), torch.cat(all_logits), torch.cat(all_labels)

print("Extracting features...")
feat_val, logits_val, y_val = extract_outputs(loader_val)
feat_test, logits_test, y_test = extract_outputs(loader_test)
feat_near, logits_near, y_near = extract_outputs(loader_near)
feat_far, logits_far, y_far = extract_outputs(loader_far)
feat_train, _, y_train = extract_outputs(loader_train_unaug)

mah_scorer = MahalanobisScorer(num_classes=10)
mah_scorer.fit(feat_train.to(DEVICE), y_train.to(DEVICE))

scores_dict = {
    'MSP': (compute_msp_score(logits_test).numpy(), 
            compute_msp_score(logits_near).numpy(), 
            compute_msp_score(logits_far).numpy()),
    'MLS': (compute_mls_score(logits_test).numpy(), 
            compute_mls_score(logits_near).numpy(), 
            compute_mls_score(logits_far).numpy()),
    'Energy': (compute_energy_score(logits_test).numpy(), 
               compute_energy_score(logits_near).numpy(), 
               compute_energy_score(logits_far).numpy()),
    'Mahalanobis': (mah_scorer.compute_score(feat_test.to(DEVICE)).cpu().numpy(),
                    mah_scorer.compute_score(feat_near.to(DEVICE)).cpu().numpy(),
                    mah_scorer.compute_score(feat_far.to(DEVICE)).cpu().numpy())
}

# 1. ROC Curves Figure
plt.figure(figsize=(7, 5))
for name, (s_test, s_near, s_far) in scores_dict.items():
    s_unk = np.concatenate([s_near, s_far])
    y_true = np.concatenate([np.zeros_like(s_test), np.ones_like(s_unk)])
    y_scores = np.concatenate([s_test, s_unk])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores) * 100
    plt.plot(fpr, tpr, label=f"{name} (AUROC = {auc:.2f}%)", lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Knowns Misclassified as Unknown)', fontsize=10)
plt.ylabel('True Positive Rate (Unknowns Correctly Identified)', fontsize=10)
plt.title('ROC Curves (Known Test vs. All Unknowns)', fontsize=12)
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('figures/task4/roc_curves.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/task4/roc_curves.png")

# 2. Failure Cases Analysis (Vanilla MLS)
val_mls = compute_mls_score(logits_val).numpy()
tau = float(np.percentile(val_mls, 95))
print(f"Validation threshold tau (95th percentile): {tau:.4f}")

mls_near = compute_mls_score(logits_near).numpy()
mls_far = compute_mls_score(logits_far).numpy()

# Incorrectly accepted unknowns: score <= tau
near_acc_mask = mls_near <= tau
far_acc_mask = mls_far <= tau

near_acc_indices = np.where(near_acc_mask)[0]
far_acc_indices = np.where(far_acc_mask)[0]

print(f"Accepted near unknowns: {len(near_acc_indices)} / {len(mls_near)}")
print(f"Accepted far unknowns: {len(far_acc_indices)} / {len(mls_far)}")

preds_near = logits_near.argmax(dim=1).numpy()
preds_far = logits_far.argmax(dim=1).numpy()

mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

def unnormalize(tensor):
    arr = tensor.cpu().numpy()
    arr = arr * std + mean
    arr = np.clip(arr, 0, 1)
    return np.transpose(arr, (1, 2, 0))

# Select representative failure cases from the top-3 most frequently accepted classes
from collections import Counter

# Near top 3 accepted classes: pickup_truck, bus, wolf
near_counts = Counter([idx_to_cifar100[y_near[i].item()] for i in near_acc_indices])
top_near_classes = [c for c, _ in near_counts.most_common(3)]
selected_near = []
for c_name in top_near_classes:
    c_indices = [i for i in near_acc_indices if idx_to_cifar100[y_near[i].item()] == c_name]
    best_idx = c_indices[np.argmin(mls_near[c_indices])]
    selected_near.append(best_idx)

# Far top 3 accepted classes: wardrobe, mushroom, chair
far_counts = Counter([idx_to_cifar100[y_far[i].item()] for i in far_acc_indices])
top_far_classes = [c for c, _ in far_counts.most_common(3)]
selected_far = []
for c_name in top_far_classes:
    c_indices = [i for i in far_acc_indices if idx_to_cifar100[y_far[i].item()] == c_name]
    best_idx = c_indices[np.argmin(mls_far[c_indices])]
    selected_far.append(best_idx)

# Plot Near Failures
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for i, idx in enumerate(selected_near):
    img_t, _ = cifar100_data['near'][idx]
    img = unnormalize(img_t)
    true_cls = idx_to_cifar100[y_near[idx].item()]
    pred_cls = cifar10_classes[preds_near[idx]]
    u_score = mls_near[idx]
    margin = tau - u_score
    
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}\n$u_{{MLS}}$: {u_score:.2f} (margin: +{margin:.2f})", fontsize=10)

plt.tight_layout()
plt.savefig('figures/task4/near_unknown_failures.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/task4/near_unknown_failures.png")

# Plot Far Failures
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for i, idx in enumerate(selected_far):
    img_t, _ = cifar100_data['far'][idx]
    img = unnormalize(img_t)
    true_cls = idx_to_cifar100[y_far[idx].item()]
    pred_cls = cifar10_classes[preds_far[idx]]
    u_score = mls_far[idx]
    margin = tau - u_score
    
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}\n$u_{{MLS}}$: {u_score:.2f} (margin: +{margin:.2f})", fontsize=10)

plt.tight_layout()
plt.savefig('figures/task4/far_unknown_failures.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/task4/far_unknown_failures.png")

# Print Table Data
print("\n" + "="*50)
print("TABLE DATA FOR tab:task4_failure_cases:")
print("="*50)
for idx in selected_near:
    true_cls = idx_to_cifar100[y_near[idx].item()]
    pred_cls = cifar10_classes[preds_near[idx]]
    u_score = mls_near[idx]
    print(f"Near & {true_cls} & {pred_cls} & {u_score:.2f} & {tau:.2f} & Plausible semantic overlap \\\\")

for idx in selected_far:
    true_cls = idx_to_cifar100[y_far[idx].item()]
    pred_cls = cifar10_classes[preds_far[idx]]
    u_score = mls_far[idx]
    print(f"Far & {true_cls} & {pred_cls} & {u_score:.2f} & {tau:.2f} & Surprising feature alias \\\\")
