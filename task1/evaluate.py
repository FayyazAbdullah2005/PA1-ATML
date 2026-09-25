import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import umap
from task1.data_utils import STL10_CLASSES

sns.set_theme(style='whitegrid', font_scale=1.1)

def compute_metrics(logits_or_probs, targets, is_probs=False):
    if is_probs:
        probs = logits_or_probs
        preds = probs.argmax(dim=-1).cpu().numpy()
        confs = probs.max(dim=-1).values.cpu().numpy()
    else:
        probs = torch.softmax(logits_or_probs, dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()
        confs = probs.max(dim=-1).values.cpu().numpy()
        
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    acc = float(accuracy_score(targets_np, preds) * 100.0)
    f1 = float(f1_score(targets_np, preds, average='macro'))
    mean_conf = float(np.mean(confs))
    return acc, f1, mean_conf, preds

def compute_prediction_consistency(clean_preds, trans_preds):
    clean_p = clean_preds.cpu().numpy() if isinstance(clean_preds, torch.Tensor) else np.array(clean_preds)
    trans_p = trans_preds.cpu().numpy() if isinstance(trans_preds, torch.Tensor) else np.array(trans_preds)
    return float((clean_p == trans_p).mean() * 100.0)

def compute_shape_bias(preds, content_classes, style_classes):
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)
    contents = np.array(content_classes)
    styles = np.array(style_classes)
    
    n_shape = int((preds == contents).sum())
    n_texture = int((preds == styles).sum())
    n_other = int(len(preds) - (n_shape + n_texture))
    n_total = int(len(preds))
    
    denom = n_shape + n_texture
    shape_bias = float(n_shape / denom * 100.0) if denom > 0 else 0.0
    coverage = float(denom / n_total * 100.0) if n_total > 0 else 0.0
    
    return n_shape, n_texture, n_other, shape_bias, coverage

def compute_cosine_stability(feat_clean, feat_trans):
    # feat_clean, feat_trans: [N, D]
    f1 = feat_clean / (feat_clean.norm(dim=-1, keepdim=True) + 1e-8)
    f2 = feat_trans / (feat_trans.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (f1 * f2).sum(dim=-1)
    return float(cos_sim.mean().item())

def plot_translation_curves(shifts, acc_dict, consistency_dict, out_dir='figures/task1'):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Accuracy vs displacement
    plt.figure(figsize=(6.5, 4.5))
    markers = {'ResNet-50': 'o', 'ViT-B/16': 's', 'OpenCLIP Probe': '^', 'OpenCLIP Zero-Shot': 'd'}
    colors = {'ResNet-50': '#1f77b4', 'ViT-B/16': '#ff7f0e', 'OpenCLIP Probe': '#2ca02c', 'OpenCLIP Zero-Shot': '#d62728'}
    
    for model_name, accs in acc_dict.items():
        plt.plot(shifts, accs, label=model_name, marker=markers.get(model_name, 'o'), color=colors.get(model_name, None), linewidth=2)
    plt.xlabel('Displacement $\\delta$ (pixels)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Top-1 Accuracy vs. Spatial Displacement')
    plt.xticks(shifts)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'translation_accuracy.png'), dpi=300)
    plt.close()
    
    # 2. Prediction Consistency vs displacement
    plt.figure(figsize=(6.5, 4.5))
    for model_name, cons in consistency_dict.items():
        plt.plot(shifts, cons, label=model_name, marker=markers.get(model_name, 'o'), color=colors.get(model_name, None), linewidth=2)
    plt.xlabel('Displacement $\\delta$ (pixels)')
    plt.ylabel('Prediction Consistency (%)')
    plt.title('Prediction Consistency vs. Spatial Displacement')
    plt.xticks(shifts)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'translation_consistency.png'), dpi=300)
    plt.close()

def plot_patch_shuffle_samples(clean_imgs, shuffled_imgs, clean_preds_dict, shuffled_preds_dict, out_dir='figures/task1'):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        # Clean
        img_c = clean_imgs[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(img_c)
        axes[0, i].set_title(f"Clean: {STL10_CLASSES[clean_preds_dict['ResNet-50'][i]]}", fontsize=10)
        axes[0, i].axis('off')
        
        # Shuffled
        img_s = shuffled_imgs[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(img_s)
        axes[1, i].set_title(f"Shuffled (4x4)\nRN: {STL10_CLASSES[shuffled_preds_dict['ResNet-50'][i]]}\nViT: {STL10_CLASSES[shuffled_preds_dict['ViT-B/16'][i]]}", fontsize=9)
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'patch_shuffle_examples.png'), dpi=300)
    plt.close()

def plot_cue_conflict_figures(conflicts, model_predictions, out_dir='figures/task1'):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Stylized cue-conflict examples (Accepted)
    accepted = [c for c in conflicts if c['status'] == 'accepted']
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for i in range(min(4, len(accepted))):
        c = accepted[i]
        c_cls = STL10_CLASSES[c['content_class']]
        s_cls = STL10_CLASSES[c['style_class']]
        
        axes[0, i].imshow(c['content_image'].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f"Content: {c_cls}", fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(c['image'].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f"Stylized (Shape:{c_cls}, Tex:{s_cls})\nSSIM={c['ssim']:.2f}, Edge={c['edge_corr']:.2f}", fontsize=9)
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cue_conflict_samples.png'), dpi=300)
    plt.close()
    
    # 2. Rejection rule examples (Accepted vs Rejected)
    rejected = [c for c in conflicts if c['status'] == 'rejected']
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    if len(accepted) > 0:
        c_acc = accepted[0]
        axes[0, 0].imshow(c_acc['content_image'].permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title(f"Accepted Content ({STL10_CLASSES[c_acc['content_class']]})", fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(c_acc['image'].permute(1, 2, 0).cpu().numpy())
        axes[0, 1].set_title(f"ACCEPTED (SSIM={c_acc['ssim']:.2f} >= 0.35)", color='green', fontsize=10)
        axes[0, 1].axis('off')
        
    if len(rejected) > 0:
        c_rej = rejected[0]
        axes[1, 0].imshow(c_rej['content_image'].permute(1, 2, 0).cpu().numpy())
        axes[1, 0].set_title(f"Rejected Content ({STL10_CLASSES[c_rej['content_class']]})", fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(c_rej['image'].permute(1, 2, 0).cpu().numpy())
        axes[1, 1].set_title(f"REJECTED (SSIM={c_rej['ssim']:.2f} or Edge={c_rej['edge_corr']:.2f})", color='red', fontsize=10)
        axes[1, 1].axis('off')
    else:
        # Placeholder if no rejections occurred
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rejection_rule_examples.png'), dpi=300)
    plt.close()

    # 3. Cue-conflict failure / disagreement cases
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i in range(min(4, len(accepted))):
        c = accepted[i]
        c_cls = STL10_CLASSES[c['content_class']]
        s_cls = STL10_CLASSES[c['style_class']]
        
        rn_pred = STL10_CLASSES[model_predictions['ResNet-50'][i]]
        vit_pred = STL10_CLASSES[model_predictions['ViT-B/16'][i]]
        clip_pred = STL10_CLASSES[model_predictions['OpenCLIP Zero-Shot'][i]]
        
        axes[i].imshow(c['image'].permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"Shape: {c_cls} | Tex: {s_cls}\nRN: {rn_pred}\nViT: {vit_pred}\nCLIP: {clip_pred}", fontsize=9)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cue_conflict_failures.png'), dpi=300)
    plt.close()

def plot_umap_embeddings(clean_feats, trans_feats, targets, model_name, filename, out_dir='figures/task1', seed=6304):
    os.makedirs(out_dir, exist_ok=True)
    
    # Subsample 150 points from clean and 150 from transformed for readable visualization
    n_sample = min(150, len(clean_feats))
    idx = np.random.RandomState(seed).choice(len(clean_feats), n_sample, replace=False)
    
    feats_comb = np.vstack([clean_feats[idx], trans_feats[idx]])
    conditions = ['Clean'] * n_sample + ['Transformed'] * n_sample
    class_labels = [STL10_CLASSES[targets[i]] for i in idx] * 2
    
    # Fit 2D UMAP with Cosine metric
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=seed)
    embedding = reducer.fit_transform(feats_comb)
    
    plt.figure(figsize=(8, 6.5))
    palette = sns.color_palette("tab10", 10)
    class_to_color = {STL10_CLASSES[i]: palette[i] for i in range(10)}
    
    # Plot clean with circle 'o', transformed with cross 'X'
    for c_name in STL10_CLASSES:
        # Clean
        mask_clean = [(c == 'Clean' and l == c_name) for c, l in zip(conditions, class_labels)]
        if any(mask_clean):
            plt.scatter(embedding[mask_clean, 0], embedding[mask_clean, 1],
                        c=[class_to_color[c_name]], marker='o', s=55, edgecolors='k', linewidth=0.5,
                        label=f"{c_name}" if c_name in [STL10_CLASSES[0], STL10_CLASSES[1]] else "")
            
        # Transformed
        mask_trans = [(c == 'Transformed' and l == c_name) for c, l in zip(conditions, class_labels)]
        if any(mask_trans):
            plt.scatter(embedding[mask_trans, 0], embedding[mask_trans, 1],
                        c=[class_to_color[c_name]], marker='X', s=70, edgecolors='k', linewidth=0.5)

    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Clean', markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker='X', color='w', label='Transformed', markerfacecolor='gray', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.title(f"{model_name} 2D UMAP Projection (Cosine Metric)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
