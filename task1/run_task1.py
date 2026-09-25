import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from task1.data_utils import get_stl10_datasets, STL10_CLASSES, set_seed, SEED
from task1.models import (
    ResNet50Extractor, ViTB16Extractor, OpenCLIPExtractor,
    extract_features, train_linear_head
)
from task1.transforms import (
    apply_grayscale, apply_hue_rotation_180, apply_translation, apply_patch_shuffle_4x4
)
from task1.cue_conflicts import generate_cue_conflicts
from task1.evaluate import (
    compute_metrics, compute_prediction_consistency, compute_shape_bias,
    compute_cosine_stability, plot_translation_curves, plot_patch_shuffle_samples,
    plot_cue_conflict_figures, plot_umap_embeddings
)

def run_experiment():
    print("=" * 75)
    print("STARTING TASK 1: INDUCTIVE BIASES AND FEATURE REPRESENTATIONS")
    print("=" * 75)
    
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/task1', exist_ok=True)
    os.makedirs('cache/task1', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: DATASET PREPARATION
    # -------------------------------------------------------------------------
    train_sub, val_sub, test_500_sub, test_dataset = get_stl10_datasets()
    
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_500_loader = DataLoader(test_500_sub, batch_size=128, shuffle=False, num_workers=0)
    
    # Extract clean test images and targets into memory (only 500 images)
    clean_test_imgs = []
    clean_test_targets = []
    for imgs, tgts in test_500_loader:
        clean_test_imgs.append(imgs)
        clean_test_targets.append(tgts)
    clean_test_imgs = torch.cat(clean_test_imgs, dim=0) # [500, 3, 224, 224]
    clean_test_targets = torch.cat(clean_test_targets, dim=0) # [500]
    
    # -------------------------------------------------------------------------
    # STEP 2: BACKBONE MODELS AND FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    print("\n--- Initializing Backbones ---")
    rn_ext = ResNet50Extractor().to(device)
    vit_ext = ViTB16Extractor().to(device)
    clip_ext = OpenCLIPExtractor(device=device)
    
    backbones = {
        'ResNet-50': rn_ext,
        'ViT-B/16': vit_ext,
        'OpenCLIP ViT-B-32': clip_ext
    }
    
    # Extract / cache features
    cache_path = 'cache/task1/clean_features.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        features_cache = torch.load(cache_path, map_location='cpu')
    else:
        print("Extracting train, val, and test clean features...")
        features_cache = {}
        for name, ext in backbones.items():
            print(f"Extracting for {name}...")
            tr_f, tr_y = extract_features(ext, train_loader, device)
            va_f, va_y = extract_features(ext, val_loader, device)
            te_f, te_y = extract_features(ext, test_500_loader, device)
            features_cache[name] = {
                'train': (tr_f, tr_y),
                'val': (va_f, va_y),
                'test': (te_f, te_y)
            }
        torch.save(features_cache, cache_path)
        print("Features cached successfully.")
        
    # -------------------------------------------------------------------------
    # STEP 3: TRAIN LINEAR HEADS & CLEAN BASELINE EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 1: CLEAN BASELINE EVALUATION")
    print("=" * 50)
    
    trained_heads = {}
    clean_metrics = {}
    clean_predictions = {}
    
    for name in ['ResNet-50', 'ViT-B/16', 'OpenCLIP ViT-B-32']:
        tr_f, tr_y = features_cache[name]['train']
        va_f, va_y = features_cache[name]['val']
        te_f, te_y = features_cache[name]['test']
        
        print(f"Training Linear Head for {name}...")
        head, best_val_acc = train_linear_head(
            tr_f, tr_y, va_f, va_y, num_classes=10,
            max_epochs=50, patience=5, lr=1e-3, wd=1e-4, seed=SEED, device=device
        )
        trained_heads[name] = head
        
        # Test evaluation
        with torch.no_grad():
            test_logits = head(te_f.to(device)).cpu()
            acc, f1, conf, preds = compute_metrics(test_logits, te_y)
            clean_metrics[name] = {'acc': acc, 'f1': f1, 'conf': conf}
            clean_predictions[name] = preds
            print(f"  {name} (Linear Head): Top-1={acc:.2f}%, Macro-F1={f1:.4f}, Mean Conf={conf:.4f}")
            
    # Zero-Shot CLIP
    print("Evaluating Zero-Shot OpenCLIP ViT-B-32...")
    with torch.no_grad():
        clip_te_f, clip_te_y = features_cache['OpenCLIP ViT-B-32']['test']
        text_feats = clip_ext.get_text_classifier().cpu() # [10, 512]
        
        # Scaled cosine similarity: logit_scale.exp() * (image @ text.T)
        scale = clip_ext.model.logit_scale.exp().item()
        sims = scale * (clip_te_f @ text_feats.T)
        zs_probs = torch.softmax(sims, dim=-1)
        
        zs_acc, zs_f1, zs_conf, zs_preds = compute_metrics(zs_probs, clip_te_y, is_probs=True)
        clean_metrics['OpenCLIP Zero-Shot'] = {'acc': zs_acc, 'f1': zs_f1, 'conf': zs_conf}
        clean_predictions['OpenCLIP Zero-Shot'] = zs_preds
        print(f"  OpenCLIP ViT-B-32 (Zero-Shot): Top-1={zs_acc:.2f}%, Macro-F1={zs_f1:.4f}, Mean Conf={zs_conf:.4f}")
        
    # -------------------------------------------------------------------------
    # STEP 4: COLOR BIAS (GRAYSCALE & 180 DEG HUE ROTATION)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 2: COLOR BIAS INTERVENTIONS")
    print("=" * 50)
    
    color_results = {}
    
    # Generate transformed image batches
    print("Generating Grayscale and 180° Hue Rotated images...")
    gray_imgs = apply_grayscale(clean_test_imgs)
    hue_imgs = apply_hue_rotation_180(clean_test_imgs)
    
    # Helper to evaluate model on a batch of transformed images
    def evaluate_on_transformed_images(trans_imgs):
        preds_out = {}
        feats_out = {}
        with torch.no_grad():
            for name, ext in backbones.items():
                feats = []
                for b in range(0, len(trans_imgs), 64):
                    batch = trans_imgs[b : b + 64].to(device)
                    f = ext(batch)
                    feats.append(f.cpu())
                feats = torch.cat(feats, dim=0)
                feats_out[name] = feats
                
                # Classify
                logits = trained_heads[name](feats.to(device)).cpu()
                probs = torch.softmax(logits, dim=-1)
                preds_out[name] = probs.argmax(dim=-1).numpy()
                
            # CLIP Zero-Shot
            clip_f = feats_out['OpenCLIP ViT-B-32']
            sims = clip_ext.model.logit_scale.exp().item() * (clip_f @ text_feats.T)
            preds_out['OpenCLIP Zero-Shot'] = torch.softmax(sims, dim=-1).argmax(dim=-1).numpy()
            
        return preds_out, feats_out

    preds_gray, feats_gray = evaluate_on_transformed_images(gray_imgs)
    preds_hue, feats_hue = evaluate_on_transformed_images(hue_imgs)
    
    model_eval_list = ['ResNet-50', 'ViT-B/16', 'OpenCLIP ViT-B-32', 'OpenCLIP Zero-Shot']
    for m in model_eval_list:
        clean_acc = clean_metrics[m]['acc']
        
        # Grayscale
        gray_acc = accuracy_score(clean_test_targets.numpy(), preds_gray[m]) * 100.0
        gray_cons = compute_prediction_consistency(clean_predictions[m], preds_gray[m])
        gray_delta = gray_acc - clean_acc
        
        # Hue
        hue_acc = accuracy_score(clean_test_targets.numpy(), preds_hue[m]) * 100.0
        hue_cons = compute_prediction_consistency(clean_predictions[m], preds_hue[m])
        hue_delta = hue_acc - clean_acc
        
        color_results[m] = {
            'gray_delta': gray_delta, 'gray_cons': gray_cons,
            'hue_delta': hue_delta, 'hue_cons': hue_cons
        }
        print(f"{m:22s} | Gray: dAcc={gray_delta:+.2f}%, Cons={gray_cons:.1f}% | Hue: dAcc={hue_delta:+.2f}%, Cons={hue_cons:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 5: SHAPE VS. TEXTURE (AdaIN CUE CONFLICTS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 3: SHAPE VS. TEXTURE BIAS (AdaIN CUE CONFLICTS)")
    print("=" * 50)
    
    cue_cache_path = 'cache/task1/cue_conflicts.pt'
    if os.path.exists(cue_cache_path):
        print(f"Loading cached cue conflicts from {cue_cache_path}...")
        cue_data = torch.load(cue_cache_path, map_location='cpu')
        conflicts = cue_data['conflicts']
        acc_count = cue_data['acc_count']
        rej_count = cue_data['rej_count']
    else:
        conflicts, acc_count, rej_count = generate_cue_conflicts(
            test_dataset, alpha=0.8, target_per_dir=22, ssim_thresh=0.35, edge_thresh=0.40, device=device
        )
        torch.save({'conflicts': conflicts, 'acc_count': acc_count, 'rej_count': rej_count}, cue_cache_path)
        
    valid_conflicts = [c for c in conflicts if c['status'] == 'accepted']
    valid_conflict_imgs = torch.stack([c['image'] for c in valid_conflicts], dim=0)
    content_classes = [c['content_class'] for c in valid_conflicts]
    style_classes = [c['style_class'] for c in valid_conflicts]
    
    print(f"\nEvaluating on {len(valid_conflicts)} valid cue-conflict images...")
    preds_cue, feats_cue = evaluate_on_transformed_images(valid_conflict_imgs)
    
    shape_bias_results = {}
    for m in model_eval_list:
        n_shape, n_tex, n_other, s_bias, cov = compute_shape_bias(
            preds_cue[m], content_classes, style_classes
        )
        shape_bias_results[m] = {
            'n_shape': n_shape, 'n_texture': n_tex, 'n_other': n_other,
            'shape_bias': s_bias, 'coverage': cov
        }
        print(f"{m:22s} | N_shape={n_shape:3d}, N_tex={n_tex:3d}, N_other={n_other:3d} | Shape Bias={s_bias:.2f}%, Coverage={cov:.2f}%")
        
    plot_cue_conflict_figures(conflicts, preds_cue, out_dir='figures/task1')
    print("Cue-conflict figures saved to figures/task1/")

    # -------------------------------------------------------------------------
    # STEP 6: TRANSLATION SENSITIVITY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 4: TRANSLATION SENSITIVITY")
    print("=" * 50)
    
    shifts = [0, 8, 16, 32]
    directions = ['north', 'south', 'east', 'west']
    
    trans_acc = {m: [] for m in model_eval_list}
    trans_cons = {m: [] for m in model_eval_list}
    feats_shift32 = None # save shift=32 features for cosine stability
    
    for shift in shifts:
        if shift == 0:
            for m in model_eval_list:
                trans_acc[m].append(clean_metrics[m]['acc'])
                trans_cons[m].append(100.0)
            continue
            
        dir_accs = {m: [] for m in model_eval_list}
        dir_conss = {m: [] for m in model_eval_list}
        
        last_feats = None
        for direction in directions:
            shifted_imgs = apply_translation(clean_test_imgs, shift=shift, direction=direction)
            preds_sh, feats_sh = evaluate_on_transformed_images(shifted_imgs)
            last_feats = feats_sh
            
            for m in model_eval_list:
                acc = accuracy_score(clean_test_targets.numpy(), preds_sh[m]) * 100.0
                cons = compute_prediction_consistency(clean_predictions[m], preds_sh[m])
                dir_accs[m].append(acc)
                dir_conss[m].append(cons)
                
        if shift == 32:
            feats_shift32 = last_feats
            
        for m in model_eval_list:
            mean_a = float(np.mean(dir_accs[m]))
            mean_c = float(np.mean(dir_conss[m]))
            trans_acc[m].append(mean_a)
            trans_cons[m].append(mean_c)
            print(f"Shift={shift:2d}px | {m:20s} Acc={mean_a:.2f}%, Consistency={mean_c:.2f}%")
            
    plot_translation_curves(shifts, trans_acc, trans_cons, out_dir='figures/task1')
    print("Translation curves saved to figures/task1/")

    # -------------------------------------------------------------------------
    # STEP 7: PATCH STRUCTURE PERMUTATION (4x4 SHUFFLE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 5: PATCH STRUCTURE PERMUTATION (4x4 GRID)")
    print("=" * 50)
    
    shuffled_imgs = apply_patch_shuffle_4x4(clean_test_imgs, seed=SEED)
    preds_shuf, feats_shuf = evaluate_on_transformed_images(shuffled_imgs)
    
    patch_results = {}
    for m in model_eval_list:
        clean_acc = clean_metrics[m]['acc']
        shuf_acc = accuracy_score(clean_test_targets.numpy(), preds_shuf[m]) * 100.0
        shuf_cons = compute_prediction_consistency(clean_predictions[m], preds_shuf[m])
        delta_acc = shuf_acc - clean_acc
        patch_results[m] = {
            'shuf_acc': shuf_acc, 'delta_acc': delta_acc, 'consistency': shuf_cons
        }
        print(f"{m:22s} | Shuffled Acc={shuf_acc:.2f}%, Delta={delta_acc:+.2f}%, Consistency={shuf_cons:.2f}%")
        
    plot_patch_shuffle_samples(clean_test_imgs, shuffled_imgs, clean_predictions, preds_shuf, out_dir='figures/task1')
    print("Patch-shuffled visualizations saved to figures/task1/")

    # -------------------------------------------------------------------------
    # STEP 8: REPRESENTATION ANALYSIS (COSINE STABILITY & UMAP)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 6: REPRESENTATION ANALYSIS (COSINE STABILITY & UMAP)")
    print("=" * 50)
    
    cosine_stability = {}
    for name in ['ResNet-50', 'ViT-B/16', 'OpenCLIP ViT-B-32']:
        clean_f, _ = features_cache[name]['test']
        
        i_gray = compute_cosine_stability(clean_f, feats_gray[name])
        # For cue conflict, compare on the subset of clean content images vs stylized
        cue_clean_content_imgs = torch.stack([c['content_image'] for c in valid_conflicts], dim=0)
        with torch.no_grad():
            ext = backbones[name]
            f_content = []
            for b in range(0, len(cue_clean_content_imgs), 64):
                f_content.append(ext(cue_clean_content_imgs[b:b+64].to(device)).cpu())
            f_content = torch.cat(f_content, dim=0)
        i_cue = compute_cosine_stability(f_content, feats_cue[name])
        
        i_trans = compute_cosine_stability(clean_f, feats_shift32[name])
        i_shuf = compute_cosine_stability(clean_f, feats_shuf[name])
        
        cosine_stability[name] = {
            'grayscale': i_gray, 'cue_conflict': i_cue,
            'translation_32': i_trans, 'patch_shuffle': i_shuf
        }
        print(f"{name:20s} | Gray={i_gray:.4f}, Cue={i_cue:.4f}, Trans(32)={i_trans:.4f}, Shuf={i_shuf:.4f}")

    print("\nFitting and generating 2D UMAP projections (Cosine metric)...")
    umap_files = {
        'ResNet-50': 'umap_resnet50.png',
        'ViT-B/16': 'umap_vitb16.png',
        'OpenCLIP ViT-B-32': 'umap_clip.png'
    }
    for name, fname in umap_files.items():
        clean_f, targets_t = features_cache[name]['test']
        # Combine clean with hue-rotated transformed features for clear manifold visualization
        plot_umap_embeddings(
            clean_f.numpy(), feats_hue[name].numpy(), targets_t.numpy(),
            model_name=name, filename=fname, out_dir='figures/task1', seed=SEED
        )
        print(f"  Saved {fname}")
        
    # -------------------------------------------------------------------------
    # STEP 9: SAVE ALL NUMERICAL RESULTS
    # -------------------------------------------------------------------------
    all_results = {
        'clean_metrics': clean_metrics,
        'color_bias': color_results,
        'shape_vs_texture': shape_bias_results,
        'cue_conflict_counts': {'accepted': acc_count, 'rejected': rej_count, 'valid_evaluated': len(valid_conflicts)},
        'translation': {'shifts': shifts, 'accuracy': trans_acc, 'consistency': trans_cons},
        'patch_shuffling': patch_results,
        'cosine_stability': cosine_stability
    }
    
    with open('results/task1_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nAll numeric results successfully saved to results/task1_results.json!")
    print("=" * 75)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == '__main__':
    run_experiment()
