import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.seed import set_seed, SEED
from task4.data.cifar10 import get_cifar10_datasets
from task4.data.cifar100_unknowns import get_cifar100_unknowns
from task4.models.resnet_cifar import CIFARResNet18
from task4.scores.msp import compute_msp_score
from task4.scores.mls import compute_mls_score
from task4.scores.energy import compute_energy_score
from task4.scores.mahalanobis import MahalanobisScorer
from task4.scores.proser_score import compute_proser_score
from task4.evaluation.metrics import compute_auroc, compute_fpr_at_95_tpr

def evaluate_all_osr_models(
    checkpoints={
        'vanilla': 'checkpoints/task4_vanilla.pt',
        'gcsc': 'checkpoints/task4_gcsc.pt',
        'proser': 'checkpoints/task4_proser.pt'
    },
    results_path='results/task4_results.json',
    seed=SEED,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_seed(seed)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Datasets
    cifar10_data = get_cifar10_datasets(use_randaug=False)
    cifar100_data = get_cifar100_unknowns()
    
    loader_val = DataLoader(cifar10_data['val'], batch_size=256, shuffle=False)
    loader_test = DataLoader(cifar10_data['test'], batch_size=256, shuffle=False)
    loader_near = DataLoader(cifar100_data['near'], batch_size=256, shuffle=False)
    loader_far = DataLoader(cifar100_data['far'], batch_size=256, shuffle=False)
    loader_train_unaug = DataLoader(cifar10_data['train'], batch_size=256, shuffle=False)
    
    # Models
    models = {}
    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint not found: {ckpt_path}. Skipping {name}.")
            continue
        model = CIFARResNet18(num_classes=10).to(device)
        if name == "proser":
            model.backbone.dummy_fc = nn.Linear(512, 5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models[name] = model

    def extract_outputs(m, loader):
        all_feats, all_logits, all_labels = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feat, logits = m(x, return_feature=True)
                all_feats.append(feat.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        return torch.cat(all_feats), torch.cat(all_logits), torch.cat(all_labels)

    cache = {}
    for name, m in models.items():
        cache[name] = {
            'val': extract_outputs(m, loader_val),
            'test': extract_outputs(m, loader_test),
            'near': extract_outputs(m, loader_near),
            'far': extract_outputs(m, loader_far)
        }
        if name == 'vanilla':
            cache[name]['train'] = extract_outputs(m, loader_train_unaug)

    # Fit Mahalanobis on unaugmented training features
    mah_scorer = MahalanobisScorer(num_classes=10)
    feat_tr, _, label_tr = cache['vanilla']['train']
    mah_scorer.fit(feat_tr.to(device), label_tr.to(device))

    results = []
    def eval_score(model_name, score_name, score_fn, use_mah=False, use_proser=False):
        c = cache[model_name]
        
        def get_score(split):
            feat, logits, _ = c[split]
            if use_mah:
                return score_fn(feat.to(device)).detach().cpu().numpy()
            elif use_proser:
                return score_fn(models[model_name], logits.to(device), feat.to(device)).detach().cpu().numpy()
            else:
                return score_fn(logits).detach().numpy()
                
        val_scores = get_score('val')
        test_scores = get_score('test')
        near_scores = get_score('near')
        far_scores = get_score('far')
        all_unknown_scores = np.concatenate([near_scores, far_scores])
        
        auroc_near = compute_auroc(test_scores, near_scores) * 100
        auroc_far = compute_auroc(test_scores, far_scores) * 100
        auroc_all = compute_auroc(test_scores, all_unknown_scores) * 100
        
        _, fpr_test, rej_near = compute_fpr_at_95_tpr(val_scores, test_scores, near_scores)
        _, _, rej_far = compute_fpr_at_95_tpr(val_scores, test_scores, far_scores)
        
        logits_test = c['test'][1]
        labels_test = c['test'][2]
        preds = logits_test.argmax(dim=1)
        csa = (preds == labels_test).float().mean().item() * 100
        
        results.append({
            'Model': model_name.upper(),
            'Score': score_name,
            'CSA (%)': csa,
            'AUROC Near': auroc_near,
            'AUROC Far': auroc_far,
            'AUROC All': auroc_all,
            'FPR@95TPR (%)': fpr_test * 100,
            'Near Rej. (%)': rej_near * 100,
            'Far Rej. (%)': rej_far * 100
        })

    # Vanilla post-hoc scores
    eval_score('vanilla', 'MSP', compute_msp_score)
    eval_score('vanilla', 'MLS', compute_mls_score)
    eval_score('vanilla', 'Energy', compute_energy_score)
    eval_score('vanilla', 'Mahalanobis', mah_scorer.compute_score, use_mah=True)

    # GCSC
    if 'gcsc' in models:
        eval_score('gcsc', 'MLS', compute_mls_score)

    # PROSER
    if 'proser' in models:
        eval_score('proser', 'MLS', compute_mls_score)
        eval_score('proser', 'Score', compute_proser_score, use_proser=True)

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TASK 4 OSR EVALUATION SUMMARY")
    print("="*80)
    print(df.round(2).to_string(index=False))
    print("="*80)
    
    return results

if __name__ == '__main__':
    evaluate_all_osr_models()
