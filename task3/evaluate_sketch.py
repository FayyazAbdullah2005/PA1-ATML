import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import set_seed, SEED
from common.logging import get_logger
from common.pacs import PACS_CLASSES
from common.pacs_protocol import get_pacs_datasets
from task3.models.classifier_head import PACSResNet18
from task2.evaluation.metrics import evaluate_model
from task3.evaluation.source_domain_separability import compute_source_domain_separability
from task3.evaluation.sharpness import compute_sharpness_proxy

logger = get_logger("task3_eval")

def evaluate_all_dg_models(
    checkpoints={
        'ERM': 'checkpoints/pacs_erm_baseline.pt',
        'DAN-DG': 'checkpoints/task3_DAN-DG.pt',
        'SAM': 'checkpoints/task3_SAM.pt',
        'SAM (rho=0.01)': 'checkpoints/task3_SAM_rho0.01.pt',
        'SAM (rho=0.1)': 'checkpoints/task3_SAM_rho0.1.pt',
    },
    results_path='task3/results/task3_main_results.json',
    seed=SEED,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_seed(seed)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    pacs_data = get_pacs_datasets(seed=seed)
    
    val_loaders = {
        'Photo': DataLoader(pacs_data['source_val']['photo'], batch_size=32, shuffle=False),
        'Art': DataLoader(pacs_data['source_val']['art_painting'], batch_size=32, shuffle=False),
        'Cartoon': DataLoader(pacs_data['source_val']['cartoon'], batch_size=32, shuffle=False)
    }
    tgt_loader = DataLoader(pacs_data['target_eval'], batch_size=32, shuffle=False)
    
    results = []
    erm_tgt_acc = None
    
    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}. Skipping {name}.")
            continue
            
        logger.info(f"Evaluating {name} from {ckpt_path}...")
        model = PACSResNet18(num_classes=7).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        # Source validation
        src_accs = {}
        src_f1s = {}
        for d, loader in val_loaders.items():
            acc, f1, _, _, _, _, _ = evaluate_model(model, loader, device)
            src_accs[d] = float(acc)
            src_f1s[d] = float(f1)
            
        mean_src_acc = float(np.mean(list(src_accs.values())))
        worst_src_acc = float(np.min(list(src_accs.values())))
        
        # Target Sketch evaluation
        tgt_acc, tgt_f1, _, _, _, class_recalls, _ = evaluate_model(model, tgt_loader, device)
        tgt_acc = float(tgt_acc)
        tgt_f1 = float(tgt_f1)
        
        if name == 'ERM':
            erm_tgt_acc = tgt_acc
            delta_tgt_acc = 0.0
        else:
            delta_tgt_acc = float(tgt_acc - erm_tgt_acc) if erm_tgt_acc is not None else 0.0
            
        # Diagnostics
        logger.info(f"Computing Domain Separability for {name}...")
        sep_score = compute_source_domain_separability(
            model, val_loaders['Photo'], val_loaders['Art'], val_loaders['Cartoon'], device
        )
        
        logger.info(f"Computing Sharpness Proxy for {name}...")
        sharp_rho = 0.01 if 'rho=0.01' in name else (0.1 if 'rho=0.1' in name else 0.05)
        delta_sharp = compute_sharpness_proxy(
            model, val_loaders['Photo'], val_loaders['Art'], val_loaders['Cartoon'], device, rho=sharp_rho
        )
        
        results.append({
            'Method': name,
            'Photo Acc': src_accs['Photo'],
            'Art Acc': src_accs['Art'],
            'Cartoon Acc': src_accs['Cartoon'],
            'Mean Source Acc': mean_src_acc,
            'Worst Source Acc': worst_src_acc,
            'Sketch Target Acc': tgt_acc,
            'Sketch Target F1': tgt_f1,
            'Domain Sep Score': sep_score,
            'Sharpness Proxy': delta_sharp,
            'Target Class Accs': {str(k): float(v * 100) for k, v in class_recalls.items()},
            'Delta Sketch Acc': delta_tgt_acc
        })
        
        logger.info(
            f"[{name:<15}] Mean Src: {mean_src_acc:.2f}% | Worst Src: {worst_src_acc:.2f}% | "
            f"Sketch Acc: {tgt_acc:.2f}% (Δ: {delta_tgt_acc:+.2f}%) | "
            f"Sep Score: {sep_score:.2f}% | Sharpness: {delta_sharp:.4f}"
        )
        
    print("\n" + "="*80)
    print("TASK 3 EVALUATION SUMMARY (on unseen Sketch)")
    print("="*80)
    for r in results:
        print(f"{r['Method']:<18} | Mean Src: {r['Mean Source Acc']:.2f}% | Sketch: {r['Sketch Target Acc']:.2f}% | Δ: {r['Delta Sketch Acc']:+.2f}% | Sep: {r['Domain Sep Score']:.2f}% | Sharp: {r['Sharpness Proxy']:.4f}")
    print("="*80)
    
    return results

if __name__ == '__main__':
    evaluate_all_dg_models()
