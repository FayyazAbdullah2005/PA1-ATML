import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import set_seed, SEED
from common.logging import get_logger
from common.plotting import plot_training_curves, plot_confusion_heatmap
from common.pacs import PACS_CLASSES
from common.pacs_protocol import get_pacs_datasets
from task2.models.classifier_head import PACSResNet18
from task2.evaluation.metrics import evaluate_model
from task2.evaluation.domain_separability import compute_domain_separability
from task2.evaluation.class_analysis import analyze_per_class_transfer, find_dominant_confusions

logger = get_logger("task2_eval")

def evaluate_all_adaptation_models(
    checkpoints={
        'Source-only (ERM)': 'checkpoints/pacs_erm_baseline.pt',
        'DAN': 'checkpoints/pacs_dan.pt',
        'DANN': 'checkpoints/pacs_dann.pt',
        'CDAN': 'checkpoints/pacs_cdan.pt'
    },
    results_path='task2/results/task2_results.json',
    figures_dir='figures/task2',
    seed=SEED,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_seed(seed)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    datasets = get_pacs_datasets(seed=seed)
    
    val_loaders = {
        d: DataLoader(datasets['source_val'][d], batch_size=32, shuffle=False)
        for d in ['photo', 'art_painting', 'cartoon']
    }
    tgt_loader = DataLoader(datasets['target_eval'], batch_size=32, shuffle=False)
    
    results = {'main_results': {}, 'per_class_comparison': {}, 'dominant_confusions': {}}
    erm_tgt_acc = None
    erm_per_class = None
    
    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}. Skipping {name}.")
            continue
            
        logger.info(f"Evaluating {name} from {ckpt_path}...")
        model = PACSResNet18(num_classes=7).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        # 1. Source validation evaluation
        src_domain_accs = {}
        src_accs = []
        src_f1s = []
        src_val_feats_list = []
        
        for d_name, v_loader in val_loaders.items():
            _, f1, acc, _, _, _, feats = evaluate_model(model, v_loader, device)
            display_name = d_name.replace('_painting', '').capitalize()
            src_domain_accs[display_name] = acc
            src_accs.append(acc)
            src_f1s.append(f1)
            src_val_feats_list.append(feats)
            
        mean_src_acc = float(np.mean(src_accs))
        mean_src_f1 = float(np.mean(src_f1s))
        src_val_feats = np.concatenate(src_val_feats_list, axis=0)
        
        # 2. Target domain evaluation (unlabeled during adaptation; transductive evaluation here)
        tgt_loss, tgt_f1, tgt_acc, tgt_per_class, tgt_preds, tgt_labels, tgt_feats = evaluate_model(
            model, tgt_loader, device
        )
        
        if 'ERM' in name or 'Source-only' in name:
            erm_tgt_acc = tgt_acc
            erm_per_class = {PACS_CLASSES[k]: v for k, v in tgt_per_class.items()}
            tgt_gain = 0.0
            
            # Save ERM baseline confusion matrix
            plot_confusion_heatmap(
                tgt_labels, tgt_preds, PACS_CLASSES,
                title=f"Normalized Confusion Matrix on Sketch (Source-only ERM)",
                save_path=os.path.join(figures_dir, "confusion_failures.png")
            )
        else:
            tgt_gain = float(tgt_acc - (erm_tgt_acc if erm_tgt_acc is not None else tgt_acc))
            
        # 3. Domain separability score
        dom_sep = compute_domain_separability(src_val_feats, tgt_feats, seed=seed)
        
        results['main_results'][name] = {
            'source_val': src_domain_accs,
            'mean_src_acc': mean_src_acc,
            'mean_src_f1': mean_src_f1,
            'target_acc': tgt_acc,
            'target_f1': tgt_f1,
            'target_gain': tgt_gain,
            'domain_separability': dom_sep,
            'per_class_target': {PACS_CLASSES[k]: v for k, v in tgt_per_class.items()}
        }
        
        results['dominant_confusions'][name] = find_dominant_confusions(
            tgt_labels, tgt_preds, class_names=PACS_CLASSES
        )
        
    # Per-class transfer analysis relative to ERM
    if erm_per_class is not None:
        for name, data in results['main_results'].items():
            if name != 'Source-only (ERM)':
                results['per_class_comparison'][name] = analyze_per_class_transfer(
                    erm_per_class, data['per_class_target'], class_names=PACS_CLASSES
                )
                
    # Preserve existing controlled study results if available
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                old_data = json.load(f)
                if 'controlled_study' in old_data:
                    results['controlled_study'] = old_data['controlled_study']
        except Exception:
            pass
            
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {results_path}")
    return results

if __name__ == "__main__":
    evaluate_all_adaptation_models()
