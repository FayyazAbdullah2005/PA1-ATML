import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import torch
import numpy as np

from common.seed import set_seed, SEED
from common.logging import get_logger
from common.plotting import plot_training_curves
from task2.train import train_adaptation
from task2.evaluate_final import evaluate_all_adaptation_models

logger = get_logger("run_task2")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('figures/task2', exist_ok=True)
    os.makedirs('task2/results', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    # 1. Base histories for Source-only and DAN from existing stable runs
    hist_source_only = {
        'cls_loss': [0.4664, 0.2154, 0.1440, 0.1129, 0.0893, 0.0733, 0.0776, 0.0699, 0.0706, 0.0598],
        'align_loss': [0.0] * 10,
        'mean_val_f1': [0.8809, 0.9129, 0.9203, 0.9330, 0.9439, 0.9333, 0.9136, 0.9172, 0.9375, 0.9291]
    }
    
    hist_dan = {
        'cls_loss': [0.4676, 0.2305, 0.1620, 0.1220, 0.0754, 0.0828, 0.0718, 0.0635, 0.0644, 0.0599, 0.0581, 0.0588],
        'align_loss': [0.1603, 0.1314, 0.1255, 0.1289, 0.1155, 0.1170, 0.1173, 0.1153, 0.1129, 0.1139, 0.1153, 0.1114],
        'mean_val_f1': [0.8821, 0.9005, 0.9074, 0.9313, 0.9014, 0.9238, 0.9340, 0.9201, 0.9065, 0.9139, 0.9213, 0.9136]
    }
    
    # 2. Train DANN with gradient clipping
    logger.info("=" * 60)
    logger.info("Training DANN (Domain-Adversarial Neural Network)...")
    logger.info("=" * 60)
    model_dann, hist_dann = train_adaptation(
        method='dann',
        max_epochs=30,
        patience=5,
        grad_clip_norm=1.0,
        checkpoint_path='checkpoints/pacs_dann.pt',
        seed=SEED,
        device=device
    )
    
    # 3. Train CDAN with feature normalization and gradient clipping
    logger.info("=" * 60)
    logger.info("Training CDAN (Conditional Domain Adversarial Network)...")
    logger.info("=" * 60)
    model_cdan, hist_cdan = train_adaptation(
        method='cdan',
        max_epochs=30,
        patience=5,
        grad_clip_norm=1.0,
        checkpoint_path='checkpoints/pacs_cdan.pt',
        seed=SEED,
        device=device
    )
    
    # 4. Save histories
    histories = {
        'Source-only': hist_source_only,
        'DAN': hist_dan,
        'DANN': hist_dann,
        'CDAN': hist_cdan
    }
    with open('cache/task2_training_histories.json', 'w') as f:
        json.dump(histories, f, indent=2)
        
    # 5. Plot Loss Curves
    # Classification loss
    cls_curves = {m: h['cls_loss'] for m, h in histories.items()}
    plot_training_curves(
        cls_curves,
        title="Source Classification Loss Dynamics",
        xlabel="Epoch",
        ylabel="Classification Loss (Cross-Entropy)",
        save_path="figures/task2/training_loss_curves.png"
    )
    
    # Alignment loss (DAN: MMD; DANN: Domain Discriminator CE; CDAN: Conditioned Discriminator CE)
    align_curves = {
        'DAN (MMD)': hist_dan['align_loss'],
        'DANN (Adversarial)': hist_dann['align_loss'],
        'CDAN (Class-Conditional)': hist_cdan['align_loss']
    }
    plot_training_curves(
        align_curves,
        title="Domain Alignment Objective Dynamics",
        xlabel="Epoch",
        ylabel="Alignment / Discriminator Loss",
        save_path="figures/task2/alignment_loss_curves.png"
    )
    logger.info("Loss curves successfully exported to figures/task2/")
    
    # 6. Evaluate all models on target domain Sketch and save results
    logger.info("=" * 60)
    logger.info("Running Transductive Evaluation on Target Domain (Sketch)...")
    logger.info("=" * 60)
    evaluate_all_adaptation_models(
        checkpoints={
            'Source-only (ERM)': 'checkpoints/pacs_erm_baseline.pt',
            'DAN': 'checkpoints/pacs_dan.pt',
            'DANN': 'checkpoints/pacs_dann.pt',
            'CDAN': 'checkpoints/pacs_cdan.pt'
        },
        results_path='task2/results/task2_results.json',
        figures_dir='figures/task2',
        seed=SEED,
        device=device
    )
    logger.info("TASK 2 EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
