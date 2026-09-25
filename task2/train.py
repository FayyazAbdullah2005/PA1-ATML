import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.seed import set_seed, SEED
from common.logging import get_logger
from shared.pacs import PACS_CLASSES
from shared.pacs_protocol import get_pacs_datasets, BalancedDomainBatchSampler
from task2.models.classifier_head import PACSResNet18
from task2.models.domain_discriminator import DomainDiscriminator, get_grl_alpha
from task2.methods.source_only import SourceOnlyMethod
from task2.methods.dan import DANMethod
from task2.methods.dann import DANNMethod
from task2.methods.cdan import CDANMethod
from task2.evaluation.metrics import evaluate_model

logger = get_logger("task2_train")

def train_adaptation(
    method="source_only",
    lambda_mmd=1.0,
    max_epochs=30,
    patience=5,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    checkpoint_path=None,
    seed=SEED,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_seed(seed)
    logger.info(f"Initializing training for method={method} on device={device}")
    
    # 1. Datasets & Loaders
    datasets = get_pacs_datasets(seed=seed)
    loader_p = DataLoader(datasets['source_train']['photo'], batch_size=8, shuffle=True, drop_last=True)
    loader_a = DataLoader(datasets['source_train']['art_painting'], batch_size=8, shuffle=True, drop_last=True)
    loader_c = DataLoader(datasets['source_train']['cartoon'], batch_size=8, shuffle=True, drop_last=True)
    loader_t = DataLoader(datasets['target_adapt'], batch_size=24, shuffle=True, drop_last=True)
    
    val_loaders = {
        d: DataLoader(datasets['source_val'][d], batch_size=32, shuffle=False)
        for d in ['photo', 'art_painting', 'cartoon']
    }
    
    # 2. Model & Discriminator Setup
    model = PACSResNet18(num_classes=7).to(device)
    discriminator = None
    
    if method == "source_only":
        runner = SourceOnlyMethod(model)
    elif method == "dan":
        runner = DANMethod(model, lambda_mmd=lambda_mmd)
    elif method == "dann":
        discriminator = DomainDiscriminator(in_features=512, hidden_dim=256).to(device)
        runner = DANNMethod(model, discriminator=discriminator)
    elif method == "cdan":
        discriminator = DomainDiscriminator(in_features=512 * 7, hidden_dim=256).to(device)
        runner = CDANMethod(model, discriminator=discriminator)
    else:
        raise ValueError(f"Unknown adaptation method: {method}")
        
    # Collect trainable parameters
    params = list(model.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # 3. Training Loop with GRL Schedule and Validation Tracking
    total_steps = max_epochs * max(len(loader_p), len(loader_a), len(loader_c))
    step = 0
    history = {'cls_loss': [], 'align_loss': [], 'mean_val_f1': []}
    
    best_val_f1 = -1.0
    best_model_weights = None
    best_disc_weights = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
            
        epoch_cls_loss = 0.0
        epoch_align_loss = 0.0
        batch_count = 0
        
        sampler = BalancedDomainBatchSampler(loader_p, loader_a, loader_c, loader_t)
        
        for x_src, y_src, x_tgt in sampler:
            x_src, y_src = x_src.to(device), y_src.to(device)
            x_tgt = x_tgt.to(device)
            
            optimizer.zero_grad()
            step += 1
            progress = min(1.0, step / total_steps)
            alpha = get_grl_alpha(progress)
            
            loss_total, loss_cls, loss_align = runner.compute_loss(x_src, y_src, x_tgt, alpha=alpha)
            
            loss_total.backward()
            
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                
            optimizer.step()
            
            epoch_cls_loss += loss_cls.item()
            epoch_align_loss += loss_align.item()
            batch_count += 1
            
        epoch_cls_loss /= batch_count
        epoch_align_loss /= batch_count
        history['cls_loss'].append(epoch_cls_loss)
        history['align_loss'].append(epoch_align_loss)
        
        # Source-validation checkpoint selection (Mean Macro-F1 across Photo, Art, Cartoon)
        model.eval()
        val_f1s = []
        for d_name, v_loader in val_loaders.items():
            _, f1, _, _, _, _, _ = evaluate_model(model, v_loader, device)
            val_f1s.append(f1)
        mean_v_f1 = float(np.mean(val_f1s))
        history['mean_val_f1'].append(mean_v_f1)
        
        logger.info(
            f"[{method.upper():<11}] Epoch {epoch:02d}/{max_epochs} | "
            f"Cls Loss: {epoch_cls_loss:.4f} | Align Loss: {epoch_align_loss:.4f} | "
            f"Mean Val F1: {mean_v_f1:.4f}"
        )
        
        # Check early stopping
        if mean_v_f1 > best_val_f1:
            best_val_f1 = mean_v_f1
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            if discriminator is not None:
                best_disc_weights = copy.deepcopy(discriminator.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (Best Epoch: {best_epoch}).")
                break
                
    # Restore best checkpoint
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        if discriminator is not None and best_disc_weights is not None:
            discriminator.load_state_dict(best_disc_weights)
            
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PACS Adaptation Model")
    parser.add_argument("--method", type=str, default="source_only", choices=["source_only", "dan", "dann", "cdan"])
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    ckpt = args.checkpoint or f"checkpoints/pacs_{args.method}.pt"
    train_adaptation(
        method=args.method,
        lambda_mmd=args.lambda_mmd,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_path=ckpt
    )
