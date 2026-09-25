import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.pacs_protocol import BalancedDomainBatchSampler
from task3.models.classifier_head import PACSResNet18
from task3.methods.erm import ERMMethod
from task3.methods.dan_dg import DANDGMethod
from task3.methods.sam import SAMMethod, SAM
from task3.evaluation.domain_metrics import evaluate_model
from task3.evaluation.sharpness import compute_sharpness_proxy
from task3.evaluation.source_domain_separability import compute_source_domain_separability

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_dg_method(
    method,
    pacs_data,
    device,
    max_epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    lambda_dg=1.0,
    rho=0.05
):
    """
    Trains a Domain Generalization method on the three source domains.
    - No Sketch (target) data is ever loaded during training or checkpoint selection.
    """
    # 1. Dataloaders
    batch_size_per_source = 8
    loader_p = DataLoader(pacs_data['source_train']['photo'], batch_size=batch_size_per_source, shuffle=True, drop_last=True)
    loader_a = DataLoader(pacs_data['source_train']['art_painting'], batch_size=batch_size_per_source, shuffle=True, drop_last=True)
    loader_c = DataLoader(pacs_data['source_train']['cartoon'], batch_size=batch_size_per_source, shuffle=True, drop_last=True)
    
    val_loaders = {
        'Photo': DataLoader(pacs_data['source_val']['photo'], batch_size=32, shuffle=False),
        'Art': DataLoader(pacs_data['source_val']['art_painting'], batch_size=32, shuffle=False),
        'Cartoon': DataLoader(pacs_data['source_val']['cartoon'], batch_size=32, shuffle=False)
    }
    
    # 2. Model Setup
    model = PACSResNet18(num_classes=7).to(device)
    params = list(model.parameters())
    
    if method == "erm":
        runner = ERMMethod(model)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif method == "dan_dg":
        runner = DANDGMethod(model, lambda_dg=lambda_dg)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif method == "sam":
        runner = SAMMethod(model)
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params, base_optimizer, rho=rho, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown DG method: {method}")
        
    history = {'cls_loss': [], 'align_loss': [], 'mean_val_f1': []}
    
    if method == "erm":
        logger.info("Loading pre-trained ERM baseline from Task 2...")
        model.load_state_dict(torch.load("checkpoints/pacs_erm_baseline.pt", map_location=device))
        best_val_f1 = 0.0 # Just a placeholder
        best_epoch = 0
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        best_val_f1 = -1.0
        best_model_weights = None
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, max_epochs + 1):
            model.train()
                
            epoch_cls_loss = 0.0
            epoch_align_loss = 0.0
            batch_count = 0
            
            # Sampler with no target loader!
            sampler = BalancedDomainBatchSampler(loader_p, loader_a, loader_c, loader_t=None)
            
            for batch_data in sampler:
                # When loader_t is None, sampler yields (x_src, y_src)
                x_src, y_src = batch_data
                x_src, y_src = x_src.to(device), y_src.to(device)
                
                if method == "sam":
                    # SAM requires closure
                    def closure():
                        loss_total, loss_cls, _ = runner.compute_loss(x_src, y_src)
                        loss_total.backward()
                        return loss_total, loss_cls
                    
                    loss_total, loss_cls, loss_align = runner.compute_loss(x_src, y_src)
                    loss_total.backward()
                    optimizer.first_step(zero_grad=True)
                    
                    # Second step
                    loss_total_2, loss_cls_2, _ = runner.compute_loss(x_src, y_src)
                    loss_total_2.backward()
                    optimizer.second_step(zero_grad=True)
                    
                else:
                    optimizer.zero_grad()
                    loss_total, loss_cls, loss_align = runner.compute_loss(x_src, y_src)
                    loss_total.backward()
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
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 5:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        logger.info(f"Training completed. Best Mean Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Load best weights
    model.load_state_dict(best_model_weights)
    
    # After model is trained, compute diagnostics
    logger.info("Computing Domain Separability...")
    sep_score = compute_source_domain_separability(
        model, 
        val_loaders['Photo'], val_loaders['Art'], val_loaders['Cartoon'], 
        device
    )
    
    logger.info("Computing Sharpness Proxy...")
    delta_sharp = compute_sharpness_proxy(
        model, 
        val_loaders['Photo'], val_loaders['Art'], val_loaders['Cartoon'], 
        device, 
        rho=0.05
    )
    
    return model, history, sep_score, delta_sharp
