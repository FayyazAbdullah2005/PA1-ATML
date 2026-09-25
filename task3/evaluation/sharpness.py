import torch
import torch.nn as nn
from common.seed import set_seed

def compute_sharpness_proxy(model, loader_p, loader_a, loader_c, device, rho=0.05, seed=6304):
    """
    Computes the standardized local sharpness proxy:
    Delta_sharp = L_val(theta + epsilon) - L_val(theta)
    where epsilon = rho * (grad(L_val) / ||grad(L_val)||_2).
    
    A fixed validation batch containing 32 examples from each source (seed 6304) is used.
    """
    set_seed(seed)
    
    # 1. Collect fixed batch of 32 from each source
    def get_32(loader):
        x_list, y_list = [], []
        count = 0
        for x, y in loader:
            take = min(32 - count, x.size(0))
            x_list.append(x[:take])
            y_list.append(y[:take])
            count += take
            if count >= 32:
                break
        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

    x_p, y_p = get_32(loader_p)
    x_a, y_a = get_32(loader_a)
    x_c, y_c = get_32(loader_c)
    
    x_val = torch.cat([x_p, x_a, x_c], dim=0).to(device)
    y_val = torch.cat([y_p, y_a, y_c], dim=0).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Place model in evaluation mode
    model.eval()
    
    # 2. Compute L_val(theta) and its gradients
    model.zero_grad()
    _, logits = model(x_val)
    loss_val_theta = criterion(logits, y_val)
    loss_val_theta.backward()
    
    # 3. Compute epsilon and perturb weights
    # We must save original weights to restore them later
    original_weights = {}
    grad_norm_sq = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm_sq += param.grad.norm(p=2).item() ** 2
            
    grad_norm = grad_norm_sq ** 0.5
    scale = rho / (grad_norm + 1e-12)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_weights[name] = param.data.clone()
                e_w = param.grad * scale
                param.add_(e_w)
                
    # 4. Compute L_val(theta + epsilon)
    # Forward pass again in eval mode
    with torch.no_grad():
        _, logits_perturbed = model(x_val)
        loss_val_perturbed = criterion(logits_perturbed, y_val)
        
    delta_sharp = loss_val_perturbed.item() - loss_val_theta.item()
    
    # 5. Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data = original_weights[name]
                
    model.zero_grad()
    return delta_sharp
