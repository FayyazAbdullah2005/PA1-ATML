import torch
import torch.nn as nn

def compute_pairwise_distances(x, y):
    """
    Computes pairwise squared Euclidean distances: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y^T.
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist_sq = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist_sq, min=0.0)

def compute_mmd_loss(source_feats, target_feats, kernel_multipliers=[0.5, 1.0, 2.0]):
    """
    Computes multi-kernel RBF Maximum Mean Discrepancy (MMD)
    between source and target feature representations.
    Bandwidths: 0.5, 1.0, 2.0 times the median pairwise squared feature distance.
    """
    n_s = source_feats.size(0)
    n_t = target_feats.size(0)
    
    combined = torch.cat([source_feats, target_feats], dim=0)
    total_dist = compute_pairwise_distances(combined, combined)
    
    non_diag = total_dist[total_dist > 0.0]
    if len(non_diag) > 0:
        median_dist = torch.median(non_diag)
    else:
        median_dist = torch.tensor(1.0, device=combined.device)
        
    if median_dist.item() == 0 or torch.isnan(median_dist):
        median_dist = torch.tensor(1.0, device=combined.device)
        
    dist_ss = compute_pairwise_distances(source_feats, source_feats)
    dist_tt = compute_pairwise_distances(target_feats, target_feats)
    dist_st = compute_pairwise_distances(source_feats, target_feats)
    
    loss_mmd = 0.0
    for m in kernel_multipliers:
        bandwidth = 2.0 * (m * median_dist + 1e-8)
        k_ss = torch.exp(-dist_ss / bandwidth)
        k_tt = torch.exp(-dist_tt / bandwidth)
        k_st = torch.exp(-dist_st / bandwidth)
        
        loss_mmd = loss_mmd + (k_ss.sum() / (n_s * n_s) + k_tt.sum() / (n_t * n_t) - 2.0 * k_st.sum() / (n_s * n_t))
        
    return loss_mmd

class DANMethod:
    """
    Deep Adaptation Network:
    L_DAN = L_cls + lambda_mmd * MMD^2(F(x_s), F(x_t))
    """
    def __init__(self, model, lambda_mmd=1.0, kernel_multipliers=[0.5, 1.0, 2.0]):
        self.model = model
        self.lambda_mmd = lambda_mmd
        self.kernel_multipliers = kernel_multipliers
        self.criterion_cls = nn.CrossEntropyLoss()

    def compute_loss(self, x_src, y_src, x_tgt, alpha=0.0):
        feat_src, logits_src = self.model(x_src)
        feat_tgt, _ = self.model(x_tgt)
        
        loss_cls = self.criterion_cls(logits_src, y_src)
        mmd_val = compute_mmd_loss(feat_src, feat_tgt, self.kernel_multipliers)
        loss_align = self.lambda_mmd * mmd_val
        loss_total = loss_cls + loss_align
        return loss_total, loss_cls, loss_align
