import torch
import torch.nn as nn

def compute_pairwise_distances(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist_sq = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist_sq, min=0.0)

def compute_mmd_loss(source_feats, target_feats, kernel_multipliers=[0.5, 1.0, 2.0]):
    n_s = source_feats.size(0)
    n_t = target_feats.size(0)
    
    if n_s == 0 or n_t == 0:
        return torch.tensor(0.0, device=source_feats.device)
        
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

class DANDGMethod:
    """
    DAN-DG: Pairwise Source-Domain Alignment.
    L_DAN-DG = L_cls + (lambda_dg / 3) * [MMD^2(P,A) + MMD^2(A,C) + MMD^2(P,C)]
    """
    def __init__(self, model, lambda_dg=1.0, kernel_multipliers=[0.5, 1.0, 2.0]):
        self.model = model
        self.lambda_dg = lambda_dg
        self.kernel_multipliers = kernel_multipliers
        self.criterion_cls = nn.CrossEntropyLoss()

    def compute_loss(self, x_src, y_src, **kwargs):
        feat_src, logits_src = self.model(x_src)
        
        # Cross entropy over all source examples
        loss_cls = self.criterion_cls(logits_src, y_src)
        
        # Pairwise MMD
        # The sampler yields batches by concatenating: Photo, Art, Cartoon.
        # Assuming equal batch sizes of N for each source domain (typically 8)
        N = x_src.size(0) // 3
        
        feat_p = feat_src[0:N]
        feat_a = feat_src[N:2*N]
        feat_c = feat_src[2*N:]
        
        mmd_pa = compute_mmd_loss(feat_p, feat_a, self.kernel_multipliers)
        mmd_ac = compute_mmd_loss(feat_a, feat_c, self.kernel_multipliers)
        mmd_pc = compute_mmd_loss(feat_p, feat_c, self.kernel_multipliers)
        
        mmd_avg = (mmd_pa + mmd_ac + mmd_pc) / 3.0
        loss_align = self.lambda_dg * mmd_avg
        
        loss_total = loss_cls + loss_align
        return loss_total, loss_cls, loss_align
