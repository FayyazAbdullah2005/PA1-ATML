import torch
import torch.nn as nn
import torch.nn.functional as F
from task2.models.domain_discriminator import DomainDiscriminator, grad_reverse

def compute_cdan_features(features, logits, scale=True):
    """
    Multilinear conditioning per Long et al. (2018):
    g(x) = vec(f (x) p)
    f in R^512, p in R^7 -> g(x) in R^3584
    Normalized by 1/sqrt(d) for numerical stability under AdamW.
    """
    softmax_p = F.softmax(logits, dim=-1) # [B, 7]
    op = torch.bmm(features.unsqueeze(2), softmax_p.unsqueeze(1)) # [B, 512, 7]
    flat_op = torch.flatten(op, 1) # [B, 3584]
    if scale:
        flat_op = flat_op / (features.size(1) ** 0.5)
    return flat_op

class CDANMethod:
    """
    Conditional Domain Adversarial Network (Long et al., 2018):
    Conditions the domain discriminator on classifier predictions.
    """
    def __init__(self, model, discriminator=None, in_features=512*7, hidden_dim=256, dropout=0.5):
        self.model = model
        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = DomainDiscriminator(in_features=in_features, hidden_dim=hidden_dim)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_domain = nn.CrossEntropyLoss()

    def compute_loss(self, x_src, y_src, x_tgt, alpha):
        # 1. Source forward pass
        feat_src, logits_src = self.model(x_src)
        loss_cls = self.criterion_cls(logits_src, y_src)
        
        # 2. Target forward pass
        feat_tgt, logits_tgt = self.model(x_tgt)
        
        # 3. Multilinear conditioning on normalized representations
        f_src_norm = F.normalize(feat_src, p=2, dim=-1) * 20.0
        f_tgt_norm = F.normalize(feat_tgt, p=2, dim=-1) * 20.0
        g_src = compute_cdan_features(f_src_norm, logits_src)
        g_tgt = compute_cdan_features(f_tgt_norm, logits_tgt)
        all_g = torch.cat([g_src, g_tgt], dim=0)
        
        # 4. GRL and Discriminator pass
        rev_g = grad_reverse(all_g, alpha)
        d_logits = self.discriminator(rev_g)
        
        n_src = len(x_src)
        n_tgt = len(x_tgt)
        d_labels = torch.tensor([0] * n_src + [1] * n_tgt, dtype=torch.long, device=x_src.device)
        
        loss_align = self.criterion_domain(d_logits, d_labels)
        loss_total = loss_cls + loss_align
        return loss_total, loss_cls, loss_align
