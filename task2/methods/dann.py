import torch
import torch.nn as nn
from task2.models.domain_discriminator import DomainDiscriminator, grad_reverse

class DANNMethod:
    """
    Domain-Adversarial Neural Network (Ganin et al., 2016):
    Minimizes source classification loss while maximizing domain confusion via GRL.
    """
    def __init__(self, model, discriminator=None, in_features=512, hidden_dim=256, dropout=0.5):
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
        feat_tgt, _ = self.model(x_tgt)
        
        # 3. Concatenate and normalize representations into discriminator
        all_feats = torch.cat([feat_src, feat_tgt], dim=0)
        norm_feats = torch.nn.functional.normalize(all_feats, p=2, dim=-1) * 20.0
        
        # 4. Apply Gradient Reversal
        rev_feats = grad_reverse(norm_feats, alpha)
        d_logits = self.discriminator(rev_feats)
        
        # Domain targets: 0 = Source, 1 = Target
        n_src = len(x_src)
        n_tgt = len(x_tgt)
        d_labels = torch.tensor([0] * n_src + [1] * n_tgt, dtype=torch.long, device=x_src.device)
        
        loss_align = self.criterion_domain(d_logits, d_labels)
        loss_total = loss_cls + loss_align
        return loss_total, loss_cls, loss_align
