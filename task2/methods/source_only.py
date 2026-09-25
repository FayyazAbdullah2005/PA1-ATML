import torch
import torch.nn as nn

class SourceOnlyMethod:
    """
    Empirical Risk Minimization baseline:
    Trains exclusively on labeled source domains with cross-entropy loss.
    """
    def __init__(self, model):
        self.model = model
        self.criterion_cls = nn.CrossEntropyLoss()

    def compute_loss(self, x_src, y_src, x_tgt=None, alpha=0.0):
        feat_src, logits_src = self.model(x_src)
        loss_cls = self.criterion_cls(logits_src, y_src)
        loss_align = torch.tensor(0.0, device=x_src.device)
        loss_total = loss_cls
        return loss_total, loss_cls, loss_align
