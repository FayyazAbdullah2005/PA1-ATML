import torch
import torch.nn as nn

class ERMMethod:
    """
    Empirical Risk Minimization baseline:
    Trained exclusively on labeled source domains with cross-entropy loss.
    In Task 3, we simply evaluate its checkpoint from Task 2.
    """
    def __init__(self, model):
        self.model = model
        self.criterion_cls = nn.CrossEntropyLoss()

    def compute_loss(self, x_src, y_src, **kwargs):
        feat_src, logits_src = self.model(x_src)
        loss_cls = self.criterion_cls(logits_src, y_src)
        return loss_cls, loss_cls, torch.tensor(0.0, device=x_src.device)
