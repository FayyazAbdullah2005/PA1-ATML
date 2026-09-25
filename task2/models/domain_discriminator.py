import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output.clone(), None

def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

def get_grl_alpha(progress):
    """
    Standard GRL schedule per Ganin et al. (2016) and assignment manual:
    alpha(p) = 2 / (1 + exp(-10*p)) - 1, where p in [0, 1].
    """
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0)

class DomainDiscriminator(nn.Module):
    """
    Standard domain discriminator per assignment specifications:
    - 256-unit hidden layer
    - ReLU activation
    - Dropout of 0.5
    - Two-class output layer (0 = Source, 1 = Target)
    """
    def __init__(self, in_features=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)
