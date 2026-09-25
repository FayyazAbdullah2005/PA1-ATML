import torch

def compute_energy_score(logits):
    """
    Energy novelty score:
    u_Energy(x) = - log sum_k exp(z_k(x))
    """
    return -torch.logsumexp(logits, dim=1)
