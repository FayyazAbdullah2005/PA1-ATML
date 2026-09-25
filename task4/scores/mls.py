import torch

def compute_mls_score(logits):
    """
    Maximum Logit Score novelty score:
    u_MLS(x) = - max_k z_k(x)
    """
    max_logits, _ = torch.max(logits, dim=1)
    return -max_logits
