import torch
import torch.nn.functional as F

def compute_msp_score(logits):
    """
    Maximum Softmax Probability novelty score:
    u_MSP(x) = 1 - max_k p_k(x)
    """
    probs = F.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return 1.0 - max_probs
