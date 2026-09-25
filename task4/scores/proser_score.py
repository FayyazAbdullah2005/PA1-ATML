import torch

def compute_proser_score(model, logits, features):
    """
    PROSER placeholder novelty score:
    u_PROSER(x) = max_{k in dummy} z_k(x) - max_{k in known} z_k(x)
    """
    # dummy logits
    dummy_logits = model.backbone.dummy_fc(features)
    
    max_known, _ = torch.max(logits, dim=1)
    max_dummy, _ = torch.max(dummy_logits, dim=1)
    
    # Larger score means more unknown
    return max_dummy - max_known
