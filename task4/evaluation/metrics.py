import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(known_scores, unknown_scores):
    """
    Computes AUROC where unknown class is the positive class.
    """
    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    y_scores = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_scores)

def compute_fpr_at_95_tpr(known_val_scores, known_test_scores, unknown_scores):
    """
    Calibrates threshold tau on known_val_scores to accept 95% of knowns 
    (i.e., reject 5% of knowns).
    Then evaluates rejection rate on unknown_scores (which is TPR if unknowns are positive).
    """
    # Threshold tau is the 95th percentile of known validation scores.
    # Because larger score means more unknown, accepting 95% of knowns means 
    # we threshold at the 95th percentile of the known score distribution.
    tau = np.percentile(known_val_scores, 95)
    
    # False Positive Rate on known test set (knowns incorrectly rejected)
    fpr = np.mean(known_test_scores > tau)
    
    # True Positive Rate on unknowns (unknowns correctly rejected)
    reject_rate_unknowns = np.mean(unknown_scores > tau)
    
    return tau, fpr, reject_rate_unknowns
