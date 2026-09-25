import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from common.seed import SEED

def compute_domain_separability(source_val_feats, target_feats, seed=SEED):
    """
    Measures domain separability per assignment specification:
    - Collect equal numbers of source-validation and target features.
    - Balanced 70/30 train/test split using seed 6304.
    - Logistic regression classifier with C=1.
    - Held-out accuracy is the domain separability score (50% = chance).
    """
    n_samples = min(len(source_val_feats), len(target_feats))
    
    rng = np.random.RandomState(seed)
    idx_src = rng.choice(len(source_val_feats), size=n_samples, replace=False)
    idx_tgt = rng.choice(len(target_feats), size=n_samples, replace=False)
    
    X_src = source_val_feats[idx_src]
    X_tgt = target_feats[idx_tgt]
    
    y_src = np.zeros(n_samples, dtype=int)
    y_tgt = np.ones(n_samples, dtype=int)
    
    X = np.concatenate([X_src, X_tgt], axis=0)
    y = np.concatenate([y_src, y_tgt], axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    held_out_acc = float(clf.score(X_test, y_test) * 100.0)
    
    return held_out_acc
