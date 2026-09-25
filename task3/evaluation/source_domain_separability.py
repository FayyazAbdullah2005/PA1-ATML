import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from common.seed import set_seed

def extract_features(model, loader, device):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feat, _ = model(x)
            all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)

def compute_source_domain_separability(model, loader_p, loader_a, loader_c, device, seed=6304):
    """
    Computes source-domain separability score using balanced features from validation sets.
    """
    set_seed(seed)
    
    feats_p = extract_features(model, loader_p, device)
    feats_a = extract_features(model, loader_a, device)
    feats_c = extract_features(model, loader_c, device)
    
    # Collect balanced features from the three source validation sets
    min_len = min(len(feats_p), len(feats_a), len(feats_c))
    
    # Take first min_len to balance
    feats_p = feats_p[:min_len]
    feats_a = feats_a[:min_len]
    feats_c = feats_c[:min_len]
    
    X = np.concatenate([feats_p, feats_a, feats_c], axis=0)
    # Labels: 0 for Photo, 1 for Art, 2 for Cartoon
    y = np.concatenate([
        np.zeros(min_len),
        np.ones(min_len),
        np.full(min_len, 2)
    ], axis=0)
    
    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    
    # Train multinomial logistic-regression classifier with C = 1
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100.0
    return acc
