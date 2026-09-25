import torch
import torch.nn as nn
import numpy as np
from common.metrics import compute_classification_metrics, compute_per_class_accuracy

def evaluate_model(model, dataloader, device):
    """
    Evaluates model performance: returns (mean_loss, macro_f1, top1_acc, per_class_acc, all_preds, all_labels, all_feats).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_feats = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            feat, logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            all_feats.append(feat.cpu().numpy())
            
    total_samples = len(all_labels)
    mean_loss = float(total_loss / total_samples)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_feats = np.concatenate(all_feats, axis=0)
    
    acc, f1 = compute_classification_metrics(all_labels, all_preds)
    per_class = compute_per_class_accuracy(all_labels, all_preds, num_classes=7)
    
    return mean_loss, f1, acc, per_class, all_preds, all_labels, all_feats
