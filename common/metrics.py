import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred):
    """
    Computes top-1 accuracy and macro-F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    return acc, macro_f1

def compute_per_class_accuracy(y_true, y_pred, num_classes=7):
    """
    Computes accuracy individually per class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = {}
    for c in range(num_classes):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            per_class[c] = float(np.mean(y_pred[mask] == y_true[mask]) * 100.0)
        else:
            per_class[c] = 0.0
    return per_class
