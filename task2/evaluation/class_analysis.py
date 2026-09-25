import numpy as np
from sklearn.metrics import confusion_matrix
from shared.pacs import PACS_CLASSES

def analyze_per_class_transfer(baseline_per_class, method_per_class, class_names=PACS_CLASSES):
    """
    Compares per-class target accuracies against the Source-only ERM baseline.
    Returns sorted improvements and degradations.
    """
    gains = {}
    for i, name in enumerate(class_names):
        base_acc = baseline_per_class.get(i, baseline_per_class.get(name, 0.0))
        m_acc = method_per_class.get(i, method_per_class.get(name, 0.0))
        gains[name] = m_acc - base_acc
        
    sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    best_improved = sorted_gains[0]
    worst_degraded = sorted_gains[-1]
    
    return {
        'per_class_gains': gains,
        'best_improved': {'class': best_improved[0], 'gain': best_improved[1]},
        'worst_degraded': {'class': worst_degraded[0], 'gain': worst_degraded[1]}
    }

def find_dominant_confusions(y_true, y_pred, class_names=PACS_CLASSES, top_k=3):
    """
    Finds top-k off-diagonal confusions (true -> pred).
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'true_class': class_names[i],
                    'pred_class': class_names[j],
                    'count': int(cm[i, j])
                })
                
    confusions = sorted(confusions, key=lambda x: x['count'], reverse=True)
    return confusions[:top_k]
