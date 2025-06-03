import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

def get_metrics(y_true, y_pred, y_prob=None, average='macro', n_classes=None):
    """
    Calculate standard metrics for classification tasks.
    """
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average),
    }
    if y_prob is not None and n_classes is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, np.array(y_prob), multi_class='ovr', average=average)
        except:
            metrics['roc_auc'] = float('nan')
    return metrics

def save_classification_report(y_true, y_pred, output_path):
    """
    Save sklearn's classification report as a JSON file.
    """
    import json
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
