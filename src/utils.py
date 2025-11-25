import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(y_true, y_pred_proba):
    out = {}
    try:
        out['auc'] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        out['auc'] = None
    try:
        out['apr'] = float(average_precision_score(y_true, y_pred_proba))
    except Exception:
        out['apr'] = None
    return out
