import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

def metrics(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_f1": f1_score(y_true, y_pred, average=None, labels=list(range(n_classes))).tolist(),
    }
