"""Performance monitoring (needs ground-truth labels)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import config as C

METRICS = ["precision", "recall", "f1", "roc_auc"]


def label_status(y):
    """'ok' | 'none' | 'single_class'"""
    if y is None:
        return "none"
    y = pd.Series(y).dropna()
    if len(y) == 0:
        return "none"
    return "ok" if y.nunique() == 2 else "single_class"


def compute_metrics(y, proba, thr=C.THRESHOLD):
    y, proba = np.asarray(y).astype(int), np.asarray(proba)
    pred = (proba >= thr).astype(int)
    return dict(precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, proba)),
                confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist(), n=int(len(y)))


def compare(current: dict, baseline: dict):
    rows = []
    for m in METRICS:
        drop = baseline[m] - current[m]
        status = "critical" if drop >= C.PERF_CRIT_DROP else "warning" if drop >= C.PERF_WARN_DROP else "ok"
        rows.append(dict(metric=m, baseline=baseline[m], current=current[m], change=-drop, status=status))
    return rows
