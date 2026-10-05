"""Threshold-selection and scoring helpers shared by training and evaluation."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return float(p), float(r), float(f)


def best_threshold(y_true, scores, grid=None):
    """Threshold maximising F1. Returns (threshold, precision, recall, f1)."""
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    best = (0.5, 0.0, 0.0, -1.0)
    for t in grid:
        p, r, f = prf(y_true, scores >= t)
        if f > best[3]:
            best = (float(t), p, r, f)
    return best


def summarize(y_true, scores, threshold):
    p, r, f = prf(y_true, scores >= threshold)
    ap = float(average_precision_score(y_true, scores)) if y_true.sum() else 0.0
    return {"precision": p, "recall": r, "f1": f, "average_precision": ap, "threshold": float(threshold)}
