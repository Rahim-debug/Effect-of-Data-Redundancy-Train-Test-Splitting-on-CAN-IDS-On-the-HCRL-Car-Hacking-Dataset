"""Task M1 - evaluation metrics (proposal Step 8), written from the formulas.

Rows of the confusion matrix = true class, columns = predicted class
(same convention as scikit-learn). Class order is fixed in common.CLASSES.

Undefined cases (documented on purpose, they matter for invalid P2 splits):
  * precision is NaN for a class that is never predicted, recall is NaN for a
    class with no true frames (support 0)  -> shown as blank, not as a fake 0;
  * F1_c = 2TP / (2TP + FP + FN) is set to 0 when its denominator is 0
    (this is scikit-learn's zero_division=0), so macro-F1 always averages
    over all C classes, exactly as in the proposal: (1/C) * sum_c F1_c.
"""
from __future__ import annotations

import numpy as np

from .common import CLASSES, N_CLASSES


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    for a in (y_true, y_pred):
        if a.size and (a.min() < 0 or a.max() >= n_classes):
            raise ValueError("labels must be integers in 0..n_classes-1")
    idx = y_true * n_classes + y_pred
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class(cm: np.ndarray) -> dict:
    """Per-class precision, recall, F1 and support from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)       # TP + FN
    predicted = cm.sum(axis=0)     # TP + FP
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        denom = support + predicted  # = 2TP + FP + FN
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1(cm: np.ndarray) -> float:
    return float(per_class(cm)["f1"].mean())


def accuracy(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else float("nan")


def trivial_baseline_macro_f1(pi_normal: float, n_classes: int = N_CLASSES) -> float:
    """Macro-F1 of 'always predict Normal': 2*pi / (C * (1 + pi)). Accuracy = pi."""
    return 2.0 * pi_normal / (n_classes * (1.0 + pi_normal))


def class_weights(y_train, n_classes: int = N_CLASSES) -> np.ndarray:
    """w_c = N / (C * N_c) on the training side (proposal Step 6).
    Equals scikit-learn's class_weight='balanced'. 0 for a class absent from train."""
    counts = np.bincount(np.asarray(y_train).astype(np.int64), minlength=n_classes).astype(float)
    n = counts.sum()
    with np.errstate(divide="ignore"):
        return np.where(counts > 0, n / (n_classes * counts), 0.0)


def evaluate(y_true, y_pred, class_names=CLASSES) -> dict:
    """One flat dict of everything a results row needs."""
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    pc = per_class(cm)
    out = {"accuracy": accuracy(cm), "macro_f1": macro_f1(cm), "n_test": int(cm.sum())}
    for i, name in enumerate(class_names):
        out[f"recall_{name}"] = float(pc["recall"][i])
        out[f"precision_{name}"] = float(pc["precision"][i])
        out[f"f1_{name}"] = float(pc["f1"][i])
        out[f"support_{name}"] = int(pc["support"][i])
    return out


def all_normal_baseline(y_true, n_classes: int = N_CLASSES) -> dict:
    """Empirical all-Normal predictor on the given labels (should match the formula)."""
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, np.zeros_like(y_true), n_classes)
    return {"accuracy": accuracy(cm), "macro_f1": macro_f1(cm), "pi_normal": float((y_true == 0).mean())}
