from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = labels if labels is not None else sorted(np.unique(y_true).tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[str(label)] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx]),
        }

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'per_class': per_class,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compute_binary_normal_abnormal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_labels: list[int] | tuple[int, ...] = (1, 2),
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_bin = np.isin(y_true, positive_labels).astype(int)
    y_pred_bin = np.isin(y_pred, positive_labels).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        'accuracy': float(accuracy_score(y_true_bin, y_pred_bin)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        'macro_f1': float(f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)),
        'negative_class': {
            'precision': float(precision[0]),
            'recall': float(recall[0]),
            'f1': float(f1[0]),
            'support': int(support[0]),
        },
        'positive_class': {
            'precision': float(precision[1]),
            'recall': float(recall[1]),
            'f1': float(f1[1]),
            'support': int(support[1]),
        },
    }
