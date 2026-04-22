import numpy as np

from ecg_repo.evaluation.metrics import (
    compute_binary_normal_abnormal_metrics,
    compute_classification_metrics,
)


def test_multiclass_metrics_basic():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert 'macro_f1' in metrics
    assert 'per_class' in metrics
    assert len(metrics['confusion_matrix']) == 3


def test_binary_normal_abnormal_metrics_basic():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    metrics = compute_binary_normal_abnormal_metrics(y_true, y_pred, positive_labels=[1, 2])
    assert 'positive_class' in metrics
    assert 'negative_class' in metrics
