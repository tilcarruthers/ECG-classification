from __future__ import annotations

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(labels: np.ndarray, classes: list[int] | np.ndarray) -> torch.Tensor:
    classes_np = np.asarray(classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes_np, y=labels)
    return torch.tensor(weights, dtype=torch.float32)
