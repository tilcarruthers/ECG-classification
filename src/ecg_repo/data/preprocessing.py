from __future__ import annotations

import numpy as np


def zscore_per_beat(beat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    std = float(beat.std())
    if std < eps:
        return beat - beat.mean()
    return (beat - beat.mean()) / std
