from __future__ import annotations

import numpy as np


def add_gaussian_noise(signal: np.ndarray, std: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return signal + rng.normal(0.0, std, size=signal.shape).astype(signal.dtype)


def amplitude_jitter(signal: np.ndarray, scale_range: tuple[float, float] = (0.9, 1.1), rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    return signal * scale


def augment_beat(signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    augmented = signal.copy()
    if rng.uniform() < 0.5:
        augmented = add_gaussian_noise(augmented, std=0.01, rng=rng)
    if rng.uniform() < 0.5:
        augmented = amplitude_jitter(augmented, rng=rng)
    return augmented.astype(signal.dtype)
