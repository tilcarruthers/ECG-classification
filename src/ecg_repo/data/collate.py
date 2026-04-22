from __future__ import annotations

from typing import Any

import torch

from ecg_repo.data.dataset import ECGSample


def pad_collate(
    batch: list[ECGSample],
    max_length: int = 256,
    pad_value: float = 0.0,
) -> dict[str, Any]:
    lengths = torch.tensor([min(sample.length, max_length) for sample in batch], dtype=torch.long)
    batch_size = len(batch)
    padded = torch.full((batch_size, max_length, 1), fill_value=pad_value, dtype=torch.float32)
    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)

    beat_ids: list[str] = []
    record_ids: list[int] = []

    for i, sample in enumerate(batch):
        seq = sample.inputs[:max_length]
        padded[i, : seq.shape[0], :] = seq
        beat_ids.append(sample.beat_id)
        record_ids.append(sample.record_id)

    return {
        "inputs": padded,
        "labels": labels,
        "lengths": lengths,
        "beat_ids": beat_ids,
        "record_ids": record_ids,
    }
