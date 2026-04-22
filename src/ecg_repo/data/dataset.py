from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecg_repo.data.augmentation import augment_beat
from ecg_repo.data.preprocessing import zscore_per_beat


@dataclass
class ECGSample:
    inputs: torch.Tensor
    label: int
    length: int
    beat_id: str
    record_id: int


class ECGBeatDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        beat_table: pd.DataFrame,
        signal_key: str = "signal",
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.beat_table = beat_table.reset_index(drop=True)
        self.signal_key = signal_key
        self.normalize = normalize
        self.augment = augment

    def __len__(self) -> int:
        return len(self.beat_table)

    def __getitem__(self, idx: int) -> ECGSample:
        row = self.beat_table.iloc[idx]
        record = self.records[int(row["record_id"])]
        signal = np.asarray(record[self.signal_key], dtype=np.float32)
        beat = signal[int(row["start_idx"]) : int(row["end_idx"])].copy()

        if self.normalize:
            beat = zscore_per_beat(beat)
        if self.augment:
            beat = augment_beat(beat)

        inputs = torch.from_numpy(beat).float().unsqueeze(-1)
        return ECGSample(
            inputs=inputs,
            label=int(row["label_mapped"]),
            length=len(beat),
            beat_id=str(row["beat_id"]),
            record_id=int(row["record_id"]),
        )
