import numpy as np
import pandas as pd
import torch

from ecg_repo.data.collate import pad_collate
from ecg_repo.data.dataset import ECGBeatDataset


def test_dataset_and_collate_shapes():
    records = [{'signal': np.linspace(0, 1, 100, dtype=np.float32)}]
    beat_table = pd.DataFrame(
        [
            {'beat_id': '0_0', 'record_id': 0, 'start_idx': 0, 'end_idx': 30, 'label_mapped': 0},
            {'beat_id': '0_1', 'record_id': 0, 'start_idx': 10, 'end_idx': 50, 'label_mapped': 1},
        ]
    )
    ds = ECGBeatDataset(records=records, beat_table=beat_table, signal_key='signal', normalize=False, augment=False)
    batch = pad_collate([ds[0], ds[1]], max_length=64)
    assert batch['inputs'].shape == (2, 64, 1)
    assert torch.equal(batch['lengths'], torch.tensor([30, 40]))
