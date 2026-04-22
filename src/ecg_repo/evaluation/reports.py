from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecg_repo.utils.io import ensure_dir, write_json


def save_metrics_report(metrics: dict, path: str | Path) -> None:
    write_json(metrics, path)


def save_predictions(
    beat_ids: list[str],
    record_ids: list[int],
    y_true: list[int],
    y_pred: list[int],
    split: str,
    path: str | Path,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(
        {
            'beat_id': beat_ids,
            'record_id': record_ids,
            'y_true': y_true,
            'y_pred': y_pred,
            'split': split,
        }
    )
    df.to_csv(path, index=False)
