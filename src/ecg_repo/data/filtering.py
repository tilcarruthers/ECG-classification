from __future__ import annotations

import pandas as pd


def filter_beats_by_length(
    beat_table: pd.DataFrame,
    min_length: int = 32,
    max_length: int = 256,
) -> pd.DataFrame:
    mask = beat_table["length"].between(min_length, max_length, inclusive="both")
    return beat_table.loc[mask].reset_index(drop=True)
