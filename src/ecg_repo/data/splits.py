from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def choose_group_key(df: pd.DataFrame, preferred_key: str = "patient_id") -> str:
    if preferred_key in df.columns and df[preferred_key].notna().all():
        return preferred_key
    if "record_id" in df.columns:
        return "record_id"
    raise KeyError("Neither preferred group key nor 'record_id' exists in dataframe.")


def make_grouped_splits(
    df: pd.DataFrame,
    group_key: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be >0 and sum to less than 1.")

    df = df.copy()
    groups = df[group_key].astype(str).to_numpy()

    indices = np.arange(len(df))
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(test_splitter.split(indices, groups=groups))

    trainval_df = df.iloc[trainval_idx].copy()
    remaining_val_fraction = val_size / (1.0 - test_size)

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=remaining_val_fraction, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(
        val_splitter.split(
            np.arange(len(trainval_df)),
            groups=trainval_df[group_key].astype(str).to_numpy(),
        )
    )

    df["split"] = "unassigned"
    df.loc[trainval_df.iloc[train_idx_rel].index, "split"] = "train"
    df.loc[trainval_df.iloc[val_idx_rel].index, "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"

    return df


def assert_group_disjoint(df: pd.DataFrame, group_key: str, split_column: str = "split") -> None:
    split_to_groups: dict[str, set[str]] = {}
    for split_name, part in df.groupby(split_column):
        split_to_groups[split_name] = set(part[group_key].astype(str).tolist())

    splits = list(split_to_groups)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            if split_to_groups[splits[i]].intersection(split_to_groups[splits[j]]):
                raise AssertionError(f"Group leakage detected between {splits[i]} and {splits[j]}.")
