from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecg_repo.data.loading import get_records_list, infer_record_keys, load_ecg_dataset
from ecg_repo.data.splits import assert_group_disjoint, choose_group_key, make_grouped_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', default='dpelacani/ecg-led2-cleaned')
    parser.add_argument('--split', default='train')
    parser.add_argument('--group-key', default='patient_id')
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outpath', default='outputs/splits/record_splits.csv')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_ecg_dataset(args.dataset_id)
    records = get_records_list(dataset, split=args.split)
    keys = infer_record_keys(records[0])

    rows = []
    for record_id, record in enumerate(records):
        rows.append(
            {
                'record_id': record_id,
                'patient_id': record[keys['patient_id']],
                'source_db': record[keys['source_db']],
            }
        )

    df = pd.DataFrame(rows)
    group_key = choose_group_key(df, preferred_key=args.group_key)
    split_df = make_grouped_splits(
        df=df,
        group_key=group_key,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    assert_group_disjoint(split_df, group_key=group_key)

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(outpath, index=False)

    print(f'Saved grouped splits to {outpath}')
    print(split_df['split'].value_counts().to_dict())
    print(f'Group key used: {group_key}')


if __name__ == '__main__':
    main()
